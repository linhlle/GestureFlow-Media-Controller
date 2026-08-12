"""Record and replay landmark streams.

Why record landmarks and not video
----------------------------------
The thing worth making reproducible is everything *downstream* of MediaPipe:
normalization, the classifier, the debouncer, the three FSMs, and the routing
rules.  Recording landmarks freezes MediaPipe's output as a fixed input, so a
replay exercises exactly the code whose behaviour is in question, runs in
milliseconds instead of real time, needs no camera, and produces byte-identical
results on every machine.  Recording video would drag MediaPipe's own
version-to-version drift into every comparison.

What this unlocks
-----------------
* **Regression tests** over real recorded hand motion, not synthetic landmarks.
* **A/B threshold comparison** -- run the same take through two configs and
  diff the actions.  This is the only honest way to claim a threshold change
  improved anything.
* **False-trigger rate.**  Record footage where you are typing, talking, and
  gesturing with no intent to control anything; label it no-intent; replay it;
  count the actions that fired.  That count over the take's duration is a
  number worth quoting, because it measures precisely what the debouncer and
  the FSM hysteresis exist to prevent.

Format: JSONL.  One header line, then one line per frame.  Text so takes diff
cleanly in git; line-oriented so a long take streams instead of loading whole.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.config import DEFAULT_CONFIG, AppConfig
from gestureflow.inference import InferenceThread

RECORDING_SCHEMA = "gestureflow.recording/1"


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

@dataclass
class RecordingHeader:
    schema: str = RECORDING_SCHEMA
    label: str = "unlabelled"
    note: str = ""
    frame_width: int = 640
    frame_height: int = 480
    recorded_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "label": self.label,
            "note": self.note,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "recorded_at": self.recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


class RecordingWriter:
    """Appends landmark frames to a JSONL file."""

    def __init__(self, path: Path, header: Optional[RecordingHeader] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header = header or RecordingHeader()
        self._fh = self.path.open("w")
        self._fh.write(json.dumps(self._header.to_dict()) + "\n")
        self.frames = 0
        self._t0: Optional[float] = None

    def write(self, capture: CaptureResult) -> None:
        if self._t0 is None:
            self._t0 = capture.timestamp
        row: Dict[str, Any] = {"t": round(capture.timestamp - self._t0, 6)}
        if capture.landmarks is None:
            row["lm"] = None
        else:
            # Rounded to 6 dp: MediaPipe's precision is nowhere near that, and
            # it keeps takes about a third smaller.
            row["lm"] = [
                [round(float(p.x), 6), round(float(p.y), 6), round(float(p.z), 6)]
                for p in capture.landmarks
            ]
        self._fh.write(json.dumps(row) + "\n")
        self.frames += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> RecordingWriter:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    header: Dict[str, Any]
    frames: List[Dict[str, Any]]
    path: Optional[Path] = None

    @property
    def label(self) -> str:
        return self.header.get("label", "unlabelled")

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return float(self.frames[-1].get("t", 0.0))

    def __len__(self) -> int:
        return len(self.frames)


def read_recording(path: Path) -> Recording:
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: recording is empty")

    header = json.loads(lines[0])
    if header.get("schema") != RECORDING_SCHEMA:
        raise ValueError(
            f"{path}: unsupported recording schema {header.get('schema')!r} "
            f"(expected {RECORDING_SCHEMA})"
        )
    frames = [json.loads(ln) for ln in lines[1:]]
    return Recording(header=header, frames=frames, path=path)


def _to_capture(row: Dict[str, Any], t0: float, width: int,
                height: int) -> CaptureResult:
    lm = row.get("lm")
    landmarks = None
    if lm is not None:
        landmarks = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in lm]
    # A 1x1x3 stand-in: the replay path only ever reads frame.shape, and
    # allocating real frames would dominate replay time for no benefit.
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return CaptureResult(
        frame=frame,
        landmarks=landmarks,
        hand_lm_obj=None,
        timestamp=t0 + float(row.get("t", 0.0)),
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class VirtualClock:
    """A clock the replay drives, so cooldowns advance with recorded time.

    Every timing decision in the pipeline -- click cooldown, command cooldown,
    scroll rate limit, volume rate limit -- reads its clock through injection.
    Pointing them all at this makes a replay deterministic: the same take
    always produces the same actions, at the same recorded timestamps, no
    matter how fast the host machine actually is.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_to(self, t: float) -> None:
        self.now = max(self.now, t)


@dataclass
class ReplayResult:
    recording: Recording
    actions: List[Any]
    frames: int
    duration: float

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for action in self.actions:
            name = type(action).__name__
            out[name] = out.get(name, 0) + 1
        return out

    def discrete_actions(self) -> List[Any]:
        """Everything except cursor movement.

        Cursor motion is continuous and expected; a false-trigger count is
        about discrete events that fired when the user meant nothing.
        """
        return [a for a in self.actions
                if type(a).__name__ != "MoveCursor"]


def replay(recording: Recording, model: Any,
           cfg: AppConfig = DEFAULT_CONFIG) -> ReplayResult:
    """Push a recording through the real pipeline and collect the actions.

    Uses the same InferenceThread.process and GestureRouter.route the live app
    uses -- not a reimplementation -- so a replay proves something about the
    shipping code path rather than about a test double.
    """
    clock = VirtualClock()
    inference = InferenceThread(
        model=model,
        in_queue=None,
        out_queue=None,
        config=cfg,
        clock=clock,
    )
    router = GestureRouter(cfg, screen_w=1920, screen_h=1080)

    width = int(recording.header.get("frame_width", 640))
    height = int(recording.header.get("frame_height", 480))

    collected: List[Any] = []
    for row in recording.frames:
        clock.advance_to(float(row.get("t", 0.0)))
        capture = _to_capture(row, 0.0, width, height)
        result = inference.process(capture)
        collected.extend(router.route(result, now=clock.now))

    return ReplayResult(
        recording=recording,
        actions=collected,
        frames=len(recording.frames),
        duration=recording.duration,
    )


def false_trigger_report(results: List[ReplayResult]) -> Dict[str, Any]:
    """Summarize discrete actions fired over footage labelled 'no-intent'.

    Any action at all over a no-intent take is a false trigger by definition:
    the user was not trying to control anything.
    """
    total_seconds = sum(r.duration for r in results)
    total_frames = sum(r.frames for r in results)
    per_take = []
    total_false = 0

    for r in results:
        discrete = r.discrete_actions()
        total_false += len(discrete)
        per_take.append({
            "path": str(r.recording.path) if r.recording.path else None,
            "label": r.recording.label,
            "duration_s": round(r.duration, 2),
            "frames": r.frames,
            "false_triggers": len(discrete),
            "by_type": {
                k: v for k, v in r.counts().items() if k != "MoveCursor"
            },
        })

    rate_per_min = (
        total_false / (total_seconds / 60.0) if total_seconds > 0 else float("nan")
    )
    return {
        "schema": "gestureflow.false_trigger/1",
        "takes": per_take,
        "totals": {
            "takes": len(results),
            "duration_s": round(total_seconds, 2),
            "frames": total_frames,
            "false_triggers": total_false,
            "false_triggers_per_minute": round(rate_per_min, 3)
            if total_seconds > 0 else None,
        },
        "note": (
            "Counted over recordings the operator labelled 'no-intent'. The "
            "number describes those specific takes and the config they were "
            "replayed against; it is not a general property of the system."
        ),
    }
