"""Pipeline instrumentation.

Nothing in this repo measured itself before: `CaptureResult.timestamp` was
written on every frame and read by nobody, so there was no basis for any claim
about frame rate or latency.  This module supplies that basis.

What it records
---------------
* per-stage durations (frame grab, MediaPipe, normalize, predict, FSMs, dispatch)
* end-to-end capture -> action latency
* per-stage frame counts, from which FPS is derived
* queue-depth histograms at each hand-off
* dropped-frame counters at each hand-off

Latency is reported as p50/p95/p99, never as a mean.  For an interactive system
the mean hides exactly the stalls a user notices: a pipeline that is smooth 95%
of the time and hitches for 300 ms the rest still has a flattering mean.

Everything here is deliberately cheap -- a monotonic clock read and a deque
append per stage -- so leaving it on does not change what it measures.  It is
still opt-in: `MetricsRecorder(enabled=False)` compiles down to near no-ops.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

# Stage names used across the pipeline.  Kept as constants so a typo in one
# call site cannot silently create a second, separate histogram.
STAGE_GRAB = "capture.grab"
STAGE_MEDIAPIPE = "capture.mediapipe"
STAGE_NORMALIZE = "inference.normalize"
STAGE_PREDICT = "inference.predict"
STAGE_FSM = "inference.fsm"
STAGE_RENDER = "render.draw"
STAGE_DISPATCH = "action.dispatch"

HANDOFF_CAPTURE = "capture->inference"
HANDOFF_INFERENCE = "inference->render"
HANDOFF_ACTION = "render->action"

# Cap on retained samples per series.  At 30 FPS this is ~10 minutes of history,
# which is plenty for percentiles while keeping memory flat during long runs.
_MAX_SAMPLES = 20_000


def percentile(sorted_values: List[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list.

    Deliberately not interpolating: with sub-millisecond timings, interpolation
    invents values that were never observed.  Nearest-rank always returns a
    measurement that actually happened.
    """
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    # Standard nearest-rank: rank = ceil(q/100 * n), clamped into [1, n].
    rank = max(1, min(n, math.ceil(q / 100.0 * n)))
    return sorted_values[rank - 1]


@dataclass
class Series:
    """One named collection of duration samples, in seconds."""

    name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=_MAX_SAMPLES))
    total_observed: int = 0

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)
        self.total_observed += 1

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0}
        ordered = sorted(self.samples)
        ms = [v * 1000.0 for v in ordered]
        return {
            "count": self.total_observed,
            "retained": len(ordered),
            "p50_ms": round(percentile(ms, 50), 3),
            "p95_ms": round(percentile(ms, 95), 3),
            "p99_ms": round(percentile(ms, 99), 3),
            "min_ms": round(ms[0], 3),
            "max_ms": round(ms[-1], 3),
            "mean_ms": round(statistics.fmean(ms), 3),
        }


class MetricsRecorder:
    """Thread-safe collector for pipeline timings and counters.

    Shared across the capture, inference, render, and action threads, so every
    mutation takes a lock.  The lock is held for a deque append at most.
    """

    def __init__(
        self,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._series: Dict[str, Series] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._queue_depths: Dict[str, Counter] = defaultdict(Counter)
        self._stage_first_ts: Dict[str, float] = {}
        self._stage_last_ts: Dict[str, float] = {}
        self._started = self._clock()

    # -- recording ---------------------------------------------------------

    def record(self, stage: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            series = self._series.get(stage)
            if series is None:
                series = self._series[stage] = Series(stage)
            series.add(seconds)
            now = self._clock()
            self._stage_first_ts.setdefault(stage, now)
            self._stage_last_ts[stage] = now

    def timer(self, stage: str) -> "_Timer":
        """Context manager that records how long its block took.

        Usage::

            with metrics.timer(STAGE_PREDICT):
                probs = model.predict_proba([features])[0]
        """
        return _Timer(self, stage)

    def count(self, name: str, amount: int = 1) -> None:
        if not self.enabled or amount == 0:
            return
        with self._lock:
            self._counters[name] += amount

    def observe_queue(self, handoff: str, depth: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._queue_depths[handoff][depth] += 1

    def record_end_to_end(self, captured_at: float, now: Optional[float] = None) -> None:
        """Record capture -> action latency for one frame.

        `captured_at` is CaptureResult.timestamp, taken the instant the frame
        left the camera.  This is the number a user actually feels.
        """
        if not self.enabled:
            return
        end = self._clock() if now is None else now
        self.record("end_to_end", max(0.0, end - captured_at))

    # -- reporting ---------------------------------------------------------

    def elapsed(self) -> float:
        return self._clock() - self._started

    def fps(self, stage: str) -> float:
        """Throughput of one stage, over the window it was actually active."""
        with self._lock:
            series = self._series.get(stage)
            if series is None or series.total_observed < 2:
                return 0.0
            first = self._stage_first_ts.get(stage)
            last = self._stage_last_ts.get(stage)
        if first is None or last is None or last <= first:
            return 0.0
        return (series.total_observed - 1) / (last - first)

    def snapshot(self) -> Dict:
        with self._lock:
            stages = {name: s.summary() for name, s in self._series.items()}
            counters = dict(self._counters)
            depths = {
                handoff: {str(k): v for k, v in sorted(hist.items())}
                for handoff, hist in self._queue_depths.items()
            }
            stage_names = list(self._series)

        return {
            "elapsed_s": round(self.elapsed(), 3),
            "stages": stages,
            "fps": {name: round(self.fps(name), 2) for name in stage_names},
            "counters": counters,
            "queue_depth_histogram": depths,
        }

    def report(self, extra: Optional[Dict] = None) -> Dict:
        data = self.snapshot()
        data["schema"] = "gestureflow.metrics/1"
        data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data["note"] = (
            "All timings measured on the machine that produced this file. "
            "They describe that machine's camera, CPU, and load at that moment "
            "and do not generalize to other hardware."
        )
        if extra:
            data.update(extra)
        return data

    def write_json(self, path: Path, extra: Optional[Dict] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.report(extra), f, indent=2)
        return path

    def format_text(self) -> str:
        """Human-readable summary for stdout at the end of a bench run."""
        snap = self.snapshot()
        lines = [
            f"Elapsed: {snap['elapsed_s']:.1f}s",
            "",
            f"{'stage':<24}{'n':>7}{'p50':>10}{'p95':>10}{'p99':>10}{'fps':>9}",
            "-" * 70,
        ]
        for name in sorted(snap["stages"]):
            s = snap["stages"][name]
            if not s.get("count"):
                continue
            fps = snap["fps"].get(name, 0.0)
            lines.append(
                f"{name:<24}{s['count']:>7}"
                f"{s['p50_ms']:>9.2f}m{s['p95_ms']:>9.2f}m{s['p99_ms']:>9.2f}m"
                f"{fps:>9.1f}"
            )

        if snap["counters"]:
            lines += ["", "counters:"]
            for name in sorted(snap["counters"]):
                lines.append(f"  {name:<40}{snap['counters'][name]:>8}")

        if snap["queue_depth_histogram"]:
            lines += ["", "queue depth (depth: observations):"]
            for handoff in sorted(snap["queue_depth_histogram"]):
                hist = snap["queue_depth_histogram"][handoff]
                rendered = "  ".join(f"{k}:{v}" for k, v in hist.items())
                lines.append(f"  {handoff:<24}{rendered}")

        return "\n".join(lines)


class _Timer:
    __slots__ = ("_recorder", "_stage", "_start")

    def __init__(self, recorder: MetricsRecorder, stage: str) -> None:
        self._recorder = recorder
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = self._recorder._clock()
        return self

    def __exit__(self, *exc) -> bool:
        self._recorder.record(self._stage, self._recorder._clock() - self._start)
        return False


class NullMetrics(MetricsRecorder):
    """A recorder that measures nothing, for callers that do not want overhead."""

    def __init__(self) -> None:
        super().__init__(enabled=False)
