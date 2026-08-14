"""Per-user calibration.

Thresholds are already fractions of hand scale, which fixed *distance from the
camera*: a hand twice as far away produces half the raw distances, and dividing
by hand scale cancels that out.

What it did not fix is that hands differ from each other. One person's fully
closed pinch measures 0.20 of their hand scale; another's measures 0.35, because
finger length relative to palm width is not the same across people. A single
shipped default cannot suit both, and the person it does not suit experiences it
as "clicking is unreliable" with no way to tell why.

Calibration measures the two states directly -- pinch closed, hand open -- and
puts the thresholds in the gap between them, keeping the hysteresis band that
stops the click chattering.

Failure is always non-fatal
---------------------------
A missing, unreadable, or nonsensical calibration file falls back to the shipped
defaults. Calibration is a refinement; it must never be the reason the app will
not start. That is also why derivation refuses to write a file it cannot justify
-- too few samples, or an open hand that measured tighter than the pinch, means
the recording went wrong, and silently encoding that would be worse than having
no calibration at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CALIBRATION_SCHEMA = "gestureflow.calibration/1"
CALIBRATION_PATH = Path.home() / ".gestureflow" / "calibration.json"

# Fewest samples per state worth trusting. At 30 FPS this is well under a
# second, so it is a floor against an aborted recording, not a real ask.
MIN_SAMPLES = 15


class CalibrationError(ValueError):
    """Raised when recorded samples cannot produce sensible thresholds."""


@dataclass
class Calibration:
    """Per-user thresholds, all as fractions of that user's hand scale."""

    click_close: float
    click_open: float
    right_click_close: float
    right_click_open: float
    hand_scale: float = 0.0
    samples: Dict[str, int] = field(default_factory=dict)
    schema: str = CALIBRATION_SCHEMA

    def to_dict(self) -> Dict:
        return {
            "schema": self.schema,
            "click_close": round(self.click_close, 4),
            "click_open": round(self.click_open, 4),
            "right_click_close": round(self.right_click_close, 4),
            "right_click_open": round(self.right_click_open, 4),
            "hand_scale": round(self.hand_scale, 4),
            "samples": self.samples,
            "note": (
                "Written by 'gestureflow calibrate'. Thresholds are fractions "
                "of your hand scale (wrist to middle knuckle). Delete this "
                "file to go back to the shipped defaults."
            ),
        }

    def write(self, path: Optional[Path] = None) -> Path:
        target = Path(path or CALIBRATION_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target


def derive(
    closed: Sequence[float],
    opened: Sequence[float],
    right_closed: Optional[Sequence[float]] = None,
    right_opened: Optional[Sequence[float]] = None,
    hand_scale: float = 0.0,
) -> Calibration:
    """Turn recorded pinch distances into thresholds.

    Every sequence is thumb-index (or middle-index) distance already divided by
    hand scale, so everything here is in hand-widths.

    The close threshold sits a little above the closed distribution and the open
    threshold a little below the open one, which leaves the dead band between
    them wide -- the whole point of the exercise, since that band is what stops
    a hand hovering at the boundary chattering clicks.
    """
    closed_p = _percentile(closed, 90, "closed pinch")
    opened_p = _percentile(opened, 10, "open hand")

    if opened_p <= closed_p:
        raise CalibrationError(
            f"The open hand measured no wider than the pinch "
            f"({opened_p:.3f} vs {closed_p:.3f} hand-widths). That usually "
            f"means the two recordings captured the same pose. Run it again "
            f"and make sure your fingers are fully apart for the open step."
        )

    close_t, open_t = _thresholds(closed_p, opened_p)

    if right_closed and right_opened:
        r_closed_p = _percentile(right_closed, 90, "closed middle-finger pinch")
        r_opened_p = _percentile(right_opened, 10, "open hand")
        if r_opened_p <= r_closed_p:
            raise CalibrationError(
                "The middle-finger pinch samples overlap the open-hand ones."
            )
        r_close_t, r_open_t = _thresholds(r_closed_p, r_opened_p)
    else:
        # Right-click wants a deliberate touch, since middle and index sit close
        # together in any curled hand. Same proportional gap as the shipped
        # defaults keep between the two.
        r_close_t = close_t * (0.22 / 0.28)
        r_open_t = open_t * (0.38 / 0.41)

    return Calibration(
        click_close=close_t,
        click_open=open_t,
        right_click_close=r_close_t,
        right_click_open=r_open_t,
        hand_scale=hand_scale,
        samples={
            "closed": len(closed),
            "opened": len(opened),
            "right_closed": len(right_closed or []),
            "right_opened": len(right_opened or []),
        },
    )


def _thresholds(closed_p: float, opened_p: float) -> tuple:
    """Place close and open inside the measured gap.

    A third of the way up for close, two thirds for open. Splitting it evenly
    would leave no dead band; putting them at the extremes would make the
    gesture need a wider range of motion than the user demonstrated.
    """
    span = opened_p - closed_p
    return closed_p + span * 0.33, closed_p + span * 0.66


def _percentile(values: Sequence[float], q: float, label: str) -> float:
    if len(values) < MIN_SAMPLES:
        raise CalibrationError(
            f"Only {len(values)} {label} sample(s); at least {MIN_SAMPLES} are "
            f"needed. Hold the pose steadily while it records."
        )
    ordered = sorted(values)
    # Nearest-rank, matching metrics.py: with this few samples an interpolated
    # value would be inventing a measurement that never happened.
    rank = max(1, min(len(ordered), int(round(q / 100.0 * len(ordered)))))
    return ordered[rank - 1]


def load(path: Optional[Path] = None) -> Optional[Calibration]:
    """Read a calibration file, or return None if there is not a usable one.

    Never raises. A corrupt file is the same situation as no file: fall back to
    the shipped defaults rather than refusing to start.
    """
    target = Path(path or CALIBRATION_PATH)
    if not target.exists():
        return None

    try:
        raw = json.loads(target.read_text())
    except (OSError, ValueError):
        print(f"[calibration] Ignoring unreadable {target}")
        return None

    if not isinstance(raw, dict) or raw.get("schema") != CALIBRATION_SCHEMA:
        print(f"[calibration] Ignoring {target}: unrecognized format")
        return None

    try:
        values = {k: float(raw[k]) for k in
                  ("click_close", "click_open",
                   "right_click_close", "right_click_open")}
    except (KeyError, TypeError, ValueError):
        print(f"[calibration] Ignoring {target}: missing or malformed values")
        return None

    if not _sane(values):
        print(f"[calibration] Ignoring {target}: values are out of range")
        return None

    return Calibration(
        click_close=values["click_close"],
        click_open=values["click_open"],
        right_click_close=values["right_click_close"],
        right_click_open=values["right_click_open"],
        hand_scale=float(raw.get("hand_scale", 0.0) or 0.0),
        samples=raw.get("samples", {}) if isinstance(raw.get("samples"), dict) else {},
    )


def _sane(values: Dict[str, float]) -> bool:
    """Reject values that would make a gesture impossible to perform.

    A close threshold of zero means the click can never trigger; one above the
    open threshold inverts the hysteresis and makes the FSM chatter. Neither is
    something to discover at runtime.
    """
    for value in values.values():
        if not (0.0 < value < 3.0):
            return False
    return (values["click_close"] < values["click_open"]
            and values["right_click_close"] < values["right_click_open"])


def summarize(calibration: Optional[Calibration]) -> List[str]:
    """Human-readable lines for the CLI."""
    if calibration is None:
        return ["No calibration found; using the shipped defaults."]
    return [
        f"click       close {calibration.click_close:.3f}  "
        f"open {calibration.click_open:.3f}",
        f"right click close {calibration.right_click_close:.3f}  "
        f"open {calibration.right_click_open:.3f}",
        f"hand scale  {calibration.hand_scale:.3f}",
    ]
