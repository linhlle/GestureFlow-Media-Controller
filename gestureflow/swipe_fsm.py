"""Horizontal swipe on a held fist.

Why the same pose as scroll
---------------------------
Scroll and swipe share the fist and are separated by which way the hand moves.
The alternative was a second whole-hand pose, and the ones left over are either
trained classes the classifier would claim first, or too close to a fist to
separate reliably.

Sharing the pose also makes the exclusivity structural. A single dominance rule
decides whether a given motion is vertical or horizontal, and the two detectors
read opposite sides of it, so one movement can never be both. Two independent
detectors on two poses would need to agree about which one you meant, and a
diagonal drag would fire both.

The dominance rule has a deliberate gap. Scroll wants vertical to at least tie;
swipe wants horizontal to win clearly. Motion in between is genuinely ambiguous,
and firing nothing is better than guessing -- a page that scrolls while you
meant to change desktop is worse than one that does nothing and lets you try
again.

One swipe, one fire
-------------------
A swipe is a flick, and a flick spans many frames above the velocity threshold.
Emitting per frame would fire a dozen desktop switches. So after firing, the
hand has to slow below a release threshold before another swipe can start --
hysteresis in the velocity domain, the same shape as the pinch thresholds.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Callable, Optional

from gestureflow.config import DEFAULT_CONFIG, SwipeConfig
from gestureflow.scroll_fsm import _is_true_scroll_fist, hand_scale


class SwipeState(Enum):
    IDLE = auto()          # no fist
    ARMED = auto()         # fist held, watching for a flick
    COOLING = auto()       # just fired; waiting for the hand to slow down


def vertical_dominates(vx: float, vy: float, ratio: float = 1.0) -> bool:
    """Scroll's side of the arbitration: vertical at least ties."""
    return abs(vy) >= ratio * abs(vx)


def horizontal_dominates(vx: float, vy: float, ratio: float = 1.5) -> bool:
    """Swipe's side: horizontal has to win clearly, not just edge ahead."""
    return abs(vx) > ratio * abs(vy)


class SwipeFSM:
    """Fires ``swipe_left`` / ``swipe_right`` once per horizontal flick."""

    def __init__(
        self,
        config: Optional[SwipeConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.swipe
        self._clock = clock
        self._state = SwipeState.IDLE
        self._hold_frames = 0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._have_prev = False
        self._last_fire = -math.inf
        self._direction: Optional[str] = None

    # -- state -------------------------------------------------------------

    @property
    def direction(self) -> Optional[str]:
        """'left' or 'right' on the frame a swipe fired, else None."""
        return self._direction

    @property
    def fired(self) -> bool:
        return self._direction is not None

    @property
    def state(self) -> SwipeState:
        return self._state

    @property
    def is_armed(self) -> bool:
        return self._state in (SwipeState.ARMED, SwipeState.COOLING)

    # -- driving -----------------------------------------------------------

    def update(self, landmarks: Any | None) -> None:
        self._direction = None

        if not self._cfg.enabled or landmarks is None:
            self._reset()
            return

        if not _is_true_scroll_fist(landmarks):
            self._reset()
            return

        scale = hand_scale(landmarks)
        x = landmarks[0].x
        y = landmarks[0].y

        if self._state is SwipeState.IDLE:
            self._state = SwipeState.ARMED
            self._hold_frames = 1
            self._remember(x, y)
            return

        vx = (x - self._prev_x) / scale if self._have_prev else 0.0
        vy = (y - self._prev_y) / scale if self._have_prev else 0.0
        self._remember(x, y)

        speed = abs(vx)

        if self._state is SwipeState.COOLING:
            # Wait for the flick to finish before another can start.
            if speed < self._cfg.sensitivity * self._cfg.release_ratio:
                self._state = SwipeState.ARMED
            return

        self._hold_frames += 1
        if self._hold_frames < self._cfg.min_hold_frames:
            return

        now = self._clock()
        if now - self._last_fire < self._cfg.cooldown:
            return

        if speed <= self._cfg.sensitivity:
            return
        if not horizontal_dominates(vx, vy, self._cfg.axis_ratio):
            return

        # Landmark x grows to the right in the mirrored frame the pipeline
        # feeds it, so a positive vx is a rightward hand movement.
        self._direction = "right" if vx > 0 else "left"
        self._last_fire = now
        self._state = SwipeState.COOLING

    # -- internals ---------------------------------------------------------

    def _remember(self, x: float, y: float) -> None:
        self._prev_x = x
        self._prev_y = y
        self._have_prev = True

    def _reset(self) -> None:
        self._state = SwipeState.IDLE
        self._hold_frames = 0
        self._have_prev = False
