from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Union

from gestureflow.config import (
    DEFAULT_CONFIG,
    ClickConfig,
    DragConfig,
    RightClickConfig,
)
from gestureflow.scroll_fsm import hand_scale, is_degenerate

PinchConfig = Union[ClickConfig, RightClickConfig]

class ClickState(Enum):
    IDLE        = auto()
    PRESSING    = auto()
    HELD        = auto()
    DRAGGING    = auto()   # held past the drag threshold; button is down


class ClickFSM:
    def __init__(
            self,
            config: PinchConfig | None = None,
            landmark_a: int = 4,
            landmark_b: int = 8,
            clock: Callable[[], float] = time.monotonic,
            drag: Optional[DragConfig] = None,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.click
        # Drag is opt-in per FSM: only the left pinch drags. Giving the
        # right-click pinch a drag state would mean a long right-pinch holds
        # the context menu button down, which is not a gesture anyone wants.
        self._drag = drag
        self._lm_a = landmark_a
        self._lm_b = landmark_b
        # Injected so tests and replays drive time explicitly instead of
        # sleeping, and so a recorded session replays at its original cadence.
        self._clock = clock
        self._state: ClickState = ClickState.IDLE
        self._hold_frames: int = 0
        self._click_fired: bool = False
        self._drag_started: bool = False
        self._drag_ended: bool = False
        self._held_since: float = math.inf
        # -inf, not 0.0: time.monotonic()'s reference point is undefined, and on
        # macOS it reads near zero at process start.  Seeding with 0.0 made
        # "now - last >= cooldown" false for the first cooldown seconds of the
        # process, silently swallowing every click made right after launch.
        self._last_click_time: float = -math.inf


    def update(self, landmarks: Any | None) -> None:
        self._click_fired = False
        self._drag_started = False
        self._drag_ended = False
        if landmarks is None or is_degenerate(landmarks):
            # A hand leaving frame mid-drag must still release the button.
            # Otherwise walking away from the camera leaves the mouse held down
            # with no way to let go except quitting.
            if self._state is ClickState.DRAGGING:
                self._drag_ended = True
            self._reset()
            return
        # Measured in hand-widths, not image units: a hand further from the
        # camera produces smaller raw distances, so an absolute threshold
        # silently gets easier to cross as the user leans back.
        dist = _pinch_distance(landmarks, self._lm_a, self._lm_b) / hand_scale(landmarks)
        self._transition(dist)

    @property
    def click_fired(self) -> bool:
        return self._click_fired

    @property
    def drag_started(self) -> bool:
        """True on the frame the button went down. Emit a press."""
        return self._drag_started

    @property
    def drag_ended(self) -> bool:
        """True on the frame the button came back up. Emit a release."""
        return self._drag_ended

    @property
    def dragging(self) -> bool:
        return self._state is ClickState.DRAGGING

    @property
    def is_active(self) -> bool:
        return self._state in (ClickState.PRESSING, ClickState.HELD,
                               ClickState.DRAGGING)

    @property
    def drag_progress(self) -> float:
        """0..1 from reaching HELD toward starting a drag, for the HUD."""
        if self._drag is None or not self._drag.enabled:
            return 0.0
        if self._state is ClickState.DRAGGING:
            return 1.0
        if self._state is not ClickState.HELD or self._held_since == math.inf:
            return 0.0
        hold_for = self._drag.hold_seconds
        if hold_for <= 0:
            return 1.0
        return min(1.0, (self._clock() - self._held_since) / hold_for)

    @property
    def state(self) -> ClickState:
        return self._state

    @property
    def hold_progress(self) -> float:
        if self._cfg.min_hold_frames == 0:
            return 1.0
        return min(1.0, self._hold_frames / self._cfg.min_hold_frames)


    # ------------------------------------------------------------------
    # FSM transitions
    # ------------------------------------------------------------------

    def _transition(self, dist: float) -> None:
        cfg = self._cfg

        if self._state is ClickState.IDLE:
            if dist < cfg.close_threshold:
                self._hold_frames = 1
                self._state = ClickState.PRESSING

        elif self._state is ClickState.PRESSING:
            if dist < cfg.close_threshold:
                self._hold_frames += 1
                if self._hold_frames >= cfg.min_hold_frames:
                    self._state = ClickState.HELD
                    self._held_since = self._clock()
            else:
                self._reset()

        elif self._state is ClickState.HELD:
            # The click fires on the release edge, never on the press, so a held
            # pinch does not auto-repeat.
            if dist > cfg.open_threshold:
                now = self._clock()
                if now - self._last_click_time >= cfg.cooldown:
                    self._click_fired = True
                    self._last_click_time = now
                self._reset()
            elif self._drag_due():
                # Holding past the threshold turns the click into a press that
                # stays down. The click path above is untouched: a pinch
                # released before this point is still exactly a click.
                self._state = ClickState.DRAGGING
                self._drag_started = True

        elif self._state is ClickState.DRAGGING:
            if dist > cfg.open_threshold:
                self._drag_ended = True
                self._reset()

    def _drag_due(self) -> bool:
        if self._drag is None or not self._drag.enabled:
            return False
        if self._held_since == math.inf:
            return False
        return (self._clock() - self._held_since) >= self._drag.hold_seconds

    def _reset(self) -> None:
        self._state = ClickState.IDLE
        self._hold_frames = 0
        self._held_since = math.inf


def _pinch_distance(landmarks: Any, lm_a: int, lm_b: int) -> float:
    """Raw euclidean distance between two landmarks, in image units."""
    a = landmarks[lm_a]
    b = landmarks[lm_b]
    return math.sqrt(
        (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2
    )

