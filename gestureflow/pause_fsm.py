"""Pause / resume kill switch.

A safety feature, so the design priority is *never fires by accident*, ahead of
*easy to fire*. A control system you cannot switch off is worse than one that
takes an extra half-second to switch off.

The pose
--------
Index and pinky extended, middle and ring curled -- "rock horns".

Not the open palm that first suggests itself, for a concrete reason: open palm
is the model's High-Five class, bound to Mission Control, and the debouncer's
command cooldown is 1.3 s. Holding an open palm for the 1.5 s a pause needs
would fire Mission Control once or twice on the way there.

Rock horns is not one of the four trained classes, so the classifier reads it as
Neutral and the geometric detectors stay live. It needs four simultaneous
conditions, which is why it does not happen while typing, reaching for a mug, or
gesturing at a colleague. And it uses no thumb landmark at all -- thumb geometry
has already caused one total silent failure in this codebase, and a kill switch
should not be built on the least reliable part of the hand.

Firing
------
The pose must be held continuously for `hold_seconds`. A single frame that
breaks the pose resets the timer to zero rather than pausing it, so a fumbled
approach cannot accumulate toward a toggle across several attempts.

After toggling, the pose must be fully released before it can toggle again.
Without that latch, holding rock horns for three seconds would toggle twice and
land back where it started.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Callable, Optional

from gestureflow.config import DEFAULT_CONFIG, PauseConfig
from gestureflow.scroll_fsm import (
    _INDEX_PIP,
    _INDEX_TIP,
    _MIDDLE_MCP,
    _MIDDLE_TIP,
    _RING_MCP,
    _RING_TIP,
    hand_scale,
)

_PINKY_PIP = 18
_PINKY_TIP = 20


class PauseState(Enum):
    IDLE = auto()        # pose absent
    HOLDING = auto()     # pose present, timer running
    LATCHED = auto()     # toggled; waiting for the pose to be released


def rock_horns(landmarks: Any, margin: float = 0.25) -> bool:
    """Index and pinky up, middle and ring down.

    Margins are fractions of hand scale, like every other threshold here, so
    the pose is recognized the same near and far from the camera.
    """
    scale = hand_scale(landmarks)
    up = margin * scale
    down = margin * 0.6 * scale

    index_up = landmarks[_INDEX_TIP].y < landmarks[_INDEX_PIP].y - up
    pinky_up = landmarks[_PINKY_TIP].y < landmarks[_PINKY_PIP].y - up
    middle_down = landmarks[_MIDDLE_TIP].y > landmarks[_MIDDLE_MCP].y + down
    ring_down = landmarks[_RING_TIP].y > landmarks[_RING_MCP].y + down

    return index_up and pinky_up and middle_down and ring_down


class PauseFSM:
    """Tracks the kill-switch pose and owns the paused/running flag."""

    def __init__(
        self,
        config: Optional[PauseConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.pause
        self._clock = clock
        self._state = PauseState.IDLE
        self._hold_started: float = math.inf
        self._paused = False
        self._toggled = False
        self._progress = 0.0

    # -- state -------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def toggled(self) -> bool:
        """True only on the frame the toggle happened."""
        return self._toggled

    @property
    def progress(self) -> float:
        """0..1 toward a toggle, for the HUD."""
        return self._progress

    @property
    def state(self) -> PauseState:
        return self._state

    def force(self, paused: bool) -> None:
        """Set the flag directly. For tests and for a future hotkey."""
        self._paused = paused
        self._state = PauseState.IDLE
        self._hold_started = math.inf
        self._progress = 0.0

    # -- driving -----------------------------------------------------------

    def update(self, landmarks: Any | None) -> None:
        self._toggled = False

        if not self._cfg.enabled:
            self._progress = 0.0
            return

        present = landmarks is not None and rock_horns(landmarks, self._cfg.margin)

        if not present:
            # A broken pose resets the timer rather than pausing it: a fumbled
            # approach must not accumulate toward a toggle across attempts.
            self._state = PauseState.IDLE
            self._hold_started = math.inf
            self._progress = 0.0
            return

        if self._state is PauseState.LATCHED:
            # Already toggled on this hold; wait for a release.
            self._progress = 1.0
            return

        now = self._clock()
        if self._state is not PauseState.HOLDING:
            self._state = PauseState.HOLDING
            self._hold_started = now

        held = now - self._hold_started
        hold_for = self._cfg.hold_seconds
        self._progress = 1.0 if hold_for <= 0 else min(1.0, held / hold_for)

        if held >= hold_for:
            self._paused = not self._paused
            self._toggled = True
            self._state = PauseState.LATCHED
            self._progress = 1.0
