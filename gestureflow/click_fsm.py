from __future__ import annotations

import time
import math

from enum import Enum, auto
from typing import Union, Any

from gestureflow.config import DEFAULT_CONFIG, ClickConfig, RightClickConfig

PinchConfig = Union[ClickConfig, RightClickConfig]

class ClickState(Enum):
    IDLE        = auto()
    PRESSING    = auto()
    HELD        = auto()
    RELEASING   = auto()


class ClickFSM:
    def __init__(
            self,
            config: PinchConfig | None = None,
            landmark_a: int = 4,
            landmark_b: int = 8,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.click
        self._lm_a = landmark_a
        self._lm_b = landmark_b
        self._state: ClickState = ClickState.IDLE
        self._hold_frames: int = 0
        self._click_fired: bool = False
        self._last_click_time: float = 0.0

    
    def update(self, landmarks: Any | None) -> None:
        self._click_fired = False
        if landmarks is None:
            self._reset()
            return
        dist = _pinch_distance(landmarks, self._lm_a, self._lm_b)
        self._transition(dist)

    @property
    def click_fired(self) -> bool:
        return self._click_fired
    
    @property 
    def is_active(self) -> bool:
        return self._state in (ClickState.PRESSING, ClickState.HELD)
    
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
            else:
                self._reset()

        elif self._state is ClickState.HELD:
            if dist > cfg.open_threshold:
                self._state = ClickState.RELEASING
                now = time.monotonic()
                if now - self._last_click_time >= cfg.cooldown:
                    self._click_fired = True
                    self._last_click_time = now
                self._reset()


    def _reset(self) -> None:
        self._state = ClickState.IDLE
        self._hold_frames = 0


def _pinch_distance(landmarks: Any, lm_a: int, lm_b: int) -> float:
    a = landmarks[lm_a]
    b = landmarks[lm_b]
    return math.sqrt(
        (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2
    )

