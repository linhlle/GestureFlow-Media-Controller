from __future__ import annotations

import time

from enum import Enum, auto
from typing import Any

from gestureflow.config import ScrollConfig, DEFAULT_CONFIG

class ScrollState(Enum):
    IDLE            = auto()
    FIST_DETECTED   = auto()
    SCROLLING       = auto()

class ScrollFSM:
    def __init__(self, config: ScrollConfig | None = None) -> None:
        self._cfg = config or DEFAULT_CONFIG.scroll
        self._state: ScrollState = ScrollState.IDLE
        self._hold_frames: int = 0
        self._anchor_y: float = 0.0
        self._last_scroll_time: float= 0.0
        self._scroll_delta: int = 0


    def update(self, landmarks: Any | None) -> None:
        self._scroll_delta = 0
        if landmarks is None:
            self._reset()
            return 
        fist = _is_fist(landmarks)
        wrist_y: float = landmarks[0].y
        self._transition(fist, wrist_y)

    @property
    def scroll_delta(self) -> int:
        return self._scroll_delta
    
    @property
    def is_active(self) -> bool:
        return self._state in (ScrollState.FIST_DETECTED, ScrollState.SCROLLING)
    
    @property
    def state(self) -> ScrollState:
        return self._state
    
    # ------------------------------------------------------------------
    # FSM transitions
    # ------------------------------------------------------------------
    def _transition(self, fist: bool, wrist_y: float) -> None:
        cfg = self._cfg

        if self._state is ScrollState.IDLE:
            if fist:
                self._hold_frames = 1
                self._state = ScrollState.FIST_DETECTED

        elif self._state is ScrollState.FIST_DETECTED:
            if fist:
                self._hold_frames += 1
                if self._hold_frames >= cfg.min_hold_frames:
                    self._anchor_y = wrist_y
                    self._state = ScrollState.SCROLLING
            else:
                self._reset()

        elif self._state is ScrollState.SCROLLING:
            if not fist:
                self._reset()
                return
            
            now = time.monotonic()
            if now - self._last_scroll_time < cfg.cooldown:
                return
            
            delta_y = self._anchor_y - wrist_y
            if abs(delta_y) > cfg.sensitivity:
                clicks = int(delta_y / cfg.sensitivity) * cfg.step
                self._scroll_delta = clicks
                self._anchor_y = wrist_y          
                self._last_scroll_time = now


    def _reset(self) -> None:
        self._state = ScrollState.IDLE
        self._hold_frames = 0
        self._anchor_y = 0.0

    
_FINGERTIP_IDS = (8, 12, 16, 20)
_KNUCKLE_IDS = (5, 9, 13, 17)

def _is_fist(landmarks: Any, threshold: float = 0.02) -> bool:
    curled = 0
    for tip_id, knuckle_id in zip(_FINGERTIP_IDS, _KNUCKLE_IDS):
        if landmarks[tip_id].y > landmarks[knuckle_id].y + threshold:
            curled += 1
    return curled >= 3
 

