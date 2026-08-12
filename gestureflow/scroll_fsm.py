from __future__ import annotations

import time
import math

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
        self._prev_wrist_y: float = 0.0
        # See ClickFSM._last_click_time: monotonic()'s epoch is undefined.
        self._last_scroll_time: float = -math.inf
        self._scroll_delta: int = 0


    def update(self, landmarks: Any | None) -> None:
        self._scroll_delta = 0
        if landmarks is None:
            self._reset()
            return 
        fist = _is_true_scroll_fist(landmarks)
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
                self._prev_wrist_y = wrist_y
                self._state = ScrollState.FIST_DETECTED

        elif self._state is ScrollState.FIST_DETECTED:
            if fist:
                self._hold_frames += 1
                if self._hold_frames >= cfg.min_hold_frames:
                    self._prev_wrist_y = wrist_y
                    self._state = ScrollState.SCROLLING
            else:
                self._reset()

        elif self._state is ScrollState.SCROLLING:
            if not fist:
                self._reset()
                return
            
            now = time.monotonic()
            if now - self._last_scroll_time < cfg.cooldown:
                self._prev_wrist_y = wrist_y
                return
            
            velocity = self._prev_wrist_y - wrist_y
            self._prev_wrist_y = wrist_y
            
            if abs(velocity) > cfg.sensitivity:
                clicks = _velocity_to_clicks(velocity, cfg)
                if clicks != 0:
                    self._scroll_delta = clicks
                    self._last_scroll_time = now


    def _reset(self) -> None:
        self._state = ScrollState.IDLE
        self._hold_frames = 0
        self._prev_wrist_y = 0.0


# Landmark coordinates arrive as float32-ish values, so a wrist that moved
# exactly 2 * sensitivity computes as 2.0000000000000018 rather than 2.0.
# ceil() then rounds that residue up to a whole extra click -- a 50% overshoot
# on small movements.  Quantizing first keeps ceil() meaning "round up a real
# fraction" instead of "round up floating-point noise".
_CLICK_PRECISION = 9


def _velocity_to_clicks(velocity: float, cfg: ScrollConfig) -> int:
    """Map wrist velocity to a signed scroll-click count."""
    ratio = round(velocity / cfg.sensitivity, _CLICK_PRECISION)
    magnitude = round(abs(ratio) ** cfg.velocity_exponent, _CLICK_PRECISION)
    return int(math.copysign(math.ceil(magnitude), ratio)) * cfg.step



_INDEX_TIP   = 8    
_INDEX_PIP   = 6   
_THUMB_TIP   = 4    
_THUMB_MCP   = 2    
_MIDDLE_TIP  = 12  
_MIDDLE_MCP  = 9    
_RING_TIP    = 16   
_RING_MCP    = 13  
_PINKY_TIP   = 20   
_PINKY_MCP   = 17  
 
_CURL_PAIRS = (
    (_INDEX_TIP,  5),   
    (_MIDDLE_TIP, 9),   
    (_RING_TIP,   13),  
    (_PINKY_TIP,  17),  
)
 
def _index_extended(landmarks: Any, margin: float = 0.04) -> bool:
    return landmarks[_INDEX_TIP].y < landmarks[_INDEX_PIP].y - margin

def _thumb_raised(landmarks: Any, margin: float = 0.04) -> bool:
    return landmarks[_THUMB_TIP].y < landmarks[_THUMB_MCP].y - margin


def _strict_fist(landmarks: Any, threshold: float = 0.03) -> bool:
    curled = 0
    for tip_id, knuckle_id in _CURL_PAIRS:
        if landmarks[tip_id].y > landmarks[knuckle_id].y + threshold:
            curled += 1
    return curled == 4
 
def _is_true_scroll_fist(landmarks: Any) -> bool:
    if _index_extended(landmarks):
        return False
    if _thumb_raised(landmarks):
        return False
    return _strict_fist(landmarks)
