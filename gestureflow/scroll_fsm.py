from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Callable

from gestureflow.config import DEFAULT_CONFIG, ScrollConfig


class ScrollState(Enum):
    IDLE            = auto()
    FIST_DETECTED   = auto()
    SCROLLING       = auto()

class ScrollFSM:
    def __init__(
        self,
        config: ScrollConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.scroll
        self._clock = clock
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
        # The wrist position stays in raw units; the *velocity* derived from it
        # is what gets scaled, in _transition. Dividing the position by hand
        # scale would couple the two, since hand scale is itself measured from
        # the wrist.
        self._transition(fist, landmarks[0].y, hand_scale(landmarks))

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
    def _transition(self, fist: bool, wrist_y: float, scale: float = 1.0) -> None:
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

            now = self._clock()
            if now - self._last_scroll_time < cfg.cooldown:
                self._prev_wrist_y = wrist_y
                return

            # Measured in hand-widths per frame, so the same physical hand
            # movement scrolls the same amount whether the user is close to the
            # camera or far from it.
            velocity = (self._prev_wrist_y - wrist_y) / scale
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



_WRIST       = 0
_INDEX_TIP   = 8
_INDEX_PIP   = 6
_THUMB_TIP   = 4
_THUMB_IP    = 3
_THUMB_MCP   = 2
_INDEX_MCP   = 5
_MIDDLE_TIP  = 12
_MIDDLE_MCP  = 9
_RING_TIP    = 16
_RING_MCP    = 13
_PINKY_TIP   = 20
_PINKY_MCP   = 17

_CURL_PAIRS = (
    (_INDEX_TIP,  _INDEX_MCP),
    (_MIDDLE_TIP, _MIDDLE_MCP),
    (_RING_TIP,   _RING_MCP),
    (_PINKY_TIP,  _PINKY_MCP),
)

# Margins and thresholds below are ratios of the hand's own size, not absolute
# distances.  MediaPipe reports landmarks in image-normalized coordinates, so a
# hand twice as far from the camera produces every distance at half the size.
# Absolute thresholds therefore encode "one particular hand at one particular
# distance", which is why a fist used to register as a right-click: the
# fingertips of a curled hand fall inside a fixed 0.045 radius even though,
# relative to that hand, they are no closer than they ever were.
_MIN_HAND_SCALE = 1e-6


def hand_scale(landmarks: Any) -> float:
    """Distance from the wrist to the middle-finger MCP.

    The reference length everything else is measured against.  It is chosen
    because it spans the palm, which is rigid: unlike a fingertip span it does
    not change when the hand opens, closes, or points, so it stays a stable
    unit across every pose.
    """
    a = landmarks[_WRIST]
    b = landmarks[_MIDDLE_MCP]
    scale = math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )
    return scale if scale > _MIN_HAND_SCALE else _MIN_HAND_SCALE


def _index_extended(landmarks: Any, margin: float = 0.25) -> bool:
    """Index fingertip clearly above its own PIP joint."""
    return (landmarks[_INDEX_TIP].y
            < landmarks[_INDEX_PIP].y - margin * hand_scale(landmarks))


def _thumb_raised(landmarks: Any, margin: float = 0.25) -> bool:
    """A straight thumb pointing up and clear of the hand.

    The previous form compared the thumb tip against its own MCP and nothing
    else, which is true for almost any posture that is not actively pointing
    the thumb downward.  Critically it is true for a closed fist, where the
    thumb folds across the curled fingers and its tip ends up above its own
    knuckle -- measured at 79% of Neutral frames and 89% of genuine fists,
    which is what made scroll impossible to trigger.

    Requiring the thumb to be *straight* (tip above IP above MCP) and *clear of
    the hand* (tip above the index knuckle) distinguishes a deliberate
    thumbs-up from a thumb tucked into a fist.
    """
    scale = hand_scale(landmarks)
    tip, ip, mcp = landmarks[_THUMB_TIP], landmarks[_THUMB_IP], landmarks[_THUMB_MCP]

    straight = (tip.y < ip.y - margin * 0.5 * scale
                and ip.y < mcp.y - margin * 0.3 * scale)
    clear_of_hand = tip.y < landmarks[_INDEX_MCP].y - margin * scale
    return straight and clear_of_hand


def _strict_fist(landmarks: Any, threshold: float = 0.19) -> bool:
    """All four non-thumb fingertips curled below their knuckles."""
    limit = threshold * hand_scale(landmarks)
    curled = 0
    for tip_id, knuckle_id in _CURL_PAIRS:
        if landmarks[tip_id].y > landmarks[knuckle_id].y + limit:
            curled += 1
    return curled == 4


def _is_true_scroll_fist(landmarks: Any) -> bool:
    """A fist held for scrolling, as opposed to any other closed-ish hand."""
    if _index_extended(landmarks):
        return False
    if _thumb_raised(landmarks):
        return False
    return _strict_fist(landmarks)
