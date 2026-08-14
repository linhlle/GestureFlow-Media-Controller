"""Pinch-to-zoom on thumb and index spread.

The collision problem
---------------------
Thumb and index are already the left-click pinch. The same two landmarks cannot
mean two things unless the two meanings occupy disjoint ranges, so:

* the click FSM lives *below* its close threshold -- fingers touching;
* zoom lives *above* the open threshold -- fingers apart and being moved apart
  or together.

A pinch closed enough to click is, by construction, too closed to arm zoom. The
gap between the two thresholds is the same hysteresis band that already stops
the click chattering, so nothing new had to be invented to separate them.

Zoom also demands the other three fingers curled, which a click does not care
about, and it *arms* over several frames rather than firing instantly -- the
same shape as the scroll fist. Arming matters because the hand passes through
"thumb and index apart" on the way to almost every other gesture; requiring the
pose to persist is what stops zoom claiming those transits.

Once armed, ZOOM outranks CLICK and CURSOR in the mode ladder, so nothing else
can fire while the user is zooming.

An honest overlap
-----------------
The zoom pose is a wide L-shape, and L-Shape is one of the four classes the
model is trained on -- bound to Spotlight by default. Measured over the
recorded dataset, roughly a quarter of L-Shape frames also satisfy the zoom
pose at the shipped angle threshold.

That overlap is handled rather than eliminated: when the classifier settles on
L-Shape, COMMAND outranks ZOOM in the ladder and the geometric detectors are
parked, so Spotlight wins. Zoom only gets the frame when the classifier is *not*
confident it is L-Shape -- in practice, when the thumb is splayed wider than the
trained pose. Users who want zoom and Spotlight to feel cleanly separate should
rebind one of them, and the guide says so.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Callable, Optional

from gestureflow.config import DEFAULT_CONFIG, ZoomConfig
from gestureflow.scroll_fsm import (
    _INDEX_MCP,
    _INDEX_PIP,
    _INDEX_TIP,
    _MIDDLE_MCP,
    _MIDDLE_TIP,
    _PINKY_MCP,
    _PINKY_TIP,
    _RING_MCP,
    _RING_TIP,
    _THUMB_MCP,
    _THUMB_TIP,
    hand_scale,
    is_degenerate,
)


class ZoomState(Enum):
    IDLE = auto()
    ARMING = auto()      # pose present, not yet held long enough
    ZOOMING = auto()     # emitting on distance changes


def _spread(landmarks: Any) -> float:
    """Thumb-to-index distance, in hand-widths."""
    a = landmarks[_THUMB_TIP]
    b = landmarks[_INDEX_TIP]
    raw = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    return raw / hand_scale(landmarks)


def thumb_index_angle(landmarks: Any) -> float:
    """Angle in degrees between the thumb and the index finger.

    This is what separates a zoom pose from ordinary pointing, and the
    separation is not optional: a pointing hand has the thumb apart from the
    index too, so distance alone reads a plain cursor gesture as a zoom and
    steals the pointer. What actually distinguishes them is the *angle* -- an
    L-shape holds the thumb roughly perpendicular to the index, while a
    relaxed pointing hand keeps it near-parallel to the fingers.
    """
    tx = landmarks[_THUMB_TIP].x - landmarks[_THUMB_MCP].x
    ty = landmarks[_THUMB_TIP].y - landmarks[_THUMB_MCP].y
    ix = landmarks[_INDEX_TIP].x - landmarks[_INDEX_MCP].x
    iy = landmarks[_INDEX_TIP].y - landmarks[_INDEX_MCP].y

    tn = math.hypot(tx, ty)
    inn = math.hypot(ix, iy)
    if tn == 0.0 or inn == 0.0:
        return 0.0

    cosine = (tx * ix + ty * iy) / (tn * inn)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def zoom_pose(landmarks: Any, cfg: ZoomConfig) -> bool:
    """Thumb and index held apart at an angle, other three curled."""
    if is_degenerate(landmarks):
        return False

    scale = hand_scale(landmarks)
    curl = cfg.curl_margin * scale

    index_out = landmarks[_INDEX_TIP].y < landmarks[_INDEX_PIP].y
    others_curled = (
        landmarks[_MIDDLE_TIP].y > landmarks[_MIDDLE_MCP].y + curl
        and landmarks[_RING_TIP].y > landmarks[_RING_MCP].y + curl
        and landmarks[_PINKY_TIP].y > landmarks[_PINKY_MCP].y + curl
    )
    # Above the click's open threshold by construction: a pinch closed enough
    # to click can never arm zoom.
    apart = _spread(landmarks) > cfg.min_separation
    splayed = thumb_index_angle(landmarks) >= cfg.min_angle_degrees

    return index_out and others_curled and apart and splayed


class ZoomFSM:
    """Emits ``zoom_in`` / ``zoom_out`` as the thumb-index gap changes."""

    def __init__(
        self,
        config: Optional[ZoomConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.zoom
        self._clock = clock
        self._state = ZoomState.IDLE
        self._hold_frames = 0
        self._prev_spread = 0.0
        self._last_fire = -math.inf
        self._direction: Optional[str] = None

    # -- state -------------------------------------------------------------

    @property
    def direction(self) -> Optional[str]:
        """'in' or 'out' on the frame a zoom step fired, else None."""
        return self._direction

    @property
    def fired(self) -> bool:
        return self._direction is not None

    @property
    def is_active(self) -> bool:
        """Armed or zooming. The mode ladder reads this."""
        return self._state in (ZoomState.ARMING, ZoomState.ZOOMING)

    @property
    def state(self) -> ZoomState:
        return self._state

    # -- driving -----------------------------------------------------------

    def update(self, landmarks: Any | None) -> None:
        self._direction = None

        if not self._cfg.enabled or landmarks is None:
            self._reset()
            return

        if not zoom_pose(landmarks, self._cfg):
            self._reset()
            return

        spread = _spread(landmarks)

        if self._state is ZoomState.IDLE:
            self._state = ZoomState.ARMING
            self._hold_frames = 1
            self._prev_spread = spread
            return

        if self._state is ZoomState.ARMING:
            self._hold_frames += 1
            self._prev_spread = spread
            if self._hold_frames >= self._cfg.min_hold_frames:
                self._state = ZoomState.ZOOMING
            return

        now = self._clock()
        if now - self._last_fire < self._cfg.cooldown:
            self._prev_spread = spread
            return

        delta = spread - self._prev_spread
        self._prev_spread = spread

        if abs(delta) <= self._cfg.sensitivity:
            return

        self._direction = "in" if delta > 0 else "out"
        self._last_fire = now

    # -- internals ---------------------------------------------------------

    def _reset(self) -> None:
        self._state = ZoomState.IDLE
        self._hold_frames = 0
        self._prev_spread = 0.0
