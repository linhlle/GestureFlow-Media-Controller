"""Dwell click: hold the pointer still to click.

An accessibility feature. Pinching needs fine motor control in two fingers
simultaneously; holding a pointer roughly still needs neither, so this is the
path for users who cannot pinch reliably -- or at all.

**Off by default**, because when it is on, resting the pointer somewhere clicks
there. That is the intended behaviour and it is also deeply surprising if you
did not ask for it, so it stays opt-in.

It watches the *screen target*, not the hand. Two reasons: the target is what
the user is actually aiming, already smoothed and mapped through the same
filter the pointer uses, so the radius is in the pixels the user sees rather
than in hand-widths. And it means the dwell radius means the same thing however
far from the camera the hand is, without a second scale conversion.

The re-arm rule mirrors the pinch hysteresis: after firing, the pointer must
leave the radius before another dwell can start. Without it, resting a hand
somewhere would click once per dwell period, forever.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from gestureflow.config import DEFAULT_CONFIG, DwellConfig


class DwellFSM:
    """Fires a click when the pointer target stays put long enough."""

    def __init__(self, config: Optional[DwellConfig] = None) -> None:
        # No internal clock. The router already knows what time the frame is
        # and threads it through route(); a second clock here would be a second
        # source of truth, and it would silently ignore an injected one -- which
        # is exactly how the first version failed under a virtual clock.
        self._cfg = config or DEFAULT_CONFIG.dwell
        self._now = 0.0
        self._anchor: Optional[Tuple[float, float]] = None
        self._since: float = math.inf
        self._latched = False
        self._fired = False

    # -- state -------------------------------------------------------------

    @property
    def fired(self) -> bool:
        """True only on the frame the dwell completed."""
        return self._fired

    @property
    def progress(self) -> float:
        """0..1 toward a click, for the HUD ring."""
        if not self._cfg.enabled or self._anchor is None or self._latched:
            return 0.0
        if self._since == math.inf:
            return 0.0
        seconds = self._cfg.seconds
        if seconds <= 0:
            return 1.0
        return min(1.0, (self._now - self._since) / seconds)

    @property
    def anchor(self) -> Optional[Tuple[float, float]]:
        return self._anchor

    # -- driving -----------------------------------------------------------

    def update(self, target: Optional[Tuple[float, float]],
               now: float) -> None:
        """Feed the current pointer target, or None when not pointing."""
        self._fired = False
        self._now = now

        if not self._cfg.enabled or target is None:
            self._reset()
            return

        if self._anchor is None:
            self._anchor = target
            self._since = now
            return

        if self._distance(target, self._anchor) > self._cfg.radius_px:
            # Moved on. Re-anchor here and start counting again -- and clear
            # the latch, since leaving the radius is what re-arms a dwell.
            self._anchor = target
            self._since = now
            self._latched = False
            return

        if self._latched:
            return

        if now - self._since >= self._cfg.seconds:
            self._fired = True
            self._latched = True

    def reset(self) -> None:
        self._reset()

    # -- internals ---------------------------------------------------------

    def _reset(self) -> None:
        self._anchor = None
        self._since = math.inf
        self._latched = False

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
