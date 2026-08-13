"""Cursor smoothing.

The problem
-----------
Hand landmarks jitter by a pixel or two every frame even when the hand is
perfectly still, and that jitter is what makes a gesture cursor feel broken.
Smoothing it away is easy; smoothing it away *without* making the pointer lag
behind fast movement is the actual problem, and a fixed-time-constant low-pass
cannot do both. Set the constant high enough to settle a still hand and the
pointer drags behind a moving one; set it low enough to keep up and the jitter
comes straight through.

The previous implementation was a fixed-tau exponential filter, and it landed on
the wrong side of that trade twice over: `tau = 0.15` was slow enough to feel
sluggish, while an irregular `dt` let it lurch. See DIAGNOSIS.md.

The fix
-------
A One Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts its cutoff
frequency to how fast the signal is moving:

    cutoff = min_cutoff + beta * |estimated speed|

Nearly-still hand -> low cutoff -> heavy smoothing -> jitter disappears.
Fast-moving hand  -> high cutoff -> light smoothing -> pointer keeps up.

Which is exactly the right behaviour, because jitter is only visible when the
pointer is nearly still: nobody notices a pixel of noise on something crossing
the screen.
"""

from __future__ import annotations

import math
from typing import Optional


class LowPass:
    """A plain exponential filter with an externally supplied alpha."""

    def __init__(self) -> None:
        self.value: Optional[float] = None

    def __call__(self, sample: float, alpha: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None

    @property
    def initialized(self) -> bool:
        return self.value is not None


def alpha_for(cutoff: float, dt: float) -> float:
    """Blend factor for a given cutoff frequency and time step.

    tau = 1 / (2*pi*cutoff) is the filter's time constant; the discrete
    equivalent over a step dt is dt / (dt + tau).
    """
    if dt <= 0.0:
        # No time passed, so nothing should move. Returning 1.0 here (the
        # textbook convention for a first sample) would instead snap straight
        # to the target and defeat seeding the filter at a known position.
        return 0.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return dt / (dt + tau)


class OneEuroFilter:
    """Speed-adaptive low-pass filter for a single scalar channel."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be positive")
        if derivative_cutoff <= 0.0:
            raise ValueError("derivative_cutoff must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._x = LowPass()
        self._dx = LowPass()
        self._last_value: Optional[float] = None

    def reset(self, value: Optional[float] = None) -> None:
        """Clear filter state, optionally seeding it at a known value."""
        self._x.reset()
        self._dx.reset()
        self._last_value = None
        if value is not None:
            self._x(value, 1.0)
            self._last_value = value

    @property
    def initialized(self) -> bool:
        return self._x.initialized

    def __call__(self, sample: float, dt: float) -> float:
        if not self._x.initialized:
            self._x(sample, 1.0)
            self._last_value = sample
            return sample

        # Speed estimate, itself smoothed -- an unfiltered derivative of a
        # noisy signal is noise, and would make the cutoff flap around.
        rate = 0.0 if dt <= 0.0 else (sample - self._last_value) / dt
        smoothed_rate = self._dx(rate, alpha_for(self.derivative_cutoff, dt))
        self._last_value = sample

        cutoff = self.min_cutoff + self.beta * abs(smoothed_rate)
        return self._x(sample, alpha_for(cutoff, dt))


class CursorFilter:
    """Two One Euro filters, one per screen axis, sharing a clock.

    Kept as one object so both axes always see the same dt. Filtering x and y
    with independently measured intervals would let them disagree about how
    much time passed, which shows up as diagonal drift.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.012,
        derivative_cutoff: float = 1.0,
        max_dt: float = 0.1,
    ) -> None:
        self.max_dt = max_dt
        self._fx = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._last_time: Optional[float] = None

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self._fx.reset(x)
        self._fy.reset(y)
        self._last_time = None

    @property
    def initialized(self) -> bool:
        return self._fx.initialized

    def is_stale(self, now: float) -> bool:
        """True when more time has passed than any real frame interval.

        A gap this long means tracking was lost -- the hand left frame, or the
        pipeline stalled. The filter's stored position and velocity describe a
        hand that is no longer there, so continuing to filter against them
        produces one enormous step. The caller should reset and re-seed.
        """
        if self._last_time is None:
            return False
        return (now - self._last_time) > self.max_dt

    def __call__(self, x: float, y: float, now: float) -> tuple:
        if self._last_time is None:
            dt = 0.0
        else:
            dt = now - self._last_time
            # A clock that went backwards is not a real interval either.
            if dt < 0.0:
                dt = 0.0
            dt = min(dt, self.max_dt)
        self._last_time = now

        return self._fx(x, dt), self._fy(y, dt)
