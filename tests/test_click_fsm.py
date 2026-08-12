"""Tests for ClickFSM.

This class drove both the left- and right-click paths and had no tests at all.
Every timing assertion here uses an injected clock rather than sleeping, so the
suite stays fast and the cooldown behaviour is exercised exactly rather than
approximately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gestureflow.click_fsm import ClickFSM, ClickState, _pinch_distance
from gestureflow.config import ClickConfig


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cfg(close=0.045, open_=0.065, hold=4, cooldown=0.4) -> ClickConfig:
    return ClickConfig(close_threshold=close, open_threshold=open_,
                       min_hold_frames=hold, cooldown=cooldown)


def lms(distance: float, lm_a: int = 4, lm_b: int = 8):
    """21 landmarks with the two pinch landmarks `distance` apart on x.

    Every other landmark is spread far apart, so a test that pinches one pair
    cannot accidentally bring an unrelated pair within the close threshold.
    """
    out = [SimpleNamespace(x=float(i), y=float(i), z=0.0) for i in range(21)]
    out[lm_a] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    out[lm_b] = SimpleNamespace(x=0.5 + distance, y=0.5, z=0.0)
    return out


def press_and_hold(fsm: ClickFSM, frames: int, distance: float = 0.01) -> None:
    for _ in range(frames):
        fsm.update(lms(distance))


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

class TestTransitions:
    def test_starts_idle(self):
        assert ClickFSM(cfg()).state is ClickState.IDLE

    def test_close_pinch_enters_pressing(self):
        fsm = ClickFSM(cfg())
        fsm.update(lms(0.01))
        assert fsm.state is ClickState.PRESSING

    def test_sustained_pinch_reaches_held(self):
        fsm = ClickFSM(cfg(hold=4))
        press_and_hold(fsm, 5)
        assert fsm.state is ClickState.HELD

    def test_release_before_hold_threshold_resets(self):
        fsm = ClickFSM(cfg(hold=4))
        press_and_hold(fsm, 2)
        fsm.update(lms(0.9))
        assert fsm.state is ClickState.IDLE
        assert not fsm.click_fired

    def test_none_landmarks_resets(self):
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 4)
        assert fsm.state is ClickState.HELD
        fsm.update(None)
        assert fsm.state is ClickState.IDLE

    def test_hand_leaving_frame_does_not_fire_a_click(self):
        # A hand vanishing mid-pinch is not a click. Firing on disappearance
        # would make walking away from the camera left-click.
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 4)
        fsm.update(None)
        assert not fsm.click_fired


# ---------------------------------------------------------------------------
# The release edge
# ---------------------------------------------------------------------------

class TestReleaseEdge:
    def test_click_fires_on_release_not_press(self):
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 5)
        assert not fsm.click_fired, "holding a pinch must not click"
        fsm.update(lms(0.9))
        assert fsm.click_fired, "releasing a held pinch must click"

    def test_holding_does_not_auto_repeat(self):
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 60)
        assert not fsm.click_fired

    def test_click_fired_is_only_true_for_one_frame(self):
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 4)
        fsm.update(lms(0.9))
        assert fsm.click_fired
        fsm.update(lms(0.9))
        assert not fsm.click_fired

    def test_hysteresis_band_does_not_release(self):
        # Between close (0.045) and open (0.065) the state must hold. Without
        # the dead band a hand hovering at the boundary chatters clicks.
        fsm = ClickFSM(cfg(close=0.045, open_=0.065, hold=2))
        press_and_hold(fsm, 4, distance=0.01)
        assert fsm.state is ClickState.HELD
        fsm.update(lms(0.055))
        assert fsm.state is ClickState.HELD
        assert not fsm.click_fired

    def test_opening_past_the_band_releases(self):
        fsm = ClickFSM(cfg(close=0.045, open_=0.065, hold=2))
        press_and_hold(fsm, 4, distance=0.01)
        fsm.update(lms(0.08))
        assert fsm.click_fired


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def _click(self, fsm: ClickFSM) -> bool:
        press_and_hold(fsm, 4)
        fsm.update(lms(0.9))
        return fsm.click_fired

    def test_second_click_inside_cooldown_is_suppressed(self):
        clock = FakeClock()
        fsm = ClickFSM(cfg(hold=2, cooldown=0.4), clock=clock)
        assert self._click(fsm)
        clock.advance(0.1)
        assert not self._click(fsm)

    def test_second_click_after_cooldown_fires(self):
        clock = FakeClock()
        fsm = ClickFSM(cfg(hold=2, cooldown=0.4), clock=clock)
        assert self._click(fsm)
        clock.advance(0.5)
        assert self._click(fsm)

    def test_just_short_of_cooldown_is_suppressed(self):
        clock = FakeClock()
        fsm = ClickFSM(cfg(hold=2, cooldown=0.4), clock=clock)
        assert self._click(fsm)
        clock.advance(0.399)
        assert not self._click(fsm)

    def test_just_past_cooldown_fires(self):
        # Deliberately not testing exact float equality at the boundary: with
        # a wall clock the odds of landing on it are nil, and asserting it
        # would only be testing IEEE 754 rounding.
        clock = FakeClock()
        fsm = ClickFSM(cfg(hold=2, cooldown=0.4), clock=clock)
        assert self._click(fsm)
        clock.advance(0.401)
        assert self._click(fsm)

    def test_suppressed_click_does_not_extend_the_cooldown(self):
        # A rejected attempt must not reset the timer, or rapid pinching
        # would lock the user out indefinitely.
        clock = FakeClock()
        fsm = ClickFSM(cfg(hold=2, cooldown=0.4), clock=clock)
        assert self._click(fsm)
        clock.advance(0.2)
        assert not self._click(fsm)
        clock.advance(0.2)
        assert self._click(fsm)


# ---------------------------------------------------------------------------
# Progress reporting and geometry
# ---------------------------------------------------------------------------

class TestHoldProgress:
    def test_progress_climbs_to_one(self):
        fsm = ClickFSM(cfg(hold=4))
        seen = []
        for _ in range(5):
            fsm.update(lms(0.01))
            seen.append(fsm.hold_progress)
        assert seen == sorted(seen)
        assert seen[-1] == 1.0

    def test_progress_never_exceeds_one(self):
        fsm = ClickFSM(cfg(hold=2))
        press_and_hold(fsm, 30)
        assert fsm.hold_progress == 1.0

    def test_zero_hold_frames_reports_complete(self):
        assert ClickFSM(cfg(hold=0)).hold_progress == 1.0


class TestPinchDistance:
    def test_is_euclidean_in_three_dimensions(self):
        a = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        b = SimpleNamespace(x=3.0, y=4.0, z=12.0)
        assert _pinch_distance({4: a, 8: b}, 4, 8) == pytest.approx(13.0)

    def test_identical_points_are_zero_apart(self):
        p = SimpleNamespace(x=0.2, y=0.3, z=0.4)
        assert _pinch_distance({4: p, 8: p}, 4, 8) == 0.0


class TestLandmarkPairing:
    def test_right_click_pair_is_independent(self):
        # The right-click FSM watches middle(12)+index(8). Pinching thumb to
        # index must not drive it.
        right = ClickFSM(cfg(hold=2), landmark_a=12, landmark_b=8)
        for _ in range(6):
            right.update(lms(0.01, lm_a=4, lm_b=8))
        assert right.state is ClickState.IDLE

    def test_right_click_pair_responds_to_its_own_landmarks(self):
        right = ClickFSM(cfg(hold=2), landmark_a=12, landmark_b=8)
        for _ in range(4):
            right.update(lms(0.01, lm_a=12, lm_b=8))
        assert right.state is ClickState.HELD
