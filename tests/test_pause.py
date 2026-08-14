"""The pause / resume kill switch.

This is a safety feature, so the tests weight accordingly: most of them are
about it *not* firing. A kill switch that engages by accident is worse than no
kill switch, because the user cannot tell the difference between "paused" and
"broken".
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gestureflow import actions as act
from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.config import AppConfig, PauseConfig
from gestureflow.inference import InferenceThread
from gestureflow.modes import Mode
from gestureflow.pause_fsm import PauseFSM, PauseState, rock_horns

HAND_SCALE = 0.20


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Landmark builders
# ---------------------------------------------------------------------------

def _base(wrist_y: float = 0.75):
    lms = [SimpleNamespace(x=0.5, y=wrist_y - 0.10, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    lms[9] = SimpleNamespace(x=0.5, y=wrist_y - HAND_SCALE, z=0.0)
    return lms


def horns():
    """Index and pinky extended, middle and ring curled."""
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    lms[6] = SimpleNamespace(x=0.45, y=knuckle, z=0.0)          # index PIP
    lms[8] = SimpleNamespace(x=0.45, y=knuckle - 0.12, z=0.0)   # index tip up
    lms[18] = SimpleNamespace(x=0.62, y=knuckle, z=0.0)         # pinky PIP
    lms[20] = SimpleNamespace(x=0.62, y=knuckle - 0.12, z=0.0)  # pinky tip up
    lms[9] = SimpleNamespace(x=0.52, y=knuckle, z=0.0)          # middle MCP
    lms[12] = SimpleNamespace(x=0.52, y=knuckle + 0.10, z=0.0)  # middle curled
    lms[13] = SimpleNamespace(x=0.57, y=knuckle, z=0.0)         # ring MCP
    lms[16] = SimpleNamespace(x=0.57, y=knuckle + 0.10, z=0.0)  # ring curled
    return lms


def open_palm():
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        lms[mcp] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
        lms[pip] = SimpleNamespace(x=0.5, y=knuckle - 0.06, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=knuckle - 0.14, z=0.0)
    return lms


def fist_hand():
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        lms[mcp] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
        lms[pip] = SimpleNamespace(x=0.5, y=knuckle + 0.03, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=knuckle + 0.08, z=0.0)
    return lms


def pointing():
    """Index up only -- the cursor pose. Must never read as horns."""
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    lms[6] = SimpleNamespace(x=0.45, y=knuckle, z=0.0)
    lms[8] = SimpleNamespace(x=0.45, y=knuckle - 0.14, z=0.0)
    for tip, pip, mcp in ((12, 10, 9), (16, 14, 13), (20, 18, 17)):
        lms[mcp] = SimpleNamespace(x=0.55, y=knuckle, z=0.0)
        lms[pip] = SimpleNamespace(x=0.55, y=knuckle + 0.03, z=0.0)
        lms[tip] = SimpleNamespace(x=0.55, y=knuckle + 0.08, z=0.0)
    return lms


# ---------------------------------------------------------------------------
# The pose
# ---------------------------------------------------------------------------

class TestRockHornsPose:
    def test_recognized(self):
        assert rock_horns(horns()) is True

    @pytest.mark.parametrize("builder,name", [
        (open_palm, "open palm"),
        (fist_hand, "fist"),
        (pointing, "pointing"),
    ])
    def test_other_poses_are_not_horns(self, builder, name):
        assert rock_horns(builder()) is False, f"{name} read as the pause pose"

    def test_needs_all_four_conditions(self):
        """Any one finger in the wrong place must disqualify the pose."""
        knuckle = 0.75 - HAND_SCALE
        breakages = {
            "index down": (8, SimpleNamespace(x=0.45, y=knuckle + 0.08, z=0.0)),
            "pinky down": (20, SimpleNamespace(x=0.62, y=knuckle + 0.08, z=0.0)),
            "middle up": (12, SimpleNamespace(x=0.52, y=knuckle - 0.12, z=0.0)),
            "ring up": (16, SimpleNamespace(x=0.57, y=knuckle - 0.12, z=0.0)),
        }
        for label, (idx, replacement) in breakages.items():
            lms = horns()
            lms[idx] = replacement
            assert rock_horns(lms) is False, f"still horns with {label}"

    @pytest.mark.parametrize("factor", [0.4, 0.7, 1.0, 1.6, 2.5])
    def test_survives_any_hand_size(self, factor):
        lms = horns()
        wx, wy = lms[0].x, lms[0].y
        scaled = [SimpleNamespace(x=wx + (p.x - wx) * factor,
                                  y=wy + (p.y - wy) * factor, z=p.z * factor)
                  for p in lms]
        assert rock_horns(scaled) is True


# ---------------------------------------------------------------------------
# The hold timer
# ---------------------------------------------------------------------------

def _cfg(hold=1.5, enabled=True):
    return PauseConfig(enabled=enabled, hold_seconds=hold, margin=0.25)


class TestHoldTimer:
    def test_does_not_toggle_before_the_hold_elapses(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.5), clock=clock)
        for _ in range(40):                      # ~1.3s at 30fps
            fsm.update(horns())
            clock.advance(1 / 30.0)
            assert not fsm.toggled
        assert not fsm.paused

    def test_toggles_once_the_hold_elapses(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.5), clock=clock)
        toggles = 0
        for _ in range(60):                      # 2s
            fsm.update(horns())
            toggles += fsm.toggled
            clock.advance(1 / 30.0)
        assert toggles == 1
        assert fsm.paused

    def test_a_single_broken_frame_resets_the_timer(self):
        """A fumbled approach must not accumulate toward a toggle."""
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)
        for i in range(90):
            # Break the pose every 20th frame; the timer should never reach 1s.
            fsm.update(pointing() if i % 20 == 19 else horns())
            clock.advance(1 / 30.0)
            assert not fsm.toggled
        assert not fsm.paused

    def test_progress_climbs_monotonically_while_held(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)
        seen = []
        for _ in range(20):
            fsm.update(horns())
            seen.append(fsm.progress)
            clock.advance(1 / 30.0)
        assert seen == sorted(seen)
        assert 0.0 < seen[-1] <= 1.0

    def test_progress_resets_when_the_pose_breaks(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)
        for _ in range(10):
            fsm.update(horns())
            clock.advance(1 / 30.0)
        assert fsm.progress > 0
        fsm.update(pointing())
        assert fsm.progress == 0.0


class TestLatch:
    def test_holding_past_the_toggle_does_not_toggle_again(self):
        """Otherwise a three-second hold toggles twice and changes nothing."""
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)
        toggles = 0
        for _ in range(150):                     # 5s of continuous holding
            fsm.update(horns())
            toggles += fsm.toggled
            clock.advance(1 / 30.0)
        assert toggles == 1
        assert fsm.paused

    def test_releasing_then_holding_again_toggles_back(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)

        for _ in range(40):
            fsm.update(horns())
            clock.advance(1 / 30.0)
        assert fsm.paused

        fsm.update(pointing())                   # release
        clock.advance(1 / 30.0)

        for _ in range(40):
            fsm.update(horns())
            clock.advance(1 / 30.0)
        assert not fsm.paused

    def test_hand_leaving_frame_releases_the_latch(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=1.0), clock=clock)
        for _ in range(40):
            fsm.update(horns())
            clock.advance(1 / 30.0)
        assert fsm.state is PauseState.LATCHED
        fsm.update(None)
        assert fsm.state is PauseState.IDLE


class TestDisabled:
    def test_disabled_never_toggles(self):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=0.1, enabled=False), clock=clock)
        for _ in range(120):
            fsm.update(horns())
            clock.advance(1 / 30.0)
            assert not fsm.toggled
        assert not fsm.paused


class TestPauseProperties:
    @given(
        pattern=st.lists(st.booleans(), min_size=1, max_size=200),
    )
    @settings(max_examples=150, deadline=None)
    def test_toggles_only_after_a_continuous_hold(self, pattern):
        """For any sequence of pose-present/absent frames, a toggle can only
        happen on a frame preceded by an unbroken run long enough to cover the
        configured hold."""
        clock = Clock()
        hold = 0.5
        fsm = PauseFSM(_cfg(hold=hold), clock=clock)

        run_start = None
        for present in pattern:
            if present and run_start is None:
                run_start = clock.now
            elif not present:
                run_start = None

            fsm.update(horns() if present else pointing())
            if fsm.toggled:
                assert run_start is not None
                assert clock.now - run_start >= hold - 1e-9
            clock.advance(1 / 30.0)

    @given(pattern=st.lists(st.booleans(), min_size=1, max_size=200))
    @settings(max_examples=150, deadline=None)
    def test_paused_only_changes_on_a_toggle_frame(self, pattern):
        clock = Clock()
        fsm = PauseFSM(_cfg(hold=0.5), clock=clock)
        previous = fsm.paused
        for present in pattern:
            fsm.update(horns() if present else pointing())
            if fsm.paused != previous:
                assert fsm.toggled
            previous = fsm.paused
            clock.advance(1 / 30.0)


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

class _StubModel:
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


def _capture(landmarks, t=0.0):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


class TestPausedPipeline:
    def _pipeline(self, clock, hold=1.0):
        cfg = AppConfig(pause=PauseConfig(enabled=True, hold_seconds=hold,
                                          margin=0.25))
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    cfg, clock=clock)
        return inference, GestureRouter(cfg, 1920, 1080)

    def _hold_horns(self, inference, router, clock, hold=1.0):
        """Hold the pose long enough to toggle, whichever way it toggles."""
        toggled = None
        for _ in range(int(hold * 30) + 10):
            clock.now += 1 / 30.0
            result = inference.process(_capture(horns(), t=clock.now))
            router.route(result, now=clock.now)
            if result.pause_toggled:
                toggled = result
        assert toggled is not None, "the hold never produced a toggle"
        return toggled

    def _release(self, inference, router, clock, frames=5):
        for _ in range(frames):
            clock.now += 1 / 30.0
            result = inference.process(_capture(pointing(), t=clock.now))
            router.route(result, now=clock.now)

    def test_pausing_engages_from_the_pipeline(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        result = self._hold_horns(inference, router, clock)
        assert result.paused
        assert router.active_mode(result) is Mode.PAUSED

    def test_nothing_dispatches_while_paused(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        self._hold_horns(inference, router, clock)

        # Now do everything that would normally produce actions.
        for builder in (pointing, fist_hand, open_palm):
            for _ in range(40):
                clock.now += 1 / 30.0
                result = inference.process(_capture(builder(), t=clock.now))
                emitted = router.route(result, now=clock.now)
                assert emitted == [], (
                    f"{builder.__name__} dispatched {emitted} while paused"
                )

    def test_the_toggle_frame_dispatches_nothing_else(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        for _ in range(60):
            clock.now += 1 / 30.0
            result = inference.process(_capture(horns(), t=clock.now))
            emitted = router.route(result, now=clock.now)
            if result.pause_toggled:
                assert all(isinstance(a, act.ReleaseCursor) for a in emitted)
                return
        raise AssertionError("never toggled")

    def test_resuming_restores_dispatch(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        assert self._hold_horns(inference, router, clock).paused

        # Release the pose, then hold it again to resume.
        self._release(inference, router, clock)
        assert not self._hold_horns(inference, router, clock).paused
        self._release(inference, router, clock)

        moves = 0
        for _ in range(40):
            clock.now += 1 / 30.0
            result = inference.process(
                _capture(pointing(), t=clock.now))
            moves += sum(1 for a in router.route(result, now=clock.now)
                         if isinstance(a, act.MoveCursor))
        assert moves > 0, "cursor did not resume after unpausing"

    def test_pausing_releases_the_cursor_filter(self):
        """Otherwise resuming smooths against a position from before the pause."""
        clock = Clock()
        inference, router = self._pipeline(clock)

        for _ in range(10):
            clock.now += 1 / 30.0
            router.route(inference.process(_capture(pointing(), t=clock.now)),
                         now=clock.now)

        released = False
        for _ in range(60):
            clock.now += 1 / 30.0
            result = inference.process(_capture(horns(), t=clock.now))
            emitted = router.route(result, now=clock.now)
            released |= any(isinstance(a, act.ReleaseCursor) for a in emitted)
            if result.paused:
                break
        assert released
