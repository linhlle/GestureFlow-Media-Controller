"""Horizontal swipe, and its arbitration against scroll.

Scroll and swipe share the fist and are told apart by which way the hand moves.
The invariant that matters is that one movement can never be both -- a page that
scrolls while you meant to switch desktop is exactly the kind of collision the
mode separation exists to prevent.

The other half is debounce: a flick spans many frames above the velocity
threshold, and emitting per frame would fire a dozen desktop switches from one
hand movement.
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
from gestureflow.config import DEFAULT_CONFIG, AppConfig, SwipeConfig
from gestureflow.inference import InferenceThread
from gestureflow.modes import Mode
from gestureflow.scroll_fsm import ScrollFSM, ScrollState
from gestureflow.swipe_fsm import (
    SwipeFSM,
    SwipeState,
    horizontal_dominates,
    vertical_dominates,
)

HAND_SCALE = 0.20


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def fist(wrist_x: float = 0.5, wrist_y: float = 0.75):
    """A closed fist, translated rigidly to (wrist_x, wrist_y)."""
    dx = wrist_x - 0.5
    dy = wrist_y - 0.75
    lms = [SimpleNamespace(x=0.5 + dx, y=0.65 + dy, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=wrist_x, y=wrist_y, z=0.0)
    knuckle = 0.75 - HAND_SCALE + dy
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        lms[mcp] = SimpleNamespace(x=0.5 + dx, y=knuckle, z=0.0)
        lms[pip] = SimpleNamespace(x=0.5 + dx, y=knuckle + 0.03, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5 + dx, y=knuckle + 0.08, z=0.0)
    lms[9] = SimpleNamespace(x=0.5 + dx, y=knuckle, z=0.0)
    lms[2] = SimpleNamespace(x=0.5 + dx, y=knuckle + 0.10, z=0.0)
    lms[3] = SimpleNamespace(x=0.5 + dx, y=knuckle + 0.07, z=0.0)
    lms[4] = SimpleNamespace(x=0.5 + dx, y=knuckle + 0.05, z=0.0)
    return lms


def open_hand():
    lms = fist()
    knuckle = 0.75 - HAND_SCALE
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lms[mcp] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=knuckle - 0.12, z=0.0)
    return lms


def _swipe_cfg(**kw):
    base = {"enabled": True, "sensitivity": 0.16, "min_hold_frames": 3,
            "cooldown": 0.6, "axis_ratio": 1.5, "release_ratio": 0.5}
    base.update(kw)
    return SwipeConfig(**base)


def _armed(clock, cfg=None, frames=6):
    fsm = SwipeFSM(cfg or _swipe_cfg(), clock=clock)
    for _ in range(frames):
        fsm.update(fist())
        clock.advance(1 / 30.0)
    return fsm


# ---------------------------------------------------------------------------
# The arbitration rule itself
# ---------------------------------------------------------------------------

class TestAxisArbitration:
    def test_pure_vertical_is_a_scroll_not_a_swipe(self):
        assert vertical_dominates(0.0, 0.5) is True
        assert horizontal_dominates(0.0, 0.5) is False

    def test_pure_horizontal_is_a_swipe_not_a_scroll(self):
        assert horizontal_dominates(0.5, 0.0) is True
        assert vertical_dominates(0.5, 0.0) is False

    def test_the_ambiguous_band_belongs_to_neither(self):
        """Diagonal motion fires nothing rather than guessing."""
        vx, vy = 0.30, 0.25          # horizontal ahead, but not by 1.5x
        assert not vertical_dominates(vx, vy)
        assert not horizontal_dominates(vx, vy)

    @given(
        vx=st.floats(min_value=-2, max_value=2, allow_nan=False),
        vy=st.floats(min_value=-2, max_value=2, allow_nan=False),
    )
    @settings(max_examples=400, deadline=None)
    def test_a_motion_is_never_both(self, vx, vy):
        assert not (vertical_dominates(vx, vy) and horizontal_dominates(vx, vy))


# ---------------------------------------------------------------------------
# Scroll must be unchanged by the arbitration
# ---------------------------------------------------------------------------

class TestScrollStillWorks:
    def _scroll(self, clock, positions):
        fsm = ScrollFSM(DEFAULT_CONFIG.scroll, clock=clock)
        deltas = []
        for x, y in positions:
            fsm.update(fist(x, y))
            deltas.append(fsm.scroll_delta)
            clock.advance(1 / 30.0)
        return fsm, deltas

    def test_pure_vertical_scroll_is_unaffected(self):
        clock = Clock()
        fsm, deltas = self._scroll(
            clock, [(0.5, 0.75 - i * 0.015) for i in range(30)])
        assert fsm.state is ScrollState.SCROLLING
        assert any(d != 0 for d in deltas)
        assert all(d >= 0 for d in deltas)

    def test_scrolling_down_still_works(self):
        clock = Clock()
        _, deltas = self._scroll(
            clock, [(0.5, 0.75 + i * 0.015) for i in range(30)])
        assert any(d < 0 for d in deltas)
        assert all(d <= 0 for d in deltas)

    def test_a_horizontal_flick_produces_no_scroll(self):
        """Without arbitration a swipe would scroll the page on its way past."""
        clock = Clock()
        _, deltas = self._scroll(
            clock, [(0.5 + i * 0.02, 0.75) for i in range(30)])
        assert all(d == 0 for d in deltas), f"swipe leaked scroll: {deltas}"


# ---------------------------------------------------------------------------
# Swipe
# ---------------------------------------------------------------------------

class TestSwipeDetection:
    def test_a_rightward_flick_fires_right(self):
        clock = Clock()
        fsm = _armed(clock)
        fired = []
        for i in range(10):
            fsm.update(fist(0.5 + i * 0.05))
            if fsm.fired:
                fired.append(fsm.direction)
            clock.advance(1 / 30.0)
        assert fired == ["right"]

    def test_a_leftward_flick_fires_left(self):
        clock = Clock()
        fsm = _armed(clock)
        fired = []
        for i in range(10):
            fsm.update(fist(0.5 - i * 0.05))
            if fsm.fired:
                fired.append(fsm.direction)
            clock.advance(1 / 30.0)
        assert fired == ["left"]

    def test_one_continuous_sweep_fires_exactly_once(self):
        """A flick spans many frames above threshold; it is still one swipe."""
        clock = Clock()
        fsm = _armed(clock)
        fires = 0
        for i in range(40):
            fsm.update(fist(0.5 + i * 0.04))
            fires += fsm.fired
            clock.advance(1 / 30.0)
        assert fires == 1, f"one sweep fired {fires} times"

    def test_a_slow_drift_never_fires(self):
        clock = Clock()
        fsm = _armed(clock)
        for i in range(60):
            fsm.update(fist(0.5 + i * 0.002))
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_an_open_hand_never_swipes(self):
        clock = Clock()
        fsm = SwipeFSM(_swipe_cfg(), clock=clock)
        for i in range(30):
            lms = open_hand()
            for p in lms:
                p.x += i * 0.05
            fsm.update(lms)
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_needs_the_fist_held_before_it_arms(self):
        clock = Clock()
        fsm = SwipeFSM(_swipe_cfg(min_hold_frames=5), clock=clock)
        for i in range(4):
            fsm.update(fist(0.5 + i * 0.06))
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_releasing_the_fist_disarms(self):
        clock = Clock()
        fsm = _armed(clock)
        assert fsm.is_armed
        fsm.update(open_hand())
        assert fsm.state is SwipeState.IDLE
        assert not fsm.is_armed

    def test_disabled_never_fires(self):
        clock = Clock()
        fsm = SwipeFSM(_swipe_cfg(enabled=False), clock=clock)
        for i in range(40):
            fsm.update(fist(0.5 + i * 0.05))
            assert not fsm.fired
            clock.advance(1 / 30.0)


class TestSwipeDebounce:
    def test_a_second_swipe_needs_the_hand_to_slow_down_first(self):
        clock = Clock()
        fsm = _armed(clock, _swipe_cfg(cooldown=0.0))
        fires = 0
        # Sustained fast motion: still one swipe, because speed never drops.
        for i in range(60):
            fsm.update(fist(0.5 + i * 0.04))
            fires += fsm.fired
            clock.advance(1 / 30.0)
        assert fires == 1

    def test_slowing_down_then_flicking_again_fires_twice(self):
        clock = Clock()
        fsm = _armed(clock, _swipe_cfg(cooldown=0.0))
        fires = 0

        x = 0.5
        for _ in range(8):                    # flick
            x += 0.05
            fsm.update(fist(x))
            fires += fsm.fired
            clock.advance(1 / 30.0)
        for _ in range(8):                    # hold still
            fsm.update(fist(x))
            fires += fsm.fired
            clock.advance(1 / 30.0)
        for _ in range(8):                    # flick again
            x += 0.05
            fsm.update(fist(x))
            fires += fsm.fired
            clock.advance(1 / 30.0)

        assert fires == 2

    def test_the_cooldown_blocks_a_rapid_second_flick(self):
        clock = Clock()
        fsm = _armed(clock, _swipe_cfg(cooldown=5.0))
        fires = 0
        x = 0.5
        for _ in range(3):
            for _ in range(6):
                x += 0.05
                fsm.update(fist(x))
                fires += fsm.fired
                clock.advance(1 / 30.0)
            for _ in range(6):
                fsm.update(fist(x))
                fires += fsm.fired
                clock.advance(1 / 30.0)
        assert fires == 1


class TestSwipeProperties:
    @given(
        steps=st.lists(
            st.tuples(st.floats(min_value=-0.06, max_value=0.06,
                                allow_nan=False),
                      st.floats(min_value=-0.06, max_value=0.06,
                                allow_nan=False)),
            min_size=1, max_size=80),
    )
    @settings(max_examples=200, deadline=None)
    def test_scroll_and_swipe_never_fire_on_the_same_frame(self, steps):
        """The headline invariant: one hand movement, at most one meaning."""
        clock = Clock()
        scroll = ScrollFSM(DEFAULT_CONFIG.scroll, clock=clock)
        swipe = SwipeFSM(_swipe_cfg(), clock=clock)

        x, y = 0.5, 0.75
        for dx, dy in steps:
            x += dx
            y += dy
            lms = fist(x, y)
            scroll.update(lms)
            swipe.update(lms)
            assert not (scroll.scroll_delta != 0 and swipe.fired), (
                f"both fired on dx={dx:.4f} dy={dy:.4f}"
            )
            clock.advance(1 / 30.0)

    @given(steps=st.lists(
        st.floats(min_value=-0.08, max_value=0.08, allow_nan=False),
        min_size=1, max_size=80))
    @settings(max_examples=200, deadline=None)
    def test_direction_always_matches_the_movement(self, steps):
        clock = Clock()
        fsm = _armed(clock)
        x = 0.5
        for dx in steps:
            previous = x
            x += dx
            fsm.update(fist(x))
            if fsm.fired:
                assert fsm.direction == ("right" if x > previous else "left")
            clock.advance(1 / 30.0)


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

class _StubModel:
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


def _capture(landmarks, t=0.0):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


class TestSwipeThroughThePipeline:
    def _pipeline(self, clock):
        conf = AppConfig()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    conf, clock=clock)
        return inference, GestureRouter(conf, 1920, 1080)

    def test_a_flick_emits_one_named_command(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []
        for i in range(30):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(fist(0.5 + i * 0.04),
                                                t=clock.now))
            emitted += router.route(result, now=clock.now)

        commands = [a for a in emitted if isinstance(a, act.NamedCommand)]
        assert len(commands) == 1
        assert commands[0].gesture == "swipe_right"

    def test_a_flick_emits_no_scroll(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []
        for i in range(30):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(fist(0.5 + i * 0.04),
                                                t=clock.now))
            emitted += router.route(result, now=clock.now)
        assert not [a for a in emitted if isinstance(a, act.Scroll)]

    def test_a_vertical_drag_emits_scroll_and_no_swipe(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []
        for i in range(30):
            clock.advance(1 / 30.0)
            result = inference.process(
                _capture(fist(0.5, 0.75 - i * 0.015), t=clock.now))
            emitted += router.route(result, now=clock.now)

        assert [a for a in emitted if isinstance(a, act.Scroll)]
        assert not [a for a in emitted if isinstance(a, act.NamedCommand)]

    def test_a_held_fist_claims_scroll_or_swipe_never_cursor(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        for i in range(30):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(fist(0.5 + i * 0.04),
                                                t=clock.now))
            mode = router.active_mode(result)
            assert mode in (Mode.SCROLL, Mode.SWIPE, Mode.TRACKING), mode


class TestNamedBindings:
    def test_an_unbound_swipe_is_not_an_error(self):
        """A user who does not want swipes just leaves them out."""
        from gestureflow.commands import parse_commands
        commands = parse_commands({
            "version": 2, "neutral_label": 0,
            "gestures": [{"label": 1, "name": "X",
                          "action": {"type": "keypress", "key": "a"}}],
        })
        assert commands.get_named("swipe_left") is None
        assert not commands.has_named("swipe_left")

    def test_a_bound_swipe_resolves(self):
        from gestureflow.commands import parse_commands
        commands = parse_commands({
            "version": 2, "neutral_label": 0,
            "gestures": [{"gesture": "swipe_left", "name": "Back",
                          "action": {"type": "hotkey",
                                     "keys": ["command", "["]}}],
        })
        binding = commands.get_named("swipe_left")
        assert binding is not None
        assert binding.action.keys == ("command", "[")

    def test_unknown_gesture_names_are_rejected(self):
        from gestureflow.commands import CommandConfigError, parse_commands
        with pytest.raises(CommandConfigError, match="unknown gesture"):
            parse_commands({
                "version": 2, "neutral_label": 0,
                "gestures": [{"gesture": "backflip", "name": "X",
                              "action": {"type": "keypress", "key": "a"}}],
            })

    def test_a_binding_cannot_have_both_keys(self):
        from gestureflow.commands import CommandConfigError, parse_commands
        with pytest.raises(CommandConfigError, match="both"):
            parse_commands({
                "version": 2, "neutral_label": 0,
                "gestures": [{"label": 1, "gesture": "swipe_left", "name": "X",
                              "action": {"type": "keypress", "key": "a"}}],
            })

    def test_named_gestures_need_version_2(self):
        from gestureflow.commands import CommandConfigError, parse_commands
        with pytest.raises(CommandConfigError, match="version 2"):
            parse_commands({
                "version": 1, "neutral_label": 0,
                "gestures": [{"gesture": "swipe_left", "name": "X",
                              "action": {"type": "keypress", "key": "a"}}],
            })
