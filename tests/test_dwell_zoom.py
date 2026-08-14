"""Dwell click and zoom.

Both are modes layered onto hands that already mean something else -- dwell onto
the cursor, zoom onto the two landmarks that are already the left-click pinch --
so most of these tests are about the boundaries rather than the happy path.
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
from gestureflow.click_fsm import ClickFSM
from gestureflow.config import (
    DEFAULT_CONFIG,
    AppConfig,
    DwellConfig,
    ZoomConfig,
)
from gestureflow.dwell_fsm import DwellFSM
from gestureflow.inference import InferenceThread
from gestureflow.modes import Mode
from gestureflow.zoom_fsm import ZoomFSM, ZoomState, thumb_index_angle, zoom_pose

HAND_SCALE = 0.20


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ===========================================================================
# Dwell click
# ===========================================================================

def _dwell_cfg(enabled=True, seconds=1.0, radius=40.0):
    return DwellConfig(enabled=enabled, seconds=seconds, radius_px=radius)


class TestDwellDisabledByDefault:
    def test_the_shipped_default_is_off(self):
        """Resting the pointer must not click unless the user asked for it."""
        assert DEFAULT_CONFIG.dwell.enabled is False

    def test_disabled_never_fires(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(enabled=False, seconds=0.1))
        for _ in range(200):
            fsm.update((500.0, 500.0), clock.now)
            assert not fsm.fired
            clock.advance(1 / 30.0)


class TestDwellFiring:
    def test_holding_still_fires_after_the_configured_time(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0))
        fires = 0
        for _ in range(45):                     # 1.5s
            fsm.update((500.0, 500.0), clock.now)
            fires += fsm.fired
            clock.advance(1 / 30.0)
        assert fires == 1

    def test_does_not_fire_early(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0))
        for _ in range(25):                     # ~0.83s
            fsm.update((500.0, 500.0), clock.now)
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_small_jitter_inside_the_radius_still_fires(self):
        """The pointer is never perfectly still; the radius exists for that."""
        import random
        rng = random.Random(3)
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0, radius=40.0))
        fires = 0
        for _ in range(45):
            fsm.update((500.0 + rng.uniform(-10, 10),
                        500.0 + rng.uniform(-10, 10)), clock.now)
            fires += fsm.fired
            clock.advance(1 / 30.0)
        assert fires == 1

    def test_leaving_the_radius_restarts_the_timer(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0, radius=40.0))
        x = 500.0
        for _ in range(90):
            x += 20.0                            # steadily walking away
            fsm.update((x, 500.0), clock.now)
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_no_target_resets(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0))
        for _ in range(20):
            fsm.update((500.0, 500.0), clock.now)
            clock.advance(1 / 30.0)
        fsm.update(None, clock.now)
        assert fsm.anchor is None
        assert fsm.progress == 0.0

    def test_progress_climbs_toward_one(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=1.0))
        seen = []
        for _ in range(25):
            fsm.update((500.0, 500.0), clock.now)
            seen.append(fsm.progress)
            clock.advance(1 / 30.0)
        assert seen == sorted(seen)
        assert 0 < seen[-1] < 1.0


class TestDwellReArm:
    def test_resting_forever_clicks_once_not_repeatedly(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=0.5))
        fires = 0
        for _ in range(300):                    # 10 seconds of resting
            fsm.update((500.0, 500.0), clock.now)
            fires += fsm.fired
            clock.advance(1 / 30.0)
        assert fires == 1, f"resting the pointer fired {fires} clicks"

    def test_moving_away_then_resting_again_fires_again(self):
        clock = Clock()
        fsm = DwellFSM(_dwell_cfg(seconds=0.5, radius=40.0))
        fires = 0

        for _ in range(25):                     # dwell here
            fsm.update((500.0, 500.0), clock.now)
            fires += fsm.fired
            clock.advance(1 / 30.0)
        for _ in range(25):                     # dwell somewhere else
            fsm.update((900.0, 900.0), clock.now)
            fires += fsm.fired
            clock.advance(1 / 30.0)

        assert fires == 2

    @given(
        moves=st.lists(
            st.tuples(st.floats(min_value=0, max_value=2000, allow_nan=False),
                      st.floats(min_value=0, max_value=2000, allow_nan=False)),
            min_size=1, max_size=200),
    )
    @settings(max_examples=150, deadline=None)
    def test_never_fires_twice_without_leaving_the_radius(self, moves):
        clock = Clock()
        radius = 40.0
        fsm = DwellFSM(_dwell_cfg(seconds=0.3, radius=radius))

        last_fire_at = None
        for target in moves:
            fsm.update(target, clock.now)
            if fsm.fired:
                if last_fire_at is not None:
                    moved = ((target[0] - last_fire_at[0]) ** 2
                             + (target[1] - last_fire_at[1]) ** 2) ** 0.5
                    assert moved > radius, (
                        "fired twice without the pointer leaving the radius"
                    )
                last_fire_at = target
            clock.advance(1 / 30.0)


# ===========================================================================
# Zoom
# ===========================================================================

def _base(wrist_y: float = 0.75):
    lms = [SimpleNamespace(x=0.5, y=wrist_y - 0.10, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    lms[9] = SimpleNamespace(x=0.5, y=wrist_y - HAND_SCALE, z=0.0)
    return lms


def zoom_hand(spread: float = 0.20):
    """An L-shape: index up, thumb out sideways, other three curled."""
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    lms[5] = SimpleNamespace(x=0.50, y=knuckle, z=0.0)
    lms[6] = SimpleNamespace(x=0.50, y=knuckle - 0.06, z=0.0)
    lms[8] = SimpleNamespace(x=0.50, y=knuckle - 0.16, z=0.0)   # index straight up
    # Thumb points sideways from its MCP -- roughly 90 degrees off the index.
    lms[2] = SimpleNamespace(x=0.52, y=knuckle + 0.06, z=0.0)
    lms[3] = SimpleNamespace(x=0.52 + spread * 0.4, y=knuckle + 0.05, z=0.0)
    lms[4] = SimpleNamespace(x=0.52 + spread, y=knuckle + 0.04, z=0.0)
    for tip, mcp in ((12, 9), (16, 13), (20, 17)):
        lms[mcp] = SimpleNamespace(x=0.56, y=knuckle, z=0.0)
        lms[tip] = SimpleNamespace(x=0.56, y=knuckle + 0.08, z=0.0)
    return lms


def pinched_hand():
    """Thumb touching index -- the click pose, which must never zoom."""
    lms = zoom_hand(spread=0.01)
    lms[4] = SimpleNamespace(x=lms[8].x + 0.004, y=lms[8].y, z=0.0)
    lms[3] = SimpleNamespace(x=lms[8].x + 0.02, y=lms[8].y + 0.02, z=0.0)
    return lms


def _zoom_cfg(**kw):
    base = {"enabled": True, "min_separation": 0.55, "sensitivity": 0.06,
            "min_hold_frames": 4, "cooldown": 0.0, "curl_margin": 0.12,
            "min_angle_degrees": 65.0}
    base.update(kw)
    return ZoomConfig(**base)


class TestZoomPose:
    def test_the_l_shape_is_recognized(self):
        assert zoom_pose(zoom_hand(0.25), _zoom_cfg()) is True

    def test_a_closed_pinch_is_never_a_zoom_pose(self):
        """The click and zoom ranges are disjoint by construction."""
        assert zoom_pose(pinched_hand(), _zoom_cfg()) is False

    def test_the_angle_separates_zoom_from_pointing(self):
        """Distance alone reads a pointing hand as a zoom; the angle does not."""
        assert thumb_index_angle(zoom_hand(0.25)) >= 65.0

    def test_a_degenerate_hand_is_not_a_zoom_pose(self):
        collapsed = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        assert zoom_pose(collapsed, _zoom_cfg()) is False

    @pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
    def test_survives_any_hand_size(self, factor):
        lms = zoom_hand(0.25)
        wx, wy = lms[0].x, lms[0].y
        scaled = [SimpleNamespace(x=wx + (p.x - wx) * factor,
                                  y=wy + (p.y - wy) * factor, z=p.z * factor)
                  for p in lms]
        assert zoom_pose(scaled, _zoom_cfg()) is True


class TestZoomFiring:
    def _armed(self, clock, cfg=None):
        fsm = ZoomFSM(cfg or _zoom_cfg(), clock=clock)
        for _ in range(6):
            fsm.update(zoom_hand(0.25))
            clock.advance(1 / 30.0)
        assert fsm.state is ZoomState.ZOOMING
        return fsm

    def test_spreading_zooms_in(self):
        clock = Clock()
        fsm = self._armed(clock)
        directions = []
        for i in range(10):
            fsm.update(zoom_hand(0.25 + i * 0.03))
            if fsm.fired:
                directions.append(fsm.direction)
            clock.advance(1 / 30.0)
        assert directions and all(d == "in" for d in directions)

    def test_closing_zooms_out(self):
        clock = Clock()
        fsm = ZoomFSM(_zoom_cfg(), clock=clock)
        for _ in range(6):
            fsm.update(zoom_hand(0.55))
            clock.advance(1 / 30.0)
        directions = []
        for i in range(8):
            fsm.update(zoom_hand(0.55 - i * 0.03))
            if fsm.fired:
                directions.append(fsm.direction)
            clock.advance(1 / 30.0)
        assert directions and all(d == "out" for d in directions)

    def test_a_static_pose_emits_nothing(self):
        clock = Clock()
        fsm = self._armed(clock)
        for _ in range(30):
            fsm.update(zoom_hand(0.25))
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_needs_the_pose_held_before_arming(self):
        clock = Clock()
        fsm = ZoomFSM(_zoom_cfg(min_hold_frames=6), clock=clock)
        for i in range(5):
            fsm.update(zoom_hand(0.25 + i * 0.05))
            assert not fsm.fired
            clock.advance(1 / 30.0)

    def test_breaking_the_pose_disarms(self):
        clock = Clock()
        fsm = self._armed(clock)
        fsm.update(pinched_hand())
        assert fsm.state is ZoomState.IDLE
        assert not fsm.is_active

    def test_disabled_never_fires(self):
        clock = Clock()
        fsm = ZoomFSM(_zoom_cfg(enabled=False), clock=clock)
        for i in range(30):
            fsm.update(zoom_hand(0.25 + i * 0.03))
            assert not fsm.fired
            clock.advance(1 / 30.0)


class TestZoomVersusClick:
    """The two share thumb and index; they must never both engage."""

    def test_a_pinch_that_clicks_never_arms_zoom(self):
        clock = Clock()
        zoom = ZoomFSM(_zoom_cfg(), clock=clock)
        click = ClickFSM(DEFAULT_CONFIG.click, clock=clock)

        for _ in range(20):
            hand = pinched_hand()
            zoom.update(hand)
            click.update(hand)
            assert not zoom.is_active
            clock.advance(1 / 30.0)
        assert click.is_active

    def test_a_zoom_pose_never_arms_the_click(self):
        clock = Clock()
        zoom = ZoomFSM(_zoom_cfg(), clock=clock)
        click = ClickFSM(DEFAULT_CONFIG.click, clock=clock)

        for i in range(20):
            hand = zoom_hand(0.25 + i * 0.02)
            zoom.update(hand)
            click.update(hand)
            assert not click.is_active
            clock.advance(1 / 30.0)
        assert zoom.is_active


# ===========================================================================
# Through the pipeline
# ===========================================================================

class _StubModel:
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


def _capture(landmarks, t=0.0):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


class TestZoomThroughThePipeline:
    def _pipeline(self, clock, zoom_cfg=None):
        conf = AppConfig(zoom=zoom_cfg or _zoom_cfg())
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    conf, clock=clock)
        return inference, GestureRouter(conf, 1920, 1080)

    def test_spreading_emits_zoom_in_commands(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []
        for i in range(24):
            clock.advance(1 / 30.0)
            result = inference.process(
                _capture(zoom_hand(0.25 + i * 0.03), t=clock.now))
            emitted += router.route(result, now=clock.now)

        commands = [a for a in emitted if isinstance(a, act.NamedCommand)]
        assert commands
        assert all(c.gesture == "zoom_in" for c in commands)

    def test_zooming_claims_the_mode_and_suppresses_the_cursor(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        for i in range(24):
            clock.advance(1 / 30.0)
            result = inference.process(
                _capture(zoom_hand(0.25 + i * 0.03), t=clock.now))
            emitted = router.route(result, now=clock.now)
            if result.zoom_active:
                assert router.active_mode(result) is Mode.ZOOM
                assert not any(isinstance(a, act.MoveCursor) for a in emitted)

    def test_zooming_emits_no_clicks(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []
        for i in range(30):
            clock.advance(1 / 30.0)
            result = inference.process(
                _capture(zoom_hand(0.25 + i * 0.02), t=clock.now))
            emitted += router.route(result, now=clock.now)
        assert not [a for a in emitted if isinstance(a, act.Click)]


class TestDwellThroughThePipeline:
    def _pipeline(self, clock, enabled):
        conf = AppConfig(dwell=_dwell_cfg(enabled=enabled, seconds=0.5))
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    conf, clock=clock)
        return inference, GestureRouter(conf, 1920, 1080)

    def _pointing(self):
        lms = _base()
        knuckle = 0.75 - HAND_SCALE
        lms[5] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
        lms[6] = SimpleNamespace(x=0.45, y=knuckle, z=0.0)
        lms[8] = SimpleNamespace(x=0.45, y=knuckle - 0.14, z=0.0)
        for tip, mcp in ((12, 9), (16, 13), (20, 17)):
            lms[mcp] = SimpleNamespace(x=0.55, y=knuckle, z=0.0)
            lms[tip] = SimpleNamespace(x=0.55, y=knuckle + 0.08, z=0.0)
        lms[2] = SimpleNamespace(x=0.62, y=knuckle + 0.06, z=0.0)
        lms[3] = SimpleNamespace(x=0.62, y=knuckle + 0.04, z=0.0)
        lms[4] = SimpleNamespace(x=0.62, y=knuckle + 0.02, z=0.0)
        return lms

    def test_a_still_pointing_hand_clicks_when_enabled(self):
        clock = Clock()
        inference, router = self._pipeline(clock, enabled=True)
        emitted = []
        for _ in range(40):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(self._pointing(), t=clock.now))
            emitted += router.route(result, now=clock.now)
        clicks = [a for a in emitted if isinstance(a, act.Click)]
        assert len(clicks) == 1

    def test_the_same_hand_never_clicks_when_disabled(self):
        clock = Clock()
        inference, router = self._pipeline(clock, enabled=False)
        emitted = []
        for _ in range(120):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(self._pointing(), t=clock.now))
            emitted += router.route(result, now=clock.now)
        assert not [a for a in emitted if isinstance(a, act.Click)]
