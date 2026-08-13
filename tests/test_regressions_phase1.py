"""Regression tests for the Phase 1 bug fixes.

Each test here fails against the code as it was before the fix.  They are kept
in one file so the link between "known bug" and "test that proves it stays
fixed" stays obvious.
"""

from __future__ import annotations

import math
import queue
from types import SimpleNamespace

import pytest

from gestureflow.click_fsm import ClickFSM, ClickState
from gestureflow.config import ClickConfig, DebounceConfig, ScrollConfig
from gestureflow.debouncer import GestureDebouncer
from gestureflow.scroll_fsm import ScrollFSM, _velocity_to_clicks
from gestureflow.utils import (
    BindingError,
    drop_oldest_put,
    validate_bindings,
)

# ---------------------------------------------------------------------------
# Bug 1: monotonic() epoch — cooldowns suppressed every action after launch
# ---------------------------------------------------------------------------

class TestMonotonicEpoch:
    """time.monotonic()'s reference point is undefined.

    Seeding a "last fired at" timestamp with 0.0 assumes monotonic() starts
    large.  On macOS it starts near zero, so `now - last >= cooldown` was false
    for the first cooldown seconds of the process and every gated action in
    that window was silently dropped.
    """

    def test_click_can_fire_immediately_after_construction(self):
        fsm = ClickFSM(ClickConfig(close_threshold=0.28, open_threshold=0.41,
                                   min_hold_frames=1, cooldown=1.3))
        fsm.update(_pinch(0.01))          # IDLE -> PRESSING
        fsm.update(_pinch(0.01))          # PRESSING -> HELD
        assert fsm.state is ClickState.HELD
        fsm.update(_pinch(0.9))           # release edge
        assert fsm.click_fired, (
            "a click made in the first cooldown-seconds of the process must "
            "still fire"
        )

    def test_debouncer_can_fire_immediately_after_construction(self):
        db = GestureDebouncer(
            DebounceConfig(vote_window_size=3, vote_threshold=3,
                           cmd_cooldown=1.3),
            confidence_threshold=0.5,
        )
        for _ in range(2):
            assert db.update(1, 0.99) is None
        assert db.update(1, 0.99) == 1, (
            "a command gesture completed right after launch must still fire"
        )

    def test_scroll_can_fire_immediately_after_construction(self):
        fsm = _armed_scroll(ScrollConfig(sensitivity=0.05, min_hold_frames=2,
                                         cooldown=10.0, step=2,
                                         velocity_exponent=1.6))
        fsm.update(_fist(0.50))
        fsm.update(_fist(0.44))
        assert fsm.scroll_delta != 0

    @pytest.mark.parametrize("attr, obj", [
        ("_last_click_time", ClickFSM()),
        ("_last_cmd_time", GestureDebouncer()),
        ("_last_scroll_time", ScrollFSM()),
    ])
    def test_seed_is_negative_infinity(self, attr, obj):
        assert getattr(obj, attr) == -math.inf


# ---------------------------------------------------------------------------
# Bug 2: float residue inflated the scroll click count
# ---------------------------------------------------------------------------

class TestScrollRounding:
    def test_exact_multiple_does_not_round_up(self):
        cfg = ScrollConfig(sensitivity=0.010, min_hold_frames=1, cooldown=0.0,
                           step=1, velocity_exponent=1.0)
        # 0.500 - 0.480 is 0.020000000000000018 in binary floating point.
        velocity = 0.500 - 0.480
        assert _velocity_to_clicks(velocity, cfg) == 2, (
            "float residue must not buy an extra scroll click"
        )

    def test_linear_exponent_stays_proportional(self):
        cfg = ScrollConfig(sensitivity=0.010, min_hold_frames=1, cooldown=0.0,
                           step=1, velocity_exponent=1.0)
        c1 = _velocity_to_clicks(0.500 - 0.480, cfg)
        c2 = _velocity_to_clicks(0.500 - 0.460, cfg)
        assert c2 == 2 * c1

    def test_genuine_fraction_still_rounds_up(self):
        cfg = ScrollConfig(sensitivity=0.010, min_hold_frames=1, cooldown=0.0,
                           step=1, velocity_exponent=1.0)
        # 2.5x the threshold is a real fraction, not float noise.
        assert _velocity_to_clicks(0.025, cfg) == 3

    def test_direction_is_preserved(self):
        cfg = ScrollConfig(sensitivity=0.010, min_hold_frames=1, cooldown=0.0,
                           step=1, velocity_exponent=1.0)
        assert _velocity_to_clicks(0.02, cfg) > 0
        assert _velocity_to_clicks(-0.02, cfg) < 0


# ---------------------------------------------------------------------------
# Bug 3: the capture queue dropped the newest frame instead of the oldest
# ---------------------------------------------------------------------------

class TestFreshnessQueue:
    def test_put_on_empty_queue_drops_nothing(self):
        q = queue.Queue(maxsize=2)
        assert drop_oldest_put(q, "a") == 0
        assert list(q.queue) == ["a"]

    def test_full_queue_evicts_oldest_and_keeps_newest(self):
        q = queue.Queue(maxsize=2)
        drop_oldest_put(q, "old")
        drop_oldest_put(q, "mid")
        dropped = drop_oldest_put(q, "new")

        assert dropped == 1
        contents = list(q.queue)
        assert "new" in contents, "the freshest item must survive"
        assert "old" not in contents, "the stalest item must be the one dropped"

    def test_repeated_overflow_always_leaves_the_latest(self):
        q = queue.Queue(maxsize=1)
        for i in range(10):
            drop_oldest_put(q, i)
        assert list(q.queue) == [9]

    def test_drop_count_accumulates(self):
        q = queue.Queue(maxsize=1)
        dropped = sum(drop_oldest_put(q, i) for i in range(5))
        assert dropped == 4


# ---------------------------------------------------------------------------
# Bug 6 / startup validation: bindings must cover predictable classes
# ---------------------------------------------------------------------------

class TestBindingValidation:
    def test_full_coverage_passes_silently(self):
        assert validate_bindings([0, 1, 2, 3], [1, 2, 3]) == []

    def test_unbound_predictable_class_is_fatal(self):
        with pytest.raises(BindingError) as exc:
            validate_bindings([0, 1, 2, 3], [1, 2])
        assert "[3]" in str(exc.value)

    def test_binding_for_unpredictable_class_warns(self):
        # The defect this repo shipped with: labels 4 and 5 were bound to
        # Screenshot and Do Not Disturb, but classes_ is [0 1 2 3].
        warnings_ = validate_bindings([0, 1, 2, 3], [1, 2, 3, 4, 5])
        assert len(warnings_) == 2
        assert any("4" in w for w in warnings_)
        assert any("5" in w for w in warnings_)

    def test_neutral_needs_no_binding(self):
        assert validate_bindings([0, 1], [1]) == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pinch(distance: float):
    """21 landmarks where thumb(4) and index(8) sit `distance` apart.

    Includes a real hand scale (0.20) because pinch thresholds are ratios of it.
    """
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=0.75, z=0.0)
    lms[9] = SimpleNamespace(x=0.5, y=0.55, z=0.0)
    lms[4] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    lms[8] = SimpleNamespace(x=0.5 + distance, y=0.5, z=0.0)
    return lms


def _fist(wrist_y: float):
    """A landmark set that satisfies every _is_true_scroll_fist gate."""
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lms[mcp] = SimpleNamespace(x=0.5, y=0.35, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=0.45, z=0.0)
    lms[6] = SimpleNamespace(x=0.5, y=0.40, z=0.0)   # index not extended
    lms[2] = SimpleNamespace(x=0.5, y=0.40, z=0.0)   # thumb not raised
    lms[3] = SimpleNamespace(x=0.5, y=0.42, z=0.0)
    lms[4] = SimpleNamespace(x=0.5, y=0.45, z=0.0)
    return lms


def _armed_scroll(cfg: ScrollConfig) -> ScrollFSM:
    fsm = ScrollFSM(cfg)
    for _ in range(cfg.min_hold_frames):
        fsm.update(_fist(0.5))
    return fsm
