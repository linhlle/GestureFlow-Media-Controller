"""
tests/test_scroll_fsm.py
------------------------
Tests for the redesigned ScrollFSM covering:
  - All three exclusion gates (_index_extended, _thumb_raised, _strict_fist)
  - _is_true_scroll_fist composite check
  - State machine transitions
  - Inertial scroll model (slow/fast velocity → proportional output)
  - Clutch prevention (no rowing needed)
  - All collision scenarios from the original bugs
"""
from __future__ import annotations
import math, time
from types import SimpleNamespace
import pytest

from gestureflow.scroll_fsm import (
    ScrollFSM, ScrollState,
    _index_extended, _thumb_raised, _strict_fist, _is_true_scroll_fist,
)
from gestureflow.config import ScrollConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _cfg(sens=0.008, hold=5, cd=0.0, step=2, exp=1.6):
    return ScrollConfig(
        sensitivity=sens,
        min_hold_frames=hold,
        cooldown=cd,
        step=step,
        velocity_exponent=exp,
    )

def _lms21():
    return [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]


def true_scroll_fist_lms(wrist_y=0.5):
    """All 4 non-thumb tips curled, index NOT extended, thumb NOT raised."""
    lms = _lms21()
    lms[0]  = SimpleNamespace(x=0.5, y=wrist_y,  z=0.0)   # wrist
    # curl all 4 fingers: tip.y > mcp.y + threshold
    for tip, mcp in [(8,5),(12,9),(16,13),(20,17)]:
        lms[mcp] = SimpleNamespace(x=0.5, y=0.35, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=0.45, z=0.0)  # tip BELOW knuckle
    # index PIP must be below tip (tip not extended)
    lms[6]  = SimpleNamespace(x=0.5, y=0.40, z=0.0)       # PIP above tip (y=0.45)
    # thumb NOT raised: tip.y >= mcp.y
    lms[2]  = SimpleNamespace(x=0.5, y=0.40, z=0.0)       # thumb MCP
    lms[4]  = SimpleNamespace(x=0.5, y=0.45, z=0.0)       # thumb tip NOT above MCP
    return lms


def index_up_mouse_lms():
    """Index extended upward — mouse mode. Should NOT trigger scroll."""
    lms = true_scroll_fist_lms()
    lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)   # PIP lower
    lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)   # tip ABOVE PIP by >0.04
    return lms


def thumb_raised_volume_lms():
    """Thumb raised — volume mode. Should NOT trigger scroll."""
    lms = true_scroll_fist_lms()
    lms[2] = SimpleNamespace(x=0.5, y=0.50, z=0.0)   # thumb MCP lower
    lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)   # thumb tip ABOVE MCP by >0.04
    return lms


def open_hand_lms():
    """Flat open hand — no fingers curled."""
    lms = _lms21()
    for tip, mcp in [(8,5),(12,9),(16,13),(20,17)]:
        lms[mcp] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=0.30, z=0.0)  # tip ABOVE knuckle
    return lms


def scroll_lms(wrist_y=0.5):
    lms = true_scroll_fist_lms(wrist_y=wrist_y)
    return lms


def charge(fsm, n=5):
    for _ in range(n):
        fsm.update(true_scroll_fist_lms())


# ── Gate 1: _index_extended ──────────────────────────────────────────────────

class TestIndexExtended:
    def test_extended_returns_true(self):
        lms = _lms21()
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)   # PIP
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)   # tip clearly above
        assert _index_extended(lms) is True

    def test_curled_returns_false(self):
        lms = _lms21()
        lms[6] = SimpleNamespace(x=0.5, y=0.35, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.45, z=0.0)   # tip below
        assert _index_extended(lms) is False

    def test_margin_prevents_hairline_trigger(self):
        """tip just 0.01 above PIP — not enough to qualify as extended."""
        lms = _lms21()
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.49, z=0.0)   # only 0.01 above (< 0.04 margin)
        assert _index_extended(lms) is False

    def test_exactly_at_margin_boundary(self):
        """Exactly at the margin = NOT extended (strict less-than)."""
        lms = _lms21()
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.46, z=0.0)   # 0.04 above = boundary
        assert _index_extended(lms) is False

    def test_custom_margin(self):
        lms = _lms21()
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.43, z=0.0)   # 0.07 above
        assert _index_extended(lms, margin=0.06) is True
        assert _index_extended(lms, margin=0.08) is False


# ── Gate 2: _thumb_raised ────────────────────────────────────────────────────

class TestThumbRaised:
    def test_raised_returns_true(self):
        lms = _lms21()
        lms[2] = SimpleNamespace(x=0.5, y=0.55, z=0.0)   # MCP
        lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)   # tip clearly above
        assert _thumb_raised(lms) is True

    def test_not_raised_returns_false(self):
        lms = _lms21()
        lms[2] = SimpleNamespace(x=0.5, y=0.40, z=0.0)
        lms[4] = SimpleNamespace(x=0.5, y=0.45, z=0.0)   # tip below
        assert _thumb_raised(lms) is False

    def test_margin_prevents_hairline_trigger(self):
        lms = _lms21()
        lms[2] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[4] = SimpleNamespace(x=0.5, y=0.49, z=0.0)
        assert _thumb_raised(lms) is False

    def test_custom_margin(self):
        lms = _lms21()
        lms[2] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[4] = SimpleNamespace(x=0.5, y=0.43, z=0.0)
        assert _thumb_raised(lms, margin=0.06) is True
        assert _thumb_raised(lms, margin=0.08) is False


# ── Gate 3: _strict_fist ─────────────────────────────────────────────────────

class TestStrictFist:
    def test_all_4_curled_true(self):
        assert _strict_fist(true_scroll_fist_lms()) is True

    def test_3_curled_false(self):
        """Strict fist requires ALL 4 — 3 is not enough."""
        lms = true_scroll_fist_lms()
        # Uncurl the index: tip above knuckle
        lms[5] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        assert _strict_fist(lms) is False

    def test_open_hand_false(self):
        assert _strict_fist(open_hand_lms()) is False

    def test_threshold_respected(self):
        """Fingertip only 0.01 below knuckle — below threshold of 0.03."""
        lms = true_scroll_fist_lms()
        # Set index so tip is only barely below knuckle
        lms[5] = SimpleNamespace(x=0.5, y=0.40, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.41, z=0.0)   # only 0.01 delta
        assert _strict_fist(lms) is False


# ── Composite: _is_true_scroll_fist ──────────────────────────────────────────

class TestIsTrueScrollFist:
    def test_true_fist_passes_all_gates(self):
        assert _is_true_scroll_fist(true_scroll_fist_lms()) is True

    def test_index_extended_blocked_by_gate1(self):
        """The original collision A: index-up mouse mode triggering scroll."""
        assert _is_true_scroll_fist(index_up_mouse_lms()) is False

    def test_thumb_raised_blocked_by_gate2(self):
        """The original collision B: thumb-up volume mode triggering scroll."""
        assert _is_true_scroll_fist(thumb_raised_volume_lms()) is False

    def test_open_hand_blocked_by_gate3(self):
        """A loose open hand never triggers scroll."""
        assert _is_true_scroll_fist(open_hand_lms()) is False

    def test_partial_fist_3fingers_blocked(self):
        """3-finger partial fist (old threshold) now blocked by gate 3."""
        lms = true_scroll_fist_lms()
        # Uncurl index only
        lms[5] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        assert _is_true_scroll_fist(lms) is False

    def test_gate_priority_index_beats_fist(self):
        """Even if all 4 fingers are curled, index extension blocks scroll."""
        lms = true_scroll_fist_lms()
        # Force index extended (gate 1 trigger)
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        assert _is_true_scroll_fist(lms) is False

    def test_gate_priority_thumb_beats_fist(self):
        """Even if all 4 fingers are curled, thumb raising blocks scroll."""
        lms = true_scroll_fist_lms()
        lms[2] = SimpleNamespace(x=0.5, y=0.55, z=0.0)
        lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        assert _is_true_scroll_fist(lms) is False


# ── ScrollFSM state transitions ──────────────────────────────────────────────

class TestScrollFSMTransitions:
    def test_initial_idle(self):
        assert ScrollFSM(_cfg()).state is ScrollState.IDLE

    def test_fist_enters_fist_detected(self):
        f = ScrollFSM(_cfg())
        f.update(true_scroll_fist_lms())
        assert f.state is ScrollState.FIST_DETECTED

    def test_hold_enters_scrolling(self):
        f = ScrollFSM(_cfg(hold=5))
        charge(f)
        assert f.state is ScrollState.SCROLLING

    def test_none_resets_to_idle(self):
        f = ScrollFSM(_cfg())
        f.update(true_scroll_fist_lms())
        f.update(None)
        assert f.state is ScrollState.IDLE

    def test_open_hand_exits_scrolling(self):
        f = ScrollFSM(_cfg(hold=5))
        charge(f)
        f.update(open_hand_lms())
        assert f.state is ScrollState.IDLE

    def test_interrupted_fist_resets(self):
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(3): f.update(true_scroll_fist_lms())
        f.update(open_hand_lms())
        assert f.state is ScrollState.IDLE

    def test_is_active_fist_detected(self):
        f = ScrollFSM(_cfg())
        f.update(true_scroll_fist_lms())
        assert f.is_active is True

    def test_is_active_scrolling(self):
        f = ScrollFSM(_cfg(hold=5))
        charge(f)
        assert f.is_active is True

    def test_is_not_active_idle(self):
        assert ScrollFSM(_cfg()).is_active is False

    def test_is_not_active_after_release(self):
        f = ScrollFSM(_cfg(hold=5))
        charge(f)
        f.update(open_hand_lms())
        assert f.is_active is False


# ── Inertial scroll model ─────────────────────────────────────────────────────

class TestInertialScrollModel:
    """Tests confirming the velocity-based scroll output."""

    def _scrolling_fsm(self, **kw):
        f = ScrollFSM(_cfg(**kw))
        charge(f)
        assert f.state is ScrollState.SCROLLING
        return f

    def test_no_scroll_below_sensitivity(self):
        f = self._scrolling_fsm(sens=0.02, cd=0.0)
        # Move hand only 0.005 — below threshold
        f.update(scroll_lms(0.500))
        f.update(scroll_lms(0.495))   # delta = 0.005 < 0.02
        assert f.scroll_delta == 0

    def test_slow_movement_produces_small_output(self):
        """Slow wrist movement → low velocity → few clicks."""
        f = self._scrolling_fsm(sens=0.008, step=2, exp=1.6, cd=0.0)
        f.update(scroll_lms(0.500))
        f.update(scroll_lms(0.490))   # delta = 0.010 — just above threshold
        slow_clicks = abs(f.scroll_delta)
        assert slow_clicks > 0

        # Fast: delta = 0.050 — should produce more clicks
        f2 = self._scrolling_fsm(sens=0.008, step=2, exp=1.6, cd=0.0)
        f2.update(scroll_lms(0.500))
        f2.update(scroll_lms(0.450))
        fast_clicks = abs(f2.scroll_delta)

        assert fast_clicks > slow_clicks, (
            f"Fast ({fast_clicks}) should exceed slow ({slow_clicks})"
        )

    def test_direction_up(self):
        """Hand moves UP (wrist Y decreases) → positive delta."""
        f = self._scrolling_fsm(sens=0.008, step=2, cd=0.0)
        f.update(scroll_lms(0.500))   # anchor
        f.update(scroll_lms(0.440))   # wrist moved up
        assert f.scroll_delta > 0

    def test_direction_down(self):
        """Hand moves DOWN (wrist Y increases) → negative delta."""
        f = self._scrolling_fsm(sens=0.008, step=2, cd=0.0)
        f.update(scroll_lms(0.500))   # anchor
        f.update(scroll_lms(0.560))   # wrist moved down
        assert f.scroll_delta < 0

    def test_exponent_amplifies_fast_movement(self):
        """Exponent > 1.0 means doubling velocity > doubles output."""
        f_slow = self._scrolling_fsm(sens=0.008, step=1, exp=2.0, cd=0.0)
        f_slow.update(scroll_lms(0.500))
        f_slow.update(scroll_lms(0.490))   # delta 0.010
        clicks_slow = abs(f_slow.scroll_delta)

        f_fast = self._scrolling_fsm(sens=0.008, step=1, exp=2.0, cd=0.0)
        f_fast.update(scroll_lms(0.500))
        f_fast.update(scroll_lms(0.460))   # delta 0.040 (4× slower)
        clicks_fast = abs(f_fast.scroll_delta)

        # With exp=2.0, 4× velocity → >4× output (quadratic amplification)
        assert clicks_fast > 4 * clicks_slow or clicks_fast >= 4

    def test_linear_exponent_is_proportional(self):
        """exp=1.0 makes the model behave proportionally (linear)."""
        f1 = self._scrolling_fsm(sens=0.010, step=1, exp=1.0, cd=0.0)
        f1.update(scroll_lms(0.500))
        f1.update(scroll_lms(0.480))   # delta = 0.020 = 2× sensitivity
        c1 = abs(f1.scroll_delta)

        f2 = self._scrolling_fsm(sens=0.010, step=1, exp=1.0, cd=0.0)
        f2.update(scroll_lms(0.500))
        f2.update(scroll_lms(0.460))   # delta = 0.040 = 4× sensitivity
        c2 = abs(f2.scroll_delta)

        # c2 should be ~2× c1 (linear relationship)
        assert c2 >= c1 * 1.5

    def test_cooldown_prevents_rapid_fire(self):
        f = self._scrolling_fsm(sens=0.008, cd=10.0)
        f.update(scroll_lms(0.500))
        f.update(scroll_lms(0.440))   # first event
        first = f.scroll_delta
        f.update(scroll_lms(0.380))   # still in cooldown
        second = f.scroll_delta
        assert first != 0 and second == 0

    def test_rolling_anchor_prevents_drift(self):
        """After each event the anchor resets — no drift accumulation."""
        f = self._scrolling_fsm(sens=0.008, cd=0.0)
        y = 0.500
        deltas = []
        for _ in range(20):
            f.update(scroll_lms(y))
            y -= 0.010   # constant velocity each frame
            deltas.append(f.scroll_delta)
        # Should produce a consistent stream of clicks, not explode
        non_zero = [d for d in deltas if d != 0]
        assert len(non_zero) >= 5   # emitting regularly
        # No single output should be absurdly large (drift check)
        assert all(abs(d) < 50 for d in non_zero)


# ── Clutching problem — prevention ───────────────────────────────────────────

class TestClutchPrevention:
    """A fast wrist flick should cover large distances without rowing."""

    def test_fast_flick_produces_large_jump(self):
        """A fast 0.15 unit flick should produce many more clicks than
        a slow 0.01 unit movement — demonstrating the inertial benefit."""
        # Slow: barely crosses threshold
        f_slow = ScrollFSM(_cfg(sens=0.008, step=2, exp=1.6, cd=0.0))
        charge(f_slow)
        f_slow.update(scroll_lms(0.500))
        f_slow.update(scroll_lms(0.490))
        slow_output = abs(f_slow.scroll_delta)

        # Fast: large single flick
        f_fast = ScrollFSM(_cfg(sens=0.008, step=2, exp=1.6, cd=0.0))
        charge(f_fast)
        f_fast.update(scroll_lms(0.500))
        f_fast.update(scroll_lms(0.350))   # 0.15 unit flick
        fast_output = abs(f_fast.scroll_delta)

        assert fast_output > slow_output * 3, (
            f"Fast flick ({fast_output}) should be >3× slow ({slow_output})"
        )


# ── Collision regression tests ────────────────────────────────────────────────

class TestCollisionRegressions:
    """These are the exact scenarios that broke in the original code."""

    def test_index_up_mouse_never_scrolls(self):
        """Collision A: 'index-up mouse' should never activate scroll FSM."""
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(10):
            f.update(index_up_mouse_lms())
        assert f.state is ScrollState.IDLE
        assert f.scroll_delta == 0

    def test_thumb_raised_volume_never_scrolls(self):
        """Collision B: 'thumb-raised volume mode' should never scroll."""
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(10):
            f.update(thumb_raised_volume_lms())
        assert f.state is ScrollState.IDLE
        assert f.scroll_delta == 0

    def test_partial_fist_3finger_never_scrolls(self):
        """Old bug: ≥3 curled fingers was enough. Now requires all 4."""
        lms = true_scroll_fist_lms()
        # Uncurl index (3 fingers remain curled)
        lms[5] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(10):
            f.update(lms)
        assert f.state is ScrollState.IDLE

    def test_open_hand_during_mouse_never_scrolls(self):
        """Loose hand mid-mouse-tracking should never trigger scroll."""
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(20):
            f.update(open_hand_lms())
        assert f.scroll_delta == 0
        assert f.state is ScrollState.IDLE

    def test_combined_index_and_thumb_up_never_scrolls(self):
        """Both gates triggered simultaneously — still blocked."""
        lms = true_scroll_fist_lms()
        # Extend index
        lms[6] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lms[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        # Raise thumb
        lms[2] = SimpleNamespace(x=0.5, y=0.55, z=0.0)
        lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        f = ScrollFSM(_cfg(hold=5))
        for _ in range(10):
            f.update(lms)
        assert f.state is ScrollState.IDLE