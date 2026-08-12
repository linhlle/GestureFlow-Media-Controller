"""
tests/test_debouncer.py
-----------------------
Unit tests for gestureflow.debouncer.GestureDebouncer.

All tests use a custom DebounceConfig with a very short cooldown (0.0) so
we don't have to sleep in tests.
"""

from __future__ import annotations

import time

import pytest

from gestureflow.config import DebounceConfig
from gestureflow.debouncer import GestureDebouncer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_debouncer(
    window: int = 10,
    threshold: int = 7,
    cooldown: float = 0.0,
    confidence: float = 0.8,
) -> GestureDebouncer:
    cfg = DebounceConfig(
        vote_window_size=window,
        vote_threshold=threshold,
        cmd_cooldown=cooldown,
    )
    return GestureDebouncer(config=cfg, confidence_threshold=confidence)


# ---------------------------------------------------------------------------
# Basic voting
# ---------------------------------------------------------------------------

class TestVoting:
    def test_no_action_below_threshold(self):
        db = _make_debouncer(window=10, threshold=7)
        for _ in range(6):
            action = db.update(1, 0.9)
        assert action is None

    def test_action_fires_at_threshold(self):
        db = _make_debouncer(window=10, threshold=7, cooldown=0.0)
        action = None
        for _ in range(7):
            action = db.update(1, 0.9)
        assert action == 1

    def test_neutral_never_fires_action(self):
        db = _make_debouncer(window=10, threshold=1, cooldown=0.0)
        for _ in range(10):
            action = db.update(0, 1.0)
        assert action is None

    def test_low_confidence_counts_as_neutral(self):
        """Predictions below confidence threshold should not build votes."""
        db = _make_debouncer(window=10, threshold=7, cooldown=0.0)
        action = None
        # Feed 10 low-confidence predictions for gesture 1
        for _ in range(10):
            action = db.update(1, confidence=0.5)  # below default 0.8
        assert action is None

    def test_window_is_rolling(self):
        """Old votes fall off as the window fills with new ones."""
        db = _make_debouncer(window=5, threshold=4, cooldown=0.0)
        # 4 votes fires gesture 1 and clears history
        for _ in range(4):
            db.update(1, 0.9)
        # Now flood with Neutral — gesture 1 must no longer be stable
        for _ in range(6):
            db.update(0, 1.0)
        assert db.stable_gesture == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_second_fire_blocked_during_cooldown(self):
        db = _make_debouncer(window=10, threshold=7, cooldown=10.0)
        # First fire
        for _ in range(7):
            db.update(1, 0.9)
        # Second attempt immediately after — still in cooldown
        action = None
        for _ in range(7):
            action = db.update(1, 0.9)
        assert action is None

    def test_fires_again_after_cooldown(self):
        db = _make_debouncer(window=10, threshold=7, cooldown=0.01)
        for _ in range(7):
            db.update(1, 0.9)
        time.sleep(0.02)   # wait out the 10ms cooldown
        action = None
        for _ in range(7):
            action = db.update(1, 0.9)
        assert action == 1


# ---------------------------------------------------------------------------
# State & properties
# ---------------------------------------------------------------------------

class TestState:
    def test_vote_score_increases(self):
        db = _make_debouncer()
        db.update(1, 0.9)
        db.update(1, 0.9)
        assert db.vote_score == 2

    def test_stable_gesture_reflects_majority(self):
        db = _make_debouncer()
        for _ in range(3):
            db.update(2, 0.9)
        for _ in range(1):
            db.update(1, 0.9)
        assert db.stable_gesture == 2

    def test_reset_clears_history(self):
        db = _make_debouncer()
        for _ in range(5):
            db.update(1, 0.9)
        db.reset()
        assert db.vote_score == 0
        assert db.stable_gesture == 0

    def test_history_cleared_after_fire(self):
        """After a gesture fires the history is cleared to prevent re-fire."""
        db = _make_debouncer(window=10, threshold=7, cooldown=0.0)
        for _ in range(7):
            db.update(1, 0.9)
        # History cleared — next single update should not fire
        action = db.update(1, 0.9)
        assert action is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_different_gestures_compete(self):
        """Tie-breaking: whichever label Counter picks as most_common wins."""
        db = _make_debouncer(window=10, threshold=7, cooldown=0.0)
        for _ in range(5):
            db.update(1, 0.9)
        for _ in range(5):
            db.update(2, 0.9)
        # 5 vs 5 — neither exceeds threshold of 7, so no action
        assert db.update(1, 0.9) is None

    def test_window_size_property(self):
        db = _make_debouncer(window=15)
        assert db.window_size == 15