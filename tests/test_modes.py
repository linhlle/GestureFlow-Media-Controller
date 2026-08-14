"""The mode ladder.

Exclusivity used to be emergent: three independent predicates that happened to
exclude each other, checked by a property test. With eight modes that structure
does not survive -- every new mode must be taught to exclude every existing one.

active_mode() returns a single value, so two modes being active at once is
unrepresentable. These tests guard the two things that can still go wrong: the
ladder not being total, and the derived predicates drifting from it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.click_fsm import ClickState
from gestureflow.config import DEFAULT_CONFIG
from gestureflow.inference import InferenceResult
from gestureflow.modes import SUPPRESSES_GEOMETRY, Mode
from gestureflow.scroll_fsm import ScrollState

coord = st.floats(min_value=-0.5, max_value=1.5,
                  allow_nan=False, allow_infinity=False)

landmark_set = st.lists(
    st.tuples(coord, coord,
              st.floats(min_value=-0.5, max_value=0.5,
                        allow_nan=False, allow_infinity=False)),
    min_size=21, max_size=21,
)


def to_landmarks(triples):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in triples]


def make_result(landmarks, *, stable=0, action=None, fsm_active=False,
                right_fsm_active=False, scroll_active=False,
                index_extended=False, thumb_raised=False):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = CaptureResult(frame=frame, landmarks=landmarks,
                            hand_lm_obj=None, timestamp=0.0)
    return InferenceResult(
        capture=capture, stable_gesture=stable, vote_score=0, confidence=1.0,
        raw_prediction=stable, action=action,
        click_fired=False, fsm_active=fsm_active,
        fsm_state=ClickState.HELD if fsm_active else ClickState.IDLE,
        hold_progress=0.0,
        right_click_fired=False, right_fsm_active=right_fsm_active,
        right_fsm_state=ClickState.HELD if right_fsm_active else ClickState.IDLE,
        right_hold_progress=0.0,
        scroll_delta=0, scroll_active=scroll_active,
        scroll_state=ScrollState.SCROLLING if scroll_active else ScrollState.IDLE,
        index_extended=index_extended, thumb_raised=thumb_raised,
    )


class TestLadderIsTotal:
    @given(
        triples=landmark_set,
        stable=st.integers(min_value=0, max_value=3),
        action=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
        fsm_active=st.booleans(),
        right_fsm_active=st.booleans(),
        scroll_active=st.booleans(),
        index_extended=st.booleans(),
        thumb_raised=st.booleans(),
    )
    @settings(max_examples=250, deadline=None)
    def test_always_returns_exactly_one_mode(
        self, triples, stable, action, fsm_active, right_fsm_active,
        scroll_active, index_extended, thumb_raised,
    ):
        router = GestureRouter(DEFAULT_CONFIG)
        mode = router.active_mode(make_result(
            to_landmarks(triples), stable=stable, action=action,
            fsm_active=fsm_active, right_fsm_active=right_fsm_active,
            scroll_active=scroll_active, index_extended=index_extended,
            thumb_raised=thumb_raised,
        ))
        assert isinstance(mode, Mode)

    def test_no_hand_is_always_none(self):
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(None, index_extended=True, thumb_raised=True,
                             scroll_active=True, fsm_active=True)
        assert router.active_mode(result) is Mode.NONE


class TestDerivedPredicatesAgreeWithTheLadder:
    """The wrappers must never disagree with the function they wrap."""

    @given(
        triples=landmark_set,
        stable=st.integers(min_value=0, max_value=3),
        action=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
        fsm_active=st.booleans(),
        right_fsm_active=st.booleans(),
        scroll_active=st.booleans(),
        index_extended=st.booleans(),
        thumb_raised=st.booleans(),
    )
    @settings(max_examples=250, deadline=None)
    def test_each_predicate_is_its_mode(
        self, triples, stable, action, fsm_active, right_fsm_active,
        scroll_active, index_extended, thumb_raised,
    ):
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(
            to_landmarks(triples), stable=stable, action=action,
            fsm_active=fsm_active, right_fsm_active=right_fsm_active,
            scroll_active=scroll_active, index_extended=index_extended,
            thumb_raised=thumb_raised,
        )
        mode = router.active_mode(result)
        assert router.cursor_enabled(result) == (mode is Mode.CURSOR)
        assert router.scroll_enabled(result) == (mode is Mode.SCROLL)
        assert router.volume_enabled(result) == (mode is Mode.VOLUME)

    @given(
        triples=landmark_set,
        fsm_active=st.booleans(),
        scroll_active=st.booleans(),
        index_extended=st.booleans(),
        thumb_raised=st.booleans(),
    )
    @settings(max_examples=200, deadline=None)
    def test_at_most_one_of_the_three_is_ever_enabled(
        self, triples, fsm_active, scroll_active, index_extended, thumb_raised,
    ):
        """The original invariant, now a consequence rather than a hope."""
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(
            to_landmarks(triples), fsm_active=fsm_active,
            scroll_active=scroll_active, index_extended=index_extended,
            thumb_raised=thumb_raised,
        )
        enabled = [router.cursor_enabled(result),
                   router.scroll_enabled(result),
                   router.volume_enabled(result)]
        assert sum(enabled) <= 1


class TestPrecedence:
    """Each rung must beat every rung below it."""

    def _router(self):
        return GestureRouter(DEFAULT_CONFIG)

    def test_command_beats_every_geometric_mode(self):
        for stable, action in ((2, None), (0, 3)):
            result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21),
                                 stable=stable, action=action,
                                 scroll_active=True, fsm_active=True,
                                 index_extended=True, thumb_raised=True)
            assert self._router().active_mode(result) is Mode.COMMAND

    def test_scroll_beats_click_and_cursor(self):
        result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21),
                             scroll_active=True, fsm_active=True,
                             index_extended=True)
        assert self._router().active_mode(result) is Mode.SCROLL

    def test_click_beats_cursor(self):
        result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21),
                             fsm_active=True, index_extended=True)
        assert self._router().active_mode(result) is Mode.CLICK

    def test_right_click_also_beats_cursor(self):
        result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21),
                             right_fsm_active=True, index_extended=True)
        assert self._router().active_mode(result) is Mode.CLICK

    def test_cursor_wins_when_only_the_index_is_up(self):
        result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21),
                             index_extended=True)
        assert self._router().active_mode(result) is Mode.CURSOR

    def test_a_visible_but_idle_hand_is_tracking(self):
        result = make_result(to_landmarks([(0.5, 0.5, 0.0)] * 21))
        assert self._router().active_mode(result) is Mode.TRACKING


class TestVolumePose:
    def test_volume_needs_the_index_down(self):
        lms = to_landmarks([(0.5, 0.5, 0.0)] * 21)
        lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        lms[5] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        result = make_result(lms, thumb_raised=True, index_extended=True)
        assert GestureRouter(DEFAULT_CONFIG).active_mode(result) is Mode.CURSOR

    def test_volume_claims_a_raised_thumb_with_the_index_down(self):
        lms = to_landmarks([(0.5, 0.5, 0.0)] * 21)
        lms[4] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        lms[5] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        result = make_result(lms, thumb_raised=True)
        assert GestureRouter(DEFAULT_CONFIG).active_mode(result) is Mode.VOLUME


class TestModeEnum:
    def test_suppression_set_is_explicit(self):
        assert SUPPRESSES_GEOMETRY == {Mode.NONE, Mode.PAUSED, Mode.COMMAND}

    def test_every_mode_stringifies_to_its_value(self):
        for mode in Mode:
            assert str(mode) == mode.value

    def test_mode_values_are_unique(self):
        values = [m.value for m in Mode]
        assert len(values) == len(set(values))
