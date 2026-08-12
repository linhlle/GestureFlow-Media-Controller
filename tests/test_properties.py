"""Property-based tests for the invariants the architecture depends on.

Example-based tests prove the cases someone thought to write down.  These
generate arbitrary hand configurations and assert the properties must hold for
all of them -- which is the right tool here, because the interesting failures
in this system are collisions between modes at landmark positions nobody
anticipated.

The headline property is MODE EXCLUSIVITY.  Cursor, scroll, and volume are
three interpretations of the same hand, separated only by geometric gates.  If
two can be active at once, the user gets a cursor that jumps while the page
scrolls, and no example-based test is likely to find the configuration that
does it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.click_fsm import ClickFSM, ClickState
from gestureflow.config import ClickConfig, DEFAULT_CONFIG, ScrollConfig
from gestureflow.inference import InferenceResult
from gestureflow.scroll_fsm import ScrollFSM, ScrollState
from gestureflow.utils import normalize_landmarks

# Landmarks live in normalized image space; MediaPipe can report slightly
# outside [0, 1] when a hand is partly out of frame, so the generators allow it.
coord = st.floats(min_value=-0.5, max_value=1.5, allow_nan=False,
                  allow_infinity=False)

landmark_set = st.lists(
    st.tuples(coord, coord, st.floats(min_value=-0.5, max_value=0.5,
                                      allow_nan=False, allow_infinity=False)),
    min_size=21, max_size=21,
)


def to_landmarks(triples):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in triples]


def make_result(landmarks, *, stable=0, fsm_active=False,
                right_fsm_active=False, scroll_active=False,
                index_extended=False, thumb_raised=False,
                scroll_delta=0, click_fired=False, right_click_fired=False,
                action=None, timestamp=0.0):
    """Build an InferenceResult without running the model."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = CaptureResult(frame=frame, landmarks=landmarks,
                            hand_lm_obj=None, timestamp=timestamp)
    return InferenceResult(
        capture=capture,
        stable_gesture=stable,
        vote_score=0,
        confidence=1.0,
        raw_prediction=stable,
        action=action,
        click_fired=click_fired,
        fsm_active=fsm_active,
        fsm_state=ClickState.HELD if fsm_active else ClickState.IDLE,
        hold_progress=1.0 if fsm_active else 0.0,
        right_click_fired=right_click_fired,
        right_fsm_active=right_fsm_active,
        right_fsm_state=ClickState.HELD if right_fsm_active else ClickState.IDLE,
        right_hold_progress=1.0 if right_fsm_active else 0.0,
        scroll_delta=scroll_delta,
        scroll_active=scroll_active,
        scroll_state=ScrollState.SCROLLING if scroll_active else ScrollState.IDLE,
        index_extended=index_extended,
        thumb_raised=thumb_raised,
    )


# ---------------------------------------------------------------------------
# MODE EXCLUSIVITY
# ---------------------------------------------------------------------------

class TestModeExclusivity:
    """At most one of {cursor, scroll, volume} may be active on any frame."""

    @given(
        triples=landmark_set,
        stable=st.integers(min_value=0, max_value=3),
        fsm_active=st.booleans(),
        right_fsm_active=st.booleans(),
        scroll_active=st.booleans(),
        index_extended=st.booleans(),
        thumb_raised=st.booleans(),
    )
    @settings(max_examples=250, deadline=None)
    def test_at_most_one_mode_is_ever_enabled(
        self, triples, stable, fsm_active, right_fsm_active,
        scroll_active, index_extended, thumb_raised,
    ):
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(
            to_landmarks(triples), stable=stable, fsm_active=fsm_active,
            right_fsm_active=right_fsm_active, scroll_active=scroll_active,
            index_extended=index_extended, thumb_raised=thumb_raised,
        )

        enabled = [
            router.cursor_enabled(result),
            router.scroll_enabled(result),
            router.volume_enabled(result),
        ]
        assert sum(enabled) <= 1, (
            f"modes overlapped: cursor={enabled[0]} scroll={enabled[1]} "
            f"volume={enabled[2]}"
        )

    @given(triples=landmark_set, stable=st.integers(min_value=1, max_value=3))
    @settings(max_examples=120, deadline=None)
    def test_named_gesture_disables_every_geometric_mode(self, triples, stable):
        # A recognized command gesture must park cursor, scroll, and volume --
        # otherwise raising a hand to trigger Spotlight also flings the cursor.
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(to_landmarks(triples), stable=stable,
                             index_extended=True, thumb_raised=True,
                             scroll_active=True)
        assert not router.cursor_enabled(result)
        assert not router.scroll_enabled(result)
        assert not router.volume_enabled(result)

    @given(triples=landmark_set)
    @settings(max_examples=120, deadline=None)
    def test_no_hand_means_no_mode(self, triples):
        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(None, index_extended=True, thumb_raised=True,
                             scroll_active=True)
        assert not router.cursor_enabled(result)
        assert not router.scroll_enabled(result)
        assert not router.volume_enabled(result)

    @given(triples=landmark_set, thumb_raised=st.booleans())
    @settings(max_examples=120, deadline=None)
    def test_pinching_suppresses_cursor(self, triples, thumb_raised):
        # While a click pinch is charging the cursor must hold still, or the
        # click lands somewhere other than where the user aimed.
        router = GestureRouter(DEFAULT_CONFIG)
        for active in ("fsm_active", "right_fsm_active"):
            result = make_result(to_landmarks(triples), index_extended=True,
                                 thumb_raised=thumb_raised, **{active: True})
            assert not router.cursor_enabled(result)

    @given(
        triples=landmark_set,
        stable=st.integers(min_value=0, max_value=3),
        fsm_active=st.booleans(),
        scroll_active=st.booleans(),
        index_extended=st.booleans(),
        thumb_raised=st.booleans(),
    )
    @settings(max_examples=150, deadline=None)
    def test_route_never_emits_two_continuous_actions(
        self, triples, stable, fsm_active, scroll_active,
        index_extended, thumb_raised,
    ):
        from gestureflow import actions as act

        router = GestureRouter(DEFAULT_CONFIG)
        result = make_result(
            to_landmarks(triples), stable=stable, fsm_active=fsm_active,
            scroll_active=scroll_active, index_extended=index_extended,
            thumb_raised=thumb_raised,
        )
        emitted = router.route(result, now=1.0)
        moves = [a for a in emitted if isinstance(a, act.MoveCursor)]
        volumes = [a for a in emitted if isinstance(a, act.SetVolume)]
        assert len(moves) <= 1
        assert len(volumes) <= 1
        assert not (moves and volumes), "cursor and volume both moved"


# ---------------------------------------------------------------------------
# ClickFSM invariants
# ---------------------------------------------------------------------------

distance_seq = st.lists(
    st.floats(min_value=0.0, max_value=0.4, allow_nan=False,
              allow_infinity=False),
    min_size=1, max_size=80,
)


def _pinch_lms(distance: float):
    out = [SimpleNamespace(x=float(i), y=float(i), z=0.0) for i in range(21)]
    out[4] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    out[8] = SimpleNamespace(x=0.5 + distance, y=0.5, z=0.0)
    return out


class TestClickInvariants:
    @given(distances=distance_seq)
    @settings(max_examples=150, deadline=None)
    def test_click_only_ever_follows_a_held_state(self, distances):
        """A click must be preceded by HELD in the previous frame.

        This is the release-edge contract: no sequence of distances, however
        adversarial, may produce a click from IDLE or PRESSING.
        """
        fsm = ClickFSM(ClickConfig(close_threshold=0.045, open_threshold=0.065,
                                   min_hold_frames=3, cooldown=0.0))
        previous_state = fsm.state
        for d in distances:
            fsm.update(_pinch_lms(d))
            if fsm.click_fired:
                assert previous_state is ClickState.HELD, (
                    f"click fired out of {previous_state}, not HELD"
                )
            previous_state = fsm.state

    @given(distances=distance_seq)
    @settings(max_examples=150, deadline=None)
    def test_no_two_clicks_within_cooldown(self, distances):
        clock = _Clock()
        cooldown = 0.4
        fsm = ClickFSM(
            ClickConfig(close_threshold=0.045, open_threshold=0.065,
                        min_hold_frames=2, cooldown=cooldown),
            clock=clock,
        )
        fire_times = []
        for d in distances:
            clock.now += 1 / 30.0     # a frame at 30 FPS
            fsm.update(_pinch_lms(d))
            if fsm.click_fired:
                fire_times.append(clock.now)

        gaps = [b - a for a, b in zip(fire_times, fire_times[1:])]
        assert all(g >= cooldown - 1e-9 for g in gaps), (
            f"clicks fired {min(gaps, default=0):.4f}s apart, "
            f"cooldown is {cooldown}s"
        )

    @given(distances=distance_seq)
    @settings(max_examples=120, deadline=None)
    def test_hold_progress_stays_in_unit_range(self, distances):
        fsm = ClickFSM(ClickConfig(min_hold_frames=4))
        for d in distances:
            fsm.update(_pinch_lms(d))
            assert 0.0 <= fsm.hold_progress <= 1.0

    @given(distances=distance_seq)
    @settings(max_examples=120, deadline=None)
    def test_is_active_matches_state(self, distances):
        fsm = ClickFSM(ClickConfig(min_hold_frames=3))
        for d in distances:
            fsm.update(_pinch_lms(d))
            expected = fsm.state in (ClickState.PRESSING, ClickState.HELD)
            assert fsm.is_active is expected


class _Clock:
    def __init__(self, start: float = 500.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

class TestNormalizationInvariants:
    @given(triples=landmark_set)
    @settings(max_examples=150, deadline=None)
    def test_always_63_values_in_unit_range(self, triples):
        out = normalize_landmarks(to_landmarks(triples))
        assert len(out) == 63
        assert all(-1.0 <= v <= 1.0 for v in out)

    @given(triples=landmark_set)
    @settings(max_examples=150, deadline=None)
    def test_wrist_is_always_the_origin(self, triples):
        out = normalize_landmarks(to_landmarks(triples))
        assert out[:3] == [0.0, 0.0, 0.0]

    @given(
        triples=landmark_set,
        dx=st.floats(min_value=-2, max_value=2, allow_nan=False),
        dy=st.floats(min_value=-2, max_value=2, allow_nan=False),
        dz=st.floats(min_value=-2, max_value=2, allow_nan=False),
    )
    @settings(max_examples=150, deadline=None)
    def test_translation_invariant(self, triples, dx, dy, dz):
        """Moving the whole hand must not change the feature vector.

        This is what lets the classifier work regardless of where in frame the
        hand is, and it is the reason training data collected in one corner
        generalizes to the other.
        """
        base = normalize_landmarks(to_landmarks(triples))
        shifted = normalize_landmarks(to_landmarks(
            [(x + dx, y + dy, z + dz) for x, y, z in triples]
        ))
        assert base == pytest.approx(shifted, abs=1e-5)

    @given(
        triples=landmark_set,
        scale=st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=150, deadline=None)
    def test_scale_invariant(self, triples, scale):
        """A hand twice as far from the camera must produce the same vector."""
        base = normalize_landmarks(to_landmarks(triples))
        # Scale about the wrist so this is a pure size change.
        wx, wy, wz = triples[0]
        scaled = normalize_landmarks(to_landmarks([
            (wx + (x - wx) * scale,
             wy + (y - wy) * scale,
             wz + (z - wz) * scale)
            for x, y, z in triples
        ]))
        assume(max(abs(v) for v in base) > 0)
        assert base == pytest.approx(scaled, abs=1e-4)

    @given(triples=landmark_set)
    @settings(max_examples=120, deadline=None)
    def test_output_is_never_nan(self, triples):
        out = normalize_landmarks(to_landmarks(triples))
        assert not any(math.isnan(v) or math.isinf(v) for v in out)


# ---------------------------------------------------------------------------
# ScrollFSM invariants
# ---------------------------------------------------------------------------

class TestScrollInvariants:
    @given(
        wrist_ys=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1, max_size=60,
        ),
    )
    @settings(max_examples=120, deadline=None)
    def test_scroll_never_fires_from_idle(self, wrist_ys):
        fsm = ScrollFSM(ScrollConfig(min_hold_frames=5, cooldown=0.0))
        for y in wrist_ys:
            fsm.update(_open_hand(y))       # never a fist
            assert fsm.scroll_delta == 0
            assert fsm.state is ScrollState.IDLE

    @given(
        wrist_ys=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=10, max_size=60,
        ),
    )
    @settings(max_examples=120, deadline=None)
    def test_delta_is_zero_whenever_not_scrolling(self, wrist_ys):
        fsm = ScrollFSM(ScrollConfig(min_hold_frames=5, cooldown=0.0))
        for y in wrist_ys:
            fsm.update(_fist_lms(y))
            if fsm.state is not ScrollState.SCROLLING:
                assert fsm.scroll_delta == 0


def _fist_lms(wrist_y: float):
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lm[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lm[mcp] = SimpleNamespace(x=0.5, y=0.35, z=0.0)
        lm[tip] = SimpleNamespace(x=0.5, y=0.45, z=0.0)
    lm[6] = SimpleNamespace(x=0.5, y=0.40, z=0.0)
    lm[2] = SimpleNamespace(x=0.5, y=0.40, z=0.0)
    lm[4] = SimpleNamespace(x=0.5, y=0.45, z=0.0)
    return lm


def _open_hand(wrist_y: float):
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lm[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lm[mcp] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        lm[tip] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
    return lm
