"""Drag and drop.

The risk here is not that drag fails to work -- it is that adding a state to
ClickFSM changes what a plain click does. So the first class asserts the click
path is untouched, and the rest cover the new edges.

The other thing worth being paranoid about: a drag that starts and never ends
leaves the mouse button physically held down, which the user cannot undo without
quitting the app. Several tests exist purely to prove that cannot happen.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gestureflow import actions as act
from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.click_fsm import ClickFSM, ClickState
from gestureflow.config import AppConfig, ClickConfig, DragConfig
from gestureflow.inference import InferenceThread
from gestureflow.modes import Mode

HAND_SCALE = 0.20


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cfg(close=0.28, open_=0.41, hold=4, cooldown=0.0) -> ClickConfig:
    return ClickConfig(close_threshold=close, open_threshold=open_,
                       min_hold_frames=hold, cooldown=cooldown)


def lms(distance: float):
    """A realistic hand with thumb(4) and index(8) `distance` apart."""
    out = [SimpleNamespace(x=0.5, y=0.65, z=0.0) for _ in range(21)]
    out[0] = SimpleNamespace(x=0.5, y=0.75, z=0.0)
    out[9] = SimpleNamespace(x=0.5, y=0.55, z=0.0)      # hand_scale 0.20
    out[4] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    out[8] = SimpleNamespace(x=0.5 + distance, y=0.5, z=0.0)
    out[12] = SimpleNamespace(x=0.9, y=0.9, z=0.0)      # middle far away
    return out


PINCH = 0.01        # well inside close (0.28 * 0.20 = 0.056)
OPEN = 0.20         # well outside open (0.41 * 0.20 = 0.082)


def _drag_fsm(clock, hold_seconds=0.55, enabled=True, hold_frames=4):
    return ClickFSM(cfg(hold=hold_frames), clock=clock,
                    drag=DragConfig(enabled=enabled, hold_seconds=hold_seconds))


# ---------------------------------------------------------------------------
# The click path must be bit-for-bit unchanged
# ---------------------------------------------------------------------------

class TestClickStillWorks:
    def test_a_quick_pinch_and_release_is_still_a_click(self):
        clock = Clock()
        fsm = _drag_fsm(clock)
        for _ in range(5):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert not fsm.drag_started, "a short pinch must not start a drag"
        fsm.update(lms(OPEN))
        assert fsm.click_fired
        assert not fsm.drag_ended

    def test_a_click_never_emits_drag_edges(self):
        clock = Clock()
        fsm = _drag_fsm(clock)
        edges = []
        for _ in range(5):
            fsm.update(lms(PINCH))
            edges.append((fsm.drag_started, fsm.drag_ended))
            clock.advance(1 / 30.0)
        fsm.update(lms(OPEN))
        edges.append((fsm.drag_started, fsm.drag_ended))
        assert not any(a or b for a, b in edges)

    def test_disabling_drag_restores_the_old_behaviour_exactly(self):
        """With drag off, holding forever must still never click or press."""
        clock = Clock()
        fsm = _drag_fsm(clock, enabled=False)
        for _ in range(200):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
            assert not fsm.click_fired
            assert not fsm.drag_started
        assert fsm.state is ClickState.HELD


# ---------------------------------------------------------------------------
# The new edges
# ---------------------------------------------------------------------------

class TestDragEdges:
    def _hold_into_drag(self, fsm, clock, frames=40):
        for _ in range(frames):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)

    def test_holding_past_the_threshold_presses_once(self):
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        presses = 0
        for _ in range(60):
            fsm.update(lms(PINCH))
            presses += fsm.drag_started
            clock.advance(1 / 30.0)
        assert presses == 1
        assert fsm.dragging
        assert fsm.state is ClickState.DRAGGING

    def test_releasing_a_drag_releases_once(self):
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        self._hold_into_drag(fsm, clock)
        assert fsm.dragging

        fsm.update(lms(OPEN))
        assert fsm.drag_ended
        assert not fsm.dragging
        assert fsm.state is ClickState.IDLE

    def test_releasing_a_drag_does_not_also_click(self):
        """A drop is not a click. Emitting both would double-activate."""
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        self._hold_into_drag(fsm, clock)
        fsm.update(lms(OPEN))
        assert fsm.drag_ended
        assert not fsm.click_fired

    def test_the_drag_timer_starts_from_held_not_from_first_contact(self):
        """Otherwise min_hold_frames and the drag threshold race each other."""
        clock = Clock()
        # 10 frames to reach HELD, then 0.5s more before the drag starts.
        fsm = _drag_fsm(clock, hold_seconds=0.5, hold_frames=10)
        for _ in range(10):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert fsm.state is ClickState.HELD
        assert not fsm.drag_started

        for _ in range(14):                      # ~0.47s, still short
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert not fsm.drag_started

        for _ in range(4):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert fsm.dragging

    def test_drag_progress_climbs_then_pins(self):
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        seen = []
        for _ in range(30):
            fsm.update(lms(PINCH))
            seen.append(fsm.drag_progress)
            clock.advance(1 / 30.0)
        assert seen == sorted(seen)
        assert seen[-1] == 1.0

    def test_progress_is_zero_when_drag_is_disabled(self):
        clock = Clock()
        fsm = _drag_fsm(clock, enabled=False)
        for _ in range(30):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert fsm.drag_progress == 0.0


class TestDragCannotStick:
    """A button left down is unrecoverable without quitting. It must not happen."""

    def test_hand_leaving_frame_mid_drag_releases(self):
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        for _ in range(40):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert fsm.dragging

        fsm.update(None)
        assert fsm.drag_ended, "walking away from the camera left the button down"
        assert not fsm.dragging

    def test_a_degenerate_hand_mid_drag_releases(self):
        """Tracking collapsing is the same situation as the hand vanishing."""
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.5)
        for _ in range(40):
            fsm.update(lms(PINCH))
            clock.advance(1 / 30.0)
        assert fsm.dragging

        collapsed = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        fsm.update(collapsed)
        assert fsm.drag_ended
        assert not fsm.dragging

    @given(distances=st.lists(
        st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        min_size=1, max_size=120))
    @settings(max_examples=200, deadline=None)
    def test_presses_and_releases_always_balance(self, distances):
        """Over any input sequence, the button is never left down.

        Formally: presses and releases alternate, starting with a press, and
        the FSM is dragging exactly when there is one more press than release.
        """
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.2)
        depth = 0
        for d in distances:
            fsm.update(lms(d))
            clock.advance(1 / 30.0)
            if fsm.drag_started:
                depth += 1
            if fsm.drag_ended:
                depth -= 1
            assert depth in (0, 1), f"button depth went to {depth}"
            assert fsm.dragging == (depth == 1)

        # Whatever state we ended in, letting go must settle it.
        fsm.update(None)
        if fsm.drag_ended:
            depth -= 1
        assert depth == 0

    @given(distances=st.lists(
        st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        min_size=1, max_size=120))
    @settings(max_examples=200, deadline=None)
    def test_a_click_and_a_drag_are_never_emitted_on_the_same_frame(self, distances):
        clock = Clock()
        fsm = _drag_fsm(clock, hold_seconds=0.2)
        for d in distances:
            fsm.update(lms(d))
            clock.advance(1 / 30.0)
            assert not (fsm.click_fired and fsm.drag_started)
            assert not (fsm.click_fired and fsm.drag_ended)


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


class TestDragThroughThePipeline:
    def _pipeline(self, clock, hold_seconds=0.4):
        conf = AppConfig(drag=DragConfig(enabled=True,
                                         hold_seconds=hold_seconds))
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    conf, clock=clock)
        return inference, GestureRouter(conf, 1920, 1080)

    def test_a_long_pinch_emits_mouse_down_then_up(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        emitted = []

        for _ in range(40):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(lms(PINCH), t=clock.now))
            emitted += router.route(result, now=clock.now)
        for _ in range(3):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(lms(OPEN), t=clock.now))
            emitted += router.route(result, now=clock.now)

        downs = [a for a in emitted if isinstance(a, act.MouseDown)]
        ups = [a for a in emitted if isinstance(a, act.MouseUp)]
        assert len(downs) == 1
        assert len(ups) == 1

    def test_the_pointer_keeps_tracking_while_dragging(self):
        """Without this a drag can press and release but never move anything."""
        clock = Clock()
        inference, router = self._pipeline(clock)

        moves_while_dragging = 0
        for i in range(60):
            clock.advance(1 / 30.0)
            hand = lms(PINCH)
            # Slide the whole pinch across the frame.
            for p in (hand[4], hand[8]):
                p.x += i * 0.004
            result = inference.process(_capture(hand, t=clock.now))
            emitted = router.route(result, now=clock.now)
            if result.dragging:
                moves_while_dragging += sum(
                    1 for a in emitted if isinstance(a, act.MoveCursor))

        assert moves_while_dragging > 0, "the pointer froze during the drag"

    def test_dragging_claims_the_mode(self):
        clock = Clock()
        inference, router = self._pipeline(clock)
        for _ in range(40):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(lms(PINCH), t=clock.now))
            router.route(result, now=clock.now)
        assert router.active_mode(result) is Mode.DRAG

    def test_mouse_down_is_never_dropped_by_the_dispatcher(self):
        from gestureflow.actions import ActionDispatcher

        class Recorder:
            screen_w, screen_h = 1920, 1080
            volume = 50

            def __init__(self):
                self.calls = []

            def move_mouse_smooth(self, x, y, now=None): self.calls.append("move")
            def release_cursor(self): self.calls.append("release")
            def click(self): self.calls.append("click")
            def right_click(self): self.calls.append("right_click")
            def mouse_down(self, button="left"): self.calls.append("down")
            def mouse_up(self, button="left"): self.calls.append("up")
            def scroll(self, delta): self.calls.append("scroll")
            def set_volume(self, value): self.calls.append("vol")
            def execute_command(self, gid): self.calls.append("cmd")

        ctrl = Recorder()
        dispatcher = ActionDispatcher(ctrl)
        for i in range(40):
            dispatcher.submit(act.MoveCursor(float(i), 0.0))
        dispatcher.submit(act.MouseDown("left"))
        for i in range(40):
            dispatcher.submit(act.MoveCursor(float(i), 0.0))
        dispatcher.submit(act.MouseUp("left"))
        dispatcher._drain()

        assert ctrl.calls.count("down") == 1
        assert ctrl.calls.count("up") == 1
        assert ctrl.calls.index("down") < ctrl.calls.index("up")


class TestRightClickDoesNotDrag:
    def test_the_right_pinch_has_no_drag_state(self):
        """A long right-pinch holding the context-menu button down helps nobody."""
        clock = Clock()
        conf = AppConfig(drag=DragConfig(enabled=True, hold_seconds=0.2))
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    conf, clock=clock)
        hand = lms(0.30)
        hand[12] = SimpleNamespace(x=hand[8].x + 0.005, y=hand[8].y, z=0.0)

        for _ in range(60):
            clock.advance(1 / 30.0)
            result = inference.process(_capture(hand, t=clock.now))
            assert not result.drag_started
        assert result.right_fsm_active
