"""Regressions for the cursor and scroll failures described in DIAGNOSIS.md.

Each test here fails against the code as it was before the fix, and each one
names the symptom it guards so a future change that reintroduces it says so.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gestureflow import actions as act
from gestureflow.actions import ActionDispatcher
from gestureflow.app import GestureRouter
from gestureflow.capture import CaptureResult
from gestureflow.click_fsm import ClickFSM
from gestureflow.config import DEFAULT_CONFIG
from gestureflow.inference import InferenceThread
from gestureflow.scroll_fsm import (
    ScrollFSM,
    ScrollState,
    _is_true_scroll_fist,
    _thumb_raised,
)

# ---------------------------------------------------------------------------
# Landmark builders — anatomically plausible, with a real hand scale
# ---------------------------------------------------------------------------

HAND_SCALE = 0.20


def _base(wrist_y: float = 0.75):
    """21 landmarks with wrist(0) and middle MCP(9) a real hand apart."""
    lms = [SimpleNamespace(x=0.5, y=wrist_y - 0.10, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    lms[9] = SimpleNamespace(x=0.5, y=wrist_y - HAND_SCALE, z=0.0)
    return lms


def fist(wrist_y: float = 0.75):
    """A closed fist with the thumb folded across, as a real fist has it.

    This is the shape that used to be impossible to scroll with: the folded
    thumb sits above its own MCP, which the old _thumb_raised read as a
    deliberate thumbs-up and used to veto the scroll gate.
    """
    lms = _base(wrist_y)
    knuckle = wrist_y - HAND_SCALE          # MCP row
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lms[mcp] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
        lms[tip] = SimpleNamespace(x=0.5, y=knuckle + 0.06, z=0.0)   # curled
    lms[6] = SimpleNamespace(x=0.5, y=knuckle + 0.03, z=0.0)         # index PIP
    # Thumb folded across the fingers: tip above its own MCP (the old false
    # positive) but bent, and not clear of the index knuckle.
    lms[2] = SimpleNamespace(x=0.5, y=knuckle + 0.10, z=0.0)
    lms[3] = SimpleNamespace(x=0.5, y=knuckle + 0.07, z=0.0)
    lms[4] = SimpleNamespace(x=0.5, y=knuckle + 0.05, z=0.0)
    return lms


def pointing(x: float = 0.5, tip_y: float = 0.40):
    """Index finger up, thumb tucked."""
    lms = _base()
    knuckle = 0.75 - HAND_SCALE
    lms[5] = SimpleNamespace(x=0.5, y=knuckle, z=0.0)
    lms[6] = SimpleNamespace(x=x, y=tip_y + 0.12, z=0.0)
    lms[8] = SimpleNamespace(x=x, y=tip_y, z=0.0)
    for tip, mcp in ((12, 9), (16, 13), (20, 17)):
        lms[mcp] = SimpleNamespace(x=0.62, y=knuckle, z=0.0)
        lms[tip] = SimpleNamespace(x=0.62, y=knuckle + 0.06, z=0.0)
    lms[2] = SimpleNamespace(x=0.68, y=knuckle + 0.10, z=0.0)
    lms[3] = SimpleNamespace(x=0.68, y=knuckle + 0.08, z=0.0)
    lms[4] = SimpleNamespace(x=0.68, y=knuckle + 0.06, z=0.0)
    return lms


def thumbs_up():
    """A genuine thumbs-up: straight thumb, clear of the hand."""
    lms = fist()
    knuckle = 0.75 - HAND_SCALE
    lms[2] = SimpleNamespace(x=0.5, y=knuckle + 0.02, z=0.0)
    lms[3] = SimpleNamespace(x=0.5, y=knuckle - 0.06, z=0.0)
    lms[4] = SimpleNamespace(x=0.5, y=knuckle - 0.14, z=0.0)
    return lms


def _capture(landmarks, t=0.0):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


class _StubModel:
    """Always predicts Neutral, so the geometric modes stay live."""
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


class _Clock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now


class RecordingController:
    def __init__(self):
        self.calls = []
        self.screen_w, self.screen_h = 1920, 1080
        self.volume = 50
        self._lock = threading.Lock()

    def _log(self, name, *args):
        with self._lock:
            self.calls.append((name, args))

    def move_mouse_smooth(self, x, y, now=None): self._log("move", x, y)
    def release_cursor(self): self._log("release")
    def click(self): self._log("click")
    def right_click(self): self._log("right_click")
    def scroll(self, delta): self._log("scroll", delta)
    def set_volume(self, value): self._log("set_volume", value)
    def execute_command(self, gid): self._log("command", gid)

    def names(self):
        with self._lock:
            return [c[0] for c in self.calls]


# ---------------------------------------------------------------------------
# Part 3 — the shared root cause
# ---------------------------------------------------------------------------

class TestThumbDetection:
    def test_a_folded_fist_thumb_is_not_raised(self):
        """The bug that made scroll impossible.

        Comparing the thumb tip against its own MCP is true for a fist, because
        the thumb folds across the curled fingers and lands above its knuckle.
        Measured on real data, that vetoed 89% of genuine scroll fists.
        """
        assert _thumb_raised(fist()) is False

    def test_a_real_thumbs_up_is_still_detected(self):
        assert _thumb_raised(thumbs_up()) is True

    def test_a_pointing_hand_does_not_read_as_thumb_raised(self):
        assert _thumb_raised(pointing()) is False


class TestHandScaleInvariance:
    """Thresholds must not depend on how far away the hand is."""

    @staticmethod
    def _scaled(landmarks, factor):
        wx, wy, wz = landmarks[0].x, landmarks[0].y, landmarks[0].z
        return [SimpleNamespace(x=wx + (p.x - wx) * factor,
                                y=wy + (p.y - wy) * factor,
                                z=wz + (p.z - wz) * factor)
                for p in landmarks]

    @pytest.mark.parametrize("factor", [0.4, 0.7, 1.0, 1.6, 2.5])
    def test_fist_detection_survives_any_hand_size(self, factor):
        assert _is_true_scroll_fist(self._scaled(fist(), factor)) is True

    @pytest.mark.parametrize("factor", [0.4, 0.7, 1.0, 1.6, 2.5])
    def test_thumbs_up_detection_survives_any_hand_size(self, factor):
        assert _thumb_raised(self._scaled(thumbs_up(), factor)) is True

    @pytest.mark.parametrize("factor", [0.4, 0.7, 1.0, 1.6, 2.5])
    def test_pinch_threshold_survives_any_hand_size(self, factor):
        """A pinch is a pinch whether the hand is near or far.

        With absolute thresholds a hand further from the camera produced
        smaller raw distances, so the click threshold silently got easier to
        cross as the user leaned back.
        """
        lms = pointing()
        lms[4] = SimpleNamespace(x=lms[8].x + 0.01, y=lms[8].y, z=0.0)
        scaled = self._scaled(lms, factor)
        fsm = ClickFSM(DEFAULT_CONFIG.click, 4, 8)
        for _ in range(6):
            fsm.update(scaled)
        assert fsm.is_active, f"pinch not detected at hand size x{factor}"


class TestFistNeverRightClicks:
    """A fist resolves to scroll, never to a middle+index pinch."""

    def _thread(self, clock):
        return InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                               DEFAULT_CONFIG, clock=clock)

    def test_holding_a_fist_never_fires_a_right_click(self):
        clock = _Clock()
        t = self._thread(clock)
        for i in range(90):
            clock.now = i / 30.0
            result = t.process(_capture(fist(), t=clock.now))
            assert not result.right_click_fired, f"right click at frame {i}"
            assert not result.click_fired, f"left click at frame {i}"

    def test_holding_a_fist_never_arms_the_click_fsms(self):
        clock = _Clock()
        t = self._thread(clock)
        for i in range(30):
            clock.now = i / 30.0
            result = t.process(_capture(fist(), t=clock.now))
            assert not result.fsm_active
            assert not result.right_fsm_active

    def test_holding_a_fist_does_reach_the_scrolling_state(self):
        clock = _Clock()
        t = self._thread(clock)
        for i in range(30):
            clock.now = i / 30.0
            result = t.process(_capture(fist(0.75 - i * 0.004), t=clock.now))
        assert result.scroll_state is ScrollState.SCROLLING


# ---------------------------------------------------------------------------
# Part 2 — scroll, end to end
# ---------------------------------------------------------------------------

class TestScrollEndToEnd:
    def test_a_fist_arms_the_scroll_gate(self):
        assert _is_true_scroll_fist(fist()) is True

    def test_scroll_fsm_reaches_scrolling_and_emits_a_delta(self):
        clock = _Clock()
        fsm = ScrollFSM(DEFAULT_CONFIG.scroll, clock=clock)
        deltas = []
        for i in range(30):
            clock.now = i / 30.0
            fsm.update(fist(0.75 - i * 0.015))   # ~0.075 hand-widths/frame
            deltas.append(fsm.scroll_delta)

        assert fsm.state is ScrollState.SCROLLING
        assert any(d != 0 for d in deltas), "the wrist anchor never produced output"

    def test_moving_up_scrolls_up_and_down_scrolls_down(self):
        for direction, sign in (("up", 1), ("down", -1)):
            clock = _Clock()
            fsm = ScrollFSM(DEFAULT_CONFIG.scroll, clock=clock)
            deltas = []
            for i in range(30):
                clock.now = i / 30.0
                fsm.update(fist(0.75 - sign * i * 0.015))
                if fsm.scroll_delta:
                    deltas.append(fsm.scroll_delta)
            assert deltas, f"no scroll output moving {direction}"
            assert all((d > 0) == (sign > 0) for d in deltas), (
                f"moving {direction} produced {deltas}"
            )

    def test_full_pipeline_fist_plus_motion_dispatches_scrolls(self):
        """Fake landmarks -> real recognizer -> real router -> real dispatcher."""
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)

        for i in range(40):
            clock.now = i / 30.0
            result = inference.process(_capture(fist(0.75 - i * 0.015),
                                                t=clock.now))
            for action in router.route(result, now=clock.now):
                dispatcher.submit(action)
        dispatcher._drain()

        scrolls = [c for c in ctrl.calls if c[0] == "scroll"]
        assert scrolls, (
            "a held fist moving vertically dispatched no scroll actions"
        )
        assert all(c[1][0] > 0 for c in scrolls), "wrong scroll direction"
        assert "right_click" not in ctrl.names()

    def test_scroll_actions_are_never_dropped_by_the_dispatcher(self):
        """Scroll is discrete: the cursor coalescing policy must not apply."""
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)
        for i in range(1, 21):
            dispatcher.submit(act.Scroll(i))
        dispatcher._drain()

        deltas = [c[1][0] for c in ctrl.calls if c[0] == "scroll"]
        assert deltas == list(range(1, 21)), (
            f"scroll events were dropped or reordered: {deltas}"
        )

    def test_scroll_survives_a_flood_of_cursor_moves(self):
        """Cursor coalescing must not starve or reorder discrete events."""
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)
        for i in range(50):
            dispatcher.submit(act.MoveCursor(float(i), float(i)))
            if i % 10 == 0:
                dispatcher.submit(act.Scroll(i))
        dispatcher._drain()

        deltas = [c[1][0] for c in ctrl.calls if c[0] == "scroll"]
        assert deltas == [0, 10, 20, 30, 40]


# ---------------------------------------------------------------------------
# Part 1 — cursor
# ---------------------------------------------------------------------------

class TestCursorGating:
    def test_pointing_enables_cursor_mode(self):
        router = GestureRouter(DEFAULT_CONFIG)
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        result = inference.process(_capture(pointing()))
        assert router.cursor_enabled(result) is True

    def test_cursor_mode_stays_on_across_a_whole_sweep(self):
        """The regression: cursor mode used to drop out ~3 frames in 4.

        Sparse enablement is what starved the filter and turned smooth motion
        into a handful of large hops.
        """
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)

        enabled = 0
        frames = 40
        for i in range(frames):
            clock.now = i / 30.0
            result = inference.process(
                _capture(pointing(x=0.30 + i * 0.008), t=clock.now))
            if router.cursor_enabled(result):
                enabled += 1

        assert enabled == frames, (
            f"cursor mode was enabled on only {enabled}/{frames} frames; "
            f"sparse enablement is what made the pointer hop"
        )

    def test_a_sweep_emits_a_move_every_frame(self):
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)

        moves = 0
        for i in range(40):
            clock.now = i / 30.0
            result = inference.process(
                _capture(pointing(x=0.30 + i * 0.008), t=clock.now))
            moves += sum(1 for a in router.route(result, now=clock.now)
                         if isinstance(a, act.MoveCursor))
        assert moves == 40

    def test_moves_carry_the_capture_timestamp(self):
        """The filter needs frame time, not dispatch time."""
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)
        clock.now = 4.0
        result = inference.process(_capture(pointing(), t=4.0))
        moves = [a for a in router.route(result, now=4.0)
                 if isinstance(a, act.MoveCursor)]
        assert moves and moves[0].captured_at == 4.0

    def test_disengaging_cursor_mode_emits_a_release(self):
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)

        clock.now = 0.0
        router.route(inference.process(_capture(pointing(), t=0.0)), now=0.0)
        clock.now = 0.1
        emitted = router.route(inference.process(_capture(None, t=0.1)), now=0.1)
        assert any(isinstance(a, act.ReleaseCursor) for a in emitted)

    def test_release_is_emitted_once_not_every_idle_frame(self):
        clock = _Clock()
        inference = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                                    DEFAULT_CONFIG, clock=clock)
        router = GestureRouter(DEFAULT_CONFIG, 1920, 1080)

        clock.now = 0.0
        router.route(inference.process(_capture(pointing(), t=0.0)), now=0.0)

        releases = 0
        for i in range(1, 20):
            clock.now = i / 30.0
            emitted = router.route(
                inference.process(_capture(None, t=clock.now)), now=clock.now)
            releases += sum(1 for a in emitted
                            if isinstance(a, act.ReleaseCursor))
        assert releases == 1


class TestCoalescingPolicy:
    """Latest target wins, but a dispatch must actually happen each tick."""

    def test_only_the_newest_target_is_delivered(self):
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)
        for i in range(10):
            dispatcher.submit(act.MoveCursor(float(i), float(i)))
        dispatcher._drain()

        moves = [c for c in ctrl.calls if c[0] == "move"]
        assert len(moves) == 1
        assert moves[0][1] == (9.0, 9.0)

    def test_a_move_is_dispatched_on_every_tick_not_batched_away(self):
        """Coalescing must not turn a steady stream into occasional hops.

        One producer frame followed by one drain must produce one move. If the
        dispatcher only delivered on some ticks, the pointer would travel in
        bursts -- which is the reported symptom.
        """
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)
        for i in range(30):
            dispatcher.submit(act.MoveCursor(float(i), 0.0))
            dispatcher._drain()

        moves = [c for c in ctrl.calls if c[0] == "move"]
        assert len(moves) == 30, (
            f"30 producer frames yielded {len(moves)} dispatches"
        )
        assert [m[1][0] for m in moves] == [float(i) for i in range(30)]

    def test_running_dispatcher_keeps_up_with_a_steady_producer(self):
        """The same property against the real thread, not a manual drain."""
        ctrl = RecordingController()
        stop = threading.Event()
        dispatcher = ActionDispatcher(ctrl, stop_event=stop)
        dispatcher.start()
        try:
            for i in range(30):
                dispatcher.submit(act.MoveCursor(float(i), 0.0))
                time.sleep(0.005)
            dispatcher.flush(timeout=2.0)
            time.sleep(0.05)
        finally:
            dispatcher.stop()
            dispatcher.join(timeout=2.0)

        moves = [c for c in ctrl.calls if c[0] == "move"]
        # Some coalescing is expected and correct; near-total collapse is not.
        assert len(moves) >= 15, (
            f"only {len(moves)} of 30 cursor updates were dispatched; "
            f"coalescing is starving motion"
        )
        assert ctrl.calls[-1][1][0] == 29.0, "the final position was not delivered"

    def test_release_is_never_coalesced_away(self):
        ctrl = RecordingController()
        dispatcher = ActionDispatcher(ctrl)
        dispatcher.submit(act.MoveCursor(1.0, 1.0))
        dispatcher.submit(act.ReleaseCursor())
        dispatcher.submit(act.MoveCursor(2.0, 2.0))
        dispatcher._drain()
        assert "release" in ctrl.names()
