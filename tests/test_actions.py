"""Tests for the action dispatcher.

The property that matters here is the asymmetry in drop policy: cursor moves
are allowed to be superseded, discrete events are not.  A dispatcher that
dropped clicks under load would feel like a broken mouse, and the failure would
be invisible in any aggregate metric.
"""

from __future__ import annotations

import threading
import time

import pytest

from gestureflow import actions as act
from gestureflow.actions import ActionDispatcher
from gestureflow.metrics import MetricsRecorder


class RecordingController:
    """Records what it was asked to do, and can be made artificially slow."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls = []
        self.delay = delay
        self.screen_w = 1920
        self.screen_h = 1080
        self.volume = 50
        self._lock = threading.Lock()

    def _log(self, name, *args):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.calls.append((name, args))

    def move_mouse_smooth(self, x, y, now=None): self._log("move", x, y)
    def click(self): self._log("click")
    def right_click(self): self._log("right_click")
    def scroll(self, delta): self._log("scroll", delta)
    def set_volume(self, value): self._log("set_volume", value)
    def execute_command(self, gesture_id): self._log("command", gesture_id)

    def names(self):
        with self._lock:
            return [c[0] for c in self.calls]


@pytest.fixture
def dispatcher():
    ctrl = RecordingController()
    stop = threading.Event()
    d = ActionDispatcher(ctrl, stop_event=stop)
    d.start()
    yield d, ctrl
    d.stop()
    d.join(timeout=2.0)


class TestDispatch:
    def test_click_is_performed(self, dispatcher):
        d, ctrl = dispatcher
        d.submit(act.Click("left"))
        assert d.flush(timeout=1.0)
        time.sleep(0.05)
        assert "click" in ctrl.names()

    def test_right_click_routes_to_right_click(self, dispatcher):
        d, ctrl = dispatcher
        d.submit(act.Click("right"))
        d.flush(timeout=1.0)
        time.sleep(0.05)
        assert "right_click" in ctrl.names()

    def test_every_action_type_reaches_the_controller(self, dispatcher):
        d, ctrl = dispatcher
        for action in (act.Click("left"), act.Scroll(3), act.SetVolume(70),
                       act.Command(1), act.MoveCursor(10.0, 20.0)):
            d.submit(action)
        d.flush(timeout=1.0)
        time.sleep(0.1)
        names = set(ctrl.names())
        assert {"click", "scroll", "set_volume", "command", "move"} <= names

    def test_unknown_action_does_not_kill_the_thread(self, dispatcher):
        d, ctrl = dispatcher
        d.submit(object())          # not an action type
        d.submit(act.Click("left"))
        d.flush(timeout=1.0)
        time.sleep(0.05)
        assert "click" in ctrl.names(), "dispatcher survived a bad action"


class TestDropSemantics:
    """Cursor moves coalesce; discrete events do not."""

    def test_cursor_moves_coalesce_to_the_latest(self):
        ctrl = RecordingController()
        d = ActionDispatcher(ctrl)          # not started: nothing drains
        for i in range(10):
            d.submit(act.MoveCursor(float(i), float(i)))

        assert d.pending == 1, "only the newest cursor position should be held"
        assert d.coalesced_moves == 9

        d._drain()
        moves = [c for c in ctrl.calls if c[0] == "move"]
        assert len(moves) == 1
        assert moves[0][1] == (9.0, 9.0), "the surviving move must be the newest"

    def test_discrete_events_are_never_coalesced(self):
        ctrl = RecordingController()
        d = ActionDispatcher(ctrl)
        for _ in range(5):
            d.submit(act.Click("left"))

        assert d.pending == 5
        d._drain()
        assert ctrl.names().count("click") == 5, (
            "a click the user made must never be silently dropped"
        )

    def test_discrete_events_are_delivered_in_order(self):
        ctrl = RecordingController()
        d = ActionDispatcher(ctrl)
        for i in range(5):
            d.submit(act.Scroll(i))
        d._drain()
        deltas = [c[1][0] for c in ctrl.calls if c[0] == "scroll"]
        assert deltas == [0, 1, 2, 3, 4]

    def test_discrete_events_win_over_a_pending_move(self):
        ctrl = RecordingController()
        d = ActionDispatcher(ctrl)
        d.submit(act.MoveCursor(1.0, 1.0))
        d.submit(act.Click("left"))
        d._drain()
        assert ctrl.names() == ["click", "move"], (
            "a click must not wait behind a cursor move that is about to be "
            "superseded anyway"
        )

    def test_queue_full_drops_and_counts_rather_than_growing(self):
        ctrl = RecordingController()
        d = ActionDispatcher(ctrl, max_pending=4)
        accepted = [d.submit(act.Click("left")) for _ in range(10)]

        assert accepted.count(True) == 4
        assert d.dropped_discrete == 6
        assert d.pending == 4, "bounded means bounded"

    def test_submit_reports_success_for_moves_always(self):
        d = ActionDispatcher(RecordingController(), max_pending=1)
        assert all(d.submit(act.MoveCursor(float(i), 0.0)) for i in range(20))


class TestMetricsIntegration:
    def test_coalesced_moves_are_counted(self):
        metrics = MetricsRecorder()
        d = ActionDispatcher(RecordingController(), metrics=metrics)
        for i in range(4):
            d.submit(act.MoveCursor(float(i), 0.0))
        assert metrics.snapshot()["counters"]["action.moves_coalesced"] == 3

    def test_dropped_discrete_events_are_counted(self):
        metrics = MetricsRecorder()
        d = ActionDispatcher(RecordingController(), metrics=metrics,
                             max_pending=2)
        for _ in range(5):
            d.submit(act.Click("left"))
        assert metrics.snapshot()["counters"]["action.discrete_dropped"] == 3

    def test_dispatch_duration_is_recorded(self):
        metrics = MetricsRecorder()
        d = ActionDispatcher(RecordingController(), metrics=metrics)
        d.submit(act.Click("left"))
        d._drain()
        assert metrics.snapshot()["stages"]["action.dispatch"]["count"] == 1

    def test_end_to_end_latency_is_recorded_when_a_timestamp_is_present(self):
        metrics = MetricsRecorder()
        d = ActionDispatcher(RecordingController(), metrics=metrics)
        d.submit(act.Click("left", captured_at=metrics._clock() - 0.05))
        d._drain()
        e2e = metrics.snapshot()["stages"]["end_to_end"]
        assert e2e["count"] == 1
        assert e2e["p50_ms"] >= 45.0


class TestOffTheRenderThread:
    def test_a_slow_controller_does_not_block_the_producer(self):
        """The point of the whole module: submitting must not wait on the OS."""
        ctrl = RecordingController(delay=0.05)
        stop = threading.Event()
        d = ActionDispatcher(ctrl, stop_event=stop)
        d.start()
        try:
            start = time.monotonic()
            for _ in range(10):
                d.submit(act.Click("left"))
            elapsed = time.monotonic() - start
            # 10 clicks x 50ms = 500ms of work; submitting must not pay it.
            assert elapsed < 0.1, (
                f"submit() blocked for {elapsed:.3f}s -- the render thread "
                f"would have stalled"
            )
        finally:
            d.stop()
            d.join(timeout=3.0)
