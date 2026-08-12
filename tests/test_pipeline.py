"""Tests for capture, inference, the controller, and the pipeline end to end.

These four were entirely untested. The integration test at the bottom is the
one that matters most: it drives a fake landmark sequence through the real
InferenceThread and the real GestureRouter with an injected clock, and asserts
the exact action sequence that comes out.
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
from gestureflow.capture import CaptureResult, CaptureThread
from gestureflow.config import (
    DEFAULT_CONFIG,
    AppConfig,
    ClickConfig,
    DebounceConfig,
)
from gestureflow.inference import InferenceThread
from gestureflow.metrics import MetricsRecorder

# ---------------------------------------------------------------------------
# Regression: thread subclasses must not shadow threading.Thread._stop
# ---------------------------------------------------------------------------

class TestThreadAttributeShadowing:
    """`self._stop = Event()` silently breaks Thread.join().

    threading.Thread has a private _stop() method that join() calls once the
    thread has finished. Binding an Event over it made join() raise
    "TypeError: 'Event' object is not callable" on every clean shutdown -- and
    only on a *clean* shutdown, which is why it went unnoticed.
    """

    @pytest.mark.parametrize("cls", [CaptureThread, InferenceThread,
                                     ActionDispatcher])
    def test_stop_method_is_not_shadowed(self, cls):
        assert callable(getattr(cls, "_stop", None)), (
            f"{cls.__name__} shadows threading.Thread._stop; join() will raise"
        )

    def test_dispatcher_joins_cleanly_after_finishing(self):
        d = ActionDispatcher(_NullController())
        d.start()
        d.stop()
        d.join(timeout=2.0)
        assert not d.is_alive()

    def test_inference_thread_joins_cleanly_after_finishing(self):
        stop = threading.Event()
        t = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                            DEFAULT_CONFIG, stop)
        t.start()
        time.sleep(0.05)
        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class _StubModel:
    """Returns a scripted probability vector, so tests control the classifier.

    Real model behaviour is not what these tests are about -- the pipeline's
    reaction to a given prediction is.
    """

    classes_ = np.array([0, 1, 2, 3])

    def __init__(self, sequence=None) -> None:
        self.sequence = list(sequence or [])
        self.calls = 0

    def predict_proba(self, features):
        self.calls += 1
        if self.sequence:
            label, conf = self.sequence.pop(0)
        else:
            label, conf = 0, 1.0
        probs = [(1.0 - conf) / 3.0] * 4
        probs[label] = conf
        return np.array([probs])


def _capture(landmarks, t=0.0):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


def _neutral_hand():
    """A hand in no particular pose: no fist, no pinch, no raised digits."""
    lm = [SimpleNamespace(x=0.5 + i * 0.01, y=0.5, z=0.0) for i in range(21)]
    lm[8] = SimpleNamespace(x=0.30, y=0.50, z=0.0)   # index tip level with PIP
    lm[6] = SimpleNamespace(x=0.30, y=0.50, z=0.0)
    lm[4] = SimpleNamespace(x=0.70, y=0.50, z=0.0)   # thumb down
    lm[2] = SimpleNamespace(x=0.70, y=0.45, z=0.0)
    return lm


def _pointing_hand(x=0.5, y=0.5):
    lm = [SimpleNamespace(x=0.5, y=0.6, z=0.0) for _ in range(21)]
    lm[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    lm[8] = SimpleNamespace(x=x, y=y, z=0.0)          # index tip up
    lm[6] = SimpleNamespace(x=x, y=y + 0.20, z=0.0)   # PIP well below tip
    lm[4] = SimpleNamespace(x=0.75, y=0.70, z=0.0)    # thumb NOT raised
    lm[2] = SimpleNamespace(x=0.75, y=0.66, z=0.0)
    lm[5] = SimpleNamespace(x=0.5, y=0.60, z=0.0)
    lm[12] = SimpleNamespace(x=0.60, y=0.62, z=0.0)   # middle away from index
    return lm


def _pinching_hand():
    lm = _pointing_hand()
    tip = lm[8]
    lm[4] = SimpleNamespace(x=tip.x + 0.005, y=tip.y, z=0.0)
    lm[2] = SimpleNamespace(x=tip.x + 0.005, y=tip.y + 0.10, z=0.0)
    return lm


class TestInferenceThread:
    def _thread(self, model, cfg=DEFAULT_CONFIG, clock=None):
        return InferenceThread(model, queue.Queue(), queue.Queue(), cfg,
                               clock=clock or time.monotonic)

    def test_no_hand_yields_neutral_and_no_action(self):
        t = self._thread(_StubModel())
        result = t.process(_capture(None))
        assert result.stable_gesture == 0
        assert result.action is None
        assert result.raw_prediction == 0
        assert not result.click_fired

    def test_no_hand_does_not_call_the_model(self):
        model = _StubModel()
        t = self._thread(model)
        t.process(_capture(None))
        assert model.calls == 0, "a frame with no hand must not run inference"

    def test_confident_repeated_prediction_produces_an_action(self):
        cfg = AppConfig(debounce=DebounceConfig(vote_window_size=5,
                                                vote_threshold=3,
                                                cmd_cooldown=0.0))
        model = _StubModel([(1, 0.99)] * 5)
        t = self._thread(model, cfg)
        fired = [t.process(_capture(_neutral_hand())).action for _ in range(5)]
        assert 1 in fired

    def test_low_confidence_never_produces_an_action(self):
        cfg = AppConfig(debounce=DebounceConfig(vote_window_size=5,
                                                vote_threshold=3,
                                                cmd_cooldown=0.0))
        model = _StubModel([(1, 0.30)] * 10)
        t = self._thread(model, cfg)
        fired = [t.process(_capture(_neutral_hand())).action for _ in range(10)]
        assert all(f is None for f in fired)

    def test_named_gesture_parks_the_geometric_fsms(self):
        cfg = AppConfig(debounce=DebounceConfig(vote_window_size=3,
                                                vote_threshold=2,
                                                cmd_cooldown=0.0))
        model = _StubModel([(2, 0.99)] * 10)
        t = self._thread(model, cfg)

        saw_stable = False
        for _ in range(10):
            result = t.process(_capture(_pinching_hand()))
            if result.stable_gesture != 0:
                saw_stable = True
                assert not result.fsm_active, (
                    "a recognized command gesture must suspend the pinch FSM"
                )
        assert saw_stable, "the stub model should have produced a stable gesture"

    def test_stable_gesture_drops_to_zero_on_the_frame_a_command_fires(self):
        """Documents a real quirk rather than asserting it away.

        GestureDebouncer.update() clears its vote history after firing, so the
        very next read of stable_gesture sees an empty window and reports 0.
        That briefly un-parks the geometric FSMs right after a command. It is
        harmless in practice -- the window refills in two or three frames while
        a click needs four held frames -- but it is real, and a future change to
        the hold thresholds needs to know about it.
        """
        cfg = AppConfig(debounce=DebounceConfig(vote_window_size=3,
                                                vote_threshold=3,
                                                cmd_cooldown=0.0))
        t = self._thread(_StubModel([(2, 0.99)] * 6), cfg)
        results = [t.process(_capture(_neutral_hand())) for _ in range(6)]

        fired = [r for r in results if r.action is not None]
        assert fired, "expected the command to fire"
        assert fired[0].stable_gesture == 0

    def test_metrics_are_recorded_for_each_stage(self):
        metrics = MetricsRecorder()
        t = InferenceThread(_StubModel(), queue.Queue(), queue.Queue(),
                            DEFAULT_CONFIG, metrics=metrics)
        t.process(_capture(_neutral_hand()))
        stages = metrics.snapshot()["stages"]
        assert stages["inference.normalize"]["count"] == 1
        assert stages["inference.predict"]["count"] == 1
        assert stages["inference.fsm"]["count"] == 1

    def test_emit_drops_stale_results_and_counts_them(self):
        metrics = MetricsRecorder()
        out: queue.Queue = queue.Queue(maxsize=1)
        t = InferenceThread(_StubModel(), queue.Queue(), out, DEFAULT_CONFIG,
                            metrics=metrics)
        for _ in range(5):
            t._emit(t.process(_capture(_neutral_hand())))
        assert out.qsize() == 1
        assert t.dropped == 4
        assert metrics.snapshot()["counters"]["inference.results_dropped"] == 4


# ---------------------------------------------------------------------------
# Capture (without a camera)
# ---------------------------------------------------------------------------

class TestCaptureThread:
    def test_reports_starting_before_run(self):
        assert CaptureThread(queue.Queue()).status == "starting"

    def test_stop_is_observed_by_the_open_loop(self):
        """A stop during camera-open must not wait out the open timeout."""
        stop = threading.Event()
        cfg = AppConfig(camera=type(DEFAULT_CONFIG.camera)(
            device_index=999, open_timeout=30.0,
            reconnect_backoff_min=0.1, reconnect_backoff_max=0.2,
        ))
        t = CaptureThread(queue.Queue(), cfg, stop)
        t.start()
        time.sleep(0.2)
        stop.set()
        t.join(timeout=5.0)
        assert not t.is_alive(), "capture thread ignored the stop event"

    def test_dropped_counter_starts_at_zero(self):
        assert CaptureThread(queue.Queue()).dropped == 0


# ---------------------------------------------------------------------------
# Controller (no OS calls)
# ---------------------------------------------------------------------------

class _NullController:
    screen_w, screen_h = 1920, 1080
    volume = 50

    def move_mouse_smooth(self, x, y, now=None): pass
    def click(self): pass
    def right_click(self): pass
    def scroll(self, delta): pass
    def set_volume(self, value): pass
    def execute_command(self, gesture_id): pass


class TestCursorSmoothing:
    """move_mouse_smooth's filter must be frame-rate independent."""

    def _controller(self):
        from gestureflow.controller import SystemController
        ctrl = SystemController.__new__(SystemController)
        ctrl._cfg = DEFAULT_CONFIG
        ctrl._tau = 0.15
        ctrl._ploc_x = 0.0
        ctrl._ploc_y = 0.0
        ctrl._last_move_time = None
        ctrl.screen_w, ctrl.screen_h = 1920, 1080
        ctrl._pag = _FakePyAutoGUI()
        return ctrl

    def test_first_move_seeds_from_the_live_pointer(self):
        ctrl = self._controller()
        ctrl._pag.pos = (800.0, 600.0)
        ctrl.move_mouse_smooth(1000.0, 700.0, now=0.0)
        # Starting from (800, 600), not (0, 0): the cursor must not sweep in
        # from the screen corner on the first frame of a session.
        assert ctrl._pag.moved[-1] == (800.0, 600.0)

    def test_same_elapsed_time_gives_the_same_result_at_any_frame_rate(self):
        target = 1000.0

        slow = self._controller()
        slow._pag.pos = (0.0, 0.0)
        slow.move_mouse_smooth(target, 0.0, now=0.0)
        for i in range(1, 16):                       # 15 FPS for one second
            slow.move_mouse_smooth(target, 0.0, now=i / 15.0)

        fast = self._controller()
        fast._pag.pos = (0.0, 0.0)
        fast.move_mouse_smooth(target, 0.0, now=0.0)
        for i in range(1, 61):                       # 60 FPS for one second
            fast.move_mouse_smooth(target, 0.0, now=i / 60.0)

        assert slow._ploc_x == pytest.approx(fast._ploc_x, rel=0.02), (
            f"15 FPS reached {slow._ploc_x:.1f}, 60 FPS reached "
            f"{fast._ploc_x:.1f} -- smoothing is still frame-rate dependent"
        )

    def test_converges_toward_the_target(self):
        ctrl = self._controller()
        ctrl._pag.pos = (0.0, 0.0)
        for i in range(60):
            ctrl.move_mouse_smooth(500.0, 0.0, now=i / 30.0)
        assert ctrl._ploc_x == pytest.approx(500.0, abs=1.0)

    def test_zero_elapsed_time_does_not_move(self):
        ctrl = self._controller()
        ctrl._pag.pos = (100.0, 100.0)
        ctrl.move_mouse_smooth(900.0, 900.0, now=5.0)
        ctrl.move_mouse_smooth(900.0, 900.0, now=5.0)
        assert ctrl._ploc_x == pytest.approx(100.0)


class _FakePyAutoGUI:
    def __init__(self) -> None:
        self.pos = (0.0, 0.0)
        self.moved = []

    def position(self):
        return self.pos

    def moveTo(self, x, y, _pause=True):
        self.moved.append((x, y))


# ---------------------------------------------------------------------------
# Full-pipeline integration
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestFullPipelineIntegration:
    """Fake landmarks + stub model + injected clock -> asserted action sequence.

    Nothing here is mocked out of the path under test: this is the real
    InferenceThread.process, the real debouncer, the real three FSMs, the real
    GestureRouter, and the real ActionDispatcher.  Only the camera, the model,
    the clock, and the OS are substituted.
    """

    def _cfg(self):
        return AppConfig(
            debounce=DebounceConfig(vote_window_size=4, vote_threshold=3,
                                    cmd_cooldown=1.0),
            click=ClickConfig(close_threshold=0.045, open_threshold=0.065,
                              min_hold_frames=2, cooldown=0.3),
        )

    def _run(self, frames, model, cfg, clock):
        inference = InferenceThread(model, queue.Queue(), queue.Queue(), cfg,
                                    clock=clock)
        router = GestureRouter(cfg, screen_w=1920, screen_h=1080)
        emitted = []
        for landmarks in frames:
            clock.now += 1 / 30.0
            result = inference.process(_capture(landmarks, t=clock.now))
            emitted.extend(router.route(result, now=clock.now))
        return emitted

    def test_pinch_and_release_produces_exactly_one_left_click(self):
        clock = _Clock()
        cfg = self._cfg()
        frames = ([_pointing_hand()] * 2
                  + [_pinching_hand()] * 4      # charge past min_hold_frames
                  + [_pointing_hand()] * 2)     # release
        emitted = self._run(frames, _StubModel(), cfg, clock)

        clicks = [a for a in emitted if isinstance(a, act.Click)]
        assert len(clicks) == 1, f"expected 1 click, got {len(clicks)}"
        assert clicks[0].button == "left"

    def test_holding_a_pinch_never_clicks(self):
        clock = _Clock()
        emitted = self._run([_pinching_hand()] * 40, _StubModel(),
                            self._cfg(), clock)
        assert not [a for a in emitted if isinstance(a, act.Click)]

    def test_pointing_produces_cursor_moves_and_nothing_else(self):
        clock = _Clock()
        frames = [_pointing_hand(x=0.3 + i * 0.01) for i in range(12)]
        emitted = self._run(frames, _StubModel(), self._cfg(), clock)

        assert emitted, "pointing should move the cursor"
        assert all(isinstance(a, act.MoveCursor) for a in emitted), (
            f"pointing emitted non-cursor actions: "
            f"{ {type(a).__name__ for a in emitted} }"
        )

    def test_a_recognized_gesture_emits_a_command_and_no_cursor_move(self):
        clock = _Clock()
        cfg = self._cfg()
        model = _StubModel([(3, 0.99)] * 12)
        emitted = self._run([_pointing_hand()] * 12, model, cfg, clock)

        commands = [a for a in emitted if isinstance(a, act.Command)]
        assert len(commands) == 1
        assert commands[0].gesture_id == 3

        # Cursor moves before the vote accumulates are correct -- until the
        # gesture is recognized, the hand is just pointing. What must never
        # happen is a move on the same frame as the command.
        command_frames = {a.captured_at for a in commands}
        collisions = [a for a in emitted
                      if isinstance(a, act.MoveCursor)
                      and a.captured_at in command_frames]
        assert not collisions, "a command frame must not also drag the cursor"

    def test_command_cooldown_limits_repeat_fires(self):
        clock = _Clock()
        cfg = self._cfg()          # cmd_cooldown = 1.0s, frames at 30 FPS
        model = _StubModel([(1, 0.99)] * 60)
        emitted = self._run([_pointing_hand()] * 60, model, cfg, clock)

        commands = [a for a in emitted if isinstance(a, act.Command)]
        # 60 frames is 2 seconds of recorded time, so at most 2 fires.
        assert 1 <= len(commands) <= 2

    def test_end_to_end_through_the_dispatcher(self):
        """The same sequence, but delivered to a controller double."""
        clock = _Clock()
        cfg = self._cfg()
        ctrl = _CountingController()
        dispatcher = ActionDispatcher(ctrl)

        frames = ([_pointing_hand()] * 2 + [_pinching_hand()] * 4
                  + [_pointing_hand()] * 2)
        for action in self._run(frames, _StubModel(), cfg, clock):
            dispatcher.submit(action)
        dispatcher._drain()

        assert ctrl.clicks == 1
        assert ctrl.moves >= 1


class _CountingController(_NullController):
    def __init__(self) -> None:
        self.clicks = 0
        self.moves = 0
        self.commands = []

    def click(self): self.clicks += 1
    def move_mouse_smooth(self, x, y, now=None): self.moves += 1
    def execute_command(self, gesture_id): self.commands.append(gesture_id)
