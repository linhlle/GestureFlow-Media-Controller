"""The GestureFlow application: pipeline wiring, HUD, and gesture handlers.

Thread layout
-------------

  ┌───────────────┐  Queue[2]  ┌────────────────┐  Queue[1]  ┌──────────────┐
  │ CaptureThread │ ─────────► │ InferenceThread│ ─────────► │ Main (render)│
  │ camera + MP   │            │ RF + FSMs      │            │ HUD only     │
  └───────────────┘            └────────────────┘            └──────┬───────┘
                                                                    │ actions
                                                             ┌──────▼───────┐
                                                             │ ActionThread │
                                                             │ pyautogui,   │
                                                             │ osascript    │
                                                             └──────────────┘

The main thread is the only one that calls OpenCV GUI functions, because
HighGUI is not thread-safe.  It is also, deliberately, the thread that does the
*least* work: it decides what should happen and hands the doing to the action
thread, so a slow hotkey cannot stall rendering or stop the inference queue
draining.

`run()` drives the GUI.  `run_headless()` drives the same pipeline with no
window, for `gestureflow bench`.
"""

from __future__ import annotations

import os
import pickle
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

from gestureflow import actions as act
from gestureflow.actions import ActionDispatcher
from gestureflow.capture import CaptureThread
from gestureflow.commands import CommandSet, load_commands
from gestureflow.config import AppConfig, DEFAULT_CONFIG
from gestureflow.controller import SystemController
from gestureflow.click_fsm import ClickState
from gestureflow.inference import InferenceResult, InferenceThread
from gestureflow.metrics import STAGE_RENDER, MetricsRecorder
from gestureflow.utils import BindingError, models_path, validate_bindings


# ============================================================================
# HUD rendering
# ============================================================================

def _draw_active_zone(frame: np.ndarray, margin: int) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (margin, margin), (w - margin, h - margin),
                  (255, 0, 0), 2)


def _draw_status(frame: np.ndarray, result: InferenceResult,
                 window_size: int, commands: CommandSet) -> None:
    stable = result.stable_gesture
    score = result.vote_score

    if result.click_fired:
        text, color = "LEFT CLICK", (0, 255, 255)
    elif result.right_click_fired:
        text, color = "RIGHT CLICK", (0, 200, 180)
    elif result.scroll_active:
        if result.scroll_delta > 0:
            direction = f"UP x{abs(result.scroll_delta)}"
        elif result.scroll_delta < 0:
            direction = f"DOWN x{abs(result.scroll_delta)}"
        else:
            direction = "ready"
        text, color = f"SCROLL {direction}", (100, 255, 150)
    elif result.fsm_active:
        pct = int(result.hold_progress * 100)
        text, color = f"L-Pinch {pct}%", (0, 200, 255)
    elif result.right_fsm_active:
        pct = int(result.right_hold_progress * 100)
        text, color = f"R-Pinch {pct}%", (0, 200, 180)
    elif stable != 0 and commands.has(stable):
        text, color = f"ACTION: {commands.name_for(stable)}", (0, 255, 0)
    elif result.capture.landmarks is None:
        text, color = "No hand detected", (128, 128, 128)
    else:
        text, color = "Tracking", (255, 255, 255)

    cv2.putText(frame, f"{text}  ({score}/{window_size})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_camera_banner(frame: np.ndarray, status: str) -> None:
    """Say out loud when the camera has gone away.

    Previously a disconnected camera just froze the last frame with no
    explanation, which is indistinguishable from the app hanging.
    """
    if status in ("running", "stopped"):
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    label = {
        "starting": "Opening camera...",
        "reconnecting": "Camera lost - reconnecting...",
        "failed": "Camera unavailable",
    }.get(status, status)
    cv2.putText(frame, label, (30, h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)


def _draw_click_arc(frame: np.ndarray, result: InferenceResult,
                    lm_idx: int, progress: float, color: tuple) -> None:
    """Generic charging arc near a fingertip."""
    if progress <= 0 or result.capture.landmarks is None:
        return
    h, w = frame.shape[:2]
    lm = result.capture.landmarks
    ix = int(lm[lm_idx].x * w)
    iy = int(lm[lm_idx].y * h)
    cv2.circle(frame, (ix, iy), 22, (60, 60, 60), 2)
    angle = int(progress * 360)
    if angle > 0:
        cv2.ellipse(frame, (ix, iy), (22, 22), -90, 0, angle, color, 2)
    cv2.circle(frame, (ix, iy), 5, color, -1)


def _draw_pinch_line(frame: np.ndarray, result: InferenceResult,
                     lm_a: int, lm_b: int, state: ClickState,
                     color_held: tuple, color_pressing: tuple) -> None:
    if result.capture.landmarks is None:
        return
    if state not in (ClickState.PRESSING, ClickState.HELD):
        return
    h, w = frame.shape[:2]
    lm = result.capture.landmarks
    a = (int(lm[lm_a].x * w), int(lm[lm_a].y * h))
    b = (int(lm[lm_b].x * w), int(lm[lm_b].y * h))
    color = color_held if state is ClickState.HELD else color_pressing
    cv2.line(frame, a, b, color, 2)


def _draw_scroll_indicator(frame: np.ndarray, result: InferenceResult) -> None:
    """Arrow near the wrist pointing in scroll direction."""
    if not result.scroll_active or result.capture.landmarks is None:
        return
    h, w = frame.shape[:2]
    lm = result.capture.landmarks
    wx = int(lm[0].x * w)
    wy = int(lm[0].y * h)
    arrow_len = 30
    tip_y = wy - arrow_len if result.scroll_delta >= 0 else wy + arrow_len
    cv2.arrowedLine(frame, (wx, wy), (wx, tip_y), (100, 255, 150), 2,
                    tipLength=0.4)


def _draw_volume_bar(frame: np.ndarray, vol: int) -> None:
    h = frame.shape[0]
    bar_top, bar_bot = int(h * 0.31), int(h * 0.83)
    bx1, bx2 = frame.shape[1] - 50, frame.shape[1] - 30
    bar_y = int(np.interp(vol, [0, 100], [bar_bot, bar_top]))
    cv2.rectangle(frame, (bx1, bar_top), (bx2, bar_bot), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx1, bar_y), (bx2, bar_bot), (0, 255, 255), -1)
    cv2.putText(frame, f"{vol}%", (bx1 - 10, bar_bot + 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)


def _draw_landmarks(frame: np.ndarray, result: InferenceResult) -> None:
    if result.capture.hand_lm_obj is None:
        return
    import mediapipe as mp
    mp.solutions.drawing_utils.draw_landmarks(
        frame, result.capture.hand_lm_obj, mp.solutions.hands.HAND_CONNECTIONS,
    )


def draw_overlay(frame: np.ndarray, result: InferenceResult, volume: int,
                 commands: CommandSet, cfg: AppConfig = DEFAULT_CONFIG,
                 camera_status: str = "running",
                 draw_landmarks: bool = True) -> None:
    _draw_active_zone(frame, cfg.mouse.frame_margin)
    _draw_status(frame, result, cfg.debounce.vote_window_size, commands)

    # Left-click arc (cyan) at index tip (8)
    _draw_click_arc(frame, result, 8, result.hold_progress, (0, 220, 255))
    # Right-click arc (teal) at middle tip (12)
    _draw_click_arc(frame, result, 12, result.right_hold_progress, (0, 200, 180))

    _draw_pinch_line(frame, result, 4, 8, result.fsm_state,
                     (0, 255, 255), (0, 180, 200))
    _draw_pinch_line(frame, result, 12, 8, result.right_fsm_state,
                     (0, 200, 180), (0, 150, 150))

    _draw_scroll_indicator(frame, result)
    _draw_volume_bar(frame, volume)
    # Landmark drawing is the single most expensive HUD element; it is the
    # first thing dropped when the frame budget is blown.
    if draw_landmarks:
        _draw_landmarks(frame, result)
    _draw_camera_banner(frame, camera_status)


# ============================================================================
# Gesture handlers -- these decide, the action thread does
# ============================================================================

class GestureRouter:
    """Turns an InferenceResult into zero or more actions.

    Pulled out of the render loop so it can be exercised with no camera, no
    window, and no OS side effects -- the replay harness and the integration
    test both drive this class directly and assert on what it emits.
    """

    def __init__(self, cfg: AppConfig = DEFAULT_CONFIG,
                 screen_w: int = 1920, screen_h: int = 1080) -> None:
        self._cfg = cfg
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._prev_wrist_y = 0.0
        self._last_vol_update = -float("inf")
        self._volume = 50

    def set_volume_reference(self, volume: int) -> None:
        self._volume = volume

    def route(self, result: InferenceResult, now: Optional[float] = None) -> list:
        """Return the actions this frame should produce, in priority order."""
        now = time.monotonic() if now is None else now
        out: list = []
        captured_at = result.capture.timestamp

        if result.action is not None:
            out.append(act.Command(result.action, captured_at))

        if result.click_fired:
            out.append(act.Click("left", captured_at))
        if result.right_click_fired:
            out.append(act.Click("right", captured_at))
        if result.scroll_delta != 0:
            out.append(act.Scroll(result.scroll_delta, captured_at))

        vol = self._route_volume(result, now)
        if vol is not None:
            out.append(act.SetVolume(vol, captured_at))

        move = self._route_cursor(result)
        if move is not None:
            out.append(act.MoveCursor(move[0], move[1], captured_at))

        return out

    # -- mode predicates ---------------------------------------------------
    #
    # These three are the mutual-exclusion contract.  Exactly one of
    # cursor/scroll/volume may be enabled for any given frame; the property
    # test in tests/test_properties.py asserts that over arbitrary landmarks.

    def cursor_enabled(self, result: InferenceResult) -> bool:
        if result.capture.landmarks is None or result.stable_gesture != 0:
            return False
        if result.fsm_active or result.right_fsm_active:
            return False
        if result.scroll_active:
            return False
        if result.thumb_raised:
            # Thumb up means the user is reaching for volume, not pointing.
            return False
        return result.index_extended

    def scroll_enabled(self, result: InferenceResult) -> bool:
        if result.capture.landmarks is None or result.stable_gesture != 0:
            return False
        return result.scroll_active

    def volume_enabled(self, result: InferenceResult) -> bool:
        if result.capture.landmarks is None or result.stable_gesture != 0:
            return False
        if result.scroll_active:
            return False
        if result.index_extended:
            # Index up means cursor mode; do not also grab the volume.
            return False
        if not result.thumb_raised:
            return False
        lm = result.capture.landmarks
        return lm[4].y < lm[5].y

    # -- routing helpers ---------------------------------------------------

    def _route_volume(self, result: InferenceResult, now: float) -> Optional[int]:
        if not self.volume_enabled(result):
            self._prev_wrist_y = 0.0
            return None

        cfg = self._cfg.volume
        wrist_y = result.capture.landmarks[0].y
        target = None
        if self._prev_wrist_y != 0.0:
            diff = self._prev_wrist_y - wrist_y
            if (abs(diff) > cfg.sensitivity
                    and now - self._last_vol_update > cfg.cooldown):
                step = cfg.step if diff > 0 else -cfg.step
                self._volume = max(0, min(100, self._volume + step))
                self._last_vol_update = now
                target = self._volume
        self._prev_wrist_y = wrist_y
        return target

    def _route_cursor(self, result: InferenceResult) -> Optional[tuple]:
        if not self.cursor_enabled(result):
            return None

        landmarks = result.capture.landmarks
        h, w = result.capture.frame.shape[:2]
        margin = self._cfg.mouse.frame_margin

        index_tip = landmarks[8]
        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        x_target = float(np.interp(ix, (margin, w - margin), (0, self.screen_w)))
        y_target = float(np.interp(iy, (margin, h - margin), (0, self.screen_h)))
        return x_target, y_target


# ============================================================================
# Pipeline assembly
# ============================================================================

class Pipeline:
    """Owns every thread and tears them all down together."""

    def __init__(self, cfg: AppConfig, commands: CommandSet,
                 metrics: MetricsRecorder, model) -> None:
        self.cfg = cfg
        self.commands = commands
        self.metrics = metrics
        self.stop_event = threading.Event()

        self.capture_q: queue.Queue = queue.Queue(
            maxsize=cfg.queues.inference_queue_size)
        self.inference_q: queue.Queue = queue.Queue(
            maxsize=cfg.queues.action_queue_size)

        self.controller = SystemController(cfg, stop_event=self.stop_event,
                                           commands=commands)
        self.controller.prime_volume()

        self.capture = CaptureThread(self.capture_q, cfg, self.stop_event,
                                     metrics=metrics)
        self.inference = InferenceThread(model, self.capture_q,
                                         self.inference_q, cfg,
                                         self.stop_event, metrics=metrics)
        self.dispatcher = ActionDispatcher(self.controller, self.stop_event,
                                           metrics=metrics)
        self.router = GestureRouter(cfg, self.controller.screen_w,
                                    self.controller.screen_h)
        self.router.set_volume_reference(self.controller.volume)

    def start(self) -> None:
        self.capture.start()
        self.inference.start()
        self.dispatcher.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.dispatcher.stop()
        self.capture.join(timeout=3.0)
        self.inference.join(timeout=3.0)
        self.dispatcher.join(timeout=3.0)
        self.controller.shutdown()


def load_model(model_file: Optional[Path] = None):
    model_file = model_file or models_path("gesture_classifier.pkl")
    if not model_file.exists():
        raise FileNotFoundError(
            f"No model at {model_file}. Run scripts/train_model.py first."
        )
    with open(model_file, "rb") as f:
        return pickle.load(f)


def prepare(cfg: AppConfig, commands_path: Optional[Path] = None,
            model_file: Optional[Path] = None,
            metrics_enabled: bool = True):
    """Load the model and commands, and check they agree about labels."""
    model = load_model(model_file)
    commands = load_commands(commands_path)

    warnings_ = validate_bindings(model.classes_, commands.labels(),
                                  neutral_label=commands.neutral_label)
    for warning in warnings_:
        print(f"[gestureflow] WARNING: {warning}")

    metrics = MetricsRecorder(enabled=metrics_enabled)
    return Pipeline(cfg, commands, metrics, model)


# ============================================================================
# Run loops
# ============================================================================

def run(cfg: AppConfig = DEFAULT_CONFIG,
        commands_path: Optional[Path] = None,
        metrics_out: Optional[Path] = None) -> int:
    """Interactive run with the OpenCV HUD."""
    try:
        pipe = prepare(cfg, commands_path)
    except (FileNotFoundError, BindingError) as exc:
        print(f"[gestureflow] ERROR: {exc}")
        return 1

    pipe.start()
    print("[gestureflow] Running. Press 'q' in the window to quit.")

    budget = 1.0 / cfg.hud.min_fps
    draw_landmarks = True

    try:
        while not pipe.stop_event.is_set():
            try:
                result: InferenceResult = pipe.inference_q.get(timeout=0.05)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            for action in pipe.router.route(result):
                pipe.dispatcher.submit(action)

            frame = result.capture.frame.copy()
            render_start = time.monotonic()
            with pipe.metrics.timer(STAGE_RENDER):
                draw_overlay(frame, result, pipe.controller.volume,
                             pipe.commands, cfg,
                             camera_status=pipe.capture.status,
                             draw_landmarks=draw_landmarks)
                cv2.imshow("GestureFlow", frame)
            render_cost = time.monotonic() - render_start

            # Graceful degradation: if drawing alone is eating the frame
            # budget, stop drawing the skeleton. Recognition is unaffected --
            # only the overlay gets simpler.
            if render_cost > budget and draw_landmarks:
                draw_landmarks = False
                pipe.metrics.count("hud.degraded")
                print("[gestureflow] Frame budget exceeded; "
                      "dropping landmark overlay.")
            elif render_cost < budget * 0.5 and not draw_landmarks:
                draw_landmarks = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("[gestureflow] Shutting down…")
        pipe.shutdown()
        cv2.destroyAllWindows()

    if metrics_out:
        path = pipe.metrics.write_json(metrics_out)
        print(f"[gestureflow] Metrics written to {path}")
    return 0


def run_headless(seconds: float, cfg: AppConfig = DEFAULT_CONFIG,
                 commands_path: Optional[Path] = None,
                 metrics_out: Optional[Path] = None,
                 dry_run: bool = True) -> int:
    """Run the pipeline with no window for `seconds`, then report timings.

    `dry_run` (the default) routes actions through a controller stub that
    performs nothing, so benchmarking does not fling the real cursor around.
    """
    try:
        pipe = prepare(cfg, commands_path)
    except (FileNotFoundError, BindingError) as exc:
        print(f"[gestureflow] ERROR: {exc}")
        return 1

    if dry_run:
        pipe.dispatcher._ctrl = _NullController(pipe.controller.screen_w,
                                                pipe.controller.screen_h)

    pipe.start()
    print(f"[gestureflow] Benchmarking for {seconds:.0f}s "
          f"({'dry run' if dry_run else 'LIVE - will control your Mac'})…")

    deadline = time.monotonic() + seconds
    frames = 0
    try:
        while time.monotonic() < deadline:
            try:
                result = pipe.inference_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frames += 1
            for action in pipe.router.route(result):
                pipe.dispatcher.submit(action)
    finally:
        pipe.dispatcher.flush(timeout=2.0)
        pipe.shutdown()

    pipe.metrics.count("render.frames", frames)
    print()
    print(pipe.metrics.format_text())

    out = metrics_out or Path("bench") / f"bench-{int(time.time())}.json"
    path = pipe.metrics.write_json(out, extra={
        "mode": "bench",
        "requested_seconds": seconds,
        "dry_run": dry_run,
        "camera": {"width": cfg.camera.width, "height": cfg.camera.height},
    })
    print(f"\n[gestureflow] Report written to {path}")
    return 0


class _NullController:
    """Accepts every action and performs none, for dry-run benchmarking."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.volume = 50

    def move_mouse_smooth(self, x, y, now=None): pass
    def click(self): pass
    def right_click(self): pass
    def scroll(self, delta): pass
    def set_volume(self, value): self.volume = value
    def execute_command(self, gesture_id): pass
