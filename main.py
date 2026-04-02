"""
main.py
-------
GestureFlow entry point.

Thread layout
-------------

  ┌──────────────────┐    Queue[2]    ┌──────────────────┐    Queue[1]    ┌──────────────────────┐
  │  CaptureThread   │ ─────────────► │ InferenceThread  │ ─────────────► │   Main (render loop) │
  │  camera + MP     │                │ RF + debounce    │                │   OpenCV overlay      │
  └──────────────────┘                └──────────────────┘                │   SystemController   │
                                                                           └──────────────────────┘

The main thread is the only thread that calls OpenCV GUI functions (cv2.imshow,
cv2.waitKey) because OpenCV's HighGUI is not thread-safe.

The SystemController's volume worker and sync worker are additional daemon
threads managed inside the controller itself.
"""

from __future__ import annotations

import os
import pickle
import queue
import sys
import threading
import time
import warnings

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

from gestureflow.capture import CaptureThread
from gestureflow.config import DEFAULT_CONFIG, GESTURE_MAP
from gestureflow.controller import SystemController
from gestureflow.inference import InferenceResult, InferenceThread
from gestureflow.utils import models_path
from gestureflow.click_fsm import ClickState



# ============================================================================
# HUD rendering
# ============================================================================
 
def _draw_active_zone(frame: np.ndarray, margin: int) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (margin, margin), (w - margin, h - margin),
                  (255, 0, 0), 2)
 
 
def _draw_status(frame: np.ndarray, result: InferenceResult,
                 window_size: int) -> None:
    stable = result.stable_gesture
    score  = result.vote_score
 
    if result.click_fired:
        text, color = "LEFT CLICK", (0, 255, 255)
    elif result.right_click_fired:
        text, color = "RIGHT CLICK", (0, 200, 180)
    # elif result.scroll_active:
    #     direction = "UP" if result.scroll_delta > 0 else ("DOWN" if result.scroll_delta < 0 else "...")
    #     text, color = f"SCROLL {direction}", (100, 255, 150)
    elif result.fsm_active:
        pct = int(result.hold_progress * 100)
        text, color = f"L-Pinch {pct}%", (0, 200, 255)
    elif result.right_fsm_active:
        pct = int(result.right_hold_progress * 100)
        text, color = f"R-Pinch {pct}%", (0, 200, 180)
    elif stable != 0 and stable in GESTURE_MAP:
        text, color = f"ACTION: {GESTURE_MAP[stable]['name']}", (0, 255, 0)
    elif result.capture.landmarks is None:
        text, color = "No hand detected", (128, 128, 128)
    else:
        text, color = "Tracking", (255, 255, 255)
 
    cv2.putText(frame, f"{text}  ({score}/{window_size})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
 
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
    cv2.arrowedLine(frame, (wx, wy), (wx, tip_y), (100, 255, 150), 2, tipLength=0.4)
 
 
def _draw_volume_bar(frame: np.ndarray, vol: int) -> None:
    bar_top, bar_bot = 150, 400
    bx1, bx2 = 590, 610
    bar_y = int(np.interp(vol, [0, 100], [bar_bot, bar_top]))
    cv2.rectangle(frame, (bx1, bar_top), (bx2, bar_bot), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx1, bar_y),   (bx2, bar_bot), (0, 255, 255), -1)
    cv2.putText(frame, f"{vol}%", (bx1 - 10, bar_bot + 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
 
 
def _draw_landmarks(frame: np.ndarray, result: InferenceResult) -> None:
    if result.capture.hand_lm_obj is None:
        return
    import mediapipe as mp
    mp.solutions.drawing_utils.draw_landmarks(
        frame, result.capture.hand_lm_obj, mp.solutions.hands.HAND_CONNECTIONS,
    )
 
 
def _draw_overlay(frame: np.ndarray, result: InferenceResult,
                  ctrl: SystemController, cfg=DEFAULT_CONFIG) -> None:
    _draw_active_zone(frame, cfg.mouse.frame_margin)
    _draw_status(frame, result, cfg.debounce.vote_window_size)
 
    # Left-click arc (cyan) at index tip (8)
    _draw_click_arc(frame, result, 8, result.hold_progress,       (0, 220, 255))
    # Right-click arc (teal) at middle tip (12)
    _draw_click_arc(frame, result, 12, result.right_hold_progress, (0, 200, 180))
 
    # Pinch lines
    _draw_pinch_line(frame, result, 4, 8,  result.fsm_state,
                     (0, 255, 255), (0, 180, 200))
    _draw_pinch_line(frame, result, 12, 8, result.right_fsm_state,
                     (0, 200, 180), (0, 150, 150))
 
    # _draw_scroll_indicator(frame, result)
    _draw_volume_bar(frame, ctrl.volume)
    _draw_landmarks(frame, result)
 
 

# ============================================================================
# Gesture handlers
# ============================================================================
 
def _handle_left_click(result: InferenceResult, ctrl: SystemController) -> None:
    if result.click_fired:
        ctrl.click()
        print("[main] Left click")
 
 
def _handle_right_click(result: InferenceResult, ctrl: SystemController) -> None:
    if result.right_click_fired:
        ctrl.right_click()
        print("[main] Right click")

def _handle_volume(result: InferenceResult, ctrl: SystemController,
                   prev_y: list, last_update: list, cfg=DEFAULT_CONFIG) -> None:
    if result.capture.landmarks is None or result.stable_gesture != 0:
        prev_y[0] = 0.0
        return
    lm = result.capture.landmarks
    if lm[4].y >= lm[5].y:
        prev_y[0] = 0.0
        return
    wrist_y = lm[0].y
    if prev_y[0] != 0.0:
        diff = prev_y[0] - wrist_y
        now  = time.monotonic()
        if abs(diff) > cfg.volume.sensitivity and now - last_update[0] > cfg.volume.cooldown:
            step = cfg.volume.step if diff > 0 else -cfg.volume.step
            ctrl.set_volume(max(0, min(100, ctrl.volume + step)))
            last_update[0] = now
    prev_y[0] = wrist_y



# ---------------------------------------------------------------------------
# Mouse tracking (runs in the main thread — pure math, no I/O)
# ---------------------------------------------------------------------------

def _handle_mouse(
    result: InferenceResult,
    ctrl: SystemController,
    cfg=DEFAULT_CONFIG,
) -> None:
    """Move the mouse based on index-finger tip position."""
    if result.capture.landmarks is None or result.stable_gesture != 0:
        return
    
    if result.fsm_active:                    return   # left-click pinch
    if result.right_fsm_active:              return   # right-click pinch


    landmarks = result.capture.landmarks
    h, w = result.capture.frame.shape[:2]
    margin = cfg.mouse.frame_margin

    index_tip = landmarks[8]
    ix = int(index_tip.x * w)
    iy = int(index_tip.y * h)

    x_target = float(np.interp(ix, (margin, w - margin), (0, ctrl.screen_w)))
    y_target = float(np.interp(iy, (margin, h - margin), (0, ctrl.screen_h + 50)))
    ctrl.move_mouse_smooth(x_target, y_target)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = DEFAULT_CONFIG

    # --- Load model ---
    model_file = models_path("gesture_classifier.pkl")
    if not model_file.exists():
        print(f"[main] ERROR: Model not found at {model_file}")
        print("[main] Run scripts/train_model.py first.")
        sys.exit(1)

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    print(f"[main] Loaded model from {model_file}")

    # --- Shared shutdown event ---
    stop_event = threading.Event()

    # --- Queues ---
    capture_q: queue.Queue = queue.Queue(maxsize=cfg.queues.inference_queue_size)
    inference_q: queue.Queue = queue.Queue(maxsize=cfg.queues.action_queue_size)

    # --- Controller (starts its own background threads) ---
    ctrl = SystemController(cfg)
    ctrl.prime_volume()  

    # --- Worker threads ---
    cap_thread = CaptureThread(capture_q, cfg, stop_event)
    inf_thread = InferenceThread(model, capture_q, inference_q, cfg, stop_event)

    cap_thread.start()
    inf_thread.start()

    # --- Mutable state that lives in the main thread only ---
    prev_y: list[float] = [0.0]
    last_vol: list[float] = [0.0]


    print("[main] Pipeline running. Press 'q' in the OpenCV window to quit.")

    # --- Render loop (main thread only) ---
    while not stop_event.is_set():
        try:
            result: InferenceResult = inference_q.get(timeout=0.05)
        except queue.Empty:
            # No new result yet — still call waitKey so the window stays alive
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        frame = result.capture.frame.copy()

        # Execute any confirmed gesture action
        if result.action is not None:
            ctrl.execute_command(result.action)

        _handle_left_click(result, ctrl)
        _handle_right_click(result, ctrl)
        _handle_volume(result, ctrl, prev_y, last_vol, cfg)
        _handle_mouse(result, ctrl, cfg)

        _draw_overlay(frame, result, ctrl, cfg)



        cv2.imshow("GestureFlow", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Clean shutdown ---
    print("[main] Shutting down…")
    stop_event.set()
    cap_thread.join(timeout=3.0)
    inf_thread.join(timeout=3.0)
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    main()