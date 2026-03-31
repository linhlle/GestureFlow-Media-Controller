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


# ---------------------------------------------------------------------------
# Overlay rendering helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    result: InferenceResult,
    ctrl: SystemController,
    cfg=DEFAULT_CONFIG,
) -> None:
    """Draw all HUD elements onto ``frame`` in-place."""
    h, w = frame.shape[:2]
    margin = cfg.mouse.frame_margin

    cv2.rectangle(
        frame,
        (margin, margin),
        (w - margin, h - margin),
        (255, 0, 0), 2,
    )

    stable = result.stable_gesture
    score = result.vote_score
    window = cfg.debounce.vote_window_size

    if stable != 0 and stable in GESTURE_MAP:
        status = f"ACTION: {GESTURE_MAP[stable]['name']}"
        color = (0, 255, 0)
    elif result.capture.landmarks is None:
        status = "No hand detected"
        color = (128, 128, 128)
    else:
        status = "Tracking..."
        color = (255, 255, 255)

    cv2.putText(
        frame,
        f"{status}  ({score}/{window})",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2,
    )

    _draw_volume_bar(frame, ctrl.volume, h)

    if result.capture.hand_lm_obj is not None:
        import mediapipe as mp
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.capture.hand_lm_obj,
            mp.solutions.hands.HAND_CONNECTIONS,
        )


def _draw_volume_bar(frame: np.ndarray, vol: int, frame_h: int) -> None:
    bar_top, bar_bottom = 150, 400
    bar_x1, bar_x2 = 590, 610
    bar_y = int(np.interp(vol, [0, 100], [bar_bottom, bar_top]))

    cv2.rectangle(frame, (bar_x1, bar_top), (bar_x2, bar_bottom), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_bottom), (0, 255, 255), -1)
    cv2.putText(
        frame, f"{vol}%", (bar_x1 - 10, bar_bottom + 20),
        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2,
    )


# ---------------------------------------------------------------------------
# Volume gesture detection (runs in the main thread — pure math, no I/O)
# ---------------------------------------------------------------------------

def _handle_volume_gesture(
    result: InferenceResult,
    ctrl: SystemController,
    prev_y: list,              
    last_vol_update: list,
    cfg=DEFAULT_CONFIG,
) -> None:
    if result.capture.landmarks is None or result.stable_gesture != 0:
        prev_y[0] = 0.0
        return

    landmarks = result.capture.landmarks
    thumb_tip = landmarks[4]
    index_knuckle = landmarks[5]

    if thumb_tip.y >= index_knuckle.y:
        prev_y[0] = 0.0
        return

    wrist_y = landmarks[0].y
    if prev_y[0] != 0.0:
        diff = prev_y[0] - wrist_y
        now = time.monotonic()
        if (
            abs(diff) > cfg.volume.sensitivity
            and (now - last_vol_update[0]) > cfg.volume.cooldown
        ):
            change = cfg.volume.step if diff > 0 else -cfg.volume.step
            new_vol = max(0, min(100, ctrl.volume + change))
            ctrl.set_volume(new_vol)
            last_vol_update[0] = now

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
    ctrl.prime_volume()   # blocking one-shot read at startup

    # --- Worker threads ---
    cap_thread = CaptureThread(capture_q, cfg, stop_event)
    inf_thread = InferenceThread(model, capture_q, inference_q, cfg, stop_event)

    cap_thread.start()
    inf_thread.start()

    # --- Mutable state that lives in the main thread only ---
    prev_y: list[float] = [0.0]
    last_vol_update: list[float] = [0.0]

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
            ctrl.execute_gesture(result.action)

        # Volume gesture (thumb slide)
        _handle_volume_gesture(result, ctrl, prev_y, last_vol_update, cfg)

        # Mouse tracking (neutral mode only)
        _handle_mouse(result, ctrl, cfg)

        # Draw HUD
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