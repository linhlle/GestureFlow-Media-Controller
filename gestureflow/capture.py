from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np

from gestureflow.config import AppConfig, DEFAULT_CONFIG
from gestureflow.metrics import (
    HANDOFF_CAPTURE,
    STAGE_GRAB,
    STAGE_MEDIAPIPE,
    MetricsRecorder,
    NullMetrics,
)
from gestureflow.utils import drop_oldest_put


@dataclass
class CaptureResult:
    frame: np.ndarray
    landmarks: Optional[Any]
    hand_lm_obj: Optional[Any]
    timestamp: float


class CameraUnavailable(RuntimeError):
    pass


class CaptureThread(threading.Thread):
    """Grabs frames, runs MediaPipe, and publishes landmark results.

    Survives the camera going away.  A USB camera being unplugged, or macOS
    handing the device to another app, used to leave this thread spinning on
    `cap.read()` returning False forever with nothing on screen explaining why.
    It now reconnects with exponential backoff and exposes the failure through
    `status` so the HUD can say what is happening.
    """

    def __init__(
            self,
            out_queue: queue.Queue,
            config: AppConfig | None = None,
            stop_event: threading.Event | None = None,
            metrics: MetricsRecorder | None = None,
    ) -> None:
        super().__init__(name="capture-thread", daemon=True)
        self._q = out_queue
        self._cfg = config or DEFAULT_CONFIG
        self._stop = stop_event or threading.Event()
        self._metrics = metrics or NullMetrics()
        self._dropped = 0
        self._status = "starting"
        self._status_lock = threading.Lock()

    @property
    def dropped(self) -> int:
        """Frames discarded because inference could not keep up."""
        return self._dropped

    @property
    def status(self) -> str:
        """One of: starting, running, reconnecting, failed, stopped."""
        with self._status_lock:
            return self._status

    def _set_status(self, value: str) -> None:
        with self._status_lock:
            self._status = value

    # -- camera lifecycle --------------------------------------------------

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        cam_cfg = self._cfg.camera
        cap = cv2.VideoCapture(cam_cfg.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)

        deadline = time.monotonic() + cam_cfg.open_timeout
        while not cap.isOpened():
            if self._stop.is_set():
                cap.release()
                return None
            if time.monotonic() > deadline:
                cap.release()
                return None
            time.sleep(0.05)
        return cap

    def run(self) -> None:
        mp_cfg = self._cfg.mediapipe
        cam_cfg = self._cfg.camera

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=mp_cfg.max_num_hands,
            min_detection_confidence=mp_cfg.min_detection_confidence,
            min_tracking_confidence=mp_cfg.min_tracking_confidence,
        )

        cap: Optional[cv2.VideoCapture] = None
        backoff = cam_cfg.reconnect_backoff_min
        consecutive_read_failures = 0

        try:
            while not self._stop.is_set():
                if cap is None:
                    self._set_status("reconnecting" if backoff >
                                     cam_cfg.reconnect_backoff_min else "starting")
                    cap = self._open_camera()
                    if cap is None:
                        if self._stop.is_set():
                            break
                        self._metrics.count("capture.reconnect_attempts")
                        self._set_status("reconnecting")
                        print(f"[capture] Camera unavailable; retrying in "
                              f"{backoff:.1f}s")
                        # wait() rather than sleep() so shutdown is immediate
                        if self._stop.wait(backoff):
                            break
                        backoff = min(backoff * 2, cam_cfg.reconnect_backoff_max)
                        continue

                    backoff = cam_cfg.reconnect_backoff_min
                    consecutive_read_failures = 0
                    self._set_status("running")
                    print("[capture] Camera opened. Starting capture loop.")

                with self._metrics.timer(STAGE_GRAB):
                    success, frame = cap.read()

                if not success:
                    consecutive_read_failures += 1
                    self._metrics.count("capture.read_failures")
                    if consecutive_read_failures >= cam_cfg.read_failure_limit:
                        # The device is gone, not merely slow. Tear down and
                        # let the reconnect path handle it.
                        print("[capture] Lost camera; attempting reconnect.")
                        cap.release()
                        cap = None
                        self._set_status("reconnecting")
                    else:
                        time.sleep(0.01)
                    continue

                consecutive_read_failures = 0
                frame = cv2.flip(frame, 1)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False         # avoid a copy inside MP
                with self._metrics.timer(STAGE_MEDIAPIPE):
                    results = hands.process(img_rgb)
                img_rgb.flags.writeable = True

                landmarks = None
                hand_lm_obj = None
                if results.multi_hand_landmarks:
                    hand_lm_obj = results.multi_hand_landmarks[0]
                    landmarks = hand_lm_obj.landmark

                result = CaptureResult(
                    frame=frame,
                    landmarks=landmarks,
                    hand_lm_obj=hand_lm_obj,
                    timestamp=time.monotonic(),
                )

                self._metrics.observe_queue(HANDOFF_CAPTURE, self._q.qsize())
                dropped = drop_oldest_put(self._q, result)
                if dropped:
                    self._dropped += dropped
                    self._metrics.count("capture.frames_dropped", dropped)
        finally:
            hands.close()
            if cap is not None:
                cap.release()
            self._set_status("stopped")
            print("[capture] Camera released.")

    def stop(self) -> None:
        self._stop.set()
