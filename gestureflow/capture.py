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

@dataclass
class CaptureResult:
    frame: np.array
    landmarks: Optional[Any]
    hand_lm_obj: Optional[Any]
    timestamp: float


class CaptureThread(threading.Thread):

    def __init__(
            self,
            out_queue: queue.Queue,
            config: AppConfig | None = None,
            stop_event: threading.Event | None = None
    ) -> None:
        super().__init__(name="capture-thread", daemon=True)
        self._q = out_queue
        self._cfg = config or DEFAULT_CONFIG
        self._stop = stop_event or threading.Event()


    def run(self) -> None:
        cam_cfg = self._cfg.camera
        mp_cfg = self._cfg.mediapipe

        cap = cv2.VideoCapture(cam_cfg.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)

        timeout = time.monotonic() + 5.0
        while not cap.isOpened():
            if time.monotonic() > timeout:
                print("[capture] ERROR: Could not open camera.")
                return
        time.sleep(0.1)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=mp_cfg.max_num_hands,
            min_detection_confidence=mp_cfg.min_detection_tracking,
            min_tracking_confidence=mp_cfg.min_tracking_confidence
        )

        print("[capture] Camera opened. Starting capture loop.")
        try:
            while not self._stop.is_set():
                success, frame = cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False         # avoid a copy inside MP
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

                # Non-blocking put — drop frame if inference is backed up
                try:
                    self._q.put_nowait(result)
                except queue.Full:
                    pass   # intentional: always use freshest data


        finally:
            hands.close()
            cap.release()
            print("[capture] Camera released.")


    def stop(self) -> None:
        self._stop.set()
