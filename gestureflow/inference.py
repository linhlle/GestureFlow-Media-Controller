from __future__ import annotations

import threading
import queue
import numpy as np

from dataclasses import dataclass
from typing import Optional, Any

from gestureflow.config import AppConfig, DEFAULT_CONFIG
from gestureflow.capture import CaptureResult
from gestureflow.debouncer import GestureDebouncer
from gestureflow.utils import normalize_landmarks


@dataclass
class InferenceResult:
    capture: CaptureResult
    stable_gesture: int
    vote_score: int
    confidence: float
    raw_prediction: int
    action: Optional[int]

class InferenceThread(threading.Thread):
    def __init__(
            self,
            model: Any,
            in_queue: queue.Queue,
            out_queue: queue.Queue,
            config: AppConfig | None = None,
            stop_event: threading.Event | None = None
    ) -> None:
        super().__init__(name="inference-thread", daemon=True)
        self._model = model
        self._in_q = in_queue
        self._out_q = out_queue
        self._cfg = config or DEFAULT_CONFIG
        self._stop = stop_event or threading.Event()
        self._debouncer = GestureDebouncer(
            config=config.debounce if config else None,
            confidence_threshold=config.inference.confidence_threshold if config else None,
        )

    def run(self) -> None:
        print("[inference] Starting inference loop.")
        while not self._stop.is_set():
            try:
                capture: CaptureResult = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if capture.landmarks is None:
                # No hand detected -> feed Neutral to debouncer so the vote 
                # window drains naturally
                self._debouncer.update(0, 1.0)
                result = InferenceResult(
                    capture=capture,
                    stable_gesture=0,
                    vote_score=self._debouncer.vote_score,
                    confidence=0.0,
                    raw_prediction=0,
                    action=None
                )
            else:
                normalized_feat = normalize_landmarks(capture.landmarks)
                probs: np.ndarray = self._model.predict_proba([normalized_feat])[0]
                raw_pred = int(np.argmax(probs))
                confidence = float(probs[raw_pred])

                action = self._debouncer.update(raw_pred, confidence)

                result = InferenceResult(
                    capture=capture,
                    stable_gesture=self._debouncer.stable_gesture,
                    vote_score=self._debouncer.vote_score,
                    confidence=confidence,
                    raw_prediction=raw_pred,
                    action=action
                )

            try:
                self._out_q.put_nowait(result)
            except queue.Full:
                # Replace state result with fresher one
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_q.put_nowait(result)
                except queue.Full:
                    pass
            
        print("[inference] Inference loop stopped.")

    def stop(self) -> None:
        self._stop.set()