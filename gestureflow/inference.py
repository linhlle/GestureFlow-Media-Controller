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
from gestureflow.click_fsm import ClickFSM, ClickState


@dataclass
class InferenceResult:
    capture: CaptureResult
    stable_gesture: int
    vote_score: int
    confidence: float
    raw_prediction: int
    action: Optional[int]

    # Left click FSM
    click_fired: bool
    fsm_active: bool
    fsm_state: ClickState
    hold_progress: float

    #Right click
    right_click_fired: bool
    right_fsm_active: bool
    right_fsm_state: ClickState
    right_hold_progress: float

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

        cfg = config or DEFAULT_CONFIG
        self._debouncer = GestureDebouncer(
            config=cfg.debounce,
            confidence_threshold=cfg.inference.confidence_threshold,
        )

        # Left-click: thumb(4) + index(8)
        self._left_fsm  = ClickFSM(config=cfg.click,       landmark_a=4,  landmark_b=8)
        # Right-click: middle(12) + index(8)
        self._right_fsm = ClickFSM(config=cfg.right_click,  landmark_a=12, landmark_b=8)


    def run(self) -> None:
        print("[inference] Starting inference loop.")
        while not self._stop.is_set():
            try:
                capture: CaptureResult = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue



            self._emit(self._process(capture))
            
        print("[inference] Inference loop stopped.")

    def stop(self) -> None:
        self._stop.set()


    def _process(self, capture: CaptureResult) -> InferenceResult:
        lm = capture.landmarks
        if lm is None:
            # No hand detected -> feed Neutral to debouncer so the vote 
            # window drains naturally
            self._debouncer.update(0, 1.0)
            self._left_fsm.update(None)
            self._right_fsm.update(None)
            return InferenceResult(
                capture=capture,
                stable_gesture=0,
                vote_score=self._debouncer.vote_score,
                confidence=0.0,
                raw_prediction=0,
                action=None, 
                click_fired=False, fsm_active=False,
                fsm_state=self._left_fsm.state, hold_progress=0.0,
                right_click_fired=False, right_fsm_active=False,
                right_fsm_state=self._right_fsm.state, right_hold_progress=0.0,

            )
        

        normalized_feat = normalize_landmarks(lm)
        probs: np.ndarray = self._model.predict_proba([normalized_feat])[0]
        raw_pred = int(np.argmax(probs))
        confidence = float(probs[raw_pred])

        action = self._debouncer.update(raw_pred, confidence)
        stable = self._debouncer.stable_gesture

        # Pause all Neutral-mode FSMs when a named gesture is active
        if stable != 0:
            self._left_fsm.update(None)
            self._right_fsm.update(None)
            # self._scroll_fsm.update(None)
        else:
            self._left_fsm.update(lm)
            self._right_fsm.update(lm)
            # self._scroll_fsm.update(lm)
 

        return InferenceResult(
            capture=capture,
            stable_gesture=stable,
            vote_score=self._debouncer.vote_score,
            confidence=confidence,
            raw_prediction=raw_pred,
            action=action,
            click_fired=self._left_fsm.click_fired,
            fsm_active=self._left_fsm.is_active,
            fsm_state=self._left_fsm.state,
            hold_progress=self._left_fsm.hold_progress,
            right_click_fired=self._right_fsm.click_fired,
            right_fsm_active=self._right_fsm.is_active,
            right_fsm_state=self._right_fsm.state,
            right_hold_progress=self._right_fsm.hold_progress,
        )



    def _emit(self, result: InferenceResult) -> None:
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
