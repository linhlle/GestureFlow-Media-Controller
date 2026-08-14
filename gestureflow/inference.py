from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from gestureflow.capture import CaptureResult
from gestureflow.click_fsm import ClickFSM, ClickState
from gestureflow.config import DEFAULT_CONFIG, AppConfig
from gestureflow.debouncer import GestureDebouncer
from gestureflow.metrics import (
    HANDOFF_INFERENCE,
    STAGE_FSM,
    STAGE_NORMALIZE,
    STAGE_PREDICT,
    MetricsRecorder,
    NullMetrics,
)
from gestureflow.pause_fsm import PauseFSM
from gestureflow.scroll_fsm import (
    ScrollFSM,
    ScrollState,
    _index_extended,
    _is_true_scroll_fist,
    _thumb_raised,
)
from gestureflow.swipe_fsm import SwipeFSM
from gestureflow.utils import drop_oldest_put, normalize_landmarks


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

    # Scroll FSM
    scroll_delta: int
    scroll_active: bool
    scroll_state: ScrollState
    index_extended: bool
    thumb_raised: bool

    # Swipe
    swipe_direction: Optional[str] = None
    swipe_armed: bool = False

    # Drag
    drag_started: bool = False
    drag_ended: bool = False
    dragging: bool = False
    drag_progress: float = 0.0

    # Pause kill switch
    paused: bool = False
    pause_toggled: bool = False
    pause_progress: float = 0.0

class InferenceThread(threading.Thread):
    def __init__(
            self,
            model: Any,
            in_queue: queue.Queue,
            out_queue: queue.Queue,
            config: AppConfig | None = None,
            stop_event: threading.Event | None = None,
            metrics: MetricsRecorder | None = None,
            clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="inference-thread", daemon=True)
        self._model = model
        self._in_q = in_queue
        self._out_q = out_queue
        self._cfg = config or DEFAULT_CONFIG
        self._stop_event = stop_event or threading.Event()
        self._metrics = metrics or NullMetrics()
        self._dropped = 0

        cfg = config or DEFAULT_CONFIG
        self._debouncer = GestureDebouncer(
            config=cfg.debounce,
            confidence_threshold=cfg.inference.confidence_threshold,
            clock=clock,
        )

        # Left-click: thumb(4) + index(8)
        self._left_fsm  = ClickFSM(config=cfg.click,       landmark_a=4,  landmark_b=8,
                                   clock=clock, drag=cfg.drag)
        # Right-click: middle(12) + index(8)
        self._right_fsm = ClickFSM(config=cfg.right_click,  landmark_a=12, landmark_b=8,
                                   clock=clock)
        self._scroll_fsm = ScrollFSM(config=cfg.scroll, clock=clock)
        self._pause_fsm = PauseFSM(config=cfg.pause, clock=clock)
        self._swipe_fsm = SwipeFSM(config=cfg.swipe, clock=clock)

    def run(self) -> None:
        print("[inference] Starting inference loop.")
        while not self._stop_event.is_set():
            try:
                capture: CaptureResult = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue



            self._emit(self.process(capture))

        print("[inference] Inference loop stopped.")

    def process(self, capture: CaptureResult) -> InferenceResult:
        """Run one frame through the recognizer.

        Public so the replay harness and integration tests can drive the exact
        same code path synchronously, with no threads and no camera.
        """
        return self._process(capture)

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def dropped(self) -> int:
        """Results discarded because the render loop could not keep up."""
        return self._dropped

    def _process(self, capture: CaptureResult) -> InferenceResult:
        lm = capture.landmarks
        if lm is None:
            # No hand detected -> feed Neutral to debouncer so the vote
            # window drains naturally
            self._debouncer.update(0, 1.0)
            self._left_fsm.update(None)
            self._right_fsm.update(None)
            self._scroll_fsm.update(None)
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
                scroll_delta=0, scroll_active=False,
                scroll_state=self._scroll_fsm.state,
                index_extended=False,
                thumb_raised=False
            )


        # Runs above the classifier on purpose: the kill switch has to work
        # even if the model reads the pose as something else entirely.
        self._pause_fsm.update(lm)

        with self._metrics.timer(STAGE_NORMALIZE):
            normalized_feat = normalize_landmarks(lm)

        with self._metrics.timer(STAGE_PREDICT):
            probs: np.ndarray = self._model.predict_proba([normalized_feat])[0]
            raw_pred = int(np.argmax(probs))
            confidence = float(probs[raw_pred])

        action = self._debouncer.update(raw_pred, confidence)
        stable = self._debouncer.stable_gesture

        with self._metrics.timer(STAGE_FSM):
            # Pause all Neutral-mode FSMs when a named gesture is active.  This
            # is what makes the modes mutually exclusive: a recognized command
            # gesture parks the pinch, fist, and thumb detectors entirely.
            if stable != 0:
                self._left_fsm.update(None)
                self._right_fsm.update(None)
                self._scroll_fsm.update(None)
                self._swipe_fsm.update(None)
            else:
                # A fist resolves to scroll and nothing else. In a closed hand
                # the middle and index fingertips sit side by side, which the
                # right-click pinch reads as a deliberate touch -- measured on
                # 22% of real fist frames, enough to latch the FSM. Deciding
                # this once, here, means the two interpretations can never
                # race: scroll wins by construction rather than by threshold
                # tuning.
                fist = _is_true_scroll_fist(lm)
                self._left_fsm.update(None if fist else lm)
                self._right_fsm.update(None if fist else lm)
                self._scroll_fsm.update(lm)
                self._swipe_fsm.update(lm)


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
            scroll_delta=self._scroll_fsm.scroll_delta,
            scroll_active=self._scroll_fsm.is_active,
            scroll_state=self._scroll_fsm.state,
            index_extended=_index_extended(lm),
            thumb_raised=_thumb_raised(lm),
            swipe_direction=self._swipe_fsm.direction,
            swipe_armed=self._swipe_fsm.is_armed,
            drag_started=self._left_fsm.drag_started,
            drag_ended=self._left_fsm.drag_ended,
            dragging=self._left_fsm.dragging,
            drag_progress=self._left_fsm.drag_progress,
            paused=self._pause_fsm.paused,
            pause_toggled=self._pause_fsm.toggled,
            pause_progress=self._pause_fsm.progress,
        )



    def _emit(self, result: InferenceResult) -> None:
        self._metrics.observe_queue(HANDOFF_INFERENCE, self._out_q.qsize())
        dropped = drop_oldest_put(self._out_q, result)
        if dropped:
            self._dropped += dropped
            self._metrics.count("inference.results_dropped", dropped)
