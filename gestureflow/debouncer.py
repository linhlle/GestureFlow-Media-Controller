from __future__ import annotations

import time
from collections import Counter, deque

from gestureflow.config import AppConfig, DebounceConfig, DEFAULT_CONFIG


class GestureDebouncer:
    """Temporal majority-voting filter for raw classifier predictions.

    Typical usage
    -------------
    ::

        debouncer = GestureDebouncer()

        # Inside the frame loop:
        action = debouncer.update(predicted_label, confidence)
        if action is not None:
            controller.execute_gesture(action)

    """

    def __init__(
        self,
        config: DebounceConfig | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG.debounce
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else DEFAULT_CONFIG.inference.confidence_threshold
        )
        self._history: deque[int] = deque(maxlen=self._cfg.vote_window_size)
        self._last_cmd_time: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, prediction: int, confidence: float) -> int | None:
        # Low-confidence predictions are clamped to Neutral so they don't
        # accumulate spurious votes for real gestures.
        effective = prediction if confidence >= self._confidence_threshold else 0
        self._history.append(effective)

        stable, score = self._majority()

        if stable == 0 or score < self._cfg.vote_threshold:
            return None  

        now = time.monotonic()
        if now - self._last_cmd_time < self._cfg.cmd_cooldown:
            return None  

        self._last_cmd_time = now
        self._history.clear()
        return stable

    @property
    def stable_gesture(self) -> int:
        label, _ = self._majority()
        return label

    @property
    def vote_score(self) -> int:
        _, score = self._majority()
        return score

    @property
    def window_size(self) -> int:
        return self._cfg.vote_window_size

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _majority(self) -> tuple[int, int]:
        if not self._history:
            return 0, 0
        label, count = Counter(self._history).most_common(1)[0]
        return int(label), int(count)