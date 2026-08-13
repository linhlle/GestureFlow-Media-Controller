"""Action dispatch on a dedicated thread.

Why this exists
---------------
Every OS call -- `pyautogui.moveTo`, `click`, `hotkey`, `scroll`, and the
`osascript` subprocess spawns -- used to run inline in the render loop.  A
hotkey that takes 40 ms to deliver stalled the HUD for 40 ms and, worse, stopped
the loop draining the inference queue, so the backlog grew while the pipeline
was already behind.  Dispatch now happens on its own thread.

Drop semantics are per action type, and that distinction is the whole point of
this module.  A queue that treats every item the same is wrong here:

* **Cursor moves coalesce.** Only the newest position matters.  If three moves
  are pending, delivering all three walks the cursor through two stale
  positions on the way to the right one, which reads as lag.  Superseding is
  strictly better than delivering.

* **Clicks, hotkeys, scrolls, and volume changes are never dropped.** They are
  discrete events a user consciously performed.  Silently swallowing a click
  under load is a correctness bug, not a performance trade-off.  These queue up
  and are delivered in order, even if slightly late.

So the queue is really two: an unbounded-ish FIFO of discrete events, and a
single-slot register holding the latest cursor position.  Discrete events win
when both are pending, because a dropped-feeling click is worse than a
few-milliseconds-later cursor.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gestureflow.metrics import (
    HANDOFF_ACTION,
    STAGE_DISPATCH,
    MetricsRecorder,
    NullMetrics,
)

# --------------------------------------------------------------------------
# Action types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveCursor:
    """Absolute cursor target. Coalesces: only the newest one is delivered."""
    x: float
    y: float
    captured_at: float = 0.0


@dataclass(frozen=True)
class ReleaseCursor:
    """Cursor mode disengaged; the smoothing filter should forget its state.

    Discrete rather than coalescing: if this were dropped the next pointing
    gesture would be filtered against a stale position from before the gap.
    """
    captured_at: float = 0.0


@dataclass(frozen=True)
class Click:
    button: str = "left"          # "left" | "right"
    captured_at: float = 0.0


@dataclass(frozen=True)
class Scroll:
    delta: int = 0
    captured_at: float = 0.0


@dataclass(frozen=True)
class SetVolume:
    value: int = 0
    captured_at: float = 0.0


@dataclass(frozen=True)
class Command:
    """A bound gesture command, resolved by the controller."""
    gesture_id: int = 0
    captured_at: float = 0.0


# Discrete events are never dropped; cursor moves are.
DISCRETE_TYPES = (Click, Scroll, SetVolume, Command, ReleaseCursor)


class ActionDispatcher(threading.Thread):
    """Consumes actions off the render thread and performs them.

    Parameters
    ----------
    controller:
        Anything exposing ``move_mouse_smooth``, ``click``, ``right_click``,
        ``scroll``, ``set_volume`` and ``execute_command``.  Kept structural
        rather than a hard type so tests can pass a recording double.
    max_pending:
        Cap on queued discrete events.  Reaching it means the OS layer is
        badly wedged; past that point events *are* dropped and counted, because
        an unbounded queue would trade a visible stall for an invisible leak.
    """

    def __init__(
        self,
        controller: Any,
        stop_event: Optional[threading.Event] = None,
        metrics: Optional[MetricsRecorder] = None,
        max_pending: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="action-thread", daemon=True)
        self._ctrl = controller
        self._stop_event = stop_event or threading.Event()
        self._metrics = metrics or NullMetrics()
        self._clock = clock

        self._discrete: queue.Queue = queue.Queue(maxsize=max_pending)
        self._pending_move: Optional[MoveCursor] = None
        self._move_lock = threading.Lock()
        # Signalled whenever either the queue or the move register gains work,
        # so the thread can block instead of spinning.
        self._work = threading.Event()

        self.coalesced_moves = 0
        self.dropped_discrete = 0
        self.performed = 0

    # -- producer side (called from the render thread) ---------------------

    def submit(self, action: Any) -> bool:
        """Queue an action. Returns False only if a discrete event was dropped."""
        if isinstance(action, MoveCursor):
            with self._move_lock:
                if self._pending_move is not None:
                    # Superseding an undelivered move is the desired outcome,
                    # not a failure -- count it separately from a real drop.
                    self.coalesced_moves += 1
                    self._metrics.count("action.moves_coalesced")
                self._pending_move = action
            self._work.set()
            return True

        try:
            self._discrete.put_nowait(action)
        except queue.Full:
            self.dropped_discrete += 1
            self._metrics.count("action.discrete_dropped")
            self._work.set()
            return False

        self._metrics.observe_queue(HANDOFF_ACTION, self._discrete.qsize())
        self._work.set()
        return True

    # -- consumer side -----------------------------------------------------

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self._work.wait(timeout=0.1):
                continue
            self._work.clear()
            self._drain()
        self._drain()   # deliver anything already accepted before shutting down

    def _drain(self) -> None:
        # Discrete events first: a click the user made must not sit behind a
        # cursor move that is about to be superseded anyway.
        while True:
            try:
                action = self._discrete.get_nowait()
            except queue.Empty:
                break
            self._perform(action)

        with self._move_lock:
            move, self._pending_move = self._pending_move, None
        if move is not None:
            self._perform(move)

    def _perform(self, action: Any) -> None:
        with self._metrics.timer(STAGE_DISPATCH):
            try:
                self._dispatch(action)
            except Exception as exc:
                # One failing hotkey must not take the dispatcher down with it.
                print(f"[action] {type(action).__name__} failed: {exc}")
                self._metrics.count("action.errors")
        self.performed += 1
        if getattr(action, "captured_at", 0.0):
            self._metrics.record_end_to_end(action.captured_at)

    def _dispatch(self, action: Any) -> None:
        if isinstance(action, MoveCursor):
            # Pass the capture timestamp through: the filter must measure the
            # interval the hand moved over, not however long this thread took
            # to reach the action.
            self._ctrl.move_mouse_smooth(action.x, action.y,
                                         now=action.captured_at or None)
        elif isinstance(action, ReleaseCursor):
            self._ctrl.release_cursor()
        elif isinstance(action, Click):
            if action.button == "right":
                self._ctrl.right_click()
            else:
                self._ctrl.click()
        elif isinstance(action, Scroll):
            self._ctrl.scroll(action.delta)
        elif isinstance(action, SetVolume):
            self._ctrl.set_volume(action.value)
        elif isinstance(action, Command):
            self._ctrl.execute_command(action.gesture_id)
        else:
            raise TypeError(f"unknown action type: {type(action).__name__}")

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()
        self._work.set()

    def flush(self, timeout: float = 1.0) -> bool:
        """Block until the queue is empty. Test and shutdown helper."""
        deadline = self._clock() + timeout
        while self._clock() < deadline:
            with self._move_lock:
                move_pending = self._pending_move is not None
            if self._discrete.empty() and not move_pending:
                return True
            self._work.set()
            time.sleep(0.005)
        return False

    @property
    def pending(self) -> int:
        with self._move_lock:
            move = 1 if self._pending_move is not None else 0
        return self._discrete.qsize() + move
