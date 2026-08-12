from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
from typing import Optional

from gestureflow.config import AppConfig, DEFAULT_CONFIG, GESTURE_MAP
from gestureflow.utils import drop_oldest_put


class _Shutdown:
    """Sentinel pushed onto the volume queue to wake a blocked worker."""


_SHUTDOWN = _Shutdown()


class SystemController:
    def __init__(
        self,
        config: AppConfig | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._stop = stop_event or threading.Event()

        import pyautogui as _pag

        self._pag = _pag
        self._pag.FAILSAFE = True
        self._pag.PAUSE = 0

        self.screen_w, self.screen_h = self._pag.size()

        self._volume: int = 50
        self._vol_lock = threading.Lock()
        self._vol_queue: queue.Queue = queue.Queue(maxsize=1)

        self._vol_thread = threading.Thread(
            target=self._volume_worker, daemon=True, name="vol-worker"
        )
        self._vol_thread.start()

        self._sync_thread = threading.Thread(
            target=self._volume_sync_worker, daemon=True, name="vol-sync"
        )
        self._sync_thread.start()

        self._ploc_x: float = 0.0
        self._ploc_y: float = 0.0
        self._last_move_time: Optional[float] = None
        self._tau: float = self._cfg.mouse.smoothing_tau


    # ------------------------------------------------------------------
    # Public: volume
    # ------------------------------------------------------------------


    @property
    def volume(self) -> int:
        with self._vol_lock:
            return self._volume
        
    def set_volume(self, value: int) -> None:
        """Request a volume change.  Non-blocking — returns immediately.
 
        If a previous request is still being processed, the old value is
        discarded and replaced with the new one so we always move toward
        the most recent desired state.
        """

        value = max(0, min(100, value))
        with self._vol_lock:
            self._volume = value
        drop_oldest_put(self._vol_queue, value)


    # ------------------------------------------------------------------
    # Public: gestures
    # ------------------------------------------------------------------
 
    @property
    def gesture_map(self) -> dict:
        return GESTURE_MAP
 
    def execute_command(self, gesture_id: int) -> None:
        """Fire the hotkey associated with ``gesture_id``."""
        if gesture_id not in GESTURE_MAP:
            return
        
        entry = GESTURE_MAP[gesture_id]
        name = entry["name"]
        action_type = entry.get("type", "hotkey")

        if action_type == "hotkey":
            self._pag.hotkey(*entry["keys"])
            print(f"[controller] Executed: {name}")

        elif action_type == "osascript":
            script = entry.get("script", "")
            threading.Thread(
                target=lambda: subprocess.run(
                    ["osascript", "-e", script],
                    check=False, capture_output=True
                ),
                daemon=True,
            ).start()
            print(f"[controller] osascript: {name}")

        elif action_type == "shell":
            cmd = entry.get("cmd", [])
            threading.Thread(
                target=lambda: subprocess.run(cmd, check=False, capture_output=True),
                daemon=True,
            ).start()
            print(f"[controller] shell: {name}")


    # ------------------------------------------------------------------
    # Public: mouse
    # ------------------------------------------------------------------
 
    def click(self) -> None:
        self._pag.click()
 
    def right_click(self) -> None:
        self._pag.rightClick()

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------
 
    def scroll(self, delta: int) -> None:
        if delta != 0:
            self._pag.scroll(delta)

    # ------------------------------------------------------------------
    # Mouse movement
    # ------------------------------------------------------------------

    def move_mouse_smooth(
        self,
        target_x: float,
        target_y: float,
        now: float | None = None,
    ) -> None:
        """Move the cursor toward a target with frame-rate-independent smoothing.

        The old form applied a fixed 1/smooth_factor step per *frame*, so the
        cursor felt sluggish at 15 FPS and twitchy at 60 -- the same config
        produced a different feel on every machine.  This uses an exponential
        filter over elapsed *time* instead: alpha = 1 - exp(-dt / tau).  The
        cursor now covers the same fraction of the remaining distance per
        second regardless of how many frames that second contained.
        """
        now = time.monotonic() if now is None else now

        if self._last_move_time is None:
            # First move of the session: adopt the cursor's real position so we
            # filter from where the pointer actually is.  Seeding from (0, 0)
            # made the very first movement sweep in from the screen corner.
            self._ploc_x, self._ploc_y = self._current_pointer()
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_move_time)
        self._last_move_time = now

        alpha = 1.0 if self._tau <= 0.0 else 1.0 - math.exp(-dt / self._tau)
        alpha = min(1.0, max(0.0, alpha))

        cloc_x = self._ploc_x + (target_x - self._ploc_x) * alpha
        cloc_y = self._ploc_y + (target_y - self._ploc_y) * alpha
        self._pag.moveTo(cloc_x, cloc_y, _pause=False)
        self._ploc_x = cloc_x
        self._ploc_y = cloc_y

    def _current_pointer(self) -> tuple[float, float]:
        try:
            pos = self._pag.position()
            return float(pos[0]), float(pos[1])
        except Exception:
            # Headless or permission-denied: centre of the screen is a saner
            # fallback than the top-left corner.
            return self.screen_w / 2.0, self.screen_h / 2.0


    # ------------------------------------------------------------------
    # Private: background workers
    # ------------------------------------------------------------------
 
    def _volume_worker(self) -> None:
        while not self._stop.is_set():
            try:
                # Timed get rather than a blocking one, so the thread notices
                # the stop event even when no volume change is pending.
                value = self._vol_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if value is _SHUTDOWN:
                break
            try:
                cmd = ["osascript", "-e", f"set volume output volume {value}"]
                subprocess.run(cmd, check=False, capture_output=True)
            except Exception as exc:
                print(f"[controller] Volume set error: {exc}")

    def _volume_sync_worker(self) -> None:
        """Periodically re-read real system volume to stay in sync."""
        interval = self._cfg.volume.sync_interval
        while not self._stop.is_set():
            # wait() rather than sleep() so shutdown does not have to wait out
            # a full sync interval.
            if self._stop.wait(interval):
                break
            try:
                cmd = ["osascript", "-e", "output volume of (get volume settings)"]
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=2.0
                )
                if result.returncode == 0:
                    real_vol = int(result.stdout.strip())
                    with self._vol_lock:
                        self._volume = real_vol
            except Exception as exc:
                print(f"[controller] Volume sync error: {exc}")
 
    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop both volume workers and wait for them to exit.

        These used to be `while True:` daemon threads that were never joined --
        they only died because the interpreter killed them at exit, which meant
        an osascript call could be interrupted mid-flight.
        """
        self._stop.set()
        try:
            self._vol_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass
        self._vol_thread.join(timeout=timeout)
        self._sync_thread.join(timeout=timeout)

    def prime_volume(self) -> None:
        try:
            r = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                check=False, capture_output=True, text=True, timeout=2.0
            )
            if r.returncode == 0:
                with self._vol_lock:
                    self._volume = int(r.stdout.strip())
        except Exception:
            pass
