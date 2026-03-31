from __future__ import annotations

import subprocess
import threading
import queue
import time
from typing import Optional

from gestureflow.config import AppConfig, DEFAULT_CONFIG, GESTURE_MAP


class SystemController:
    def __init__(self, config: AppConfig | None = None) -> None:
        self._cfg = config or DEFAULT_CONFIG
        
        import pyautogui as _pag

        self._pag = _pag
        self._pag.FAILSAFE = True
        self._pag.PAUSE = 0

        self.screen_w, self.screen_h = self._pag.size()

        self._volume: int = 50
        self._vol_lock = threading.Lock()
        self._vol_queue: queue.Queue[int] = queue.Queue(maxsize=1)

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
        try:
            self._vol_queue.get_nowait()
        except queue.Empty:
            pass
        self._vol_queue.put_nowait(value)


    # ------------------------------------------------------------------
    # Public: gestures
    # ------------------------------------------------------------------
 
    @property
    def gesture_map(self) -> dict:
        return GESTURE_MAP
 
    def execute_gesture(self, gesture_id: int) -> None:
        """Fire the hotkey associated with ``gesture_id``."""
        if gesture_id not in GESTURE_MAP:
            return
        keys = GESTURE_MAP[gesture_id]["keys"]
        name = GESTURE_MAP[gesture_id]["name"]
        self._pag.hotkey(*keys)
        print(f"[controller] Executed: {name}")

    # ------------------------------------------------------------------
    # Public: mouse
    # ------------------------------------------------------------------
 
    def move_mouse_smooth(
        self,
        target_x: float,
        target_y: float,
    ) -> None:
        smooth = self._cfg.mouse.smooth_factor
        cloc_x = self._ploc_x + (target_x - self._ploc_x) / smooth
        cloc_y = self._ploc_y + (target_y - self._ploc_y) / smooth
        self._pag.moveTo(cloc_x, cloc_y, _pause=False)
        self._ploc_x = cloc_x
        self._ploc_y = cloc_y


    # ------------------------------------------------------------------
    # Private: background workers
    # ------------------------------------------------------------------
 
    def _volume_worker(self) -> None:
        while True:
            value = self._vol_queue.get()   # blocks until a request arrives
            try:
                cmd = ["osascript", "-e", f"set volume output volume {value}"]
                subprocess.run(cmd, check=False, capture_output=True)
            except Exception as exc:
                print(f"[controller] Volume set error: {exc}")
 
    def _volume_sync_worker(self) -> None:
        """Periodically re-read real system volume to stay in sync."""
        interval = self._cfg.volume.sync_interval
        while True:
            time.sleep(interval)
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
 
    def _get_volume_once(self) -> Optional[int]:
        """One-shot blocking volume read.  Only used during startup."""
        try:
            cmd = ["osascript", "-e", "output volume of (get volume settings)"]
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=2.0
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return None
 
    def prime_volume(self) -> None:
        """Synchronously read the real volume once at startup.
 
        Call this before the main loop starts so the UI shows the correct
        initial value rather than the 50 default.
        """
        val = self._get_volume_once()
        if val is not None:
            with self._vol_lock:
                self._volume = val
 

