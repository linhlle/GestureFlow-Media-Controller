from __future__ import annotations

import queue
import subprocess
import threading
import time

from gestureflow.commands import Action, CommandSet, load_commands
from gestureflow.config import DEFAULT_CONFIG, AppConfig
from gestureflow.smoothing import CursorFilter
from gestureflow.utils import drop_oldest_put


class _Shutdown:
    """Sentinel pushed onto the volume queue to wake a blocked worker."""


_SHUTDOWN = _Shutdown()


class SystemController:
    def __init__(
        self,
        config: AppConfig | None = None,
        stop_event: threading.Event | None = None,
        commands: CommandSet | None = None,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._stop = stop_event or threading.Event()
        self._commands = commands if commands is not None else load_commands()

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
        self._filter = CursorFilter(
            min_cutoff=self._cfg.mouse.min_cutoff,
            beta=self._cfg.mouse.beta,
            derivative_cutoff=self._cfg.mouse.derivative_cutoff,
            max_dt=self._cfg.mouse.max_dt,
        )


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
    def commands(self) -> CommandSet:
        return self._commands

    def set_commands(self, commands: CommandSet) -> None:
        """Swap the binding set at runtime (used by hot-reload)."""
        self._commands = commands

    def execute_command(self, gesture_id: int) -> None:
        """Perform the action bound to ``gesture_id``, if any."""
        binding = self._commands.get(gesture_id)
        if binding is None:
            return
        try:
            self.perform_action(binding.action)
        except Exception as exc:
            print(f"[controller] {binding.name} failed: {exc}")
        else:
            print(f"[controller] Executed: {binding.name}")

    def perform_action(self, action: Action) -> None:
        """Dispatch one validated action.

        Every branch here handles a type that commands.parse_action already
        validated, so this function does no parsing and takes no strings from
        the config that have not been checked against a whitelist.
        """
        kind = action.type

        if kind == "hotkey":
            self._pag.hotkey(*action.keys)

        elif kind == "keypress":
            self._pag.press(action.key)

        elif kind == "media":
            self._perform_media(action.media)

        elif kind == "launch":
            # argv form, no shell: the app name cannot break out into a
            # second command however it is quoted.
            self._spawn(["open", "-a", action.app])

        elif kind == "applescript":
            self._spawn(["osascript", "-e", action.script])

        elif kind == "shell":
            # argv list, executed without shell=True. Pipes, redirects, and
            # command chaining are structurally impossible here.
            self._spawn(list(action.argv))

        else:
            raise ValueError(f"unsupported action type: {kind!r}")

    def _perform_media(self, media: str) -> None:
        if media == "mute":
            self._spawn_sync(["osascript", "-e",
                              "set volume output muted not "
                              "(output muted of (get volume settings))"])
            return
        if media in ("volumeup", "volumedown"):
            step = self._cfg.volume.step
            delta = step if media == "volumeup" else -step
            self.set_volume(max(0, min(100, self.volume + delta)))
            return
        key = {
            "playpause": "playpause",
            "next": "nexttrack",
            "previous": "prevtrack",
        }[media]
        self._pag.press(key)

    def _spawn(self, argv: list) -> None:
        """Run a subprocess off the calling thread and never wait on it."""
        threading.Thread(
            target=self._spawn_sync, args=(argv,), daemon=True
        ).start()

    @staticmethod
    def _spawn_sync(argv: list) -> None:
        try:
            subprocess.run(argv, check=False, capture_output=True, timeout=10.0)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[controller] subprocess failed ({argv[0]}): {exc}")

    # ------------------------------------------------------------------
    # Public: mouse
    # ------------------------------------------------------------------

    def click(self) -> None:
        self._pag.click()

    def right_click(self) -> None:
        self._pag.rightClick()

    def mouse_down(self, button: str = "left") -> None:
        self._pag.mouseDown(button=button)

    def mouse_up(self, button: str = "left") -> None:
        self._pag.mouseUp(button=button)

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
        """Move the cursor toward a target, filtered for hand jitter.

        `now` should be the timestamp of the frame the target came from, not
        the moment of dispatch. The filter needs the interval over which the
        hand actually moved; using dispatch time instead makes dt depend on how
        long the action thread took to get here, which is unrelated to the
        motion being filtered.
        """
        now = time.monotonic() if now is None else now

        if self._filter.is_stale(now):
            # Tracking was lost for longer than any real frame interval.
            # Filtering the new target against the pre-gap position would take
            # one huge step; starting fresh glides there over the next few
            # frames instead.
            self._filter.reset()

        if not self._filter.initialized:
            # Seed at the pointer's real position so the first movement of a
            # session does not sweep in from wherever the filter started.
            px, py = self._current_pointer()
            self._filter.reset(px, py)

        x, y = self._filter(target_x, target_y, now)
        self._pag.moveTo(x, y, _pause=False)
        self._ploc_x, self._ploc_y = x, y

    def release_cursor(self) -> None:
        """Called when cursor mode disengages.

        Without this the filter would carry a stale position and a stale
        timestamp across the gap, so the first frame after the user stopped
        pointing would be filtered against wherever the pointer was seconds
        ago.
        """
        self._filter.reset()

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
