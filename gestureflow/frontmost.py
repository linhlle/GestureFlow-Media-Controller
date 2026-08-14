"""Which application is in front.

Used to pick a command profile, so a swipe can mean "next slide" in Keynote and
"next desktop" everywhere else.

Two things shape the design:

**It must never block the pipeline.** Asking macOS which app is frontmost means
spawning `osascript`, which takes long enough that doing it per frame would
dominate the frame budget. So the lookup runs on its own thread on a slow poll,
and the render path only ever reads a cached string.

**It must never be the reason something breaks.** Automation permission may be
denied, `osascript` may be missing, the user may not be on macOS at all. Every
one of those degrades to "unknown app", which resolves to the default profile --
the same behaviour as having no profiles configured.

The provider is an interface with a fake implementation, so profile resolution
is fully testable without an operating system in the loop.
"""

from __future__ import annotations

import platform
import subprocess
import threading
from typing import Callable, Optional

# Asking more often than this buys nothing: a human cannot switch apps and
# perform a gesture faster than the poll, and each check spawns a process.
DEFAULT_POLL_SECONDS = 1.0

_FRONTMOST_SCRIPT = (
    'tell application "System Events" to get name of first application '
    "process whose frontmost is true"
)


class FrontmostApp:
    """Base provider. Returns None when the frontmost app is unknown."""

    def current(self) -> Optional[str]:
        return None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class StaticFrontmostApp(FrontmostApp):
    """A provider that always reports the same app. For tests and replays."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def current(self) -> Optional[str]:
        return self.name


class PollingFrontmostApp(FrontmostApp):
    """Polls on a background thread and caches the answer.

    The render path reads `current()`, which is a lock-guarded string read and
    nothing more -- the subprocess never happens on a thread anyone is waiting
    on.
    """

    def __init__(
        self,
        interval: float = DEFAULT_POLL_SECONDS,
        query: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._interval = interval
        self._query = query or query_frontmost_macos
        self._lock = threading.Lock()
        self._current: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failures = 0

    def current(self) -> Optional[str]:
        with self._lock:
            return self._current

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="frontmost-app",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            name = None
            try:
                name = self._query()
            except Exception:
                name = None

            if name is None:
                self._failures += 1
                if self._failures == 3:
                    # Say it once. Repeating every second would bury the app's
                    # own output, and the degraded behaviour is harmless.
                    print("[frontmost] Cannot read the frontmost app; "
                          "using the default profile. On macOS this usually "
                          "means Automation permission was declined.")
            else:
                self._failures = 0

            with self._lock:
                self._current = name

            if self._stop.wait(self._interval):
                return


def query_frontmost_macos(timeout: float = 2.0) -> Optional[str]:
    """Ask macOS for the frontmost application's name.

    Returns None on any failure, which the caller reads as "unknown".
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["osascript", "-e", _FRONTMOST_SCRIPT],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    name = result.stdout.strip()
    return name or None


def make_provider(enabled: bool, interval: float = DEFAULT_POLL_SECONDS) -> FrontmostApp:
    """The provider the app should use, given config."""
    if not enabled:
        return FrontmostApp()
    provider = PollingFrontmostApp(interval=interval)
    provider.start()
    return provider
