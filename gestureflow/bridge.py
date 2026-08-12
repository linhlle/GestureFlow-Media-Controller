"""Optional localhost bridge: serves the web UI and pushes live gesture state.

This is the *secondary* handoff. The primary one is downloading a config file
from the deployed site, which needs no server, no port, and no browser
compatibility caveat (see PLAN.md).

Why serving the UI ourselves solves the bridge's main problem
-------------------------------------------------------------
A page on ``https://gestureflow.vercel.app`` opening ``ws://localhost`` is
mixed content. Chrome and Firefox carve out loopback; Safari -- the default
browser on the macOS this app targets -- does not reliably, and Chrome's
Private Network Access work is tightening loopback access further. Building the
main handoff on that would be building on sand.

Serving the same UI from ``http://127.0.0.1:8765`` sidesteps it entirely: the
page origin is already loopback, so the WebSocket is same-origin and no
mixed-content rule applies anywhere.

Security posture
----------------
* Binds to loopback only. Never ``0.0.0.0``.
* Rejects WebSocket upgrades whose ``Origin`` is not this server.
* Refuses to serve files outside ``web/``.
* Pushed configs go through the same ``commands.parse_commands`` validation as
  a config read from disk, and are written to a file the user can inspect
  rather than applied invisibly.

The WebSocket implementation is stdlib-only (RFC 6455 handshake plus frame
codec) to avoid adding a dependency for an optional feature.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import socket
import struct
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from gestureflow.commands import (
    USER_COMMANDS_PATH,
    CommandConfigError,
    dump_yaml,
    parse_commands,
)
from gestureflow.utils import PROJECT_ROOT

WEB_ROOT = PROJECT_ROOT / "web"

# RFC 6455's fixed handshake GUID.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OPCODE_TEXT = 0x1
_OPCODE_CLOSE = 0x8
_OPCODE_PING = 0x9
_OPCODE_PONG = 0xA

# A pushed config is bounded well below anything legitimate.
_MAX_FRAME_BYTES = 256 * 1024


class GestureState:
    """Latest recognizer state, shared between the pipeline and the sockets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "mode": "none",
            "stable_gesture": 0,
            "vote_score": 0,
            "confidence": 0.0,
            "hand_present": False,
            "updated_at": 0.0,
        }

    def update(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)
            self._state["updated_at"] = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


class _Broadcaster:
    """Tracks connected sockets and fans state out to them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: List[socket.socket] = []

    def add(self, conn: socket.socket) -> None:
        with self._lock:
            self._clients.append(conn)

    def remove(self, conn: socket.socket) -> None:
        with self._lock:
            if conn in self._clients:
                self._clients.remove(conn)

    def broadcast(self, payload: Dict[str, Any]) -> None:
        frame = _encode_frame(json.dumps(payload))
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.sendall(frame)
            except OSError:
                self.remove(conn)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._clients)


# ---------------------------------------------------------------------------
# WebSocket frame codec
# ---------------------------------------------------------------------------

def _encode_frame(text: str) -> bytes:
    payload = text.encode("utf-8")
    header = bytearray([0x80 | _OPCODE_TEXT])
    length = len(payload)
    # Server-to-client frames are never masked.
    if length < 126:
        header.append(length)
    elif length < (1 << 16):
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


def _recv_exactly(conn: socket.socket, n: int) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = conn.recv(n - len(chunks))
        if not chunk:
            return None
        chunks += chunk
    return bytes(chunks)


def _read_frame(conn: socket.socket):
    """Return (opcode, payload) or None when the peer went away."""
    head = _recv_exactly(conn, 2)
    if head is None:
        return None

    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F

    if length == 126:
        ext = _recv_exactly(conn, 2)
        if ext is None:
            return None
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = _recv_exactly(conn, 8)
        if ext is None:
            return None
        length = struct.unpack(">Q", ext)[0]

    if length > _MAX_FRAME_BYTES:
        return None

    mask = None
    if masked:
        mask = _recv_exactly(conn, 4)
        if mask is None:
            return None

    payload = _recv_exactly(conn, length) if length else b""
    if payload is None:
        return None

    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    return opcode, payload


# ---------------------------------------------------------------------------
# HTTP + WebSocket handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "GestureFlowBridge/1.0"
    protocol_version = "HTTP/1.1"

    # Injected by serve()
    state: GestureState = None            # type: ignore[assignment]
    broadcaster: _Broadcaster = None      # type: ignore[assignment]
    allowed_origins: set = frozenset()
    config_target: Path = USER_COMMANDS_PATH
    on_config = None

    def log_message(self, fmt, *args) -> None:      # noqa: A003
        # BaseHTTPRequestHandler logs every request to stderr by default, which
        # buries the app's own output.
        return

    def handle_one_request(self) -> None:
        # A browser tab closing mid-request resets the connection, which
        # socketserver reports as an unhandled exception with a full traceback.
        # That is normal client behaviour, not an error worth printing over the
        # app's own output.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError,
                socket.timeout):
            self.close_connection = True

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:                        # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/ws":
            self._handle_websocket()
            return
        if parsed.path == "/api/state":
            self._send_json(self.state.snapshot())
            return
        self._serve_static(parsed.path)

    # -- static files ------------------------------------------------------

    def _serve_static(self, url_path: str) -> None:
        rel = unquote(url_path).lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"

        try:
            # resolve() then a containment check: the canonical defence against
            # ../ traversal and symlink escapes.
            target = (WEB_ROOT / rel).resolve()
            target.relative_to(WEB_ROOT.resolve())
        except (ValueError, OSError):
            self._send_error(403, "Forbidden")
            return

        if not target.is_file():
            # Bare paths like /guide also resolve to guide.html.
            alt = target.with_suffix(".html")
            if alt.is_file():
                target = alt
            else:
                self._send_error(404, "Not found")
                return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix in (".js", ".mjs"):
            content_type = "text/javascript"
        body = target.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8"
                         if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- websocket ---------------------------------------------------------

    def _handle_websocket(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.allowed_origins:
            # Any page in the browser can attempt a connection to loopback;
            # only pages this server itself served are accepted.
            print(f"[bridge] Rejected WebSocket from origin {origin!r}")
            self._send_error(403, "Origin not allowed")
            return

        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._send_error(400, "Missing Sec-WebSocket-Key")
            return

        accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()
        ).decode()

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        conn = self.connection
        self.broadcaster.add(conn)
        print(f"[bridge] Client connected ({self.broadcaster.count} total)")

        try:
            conn.sendall(_encode_frame(json.dumps({
                "type": "state", "state": self.state.snapshot(),
            })))
            self._websocket_loop(conn)
        except OSError:
            pass
        finally:
            self.broadcaster.remove(conn)
            print(f"[bridge] Client disconnected "
                  f"({self.broadcaster.count} remaining)")
            self.close_connection = True

    def _websocket_loop(self, conn: socket.socket) -> None:
        while True:
            frame = _read_frame(conn)
            if frame is None:
                return
            opcode, payload = frame

            if opcode == _OPCODE_CLOSE:
                return
            if opcode == _OPCODE_PING:
                conn.sendall(bytes([0x80 | _OPCODE_PONG, 0]))
                continue
            if opcode != _OPCODE_TEXT:
                continue

            try:
                message = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

            if message.get("type") == "set_config":
                self._apply_config(conn, message.get("config"))

    def _apply_config(self, conn: socket.socket, raw: Any) -> None:
        try:
            # Exactly the validation a config read from disk goes through.
            commands = parse_commands(raw)
        except CommandConfigError as exc:
            conn.sendall(_encode_frame(json.dumps({
                "type": "config_result", "ok": False, "error": str(exc),
            })))
            return

        try:
            target = Path(self.config_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(dump_yaml(commands.to_dict()))
        except OSError as exc:
            conn.sendall(_encode_frame(json.dumps({
                "type": "config_result", "ok": False,
                "error": f"could not write {self.config_target}: {exc}",
            })))
            return

        if callable(self.on_config):
            self.on_config(commands)

        print(f"[bridge] Wrote {len(commands.bindings)} binding(s) to "
              f"{self.config_target}")
        conn.sendall(_encode_frame(json.dumps({
            "type": "config_result", "ok": True,
            "path": str(self.config_target),
            "bindings": len(commands.bindings),
        })))


class _QuietHTTPServer(ThreadingHTTPServer):
    """Threading server that does not dump a traceback on a dropped client."""

    def handle_error(self, request, client_address) -> None:
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError,
                            TimeoutError, socket.timeout)):
            return
        super().handle_error(request, client_address)


class BridgeServer:
    """Serves the web UI on loopback and broadcasts gesture state."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 config_target: Optional[Path] = None) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"The bridge binds to loopback only; refusing host {host!r}. "
                f"Exposing gesture state and config writes to the network is "
                f"not something this should make easy."
            )
        self.host = host
        self.port = port
        self.state = GestureState()
        self.broadcaster = _Broadcaster()
        self.config_target = Path(config_target or USER_COMMANDS_PATH)
        self.on_config = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def origins(self) -> set:
        return {
            f"http://{self.host}:{self.port}",
            f"http://localhost:{self.port}",
            f"http://127.0.0.1:{self.port}",
        }

    def start(self) -> None:
        handler = type("_BoundHandler", (_Handler,), {
            "state": self.state,
            "broadcaster": self.broadcaster,
            "allowed_origins": self.origins,
            "config_target": self.config_target,
            "on_config": staticmethod(self.on_config) if self.on_config else None,
        })
        self._httpd = _QuietHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="bridge-http", daemon=True)
        self._thread.start()

    def publish(self, **fields) -> None:
        self.state.update(**fields)
        if self.broadcaster.count:
            self.broadcaster.broadcast({"type": "state",
                                        "state": self.state.snapshot()})

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"


def serve(host: str = "127.0.0.1", port: int = 8765,
          commands_path: Optional[Path] = None,
          open_browser: bool = True) -> int:
    """Run the bridge alongside the live pipeline."""
    from gestureflow.app import prepare
    from gestureflow.config import DEFAULT_CONFIG

    if not WEB_ROOT.is_dir():
        print(f"[bridge] ERROR: no web/ directory at {WEB_ROOT}")
        return 1

    server = BridgeServer(host=host, port=port,
                          config_target=commands_path or USER_COMMANDS_PATH)

    try:
        server.start()
    except OSError as exc:
        print(f"[bridge] Could not bind {host}:{port} — {exc}")
        print("[bridge] Another instance may already be running.")
        return 1

    print(f"[bridge] Serving the GestureFlow UI at {server.url}")
    print(f"[bridge] Config pushes are written to {server.config_target}")
    print("[bridge] Loopback only. Press Ctrl-C to stop.")

    if open_browser:
        webbrowser.open(server.url)

    try:
        pipe = prepare(DEFAULT_CONFIG, commands_path)
    except Exception as exc:
        print(f"[bridge] Pipeline unavailable ({exc})")
        print("[bridge] Serving the UI without live gesture state.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        server.stop()
        return 0

    server.on_config = pipe.controller.set_commands
    pipe.start()

    import queue as _queue
    try:
        while True:
            try:
                result = pipe.inference_q.get(timeout=0.2)
            except _queue.Empty:
                continue

            for action in pipe.router.route(result):
                pipe.dispatcher.submit(action)

            server.publish(
                mode=_mode_of(pipe.router, result),
                stable_gesture=int(result.stable_gesture),
                vote_score=int(result.vote_score),
                confidence=round(float(result.confidence), 3),
                hand_present=result.capture.landmarks is not None,
                camera=pipe.capture.status,
            )
    except KeyboardInterrupt:
        print("\n[bridge] Stopping…")
    finally:
        pipe.shutdown()
        server.stop()
    return 0


def _mode_of(router, result) -> str:
    if result.capture.landmarks is None:
        return "none"
    if result.stable_gesture != 0 or result.action is not None:
        return "command"
    if result.fsm_active:
        return "left-click"
    if result.right_fsm_active:
        return "right-click"
    if router.scroll_enabled(result):
        return "scroll"
    if router.volume_enabled(result):
        return "volume"
    if router.cursor_enabled(result):
        return "cursor"
    return "tracking"
