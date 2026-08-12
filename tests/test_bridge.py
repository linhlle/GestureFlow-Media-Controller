"""Tests for the localhost bridge.

The bridge writes config files and exposes gesture state on a port, so its
security posture is the thing most worth testing: loopback only, origin
checked, no path traversal, and pushed configs validated exactly as they would
be from disk.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gestureflow.bridge import (
    WEB_ROOT,
    BridgeServer,
    GestureState,
    _encode_frame,
    _read_frame,
)


@pytest.fixture
def server(tmp_path):
    srv = BridgeServer(port=_free_port(), config_target=tmp_path / "commands.yaml")
    srv.start()
    time.sleep(0.1)
    yield srv
    srv.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(server, path, headers=None):
    req = urllib.request.Request(f"{server.url.rstrip('/')}{path}",
                                 headers=headers or {})
    return urllib.request.urlopen(req, timeout=3)


# ---------------------------------------------------------------------------
# Binding and reachability
# ---------------------------------------------------------------------------

class TestBinding:
    def test_refuses_to_bind_a_non_loopback_host(self):
        # Exposing gesture state and config writes to the LAN should not be one
        # flag away.
        with pytest.raises(ValueError, match="loopback"):
            BridgeServer(host="0.0.0.0")

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
    def test_accepts_loopback_hosts(self, host):
        BridgeServer(host=host)

    def test_serves_the_landing_page(self, server):
        response = _get(server, "/")
        assert response.status == 200
        assert b"GestureFlow" in response.read()

    def test_serves_javascript_with_a_module_friendly_type(self, server):
        # A wrong Content-Type makes the browser refuse an ES module import.
        response = _get(server, "/js/recognizer.js")
        assert response.status == 200
        assert "text/javascript" in response.headers["Content-Type"]

    def test_serves_the_exported_model(self, server):
        if not (WEB_ROOT / "models" / "forest.json").exists():
            pytest.skip("forest.json not exported")
        payload = json.loads(_get(server, "/models/forest.json").read())
        assert payload["schema"] == "gestureflow.forest/1"

    def test_state_endpoint_returns_json(self, server):
        state = json.loads(_get(server, "/api/state").read())
        assert "mode" in state and "hand_present" in state

    def test_unknown_path_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(server, "/definitely-not-here")
        assert exc.value.code == 404


class TestPathTraversal:
    @pytest.mark.parametrize("path", [
        "/../pyproject.toml",
        "/../../etc/passwd",
        "/js/../../gestureflow/controller.py",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/....//....//etc/passwd",
    ])
    def test_cannot_escape_the_web_root(self, server, path):
        try:
            response = _get(server, path)
        except urllib.error.HTTPError as exc:
            assert exc.code in (403, 404)
            return
        # A 200 is only acceptable if the server normalized the path back to
        # something inside web/ -- never if it leaked a repo file.
        body = response.read()
        assert b"[project]" not in body
        assert b"root:" not in body
        assert b"class SystemController" not in body


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class WSClient:
    """A minimal RFC 6455 client, enough to exercise the server."""

    def __init__(self, host: str, port: int, origin: str = None) -> None:
        self.sock = socket.create_connection((host, port), timeout=3)
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [
            "GET /ws HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if origin:
            lines.append(f"Origin: {origin}")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        self.response = self._read_headers()

    def _read_headers(self) -> bytes:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            buf += chunk
        return buf

    @property
    def upgraded(self) -> bool:
        return b"101" in self.response.split(b"\r\n")[0]

    def send(self, obj) -> None:
        payload = json.dumps(obj).encode()
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        else:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def recv(self, timeout: float = 3.0):
        self.sock.settimeout(timeout)
        frame = _read_frame(self.sock)
        if frame is None:
            return None
        return json.loads(frame[1].decode())

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class TestWebSocketHandshake:
    def test_same_origin_is_accepted(self, server):
        client = WSClient("127.0.0.1", server.port,
                          origin=f"http://127.0.0.1:{server.port}")
        try:
            assert client.upgraded
        finally:
            client.close()

    def test_foreign_origin_is_rejected(self, server):
        # Any page in the browser can try to reach loopback; only pages this
        # server itself served may connect.
        client = WSClient("127.0.0.1", server.port,
                          origin="https://evil.example.com")
        try:
            assert not client.upgraded
            assert b"403" in client.response
        finally:
            client.close()

    def test_missing_origin_is_allowed_for_non_browser_clients(self, server):
        client = WSClient("127.0.0.1", server.port)
        try:
            assert client.upgraded
        finally:
            client.close()


class TestWebSocketState:
    def test_state_is_sent_on_connect(self, server):
        client = WSClient("127.0.0.1", server.port,
                          origin=f"http://127.0.0.1:{server.port}")
        try:
            message = client.recv()
            assert message["type"] == "state"
            assert "mode" in message["state"]
        finally:
            client.close()

    def test_published_updates_are_broadcast(self, server):
        client = WSClient("127.0.0.1", server.port,
                          origin=f"http://127.0.0.1:{server.port}")
        try:
            client.recv()                       # initial snapshot
            server.publish(mode="scroll", stable_gesture=0, hand_present=True)
            message = client.recv()
            assert message["state"]["mode"] == "scroll"
            assert message["state"]["hand_present"] is True
        finally:
            client.close()


class TestConfigPush:
    VALID = {
        "version": 1,
        "neutral_label": 0,
        "gestures": [
            {"label": 1, "name": "Spotlight",
             "action": {"type": "hotkey", "keys": ["command", "space"]}},
        ],
    }

    def _client(self, server):
        client = WSClient("127.0.0.1", server.port,
                          origin=f"http://127.0.0.1:{server.port}")
        client.recv()                            # drain the initial state
        return client

    def test_valid_config_is_written(self, server, tmp_path):
        client = self._client(server)
        try:
            client.send({"type": "set_config", "config": self.VALID})
            reply = client.recv()
            assert reply["ok"] is True
            assert reply["bindings"] == 1
        finally:
            client.close()

        written = Path(server.config_target)
        assert written.exists()
        assert "Spotlight" in written.read_text()

    def test_invalid_config_is_rejected_and_nothing_is_written(self, server):
        client = self._client(server)
        try:
            client.send({"type": "set_config", "config": {
                "version": 1, "neutral_label": 0,
                "gestures": [{"label": 1, "name": "Bad",
                              "action": {"type": "eval", "code": "whatever"}}],
            }})
            reply = client.recv()
            assert reply["ok"] is False
            assert "unknown action type" in reply["error"]
        finally:
            client.close()
        assert not Path(server.config_target).exists()

    def test_config_with_an_injection_shaped_app_name_is_rejected(self, server):
        client = self._client(server)
        try:
            client.send({"type": "set_config", "config": {
                "version": 1, "neutral_label": 0,
                "gestures": [{"label": 1, "name": "Bad", "action": {
                    "type": "launch", "app": 'Notes"; rm -rf ~; echo "'}}],
            }})
            reply = client.recv()
            assert reply["ok"] is False
        finally:
            client.close()
        assert not Path(server.config_target).exists()

    def test_written_config_reloads_cleanly(self, server):
        from gestureflow.commands import load_commands

        client = self._client(server)
        try:
            client.send({"type": "set_config", "config": self.VALID})
            assert client.recv()["ok"] is True
        finally:
            client.close()

        # The round trip must survive: what the bridge writes, the app reads.
        commands = load_commands(Path(server.config_target))
        assert commands.get(1).action.keys == ("command", "space")

    def test_unknown_message_types_are_ignored(self, server):
        client = self._client(server)
        try:
            client.send({"type": "please_do_something_else"})
            server.publish(mode="cursor")
            assert client.recv()["type"] == "state"
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Frame codec
# ---------------------------------------------------------------------------

class TestFrameCodec:
    @pytest.mark.parametrize("size", [0, 5, 125, 126, 300, 70000])
    def test_encodes_each_length_class(self, size):
        text = "x" * size
        frame = _encode_frame(text)
        assert frame[0] == 0x81
        # Server frames must never set the mask bit.
        assert not frame[1] & 0x80

    def test_length_header_widens_correctly(self):
        assert _encode_frame("x" * 10)[1] == 10
        assert _encode_frame("x" * 200)[1] == 126
        assert _encode_frame("x" * 70000)[1] == 127


class TestGestureState:
    def test_snapshot_is_a_copy(self):
        state = GestureState()
        snap = state.snapshot()
        snap["mode"] = "mutated"
        assert state.snapshot()["mode"] != "mutated"

    def test_update_stamps_a_time(self):
        state = GestureState()
        state.update(mode="cursor")
        assert state.snapshot()["updated_at"] > 0
