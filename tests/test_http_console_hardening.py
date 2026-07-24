"""Hardening tests for the HTTP console: body limits, JSON errors, parameter bounds.

Starts a real ``EngineHTTPConsole`` on port=0 like the console E2E tests in
``test_engine_core.py``, but drives it with raw sockets / ``http.client`` so
malformed requests (bad Content-Length, broken JSON) can be constructed.
"""

from __future__ import annotations

import http.client
import json
import socket

from oracle_game.http_console import MAX_BODY_BYTES, EngineHTTPConsole, EngineRunState
from oracle_game.world import WorldEngine


def _post_json(port: int, path: str, body: bytes) -> tuple[int, dict[str, object]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read()
        return response.status, json.loads(payload.decode("utf-8"))
    finally:
        conn.close()


def _raw_post(port: int, request: bytes) -> tuple[int, dict[str, object]]:
    """Send a hand-crafted HTTP request; the console is HTTP/1.0 so it closes
    the connection after the response, terminating the read loop."""
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
        sock.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(body.decode("utf-8"))


def test_http_console_rejects_oversized_body_before_reading() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        # Declare a body larger than MAX_BODY_BYTES but never send it: the 413
        # must come back without the console reading the body into memory.
        request = (
            b"POST /api/control/pause HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(MAX_BODY_BYTES + 1).encode("ascii") + b"\r\n"
            b"\r\n"
        )
        status, body = _raw_post(console.port, request)
        assert status == 413
        assert "error" in body

        # The console is unaffected and keeps serving normal requests.
        ok_status, ok_body = _post_json(console.port, "/api/control/pause", b"{}")
        assert ok_status == 200
        assert ok_body == {"paused": True}
    finally:
        console.stop()


def test_http_console_accepts_body_at_max_size_limit() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        # A body of exactly MAX_BODY_BYTES is still accepted (limit is exclusive).
        prefix = '{"speed": 1.5, "pad": "'
        suffix = '"}'
        pad_len = MAX_BODY_BYTES - len(prefix) - len(suffix)
        body = (prefix + "x" * pad_len + suffix).encode("utf-8")
        assert len(body) == MAX_BODY_BYTES
        status, response_body = _post_json(console.port, "/api/control/speed", body)
        assert status == 200
        assert response_body["speed"] == 1.5
    finally:
        console.stop()


def test_http_console_requires_content_length() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        request = b"POST /api/control/pause HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        status, body = _raw_post(console.port, request)
        assert status == 411
        assert "error" in body
    finally:
        console.stop()


def test_http_console_rejects_invalid_content_length() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        for declared in (b"not-a-number", b"-5"):
            request = (
                b"POST /api/control/pause HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Length: " + declared + b"\r\n"
                b"\r\n"
            )
            status, body = _raw_post(console.port, request)
            assert status == 400, declared
            assert "error" in body
    finally:
        console.stop()


def test_http_console_malformed_json_returns_400() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        for broken in (
            b'{"speed": 1.0,',  # truncated JSON
            b'{"speed": "\xff\xfe"}',  # invalid UTF-8
            b"[1, 2, 3]",  # valid JSON but not an object
        ):
            status, body = _post_json(console.port, "/api/control/speed", broken)
            assert status == 400, broken
            assert "error" in body
        # Engine state untouched by the rejected requests.
        state_status, state_body = _post_json(console.port, "/api/control/speed", b'{"speed": 2.0}')
        assert state_status == 200
        assert state_body == {"speed": 2.0}
    finally:
        console.stop()


def test_http_console_speed_rejects_non_finite_and_negative_values() -> None:
    engine = WorldEngine(width=32, height=24)
    state = EngineRunState()
    console = EngineHTTPConsole(engine, state, port=0)
    console.start()
    try:
        for bad in ("NaN", "Infinity", "-Infinity", "-1.5"):
            status, body = _post_json(
                console.port, "/api/control/speed", f'{{"speed": {bad}}}'.encode("utf-8")
            )
            assert status == 400, bad
            assert "error" in body
            assert float(state.speed) == 1.0
        for good in ("0", "2.5"):
            status, body = _post_json(
                console.port, "/api/control/speed", f'{{"speed": {good}}}'.encode("utf-8")
            )
            assert status == 200, good
            assert body == {"speed": float(good)}
        assert float(state.speed) == 2.5
    finally:
        console.stop()


def test_http_console_rejects_out_of_bounds_geometry() -> None:
    engine = WorldEngine(width=32, height=24)  # geometry limit = max(32, 24) = 32
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        cases = [
            ("/api/material/write", {"x": 4, "y": 4, "material": "sand_powder", "radius": 33}),
            ("/api/material/write", {"x": 4, "y": 4, "material": "sand_powder", "radius": -1}),
            (
                "/api/material/fill",
                {"x": 0, "y": 0, "width": 33, "height": 4, "material": "sand_powder"},
            ),
            (
                "/api/material/fill",
                {"x": 0, "y": 0, "width": 4, "height": -3, "material": "sand_powder"},
            ),
            ("/api/inject/temperature", {"x": 4, "y": 4, "delta": 5.0, "radius": 1000}),
            ("/api/inject/velocity", {"x": 4, "y": 4, "velocity": [1.0, 0.0], "radius": 1000}),
            (
                "/api/inject/gas",
                {"x": 4, "y": 4, "species": "water_gas", "amount": 1.0, "radius": 33},
            ),
            ("/api/inject/force", {"x": 4, "y": 4, "radius": 1e9}),
            ("/api/inject/force", {"x": 4, "y": 4, "radius": -2.0}),
            ("/api/inject/force", {"x": 4, "y": 4, "radius": "NaN"}),
            ("/api/force_sources/set", {"force_sources": [{"x": 4, "y": 4, "radius": 100.0}]}),
            (
                "/api/emitters/set",
                {"emitters": [{"x": 4, "y": 4, "light_type": "visible_light", "radius": 100}]},
            ),
            ("/api/inject/light", {"x": 4, "y": 4, "light_type": "visible_light", "radius": 100}),
            (
                "/api/entity/placeholders",
                {"placeholders": [{"entity_id": 1, "x": 0, "y": 0, "width": 10**9, "height": 1}]},
            ),
            (
                "/api/entity/states/set",
                {"entities": [{"entity_id": 1, "x": 0, "y": 0, "width": 1, "height": 10**9}]},
            ),
            ("/api/entity/states/patch", {"patches": [{"entity_id": 1, "fields": {"width": 33}}]}),
            (
                "/api/frame/preview",
                {
                    "entity_placeholders": [
                        {"entity_id": 1, "x": 0, "y": 0, "width": 10**9, "height": 10**9}
                    ]
                },
            ),
            (
                "/api/frame/submit",
                {"entities": [{"entity_id": 1, "x": 0, "y": 0, "width": 10**9, "height": 1}]},
            ),
        ]
        for path, payload in cases:
            status, body = _post_json(console.port, path, json.dumps(payload).encode("utf-8"))
            assert status == 400, (path, status, body)
            assert "error" in body
    finally:
        console.stop()


def test_http_console_accepts_in_bounds_geometry() -> None:
    engine = WorldEngine(width=32, height=24)
    console = EngineHTTPConsole(engine, EngineRunState(), port=0)
    console.start()
    try:
        cases = [
            # Bounds are inclusive; coordinates may be negative (paging world).
            ("/api/material/write", {"x": -4, "y": 4, "material": "sand_powder", "radius": 32}),
            (
                "/api/material/fill",
                {"x": 0, "y": 0, "width": 32, "height": 24, "material": "sand_powder"},
            ),
            ("/api/inject/force", {"x": 4, "y": 4, "radius": 32.0}),
            (
                "/api/entity/placeholders",
                {"placeholders": [{"entity_id": 7, "x": 1, "y": 1, "width": 4, "height": 4}]},
            ),
        ]
        for path, payload in cases:
            status, body = _post_json(console.port, path, json.dumps(payload).encode("utf-8"))
            assert status == 200, (path, status, body)
            assert body["ok"] is True
    finally:
        console.stop()
