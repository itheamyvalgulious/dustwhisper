from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from oracle_game.gpu import moderngl


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{value!r} is not JSON serializable")


def _controller_turn_kwargs(payload: dict[str, object]) -> dict[str, object]:
    focus_center_payload = payload.get("focus_center")
    focus_center = None if focus_center_payload is None else (
        int(focus_center_payload[0]),  # type: ignore[index]
        int(focus_center_payload[1]),  # type: ignore[index]
    )
    return {
        "controller_state": payload.get("controller_state"),
        "controller_state_provided": "controller_state" in payload,
        "focus_center": focus_center,
        "entities": payload.get("entities"),
        "entity_placeholders": payload.get("entity_placeholders"),
        "patches": payload.get("patches"),
        "observation_specs": payload.get("observation_specs"),
        "force_sources": payload.get("force_sources"),
        "emitters": payload.get("emitters"),
        "target_queries": payload.get("target_queries"),
        "change_intents": payload.get("change_intents"),
        "carrier_intents": payload.get("carrier_intents"),
        "observation_targets": payload.get("observation_targets"),
        "readback_requests": payload.get("readback_requests"),
        "commands": payload.get("commands"),
    }


def _target_query_kwargs(payload: dict[str, object]) -> dict[str, object]:
    return {
        "target_query_id": None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
        "target_dx": int(payload.get("target_dx", 0)),
        "target_dy": int(payload.get("target_dy", 0)),
        "target_queries": payload.get("target_queries"),
    }


@dataclass(slots=True)
class EngineRunState:
    paused: bool = False
    speed: float = 1.0
    single_step: bool = False


from oracle_game.http_console_controller import (
    route_controller_get,
    route_controller_post,
)
from oracle_game.http_console_entity import (
    route_entity_get,
    route_entity_post,
)
from oracle_game.http_console_frame import (
    route_frame_get,
    route_frame_post,
)
from oracle_game.http_console_inject import route_inject_post
from oracle_game.http_console_paging import (
    route_paging_get,
    route_paging_post,
)
from oracle_game.http_console_readback import (
    route_readback_get,
    route_readback_post,
)
from oracle_game.http_console_tables import (
    route_tables_get,
    route_tables_post,
)


class EngineHTTPConsole:
    def __init__(
        self,
        engine: "WorldEngine",
        state: EngineRunState,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        own_gpu_context: bool = True,
    ) -> None:
        self.engine = engine
        self.state = state
        self.host = host
        self.own_gpu_context = bool(own_gpu_context)
        self._server = HTTPServer((host, port), self._make_handler())
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._startup_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=5.0):
            raise RuntimeError("HTTP console failed to initialize GPU context")
        if self._startup_error is not None:
            raise RuntimeError("HTTP console failed to initialize GPU context") from self._startup_error

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _serve(self) -> None:
        http_ctx = None
        try:
            if getattr(self.engine, "simulation_backend", "") == "gpu" and self.own_gpu_context:
                if moderngl is None:
                    raise RuntimeError("GPU HTTP console requires ModernGL; CPU fallback is disabled")
                errors: list[Exception] = []
                for kwargs in ({"require": 430, "backend": "egl"}, {"require": 430}):
                    try:
                        http_ctx = moderngl.create_standalone_context(**kwargs)
                        break
                    except Exception as exc:
                        errors.append(exc)
                if http_ctx is None:
                    raise RuntimeError("GPU HTTP console failed to initialize a ModernGL context") from (errors[-1] if errors else None)
                with self.engine.state_lock:
                    self.engine.bridge.attach_context(http_ctx)
                    self.engine.bridge.sync_world(self.engine, force_cpu_resource_upload=True)
                    self.engine.bridge.mark_gpu_authoritative(
                        "cell_core",
                        "material",
                        "island_id",
                        "entity_id",
                        "placeholder_displaced_material",
                        "collapse_delay_pending",
                        "gas_concentration",
                        "ambient_temperature",
                        "flow_velocity",
                        "pressure_ping",
                        "visible_illumination",
                        "cell_optical_dose",
                        "gas_optical_dose",
                        "active_meta",
                        "active_tile_ttl",
                        "active_chunk_mask",
                    )
                    self.engine._gpu_cpu_dirty_resources.clear()
        except BaseException as exc:
            self._startup_error = exc
            self._ready_event.set()
            return
        self._ready_event.set()
        try:
            self._server.serve_forever()
        finally:
            if http_ctx is not None:
                with self.engine.state_lock:
                    self.engine.bridge.release_resources()
                    try:
                        http_ctx.release()
                    except Exception:
                        pass
                    self.engine.bridge.ctx = None
                    self.engine.bridge.enabled = False
                    self.engine.bridge.owner_thread_id = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        engine = self.engine
        state = self.state
        console = self

        class Handler(BaseHTTPRequestHandler):
            @staticmethod
            def _payload_immediate(payload: dict[str, object]) -> bool:
                return bool(payload.get("immediate", False))

            def do_GET(self) -> None:  # noqa: N802
                with engine.state_lock:
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    if console.route_readback_get(self, parsed, query):
                        return
                    if console.route_frame_get(self, parsed, query):
                        return
                    if console.route_controller_get(self, parsed, query):
                        return
                    if console.route_entity_get(self, parsed, query):
                        return
                    if console.route_tables_get(self, parsed, query):
                        return
                    if console.route_paging_get(self, parsed, query):
                        return
                    if console.route_control_get(self, parsed, query):
                        return
                    self._send({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                with engine.state_lock:
                    parsed = urlparse(self.path)
                    payload = self._read_json()
                    try:
                        if console.route_controller_post(self, parsed, payload):
                            return
                        if console.route_inject_post(self, parsed, payload):
                            return
                        if console.route_readback_post(self, parsed, payload):
                            return
                        if console.route_frame_post(self, parsed, payload):
                            return
                        if console.route_entity_post(self, parsed, payload):
                            return
                        if console.route_paging_post(self, parsed, payload):
                            return
                        if console.route_tables_post(self, parsed, payload):
                            return
                        if console.route_control_post(self, parsed, payload):
                            return
                        self._send({"error": "not found"}, status=404)
                    except (KeyError, TypeError, ValueError) as exc:
                        self._send_bad_request(exc)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _read_json(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _send(self, payload: dict[str, object], status: int = 200) -> None:
                body = json.dumps(payload, default=_json_default).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_bad_request(self, exc: Exception) -> None:
                if isinstance(exc, KeyError) and exc.args:
                    message = str(exc.args[0])
                else:
                    message = str(exc)
                self._send({"error": message}, status=400)

            def _send_queued(self, **extra: object) -> None:
                payload: dict[str, object] = {
                    "ok": True,
                    "queued": True,
                    "pending_commands": len(engine.command_queue),
                }
                payload.update(extra)
                self._send(payload)

            def _send_mutation_result(self, *, immediate: bool, **extra: object) -> None:
                if not immediate:
                    self._send_queued(**extra)
                    return
                payload: dict[str, object] = {
                    "ok": True,
                    "queued": False,
                    "pending_commands": len(engine.command_queue),
                }
                payload.update(extra)
                self._send(payload)

        return Handler

    # Shared payload helpers exposed for bucket modules (avoid circular imports).
    _controller_turn_kwargs = staticmethod(_controller_turn_kwargs)
    _target_query_kwargs = staticmethod(_target_query_kwargs)

    # Readback / observation endpoints.
    route_readback_get = route_readback_get
    route_readback_post = route_readback_post

    # Controller-turn / targets / commands / intents endpoints.
    route_controller_get = route_controller_get
    route_controller_post = route_controller_post

    # Table mutation endpoints.
    route_tables_get = route_tables_get
    route_tables_post = route_tables_post

    # Frame I/O endpoints.
    route_frame_get = route_frame_get
    route_frame_post = route_frame_post

    # Paging / page-store endpoints.
    route_paging_get = route_paging_get
    route_paging_post = route_paging_post

    # Entity (non-controller) endpoints.
    route_entity_get = route_entity_get
    route_entity_post = route_entity_post

    # Material / inject / force / emitter mutation endpoints.
    route_inject_post = route_inject_post

    def route_control_get(
        self,
        handler: BaseHTTPRequestHandler,
        parsed: object,
        query: dict[str, list[str]],
    ) -> bool:
        if parsed.path == "/api/control/state":  # type: ignore[attr-defined]
            handler._send(
                {
                    "paused": bool(self.state.paused),
                    "speed": float(self.state.speed),
                    "single_step": bool(self.state.single_step),
                }
            )
            return True
        return False

    def route_control_post(
        self,
        handler: BaseHTTPRequestHandler,
        parsed: object,
        payload: dict[str, object],
    ) -> bool:
        engine = self.engine
        state = self.state
        path = parsed.path  # type: ignore[attr-defined]
        if path == "/api/control/pause":
            state.paused = True
            handler._send({"paused": True})
            return True
        if path == "/api/control/resume":
            state.paused = False
            handler._send({"paused": False})
            return True
        if path == "/api/control/step":
            state.single_step = True
            handler._send({"single_step": True})
            return True
        if path == "/api/control/tick":
            engine.step()
            handler._send(
                {
                    "ok": True,
                    "frame_id": int(engine.frame_id),
                    "pending_commands": len(engine.command_queue),
                }
            )
            return True
        if path == "/api/control/speed":
            state.speed = float(payload["speed"])
            handler._send({"speed": state.speed})
            return True
        if path == "/api/control/reset":
            engine.reset_world(immediate=False)
            handler._send_queued()
            return True
        return False
