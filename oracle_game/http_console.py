from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from oracle_game.gpu import moderngl
from oracle_game.types import EntityPlaceholder, PageStripeUpdate


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

        class Handler(BaseHTTPRequestHandler):
            @staticmethod
            def _payload_immediate(payload: dict[str, object]) -> bool:
                return bool(payload.get("immediate", False))

            def do_GET(self) -> None:  # noqa: N802
                with engine.state_lock:
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    if parsed.path == "/api/read/cells":
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", ["16"])[0])
                        h = int(query.get("h", ["16"])[0])
                        self._send(engine.serialize_local_cells(x, y, w, h))
                        return
                    if parsed.path == "/api/read/temperature":
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", [str(engine.width)])[0])
                        h = int(query.get("h", [str(engine.height)])[0])
                        self._send(engine.serialize_temperature_window(x, y, w, h))
                        return
                    if parsed.path == "/api/read/gas":
                        species = query.get("species", ["water_gas"])[0]
                        try:
                            self._send({"species": species, "concentration": engine.serialize_gas(species)})
                        except KeyError as exc:
                            message = str(exc.args[0]) if exc.args else "invalid gas species"
                            self._send({"error": message}, status=400)
                        return
                    if parsed.path == "/api/read/pressure":
                        self._send({"pressure": engine.serialize_pressure()})
                        return
                    if parsed.path == "/api/read/gas_runtime":
                        self._send(engine.serialize_gas_runtime())
                        return
                    if parsed.path == "/api/read/heat_runtime":
                        self._send(engine.serialize_heat_runtime())
                        return
                    if parsed.path == "/api/read/liquid_runtime":
                        self._send(engine.serialize_liquid_runtime())
                        return
                    if parsed.path == "/api/read/reaction_runtime":
                        self._send(engine.serialize_reaction_runtime())
                        return
                    if parsed.path == "/api/read/collapse_runtime":
                        self._send(engine.serialize_collapse_runtime())
                        return
                    if parsed.path == "/api/read/optics_runtime":
                        self._send(engine.serialize_optics_runtime())
                        return
                    if parsed.path == "/api/read/light":
                        self._send({"illumination": engine.serialize_visible_illumination()})
                        return
                    if parsed.path == "/api/read/demo_runtime":
                        self._send(dict(getattr(engine, "demo_runtime_state", {})))
                        return
                    if parsed.path == "/api/read/debug_frame":
                        view_name = query.get("view", [engine.default_debug_view.value])[0]
                        gas_species = query.get("gas_species", [None])[0]
                        light_type = query.get("light", [None])[0]
                        try:
                            self._send(
                                engine.serialize_debug_frame(
                                    str(view_name),
                                    gas_species=None if gas_species is None else str(gas_species),
                                    light_type=None if light_type is None else str(light_type),
                                )
                            )
                        except ValueError:
                            self._send({"error": "unknown debug view", "view": str(view_name)}, status=400)
                        except KeyError as exc:
                            message = str(exc.args[0]) if exc.args else "invalid debug frame parameter"
                            self._send({"error": message}, status=400)
                        return
                    if parsed.path == "/api/read/optics":
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", [str(engine.width)])[0])
                        h = int(query.get("h", [str(engine.height)])[0])
                        light_type = query.get("light", [None])[0]
                        try:
                            self._send(engine.serialize_optics(x, y, w, h, light_type=light_type))
                        except KeyError as exc:
                            message = str(exc.args[0]) if exc.args else "invalid light type"
                            self._send({"error": message}, status=400)
                        return
                    if parsed.path == "/api/read/velocity":
                        self._send({"velocity": engine.serialize_velocity()})
                        return
                    if parsed.path == "/api/read/forces":
                        self._send({"force_sources": engine.serialize_force_sources()})
                        return
                    if parsed.path == "/api/read/emitters":
                        self._send(engine.serialize_emitters())
                        return
                    if parsed.path == "/api/read/active":
                        self._send(engine.serialize_active_runtime())
                        return
                    if parsed.path == "/api/read/motion":
                        self._send(engine.serialize_motion_runtime())
                        return
                    if parsed.path == "/api/read/bridge_runtime":
                        self._send(engine.serialize_bridge_runtime())
                        return
                    if parsed.path == "/api/read/bridge_resources":
                        self._send(engine.serialize_bridge_resources())
                        return
                    if parsed.path == "/api/read/bridge_typed_table":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing typed table name"}, status=400)
                            return
                        try:
                            self._send(engine.serialize_bridge_typed_table(str(name)))
                        except KeyError:
                            self._send({"error": "typed table not found", "name": str(name)}, status=404)
                        return
                    if parsed.path == "/api/read/bridge_typed_table_slice":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing typed table name"}, status=400)
                            return
                        offset = int(query.get("offset", ["0"])[0])
                        limit = int(query.get("limit", ["64"])[0])
                        try:
                            self._send(engine.serialize_bridge_typed_table_slice(str(name), offset=offset, limit=limit))
                        except KeyError:
                            self._send({"error": "typed table not found", "name": str(name)}, status=404)
                        return
                    if parsed.path == "/api/read/bridge_shadow_buffer":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing shadow buffer name"}, status=400)
                            return
                        try:
                            self._send(engine.serialize_bridge_shadow_buffer(str(name)))
                        except KeyError:
                            self._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
                        return
                    if parsed.path == "/api/read/bridge_shadow_buffer_slice":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing shadow buffer name"}, status=400)
                            return
                        offset = int(query.get("offset", ["0"])[0])
                        limit = int(query.get("limit", ["64"])[0])
                        try:
                            self._send(engine.serialize_bridge_shadow_buffer_slice(str(name), offset=offset, limit=limit))
                        except KeyError:
                            self._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
                        return
                    if parsed.path == "/api/read/bridge_shadow_buffer_window":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing shadow buffer name"}, status=400)
                            return
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", ["16"])[0])
                        h = int(query.get("h", ["16"])[0])
                        try:
                            self._send(engine.serialize_bridge_shadow_buffer_window(str(name), x=x, y=y, w=w, h=h))
                        except KeyError:
                            self._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
                        except ValueError as exc:
                            self._send({"error": str(exc), "name": str(name)}, status=400)
                        return
                    if parsed.path == "/api/read/bridge_shadow_buffer_world_window":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing shadow buffer name"}, status=400)
                            return
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", ["16"])[0])
                        h = int(query.get("h", ["16"])[0])
                        try:
                            self._send(engine.serialize_bridge_shadow_buffer_world_window(str(name), x=x, y=y, w=w, h=h))
                        except KeyError:
                            self._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
                        except ValueError as exc:
                            self._send({"error": str(exc), "name": str(name)}, status=400)
                        return
                    if parsed.path == "/api/read/bridge_shadow_buffer_gas_window":
                        name = query.get("name", [None])[0]
                        if name is None:
                            self._send({"error": "missing shadow buffer name"}, status=400)
                            return
                        x = int(query.get("x", ["0"])[0])
                        y = int(query.get("y", ["0"])[0])
                        w = int(query.get("w", ["4"])[0])
                        h = int(query.get("h", ["4"])[0])
                        try:
                            self._send(engine.serialize_bridge_shadow_buffer_gas_window(str(name), x=x, y=y, w=w, h=h))
                        except KeyError:
                            self._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
                        except ValueError as exc:
                            self._send({"error": str(exc), "name": str(name)}, status=400)
                        return
                    if parsed.path == "/api/read/bridge_uploads":
                        self._send(engine.serialize_bridge_upload_snapshot())
                        return
                    if parsed.path == "/api/read/bridge_frame":
                        self._send(engine.serialize_bridge_frame_snapshot())
                        return
                    if parsed.path == "/api/read/paging":
                        self._send(engine.serialize_paging_state())
                        return
                    if parsed.path == "/api/paging/store/state":
                        self._send(engine.serialize_page_store_state())
                        return
                    if parsed.path == "/api/paging/store/export":
                        self._send(engine.export_page_store_entries())
                        return
                    if parsed.path == "/api/control/state":
                        self._send(
                            {
                                "paused": bool(state.paused),
                                "speed": float(state.speed),
                                "single_step": bool(state.single_step),
                            }
                        )
                        return
                    if parsed.path == "/api/commands/pending":
                        self._send(engine.serialize_pending_commands())
                        return
                    if parsed.path == "/api/readback/pending":
                        self._send(engine.serialize_readback_state())
                        return
                    if parsed.path == "/api/readback/ready":
                        self._send(engine.serialize_ready_readbacks())
                        return
                    if parsed.path == "/api/meta/capabilities":
                        self._send(engine.serialize_engine_capabilities())
                        return
                    if parsed.path == "/api/readback/poll":
                        request_id = query.get("request_id", [None])[0]
                        status = None
                        result = engine.poll_readbacks(None if request_id is None else int(request_id))
                        if request_id is not None and result is None:
                            status = engine.readback_request_status(int(request_id))
                        self._send(
                            {
                                "ready": result is not None,
                                "status": "ready" if result is not None else status,
                                "result": None if result is None else engine.serialize_readback_result(result),
                            }
                        )
                        return
                    if parsed.path == "/api/readback/poll_all":
                        results = engine.poll_all_readbacks()
                        self._send({"results": [engine.serialize_readback_result(result) for result in results]})
                        return
                    if parsed.path == "/api/readback/status":
                        request_id = int(query["request_id"][0])
                        self._send({"request_id": request_id, "status": engine.readback_request_status(request_id)})
                        return
                    if parsed.path == "/api/frame/pending":
                        self._send({"pending": len(engine.pending_frame_inputs), "submission_ids": engine.pending_frame_submission_ids()})
                        return
                    if parsed.path == "/api/frame/pending/detail":
                        self._send(engine.serialize_pending_frame_inputs())
                        return
                    if parsed.path == "/api/frame/state":
                        self._send(engine.serialize_frame_state())
                        return
                    if parsed.path == "/api/frame/output/poll":
                        submission_id = query.get("submission_id", [None])[0]
                        status = None
                        output = engine.poll_frame_output(None if submission_id is None else int(submission_id))
                        if submission_id is not None and output is None:
                            status = engine.frame_submission_status(int(submission_id))
                        self._send(
                            {
                                "ready": output is not None,
                                "status": "ready" if output is not None else status,
                                "output": None if output is None else engine.serialize_frame_output(output),
                            }
                        )
                        return
                    if parsed.path == "/api/frame/output/ready":
                        self._send(engine.serialize_ready_frame_outputs())
                        return
                    if parsed.path == "/api/frame/output/poll_all":
                        outputs = engine.poll_all_frame_outputs()
                        self._send({"outputs": [engine.serialize_frame_output(output) for output in outputs]})
                        return
                    if parsed.path == "/api/frame/output/status":
                        submission_id = int(query["submission_id"][0])
                        self._send({"submission_id": submission_id, "status": engine.frame_submission_status(submission_id)})
                        return
                    if parsed.path == "/api/entity/states":
                        self._send(engine.serialize_entity_states())
                        return
                    if parsed.path == "/api/entity/observations/state":
                        self._send(engine.serialize_entity_observation_state())
                        return
                    if parsed.path == "/api/entity/observations/consumed":
                        self._send(engine.serialize_entity_observation_consume_state())
                        return
                    if parsed.path == "/api/entity/placeholders/state":
                        self._send(engine.serialize_entity_placeholder_index_snapshot())
                        return
                    if parsed.path == "/api/entity/feedback":
                        self._send(engine.serialize_consumed_entity_feedback_snapshot())
                        return
                    if parsed.path == "/api/entity/controller/state":
                        self._send(engine.serialize_controller_state())
                        return
                    if parsed.path == "/api/table/materials":
                        self._send({"materials": engine.serialize_material_table()})
                        return
                    if parsed.path == "/api/table/gases":
                        self._send({"gases": engine.serialize_gas_species_table()})
                        return
                    if parsed.path == "/api/table/lights":
                        self._send({"lights": engine.serialize_light_type_table()})
                        return
                    if parsed.path == "/api/table/optics":
                        self._send({"optics": engine.serialize_material_optics_table()})
                        return
                    if parsed.path == "/api/table/reactions":
                        self._send(engine.serialize_reaction_table())
                        return
                    self._send({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                with engine.state_lock:
                    parsed = urlparse(self.path)
                    payload = self._read_json()
                    try:
                        if parsed.path == "/api/targets/preview":
                            resolved_targets = engine.preview_target_queries(payload["target_queries"])
                            self._send(
                                {
                                    "ok": True,
                                    "resolved_targets": {
                                        query_id: engine.serialize_resolved_target(target)
                                        for query_id, target in resolved_targets.items()
                                    },
                                }
                            )
                            return
                        if parsed.path == "/api/commands/preview":
                            command = engine.preview_world_command(
                                payload["command"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, "command": engine.serialize_world_command(command)})
                            return
                        if parsed.path == "/api/commands/request":
                            command = engine.request_world_command(
                                payload["command"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send_queued(command=engine.serialize_world_command(command))
                            return
                        if parsed.path == "/api/change_intents/preview":
                            resolved_intent = engine.preview_change_intent(
                                payload["intent"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, "resolved_intent": engine.serialize_resolved_change_intent(resolved_intent)})
                            return
                        if parsed.path == "/api/change_intents/request":
                            resolved_intent = engine.request_change_intent(
                                payload["intent"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send_queued(
                                resolved_intent=engine.serialize_resolved_change_intent(resolved_intent),
                                queued=bool(resolved_intent.generated_commands),
                            )
                            return
                        if parsed.path == "/api/carrier_intents/preview":
                            resolved_intent = engine.preview_carrier_intent(
                                payload["intent"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, "resolved_intent": engine.serialize_resolved_carrier_intent(resolved_intent)})
                            return
                        if parsed.path == "/api/carrier_intents/request":
                            resolved_intent = engine.request_carrier_intent(
                                payload["intent"],
                                target_queries=payload.get("target_queries"),
                            )
                            self._send_queued(
                                resolved_intent=engine.serialize_resolved_carrier_intent(resolved_intent),
                                queued=bool(resolved_intent.generated_commands),
                            )
                            return
                        if parsed.path == "/api/material/write":
                            immediate = self._payload_immediate(payload)
                            engine.inject_material(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                payload["material"],
                                int(payload.get("radius", 2)),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/material/fill":
                            immediate = self._payload_immediate(payload)
                            engine.write_material_region(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                int(payload["width"]),
                                int(payload["height"]),
                                payload["material"],
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/inject/temperature":
                            immediate = self._payload_immediate(payload)
                            engine.inject_temperature(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                float(payload["delta"]),
                                int(payload.get("radius", 2)),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/inject/velocity":
                            immediate = self._payload_immediate(payload)
                            engine.inject_velocity(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                tuple(payload.get("velocity", [0.0, 0.0])),
                                int(payload.get("radius", 2)),
                                carrier=str(payload.get("carrier", "cell")),
                                mode=str(payload.get("mode", "add")),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/inject/gas":
                            immediate = self._payload_immediate(payload)
                            engine.inject_gas(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                payload["species"],
                                float(payload["amount"]),
                                int(payload.get("radius", 1)),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/inject/force":
                            immediate = self._payload_immediate(payload)
                            engine.inject_force(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                tuple(payload.get("direction", [0.0, -1.0])),
                                float(payload.get("radius", 8.0)),
                                float(payload.get("strength", 2.0)),
                                float(payload.get("lifetime", 0.4)),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/force_sources/set":
                            engine.set_force_sources(
                                [
                                    {
                                        "x": float(item["x"]),
                                        "y": float(item["y"]),
                                        "direction": tuple(item.get("direction", [0.0, -1.0])),
                                        "radius": float(item.get("radius", 8.0)),
                                        "strength": float(item.get("strength", 2.0)),
                                        "lifetime": float(item.get("lifetime", 0.4)),
                                    }
                                    for item in payload.get("force_sources", [])
                                ]
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/emitters/set":
                            engine.set_emitters(
                                [
                                    {
                                        "x": int(item["x"]),
                                        "y": int(item["y"]),
                                        "light_type": item["light_type"],
                                        "direction": tuple(item.get("direction", [0.0, 0.0])),
                                        "spread": float(item.get("spread", 0.25)),
                                        "strength": float(item.get("strength", 1.0)),
                                        "radius": int(item.get("radius", 8)),
                                    }
                                    for item in payload.get("emitters", [])
                                ]
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/inject/light":
                            immediate = self._payload_immediate(payload)
                            engine.inject_light(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                payload["light_type"],
                                float(payload.get("strength", 1.0)),
                                None if payload.get("radius") is None else int(payload["radius"]),
                                direction=tuple(payload.get("direction", [0.0, 0.0])),
                                spread=float(payload.get("spread", 0.25)),
                                immediate=immediate,
                                **_target_query_kwargs(payload),
                            )
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/readback/request":
                            channels = tuple(str(channel) for channel in payload.get("channels", []))
                            request_id = engine.request_readback(
                                None if payload.get("center_x") is None else int(payload["center_x"]),
                                None if payload.get("center_y") is None else int(payload["center_y"]),
                                int(payload["width"]),
                                int(payload["height"]),
                                channels,
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                observer_id=None if payload.get("observer_id") is None else int(payload["observer_id"]),
                                label=None if payload.get("label") is None else str(payload["label"]),
                                target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                target_dx=int(payload.get("target_dx", 0)),
                                target_dy=int(payload.get("target_dy", 0)),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send_queued(request_id=request_id)
                            return
                        if parsed.path == "/api/readback/plan":
                            channels = tuple(str(channel) for channel in payload.get("channels", []))
                            plan = engine.serialize_readback_plan(
                                None if payload.get("center_x") is None else int(payload["center_x"]),
                                None if payload.get("center_y") is None else int(payload["center_y"]),
                                int(payload["width"]),
                                int(payload["height"]),
                                channels,
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                observer_id=None if payload.get("observer_id") is None else int(payload["observer_id"]),
                                label=None if payload.get("label") is None else str(payload["label"]),
                                target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                target_dx=int(payload.get("target_dx", 0)),
                                target_dy=int(payload.get("target_dy", 0)),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, **plan})
                            return
                        if parsed.path == "/api/readback/preview":
                            channels = tuple(str(channel) for channel in payload.get("channels", []))
                            request = engine.preview_readback(
                                None if payload.get("center_x") is None else int(payload["center_x"]),
                                None if payload.get("center_y") is None else int(payload["center_y"]),
                                int(payload["width"]),
                                int(payload["height"]),
                                channels,
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                observer_id=None if payload.get("observer_id") is None else int(payload["observer_id"]),
                                label=None if payload.get("label") is None else str(payload["label"]),
                                target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                target_dx=int(payload.get("target_dx", 0)),
                                target_dy=int(payload.get("target_dy", 0)),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, "request": engine.serialize_readback_request(request)})
                            return
                        if parsed.path == "/api/readback/cancel":
                            request_id = int(payload["request_id"])
                            canceled = engine.cancel_readback_request(request_id)
                            self._send(
                                {
                                    "ok": canceled,
                                    "request_id": request_id,
                                    "status": engine.readback_request_status(request_id),
                                }
                            )
                            return
                        if parsed.path == "/api/frame/preview":
                            preview = engine.preview_frame_input(payload)
                            self._send({"ok": True, "preview": engine.serialize_frame_preview(preview)})
                            return
                        if parsed.path == "/api/frame/request":
                            queued = engine.request_frame_input(payload)
                            self._send(
                                {
                                    "ok": True,
                                    "queued": queued["queued"],
                                    "pending_frames": queued["pending_frames"],
                                    "submission_id": queued["submission_id"],
                                    "preview": engine.serialize_frame_preview(queued["preview"]),
                                }
                            )
                            return
                        if parsed.path == "/api/frame/submit":
                            submission_id = engine.submit_frame_input(payload)
                            self._send({"ok": True, "queued": True, "pending_frames": len(engine.pending_frame_inputs), "submission_id": submission_id})
                            return
                        if parsed.path == "/api/frame/cycle":
                            cycle = engine.request_frame_cycle(payload, apply_frame=bool(payload.get("apply_frame", True)))
                            self._send(
                                {
                                    "ok": True,
                                    "applied": cycle["applied"],
                                    "queued": cycle["queued"],
                                    "pending_frames": cycle["pending_frames"],
                                    "submission_id": cycle["submission_id"],
                                    "preview": engine.serialize_frame_preview(cycle["preview"]),
                                    "result": None,
                                }
                            )
                            return
                        if parsed.path == "/api/frame/cancel":
                            submission_id = int(payload["submission_id"])
                            canceled = engine.cancel_frame_submission(submission_id)
                            self._send(
                                {
                                    "ok": canceled,
                                    "submission_id": submission_id,
                                    "status": engine.frame_submission_status(submission_id),
                                    "pending_frames": len(engine.pending_frame_inputs),
                                }
                            )
                            return
                        if parsed.path == "/api/frame/cancel_all":
                            canceled = engine.cancel_all_pending_frame_submissions()
                            self._send({"ok": True, "canceled_submission_ids": canceled, "pending_frames": len(engine.pending_frame_inputs)})
                            return
                        if parsed.path == "/api/entity/placeholders":
                            engine.sync_entity_placeholders(
                                [EntityPlaceholder(**item) for item in payload.get("placeholders", [])]
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/entity/states/set":
                            engine.sync_entity_states(payload.get("entities", []))
                            self._send_queued()
                            return
                        if parsed.path == "/api/entity/states/patch":
                            engine.patch_entity_states(payload.get("patches", []))
                            self._send_queued()
                            return
                        if parsed.path == "/api/entity/controller/state/set":
                            submission_id = engine.submit_frame_input(
                                {
                                    "controller_state": payload.get("controller_state"),
                                    "controller_state_provided": "controller_state" in payload,
                                }
                            )
                            self._send(
                                {
                                    "ok": True,
                                    "queued": True,
                                    "pending_frames": len(engine.pending_frame_inputs),
                                    "submission_id": submission_id,
                                }
                            )
                            return
                        if parsed.path == "/api/entity/controller/preview":
                            controller_kwargs = _controller_turn_kwargs(payload)
                            self._send(
                                {
                                    "ok": True,
                                    "preview": engine.preview_entity_controller_turn(**controller_kwargs),
                                }
                            )
                            return
                        if parsed.path == "/api/entity/controller/submit":
                            controller_kwargs = _controller_turn_kwargs(payload)
                            self._send({"ok": True, **engine.request_entity_controller_turn(**controller_kwargs)})
                            return
                        if parsed.path == "/api/entity/controller/turn":
                            controller_kwargs = _controller_turn_kwargs(payload)
                            self._send({"ok": True, **engine.request_entity_controller_turn(**controller_kwargs)})
                            return
                        if parsed.path == "/api/entity/controller/cycle":
                            controller_kwargs = _controller_turn_kwargs(payload)
                            self._send(
                                {
                                    "ok": True,
                                    **engine.request_entity_controller_cycle(
                                        apply_turn=bool(payload.get("apply_turn", True)),
                                        **controller_kwargs,
                                    ),
                                }
                            )
                            return
                        if parsed.path == "/api/entity/observations/set":
                            engine.sync_entity_observation_specs(payload.get("observations", []))
                            self._send_queued()
                            return
                        if parsed.path == "/api/entity/observations/request":
                            request_id = engine.request_observation(
                                {
                                    "observer_id": int(payload["observer_id"]),
                                    "channels": tuple(str(channel) for channel in payload.get("channels", [])),
                                    "center_x": None if payload.get("center_x") is None else int(payload["center_x"]),
                                    "center_y": None if payload.get("center_y") is None else int(payload["center_y"]),
                                    "width": None if payload.get("width") is None else int(payload["width"]),
                                    "height": None if payload.get("height") is None else int(payload["height"]),
                                    "entity_id": None if payload.get("entity_id") is None else int(payload["entity_id"]),
                                    "pad_cells": int(payload.get("pad_cells", 0)),
                                    "label": None if payload.get("label") is None else str(payload["label"]),
                                    "target_query_id": None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                    "target_dx": int(payload.get("target_dx", 0)),
                                    "target_dy": int(payload.get("target_dy", 0)),
                                },
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send_queued(request_id=request_id)
                            return
                        if parsed.path == "/api/entity/observations/preview":
                            request = engine.preview_observation(
                                {
                                    "observer_id": int(payload["observer_id"]),
                                    "channels": tuple(str(channel) for channel in payload.get("channels", [])),
                                    "center_x": None if payload.get("center_x") is None else int(payload["center_x"]),
                                    "center_y": None if payload.get("center_y") is None else int(payload["center_y"]),
                                    "width": None if payload.get("width") is None else int(payload["width"]),
                                    "height": None if payload.get("height") is None else int(payload["height"]),
                                    "entity_id": None if payload.get("entity_id") is None else int(payload["entity_id"]),
                                    "pad_cells": int(payload.get("pad_cells", 0)),
                                    "label": None if payload.get("label") is None else str(payload["label"]),
                                    "target_query_id": None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                    "target_dx": int(payload.get("target_dx", 0)),
                                    "target_dy": int(payload.get("target_dy", 0)),
                                },
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, "request": engine.serialize_readback_request(request)})
                            return
                        if parsed.path == "/api/entity/observations/plan":
                            plan = engine.serialize_observation_plan(
                                {
                                    "observer_id": int(payload["observer_id"]),
                                    "channels": tuple(str(channel) for channel in payload.get("channels", [])),
                                    "center_x": None if payload.get("center_x") is None else int(payload["center_x"]),
                                    "center_y": None if payload.get("center_y") is None else int(payload["center_y"]),
                                    "width": None if payload.get("width") is None else int(payload["width"]),
                                    "height": None if payload.get("height") is None else int(payload["height"]),
                                    "entity_id": None if payload.get("entity_id") is None else int(payload["entity_id"]),
                                    "pad_cells": int(payload.get("pad_cells", 0)),
                                    "label": None if payload.get("label") is None else str(payload["label"]),
                                    "target_query_id": None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                                    "target_dx": int(payload.get("target_dx", 0)),
                                    "target_dy": int(payload.get("target_dy", 0)),
                                },
                                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                                target_queries=payload.get("target_queries"),
                            )
                            self._send({"ok": True, **plan})
                            return
                        if parsed.path == "/api/entity/observations/consume":
                            self._send(engine.consume_entity_observation_results())
                            return
                        if parsed.path == "/api/paging/focus":
                            engine.advance_paging(
                                None if payload.get("x") is None else int(payload["x"]),
                                None if payload.get("y") is None else int(payload["y"]),
                                **_target_query_kwargs(payload),
                            )
                            queued_center = None
                            if engine.command_queue and engine.command_queue[-1].kind == "advance_paging":
                                queued_payload = engine.serialize_world_command(engine.command_queue[-1])["payload"]
                                queued_center = [int(queued_payload["center_x"]), int(queued_payload["center_y"])]
                            self._send_queued(target_center=queued_center)
                            return
                        if parsed.path == "/api/paging/store/has":
                            update = PageStripeUpdate(**payload["update"])
                            self._send({"stored": engine.page_store_has_stripe(update)})
                            return
                        if parsed.path == "/api/paging/store/capture":
                            update = PageStripeUpdate(**payload["update"])
                            stripe_payload = engine.capture_page_stripe_to_store(update)
                            self._send(
                                {
                                    "ok": True,
                                    "stored_stripes": engine.page_store.stored_count(),
                                    "payload": engine.serialize_page_stripe_payload(stripe_payload),
                                }
                            )
                            return
                        if parsed.path == "/api/paging/store/load":
                            update = PageStripeUpdate(**payload["update"])
                            stripe_payload = engine.load_page_stripe(update)
                            self._send(
                                {
                                    "ok": True,
                                    "stored": stripe_payload is not None,
                                    "payload": None if stripe_payload is None else engine.serialize_page_stripe_payload(stripe_payload),
                                }
                            )
                            return
                        if parsed.path == "/api/paging/store/apply":
                            update = PageStripeUpdate(**payload["update"])
                            immediate = bool(payload.get("immediate", False))
                            stripe_payload = engine.apply_stored_page_stripe(update, immediate=immediate)
                            self._send(
                                {
                                    "ok": True,
                                    "stored": stripe_payload is not None,
                                    "queued": bool(stripe_payload is not None and not immediate),
                                    "pending_commands": len(engine.command_queue),
                                }
                            )
                            return
                        if parsed.path == "/api/paging/store/save":
                            update = PageStripeUpdate(**payload["update"])
                            engine.store_page_stripe(update, dict(payload["payload"]))
                            self._send({"ok": True, "stored_stripes": engine.page_store.stored_count()})
                            return
                        if parsed.path == "/api/paging/store/import":
                            result = engine.import_page_store_entries(
                                payload.get("entries", []),
                                clear=bool(payload.get("clear", False)),
                            )
                            self._send({"ok": True, **result})
                            return
                        if parsed.path == "/api/paging/store/clear":
                            cleared = engine.clear_page_store()
                            self._send({"ok": True, "cleared": cleared, "stored_stripes": engine.page_store.stored_count()})
                            return
                        if parsed.path == "/api/paging/stripe/capture":
                            update = PageStripeUpdate(**payload["update"])
                            stripe_payload = engine.capture_page_stripe(update)
                            self._send({"ok": True, "payload": engine.serialize_page_stripe_payload(stripe_payload)})
                            return
                        if parsed.path == "/api/paging/stripe/apply":
                            update = PageStripeUpdate(**payload["update"])
                            immediate = self._payload_immediate(payload)
                            engine.apply_page_stripe(update, dict(payload["payload"]), immediate=immediate)
                            self._send_mutation_result(immediate=immediate)
                            return
                        if parsed.path == "/api/table/material":
                            engine.patch_material(payload["name"], immediate=False, **payload["fields"])
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/light":
                            engine.patch_light(payload["name"], immediate=False, **payload["fields"])
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/gas":
                            engine.patch_gas(payload["name"], immediate=False, **payload["fields"])
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/optic":
                            engine.patch_material_optics(
                                payload["material_name"],
                                payload["light_type"],
                                immediate=False,
                                **payload["fields"],
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reaction":
                            engine.patch_reaction_action(int(payload["index"]), immediate=False, **payload["fields"])
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reaction/delete":
                            engine.delete_reaction_action(int(payload["index"]), immediate=False)
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reaction_rule":
                            engine.patch_reaction_rule(
                                str(payload["rule_set"]),
                                int(payload["index"]),
                                immediate=False,
                                **payload["fields"],
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reaction_rule/delete":
                            engine.delete_reaction_rule(
                                str(payload["rule_set"]),
                                int(payload["index"]),
                                immediate=False,
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/materials/update":
                            engine.update_material_table(payload.get("materials", []), immediate=False)
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/gases/update":
                            engine.update_gas_species_table(payload.get("gases", []), immediate=False)
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/lights/update":
                            engine.update_light_type_table(payload.get("lights", []), immediate=False)
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/optics/update":
                            engine.update_material_optics_table(payload.get("optics", []), immediate=False)
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reactions/update":
                            engine.update_reaction_table(
                                payload.get("actions", []),
                                payload.get("rules", {}),
                                immediate=False,
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/table/reactions/replace":
                            engine.replace_reaction_table(
                                payload.get("actions", []),
                                payload.get("rules", {}),
                                immediate=False,
                            )
                            self._send_queued()
                            return
                        if parsed.path == "/api/control/pause":
                            state.paused = True
                            self._send({"paused": True})
                            return
                        if parsed.path == "/api/control/resume":
                            state.paused = False
                            self._send({"paused": False})
                            return
                        if parsed.path == "/api/control/step":
                            state.single_step = True
                            self._send({"single_step": True})
                            return
                        if parsed.path == "/api/control/tick":
                            engine.step()
                            self._send(
                                {
                                    "ok": True,
                                    "frame_id": int(engine.frame_id),
                                    "pending_commands": len(engine.command_queue),
                                }
                            )
                            return
                        if parsed.path == "/api/control/speed":
                            state.speed = float(payload["speed"])
                            self._send({"speed": state.speed})
                            return
                        if parsed.path == "/api/control/reset":
                            engine.reset_world(immediate=False)
                            self._send_queued()
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
