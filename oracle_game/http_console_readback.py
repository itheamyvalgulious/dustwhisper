from __future__ import annotations

from typing import TYPE_CHECKING

from urllib.parse import parse_qs

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_readback_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/read/cells":
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", ["16"])[0])
        h = int(query.get("h", ["16"])[0])
        handler._send(engine.serialize_local_cells(x, y, w, h))
        return True
    if path == "/api/read/temperature":
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", [str(engine.width)])[0])
        h = int(query.get("h", [str(engine.height)])[0])
        handler._send(engine.serialize_temperature_window(x, y, w, h))
        return True
    if path == "/api/read/gas":
        species = query.get("species", ["water_gas"])[0]
        try:
            handler._send({"species": species, "concentration": engine.serialize_gas(species)})
        except KeyError as exc:
            message = str(exc.args[0]) if exc.args else "invalid gas species"
            handler._send({"error": message}, status=400)
        return True
    if path == "/api/read/pressure":
        handler._send({"pressure": engine.serialize_pressure()})
        return True
    if path == "/api/read/gas_runtime":
        handler._send(engine.serialize_gas_runtime())
        return True
    if path == "/api/read/heat_runtime":
        handler._send(engine.serialize_heat_runtime())
        return True
    if path == "/api/read/liquid_runtime":
        handler._send(engine.serialize_liquid_runtime())
        return True
    if path == "/api/read/reaction_runtime":
        handler._send(engine.serialize_reaction_runtime())
        return True
    if path == "/api/read/collapse_runtime":
        handler._send(engine.serialize_collapse_runtime())
        return True
    if path == "/api/read/optics_runtime":
        handler._send(engine.serialize_optics_runtime())
        return True
    if path == "/api/read/light":
        handler._send({"illumination": engine.serialize_visible_illumination()})
        return True
    if path == "/api/read/demo_runtime":
        handler._send(dict(getattr(engine, "demo_runtime_state", {})))
        return True
    if path == "/api/read/debug_frame":
        view_name = query.get("view", [engine.default_debug_view.value])[0]
        gas_species = query.get("gas_species", [None])[0]
        light_type = query.get("light", [None])[0]
        try:
            handler._send(
                engine.serialize_debug_frame(
                    str(view_name),
                    gas_species=None if gas_species is None else str(gas_species),
                    light_type=None if light_type is None else str(light_type),
                )
            )
        except ValueError:
            handler._send({"error": "unknown debug view", "view": str(view_name)}, status=400)
        except KeyError as exc:
            message = str(exc.args[0]) if exc.args else "invalid debug frame parameter"
            handler._send({"error": message}, status=400)
        return True
    if path == "/api/read/optics":
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", [str(engine.width)])[0])
        h = int(query.get("h", [str(engine.height)])[0])
        light_type = query.get("light", [None])[0]
        try:
            handler._send(engine.serialize_optics(x, y, w, h, light_type=light_type))
        except KeyError as exc:
            message = str(exc.args[0]) if exc.args else "invalid light type"
            handler._send({"error": message}, status=400)
        return True
    if path == "/api/read/velocity":
        handler._send({"velocity": engine.serialize_velocity()})
        return True
    if path == "/api/read/forces":
        handler._send({"force_sources": engine.serialize_force_sources()})
        return True
    if path == "/api/read/emitters":
        handler._send(engine.serialize_emitters())
        return True
    if path == "/api/read/active":
        handler._send(engine.serialize_active_runtime())
        return True
    if path == "/api/read/motion":
        handler._send(engine.serialize_motion_runtime())
        return True
    if path == "/api/read/bridge_runtime":
        handler._send(engine.serialize_bridge_runtime())
        return True
    if path == "/api/read/bridge_resources":
        handler._send(engine.serialize_bridge_resources())
        return True
    if path == "/api/read/bridge_typed_table":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing typed table name"}, status=400)
            return True
        try:
            handler._send(engine.serialize_bridge_typed_table(str(name)))
        except KeyError:
            handler._send({"error": "typed table not found", "name": str(name)}, status=404)
        return True
    if path == "/api/read/bridge_typed_table_slice":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing typed table name"}, status=400)
            return True
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", ["64"])[0])
        try:
            handler._send(engine.serialize_bridge_typed_table_slice(str(name), offset=offset, limit=limit))
        except KeyError:
            handler._send({"error": "typed table not found", "name": str(name)}, status=404)
        return True
    if path == "/api/read/bridge_shadow_buffer":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing shadow buffer name"}, status=400)
            return True
        try:
            handler._send(engine.serialize_bridge_shadow_buffer(str(name)))
        except KeyError:
            handler._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
        return True
    if path == "/api/read/bridge_shadow_buffer_slice":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing shadow buffer name"}, status=400)
            return True
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", ["64"])[0])
        try:
            handler._send(engine.serialize_bridge_shadow_buffer_slice(str(name), offset=offset, limit=limit))
        except KeyError:
            handler._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
        return True
    if path == "/api/read/bridge_shadow_buffer_window":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing shadow buffer name"}, status=400)
            return True
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", ["16"])[0])
        h = int(query.get("h", ["16"])[0])
        try:
            handler._send(engine.serialize_bridge_shadow_buffer_window(str(name), x=x, y=y, w=w, h=h))
        except KeyError:
            handler._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
        except ValueError as exc:
            handler._send({"error": str(exc), "name": str(name)}, status=400)
        return True
    if path == "/api/read/bridge_shadow_buffer_world_window":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing shadow buffer name"}, status=400)
            return True
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", ["16"])[0])
        h = int(query.get("h", ["16"])[0])
        try:
            handler._send(engine.serialize_bridge_shadow_buffer_world_window(str(name), x=x, y=y, w=w, h=h))
        except KeyError:
            handler._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
        except ValueError as exc:
            handler._send({"error": str(exc), "name": str(name)}, status=400)
        return True
    if path == "/api/read/bridge_shadow_buffer_gas_window":
        name = query.get("name", [None])[0]
        if name is None:
            handler._send({"error": "missing shadow buffer name"}, status=400)
            return True
        x = int(query.get("x", ["0"])[0])
        y = int(query.get("y", ["0"])[0])
        w = int(query.get("w", ["4"])[0])
        h = int(query.get("h", ["4"])[0])
        try:
            handler._send(engine.serialize_bridge_shadow_buffer_gas_window(str(name), x=x, y=y, w=w, h=h))
        except KeyError:
            handler._send({"error": "shadow buffer not found", "name": str(name)}, status=404)
        except ValueError as exc:
            handler._send({"error": str(exc), "name": str(name)}, status=400)
        return True
    if path == "/api/read/bridge_uploads":
        handler._send(engine.serialize_bridge_upload_snapshot())
        return True
    if path == "/api/read/bridge_frame":
        handler._send(engine.serialize_bridge_frame_snapshot())
        return True
    if path == "/api/readback/pending":
        handler._send(engine.serialize_readback_state())
        return True
    if path == "/api/readback/ready":
        handler._send(engine.serialize_ready_readbacks())
        return True
    if path == "/api/readback/poll":
        request_id = query.get("request_id", [None])[0]
        status = None
        result = engine.poll_readbacks(None if request_id is None else int(request_id))
        if request_id is not None and result is None:
            status = engine.readback_request_status(int(request_id))
        handler._send(
            {
                "ready": result is not None,
                "status": "ready" if result is not None else status,
                "result": None if result is None else engine.serialize_readback_result(result),
            }
        )
        return True
    if path == "/api/readback/poll_all":
        results = engine.poll_all_readbacks()
        handler._send({"results": [engine.serialize_readback_result(result) for result in results]})
        return True
    if path == "/api/readback/status":
        request_id = int(query["request_id"][0])
        handler._send({"request_id": request_id, "status": engine.readback_request_status(request_id)})
        return True
    if path == "/api/commands/pending":
        handler._send(engine.serialize_pending_commands())
        return True
    if path == "/api/meta/capabilities":
        handler._send(engine.serialize_engine_capabilities())
        return True
    if path == "/api/entity/observations/state":
        handler._send(engine.serialize_entity_observation_state())
        return True
    if path == "/api/entity/observations/consumed":
        handler._send(engine.serialize_entity_observation_consume_state())
        return True
    return False


def route_readback_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/readback/request":
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
        handler._send_queued(request_id=request_id)
        return True
    if path == "/api/readback/plan":
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
        handler._send({"ok": True, **plan})
        return True
    if path == "/api/readback/preview":
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
        handler._send({"ok": True, "request": engine.serialize_readback_request(request)})
        return True
    if path == "/api/readback/cancel":
        request_id = int(payload["request_id"])
        canceled = engine.cancel_readback_request(request_id)
        handler._send(
            {
                "ok": canceled,
                "request_id": request_id,
                "status": engine.readback_request_status(request_id),
            }
        )
        return True
    if path == "/api/entity/observations/request":
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
        handler._send_queued(request_id=request_id)
        return True
    if path == "/api/entity/observations/preview":
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
        handler._send({"ok": True, "request": engine.serialize_readback_request(request)})
        return True
    if path == "/api/entity/observations/plan":
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
        handler._send({"ok": True, **plan})
        return True
    if path == "/api/entity/observations/consume":
        handler._send(engine.consume_entity_observation_results())
        return True
    return False
