from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_inject_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/material/write":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_material(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            payload["material"],
            console._bounded_int(payload.get("radius", 2), "radius"),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/material/fill":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.write_material_region(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            console._bounded_int(payload["width"], "width"),
            console._bounded_int(payload["height"], "height"),
            payload["material"],
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/inject/temperature":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_temperature(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            float(payload["delta"]),
            console._bounded_int(payload.get("radius", 2), "radius"),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/inject/velocity":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_velocity(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            tuple(payload.get("velocity", [0.0, 0.0])),
            console._bounded_int(payload.get("radius", 2), "radius"),
            carrier=str(payload.get("carrier", "cell")),
            mode=str(payload.get("mode", "add")),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/inject/gas":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_gas(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            payload["species"],
            float(payload["amount"]),
            console._bounded_int(payload.get("radius", 1), "radius"),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/inject/force":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_force(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            tuple(payload.get("direction", [0.0, -1.0])),
            console._bounded_float(payload.get("radius", 8.0), "radius"),
            float(payload.get("strength", 2.0)),
            float(payload.get("lifetime", 0.4)),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    if path == "/api/force_sources/set":
        engine.set_force_sources(
            [
                {
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "direction": tuple(item.get("direction", [0.0, -1.0])),
                    "radius": console._bounded_float(item.get("radius", 8.0), "radius"),
                    "strength": float(item.get("strength", 2.0)),
                    "lifetime": float(item.get("lifetime", 0.4)),
                }
                for item in payload.get("force_sources", [])
            ]
        )
        handler._send_queued()
        return True
    if path == "/api/emitters/set":
        engine.set_emitters(
            [
                {
                    "x": int(item["x"]),
                    "y": int(item["y"]),
                    "light_type": item["light_type"],
                    "direction": tuple(item.get("direction", [0.0, 0.0])),
                    "spread": float(item.get("spread", 0.25)),
                    "strength": float(item.get("strength", 1.0)),
                    "radius": console._bounded_int(item.get("radius", 8), "radius"),
                }
                for item in payload.get("emitters", [])
            ]
        )
        handler._send_queued()
        return True
    if path == "/api/inject/light":
        immediate = handler._payload_immediate(payload)
        if immediate:
            console._ensure_gpu_attached()
        engine.inject_light(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            payload["light_type"],
            float(payload.get("strength", 1.0)),
            None
            if payload.get("radius") is None
            else console._bounded_int(payload["radius"], "radius"),
            direction=tuple(payload.get("direction", [0.0, 0.0])),
            spread=float(payload.get("spread", 0.25)),
            immediate=immediate,
            **console._target_query_kwargs(payload),
        )
        handler._send_mutation_result(immediate=immediate)
        return True
    return False
