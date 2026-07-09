from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_controller_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    if parsed.path == "/api/entity/controller/state":  # type: ignore[attr-defined]
        handler._send(engine.serialize_controller_state())
        return True
    return False


def route_controller_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/targets/preview":
        resolved_targets = engine.preview_target_queries(payload["target_queries"])
        handler._send(
            {
                "ok": True,
                "resolved_targets": {
                    query_id: engine.serialize_resolved_target(target)
                    for query_id, target in resolved_targets.items()
                },
            }
        )
        return True
    if path == "/api/commands/preview":
        command = engine.preview_world_command(
            payload["command"],
            target_queries=payload.get("target_queries"),
        )
        handler._send({"ok": True, "command": engine.serialize_world_command(command)})
        return True
    if path == "/api/commands/request":
        command = engine.request_world_command(
            payload["command"],
            target_queries=payload.get("target_queries"),
        )
        handler._send_queued(command=engine.serialize_world_command(command))
        return True
    if path == "/api/change_intents/preview":
        resolved_intent = engine.preview_change_intent(
            payload["intent"],
            target_queries=payload.get("target_queries"),
        )
        handler._send({"ok": True, "resolved_intent": engine.serialize_resolved_change_intent(resolved_intent)})
        return True
    if path == "/api/change_intents/request":
        resolved_intent = engine.request_change_intent(
            payload["intent"],
            target_queries=payload.get("target_queries"),
        )
        handler._send_queued(
            resolved_intent=engine.serialize_resolved_change_intent(resolved_intent),
            queued=bool(resolved_intent.generated_commands),
        )
        return True
    if path == "/api/carrier_intents/preview":
        resolved_intent = engine.preview_carrier_intent(
            payload["intent"],
            target_queries=payload.get("target_queries"),
        )
        handler._send({"ok": True, "resolved_intent": engine.serialize_resolved_carrier_intent(resolved_intent)})
        return True
    if path == "/api/carrier_intents/request":
        resolved_intent = engine.request_carrier_intent(
            payload["intent"],
            target_queries=payload.get("target_queries"),
        )
        handler._send_queued(
            resolved_intent=engine.serialize_resolved_carrier_intent(resolved_intent),
            queued=bool(resolved_intent.generated_commands),
        )
        return True
    if path == "/api/entity/controller/state/set":
        submission_id = engine.submit_frame_input(
            {
                "controller_state": payload.get("controller_state"),
                "controller_state_provided": "controller_state" in payload,
            }
        )
        handler._send(
            {
                "ok": True,
                "queued": True,
                "pending_frames": len(engine.pending_frame_inputs),
                "submission_id": submission_id,
            }
        )
        return True
    if path == "/api/entity/controller/preview":
        controller_kwargs = console._controller_turn_kwargs(payload)
        handler._send(
            {
                "ok": True,
                "preview": engine.preview_entity_controller_turn(**controller_kwargs),
            }
        )
        return True
    if path == "/api/entity/controller/submit":
        controller_kwargs = console._controller_turn_kwargs(payload)
        handler._send({"ok": True, **engine.request_entity_controller_turn(**controller_kwargs)})
        return True
    if path == "/api/entity/controller/turn":
        controller_kwargs = console._controller_turn_kwargs(payload)
        handler._send({"ok": True, **engine.request_entity_controller_turn(**controller_kwargs)})
        return True
    if path == "/api/entity/controller/cycle":
        controller_kwargs = console._controller_turn_kwargs(payload)
        handler._send(
            {
                "ok": True,
                **engine.request_entity_controller_cycle(
                    apply_turn=bool(payload.get("apply_turn", True)),
                    **controller_kwargs,
                ),
            }
        )
        return True
    return False
