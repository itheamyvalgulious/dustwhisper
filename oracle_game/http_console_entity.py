from __future__ import annotations

from typing import TYPE_CHECKING

from oracle_game.types import EntityPlaceholder

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_entity_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/entity/states":
        handler._send(engine.serialize_entity_states())
        return True
    if path == "/api/entity/placeholders/state":
        handler._send(engine.serialize_entity_placeholder_index_snapshot())
        return True
    if path == "/api/entity/feedback":
        handler._send(engine.serialize_consumed_entity_feedback_snapshot())
        return True
    return False


def route_entity_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/entity/placeholders":
        engine.sync_entity_placeholders(
            [EntityPlaceholder(**item) for item in payload.get("placeholders", [])]
        )
        handler._send_queued()
        return True
    if path == "/api/entity/states/set":
        engine.sync_entity_states(payload.get("entities", []))
        handler._send_queued()
        return True
    if path == "/api/entity/states/patch":
        engine.patch_entity_states(payload.get("patches", []))
        handler._send_queued()
        return True
    if path == "/api/entity/observations/set":
        engine.sync_entity_observation_specs(payload.get("observations", []))
        handler._send_queued()
        return True
    return False
