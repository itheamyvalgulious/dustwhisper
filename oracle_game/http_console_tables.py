from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_tables_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/table/materials":
        handler._send({"materials": engine.serialize_material_table()})
        return True
    if path == "/api/table/gases":
        handler._send({"gases": engine.serialize_gas_species_table()})
        return True
    if path == "/api/table/lights":
        handler._send({"lights": engine.serialize_light_type_table()})
        return True
    if path == "/api/table/optics":
        handler._send({"optics": engine.serialize_material_optics_table()})
        return True
    if path == "/api/table/reactions":
        handler._send(engine.serialize_reaction_table())
        return True
    return False


def route_tables_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/table/material":
        engine.patch_material(payload["name"], immediate=False, **payload["fields"])
        handler._send_queued()
        return True
    if path == "/api/table/light":
        engine.patch_light(payload["name"], immediate=False, **payload["fields"])
        handler._send_queued()
        return True
    if path == "/api/table/gas":
        engine.patch_gas(payload["name"], immediate=False, **payload["fields"])
        handler._send_queued()
        return True
    if path == "/api/table/optic":
        engine.patch_material_optics(
            payload["material_name"],
            payload["light_type"],
            immediate=False,
            **payload["fields"],
        )
        handler._send_queued()
        return True
    if path == "/api/table/reaction":
        engine.patch_reaction_action(int(payload["index"]), immediate=False, **payload["fields"])
        handler._send_queued()
        return True
    if path == "/api/table/reaction/delete":
        engine.delete_reaction_action(int(payload["index"]), immediate=False)
        handler._send_queued()
        return True
    if path == "/api/table/reaction_rule":
        engine.patch_reaction_rule(
            str(payload["rule_set"]),
            int(payload["index"]),
            immediate=False,
            **payload["fields"],
        )
        handler._send_queued()
        return True
    if path == "/api/table/reaction_rule/delete":
        engine.delete_reaction_rule(
            str(payload["rule_set"]),
            int(payload["index"]),
            immediate=False,
        )
        handler._send_queued()
        return True
    if path == "/api/table/materials/update":
        engine.update_material_table(payload.get("materials", []), immediate=False)
        handler._send_queued()
        return True
    if path == "/api/table/gases/update":
        engine.update_gas_species_table(payload.get("gases", []), immediate=False)
        handler._send_queued()
        return True
    if path == "/api/table/lights/update":
        engine.update_light_type_table(payload.get("lights", []), immediate=False)
        handler._send_queued()
        return True
    if path == "/api/table/optics/update":
        engine.update_material_optics_table(payload.get("optics", []), immediate=False)
        handler._send_queued()
        return True
    if path == "/api/table/reactions/update":
        engine.update_reaction_table(
            payload.get("actions", []),
            payload.get("rules", {}),
            immediate=False,
        )
        handler._send_queued()
        return True
    if path == "/api/table/reactions/replace":
        engine.replace_reaction_table(
            payload.get("actions", []),
            payload.get("rules", {}),
            immediate=False,
        )
        handler._send_queued()
        return True
    return False
