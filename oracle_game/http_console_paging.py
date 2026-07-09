from __future__ import annotations

from typing import TYPE_CHECKING

from oracle_game.types import PageStripeUpdate

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_paging_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/read/paging":
        handler._send(engine.serialize_paging_state())
        return True
    if path == "/api/paging/store/state":
        handler._send(engine.serialize_page_store_state())
        return True
    if path == "/api/paging/store/export":
        handler._send(engine.export_page_store_entries())
        return True
    return False


def route_paging_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/paging/focus":
        engine.advance_paging(
            None if payload.get("x") is None else int(payload["x"]),
            None if payload.get("y") is None else int(payload["y"]),
            **console._target_query_kwargs(payload),
        )
        queued_center = None
        if engine.command_queue and engine.command_queue[-1].kind == "advance_paging":
            queued_payload = engine.serialize_world_command(engine.command_queue[-1])["payload"]
            queued_center = [int(queued_payload["center_x"]), int(queued_payload["center_y"])]
        handler._send_queued(target_center=queued_center)
        return True
    if path == "/api/paging/store/has":
        update = PageStripeUpdate(**payload["update"])
        handler._send({"stored": engine.page_store_has_stripe(update)})
        return True
    if path == "/api/paging/store/capture":
        update = PageStripeUpdate(**payload["update"])
        stripe_payload = engine.capture_page_stripe_to_store(update)
        handler._send(
            {
                "ok": True,
                "stored_stripes": engine.page_store.stored_count(),
                "payload": engine.serialize_page_stripe_payload(stripe_payload),
            }
        )
        return True
    if path == "/api/paging/store/load":
        update = PageStripeUpdate(**payload["update"])
        stripe_payload = engine.load_page_stripe(update)
        handler._send(
            {
                "ok": True,
                "stored": stripe_payload is not None,
                "payload": None if stripe_payload is None else engine.serialize_page_stripe_payload(stripe_payload),
            }
        )
        return True
    if path == "/api/paging/store/apply":
        update = PageStripeUpdate(**payload["update"])
        immediate = bool(payload.get("immediate", False))
        stripe_payload = engine.apply_stored_page_stripe(update, immediate=immediate)
        handler._send(
            {
                "ok": True,
                "stored": stripe_payload is not None,
                "queued": bool(stripe_payload is not None and not immediate),
                "pending_commands": len(engine.command_queue),
            }
        )
        return True
    if path == "/api/paging/store/save":
        update = PageStripeUpdate(**payload["update"])
        engine.store_page_stripe(update, dict(payload["payload"]))
        handler._send({"ok": True, "stored_stripes": engine.page_store.stored_count()})
        return True
    if path == "/api/paging/store/import":
        result = engine.import_page_store_entries(
            payload.get("entries", []),
            clear=bool(payload.get("clear", False)),
        )
        handler._send({"ok": True, **result})
        return True
    if path == "/api/paging/store/clear":
        cleared = engine.clear_page_store()
        handler._send({"ok": True, "cleared": cleared, "stored_stripes": engine.page_store.stored_count()})
        return True
    if path == "/api/paging/stripe/capture":
        update = PageStripeUpdate(**payload["update"])
        stripe_payload = engine.capture_page_stripe(update)
        handler._send({"ok": True, "payload": engine.serialize_page_stripe_payload(stripe_payload)})
        return True
    if path == "/api/paging/stripe/apply":
        update = PageStripeUpdate(**payload["update"])
        immediate = handler._payload_immediate(payload)
        engine.apply_page_stripe(update, dict(payload["payload"]), immediate=immediate)
        handler._send_mutation_result(immediate=immediate)
        return True
    return False
