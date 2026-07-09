from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from oracle_game.http_console import EngineHTTPConsole


def route_frame_get(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    query: dict[str, list[str]],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/frame/pending":
        handler._send({"pending": len(engine.pending_frame_inputs), "submission_ids": engine.pending_frame_submission_ids()})
        return True
    if path == "/api/frame/pending/detail":
        handler._send(engine.serialize_pending_frame_inputs())
        return True
    if path == "/api/frame/state":
        handler._send(engine.serialize_frame_state())
        return True
    if path == "/api/frame/output/poll":
        submission_id = query.get("submission_id", [None])[0]
        status = None
        output = engine.poll_frame_output(None if submission_id is None else int(submission_id))
        if submission_id is not None and output is None:
            status = engine.frame_submission_status(int(submission_id))
        handler._send(
            {
                "ready": output is not None,
                "status": "ready" if output is not None else status,
                "output": None if output is None else engine.serialize_frame_output(output),
            }
        )
        return True
    if path == "/api/frame/output/ready":
        handler._send(engine.serialize_ready_frame_outputs())
        return True
    if path == "/api/frame/output/poll_all":
        outputs = engine.poll_all_frame_outputs()
        handler._send({"outputs": [engine.serialize_frame_output(output) for output in outputs]})
        return True
    if path == "/api/frame/output/status":
        submission_id = int(query["submission_id"][0])
        handler._send({"submission_id": submission_id, "status": engine.frame_submission_status(submission_id)})
        return True
    return False


def route_frame_post(
    console: "EngineHTTPConsole",
    handler: "BaseHTTPRequestHandler",
    parsed: object,
    payload: dict[str, object],
) -> bool:
    engine = console.engine
    path = parsed.path  # type: ignore[attr-defined]
    if path == "/api/frame/preview":
        preview = engine.preview_frame_input(payload)
        handler._send({"ok": True, "preview": engine.serialize_frame_preview(preview)})
        return True
    if path == "/api/frame/request":
        queued = engine.request_frame_input(payload)
        handler._send(
            {
                "ok": True,
                "queued": queued["queued"],
                "pending_frames": queued["pending_frames"],
                "submission_id": queued["submission_id"],
                "preview": engine.serialize_frame_preview(queued["preview"]),
            }
        )
        return True
    if path == "/api/frame/submit":
        submission_id = engine.submit_frame_input(payload)
        handler._send({"ok": True, "queued": True, "pending_frames": len(engine.pending_frame_inputs), "submission_id": submission_id})
        return True
    if path == "/api/frame/cycle":
        cycle = engine.request_frame_cycle(payload, apply_frame=bool(payload.get("apply_frame", True)))
        handler._send(
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
        return True
    if path == "/api/frame/cancel":
        submission_id = int(payload["submission_id"])
        canceled = engine.cancel_frame_submission(submission_id)
        handler._send(
            {
                "ok": canceled,
                "submission_id": submission_id,
                "status": engine.frame_submission_status(submission_id),
                "pending_frames": len(engine.pending_frame_inputs),
            }
        )
        return True
    if path == "/api/frame/cancel_all":
        canceled = engine.cancel_all_pending_frame_submissions()
        handler._send({"ok": True, "canceled_submission_ids": canceled, "pending_frames": len(engine.pending_frame_inputs)})
        return True
    return False
