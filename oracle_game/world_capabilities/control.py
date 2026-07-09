"""Control, frame, readback-cancel, deferred, paging, and speed result keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_control_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the control portion of the engine capabilities schema."""

    target_query_fields = ctx.target_query_fields
    inline_target_query_optional_fields = ctx.inline_target_query_optional_fields
    control_state_fields = ctx.control_state_fields
    readback_status_fields = ctx.readback_status_fields
    readback_cancel_request_fields = ctx.readback_cancel_request_fields
    readback_cancel_result_fields = ctx.readback_cancel_result_fields
    frame_pending_state_fields = ctx.frame_pending_state_fields
    frame_state_fields = ctx.frame_state_fields
    frame_submission_status_fields = ctx.frame_submission_status_fields
    frame_cancel_request_fields = ctx.frame_cancel_request_fields
    frame_cancel_result_fields = ctx.frame_cancel_result_fields
    frame_cancel_all_result_fields = ctx.frame_cancel_all_result_fields
    deferred_readback_request_result_fields = ctx.deferred_readback_request_result_fields
    deferred_observation_request_result_fields = ctx.deferred_observation_request_result_fields
    deferred_frame_submit_ack_fields = ctx.deferred_frame_submit_ack_fields
    deferred_controller_state_result_fields = ctx.deferred_controller_state_result_fields
    material_fill_request_fields = ctx.material_fill_request_fields
    paging_focus_request_fields = ctx.paging_focus_request_fields
    paging_focus_result_fields = ctx.paging_focus_result_fields
    page_store_has_result_fields = ctx.page_store_has_result_fields
    page_store_apply_result_fields = ctx.page_store_apply_result_fields
    page_store_clear_result_fields = ctx.page_store_clear_result_fields
    page_stripe_apply_result_fields = ctx.page_stripe_apply_result_fields
    speed_request_fields = ctx.speed_request_fields
    pause_result_fields = ctx.pause_result_fields
    resume_result_fields = ctx.resume_result_fields
    step_result_fields = ctx.step_result_fields
    speed_result_fields = ctx.speed_result_fields
    queued_mutation_result_fields = ctx.queued_mutation_result_fields
    control_reset_result_fields = ctx.control_reset_result_fields

    return {
        "control_state": {
            "fields": control_state_fields,
            "field_types": {
                "paused": {"type": "bool"},
                "speed": {"type": "float"},
                "single_step": {"type": "bool"},
            },
        },
        "readback_status": {
            "fields": readback_status_fields,
            "field_types": {
                "request_id": {"type": "int"},
                "status": {"type": "str"},
            },
        },
        "readback_cancel_request": {
            "fields": readback_cancel_request_fields,
            "field_types": {
                "request_id": {"type": "int"},
            },
        },
        "readback_cancel_result": {
            "fields": readback_cancel_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "request_id": {"type": "int"},
                "status": {"type": "str"},
            },
        },
        "frame_pending_state": {
            "fields": frame_pending_state_fields,
            "field_types": {
                "pending": {"type": "int"},
                "submission_ids": {"type": "int[]"},
            },
        },
        "pending_frame_details": {
            "fields": ["pending", "frames"],
            "field_types": {
                "pending": {"type": "int"},
                "frames": {"type": "pending_frame_detail[]"},
            },
        },
        "frame_state": {
            "fields": frame_state_fields,
            "field_types": {
                "pending": {"type": "int"},
                "pending_submission_ids": {"type": "int[]"},
                "ready": {"type": "int"},
                "ready_submission_ids": {"type": "int[]"},
                "canceled_submission_ids": {"type": "int[]"},
            },
        },
        "frame_submission_status": {
            "fields": frame_submission_status_fields,
            "field_types": {
                "submission_id": {"type": "int"},
                "status": {"type": "str"},
            },
        },
        "frame_cancel_request": {
            "fields": frame_cancel_request_fields,
            "field_types": {
                "submission_id": {"type": "int"},
            },
        },
        "frame_cancel_result": {
            "fields": frame_cancel_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "submission_id": {"type": "int"},
                "status": {"type": "str"},
                "pending_frames": {"type": "int"},
            },
        },
        "frame_cancel_all_result": {
            "fields": frame_cancel_all_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "canceled_submission_ids": {"type": "int[]"},
                "pending_frames": {"type": "int"},
            },
        },
        "queued_mutation_result": {
            "fields": queued_mutation_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
            },
        },
        "deferred_readback_request_result": {
            "fields": deferred_readback_request_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "request_id": {"type": "int"},
            },
        },
        "deferred_observation_request_result": {
            "fields": deferred_observation_request_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "request_id": {"type": "int"},
            },
        },
        "deferred_frame_submit_ack": {
            "fields": deferred_frame_submit_ack_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_frames": {"type": "int"},
                "submission_id": {"type": "int"},
            },
        },
        "deferred_controller_state_result": {
            "fields": deferred_controller_state_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_frames": {"type": "int"},
                "submission_id": {"type": "int"},
            },
        },
        "material_fill_request": {
            "fields": material_fill_request_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "material_alias_fields": ["material"],
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "width": {"type": "int"},
                "height": {"type": "int"},
                "material": {"type": "str"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "paging_focus_request": {
            "fields": paging_focus_request_fields,
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "field_types": {
                "x": {"type": "int"},
                "y": {"type": "int"},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "paging_focus_result": {
            "fields": paging_focus_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "target_center": {"type": "cell_xy", "optional": True},
            },
        },
        "page_store_has_result": {
            "fields": page_store_has_result_fields,
            "field_types": {
                "stored": {"type": "bool"},
            },
        },
        "page_store_apply_result": {
            "fields": page_store_apply_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "stored": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
            },
        },
        "page_store_clear_result": {
            "fields": page_store_clear_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "cleared": {"type": "int"},
                "stored_stripes": {"type": "int"},
            },
        },
        "page_stripe_apply_result": {
            "fields": page_stripe_apply_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
            },
        },
        "speed_request": {
            "fields": speed_request_fields,
            "field_types": {
                "speed": {"type": "float"},
            },
        },
        "pause_result": {
            "fields": pause_result_fields,
            "field_types": {
                "paused": {"type": "bool"},
            },
        },
        "resume_result": {
            "fields": resume_result_fields,
            "field_types": {
                "paused": {"type": "bool"},
            },
        },
        "step_result": {
            "fields": step_result_fields,
            "field_types": {
                "single_step": {"type": "bool"},
            },
        },
        "speed_result": {
            "fields": speed_result_fields,
            "field_types": {
                "speed": {"type": "float"},
            },
        },
        "control_reset_result": {
            "fields": control_reset_result_fields,
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
            },
        },
    }
