"""Entity state, controller, observation, and resolved-intent schema keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from oracle_game.world_constants import (
    ENTITY_STATE_PATCHABLE_FIELDS,
)


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_entities_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the entities portion of the engine capabilities schema."""

    force_source_fields = ctx.force_source_fields
    entity_state_fields = ctx.entity_state_fields
    entity_state_patch_fields = ctx.entity_state_patch_fields
    entity_observation_spec_fields = ctx.entity_observation_spec_fields
    entity_placeholder_fields = ctx.entity_placeholder_fields
    readback_plan_fields = ctx.readback_plan_fields
    observation_plan_fields = ctx.observation_plan_fields
    resolved_target_fields = ctx.resolved_target_fields
    resolved_change_intent_fields = ctx.resolved_change_intent_fields
    resolved_carrier_intent_fields = ctx.resolved_carrier_intent_fields
    observation_result_fields = ctx.observation_result_fields
    entity_observation_runtime_fields = ctx.entity_observation_runtime_fields
    entity_observation_consume_result_fields = ctx.entity_observation_consume_result_fields

    return {
        "force_sources": {
            "fields": force_source_fields,
            "replace_semantics": "replace_all",
        },
        "entity_state": {
            "fields": entity_state_fields,
            "replace_semantics": "replace_all",
            "material_alias_fields": ["placeholder_material"],
            "field_types": {
                "entity_id": {"type": "int"},
                "x": {"type": "int"},
                "y": {"type": "int"},
                "width": {"type": "int"},
                "height": {"type": "int"},
                "velocity_xy": {"type": "float2"},
                "facing_xy": {"type": "float2", "optional": True},
                "placeholder_material": {"type": "str"},
                "tags": {"type": "str[]"},
                "observe_channels": {"type": "str[]"},
                "observe_pad_cells": {"type": "int"},
                "observe_width": {"type": "int", "optional": True},
                "observe_height": {"type": "int", "optional": True},
                "observe_label": {"type": "str", "optional": True},
            },
        },
        "entity_placeholder": {
            "fields": entity_placeholder_fields,
            "replace_semantics": "replace_all",
            "material_alias_fields": ["material"],
            "field_types": {
                "entity_id": {"type": "int"},
                "x": {"type": "int"},
                "y": {"type": "int"},
                "width": {"type": "int"},
                "height": {"type": "int"},
                "material": {"type": "str"},
            },
        },
        "entity_placeholder_runtime": {
            "fields": ["entity_id", "bbox", "cells"],
            "cell_fields": [
                "x",
                "y",
                "material_id",
                "material",
                "phase",
                "displaced_material_id",
                "displaced_material",
            ],
        },
        "entity_feedback": {
            "fields": ["entity_id", "bbox", "cells"],
            "cell_fields": ["x", "y", "present", "material_id", "phase", "integrity", "entity_id"],
        },
        "entity_controller_state": {
            "fields": ["controller_state"],
            "controller_state_type": "json",
            "replace_semantics": "replace_all",
            "persistence": "cpu_side",
        },
        "entity_observation_spec": {
            "fields": entity_observation_spec_fields,
            "replace_semantics": "replace_all",
        },
        "entity_state_patch": {
            "fields": entity_state_patch_fields,
            "patchable_fields": sorted(ENTITY_STATE_PATCHABLE_FIELDS),
            "material_alias_fields": ["placeholder_material"],
            "field_types": {
                "entity_id": {"type": "int"},
                "fields": {"type": "json"},
            },
        },
        "entity_controller_turn": {
            "fields": [
                "controller_state",
                "focus_center",
                "entities",
                "entity_placeholders",
                "patches",
                "observation_specs",
                "force_sources",
                "emitters",
                "target_queries",
                "change_intents",
                "carrier_intents",
                "observation_targets",
                "readback_requests",
                "commands",
            ],
        },
        "entity_controller_cycle": {
            "fields": [
                "apply_turn",
                "controller_state",
                "focus_center",
                "entities",
                "entity_placeholders",
                "patches",
                "observation_specs",
                "force_sources",
                "emitters",
                "target_queries",
                "change_intents",
                "carrier_intents",
                "observation_targets",
                "readback_requests",
                "commands",
            ],
        },
        "resolved_target": {
            "field_order": resolved_target_fields,
            "fields": {
                "query_id": {"type": "str"},
                "status": {"type": "str"},
                "anchor_filters": {"type": "str[]"},
                "direction": {"type": "str", "optional": True},
                "distance_cells": {"type": "int"},
                "distance_meters": {"type": "float", "optional": True},
                "distance_hint": {"type": "str", "optional": True},
                "label": {"type": "str", "optional": True},
                "source_position": {"type": "cell_xy", "optional": True},
                "source_world_position": {"type": "cell_xy", "optional": True},
                "anchor_kind": {"type": "str", "optional": True},
                "anchor_entity_id": {"type": "int", "optional": True},
                "anchor_position": {"type": "cell_xy", "optional": True},
                "anchor_world_position": {"type": "cell_xy", "optional": True},
                "resolved_position": {"type": "cell_xy", "optional": True},
                "resolved_world_position": {"type": "cell_xy", "optional": True},
                "note": {"type": "str", "optional": True},
            },
        },
        "resolved_change_intent": {
            "field_order": resolved_change_intent_fields,
            "fields": {
                "intent_id": {"type": "str"},
                "status": {"type": "str"},
                "target_query_id": {"type": "str", "optional": True},
                "label": {"type": "str", "optional": True},
                "potency": {"type": "float"},
                "stability": {"type": "float"},
                "center_position": {"type": "cell_xy", "optional": True},
                "center_world_position": {"type": "cell_xy", "optional": True},
                "effective_radius": {"type": "int"},
                "material": {"type": "str", "optional": True},
                "temperature_delta": {"type": "float"},
                "velocity": {"type": "float2", "optional": True},
                "velocity_carrier": {"type": "str"},
                "velocity_mode": {"type": "str"},
                "require_empty": {"type": "bool"},
                "fallback_mode": {"type": "str"},
                "fallback_applied": {"type": "bool"},
                "effect_shape": {"type": "str"},
                "effect_cells": {"type": "cell_xy[]"},
                "effect_bounds": {"type": "cell_rect", "optional": True},
                "generated_commands": {"type": "world_command[]"},
                "note": {"type": "str", "optional": True},
            },
        },
        "resolved_carrier_intent": {
            "field_order": resolved_carrier_intent_fields,
            "fields": {
                "intent_id": {"type": "str"},
                "status": {"type": "str"},
                "kind": {"type": "str"},
                "target_query_id": {"type": "str", "optional": True},
                "label": {"type": "str", "optional": True},
                "release_mode": {"type": "str"},
                "potency": {"type": "float"},
                "stability": {"type": "float"},
                "source_position": {"type": "cell_xy", "optional": True},
                "source_world_position": {"type": "cell_xy", "optional": True},
                "impact_position": {"type": "cell_xy", "optional": True},
                "impact_world_position": {"type": "cell_xy", "optional": True},
                "effective_radius": {"type": "int"},
                "material": {"type": "str", "optional": True},
                "gas_species": {"type": "str", "optional": True},
                "gas_amount": {"type": "float"},
                "light_type": {"type": "str", "optional": True},
                "light_strength": {"type": "float"},
                "light_spread": {"type": "float"},
                "force_radius": {"type": "float"},
                "force_strength": {"type": "float"},
                "force_lifetime": {"type": "float"},
                "direction": {"type": "float2", "optional": True},
                "require_empty": {"type": "bool"},
                "fallback_mode": {"type": "str"},
                "fallback_applied": {"type": "bool"},
                "effect_shape": {"type": "str"},
                "effect_cells": {"type": "cell_xy[]"},
                "effect_bounds": {"type": "cell_rect", "optional": True},
                "generated_commands": {"type": "world_command[]"},
                "note": {"type": "str", "optional": True},
            },
        },
        "observation_result": {
            "fields": observation_result_fields,
            "request_type": "readback_request",
            "payload_type": "json",
            "payload_schema_type": "readback_payload",
        },
        "entity_observation_runtime": {
            "fields": entity_observation_runtime_fields,
            "field_types": {
                "observations": {"type": "entity_observation_spec[]"},
                "targets": {"type": "observation_target[]"},
                "requests": {"type": "readback_request[]"},
            },
        },
        "entity_observation_consume_result": {
            "fields": entity_observation_consume_result_fields,
            "field_types": {
                "frame_id": {"type": "int"},
                "consumed": {"type": "int"},
                "consumed_readbacks": {"type": "readback_result[]"},
                "observations": {"type": "observation_result{}", "key": "observer_id"},
                "entity_feedback": {"type": "entity_feedback{}", "key": "entity_id"},
            },
        },
        "entity_controller_turn_result": {
            "fields": [
                "frame_id",
                "controller_state",
                "consumed",
                "paging_updates",
                "resolved_targets",
                "resolved_change_intents",
                "resolved_carrier_intents",
                "resolved_commands",
                "observation_requests",
                "readback_requests",
                "queued_observations",
                "queued_readbacks",
                "queued_commands",
                "entities",
                "placeholders",
                "observation_state",
                "paging_state",
                "readback_state",
                "force_sources",
                "emitters",
                "pending_commands",
            ],
            "field_types": {
                "frame_id": {"type": "int"},
                "controller_state": {"type": "json"},
                "consumed": {"type": "entity_observation_consume_result"},
                "paging_updates": {"type": "page_stripe_update[]"},
                "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                "resolved_commands": {"type": "world_command[]"},
                "observation_requests": {"type": "readback_request[]"},
                "readback_requests": {"type": "readback_request[]"},
                "queued_observations": {"type": "int"},
                "queued_readbacks": {"type": "int"},
                "queued_commands": {"type": "int"},
                "entities": {"type": "entity_state[]"},
                "placeholders": {"type": "entity_placeholder_runtime[]"},
                "observation_state": {"type": "entity_observation_runtime"},
                "paging_state": {"type": "paging_state"},
                "readback_state": {"type": "readback_state"},
                "force_sources": {"type": "force_source[]"},
                "emitters": {"type": "emitters"},
                "pending_commands": {"type": "pending_commands"},
            },
        },
        "entity_controller_turn_preview": {
            "fields": [
                "frame_id",
                "controller_state",
                "consumed",
                "paging_updates",
                "resolved_targets",
                "resolved_change_intents",
                "resolved_carrier_intents",
                "resolved_commands",
                "observation_requests",
                "observation_plans",
                "readback_requests",
                "readback_plans",
                "bridge_frame_snapshot",
                "queued_observations",
                "queued_readbacks",
                "queued_commands",
                "placeholder_count",
                "entities",
                "placeholders",
                "observation_state",
                "paging_state",
                "force_sources",
                "emitters",
                "pending_commands",
            ],
            "field_types": {
                "frame_id": {"type": "int"},
                "controller_state": {"type": "json"},
                "consumed": {"type": "entity_observation_consume_result"},
                "paging_updates": {"type": "page_stripe_update[]"},
                "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                "resolved_commands": {"type": "world_command[]"},
                "observation_requests": {"type": "readback_request[]"},
                "observation_plans": {"type": "observation_plan[]"},
                "readback_requests": {"type": "readback_request[]"},
                "readback_plans": {"type": "readback_plan[]"},
                "bridge_frame_snapshot": {"type": "bridge_frame_snapshot"},
                "queued_observations": {"type": "int"},
                "queued_readbacks": {"type": "int"},
                "queued_commands": {"type": "int"},
                "placeholder_count": {"type": "int"},
                "entities": {"type": "entity_state[]"},
                "placeholders": {"type": "entity_placeholder_runtime[]"},
                "observation_state": {"type": "entity_observation_runtime"},
                "paging_state": {"type": "paging_state"},
                "force_sources": {"type": "force_source[]"},
                "emitters": {"type": "emitters"},
                "pending_commands": {"type": "pending_commands"},
            },
        },
        "entity_controller_cycle_result": {
            "fields": ["applied", "queued", "pending_frames", "submission_id", "preview", "result"],
            "preview_type": "entity_controller_turn_preview",
            "result_type": "entity_controller_turn_result",
            "result_optional_when_unapplied": True,
            "result_optional_when_deferred": True,
            "submission_id_optional_when_unapplied": True,
            "queueing": "deferred_frame_when_applied",
        },
        "entity_controller_submit_result": {
            "fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
            "preview_type": "entity_controller_turn_preview",
            "submission_id_optional": False,
            "queueing": "deferred_frame",
        },
        "entity_states_response": {
            "fields": ["entities"],
            "field_types": {
                "entities": {"type": "entity_state[]"},
            },
        },
        "entity_placeholders_response": {
            "fields": ["placeholders"],
            "field_types": {
                "placeholders": {"type": "entity_placeholder_runtime[]"},
            },
        },
        "entity_feedback_response": {
            "fields": ["feedback"],
            "field_types": {
                "feedback": {"type": "entity_feedback{}", "key": "entity_id"},
            },
        },
        "target_preview_result": {
            "fields": ["ok", "resolved_targets"],
            "field_types": {
                "ok": {"type": "bool"},
                "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
            },
        },
        "command_preview_result": {
            "fields": ["ok", "command"],
            "field_types": {
                "ok": {"type": "bool"},
                "command": {"type": "world_command"},
            },
        },
        "command_request_result": {
            "fields": ["ok", "queued", "pending_commands", "command"],
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "command": {"type": "world_command"},
            },
            "queueing": "deferred_command",
        },
        "change_intent_preview_result": {
            "fields": ["ok", "resolved_intent"],
            "field_types": {
                "ok": {"type": "bool"},
                "resolved_intent": {"type": "resolved_change_intent"},
            },
        },
        "change_intent_request_result": {
            "fields": ["ok", "queued", "pending_commands", "resolved_intent"],
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "resolved_intent": {"type": "resolved_change_intent"},
            },
            "queueing": "deferred_command",
        },
        "carrier_intent_preview_result": {
            "fields": ["ok", "resolved_intent"],
            "field_types": {
                "ok": {"type": "bool"},
                "resolved_intent": {"type": "resolved_carrier_intent"},
            },
        },
        "carrier_intent_request_result": {
            "fields": ["ok", "queued", "pending_commands", "resolved_intent"],
            "field_types": {
                "ok": {"type": "bool"},
                "queued": {"type": "bool"},
                "pending_commands": {"type": "int"},
                "resolved_intent": {"type": "resolved_carrier_intent"},
            },
            "queueing": "deferred_command",
        },
        "readback_preview_result": {
            "fields": ["ok", "request"],
            "field_types": {
                "ok": {"type": "bool"},
                "request": {"type": "readback_request"},
            },
        },
        "readback_plan_result": {
            "fields": ["ok", *readback_plan_fields],
            "field_types": {
                "ok": {"type": "bool"},
                "request": {"type": "readback_request"},
                "layout": {"type": "json"},
                "nbytes": {"type": "int"},
                "gpu_source_count": {"type": "int"},
                "cpu_chunk_count": {"type": "int"},
                "payload": {"type": "json"},
            },
        },
        "observation_plan_result": {
            "fields": ["ok", *observation_plan_fields],
            "field_types": {
                "ok": {"type": "bool"},
                "target": {"type": "observation_target"},
                "request": {"type": "readback_request"},
                "layout": {"type": "json"},
                "nbytes": {"type": "int"},
                "gpu_source_count": {"type": "int"},
                "cpu_chunk_count": {"type": "int"},
                "payload": {"type": "json"},
            },
        },
        "frame_preview_result": {
            "fields": ["ok", "preview"],
            "field_types": {
                "ok": {"type": "bool"},
                "preview": {"type": "frame_preview"},
            },
        },
        "entity_controller_preview_result": {
            "fields": ["ok", "preview"],
            "field_types": {
                "ok": {"type": "bool"},
                "preview": {"type": "entity_controller_turn_preview"},
            },
        },
    }
