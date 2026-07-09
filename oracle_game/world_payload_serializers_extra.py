"""Payload serializers for frame I/O, entity, resolved-intent, observation, and debug views.

Split from `world_payload_serializers` to keep each module under the 1000-line
limit. All names are re-exported by `world_payload_serializers` so the public
import path (`from oracle_game.world_payload_serializers import X`) is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from copy import deepcopy
from dataclasses import asdict
from oracle_game.page_store import StoredStripeKey
from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    DebugView,
    EntityFeedback,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ObservationResult,
    ObservationTarget,
    PageStripeUpdate,
    Phase,
    ReadbackResult,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    TargetQuery,
    WorldCommand,
    WorldFrameInput,
    WorldFrameOutput,
    WorldFramePreview,
)
from oracle_game.world_constants import ENTITY_STATE_PATCH_METADATA_FIELDS


def serialize_world_command(engine, command: WorldCommand) -> dict[str, Any]:
    public_command = engine._public_world_command(command)
    return {"kind": public_command.kind, "payload": engine._normalize_json_payload_value(public_command.payload)}


def serialize_entity_placeholder_input(placeholder: EntityPlaceholder) -> dict[str, Any]:
    return {
        "entity_id": int(placeholder.entity_id),
        "x": int(placeholder.world_x) if placeholder.world_x is not None else int(placeholder.x),
        "y": int(placeholder.world_y) if placeholder.world_y is not None else int(placeholder.y),
        "width": int(placeholder.width),
        "height": int(placeholder.height),
    }


def serialize_target_query_input(query: TargetQuery) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "anchor_filters": list(query.anchor_filters),
        "source_entity_id": None if query.source_entity_id is None else int(query.source_entity_id),
        "source_x": None if query.source_x is None else int(query.source_x),
        "source_y": None if query.source_y is None else int(query.source_y),
        "anchor_entity_id": None if query.anchor_entity_id is None else int(query.anchor_entity_id),
        "direction": query.direction,
        "distance_cells": int(query.distance_cells),
        "distance_meters": None if query.distance_meters is None else float(query.distance_meters),
        "distance_hint": query.distance_hint,
        "require_empty": bool(query.require_empty),
        "search_radius": int(query.search_radius),
        "label": query.label,
    }


def serialize_page_stripe_update(update: PageStripeUpdate) -> dict[str, Any]:
    return {
        "axis": str(update.axis),
        "world_start": int(update.world_start),
        "world_end": int(update.world_end),
        "buffer_start": int(update.buffer_start),
        "buffer_end": int(update.buffer_end),
        "kind": str(update.kind),
        "cross_world_start": 0 if update.cross_world_start is None else int(update.cross_world_start),
        "cross_world_end": 0 if update.cross_world_end is None else int(update.cross_world_end),
    }


def serialize_page_store_key(key: StoredStripeKey) -> dict[str, Any]:
    return {
        "axis": str(key.axis),
        "world_start": int(key.world_start),
        "world_end": int(key.world_end),
        "cross_world_start": int(getattr(key, "cross_world_start", 0)),
        "cross_world_end": int(getattr(key, "cross_world_end", 0)),
    }


def serialize_page_stripe_payload(engine, payload: dict[str, Any]) -> dict[str, Any]:
    return engine._normalize_json_payload_value(payload)


def serialize_change_intent_input(intent: ChangeIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "target_query_id": intent.target_query_id,
        "center_x": None if intent.center_x is None else int(intent.center_x),
        "center_y": None if intent.center_y is None else int(intent.center_y),
        "target_dx": int(intent.target_dx),
        "target_dy": int(intent.target_dy),
        "radius": int(intent.radius),
        "material": intent.material,
        "temperature_delta": float(intent.temperature_delta),
        "velocity": None if intent.velocity is None else [float(intent.velocity[0]), float(intent.velocity[1])],
        "velocity_carrier": intent.velocity_carrier,
        "velocity_mode": intent.velocity_mode,
        "require_empty": bool(intent.require_empty),
        "fallback_mode": intent.fallback_mode,
        "fallback_radius": int(intent.fallback_radius),
        "potency": float(intent.potency),
        "stability": float(intent.stability),
        "label": intent.label,
    }


def serialize_carrier_intent_input(intent: CarrierIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "kind": intent.kind,
        "target_query_id": intent.target_query_id,
        "center_x": None if intent.center_x is None else int(intent.center_x),
        "center_y": None if intent.center_y is None else int(intent.center_y),
        "source_entity_id": None if intent.source_entity_id is None else int(intent.source_entity_id),
        "source_x": None if intent.source_x is None else int(intent.source_x),
        "source_y": None if intent.source_y is None else int(intent.source_y),
        "target_dx": int(intent.target_dx),
        "target_dy": int(intent.target_dy),
        "radius": int(intent.radius),
        "material": intent.material,
        "gas_species": intent.gas_species,
        "gas_amount": float(intent.gas_amount),
        "light_type": intent.light_type,
        "light_strength": float(intent.light_strength),
        "light_spread": float(intent.light_spread),
        "force_radius": float(intent.force_radius),
        "force_strength": float(intent.force_strength),
        "force_lifetime": float(intent.force_lifetime),
        "release_mode": intent.release_mode,
        "require_empty": bool(intent.require_empty),
        "fallback_mode": intent.fallback_mode,
        "fallback_radius": int(intent.fallback_radius),
        "potency": float(intent.potency),
        "stability": float(intent.stability),
        "label": intent.label,
    }


def serialize_frame_input(engine, frame_input: WorldFrameInput) -> dict[str, Any]:
    return {
        "submission_id": frame_input.submission_id,
        "focus_center": None if frame_input.focus_center is None else list(frame_input.focus_center),
        "controller_state": deepcopy(frame_input.controller_state),
        "controller_state_provided": bool(frame_input.controller_state_provided),
        "entities": [engine.serialize_entity_state_input(entity) for entity in frame_input.entities]
        if frame_input.entities is not None
        else None,
        "entity_placeholders": [engine.serialize_entity_placeholder_input(placeholder) for placeholder in frame_input.entity_placeholders]
        if frame_input.entity_placeholders is not None
        else None,
        "force_sources": None
        if frame_input.force_sources is None
        else [
            {
                "x": float(force_source.x),
                "y": float(force_source.y),
                "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
                "radius": float(force_source.radius),
                "strength": float(force_source.strength),
                "lifetime": float(force_source.lifetime),
            }
            for force_source in frame_input.force_sources
        ],
        "emitters": None
        if frame_input.emitters is None
        else [engine._serialize_emitter_record(emitter) for emitter in frame_input.emitters],
        "target_queries": [engine.serialize_target_query_input(query) for query in frame_input.target_queries],
        "change_intents": [engine.serialize_change_intent_input(intent) for intent in frame_input.change_intents],
        "carrier_intents": [engine.serialize_carrier_intent_input(intent) for intent in frame_input.carrier_intents],
        "observation_targets": [engine.serialize_observation_target(target) for target in frame_input.observation_targets],
        "readback_requests": [engine.serialize_readback_request(request) for request in frame_input.readback_requests],
        "commands": [engine.serialize_world_command(command) for command in frame_input.commands],
    }


def _serialize_readback_payload(engine, payload: Any) -> Any:
    return engine._normalize_json_payload_value(payload)


def serialize_readback_result(engine, result: ReadbackResult) -> dict[str, Any]:
    return {
        "frame_id": int(result.frame_id),
        "request": engine.serialize_readback_request(result.request),
        "payload": engine._serialize_readback_payload(result.payload),
    }


def serialize_resolved_target(engine, target: ResolvedTarget) -> dict[str, Any]:
    target = engine._public_resolved_target(target)
    return {
        "query_id": target.query_id,
        "status": target.status,
        "anchor_filters": list(target.anchor_filters),
        "direction": target.direction,
        "distance_cells": int(target.distance_cells),
        "distance_meters": None if target.distance_meters is None else float(target.distance_meters),
        "distance_hint": target.distance_hint,
        "label": target.label,
        "source_position": None if target.source_position is None else list(target.source_position),
        "source_world_position": None
        if target.source_world_position is None
        else list(target.source_world_position),
        "anchor_kind": target.anchor_kind,
        "anchor_entity_id": target.anchor_entity_id,
        "anchor_position": None if target.anchor_position is None else list(target.anchor_position),
        "anchor_world_position": None
        if target.anchor_world_position is None
        else list(target.anchor_world_position),
        "resolved_position": None if target.resolved_position is None else list(target.resolved_position),
        "resolved_world_position": None
        if target.resolved_world_position is None
        else list(target.resolved_world_position),
        "note": target.note,
    }


def serialize_resolved_change_intent(engine, intent: ResolvedChangeIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "status": intent.status,
        "target_query_id": intent.target_query_id,
        "label": intent.label,
        "potency": float(intent.potency),
        "stability": float(intent.stability),
        "center_position": None if intent.center_position is None else list(intent.center_position),
        "center_world_position": None
        if intent.center_world_position is None
        else list(intent.center_world_position),
        "effective_radius": int(intent.effective_radius),
        "material": intent.material,
        "temperature_delta": float(intent.temperature_delta),
        "velocity": None if intent.velocity is None else [float(intent.velocity[0]), float(intent.velocity[1])],
        "velocity_carrier": intent.velocity_carrier,
        "velocity_mode": intent.velocity_mode,
        "require_empty": bool(intent.require_empty),
        "fallback_mode": intent.fallback_mode,
        "fallback_applied": bool(intent.fallback_applied),
        "effect_shape": intent.effect_shape,
        "effect_cells": [list(cell) for cell in intent.effect_cells],
        "effect_bounds": None if intent.effect_bounds is None else list(intent.effect_bounds),
        "generated_commands": [engine.serialize_world_command(command) for command in intent.generated_commands],
        "note": intent.note,
    }


def serialize_resolved_carrier_intent(engine, intent: ResolvedCarrierIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "status": intent.status,
        "kind": intent.kind,
        "target_query_id": intent.target_query_id,
        "label": intent.label,
        "release_mode": intent.release_mode,
        "potency": float(intent.potency),
        "stability": float(intent.stability),
        "source_position": None if intent.source_position is None else list(intent.source_position),
        "source_world_position": None
        if intent.source_world_position is None
        else list(intent.source_world_position),
        "impact_position": None if intent.impact_position is None else list(intent.impact_position),
        "impact_world_position": None
        if intent.impact_world_position is None
        else list(intent.impact_world_position),
        "effective_radius": int(intent.effective_radius),
        "material": intent.material,
        "gas_species": intent.gas_species,
        "gas_amount": float(intent.gas_amount),
        "light_type": intent.light_type,
        "light_strength": float(intent.light_strength),
        "light_spread": float(intent.light_spread),
        "force_radius": float(intent.force_radius),
        "force_strength": float(intent.force_strength),
        "force_lifetime": float(intent.force_lifetime),
        "direction": None if intent.direction is None else [float(intent.direction[0]), float(intent.direction[1])],
        "require_empty": bool(intent.require_empty),
        "fallback_mode": intent.fallback_mode,
        "fallback_applied": bool(intent.fallback_applied),
        "effect_shape": intent.effect_shape,
        "effect_cells": [list(cell) for cell in intent.effect_cells],
        "effect_bounds": None if intent.effect_bounds is None else list(intent.effect_bounds),
        "generated_commands": [engine.serialize_world_command(command) for command in intent.generated_commands],
        "note": intent.note,
    }


def serialize_observation_result(engine, result: ObservationResult) -> dict[str, Any]:
    return {
        "observer_id": int(result.observer_id),
        "frame_id": int(result.frame_id),
        "request": engine.serialize_readback_request(result.request),
        "payload": engine._serialize_readback_payload(result.payload),
    }


def serialize_entity_observation_spec(spec: EntityObservationSpec) -> dict[str, Any]:
    return {
        "entity_id": int(spec.entity_id),
        "observe_channels": list(spec.observe_channels),
        "observe_pad_cells": int(spec.observe_pad_cells),
        "observe_width": None if spec.observe_width is None else int(spec.observe_width),
        "observe_height": None if spec.observe_height is None else int(spec.observe_height),
        "observe_label": spec.observe_label,
    }


def serialize_entity_state_patch(engine, patch: EntityStatePatch) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in patch.fields.items():
        if name in ENTITY_STATE_PATCH_METADATA_FIELDS:
            continue
        if name == "x":
            fields[name] = int(patch.fields.get("_world_x", value))
        elif name == "y":
            fields[name] = int(patch.fields.get("_world_y", value))
        elif name in {"velocity_xy", "facing_xy"}:
            fields[name] = None if value is None else [float(item) for item in value]
        elif name == "tags":
            fields[name] = list(value)
        elif name == "observe_channels":
            fields[name] = list(value)
        else:
            fields[name] = value
    return {
        "entity_id": int(patch.entity_id),
        "fields": fields,
    }


def serialize_observation_target(target: ObservationTarget) -> dict[str, Any]:
    return {
        "observer_id": int(target.observer_id),
        "channels": list(target.channels),
        "center_x": None if target.center_x is None else int(target.center_x),
        "center_y": None if target.center_y is None else int(target.center_y),
        "width": None if target.width is None else int(target.width),
        "height": None if target.height is None else int(target.height),
        "entity_id": None if target.entity_id is None else int(target.entity_id),
        "pad_cells": int(target.pad_cells),
        "label": target.label,
        "target_query_id": target.target_query_id,
        "target_dx": int(target.target_dx),
        "target_dy": int(target.target_dy),
    }


def serialize_entity_state_input(entity: EntityState) -> dict[str, Any]:
    return {
        "entity_id": int(entity.entity_id),
        "x": int(entity.world_x) if entity.world_x is not None else int(entity.x),
        "y": int(entity.world_y) if entity.world_y is not None else int(entity.y),
        "width": int(entity.width),
        "height": int(entity.height),
        "velocity_xy": [float(entity.velocity_xy[0]), float(entity.velocity_xy[1])],
        "facing_xy": None if entity.facing_xy is None else [float(entity.facing_xy[0]), float(entity.facing_xy[1])],
        "placeholder_material": str(entity.placeholder_material),
        "tags": list(entity.tags),
        "observe_channels": list(entity.observe_channels),
        "observe_pad_cells": int(entity.observe_pad_cells),
        "observe_width": None if entity.observe_width is None else int(entity.observe_width),
        "observe_height": None if entity.observe_height is None else int(entity.observe_height),
        "observe_label": entity.observe_label,
    }


def serialize_entity_state(engine, entity: EntityState) -> dict[str, Any]:
    if entity.world_x is not None and entity.world_y is not None:
        world_x = int(entity.world_x)
        world_y = int(entity.world_y)
    else:
        world_x, world_y = engine._buffer_to_world_position((int(entity.x), int(entity.y)))
    payload = engine.serialize_entity_state_input(entity)
    payload["x"] = int(world_x)
    payload["y"] = int(world_y)
    return payload


def serialize_entity_states(engine) -> dict[str, Any]:
    entities = [engine.serialize_entity_state(entity) for entity in sorted(engine.entity_states.values(), key=lambda item: item.entity_id)]
    return {"entities": entities}


def serialize_entity_observation_state(engine) -> dict[str, Any]:
    entities = [entity for _, entity in sorted(engine.entity_states.items())]
    _, targets = engine._frame_entities_to_placeholders_and_observations(entities)
    requests = engine._build_observation_requests(targets, {})
    return {
        "observations": [
            engine.serialize_entity_observation_spec(
                EntityObservationSpec(
                    entity_id=entity.entity_id,
                    observe_channels=entity.observe_channels,
                    observe_pad_cells=entity.observe_pad_cells,
                    observe_width=entity.observe_width,
                    observe_height=entity.observe_height,
                    observe_label=entity.observe_label,
                )
            )
            for entity in entities
            if entity.observe_channels
        ],
        "targets": [engine.serialize_observation_target(target) for target in targets],
        "requests": [engine.serialize_readback_request(request) for request in requests],
    }


def serialize_entity_placeholders(engine, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
    if not allow_gpu_sync_readback and engine._entity_placeholder_state_gpu_authoritative():
        return engine.serialize_entity_placeholder_index_snapshot()
    payload: list[dict[str, Any]] = []
    cell_state = engine._current_cell_state_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback)
    entity_runtime = engine._current_entity_runtime_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback)
    material_id_grid = cell_state["material_id"]
    phase_grid = cell_state["phase"]
    displaced_grid = entity_runtime["placeholder_displaced_material"]
    for entity_id in sorted(engine.entity_placeholders):
        cells = sorted(engine.entity_placeholders[entity_id], key=lambda cell: (cell[1], cell[0]))
        if not cells:
            continue
        world_cells: list[tuple[int, int, int, int]] = []
        for buffer_x, buffer_y in cells:
            world_x, world_y = engine._buffer_to_world_position((buffer_x, buffer_y))
            world_cells.append((int(world_x), int(world_y), int(buffer_x), int(buffer_y)))
        world_cells.sort(key=lambda cell: (cell[1], cell[0]))
        xs = [cell[0] for cell in world_cells]
        ys = [cell[1] for cell in world_cells]
        payload.append(
            {
                "entity_id": int(entity_id),
                "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                "cells": [
                    {
                        "x": int(world_x),
                        "y": int(world_y),
                        "material_id": int(material_id_grid[buffer_y, buffer_x]),
                        "material": engine._shadow_material_name(int(material_id_grid[buffer_y, buffer_x])),
                        "phase": int(phase_grid[buffer_y, buffer_x]),
                        "displaced_material_id": int(displaced_grid[buffer_y, buffer_x]),
                        "displaced_material": (
                            engine._shadow_material_name(int(displaced_grid[buffer_y, buffer_x]))
                            if int(displaced_grid[buffer_y, buffer_x]) > 0
                            else None
                        ),
                    }
                    for world_x, world_y, buffer_x, buffer_y in world_cells
                ],
            }
        )
    return {"placeholders": payload}


def serialize_entity_placeholder_index_snapshot(engine) -> dict[str, Any]:
    payload: list[dict[str, Any]] = []
    for entity_id in sorted(engine.entity_placeholders):
        cells = sorted(engine.entity_placeholders[entity_id], key=lambda cell: (cell[1], cell[0]))
        if not cells:
            continue
        entity = engine.entity_states.get(int(entity_id))
        material_name = str(entity.placeholder_material) if entity is not None else "placeholder_solid"
        material_id = engine._resolve_sanctioned_placeholder_material_id(material_name)
        if material_id <= 0:
            material_id = int(engine.placeholder_material_id)
        world_cells: list[tuple[int, int]] = []
        for buffer_x, buffer_y in cells:
            world_x, world_y = engine._buffer_to_world_position((buffer_x, buffer_y))
            world_cells.append((int(world_x), int(world_y)))
        world_cells.sort(key=lambda cell: (cell[1], cell[0]))
        xs = [cell[0] for cell in world_cells]
        ys = [cell[1] for cell in world_cells]
        payload.append(
            {
                "entity_id": int(entity_id),
                "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                "cells": [
                    {
                        "x": int(world_x),
                        "y": int(world_y),
                        "material_id": int(material_id),
                        "material": engine._shadow_material_name(int(material_id)),
                        "phase": int(Phase.STATIC_SOLID),
                        "displaced_material_id": 0,
                        "displaced_material": None,
                    }
                    for world_x, world_y in world_cells
                ],
            }
        )
    return {"placeholders": payload}


def serialize_entity_feedback_snapshot(engine, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
    if not allow_gpu_sync_readback and engine._entity_placeholder_state_gpu_authoritative():
        return engine.serialize_consumed_entity_feedback_snapshot()
    feedback = {}
    for entity_id, entity in sorted(engine.entity_states.items()):
        snapshot = engine._build_entity_feedback_from_current_state(
            entity,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        if snapshot is None:
            continue
        feedback[str(entity_id)] = engine.serialize_entity_feedback(snapshot)
    return {"feedback": feedback}


def serialize_consumed_entity_feedback_snapshot(engine) -> dict[str, Any]:
    feedback = engine.last_entity_observation_consume_snapshot.get("entity_feedback", {})
    if isinstance(feedback, dict):
        return {"feedback": deepcopy(feedback)}
    return {"feedback": {}}


def _serialize_cpu_visible_entity_placeholders(engine) -> dict[str, Any]:
    if engine.simulation_backend == "gpu":
        return engine.serialize_entity_placeholder_index_snapshot()
    return engine.serialize_entity_placeholders()


def serialize_entity_feedback(engine, feedback: EntityFeedback) -> dict[str, Any]:
    return {
        "entity_id": int(feedback.entity_id),
        "bbox": list(feedback.bbox),
        "cells": [
            {
                "x": int(cell.x),
                "y": int(cell.y),
                "present": bool(cell.present),
                "material_id": int(cell.material_id),
                "phase": int(cell.phase),
                "integrity": float(cell.integrity),
                "entity_id": int(cell.entity_id),
            }
            for cell in feedback.cells
        ],
    }


def serialize_entity_observation_consume_state(engine) -> dict[str, Any]:
    return deepcopy(engine.last_entity_observation_consume_snapshot)


def serialize_frame_output(engine, output: WorldFrameOutput) -> dict[str, Any]:
    return {
        "frame_id": int(output.frame_id),
        "submission_id": output.submission_id,
        "controller_state": deepcopy(output.controller_state),
        "consumed_readbacks": [engine.serialize_readback_result(result) for result in output.consumed_readbacks],
        "resolved_targets": {
            query_id: engine.serialize_resolved_target(target)
            for query_id, target in output.resolved_targets.items()
        },
        "resolved_change_intents": {
            intent_id: engine.serialize_resolved_change_intent(intent)
            for intent_id, intent in output.resolved_change_intents.items()
        },
        "resolved_carrier_intents": {
            intent_id: engine.serialize_resolved_carrier_intent(intent)
            for intent_id, intent in output.resolved_carrier_intents.items()
        },
        "observations": {
            str(observer_id): engine.serialize_observation_result(result)
            for observer_id, result in output.observations.items()
        },
        "entity_feedback": {
            str(entity_id): engine.serialize_entity_feedback(feedback)
            for entity_id, feedback in output.entity_feedback.items()
        },
        "paging_updates": [asdict(update) for update in output.paging_updates],
        "observation_plans": [
            engine._normalize_json_payload_value(plan)
            for plan in output.observation_plans
        ],
        "readback_plans": [
            engine._normalize_json_payload_value(plan)
            for plan in output.readback_plans
        ],
        "bridge_upload_snapshot": engine._normalize_json_payload_value(output.bridge_upload_snapshot),
        "bridge_frame_snapshot": engine._normalize_json_payload_value(output.bridge_frame_snapshot),
        "queued_observations": int(output.queued_observations),
        "queued_readbacks": int(output.queued_readbacks),
        "queued_commands": int(output.queued_commands),
        "placeholder_count": int(output.placeholder_count),
    }


def serialize_frame_preview(engine, preview: WorldFramePreview) -> dict[str, Any]:
    return {
        "controller_state": deepcopy(preview.controller_state),
        "resolved_targets": {
            query_id: engine.serialize_resolved_target(target)
            for query_id, target in preview.resolved_targets.items()
        },
        "resolved_change_intents": {
            intent_id: engine.serialize_resolved_change_intent(intent)
            for intent_id, intent in preview.resolved_change_intents.items()
        },
        "resolved_carrier_intents": {
            intent_id: engine.serialize_resolved_carrier_intent(intent)
            for intent_id, intent in preview.resolved_carrier_intents.items()
        },
        "resolved_commands": [engine.serialize_world_command(command) for command in preview.resolved_commands],
        "observation_requests": [engine.serialize_readback_request(request) for request in preview.observation_requests],
        "observation_plans": [
            engine._normalize_json_payload_value(plan)
            for plan in preview.observation_plans
        ],
        "readback_requests": [engine.serialize_readback_request(request) for request in preview.readback_requests],
        "readback_plans": [
            engine._normalize_json_payload_value(plan)
            for plan in preview.readback_plans
        ],
        "bridge_frame_snapshot": engine._normalize_json_payload_value(preview.bridge_frame_snapshot),
        "paging_updates": [asdict(update) for update in preview.paging_updates],
        "placeholder_count": int(preview.placeholder_count),
    }


def serialize_debug_frame(
    engine,
    view: DebugView | str,
    *,
    gas_species: str | None = None,
    light_type: str | None = None,
) -> dict[str, Any]:
    resolved_view = view if isinstance(view, DebugView) else DebugView(str(view).lower())
    if resolved_view == DebugView.GAS and gas_species is not None:
        if engine._resolve_sanctioned_gas_id(str(gas_species)) < 0:
            raise KeyError(str(gas_species))
    if resolved_view in {DebugView.OPTICS, DebugView.LIGHT} and light_type is not None:
        if engine._resolve_sanctioned_light_id(str(light_type)) < 0:
            raise KeyError(str(light_type))
    frame = engine.debug_frame(
        resolved_view,
        gas_species=gas_species,
        light_type=light_type,
    )
    return {
        "view": resolved_view.value,
        "origin": [int(engine.paging.origin_x), int(engine.paging.origin_y)],
        "size": [int(engine.width), int(engine.height)],
        "gas_species": None if resolved_view != DebugView.GAS else str(gas_species or "water_gas"),
        "light_type": None if resolved_view not in {DebugView.OPTICS, DebugView.LIGHT} else light_type,
        "frame": np.asarray(frame, dtype=np.float32).round(4).tolist(),
    }
