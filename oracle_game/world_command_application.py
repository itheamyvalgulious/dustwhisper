from __future__ import annotations

from typing import Any, TYPE_CHECKING

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np

from oracle_game.sim.gpu_world_commands import COMMAND_KIND_IDS as GPU_WORLD_COMMAND_KIND_IDS
from oracle_game.types import (
    EntityObservationSpec,
    EntityPlaceholder,
    EntityStatePatch,
    EntityState,
    ForceSource,
    PageStripeUpdate,
    ReadbackRequest,
    ResolvedTarget,
    WorldCommand,
)
from oracle_game.world_constants import TARGETED_COMMAND_COORD_FIELDS

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _queue_loaded_collapse_pending_regions(engine: "WorldEngine", update: PageStripeUpdate) -> None:
    if update.kind != "load":
        return
    for start, end in engine._stripe_buffer_ranges(update, gas_grid=False):
        if update.axis == "x":
            pending = engine.collapse_delay_pending[:, start:end]
            ys, xs = np.nonzero(pending)
            if ys.size == 0:
                continue
            engine.collapse_deferred_regions.append(
                (
                    max(0, start + int(xs.min()) - 1),
                    max(0, int(ys.min()) - 1),
                    min(engine.width, start + int(xs.max()) + 2),
                    min(engine.height, int(ys.max()) + 2),
                )
            )
            continue
        pending = engine.collapse_delay_pending[start:end, :]
        ys, xs = np.nonzero(pending)
        if ys.size == 0:
            continue
        engine.collapse_deferred_regions.append(
            (
                max(0, int(xs.min()) - 1),
                max(0, start + int(ys.min()) - 1),
                min(engine.width, int(xs.max()) + 2),
                min(engine.height, start + int(ys.max()) + 2),
            )
        )


def _subtract_page_stripe_range_from_region(
    region: tuple[int, int, int, int],
    *,
    axis: str,
    start: int,
    end: int,
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = (int(value) for value in region)
    if x1 <= x0 or y1 <= y0:
        return []
    if axis == "x":
        overlap0 = max(x0, int(start))
        overlap1 = min(x1, int(end))
        if overlap0 >= overlap1:
            return [(x0, y0, x1, y1)]
        remaining: list[tuple[int, int, int, int]] = []
        if x0 < overlap0:
            remaining.append((x0, y0, overlap0, y1))
        if overlap1 < x1:
            remaining.append((overlap1, y0, x1, y1))
        return remaining
    overlap0 = max(y0, int(start))
    overlap1 = min(y1, int(end))
    if overlap0 >= overlap1:
        return [(x0, y0, x1, y1)]
    remaining = []
    if y0 < overlap0:
        remaining.append((x0, y0, x1, overlap0))
    if overlap1 < y1:
        remaining.append((x0, overlap1, x1, y1))
    return remaining


def _apply_grid_world_commands(engine: "WorldEngine", commands: list[WorldCommand]) -> None:
    if not commands:
        engine.grid_command_pipeline.last_backend = "idle"
        return
    if engine._gpu_pipeline_available(
        engine.grid_command_pipeline,
        "world command",
        require=engine.simulation_backend == "gpu",
    ):
        engine.grid_command_pipeline.apply(engine, commands)
        if engine.simulation_backend == "gpu" and engine._world_simulation_frame_active:
            active_rects: list[tuple[int, int, int, int, int]] = []
            for command in commands:
                active_rect, collapse_rect = _grid_world_command_runtime_regions(engine, command)
                if active_rect is not None:
                    active_rects.append(active_rect)
                if collapse_rect is not None:
                    engine._mark_collapse_dirty_rect(*collapse_rect)
            if active_rects and not engine.bridge.mark_active_rects(engine, active_rects):
                engine._require_gpu_stage("active scheduler command marking")
        else:
            for command in commands:
                engine._mark_grid_world_command_runtime_regions(command)
        if engine.grid_command_pipeline.last_cpu_mirror_downloaded:
            engine._rebuild_island_records()
        return
    engine._require_cpu_oracle_backend("world command")
    engine.grid_command_pipeline.last_backend = "cpu"
    for command in commands:
        _apply_grid_world_command_cpu(engine, command)


def _apply_grid_world_command_cpu(engine: "WorldEngine", command: WorldCommand) -> None:
    if command.kind == "inject_material":
        x, y = engine._queued_command_xy(command)
        engine._paint_material(x, y, command.payload["material"], command.payload["radius"])
    elif command.kind == "write_material_region":
        x, y = engine._queued_command_xy(command)
        engine._write_material_region_immediate(
            x,
            y,
            command.payload["width"],
            command.payload["height"],
            command.payload["material"],
        )
    elif command.kind == "inject_temperature":
        x, y = engine._queued_command_xy(command)
        engine._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
    elif command.kind == "inject_velocity":
        x, y = engine._queued_command_xy(command)
        engine._inject_velocity_immediate(
            x,
            y,
            tuple(command.payload["velocity"]),
            command.payload["radius"],
            carrier=command.payload.get("carrier", "cell"),
            mode=command.payload.get("mode", "add"),
        )
    elif command.kind == "inject_gas":
        x, y = engine._queued_command_xy(command)
        engine._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])


def _grid_world_command_runtime_regions(
    engine: "WorldEngine",
    command: WorldCommand,
) -> tuple[tuple[int, int, int, int, int] | None, tuple[int, int, int, int] | None]:
    x, y = engine._queued_command_xy(command)
    if command.kind == "write_material_region":
        width = max(0, int(command.payload["width"]))
        height = max(0, int(command.payload["height"]))
        x0 = max(0, int(x) - 1)
        y0 = max(0, int(y) - 1)
        x1 = min(engine.width, int(x) + width + 1)
        y1 = min(engine.height, int(y) + height + 1)
        if x0 < x1 and y0 < y1:
            return (
                (x0, y0, x1, y1, 0),
                (
                    max(0, int(x)),
                    max(0, int(y)),
                    min(engine.width, int(x) + width),
                    min(engine.height, int(y) + height),
                ),
            )
        return None, None
    radius = max(0, int(command.payload.get("radius", 0)))
    pad = 1 if command.kind == "inject_material" else 0
    active_rect = (
        max(0, int(x) - radius - pad),
        max(0, int(y) - radius - pad),
        min(engine.width, int(x) + radius + pad + 1),
        min(engine.height, int(y) + radius + pad + 1),
    )
    collapse_rect = None
    if command.kind == "inject_material":
        collapse_rect = (
            max(0, int(x) - radius),
            max(0, int(y) - radius),
            min(engine.width, int(x) + radius + 1),
            min(engine.height, int(y) + radius + 1),
        )
    return (*active_rect, 0), collapse_rect


def _apply_commands(engine: "WorldEngine") -> None:
    pending_grid_commands: list[WorldCommand] = []
    def flush_pending_grid_commands() -> None:
        if not pending_grid_commands:
            return
        if engine.simulation_backend == "gpu" and not engine._world_simulation_frame_active:
            engine.bridge.sync_world(engine)
        _apply_grid_world_commands(engine, pending_grid_commands)
        pending_grid_commands.clear()

    while engine.command_queue:
        command = engine.command_queue.popleft()
        engine.bridge_frame_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
        if command.kind in GPU_WORLD_COMMAND_KIND_IDS:
            if engine.simulation_backend == "gpu":
                engine._gpu_pipeline_available(engine.grid_command_pipeline, "world command")
            pending_grid_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
            continue
        flush_pending_grid_commands()
        if command.kind == "inject_material":
            x, y = engine._queued_command_xy(command)
            engine._paint_material(x, y, command.payload["material"], command.payload["radius"])
        elif command.kind == "write_material_region":
            x, y = engine._queued_command_xy(command)
            engine._write_material_region_immediate(
                x,
                y,
                command.payload["width"],
                command.payload["height"],
                command.payload["material"],
            )
        elif command.kind == "inject_temperature":
            x, y = engine._queued_command_xy(command)
            engine._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
        elif command.kind == "inject_velocity":
            x, y = engine._queued_command_xy(command)
            engine._inject_velocity_immediate(
                x,
                y,
                tuple(command.payload["velocity"]),
                command.payload["radius"],
                carrier=command.payload.get("carrier", "cell"),
                mode=command.payload.get("mode", "add"),
            )
        elif command.kind == "inject_force":
            x, y = engine._queued_command_xy(command)
            engine._append_force_source_immediate(
                ForceSource(
                    x=float(x),
                    y=float(y),
                    direction=tuple(command.payload["direction"]),
                    radius=float(command.payload["radius"]),
                    strength=float(command.payload["strength"]),
                    lifetime=float(command.payload.get("lifetime", 0.5)),
                    world_x=float(command.payload["x"]),
                    world_y=float(command.payload["y"]),
                )
            )
        elif command.kind == "inject_gas":
            x, y = engine._queued_command_xy(command)
            engine._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])
        elif command.kind == "inject_light":
            light_type = command.payload["light_type"]
            light_id = engine._resolve_sanctioned_light_id(str(light_type))
            if light_id < 0:
                continue
            if "radius" in command.payload:
                range_cells = int(command.payload["radius"])
            else:
                shadow_default_range = engine._shadow_light_default_range(light_id)
                if shadow_default_range is None:
                    continue
                range_cells = int(shadow_default_range)
            x, y = engine._queued_command_xy(command)
            shadow_light = engine._shadow_light_name(light_id)
            if shadow_light is None:
                continue
            engine._append_transient_light_emitter_immediate(
                {
                    "light_type": shadow_light,
                    "origin": (x, y),
                    "world_origin": (int(command.payload["x"]), int(command.payload["y"])),
                    "direction": tuple(command.payload.get("direction", (0.0, 0.0))),
                    "spread": float(command.payload.get("spread", 0.25)),
                    "strength": command.payload["strength"],
                    "range_cells": range_cells,
                }
            )
        elif command.kind == "sync_entity_placeholders":
            payload = command.payload.get("placeholders", [])
            engine._sync_entity_placeholders(
                [
                    engine._frame_entity_placeholder_input(placeholder)
                    if isinstance(placeholder, EntityPlaceholder)
                    else engine._frame_entity_placeholder_input(EntityPlaceholder(**placeholder))
                    for placeholder in payload
                ]
            )
        elif command.kind == "sync_entity_states":
            payload = command.payload.get("entities", [])
            entities = [
                engine._frame_entity_state_input(entity)
                if isinstance(entity, EntityState)
                else engine._frame_entity_state_input(engine._coerce_entity_state(entity))
                for entity in payload
            ]
            placeholders, _ = engine._sync_entity_states(entities)
            engine._sync_entity_placeholders(placeholders)
        elif command.kind == "patch_entity_states":
            payload = command.payload.get("patches", [])
            engine._patch_entity_states(
                [
                    engine._frame_entity_state_patch_input(patch)
                    if isinstance(patch, EntityStatePatch)
                    else engine._frame_entity_state_patch_input(engine._coerce_entity_state_patch(patch))
                    for patch in payload
                ]
            )
        elif command.kind == "sync_entity_observation_specs":
            payload = command.payload.get("observations", [])
            engine._sync_entity_observation_specs(
                [
                    observation
                    if isinstance(observation, EntityObservationSpec)
                    else engine._coerce_entity_observation_spec(observation)
                    for observation in payload
                ]
            )
        elif command.kind == "set_force_sources":
            payload = command.payload.get("force_sources", [])
            engine._sync_force_sources(
                [
                    engine._public_force_source_input(force_source)
                    for force_source in payload
                ]
            )
        elif command.kind == "set_emitters":
            payload = command.payload.get("emitters", [])
            normalized_emitters = []
            for emitter in payload:
                record = engine._coerce_emitter(emitter)
                normalized_emitters.append(
                    {
                        **dict(record),
                        "origin": engine._world_to_buffer_clamped(
                            int(record["world_origin"][0]),
                            int(record["world_origin"][1]),
                        ),
                    }
                )
            engine._sync_persistent_emitters(normalized_emitters)
        elif command.kind == "advance_paging":
            engine._advance_paging(command.payload["center_x"], command.payload["center_y"])
        elif command.kind == "apply_page_stripe":
            update_payload = command.payload["update"]
            update = update_payload if isinstance(update_payload, PageStripeUpdate) else PageStripeUpdate(**update_payload)
            engine.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
            engine._apply_page_stripe(update, command.payload["payload"])
            engine._record_bridge_page_stripe(update, command.payload["payload"])
        elif command.kind == "reset_world":
            engine._reset_world_state(reset_bridge_frame_inputs=True, keep_command_log=True)
        elif command.kind == "request_readback":
            request = engine._assign_readback_request_id(engine._normalize_readback_request(ReadbackRequest(**command.payload)))
            engine.pending_readbacks.append(request)
            engine.bridge_frame_readback_requests.append(replace(request))
        elif command.kind == "update_material_table":
            engine.update_material_table(command.payload["materials"], immediate=True)
        elif command.kind == "update_gas_species_table":
            engine.update_gas_species_table(command.payload["gases"], immediate=True)
        elif command.kind == "update_light_type_table":
            engine.update_light_type_table(command.payload["lights"], immediate=True)
        elif command.kind == "update_material_optics_table":
            engine.update_material_optics_table(command.payload["optics"], immediate=True)
        elif command.kind == "update_reaction_table":
            engine.update_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
        elif command.kind == "replace_reaction_table":
            engine.replace_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
        elif command.kind == "patch_material":
            engine.patch_material(command.payload["name"], immediate=True, **command.payload["fields"])
        elif command.kind == "patch_light":
            engine.patch_light(command.payload["name"], immediate=True, **command.payload["fields"])
        elif command.kind == "patch_gas":
            engine.patch_gas(command.payload["name"], immediate=True, **command.payload["fields"])
        elif command.kind == "patch_material_optics":
            engine.patch_material_optics(
                command.payload["material_name"],
                command.payload["light_type"],
                immediate=True,
                **command.payload["fields"],
            )
        elif command.kind == "patch_reaction_action":
            engine.patch_reaction_action(command.payload["index"], immediate=True, **command.payload["fields"])
        elif command.kind == "delete_reaction_action":
            engine.delete_reaction_action(command.payload["index"], immediate=True)
        elif command.kind == "patch_reaction_rule":
            engine.patch_reaction_rule(
                command.payload["rule_set"],
                command.payload["index"],
                immediate=True,
                **command.payload["fields"],
            )
        elif command.kind == "delete_reaction_rule":
            engine.delete_reaction_rule(
                command.payload["rule_set"],
                command.payload["index"],
                immediate=True,
            )
    flush_pending_grid_commands()


def _queue_loaded_collapse_pending_regions_from_payload(
    engine: "WorldEngine",
    update: PageStripeUpdate,
    payload: dict[str, Any],
) -> None:
    if update.kind != "load":
        return
    pending_payload = np.asarray(payload["cell"]["collapse_delay_pending"], dtype=np.bool_)
    offset = 0
    for start, end in engine._stripe_buffer_ranges(update, gas_grid=False):
        span = int(end) - int(start)
        if span <= 0:
            continue
        if update.axis == "x":
            pending = pending_payload[:, offset : offset + span]
            ys, xs = np.nonzero(pending)
            if ys.size != 0:
                engine.collapse_deferred_regions.append(
                    (
                        max(0, start + int(xs.min()) - 1),
                        max(0, int(ys.min()) - 1),
                        min(engine.width, start + int(xs.max()) + 2),
                        min(engine.height, int(ys.max()) + 2),
                    )
                )
        else:
            pending = pending_payload[offset : offset + span, :]
            ys, xs = np.nonzero(pending)
            if ys.size != 0:
                engine.collapse_deferred_regions.append(
                    (
                        max(0, int(xs.min()) - 1),
                        max(0, start + int(ys.min()) - 1),
                        min(engine.width, int(xs.max()) + 2),
                        min(engine.height, start + int(ys.max()) + 2),
                    )
                )
        offset += span


def _resolve_targeted_commands(
    engine: "WorldEngine",
    commands: list[WorldCommand],
    resolved_targets: dict[str, ResolvedTarget],
) -> list[WorldCommand]:
    resolved_commands: list[WorldCommand] = []
    for command in commands:
        payload = deepcopy(command.payload)
        target_query_id = payload.pop("target_query_id", None)
        target_dx = int(payload.pop("target_dx", 0))
        target_dy = int(payload.pop("target_dy", 0))
        if target_query_id is None:
            resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
            continue
        target = resolved_targets.get(str(target_query_id))
        fields = TARGETED_COMMAND_COORD_FIELDS.get(command.kind)
        if target is None or target.status != "resolved" or target.resolved_world_position is None or fields is None:
            continue
        world_x = int(target.resolved_world_position[0]) + target_dx
        world_y = int(target.resolved_world_position[1]) + target_dy
        x_field, y_field = fields
        if command.kind in {"request_readback", "advance_paging"}:
            payload[x_field] = int(world_x)
            payload[y_field] = int(world_y)
            payload["target_query_id"] = str(target_query_id)
            payload["target_dx"] = int(target_dx)
            payload["target_dy"] = int(target_dy)
        else:
            payload[x_field] = int(world_x)
            payload[y_field] = int(world_y)
            payload["resolved_target_query_id"] = str(target_query_id)
        resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
    return resolved_commands
