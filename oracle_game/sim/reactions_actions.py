from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.gpu import (
    DIRECTION_IDS,
    REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE,
    REACTION_ACTION_FLAG_RANDOM_TARGET,
)
from oracle_game.sim.gpu_reactions import (
    TYPE_CONVERT_MATERIAL,
    TYPE_EMIT_LIGHT,
    TYPE_EMIT_MATERIAL,
    TYPE_HARM,
    TYPE_MODIFY_GAS,
    TYPE_MODIFY_TEMPERATURE,
)
from oracle_game.sim.reactions import REACTION_FLOW_SOURCE_LIFETIME
from oracle_game.types import CellFlag, ForceSource, ReactionType


def _rule_value(rule: object, field: str, default: object | None = None) -> object | None:
    dtype = getattr(rule, "dtype", None)
    names = None if dtype is None else getattr(dtype, "names", None)
    if names is not None and field in names:
        return rule[field]
    return getattr(rule, field, default)


def _execute_pair_rule(solver, world: "WorldEngine", rule: object, x: int, y: int, scale: float) -> None:
    trigger_slot_index = solver._rule_value(rule, "trigger_slot_index", None)
    if trigger_slot_index is not None and int(trigger_slot_index) >= 0:
        solver._trigger_material_slot(world, x, y, int(trigger_slot_index), scale=scale)
        return
    result_action = solver._rule_value(rule, "result_action", -1)
    if result_action is not None and int(result_action) >= 0:
        solver._execute_action(world, int(result_action), x, y, scale)


def _rule_scale(rule: object, base_scale: float) -> float:
    return max(0.0, float(base_scale) * float(_rule_value(rule, "rate", 1.0)))


def _consume_policy(rule: object) -> str:
    policy_id = _rule_value(rule, "consume_policy_id", None)
    if policy_id is not None:
        return {
            1: "lhs",
            2: "rhs",
            3: "both",
        }.get(int(policy_id), "none")
    return str(_rule_value(rule, "consume_policy", "none") or "none").lower()


def _phase_mask_matches_values(phase_values: np.ndarray, phase_mask: int) -> np.ndarray:
    if phase_mask == 0:
        return np.ones_like(phase_values, dtype=np.bool_)
    phase_bits = np.left_shift(np.uint32(1), phase_values.astype(np.uint32, copy=False))
    return (phase_bits & np.uint32(phase_mask)) != 0


def _apply_material_material_consume(
    solver,
    world: "WorldEngine",
    rule: object,
    x: int,
    y: int,
    rhs_xy: tuple[int, int],
    scale: float,
) -> None:
    policy = solver._consume_policy(rule)
    if policy in {"lhs", "both"}:
        solver._consume_material_cell(world, x, y, scale)
    if policy in {"rhs", "both"}:
        solver._consume_material_cell(world, rhs_xy[0], rhs_xy[1], scale)


def _apply_material_gas_consume(
    solver,
    world: "WorldEngine",
    rule: object,
    x: int,
    y: int,
    gy: int,
    gx: int,
    species_id: int | None,
    scale: float,
) -> None:
    policy = solver._consume_policy(rule)
    if policy in {"lhs", "both"}:
        solver._consume_material_cell(world, x, y, scale)
    if policy in {"rhs", "both"} and species_id is not None:
        solver._consume_gas_species(world, species_id, gy, gx, scale)


def _apply_material_light_consume(
    solver,
    world: "WorldEngine",
    rule: object,
    x: int,
    y: int,
    dose_channel: int,
    scale: float,
) -> None:
    policy = solver._consume_policy(rule)
    if policy in {"lhs", "both"}:
        solver._consume_material_cell(world, x, y, scale)
    if policy in {"rhs", "both"}:
        solver._consume_light_dose(world, dose_channel, x, y, scale)


def _apply_gas_gas_consume(
    solver,
    world: "WorldEngine",
    rule: object,
    gx: int,
    gy: int,
    lhs_species_id: int | None,
    rhs_species_id: int | None,
    scale: float,
) -> None:
    policy = solver._consume_policy(rule)
    if policy in {"lhs", "both"} and lhs_species_id is not None:
        solver._consume_gas_species(world, lhs_species_id, gy, gx, scale)
    if policy in {"rhs", "both"} and rhs_species_id is not None:
        solver._consume_gas_species(world, rhs_species_id, gy, gx, scale)


def _apply_gas_light_consume(
    solver,
    world: "WorldEngine",
    rule: object,
    gx: int,
    gy: int,
    species_id: int | None,
    scale: float,
) -> None:
    policy = solver._consume_policy(rule)
    if policy in {"rhs", "both"} and species_id is not None:
        solver._consume_gas_species(world, species_id, gy, gx, scale)


def _consume_material_cell(solver, world: "WorldEngine", x: int, y: int, amount: float) -> None:
    if amount <= 0.0 or not world.in_bounds(x, y):
        return
    material_id = int(world.material_id[y, x])
    if material_id <= 0:
        return
    world.integrity[y, x] -= float(amount)
    if world.integrity[y, x] <= 0.0:
        world.clear_cell(x, y)


def _consume_gas_species(solver, world: "WorldEngine", species_id: int, gy: int, gx: int, amount: float) -> None:
    if amount <= 0.0 or species_id < 0:
        return
    world.gas_concentration[species_id, gy, gx] = max(
        0.0,
        float(world.gas_concentration[species_id, gy, gx]) - float(amount),
    )


def _consume_light_dose(solver, world: "WorldEngine", dose_channel: int, x: int, y: int, amount: float) -> None:
    if amount <= 0.0:
        return
    world.cell_optical_dose[dose_channel, y, x] = max(
        0.0,
        float(world.cell_optical_dose[dose_channel, y, x]) - float(amount),
    )
    gy, gx = world.cell_to_gas(y, x)
    world.gas_optical_dose[dose_channel, gy, gx] = max(
        0.0,
        float(world.gas_optical_dose[dose_channel, gy, gx]) - float(amount) * 0.08,
    )


def _mask_matches(value: int, required_mask: int) -> bool:
    if required_mask == 0:
        return True
    return (int(value) & int(required_mask)) == int(required_mask)


def _trigger_material_slot(solver, world: "WorldEngine", x: int, y: int, slot_index: int, *, scale: float = 1.0) -> None:
    material_id = int(world.material_id[y, x])
    if material_id <= 0:
        return
    action_index = solver._material_reaction_slot(world, material_id, slot_index)
    if action_index <= 0:
        return
    action = solver._action_row(world, action_index)
    if action is None:
        return
    if slot_index < 4:
        if world.timer_pack[y, x, slot_index] == 0 and int(action["duration"]) > 0:
            world.timer_pack[y, x, slot_index] = int(action["duration"])
        solver._execute_action(world, action_index, x, y, scale)
    else:
        solver._execute_action(world, action_index, x, y, scale)


def _execute_action(solver, world: "WorldEngine", action_index: int, x: int, y: int, scale: float) -> None:
    action = solver._action_row(world, action_index)
    if action is None:
        return
    reaction_type_id = int(action["reaction_type_id"])
    if reaction_type_id == int(ReactionType.NONE.value):
        return
    solver.last_executed_action_count += 1
    if solver._current_stage is not None:
        solver.last_stage_action_counts[solver._current_stage] += 1
    world.cell_flags[y, x] |= int(CellFlag.REACTION_LATCHED)
    if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
        solver.last_emit_material_action_count += 1
        emit_material_id = int(action["emit_material_id"])
        tx, ty, emitted_velocity = solver._material_emit_target_and_velocity(
            world,
            emit_material_id,
            int(action["direction_id"]),
            np.asarray(action["velocity"], dtype=np.float32),
            float(action["speed"]),
            x,
            y,
        )
        if emit_material_id > 0 and world.in_bounds(tx, ty) and world.material_id[ty, tx] == 0:
            world.set_cell_by_id(tx, ty, emit_material_id)
            world.velocity[ty, tx] = emitted_velocity
            solver._stage_extra_changed_cell_mask[ty, tx] = True
            solver.last_emitted_material_count += 1
            solver.last_emitted_material_mask[ty, tx] = True
    elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value):
        solver.last_emit_light_action_count += 1
        light_id = int(action["light_type_id"])
        light_meta = solver._light_emit_metadata(world, light_id)
        if light_meta is None:
            return
        light_name, default_range = light_meta
        range_cells = int(action["range_cells"])
        if range_cells <= 0:
            range_cells = int(default_range)
        world.emitters.append(
            {
                "light_type": light_name,
                "origin": (x, y),
                "direction": solver._direction_vector_id(int(action["direction_id"]), x, y, world),
                "spread": max(0.0, float(action["beam_width"])),
                "strength": max(0.1, float(action["strength"]) * scale),
                "range_cells": range_cells,
            }
        )
        solver.last_emitted_light_count += 1
        solver.last_emitted_light_mask[y, x] = True
    elif reaction_type_id == int(ReactionType.MODIFY_GAS.value):
        solver.last_modify_gas_action_count += 1
        gy, gx = world.cell_to_gas(y, x)
        species_id = int(action["gas_species_id"])
        if 0 <= species_id < world.gas_concentration.shape[0]:
            world.gas_concentration[species_id, gy, gx] = max(
                0.0,
                world.gas_concentration[species_id, gy, gx] + float(action["speed"]) * 0.1 * scale,
            )
        solver._emit_modify_gas_flow_sources(world, action, x, y, scale)
    elif reaction_type_id == int(ReactionType.CONVERT_MATERIAL.value):
        solver.last_convert_material_action_count += 1
        harm_scale = float(scale)
        if not (int(action["flags"]) & REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE):
            harm_scale = max(1.0, harm_scale)
        world.integrity[y, x] -= float(action["harm_per_frame"]) * harm_scale
        if world.integrity[y, x] <= float(action["integrity_threshold"]):
            material_id = int(world.material_id[y, x])
            if int(action["flags"]) & REACTION_ACTION_FLAG_RANDOM_TARGET:
                target_material_id = solver._select_random_convert_material(world, material_id, x, y)
                if target_material_id > 0:
                    world.set_cell_by_id(x, y, target_material_id)
                else:
                    world.clear_cell(x, y)
            else:
                target_material_id = int(action["target_material_id"])
                if target_material_id > 0:
                    world.set_cell_by_id(x, y, target_material_id)
                else:
                    world.clear_cell(x, y)
    elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
        solver.last_modify_temperature_action_count += 1
        world.cell_temperature[y, x] += float(action["delta"]) * scale
    elif reaction_type_id == int(ReactionType.HARM.value):
        solver.last_harm_action_count += 1
        material_id = int(world.material_id[y, x])
        base_integrity = solver._material_base_integrity(world, material_id)
        if base_integrity is not None:
            next_integrity = float(world.integrity[y, x]) - float(action["value"]) * scale
            if float(action["value"]) < 0.0:
                next_integrity = min(base_integrity, next_integrity)
            world.integrity[y, x] = next_integrity
            if world.integrity[y, x] <= 0.0:
                world.clear_cell(x, y)


def _execute_gas_action(solver, world: "WorldEngine", action_index: int, gx: int, gy: int, scale: float) -> None:
    action = solver._action_row(world, action_index)
    if action is None:
        return
    reaction_type_id = int(action["reaction_type_id"])
    if reaction_type_id == int(ReactionType.NONE.value):
        return
    solver.last_executed_action_count += 1
    if solver._current_stage is not None:
        solver.last_stage_action_counts[solver._current_stage] += 1
    if reaction_type_id == int(ReactionType.MODIFY_GAS.value):
        solver.last_modify_gas_action_count += 1
        species_id = int(action["gas_species_id"])
        if 0 <= species_id < world.gas_concentration.shape[0]:
            world.gas_concentration[species_id, gy, gx] = max(
                0.0,
                world.gas_concentration[species_id, gy, gx] + float(action["speed"]) * 0.1 * scale,
            )
        cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
        solver._emit_modify_gas_flow_sources(world, action, cell_x, cell_y, scale)
        return
    if reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
        solver.last_modify_temperature_action_count += 1
        world.ambient_temperature[gy, gx] += float(action["delta"]) * scale
        return
    cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
    if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
        solver.last_emit_material_action_count += 1
        emit_material_id = int(action["emit_material_id"])
        tx, ty, emitted_velocity = solver._material_emit_target_and_velocity(
            world,
            emit_material_id,
            int(action["direction_id"]),
            np.asarray(action["velocity"], dtype=np.float32),
            float(action["speed"]),
            cell_x,
            cell_y,
        )
        if emit_material_id > 0 and world.in_bounds(tx, ty) and world.material_id[ty, tx] == 0:
            world.set_cell_by_id(tx, ty, emit_material_id)
            world.velocity[ty, tx] = emitted_velocity
            solver._stage_extra_changed_cell_mask[ty, tx] = True
            solver.last_emitted_material_count += 1
            solver.last_emitted_material_mask[ty, tx] = True
    elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value):
        solver.last_emit_light_action_count += 1
        light_id = int(action["light_type_id"])
        light_meta = solver._light_emit_metadata(world, light_id)
        if light_meta is None:
            return
        light_name, default_range = light_meta
        range_cells = int(action["range_cells"])
        if range_cells <= 0:
            range_cells = int(default_range)
        world.emitters.append(
            {
                "light_type": light_name,
                "origin": (cell_x, cell_y),
                "direction": solver._gas_direction_vector_id(world, int(action["direction_id"]), gx, gy),
                "spread": max(0.0, float(action["beam_width"])),
                "strength": max(0.1, float(action["strength"]) * scale),
                "range_cells": range_cells,
            }
        )
        solver.last_emitted_light_count += 1
        solver.last_emitted_light_mask[cell_y, cell_x] = True


def _action_row(solver, world: "WorldEngine", action_index: int) -> np.void | None:
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    if action_index < 0 or action_index >= action_table.shape[0]:
        return None
    return action_table[action_index]


def _apply_trigger_grid(solver, world: "WorldEngine", trigger_grid: np.ndarray) -> None:
    ys, xs = np.nonzero(np.any(trigger_grid > 0, axis=-1))
    used_cpu = False
    for y, x in zip(ys.tolist(), xs.tolist()):
        for local_slot, action_index in enumerate(trigger_grid[y, x].tolist()):
            if int(action_index) <= 0:
                continue
            world._require_gpu_stage("reaction trigger action execution")
            used_cpu = True
            solver._execute_action(world, int(action_index), int(x), int(y), 1.0)
    if used_cpu:
        solver._note_runtime_backend("cpu")


def _apply_deferred_batch(solver, world: "WorldEngine", batch: "GPUDeferredActionBatch") -> None:
    solver._record_gpu_local_action_counts(batch.gpu_local_action_counts)
    if solver._formal_gpu_frame(world):
        if getattr(batch, "formal_gpu_empty", False):
            return
        has_cpu_deferred_payload = bool(
            np.any(batch.action_lo > 0)
            or np.any(batch.action_hi > 0)
            or batch.emitted_lights.size > 0
            or np.any(batch.emitted_material_mask)
        )
        if has_cpu_deferred_payload:
            raise RuntimeError("GPU reaction formal frame returned CPU deferred actions; CPU fallback is disabled")
        return
    solver._append_gpu_emitted_lights(world, batch.emitted_lights)
    solver._record_gpu_emitted_materials(world, batch.emitted_material_mask)
    ys, xs = np.nonzero(
        np.any(batch.action_lo > 0, axis=-1) | np.any(batch.action_hi > 0, axis=-1)
    )
    used_cpu = False
    for y, x in zip(ys.tolist(), xs.tolist()):
        action_indices = batch.action_lo[y, x].tolist() + batch.action_hi[y, x].tolist()
        scales = batch.scale_lo[y, x].tolist() + batch.scale_hi[y, x].tolist()
        for action_index, scale in zip(action_indices, scales):
            if int(action_index) <= 0:
                continue
            if solver._deferred_action_handled_by_gpu(world, int(action_index)):
                solver._record_gpu_deferred_action(world, int(action_index))
                continue
            world._require_gpu_stage("deferred reaction action execution")
            used_cpu = True
            solver._execute_action(world, int(action_index), int(x), int(y), float(scale))
    if used_cpu:
        solver._note_runtime_backend("cpu")


def _record_gpu_local_action_counts(solver, counts: np.ndarray) -> None:
    if counts.size == 0:
        return
    values = np.asarray(counts, dtype=np.uint32).reshape(-1)
    if values.size < 8:
        return
    total = int(values[0])
    if total <= 0:
        return
    solver.last_executed_action_count += total
    if solver._current_stage is not None:
        solver.last_stage_action_counts[solver._current_stage] += total
    solver.last_harm_action_count += int(values[TYPE_HARM])
    solver.last_modify_temperature_action_count += int(values[TYPE_MODIFY_TEMPERATURE])
    solver.last_convert_material_action_count += int(values[TYPE_CONVERT_MATERIAL])
    solver.last_modify_gas_action_count += int(values[TYPE_MODIFY_GAS])
    solver.last_emit_light_action_count += int(values[TYPE_EMIT_LIGHT])
    solver.last_emit_material_action_count += int(values[TYPE_EMIT_MATERIAL])


def _append_gpu_emitted_lights(solver, world: "WorldEngine", emitted_lights: np.ndarray) -> None:
    if emitted_lights.size == 0:
        return
    for record in np.asarray(emitted_lights, dtype=np.float32):
        x = int(round(float(record[0])))
        y = int(round(float(record[1])))
        if not world.in_bounds(x, y):
            continue
        light_id = int(round(float(record[7])))
        light_meta = solver._light_emit_metadata(world, light_id)
        if light_meta is None:
            continue
        light_name, default_range = light_meta
        range_cells = int(round(float(record[5])))
        if range_cells <= 0:
            range_cells = int(default_range)
        world.emitters.append(
            {
                "light_type": light_name,
                "origin": (x, y),
                "direction": (float(record[2]), float(record[3])),
                "spread": max(0.0, float(record[6])),
                "strength": max(0.1, float(record[4])),
                "range_cells": range_cells,
            }
        )
        solver.last_emitted_light_count += 1
        solver.last_emitted_light_mask[y, x] = True


def _record_gpu_emitted_materials(solver, world: "WorldEngine", emitted_material_mask: np.ndarray) -> None:
    if emitted_material_mask.size == 0:
        return
    mask = np.asarray(emitted_material_mask, dtype=np.bool_)
    if mask.shape != solver.last_emitted_material_mask.shape or not np.any(mask):
        return
    solver.last_emitted_material_mask |= mask
    solver.last_emitted_material_count += int(np.count_nonzero(mask))
    solver._stage_extra_changed_cell_mask |= mask


def _record_gpu_deferred_action(solver, world: "WorldEngine", action_index: int) -> None:
    action = solver._action_row(world, action_index)
    if action is None:
        return
    reaction_type_id = int(action["reaction_type_id"])
    if reaction_type_id == int(ReactionType.NONE.value):
        return
    solver.last_executed_action_count += 1
    if solver._current_stage is not None:
        solver.last_stage_action_counts[solver._current_stage] += 1
    if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
        solver.last_emit_material_action_count += 1
    elif reaction_type_id == int(ReactionType.MODIFY_GAS.value):
        solver.last_modify_gas_action_count += 1


def _deferred_action_handled_by_gpu(solver, world: "WorldEngine", action_index: int) -> bool:
    action = solver._action_row(world, action_index)
    if action is None:
        return False
    if not world._gpu_pipeline_available(solver.gpu_pipeline, "reactions"):
        return False
    reaction_type_id = int(action["reaction_type_id"])
    if reaction_type_id == int(ReactionType.MODIFY_GAS.value):
        return int(action["gas_species_id"]) >= 0
    if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
        return int(action["emit_material_id"]) > 0
    return False


def _select_random_convert_material(solver, world: "WorldEngine", current_material_id: int, x: int, y: int) -> int:
    candidate_ids = solver._random_convert_candidates(world)
    if not candidate_ids:
        return 0
    selector = solver._deterministic_selector(x, y, len(candidate_ids))
    candidate = candidate_ids[selector]
    if candidate > 0 and candidate != current_material_id:
        return candidate
    for offset in range(1, len(candidate_ids)):
        material_id = candidate_ids[(selector + offset) % len(candidate_ids)]
        if material_id > 0 and material_id != current_material_id:
            return material_id
    return 0


def _emit_modify_gas_flow_sources(
    solver,
    world: "WorldEngine",
    action: np.void,
    x: int,
    y: int,
    scale: float,
) -> None:
    strength = float(action["strength"]) * max(0.0, scale)
    radius = float(action["range_cells"])
    if strength <= 0.0 or radius <= 0.0:
        return
    flow_sources: list[ForceSource] = []
    velocity = np.asarray(action["velocity"], dtype=np.float32)
    velocity_norm = float(np.hypot(velocity[0], velocity[1]))
    direction_id = int(action["direction_id"])
    if velocity_norm > 1e-5:
        direction = (float(velocity[0] / velocity_norm), float(velocity[1] / velocity_norm))
        flow_sources.append(
            ForceSource(
                x=float(x),
                y=float(y),
                direction=direction,
                radius=radius,
                strength=strength,
                lifetime=REACTION_FLOW_SOURCE_LIFETIME,
            )
        )
    elif direction_id != int(DIRECTION_IDS["all"]):
        direction = solver._direction_vector_id(direction_id, x, y, world)
        if abs(direction[0]) > 1e-5 or abs(direction[1]) > 1e-5:
            flow_sources.append(
                ForceSource(
                    x=float(x),
                    y=float(y),
                    direction=direction,
                    radius=radius,
                    strength=strength,
                    lifetime=REACTION_FLOW_SOURCE_LIFETIME,
                )
            )
    else:
        speed = float(action["speed"])
        if abs(speed) <= 1e-5:
            return
        flow_sign = 1.0 if speed > 0.0 else -1.0
        offset = max(1.0, radius * 0.45)
        for radial_x, radial_y in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
            source_x = min(max(float(x) + radial_x * offset, 0.0), float(world.width - 1))
            source_y = min(max(float(y) + radial_y * offset, 0.0), float(world.height - 1))
            flow_sources.append(
                ForceSource(
                    x=source_x,
                    y=source_y,
                    direction=(radial_x * flow_sign, radial_y * flow_sign),
                    radius=radius,
                    strength=strength,
                    lifetime=REACTION_FLOW_SOURCE_LIFETIME,
                )
            )
    if not flow_sources:
        return
    world.force_sources.extend(flow_sources)
    max_radius = int(np.ceil(radius))
    world._mark_active_rect_runtime(
        max(0, x - max_radius),
        max(0, y - max_radius),
        min(world.width, x + max_radius + 1),
        min(world.height, y + max_radius + 1),
    )
