from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.reactions import REACTION_STAGE_NAMES
from oracle_game.types import CellFlag


def step(solver, world: "WorldEngine", dt: float) -> None:
    solver.reset_runtime_state(world)
    solver._advance_timed_slots(world)
    solver._run_self_rules(world)
    solver._run_material_material(world)
    solver._run_material_gas(world)
    solver._run_material_light(world)
    solver._run_gas_gas(world)
    solver._run_gas_light(world)
    if solver.gpu_pipeline.clear_reaction_latches(world):
        solver._note_runtime_backend("gpu")
    else:
        world._require_gpu_stage("reaction latch clearing")
        world.cell_flags &= np.uint8(~int(CellFlag.REACTION_LATCHED) & 0xFF)
        solver._note_runtime_backend("cpu")


def release(solver) -> None:
    solver.gpu_pipeline.release()
    solver.reset_runtime_state()


def reset_runtime_state(solver, world: "WorldEngine" | None = None) -> None:
    tile_shape = (0, 0) if world is None else (world.active.tile_height, world.active.tile_width)
    cell_shape = (0, 0) if world is None else (world.height, world.width)
    gas_shape = (0, 0) if world is None else (world.gas_height, world.gas_width)
    solver.last_stage_tile_masks = {
        stage: np.zeros(tile_shape, dtype=np.bool_)
        for stage in REACTION_STAGE_NAMES
    }
    solver.last_solve_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver.last_solve_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
    solver.last_changed_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver.last_changed_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
    solver.last_ambient_changed_mask = np.zeros(gas_shape, dtype=np.bool_)
    solver.last_timer_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver.last_emitted_light_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver.last_emitted_material_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver.last_stage_solve_modes = {stage: "empty" for stage in REACTION_STAGE_NAMES}
    solver.last_full_gpu_authoritative_solve_stages: set[str] = set()
    solver.last_full_gpu_authoritative_changed_stages: set[str] = set()
    solver.last_stage_action_counts = {stage: 0 for stage in REACTION_STAGE_NAMES}
    solver.last_executed_action_count = 0
    solver.last_emitted_light_count = 0
    solver.last_emitted_material_count = 0
    solver.last_emit_light_action_count = 0
    solver.last_emit_material_action_count = 0
    solver.last_modify_gas_action_count = 0
    solver.last_convert_material_action_count = 0
    solver.last_modify_temperature_action_count = 0
    solver.last_harm_action_count = 0
    solver.last_runtime_backend = "idle"
    solver._stage_extra_changed_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
    solver._runtime_used_cpu = False
    solver._runtime_used_gpu = False
    solver._current_stage = None
    if world is None:
        solver._full_solve_mask_cache_signature = None
        solver._full_solve_mask_cache = None
        solver._full_gpu_authoritative_mask_cache_signature = None
        solver._full_gpu_authoritative_mask_cache = None


def runtime_snapshot(solver) -> dict[str, object]:
    return {
        "backend": solver.last_runtime_backend,
        "stage_tile_masks": {stage: mask.copy() for stage, mask in solver.last_stage_tile_masks.items()},
        "stage_solve_modes": dict(solver.last_stage_solve_modes),
        "full_gpu_authoritative_solve_stages": sorted(solver.last_full_gpu_authoritative_solve_stages),
        "full_gpu_authoritative_changed_stages": sorted(solver.last_full_gpu_authoritative_changed_stages),
        "solve_cell_mask": solver.last_solve_cell_mask.copy(),
        "solve_gas_mask": solver.last_solve_gas_mask.copy(),
        "changed_cell_mask": solver.last_changed_cell_mask.copy(),
        "changed_gas_mask": solver.last_changed_gas_mask.copy(),
        "ambient_changed_mask": solver.last_ambient_changed_mask.copy(),
        "timer_changed_mask": solver.last_timer_changed_mask.copy(),
        "emitted_light_mask": solver.last_emitted_light_mask.copy(),
        "emitted_material_mask": solver.last_emitted_material_mask.copy(),
        "stage_action_counts": dict(solver.last_stage_action_counts),
        "executed_action_count": int(solver.last_executed_action_count),
        "emitted_light_count": int(solver.last_emitted_light_count),
        "emitted_material_count": int(solver.last_emitted_material_count),
        "emit_light_action_count": int(solver.last_emit_light_action_count),
        "emit_material_action_count": int(solver.last_emit_material_action_count),
        "modify_gas_action_count": int(solver.last_modify_gas_action_count),
        "convert_material_action_count": int(solver.last_convert_material_action_count),
        "modify_temperature_action_count": int(solver.last_modify_temperature_action_count),
        "harm_action_count": int(solver.last_harm_action_count),
    }
