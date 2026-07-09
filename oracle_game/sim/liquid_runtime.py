from __future__ import annotations

import numpy as np

from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask
from oracle_game.types import Phase
from oracle_game.sim.cpu_base import material_table_row

from oracle_game.sim.liquid import (
    LIQUID_ACTIVITY_EPSILON,
    LIQUID_SOLVER_TILE_LEVEL,
    LIQUID_SOLVER_COLUMNAR,
)


def step(solver, world: "WorldEngine") -> None:
    solver.reset_runtime_state(world)
    gpu_available = world._gpu_pipeline_available(solver.gpu_pipeline, "liquid")
    formal_gpu_frame = (
        gpu_available
        and getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )
    active_scheduler_gpu_authoritative = (
        formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
    )
    if formal_gpu_frame and not active_scheduler_gpu_authoritative:
        world._require_gpu_stage("active scheduler liquid solve masks")
    if active_scheduler_gpu_authoritative:
        active_tiles = []
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
    else:
        active_tiles = list(world.active.iter_active_tiles())
        solve_tile_mask = solver._build_solve_tile_mask(world, active_tiles)
    solver.last_solve_tile_mask = solve_tile_mask.copy()
    if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
        return

    if formal_gpu_frame:
        pre_material_id = None
        pre_phase = None
        pre_velocity = None
        pre_temperature = None
        pre_integrity = None
        pre_island_id = None
        pre_placeholder = None
    else:
        pre_material_id = world.material_id.copy()
        pre_phase = world.phase.copy()
        pre_velocity = world.velocity.copy()
        pre_temperature = world.cell_temperature.copy()
        pre_integrity = world.integrity.copy()
        pre_island_id = world.island_id.copy()
        pre_placeholder = world.placeholder_displaced_material.copy()
        solver.last_pending_placeholder_count_before = int(np.count_nonzero(pre_placeholder > 0))
        solver.last_liquid_cell_count_before = int(np.count_nonzero(pre_phase == int(Phase.LIQUID)))

    post_tile_mask = expand_bool_mask(solve_tile_mask, radius=1)
    post_cell_mask = tile_mask_to_cell_mask(
        post_tile_mask,
        tile_size=world.active.tile_size,
        width=world.width,
        height=world.height,
    )
    solver.last_post_tile_mask = post_tile_mask.copy()
    solver.last_post_cell_mask = post_cell_mask.copy()
    solver.last_vertical_seam_mask = solver._vertical_seam_mask(world, post_tile_mask)
    solver.last_horizontal_seam_mask = solver._horizontal_seam_mask(world, post_tile_mask)
    if formal_gpu_frame:
        solver.last_buoyancy_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    else:
        assert pre_material_id is not None
        assert pre_phase is not None
        solver.last_buoyancy_mask = solver._buoyancy_candidate_mask(world, post_cell_mask, pre_material_id, pre_phase)

    if gpu_available:
        solver.gpu_pipeline.step(
            world,
            solve_tile_mask=solve_tile_mask,
            post_tile_mask=post_tile_mask,
        )
        solver.last_backend = "gpu"
        if not active_scheduler_gpu_authoritative:
            solver._refresh_active_tiles(world, active_tiles)
        if not formal_gpu_frame:
            solver._mark_pending_placeholder_regions(world)
    else:
        world._require_cpu_oracle_backend("liquid")
        solver.last_backend = "cpu"
        tile_size = world.active.tile_size
        for tile_x, tile_y in active_tiles:
            x0 = tile_x * tile_size
            y0 = tile_y * tile_size
            x1 = min(world.width, x0 + tile_size)
            y1 = min(world.height, y0 + tile_size)
            solver._solve_tile(world, x0, y0, x1, y1)
        solver._seam_correction(world, post_tile_mask)
        solver._apply_buoyancy(world, post_cell_mask)
        solver._apply_placeholder_displacement(world, post_cell_mask)
        solver._mark_pending_placeholder_regions(world)

    if formal_gpu_frame:
        solver.last_changed_cell_mask = post_cell_mask.copy()
        solver.last_material_changed = True
        solver.last_phase_changed = True
        solver.last_velocity_changed = True
        solver.last_temperature_changed = True
        solver.last_integrity_changed = True
        solver.last_placeholder_changed = True
        return

    assert pre_material_id is not None
    assert pre_phase is not None
    assert pre_velocity is not None
    assert pre_temperature is not None
    assert pre_integrity is not None
    assert pre_island_id is not None
    assert pre_placeholder is not None
    solver._finalize_runtime_state(
        world,
        pre_material_id,
        pre_phase,
        pre_velocity,
        pre_temperature,
        pre_integrity,
        pre_island_id,
        pre_placeholder,
        repair_runtime_state=not gpu_available,
    )

def _finalize_runtime_state(
    solver,
    world: "WorldEngine",
    pre_material_id: np.ndarray,
    pre_phase: np.ndarray,
    pre_velocity: np.ndarray,
    pre_temperature: np.ndarray,
    pre_integrity: np.ndarray,
    pre_island_id: np.ndarray,
    pre_placeholder: np.ndarray,
    *,
    repair_runtime_state: bool = True,
) -> None:
    material_changed_mask = world.material_id != pre_material_id
    phase_changed_mask = world.phase != pre_phase
    runtime_changed_mask = material_changed_mask | phase_changed_mask
    touched_island_ids = np.unique(pre_island_id[runtime_changed_mask])
    if repair_runtime_state:
        non_placeholder_mask = runtime_changed_mask & ~solver._placeholder_mask(world, world.material_id)
        world.entity_id[non_placeholder_mask] = 0
        world.placeholder_displaced_material[non_placeholder_mask] = 0
        invalid_island_mask = runtime_changed_mask & (world.island_id > 0) & (
            (world.phase != int(Phase.FALLING_ISLAND)) | (world.material_id <= 0)
        )
        world.island_id[invalid_island_mask] = 0
    world._refresh_island_records_for_ids(touched_island_ids.tolist())
    velocity_changed_mask = np.any(np.abs(world.velocity - pre_velocity) > LIQUID_ACTIVITY_EPSILON, axis=-1)
    temperature_changed_mask = np.abs(world.cell_temperature - pre_temperature) > LIQUID_ACTIVITY_EPSILON
    integrity_changed_mask = np.abs(world.integrity - pre_integrity) > LIQUID_ACTIVITY_EPSILON
    placeholder_changed_mask = world.placeholder_displaced_material != pre_placeholder
    solver.last_changed_cell_mask = (
        material_changed_mask
        | phase_changed_mask
        | velocity_changed_mask
        | temperature_changed_mask
        | integrity_changed_mask
        | placeholder_changed_mask
    )
    solver.last_material_changed = bool(np.any(material_changed_mask))
    solver.last_phase_changed = bool(np.any(phase_changed_mask))
    solver.last_velocity_changed = bool(np.any(velocity_changed_mask))
    solver.last_temperature_changed = bool(np.any(temperature_changed_mask))
    solver.last_integrity_changed = bool(np.any(integrity_changed_mask))
    solver.last_placeholder_changed = bool(np.any(placeholder_changed_mask))
    solver.last_pending_placeholder_count_after = int(np.count_nonzero(world.placeholder_displaced_material > 0))
    solver.last_liquid_cell_count_after = int(np.count_nonzero(world.phase == int(Phase.LIQUID)))

def release(solver) -> None:
    solver.gpu_pipeline.release()
    solver.reset_runtime_state()

def reset_runtime_state(solver, world: "WorldEngine" | None = None) -> None:
    if world is None:
        solver.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_post_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_post_cell_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_vertical_seam_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_horizontal_seam_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_buoyancy_mask = np.zeros((0, 0), dtype=np.bool_)
        solver.last_changed_cell_mask = np.zeros((0, 0), dtype=np.bool_)
    else:
        solver.last_solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        solver.last_post_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        solver.last_post_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        solver.last_vertical_seam_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        solver.last_horizontal_seam_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        solver.last_buoyancy_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        solver.last_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    solver.last_material_changed = False
    solver.last_phase_changed = False
    solver.last_velocity_changed = False
    solver.last_temperature_changed = False
    solver.last_integrity_changed = False
    solver.last_placeholder_changed = False
    solver.last_pending_placeholder_count_before = 0
    solver.last_pending_placeholder_count_after = 0
    solver.last_liquid_cell_count_before = 0
    solver.last_liquid_cell_count_after = 0

def runtime_snapshot(solver) -> dict[str, np.ndarray | int | bool]:
    return {
        "solve_tile_mask": solver.last_solve_tile_mask.copy(),
        "post_tile_mask": solver.last_post_tile_mask.copy(),
        "post_cell_mask": solver.last_post_cell_mask.copy(),
        "vertical_seam_mask": solver.last_vertical_seam_mask.copy(),
        "horizontal_seam_mask": solver.last_horizontal_seam_mask.copy(),
        "buoyancy_mask": solver.last_buoyancy_mask.copy(),
        "changed_cell_mask": solver.last_changed_cell_mask.copy(),
        "material_changed": bool(solver.last_material_changed),
        "phase_changed": bool(solver.last_phase_changed),
        "velocity_changed": bool(solver.last_velocity_changed),
        "temperature_changed": bool(solver.last_temperature_changed),
        "integrity_changed": bool(solver.last_integrity_changed),
        "placeholder_changed": bool(solver.last_placeholder_changed),
        "pending_placeholder_count_before": int(solver.last_pending_placeholder_count_before),
        "pending_placeholder_count_after": int(solver.last_pending_placeholder_count_after),
        "liquid_cell_count_before": int(solver.last_liquid_cell_count_before),
        "liquid_cell_count_after": int(solver.last_liquid_cell_count_after),
    }

def _material_table_row(solver, world: "WorldEngine", material_id: int) -> np.void | None:
    # Delegated to the shared helper (formerly duplicated verbatim here).
    return material_table_row(world, material_id)

def _material_density(solver, world: "WorldEngine", material_id: int) -> float:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return float(row["density"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return float(shadow_material.density)
    if world._shadow_has_table_payload("materials"):
        return 0.0
    if 0 <= material_id < world.material_density.shape[0]:
        return float(world.material_density[material_id])
    return 0.0

def _material_base_integrity(solver, world: "WorldEngine", material_id: int) -> float:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return float(row["base_integrity"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return float(shadow_material.base_integrity)
    if world._shadow_has_table_payload("materials"):
        return 0.0
    if 0 <= material_id < world.material_base_integrity.shape[0]:
        return float(world.material_base_integrity[material_id])
    return 0.0

def _material_liquid_solver_kind(solver, world: "WorldEngine", material_id: int) -> int:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return int(row["liquid_solver_kind_id"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return LIQUID_SOLVER_COLUMNAR if shadow_material.liquid_solver_kind == "columnar" else LIQUID_SOLVER_TILE_LEVEL
    if world._shadow_has_table_payload("materials"):
        return 0
    if 0 <= material_id < world.material_liquid_solver_kind.shape[0]:
        return int(world.material_liquid_solver_kind[material_id])
    return 0

def _placeholder_material_id(solver, world: "WorldEngine") -> int:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None:
        for row in material_table:
            if int(row["material_id"]) > 0 and int(row["name_hash"]) != 0 and int(row["render_group_id"]) == 7:
                return int(row["material_id"])
        return 0
    return int(world.placeholder_material_id)

def _material_is_placeholder(solver, world: "WorldEngine", material_id: int) -> bool:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return int(row["render_group_id"]) == 7
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return shadow_material.render_group == "placeholder" or "placeholder" in shadow_material.tags
    if world._shadow_has_table_payload("materials"):
        return False
    if 0 <= material_id < world.material_is_placeholder.shape[0]:
        return bool(world.material_is_placeholder[material_id])
    return False

def _placeholder_mask(solver, world: "WorldEngine", material_ids: np.ndarray) -> np.ndarray:
    result = np.zeros(material_ids.shape, dtype=np.bool_)
    positive_mask = material_ids > 0
    if not np.any(positive_mask):
        return result
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None:
        valid_mask = positive_mask & (material_ids < int(material_table.shape[0]))
        if np.any(valid_mask):
            result[valid_mask] = material_table["render_group_id"][material_ids[valid_mask]] == 7
        fallback_mask = positive_mask & ~valid_mask
    else:
        fallback_mask = positive_mask
    if np.any(fallback_mask):
        for material_id in np.unique(material_ids[fallback_mask]).tolist():
            result[fallback_mask & (material_ids == int(material_id))] = solver._material_is_placeholder(world, int(material_id))
    return result
