from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.reactions import REACTION_ACTIVITY_EPSILON, GPUAuthoritativeFullSolveMask
from oracle_game.sim.utils import expand_bool_mask
# tile_mask_to_cell_mask / tile_mask_to_gas_mask are referenced through the
# facade module (imported below) so tests that monkeypatch
# oracle_game.sim.reactions.tile_mask_to_*_mask can intercept these calls.
import oracle_game.sim.reactions as _reactions_facade


def _solve_masks(
    solver,
    world: "WorldEngine",
    *,
    seed_timer_cells: bool,
    stage: str | None = None,
):
    formal_gpu_frame = solver._formal_gpu_frame(world)
    active_scheduler_gpu_authoritative = solver._active_scheduler_gpu_authoritative(world)
    if formal_gpu_frame and not active_scheduler_gpu_authoritative:
        world._require_gpu_stage("active scheduler reaction solve masks")
    if solver._use_full_gpu_authoritative_reaction_solve_masks(world, stage):
        return solver._full_gpu_authoritative_solve_masks(world)
    if active_scheduler_gpu_authoritative:
        return solver._full_solve_masks(world)
    else:
        solve_tile_mask = solver._solve_tile_mask(world, seed_timer_cells=seed_timer_cells)
        if (
            not formal_gpu_frame
            and not bool(getattr(world, "_world_simulation_frame_active", False))
            and not np.any(solve_tile_mask)
        ):
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
    solve_cell_mask = _reactions_facade.tile_mask_to_cell_mask(
        solve_tile_mask,
        tile_size=world.active.tile_size,
        width=world.width,
        height=world.height,
    )
    solve_gas_mask = _reactions_facade.tile_mask_to_gas_mask(
        solve_tile_mask,
        tile_size=world.active.tile_size,
        gas_cell_size=world.gas_cell_size,
        width=world.width,
        height=world.height,
        gas_width=world.gas_width,
        gas_height=world.gas_height,
    )
    return solve_tile_mask, solve_cell_mask, solve_gas_mask


def _use_full_gpu_authoritative_reaction_solve_masks(solver, world: "WorldEngine", stage: str | None) -> bool:
    if not solver._active_scheduler_gpu_authoritative(world):
        return False
    if stage in {"material_material", "material_gas"}:
        return True
    return stage in {"material_light", "gas_light"} and solver.gpu_pipeline._formal_light_dose_guard_buffer(world) is not None


def _full_gpu_authoritative_solve_masks(
    solver,
    world: "WorldEngine",
) -> tuple[GPUAuthoritativeFullSolveMask, GPUAuthoritativeFullSolveMask, GPUAuthoritativeFullSolveMask]:
    tile_shape = (world.active.tile_height, world.active.tile_width)
    cell_shape = (world.height, world.width)
    gas_shape = (world.gas_height, world.gas_width)
    signature = (tile_shape, cell_shape, gas_shape)
    if (
        solver._full_gpu_authoritative_mask_cache_signature != signature
        or solver._full_gpu_authoritative_mask_cache is None
    ):
        solver._full_gpu_authoritative_mask_cache_signature = signature
        solver._full_gpu_authoritative_mask_cache = (
            GPUAuthoritativeFullSolveMask("tile", tile_shape),
            GPUAuthoritativeFullSolveMask("cell", cell_shape),
            GPUAuthoritativeFullSolveMask("gas", gas_shape),
        )
    return solver._full_gpu_authoritative_mask_cache


def _full_solve_masks(solver, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile_shape = (world.active.tile_height, world.active.tile_width)
    cell_shape = (world.height, world.width)
    gas_shape = (world.gas_height, world.gas_width)
    signature = (tile_shape, cell_shape, gas_shape)
    if solver._full_solve_mask_cache_signature != signature or solver._full_solve_mask_cache is None:
        solver._full_solve_mask_cache_signature = signature
        solver._full_solve_mask_cache = (
            np.ones(tile_shape, dtype=np.bool_),
            np.ones(cell_shape, dtype=np.bool_),
            np.ones(gas_shape, dtype=np.bool_),
        )
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._full_solve_mask_cache
    return solve_tile_mask.copy(), solve_cell_mask.copy(), solve_gas_mask.copy()


def _is_full_gpu_authoritative_mask(mask: object) -> bool:
    return bool(getattr(mask, "full_gpu_authoritative", False))


def _all_full_gpu_authoritative_masks(*masks: object) -> bool:
    return all(_is_full_gpu_authoritative_mask(mask) for mask in masks)


def _solve_mask_any(mask: object) -> bool:
    if _is_full_gpu_authoritative_mask(mask):
        return True
    return bool(np.any(mask))


def _require_materialized_cpu_solve_masks(
    solver,
    world: "WorldEngine",
    stage_name: str,
    *solve_masks: object,
) -> None:
    if not any(solver._is_full_gpu_authoritative_mask(mask) for mask in solve_masks):
        return
    world._require_gpu_stage(stage_name)
    raise RuntimeError(
        f"GPU-authoritative reaction solve masks require GPU support for {stage_name}; CPU fallback is disabled"
    )


def _formal_gpu_frame(solver, world: "WorldEngine") -> bool:
    return (
        getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )


def _active_scheduler_gpu_authoritative(solver, world: "WorldEngine") -> bool:
    return (
        solver._formal_gpu_frame(world)
        and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
    )


def _solve_tile_mask(solver, world: "WorldEngine", *, seed_timer_cells: bool) -> np.ndarray:
    active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
    seeded_tiles = active_tiles.copy()
    if seed_timer_cells:
        tile_size = world.active.tile_size
        timer_mask = np.any(world.timer_pack > 0, axis=-1)
        for y, x in np.argwhere(timer_mask):
            tile_x = min(world.active.tile_width - 1, int(x) // tile_size)
            tile_y = min(world.active.tile_height - 1, int(y) // tile_size)
            seeded_tiles[tile_y, tile_x] = True
    return expand_bool_mask(seeded_tiles, radius=1)


def _capture_activity_state(
    solver,
    world: "WorldEngine",
    solve_cell_mask,
    solve_gas_mask,
) -> dict[str, object]:
    if solver._formal_gpu_frame(world):
        if solver._all_full_gpu_authoritative_masks(solve_cell_mask, solve_gas_mask):
            solver._stage_extra_changed_cell_mask = None
        else:
            solver._stage_extra_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        return {
            "formal_gpu_frame": True,
            "full_gpu_authoritative": solver._all_full_gpu_authoritative_masks(solve_cell_mask, solve_gas_mask),
            "emitters": len(world.emitters),
        }
    solver._stage_extra_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    return {
        "formal_gpu_frame": False,
        "material_id": world.material_id[solve_cell_mask].copy(),
        "phase": world.phase[solve_cell_mask].copy(),
        "cell_temperature": world.cell_temperature[solve_cell_mask].copy(),
        "integrity": world.integrity[solve_cell_mask].copy(),
        "timer_pack": world.timer_pack[solve_cell_mask].copy(),
        "gas_concentration": world.gas_concentration[:, solve_gas_mask].copy(),
        "ambient_temperature": world.ambient_temperature[solve_gas_mask].copy(),
        "emitters": len(world.emitters),
    }


def _refresh_active_regions(
    solver,
    world: "WorldEngine",
    solve_tile_mask: np.ndarray,
    changed_cell_mask: np.ndarray,
    changed_gas_mask: np.ndarray,
    ambient_changed_mask: np.ndarray,
    timer_changed_mask: np.ndarray,
    *,
    source_emitted: bool,
) -> None:
    any_cell_changed = bool(np.any(changed_cell_mask))
    any_gas_changed = bool(np.any(changed_gas_mask))
    any_ambient_changed = bool(np.any(ambient_changed_mask))
    any_timer_changed = bool(np.any(timer_changed_mask))
    if not (
        any_cell_changed
        or any_gas_changed
        or any_ambient_changed
        or any_timer_changed
        or source_emitted
    ):
        return
    if np.any(solve_tile_mask):
        solver._mark_tiles_from_mask(world, solve_tile_mask)
    if any_cell_changed or any_timer_changed:
        solver._mark_tiles_from_cell_mask(world, changed_cell_mask | timer_changed_mask, tile_padding=1)
    if any_gas_changed or any_ambient_changed:
        solver._mark_tiles_from_gas_mask(world, changed_gas_mask | ambient_changed_mask, tile_padding=1)


def _ensure_runtime_state(solver, world: "WorldEngine") -> None:
    world.bridge.sync_rule_tables(world)
    if solver.last_solve_cell_mask.shape != (world.height, world.width):
        solver.reset_runtime_state(world)


def _record_stage_solve_masks(
    solver,
    stage: str,
    solve_tile_mask,
    solve_cell_mask,
    solve_gas_mask,
) -> None:
    if solver._all_full_gpu_authoritative_masks(solve_tile_mask, solve_cell_mask, solve_gas_mask):
        solver.last_full_gpu_authoritative_solve_stages.add(stage)
        solver.last_stage_solve_modes[stage] = "full_gpu_authoritative"
        return
    solver.last_stage_solve_modes[stage] = "materialized"
    solver.last_stage_tile_masks[stage] = np.asarray(solve_tile_mask, dtype=np.bool_).copy()
    solver.last_solve_cell_mask |= np.asarray(solve_cell_mask, dtype=np.bool_)
    solver.last_solve_gas_mask |= np.asarray(solve_gas_mask, dtype=np.bool_)


def _note_runtime_backend(solver, backend: str) -> None:
    if backend == "gpu":
        solver._runtime_used_gpu = True
    else:
        solver._runtime_used_cpu = True
    solver.last_runtime_backend = solver._current_runtime_backend()


def _current_runtime_backend(solver) -> str:
    if solver._runtime_used_cpu and solver._runtime_used_gpu:
        return "hybrid"
    if solver._runtime_used_gpu:
        return "gpu"
    return "cpu"


def _finalize_stage_runtime(
    solver,
    world: "WorldEngine",
    solve_tile_mask,
    solve_cell_mask,
    solve_gas_mask,
    previous_state: dict[str, object],
) -> None:
    if bool(previous_state.get("formal_gpu_frame", False)):
        if solver._all_full_gpu_authoritative_masks(solve_tile_mask, solve_cell_mask, solve_gas_mask):
            if solver._current_stage is not None:
                solver.last_full_gpu_authoritative_changed_stages.add(solver._current_stage)
            if solver._stage_extra_changed_cell_mask is not None and np.any(solver._stage_extra_changed_cell_mask):
                solver.last_changed_cell_mask |= solver._stage_extra_changed_cell_mask
            solver._current_stage = None
            if not solver._active_scheduler_gpu_authoritative(world):
                world._require_gpu_stage("active scheduler reaction refresh")
            return
        stage_extra_changed_cell_mask = solver._stage_extra_changed_cell_mask
        if stage_extra_changed_cell_mask is None:
            stage_extra_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        solver.last_changed_cell_mask |= solve_cell_mask | stage_extra_changed_cell_mask
        solver.last_changed_gas_mask |= solve_gas_mask
        solver.last_ambient_changed_mask |= solve_gas_mask
        solver.last_timer_changed_mask |= solve_cell_mask
        solver._current_stage = None
        if not solver._active_scheduler_gpu_authoritative(world):
            world._require_gpu_stage("active scheduler reaction refresh")
        return
    material_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    phase_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    timer_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    cell_temperature_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    integrity_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
    if np.any(solve_cell_mask):
        material_changed_mask[solve_cell_mask] = world.material_id[solve_cell_mask] != previous_state["material_id"]
        phase_changed_mask[solve_cell_mask] = world.phase[solve_cell_mask] != previous_state["phase"]
        timer_changed_mask[solve_cell_mask] = np.any(
            world.timer_pack[solve_cell_mask] != previous_state["timer_pack"],
            axis=-1,
        )
        cell_temperature_changed_mask[solve_cell_mask] = (
            np.abs(world.cell_temperature[solve_cell_mask] - previous_state["cell_temperature"]) > REACTION_ACTIVITY_EPSILON
        )
        integrity_changed_mask[solve_cell_mask] = (
            np.abs(world.integrity[solve_cell_mask] - previous_state["integrity"]) > REACTION_ACTIVITY_EPSILON
        )
    gas_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
    ambient_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
    if np.any(solve_gas_mask):
        gas_changed_mask[solve_gas_mask] = np.any(
            np.abs(world.gas_concentration[:, solve_gas_mask] - previous_state["gas_concentration"]) > REACTION_ACTIVITY_EPSILON,
            axis=0,
        )
        ambient_changed_mask[solve_gas_mask] = (
            np.abs(world.ambient_temperature[solve_gas_mask] - previous_state["ambient_temperature"]) > REACTION_ACTIVITY_EPSILON
        )
    if solver._stage_extra_changed_cell_mask is not None:
        material_changed_mask |= solver._stage_extra_changed_cell_mask
    solver.last_changed_cell_mask |= (
        material_changed_mask
        | phase_changed_mask
        | timer_changed_mask
        | cell_temperature_changed_mask
        | integrity_changed_mask
    )
    solver.last_changed_gas_mask |= gas_changed_mask
    solver.last_ambient_changed_mask |= ambient_changed_mask
    solver.last_timer_changed_mask |= timer_changed_mask
    source_emitted = len(world.emitters) > int(previous_state["emitters"])
    solver._current_stage = None
    if not solver._active_scheduler_gpu_authoritative(world):
        solver._refresh_active_regions(
            world,
            solve_tile_mask,
            material_changed_mask | phase_changed_mask | cell_temperature_changed_mask | integrity_changed_mask,
            gas_changed_mask,
            ambient_changed_mask,
            timer_changed_mask,
            source_emitted=source_emitted,
        )


def _mark_tiles_from_mask(solver, world: "WorldEngine", solve_tile_mask: np.ndarray) -> None:
    tile_size = world.active.tile_size
    rects: list[tuple[int, int, int, int]] = []
    for tile_y, tile_x in np.argwhere(solve_tile_mask):
        x0 = int(tile_x) * tile_size
        y0 = int(tile_y) * tile_size
        rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size)))
    world._mark_active_rects_runtime(rects)


def _mark_tiles_from_cell_mask(
    solver,
    world: "WorldEngine",
    cell_mask: np.ndarray,
    *,
    tile_padding: int = 0,
) -> None:
    tile_size = world.active.tile_size
    rects: list[tuple[int, int, int, int, int]] = []
    for tile_y, tile_x in {
        (int(y) // tile_size, int(x) // tile_size)
        for y, x in np.argwhere(cell_mask)
    }:
        x0 = tile_x * tile_size
        y0 = tile_y * tile_size
        rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size), tile_padding))
    world._mark_active_rects_runtime(rects)


def _mark_tiles_from_gas_mask(
    solver,
    world: "WorldEngine",
    gas_mask: np.ndarray,
    *,
    tile_padding: int = 0,
) -> None:
    gas_cell_size = world.gas_cell_size
    rects: list[tuple[int, int, int, int, int]] = []
    for gy, gx in np.argwhere(gas_mask):
        x0 = int(gx) * gas_cell_size
        y0 = int(gy) * gas_cell_size
        rects.append((x0, y0, min(world.width, x0 + gas_cell_size), min(world.height, y0 + gas_cell_size), tile_padding))
    world._mark_active_rects_runtime(rects)
