from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask
from oracle_game.types import Phase


def step(solver, world: "WorldEngine", dt: float) -> None:
    solver.reset_runtime_state()
    solver.gpu_pipeline.reset_pass_profile()
    gpu_available = world._gpu_pipeline_available(solver.gpu_pipeline, "motion")
    formal_gpu_frame = (
        gpu_available
        and getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )
    active_scheduler_gpu_authoritative = (
        formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
    )
    if formal_gpu_frame and not active_scheduler_gpu_authoritative:
        world._require_gpu_stage("active scheduler motion solve masks")
    if active_scheduler_gpu_authoritative:
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
    else:
        solve_tile_mask = solver._solve_tile_mask(world)
    if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
        solver.last_backend = "idle"
        return
    solve_cell_mask = tile_mask_to_cell_mask(
        solve_tile_mask,
        tile_size=world.active.tile_size,
        width=world.width,
        height=world.height,
    )
    used_gpu = False
    used_cpu = False
    if gpu_available:
        solver.gpu_pipeline.integrate_velocity(world, dt, solve_tile_mask=solve_tile_mask)
        used_gpu = True
    else:
        world._require_cpu_oracle_backend("motion velocity")
        solver._integrate_velocity(world, dt, solve_cell_mask)
        used_cpu = True
    island_cpu_work = bool(world.islands and not gpu_available)
    island_used_gpu = solver._move_falling_islands(world, dt=dt, use_gpu=gpu_available)
    used_gpu = used_gpu or island_used_gpu
    used_cpu = used_cpu or island_cpu_work
    powder_active = bool(gpu_available) or bool(
        np.any(solve_cell_mask & (world.material_id != 0) & (world.phase == int(Phase.POWDER)))
    )
    if powder_active:
        if gpu_available:
            powder_reservations = solver.gpu_pipeline.resolve_and_apply_powders(
                world,
                dt,
                solve_tile_mask=solve_tile_mask,
            )
            used_gpu = True
        else:
            world._require_cpu_oracle_backend("motion powder")
            powder_reservations = solver._plan_cpu_powder_reservations(world, solve_cell_mask, dt)
            powder_reservations = solver._resolve_powder_reservations(world, powder_reservations)
            solver.gpu_pipeline.upload_powder_reservations(world, powder_reservations)
            solver._apply_powder_reservations(world, powder_reservations)
            used_cpu = True
        solver.last_powder_reservations = powder_reservations.copy()
        if gpu_available and not active_scheduler_gpu_authoritative:
            solver._mark_powder_reservation_regions(world, powder_reservations)
    if used_gpu and used_cpu:
        solver.last_backend = "hybrid"
    elif used_gpu:
        solver.last_backend = "gpu"
    else:
        solver.last_backend = "cpu"
    solver.last_public_powder_reservations = solver._capture_public_powder_reservations(world, solver.last_powder_reservations)
    solver.last_public_island_reservations = solver._capture_public_island_reservations(world, solver.last_island_reservations)



def _solve_tile_mask(solver, world: "WorldEngine") -> np.ndarray:
    formal_gpu_frame = (
        getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
        and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
    )
    if formal_gpu_frame and world.bridge.enabled and world.bridge.ctx is not None and "active_tile_ttl" in world.bridge.buffers:
        active_tiles = (
            np.frombuffer(
                world.bridge.buffers["active_tile_ttl"].read(
                    size=world.active.tile_width * world.active.tile_height * 4,
                ),
                dtype=np.int32,
            ).reshape((world.active.tile_height, world.active.tile_width))
            > 0
        )
    else:
        active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
    seeded_tiles = active_tiles.copy()
    tile_size = world.active.tile_size
    for record in world.islands.values():
        x0, y0, x1, y1 = record.bbox
        tile_x0 = max(0, x0 // tile_size)
        tile_y0 = max(0, y0 // tile_size)
        tile_x1 = min(world.active.tile_width, (x1 + tile_size - 1) // tile_size)
        tile_y1 = min(world.active.tile_height, (y1 + tile_size - 1) // tile_size)
        seeded_tiles[tile_y0:tile_y1, tile_x0:tile_x1] = True
    return expand_bool_mask(seeded_tiles, radius=1)



def _integrate_velocity(solver, world: "WorldEngine", dt: float, solve_cell_mask: np.ndarray) -> None:
    non_empty = (world.material_id != 0) & solve_cell_mask
    if not non_empty.any():
        return
    gravity = solver._material_scalar_field(world, world.material_id, "gravity_scale", world.material_gravity) * dt * 24.0
    world.velocity[..., 1][non_empty] += gravity[non_empty]
    cell_flow = world.sample_flow_to_cells()
    wind_delta = (
        cell_flow
        * solver._material_scalar_field(world, world.material_id, "wind_coupling", world.material_wind)[..., None]
        * dt
        * 4.0
    )
    world.velocity[non_empty] += wind_delta[non_empty]
    drag = np.maximum(
        0.0,
        1.0 - solver._material_scalar_field(world, world.material_id, "drag_scale", world.material_drag)[..., None] * dt,
    )
    world.velocity[non_empty] *= drag[non_empty]

