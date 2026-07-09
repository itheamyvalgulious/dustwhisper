from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_motion import (
    POWDER_RESOLVE_BLOCKED,
    POWDER_RESOLVE_DDA,
    POWDER_RESOLVE_FALLBACK,
    POWDER_RESOLVE_STALE,
    powder_reservation_dtype,
)
from oracle_game.sim.motion import POWDER_SOLVER_SUSPENDED
from oracle_game.types import Phase


def _mark_powder_reservation_regions(solver, world: "WorldEngine", powder_reservations: np.ndarray) -> None:
    for reservation in powder_reservations:
        resolve_state = int(reservation["resolve_state"])
        if resolve_state == POWDER_RESOLVE_STALE:
            continue
        sx = int(reservation["source_xy"][0])
        sy = int(reservation["source_xy"][1])
        tx = int(reservation["resolved_target_xy"][0])
        ty = int(reservation["resolved_target_xy"][1])
        world._mark_active_rect_runtime(
            max(0, min(sx, tx) - 1),
            max(0, min(sy, ty) - 1),
            min(world.width, max(sx, tx) + 2),
            min(world.height, max(sy, ty) + 2),
        )



def _plan_cpu_powder_reservations(
    solver,
    world: "WorldEngine",
    solve_cell_mask: np.ndarray,
    dt: float,
) -> np.ndarray:
    reservations: list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[float, float],
            int,
            int,
        ]
    ] = []
    for y in range(world.height - 2, -1, -1):
        active_xs = np.flatnonzero(solve_cell_mask[y])
        if active_xs.size == 0:
            continue
        for x in active_xs.tolist():
            material_id = int(world.material_id[y, x])
            if material_id == 0 or int(world.phase[y, x]) != int(Phase.POWDER):
                continue
            velocity = world.velocity[y, x]
            max_dda_step = solver._material_max_dda_step(world, material_id)
            frame_delta_x = float(velocity[0]) * float(dt)
            frame_delta_y = float(velocity[1]) * float(dt)
            desired_dx = int(np.clip(np.rint(frame_delta_x), -max_dda_step, max_dda_step))
            desired_dy = int(np.clip(np.rint(frame_delta_y), -max_dda_step, max_dda_step))
            reserved_target = solver._resolve_powder_dda_target(world, x, y, max_dda_step, dt)
            if reserved_target is None:
                reserved_target = (x, y)
            reservations.append(
                (
                    (int(x), int(y)),
                    (int(x) + desired_dx, int(y) + desired_dy),
                    (int(reserved_target[0]), int(reserved_target[1])),
                    (int(x), int(y)),
                    (float(velocity[0]), float(velocity[1])),
                    material_id,
                    POWDER_RESOLVE_BLOCKED,
                )
            )
    packed = np.zeros((len(reservations),), dtype=powder_reservation_dtype())
    for index, (
        source_xy,
        desired_target_xy,
        reserved_target_xy,
        resolved_target_xy,
        velocity_xy,
        material_id,
        resolve_state,
    ) in enumerate(reservations):
        packed[index]["source_xy"] = np.asarray(source_xy, dtype=np.int32)
        packed[index]["desired_target_xy"] = np.asarray(desired_target_xy, dtype=np.int32)
        packed[index]["reserved_target_xy"] = np.asarray(reserved_target_xy, dtype=np.int32)
        packed[index]["resolved_target_xy"] = np.asarray(resolved_target_xy, dtype=np.int32)
        packed[index]["velocity_xy"] = np.asarray(velocity_xy, dtype=np.float32)
        packed[index]["material_id"] = int(material_id)
        packed[index]["resolve_state"] = int(resolve_state)
    return packed



def _move_powders(solver, world: "WorldEngine", solve_cell_mask: np.ndarray, dt: float) -> None:
    processed = np.zeros((world.height, world.width), dtype=bool)
    for y in range(world.height - 2, -1, -1):
        active_xs = np.flatnonzero(solve_cell_mask[y])
        if active_xs.size == 0:
            continue
        for x in active_xs.tolist():
            if processed[y, x]:
                continue
            material_id = int(world.material_id[y, x])
            if material_id == 0:
                continue
            if int(world.phase[y, x]) != int(Phase.POWDER):
                continue
            dda_target = solver._resolve_powder_dda_target(
                world,
                x,
                y,
                solver._material_max_dda_step(world, material_id),
                dt,
            )
            moved = False
            if dda_target is not None and dda_target != (x, y):
                world.swap_cells(x, y, dda_target[0], dda_target[1])
                processed[dda_target[1], dda_target[0]] = True
                moved = True
            if moved:
                continue
            candidates = (
                [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]
                if solver._material_gravity(world, material_id) >= 0.0
                else [(x, y - 1), (x - 1, y - 1), (x + 1, y - 1)]
            )
            for tx, ty in candidates:
                if world.in_bounds(tx, ty) and world.material_id[ty, tx] == 0:
                    world.swap_cells(x, y, tx, ty)
                    processed[ty, tx] = True
                    moved = True
                    break
            if not moved:
                world.velocity[y, x] *= 0.2
                processed[y, x] = True



def _apply_powder_reservations(
    solver,
    world: "WorldEngine",
    powder_reservations: np.ndarray,
) -> None:
    for reservation in powder_reservations:
        x = int(reservation["source_xy"][0])
        y = int(reservation["source_xy"][1])
        resolve_state = int(reservation["resolve_state"])
        if resolve_state == POWDER_RESOLVE_STALE:
            continue
        material_id = int(world.material_id[y, x])
        if material_id == 0 or material_id != int(reservation["material_id"]):
            continue
        if int(world.phase[y, x]) != int(Phase.POWDER):
            continue
        target_x = int(reservation["resolved_target_xy"][0])
        target_y = int(reservation["resolved_target_xy"][1])
        if resolve_state in {POWDER_RESOLVE_DDA, POWDER_RESOLVE_FALLBACK} and (target_x != x or target_y != y):
            world.swap_cells(x, y, target_x, target_y)
            material_id = int(world.material_id[target_y, target_x])
            desired_dx = int(reservation["desired_target_xy"][0]) - x
            desired_dy = int(reservation["desired_target_xy"][1]) - y
            actual_dx = target_x - x
            actual_dy = target_y - y
            if material_id != 0 and (desired_dx != actual_dx or desired_dy != actual_dy):
                world.velocity[target_y, target_x] = solver._collision_response(
                    world.velocity[target_y, target_x],
                    (desired_dx, desired_dy),
                    (actual_dx, actual_dy),
                    friction=solver._material_friction(world, material_id),
                    elasticity=solver._material_elasticity(world, material_id),
                )
            continue
        desired_dx = int(reservation["desired_target_xy"][0]) - x
        desired_dy = int(reservation["desired_target_xy"][1]) - y
        if desired_dx != 0 or desired_dy != 0:
            world.velocity[y, x] = solver._collision_response(
                world.velocity[y, x],
                (desired_dx, desired_dy),
                (0, 0),
                friction=solver._material_friction(world, material_id),
                elasticity=solver._material_elasticity(world, material_id),
            )
        else:
            world.velocity[y, x] *= 0.2



def _resolve_powder_reservations(
    solver,
    world: "WorldEngine",
    powder_reservations: np.ndarray,
) -> np.ndarray:
    if len(powder_reservations) == 0:
        return powder_reservations
    resolved = powder_reservations.copy()
    shadow_material = world.material_id.copy()
    for index, reservation in enumerate(resolved):
        x = int(reservation["source_xy"][0])
        y = int(reservation["source_xy"][1])
        material_id = int(reservation["material_id"])
        resolved[index]["resolved_target_xy"] = np.asarray((x, y), dtype=np.int32)
        resolved[index]["resolve_state"] = POWDER_RESOLVE_BLOCKED
        if not world.in_bounds(x, y) or int(shadow_material[y, x]) != material_id:
            resolved[index]["resolve_state"] = POWDER_RESOLVE_STALE
            continue
        target_x = int(reservation["reserved_target_xy"][0])
        target_y = int(reservation["reserved_target_xy"][1])
        if (
            (target_x != x or target_y != y)
            and world.in_bounds(target_x, target_y)
            and solver._path_is_clear_material(shadow_material, x, y, target_x, target_y)
        ):
            shadow_material[y, x] = 0
            shadow_material[target_y, target_x] = material_id
            resolved[index]["resolved_target_xy"] = np.asarray((target_x, target_y), dtype=np.int32)
            resolved[index]["resolve_state"] = POWDER_RESOLVE_DDA
            continue
        candidates = solver._powder_fallback_candidates(world, x, y, material_id)
        for fallback_x, fallback_y in candidates:
            if not world.in_bounds(fallback_x, fallback_y) or int(shadow_material[fallback_y, fallback_x]) != 0:
                continue
            shadow_material[y, x] = 0
            shadow_material[fallback_y, fallback_x] = material_id
            resolved[index]["resolved_target_xy"] = np.asarray((fallback_x, fallback_y), dtype=np.int32)
            resolved[index]["resolve_state"] = POWDER_RESOLVE_FALLBACK
            break
    return resolved



def _powder_fallback_candidates(
    solver,
    world: "WorldEngine",
    x: int,
    y: int,
    material_id: int,
) -> list[tuple[int, int]]:
    if material_id > 0 and solver._material_powder_solver_kind(world, material_id) == POWDER_SOLVER_SUSPENDED:
        return []
    if material_id > 0 and solver._material_gravity(world, material_id) >= 0.0:
        return [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]
    return [(x, y - 1), (x - 1, y - 1), (x + 1, y - 1)]



def _resolve_powder_dda_target(
    solver,
    world: "WorldEngine",
    x: int,
    y: int,
    max_dda_step: int,
    dt: float,
) -> tuple[int, int] | None:
    velocity = world.velocity[y, x]
    max_step = max(0, int(max_dda_step))
    if max_step <= 0:
        return None
    frame_delta_x = float(velocity[0]) * float(dt)
    frame_delta_y = float(velocity[1]) * float(dt)
    desired_dx = int(np.clip(np.rint(frame_delta_x), -max_step, max_step))
    desired_dy = int(np.clip(np.rint(frame_delta_y), -max_step, max_step))
    if desired_dx == 0 and desired_dy == 0:
        return None
    target_x = x + desired_dx
    target_y = y + desired_dy
    furthest_free = (x, y)
    for cell_x, cell_y in solver._dda_line_cells(x, y, target_x, target_y):
        if not world.in_bounds(cell_x, cell_y):
            break
        if world.material_id[cell_y, cell_x] != 0:
            break
        furthest_free = (cell_x, cell_y)
    return furthest_free



def _path_is_clear(solver, world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> bool:
    for cell_x, cell_y in solver._dda_line_cells(x0, y0, x1, y1):
        if not world.in_bounds(cell_x, cell_y):
            return False
        if world.material_id[cell_y, cell_x] != 0:
            return False
    return True



def _path_is_clear_material(solver, material_id: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    height, width = material_id.shape
    for cell_x, cell_y in solver._dda_line_cells(x0, y0, x1, y1):
        if cell_x < 0 or cell_y < 0 or cell_x >= width or cell_y >= height:
            return False
        if material_id[cell_y, cell_x] != 0:
            return False
    return True

