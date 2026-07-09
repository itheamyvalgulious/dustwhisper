from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.cpu_base import material_table_row


def _collision_response(
    solver,
    velocity_xy: tuple[float, float] | np.ndarray,
    attempted_delta: tuple[int, int],
    actual_delta: tuple[int, int],
    *,
    friction: float,
    elasticity: float,
) -> tuple[float, float]:
    vx = float(velocity_xy[0])
    vy = float(velocity_xy[1])
    tangential_scale = max(0.0, 1.0 - float(np.clip(friction, 0.0, 1.0)))
    bounce = max(0.0, float(elasticity))
    if attempted_delta[0] != actual_delta[0] and attempted_delta[0] != 0:
        normal_vx = vx if abs(vx) > 1.0e-6 else float(attempted_delta[0])
        vx = -normal_vx * bounce
        vy *= tangential_scale
    if attempted_delta[1] != actual_delta[1] and attempted_delta[1] != 0:
        normal_vy = vy if abs(vy) > 1.0e-6 else float(attempted_delta[1])
        vy = -normal_vy * bounce
        vx *= tangential_scale
    if abs(vx) < 1.0e-6:
        vx = 0.0
    if abs(vy) < 1.0e-6:
        vy = 0.0
    return (vx, vy)



def _material_table_row(solver, world: "WorldEngine", material_id: int) -> np.void | None:
    # Delegated to the shared helper (formerly duplicated verbatim here).
    return material_table_row(world, material_id)



def _material_scalar(solver, world: "WorldEngine", material_id: int, field: str, fallback: np.ndarray) -> float:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return float(row[field])
    if 0 <= material_id < fallback.shape[0]:
        return float(fallback[material_id])
    return 0.0



def _material_int(solver, world: "WorldEngine", material_id: int, field: str, fallback: np.ndarray) -> int:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return int(row[field])
    if 0 <= material_id < fallback.shape[0]:
        return int(fallback[material_id])
    return 0



def _material_scalar_field(
    solver,
    world: "WorldEngine",
    material_ids: np.ndarray,
    field: str,
    fallback: np.ndarray,
) -> np.ndarray:
    values = fallback[material_ids].astype(np.float32, copy=True)
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is None:
        return values
    valid_mask = (
        (material_ids >= 0)
        & (material_ids < int(material_table.shape[0]))
        & (material_table["name_hash"][np.clip(material_ids, 0, max(0, int(material_table.shape[0]) - 1))] != 0)
    )
    if np.any(valid_mask):
        values[valid_mask] = material_table[field][material_ids[valid_mask]].astype(np.float32, copy=False)
    return values



def _material_default_phase(solver, world: "WorldEngine", material_id: int) -> int | None:
    row = solver._material_table_row(world, material_id)
    if row is not None:
        return int(row["default_phase"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return int(shadow_material.default_phase)
    if world._shadow_has_table_payload("materials"):
        return None
    if 0 <= material_id < world.material_default_phase.shape[0]:
        return int(world.material_default_phase[material_id])
    return None



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



def _material_powder_generation_id(solver, world: "WorldEngine", material_id: int) -> int:
    if material_id <= 0:
        return 0
    return solver._material_int(world, material_id, "powder_generation_id", world.material_powder_generation_id)



def _material_falling_island_break_kind(solver, world: "WorldEngine", material_id: int) -> int:
    return solver._material_int(
        world,
        material_id,
        "falling_island_break_kind_id",
        world.material_falling_island_break_kind,
    )



def _material_max_dda_step(solver, world: "WorldEngine", material_id: int) -> int:
    return max(0, solver._material_int(world, material_id, "max_dda_step", world.material_max_dda_step))



def _material_powder_solver_kind(solver, world: "WorldEngine", material_id: int) -> int:
    return solver._material_int(world, material_id, "powder_solver_kind_id", world.material_powder_solver_kind)



def _material_gravity(solver, world: "WorldEngine", material_id: int) -> float:
    return solver._material_scalar(world, material_id, "gravity_scale", world.material_gravity)



def _material_friction(solver, world: "WorldEngine", material_id: int) -> float:
    return solver._material_scalar(world, material_id, "friction", world.material_friction)



def _material_elasticity(solver, world: "WorldEngine", material_id: int) -> float:
    return solver._material_scalar(world, material_id, "elasticity", world.material_elasticity)



def _dda_line_cells(solver, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
        cells.append((x0, y0))
    return cells
