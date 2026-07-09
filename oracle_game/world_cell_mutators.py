from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from oracle_game.types import MaterialDef, Phase

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def material_by_id(engine: "WorldEngine", material_id: int) -> MaterialDef:
    material = engine._shadow_material_def(int(material_id))
    if material is None:
        raise KeyError(material_id)
    return material


def allocate_island_id(engine: "WorldEngine") -> int:
    island_id = max(1, int(engine.next_island_id))
    while island_id in engine.islands:
        island_id += 1
    engine.next_island_id = island_id + 1
    return island_id


def in_bounds(engine: "WorldEngine", x: int, y: int) -> bool:
    return 0 <= x < engine.width and 0 <= y < engine.height


def cell_xy_to_gas(engine: "WorldEngine", x: int, y: int) -> tuple[int, int]:
    """Map a cell-space (x, y) pair onto the lower-resolution gas grid."""
    return (
        min(engine.gas_height - 1, max(0, y // engine.gas_cell_size)),
        min(engine.gas_width - 1, max(0, x // engine.gas_cell_size)),
    )


def sample_ambient_to_cells(engine: "WorldEngine") -> np.ndarray:
    return np.repeat(np.repeat(engine.ambient_temperature, engine.gas_cell_size, axis=0), engine.gas_cell_size, axis=1)[: engine.height, : engine.width]


def sample_flow_to_cells(engine: "WorldEngine") -> np.ndarray:
    return np.repeat(np.repeat(engine.flow_velocity, engine.gas_cell_size, axis=0), engine.gas_cell_size, axis=1)[: engine.height, : engine.width]


def add_gas_from_cells(engine: "WorldEngine", mask: np.ndarray, species: str, amount: float) -> None:
    species_id = engine._resolve_sanctioned_gas_id(species)
    if species_id < 0:
        raise KeyError(species)
    engine._invalidate_gpu_authoritative_resources("gas_concentration")
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        gy, gx = engine.cell_to_gas(y, x)
        engine.gas_concentration[species_id, gy, gx] += amount


def set_cell_by_id(engine: "WorldEngine", x: int, y: int, material_id: int, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
    engine._invalidate_gpu_authoritative_cell_resources()
    previous_material = int(engine.material_id[y, x])
    previous_phase = int(engine.phase[y, x])
    previous_island_id = int(engine.island_id[y, x])
    previous_displaced = int(engine.placeholder_displaced_material[y, x])
    engine.material_id[y, x] = int(material_id)
    if phase is not None:
        resolved_phase = int(phase)
    else:
        shadow_phase = engine._shadow_material_default_phase(int(material_id))
        resolved_phase = int(shadow_phase) if shadow_phase is not None else 0
    engine.phase[y, x] = resolved_phase
    engine.cell_flags[y, x] = 0
    engine.timer_pack[y, x] = 0
    shadow_integrity = engine._shadow_material_base_integrity(int(material_id))
    engine.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
    spawn_temperature = engine._shadow_material_spawn_temperature(int(material_id))
    if spawn_temperature is not None:
        engine.cell_temperature[y, x] = max(float(engine.cell_temperature[y, x]), spawn_temperature)
    current_is_placeholder = engine._shadow_material_is_placeholder(int(material_id))
    previous_is_placeholder = engine._shadow_material_is_placeholder(previous_material)
    if current_is_placeholder:
        if previous_is_placeholder:
            engine.placeholder_displaced_material[y, x] = previous_displaced
        elif previous_material != 0 and previous_phase == int(Phase.LIQUID):
            engine.placeholder_displaced_material[y, x] = previous_material
        else:
            engine.placeholder_displaced_material[y, x] = 0
    else:
        engine.entity_id[y, x] = 0
        engine.placeholder_displaced_material[y, x] = 0
    current_displaced = int(engine.placeholder_displaced_material[y, x])
    if current_is_placeholder or previous_is_placeholder or previous_displaced != current_displaced:
        engine._pending_placeholder_dirty_rects.append((int(x), int(y), int(x) + 1, int(y) + 1))
    engine.island_id[y, x] = 0 if engine.phase[y, x] != int(Phase.FALLING_ISLAND) else engine.island_id[y, x]
    engine._refresh_island_records_for_ids((previous_island_id, int(engine.island_id[y, x])))
    engine._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(engine.width, x + 2), min(engine.height, y + 2))
    if mark_dirty and (
        engine._cell_participates_in_collapse(previous_material, previous_phase)
        or engine._cell_participates_in_collapse(int(engine.material_id[y, x]), int(engine.phase[y, x]))
    ):
        engine._mark_collapse_dirty_rect(x, y, x + 1, y + 1)


def _inject_velocity_immediate(
    engine: "WorldEngine",
    x: int,
    y: int,
    velocity: tuple[float, float],
    radius: int,
    *,
    carrier: str,
    mode: str,
) -> None:
    velocity_vec = np.asarray(velocity, dtype=np.float32)
    if mode not in {"add", "set"}:
        raise ValueError(f"unsupported velocity mode: {mode}")
    if carrier not in {"cell", "flow", "both"}:
        raise ValueError(f"unsupported velocity carrier: {carrier}")
    if carrier in {"cell", "both"}:
        engine._invalidate_gpu_authoritative_resources("cell_core")
        yy, xx = np.mgrid[0 : engine.height, 0 : engine.width]
        cell_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        if mode == "set":
            engine.velocity[cell_mask] = velocity_vec
        else:
            engine.velocity[cell_mask] += velocity_vec
    if carrier in {"flow", "both"}:
        engine._invalidate_gpu_authoritative_resources("flow_velocity")
        gas_center_x = min(engine.gas_width - 1, max(0, x // engine.gas_cell_size))
        gas_center_y = min(engine.gas_height - 1, max(0, y // engine.gas_cell_size))
        gas_radius = max(0, (radius + engine.gas_cell_size - 1) // engine.gas_cell_size)
        yy, xx = np.mgrid[0 : engine.gas_height, 0 : engine.gas_width]
        flow_mask = (xx - gas_center_x) ** 2 + (yy - gas_center_y) ** 2 <= gas_radius ** 2
        if mode == "set":
            engine.flow_velocity[flow_mask] = velocity_vec
        else:
            engine.flow_velocity[flow_mask] += velocity_vec
    engine._mark_active_rect_runtime(
        max(0, x - radius),
        max(0, y - radius),
        min(engine.width, x + radius + 1),
        min(engine.height, y + radius + 1),
    )


def _inject_temperature_immediate(engine: "WorldEngine", x: int, y: int, delta: float, radius: int) -> None:
    engine._invalidate_gpu_authoritative_resources("cell_core")
    yy, xx = np.mgrid[0 : engine.height, 0 : engine.width]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    engine.cell_temperature[mask] += delta
    engine._mark_active_rect_runtime(max(0, x - radius), max(0, y - radius), min(engine.width, x + radius + 1), min(engine.height, y + radius + 1))


def _inject_gas_immediate(engine: "WorldEngine", x: int, y: int, species: str, amount: float, radius: int) -> None:
    gy, gx = cell_xy_to_gas(engine, x, y)
    species_id = engine._resolve_sanctioned_gas_id(species)
    if species_id < 0:
        raise KeyError(species)
    engine._invalidate_gpu_authoritative_resources("gas_concentration")
    engine.gas_concentration[species_id, gy, gx] = max(0.0, engine.gas_concentration[species_id, gy, gx] + amount)
    engine._mark_active_rect_runtime(max(0, x - radius), max(0, y - radius), min(engine.width, x + radius + 1), min(engine.height, y + radius + 1))


def set_cell(engine: "WorldEngine", x: int, y: int, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
    material_id = engine._resolve_sanctioned_material_id(material_name)
    if material_id <= 0:
        raise KeyError(material_name)
    set_cell_by_id(engine, x, y, material_id, phase=phase, mark_dirty=mark_dirty)


def clear_cell(engine: "WorldEngine", x: int, y: int, *, mark_dirty: bool = True) -> None:
    engine._invalidate_gpu_authoritative_cell_resources()
    previous_material = int(engine.material_id[y, x])
    previous_phase = int(engine.phase[y, x])
    previous_island_id = int(engine.island_id[y, x])
    previous_displaced = int(engine.placeholder_displaced_material[y, x])
    previous_is_placeholder = engine._shadow_material_is_placeholder(previous_material)
    engine.material_id[y, x] = 0
    engine.phase[y, x] = 0
    engine.cell_flags[y, x] = 0
    engine.velocity[y, x] = 0.0
    engine.cell_temperature[y, x] = engine.ambient_temperature_at_cell(x, y)
    engine.timer_pack[y, x] = 0
    engine.integrity[y, x] = 0.0
    engine.island_id[y, x] = 0
    engine.entity_id[y, x] = 0
    engine.placeholder_displaced_material[y, x] = 0
    if previous_is_placeholder or previous_displaced != 0:
        engine._pending_placeholder_dirty_rects.append((int(x), int(y), int(x) + 1, int(y) + 1))
    engine._refresh_island_records_for_ids((previous_island_id,))
    engine._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(engine.width, x + 2), min(engine.height, y + 2))
    if mark_dirty and engine._cell_participates_in_collapse(previous_material, previous_phase):
        engine._mark_collapse_dirty_rect(x, y, x + 1, y + 1)


def clear_cells(engine: "WorldEngine", mask: np.ndarray, *, mark_dirty: bool = True) -> None:
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        clear_cell(engine, int(x), int(y), mark_dirty=mark_dirty)


def set_material_by_mask(engine: "WorldEngine", mask: np.ndarray, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        set_cell(engine, int(x), int(y), material_name, phase=phase, mark_dirty=mark_dirty)


def swap_cells(engine: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> None:
    engine._invalidate_gpu_authoritative_cell_resources()
    previous_placeholder_cells = (
        (x0, y0, int(engine.entity_id[y0, x0]), int(engine.material_id[y0, x0])),
        (x1, y1, int(engine.entity_id[y1, x1]), int(engine.material_id[y1, x1])),
    )
    previous_island_ids = (
        int(engine.island_id[y0, x0]),
        int(engine.island_id[y1, x1]),
    )
    material0 = int(engine.material_id[y0, x0])
    material1 = int(engine.material_id[y1, x1])
    if (material0 == 0) != (material1 == 0):
        src_x, src_y, dst_x, dst_y = (x1, y1, x0, y0) if material0 == 0 else (x0, y0, x1, y1)
        for array in (
            engine.material_id,
            engine.phase,
            engine.cell_flags,
            engine.cell_temperature,
            engine.integrity,
            engine.island_id,
            engine.entity_id,
            engine.placeholder_displaced_material,
        ):
            value = array[src_y, src_x].copy() if hasattr(array[src_y, src_x], "copy") else array[src_y, src_x]
            array[dst_y, dst_x] = value
        for array in (engine.velocity, engine.timer_pack):
            array[dst_y, dst_x] = array[src_y, src_x].copy()
        engine.material_id[src_y, src_x] = 0
        engine.phase[src_y, src_x] = 0
        engine.cell_flags[src_y, src_x] = 0
        engine.cell_temperature[src_y, src_x] = 0.0
        engine.integrity[src_y, src_x] = 0.0
        engine.island_id[src_y, src_x] = 0
        engine.entity_id[src_y, src_x] = 0
        engine.placeholder_displaced_material[src_y, src_x] = 0
        engine.velocity[src_y, src_x] = 0.0
        engine.timer_pack[src_y, src_x] = 0
    else:
        for array in (
            engine.material_id,
            engine.phase,
            engine.cell_flags,
            engine.cell_temperature,
            engine.integrity,
            engine.island_id,
            engine.entity_id,
            engine.placeholder_displaced_material,
        ):
            array[y0, x0], array[y1, x1] = (
                array[y1, x1].copy() if hasattr(array[y1, x1], "copy") else array[y1, x1],
                array[y0, x0].copy() if hasattr(array[y0, x0], "copy") else array[y0, x0],
            )
        for array in (engine.velocity, engine.timer_pack):
            temp = array[y0, x0].copy()
            array[y0, x0] = array[y1, x1]
            array[y1, x1] = temp
    for cell_x, cell_y, entity_id, material_id in previous_placeholder_cells:
        if entity_id > 0:
            cells = engine.entity_placeholders.get(entity_id)
            if cells is not None:
                cells.discard((cell_x, cell_y))
                if not cells:
                    engine.entity_placeholders.pop(entity_id, None)
    for cell_x, cell_y in ((x0, y0), (x1, y1)):
        entity_id = int(engine.entity_id[cell_y, cell_x])
        material_id = int(engine.material_id[cell_y, cell_x])
        if entity_id > 0 and material_id > 0 and engine._shadow_material_is_placeholder(material_id):
            engine.entity_placeholders.setdefault(entity_id, set()).add((cell_x, cell_y))
    engine._refresh_island_records_for_ids(
        previous_island_ids
        + (
            int(engine.island_id[y0, x0]),
            int(engine.island_id[y1, x1]),
        )
    )
    engine._mark_active_rect_runtime(max(0, min(x0, x1) - 1), max(0, min(y0, y1) - 1), min(engine.width, max(x0, x1) + 2), min(engine.height, max(y0, y1) + 2))


def clear_cell_region(engine: "WorldEngine", x0: int, y0: int, x1: int, y1: int, *, mark_dirty: bool = True) -> None:
    engine._invalidate_gpu_authoritative_cell_resources()
    region_material = engine.material_id[y0:y1, x0:x1]
    region_phase = engine.phase[y0:y1, x0:x1]
    region_island_id = engine.island_id[y0:y1, x0:x1].copy()
    affects_collapse = bool(
        mark_dirty
        and region_material.size
        and np.any(
            (region_material != 0)
            & (region_phase != int(Phase.FALLING_ISLAND))
            & (engine.material_is_structural[region_material] | engine.material_is_support_anchor[region_material])
        )
    )
    engine.material_id[y0:y1, x0:x1] = 0
    engine.phase[y0:y1, x0:x1] = 0
    engine.cell_flags[y0:y1, x0:x1] = 0
    engine.velocity[y0:y1, x0:x1] = 0.0
    engine.cell_temperature[y0:y1, x0:x1] = engine.ambient_temperature_region(x0, y0, x1, y1)
    engine.timer_pack[y0:y1, x0:x1] = 0
    engine.integrity[y0:y1, x0:x1] = 0.0
    engine.island_id[y0:y1, x0:x1] = 0
    engine.entity_id[y0:y1, x0:x1] = 0
    engine.placeholder_displaced_material[y0:y1, x0:x1] = 0
    engine._refresh_island_records_for_ids(np.unique(region_island_id))
    engine._mark_active_rect_runtime(x0, y0, x1, y1)
    if affects_collapse:
        engine._mark_collapse_dirty_rect(x0, y0, x1, y1)
