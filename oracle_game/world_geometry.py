from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import zlib

from oracle_game.types import ForceSource, Phase, TargetQuery
from oracle_game.world_constants import CARDINAL_DIRECTION_VECTORS

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _disk_world_cells(engine: "WorldEngine", center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    radius = max(0, int(radius))
    cx, cy = center_world_position
    cells: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            cells.append(engine._clamp_world_position(cx + dx, cy + dy))
    if not cells:
        cells.append(engine._clamp_world_position(cx, cy))
    return sorted(set(cells))


def _disk_world_cells_raw(center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    radius = max(0, int(radius))
    cx, cy = center_world_position
    cells: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            cells.append((int(cx + dx), int(cy + dy)))
    if not cells:
        cells.append((int(cx), int(cy)))
    return sorted(set(cells))


def _line_world_cells(
    engine: "WorldEngine",
    start_world_position: tuple[int, int],
    end_world_position: tuple[int, int],
) -> list[tuple[int, int]]:
    x0, y0 = (int(value) for value in start_world_position)
    x1, y1 = (int(value) for value in end_world_position)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append(engine._clamp_world_position(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return cells


def _line_world_cells_raw(
    start_world_position: tuple[int, int],
    end_world_position: tuple[int, int],
) -> list[tuple[int, int]]:
    x0, y0 = (int(value) for value in start_world_position)
    x1, y1 = (int(value) for value in end_world_position)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((int(x0), int(y0)))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return cells


def _capsule_world_cells(
    engine: "WorldEngine",
    start_world_position: tuple[int, int],
    end_world_position: tuple[int, int],
    radius: int,
) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for world_position in _line_world_cells(engine, start_world_position, end_world_position):
        cells.update(_disk_world_cells(engine, world_position, radius))
    return sorted(cells)


def _capsule_world_cells_raw(
    engine: "WorldEngine",
    start_world_position: tuple[int, int],
    end_world_position: tuple[int, int],
    radius: int,
) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for world_position in _line_world_cells_raw(start_world_position, end_world_position):
        cells.update(_disk_world_cells_raw(world_position, radius))
    return sorted(cells)


def _buffer_cell_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _apply_change_stability_drift(
    engine: "WorldEngine",
    intent_id: str,
    world_position: tuple[int, int],
    *,
    effective_radius: int,
    stability: float,
) -> tuple[int, int]:
    drift_radius = int(round((1.0 - stability) * max(1, effective_radius + 1)))
    if drift_radius <= 0:
        return world_position
    seed = zlib.crc32(intent_id.encode("utf-8")) & 0xFFFFFFFF
    span = drift_radius * 2 + 1
    dx = int(seed % span) - drift_radius
    dy = int((seed // span) % span) - drift_radius
    return engine._clamp_world_position(world_position[0] + dx, world_position[1] + dy)


def _resolve_legal_world_position(
    engine: "WorldEngine",
    world_position: tuple[int, int],
    *,
    require_empty: bool,
    fallback_mode: str,
    fallback_radius: int,
    effective_radius: int,
    source_world_position: tuple[int, int] | None,
) -> tuple[tuple[int, int] | None, bool, str | None]:
    if not require_empty:
        return world_position, False, None

    clamped_world_position = engine._clamp_world_position(*world_position)
    if _world_cell_is_empty(engine, *clamped_world_position):
        return clamped_world_position, False, None

    if fallback_mode == "nearest_empty":
        search_radius = max(0, int(fallback_radius))
        if search_radius <= 0:
            search_radius = max(1, int(effective_radius) + 1)
        empty_world_position = _find_nearest_empty_world_position(engine,
            clamped_world_position,
            radius=search_radius,
        )
        if empty_world_position is not None:
            return empty_world_position, True, "occupied target fell back to nearest empty cell"
        return None, False, "occupied target had no empty fallback cell"

    if fallback_mode == "source":
        if source_world_position is None:
            return None, False, "occupied target requested source fallback without a source position"
        fallback_world_position = engine._clamp_world_position(*source_world_position)
        if _world_cell_is_empty(engine, *fallback_world_position):
            return fallback_world_position, True, "occupied target fell back to source cell"
        return None, False, "occupied target could not fall back to the source cell"

    return None, False, f"occupied target requested unsupported fallback mode '{fallback_mode}'"


def _resolve_entity_anchor(
    engine: "WorldEngine",
    query: TargetQuery,
    source_world_position: tuple[int, int],
    *,
    direction_filter: str | None,
) -> dict[str, Any] | None:
    best: tuple[float, int, tuple[int, int], tuple[int, int]] | None = None
    for entity in engine.entity_states.values():
        if query.anchor_entity_id is not None:
            if entity.entity_id != int(query.anchor_entity_id):
                continue
        else:
            if query.source_entity_id is not None and entity.entity_id == int(query.source_entity_id):
                continue
            if not engine._entity_matches_anchor_filters(entity, query.anchor_filters):
                continue
        world_position = engine._entity_center_world_position(entity)
        buffer_position = _world_to_buffer_clamped(engine, *world_position)
        if direction_filter is not None and not _matches_direction_filter(engine,
            source_world_position,
            world_position,
            direction_filter,
            source_entity_id=query.source_entity_id,
        ):
            continue
        distance_sq = engine._world_distance_sq(source_world_position, world_position)
        candidate = (distance_sq, int(entity.entity_id), buffer_position, world_position)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    _, entity_id, buffer_position, world_position = best
    return {
        "kind": "entity",
        "entity_id": entity_id,
        "buffer_position": buffer_position,
        "world_position": world_position,
    }


def _resolve_terrain_anchor(
    engine: "WorldEngine",
    source_world_position: tuple[int, int],
    terrain_filters: list[str],
    *,
    direction_filter: str | None,
) -> dict[str, Any] | None:
    for terrain_filter in terrain_filters:
        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for y in range(engine.height):
            for x in range(engine.width):
                if not _terrain_cell_matches(engine, x, y, terrain_filter):
                    continue
                world_position = _buffer_to_world_position(engine, (x, y))
                if direction_filter is not None and not _matches_direction_filter(engine,
                    source_world_position,
                    world_position,
                    direction_filter,
                    source_entity_id=None,
                ):
                    continue
                distance_sq = engine._world_distance_sq(source_world_position, world_position)
                candidate = (distance_sq, (x, y), world_position)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            _, buffer_position, world_position = best
            return {
                "kind": "terrain",
                "entity_id": None,
                "buffer_position": buffer_position,
                "world_position": world_position,
            }
    return None


def _terrain_cell_matches(engine: "WorldEngine", x: int, y: int, terrain_filter: str) -> bool:
    material_id, phase = engine._material_state_for_position(
        x,
        y,
        blocked_cells=engine._resolver_blocked_cells,
        released_cells=engine._resolver_released_cells,
    )
    if terrain_filter == "empty":
        return material_id == 0
    if terrain_filter == "tree":
        return engine._terrain_tree_cell_matches(x, y, material_id, phase)
    if terrain_filter in {"liquid", "pool"}:
        return material_id != 0 and phase == int(Phase.LIQUID)
    if terrain_filter == "solid":
        return material_id != 0 and phase != int(Phase.LIQUID)
    if terrain_filter == "hill":
        return engine._terrain_hill_cell_matches(x, y, material_id, phase)
    if terrain_filter == "wall":
        if material_id == 0 or phase == int(Phase.LIQUID):
            return False
        above_material, above_phase = (0, 0) if y <= 0 else engine._material_state_for_position(
            x,
            y - 1,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        below_material, below_phase = (0, 0) if y + 1 >= engine.height else engine._material_state_for_position(
            x,
            y + 1,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        left_material, _ = (0, 0) if x <= 0 else engine._material_state_for_position(
            x - 1,
            y,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        right_material, _ = (0, 0) if x + 1 >= engine.width else engine._material_state_for_position(
            x + 1,
            y,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        vertical_neighbor = (
            (above_material != 0 and above_phase != int(Phase.LIQUID))
            or (below_material != 0 and below_phase != int(Phase.LIQUID))
        )
        horizontal_edge = (
            (left_material == 0)
            or (right_material == 0)
        )
        return vertical_neighbor and horizontal_edge
    if terrain_filter == "hole":
        if material_id != 0 or y + 1 >= engine.height or x == 0 or x + 1 >= engine.width:
            return False
        below_material, below_phase = engine._material_state_for_position(
            x,
            y + 1,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        left_material, left_phase = engine._material_state_for_position(
            x - 1,
            y,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        right_material, right_phase = engine._material_state_for_position(
            x + 1,
            y,
            blocked_cells=engine._resolver_blocked_cells,
            released_cells=engine._resolver_released_cells,
        )
        below_solid = below_material != 0 and below_phase != int(Phase.LIQUID)
        left_solid = left_material != 0 and left_phase != int(Phase.LIQUID)
        right_solid = right_material != 0 and right_phase != int(Phase.LIQUID)
        return bool(below_solid and left_solid and right_solid)
    return False


def _world_cell_is_solid_local(engine: "WorldEngine", x: int, y: int) -> bool:
    material_id, phase = _bounded_material_state_for_position(engine, x, y)
    return material_id != 0 and phase != int(Phase.LIQUID)


def _world_cell_is_empty_local(engine: "WorldEngine", x: int, y: int) -> bool:
    material_id, _ = _bounded_material_state_for_position(engine, x, y)
    return material_id == 0


def _bounded_material_state_for_position(engine: "WorldEngine", x: int, y: int) -> tuple[int, int]:
    if x < 0 or x >= engine.width or y < 0 or y >= engine.height:
        return (0, 0)
    return engine._material_state_for_position(
        x,
        y,
        blocked_cells=engine._resolver_blocked_cells,
        released_cells=engine._resolver_released_cells,
    )


def _matches_direction_filter(
    engine: "WorldEngine",
    source_world_position: tuple[int, int],
    candidate_world_position: tuple[int, int],
    direction_name: str,
    *,
    source_entity_id: int | None,
) -> bool:
    direction = _direction_vector(engine, direction_name, source_entity_id=source_entity_id)
    if direction is None:
        return True
    delta_x = candidate_world_position[0] - source_world_position[0]
    delta_y = candidate_world_position[1] - source_world_position[1]
    if direction[0] < 0:
        return delta_x < 0
    if direction[0] > 0:
        return delta_x > 0
    if direction[1] < 0:
        return delta_y < 0
    if direction[1] > 0:
        return delta_y > 0
    return True


def _query_direction_vector(
    engine: "WorldEngine",
    query: TargetQuery,
    *,
    source_entity_id: int | None,
) -> tuple[int, int] | None:
    if query.direction is None:
        return None
    return _direction_vector(engine, query.direction, source_entity_id=source_entity_id)


def _direction_vector(
    engine: "WorldEngine",
    direction_name: str,
    *,
    source_entity_id: int | None,
) -> tuple[int, int] | None:
    direction_key = direction_name.lower()
    if direction_key in CARDINAL_DIRECTION_VECTORS:
        return CARDINAL_DIRECTION_VECTORS[direction_key]
    if direction_key not in {"forward", "backward"}:
        return None
    facing_x, facing_y = engine._source_facing_vector(source_entity_id)
    if abs(facing_x) >= abs(facing_y):
        direction = (1, 0) if facing_x >= 0.0 else (-1, 0)
    else:
        direction = (0, 1) if facing_y >= 0.0 else (0, -1)
    if direction_key == "backward":
        return (-direction[0], -direction[1])
    return direction


def _buffer_to_world_position(engine: "WorldEngine", position: tuple[int, int]) -> tuple[int, int]:
    world_x, world_y = engine.paging.buffer_to_world(int(position[0]), int(position[1]))
    return (int(world_x), int(world_y))


def _buffer_to_world_float_position(engine: "WorldEngine", position: tuple[float, float]) -> tuple[float, float]:
    world_x = float(engine.paging.origin_x) + ((float(position[0]) - float(engine.paging.buffer_origin_x)) % float(engine.width))
    world_y = float(engine.paging.origin_y) + ((float(position[1]) - float(engine.paging.buffer_origin_y)) % float(engine.height))
    return (float(world_x), float(world_y))


def _world_to_buffer_float_position(engine: "WorldEngine", position: tuple[float, float]) -> tuple[float, float]:
    buffer_x = (
        float(position[0]) - float(engine.paging.origin_x) + float(engine.paging.buffer_origin_x)
    ) % float(engine.width)
    buffer_y = (
        float(position[1]) - float(engine.paging.origin_y) + float(engine.paging.buffer_origin_y)
    ) % float(engine.height)
    return (float(buffer_x), float(buffer_y))


def _force_source_world_position(engine: "WorldEngine", force_source: ForceSource) -> tuple[float, float]:
    if force_source.world_x is not None and force_source.world_y is not None:
        return (float(force_source.world_x), float(force_source.world_y))
    return _buffer_to_world_float_position(engine, (float(force_source.x), float(force_source.y)))


def _force_source_buffer_position(engine: "WorldEngine", force_source: ForceSource) -> tuple[float, float]:
    if force_source.world_x is not None and force_source.world_y is not None:
        return _world_to_buffer_float_position(engine, (float(force_source.world_x), float(force_source.world_y)))
    return (float(force_source.x), float(force_source.y))


def _buffer_gas_to_world_position(engine: "WorldEngine", position: tuple[int, int]) -> tuple[int, int]:
    cell_x = int(position[0]) * int(engine.gas_cell_size)
    cell_y = int(position[1]) * int(engine.gas_cell_size)
    world_x, world_y = _buffer_to_world_position(engine, (cell_x, cell_y))
    return (int(world_x // engine.gas_cell_size), int(world_y // engine.gas_cell_size))


def _buffer_bbox_to_world_bbox(engine: "WorldEngine", bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    world_x0, world_y0 = _buffer_to_world_position(engine, (x0, y0))
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    return (int(world_x0), int(world_y0), int(world_x0) + width, int(world_y0) + height)


def _clamped_world_window(engine: "WorldEngine", world_x: int, world_y: int, width: int, height: int) -> tuple[int, int, int, int]:
    min_world_x = int(engine.paging.origin_x)
    min_world_y = int(engine.paging.origin_y)
    max_world_x = min_world_x + engine.width
    max_world_y = min_world_y + engine.height
    clamped_world_x = min_world_x if engine.width <= 0 else max(min_world_x, min(max_world_x - 1, int(world_x)))
    clamped_world_y = min_world_y if engine.height <= 0 else max(min_world_y, min(max_world_y - 1, int(world_y)))
    span_x = max(0, int(width))
    span_y = max(0, int(height))
    return (
        int(clamped_world_x),
        int(clamped_world_y),
        int(min(max_world_x, clamped_world_x + span_x)),
        int(min(max_world_y, clamped_world_y + span_y)),
    )


def _centered_world_window(engine: "WorldEngine", center_x: int, center_y: int, width: int, height: int) -> tuple[int, int, int, int]:
    clamped_center_x, clamped_center_y = engine._clamp_world_position(center_x, center_y)
    min_world_x = int(engine.paging.origin_x)
    min_world_y = int(engine.paging.origin_y)
    max_world_x = min_world_x + engine.width
    max_world_y = min_world_y + engine.height
    span_x = max(0, int(width))
    span_y = max(0, int(height))
    world_x0 = max(min_world_x, int(clamped_center_x) - span_x // 2)
    world_y0 = max(min_world_y, int(clamped_center_y) - span_y // 2)
    return (
        int(world_x0),
        int(world_y0),
        int(min(max_world_x, world_x0 + span_x)),
        int(min(max_world_y, world_y0 + span_y)),
    )


def _world_axis_spans(
    engine: "WorldEngine",
    world_start: int,
    world_end: int,
    *,
    axis: str,
    gas_grid: bool = False,
) -> list[tuple[int, int]]:
    span = max(0, int(world_end) - int(world_start))
    if span <= 0:
        return []
    if not gas_grid:
        if axis == "x":
            size = engine.width
            origin = int(engine.paging.origin_x)
            buffer_origin = int(engine.paging.buffer_origin_x)
        else:
            size = engine.height
            origin = int(engine.paging.origin_y)
            buffer_origin = int(engine.paging.buffer_origin_y)
    else:
        if axis == "x":
            size = engine.gas_width
            origin = int(engine.paging.origin_x) // int(engine.gas_cell_size)
            buffer_origin = int(engine.paging.buffer_origin_x) // int(engine.gas_cell_size)
        else:
            size = engine.gas_height
            origin = int(engine.paging.origin_y) // int(engine.gas_cell_size)
            buffer_origin = int(engine.paging.buffer_origin_y) // int(engine.gas_cell_size)
    start = (int(world_start) - origin + buffer_origin) % size
    if span >= size:
        return [(0, size)]
    end = (start + span) % size
    if start < end:
        return [(int(start), int(end))]
    spans = [(int(start), int(size))]
    if end > 0:
        spans.append((0, int(end)))
    return spans


def _world_axis_indices(engine: "WorldEngine", world_start: int, world_end: int, *, axis: str, gas_grid: bool = False) -> np.ndarray:
    span = max(0, int(world_end) - int(world_start))
    if span <= 0:
        return np.empty((0,), dtype=np.intp)
    if not gas_grid:
        if axis == "x":
            size = engine.width
            origin = int(engine.paging.origin_x)
            buffer_origin = int(engine.paging.buffer_origin_x)
        else:
            size = engine.height
            origin = int(engine.paging.origin_y)
            buffer_origin = int(engine.paging.buffer_origin_y)
    else:
        if axis == "x":
            size = engine.gas_width
            origin = int(engine.paging.origin_x) // int(engine.gas_cell_size)
            buffer_origin = int(engine.paging.buffer_origin_x) // int(engine.gas_cell_size)
        else:
            size = engine.gas_height
            origin = int(engine.paging.origin_y) // int(engine.gas_cell_size)
            buffer_origin = int(engine.paging.buffer_origin_y) // int(engine.gas_cell_size)
    coords = np.arange(int(world_start), int(world_end), dtype=np.int64)
    return ((coords - origin + buffer_origin) % size).astype(np.intp, copy=False)


def _extract_world_window(
    engine: "WorldEngine",
    array: np.ndarray,
    world_x0: int,
    world_y0: int,
    world_x1: int,
    world_y1: int,
    *,
    x_axis: int,
    y_axis: int,
    gas_grid: bool = False,
) -> np.ndarray:
    x_indices = _world_axis_indices(engine, world_x0, world_x1, axis="x", gas_grid=gas_grid)
    y_indices = _world_axis_indices(engine, world_y0, world_y1, axis="y", gas_grid=gas_grid)
    window = np.take(array, y_indices, axis=y_axis)
    window = np.take(window, x_indices, axis=x_axis)
    return np.ascontiguousarray(window)


def _pack_cell_core_world_window(engine: "WorldEngine", world_x0: int, world_y0: int, world_x1: int, world_y1: int) -> np.ndarray:
    material_id = _extract_world_window(engine,
        engine.material_id,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    phase = _extract_world_window(engine, engine.phase, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    cell_flags = _extract_world_window(engine, engine.cell_flags, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    velocity = _extract_world_window(engine, engine.velocity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    cell_temperature = _extract_world_window(engine,
        engine.cell_temperature,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    timer_pack = _extract_world_window(engine, engine.timer_pack, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    integrity = _extract_world_window(engine, engine.integrity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)

    packed = np.zeros((max(0, world_y1 - world_y0), max(0, world_x1 - world_x0), 5), dtype=np.uint32)
    packed[..., 0] = (
        material_id.astype(np.uint32)
        | (phase.astype(np.uint32) << 16)
        | (cell_flags.astype(np.uint32) << 24)
    )
    half = velocity.astype(np.float16)
    raw_half = half.view(np.uint16)
    packed[..., 1] = raw_half[..., 0].astype(np.uint32) | (raw_half[..., 1].astype(np.uint32) << 16)
    packed[..., 2] = cell_temperature.astype(np.float32).view(np.uint32)
    packed[..., 3] = (
        timer_pack[..., 0].astype(np.uint32)
        | (timer_pack[..., 1].astype(np.uint32) << 8)
        | (timer_pack[..., 2].astype(np.uint32) << 16)
        | (timer_pack[..., 3].astype(np.uint32) << 24)
    )
    packed[..., 4] = np.clip(np.rint(integrity), 0, 65535).astype(np.uint32)
    return packed


def _world_to_buffer_clamped(engine: "WorldEngine", world_x: int, world_y: int) -> tuple[int, int]:
    clamped_world_x, clamped_world_y = engine._clamp_world_position(world_x, world_y)
    buffer_x, buffer_y = engine.paging.world_to_buffer(clamped_world_x, clamped_world_y)
    return (int(buffer_x), int(buffer_y))


def _find_nearest_empty_world_position(
    engine: "WorldEngine",
    start_world_position: tuple[int, int],
    *,
    radius: int,
) -> tuple[int, int] | None:
    start_world_position = engine._clamp_world_position(*start_world_position)
    if _world_cell_is_empty(engine, *start_world_position):
        return start_world_position
    if radius <= 0:
        return None
    for step in range(1, radius + 1):
        seen: set[tuple[int, int]] = set()
        for dy in range(-step, step + 1):
            for dx in range(-step, step + 1):
                if max(abs(dx), abs(dy)) != step:
                    continue
                world_position = engine._clamp_world_position(
                    start_world_position[0] + dx,
                    start_world_position[1] + dy,
                )
                if world_position in seen:
                    continue
                seen.add(world_position)
                if _world_cell_is_empty(engine, *world_position):
                    return world_position
    return None


def _world_cell_is_empty(engine: "WorldEngine", world_x: int, world_y: int) -> bool:
    buffer_x, buffer_y = _world_to_buffer_clamped(engine, world_x, world_y)
    material_id, _ = engine._material_state_for_position(
        buffer_x,
        buffer_y,
        blocked_cells=engine._resolver_blocked_cells,
        released_cells=engine._resolver_released_cells,
    )
    return material_id == 0

