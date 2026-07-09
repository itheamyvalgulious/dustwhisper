from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_motion import (
    ISLAND_RESOLVE_BLOCKED,
    ISLAND_RESOLVE_DIRECT,
    ISLAND_RESOLVE_RERESOLVED,
    ISLAND_RESOLVE_STALE,
    falling_island_reservation_dtype,
)
from oracle_game.sim.motion import (
    FALLING_ISLAND_BREAK_STABLE,
    MAX_ISLAND_DDA_STEP,
    _IslandComponentEntry,
)
from oracle_game.types import Phase


def _falling_island_contact_material_response(
    solver,
    world: "WorldEngine",
    coords: np.ndarray,
    velocity_xy: tuple[float, float],
    attempted_delta: tuple[int, int],
    actual_delta: tuple[int, int],
) -> tuple[float, float]:
    if len(coords) == 0:
        return velocity_xy
    material_ids = world.material_id[coords[:, 0], coords[:, 1]]
    friction = float(np.mean(solver._material_scalar_field(world, material_ids, "friction", world.material_friction)))
    elasticity = float(np.mean(solver._material_scalar_field(world, material_ids, "elasticity", world.material_elasticity)))
    return solver._collision_response(
        velocity_xy,
        attempted_delta,
        actual_delta,
        friction=friction,
        elasticity=elasticity,
    )



def _move_falling_islands(solver, world: "WorldEngine", *, dt: float, use_gpu: bool) -> bool:
    if use_gpu and solver.gpu_pipeline._formal_gpu_frame(world):
        bridge_authoritative = {"cell_core", "island_id", "island_runtime"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
        if (
            not bridge_authoritative
            and {"cell_core", "island_id"}.issubset(world.bridge.gpu_authoritative_resources)
            and solver._can_seed_bridge_runtime_fast_path(world)
        ):
            seeded_count = solver.gpu_pipeline.seed_bridge_falling_island_runtime_from_cpu(world)
            bridge_authoritative = seeded_count > 0 and {"cell_core", "island_id", "island_runtime"}.issubset(
                world.bridge.gpu_authoritative_resources
            )
        runtime_capacity = int(getattr(solver.gpu_pipeline, "last_published_island_runtime_capacity", 0))
        if bridge_authoritative and runtime_capacity > 0:
            reservation_capacity = solver.gpu_pipeline.plan_uploaded_falling_island_reservations_from_bridge_runtime(
                world,
                dt,
                runtime_capacity,
            )
            if reservation_capacity <= 0:
                solver.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
                world.islands.clear()
                return False
            solver.gpu_pipeline.resolve_uploaded_falling_island_reservations(
                world,
                reservation_capacity,
            )
            solver.gpu_pipeline.apply_uploaded_falling_island_settlements(world, reservation_capacity)
            solver.gpu_pipeline.apply_uploaded_falling_island_reservations(world, reservation_capacity)
            solver.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
            world.islands.clear()
            return True
        solver.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
        return False
    used_gpu = False
    processed_ids: set[int] = set()
    pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]] = []
    gpu_shed_fragments = bool(use_gpu and world.islands)
    if gpu_shed_fragments:
        used_gpu = solver.gpu_pipeline.shed_falling_island_fragments(world) or used_gpu
        for record in list(world.islands.values()):
            world._mark_active_rect_runtime(
                max(0, int(record.bbox[0]) - 1),
                max(0, int(record.bbox[1]) - 1),
                min(world.width, int(record.bbox[2]) + 1),
                min(world.height, int(record.bbox[3]) + 1),
            )
    for island_id in list(world.islands):
        if island_id in processed_ids:
            continue
        record = world.islands.get(island_id)
        if record is None:
            continue
        for component_id, coords in solver._resolve_falling_island_components(
            world,
            island_id,
            record,
            skip_shed=gpu_shed_fragments,
            use_gpu_components=use_gpu,
        ):
            processed_ids.add(component_id)
            component_record = world.islands.get(component_id)
            if component_record is None or len(coords) == 0:
                continue
            vx, vy = component_record.velocity_xy
            vy += 0.9
            if use_gpu:
                target_dx = 0
                target_dy = 0
                residual = tuple(float(value) for value in component_record.subcell_offset)
            else:
                target_dx, target_dy, residual = solver._resolve_island_dda_shift(
                    component_record.subcell_offset,
                    (vx, vy),
                    dt,
                )
                if target_dx == 0 and target_dy == 0:
                    gravity_dy = solver._falling_island_gravity_fallback_dy(world, coords, vy)
                    if gravity_dy != 0:
                        target_dy = gravity_dy
                        residual = (float(residual[0]), 0.0)
            pending_components.append(
                (
                    component_id,
                    coords,
                    component_record,
                    (target_dx, target_dy),
                    residual,
                    (vx, vy),
                )
            )
    island_reservations, used_gpu = solver._plan_falling_island_reservations(
        world,
        pending_components,
        dt=dt,
        use_gpu=use_gpu,
    )
    solver.last_island_reservations = island_reservations.copy()
    reservation_by_id = {int(record["island_id"]): record for record in island_reservations}
    ordered_components = sorted(
        pending_components,
        key=lambda item: solver._falling_island_reservation_order_key(reservation_by_id.get(item[0])),
    )
    gpu_apply_shifted = bool(
        use_gpu
        and len(island_reservations) > 0
        and np.any(np.any(island_reservations["resolved_shift"] != 0, axis=1))
    )
    gpu_apply_settled = bool(
        use_gpu
        and len(island_reservations) > 0
        and np.any(
            np.any(island_reservations["target_shift"] != 0, axis=1)
            & ~np.any(island_reservations["resolved_shift"] != 0, axis=1)
        )
    )
    for component_id, coords, component_record, _, residual, (vx, vy) in ordered_components:
        reservation = reservation_by_id.get(component_id)
        if reservation is None:
            target_dx = 0
            target_dy = 0
            actual_dx = 0
            actual_dy = 0
        else:
            target_dx = int(reservation["target_shift"][0])
            target_dy = int(reservation["target_shift"][1])
            actual_dx = int(reservation["resolved_shift"][0])
            actual_dy = int(reservation["resolved_shift"][1])
            if use_gpu:
                residual = (
                    float(reservation["subcell_offset"][0]),
                    float(reservation["subcell_offset"][1]),
                )
        if actual_dx == 0 and actual_dy == 0 and target_dx == 0 and target_dy == 0:
            component_record.velocity_xy = (vx * 0.98, vy)
            component_record.subcell_offset = residual
            component_record.bbox = solver._bbox_from_coords(coords, 0, 0)
            world.islands[component_id] = component_record
            world._mark_active_rect_runtime(*component_record.bbox)
            continue
        if actual_dx != 0 or actual_dy != 0:
            collision_velocity_xy = None
            if actual_dx != target_dx or actual_dy != target_dy:
                if use_gpu and reservation is not None:
                    collision_velocity_xy = (
                        float(reservation["velocity_xy"][0]),
                        float(reservation["velocity_xy"][1]),
                    )
                else:
                    collision_velocity_xy = solver._falling_island_contact_material_response(
                        world,
                        coords,
                        (vx, vy),
                        (target_dx, target_dy),
                        (actual_dx, actual_dy),
                    )
            old_bbox = solver._bbox_from_coords(coords, 0, 0)
            new_bbox = solver._bbox_from_coords(coords, actual_dx, actual_dy)
            if gpu_apply_shifted:
                world._mark_active_rect_runtime(
                    max(0, min(old_bbox[0], new_bbox[0]) - 1),
                    max(0, min(old_bbox[1], new_bbox[1]) - 1),
                    min(world.width, max(old_bbox[2], new_bbox[2]) + 1),
                    min(world.height, max(old_bbox[3], new_bbox[3]) + 1),
                )
            else:
                solver._shift_island(world, coords, actual_dx, actual_dy, component_id)
            if actual_dx == target_dx and actual_dy == target_dy:
                component_record.velocity_xy = (vx * 0.98, vy)
                component_record.subcell_offset = residual
            else:
                assert collision_velocity_xy is not None
                component_record.velocity_xy = collision_velocity_xy
                component_record.subcell_offset = (0.0, 0.0)
            component_record.bbox = new_bbox
        else:
            if gpu_apply_settled:
                world._mark_active_rect_runtime(*solver._bbox_from_coords(coords, 0, 0))
            else:
                for y, x in coords:
                    material_id = int(world.material_id[y, x])
                    powder_generation_id = solver._material_powder_generation_id(world, material_id)
                    if solver._same_island_neighbors(world, x, y, component_id) < 2 and powder_generation_id > 0:
                        world.set_cell_by_id(x, y, powder_generation_id, phase=Phase.POWDER)
                    else:
                        default_phase = solver._material_default_phase(world, material_id)
                        world.phase[y, x] = int(Phase.STATIC_SOLID) if default_phase is None else default_phase
                        if material_id <= 0 or not solver._material_is_placeholder(world, material_id):
                            world.entity_id[y, x] = 0
                            world.placeholder_displaced_material[y, x] = 0
                    world.island_id[y, x] = 0
            world._mark_active_rect_runtime(*solver._bbox_from_coords(coords, 0, 0))
            world.islands.pop(component_id, None)
            continue
        world.islands[component_id] = component_record
        world._mark_active_rect_runtime(*component_record.bbox)
    if gpu_apply_settled:
        used_gpu = solver.gpu_pipeline.apply_uploaded_falling_island_settlements(
            world,
            int(len(island_reservations)),
        ) or used_gpu
    if gpu_apply_shifted:
        used_gpu = solver.gpu_pipeline.apply_uploaded_falling_island_reservations(
            world,
            int(len(island_reservations)),
        ) or used_gpu
    return used_gpu



def _can_seed_bridge_runtime_fast_path(solver, world: "WorldEngine") -> bool:
    if not world.islands:
        return False
    bridge_authoritative_island_grid = {"cell_core", "island_id"}.issubset(
        world.bridge.gpu_authoritative_resources
    )
    for island_id, record in world.islands.items():
        if int(island_id) <= 0:
            return False
        x0, y0, x1, y1 = (int(value) for value in record.bbox)
        if x0 < 0 or y0 < 0 or x1 > world.width or y1 > world.height or x1 <= x0 or y1 <= y0:
            return False
        if bridge_authoritative_island_grid:
            continue
        island_mask = world.island_id[y0:y1, x0:x1] == int(island_id)
        material_mask = world.material_id[y0:y1, x0:x1] > 0
        if not bool(np.any(island_mask & material_mask)):
            return False
    return True



def _plan_falling_island_reservations(
    solver,
    world: "WorldEngine",
    pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]],
    *,
    dt: float,
    use_gpu: bool,
) -> tuple[np.ndarray, bool]:
    reservations = np.zeros((len(pending_components),), dtype=falling_island_reservation_dtype())
    if len(pending_components) == 0:
        solver.gpu_pipeline.upload_falling_island_reservations(world, reservations)
        return reservations, False
    reservation_index_by_id: dict[int, int] = {}
    motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    gpu_island_ids: list[int] = []
    for index, (component_id, _, component_record, (target_dx, target_dy), _, (vx, vy)) in enumerate(pending_components):
        reservations[index]["island_id"] = int(component_id)
        reservations[index]["buffer_bbox"] = np.asarray(component_record.bbox, dtype=np.int32)
        reservations[index]["velocity_xy"] = np.asarray((vx, vy), dtype=np.float32)
        reservations[index]["subcell_offset"] = np.asarray(component_record.subcell_offset, dtype=np.float32)
        reservations[index]["target_shift"] = np.asarray((target_dx, target_dy), dtype=np.int32)
        reservations[index]["resolve_state"] = ISLAND_RESOLVE_DIRECT if target_dx == 0 and target_dy == 0 else ISLAND_RESOLVE_BLOCKED
        reservation_index_by_id[component_id] = index
        if use_gpu:
            gpu_island_ids.append(component_id)
            motion_overrides[component_id] = (
                (float(vx), float(vy)),
                tuple(float(value) for value in component_record.subcell_offset),
            )
        elif target_dx != 0 or target_dy != 0:
            gpu_island_ids.append(component_id)
            motion_overrides[component_id] = (
                (float(vx), float(vy)),
                tuple(float(value) for value in component_record.subcell_offset),
            )
    gpu_handled: set[int] = set()
    used_gpu = False
    gpu_available = bool(use_gpu and world._gpu_pipeline_available(solver.gpu_pipeline, "motion"))
    if gpu_available and gpu_island_ids:
        gpu_reservations = solver.gpu_pipeline.plan_falling_island_reservations(
            world,
            dt,
            island_ids=gpu_island_ids,
            motion_overrides=motion_overrides,
        )
        for reservation in gpu_reservations:
            island_id = int(reservation["island_id"])
            index = reservation_index_by_id.get(island_id)
            if index is None:
                continue
            reservations[index]["target_shift"] = np.asarray(reservation["target_shift"], dtype=np.int32)
            reservations[index]["reserved_shift"] = np.asarray(reservation["reserved_shift"], dtype=np.int32)
            reservations[index]["resolved_shift"] = np.asarray(reservation["reserved_shift"], dtype=np.int32)
            reservations[index]["velocity_xy"] = np.asarray(reservation["velocity_xy"], dtype=np.float32)
            reservations[index]["subcell_offset"] = np.asarray(reservation["subcell_offset"], dtype=np.float32)
            reservations[index]["resolve_state"] = int(reservation["resolve_state"])
            gpu_handled.add(island_id)
        used_gpu = True
    for component_id, coords, _, (target_dx, target_dy), _, _ in pending_components:
        if component_id in gpu_handled or (target_dx == 0 and target_dy == 0):
            continue
        if use_gpu:
            raise RuntimeError(
                "GPU motion pipeline did not return a falling-island reservation; CPU fallback is disabled"
            )
        actual_dx, actual_dy = solver._resolve_island_dda_target(
            world,
            coords,
            component_id,
            target_dx,
            target_dy,
        )
        reservations[reservation_index_by_id[component_id]]["reserved_shift"] = np.asarray(
            (actual_dx, actual_dy),
            dtype=np.int32,
        )
    if gpu_available:
        reservations = solver.gpu_pipeline.resolve_falling_island_reservations(world, reservations)
        used_gpu = True
    else:
        world._require_cpu_oracle_backend("motion falling island reservations")
        reservations = solver._resolve_falling_island_reservations(world, pending_components, reservations)
        solver.gpu_pipeline.upload_falling_island_reservations(world, reservations)
    return reservations, used_gpu



def _falling_island_reservation_order_key(solver, reservation: np.ndarray | None) -> tuple[int, int, int, int]:
    if reservation is None:
        return (4, 0, 0, 0)
    island_id = int(reservation["island_id"])
    x0, y0, x1, y1 = (int(value) for value in reservation["buffer_bbox"])
    target_dx = int(reservation["target_shift"][0])
    target_dy = int(reservation["target_shift"][1])
    if target_dy > 0:
        return (0, -y1, x0, island_id)
    if target_dy < 0:
        return (1, y0, x0, island_id)
    if target_dx > 0:
        return (2, -x1, y0, island_id)
    if target_dx < 0:
        return (3, x0, y0, island_id)
    return (4, y0, x0, island_id)



def _resolve_falling_island_reservations(
    solver,
    world: "WorldEngine",
    pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]],
    reservations: np.ndarray,
) -> np.ndarray:
    if len(reservations) == 0:
        return reservations
    resolved = reservations.copy()
    coords_by_id = {int(component_id): coords for component_id, coords, *_ in pending_components}
    shadow_material = world.material_id.copy()
    order = sorted(range(len(resolved)), key=lambda index: solver._falling_island_reservation_order_key(resolved[index]))
    for index in order:
        island_id = int(resolved[index]["island_id"])
        coords = coords_by_id.get(island_id)
        if coords is None or len(coords) == 0:
            resolved[index]["resolve_state"] = ISLAND_RESOLVE_STALE
            continue
        target_dx = int(resolved[index]["target_shift"][0])
        target_dy = int(resolved[index]["target_shift"][1])
        candidate_dx = int(resolved[index]["reserved_shift"][0])
        candidate_dy = int(resolved[index]["reserved_shift"][1])
        actual_dx, actual_dy = solver._resolve_island_dda_target_material(
            shadow_material,
            coords,
            target_dx,
            target_dy,
        )
        resolved[index]["resolved_shift"] = np.asarray((actual_dx, actual_dy), dtype=np.int32)
        if actual_dx == 0 and actual_dy == 0 and target_dx == 0 and target_dy == 0:
            resolved[index]["resolve_state"] = ISLAND_RESOLVE_DIRECT
            continue
        if actual_dx == 0 and actual_dy == 0:
            resolved[index]["resolve_state"] = ISLAND_RESOLVE_BLOCKED
            continue
        if actual_dx == candidate_dx and actual_dy == candidate_dy:
            resolved[index]["resolve_state"] = ISLAND_RESOLVE_DIRECT
        else:
            resolved[index]["resolve_state"] = ISLAND_RESOLVE_RERESOLVED
        solver._shadow_shift_island_material(shadow_material, coords, actual_dx, actual_dy)
    return resolved



def _shed_falling_island_fragments(solver, world: "WorldEngine", island_id: int) -> np.ndarray:
    coords = solver._falling_island_coords(world, island_id)
    if len(coords) == 0:
        return coords
    fragments: list[tuple[int, int, str]] = []
    for y, x in coords:
        material_id = int(world.material_id[y, x])
        neighbor_count = solver._same_island_neighbors(world, int(x), int(y), island_id)
        if neighbor_count >= solver._falling_island_fragment_neighbor_threshold(world, material_id):
            continue
        powder_generation_id = solver._material_powder_generation_id(world, material_id)
        if powder_generation_id <= 0:
            continue
        fragments.append((int(x), int(y), powder_generation_id))
    for x, y, powder_generation_id in fragments:
        world.set_cell_by_id(x, y, powder_generation_id, phase=Phase.POWDER, mark_dirty=False)
    if not fragments:
        return coords
    return solver._falling_island_coords(world, island_id)



def _falling_island_fragment_neighbor_threshold(solver, world: "WorldEngine", material_id: int) -> int:
    if material_id > 0 and solver._material_falling_island_break_kind(world, material_id) == FALLING_ISLAND_BREAK_STABLE:
        return 0
    return 2



def _resolve_falling_island_components(
    solver,
    world: "WorldEngine",
    island_id: int,
    record: "FallingIslandRecord",
    *,
    skip_shed: bool = False,
    use_gpu_components: bool = False,
) -> list[tuple[int, np.ndarray]]:
    component_label_texture: Any | None = None
    if use_gpu_components and world._gpu_pipeline_available(solver.gpu_pipeline, "motion components"):
        component_label_texture, component_entries = solver._gpu_connected_island_component_entries(
            world,
            island_id,
            record.bbox,
        )
    else:
        world._require_cpu_oracle_backend("motion component labeling")
        solver._clear_stale_island_cells(world, island_id)
        coords = (
            solver._falling_island_coords(world, island_id)
            if skip_shed
            else solver._shed_falling_island_fragments(world, island_id)
        )
        if len(coords) == 0:
            world.islands.pop(island_id, None)
            return []
        component_entries = [
            solver._component_entry_from_coords(0, component)
            for component in solver._connected_island_components(coords)
        ]
    if len(component_entries) == 0:
        world.islands.pop(island_id, None)
        return []
    if len(component_entries) == 1:
        record.bbox = component_entries[0].bbox
        world.islands[island_id] = record
        return [(island_id, component_entries[0].coords)]
    component_entries.sort(
        key=lambda entry: (-entry.cell_count, int(entry.bbox[1]), int(entry.bbox[0]))
    )
    result: list[tuple[int, np.ndarray]] = []
    primary = component_entries[0]
    primary_label = int(primary.label)
    primary_component = primary.coords
    record.bbox = primary.bbox
    world.islands[island_id] = record
    result.append((island_id, primary_component))
    component_labels = [int(primary_label)]
    component_island_ids = [int(island_id)]
    split_components: list[tuple[int, np.ndarray]] = []
    for entry in component_entries[1:]:
        label = int(entry.label)
        component = entry.coords
        new_island_id = world.allocate_island_id()
        component_labels.append(int(label))
        component_island_ids.append(int(new_island_id))
        split_components.append((new_island_id, component))
        world.islands[new_island_id] = type(record)(
            island_id=new_island_id,
            bbox=entry.bbox,
            velocity_xy=(float(record.velocity_xy[0]), float(record.velocity_xy[1])),
            subcell_offset=tuple(record.subcell_offset),
        )
        result.append((new_island_id, component))
    relabeled_on_gpu = False
    if component_label_texture is not None:
        relabeled_on_gpu = solver.gpu_pipeline.relabel_falling_island_component_texture(
            world,
            component_label_texture,
            np.asarray(component_labels, dtype=np.int32),
            np.asarray(component_island_ids, dtype=np.int32),
        )
    if not relabeled_on_gpu and use_gpu_components:
        raise RuntimeError("GPU motion pipeline did not relabel falling-island components; CPU fallback is disabled")
    if not relabeled_on_gpu:
        world._require_cpu_oracle_backend("motion component relabeling")
        for new_island_id, component in split_components:
            solver._assign_split_component_cells_cpu(world, component, new_island_id)
    return result



def _component_entry_from_coords(solver, label: int, coords: np.ndarray) -> _IslandComponentEntry:
    return _IslandComponentEntry(
        label=int(label),
        coords=coords,
        bbox=solver._bbox_from_coords(coords, 0, 0),
        cell_count=int(len(coords)),
    )



def _component_entry_from_gpu_metadata(solver, metadata: np.ndarray) -> _IslandComponentEntry:
    label, min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata)
    bbox = (min_x, min_y, max_x, max_y)
    width = max_x - min_x
    height = max_y - min_y
    bbox_area = width * height
    if width <= 0 or height <= 0 or cell_count <= 0 or cell_count > bbox_area:
        raise RuntimeError("GPU falling-island component metadata is invalid")
    if cell_count != bbox_area:
        raise RuntimeError(
            "GPU falling-island component metadata does not include exact non-rectangular coords"
        )
    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    coords = np.stack((yy.ravel(), xx.ravel()), axis=1).astype(np.int32)
    return _IslandComponentEntry(
        label=label,
        coords=coords,
        bbox=bbox,
        cell_count=cell_count,
    )



def _assign_split_component_cells_cpu(
    solver,
    world: "WorldEngine",
    component: np.ndarray,
    island_id: int,
) -> None:
    for y, x in component:
        world.island_id[int(y), int(x)] = int(island_id)



def _clear_stale_island_cells(solver, world: "WorldEngine", island_id: int) -> None:
    invalid_mask = (world.island_id == island_id) & (
        (world.phase != int(Phase.FALLING_ISLAND)) | (world.material_id <= 0)
    )
    if np.any(invalid_mask):
        world.island_id[invalid_mask] = 0



def _gpu_connected_island_components(
    solver,
    world: "WorldEngine",
    island_id: int,
    bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    _, entries = solver._gpu_connected_island_component_entries(world, island_id, bbox)
    return [entry.coords for entry in entries]



def _gpu_connected_island_component_entries(
    solver,
    world: "WorldEngine",
    island_id: int,
    bbox: tuple[int, int, int, int],
) -> tuple[Any, list[_IslandComponentEntry]]:
    labels, metadata = solver.gpu_pipeline.label_falling_island_component_metadata_texture(world, island_id, bbox)
    if metadata.size == 0:
        return labels, []
    return labels, [solver._component_entry_from_gpu_metadata(record) for record in metadata]



def _resolve_island_dda_shift(
    solver,
    subcell_offset: tuple[float, float],
    velocity_xy: tuple[float, float],
    dt: float,
) -> tuple[int, int, tuple[float, float]]:
    total_x = float(subcell_offset[0]) + float(velocity_xy[0]) * float(dt)
    total_y = float(subcell_offset[1]) + float(velocity_xy[1]) * float(dt)
    target_dx = int(np.clip(np.rint(total_x), -MAX_ISLAND_DDA_STEP, MAX_ISLAND_DDA_STEP))
    target_dy = int(np.clip(np.rint(total_y), -MAX_ISLAND_DDA_STEP, MAX_ISLAND_DDA_STEP))
    residual = (float(total_x - target_dx), float(total_y - target_dy))
    return target_dx, target_dy, residual



def _falling_island_gravity_fallback_dy(
    solver,
    world: "WorldEngine",
    coords: np.ndarray,
    velocity_y: float,
) -> int:
    if len(coords) == 0:
        return 0
    material_ids = world.material_id[coords[:, 0], coords[:, 1]]
    gravity = solver._material_scalar_field(world, material_ids, "gravity_scale", world.material_gravity)
    mean_gravity = float(np.mean(gravity)) if gravity.size else 0.0
    if abs(mean_gravity) > 1.0e-6:
        return 1 if mean_gravity > 0.0 else -1
    if abs(float(velocity_y)) <= 1.0e-6:
        return 0
    return 1 if float(velocity_y) > 0.0 else -1



def _resolve_island_dda_target(
    solver,
    world: "WorldEngine",
    coords: np.ndarray,
    island_id: int,
    target_dx: int,
    target_dy: int,
) -> tuple[int, int]:
    if target_dx == 0 and target_dy == 0:
        return (0, 0)
    furthest = (0, 0)
    for dx, dy in solver._dda_line_cells(0, 0, target_dx, target_dy):
        if not solver._can_shift_island(world, coords, dx, dy, island_id):
            break
        furthest = (dx, dy)
    return furthest



def _resolve_island_dda_target_material(
    solver,
    material_id: np.ndarray,
    coords: np.ndarray,
    target_dx: int,
    target_dy: int,
) -> tuple[int, int]:
    if target_dx == 0 and target_dy == 0:
        return (0, 0)
    furthest = (0, 0)
    for dx, dy in solver._dda_line_cells(0, 0, target_dx, target_dy):
        if not solver._can_shift_island_material(material_id, coords, dx, dy):
            break
        furthest = (dx, dy)
    return furthest



def _connected_island_components(solver, coords: np.ndarray) -> list[np.ndarray]:
    coord_set = {(int(x), int(y)) for y, x in coords.tolist()}
    seen: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []
    for y, x in coords.tolist():
        start = (int(x), int(y))
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        component: list[tuple[int, int]] = []
        while queue:
            cx, cy = queue.pop()
            component.append((cy, cx))
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                neighbor = (nx, ny)
                if neighbor in seen or neighbor not in coord_set:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int32))
    return components



def _bbox_from_coords(solver, coords: np.ndarray, dx: int, dy: int) -> tuple[int, int, int, int]:
    xs = coords[:, 1] + dx
    ys = coords[:, 0] + dy
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)



def _same_island_neighbors(solver, world: "WorldEngine", x: int, y: int, island_id: int) -> int:
    count = 0
    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if (
            0 <= nx < world.width
            and 0 <= ny < world.height
            and int(world.island_id[ny, nx]) == island_id
            and int(world.phase[ny, nx]) == int(Phase.FALLING_ISLAND)
            and int(world.material_id[ny, nx]) > 0
        ):
            count += 1
    return count



def _falling_island_coords(solver, world: "WorldEngine", island_id: int) -> np.ndarray:
    solver._clear_stale_island_cells(world, island_id)
    return np.argwhere(
        (world.island_id == island_id)
        & (world.phase == int(Phase.FALLING_ISLAND))
        & (world.material_id > 0)
    )



def _can_shift_island(solver, world: "WorldEngine", coords: np.ndarray, dx: int, dy: int, island_id: int) -> bool:
    occupied = {(int(x), int(y)) for y, x in coords}
    for y, x in coords:
        nx = int(x) + dx
        ny = int(y) + dy
        if nx < 0 or nx >= world.width or ny < 0 or ny >= world.height:
            return False
        other_id = int(world.material_id[ny, nx])
        if other_id == 0:
            continue
        if (nx, ny) in occupied:
            continue
        return False
    return True



def _can_shift_island_material(solver, material_id: np.ndarray, coords: np.ndarray, dx: int, dy: int) -> bool:
    height, width = material_id.shape
    occupied = {(int(x), int(y)) for y, x in coords}
    for y, x in coords:
        nx = int(x) + dx
        ny = int(y) + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            return False
        other_id = int(material_id[ny, nx])
        if other_id == 0:
            continue
        if (nx, ny) in occupied:
            continue
        return False
    return True



def _shadow_shift_island_material(solver, material_id: np.ndarray, coords: np.ndarray, dx: int, dy: int) -> None:
    ys = coords[:, 0].astype(np.int32)
    xs = coords[:, 1].astype(np.int32)
    new_ys = ys + int(dy)
    new_xs = xs + int(dx)
    payload = material_id[ys, xs].copy()
    material_id[ys, xs] = 0
    material_id[new_ys, new_xs] = payload



def _shift_island(solver, world: "WorldEngine", coords: np.ndarray, dx: int, dy: int, island_id: int) -> None:
    ys = coords[:, 0].astype(np.int32)
    xs = coords[:, 1].astype(np.int32)
    new_ys = ys + int(dy)
    new_xs = xs + int(dx)
    old_bbox = solver._bbox_from_coords(coords, 0, 0)
    new_bbox = solver._bbox_from_coords(coords, dx, dy)

    material_id = world.material_id[ys, xs].copy()
    phase = world.phase[ys, xs].copy()
    cell_flags = world.cell_flags[ys, xs].copy()
    velocity = world.velocity[ys, xs].copy()
    cell_temperature = world.cell_temperature[ys, xs].copy()
    timer_pack = world.timer_pack[ys, xs].copy()
    integrity = world.integrity[ys, xs].copy()
    entity_id = world.entity_id[ys, xs].copy()
    displaced = world.placeholder_displaced_material[ys, xs].copy()
    ambient_cell_temperature = world.ambient_temperature[
        np.clip(ys // world.gas_cell_size, 0, world.gas_height - 1),
        np.clip(xs // world.gas_cell_size, 0, world.gas_width - 1),
    ].copy()

    world.material_id[ys, xs] = 0
    world.phase[ys, xs] = 0
    world.cell_flags[ys, xs] = 0
    world.velocity[ys, xs] = 0.0
    world.cell_temperature[ys, xs] = ambient_cell_temperature
    world.timer_pack[ys, xs] = 0
    world.integrity[ys, xs] = 0.0
    world.island_id[ys, xs] = 0
    world.entity_id[ys, xs] = 0
    world.placeholder_displaced_material[ys, xs] = 0

    world.material_id[new_ys, new_xs] = material_id
    world.phase[new_ys, new_xs] = phase
    world.cell_flags[new_ys, new_xs] = cell_flags
    world.velocity[new_ys, new_xs] = velocity
    world.cell_temperature[new_ys, new_xs] = cell_temperature
    world.timer_pack[new_ys, new_xs] = timer_pack
    world.integrity[new_ys, new_xs] = integrity
    world.island_id[new_ys, new_xs] = island_id
    world.entity_id[new_ys, new_xs] = entity_id
    world.placeholder_displaced_material[new_ys, new_xs] = displaced

    world._mark_active_rect_runtime(
        max(0, min(old_bbox[0], new_bbox[0]) - 1),
        max(0, min(old_bbox[1], new_bbox[1]) - 1),
        min(world.width, max(old_bbox[2], new_bbox[2]) + 1),
        min(world.height, max(old_bbox[3], new_bbox[3]) + 1),
    )

