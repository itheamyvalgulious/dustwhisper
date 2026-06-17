from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_motion import (
    GPUMotionPipeline,
    ISLAND_RESOLVE_BLOCKED,
    ISLAND_RESOLVE_DIRECT,
    ISLAND_RESOLVE_RERESOLVED,
    ISLAND_RESOLVE_STALE,
    POWDER_RESOLVE_BLOCKED,
    POWDER_RESOLVE_DDA,
    POWDER_RESOLVE_FALLBACK,
    POWDER_RESOLVE_STALE,
    falling_island_reservation_dtype,
    powder_reservation_dtype,
)
from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask
from oracle_game.types import Phase

MAX_ISLAND_DDA_STEP = 4
POWDER_SOLVER_SUSPENDED = 2
FALLING_ISLAND_BREAK_STABLE = 2


@dataclass(slots=True)
class _IslandComponentEntry:
    label: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    cell_count: int


class MotionSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUMotionPipeline()
        self.last_backend = "idle"
        self.last_powder_reservations = np.zeros((0,), dtype=powder_reservation_dtype())
        self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
        self.last_public_powder_reservations: list[dict[str, object]] = []
        self.last_public_island_reservations: list[dict[str, object]] = []

    def step(self, world: "WorldEngine", dt: float) -> None:
        self.reset_runtime_state()
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "motion")
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
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            solve_tile_mask = self._solve_tile_mask(world)
        if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
            self.last_backend = "idle"
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
            self.gpu_pipeline.integrate_velocity(world, dt, solve_tile_mask=solve_tile_mask)
            used_gpu = True
        else:
            world._require_cpu_oracle_backend("motion velocity")
            self._integrate_velocity(world, dt, solve_cell_mask)
            used_cpu = True
        island_cpu_work = bool(world.islands and not gpu_available)
        island_used_gpu = self._move_falling_islands(world, use_gpu=gpu_available)
        used_gpu = used_gpu or island_used_gpu
        used_cpu = used_cpu or island_cpu_work
        powder_active = bool(gpu_available) or bool(
            np.any(solve_cell_mask & (world.material_id != 0) & (world.phase == int(Phase.POWDER)))
        )
        if powder_active:
            if gpu_available:
                powder_reservations = self.gpu_pipeline.resolve_and_apply_powders(
                    world,
                    solve_tile_mask=solve_tile_mask,
                )
                used_gpu = True
            else:
                world._require_cpu_oracle_backend("motion powder")
                powder_reservations = self._plan_cpu_powder_reservations(world, solve_cell_mask)
                powder_reservations = self._resolve_powder_reservations(world, powder_reservations)
                self.gpu_pipeline.upload_powder_reservations(world, powder_reservations)
                self._apply_powder_reservations(world, powder_reservations)
                used_cpu = True
            self.last_powder_reservations = powder_reservations.copy()
            if gpu_available and not active_scheduler_gpu_authoritative:
                self._mark_powder_reservation_regions(world, powder_reservations)
        if used_gpu and used_cpu:
            self.last_backend = "hybrid"
        elif used_gpu:
            self.last_backend = "gpu"
        else:
            self.last_backend = "cpu"
        self.last_public_powder_reservations = self._capture_public_powder_reservations(world, self.last_powder_reservations)
        self.last_public_island_reservations = self._capture_public_island_reservations(world, self.last_island_reservations)

    def _mark_powder_reservation_regions(self, world: "WorldEngine", powder_reservations: np.ndarray) -> None:
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

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self.last_powder_reservations = np.zeros((0,), dtype=powder_reservation_dtype())
        self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
        self.last_public_powder_reservations = []
        self.last_public_island_reservations = []

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "powder_reservations": self.last_powder_reservations.copy(),
            "island_reservations": self.last_island_reservations.copy(),
            "public_powder_reservations": [dict(record) for record in self.last_public_powder_reservations],
            "public_island_reservations": [dict(record) for record in self.last_public_island_reservations],
        }

    def _capture_public_powder_reservations(
        self,
        world: "WorldEngine",
        reservations: np.ndarray,
    ) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for record in reservations:
            item: dict[str, object] = {}
            for name in reservations.dtype.names or ():
                value = record[name]
                if isinstance(value, np.ndarray):
                    if name in {"source_xy", "desired_target_xy", "reserved_target_xy", "resolved_target_xy"}:
                        world_x, world_y = world._buffer_to_world_position((int(value[0]), int(value[1])))
                        item[name] = [int(world_x), int(world_y)]
                    else:
                        item[name] = value.tolist()
                elif isinstance(value, np.generic):
                    item[name] = value.item()
                else:
                    item[name] = value
            payload.append(item)
        return payload

    def _capture_public_island_reservations(
        self,
        world: "WorldEngine",
        reservations: np.ndarray,
    ) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for record in reservations:
            item: dict[str, object] = {}
            for name in reservations.dtype.names or ():
                value = record[name]
                if name == "buffer_bbox":
                    item["world_bbox"] = list(
                        world._buffer_bbox_to_world_bbox(tuple(int(component) for component in np.asarray(value).tolist()))
                    )
                    continue
                if isinstance(value, np.ndarray):
                    item[name] = value.tolist()
                elif isinstance(value, np.generic):
                    item[name] = value.item()
                else:
                    item[name] = value
            payload.append(item)
        return payload

    def _solve_tile_mask(self, world: "WorldEngine") -> np.ndarray:
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

    def _integrate_velocity(self, world: "WorldEngine", dt: float, solve_cell_mask: np.ndarray) -> None:
        non_empty = (world.material_id != 0) & solve_cell_mask
        if not non_empty.any():
            return
        gravity = self._material_scalar_field(world, world.material_id, "gravity_scale", world.material_gravity) * dt * 24.0
        world.velocity[..., 1][non_empty] += gravity[non_empty]
        cell_flow = world.sample_flow_to_cells()
        wind_delta = (
            cell_flow
            * self._material_scalar_field(world, world.material_id, "wind_coupling", world.material_wind)[..., None]
            * dt
            * 4.0
        )
        world.velocity[non_empty] += wind_delta[non_empty]
        drag = np.maximum(
            0.0,
            1.0 - self._material_scalar_field(world, world.material_id, "drag_scale", world.material_drag)[..., None] * dt,
        )
        world.velocity[non_empty] *= drag[non_empty]

    def _collision_response(
        self,
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

    def _falling_island_contact_material_response(
        self,
        world: "WorldEngine",
        coords: np.ndarray,
        velocity_xy: tuple[float, float],
        attempted_delta: tuple[int, int],
        actual_delta: tuple[int, int],
    ) -> tuple[float, float]:
        if len(coords) == 0:
            return velocity_xy
        material_ids = world.material_id[coords[:, 0], coords[:, 1]]
        friction = float(np.mean(self._material_scalar_field(world, material_ids, "friction", world.material_friction)))
        elasticity = float(np.mean(self._material_scalar_field(world, material_ids, "elasticity", world.material_elasticity)))
        return self._collision_response(
            velocity_xy,
            attempted_delta,
            actual_delta,
            friction=friction,
            elasticity=elasticity,
        )

    def _move_falling_islands(self, world: "WorldEngine", *, use_gpu: bool) -> bool:
        if use_gpu and self.gpu_pipeline._formal_gpu_frame(world):
            bridge_authoritative = {"cell_core", "island_id", "island_runtime"}.issubset(
                world.bridge.gpu_authoritative_resources
            )
            if (
                not bridge_authoritative
                and {"cell_core", "island_id"}.issubset(world.bridge.gpu_authoritative_resources)
                and self._can_seed_bridge_runtime_fast_path(world)
            ):
                seeded_count = self.gpu_pipeline.seed_bridge_falling_island_runtime_from_cpu(world)
                bridge_authoritative = seeded_count > 0 and {"cell_core", "island_id", "island_runtime"}.issubset(
                    world.bridge.gpu_authoritative_resources
                )
            runtime_capacity = int(getattr(self.gpu_pipeline, "last_published_island_runtime_capacity", 0))
            if bridge_authoritative and runtime_capacity > 0:
                reservation_count = self.gpu_pipeline.plan_uploaded_falling_island_reservations_from_bridge_runtime(
                    world,
                    runtime_capacity,
                )
                if reservation_count <= 0:
                    self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
                    world.islands.clear()
                    return False
                self.gpu_pipeline.resolve_uploaded_falling_island_reservations(world, reservation_count)
                self.gpu_pipeline.apply_uploaded_falling_island_settlements(world, reservation_count)
                self.gpu_pipeline.apply_uploaded_falling_island_reservations(world, reservation_count)
                self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
                world.islands.clear()
                return True
        used_gpu = False
        processed_ids: set[int] = set()
        pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]] = []
        gpu_shed_fragments = bool(use_gpu and world.islands)
        if gpu_shed_fragments:
            used_gpu = self.gpu_pipeline.shed_falling_island_fragments(world) or used_gpu
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
            for component_id, coords in self._resolve_falling_island_components(
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
                    target_dx, target_dy, residual = self._resolve_island_dda_shift(component_record.subcell_offset, (vx, vy))
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
        island_reservations, used_gpu = self._plan_falling_island_reservations(
            world,
            pending_components,
            use_gpu=use_gpu,
        )
        self.last_island_reservations = island_reservations.copy()
        reservation_by_id = {int(record["island_id"]): record for record in island_reservations}
        ordered_components = sorted(
            pending_components,
            key=lambda item: self._falling_island_reservation_order_key(reservation_by_id.get(item[0])),
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
                component_record.bbox = self._bbox_from_coords(coords, 0, 0)
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
                        collision_velocity_xy = self._falling_island_contact_material_response(
                            world,
                            coords,
                            (vx, vy),
                            (target_dx, target_dy),
                            (actual_dx, actual_dy),
                        )
                old_bbox = self._bbox_from_coords(coords, 0, 0)
                new_bbox = self._bbox_from_coords(coords, actual_dx, actual_dy)
                if gpu_apply_shifted:
                    world._mark_active_rect_runtime(
                        max(0, min(old_bbox[0], new_bbox[0]) - 1),
                        max(0, min(old_bbox[1], new_bbox[1]) - 1),
                        min(world.width, max(old_bbox[2], new_bbox[2]) + 1),
                        min(world.height, max(old_bbox[3], new_bbox[3]) + 1),
                    )
                else:
                    self._shift_island(world, coords, actual_dx, actual_dy, component_id)
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
                    world._mark_active_rect_runtime(*self._bbox_from_coords(coords, 0, 0))
                else:
                    for y, x in coords:
                        material_id = int(world.material_id[y, x])
                        powder_generation_id = self._material_powder_generation_id(world, material_id)
                        if self._same_island_neighbors(world, x, y, component_id) < 2 and powder_generation_id > 0:
                            world.set_cell_by_id(x, y, powder_generation_id, phase=Phase.POWDER)
                        else:
                            default_phase = self._material_default_phase(world, material_id)
                            world.phase[y, x] = int(Phase.STATIC_SOLID) if default_phase is None else default_phase
                            if material_id <= 0 or not self._material_is_placeholder(world, material_id):
                                world.entity_id[y, x] = 0
                                world.placeholder_displaced_material[y, x] = 0
                        world.island_id[y, x] = 0
                world._mark_active_rect_runtime(*self._bbox_from_coords(coords, 0, 0))
                world.islands.pop(component_id, None)
                continue
            world.islands[component_id] = component_record
            world._mark_active_rect_runtime(*component_record.bbox)
        if gpu_apply_settled:
            used_gpu = self.gpu_pipeline.apply_uploaded_falling_island_settlements(
                world,
                int(len(island_reservations)),
            ) or used_gpu
        if gpu_apply_shifted:
            used_gpu = self.gpu_pipeline.apply_uploaded_falling_island_reservations(
                world,
                int(len(island_reservations)),
            ) or used_gpu
        return used_gpu

    def _can_seed_bridge_runtime_fast_path(self, world: "WorldEngine") -> bool:
        if not world.islands:
            return False
        for record in world.islands.values():
            x0, y0, x1, y1 = (int(value) for value in record.bbox)
            if x1 <= x0 or y1 <= y0:
                return False
            # Larger or sparse bboxes may need split/shedding; keep them on the existing GPU component path
            # until that path is fully bridge-runtime-authoritative.
            if (x1 - x0) * (y1 - y0) > 4:
                return False
        return True

    def _plan_falling_island_reservations(
        self,
        world: "WorldEngine",
        pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]],
        *,
        use_gpu: bool,
    ) -> tuple[np.ndarray, bool]:
        reservations = np.zeros((len(pending_components),), dtype=falling_island_reservation_dtype())
        if len(pending_components) == 0:
            self.gpu_pipeline.upload_falling_island_reservations(world, reservations)
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
        gpu_available = bool(use_gpu and world._gpu_pipeline_available(self.gpu_pipeline, "motion"))
        if gpu_available and gpu_island_ids:
            gpu_reservations = self.gpu_pipeline.plan_falling_island_reservations(
                world,
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
            actual_dx, actual_dy = self._resolve_island_dda_target(
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
            reservations = self.gpu_pipeline.resolve_falling_island_reservations(world, reservations)
            used_gpu = True
        else:
            world._require_cpu_oracle_backend("motion falling island reservations")
            reservations = self._resolve_falling_island_reservations(world, pending_components, reservations)
            self.gpu_pipeline.upload_falling_island_reservations(world, reservations)
        return reservations, used_gpu

    def _plan_cpu_powder_reservations(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
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
                max_dda_step = self._material_max_dda_step(world, material_id)
                desired_dx = int(np.clip(np.rint(float(velocity[0])), -max_dda_step, max_dda_step))
                desired_dy = int(np.clip(np.rint(float(velocity[1])), -max_dda_step, max_dda_step))
                reserved_target = self._resolve_powder_dda_target(world, x, y, max_dda_step)
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

    def _falling_island_reservation_order_key(self, reservation: np.ndarray | None) -> tuple[int, int, int, int]:
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
        self,
        world: "WorldEngine",
        pending_components: list[tuple[int, np.ndarray, object, tuple[int, int], tuple[float, float], tuple[int, int]]],
        reservations: np.ndarray,
    ) -> np.ndarray:
        if len(reservations) == 0:
            return reservations
        resolved = reservations.copy()
        coords_by_id = {int(component_id): coords for component_id, coords, *_ in pending_components}
        shadow_material = world.material_id.copy()
        order = sorted(range(len(resolved)), key=lambda index: self._falling_island_reservation_order_key(resolved[index]))
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
            actual_dx, actual_dy = self._resolve_island_dda_target_material(
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
            self._shadow_shift_island_material(shadow_material, coords, actual_dx, actual_dy)
        return resolved

    def _shed_falling_island_fragments(self, world: "WorldEngine", island_id: int) -> np.ndarray:
        coords = self._falling_island_coords(world, island_id)
        if len(coords) == 0:
            return coords
        fragments: list[tuple[int, int, str]] = []
        for y, x in coords:
            material_id = int(world.material_id[y, x])
            neighbor_count = self._same_island_neighbors(world, int(x), int(y), island_id)
            if neighbor_count >= self._falling_island_fragment_neighbor_threshold(world, material_id):
                continue
            powder_generation_id = self._material_powder_generation_id(world, material_id)
            if powder_generation_id <= 0:
                continue
            fragments.append((int(x), int(y), powder_generation_id))
        for x, y, powder_generation_id in fragments:
            world.set_cell_by_id(x, y, powder_generation_id, phase=Phase.POWDER, mark_dirty=False)
        if not fragments:
            return coords
        return self._falling_island_coords(world, island_id)

    def _falling_island_fragment_neighbor_threshold(self, world: "WorldEngine", material_id: int) -> int:
        if material_id > 0 and self._material_falling_island_break_kind(world, material_id) == FALLING_ISLAND_BREAK_STABLE:
            return 0
        return 2

    def _resolve_falling_island_components(
        self,
        world: "WorldEngine",
        island_id: int,
        record: "FallingIslandRecord",
        *,
        skip_shed: bool = False,
        use_gpu_components: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        component_label_texture: Any | None = None
        if use_gpu_components and world._gpu_pipeline_available(self.gpu_pipeline, "motion components"):
            component_label_texture, component_entries = self._gpu_connected_island_component_entries(
                world,
                island_id,
                record.bbox,
            )
        else:
            world._require_cpu_oracle_backend("motion component labeling")
            self._clear_stale_island_cells(world, island_id)
            coords = (
                self._falling_island_coords(world, island_id)
                if skip_shed
                else self._shed_falling_island_fragments(world, island_id)
            )
            if len(coords) == 0:
                world.islands.pop(island_id, None)
                return []
            component_entries = [
                self._component_entry_from_coords(0, component)
                for component in self._connected_island_components(coords)
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
            relabeled_on_gpu = self.gpu_pipeline.relabel_falling_island_component_texture(
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
                self._assign_split_component_cells_cpu(world, component, new_island_id)
        return result

    def _component_entry_from_coords(self, label: int, coords: np.ndarray) -> _IslandComponentEntry:
        return _IslandComponentEntry(
            label=int(label),
            coords=coords,
            bbox=self._bbox_from_coords(coords, 0, 0),
            cell_count=int(len(coords)),
        )

    def _component_entry_from_gpu_metadata(self, metadata: np.ndarray) -> _IslandComponentEntry:
        label, min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata)
        bbox = (min_x, min_y, max_x, max_y)
        coords = np.asarray(
            (
                (min_y, min_x),
                (max_y - 1, max_x - 1),
            ),
            dtype=np.int32,
        )
        if cell_count == 1:
            coords = coords[:1]
        return _IslandComponentEntry(
            label=label,
            coords=coords,
            bbox=bbox,
            cell_count=cell_count,
        )

    def _assign_split_component_cells_cpu(
        self,
        world: "WorldEngine",
        component: np.ndarray,
        island_id: int,
    ) -> None:
        for y, x in component:
            world.island_id[int(y), int(x)] = int(island_id)

    def _clear_stale_island_cells(self, world: "WorldEngine", island_id: int) -> None:
        invalid_mask = (world.island_id == island_id) & (
            (world.phase != int(Phase.FALLING_ISLAND)) | (world.material_id <= 0)
        )
        if np.any(invalid_mask):
            world.island_id[invalid_mask] = 0

    def _gpu_connected_island_components(
        self,
        world: "WorldEngine",
        island_id: int,
        bbox: tuple[int, int, int, int],
    ) -> list[np.ndarray]:
        _, entries = self._gpu_connected_island_component_entries(world, island_id, bbox)
        return [entry.coords for entry in entries]

    def _gpu_connected_island_component_entries(
        self,
        world: "WorldEngine",
        island_id: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[Any, list[_IslandComponentEntry]]:
        labels, metadata = self.gpu_pipeline.label_falling_island_component_metadata_texture(world, island_id, bbox)
        if metadata.size == 0:
            return labels, []
        return labels, [self._component_entry_from_gpu_metadata(record) for record in metadata]

    def _resolve_island_dda_shift(
        self,
        subcell_offset: tuple[float, float],
        velocity_xy: tuple[float, float],
    ) -> tuple[int, int, tuple[float, float]]:
        total_x = float(subcell_offset[0]) + float(velocity_xy[0])
        total_y = float(subcell_offset[1]) + float(velocity_xy[1])
        target_dx = int(np.clip(np.rint(total_x), -MAX_ISLAND_DDA_STEP, MAX_ISLAND_DDA_STEP))
        target_dy = int(np.clip(np.rint(total_y), -MAX_ISLAND_DDA_STEP, MAX_ISLAND_DDA_STEP))
        residual = (float(total_x - target_dx), float(total_y - target_dy))
        return target_dx, target_dy, residual

    def _resolve_island_dda_target(
        self,
        world: "WorldEngine",
        coords: np.ndarray,
        island_id: int,
        target_dx: int,
        target_dy: int,
    ) -> tuple[int, int]:
        if target_dx == 0 and target_dy == 0:
            return (0, 0)
        furthest = (0, 0)
        for dx, dy in self._dda_line_cells(0, 0, target_dx, target_dy):
            if not self._can_shift_island(world, coords, dx, dy, island_id):
                break
            furthest = (dx, dy)
        return furthest

    def _resolve_island_dda_target_material(
        self,
        material_id: np.ndarray,
        coords: np.ndarray,
        target_dx: int,
        target_dy: int,
    ) -> tuple[int, int]:
        if target_dx == 0 and target_dy == 0:
            return (0, 0)
        furthest = (0, 0)
        for dx, dy in self._dda_line_cells(0, 0, target_dx, target_dy):
            if not self._can_shift_island_material(material_id, coords, dx, dy):
                break
            furthest = (dx, dy)
        return furthest

    def _connected_island_components(self, coords: np.ndarray) -> list[np.ndarray]:
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

    def _bbox_from_coords(self, coords: np.ndarray, dx: int, dy: int) -> tuple[int, int, int, int]:
        xs = coords[:, 1] + dx
        ys = coords[:, 0] + dy
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def _same_island_neighbors(self, world: "WorldEngine", x: int, y: int, island_id: int) -> int:
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

    def _falling_island_coords(self, world: "WorldEngine", island_id: int) -> np.ndarray:
        self._clear_stale_island_cells(world, island_id)
        return np.argwhere(
            (world.island_id == island_id)
            & (world.phase == int(Phase.FALLING_ISLAND))
            & (world.material_id > 0)
        )

    def _can_shift_island(self, world: "WorldEngine", coords: np.ndarray, dx: int, dy: int, island_id: int) -> bool:
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

    def _can_shift_island_material(self, material_id: np.ndarray, coords: np.ndarray, dx: int, dy: int) -> bool:
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

    def _shadow_shift_island_material(self, material_id: np.ndarray, coords: np.ndarray, dx: int, dy: int) -> None:
        ys = coords[:, 0].astype(np.int32)
        xs = coords[:, 1].astype(np.int32)
        new_ys = ys + int(dy)
        new_xs = xs + int(dx)
        payload = material_id[ys, xs].copy()
        material_id[ys, xs] = 0
        material_id[new_ys, new_xs] = payload

    def _shift_island(self, world: "WorldEngine", coords: np.ndarray, dx: int, dy: int, island_id: int) -> None:
        ys = coords[:, 0].astype(np.int32)
        xs = coords[:, 1].astype(np.int32)
        new_ys = ys + int(dy)
        new_xs = xs + int(dx)
        old_bbox = self._bbox_from_coords(coords, 0, 0)
        new_bbox = self._bbox_from_coords(coords, dx, dy)

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

    def _move_powders(self, world: "WorldEngine", solve_cell_mask: np.ndarray) -> None:
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
                dda_target = self._resolve_powder_dda_target(world, x, y, self._material_max_dda_step(world, material_id))
                moved = False
                if dda_target is not None and dda_target != (x, y):
                    world.swap_cells(x, y, dda_target[0], dda_target[1])
                    processed[dda_target[1], dda_target[0]] = True
                    moved = True
                if moved:
                    continue
                candidates = (
                    [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]
                    if self._material_gravity(world, material_id) >= 0.0
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
        self,
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
                    world.velocity[target_y, target_x] = self._collision_response(
                        world.velocity[target_y, target_x],
                        (desired_dx, desired_dy),
                        (actual_dx, actual_dy),
                        friction=self._material_friction(world, material_id),
                        elasticity=self._material_elasticity(world, material_id),
                    )
                continue
            desired_dx = int(reservation["desired_target_xy"][0]) - x
            desired_dy = int(reservation["desired_target_xy"][1]) - y
            if desired_dx != 0 or desired_dy != 0:
                world.velocity[y, x] = self._collision_response(
                    world.velocity[y, x],
                    (desired_dx, desired_dy),
                    (0, 0),
                    friction=self._material_friction(world, material_id),
                    elasticity=self._material_elasticity(world, material_id),
                )
            else:
                world.velocity[y, x] *= 0.2

    def _resolve_powder_reservations(
        self,
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
                and self._path_is_clear_material(shadow_material, x, y, target_x, target_y)
            ):
                shadow_material[y, x] = 0
                shadow_material[target_y, target_x] = material_id
                resolved[index]["resolved_target_xy"] = np.asarray((target_x, target_y), dtype=np.int32)
                resolved[index]["resolve_state"] = POWDER_RESOLVE_DDA
                continue
            candidates = self._powder_fallback_candidates(world, x, y, material_id)
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
        self,
        world: "WorldEngine",
        x: int,
        y: int,
        material_id: int,
    ) -> list[tuple[int, int]]:
        if material_id > 0 and self._material_powder_solver_kind(world, material_id) == POWDER_SOLVER_SUSPENDED:
            return []
        if material_id > 0 and self._material_gravity(world, material_id) >= 0.0:
            return [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]
        return [(x, y - 1), (x - 1, y - 1), (x + 1, y - 1)]

    def _material_table_row(self, world: "WorldEngine", material_id: int) -> np.void | None:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None or material_id < 0 or material_id >= int(material_table.shape[0]):
            return None
        row = material_table[material_id]
        if int(row["name_hash"]) == 0:
            return None
        return row

    def _material_scalar(self, world: "WorldEngine", material_id: int, field: str, fallback: np.ndarray) -> float:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return float(row[field])
        if 0 <= material_id < fallback.shape[0]:
            return float(fallback[material_id])
        return 0.0

    def _material_int(self, world: "WorldEngine", material_id: int, field: str, fallback: np.ndarray) -> int:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return int(row[field])
        if 0 <= material_id < fallback.shape[0]:
            return int(fallback[material_id])
        return 0

    def _material_scalar_field(
        self,
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

    def _material_default_phase(self, world: "WorldEngine", material_id: int) -> int | None:
        row = self._material_table_row(world, material_id)
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

    def _material_is_placeholder(self, world: "WorldEngine", material_id: int) -> bool:
        row = self._material_table_row(world, material_id)
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

    def _material_powder_generation_id(self, world: "WorldEngine", material_id: int) -> int:
        if material_id <= 0:
            return 0
        return self._material_int(world, material_id, "powder_generation_id", world.material_powder_generation_id)

    def _material_falling_island_break_kind(self, world: "WorldEngine", material_id: int) -> int:
        return self._material_int(
            world,
            material_id,
            "falling_island_break_kind_id",
            world.material_falling_island_break_kind,
        )

    def _material_max_dda_step(self, world: "WorldEngine", material_id: int) -> int:
        return max(0, self._material_int(world, material_id, "max_dda_step", world.material_max_dda_step))

    def _material_powder_solver_kind(self, world: "WorldEngine", material_id: int) -> int:
        return self._material_int(world, material_id, "powder_solver_kind_id", world.material_powder_solver_kind)

    def _material_gravity(self, world: "WorldEngine", material_id: int) -> float:
        return self._material_scalar(world, material_id, "gravity_scale", world.material_gravity)

    def _material_friction(self, world: "WorldEngine", material_id: int) -> float:
        return self._material_scalar(world, material_id, "friction", world.material_friction)

    def _material_elasticity(self, world: "WorldEngine", material_id: int) -> float:
        return self._material_scalar(world, material_id, "elasticity", world.material_elasticity)

    def _resolve_powder_dda_target(
        self,
        world: "WorldEngine",
        x: int,
        y: int,
        max_dda_step: int,
    ) -> tuple[int, int] | None:
        velocity = world.velocity[y, x]
        max_step = max(0, int(max_dda_step))
        if max_step <= 0:
            return None
        desired_dx = int(np.clip(np.rint(float(velocity[0])), -max_step, max_step))
        desired_dy = int(np.clip(np.rint(float(velocity[1])), -max_step, max_step))
        if desired_dx == 0 and desired_dy == 0:
            return None
        target_x = x + desired_dx
        target_y = y + desired_dy
        furthest_free = (x, y)
        for cell_x, cell_y in self._dda_line_cells(x, y, target_x, target_y):
            if not world.in_bounds(cell_x, cell_y):
                break
            if world.material_id[cell_y, cell_x] != 0:
                break
            furthest_free = (cell_x, cell_y)
        return furthest_free

    def _path_is_clear(self, world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> bool:
        for cell_x, cell_y in self._dda_line_cells(x0, y0, x1, y1):
            if not world.in_bounds(cell_x, cell_y):
                return False
            if world.material_id[cell_y, cell_x] != 0:
                return False
        return True

    def _path_is_clear_material(self, material_id: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
        height, width = material_id.shape
        for cell_x, cell_y in self._dda_line_cells(x0, y0, x1, y1):
            if cell_x < 0 or cell_y < 0 or cell_x >= width or cell_y >= height:
                return False
            if material_id[cell_y, cell_x] != 0:
                return False
        return True

    def _dda_line_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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
