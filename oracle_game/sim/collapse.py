from __future__ import annotations

from collections import deque

import numpy as np

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline
from oracle_game.sim.gpu_collapse_dirty import has_pending_collapse_structure_dirty_tiles
from oracle_game.types import CollapseBehavior, FallingIslandRecord, Phase
from oracle_game.sim.cpu_base import material_table_row


COLLAPSE_RUNTIME_MASK_RESOURCES = (
    "collapse_structural_mask",
    "collapse_support_seed_mask",
    "collapse_supported_mask",
    "collapse_unsupported_mask",
    "collapse_delayed_pending_mask",
    "collapse_immune_unsupported_mask",
    "collapse_collapsed_cell_mask",
)
COLLAPSE_COMPONENT_SNAPSHOT_RESOURCES = (
    "collapse_component_label",
    "island_id",
    "island_runtime",
    "island_runtime_count",
)
COLLAPSE_RUNTIME_SNAPSHOT_RESOURCES = COLLAPSE_RUNTIME_MASK_RESOURCES + COLLAPSE_COMPONENT_SNAPSHOT_RESOURCES


class CollapseSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUCollapsePipeline()
        self.last_backend = "idle"
        self.reset_runtime_state()

    def step(self, world: "WorldEngine") -> None:
        with self.gpu_pipeline._profile_pass(world, "solver_runtime_reset"):
            self.reset_runtime_state(world)
        pending_regions: list[tuple[tuple[int, int, int, int], bool]] = []

        def sparse_bbox_merge(
            region: tuple[int, int, int, int],
            other: tuple[int, int, int, int],
        ) -> bool:
            rx0, ry0, rx1, ry1 = region
            ox0, oy0, ox1, oy1 = other
            area = max(0, rx1 - rx0) * max(0, ry1 - ry0)
            other_area = max(0, ox1 - ox0) * max(0, oy1 - oy0)
            ix0 = max(rx0, ox0)
            iy0 = max(ry0, oy0)
            ix1 = min(rx1, ox1)
            iy1 = min(ry1, oy1)
            overlap_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union_cell_area = area + other_area - overlap_area
            bbox_area = (max(rx1, ox1) - min(rx0, ox0)) * (max(ry1, oy1) - min(ry0, oy0))
            return bbox_area > union_cell_area

        def append_region(region: tuple[int, int, int, int], from_deferred: bool) -> None:
            rx0, ry0, rx1, ry1 = (int(value) for value in region)
            if rx0 >= rx1 or ry0 >= ry1:
                return
            merged_deferred = bool(from_deferred)
            index = 0
            while index < len(pending_regions):
                (ox0, oy0, ox1, oy1), existing_deferred = pending_regions[index]
                overlaps = rx0 <= ox1 and rx1 >= ox0 and ry0 <= oy1 and ry1 >= oy0
                if not overlaps:
                    index += 1
                    continue
                if (
                    from_deferred
                    and existing_deferred
                    and sparse_bbox_merge((rx0, ry0, rx1, ry1), (ox0, oy0, ox1, oy1))
                ):
                    index += 1
                    continue
                rx0 = min(rx0, ox0)
                ry0 = min(ry0, oy0)
                rx1 = max(rx1, ox1)
                ry1 = max(ry1, oy1)
                merged_deferred = merged_deferred and bool(existing_deferred)
                pending_regions.pop(index)
            pending_regions.append(((rx0, ry0, rx1, ry1), merged_deferred))

        with self.gpu_pipeline._profile_pass(world, "solver_region_prepare"):
            dirty_regions = list(world.collapse_dirty_regions)
            deferred_regions = list(world.collapse_deferred_regions)
            gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "collapse")
            gpu_dirty_tile_queue_pending = bool(
                gpu_available and has_pending_collapse_structure_dirty_tiles(world)
            )
            world.collapse_dirty_regions.clear()
            world.collapse_deferred_regions.clear()
            self.last_dirty_region_count_before = int(
                len(dirty_regions) + len(deferred_regions) + (1 if gpu_dirty_tile_queue_pending else 0)
            )
            if dirty_regions or deferred_regions:
                for region in dirty_regions:
                    append_region(region, False)
                for region in deferred_regions:
                    append_region(region, True)
        if not dirty_regions and not deferred_regions:
            if gpu_dirty_tile_queue_pending:
                self.last_solve_region_count += 1
                self._solve_formal_gpu_dirty_tile_queue(world)
                return
            self.last_backend = "idle"
            return
        for (x0, y0, x1, y1), processing_deferred_region in pending_regions:
            self.last_solve_region_count += 1
            self._solve_region(
                world,
                x0,
                y0,
                x1,
                y1,
                formal_region_from_deferred=processing_deferred_region,
            )
        if gpu_dirty_tile_queue_pending:
            self.last_solve_region_count += 1
            self._solve_formal_gpu_dirty_tile_queue(world)

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def _solve_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        formal_region_from_deferred: bool = False,
    ) -> None:
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "collapse")
        formal_gpu_frame = bool(
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and getattr(world, "_world_simulation_frame_active", False)
        )
        structural_world: np.ndarray | None = None
        if gpu_available and formal_gpu_frame:
            self._solve_formal_gpu_region(
                world,
                x0,
                y0,
                x1,
                y1,
                formal_region_from_deferred=formal_region_from_deferred,
            )
            return
        elif gpu_available:
            structural_world = self.gpu_pipeline.classify_world_structural_mask(world)
            x0, y0, x1, y1 = self._expand_region_to_component_gpu(world, structural_world, x0, y0, x1, y1)
        else:
            world._require_cpu_oracle_backend("collapse")
            structural_world = self._world_structural_mask(world)
            x0, y0, x1, y1 = self._expand_region_to_component(structural_world, x0, y0, x1, y1)
        self.last_solve_region_mask[y0:y1, x0:x1] = True
        if gpu_available:
            structural, support_seed, behavior_region = self.gpu_pipeline.classify_region(world, x0, y0, x1, y1)
        else:
            assert structural_world is not None
            structural = structural_world[y0:y1, x0:x1]
            support_seed = self._support_seed_mask(world, structural, x0, y0, x1, y1)
            material_region = world.material_id[y0:y1, x0:x1]
            behavior_region = self._material_int_field(
                world,
                material_region,
                "collapse_behavior_id",
                world.material_collapse_behavior,
            )
        if structural.size == 0 or not structural.any():
            return
        self.last_structural_mask[y0:y1, x0:x1] |= structural
        self.last_support_seed_mask[y0:y1, x0:x1] |= support_seed
        if gpu_available:
            unsupported = self.gpu_pipeline.solve_region(world, structural, support_seed, x0=x0, y0=y0)
            self.last_backend = "gpu"
        else:
            supported = self._supported_mask_cpu(structural, support_seed)
            unsupported = structural & ~supported
            self.last_backend = "cpu"
        supported = structural & ~unsupported
        self.last_supported_mask[y0:y1, x0:x1] |= supported
        self.last_unsupported_mask[y0:y1, x0:x1] |= unsupported
        if gpu_available:
            delayed_pending, immune_unsupported, collapse_now = self.gpu_pipeline.resolve_unsupported_outcomes(
                world,
                unsupported,
                behavior_region,
                x0,
                y0,
            )
        else:
            delayed_pending, immune_unsupported, collapse_now = self._resolve_unsupported_outcomes_cpu(
                world,
                unsupported,
                behavior_region,
                x0,
                y0,
        )
        self.last_delayed_pending_mask[y0:y1, x0:x1] |= delayed_pending
        self.last_immune_unsupported_mask[y0:y1, x0:x1] |= immune_unsupported
        if gpu_available:
            delayed_metadata = self.gpu_pipeline.summarize_labeled_components(
                world,
                delayed_pending.astype(np.int32, copy=False),
                np.asarray([1], dtype=np.int32),
                x0,
                y0,
            )
            self._queue_deferred_metadata_region(world, delayed_metadata[0] if delayed_metadata.size else None)
        elif np.any(delayed_pending):
            self._queue_deferred_mask_region(world, delayed_pending, x0, y0)
        if gpu_available:
            self._collapse_unsupported_components_gpu(world, collapse_now, x0, y0)
        else:
            self._collapse_unsupported_components(world, collapse_now, x0, y0)

    def _solve_formal_gpu_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        formal_region_from_deferred: bool,
    ) -> None:
        with self.gpu_pipeline._profile_pass(world, "formal_region_prepare"):
            if formal_region_from_deferred:
                formal_event_region = self._align_formal_dirty_region(world, x0, y0, x1, y1)
            else:
                formal_event_region = self._expand_formal_dirty_region(world, x0, y0, x1, y1)
            solve_region = self.gpu_pipeline.expand_region_to_component_bbox(world, *formal_event_region)
            solve_x0, solve_y0, solve_x1, solve_y1 = solve_region
            resource_region = self._formal_connected_resource_region(world, solve_x0, solve_y0, solve_x1, solve_y1)
            mark_x0 = max(0, solve_x0 - 1)
            mark_y0 = max(0, solve_y0 - 1)
            mark_x1 = min(int(world.width), solve_x1 + 1)
            mark_y1 = min(int(world.height), solve_y1 + 1)
            self.last_solve_region_mask[mark_y0:mark_y1, mark_x0:mark_x1] = True

            self.gpu_pipeline.clear_formal_deferred_region_requests(world)
            self.last_backend = "gpu"
        component_capacity = self.gpu_pipeline.execute_formal_connected_expansion(
            world,
            formal_event_region,
            resource_region=resource_region,
        )
        motion_pipeline = getattr(getattr(world, "motion_solver", None), "gpu_pipeline", None)
        if motion_pipeline is not None:
            motion_pipeline.last_published_island_runtime_capacity = int(component_capacity)
        has_delayed_behavior = bool(np.any(world.material_collapse_behavior == int(CollapseBehavior.DELAYED)))

        if not formal_region_from_deferred and has_delayed_behavior:
            world.collapse_deferred_regions.append((solve_x0, solve_y0, solve_x1, solve_y1))

    def _formal_connected_resource_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        world_width = int(world.width)
        world_height = int(world.height)
        rx0 = max(0, min(world_width, int(x0)))
        ry0 = max(0, min(world_height, int(y0)))
        rx1 = max(0, min(world_width, int(x1)))
        ry1 = max(0, min(world_height, int(y1)))
        touches_x_edge = rx0 <= 0 or rx1 >= world_width
        touches_y_edge = ry0 <= 0 or ry1 >= world_height
        if not touches_x_edge and not touches_y_edge:
            return (rx0, ry0, rx1, ry1)

        tile_size = max(1, int(getattr(world.active, "tile_size", 32)))
        orthogonal_margin = max(1, tile_size + tile_size // 2)
        seed_x0, seed_y0, seed_x1, seed_y1 = rx0, ry0, rx1, ry1

        def bounded_span(lo: int, hi: int, limit: int) -> tuple[int, int]:
            if limit <= 0:
                return (0, 0)
            span = max(1, int(hi) - int(lo))
            expanded_lo = max(0, int(lo) - orthogonal_margin)
            expanded_hi = min(int(limit), int(hi) + orthogonal_margin)
            if expanded_lo == 0 and expanded_hi == int(limit) and span < int(limit):
                guard = max(1, min(int(limit) - 1, max(1, tile_size // 2)))
                target_span = min(int(limit) - guard, max(span, span + orthogonal_margin * 2))
                center = (int(lo) + int(hi)) // 2
                expanded_lo = max(0, min(int(limit) - target_span, center - target_span // 2))
                expanded_hi = expanded_lo + target_span
            return (expanded_lo, expanded_hi)

        if touches_x_edge:
            rx0, rx1 = 0, world_width
            ry0, ry1 = bounded_span(seed_y0, seed_y1, world_height)
        if touches_y_edge:
            ry0, ry1 = 0, world_height
            rx0, rx1 = bounded_span(seed_x0, seed_x1, world_width)
        if touches_x_edge and touches_y_edge:
            rx0, rx1 = 0, world_width
            ry0, ry1 = bounded_span(seed_y0, seed_y1, world_height)
            if ry0 == 0 and ry1 == world_height and world_width > 1:
                rx0, rx1 = bounded_span(seed_x0, seed_x1, world_width)

        return (rx0, ry0, rx1, ry1)

    def _solve_formal_gpu_dirty_tile_queue(self, world: "WorldEngine") -> None:
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "collapse")
        formal_gpu_frame = bool(
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and getattr(world, "_world_simulation_frame_active", False)
        )
        if not formal_gpu_frame:
            return
        with self.gpu_pipeline._profile_pass(world, "formal_dirty_tile_queue_prepare"):
            self.gpu_pipeline.clear_formal_deferred_region_requests(world)
            self.last_backend = "gpu"
        component_capacity = self.gpu_pipeline.execute_formal_connected_dirty_tile_queue(world)
        motion_pipeline = getattr(getattr(world, "motion_solver", None), "gpu_pipeline", None)
        if motion_pipeline is not None:
            motion_pipeline.last_published_island_runtime_capacity = int(component_capacity)

    def _resolve_unsupported_outcomes_cpu(
        self,
        world: "WorldEngine",
        unsupported: np.ndarray,
        behavior_region: np.ndarray,
        x0: int,
        y0: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y1 = y0 + unsupported.shape[0]
        x1 = x0 + unsupported.shape[1]
        pending_region = world.collapse_delay_pending[y0:y1, x0:x1]
        delayed_pending = unsupported & (behavior_region == int(CollapseBehavior.DELAYED)) & ~pending_region
        immune_unsupported = unsupported & (behavior_region == int(CollapseBehavior.IMMUNE))
        collapse_now = unsupported & (
            (behavior_region == int(CollapseBehavior.FALLING_ISLAND))
            | ((behavior_region == int(CollapseBehavior.DELAYED)) & pending_region)
        )
        pending_region[:] = delayed_pending
        return delayed_pending, immune_unsupported, collapse_now

    def _world_structural_mask(self, world: "WorldEngine") -> np.ndarray:
        material_id = world.material_id
        return (
            (material_id != 0)
            & (world.phase != int(Phase.FALLING_ISLAND))
            & self._material_bool_field(world, material_id, "is_structural", world.material_is_structural)
        )

    def _expand_region_to_component(
        self,
        structural_world: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        x0 = max(0, x0 - 1)
        y0 = max(0, y0 - 1)
        x1 = min(structural_world.shape[1], x1 + 1)
        y1 = min(structural_world.shape[0], y1 + 1)
        if x0 >= x1 or y0 >= y1:
            return (x0, y0, x1, y1)
        while True:
            changed = False
            if x0 > 0 and np.any(structural_world[y0:y1, x0] & structural_world[y0:y1, x0 - 1]):
                x0 -= 1
                changed = True
            if x1 < structural_world.shape[1] and np.any(structural_world[y0:y1, x1 - 1] & structural_world[y0:y1, x1]):
                x1 += 1
                changed = True
            if y0 > 0 and np.any(structural_world[y0, x0:x1] & structural_world[y0 - 1, x0:x1]):
                y0 -= 1
                changed = True
            if y1 < structural_world.shape[0] and np.any(structural_world[y1 - 1, x0:x1] & structural_world[y1, x0:x1]):
                y1 += 1
                changed = True
            if not changed:
                return (x0, y0, x1, y1)

    def _align_formal_dirty_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        expand_tile_count: int = 0,
    ) -> tuple[int, int, int, int]:
        tile_size = max(1, int(getattr(world.active, "tile_size", 32)))
        width = int(world.width)
        height = int(world.height)
        expand = max(0, int(expand_tile_count)) * tile_size
        x0 = max(0, int(x0) - expand)
        y0 = max(0, int(y0) - expand)
        x1 = min(width, int(x1) + expand)
        y1 = min(height, int(y1) + expand)
        x0 = max(0, (x0 // tile_size) * tile_size)
        y0 = max(0, (y0 // tile_size) * tile_size)
        x1 = min(width, ((x1 + tile_size - 1) // tile_size) * tile_size)
        y1 = min(height, ((y1 + tile_size - 1) // tile_size) * tile_size)
        if x0 >= x1 or y0 >= y1:
            return (0, 0, width, height)
        return (x0, y0, x1, y1)

    def _expand_formal_dirty_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        return self._align_formal_dirty_region(world, x0, y0, x1, y1, expand_tile_count=1)

    def _expand_region_to_component_gpu(
        self,
        world: "WorldEngine",
        structural_world: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        x0 = max(0, x0 - 1)
        y0 = max(0, y0 - 1)
        x1 = min(structural_world.shape[1], x1 + 1)
        y1 = min(structural_world.shape[0], y1 + 1)
        if x0 >= x1 or y0 >= y1:
            return (x0, y0, x1, y1)
        seed_mask = np.zeros_like(structural_world, dtype=np.bool_)
        seed_mask[y0:y1, x0:x1] = structural_world[y0:y1, x0:x1]
        unsupported = self.gpu_pipeline.solve_region(world, structural_world, seed_mask, x0=0, y0=0)
        connected_to_dirty_region = structural_world & ~unsupported
        metadata = self.gpu_pipeline.summarize_labeled_components(
            world,
            connected_to_dirty_region.astype(np.int32, copy=False),
            np.asarray([1], dtype=np.int32),
            0,
            0,
        )
        if metadata.size == 0:
            return (x0, y0, x1, y1)
        min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata[0])
        if cell_count <= 0:
            return (x0, y0, x1, y1)
        return (
            min(x0, min_x),
            min(y0, min_y),
            max(x1, max_x),
            max(y1, max_y),
        )

    def _support_seed_mask(
        self,
        world: "WorldEngine",
        structural: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> np.ndarray:
        material_region = world.material_id[y0:y1, x0:x1]
        support_anchor = self._material_bool_field(
            world,
            material_region,
            "is_support_anchor",
            world.material_is_support_anchor,
        )
        world_y = np.arange(y0, y1, dtype=np.int32)[:, None]
        on_world_floor = world_y == world.height - 1
        return structural & (support_anchor | on_world_floor)

    def _supported_mask_cpu(self, structural: np.ndarray, support_seed: np.ndarray) -> np.ndarray:
        supported = np.zeros_like(structural, dtype=bool)
        queue = deque((int(x), int(y)) for y, x in np.argwhere(support_seed))
        for x, y in queue:
            supported[y, x] = True
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or ny >= structural.shape[0] or nx >= structural.shape[1]:
                    continue
                if supported[ny, nx] or not structural[ny, nx]:
                    continue
                supported[ny, nx] = True
                queue.append((nx, ny))
        return supported

    def _collapse_unsupported_components(
        self,
        world: "WorldEngine",
        unsupported: np.ndarray,
        x0: int,
        y0: int,
    ) -> None:
        seen = np.zeros_like(unsupported, dtype=bool)
        for local_y in range(unsupported.shape[0]):
            for local_x in range(unsupported.shape[1]):
                if not unsupported[local_y, local_x] or seen[local_y, local_x]:
                    continue
                component: list[tuple[int, int]] = []
                queue = deque([(local_x, local_y)])
                seen[local_y, local_x] = True
                while queue:
                    cx, cy = queue.popleft()
                    component.append((x0 + cx, y0 + cy))
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if nx < 0 or ny < 0 or ny >= unsupported.shape[0] or nx >= unsupported.shape[1]:
                            continue
                        if seen[ny, nx] or not unsupported[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        queue.append((nx, ny))
                self._collapse_component(world, component)

    def _collapse_unsupported_components_gpu(
        self,
        world: "WorldEngine",
        unsupported: np.ndarray,
        x0: int,
        y0: int,
    ) -> None:
        component_labels, component_island_ids, component_metadata = self.gpu_pipeline.materialize_component_mask(
            world,
            unsupported,
            x0,
            y0,
        )
        if component_labels.size == 0:
            return
        self.last_collapsed_cell_mask[y0 : y0 + unsupported.shape[0], x0 : x0 + unsupported.shape[1]] |= unsupported
        self._record_gpu_collapsed_components(world, component_island_ids, component_metadata)

    def _record_gpu_collapsed_components(
        self,
        world: "WorldEngine",
        component_island_ids: np.ndarray,
        component_metadata: np.ndarray,
    ) -> None:
        for island_id, metadata in zip(component_island_ids.tolist(), component_metadata.tolist(), strict=True):
            min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata)
            if cell_count <= 0:
                continue
            world.islands[int(island_id)] = FallingIslandRecord(
                island_id=int(island_id),
                bbox=(min_x, min_y, max_x, max_y),
                velocity_xy=(0.0, 0.0),
            )
            self.last_collapsed_components.append(
                {
                    "island_id": int(island_id),
                    "bbox": (int(min_x), int(min_y), int(max_x), int(max_y)),
                    "world_bbox": list(world._buffer_bbox_to_world_bbox((int(min_x), int(min_y), int(max_x), int(max_y)))),
                    "cell_count": int(cell_count),
                }
            )
            world._mark_active_rect_runtime(min_x, min_y, max_x, max_y)

    def _summarize_collapsed_components_cpu(
        self,
        labels: np.ndarray,
        component_labels: np.ndarray,
        x0: int,
        y0: int,
    ) -> np.ndarray:
        metadata = np.zeros((int(component_labels.size), 5), dtype=np.int32)
        for index, label in enumerate(component_labels.tolist()):
            local_ys, local_xs = np.nonzero(labels == int(label))
            if local_ys.size == 0:
                continue
            metadata[index] = np.asarray(
                (
                    x0 + int(local_xs.min()),
                    y0 + int(local_ys.min()),
                    x0 + int(local_xs.max()) + 1,
                    y0 + int(local_ys.max()) + 1,
                    int(local_ys.size),
                ),
                dtype=np.int32,
            )
        return metadata

    def _queue_deferred_mask_region(
        self,
        world: "WorldEngine",
        delayed_mask: np.ndarray,
        x0: int,
        y0: int,
    ) -> None:
        ys, xs = np.nonzero(delayed_mask)
        if ys.size == 0:
            return
        world.collapse_deferred_regions.append(
            (
                max(0, x0 + int(xs.min()) - 1),
                max(0, y0 + int(ys.min()) - 1),
                min(world.width, x0 + int(xs.max()) + 2),
                min(world.height, y0 + int(ys.max()) + 2),
            )
        )

    def _queue_deferred_metadata_region(
        self,
        world: "WorldEngine",
        metadata: np.ndarray | None,
    ) -> None:
        if metadata is None:
            return
        min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata)
        if cell_count <= 0:
            return
        world.collapse_deferred_regions.append(
            (
                max(0, min_x - 1),
                max(0, min_y - 1),
                min(world.width, max_x + 1),
                min(world.height, max_y + 1),
            )
        )

    def _collapse_component(self, world: "WorldEngine", component: list[tuple[int, int]]) -> None:
        if not component:
            return
        island_id = world.allocate_island_id()
        min_x = min(x for x, _ in component)
        min_y = min(y for _, y in component)
        max_x = max(x for x, _ in component) + 1
        max_y = max(y for _, y in component) + 1
        for x, y in component:
            self.last_collapsed_cell_mask[y, x] = True
        for x, y in component:
            material_id = int(world.material_id[y, x])
            collapse_generation_id = self._material_collapse_generation_id(world, material_id)
            if collapse_generation_id > 0:
                world.set_cell_by_id(x, y, collapse_generation_id, phase=Phase.FALLING_ISLAND, mark_dirty=False)
            else:
                world.phase[y, x] = Phase.FALLING_ISLAND
                world.entity_id[y, x] = 0
                world.placeholder_displaced_material[y, x] = 0
            world.island_id[y, x] = island_id
        world.islands[island_id] = FallingIslandRecord(
            island_id=island_id,
            bbox=(min_x, min_y, max_x, max_y),
            velocity_xy=(0.0, 0.0),
        )
        self.last_collapsed_components.append(
            {
                "island_id": int(island_id),
                "bbox": (int(min_x), int(min_y), int(max_x), int(max_y)),
                "world_bbox": list(world._buffer_bbox_to_world_bbox((int(min_x), int(min_y), int(max_x), int(max_y)))),
                "cell_count": int(len(component)),
            }
        )
        world._mark_active_rect_runtime(min_x, min_y, max_x, max_y)

    def _material_table_row(self, world: "WorldEngine", material_id: int) -> np.void | None:
        # Delegated to the shared helper (formerly duplicated verbatim here).
        return material_table_row(world, material_id)

    def _material_bool_field(
        self,
        world: "WorldEngine",
        material_ids: np.ndarray,
        field: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return fallback[material_ids].astype(np.bool_, copy=True)
        values = np.zeros(material_ids.shape, dtype=np.bool_)
        valid_mask = (
            (material_ids >= 0)
            & (material_ids < int(material_table.shape[0]))
            & (material_table["name_hash"][np.clip(material_ids, 0, max(0, int(material_table.shape[0]) - 1))] != 0)
        )
        if np.any(valid_mask):
            values[valid_mask] = material_table[field][material_ids[valid_mask]] != 0
        return values

    def _material_int_field(
        self,
        world: "WorldEngine",
        material_ids: np.ndarray,
        field: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        values = fallback[material_ids].astype(np.int32, copy=True)
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return values
        valid_mask = (
            (material_ids >= 0)
            & (material_ids < int(material_table.shape[0]))
            & (material_table["name_hash"][np.clip(material_ids, 0, max(0, int(material_table.shape[0]) - 1))] != 0)
        )
        if np.any(valid_mask):
            values[valid_mask] = material_table[field][material_ids[valid_mask]].astype(np.int32, copy=False)
        return values

    def _material_collapse_generation_id(self, world: "WorldEngine", material_id: int) -> int:
        if material_id <= 0:
            return 0
        row = self._material_table_row(world, material_id)
        if row is not None:
            return int(row["collapse_generation_id"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            if not shadow_material.collapse_generation:
                return 0
            return int(world._shadow_material_id_by_name(shadow_material.collapse_generation))
        if world._shadow_has_table_payload("materials"):
            return 0
        if 0 <= material_id < world.material_collapse_generation_id.shape[0]:
            return int(world.material_collapse_generation_id[material_id])
        return 0

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        cell_shape = (0, 0) if world is None else (world.height, world.width)
        if world is not None:
            world.bridge.clear_gpu_authoritative(
                "collapse_structural_mask",
                "collapse_support_seed_mask",
                "collapse_supported_mask",
                "collapse_unsupported_mask",
                "collapse_delayed_pending_mask",
                "collapse_immune_unsupported_mask",
                "collapse_collapsed_cell_mask",
                "collapse_component_label",
            )
        self.last_solve_region_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_structural_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_support_seed_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_supported_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_unsupported_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_delayed_pending_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_immune_unsupported_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_collapsed_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_collapsed_components: list[dict[str, int | tuple[int, int, int, int]]] = []
        self.last_dirty_region_count_before = 0
        self.last_solve_region_count = 0

    def _runtime_mask_snapshot(
        self,
        world: "WorldEngine" | None,
        resource_name: str,
        fallback: np.ndarray,
        *,
        allow_gpu_sync_readback: bool,
    ) -> tuple[np.ndarray, bool, bool]:
        mask = fallback.copy()
        if world is None or getattr(world, "simulation_backend", "") != "gpu":
            return mask, False, False
        bridge = world.bridge
        if resource_name not in bridge.gpu_authoritative_resources:
            return mask, False, False
        if not allow_gpu_sync_readback:
            return mask, False, True
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError(f"GPU collapse runtime snapshot requires bridge buffer {resource_name!r}")
        cell_count = int(world.width) * int(world.height)
        raw = bridge.buffers[resource_name].read(size=cell_count * 4)
        gpu_mask = np.frombuffer(raw, dtype=np.int32, count=cell_count).reshape((world.height, world.width)) != 0
        return gpu_mask & self.last_solve_region_mask, True, False

    def _gpu_collapsed_components_snapshot(
        self,
        world: "WorldEngine",
        collapsed_cell_mask: np.ndarray,
    ) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], bool]:
        if getattr(world, "simulation_backend", "") != "gpu":
            return [], False
        bridge = world.bridge
        required = {"collapse_collapsed_cell_mask", "collapse_component_label"}
        if not required.issubset(bridge.gpu_authoritative_resources):
            return [], False
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse component snapshot requires bridge GPU resources")
        cell_count = int(world.width) * int(world.height)
        collapsed = np.asarray(collapsed_cell_mask, dtype=np.bool_)
        if collapsed.shape != (world.height, world.width) or not bool(np.any(collapsed)):
            return [], False
        labels = np.frombuffer(
            bridge.buffers["collapse_component_label"].read(size=cell_count * np.dtype(np.int32).itemsize),
            dtype=np.int32,
            count=cell_count,
        ).reshape((world.height, world.width))
        sync_readback_performed = True
        component_mask = collapsed & (labels > 0)
        if not bool(np.any(component_mask)):
            return [], sync_readback_performed

        island_ids = np.zeros((world.height, world.width), dtype=np.int32)
        if "island_id" in bridge.gpu_authoritative_resources:
            island_ids = np.frombuffer(
                bridge.buffers["island_id"].read(size=cell_count * np.dtype(np.int32).itemsize),
                dtype=np.int32,
                count=cell_count,
            ).reshape((world.height, world.width))
        runtime_island_ids: list[int] = []
        if {"island_runtime", "island_runtime_count"}.issubset(bridge.gpu_authoritative_resources):
            runtime_count = int(
                np.frombuffer(
                    bridge.buffers["island_runtime_count"].read(size=np.dtype(np.int32).itemsize),
                    dtype=np.int32,
                    count=1,
                )[0]
            )
            if runtime_count > 0:
                runtime_records = np.frombuffer(
                    bridge.buffers["island_runtime"].read(size=runtime_count * ISLAND_RUNTIME_DTYPE.itemsize),
                    dtype=ISLAND_RUNTIME_DTYPE,
                    count=runtime_count,
                )
                runtime_island_ids = sorted(
                    int(value) for value in np.unique(runtime_records["island_id"]) if int(value) > 0
                )

        components: list[dict[str, int | tuple[int, int, int, int]]] = []
        labels_in_order = sorted(int(value) for value in np.unique(labels[component_mask]) if int(value) > 0)
        for index, label in enumerate(labels_in_order):
            label_mask = component_mask & (labels == int(label))
            ys, xs = np.nonzero(label_mask)
            if ys.size == 0:
                continue
            overlapping_ids = sorted(int(value) for value in np.unique(island_ids[label_mask]) if int(value) > 0)
            island_id = (
                overlapping_ids[0]
                if overlapping_ids
                else runtime_island_ids[index]
                if index < len(runtime_island_ids)
                else index + 1
            )
            components.append(
                {
                    "island_id": int(island_id),
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    "cell_count": int(xs.size),
                }
            )
        return components, sync_readback_performed

    def _gpu_authoritative_runtime_resources(self, world: "WorldEngine" | None) -> list[str]:
        if world is None or getattr(world, "simulation_backend", "") != "gpu":
            return []
        bridge = world.bridge
        return [
            resource_name
            for resource_name in COLLAPSE_RUNTIME_SNAPSHOT_RESOURCES
            if resource_name in bridge.gpu_authoritative_resources
        ]

    def runtime_snapshot(
        self,
        world: "WorldEngine" | None = None,
        *,
        allow_gpu_sync_readback: bool = False,
    ) -> dict[str, object]:
        sync_readback_performed = False
        stale_resources: list[str] = []

        structural_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_structural_mask",
            self.last_structural_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_structural_mask")
        support_seed_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_support_seed_mask",
            self.last_support_seed_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_support_seed_mask")
        supported_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_supported_mask",
            self.last_supported_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_supported_mask")
        unsupported_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_unsupported_mask",
            self.last_unsupported_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_unsupported_mask")
        delayed_pending_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_delayed_pending_mask",
            self.last_delayed_pending_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_delayed_pending_mask")
        immune_unsupported_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_immune_unsupported_mask",
            self.last_immune_unsupported_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_immune_unsupported_mask")
        collapsed_cell_mask, did_read, is_stale = self._runtime_mask_snapshot(
            world,
            "collapse_collapsed_cell_mask",
            self.last_collapsed_cell_mask,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        sync_readback_performed = sync_readback_performed or did_read
        if is_stale:
            stale_resources.append("collapse_collapsed_cell_mask")

        gpu_authoritative_resources = self._gpu_authoritative_runtime_resources(world)
        collapsed_components = [dict(component) for component in self.last_collapsed_components]
        if world is not None and not collapsed_components:
            if allow_gpu_sync_readback:
                collapsed_components, did_read = self._gpu_collapsed_components_snapshot(world, collapsed_cell_mask)
                sync_readback_performed = sync_readback_performed or did_read
            elif "collapse_component_label" in gpu_authoritative_resources:
                stale_resources.append("collapse_component_label")
        snapshot_stale = bool(stale_resources)
        snapshot_source = (
            "synchronous_gpu_readback"
            if sync_readback_performed
            else "cpu_shadow"
            if gpu_authoritative_resources
            else "cpu"
        )
        return {
            "backend": self.last_backend,
            "gpu_authoritative": bool(gpu_authoritative_resources),
            "gpu_authoritative_resources": gpu_authoritative_resources,
            "snapshot_source": snapshot_source,
            "snapshot_stale": snapshot_stale,
            "gpu_authoritative_snapshot_stale": snapshot_stale,
            "stale_resources": stale_resources,
            "sync_readback_required": snapshot_stale,
            "sync_readback_performed": bool(sync_readback_performed),
            "dirty_region_count_before": int(self.last_dirty_region_count_before),
            "solve_region_count": int(self.last_solve_region_count),
            "solve_region_mask": self.last_solve_region_mask.copy(),
            "structural_mask": structural_mask,
            "support_seed_mask": support_seed_mask,
            "supported_mask": supported_mask,
            "unsupported_mask": unsupported_mask,
            "delayed_pending_mask": delayed_pending_mask,
            "immune_unsupported_mask": immune_unsupported_mask,
            "collapsed_cell_mask": collapsed_cell_mask,
            "collapsed_components": collapsed_components,
        }
