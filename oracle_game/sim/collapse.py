from __future__ import annotations

from collections import deque

import numpy as np

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline
from oracle_game.types import CollapseBehavior, FallingIslandRecord, Phase


class CollapseSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUCollapsePipeline()
        self.last_backend = "idle"
        self.reset_runtime_state()

    def step(self, world: "WorldEngine") -> None:
        self.reset_runtime_state(world)
        if world.collapse_deferred_regions:
            world.collapse_dirty_regions.extend(world.collapse_deferred_regions)
            world.collapse_deferred_regions.clear()
        self.last_dirty_region_count_before = int(len(world.collapse_dirty_regions))
        if not world.collapse_dirty_regions:
            self.last_backend = "idle"
            return
        while world.collapse_dirty_regions:
            x0 = min(region[0] for region in world.collapse_dirty_regions)
            y0 = min(region[1] for region in world.collapse_dirty_regions)
            x1 = max(region[2] for region in world.collapse_dirty_regions)
            y1 = max(region[3] for region in world.collapse_dirty_regions)
            world.collapse_dirty_regions.clear()
            self.last_solve_region_count += 1
            self._solve_region(world, x0, y0, x1, y1)

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def _solve_region(self, world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> None:
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "collapse")
        formal_gpu_frame = bool(
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and getattr(world, "_world_simulation_frame_active", False)
        )
        structural_world: np.ndarray | None = None
        if gpu_available and formal_gpu_frame:
            x0, y0, x1, y1 = self.gpu_pipeline.expand_region_to_component_bbox(world, x0, y0, x1, y1)
        elif gpu_available:
            structural_world = self.gpu_pipeline.classify_world_structural_mask(world)
            x0, y0, x1, y1 = self._expand_region_to_component_gpu(world, structural_world, x0, y0, x1, y1)
        else:
            world._require_cpu_oracle_backend("collapse")
            structural_world = self._world_structural_mask(world)
            x0, y0, x1, y1 = self._expand_region_to_component(structural_world, x0, y0, x1, y1)
        self.last_solve_region_mask[y0:y1, x0:x1] = True
        if gpu_available and formal_gpu_frame:
            classify_resources, region_width, region_height = self.gpu_pipeline.classify_region_textures(
                world,
                x0,
                y0,
                x1,
                y1,
            )
            structural_metadata = self.gpu_pipeline.summarize_labeled_component_texture(
                world,
                classify_resources.structural_tex,
                np.asarray([1], dtype=np.int32),
                x0,
                y0,
                region_width,
                region_height,
            )
            if structural_metadata.size == 0 or int(structural_metadata[0][4]) <= 0:
                return
            supported_texture = self.gpu_pipeline.solve_region_textures(
                world,
                classify_resources,
                region_width,
                region_height,
                x0=x0,
                y0=y0,
            )
            self.last_backend = "gpu"
            outcome_resources, outcome_width, outcome_height = self.gpu_pipeline.resolve_supported_outcome_textures(
                world,
                classify_resources,
                supported_texture,
                x0,
                y0,
                region_width,
                region_height,
            )
            delayed_metadata = self.gpu_pipeline.summarize_labeled_component_texture(
                world,
                outcome_resources.temp_out_tex,
                np.asarray([1], dtype=np.int32),
                x0,
                y0,
                outcome_width,
                outcome_height,
            )
            self._queue_deferred_metadata_region(world, delayed_metadata[0] if delayed_metadata.size else None)
            component_labels, component_island_ids, component_metadata = self.gpu_pipeline.materialize_component_texture(
                world,
                outcome_resources.phase_out_tex,
                outcome_width,
                outcome_height,
                x0,
                y0,
            )
            self._record_gpu_collapsed_components(world, component_island_ids, component_metadata)
            return
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
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None or material_id < 0 or material_id >= int(material_table.shape[0]):
            return None
        row = material_table[material_id]
        if int(row["name_hash"]) == 0:
            return None
        return row

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
    ) -> np.ndarray:
        mask = fallback.copy()
        if world is None or getattr(world, "simulation_backend", "") != "gpu":
            return mask
        bridge = world.bridge
        if resource_name not in bridge.gpu_authoritative_resources:
            return mask
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError(f"GPU collapse runtime snapshot requires bridge buffer {resource_name!r}")
        cell_count = int(world.width) * int(world.height)
        raw = bridge.buffers[resource_name].read(size=cell_count * 4)
        gpu_mask = np.frombuffer(raw, dtype=np.int32, count=cell_count).reshape((world.height, world.width)) != 0
        return gpu_mask & self.last_solve_region_mask

    def runtime_snapshot(self, world: "WorldEngine" | None = None) -> dict[str, object]:
        structural_mask = self._runtime_mask_snapshot(world, "collapse_structural_mask", self.last_structural_mask)
        support_seed_mask = self._runtime_mask_snapshot(world, "collapse_support_seed_mask", self.last_support_seed_mask)
        supported_mask = self._runtime_mask_snapshot(world, "collapse_supported_mask", self.last_supported_mask)
        unsupported_mask = self._runtime_mask_snapshot(world, "collapse_unsupported_mask", self.last_unsupported_mask)
        delayed_pending_mask = self._runtime_mask_snapshot(
            world,
            "collapse_delayed_pending_mask",
            self.last_delayed_pending_mask,
        )
        immune_unsupported_mask = self._runtime_mask_snapshot(
            world,
            "collapse_immune_unsupported_mask",
            self.last_immune_unsupported_mask,
        )
        collapsed_cell_mask = self._runtime_mask_snapshot(
            world,
            "collapse_collapsed_cell_mask",
            self.last_collapsed_cell_mask,
        )
        return {
            "backend": self.last_backend,
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
            "collapsed_components": [dict(component) for component in self.last_collapsed_components],
        }
