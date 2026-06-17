from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.types import CollapseBehavior, Phase


LOCAL_SIZE = 8


@dataclass(slots=True)
class GPUCollapseResources:
    signature: tuple[int, int]
    structural_tex: Any
    support_ping: Any
    support_pong: Any
    material_tex: Any
    material_out_tex: Any
    phase_tex: Any
    phase_out_tex: Any
    cell_flags_tex: Any
    cell_flags_out_tex: Any
    timer_tex: Any
    timer_out_tex: Any
    integrity_tex: Any
    integrity_out_tex: Any
    temp_tex: Any
    temp_out_tex: Any
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    change_flag: Any
    component_labels: Any
    component_island_ids: Any
    component_metadata: Any
    component_flags: Any
    component_count: Any
    material_structural: Any
    material_support_anchor: Any
    material_collapse_behavior: Any
    material_collapse_generation: Any
    material_base_integrity: Any
    material_spawn_temperature: Any


class GPUCollapsePipeline:
    def __init__(self) -> None:
        self.resources: GPUCollapseResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def solve_region(
        self,
        world: "WorldEngine",
        structural_mask: np.ndarray,
        support_seed_mask: np.ndarray,
        *,
        x0: int = 0,
        y0: int = 0,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = structural_mask.shape
        if width == 0 or height == 0:
            return np.zeros_like(structural_mask, dtype=bool)
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        resources.structural_tex.write(structural_mask.astype("f4", copy=False).tobytes())
        resources.support_ping.write(support_seed_mask.astype("f4", copy=False).tobytes())
        resources.support_pong.write(support_seed_mask.astype("f4", copy=False).tobytes())
        current = self.solve_region_textures(world, resources, width, height, x0=x0, y0=y0)
        supported = np.frombuffer(current.read(), dtype="f4").reshape((height, width)) > 0.5
        return structural_mask & ~supported

    def solve_region_textures(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        *,
        x0: int = 0,
        y0: int = 0,
        publish_masks: bool = True,
    ) -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        current = resources.support_ping
        scratch = resources.support_pong
        jump = 1
        max_dim = max(width, height)
        while jump < max_dim:
            jump <<= 1
        jump >>= 1
        while jump >= 2:
            current, scratch, _ = self._run_pass(ctx, resources, current, scratch, width, height, jump, read_changed=False)
            jump >>= 1
        if self._formal_gpu_frame(world):
            for _ in range(self._formal_support_unit_pass_count(width, height)):
                current, scratch, _ = self._run_pass(ctx, resources, current, scratch, width, height, 1, read_changed=False)
            if publish_masks:
                self._publish_bridge_region_mask(world, resources, current, "collapse_supported_mask", x0, y0, width, height)
                self._publish_bridge_region_mask(
                    world,
                    resources,
                    current,
                    "collapse_unsupported_mask",
                    x0,
                    y0,
                    width,
                    height,
                    mode=1,
                )
        else:
            while True:
                current, scratch, changed = self._run_pass(ctx, resources, current, scratch, width, height, 1)
                if not changed:
                    break
        return current

    @staticmethod
    def _formal_support_unit_pass_count(width: int, height: int) -> int:
        return max(1, int(width) + int(height))

    @staticmethod
    def _formal_label_unit_pass_count(width: int, height: int) -> int:
        return max(1, int(width) + int(height))

    def classify_world_structural_mask(self, world: "WorldEngine") -> np.ndarray:
        structural, _, _ = self.classify_region(world, 0, 0, world.width, world.height)
        return structural

    def expand_region_to_component_bbox(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        seed_x0 = max(0, int(x0) - 1)
        seed_y0 = max(0, int(y0) - 1)
        seed_x1 = min(world.width, int(x1) + 1)
        seed_y1 = min(world.height, int(y1) + 1)
        if seed_x0 >= seed_x1 or seed_y0 >= seed_y1:
            return (seed_x0, seed_y0, seed_x1, seed_y1)
        resources, width, height = self.classify_region_textures(
            world,
            0,
            0,
            world.width,
            world.height,
            publish_masks=False,
        )
        self.seed_structural_region_texture(world, resources, width, height, seed_x0, seed_y0, seed_x1, seed_y1)
        connected_texture = self.solve_region_textures(world, resources, width, height, x0=0, y0=0, publish_masks=False)
        metadata = self.summarize_labeled_component_texture(
            world,
            connected_texture,
            np.asarray([1], dtype=np.int32),
            0,
            0,
            width,
            height,
        )
        if metadata.size == 0:
            return (seed_x0, seed_y0, seed_x1, seed_y1)
        min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata[0])
        if cell_count <= 0:
            return (seed_x0, seed_y0, seed_x1, seed_y1)
        return (
            min(seed_x0, min_x),
            min(seed_y0, min_y),
            max(seed_x1, max_x),
            max(seed_y1, max_y),
        )

    def seed_structural_region_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        seed_x0: int,
        seed_y0: int,
        seed_x1: int,
        seed_y1: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        program = self.programs["seed_structural_region"]
        program["region_size"].value = (int(width), int(height))
        program["seed_rect"].value = (int(seed_x0), int(seed_y0), int(seed_x1), int(seed_y1))
        resources.structural_tex.use(location=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        resources.support_pong.bind_to_image(2, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def classify_region(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        width = max(0, int(x1) - int(x0))
        height = max(0, int(y1) - int(y0))
        if width == 0 or height == 0:
            empty_bool = np.zeros((height, width), dtype=np.bool_)
            empty_int = np.zeros((height, width), dtype=np.int32)
            return empty_bool, empty_bool.copy(), empty_int
        resources, width, height = self.classify_region_textures(world, x0, y0, x1, y1)
        ctx.finish()
        structural = np.frombuffer(resources.structural_tex.read(), dtype="f4").reshape((height, width)) > 0.5
        support_seed = np.frombuffer(resources.support_ping.read(), dtype="f4").reshape((height, width)) > 0.5
        behavior = np.rint(np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width))).astype(np.int32)
        return structural, support_seed, behavior

    def classify_region_textures(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        publish_masks: bool = True,
    ) -> tuple[GPUCollapseResources, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        width = max(0, int(x1) - int(x0))
        height = max(0, int(y1) - int(y0))
        if width == 0 or height == 0:
            raise ValueError("classify_region_textures requires a non-empty region")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        self._upload_region_state(world, resources, x0, y0, width, height)
        structural_params, support_params, behavior_params = self._classification_material_params(world)
        self._write_dynamic_buffer(ctx, resources, "material_structural", structural_params)
        self._write_dynamic_buffer(ctx, resources, "material_support_anchor", support_params)
        self._write_dynamic_buffer(ctx, resources, "material_collapse_behavior", behavior_params)

        program = self.programs["classify_cells"]
        program["region_size"].value = (width, height)
        program["region_origin"].value = (int(x0), int(y0))
        program["world_height"].value = int(world.height)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.structural_tex.bind_to_image(2, read=False, write=True)
        resources.support_ping.bind_to_image(3, read=False, write=True)
        resources.material_out_tex.bind_to_image(4, read=False, write=True)
        resources.material_structural.bind_to_storage_buffer(binding=0)
        resources.material_support_anchor.bind_to_storage_buffer(binding=1)
        resources.material_collapse_behavior.bind_to_storage_buffer(binding=2)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world) and publish_masks:
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.structural_tex,
                "collapse_structural_mask",
                x0,
                y0,
                width,
                height,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.support_ping,
                "collapse_support_seed_mask",
                x0,
                y0,
                width,
                height,
            )
        return resources, width, height

    def label_component_mask(
        self,
        world: "WorldEngine",
        component_mask: np.ndarray,
        *,
        x0: int = 0,
        y0: int = 0,
    ) -> np.ndarray:
        label_texture, width, height = self._label_component_mask_texture(world, component_mask, x0=x0, y0=y0)
        if width == 0 or height == 0:
            return np.zeros_like(component_mask, dtype=np.int32)
        return np.rint(np.frombuffer(label_texture.read(), dtype="f4").reshape((height, width))).astype(np.int32)

    def materialize_component_mask(
        self,
        world: "WorldEngine",
        component_mask: np.ndarray,
        x0: int,
        y0: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        label_texture, width, height = self._label_component_mask_texture(world, component_mask, x0=x0, y0=y0)
        if width == 0 or height == 0:
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty.copy(), np.zeros((0, 5), dtype=np.int32)
        component_labels = self.collect_component_labels(world, label_texture, width, height)
        if component_labels.size == 0:
            return component_labels, np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.int32)
        component_island_ids = np.asarray(
            [world.allocate_island_id() for _ in range(int(component_labels.size))],
            dtype=np.int32,
        )
        component_metadata = self.summarize_labeled_component_texture(
            world,
            label_texture,
            component_labels,
            x0,
            y0,
            width,
            height,
        )
        self.materialize_labeled_component_texture(
            world,
            label_texture,
            component_labels,
            component_island_ids,
            x0,
            y0,
            width,
            height,
        )
        return component_labels, component_island_ids, component_metadata

    def materialize_component_texture(
        self,
        world: "WorldEngine",
        component_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if width == 0 or height == 0:
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty.copy(), np.zeros((0, 5), dtype=np.int32)
        label_texture, width, height = self._label_component_texture(
            world,
            component_texture,
            width,
            height,
            x0=x0,
            y0=y0,
        )
        component_labels = self.collect_component_labels(world, label_texture, width, height)
        if component_labels.size == 0:
            return component_labels, np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.int32)
        component_island_ids = np.asarray(
            [world.allocate_island_id() for _ in range(int(component_labels.size))],
            dtype=np.int32,
        )
        component_metadata = self.summarize_labeled_component_texture(
            world,
            label_texture,
            component_labels,
            x0,
            y0,
            width,
            height,
        )
        self.materialize_labeled_component_texture(
            world,
            label_texture,
            component_labels,
            component_island_ids,
            x0,
            y0,
            width,
            height,
        )
        return component_labels, component_island_ids, component_metadata

    def _label_component_mask_texture(
        self,
        world: "WorldEngine",
        component_mask: np.ndarray,
        *,
        x0: int = 0,
        y0: int = 0,
    ) -> tuple[Any, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = component_mask.shape
        if width == 0 or height == 0:
            return None, width, height
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        resources.structural_tex.write(component_mask.astype("f4", copy=False).tobytes())
        return self._label_component_texture(world, resources.structural_tex, width, height, x0=x0, y0=y0)

    def _label_component_texture(
        self,
        world: "WorldEngine",
        component_texture: Any,
        width: int,
        height: int,
        *,
        x0: int = 0,
        y0: int = 0,
    ) -> tuple[Any, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return None, width, height
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE

        init_program = self.programs["component_label_init"]
        init_program["region_size"].value = (width, height)
        component_texture.use(location=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        init_program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

        current = resources.support_ping
        scratch = resources.support_pong
        propagate = self.programs["component_label_propagate"]
        propagate["region_size"].value = (width, height)
        if self._formal_gpu_frame(world):
            for _ in range(self._formal_label_unit_pass_count(width, height)):
                resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                self._sync_compute_writes(ctx)
                current, scratch = scratch, current
            self._publish_bridge_region_labels(world, resources, current, x0, y0, width, height)
        else:
            while True:
                resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                ctx.memory_barrier(
                    ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
                    | ctx.SHADER_STORAGE_BARRIER_BIT
                    | ctx.TEXTURE_FETCH_BARRIER_BIT
                )
                ctx.finish()
                changed = bool(np.frombuffer(resources.change_flag.read(), dtype=np.uint32, count=1)[0])
                current, scratch = scratch, current
                if not changed:
                    break

        return current, width, height

    def collect_component_labels(
        self,
        world: "WorldEngine",
        label_texture: Any,
        width: int,
        height: int,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return np.zeros((0,), dtype=np.int32)
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        cell_count = max(1, int(width) * int(height))
        required_label_bytes = cell_count * np.dtype(np.int32).itemsize
        if resources.component_labels.size < required_label_bytes:
            resources.component_labels.release()
            resources.component_labels = ctx.buffer(reserve=required_label_bytes, dynamic=True)
        else:
            resources.component_labels.orphan(required_label_bytes)
        if resources.component_flags.size < required_label_bytes:
            resources.component_flags.release()
            resources.component_flags = ctx.buffer(reserve=required_label_bytes, dynamic=True)
        else:
            resources.component_flags.orphan(required_label_bytes)
        resources.component_flags.write(np.zeros(cell_count, dtype=np.uint32).tobytes())
        resources.component_count.write(np.zeros(1, dtype=np.uint32).tobytes())

        program = self.programs["collect_component_labels"]
        program["region_size"].value = (int(width), int(height))
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_labels.bind_to_storage_buffer(binding=1)
        resources.component_count.bind_to_storage_buffer(binding=2)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        ctx.finish()

        component_count = int(np.frombuffer(resources.component_count.read(size=4), dtype=np.uint32, count=1)[0])
        if component_count <= 0:
            return np.zeros((0,), dtype=np.int32)
        return np.frombuffer(
            resources.component_labels.read(size=component_count * np.dtype(np.int32).itemsize),
            dtype=np.int32,
            count=component_count,
        ).copy()

    def summarize_labeled_components(
        self,
        world: "WorldEngine",
        labels: np.ndarray,
        component_labels: np.ndarray,
        x0: int,
        y0: int,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = labels.shape
        component_count = int(component_labels.size)
        if width == 0 or height == 0 or component_count == 0:
            return np.zeros((0, 5), dtype=np.int32)
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        resources.support_ping.write(labels.astype("f4", copy=False).tobytes())
        return self.summarize_labeled_component_texture(
            world,
            resources.support_ping,
            component_labels,
            x0,
            y0,
            width,
            height,
        )

    def summarize_labeled_component_texture(
        self,
        world: "WorldEngine",
        label_texture: Any,
        component_labels: np.ndarray,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        component_count = int(component_labels.size)
        if width == 0 or height == 0 or component_count == 0:
            return np.zeros((0, 5), dtype=np.int32)
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        metadata = np.zeros((component_count, 5), dtype=np.int32)
        metadata[:, 0] = int(x0 + width)
        metadata[:, 1] = int(y0 + height)
        metadata[:, 2] = int(x0)
        metadata[:, 3] = int(y0)
        self._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
        self._write_dynamic_buffer(ctx, resources, "component_metadata", metadata)

        program = self.programs["summarize_components"]
        program["region_size"].value = (width, height)
        program["region_origin"].value = (int(x0), int(y0))
        program["component_count"].value = component_count
        label_texture.use(location=0)
        resources.component_labels.bind_to_storage_buffer(binding=0)
        resources.component_metadata.bind_to_storage_buffer(binding=1)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        ctx.finish()
        return np.frombuffer(
            resources.component_metadata.read(size=metadata.nbytes),
            dtype=np.int32,
            count=metadata.size,
        ).reshape((component_count, 5)).copy()

    def resolve_unsupported_outcomes(
        self,
        world: "WorldEngine",
        unsupported: np.ndarray,
        behavior_region: np.ndarray,
        x0: int,
        y0: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = unsupported.shape
        if width == 0 or height == 0:
            empty = np.zeros_like(unsupported, dtype=np.bool_)
            return empty, empty.copy(), empty.copy()
        resources, width, height = self.resolve_unsupported_outcome_textures(
            world,
            unsupported,
            behavior_region,
            x0,
            y0,
        )
        delayed_pending = np.frombuffer(resources.support_pong.read(), dtype="f4").reshape((height, width)) > 0.5
        immune_unsupported = np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width)) > 0.5
        collapse_now = np.frombuffer(resources.phase_out_tex.read(), dtype="f4").reshape((height, width)) > 0.5
        if not self._formal_gpu_frame(world):
            pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
            pending_region[:] = delayed_pending
        return delayed_pending, immune_unsupported, collapse_now

    def resolve_unsupported_outcome_textures(
        self,
        world: "WorldEngine",
        unsupported: np.ndarray,
        behavior_region: np.ndarray,
        x0: int,
        y0: int,
    ) -> tuple[GPUCollapseResources, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = unsupported.shape
        if width == 0 or height == 0:
            raise ValueError("resolve_unsupported_outcome_textures requires a non-empty region")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
        resources.structural_tex.write(unsupported.astype("f4", copy=False).tobytes())
        resources.support_ping.write(behavior_region.astype("f4", copy=False).tobytes())
        resources.phase_tex.write(pending_region.astype("f4", copy=False).tobytes())
        self._load_authoritative_bridge_pending_region(world, resources, x0, y0, width, height)

        program = self.programs["resolve_outcomes"]
        program["region_size"].value = (width, height)
        program["behavior_falling_island"].value = int(CollapseBehavior.FALLING_ISLAND)
        program["behavior_delayed"].value = int(CollapseBehavior.DELAYED)
        program["behavior_immune"].value = int(CollapseBehavior.IMMUNE)
        resources.structural_tex.use(location=0)
        resources.support_ping.use(location=1)
        resources.phase_tex.use(location=2)
        resources.support_pong.bind_to_image(3, read=False, write=True)
        resources.material_out_tex.bind_to_image(4, read=False, write=True)
        resources.phase_out_tex.bind_to_image(5, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world):
            self._publish_bridge_pending_region_outputs(world, resources, x0, y0, width, height)
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.support_pong,
                "collapse_delayed_pending_mask",
                x0,
                y0,
                width,
                height,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.material_out_tex,
                "collapse_immune_unsupported_mask",
                x0,
                y0,
                width,
                height,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.phase_out_tex,
                "collapse_collapsed_cell_mask",
                x0,
                y0,
                width,
                height,
            )
        else:
            ctx.finish()
        return resources, width, height

    def resolve_supported_outcome_textures(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        supported_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> tuple[GPUCollapseResources, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            raise ValueError("resolve_supported_outcome_textures requires a non-empty region")
        pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
        resources.phase_tex.write(pending_region.astype("f4", copy=False).tobytes())
        self._load_authoritative_bridge_pending_region(world, resources, x0, y0, width, height)

        program = self.programs["resolve_outcomes_from_supported"]
        program["region_size"].value = (width, height)
        program["behavior_falling_island"].value = int(CollapseBehavior.FALLING_ISLAND)
        program["behavior_delayed"].value = int(CollapseBehavior.DELAYED)
        program["behavior_immune"].value = int(CollapseBehavior.IMMUNE)
        resources.structural_tex.use(location=0)
        supported_texture.use(location=1)
        resources.material_out_tex.use(location=2)
        resources.phase_tex.use(location=3)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
        resources.phase_out_tex.bind_to_image(6, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world):
            self._publish_bridge_pending_region_outputs_from_texture(world, resources, resources.temp_out_tex, x0, y0, width, height)
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.temp_out_tex,
                "collapse_delayed_pending_mask",
                x0,
                y0,
                width,
                height,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.integrity_out_tex,
                "collapse_immune_unsupported_mask",
                x0,
                y0,
                width,
                height,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.phase_out_tex,
                "collapse_collapsed_cell_mask",
                x0,
                y0,
                width,
                height,
            )
        else:
            ctx.finish()
        return resources, width, height

    def materialize_labeled_components(
        self,
        world: "WorldEngine",
        labels: np.ndarray,
        component_labels: np.ndarray,
        component_island_ids: np.ndarray,
        x0: int,
        y0: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        height, width = labels.shape
        if width == 0 or height == 0 or component_labels.size == 0:
            return
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        resources.support_ping.write(labels.astype("f4", copy=False).tobytes())
        self.materialize_labeled_component_texture(
            world,
            resources.support_ping,
            component_labels,
            component_island_ids,
            x0,
            y0,
            width,
            height,
        )

    def materialize_labeled_component_texture(
        self,
        world: "WorldEngine",
        label_texture: Any,
        component_labels: np.ndarray,
        component_island_ids: np.ndarray,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0 or component_labels.size == 0:
            return
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        self._upload_region_state(world, resources, x0, y0, width, height)
        collapse_generation, base_integrity, spawn_temperature = self._materialize_material_params(world)
        self._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
        self._write_dynamic_buffer(ctx, resources, "component_island_ids", component_island_ids.astype(np.int32, copy=False))
        self._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
        self._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
        self._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

        program = self.programs["materialize_components"]
        program["region_size"].value = (width, height)
        program["component_count"].value = int(component_labels.size)
        program["material_count"].value = int(collapse_generation.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        label_texture.use(location=0)
        resources.material_tex.use(location=1)
        resources.phase_tex.use(location=2)
        resources.cell_flags_tex.use(location=3)
        resources.timer_tex.use(location=4)
        resources.integrity_tex.use(location=5)
        resources.temp_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.timer_out_tex.bind_to_image(3, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(4, read=False, write=True)
        resources.temp_out_tex.bind_to_image(5, read=False, write=True)
        resources.component_labels.bind_to_storage_buffer(binding=0)
        resources.component_island_ids.bind_to_storage_buffer(binding=1)
        resources.material_collapse_generation.bind_to_storage_buffer(binding=2)
        resources.material_base_integrity.bind_to_storage_buffer(binding=3)
        resources.material_spawn_temperature.bind_to_storage_buffer(binding=4)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        aux_program = self.programs["materialize_components_aux"]
        aux_program["region_size"].value = (width, height)
        aux_program["component_count"].value = int(component_labels.size)
        label_texture.use(location=0)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        resources.component_labels.bind_to_storage_buffer(binding=0)
        resources.component_island_ids.bind_to_storage_buffer(binding=1)
        aux_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_region_state(world, resources, x0, y0, width, height)

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.structural_tex,
            self.resources.support_ping,
            self.resources.support_pong,
            self.resources.material_tex,
            self.resources.material_out_tex,
            self.resources.phase_tex,
            self.resources.phase_out_tex,
            self.resources.cell_flags_tex,
            self.resources.cell_flags_out_tex,
            self.resources.timer_tex,
            self.resources.timer_out_tex,
            self.resources.integrity_tex,
            self.resources.integrity_out_tex,
            self.resources.temp_tex,
            self.resources.temp_out_tex,
            self.resources.island_id_tex,
            self.resources.island_id_out_tex,
            self.resources.entity_id_tex,
            self.resources.entity_id_out_tex,
            self.resources.displaced_tex,
            self.resources.displaced_out_tex,
            self.resources.change_flag,
            self.resources.component_labels,
            self.resources.component_island_ids,
            self.resources.component_metadata,
            self.resources.component_flags,
            self.resources.component_count,
            self.resources.material_structural,
            self.resources.material_support_anchor,
            self.resources.material_collapse_behavior,
            self.resources.material_collapse_generation,
            self.resources.material_base_integrity,
            self.resources.material_spawn_temperature,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, ctx: Any, width: int, height: int) -> GPUCollapseResources:
        signature = (width, height)
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        structural_tex = ctx.texture((width, height), 1, dtype="f4")
        support_ping = ctx.texture((width, height), 1, dtype="f4")
        support_pong = ctx.texture((width, height), 1, dtype="f4")
        material_tex = ctx.texture((width, height), 1, dtype="f4")
        material_out_tex = ctx.texture((width, height), 1, dtype="f4")
        phase_tex = ctx.texture((width, height), 1, dtype="f4")
        phase_out_tex = ctx.texture((width, height), 1, dtype="f4")
        cell_flags_tex = ctx.texture((width, height), 1, dtype="f4")
        cell_flags_out_tex = ctx.texture((width, height), 1, dtype="f4")
        timer_tex = ctx.texture((width, height), 4, dtype="f4")
        timer_out_tex = ctx.texture((width, height), 4, dtype="f4")
        integrity_tex = ctx.texture((width, height), 1, dtype="f4")
        integrity_out_tex = ctx.texture((width, height), 1, dtype="f4")
        temp_tex = ctx.texture((width, height), 1, dtype="f4")
        temp_out_tex = ctx.texture((width, height), 1, dtype="f4")
        island_id_tex = ctx.texture((width, height), 1, dtype="f4")
        island_id_out_tex = ctx.texture((width, height), 1, dtype="f4")
        entity_id_tex = ctx.texture((width, height), 1, dtype="f4")
        entity_id_out_tex = ctx.texture((width, height), 1, dtype="f4")
        displaced_tex = ctx.texture((width, height), 1, dtype="f4")
        displaced_out_tex = ctx.texture((width, height), 1, dtype="f4")
        for texture in (
            structural_tex,
            support_ping,
            support_pong,
            material_tex,
            material_out_tex,
            phase_tex,
            phase_out_tex,
            cell_flags_tex,
            cell_flags_out_tex,
            timer_tex,
            timer_out_tex,
            integrity_tex,
            integrity_out_tex,
            temp_tex,
            temp_out_tex,
            island_id_tex,
            island_id_out_tex,
            entity_id_tex,
            entity_id_out_tex,
            displaced_tex,
            displaced_out_tex,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        cell_count = max(1, width * height)
        self.resources = GPUCollapseResources(
            signature=signature,
            structural_tex=structural_tex,
            support_ping=support_ping,
            support_pong=support_pong,
            material_tex=material_tex,
            material_out_tex=material_out_tex,
            phase_tex=phase_tex,
            phase_out_tex=phase_out_tex,
            cell_flags_tex=cell_flags_tex,
            cell_flags_out_tex=cell_flags_out_tex,
            timer_tex=timer_tex,
            timer_out_tex=timer_out_tex,
            integrity_tex=integrity_tex,
            integrity_out_tex=integrity_out_tex,
            temp_tex=temp_tex,
            temp_out_tex=temp_out_tex,
            island_id_tex=island_id_tex,
            island_id_out_tex=island_id_out_tex,
            entity_id_tex=entity_id_tex,
            entity_id_out_tex=entity_id_out_tex,
            displaced_tex=displaced_tex,
            displaced_out_tex=displaced_out_tex,
            change_flag=ctx.buffer(reserve=4, dynamic=True),
            component_labels=ctx.buffer(reserve=cell_count * 4, dynamic=True),
            component_island_ids=ctx.buffer(reserve=cell_count * 4, dynamic=True),
            component_metadata=ctx.buffer(reserve=cell_count * 5 * 4, dynamic=True),
            component_flags=ctx.buffer(reserve=cell_count * 4, dynamic=True),
            component_count=ctx.buffer(reserve=4, dynamic=True),
            material_structural=ctx.buffer(reserve=4, dynamic=True),
            material_support_anchor=ctx.buffer(reserve=4, dynamic=True),
            material_collapse_behavior=ctx.buffer(reserve=4, dynamic=True),
            material_collapse_generation=ctx.buffer(reserve=4, dynamic=True),
            material_base_integrity=ctx.buffer(reserve=4, dynamic=True),
            material_spawn_temperature=ctx.buffer(reserve=4, dynamic=True),
        )
        return self.resources

    def _ensure_programs(self, ctx: Any | None) -> None:
        if not ctx or self.programs:
            return
        self.programs["classify_cells"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform int world_height;
            uniform int material_count;
            uniform int phase_falling_island;

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(r32f, binding=2) writeonly uniform image2D structural_img;
            layout(r32f, binding=3) writeonly uniform image2D support_seed_img;
            layout(r32f, binding=4) writeonly uniform image2D behavior_img;

            layout(std430, binding=0) buffer MaterialStructural {{
                int material_structural[];
            }};
            layout(std430, binding=1) buffer MaterialSupportAnchor {{
                int material_support_anchor[];
            }};
            layout(std430, binding=2) buffer MaterialCollapseBehavior {{
                int material_collapse_behavior[];
            }};

            int safe_material_index(int material_id) {{
                if (material_id < 0 || material_id >= material_count) {{
                    return 0;
                }}
                return material_id;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                int index = safe_material_index(material_id);
                int phase = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                bool structural = material_id > 0 && phase != phase_falling_island && material_structural[index] != 0;
                bool support_seed = structural && (material_support_anchor[index] != 0 || region_origin.y + cell.y == world_height - 1);
                int behavior = material_id > 0 ? material_collapse_behavior[index] : 0;
                imageStore(structural_img, cell, vec4(structural ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(support_seed_img, cell, vec4(support_seed ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(behavior_img, cell, vec4(float(behavior), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["propagate"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int jump;
            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D support_in_tex;
            layout(r32f, binding=2) writeonly uniform image2D support_out_img;
            layout(std430, binding=0) buffer ChangeFlagBuffer {{
                uint change_flag;
            }};

            bool line_clear(ivec2 start_cell, ivec2 end_cell) {{
                ivec2 delta = end_cell - start_cell;
                int steps = max(abs(delta.x), abs(delta.y));
                if (steps <= 1) {{
                    return true;
                }}
                for (int step_index = 1; step_index < steps; ++step_index) {{
                    float t = float(step_index) / float(steps);
                    ivec2 sample_cell = ivec2(round(mix(vec2(start_cell), vec2(end_cell), t)));
                    if (texelFetch(structural_tex, sample_cell, 0).x < 0.5) {{
                        return false;
                    }}
                }}
                return true;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                float structural = texelFetch(structural_tex, cell, 0).x;
                float current = structural > 0.5 ? texelFetch(support_in_tex, cell, 0).x : 0.0;
                float next_value = current;
                if (structural > 0.5 && current < 0.5) {{
                    for (int oy = -1; oy <= 1 && next_value < 0.5; ++oy) {{
                        for (int ox = -1; ox <= 1 && next_value < 0.5; ++ox) {{
                            if (abs(ox) + abs(oy) != 1) {{
                                continue;
                            }}
                            ivec2 sample_cell = cell + ivec2(ox * jump, oy * jump);
                            if (sample_cell.x < 0 || sample_cell.y < 0 || sample_cell.x >= region_size.x || sample_cell.y >= region_size.y) {{
                                continue;
                            }}
                            if (texelFetch(support_in_tex, sample_cell, 0).x < 0.5) {{
                                continue;
                            }}
                            if (texelFetch(structural_tex, sample_cell, 0).x < 0.5) {{
                                continue;
                            }}
                            if (jump <= 1 || line_clear(sample_cell, cell)) {{
                                next_value = 1.0;
                            }}
                        }}
                    }}
                }}
                imageStore(support_out_img, cell, vec4(next_value, 0.0, 0.0, 0.0));
                if (next_value > 0.5 && current < 0.5) {{
                    atomicOr(change_flag, 1u);
                }}
            }}
            """
        )
        self.programs["component_label_init"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            layout(binding=0) uniform sampler2D collapse_mask_tex;
            layout(r32f, binding=1) writeonly uniform image2D label_out_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                float mask_value = texelFetch(collapse_mask_tex, cell, 0).x;
                float label = mask_value > 0.5 ? float(cell.y * region_size.x + cell.x + 1) : 0.0;
                imageStore(label_out_img, cell, vec4(label, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["seed_structural_region"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec4 seed_rect;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(r32f, binding=1) writeonly uniform image2D support_seed_img;
            layout(r32f, binding=2) writeonly uniform image2D support_seed_copy_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                bool in_seed = (
                    cell.x >= seed_rect.x &&
                    cell.y >= seed_rect.y &&
                    cell.x < seed_rect.z &&
                    cell.y < seed_rect.w
                );
                bool seeded = in_seed && texelFetch(structural_tex, cell, 0).x > 0.5;
                vec4 value = vec4(seeded ? 1.0 : 0.0, 0.0, 0.0, 0.0);
                imageStore(support_seed_img, cell, value);
                imageStore(support_seed_copy_img, cell, value);
            }}
            """
        )
        self.programs["component_label_propagate"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            layout(binding=0) uniform sampler2D label_in_tex;
            layout(r32f, binding=1) writeonly uniform image2D label_out_img;
            layout(std430, binding=0) buffer ChangeFlagBuffer {{
                uint change_flag;
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                float current = texelFetch(label_in_tex, cell, 0).x;
                float next_value = current;
                if (current > 0.5) {{
                    for (int axis = 0; axis < 4; ++axis) {{
                        ivec2 offset = axis == 0 ? ivec2(-1, 0) : axis == 1 ? ivec2(1, 0) : axis == 2 ? ivec2(0, -1) : ivec2(0, 1);
                        ivec2 sample_cell = cell + offset;
                        if (sample_cell.x < 0 || sample_cell.y < 0 || sample_cell.x >= region_size.x || sample_cell.y >= region_size.y) {{
                            continue;
                        }}
                        float sample_label = texelFetch(label_in_tex, sample_cell, 0).x;
                        if (sample_label > 0.5 && sample_label < next_value) {{
                            next_value = sample_label;
                        }}
                    }}
                }}
                imageStore(label_out_img, cell, vec4(next_value, 0.0, 0.0, 0.0));
                if (next_value + 0.5 < current) {{
                    atomicOr(change_flag, 1u);
                }}
            }}
            """
        )
        self.programs["collect_component_labels"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;

            layout(binding=0) uniform sampler2D label_tex;

            layout(std430, binding=0) buffer ComponentFlags {{
                uint component_flags[];
            }};
            layout(std430, binding=1) buffer ComponentLabels {{
                int component_labels[];
            }};
            layout(std430, binding=2) buffer ComponentCount {{
                uint component_count;
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int label = int(texelFetch(label_tex, cell, 0).x + 0.5);
                if (label <= 0) {{
                    return;
                }}
                uint flag_index = uint(label - 1);
                uint previous = atomicExchange(component_flags[flag_index], 1u);
                if (previous == 0u) {{
                    uint output_index = atomicAdd(component_count, 1u);
                    component_labels[output_index] = label;
                }}
            }}
            """
        )
        self.programs["summarize_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform int component_count;

            layout(binding=0) uniform sampler2D component_label_tex;

            layout(std430, binding=0) buffer ComponentLabels {{
                int component_labels[];
            }};
            layout(std430, binding=1) buffer ComponentMetadata {{
                int component_metadata[];
            }};

            int index_for_label(int label) {{
                for (int index = 0; index < component_count; ++index) {{
                    if (component_labels[index] == label) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label <= 0) {{
                    return;
                }}
                int index = index_for_label(label);
                if (index < 0) {{
                    return;
                }}
                int base = index * 5;
                int world_x = region_origin.x + cell.x;
                int world_y = region_origin.y + cell.y;
                atomicMin(component_metadata[base + 0], world_x);
                atomicMin(component_metadata[base + 1], world_y);
                atomicMax(component_metadata[base + 2], world_x + 1);
                atomicMax(component_metadata[base + 3], world_y + 1);
                atomicAdd(component_metadata[base + 4], 1);
            }}
            """
        )
        self.programs["resolve_outcomes"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int behavior_falling_island;
            uniform int behavior_delayed;
            uniform int behavior_immune;

            layout(binding=0) uniform sampler2D unsupported_tex;
            layout(binding=1) uniform sampler2D behavior_tex;
            layout(binding=2) uniform sampler2D pending_tex;
            layout(r32f, binding=3) writeonly uniform image2D delayed_pending_img;
            layout(r32f, binding=4) writeonly uniform image2D immune_unsupported_img;
            layout(r32f, binding=5) writeonly uniform image2D collapse_now_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                bool unsupported = texelFetch(unsupported_tex, cell, 0).x > 0.5;
                int behavior = int(texelFetch(behavior_tex, cell, 0).x + 0.5);
                bool pending = texelFetch(pending_tex, cell, 0).x > 0.5;
                bool delayed_pending = unsupported && behavior == behavior_delayed && !pending;
                bool immune_unsupported = unsupported && behavior == behavior_immune;
                bool collapse_now = unsupported && (
                    behavior == behavior_falling_island || (behavior == behavior_delayed && pending)
                );
                imageStore(delayed_pending_img, cell, vec4(delayed_pending ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(immune_unsupported_img, cell, vec4(immune_unsupported ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(collapse_now_img, cell, vec4(collapse_now ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["resolve_outcomes_from_supported"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int behavior_falling_island;
            uniform int behavior_delayed;
            uniform int behavior_immune;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D supported_tex;
            layout(binding=2) uniform sampler2D behavior_tex;
            layout(binding=3) uniform sampler2D pending_tex;
            layout(r32f, binding=4) writeonly uniform image2D delayed_pending_img;
            layout(r32f, binding=5) writeonly uniform image2D immune_unsupported_img;
            layout(r32f, binding=6) writeonly uniform image2D collapse_now_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                bool structural = texelFetch(structural_tex, cell, 0).x > 0.5;
                bool supported = texelFetch(supported_tex, cell, 0).x > 0.5;
                bool unsupported = structural && !supported;
                int behavior = int(texelFetch(behavior_tex, cell, 0).x + 0.5);
                bool pending = texelFetch(pending_tex, cell, 0).x > 0.5;
                bool delayed_pending = unsupported && behavior == behavior_delayed && !pending;
                bool immune_unsupported = unsupported && behavior == behavior_immune;
                bool collapse_now = unsupported && (
                    behavior == behavior_falling_island || (behavior == behavior_delayed && pending)
                );
                imageStore(delayed_pending_img, cell, vec4(delayed_pending ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(immune_unsupported_img, cell, vec4(immune_unsupported ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(collapse_now_img, cell, vec4(collapse_now ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["materialize_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int component_count;
            uniform int material_count;
            uniform int phase_falling_island;

            layout(binding=0) uniform sampler2D component_label_tex;
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D cell_flags_tex;
            layout(binding=4) uniform sampler2D timer_tex;
            layout(binding=5) uniform sampler2D integrity_tex;
            layout(binding=6) uniform sampler2D temp_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D integrity_out_img;
            layout(r32f, binding=5) writeonly uniform image2D temp_out_img;

            layout(std430, binding=0) buffer ComponentLabels {{
                int component_labels[];
            }};
            layout(std430, binding=1) buffer ComponentIslandIds {{
                int component_island_ids[];
            }};
            layout(std430, binding=2) buffer CollapseGenerationIds {{
                int collapse_generation_ids[];
            }};
            layout(std430, binding=3) buffer BaseIntegrityValues {{
                float base_integrity_values[];
            }};
            layout(std430, binding=4) buffer SpawnTemperatureValues {{
                float spawn_temperature_values[];
            }};

            int island_id_for_label(int label) {{
                for (int index = 0; index < component_count; ++index) {{
                    if (component_labels[index] == label) {{
                        return component_island_ids[index];
                    }}
                }}
                return 0;
            }}

            int collapse_generation_id(int material_id) {{
                if (material_id <= 0 || material_id >= material_count) {{
                    return 0;
                }}
                return collapse_generation_ids[material_id];
            }}

            float base_integrity_for(int material_id) {{
                if (material_id <= 0 || material_id >= material_count) {{
                    return 0.0;
                }}
                return base_integrity_values[material_id];
            }}

            float spawn_temperature_for(int material_id) {{
                if (material_id <= 0 || material_id >= material_count) {{
                    return -340282346638528859811704183484516925440.0;
                }}
                return spawn_temperature_values[material_id];
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int source_material = int(texelFetch(material_tex, cell, 0).x + 0.5);
                float out_material = float(source_material);
                float out_phase = texelFetch(phase_tex, cell, 0).x;
                float out_flags = texelFetch(cell_flags_tex, cell, 0).x;
                vec4 out_timer = texelFetch(timer_tex, cell, 0);
                float out_integrity = texelFetch(integrity_tex, cell, 0).x;
                float out_temperature = texelFetch(temp_tex, cell, 0).x;
                float out_island_id = texelFetch(island_id_tex, cell, 0).x;
                float out_entity_id = texelFetch(entity_id_tex, cell, 0).x;
                float out_displaced = texelFetch(displaced_tex, cell, 0).x;

                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label > 0) {{
                    int generated_material = collapse_generation_id(source_material);
                    int final_material = generated_material > 0 ? generated_material : source_material;
                    out_material = float(final_material);
                    out_phase = float(phase_falling_island);
                    out_island_id = float(island_id_for_label(label));
                    out_entity_id = 0.0;
                    out_displaced = 0.0;
                    if (generated_material > 0) {{
                        out_flags = 0.0;
                        out_timer = vec4(0.0, 0.0, 0.0, 0.0);
                        out_integrity = base_integrity_for(generated_material);
                        out_temperature = max(out_temperature, spawn_temperature_for(generated_material));
                    }}
                }}

                imageStore(material_out_img, cell, vec4(out_material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(out_phase, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, cell, vec4(out_flags, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, out_timer);
                imageStore(integrity_out_img, cell, vec4(out_integrity, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, cell, vec4(out_temperature, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["materialize_components_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int component_count;
            layout(binding=0) uniform sampler2D component_label_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;
            layout(std430, binding=0) buffer ComponentLabels {{
                int component_labels[];
            }};
            layout(std430, binding=1) buffer ComponentIslandIds {{
                int component_island_ids[];
            }};

            int island_id_for_label(int label) {{
                for (int index = 0; index < component_count; ++index) {{
                    if (component_labels[index] == label) {{
                        return component_island_ids[index];
                    }}
                }}
                return 0;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                float out_island_id = texelFetch(island_id_tex, cell, 0).x;
                float out_entity_id = texelFetch(entity_id_tex, cell, 0).x;
                float out_displaced = texelFetch(displaced_tex, cell, 0).x;
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label > 0) {{
                    out_island_id = float(island_id_for_label(label));
                    out_entity_id = 0.0;
                    out_displaced = 0.0;
                }}
                imageStore(island_id_out_img, cell, vec4(out_island_id, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, cell, vec4(out_entity_id, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, cell, vec4(out_displaced, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_region_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform bool copy_cell_core;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};

            layout(r32f, binding=0) writeonly uniform image2D material_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_img;
            layout(r32f, binding=4) writeonly uniform image2D integrity_img;
            layout(r32f, binding=5) writeonly uniform image2D temp_img;

            vec4 unpack_timer(uint word) {{
                return vec4(
                    float(word & 0xFFu),
                    float((word >> 8u) & 0xFFu),
                    float((word >> 16u) & 0xFFu),
                    float((word >> 24u) & 0xFFu)
                );
            }}

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                if (copy_cell_core) {{
                    int word_index = cell_index * 5;
                    uint word0 = bridge_cell_core[word_index];
                    imageStore(material_img, local_cell, vec4(float(word0 & 0xFFFFu), 0.0, 0.0, 0.0));
                    imageStore(phase_img, local_cell, vec4(float((word0 >> 16u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(flags_img, local_cell, vec4(float((word0 >> 24u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(temp_img, local_cell, vec4(uintBitsToFloat(bridge_cell_core[word_index + 2]), 0.0, 0.0, 0.0));
                    imageStore(timer_img, local_cell, unpack_timer(bridge_cell_core[word_index + 3]));
                    imageStore(integrity_img, local_cell, vec4(float(bridge_cell_core[word_index + 4] & 0xFFFFu), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["load_bridge_region_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;
            layout(std430, binding=1) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=2) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=3) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_img;
            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                if (copy_island_id) {{
                    imageStore(island_img, local_cell, vec4(float(bridge_island_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_entity_id) {{
                    imageStore(entity_img, local_cell, vec4(float(bridge_entity_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_displaced_material) {{
                    imageStore(displaced_img, local_cell, vec4(float(bridge_displaced[cell_index]), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["publish_bridge_region_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D flags_tex;
            layout(binding=3) uniform sampler2D timer_tex;
            layout(binding=4) uniform sampler2D integrity_tex;
            layout(binding=5) uniform sampler2D temp_tex;
            layout(binding=6) uniform sampler2D island_tex;
            layout(binding=7) uniform sampler2D entity_tex;
            layout(binding=8) uniform sampler2D displaced_tex;

            layout(std430, binding=0) buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=2) writeonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=3) writeonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                int word_index = cell_index * 5;
                uint material = uint(clamp(round(texelFetch(material_tex, local_cell, 0).x), 0.0, 65535.0));
                uint phase = uint(clamp(round(texelFetch(phase_tex, local_cell, 0).x), 0.0, 255.0));
                uint flags = uint(clamp(round(texelFetch(flags_tex, local_cell, 0).x), 0.0, 255.0));
                uint preserved_velocity = bridge_cell_core[word_index + 1];
                bridge_cell_core[word_index] = material | (phase << 16u) | (flags << 24u);
                bridge_cell_core[word_index + 1] = preserved_velocity;
                bridge_cell_core[word_index + 2] = floatBitsToUint(texelFetch(temp_tex, local_cell, 0).x);
                bridge_cell_core[word_index + 3] = pack_timer(texelFetch(timer_tex, local_cell, 0));
                bridge_cell_core[word_index + 4] = uint(clamp(round(texelFetch(integrity_tex, local_cell, 0).x), 0.0, 65535.0));
                bridge_island_id[cell_index] = int(round(texelFetch(island_tex, local_cell, 0).x));
                bridge_entity_id[cell_index] = int(round(texelFetch(entity_tex, local_cell, 0).x));
                bridge_displaced[cell_index] = int(round(texelFetch(displaced_tex, local_cell, 0).x));
                imageStore(bridge_material_img, world_cell, vec4(float(material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_region_pending"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;

            layout(std430, binding=0) readonly buffer BridgePendingBuffer {{
                int bridge_pending[];
            }};
            layout(r32f, binding=1) writeonly uniform image2D pending_img;

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                imageStore(pending_img, local_cell, vec4(bridge_pending[cell_index] != 0 ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge_region_pending"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;

            layout(binding=0) uniform sampler2D pending_tex;
            layout(std430, binding=0) writeonly buffer BridgePendingBuffer {{
                int bridge_pending[];
            }};

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                bridge_pending[cell_index] = texelFetch(pending_tex, local_cell, 0).x > 0.5 ? 1 : 0;
            }}
            """
        )
        self.programs["publish_bridge_region_mask"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform int mode;

            layout(binding=0) uniform sampler2D value_tex;
            layout(binding=1) uniform sampler2D structural_tex;
            layout(std430, binding=0) writeonly buffer BridgeMaskBuffer {{
                int bridge_mask[];
            }};

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                bool value = texelFetch(value_tex, local_cell, 0).x > 0.5;
                if (mode == 1) {{
                    value = texelFetch(structural_tex, local_cell, 0).x > 0.5 && !value;
                }}
                bridge_mask[cell_index] = value ? 1 : 0;
            }}
            """
        )
        self.programs["publish_bridge_region_labels"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;

            layout(binding=0) uniform sampler2D label_tex;
            layout(std430, binding=0) writeonly buffer BridgeLabelBuffer {{
                int bridge_labels[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeCollapsedMaskBuffer {{
                int bridge_collapsed_mask[];
            }};

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                int label = int(texelFetch(label_tex, local_cell, 0).x + 0.5);
                bridge_labels[cell_index] = label;
                bridge_collapsed_mask[cell_index] = label > 0 ? 1 : 0;
            }}
            """
        )

    def _run_pass(
        self,
        ctx: Any,
        resources: GPUCollapseResources,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        jump: int,
        *,
        read_changed: bool = True,
    ) -> tuple[Any, Any, bool]:
        program = self.programs["propagate"]
        resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
        program["region_size"].value = (width, height)
        program["jump"].value = jump
        resources.structural_tex.use(location=0)
        current.use(location=1)
        scratch.bind_to_image(2, read=False, write=True)
        resources.change_flag.bind_to_storage_buffer(binding=0)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        if not read_changed:
            self._sync_compute_writes(ctx)
            return scratch, current, True
        ctx.finish()
        changed = bool(np.frombuffer(resources.change_flag.read(), dtype=np.uint32, count=1)[0])
        return scratch, current, changed

    def _upload_region_state(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ys = slice(y0, y0 + height)
        xs = slice(x0, x0 + width)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "collapse input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )
        upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
        upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
        upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
        upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
        self.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
        self.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
        if upload_cell_state_from_cpu:
            resources.material_tex.write(world.material_id[ys, xs].astype("f4").tobytes())
            resources.phase_tex.write(world.phase[ys, xs].astype("f4").tobytes())
            resources.cell_flags_tex.write(world.cell_flags[ys, xs].astype("f4").tobytes())
            resources.timer_tex.write(world.timer_pack[ys, xs].astype("f4").tobytes())
            resources.integrity_tex.write(world.integrity[ys, xs].astype("f4").tobytes())
            resources.temp_tex.write(world.cell_temperature[ys, xs].astype("f4").tobytes())
        if upload_island_id_from_cpu:
            resources.island_id_tex.write(world.island_id[ys, xs].astype("f4").tobytes())
        if upload_entity_id_from_cpu:
            resources.entity_id_tex.write(world.entity_id[ys, xs].astype("f4").tobytes())
        if upload_displaced_from_cpu:
            resources.displaced_tex.write(world.placeholder_displaced_material[ys, xs].astype("f4").tobytes())
        self._load_authoritative_bridge_region_inputs(world, resources, x0, y0, width, height)

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _load_authoritative_bridge_region_inputs(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative input state")
        if bridge.ctx is not world.bridge.ctx:
            raise RuntimeError("GPU collapse pipeline cannot consume authoritative bridge state from a separate GL context")

        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
            return

        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if copy_cell_core:
            program = self.programs["load_bridge_region_cell"]
            program["region_size"].value = (int(width), int(height))
            program["region_origin"].value = (int(x0), int(y0))
            program["cell_grid_size"].value = (int(world.width), int(world.height))
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.timer_tex.bind_to_image(3, read=False, write=True)
            resources.integrity_tex.bind_to_image(4, read=False, write=True)
            resources.temp_tex.bind_to_image(5, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(bridge.ctx)

        if copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_region_cell_aux"]
            program["region_size"].value = (int(width), int(height))
            program["region_origin"].value = (int(x0), int(y0))
            program["cell_grid_size"].value = (int(world.width), int(world.height))
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.island_id_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_tex.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(bridge.ctx)

    def _load_authoritative_bridge_pending_region(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if "collapse_delay_pending" not in bridge.gpu_authoritative_resources:
            return
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending state")
        program = self.programs["load_bridge_region_pending"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
        resources.phase_tex.bind_to_image(1, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_pending_region_outputs(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        self._publish_bridge_pending_region_outputs_from_texture(
            world,
            resources,
            resources.support_pong,
            x0,
            y0,
            width,
            height,
        )

    def _publish_bridge_pending_region_outputs_from_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        pending_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending output")
        program = self.programs["publish_bridge_region_pending"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        pending_texture.use(location=0)
        bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("collapse_delay_pending")

    def _publish_bridge_region_mask(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        texture: Any,
        resource_name: str,
        x0: int,
        y0: int,
        width: int,
        height: int,
        *,
        mode: int = 0,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative runtime masks")
        program = self.programs["publish_bridge_region_mask"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["mode"].value = int(mode)
        texture.use(location=0)
        resources.structural_tex.use(location=1)
        bridge.buffers[resource_name].bind_to_storage_buffer(binding=0)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(resource_name)

    def _publish_bridge_region_labels(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        label_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative component labels")
        program = self.programs["publish_bridge_region_labels"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        label_texture.use(location=0)
        bridge.buffers["collapse_component_label"].bind_to_storage_buffer(binding=0)
        bridge.buffers["collapse_collapsed_cell_mask"].bind_to_storage_buffer(binding=1)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("collapse_component_label", "collapse_collapsed_cell_mask")

    def _publish_bridge_region_outputs(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative output state")
            return
        if bridge.ctx is not world.bridge.ctx:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU collapse pipeline cannot publish authoritative state from a separate GL context")
            return
        if "cell_core" not in bridge.gpu_authoritative_resources:
            world._require_gpu_authoritative_resources("collapse output", "cell_core")
            bridge.sync_world(world)

        program = self.programs["publish_bridge_region_cell"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        resources.material_out_tex.use(location=0)
        resources.phase_out_tex.use(location=1)
        resources.cell_flags_out_tex.use(location=2)
        resources.timer_out_tex.use(location=3)
        resources.integrity_out_tex.use(location=4)
        resources.temp_out_tex.use(location=5)
        resources.island_id_out_tex.use(location=6)
        resources.entity_id_out_tex.use(location=7)
        resources.displaced_out_tex.use(location=8)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(ctx, "SHADER_IMAGE_ACCESS_BARRIER_BIT", 0)
            | getattr(ctx, "TEXTURE_FETCH_BARRIER_BIT", 0)
        )

    def _download_region_state(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ys = slice(y0, y0 + height)
        xs = slice(x0, x0 + width)
        world.material_id[ys, xs] = np.rint(
            np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.int32)
        world.phase[ys, xs] = np.rint(
            np.frombuffer(resources.phase_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.uint8)
        world.cell_flags[ys, xs] = np.rint(
            np.frombuffer(resources.cell_flags_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.uint8)
        world.timer_pack[ys, xs] = np.rint(
            np.frombuffer(resources.timer_out_tex.read(), dtype="f4").reshape((height, width, 4))
        ).astype(np.uint8)
        world.integrity[ys, xs] = np.frombuffer(resources.integrity_out_tex.read(), dtype="f4").reshape((height, width))
        world.cell_temperature[ys, xs] = np.frombuffer(resources.temp_out_tex.read(), dtype="f4").reshape((height, width))
        world.island_id[ys, xs] = np.rint(
            np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.int32)
        world.entity_id[ys, xs] = np.rint(
            np.frombuffer(resources.entity_id_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.int32)
        world.placeholder_displaced_material[ys, xs] = np.rint(
            np.frombuffer(resources.displaced_out_tex.read(), dtype="f4").reshape((height, width))
        ).astype(np.int32)

    def _materialize_material_params(self, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None:
            collapse_generation = material_table["collapse_generation_id"].astype(np.int32, copy=True)
            base_integrity = material_table["base_integrity"].astype(np.float32, copy=True)
            spawn_temperature = material_table["spawn_temperature"].astype(np.float32, copy=True)
            valid = material_table["name_hash"] != 0
            collapse_generation[~valid] = 0
            base_integrity[~valid] = 0.0
            spawn_temperature[~valid] = np.nan
        else:
            count = max(
                1,
                int(world.material_collapse_generation_id.shape[0]),
                int(world.material_base_integrity.shape[0]),
                int(world.material_spawn_temperature.shape[0]),
            )
            collapse_generation = np.zeros(count, dtype=np.int32)
            base_integrity = np.zeros(count, dtype=np.float32)
            spawn_temperature = np.full(count, np.nan, dtype=np.float32)
            collapse_generation[: world.material_collapse_generation_id.shape[0]] = world.material_collapse_generation_id
            base_integrity[: world.material_base_integrity.shape[0]] = world.material_base_integrity
            spawn_temperature[: world.material_spawn_temperature.shape[0]] = world.material_spawn_temperature
        spawn_temperature = np.nan_to_num(
            spawn_temperature,
            nan=np.float32(-3.4028234663852886e38),
            neginf=np.float32(-3.4028234663852886e38),
        ).astype(np.float32, copy=False)
        return (
            np.ascontiguousarray(collapse_generation, dtype=np.int32),
            np.ascontiguousarray(base_integrity, dtype=np.float32),
            np.ascontiguousarray(spawn_temperature, dtype=np.float32),
        )

    def _classification_material_params(self, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None:
            structural = material_table["is_structural"].astype(np.int32, copy=True)
            support_anchor = material_table["is_support_anchor"].astype(np.int32, copy=True)
            behavior = material_table["collapse_behavior_id"].astype(np.int32, copy=True)
            valid = material_table["name_hash"] != 0
            structural[~valid] = 0
            support_anchor[~valid] = 0
            behavior[~valid] = 0
        else:
            count = max(
                1,
                int(world.material_is_structural.shape[0]),
                int(world.material_is_support_anchor.shape[0]),
                int(world.material_collapse_behavior.shape[0]),
            )
            structural = np.zeros(count, dtype=np.int32)
            support_anchor = np.zeros(count, dtype=np.int32)
            behavior = np.zeros(count, dtype=np.int32)
            structural[: world.material_is_structural.shape[0]] = world.material_is_structural.astype(np.int32)
            support_anchor[: world.material_is_support_anchor.shape[0]] = world.material_is_support_anchor.astype(np.int32)
            behavior[: world.material_collapse_behavior.shape[0]] = world.material_collapse_behavior.astype(np.int32)
        return (
            np.ascontiguousarray(structural, dtype=np.int32),
            np.ascontiguousarray(support_anchor, dtype=np.int32),
            np.ascontiguousarray(behavior, dtype=np.int32),
        )

    def _write_dynamic_buffer(self, ctx: Any, resources: GPUCollapseResources, name: str, data: np.ndarray) -> None:
        buffer = getattr(resources, name)
        nbytes = max(4, int(data.nbytes))
        if buffer.size < nbytes:
            buffer.release()
            buffer = ctx.buffer(reserve=nbytes, dynamic=True)
            setattr(resources, name, buffer)
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())
