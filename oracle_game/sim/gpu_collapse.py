from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.gpu import ISLAND_RUNTIME_DTYPE
from oracle_game.sim.gpu_collapse_dirty import (
    clear_collapse_structure_dirty_tile_queue_on_gpu,
    ensure_collapse_structure_dirty_tile_queue,
    get_collapse_structure_dirty_tile_bounds,
)
from oracle_game.types import CollapseBehavior, Phase


LOCAL_SIZE = 8
FORMAL_CONNECTED_TILE_LOCAL_SIZE = 32
FORMAL_DEFERRED_REGION_REQUEST_CAPACITY = 256
FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER = "collapse_deferred_region_request_count"
FORMAL_DEFERRED_REGION_REQUEST_BUFFER = "collapse_deferred_region_requests"
FORMAL_CONNECTED_FRONTIER_BUFFER = "collapse_connected_frontier_mask"
FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER = "collapse_connected_frontier_scratch_mask"
FORMAL_CONNECTED_PROCESSED_BUFFER = "collapse_connected_processed_mask"
FORMAL_CONNECTED_TILE_SEED_BUFFER = "collapse_connected_tile_seed_mask"
FORMAL_CONNECTED_TILE_FRONTIER_BUFFER = "collapse_connected_tile_frontier_mask"
FORMAL_CONNECTED_TILE_SCRATCH_BUFFER = "collapse_connected_tile_scratch_mask"
FORMAL_CONNECTED_TILE_LIST_BUFFER = "collapse_connected_tile_list"
FORMAL_CONNECTED_TILE_COUNT_BUFFER = "collapse_connected_tile_count"
FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_dispatch_args"
FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER = "collapse_connected_tile_frontier_list"
FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER = "collapse_connected_tile_frontier_count"
FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_frontier_dispatch_args"
FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER = "collapse_connected_tile_scratch_list"
FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER = "collapse_connected_tile_scratch_count"
FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_scratch_dispatch_args"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER = "collapse_connected_cell_frontier_tile_list"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER = "collapse_connected_cell_frontier_tile_scratch_list"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER = "collapse_connected_cell_frontier_tile_flags"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER = "collapse_connected_cell_frontier_tile_scratch_flags"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER = "collapse_connected_cell_frontier_tile_count"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER = "collapse_connected_cell_frontier_tile_scratch_count"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER = "collapse_connected_cell_frontier_tile_dispatch_args"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER = (
    "collapse_connected_cell_frontier_tile_scratch_dispatch_args"
)
FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT = 2
FORMAL_CONNECTED_DIRTY_JUMP_ROUNDS = 4


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
    component_dispatch_args: Any
    region_flags: Any
    connected_tile_row_masks: Any
    connected_tile_column_masks: Any
    material_structural: Any
    material_support_anchor: Any
    material_collapse_behavior: Any
    material_collapse_generation: Any
    material_base_integrity: Any
    material_spawn_temperature: Any


class GPUCollapsePipeline(GPUPipelineBase):
    FORMAL_EXPAND_LEFT = 1
    FORMAL_EXPAND_TOP = 2
    FORMAL_EXPAND_RIGHT = 4
    FORMAL_EXPAND_BOTTOM = 8

    def __init__(self) -> None:
        self.resources: GPUCollapseResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self._last_formal_connected_tile_mask_name: str | None = None
        self._formal_connected_cell_frontier_generation = 0

    def _profile_enabled(self, world: "WorldEngine") -> bool:
        return bool(getattr(world, "profile_passes_enabled", False))

    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``available`` inherited from GPUPipelineBase.

    def prewarm_formal_connected_resources(self, world: "WorldEngine") -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        self._ensure_resources(ctx, int(world.width), int(world.height))
        self._ensure_formal_connected_frontier_buffers_impl(world)
        ctx.finish()

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
        if self._formal_gpu_frame(world):
            tile_mask_name = self._seed_formal_texture_region_tile_worklist(world, width, height)
            if tile_mask_name is not None:
                return self._solve_formal_connected_tile_support_textures(
                    world,
                    resources,
                    x0,
                    y0,
                    width,
                    height,
                    tile_mask_name,
                    publish_masks=publish_masks,
                )
        current = resources.support_ping
        scratch = resources.support_pong
        jumps = self._formal_jfa_jumps(width, height)
        for jump in jumps:
            current, scratch, _ = self._run_pass(ctx, resources, current, scratch, width, height, jump, read_changed=False)
        if self._formal_gpu_frame(world):
            current, scratch = self._run_formal_support_refine_passes(
                ctx,
                resources,
                current,
                scratch,
                width,
                height,
                jumps,
            )
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
    def _formal_jfa_jumps(width: int, height: int) -> tuple[int, ...]:
        jump = 1
        max_dim = max(int(width), int(height))
        while jump < max_dim:
            jump <<= 1
        jump >>= 1
        jumps: list[int] = []
        while jump >= 2:
            jumps.append(int(jump))
            jump >>= 1
        return tuple(jumps)

    @staticmethod
    def _formal_jfa_profile_jump_bands(jumps: tuple[int, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
        small_start = len(jumps)
        for index, jump in enumerate(jumps):
            if jump <= 1:
                small_start = index
                break
        large_jumps = jumps[:small_start]
        small_jumps = jumps[small_start:]
        bands: list[tuple[str, tuple[int, ...]]] = []
        if large_jumps:
            bands.append(("large", large_jumps))
        if small_jumps:
            bands.append(("small", small_jumps))
        return tuple(bands)

    @staticmethod
    def _formal_support_unit_pass_count(width: int, height: int) -> int:
        """Fixed jump=1 cleanup passes after JFA, not a region-diameter flood."""
        return 2

    @staticmethod
    def _formal_label_unit_pass_count(width: int, height: int) -> int:
        """Fixed jump=1 cleanup passes after JFA, not a region-diameter flood."""
        return 2

    @staticmethod
    def _formal_support_refine_round_count(width: int, height: int) -> int:
        """Single bounded cleanup stage after coarse JFA propagation."""
        return 1

    @staticmethod
    def _formal_label_refine_round_count(width: int, height: int) -> int:
        """Single bounded cleanup stage after coarse JFA propagation."""
        return 1

    def _run_formal_support_refine_passes(
        self,
        ctx: Any,
        resources: GPUCollapseResources,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        jumps: tuple[int, ...],
    ) -> tuple[Any, Any]:
        unit_pass_count = self._formal_support_unit_pass_count(width, height)
        refine_round_count = self._formal_support_refine_round_count(width, height)
        for round_index in range(refine_round_count):
            for _ in range(unit_pass_count):
                current, scratch, _ = self._run_pass(
                    ctx,
                    resources,
                    current,
                    scratch,
                    width,
                    height,
                    1,
                    read_changed=False,
                )
            if round_index + 1 >= refine_round_count:
                continue
            for jump in jumps:
                current, scratch, _ = self._run_pass(
                    ctx,
                    resources,
                    current,
                    scratch,
                    width,
                    height,
                    jump,
                    read_changed=False,
                )
        return current, scratch

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
        if self._formal_gpu_frame(world):
            return self._expand_formal_region_to_component_bbox(world, x0, y0, x1, y1)
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

    def _expand_formal_region_to_component_bbox(
        self,
        world: "WorldEngine",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int, int]:
        world_width = int(world.width)
        world_height = int(world.height)
        seed_x0 = max(0, min(world_width, int(x0)))
        seed_y0 = max(0, min(world_height, int(y0)))
        seed_x1 = max(0, min(world_width, int(x1)))
        seed_y1 = max(0, min(world_height, int(y1)))
        if seed_x0 >= seed_x1 or seed_y0 >= seed_y1:
            return (seed_x0, seed_y0, seed_x1, seed_y1)

        # Formal frames must not read component metadata back to the CPU to steer bbox growth.
        # The caller supplies an already halo-expanded dirty/event region; keep it tile-aligned
        # and let GPU eligibility masks restrict materialization to dirty-connected structure.
        tile_size = max(1, int(getattr(world.active, "tile_size", 32)))

        def align_down(value: int) -> int:
            return max(0, (int(value) // tile_size) * tile_size)

        def align_up(value: int, limit: int) -> int:
            return min(int(limit), ((int(value) + tile_size - 1) // tile_size) * tile_size)

        search_x0 = align_down(seed_x0)
        search_y0 = align_down(seed_y0)
        search_x1 = align_up(seed_x1, world_width)
        search_y1 = align_up(seed_y1, world_height)
        return (search_x0, search_y0, search_x1, search_y1)

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

    def connected_structural_region_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        seed_x0: int,
        seed_y0: int,
        seed_x1: int,
        seed_y1: int,
    ) -> Any:
        if width == 0 or height == 0:
            return resources.cell_flags_out_tex
        self.seed_structural_region_texture(
            world,
            resources,
            width,
            height,
            max(0, int(seed_x0)),
            max(0, int(seed_y0)),
            min(int(width), int(seed_x1)),
            min(int(height), int(seed_y1)),
        )
        connected_texture = self.solve_region_textures(
            world,
            resources,
            width,
            height,
            x0=0,
            y0=0,
            publish_masks=False,
        )
        self.copy_mask_texture(world, resources, connected_texture, resources.cell_flags_out_tex, width, height)
        return resources.cell_flags_out_tex

    def copy_mask_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        source_texture: Any,
        target_texture: Any,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return
        self._ensure_programs(ctx)
        program = self.programs["copy_mask_texture"]
        program["region_size"].value = (int(width), int(height))
        source_texture.use(location=0)
        target_texture.bind_to_image(1, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _copy_mask_texture_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        source_texture: Any,
        target_texture: Any,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["copy_mask_texture_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected mask texture copy requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
        source_texture.use(location=0)
        target_texture.bind_to_image(1, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)

    def _ensure_formal_connected_axis_mask_buffers(
        self,
        ctx: Any,
        resources: GPUCollapseResources,
        tile_count: int,
    ) -> None:
        required_bytes = max(
            4,
            int(max(1, tile_count)) * FORMAL_CONNECTED_TILE_LOCAL_SIZE * np.dtype(np.uint32).itemsize,
        )
        if resources.connected_tile_row_masks.size < required_bytes:
            resources.connected_tile_row_masks.release()
            resources.connected_tile_row_masks = ctx.buffer(reserve=required_bytes, dynamic=True)
        if resources.connected_tile_column_masks.size < required_bytes:
            resources.connected_tile_column_masks.release()
            resources.connected_tile_column_masks = ctx.buffer(reserve=required_bytes, dynamic=True)

    def _build_formal_connected_axis_masks(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        source_texture: Any,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected axis masks require tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        self._ensure_formal_connected_axis_mask_buffers(ctx, resources, tile_width * tile_height)
        program = self.programs["build_formal_connected_axis_masks"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected axis mask build requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        source_texture.use(location=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
        resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)

    def detect_connected_internal_boundary_flags(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        eligibility_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
    ) -> int:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return 0
        internal_edges = (
            1 if int(x0) > 0 else 0,
            1 if int(y0) > 0 else 0,
            1 if int(x0) + int(width) < int(world.width) else 0,
            1 if int(y0) + int(height) < int(world.height) else 0,
        )
        if not any(internal_edges):
            return 0

        self._ensure_programs(ctx)
        resources.region_flags.write(np.zeros(1, dtype=np.uint32).tobytes())
        program = self.programs["detect_connected_internal_boundary_flags"]
        program["region_size"].value = (int(width), int(height))
        program["internal_edges"].value = internal_edges
        eligibility_texture.use(location=0)
        resources.region_flags.bind_to_storage_buffer(binding=0)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        return 0

    def _ensure_formal_deferred_region_request_buffers(self, world: "WorldEngine") -> tuple[Any, Any, int]:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for deferred region requests")
        active_tile_count = int(getattr(world.active, "tile_width", 1)) * int(getattr(world.active, "tile_height", 1))
        capacity = max(4, FORMAL_DEFERRED_REGION_REQUEST_CAPACITY, active_tile_count * 4)
        request_bytes = capacity * 4 * np.dtype(np.int32).itemsize
        count_name = FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER
        request_name = FORMAL_DEFERRED_REGION_REQUEST_BUFFER
        if count_name not in bridge.buffers or bridge.buffers[count_name].size < 4:
            existing = bridge.buffers.get(count_name)
            if existing is not None:
                existing.release()
            bridge.buffers[count_name] = bridge.ctx.buffer(reserve=4, dynamic=True)
            bridge.buffers[count_name].write(np.zeros(1, dtype=np.uint32).tobytes())
        if request_name not in bridge.buffers or bridge.buffers[request_name].size < request_bytes:
            existing = bridge.buffers.get(request_name)
            if existing is not None:
                existing.release()
            bridge.buffers[request_name] = bridge.ctx.buffer(reserve=request_bytes, dynamic=True)
            bridge.buffers[request_name].write(np.zeros(capacity * 4, dtype=np.int32).tobytes())
        return bridge.buffers[count_name], bridge.buffers[request_name], capacity

    def _ensure_formal_connected_frontier_buffers(self, world: "WorldEngine") -> tuple[str, str]:
        with self._profile_pass(world, "connected_frontier_buffer_prepare"):
            return self._ensure_formal_connected_frontier_buffers_impl(world)

    def _ensure_formal_connected_frontier_buffers_impl(self, world: "WorldEngine") -> tuple[str, str]:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected frontier expansion")
        self._ensure_programs(bridge.ctx)
        cell_count = max(1, int(world.width) * int(world.height))
        tile_count = max(1, int(getattr(world.active, "tile_width", 1)) * int(getattr(world.active, "tile_height", 1)))
        cell_required_bytes = cell_count * np.dtype(np.int32).itemsize
        tile_required_bytes = tile_count * np.dtype(np.int32).itemsize
        tile_list_bytes = max(8, tile_count * 2 * np.dtype(np.int32).itemsize)
        frontier_tile_list_bytes = max(8, tile_count * 2 * np.dtype(np.int32).itemsize)
        frontier_tile_flags_bytes = max(4, tile_count * np.dtype(np.uint32).itemsize)
        for name in (
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
            FORMAL_CONNECTED_PROCESSED_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < cell_required_bytes:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=cell_required_bytes, dynamic=True)
        for name in (
            FORMAL_CONNECTED_TILE_SEED_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < tile_required_bytes:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=tile_required_bytes, dynamic=True)
                bridge.buffers[name].write(np.zeros(tile_count, dtype=np.int32).tobytes())
        for name in (
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < tile_list_bytes:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=tile_list_bytes, dynamic=True)
        for name in (
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < 4:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=4, dynamic=True)
                bridge.buffers[name].write(np.zeros(1, dtype=np.uint32).tobytes())
        for name in (
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < 12:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=12, dynamic=True)
                bridge.buffers[name].write(np.asarray([0, 1, 1], dtype=np.uint32).tobytes())
        self._clear_formal_connected_tile_mask_buffers(world)
        for name in (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < frontier_tile_list_bytes:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=frontier_tile_list_bytes, dynamic=True)
        for name in (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < frontier_tile_flags_bytes:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=frontier_tile_flags_bytes, dynamic=True)
            bridge.buffers[name].write(np.zeros(tile_count, dtype=np.uint32).tobytes())
        for name in (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < 4:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=4, dynamic=True)
            bridge.buffers[name].write(np.zeros(1, dtype=np.uint32).tobytes())
        for name in (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        ):
            if name not in bridge.buffers or bridge.buffers[name].size < 12:
                existing = bridge.buffers.get(name)
                if existing is not None:
                    existing.release()
                bridge.buffers[name] = bridge.ctx.buffer(reserve=12, dynamic=True)
            bridge.buffers[name].write(np.asarray([0, 1, 1], dtype=np.uint32).tobytes())
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
            FORMAL_CONNECTED_PROCESSED_BUFFER,
            FORMAL_CONNECTED_TILE_SEED_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )
        return FORMAL_CONNECTED_FRONTIER_BUFFER, FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER

    def _seed_formal_texture_region_tile_worklist(
        self,
        world: "WorldEngine",
        width: int,
        height: int,
    ) -> str | None:
        self._ensure_formal_connected_frontier_buffers(world)
        bridge = world.bridge
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected tile worklists")
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected texture region propagation requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        region_tile_width = (int(width) + tile_size - 1) // tile_size
        region_tile_height = (int(height) + tile_size - 1) // tile_size
        if region_tile_width <= 0 or region_tile_height <= 0:
            return None
        if region_tile_width > tile_width or region_tile_height > tile_height:
            return None

        tile_count = tile_width * tile_height
        tile_mask = np.zeros(tile_count, dtype=np.int32)
        tiles = [
            (tile_x, tile_y)
            for tile_y in range(region_tile_height)
            for tile_x in range(region_tile_width)
        ]
        tile_array = np.asarray(tiles, dtype=np.int32)
        for tile_x, tile_y in tile_array.tolist():
            tile_mask[int(tile_y) * tile_width + int(tile_x)] = 1

        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].write(tile_mask.tobytes())
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].write(tile_array.tobytes())
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].write(
            np.asarray([len(tiles)], dtype=np.uint32).tobytes()
        )
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].write(
            np.asarray([len(tiles), 1, 1], dtype=np.uint32).tobytes()
        )
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER

    def _clear_formal_connected_cell_buffer_names(self, world: "WorldEngine", buffer_names: tuple[str, ...]) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        cell_count = max(1, int(world.width) * int(world.height))
        program = self.programs["clear_formal_connected_cell_buffer"]
        program["cell_count"].value = int(cell_count)
        for name in buffer_names:
            bridge.buffers[name].bind_to_storage_buffer(binding=0)
            program.run((cell_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(*buffer_names)

    def _clear_formal_connected_tile_mask_buffers(self, world: "WorldEngine") -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        program = self.programs["clear_formal_connected_tile_masks_by_list"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected tile mask clear requires ComputeShader.run_indirect")
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_SCRATCH_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_SEED_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        self._clear_formal_connected_tile_worklists(world)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_TILE_SEED_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_BUFFER,
        )

    def _clear_formal_connected_tile_worklist(
        self,
        world: "WorldEngine",
        count_name: str,
        dispatch_args_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        program = self.programs["clear_formal_connected_tile_worklist"]
        bridge.buffers[count_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=1)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(count_name, dispatch_args_name)

    def _clear_formal_connected_tile_worklists(self, world: "WorldEngine") -> None:
        self._clear_formal_connected_tile_worklist(
            world,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        self._clear_formal_connected_tile_worklist(
            world,
            FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
        )
        self._clear_formal_connected_tile_worklist(
            world,
            FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )

    def _clear_formal_connected_cell_buffer_connected_tiles(
        self,
        world: "WorldEngine",
        buffer_name: str,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["clear_formal_connected_cell_buffer_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected tile cell clear requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        bridge.buffers[buffer_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(buffer_name)

    def reset_formal_connected_frontier(
        self,
        world: "WorldEngine",
        seed_rect: tuple[int, int, int, int],
    ) -> tuple[str, str]:
        current_name, scratch_name = self._ensure_formal_connected_frontier_buffers(world)
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        program = self.programs["seed_formal_connected_frontier_rect"]
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["seed_rect"].value = tuple(int(value) for value in seed_rect)
        world.bridge.buffers[current_name].bind_to_storage_buffer(binding=0)
        group_x = (int(world.width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (int(world.height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        world.bridge.mark_gpu_authoritative(current_name, scratch_name)
        return current_name, scratch_name

    def clear_formal_connected_frontier_buffer(self, world: "WorldEngine", buffer_name: str) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected frontier expansion")
        if buffer_name not in bridge.buffers:
            self._ensure_formal_connected_frontier_buffers(world)
        cell_count = max(1, int(world.width) * int(world.height))
        self._ensure_programs(bridge.ctx)
        program = self.programs["clear_formal_connected_cell_buffer"]
        program["cell_count"].value = int(cell_count)
        bridge.buffers[buffer_name].bind_to_storage_buffer(binding=0)
        program.run((cell_count + 255) // 256, 1, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(buffer_name)

    def clear_formal_deferred_region_requests(self, world: "WorldEngine") -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        count_buffer = bridge.buffers.get(FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER)
        if count_buffer is None:
            return
        count_buffer.write(np.zeros(1, dtype=np.uint32).tobytes())
        bridge.mark_gpu_authoritative(FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER)

    def execute_formal_connected_expansion(
        self,
        world: "WorldEngine",
        seed_rect: tuple[int, int, int, int],
        *,
        resource_region: tuple[int, int, int, int] | None = None,
    ) -> int:
        if not self._formal_gpu_frame(world):
            raise RuntimeError("formal connected collapse requires an active formal GPU frame")
        width = int(world.width)
        height = int(world.height)
        if width <= 0 or height <= 0:
            raise ValueError("formal connected collapse requires a non-empty world")
        self._ensure_formal_connected_frontier_buffers(world)
        if resource_region is None:
            resource_region = (0, 0, width, height)
        outcome_resources, outcome_x0, outcome_y0, outcome_width, outcome_height = self._solve_formal_connected_tile_textures(
            world,
            seed_rect,
            resource_region=resource_region,
        )
        return self.materialize_component_texture_formal(
            world,
            outcome_resources.phase_out_tex,
            outcome_width,
            outcome_height,
            outcome_x0,
            outcome_y0,
            tile_mask_name=self._last_formal_connected_tile_mask_name,
        )

    def execute_formal_connected_dirty_tile_queue(self, world: "WorldEngine") -> int:
        if not self._formal_gpu_frame(world):
            raise RuntimeError("formal dirty tile collapse requires an active formal GPU frame")
        width = int(world.width)
        height = int(world.height)
        if width <= 0 or height <= 0:
            raise ValueError("formal dirty tile collapse requires a non-empty world")
        ensure_collapse_structure_dirty_tile_queue(world)
        self._ensure_formal_connected_frontier_buffers(world)
        outcome_resources, outcome_x0, outcome_y0, outcome_width, outcome_height = (
            self._solve_formal_connected_dirty_tile_textures(world)
        )
        component_capacity = self.materialize_component_texture_formal(
            world,
            outcome_resources.phase_out_tex,
            outcome_width,
            outcome_height,
            outcome_x0,
            outcome_y0,
            tile_mask_name=self._last_formal_connected_tile_mask_name,
        )
        clear_collapse_structure_dirty_tile_queue_on_gpu(world)
        return component_capacity

    def solve_formal_connected_region_textures(
        self,
        world: "WorldEngine",
        seed_rect: tuple[int, int, int, int],
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        return self._solve_formal_connected_tile_textures(world, seed_rect)

    def _solve_formal_connected_dirty_tile_textures(
        self,
        world: "WorldEngine",
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        resource_region = (0, 0, int(world.width), int(world.height))
        resources, x0, y0, width, height = self._prepare_formal_connected_tile_resources_without_input_upload(
            world,
            resource_region,
        )
        with self._profile_pass(world, "tile_region_worklist"):
            tile_mask_name = self._seed_formal_texture_region_tile_worklist(world, width, height)
            if tile_mask_name is None:
                raise RuntimeError("formal dirty tile collapse requires a non-empty connected tile worklist")
            self._last_formal_connected_tile_mask_name = tile_mask_name
        with self._profile_pass(world, "connected_bridge_input_load"):
            self._load_authoritative_bridge_connected_tile_inputs(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        with self._profile_pass(world, "classify_filter"):
            self._classify_formal_connected_tile_textures(world, resources, tile_mask_name, x0, y0, width, height)
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.structural_tex,
                "collapse_structural_mask",
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name,
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
                tile_mask_name=tile_mask_name,
            )

        with self._profile_pass(world, "support_jfa"):
            supported_texture = self._solve_formal_connected_tile_support_textures(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        outcome_resources, _, _ = self.resolve_supported_outcome_textures(
            world,
            resources,
            supported_texture,
            x0,
            y0,
            width,
            height,
            eligibility_texture=resources.structural_tex,
            tile_mask_name=tile_mask_name,
        )
        return outcome_resources, x0, y0, width, height

    def _solve_formal_connected_tile_textures(
        self,
        world: "WorldEngine",
        seed_rect: tuple[int, int, int, int],
        *,
        resource_region: tuple[int, int, int, int] | None = None,
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        resources, x0, y0, width, height = self._prepare_formal_connected_tile_resources(world, resource_region)
        with self._profile_pass(world, "tile_region_worklist"):
            tile_mask_name = self._seed_formal_texture_region_tile_worklist(world, width, height)
            if tile_mask_name is None:
                raise RuntimeError("formal connected collapse requires a non-empty connected tile worklist")
            self._last_formal_connected_tile_mask_name = tile_mask_name
        with self._profile_pass(world, "classify_filter"):
            self._classify_formal_connected_tile_textures(world, resources, tile_mask_name, x0, y0, width, height)
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.structural_tex,
                "collapse_structural_mask",
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name,
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
                tile_mask_name=tile_mask_name,
            )

        with self._profile_pass(world, "support_jfa"):
            supported_texture = self._solve_formal_connected_tile_support_textures(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        outcome_resources, _, _ = self.resolve_supported_outcome_textures(
            world,
            resources,
            supported_texture,
            x0,
            y0,
            width,
            height,
            eligibility_texture=resources.structural_tex,
            tile_mask_name=tile_mask_name,
        )
        return outcome_resources, x0, y0, width, height

    def _prepare_formal_connected_tile_resources(
        self,
        world: "WorldEngine",
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        with self._profile_pass(world, "tile_resource_prepare"):
            return self._prepare_formal_connected_tile_resources_impl(world, region)

    def _prepare_formal_connected_tile_resources_without_input_upload(
        self,
        world: "WorldEngine",
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        with self._profile_pass(world, "tile_resource_prepare"):
            return self._prepare_formal_connected_tile_resources_impl(
                world,
                region,
                upload_region_state=False,
            )

    def _prepare_formal_connected_tile_resources_impl(
        self,
        world: "WorldEngine",
        region: tuple[int, int, int, int] | None = None,
        *,
        upload_region_state: bool = True,
    ) -> tuple[GPUCollapseResources, int, int, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        x0, y0, x1, y1 = self._clamp_formal_connected_region(world, region)
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            raise ValueError("formal connected world classification requires a non-empty world")
        with self._profile_pass(world, "tile_resource_prepare.ensure_programs"):
            self._ensure_programs(ctx)
        with self._profile_pass(world, "tile_resource_prepare.ensure_resources"):
            resources = self._ensure_resources(ctx, width, height)
        if upload_region_state:
            with self._profile_pass(world, "tile_resource_prepare.upload_region_state"):
                self._upload_region_state(world, resources, x0, y0, width, height)
        with self._profile_pass(world, "tile_resource_prepare.material_params"):
            structural_params, support_params, behavior_params = self._classification_material_params(world)
        with self._profile_pass(world, "tile_resource_prepare.material_buffer_writes"):
            self._write_dynamic_buffer(ctx, resources, "material_structural", structural_params)
            self._write_dynamic_buffer(ctx, resources, "material_support_anchor", support_params)
            self._write_dynamic_buffer(ctx, resources, "material_collapse_behavior", behavior_params)
        return resources, x0, y0, width, height

    def _clamp_formal_connected_region(
        self,
        world: "WorldEngine",
        region: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        world_width = int(world.width)
        world_height = int(world.height)
        if region is None:
            return (0, 0, world_width, world_height)
        x0, y0, x1, y1 = (int(value) for value in region)
        return (
            max(0, min(world_width, x0)),
            max(0, min(world_height, y0)),
            max(0, min(world_width, x1)),
            max(0, min(world_height, y1)),
        )

    def _formal_connected_dirty_tile_queue_resource_region(
        self,
        world: "WorldEngine",
    ) -> tuple[int, int, int, int]:
        dirty_tile_bounds = get_collapse_structure_dirty_tile_bounds(world)
        if dirty_tile_bounds is None:
            raise RuntimeError("formal dirty tile queue requires CPU-known dirty tile bounds")
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        tile_x0, tile_y0, tile_x1, tile_y1 = (int(value) for value in dirty_tile_bounds)
        tile_x0 = max(0, min(tile_width, tile_x0 - 1))
        tile_y0 = max(0, min(tile_height, tile_y0 - 1))
        tile_x1 = max(0, min(tile_width, tile_x1 + 1))
        tile_y1 = max(0, min(tile_height, tile_y1 + 1))
        if tile_x0 >= tile_x1 or tile_y0 >= tile_y1:
            raise RuntimeError("formal dirty tile queue requires non-empty dirty tile bounds")
        return self._formal_connected_dirty_tile_resource_region_from_tile_bounds(
            world,
            tile_x0,
            tile_y0,
            tile_x1,
            tile_y1,
        )

    def _formal_connected_dirty_tile_resource_region_from_tile_bounds(
        self,
        world: "WorldEngine",
        tile_x0: int,
        tile_y0: int,
        tile_x1: int,
        tile_y1: int,
    ) -> tuple[int, int, int, int]:
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        world_width = int(world.width)
        world_height = int(world.height)
        rx0 = max(0, min(tile_width, int(tile_x0)))
        ry0 = max(0, min(tile_height, int(tile_y0)))
        rx1 = max(0, min(tile_width, int(tile_x1)))
        ry1 = max(0, min(tile_height, int(tile_y1)))
        touches_x_edge = rx0 <= 0 or rx1 >= tile_width
        touches_y_edge = ry0 <= 0 or ry1 >= tile_height
        if rx0 >= rx1 or ry0 >= ry1:
            raise RuntimeError("formal dirty tile queue requires non-empty dirty tile bounds")
        if not touches_x_edge and not touches_y_edge:
            return (
                max(0, min(world_width, rx0 * tile_size)),
                max(0, min(world_height, ry0 * tile_size)),
                max(0, min(world_width, rx1 * tile_size)),
                max(0, min(world_height, ry1 * tile_size)),
            )

        orthogonal_margin_tiles = 2
        seed_x0, seed_y0, seed_x1, seed_y1 = rx0, ry0, rx1, ry1

        def bounded_tile_span(lo: int, hi: int, limit: int) -> tuple[int, int]:
            if limit <= 0:
                return (0, 0)
            span = max(1, int(hi) - int(lo))
            expanded_lo = max(0, int(lo) - orthogonal_margin_tiles)
            expanded_hi = min(int(limit), int(hi) + orthogonal_margin_tiles)
            if expanded_lo == 0 and expanded_hi == int(limit) and span < int(limit):
                guard = 1
                target_span = min(int(limit) - guard, max(span, span + orthogonal_margin_tiles * 2))
                center = (int(lo) + int(hi)) // 2
                expanded_lo = max(0, min(int(limit) - target_span, center - target_span // 2))
                expanded_hi = expanded_lo + target_span
            return (expanded_lo, expanded_hi)

        if touches_x_edge:
            rx0, rx1 = 0, tile_width
            ry0, ry1 = bounded_tile_span(seed_y0, seed_y1, tile_height)
        if touches_y_edge:
            ry0, ry1 = 0, tile_height
            rx0, rx1 = bounded_tile_span(seed_x0, seed_x1, tile_width)
        if touches_x_edge and touches_y_edge:
            rx0, rx1 = 0, tile_width
            ry0, ry1 = bounded_tile_span(seed_y0, seed_y1, tile_height)
            if ry0 == 0 and ry1 == tile_height and tile_width > 1:
                rx0, rx1 = bounded_tile_span(seed_x0, seed_x1, tile_width)

        return (
            max(0, min(world_width, rx0 * tile_size)),
            max(0, min(world_height, ry0 * tile_size)),
            max(0, min(world_width, rx1 * tile_size)),
            max(0, min(world_height, ry1 * tile_size)),
        )

    def _formal_connected_resource_region_from_bbox(
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

        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
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

    @staticmethod
    def _local_formal_connected_rect(
        rect: tuple[int, int, int, int],
        region_x0: int,
        region_y0: int,
        region_width: int,
        region_height: int,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = (int(value) for value in rect)
        return (
            max(0, min(int(region_width), x0 - int(region_x0))),
            max(0, min(int(region_height), y0 - int(region_y0))),
            max(0, min(int(region_width), x1 - int(region_x0))),
            max(0, min(int(region_height), y1 - int(region_y0))),
        )

    def _classify_formal_connected_tile_textures(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        tile_mask_name: str,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        _, _, behavior_params = self._classification_material_params(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile classification requires tile_size <= 32")
        program = self.programs["classify_formal_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected tile classification requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.structural_tex.bind_to_image(2, read=False, write=True)
        resources.support_ping.bind_to_image(3, read=False, write=True)
        resources.material_out_tex.bind_to_image(4, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=3)
        resources.material_structural.bind_to_storage_buffer(binding=0)
        resources.material_support_anchor.bind_to_storage_buffer(binding=1)
        resources.material_collapse_behavior.bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=5)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    def _solve_formal_connected_frontier_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        seed_rect: tuple[int, int, int, int],
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> Any:
        scratch_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )
        with self._profile_pass(world, "cell_frontier_seed_list_build"):
            self._seed_formal_connected_cell_frontier(world, resources, seed_rect, width, height, tile_mask_name)
        cell_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        current_name = FORMAL_CONNECTED_FRONTIER_BUFFER
        scratch_name = FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER
        with self._profile_pass(world, "cell_frontier_loop"):
            for jump in self._formal_connected_cell_jump_schedule(width, height):
                self._clear_formal_connected_cell_frontier_tiles(world, scratch_frontier)
                self._expand_formal_connected_cell_frontier(
                    world,
                    resources,
                    width,
                    height,
                    current_name,
                    scratch_name,
                    tile_mask_name,
                    current_frontier=cell_frontier,
                    next_frontier=scratch_frontier,
                    jump=jump,
                    jump_generation=self._next_formal_connected_cell_frontier_generation(),
                )
                cell_frontier, scratch_frontier = scratch_frontier, cell_frontier
        with self._profile_pass(world, "cell_frontier_final_copy"):
            self._copy_formal_connected_buffer_to_texture(
                world,
                resources,
                current_name,
                resources.cell_flags_out_tex,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        return resources.cell_flags_out_tex

    def _solve_formal_connected_dirty_cell_frontier_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> Any:
        scratch_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )
        with self._profile_pass(world, "cell_frontier_seed_list_build"):
            self._seed_formal_connected_cell_frontier_from_dirty_queue(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        cell_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        current_name = FORMAL_CONNECTED_FRONTIER_BUFFER
        scratch_name = FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER
        with self._profile_pass(world, "cell_frontier_loop"):
            for jump in self._formal_connected_dirty_jump_schedule(width, height):
                self._clear_formal_connected_cell_frontier_tiles(world, scratch_frontier)
                self._expand_formal_connected_cell_frontier(
                    world,
                    resources,
                    width,
                    height,
                    current_name,
                    scratch_name,
                    tile_mask_name,
                    current_frontier=cell_frontier,
                    next_frontier=scratch_frontier,
                    jump=jump,
                    jump_generation=self._next_formal_connected_cell_frontier_generation(),
                )
                cell_frontier, scratch_frontier = scratch_frontier, cell_frontier
        with self._profile_pass(world, "cell_frontier_final_copy"):
            self._copy_formal_connected_buffer_to_texture(
                world,
                resources,
                current_name,
                resources.cell_flags_out_tex,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        return resources.cell_flags_out_tex

    def _formal_connected_expansion_pass_count(self, world: "WorldEngine") -> int:
        return len(self._formal_connected_tile_jump_schedule(world))

    def _formal_connected_tile_jump_schedule(self, world: "WorldEngine") -> tuple[int, ...]:
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        return self._formal_connected_cell_jump_schedule(tile_width, tile_height)

    def _formal_connected_tile_refine_pass_count(self, world: "WorldEngine") -> int:
        return FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT

    def _formal_connected_tile_support_frontier_pass_count(self, world: "WorldEngine") -> int:
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        return max(1, tile_width + tile_height)

    def _formal_connected_component_label_frontier_pass_count(self, world: "WorldEngine") -> int:
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        return max(1, tile_width + tile_height)

    def _formal_connected_dirty_tile_jump_schedule(self, world: "WorldEngine") -> tuple[int, ...]:
        return self._formal_connected_tile_jump_schedule(world)

    @staticmethod
    def _formal_connected_dirty_jump_schedule(width: int, height: int) -> tuple[int, ...]:
        return GPUCollapsePipeline._formal_connected_cell_jump_schedule(width, height)

    @staticmethod
    def _formal_connected_cell_jump_schedule(width: int, height: int) -> tuple[int, ...]:
        jumps = GPUCollapsePipeline._formal_jfa_jumps(width, height)
        cleanup = (1,) * FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT
        round_schedule = jumps + cleanup
        if not round_schedule:
            return cleanup or (1,)
        rounds = min(FORMAL_CONNECTED_DIRTY_JUMP_ROUNDS, max(1, len(jumps)))
        return tuple(jump for _ in range(rounds) for jump in round_schedule)

    def _next_formal_connected_cell_frontier_generation(self) -> int:
        self._formal_connected_cell_frontier_generation += 1
        return self._formal_connected_cell_frontier_generation

    def _solve_formal_connected_tile_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        seed_rect: tuple[int, int, int, int],
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> str:
        self._seed_formal_connected_tile_frontier(world, resources, seed_rect, x0, y0, width, height)
        scratch_frontier = (
            FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )
        connected_frontier = (
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        for jump in self._formal_connected_tile_jump_schedule(world):
            self._clear_formal_connected_tile_worklist(world, scratch_frontier[1], scratch_frontier[2])
            self._expand_formal_connected_tile_frontier(
                world,
                resources,
                FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
                x0,
                y0,
                width,
                height,
                current_frontier=connected_frontier,
                next_frontier=scratch_frontier,
                jump=jump,
            )
        self._last_formal_connected_tile_mask_name = FORMAL_CONNECTED_TILE_FRONTIER_BUFFER
        return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER

    def _solve_formal_connected_dirty_tile_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> str:
        self._seed_formal_connected_tile_frontier_from_dirty_queue(world, resources, x0, y0, width, height)
        scratch_frontier = (
            FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
        )
        connected_frontier = (
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        for jump in self._formal_connected_dirty_tile_jump_schedule(world):
            self._clear_formal_connected_tile_worklist(world, scratch_frontier[1], scratch_frontier[2])
            self._expand_formal_connected_tile_frontier(
                world,
                resources,
                FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
                x0,
                y0,
                width,
                height,
                current_frontier=connected_frontier,
                next_frontier=scratch_frontier,
                jump=jump,
            )
        self._last_formal_connected_tile_mask_name = FORMAL_CONNECTED_TILE_FRONTIER_BUFFER
        return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER

    def _compact_formal_connected_tile_mask(self, world: "WorldEngine", tile_mask_name: str) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if tile_mask_name not in bridge.buffers:
            raise RuntimeError("formal connected tile mask is not allocated")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        tile_count = tile_width * tile_height

        clear_program = self.programs["clear_formal_connected_tile_worklist"]
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        compact_program = self.programs["compact_formal_connected_tile_mask"]
        compact_program["tile_grid_size"].value = (tile_width, tile_height)
        compact_program["tile_count"].value = int(tile_count)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=3)
        compact_program.run((tile_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )

    def _seed_formal_connected_tile_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        seed_rect: tuple[int, int, int, int],
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        self._clear_formal_connected_tile_mask_buffers(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        seed_x0, seed_y0, seed_x1, seed_y1 = (int(value) for value in seed_rect)
        tile_x0 = max(0, min(tile_width, seed_x0 // tile_size))
        tile_y0 = max(0, min(tile_height, seed_y0 // tile_size))
        tile_x1 = max(0, min(tile_width, (seed_x1 + tile_size - 1) // tile_size))
        tile_y1 = max(0, min(tile_height, (seed_y1 + tile_size - 1) // tile_size))
        if tile_x0 >= tile_x1 or tile_y0 >= tile_y1:
            return
        program = self.programs["seed_formal_connected_tile_frontier"]
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        program["seed_rect"].value = tuple(int(value) for value in seed_rect)
        program["seed_tile_origin"].value = (tile_x0, tile_y0)
        _, _, behavior_params = self._classification_material_params(world)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=1)
        resources.material_structural.bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER].bind_to_storage_buffer(binding=5)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER].bind_to_storage_buffer(binding=6)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=7)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=8)
        program.run(tile_x1 - tile_x0, tile_y1 - tile_y0, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
        )

    def _seed_formal_connected_tile_frontier_from_dirty_queue(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        self._clear_formal_connected_tile_mask_buffers(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_queue is None:
            raise RuntimeError("formal dirty tile expansion requires dirty tile queue buffers")
        dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        program = self.programs["seed_formal_connected_tile_frontier_from_dirty_queue"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal dirty tile frontier requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["region_tile_origin"].value = (int(x0) // int(tile_size), int(y0) // int(tile_size))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        _, _, behavior_params = self._classification_material_params(world)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=1)
        resources.material_structural.bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER].bind_to_storage_buffer(binding=5)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER].bind_to_storage_buffer(binding=6)
        bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=7)
        dirty_count.bind_to_storage_buffer(binding=8)
        dirty_list.bind_to_storage_buffer(binding=9)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=10)
        program.run_indirect(dirty_dispatch_args)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
        )

    def _expand_formal_connected_tile_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        tile_mask_name: str,
        x0: int,
        y0: int,
        width: int,
        height: int,
        *,
        current_frontier: tuple[str, str, str],
        next_frontier: tuple[str, str, str],
        jump: int = 1,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        program = self.programs["expand_formal_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected tile frontier requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        program["jump"].value = int(max(1, jump))
        _, _, behavior_params = self._classification_material_params(world)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        current_list_name, current_count_name, current_dispatch_args_name = current_frontier
        next_list_name, next_count_name, next_dispatch_args_name = next_frontier
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=1)
        resources.material_structural.bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[current_count_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[current_list_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[next_count_name].bind_to_storage_buffer(binding=7)
        bridge.buffers[next_list_name].bind_to_storage_buffer(binding=8)
        bridge.buffers[next_dispatch_args_name].bind_to_storage_buffer(binding=9)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=10)
        program.run_indirect(bridge.buffers[current_dispatch_args_name])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            tile_mask_name,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
            current_count_name,
            current_list_name,
            current_dispatch_args_name,
            next_count_name,
            next_list_name,
            next_dispatch_args_name,
        )

    def _clear_formal_connected_cell_frontier_tiles(
        self,
        world: "WorldEngine",
        frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        flags_name, list_name, count_name, dispatch_args_name = frontier
        clear_program = self.programs["clear_formal_connected_cell_frontier_tile_flags_by_list"]
        if not hasattr(clear_program, "run_indirect"):
            raise RuntimeError("formal connected cell frontier flag clear requires ComputeShader.run_indirect")
        clear_program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        bridge.buffers[flags_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[list_name].bind_to_storage_buffer(binding=1)
        clear_program.run_indirect(bridge.buffers[dispatch_args_name])
        self._sync_compute_writes(ctx)

        program = self.programs["reset_formal_connected_cell_frontier_tiles"]
        bridge.buffers[flags_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[count_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=2)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(flags_name, count_name, dispatch_args_name)

    def _accumulate_formal_connected_cell_frontier_tiles(
        self,
        world: "WorldEngine",
        *,
        target_frontier: tuple[str, str, str, str],
        source_frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["accumulate_formal_connected_cell_frontier_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected cell frontier accumulation requires ComputeShader.run_indirect")
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        target_flags_name, target_list_name, target_count_name, target_dispatch_args_name = target_frontier
        source_flags_name, source_list_name, source_count_name, source_dispatch_args_name = source_frontier
        bridge.buffers[target_flags_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[target_list_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[target_count_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[target_dispatch_args_name].bind_to_storage_buffer(binding=3)
        bridge.buffers[source_count_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[source_list_name].bind_to_storage_buffer(binding=5)
        program.run_indirect(bridge.buffers[source_dispatch_args_name])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            target_flags_name,
            target_list_name,
            target_count_name,
            target_dispatch_args_name,
            source_flags_name,
            source_list_name,
            source_count_name,
            source_dispatch_args_name,
        )

    def _seed_formal_connected_cell_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        seed_rect: tuple[int, int, int, int],
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["seed_formal_connected_cell_frontier"]
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        current_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        self._clear_formal_connected_cell_frontier_tiles(world, current_frontier)
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected cell frontier seed requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        program["seed_rect"].value = tuple(int(value) for value in seed_rect)
        resources.structural_tex.use(location=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=6)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=7)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=8)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )

    def _seed_formal_connected_cell_frontier_from_dirty_queue(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_queue is None:
            raise RuntimeError("formal dirty cell frontier requires dirty tile queue buffers")
        dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        current_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        self._clear_formal_connected_cell_frontier_tiles(world, current_frontier)
        self._clear_formal_connected_cell_buffer_connected_tiles(
            world,
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            width,
            height,
            tile_mask_name,
        )
        program = self.programs["seed_formal_connected_cell_frontier_from_dirty_queue"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal dirty cell frontier requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_tile_origin"].value = (int(x0) // int(tile_size), int(y0) // int(tile_size))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        resources.structural_tex.use(location=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
        bridge.buffers[FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=6)
        dirty_count.bind_to_storage_buffer(binding=7)
        dirty_list.bind_to_storage_buffer(binding=8)
        program.run_indirect(dirty_dispatch_args)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )

    def _expand_formal_connected_cell_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        current_buffer_name: str,
        scratch_buffer_name: str,
        tile_mask_name: str,
        *,
        current_frontier: tuple[str, str, str, str],
        next_frontier: tuple[str, str, str, str],
        jump: int = 1,
        jump_generation: int = 1,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["expand_formal_connected_cells_by_tile"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU collapse formal connected cell frontier requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        program["jump"].value = int(max(1, jump))
        program["jump_generation"].value = int(max(1, jump_generation))
        _, current_list_name, current_count_name, current_dispatch_args_name = current_frontier
        next_flags_name, next_list_name, next_count_name, next_dispatch_args_name = next_frontier
        resources.structural_tex.use(location=0)
        bridge.buffers[current_buffer_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[scratch_buffer_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[current_count_name].bind_to_storage_buffer(binding=3)
        bridge.buffers[current_list_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[next_flags_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[next_count_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[next_list_name].bind_to_storage_buffer(binding=7)
        bridge.buffers[next_dispatch_args_name].bind_to_storage_buffer(binding=8)
        program.run_indirect(bridge.buffers[current_dispatch_args_name])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            current_buffer_name,
            scratch_buffer_name,
            tile_mask_name,
            current_list_name,
            current_count_name,
            current_dispatch_args_name,
            next_flags_name,
            next_list_name,
            next_count_name,
            next_dispatch_args_name,
        )

    def _copy_formal_connected_buffer_to_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        connected_buffer_name: str,
        target_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["copy_formal_connected_buffer_to_texture"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected buffer copy requires ComputeShader.run_indirect")
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        resources.structural_tex.use(location=0)
        bridge.buffers[connected_buffer_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        target_texture.bind_to_image(1, read=False, write=True)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)

    def _solve_formal_connected_tile_support_textures(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
        *,
        publish_masks: bool = True,
    ) -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        current = resources.support_ping
        scratch = resources.support_pong
        with self._profile_pass(world, "support_jfa.axis_masks"):
            self._build_formal_connected_axis_masks(
                world,
                resources,
                resources.structural_tex,
                width,
                height,
                tile_mask_name,
            )
        jumps = self._formal_jfa_jumps(width, height)
        with self._profile_pass(world, "support_jfa.jfa"):
            for band_name, band_jumps in self._formal_jfa_profile_jump_bands(jumps):
                with self._profile_pass(world, f"support_jfa.jfa.{band_name}"):
                    for jump in band_jumps:
                        current, scratch = self._run_formal_connected_tile_support_pass(
                            world,
                            resources,
                            current,
                            scratch,
                            width,
                            height,
                            tile_mask_name,
                            jump,
                        )
        with self._profile_pass(world, "support_jfa.refine"):
            current, scratch = self._run_formal_connected_tile_support_refine_passes(
                world,
                resources,
                current,
                scratch,
                width,
                height,
                tile_mask_name,
            )
        if publish_masks:
            with self._profile_pass(world, "support_jfa.publish"):
                self._publish_bridge_supported_unsupported_masks_connected_tiles(
                    world,
                    resources,
                    current,
                    x0,
                    y0,
                    width,
                    height,
                    tile_mask_name=tile_mask_name,
                )
        return current

    def _seed_formal_connected_tile_support_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        tile_mask_name: str,
        frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        self._clear_formal_connected_cell_frontier_tiles(world, frontier)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected support propagation requires tile_size <= 32")
        program = self.programs["seed_formal_connected_tile_support_frontier"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected support seed requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        flags_name, list_name, count_name, dispatch_args_name = frontier
        resources.structural_tex.use(location=0)
        resources.support_ping.use(location=1)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[flags_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[list_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[count_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=7)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            flags_name,
            list_name,
            count_name,
            dispatch_args_name,
        )

    def _expand_formal_connected_tile_support_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        tile_mask_name: str,
        *,
        current_frontier: tuple[str, str, str, str],
        next_frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected support propagation requires tile_size <= 32")
        program = self.programs["expand_formal_connected_tile_support_frontier"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected support propagation requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        current_flags_name, current_list_name, current_count_name, current_dispatch_args_name = current_frontier
        next_flags_name, next_list_name, next_count_name, next_dispatch_args_name = next_frontier
        resources.structural_tex.use(location=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[current_count_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[current_list_name].bind_to_storage_buffer(binding=3)
        bridge.buffers[next_flags_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[next_list_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[next_count_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[next_dispatch_args_name].bind_to_storage_buffer(binding=7)
        program.run_indirect(bridge.buffers[current_dispatch_args_name])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            tile_mask_name,
            current_flags_name,
            current_list_name,
            current_count_name,
            current_dispatch_args_name,
            next_flags_name,
            next_list_name,
            next_count_name,
            next_dispatch_args_name,
        )

    def _run_formal_connected_tile_support_pass(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        tile_mask_name: str,
        jump: int,
    ) -> tuple[Any, Any]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected support propagation requires tile_size <= 32")
        tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
        tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
        program = self.programs["propagate_formal_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected support propagation requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = int(tile_size)
        program["jump"].value = int(jump)
        resources.structural_tex.use(location=0)
        current.use(location=1)
        scratch.bind_to_image(2, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
        resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            tile_mask_name,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        return scratch, current

    def _run_formal_connected_tile_support_refine_passes(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> tuple[Any, Any]:
        for _ in range(self._formal_connected_tile_refine_pass_count(world)):
            current, scratch = self._run_formal_connected_tile_support_pass(
                world,
                resources,
                current,
                scratch,
                width,
                height,
                tile_mask_name,
                1,
            )
        return current, scratch

    def _filter_formal_connected_eligibility(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        eligibility_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
        processed_buffer_name: str,
        *,
        tile_mask_name: str | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if processed_buffer_name not in bridge.buffers:
            self._ensure_formal_connected_frontier_buffers(world)
        self._ensure_programs(ctx)
        if tile_mask_name is not None:
            self._clear_formal_connected_cell_buffer_connected_tiles(
                world,
                processed_buffer_name,
                width,
                height,
                tile_mask_name,
            )
        else:
            self._clear_formal_connected_cell_buffer_names(world, (processed_buffer_name,))
        connected_tiles = tile_mask_name is not None
        program = self.programs[
            "filter_formal_connected_eligibility_connected_tiles"
            if connected_tiles
            else "filter_formal_connected_eligibility"
        ]
        if connected_tiles and not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected eligibility filter requires ComputeShader.run_indirect")
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        if not connected_tiles:
            program["use_tile_mask"].value = False
        resources.structural_tex.use(location=0)
        resources.support_ping.use(location=1)
        eligibility_texture.use(location=2)
        resources.integrity_out_tex.bind_to_image(3, read=False, write=True)
        resources.support_pong.bind_to_image(4, read=False, write=True)
        bridge.buffers[processed_buffer_name].bind_to_storage_buffer(binding=0)
        if tile_mask_name is not None:
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            group_x = (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(processed_buffer_name)
        if tile_mask_name is not None:
            self._copy_mask_texture_connected_tiles(
                world,
                resources,
                resources.integrity_out_tex,
                resources.structural_tex,
                width,
                height,
                tile_mask_name,
            )
            self._copy_mask_texture_connected_tiles(
                world,
                resources,
                resources.support_pong,
                resources.support_ping,
                width,
                height,
                tile_mask_name,
            )
        else:
            self.copy_mask_texture(world, resources, resources.integrity_out_tex, resources.structural_tex, width, height)
            self.copy_mask_texture(world, resources, resources.support_pong, resources.support_ping, width, height)

    def connected_structural_frontier_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        width: int,
        height: int,
        x0: int,
        y0: int,
        frontier_buffer_name: str,
    ) -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return resources.cell_flags_out_tex
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if frontier_buffer_name not in bridge.buffers:
            raise RuntimeError("formal connected frontier buffer is not allocated")
        self._ensure_programs(ctx)
        program = self.programs["seed_structural_frontier_region"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        resources.structural_tex.use(location=0)
        bridge.buffers[frontier_buffer_name].bind_to_storage_buffer(binding=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        resources.support_pong.bind_to_image(2, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        connected_texture = self.solve_region_textures(
            world,
            resources,
            width,
            height,
            x0=0,
            y0=0,
            publish_masks=False,
        )
        self.copy_mask_texture(world, resources, connected_texture, resources.cell_flags_out_tex, width, height)
        return resources.cell_flags_out_tex

    def drain_formal_deferred_region_requests(self, world: "WorldEngine") -> list[tuple[int, int, int, int]]:
        """Legacy compatibility hook; formal connected expansion is GPU-frontier driven."""
        return []

    def enqueue_connected_internal_boundary_deferred_regions(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        eligibility_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
        solve_region: tuple[int, int, int, int],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return
        internal_edges = (
            1 if int(x0) > 0 else 0,
            1 if int(y0) > 0 else 0,
            1 if int(x0) + int(width) < int(world.width) else 0,
            1 if int(y0) + int(height) < int(world.height) else 0,
        )
        if not any(internal_edges):
            return

        self._ensure_programs(ctx)
        request_count, request_buffer, request_capacity = self._ensure_formal_deferred_region_request_buffers(world)
        resources.region_flags.write(np.zeros(1, dtype=np.uint32).tobytes())
        program = self.programs["enqueue_connected_internal_boundary_deferred_regions"]
        program["region_size"].value = (int(width), int(height))
        program["internal_edges"].value = internal_edges
        program["solve_rect"].value = tuple(int(value) for value in solve_region)
        program["world_size"].value = (int(world.width), int(world.height))
        program["request_capacity"].value = int(request_capacity)
        eligibility_texture.use(location=0)
        resources.region_flags.bind_to_storage_buffer(binding=0)
        request_count.bind_to_storage_buffer(binding=1)
        request_buffer.bind_to_storage_buffer(binding=2)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        world.bridge.mark_gpu_authoritative(
            FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER,
            FORMAL_DEFERRED_REGION_REQUEST_BUFFER,
        )

    def exclude_internal_boundary_connected_texture_to_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        eligibility_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
        frontier_buffer_name: str,
    ) -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return eligibility_texture
        internal_edges = (
            1 if int(x0) > 0 else 0,
            1 if int(y0) > 0 else 0,
            1 if int(x0) + int(width) < int(world.width) else 0,
            1 if int(y0) + int(height) < int(world.height) else 0,
        )
        if not any(internal_edges):
            return eligibility_texture

        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if frontier_buffer_name not in bridge.buffers:
            raise RuntimeError("formal connected frontier buffer is not allocated")
        self.copy_mask_texture(world, resources, eligibility_texture, resources.structural_tex, width, height)
        self._ensure_programs(ctx)
        seed_program = self.programs["seed_internal_boundary_region"]
        seed_program["region_size"].value = (int(width), int(height))
        seed_program["internal_edges"].value = internal_edges
        resources.structural_tex.use(location=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        resources.support_pong.bind_to_image(2, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        seed_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        boundary_connected_texture = self.solve_region_textures(
            world,
            resources,
            width,
            height,
            x0=0,
            y0=0,
            publish_masks=False,
        )
        publish_program = self.programs["publish_internal_boundary_frontier"]
        publish_program["region_size"].value = (int(width), int(height))
        publish_program["region_origin"].value = (int(x0), int(y0))
        publish_program["cell_grid_size"].value = (int(world.width), int(world.height))
        publish_program["internal_edges"].value = internal_edges
        boundary_connected_texture.use(location=0)
        bridge.buffers[frontier_buffer_name].bind_to_storage_buffer(binding=0)
        publish_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(frontier_buffer_name)

        exclude_program = self.programs["exclude_boundary_connected_mask"]
        exclude_program["region_size"].value = (int(width), int(height))
        resources.structural_tex.use(location=0)
        boundary_connected_texture.use(location=1)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        exclude_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        return resources.cell_flags_out_tex

    def exclude_internal_boundary_connected_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        eligibility_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
    ) -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return eligibility_texture
        internal_edges = (
            1 if int(x0) > 0 else 0,
            1 if int(y0) > 0 else 0,
            1 if int(x0) + int(width) < int(world.width) else 0,
            1 if int(y0) + int(height) < int(world.height) else 0,
        )
        if not any(internal_edges):
            return eligibility_texture

        self.copy_mask_texture(world, resources, eligibility_texture, resources.structural_tex, width, height)
        self._ensure_programs(ctx)
        seed_program = self.programs["seed_internal_boundary_region"]
        seed_program["region_size"].value = (int(width), int(height))
        seed_program["internal_edges"].value = internal_edges
        resources.structural_tex.use(location=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        resources.support_pong.bind_to_image(2, read=False, write=True)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        seed_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        boundary_connected_texture = self.solve_region_textures(
            world,
            resources,
            width,
            height,
            x0=0,
            y0=0,
            publish_masks=False,
        )
        exclude_program = self.programs["exclude_boundary_connected_mask"]
        exclude_program["region_size"].value = (int(width), int(height))
        resources.structural_tex.use(location=0)
        boundary_connected_texture.use(location=1)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        exclude_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        return resources.cell_flags_out_tex

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
        treat_region_boundary_as_support: bool = False,
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
        program["world_width"].value = int(world.width)
        program["world_height"].value = int(world.height)
        program["material_count"].value = int(behavior_params.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["treat_region_boundary_as_support"].value = bool(treat_region_boundary_as_support)
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

    def materialize_component_texture_formal(
        self,
        world: "WorldEngine",
        component_texture: Any,
        width: int,
        height: int,
        x0: int,
        y0: int,
        *,
        tile_mask_name: str | None = None,
    ) -> int:
        if not self._formal_gpu_frame(world):
            raise RuntimeError("formal component texture materialization requires an active formal GPU frame")
        if width == 0 or height == 0:
            return 0
        with self._profile_pass(world, "label_collect_components"):
            with self._profile_pass(world, "label_collect_components.label"):
                label_texture, width, height = self._label_component_texture(
                    world,
                    component_texture,
                    width,
                    height,
                    x0=x0,
                    y0=y0,
                    tile_mask_name=tile_mask_name,
                )
            component_capacity = self._prepare_formal_component_list_and_metadata(
                world,
                label_texture,
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name,
            )
        if component_capacity == 0:
            return 0
        island_id_base = self._reserve_formal_component_island_ids(world, component_capacity)
        with self._profile_pass(world, "materialize"):
            self._materialize_compact_labeled_component_texture(
                world,
                label_texture,
                island_id_base,
                component_capacity,
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name,
            )
        with self._profile_pass(world, "publish_runtime"):
            self._publish_compact_component_island_runtime(
                world,
                island_id_base,
                component_capacity,
                x0,
                y0,
                width,
                height,
            )
        return component_capacity

    def _reserve_formal_component_island_ids(self, world: "WorldEngine", component_capacity: int) -> int:
        component_capacity = max(0, int(component_capacity))
        if component_capacity == 0:
            return 0
        next_id = max(1, int(getattr(world, "next_island_id", 1)))
        max_existing = max((int(island_id) for island_id in getattr(world, "islands", {})), default=0)
        island_id_base = max(next_id, max_existing + 1)
        world.next_island_id = max(next_id, island_id_base + component_capacity)
        return island_id_base

    def _ensure_component_work_buffers(
        self,
        ctx: Any,
        resources: GPUCollapseResources,
        component_capacity: int,
    ) -> None:
        component_capacity = max(1, int(component_capacity))
        label_bytes = component_capacity * np.dtype(np.int32).itemsize
        flag_bytes = component_capacity * np.dtype(np.uint32).itemsize
        metadata_bytes = component_capacity * 5 * np.dtype(np.int32).itemsize
        if resources.component_labels.size < label_bytes:
            resources.component_labels.release()
            resources.component_labels = ctx.buffer(reserve=label_bytes, dynamic=True)
        else:
            resources.component_labels.orphan(label_bytes)
        if resources.component_flags.size < flag_bytes:
            resources.component_flags.release()
            resources.component_flags = ctx.buffer(reserve=flag_bytes, dynamic=True)
        if resources.component_metadata.size < metadata_bytes:
            resources.component_metadata.release()
            resources.component_metadata = ctx.buffer(reserve=metadata_bytes, dynamic=True)
        else:
            resources.component_metadata.orphan(metadata_bytes)

    def _collect_component_labels_gpu(
        self,
        world: "WorldEngine",
        label_texture: Any,
        width: int,
        height: int,
        *,
        empty_min: tuple[int, int] | None = None,
        tile_mask_name: str | None = None,
    ) -> int:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return 0
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        component_capacity = max(1, int(width) * int(height))
        self._ensure_component_work_buffers(ctx, resources, component_capacity)
        empty_min_value = empty_min if empty_min is not None else (int(width), int(height))
        resources.component_count.write(np.zeros(1, dtype=np.uint32).tobytes())

        if self._formal_gpu_frame(world) and tile_mask_name is not None:
            program = self.programs["collect_component_labels_connected_tiles"]
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal connected component collect requires ComputeShader.run_indirect")
            program["cell_grid_size"].value = (int(width), int(height))
            program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
            program["component_capacity"].value = int(component_capacity)
            program["empty_min"].value = (int(empty_min_value[0]), int(empty_min_value[1]))
            label_texture.use(location=0)
            resources.component_flags.bind_to_storage_buffer(binding=0)
            resources.component_labels.bind_to_storage_buffer(binding=1)
            resources.component_count.bind_to_storage_buffer(binding=2)
            resources.component_metadata.bind_to_storage_buffer(binding=3)
            world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
            program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
            return component_capacity

        program = self.programs["collect_component_labels"]
        program["region_size"].value = (int(width), int(height))
        program["empty_min"].value = (int(empty_min_value[0]), int(empty_min_value[1]))
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_labels.bind_to_storage_buffer(binding=1)
        resources.component_count.bind_to_storage_buffer(binding=2)
        resources.component_metadata.bind_to_storage_buffer(binding=3)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        return component_capacity

    def _clear_component_label_flags_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        label_texture: Any,
        width: int,
        height: int,
        component_capacity: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["clear_component_label_flags_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component flag clear requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        program["component_capacity"].value = int(component_capacity)
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _build_component_dispatch_args(
        self,
        world: "WorldEngine",
        component_capacity: int,
        *,
        invocations_per_group: int = 256,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        resources = self.resources
        if resources is None:
            raise RuntimeError("GPU collapse component dispatch requires allocated resources")
        program = self.programs["build_component_dispatch_args"]
        program["component_capacity"].value = int(component_capacity)
        program["invocations_per_group"].value = int(invocations_per_group)
        resources.component_count.bind_to_storage_buffer(binding=0)
        resources.component_dispatch_args.bind_to_storage_buffer(binding=1)
        program.run(1, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))

    def _prepare_formal_component_list_and_metadata(
        self,
        world: "WorldEngine",
        label_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
        *,
        tile_mask_name: str | None = None,
    ) -> int:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        with self._profile_pass(world, "label_collect_components.collect_roots"):
            component_capacity = self._collect_component_labels_gpu(
                world,
                label_texture,
                width,
                height,
                empty_min=(int(x0 + width), int(y0 + height)),
                tile_mask_name=tile_mask_name,
            )
        if component_capacity == 0:
            return 0
        # Keep the connected-tiles summary path explicit at the prepare stage so
        # formal connected materialization stays routed to
        # summarize_compact_components_connected_tiles instead of the legacy
        # full-grid summarize shader.
        with self._profile_pass(world, "label_collect_components.summarize_metadata"):
            self._summarize_formal_component_metadata(
                world,
                label_texture,
                x0,
                y0,
                width,
                height,
                component_capacity,
                tile_mask_name=tile_mask_name,
            )
        return component_capacity

    def _summarize_formal_component_metadata(
        self,
        world: "WorldEngine",
        label_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
        component_capacity: int,
        *,
        tile_mask_name: str | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        resources = self._ensure_resources(ctx, width, height)

        summarize_program = self.programs["summarize_compact_components"]
        summarize_program["region_size"].value = (int(width), int(height))
        summarize_program["region_origin"].value = (int(x0), int(y0))
        summarize_program["component_capacity"].value = int(component_capacity)
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_metadata.bind_to_storage_buffer(binding=1)
        if self._formal_gpu_frame(world) and tile_mask_name is not None:
            summarize_program = self.programs["summarize_compact_components_connected_tiles"]
            summarize_program["cell_grid_size"].value = (int(width), int(height))
            summarize_program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            summarize_program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
            summarize_program["region_origin"].value = (int(x0), int(y0))
            summarize_program["component_capacity"].value = int(component_capacity)
            label_texture.use(location=0)
            resources.component_flags.bind_to_storage_buffer(binding=0)
            resources.component_metadata.bind_to_storage_buffer(binding=1)
            world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
            summarize_program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
            summarize_program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _materialize_compact_labeled_component_texture(
        self,
        world: "WorldEngine",
        label_texture: Any,
        island_id_base: int,
        component_capacity: int,
        x0: int,
        y0: int,
        width: int,
        height: int,
        *,
        tile_mask_name: str | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0 or component_capacity == 0:
            return
        with self._profile_pass(world, "materialize.main"):
            self._ensure_programs(ctx)
            resources = self._ensure_resources(ctx, width, height)
            if not (self._formal_gpu_frame(world) and tile_mask_name is not None):
                self._upload_region_state(world, resources, x0, y0, width, height)
            collapse_generation, base_integrity, spawn_temperature = self._materialize_material_params(world)
            self._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
            self._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
            self._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

            connected_tiles = self._formal_gpu_frame(world) and tile_mask_name is not None
            program = self.programs[
                "materialize_compact_components_connected_tiles" if connected_tiles else "materialize_compact_components"
            ]
            if connected_tiles:
                program["cell_grid_size"].value = (int(width), int(height))
                program["tile_grid_size"].value = (
                    int(getattr(world.active, "tile_width", 1)),
                    int(getattr(world.active, "tile_height", 1)),
                )
                program["tile_size"].value = int(
                    max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
                )
            else:
                program["region_size"].value = (int(width), int(height))
            program["label_capacity"].value = int(component_capacity)
            program["island_id_base"].value = int(island_id_base)
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
            resources.material_collapse_generation.bind_to_storage_buffer(binding=0)
            resources.material_base_integrity.bind_to_storage_buffer(binding=1)
            resources.material_spawn_temperature.bind_to_storage_buffer(binding=2)
            resources.component_flags.bind_to_storage_buffer(binding=3)
            if connected_tiles:
                assert tile_mask_name is not None
                world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
                world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
                world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
                program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
            else:
                group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
                group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
                program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)

        with self._profile_pass(world, "materialize.aux"):
            aux_program = self.programs[
                "materialize_compact_components_aux_connected_tiles"
                if connected_tiles
                else "materialize_compact_components_aux"
            ]
            if connected_tiles:
                aux_program["cell_grid_size"].value = (int(width), int(height))
                aux_program["tile_grid_size"].value = (
                    int(getattr(world.active, "tile_width", 1)),
                    int(getattr(world.active, "tile_height", 1)),
                )
                aux_program["tile_size"].value = int(
                    max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
                )
            else:
                aux_program["region_size"].value = (int(width), int(height))
            aux_program["label_capacity"].value = int(component_capacity)
            aux_program["island_id_base"].value = int(island_id_base)
            label_texture.use(location=0)
            resources.island_id_tex.use(location=7)
            resources.entity_id_tex.use(location=8)
            resources.displaced_tex.use(location=9)
            resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
            resources.component_flags.bind_to_storage_buffer(binding=3)
            if connected_tiles:
                assert tile_mask_name is not None
                world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
                world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
                world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
                aux_program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
            else:
                aux_program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "materialize.publish_bridge_outputs"):
            if connected_tiles:
                assert tile_mask_name is not None
                self._publish_bridge_region_outputs_connected_tiles(world, resources, x0, y0, width, height, tile_mask_name)
            else:
                self._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
        self.last_cpu_mirror_downloaded = False

    def _publish_compact_component_island_runtime(
        self,
        world: "WorldEngine",
        island_id_base: int,
        component_capacity: int,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        if component_capacity == 0:
            return
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for island runtime")
        required_bytes = max(4, int(component_capacity) * ISLAND_RUNTIME_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["island_runtime"]
        preserve_existing_runtime = "island_runtime" in bridge.gpu_authoritative_resources
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_runtime"] = bridge_buffer
            preserve_existing_runtime = False
        elif not preserve_existing_runtime:
            bridge_buffer.orphan(required_bytes)
        if not preserve_existing_runtime:
            bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())

        resources = self._ensure_resources(ctx, width, height)
        self._build_component_dispatch_args(world, component_capacity)
        program = self.programs["publish_compact_component_island_runtime"]
        program["component_capacity"].value = int(component_capacity)
        program["island_id_base"].value = int(island_id_base)
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
        program["paging_buffer_origin"].value = (
            int(world.paging.buffer_origin_x),
            int(world.paging.buffer_origin_y),
        )
        resources.component_metadata.bind_to_storage_buffer(binding=0)
        bridge_buffer.bind_to_storage_buffer(binding=1)
        bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
        resources.component_count.bind_to_storage_buffer(binding=3)
        program.run_indirect(resources.component_dispatch_args)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative("island_runtime")

    def _materialize_dense_labeled_component_texture(
        self,
        world: "WorldEngine",
        label_texture: Any,
        island_id_base: int,
        component_capacity: int,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0 or component_capacity == 0:
            return
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        self._upload_region_state(world, resources, x0, y0, width, height)
        collapse_generation, base_integrity, spawn_temperature = self._materialize_material_params(world)
        self._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
        self._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
        self._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

        program = self.programs["materialize_dense_components"]
        program["region_size"].value = (width, height)
        program["label_capacity"].value = int(component_capacity)
        program["island_id_base"].value = int(island_id_base)
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
        resources.material_collapse_generation.bind_to_storage_buffer(binding=0)
        resources.material_base_integrity.bind_to_storage_buffer(binding=1)
        resources.material_spawn_temperature.bind_to_storage_buffer(binding=2)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        aux_program = self.programs["materialize_dense_components_aux"]
        aux_program["region_size"].value = (width, height)
        aux_program["label_capacity"].value = int(component_capacity)
        aux_program["island_id_base"].value = int(island_id_base)
        label_texture.use(location=0)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        aux_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
        self.last_cpu_mirror_downloaded = False

    def _summarize_dense_component_metadata(
        self,
        world: "WorldEngine",
        label_texture: Any,
        component_capacity: int,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        init_program = self.programs["init_dense_component_metadata"]
        init_program["component_capacity"].value = int(component_capacity)
        init_program["empty_min"].value = (int(x0 + width), int(y0 + height))
        resources.component_metadata.bind_to_storage_buffer(binding=0)
        init_program.run((int(component_capacity) + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)

        summarize_program = self.programs["summarize_dense_components"]
        summarize_program["region_size"].value = (width, height)
        summarize_program["region_origin"].value = (int(x0), int(y0))
        summarize_program["component_capacity"].value = int(component_capacity)
        label_texture.use(location=0)
        resources.component_metadata.bind_to_storage_buffer(binding=0)
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        summarize_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _publish_dense_component_island_runtime(
        self,
        world: "WorldEngine",
        label_texture: Any,
        island_id_base: int,
        component_capacity: int,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> None:
        if component_capacity == 0:
            return
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._summarize_dense_component_metadata(world, label_texture, component_capacity, x0, y0, width, height)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for island runtime")
        required_bytes = max(4, int(component_capacity) * ISLAND_RUNTIME_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["island_runtime"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_runtime"] = bridge_buffer
        else:
            bridge_buffer.orphan(required_bytes)
        bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())

        resources = self._ensure_resources(ctx, width, height)
        program = self.programs["publish_dense_component_island_runtime"]
        program["component_capacity"].value = int(component_capacity)
        program["island_id_base"].value = int(island_id_base)
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
        program["paging_buffer_origin"].value = (
            int(world.paging.buffer_origin_x),
            int(world.paging.buffer_origin_y),
        )
        resources.component_metadata.bind_to_storage_buffer(binding=0)
        bridge_buffer.bind_to_storage_buffer(binding=1)
        bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
        program.run((int(component_capacity) + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative("island_runtime")

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
        tile_mask_name: str | None = None,
    ) -> tuple[Any, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            return None, width, height
        self._ensure_programs(ctx)
        resources = self._ensure_resources(ctx, width, height)
        if self._formal_gpu_frame(world) and tile_mask_name is not None:
            return self._label_component_texture_connected_tiles(
                world,
                resources,
                component_texture,
                width,
                height,
                x0=x0,
                y0=y0,
                tile_mask_name=tile_mask_name,
            )
        if self._formal_gpu_frame(world):
            region_tile_mask_name = self._seed_formal_texture_region_tile_worklist(world, width, height)
            if region_tile_mask_name is not None:
                return self._label_component_texture_connected_tiles_from_texture_init(
                    world,
                    resources,
                    component_texture,
                    width,
                    height,
                    x0=x0,
                    y0=y0,
                    tile_mask_name=region_tile_mask_name,
                )
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
            jumps = self._formal_jfa_jumps(width, height)
            for jump in jumps:
                propagate["jump"].value = int(jump)
                resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                self._sync_compute_writes(ctx)
                current, scratch = scratch, current
            current, scratch = self._run_formal_label_refine_passes(
                ctx,
                resources,
                current,
                scratch,
                width,
                height,
                jumps,
                group_x,
                group_y,
            )
            self._publish_bridge_region_labels(world, resources, current, x0, y0, width, height)
        else:
            while True:
                propagate["jump"].value = 1
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

    def _label_component_texture_connected_tiles_from_texture_init(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        width: int,
        height: int,
        *,
        x0: int,
        y0: int,
        tile_mask_name: str,
    ) -> tuple[Any, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "label_jfa.texture_init"):
            init_program = self.programs["component_label_init"]
            init_program["region_size"].value = (width, height)
            component_texture.use(location=0)
            resources.support_ping.bind_to_image(1, read=False, write=True)
            init_program.run(group_x, group_y, 1)
            ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

        current = resources.support_ping
        scratch = resources.support_pong
        with self._profile_pass(world, "label_jfa.axis_masks"):
            self._build_formal_connected_axis_masks(
                world,
                resources,
                component_texture,
                width,
                height,
                tile_mask_name,
            )
        jumps = self._formal_jfa_jumps(width, height)
        with self._profile_pass(world, "label_jfa.jfa"):
            for band_name, band_jumps in self._formal_jfa_profile_jump_bands(jumps):
                with self._profile_pass(world, f"label_jfa.jfa.{band_name}"):
                    for jump in band_jumps:
                        current, scratch = self._run_formal_connected_component_label_pass(
                            world,
                            resources,
                            component_texture,
                            current,
                            scratch,
                            width,
                            height,
                            tile_mask_name,
                            jump,
                            refine_local_labels=True,
                        )
        with self._profile_pass(world, "label_jfa.refine"):
            current, scratch = self._run_formal_connected_component_label_refine_passes(
                world,
                resources,
                component_texture,
                current,
                scratch,
                width,
                height,
                tile_mask_name,
            )
        with self._profile_pass(world, "label_jfa.publish"):
            self._publish_bridge_region_labels_connected_tiles(
                world,
                resources,
                current,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        return current, width, height

    def _label_component_texture_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        width: int,
        height: int,
        *,
        x0: int,
        y0: int,
        tile_mask_name: str,
    ) -> tuple[Any, int, int]:
        seed_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        with self._profile_pass(world, "label_jfa.seed"):
            self._seed_formal_component_label_frontier(
                world,
                resources,
                component_texture,
                width,
                height,
                tile_mask_name,
                seed_frontier,
            )
        with self._profile_pass(world, "label_jfa.materialize"):
            self._copy_formal_component_label_buffer_to_texture(
                world,
                resources,
                component_texture,
                resources.support_ping,
                width,
                height,
                tile_mask_name,
            )
        current = resources.support_ping
        scratch = resources.support_pong
        with self._profile_pass(world, "label_jfa.axis_masks"):
            self._build_formal_connected_axis_masks(
                world,
                resources,
                component_texture,
                width,
                height,
                tile_mask_name,
            )
        jumps = self._formal_jfa_jumps(width, height)
        with self._profile_pass(world, "label_jfa.jfa"):
            for band_name, band_jumps in self._formal_jfa_profile_jump_bands(jumps):
                with self._profile_pass(world, f"label_jfa.jfa.{band_name}"):
                    for jump in band_jumps:
                        current, scratch = self._run_formal_connected_component_label_pass(
                            world,
                            resources,
                            component_texture,
                            current,
                            scratch,
                            width,
                            height,
                            tile_mask_name,
                            jump,
                            refine_local_labels=True,
                        )
        with self._profile_pass(world, "label_jfa.refine"):
            current, scratch = self._run_formal_connected_component_label_refine_passes(
                world,
                resources,
                component_texture,
                current,
                scratch,
                width,
                height,
                tile_mask_name,
            )
        with self._profile_pass(world, "label_jfa.publish"):
            self._publish_bridge_region_labels_connected_tiles(
                world,
                resources,
                current,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        return current, width, height

    def _seed_formal_component_label_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        width: int,
        height: int,
        tile_mask_name: str,
        frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        self._clear_formal_connected_cell_frontier_tiles(world, frontier)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected component labeling requires tile_size <= 32")
        program = self.programs["seed_formal_component_label_frontier"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component label seed requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        component_texture.use(location=0)
        flags_name, list_name, count_name, dispatch_args_name = frontier
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[flags_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[list_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[count_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=7)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            flags_name,
            list_name,
            count_name,
            dispatch_args_name,
        )

    def _expand_formal_component_label_frontier(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        width: int,
        height: int,
        tile_mask_name: str,
        *,
        current_frontier: tuple[str, str, str, str],
        next_frontier: tuple[str, str, str, str],
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected component labeling requires tile_size <= 32")
        program = self.programs["expand_formal_component_label_frontier"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component label propagation requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        current_flags_name, current_list_name, current_count_name, current_dispatch_args_name = current_frontier
        next_flags_name, next_list_name, next_count_name, next_dispatch_args_name = next_frontier
        component_texture.use(location=0)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[current_count_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[current_list_name].bind_to_storage_buffer(binding=3)
        bridge.buffers[next_flags_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[next_list_name].bind_to_storage_buffer(binding=5)
        bridge.buffers[next_count_name].bind_to_storage_buffer(binding=6)
        bridge.buffers[next_dispatch_args_name].bind_to_storage_buffer(binding=7)
        program.run_indirect(bridge.buffers[current_dispatch_args_name])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            FORMAL_CONNECTED_FRONTIER_BUFFER,
            tile_mask_name,
            current_flags_name,
            current_list_name,
            current_count_name,
            current_dispatch_args_name,
            next_flags_name,
            next_list_name,
            next_count_name,
            next_dispatch_args_name,
        )

    def _run_formal_connected_component_label_pass(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        tile_mask_name: str,
        jump: int,
        *,
        refine_local_labels: bool,
    ) -> tuple[Any, Any]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected component labeling requires tile_size <= 32")
        program = self.programs["propagate_formal_connected_component_labels"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component label propagation requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        program["jump"].value = int(jump)
        program["refine_local_labels"].value = bool(refine_local_labels)
        component_texture.use(location=0)
        current.use(location=1)
        scratch.bind_to_image(2, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
        resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            tile_mask_name,
            FORMAL_CONNECTED_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        )
        return scratch, current

    def _run_formal_connected_component_label_refine_passes(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> tuple[Any, Any]:
        for _ in range(self._formal_connected_tile_refine_pass_count(world)):
            current, scratch = self._run_formal_connected_component_label_pass(
                world,
                resources,
                component_texture,
                current,
                scratch,
                width,
                height,
                tile_mask_name,
                1,
                refine_local_labels=True,
            )
        return current, scratch

    def _copy_formal_component_label_buffer_to_texture(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        component_texture: Any,
        target_texture: Any,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        program = self.programs["copy_formal_component_label_buffer_to_texture"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component label copy requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        component_texture.use(location=0)
        target_texture.bind_to_image(1, read=False, write=True)
        bridge.buffers[FORMAL_CONNECTED_FRONTIER_BUFFER].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(ctx)

    def _run_formal_label_refine_passes(
        self,
        ctx: Any,
        resources: GPUCollapseResources,
        current: Any,
        scratch: Any,
        width: int,
        height: int,
        jumps: tuple[int, ...],
        group_x: int,
        group_y: int,
    ) -> tuple[Any, Any]:
        propagate = self.programs["component_label_propagate"]
        unit_pass_count = self._formal_label_unit_pass_count(width, height)
        refine_round_count = self._formal_label_refine_round_count(width, height)
        for round_index in range(refine_round_count):
            for _ in range(unit_pass_count):
                propagate["jump"].value = 1
                resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                self._sync_compute_writes(ctx)
                current, scratch = scratch, current
            if round_index + 1 >= refine_round_count:
                continue
            for jump in jumps:
                propagate["jump"].value = int(jump)
                resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                self._sync_compute_writes(ctx)
                current, scratch = scratch, current
        return current, scratch

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
        self._ensure_component_work_buffers(ctx, resources, cell_count)
        resources.component_count.write(np.zeros(1, dtype=np.uint32).tobytes())

        program = self.programs["collect_component_labels"]
        program["region_size"].value = (int(width), int(height))
        program["empty_min"].value = (int(width), int(height))
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_labels.bind_to_storage_buffer(binding=1)
        resources.component_count.bind_to_storage_buffer(binding=2)
        resources.component_metadata.bind_to_storage_buffer(binding=3)
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
        *,
        eligibility_texture: Any | None = None,
        tile_mask_name: str | None = None,
    ) -> tuple[GPUCollapseResources, int, int]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        if width == 0 or height == 0:
            raise ValueError("resolve_supported_outcome_textures requires a non-empty region")
        connected_tiles = self._formal_gpu_frame(world) and tile_mask_name is not None
        if connected_tiles and "collapse_delay_pending" in world.bridge.gpu_authoritative_resources:
            assert tile_mask_name is not None
            self._load_authoritative_bridge_connected_tile_pending(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
            )
        else:
            pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
            resources.phase_tex.write(pending_region.astype("f4", copy=False).tobytes())
            self._load_authoritative_bridge_pending_region(world, resources, x0, y0, width, height)

        program = self.programs[
            "resolve_outcomes_from_supported_connected_tiles" if connected_tiles else "resolve_outcomes_from_supported"
        ]
        if connected_tiles and not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected outcome resolve requires ComputeShader.run_indirect")
        program["region_size"].value = (width, height)
        program["behavior_falling_island"].value = int(CollapseBehavior.FALLING_ISLAND)
        program["behavior_delayed"].value = int(CollapseBehavior.DELAYED)
        program["behavior_immune"].value = int(CollapseBehavior.IMMUNE)
        program["use_eligibility"].value = eligibility_texture is not None
        if connected_tiles:
            program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
        resources.structural_tex.use(location=0)
        supported_texture.use(location=1)
        resources.material_out_tex.use(location=2)
        resources.phase_tex.use(location=3)
        (eligibility_texture if eligibility_texture is not None else resources.structural_tex).use(location=7)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
        resources.phase_out_tex.bind_to_image(6, read=False, write=True)
        if connected_tiles:
            assert tile_mask_name is not None
            world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
            program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world):
            self._publish_bridge_pending_region_outputs_from_texture(
                world,
                resources,
                resources.temp_out_tex,
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name if connected_tiles else None,
            )
            self._publish_bridge_region_mask(
                world,
                resources,
                resources.temp_out_tex,
                "collapse_delayed_pending_mask",
                x0,
                y0,
                width,
                height,
                tile_mask_name=tile_mask_name if connected_tiles else None,
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
                tile_mask_name=tile_mask_name if connected_tiles else None,
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
                tile_mask_name=tile_mask_name if connected_tiles else None,
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
            self.resources.component_dispatch_args,
            self.resources.region_flags,
            self.resources.connected_tile_row_masks,
            self.resources.connected_tile_column_masks,
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
        axis_tile_width = max(1, (int(width) + FORMAL_CONNECTED_TILE_LOCAL_SIZE - 1) // FORMAL_CONNECTED_TILE_LOCAL_SIZE)
        axis_tile_height = max(1, (int(height) + FORMAL_CONNECTED_TILE_LOCAL_SIZE - 1) // FORMAL_CONNECTED_TILE_LOCAL_SIZE)
        axis_mask_bytes = max(
            4,
            axis_tile_width
            * axis_tile_height
            * FORMAL_CONNECTED_TILE_LOCAL_SIZE
            * np.dtype(np.uint32).itemsize,
        )
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
            component_dispatch_args=ctx.buffer(reserve=3 * 4, dynamic=True),
            region_flags=ctx.buffer(reserve=4, dynamic=True),
            connected_tile_row_masks=ctx.buffer(reserve=axis_mask_bytes, dynamic=True),
            connected_tile_column_masks=ctx.buffer(reserve=axis_mask_bytes, dynamic=True),
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
            uniform int world_width;
            uniform int world_height;
            uniform int material_count;
            uniform int phase_falling_island;
            uniform bool treat_region_boundary_as_support;

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
                bool external_region_boundary =
                    treat_region_boundary_as_support
                    && (
                        (cell.x == 0 && region_origin.x > 0)
                        || (cell.x == region_size.x - 1 && region_origin.x + cell.x + 1 < world_width)
                        || (cell.y == 0 && region_origin.y > 0)
                        || (cell.y == region_size.y - 1 && region_origin.y + cell.y + 1 < world_height)
                    );
                bool support_seed = structural && (
                    material_support_anchor[index] != 0
                    || region_origin.y + cell.y == world_height - 1
                    || external_region_boundary
                );
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
        self.programs["copy_mask_texture"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;

            layout(binding=0) uniform sampler2D source_tex;
            layout(r32f, binding=1) writeonly uniform image2D target_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                float value = texelFetch(source_tex, cell, 0).x > 0.5 ? 1.0 : 0.0;
                imageStore(target_img, cell, vec4(value, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["copy_mask_texture_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D source_tex;
            layout(r32f, binding=1) writeonly uniform image2D target_img;
            layout(std430, binding=0) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                float value = texelFetch(source_tex, cell, 0).x > 0.5 ? 1.0 : 0.0;
                imageStore(target_img, cell, vec4(value, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["filter_formal_connected_eligibility"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_tile_mask;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D support_seed_tex;
            layout(binding=2) uniform sampler2D eligibility_tex;
            layout(r32f, binding=3) writeonly uniform image2D filtered_structural_img;
            layout(r32f, binding=4) writeonly uniform image2D filtered_support_seed_img;
            layout(std430, binding=0) buffer ProcessedEligibility {{
                int processed[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
	                ivec2 tile = ivec2(0);
	                bool tile_allowed = true;
	                if (use_tile_mask) {{
	                    int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
	                    ivec2 group = ivec2(gl_WorkGroupID.xy);
	                    tile = ivec2(group.x / groups_per_tile_axis, group.y / groups_per_tile_axis);
	                    if (tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                        return;
	                    }}
	                    ivec2 subtile = ivec2(group.x % groups_per_tile_axis, group.y % groups_per_tile_axis);
	                    cell = tile * tile_size + subtile * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
	                    int tile_index = tile.y * tile_grid_size.x + tile.x;
	                    tile_allowed = connected_tiles[tile_index] != 0;
	                }}
	                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
	                    return;
	                }}
	                ivec2 world_cell = region_origin + cell;
	                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
	                bool structural = tile_allowed && texelFetch(structural_tex, cell, 0).x > 0.5;
	                bool connected = structural && texelFetch(eligibility_tex, cell, 0).x > 0.5;
	                bool eligible = connected && processed[cell_index] == 0;
	                bool support_seed = eligible && texelFetch(support_seed_tex, cell, 0).x > 0.5;
                imageStore(filtered_structural_img, cell, vec4(eligible ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(filtered_support_seed_img, cell, vec4(support_seed ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                if (connected) {{
                    processed[cell_index] = 1;
                }}
            }}
            """
        )
        self.programs["filter_formal_connected_eligibility_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D support_seed_tex;
            layout(binding=2) uniform sampler2D eligibility_tex;
            layout(r32f, binding=3) writeonly uniform image2D filtered_structural_img;
            layout(r32f, binding=4) writeonly uniform image2D filtered_support_seed_img;
            layout(std430, binding=0) buffer ProcessedEligibility {{
                int processed[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    cell.x >= region_size.x ||
                    cell.y >= region_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= cell_grid_size.x || world_cell.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = cell.y * region_size.x + cell.x;
                bool structural = texelFetch(structural_tex, cell, 0).x > 0.5;
                bool connected = structural && texelFetch(eligibility_tex, cell, 0).x > 0.5;
                bool eligible = connected && processed[cell_index] == 0;
                bool support_seed = eligible && texelFetch(support_seed_tex, cell, 0).x > 0.5;
                imageStore(filtered_structural_img, cell, vec4(eligible ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(filtered_support_seed_img, cell, vec4(support_seed ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                if (connected) {{
                    processed[cell_index] = 1;
                }}
            }}
            """
        )
        self.programs["seed_formal_connected_frontier_rect"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec4 seed_rect;

            layout(std430, binding=0) buffer ConnectedFrontier {{
                int frontier[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                bool in_seed =
                    cell.x >= seed_rect.x &&
                    cell.y >= seed_rect.y &&
                    cell.x < seed_rect.z &&
                    cell.y < seed_rect.w;
                frontier[cell_index] = in_seed ? 1 : 0;
            }}
            """
        )
        self.programs["seed_formal_connected_tile_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform ivec4 seed_rect;
            uniform ivec2 seed_tile_origin;
            uniform int material_count;
            uniform int phase_falling_island;

            layout(std430, binding=0) buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=2) readonly buffer MaterialStructural {{
                int material_structural[];
            }};
            layout(std430, binding=3) buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) buffer ConnectedTileDispatchArgs {{
                uint connected_tile_dispatch_args[];
            }};
            layout(std430, binding=5) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};
            layout(std430, binding=8) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};

            shared uint s_seeded;

            bool cell_structural(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return false;
                }}
                ivec2 world_cell = region_origin + cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return false;
                }}
                int word_index = (world_cell.y * world_grid_size.x + world_cell.x) * 5;
                uint word0 = bridge_cell_core[word_index];
                int material_id = int(word0 & 0xFFFFu);
                if (material_id <= 0 || material_id >= material_count) {{
                    return false;
                }}
                int phase = int((word0 >> 16u) & 0xFFu);
                return phase != phase_falling_island && material_structural[material_id] != 0;
            }}

            void main() {{
                ivec2 tile = seed_tile_origin + ivec2(gl_WorkGroupID.xy);
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (gl_LocalInvocationIndex == 0u) {{
                    s_seeded = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool in_bounds =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    cell.x < cell_grid_size.x &&
                    cell.y < cell_grid_size.y;
                bool in_seed =
                    in_bounds &&
                    cell.x >= seed_rect.x &&
                    cell.y >= seed_rect.y &&
                    cell.x < seed_rect.z &&
                    cell.y < seed_rect.w;
                if (in_seed && cell_structural(cell)) {{
                    atomicOr(s_seeded, 1u);
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_seeded != 0u) {{
                    if (atomicCompSwap(connected_tiles[tile_index], 0, 1) == 0) {{
                        uint connected_slot = atomicAdd(connected_tile_count[0], 1u);
                        connected_tile_list[int(connected_slot)] = tile;
                        atomicMax(connected_tile_dispatch_args[0], connected_slot + 1u);
                        uint frontier_slot = atomicAdd(frontier_tile_count[0], 1u);
                        frontier_tile_list[int(frontier_slot)] = tile;
                        atomicMax(frontier_tile_dispatch_args[0], frontier_slot + 1u);
                    }}
                }}
            }}
            """
        )
        self.programs["seed_formal_connected_tile_frontier_from_dirty_queue"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 region_tile_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int material_count;
            uniform int phase_falling_island;

            layout(std430, binding=0) buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=2) readonly buffer MaterialStructural {{
                int material_structural[];
            }};
            layout(std430, binding=3) buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) buffer ConnectedTileDispatchArgs {{
                uint connected_tile_dispatch_args[];
            }};
            layout(std430, binding=5) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};
            layout(std430, binding=8) readonly buffer DirtyTileCount {{
                uint dirty_tile_count[];
            }};
            layout(std430, binding=9) readonly buffer DirtyTileList {{
                ivec2 dirty_tile_list[];
            }};
            layout(std430, binding=10) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};

            shared uint s_seeded;

            bool cell_structural(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return false;
                }}
                ivec2 world_cell = region_origin + cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return false;
                }}
                int word_index = (world_cell.y * world_grid_size.x + world_cell.x) * 5;
                uint word0 = bridge_cell_core[word_index];
                int material_id = int(word0 & 0xFFFFu);
                if (material_id <= 0 || material_id >= material_count) {{
                    return false;
                }}
                int phase = int((word0 >> 16u) & 0xFFu);
                return phase != phase_falling_island && material_structural[material_id] != 0;
            }}

            void main() {{
                uint dirty_index = gl_WorkGroupID.x;
                if (dirty_index >= dirty_tile_count[0]) {{
                    return;
                }}
                ivec2 dirty_tile = dirty_tile_list[int(dirty_index)];
                if (
                    dirty_tile.x < 0 ||
                    dirty_tile.y < 0 ||
                    dirty_tile.x >= tile_grid_size.x ||
                    dirty_tile.y >= tile_grid_size.y
                ) {{
                    return;
                }}
                ivec2 tile = dirty_tile - region_tile_origin;
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_seeded = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool in_bounds =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    cell.x < cell_grid_size.x &&
                    cell.y < cell_grid_size.y;
                if (in_bounds && cell_structural(cell)) {{
                    atomicOr(s_seeded, 1u);
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_seeded != 0u) {{
                    int tile_index = tile.y * tile_grid_size.x + tile.x;
                    if (atomicCompSwap(connected_tiles[tile_index], 0, 1) == 0) {{
                        uint connected_slot = atomicAdd(connected_tile_count[0], 1u);
                        connected_tile_list[int(connected_slot)] = tile;
                        atomicMax(connected_tile_dispatch_args[0], connected_slot + 1u);
                        uint frontier_slot = atomicAdd(frontier_tile_count[0], 1u);
                        frontier_tile_list[int(frontier_slot)] = tile;
                        atomicMax(frontier_tile_dispatch_args[0], frontier_slot + 1u);
                    }}
                }}
            }}
            """
        )
        self.programs["expand_formal_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int jump;
            uniform int material_count;
            uniform int phase_falling_island;

            layout(std430, binding=0) buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=2) readonly buffer MaterialStructural {{
                int material_structural[];
            }};
            layout(std430, binding=3) buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) buffer ConnectedTileDispatchArgs {{
                uint connected_tile_dispatch_args[];
            }};
            layout(std430, binding=5) readonly buffer CurrentFrontierTileCount {{
                uint current_frontier_tile_count[];
            }};
            layout(std430, binding=6) readonly buffer CurrentFrontierTileList {{
                ivec2 current_frontier_tile_list[];
            }};
            layout(std430, binding=7) buffer NextFrontierTileCount {{
                uint next_frontier_tile_count[];
            }};
            layout(std430, binding=8) buffer NextFrontierTileList {{
                ivec2 next_frontier_tile_list[];
            }};
            layout(std430, binding=9) buffer NextFrontierDispatchArgs {{
                uint next_frontier_dispatch_args[];
            }};
            layout(std430, binding=10) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool cell_structural(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return false;
                }}
                ivec2 world_cell = region_origin + cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return false;
                }}
                int word_index = (world_cell.y * world_grid_size.x + world_cell.x) * 5;
                uint word0 = bridge_cell_core[word_index];
                int material_id = int(word0 & 0xFFFFu);
                if (material_id <= 0 || material_id >= material_count) {{
                    return false;
                }}
                int phase = int((word0 >> 16u) & 0xFFu);
                return phase != phase_falling_island && material_structural[material_id] != 0;
            }}

            bool structural_crosses_edge(ivec2 connected_tile, ivec2 candidate_tile, ivec2 direction) {{
                ivec2 connected_origin = connected_tile * tile_size;
                ivec2 candidate_origin = candidate_tile * tile_size;
                for (int offset = 0; offset < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++offset) {{
                    if (offset >= tile_size) {{
                        break;
                    }}
                    ivec2 connected_local = direction.x < 0
                        ? ivec2(0, offset)
                        : direction.x > 0
                            ? ivec2(tile_size - 1, offset)
                            : direction.y < 0
                                ? ivec2(offset, 0)
                                : ivec2(offset, tile_size - 1);
                    ivec2 candidate_local = direction.x < 0
                        ? ivec2(tile_size - 1, offset)
                        : direction.x > 0
                            ? ivec2(0, offset)
                            : direction.y < 0
                                ? ivec2(offset, tile_size - 1)
                                : ivec2(offset, 0);
                    if (cell_structural(connected_origin + connected_local) && cell_structural(candidate_origin + candidate_local)) {{
                        return true;
                    }}
                }}
                return false;
            }}

            bool structural_crosses_tile_path(ivec2 connected_tile, ivec2 direction, int distance) {{
                if (distance <= 0) {{
                    return false;
                }}
                for (int step = 0; step < distance; ++step) {{
                    ivec2 edge_tile = connected_tile + direction * step;
                    ivec2 next_tile = edge_tile + direction;
                    if (!tile_in_grid(edge_tile) || !tile_in_grid(next_tile)) {{
                        return false;
                    }}
                    if (!structural_crosses_edge(edge_tile, next_tile, direction)) {{
                        return false;
                    }}
                }}
                return true;
            }}

            void append_connected_tile(ivec2 tile) {{
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(connected_tiles[tile_index], 0, 1) != 0) {{
                    return;
                }}
                uint connected_slot = atomicAdd(connected_tile_count[0], 1u);
                connected_tile_list[int(connected_slot)] = tile;
                atomicMax(connected_tile_dispatch_args[0], connected_slot + 1u);
                uint frontier_slot = atomicAdd(next_frontier_tile_count[0], 1u);
                next_frontier_tile_list[int(frontier_slot)] = tile;
                atomicMax(next_frontier_dispatch_args[0], frontier_slot + 1u);
            }}

            void try_neighbor_jump(ivec2 connected_tile, ivec2 direction) {{
                int max_distance = max(1, jump);
                for (int distance = 1; distance <= max_distance; ++distance) {{
                    ivec2 candidate = connected_tile + direction * distance;
                    if (!tile_in_grid(candidate)) {{
                        break;
                    }}
                    int candidate_index = candidate.y * tile_grid_size.x + candidate.x;
                    if (connected_tiles[candidate_index] != 0) {{
                        continue;
                    }}
                    if (structural_crosses_tile_path(connected_tile, direction, distance)) {{
                        append_connected_tile(candidate);
                    }}
                }}
            }}

            void main() {{
                uint frontier_index = gl_WorkGroupID.x;
                if (frontier_index >= current_frontier_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = current_frontier_tile_list[int(frontier_index)];
                if (!tile_in_grid(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                try_neighbor_jump(tile, ivec2(-1, 0));
                try_neighbor_jump(tile, ivec2(1, 0));
                try_neighbor_jump(tile, ivec2(0, -1));
                try_neighbor_jump(tile, ivec2(0, 1));
            }}
            """
        )
        self.programs["classify_formal_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
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
            layout(std430, binding=3) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            int safe_material_index(int material_id) {{
                if (material_id < 0 || material_id >= material_count) {{
                    return 0;
                }}
                return material_id;
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                bool tile_connected = connected_tiles[tile_index] != 0;
                int material_id = tile_connected ? int(texelFetch(material_tex, cell, 0).x + 0.5) : 0;
                int index = safe_material_index(material_id);
                int phase = tile_connected ? int(texelFetch(phase_tex, cell, 0).x + 0.5) : 0;
                bool structural = material_id > 0 && phase != phase_falling_island && material_structural[index] != 0;
                ivec2 world_cell = region_origin + cell;
                bool world_floor = world_cell.y == world_grid_size.y - 1;
                bool support_seed = structural && (material_support_anchor[index] != 0 || world_floor);
                int behavior = material_id > 0 ? material_collapse_behavior[index] : 0;
                imageStore(structural_img, cell, vec4(structural ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(support_seed_img, cell, vec4(support_seed ? 1.0 : 0.0, 0.0, 0.0, 0.0));
                imageStore(behavior_img, cell, vec4(float(behavior), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["clear_formal_connected_tile_worklist"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;

            layout(std430, binding=0) buffer ConnectedTileCount {
                uint connected_tile_count[];
            };
            layout(std430, binding=1) buffer ConnectedTileDispatchArgs {
                uint connected_tile_dispatch_args[];
            };

            void main() {
                if (gl_GlobalInvocationID.x != 0u) {
                    return;
                }
                connected_tile_count[0] = 0u;
                connected_tile_dispatch_args[0] = 0u;
                connected_tile_dispatch_args[1] = 1u;
                connected_tile_dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["compact_formal_connected_tile_mask"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform int tile_count;

            layout(std430, binding=0) readonly buffer ConnectedTileMask {
                int connected_tiles[];
            };
            layout(std430, binding=1) buffer ConnectedTileList {
                ivec2 connected_tile_list[];
            };
            layout(std430, binding=2) buffer ConnectedTileCount {
                uint connected_tile_count[];
            };
            layout(std430, binding=3) buffer ConnectedTileDispatchArgs {
                uint connected_tile_dispatch_args[];
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index >= uint(max(tile_count, 0))) {
                    return;
                }
                if (connected_tiles[int(index)] == 0) {
                    return;
                }
                uint slot = atomicAdd(connected_tile_count[0], 1u);
                ivec2 tile = ivec2(int(index) % tile_grid_size.x, int(index) / tile_grid_size.x);
                connected_tile_list[int(slot)] = tile;
                atomicMax(connected_tile_dispatch_args[0], slot + 1u);
            }
            """
        )
        self.programs["clear_formal_connected_cell_buffer"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int cell_count;

            layout(std430, binding=0) buffer CellBuffer {
                int values[];
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index >= uint(max(cell_count, 0))) {
                    return;
                }
                values[int(index)] = 0;
            }
            """
        )
        self.programs["clear_formal_connected_tile_mask_buffer"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;

            layout(std430, binding=0) buffer TileMaskBuffer {
                int values[];
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index >= uint(max(tile_count, 0))) {
                    return;
                }
                values[int(index)] = 0;
            }
            """
        )
        self.programs["clear_formal_connected_tile_masks_by_list"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;

            layout(std430, binding=0) buffer ConnectedTileMask {
                int connected_tiles[];
            };
            layout(std430, binding=1) buffer ConnectedTileScratchMask {
                int scratch_tiles[];
            };
            layout(std430, binding=2) buffer ConnectedTileSeedMask {
                int seed_tiles[];
            };
            layout(std430, binding=3) readonly buffer ConnectedTileCount {
                uint connected_tile_count[];
            };
            layout(std430, binding=4) readonly buffer ConnectedTileList {
                ivec2 connected_tile_list[];
            };

            void main() {
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {
                    return;
                }
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0) {
                    return;
                }
                if (tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return;
                }
                int compact_index = tile.y * tile_grid_size.x + tile.x;
                connected_tiles[compact_index] = 0;
                scratch_tiles[compact_index] = 0;
                seed_tiles[compact_index] = 0;
            }
            """
        )
        self.programs["clear_formal_connected_cell_buffer_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(std430, binding=0) buffer CellBuffer {{
                int values[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                values[cell.y * cell_grid_size.x + cell.x] = 0;
            }}
            """
        )
        self.programs["clear_formal_connected_cell_frontier_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;

            layout(std430, binding=0) buffer FrontierTileFlags {
                uint tile_flags[];
            };
            layout(std430, binding=1) buffer FrontierTileCount {
                uint tile_count_buffer[];
            };
            layout(std430, binding=2) buffer FrontierTileDispatchArgs {
                uint dispatch_args[];
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index == 0u) {
                    tile_count_buffer[0] = 0u;
                    dispatch_args[0] = 0u;
                    dispatch_args[1] = 1u;
                    dispatch_args[2] = 1u;
                }
                if (index < uint(max(tile_count, 0))) {
                    tile_flags[int(index)] = 0u;
                }
            }
            """
        )
        self.programs["clear_formal_connected_cell_frontier_tile_flags_by_list"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;

            layout(std430, binding=0) buffer FrontierTileFlags {
                uint tile_flags[];
            };
            layout(std430, binding=1) readonly buffer FrontierTileList {
                ivec2 tile_list[];
            };

            void main() {
                uint work_index = gl_WorkGroupID.x;
                ivec2 tile = tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return;
                }
                tile_flags[tile.y * tile_grid_size.x + tile.x] = 0u;
            }
            """
        )
        self.programs["reset_formal_connected_cell_frontier_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std430, binding=0) buffer FrontierTileFlags {
                uint tile_flags[];
            };
            layout(std430, binding=1) buffer FrontierTileCount {
                uint tile_count_buffer[];
            };
            layout(std430, binding=2) buffer FrontierTileDispatchArgs {
                uint dispatch_args[];
            };

            void main() {
                tile_count_buffer[0] = 0u;
                dispatch_args[0] = 0u;
                dispatch_args[1] = 1u;
                dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["accumulate_formal_connected_cell_frontier_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;

            layout(std430, binding=0) buffer TargetFrontierTileFlags {
                uint target_tile_flags[];
            };
            layout(std430, binding=1) buffer TargetFrontierTileList {
                ivec2 target_tile_list[];
            };
            layout(std430, binding=2) buffer TargetFrontierTileCount {
                uint target_tile_count[];
            };
            layout(std430, binding=3) buffer TargetFrontierDispatchArgs {
                uint target_dispatch_args[];
            };
            layout(std430, binding=4) readonly buffer SourceFrontierTileCount {
                uint source_tile_count[];
            };
            layout(std430, binding=5) readonly buffer SourceFrontierTileList {
                ivec2 source_tile_list[];
            };

            void main() {
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= source_tile_count[0]) {
                    return;
                }
                ivec2 tile = source_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return;
                }
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(target_tile_flags[tile_index], 0u, 1u) != 0u) {
                    return;
                }
                uint slot = atomicAdd(target_tile_count[0], 1u);
                target_tile_list[int(slot)] = tile;
                atomicMax(target_dispatch_args[0], slot + 1u);
            }
            """
        )
        self.programs["seed_formal_connected_tile_support_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D support_seed_tex;

            layout(std430, binding=0) buffer SupportedCells {{
                int supported_cells[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=4) buffer FrontierTileFlags {{
                uint frontier_tile_flags[];
            }};
            layout(std430, binding=5) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};

            shared uint s_seeded;
            shared uint s_touch_left;
            shared uint s_touch_right;
            shared uint s_touch_top;
            shared uint s_touch_bottom;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            void emit_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(frontier_tile_count[0], 1u);
                frontier_tile_list[int(slot)] = tile;
                atomicMax(frontier_tile_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_seeded = 0u;
                    s_touch_left = 0u;
                    s_touch_right = 0u;
                    s_touch_top = 0u;
                    s_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    in_grid(cell);
                bool seeded =
                    local_valid &&
                    texelFetch(structural_tex, cell, 0).x > 0.5 &&
                    texelFetch(support_seed_tex, cell, 0).x > 0.5;
                if (local_valid) {{
                    supported_cells[cell.y * cell_grid_size.x + cell.x] = seeded ? 1 : 0;
                }}
                if (seeded) {{
                    atomicOr(s_seeded, 1u);
                    if (local_cell.x == 0) {{
                        atomicOr(s_touch_left, 1u);
                    }}
                    if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_touch_right, 1u);
                    }}
                    if (local_cell.y == 0) {{
                        atomicOr(s_touch_top, 1u);
                    }}
                    if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_touch_bottom, 1u);
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_seeded != 0u) {{
                    emit_frontier_tile(tile);
                    if (s_touch_left != 0u) {{
                        emit_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_touch_right != 0u) {{
                        emit_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_touch_top != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_touch_bottom != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["expand_formal_connected_tile_support_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D structural_tex;

            layout(std430, binding=0) buffer SupportedCells {{
                int supported_cells[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer CurrentFrontierTileCount {{
                uint current_frontier_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer CurrentFrontierTileList {{
                ivec2 current_frontier_tile_list[];
            }};
            layout(std430, binding=4) buffer NextFrontierTileFlags {{
                uint next_frontier_tile_flags[];
            }};
            layout(std430, binding=5) buffer NextFrontierTileList {{
                ivec2 next_frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer NextFrontierTileCount {{
                uint next_frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer NextFrontierDispatchArgs {{
                uint next_frontier_dispatch_args[];
            }};

            shared int s_structural[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_supported[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_next[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared uint s_changed;
            shared uint s_step_changed;
            shared uint s_touch_left;
            shared uint s_touch_right;
            shared uint s_touch_top;
            shared uint s_touch_bottom;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            bool supported_global(ivec2 cell) {{
                if (!in_grid(cell)) {{
                    return false;
                }}
                return supported_cells[cell.y * cell_grid_size.x + cell.x] != 0;
            }}

            void emit_next_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(next_frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(next_frontier_tile_count[0], 1u);
                next_frontier_tile_list[int(slot)] = tile;
                atomicMax(next_frontier_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint frontier_index = gl_WorkGroupID.x;
                if (frontier_index >= current_frontier_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = current_frontier_tile_list[int(frontier_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_changed = 0u;
                    s_touch_left = 0u;
                    s_touch_right = 0u;
                    s_touch_top = 0u;
                    s_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    in_grid(cell);
                bool structural = local_valid && texelFetch(structural_tex, cell, 0).x > 0.5;
                bool supported = local_valid && supported_global(cell);
                s_structural[local_cell.y][local_cell.x] = structural ? 1 : 0;
                s_supported[local_cell.y][local_cell.x] = supported ? 1 : 0;
                s_next[local_cell.y][local_cell.x] = supported ? 1 : 0;
                barrier();

                int max_steps = max(1, tile_size * tile_size);
                for (int step = 0; step < {FORMAL_CONNECTED_TILE_LOCAL_SIZE * FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++step) {{
                    if (step >= max_steps) {{
                        break;
                    }}
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_step_changed = 0u;
                    }}
                    barrier();
                    int previous_value = s_supported[local_cell.y][local_cell.x];
                    int next_value = previous_value;
                    if (local_valid && s_structural[local_cell.y][local_cell.x] != 0 && next_value == 0) {{
                        bool touches_supported = false;
                        if (local_cell.x > 0) {{
                            touches_supported = touches_supported || s_supported[local_cell.y][local_cell.x - 1] != 0;
                        }} else {{
                            touches_supported = touches_supported || supported_global(cell + ivec2(-1, 0));
                        }}
                        if (local_cell.x + 1 < tile_size && local_cell.x + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_supported = touches_supported || s_supported[local_cell.y][local_cell.x + 1] != 0;
                        }} else {{
                            touches_supported = touches_supported || supported_global(cell + ivec2(1, 0));
                        }}
                        if (local_cell.y > 0) {{
                            touches_supported = touches_supported || s_supported[local_cell.y - 1][local_cell.x] != 0;
                        }} else {{
                            touches_supported = touches_supported || supported_global(cell + ivec2(0, -1));
                        }}
                        if (local_cell.y + 1 < tile_size && local_cell.y + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_supported = touches_supported || s_supported[local_cell.y + 1][local_cell.x] != 0;
                        }} else {{
                            touches_supported = touches_supported || supported_global(cell + ivec2(0, 1));
                        }}
                        next_value = touches_supported ? 1 : 0;
                    }}
                    if (local_valid && previous_value == 0 && next_value != 0) {{
                        atomicOr(s_step_changed, 1u);
                    }}
                    s_next[local_cell.y][local_cell.x] = next_value;
                    barrier();
                    s_supported[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
                    barrier();
                    if (s_step_changed == 0u) {{
                        break;
                    }}
                    barrier();
                }}

                if (local_valid) {{
                    int cell_index = cell.y * cell_grid_size.x + cell.x;
                    int final_supported = s_supported[local_cell.y][local_cell.x];
                    supported_cells[cell_index] = final_supported;
                    if (final_supported != 0 && !supported) {{
                        atomicOr(s_changed, 1u);
                        if (local_cell.x == 0) {{
                            atomicOr(s_touch_left, 1u);
                        }}
                        if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_right, 1u);
                        }}
                        if (local_cell.y == 0) {{
                            atomicOr(s_touch_top, 1u);
                        }}
                        if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_bottom, 1u);
                        }}
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u) {{
                    if (s_changed != 0u) {{
                        emit_next_frontier_tile(tile);
                    }}
                    if (s_touch_left != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_touch_right != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_touch_top != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_touch_bottom != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["seed_formal_connected_cell_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform ivec4 seed_rect;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(std430, binding=0) buffer ConnectedVisited {{
                int connected[];
            }};
            layout(std430, binding=1) buffer ConnectedScratch {{
                int scratch[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) buffer FrontierTileFlags {{
                uint frontier_tile_flags[];
            }};
            layout(std430, binding=4) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=5) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=6) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};
            layout(std430, binding=7) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=8) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            shared uint s_seeded;
            shared uint s_seed_touch_left;
            shared uint s_seed_touch_right;
            shared uint s_seed_touch_top;
            shared uint s_seed_touch_bottom;

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                return connected_tiles[tile_index] != 0;
            }}

            void emit_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(frontier_tile_count[0], 1u);
                frontier_tile_list[int(slot)] = tile;
                atomicMax(frontier_tile_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_seeded = 0u;
                    s_seed_touch_left = 0u;
                    s_seed_touch_right = 0u;
                    s_seed_touch_top = 0u;
                    s_seed_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid = !(
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                );
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                bool in_seed =
                    local_valid &&
                    connected_tiles[tile_index] != 0 &&
                    cell.x >= seed_rect.x &&
                    cell.y >= seed_rect.y &&
                    cell.x < seed_rect.z &&
                    cell.y < seed_rect.w;
                bool seeded = in_seed && texelFetch(structural_tex, cell, 0).x > 0.5;
                if (local_valid) {{
                    connected[cell_index] = seeded ? 1 : 0;
                    scratch[cell_index] = seeded ? 1 : 0;
                }}
                if (seeded) {{
                    atomicOr(s_seeded, 1u);
                    if (local_cell.x == 0) {{
                        atomicOr(s_seed_touch_left, 1u);
                    }}
                    if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_seed_touch_right, 1u);
                    }}
                    if (local_cell.y == 0) {{
                        atomicOr(s_seed_touch_top, 1u);
                    }}
                    if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_seed_touch_bottom, 1u);
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_seeded != 0u) {{
                    emit_frontier_tile(tile);
                    if (s_seed_touch_left != 0u) {{
                        emit_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_seed_touch_right != 0u) {{
                        emit_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_seed_touch_top != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_seed_touch_bottom != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["seed_formal_connected_cell_frontier_from_dirty_queue"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_tile_origin;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(std430, binding=0) buffer ConnectedVisited {{
                int connected[];
            }};
            layout(std430, binding=1) buffer ConnectedScratch {{
                int scratch[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) buffer FrontierTileFlags {{
                uint frontier_tile_flags[];
            }};
            layout(std430, binding=4) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=5) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=6) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};
            layout(std430, binding=7) readonly buffer DirtyTileCount {{
                uint dirty_tile_count[];
            }};
            layout(std430, binding=8) readonly buffer DirtyTileList {{
                ivec2 dirty_tile_list[];
            }};

            shared uint s_seeded;
            shared uint s_seed_touch_left;
            shared uint s_seed_touch_right;
            shared uint s_seed_touch_top;
            shared uint s_seed_touch_bottom;

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                return connected_tiles[tile_index] != 0;
            }}

            void emit_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(frontier_tile_count[0], 1u);
                frontier_tile_list[int(slot)] = tile;
                atomicMax(frontier_tile_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint dirty_index = gl_WorkGroupID.x;
                if (dirty_index >= dirty_tile_count[0]) {{
                    return;
                }}
                ivec2 dirty_tile = dirty_tile_list[int(dirty_index)];
                if (
                    dirty_tile.x < 0 ||
                    dirty_tile.y < 0 ||
                    dirty_tile.x >= tile_grid_size.x ||
                    dirty_tile.y >= tile_grid_size.y
                ) {{
                    return;
                }}
                ivec2 tile = dirty_tile - region_tile_origin;
                if (!tile_in_grid(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_seeded = 0u;
                    s_seed_touch_left = 0u;
                    s_seed_touch_right = 0u;
                    s_seed_touch_top = 0u;
                    s_seed_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    cell.x < cell_grid_size.x &&
                    cell.y < cell_grid_size.y;
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                bool seeded =
                    local_valid &&
                    connected_tiles[tile_index] != 0 &&
                    texelFetch(structural_tex, cell, 0).x > 0.5;
                if (local_valid) {{
                    connected[cell_index] = seeded ? 1 : 0;
                    scratch[cell_index] = seeded ? 1 : 0;
                }}
                if (seeded) {{
                    atomicOr(s_seeded, 1u);
                    if (local_cell.x == 0) {{
                        atomicOr(s_seed_touch_left, 1u);
                    }}
                    if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_seed_touch_right, 1u);
                    }}
                    if (local_cell.y == 0) {{
                        atomicOr(s_seed_touch_top, 1u);
                    }}
                    if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_seed_touch_bottom, 1u);
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_seeded != 0u) {{
                    emit_frontier_tile(tile);
                    if (s_seed_touch_left != 0u) {{
                        emit_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_seed_touch_right != 0u) {{
                        emit_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_seed_touch_top != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_seed_touch_bottom != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["expand_formal_connected_cells_by_tile"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int jump;
            uniform uint jump_generation;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(std430, binding=0) buffer ConnectedCurrent {{
                int connected_current[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) readonly buffer CurrentFrontierTileCount {{
                uint current_frontier_tile_count[];
            }};
            layout(std430, binding=4) readonly buffer CurrentFrontierTileList {{
                ivec2 current_frontier_tile_list[];
            }};
            layout(std430, binding=5) buffer NextFrontierTileFlags {{
                uint next_frontier_tile_flags[];
            }};
            layout(std430, binding=6) buffer NextFrontierTileCount {{
                uint next_frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer NextFrontierTileList {{
                ivec2 next_frontier_tile_list[];
            }};
            layout(std430, binding=8) buffer NextFrontierDispatchArgs {{
                uint next_frontier_dispatch_args[];
            }};

            shared int s_structural[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_connected[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_next[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared uint s_changed;
            shared uint s_touch_left;
            shared uint s_touch_right;
            shared uint s_touch_top;
            shared uint s_touch_bottom;
            shared uint s_step_changed;
            shared uint s_jump_changed;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                return connected_tiles[tile_index] != 0;
            }}

            bool current_connected_global(ivec2 cell) {{
                if (!in_grid(cell)) {{
                    return false;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                return connected_current[cell_index] != 0;
            }}

            bool structural_connected(ivec2 cell) {{
                return in_grid(cell) && tile_connected_mask(cell / tile_size) && texelFetch(structural_tex, cell, 0).x > 0.5;
            }}

            bool connected_sample(ivec2 sample_cell) {{
                return structural_connected(sample_cell) && current_connected_global(sample_cell);
            }}

            void solve_connected_cells(ivec2 local_cell, ivec2 cell, bool local_valid) {{
                int max_steps = max(1, tile_size * tile_size);
                for (int step = 0; step < {FORMAL_CONNECTED_TILE_LOCAL_SIZE * FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++step) {{
                    if (step >= max_steps) {{
                        break;
                    }}
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_step_changed = 0u;
                    }}
                    barrier();
                    int previous_value = s_connected[local_cell.y][local_cell.x];
                    int next_value = previous_value;
                    if (local_valid && s_structural[local_cell.y][local_cell.x] != 0 && next_value == 0) {{
                        bool touches_connected = false;
                        if (local_cell.x > 0) {{
                            touches_connected = touches_connected || s_connected[local_cell.y][local_cell.x - 1] != 0;
                        }} else {{
                            touches_connected = touches_connected || connected_sample(cell + ivec2(-1, 0));
                        }}
                        if (local_cell.x + 1 < tile_size && local_cell.x + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_connected = touches_connected || s_connected[local_cell.y][local_cell.x + 1] != 0;
                        }} else {{
                            touches_connected = touches_connected || connected_sample(cell + ivec2(1, 0));
                        }}
                        if (local_cell.y > 0) {{
                            touches_connected = touches_connected || s_connected[local_cell.y - 1][local_cell.x] != 0;
                        }} else {{
                            touches_connected = touches_connected || connected_sample(cell + ivec2(0, -1));
                        }}
                        if (local_cell.y + 1 < tile_size && local_cell.y + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_connected = touches_connected || s_connected[local_cell.y + 1][local_cell.x] != 0;
                        }} else {{
                            touches_connected = touches_connected || connected_sample(cell + ivec2(0, 1));
                        }}
                        next_value = touches_connected ? 1 : 0;
                    }}
                    if (local_valid && previous_value == 0 && next_value != 0) {{
                        atomicOr(s_step_changed, 1u);
                    }}
                    s_next[local_cell.y][local_cell.x] = next_value;
                    barrier();
                    s_connected[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
                    barrier();
                    if (s_step_changed == 0u) {{
                        break;
                    }}
                    barrier();
                }}
            }}

            bool sample_in_current_tile(ivec2 tile, ivec2 sample_cell) {{
                ivec2 sample_local = sample_cell - tile * tile_size;
                return
                    sample_local.x >= 0 &&
                    sample_local.y >= 0 &&
                    sample_local.x < tile_size &&
                    sample_local.y < tile_size;
            }}

            bool connected_line_clear(ivec2 start_cell, ivec2 end_cell) {{
                ivec2 delta = end_cell - start_cell;
                int steps = max(abs(delta.x), abs(delta.y));
                if (steps <= 1) {{
                    return true;
                }}
                ivec2 axis_step = ivec2(
                    delta.x == 0 ? 0 : (delta.x > 0 ? 1 : -1),
                    delta.y == 0 ? 0 : (delta.y > 0 ? 1 : -1)
                );
                for (int step_index = 1; step_index < steps; ++step_index) {{
                    ivec2 sample_cell = start_cell + axis_step * step_index;
                    if (!structural_connected(sample_cell)) {{
                        return false;
                    }}
                }}
                return true;
            }}

            void emit_next_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicExchange(next_frontier_tile_flags[tile_index], jump_generation) == jump_generation) {{
                    return;
                }}
                uint slot = atomicAdd(next_frontier_tile_count[0], 1u);
                next_frontier_tile_list[int(slot)] = tile;
                atomicMax(next_frontier_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint frontier_index = gl_WorkGroupID.x;
                if (frontier_index >= current_frontier_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = current_frontier_tile_list[int(frontier_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_changed = 0u;
                    s_touch_left = 0u;
                    s_touch_right = 0u;
                    s_touch_top = 0u;
                    s_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    in_grid(cell);
                bool structural = local_valid && texelFetch(structural_tex, cell, 0).x > 0.5;
                bool connected = local_valid && current_connected_global(cell);
                bool boundary_connected = structural && (
                    (local_cell.x == 0 && connected_sample(cell + ivec2(-1, 0))) ||
                    (local_cell.x + 1 >= tile_size && connected_sample(cell + ivec2(1, 0))) ||
                    (local_cell.y == 0 && connected_sample(cell + ivec2(0, -1))) ||
                    (local_cell.y + 1 >= tile_size && connected_sample(cell + ivec2(0, 1)))
                );
                s_structural[local_cell.y][local_cell.x] = structural ? 1 : 0;
                s_connected[local_cell.y][local_cell.x] = connected || boundary_connected ? 1 : 0;
                s_next[local_cell.y][local_cell.x] = s_connected[local_cell.y][local_cell.x];
                barrier();

                solve_connected_cells(local_cell, cell, local_valid);

                if (jump > 1) {{
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_jump_changed = 0u;
                    }}
                    barrier();

                    int jump_value = s_connected[local_cell.y][local_cell.x];
                    int previous_jump_value = jump_value;
                    if (local_valid && structural) {{
                        for (int axis = 0; axis < 4 && jump_value == 0; ++axis) {{
                            ivec2 offset = axis == 0 ? ivec2(-jump, 0) : axis == 1 ? ivec2(jump, 0) : axis == 2 ? ivec2(0, -jump) : ivec2(0, jump);
                            ivec2 sample_cell = cell + offset;
                            if (sample_in_current_tile(tile, sample_cell)) {{
                                continue;
                            }}
                            if (!connected_sample(sample_cell)) {{
                                continue;
                            }}
                            if (connected_line_clear(sample_cell, cell)) {{
                                jump_value = 1;
                            }}
                        }}
                    }}
                    if (local_valid && previous_jump_value == 0 && jump_value != 0) {{
                        s_connected[local_cell.y][local_cell.x] = 1;
                        atomicOr(s_jump_changed, 1u);
                    }}
                    barrier();
                    if (s_jump_changed != 0u) {{
                        solve_connected_cells(local_cell, cell, local_valid);
                    }}
                }}

                if (local_valid) {{
                    int cell_index = cell.y * cell_grid_size.x + cell.x;
                    bool final_connected = s_connected[local_cell.y][local_cell.x] != 0;
                    connected_current[cell_index] = final_connected ? 1 : 0;
                    if (final_connected && !connected) {{
                        atomicOr(s_changed, 1u);
                        if (local_cell.x == 0) {{
                            atomicOr(s_touch_left, 1u);
                        }}
                        if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_right, 1u);
                        }}
                        if (local_cell.y == 0) {{
                            atomicOr(s_touch_top, 1u);
                        }}
                        if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_bottom, 1u);
                        }}
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u) {{
                    if (s_changed != 0u) {{
                        emit_next_frontier_tile(tile);
                    }}
                    if (s_touch_left != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_touch_right != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_touch_top != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_touch_bottom != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["copy_formal_connected_buffer_to_texture"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(std430, binding=0) readonly buffer ConnectedMask {{
                int connected[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTiles {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(r32f, binding=1) writeonly uniform image2D connected_img;

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= region_size.x ||
                    local_cell.y >= region_size.y
                ) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= cell_grid_size.x || world_cell.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = local_cell.y * region_size.x + local_cell.x;
                bool structural = connected_tiles[tile_index] != 0 && texelFetch(structural_tex, local_cell, 0).x > 0.5;
                bool is_connected = structural && connected[cell_index] != 0;
                imageStore(connected_img, local_cell, vec4(is_connected ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["seed_structural_frontier_region"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(std430, binding=0) readonly buffer ConnectedFrontier {{
                int frontier[];
            }};
            layout(r32f, binding=1) writeonly uniform image2D support_seed_img;
            layout(r32f, binding=2) writeonly uniform image2D support_seed_copy_img;

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                bool seeded = texelFetch(structural_tex, local_cell, 0).x > 0.5 && frontier[cell_index] != 0;
                vec4 value = vec4(seeded ? 1.0 : 0.0, 0.0, 0.0, 0.0);
                imageStore(support_seed_img, local_cell, value);
                imageStore(support_seed_copy_img, local_cell, value);
            }}
            """
        )
        self.programs["publish_internal_boundary_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec4 internal_edges;

            layout(binding=0) uniform sampler2D boundary_connected_tex;
            layout(std430, binding=0) buffer ConnectedFrontier {{
                int frontier[];
            }};

            void main() {{
                ivec2 local_cell = ivec2(gl_GlobalInvocationID.xy);
                if (local_cell.x >= region_size.x || local_cell.y >= region_size.y) {{
                    return;
                }}
                if (texelFetch(boundary_connected_tex, local_cell, 0).x < 0.5) {{
                    return;
                }}
                bool internal_boundary =
                    (internal_edges.x != 0 && local_cell.x == 0)
                    || (internal_edges.y != 0 && local_cell.y == 0)
                    || (internal_edges.z != 0 && local_cell.x == region_size.x - 1)
                    || (internal_edges.w != 0 && local_cell.y == region_size.y - 1);
                if (!internal_boundary) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                frontier[cell_index] = 1;
            }}
            """
        )
        self.programs["detect_connected_internal_boundary_flags"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec4 internal_edges;

            layout(binding=0) uniform sampler2D eligibility_tex;
            layout(std430, binding=0) buffer RegionFlags {{
                uint region_flags;
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                if (texelFetch(eligibility_tex, cell, 0).x < 0.5) {{
                    return;
                }}
                uint flags = 0u;
                if (internal_edges.x != 0 && cell.x == 0) {{
                    flags |= {GPUCollapsePipeline.FORMAL_EXPAND_LEFT}u;
                }}
                if (internal_edges.y != 0 && cell.y == 0) {{
                    flags |= {GPUCollapsePipeline.FORMAL_EXPAND_TOP}u;
                }}
                if (internal_edges.z != 0 && cell.x == region_size.x - 1) {{
                    flags |= {GPUCollapsePipeline.FORMAL_EXPAND_RIGHT}u;
                }}
                if (internal_edges.w != 0 && cell.y == region_size.y - 1) {{
                    flags |= {GPUCollapsePipeline.FORMAL_EXPAND_BOTTOM}u;
                }}
                if (flags != 0u) {{
                    atomicOr(region_flags, flags);
                }}
            }}
            """
        )
        self.programs["enqueue_connected_internal_boundary_deferred_regions"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec4 internal_edges;
            uniform ivec4 solve_rect;
            uniform ivec2 world_size;
            uniform int request_capacity;

            layout(binding=0) uniform sampler2D eligibility_tex;
            layout(std430, binding=0) buffer RegionFlags {{
                uint region_flags;
            }};
            layout(std430, binding=1) buffer DeferredRequestCount {{
                uint request_count;
            }};
            layout(std430, binding=2) buffer DeferredRequests {{
                int request_rects[];
            }};

            bool valid_request(ivec4 rect) {{
                if (rect.x >= rect.z || rect.y >= rect.w) {{
                    return false;
                }}
                if (
                    rect.x == solve_rect.x &&
                    rect.y == solve_rect.y &&
                    rect.z == solve_rect.z &&
                    rect.w == solve_rect.w
                ) {{
                    return false;
                }}
                if (rect.x == 0 && rect.y == 0 && rect.z == world_size.x && rect.w == world_size.y) {{
                    return false;
                }}
                return true;
            }}

            void append_request(uint flag, ivec4 rect) {{
                if (!valid_request(rect)) {{
                    return;
                }}
                uint previous = atomicOr(region_flags, flag);
                if ((previous & flag) != 0u) {{
                    return;
                }}
                uint index = atomicAdd(request_count, 1u);
                if (index >= uint(max(request_capacity, 0))) {{
                    return;
                }}
                int base = int(index) * 4;
                request_rects[base + 0] = rect.x;
                request_rects[base + 1] = rect.y;
                request_rects[base + 2] = rect.z;
                request_rects[base + 3] = rect.w;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                if (texelFetch(eligibility_tex, cell, 0).x < 0.5) {{
                    return;
                }}
                if (internal_edges.x != 0 && cell.x == 0 && solve_rect.x > 0) {{
                    append_request({GPUCollapsePipeline.FORMAL_EXPAND_LEFT}u, ivec4(0, solve_rect.y, solve_rect.z, solve_rect.w));
                }}
                if (internal_edges.y != 0 && cell.y == 0 && solve_rect.y > 0) {{
                    append_request({GPUCollapsePipeline.FORMAL_EXPAND_TOP}u, ivec4(solve_rect.x, 0, solve_rect.z, solve_rect.w));
                }}
                if (internal_edges.z != 0 && cell.x == region_size.x - 1 && solve_rect.z < world_size.x) {{
                    append_request({GPUCollapsePipeline.FORMAL_EXPAND_RIGHT}u, ivec4(solve_rect.x, solve_rect.y, world_size.x, solve_rect.w));
                }}
                if (internal_edges.w != 0 && cell.y == region_size.y - 1 && solve_rect.w < world_size.y) {{
                    append_request({GPUCollapsePipeline.FORMAL_EXPAND_BOTTOM}u, ivec4(solve_rect.x, solve_rect.y, solve_rect.z, world_size.y));
                }}
            }}
            """
        )
        self.programs["seed_internal_boundary_region"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec4 internal_edges;

            layout(binding=0) uniform sampler2D eligibility_tex;
            layout(r32f, binding=1) writeonly uniform image2D support_seed_img;
            layout(r32f, binding=2) writeonly uniform image2D support_seed_copy_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                bool eligible = texelFetch(eligibility_tex, cell, 0).x > 0.5;
                bool internal_boundary =
                    (internal_edges.x != 0 && cell.x == 0)
                    || (internal_edges.y != 0 && cell.y == 0)
                    || (internal_edges.z != 0 && cell.x == region_size.x - 1)
                    || (internal_edges.w != 0 && cell.y == region_size.y - 1);
                vec4 value = vec4(eligible && internal_boundary ? 1.0 : 0.0, 0.0, 0.0, 0.0);
                imageStore(support_seed_img, cell, value);
                imageStore(support_seed_copy_img, cell, value);
            }}
            """
        )
        self.programs["exclude_boundary_connected_mask"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;

            layout(binding=0) uniform sampler2D eligibility_tex;
            layout(binding=1) uniform sampler2D boundary_connected_tex;
            layout(r32f, binding=2) writeonly uniform image2D filtered_eligibility_img;

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                bool eligible = texelFetch(eligibility_tex, cell, 0).x > 0.5;
                bool boundary_connected = texelFetch(boundary_connected_tex, cell, 0).x > 0.5;
                imageStore(filtered_eligibility_img, cell, vec4(eligible && !boundary_connected ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["build_formal_connected_axis_masks"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D source_tex;
            layout(std430, binding=0) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=3) buffer ConnectedTileRowMasks {{
                uint connected_tile_row_masks[];
            }};
            layout(std430, binding=4) buffer ConnectedTileColumnMasks {{
                uint connected_tile_column_masks[];
            }};

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (!tile_in_grid(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}

                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                int mask_base = tile_index * {FORMAL_CONNECTED_TILE_LOCAL_SIZE};
                if (local_cell.x == 0) {{
                    connected_tile_row_masks[mask_base + local_cell.y] = 0u;
                }}
                if (local_cell.y == 0) {{
                    connected_tile_column_masks[mask_base + local_cell.x] = 0u;
                }}
                memoryBarrierBuffer();
                barrier();

                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    !in_grid(cell) ||
                    texelFetch(source_tex, cell, 0).x <= 0.5
                ) {{
                    return;
                }}
                atomicOr(connected_tile_row_masks[mask_base + local_cell.y], 1u << uint(local_cell.x));
                atomicOr(connected_tile_column_masks[mask_base + local_cell.x], 1u << uint(local_cell.y));
            }}
            """
        )
        self.programs["propagate_formal_connected_tiles"] = ctx.compute_shader(
            (formal_connected_tiles_source := f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int jump;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D support_in_tex;
            layout(r32f, binding=2) writeonly uniform image2D support_out_img;
            layout(std430, binding=0) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileRowMasks {{
                uint connected_tile_row_masks[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileColumnMasks {{
                uint connected_tile_column_masks[];
            }};

            shared int s_structural[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_supported[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_next[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared uint s_step_changed;
            shared uint s_jump_changed;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            bool cell_connected(ivec2 cell) {{
                if (!in_grid(cell)) {{
                    return false;
                }}
                return tile_connected_mask(cell / tile_size);
            }}

            bool structural_connected(ivec2 cell) {{
                return cell_connected(cell) && texelFetch(structural_tex, cell, 0).x > 0.5;
            }}

            bool support_global(ivec2 cell) {{
                return structural_connected(cell) && texelFetch(support_in_tex, cell, 0).x > 0.5;
            }}

            void solve_supported_cells(ivec2 local_cell, ivec2 cell, bool local_valid) {{
                int max_steps = max(1, tile_size * tile_size);
                for (int step = 0; step < {FORMAL_CONNECTED_TILE_LOCAL_SIZE * FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++step) {{
                    if (step >= max_steps) {{
                        break;
                    }}
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_step_changed = 0u;
                    }}
                    barrier();
                    int previous_value = s_supported[local_cell.y][local_cell.x];
                    int next_value = previous_value;
                    if (local_valid && s_structural[local_cell.y][local_cell.x] != 0 && next_value == 0) {{
                        bool touches_supported = false;
                        if (local_cell.x > 0) {{
                            touches_supported = touches_supported || s_supported[local_cell.y][local_cell.x - 1] != 0;
                        }} else {{
                            touches_supported = touches_supported || support_global(cell + ivec2(-1, 0));
                        }}
                        if (local_cell.x + 1 < tile_size && local_cell.x + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_supported = touches_supported || s_supported[local_cell.y][local_cell.x + 1] != 0;
                        }} else {{
                            touches_supported = touches_supported || support_global(cell + ivec2(1, 0));
                        }}
                        if (local_cell.y > 0) {{
                            touches_supported = touches_supported || s_supported[local_cell.y - 1][local_cell.x] != 0;
                        }} else {{
                            touches_supported = touches_supported || support_global(cell + ivec2(0, -1));
                        }}
                        if (local_cell.y + 1 < tile_size && local_cell.y + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            touches_supported = touches_supported || s_supported[local_cell.y + 1][local_cell.x] != 0;
                        }} else {{
                            touches_supported = touches_supported || support_global(cell + ivec2(0, 1));
                        }}
                        next_value = touches_supported ? 1 : 0;
                    }}
                    if (local_valid && previous_value == 0 && next_value != 0) {{
                        atomicOr(s_step_changed, 1u);
                    }}
                    s_next[local_cell.y][local_cell.x] = next_value;
                    barrier();
                    s_supported[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
                    barrier();
                    if (s_step_changed == 0u) {{
                        break;
                    }}
                    barrier();
                }}
            }}

            bool sample_in_current_tile(ivec2 tile, ivec2 sample_cell) {{
                ivec2 sample_local = sample_cell - tile * tile_size;
                return
                    sample_local.x >= 0 &&
                    sample_local.y >= 0 &&
                    sample_local.x < tile_size &&
                    sample_local.y < tile_size;
            }}

            bool support_sample(ivec2 sample_cell) {{
                return support_global(sample_cell);
            }}

            int connected_tile_mask_base(ivec2 tile) {{
                return (tile.y * tile_grid_size.x + tile.x) * {FORMAL_CONNECTED_TILE_LOCAL_SIZE};
            }}

            uint bit_range_mask(int local_start, int local_end) {{
                int start_bit = max(0, min(tile_size, local_start));
                int end_bit = max(0, min(tile_size, local_end));
                if (end_bit <= start_bit) {{
                    return 0u;
                }}
                uint below_start = start_bit <= 0 ? 0u : ((1u << uint(start_bit)) - 1u);
                uint below_end = 0xffffffffu;
                if (end_bit < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                    below_end = (1u << uint(end_bit)) - 1u;
                }}
                return below_end & ~below_start;
            }}

            bool row_mask_segment_clear(ivec2 segment_tile, int local_y, int local_x0, int local_x1) {{
                if (
                    !tile_connected_mask(segment_tile) ||
                    local_y < 0 ||
                    local_y >= tile_size ||
                    local_y >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}
                ) {{
                    return false;
                }}
                uint required_mask = bit_range_mask(local_x0, local_x1);
                if (required_mask == 0u) {{
                    return true;
                }}
                uint available_mask = connected_tile_row_masks[connected_tile_mask_base(segment_tile) + local_y];
                return (available_mask & required_mask) == required_mask;
            }}

            bool column_mask_segment_clear(ivec2 segment_tile, int local_x, int local_y0, int local_y1) {{
                if (
                    !tile_connected_mask(segment_tile) ||
                    local_x < 0 ||
                    local_x >= tile_size ||
                    local_x >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}
                ) {{
                    return false;
                }}
                uint required_mask = bit_range_mask(local_y0, local_y1);
                if (required_mask == 0u) {{
                    return true;
                }}
                uint available_mask = connected_tile_column_masks[connected_tile_mask_base(segment_tile) + local_x];
                return (available_mask & required_mask) == required_mask;
            }}

            bool row_masks_line_clear(int y, int x0, int x1) {{
                if (y < 0 || y >= cell_grid_size.y || x0 < 0 || x1 > cell_grid_size.x) {{
                    return false;
                }}
                int cursor = x0;
                while (cursor < x1) {{
                    ivec2 segment_tile = ivec2(cursor / tile_size, y / tile_size);
                    int segment_end = min(x1, (segment_tile.x + 1) * tile_size);
                    if (segment_end <= cursor) {{
                        return false;
                    }}
                    int local_y = y - segment_tile.y * tile_size;
                    int local_x0 = cursor - segment_tile.x * tile_size;
                    int local_x1 = segment_end - segment_tile.x * tile_size;
                    if (!row_mask_segment_clear(segment_tile, local_y, local_x0, local_x1)) {{
                        return false;
                    }}
                    cursor = segment_end;
                }}
                return true;
            }}

            bool column_masks_line_clear(int x, int y0, int y1) {{
                if (x < 0 || x >= cell_grid_size.x || y0 < 0 || y1 > cell_grid_size.y) {{
                    return false;
                }}
                int cursor = y0;
                while (cursor < y1) {{
                    ivec2 segment_tile = ivec2(x / tile_size, cursor / tile_size);
                    int segment_end = min(y1, (segment_tile.y + 1) * tile_size);
                    if (segment_end <= cursor) {{
                        return false;
                    }}
                    int local_x = x - segment_tile.x * tile_size;
                    int local_y0 = cursor - segment_tile.y * tile_size;
                    int local_y1 = segment_end - segment_tile.y * tile_size;
                    if (!column_mask_segment_clear(segment_tile, local_x, local_y0, local_y1)) {{
                        return false;
                    }}
                    cursor = segment_end;
                }}
                return true;
            }}

            bool line_clear(ivec2 tile, ivec2 start_cell, ivec2 end_cell) {{
                ivec2 delta = end_cell - start_cell;
                if (delta.x != 0 && delta.y != 0) {{
                    return false;
                }}
                int steps = max(abs(delta.x), abs(delta.y));
                if (steps <= 1) {{
                    return true;
                }}
                if (delta.y == 0) {{
                    int x0 = min(start_cell.x, end_cell.x) + 1;
                    int x1 = max(start_cell.x, end_cell.x);
                    return row_masks_line_clear(start_cell.y, x0, x1);
                }}
                int y0 = min(start_cell.y, end_cell.y) + 1;
                int y1 = max(start_cell.y, end_cell.y);
                return column_masks_line_clear(start_cell.x, y0, y1);
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid = !(
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    !in_grid(cell)
                );
                bool structural = local_valid && texelFetch(structural_tex, cell, 0).x > 0.5;
                bool supported = structural && texelFetch(support_in_tex, cell, 0).x > 0.5;
                bool boundary_supported = structural && (
                    (local_cell.x == 0 && support_global(cell + ivec2(-1, 0))) ||
                    (local_cell.x + 1 >= tile_size && support_global(cell + ivec2(1, 0))) ||
                    (local_cell.y == 0 && support_global(cell + ivec2(0, -1))) ||
                    (local_cell.y + 1 >= tile_size && support_global(cell + ivec2(0, 1)))
                );
                s_structural[local_cell.y][local_cell.x] = structural ? 1 : 0;
                s_supported[local_cell.y][local_cell.x] = (supported || boundary_supported) ? 1 : 0;
                s_next[local_cell.y][local_cell.x] = s_supported[local_cell.y][local_cell.x];
                barrier();

                solve_supported_cells(local_cell, cell, local_valid);

                if (jump > 1) {{
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_jump_changed = 0u;
                    }}
                    barrier();

                    bool previously_supported = s_supported[local_cell.y][local_cell.x] != 0;
                    int jump_value = previously_supported ? 1 : 0;
                    int previous_jump_value = jump_value;
                    if (local_valid && structural) {{
                        for (int axis = 0; axis < 4 && jump_value == 0; ++axis) {{
                            ivec2 offset = axis == 0 ? ivec2(-jump, 0) : axis == 1 ? ivec2(jump, 0) : axis == 2 ? ivec2(0, -jump) : ivec2(0, jump);
                            ivec2 sample_cell = cell + offset;
                            if (sample_in_current_tile(tile, sample_cell)) {{
                                continue;
                            }}
                            if (!support_sample(sample_cell)) {{
                                continue;
                            }}
                            if (line_clear(tile, sample_cell, cell)) {{
                                jump_value = 1;
                            }}
                        }}
                    }}
                    if (local_valid && previous_jump_value == 0 && jump_value != 0) {{
                        s_supported[local_cell.y][local_cell.x] = 1;
                        atomicOr(s_jump_changed, 1u);
                    }}
	                    barrier();
	                    if (s_jump_changed != 0u) {{
	                        solve_supported_cells(local_cell, cell, local_valid);
	                    }}
	                }}

	                if (local_valid) {{
	                    bool final_supported = s_supported[local_cell.y][local_cell.x] != 0;
	                    imageStore(support_out_img, cell, vec4(final_supported ? 1.0 : 0.0, 0.0, 0.0, 0.0));
	                }}
            }}
            """)
        )
        self.programs["propagate_formal_connected_component_labels"] = ctx.compute_shader(
            (formal_connected_component_labels_source := f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int jump;
            uniform bool refine_local_labels;

            layout(binding=0) uniform sampler2D component_mask_tex;
            layout(binding=1) uniform sampler2D label_in_tex;
            layout(r32f, binding=2) writeonly uniform image2D label_out_img;

            layout(std430, binding=0) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileRowMasks {{
                uint connected_tile_row_masks[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileColumnMasks {{
                uint connected_tile_column_masks[];
            }};

		            shared int s_component[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
		            shared int s_label[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
	            shared int s_next[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
	            shared uint s_step_changed;
	            shared uint s_jump_changed;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            bool cell_connected(ivec2 cell) {{
                if (!in_grid(cell)) {{
                    return false;
                }}
                return tile_connected_mask(cell / tile_size);
            }}

            bool component_connected(ivec2 cell) {{
                return cell_connected(cell) && texelFetch(component_mask_tex, cell, 0).x > 0.5;
            }}

            int label_global(ivec2 cell) {{
                if (!component_connected(cell)) {{
                    return 0;
                }}
                return int(texelFetch(label_in_tex, cell, 0).x + 0.5);
            }}

	            bool sample_in_current_tile(ivec2 tile, ivec2 sample_cell) {{
	                ivec2 sample_local = sample_cell - tile * tile_size;
	                return
	                    sample_local.x >= 0 &&
	                    sample_local.y >= 0 &&
	                    sample_local.x < tile_size &&
	                    sample_local.y < tile_size &&
	                    sample_local.x < {FORMAL_CONNECTED_TILE_LOCAL_SIZE} &&
	                    sample_local.y < {FORMAL_CONNECTED_TILE_LOCAL_SIZE};
	            }}

			            int label_sample(ivec2 sample_cell) {{
			                return label_global(sample_cell);
			            }}

			            int connected_tile_mask_base(ivec2 tile) {{
			                return (tile.y * tile_grid_size.x + tile.x) * {FORMAL_CONNECTED_TILE_LOCAL_SIZE};
			            }}

			            uint bit_range_mask(int local_start, int local_end) {{
			                int start_bit = max(0, min(tile_size, local_start));
			                int end_bit = max(0, min(tile_size, local_end));
			                if (end_bit <= start_bit) {{
			                    return 0u;
			                }}
			                uint below_start = start_bit <= 0 ? 0u : ((1u << uint(start_bit)) - 1u);
			                uint below_end = 0xffffffffu;
			                if (end_bit < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
			                    below_end = (1u << uint(end_bit)) - 1u;
			                }}
			                return below_end & ~below_start;
			            }}

			            bool row_mask_segment_clear(ivec2 segment_tile, int local_y, int local_x0, int local_x1) {{
			                if (
			                    !tile_connected_mask(segment_tile) ||
			                    local_y < 0 ||
			                    local_y >= tile_size ||
			                    local_y >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}
			                ) {{
			                    return false;
			                }}
			                uint required_mask = bit_range_mask(local_x0, local_x1);
			                if (required_mask == 0u) {{
			                    return true;
			                }}
			                uint available_mask = connected_tile_row_masks[connected_tile_mask_base(segment_tile) + local_y];
			                return (available_mask & required_mask) == required_mask;
			            }}

			            bool column_mask_segment_clear(ivec2 segment_tile, int local_x, int local_y0, int local_y1) {{
			                if (
			                    !tile_connected_mask(segment_tile) ||
			                    local_x < 0 ||
			                    local_x >= tile_size ||
			                    local_x >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}
			                ) {{
			                    return false;
			                }}
			                uint required_mask = bit_range_mask(local_y0, local_y1);
			                if (required_mask == 0u) {{
			                    return true;
			                }}
			                uint available_mask = connected_tile_column_masks[connected_tile_mask_base(segment_tile) + local_x];
			                return (available_mask & required_mask) == required_mask;
			            }}

			            bool row_masks_line_clear(int y, int x0, int x1) {{
			                if (y < 0 || y >= cell_grid_size.y || x0 < 0 || x1 > cell_grid_size.x) {{
			                    return false;
			                }}
			                int cursor = x0;
			                while (cursor < x1) {{
			                    ivec2 segment_tile = ivec2(cursor / tile_size, y / tile_size);
			                    int segment_end = min(x1, (segment_tile.x + 1) * tile_size);
			                    if (segment_end <= cursor) {{
			                        return false;
			                    }}
			                    int local_y = y - segment_tile.y * tile_size;
			                    int local_x0 = cursor - segment_tile.x * tile_size;
			                    int local_x1 = segment_end - segment_tile.x * tile_size;
			                    if (!row_mask_segment_clear(segment_tile, local_y, local_x0, local_x1)) {{
			                        return false;
			                    }}
			                    cursor = segment_end;
			                }}
			                return true;
			            }}

			            bool column_masks_line_clear(int x, int y0, int y1) {{
			                if (x < 0 || x >= cell_grid_size.x || y0 < 0 || y1 > cell_grid_size.y) {{
			                    return false;
			                }}
			                int cursor = y0;
			                while (cursor < y1) {{
			                    ivec2 segment_tile = ivec2(x / tile_size, cursor / tile_size);
			                    int segment_end = min(y1, (segment_tile.y + 1) * tile_size);
			                    if (segment_end <= cursor) {{
			                        return false;
			                    }}
			                    int local_x = x - segment_tile.x * tile_size;
			                    int local_y0 = cursor - segment_tile.y * tile_size;
			                    int local_y1 = segment_end - segment_tile.y * tile_size;
			                    if (!column_mask_segment_clear(segment_tile, local_x, local_y0, local_y1)) {{
			                        return false;
			                    }}
			                    cursor = segment_end;
			                }}
			                return true;
			            }}

			            bool label_line_clear(ivec2 tile, ivec2 start_cell, ivec2 end_cell) {{
			                ivec2 delta = end_cell - start_cell;
			                if (delta.x != 0 && delta.y != 0) {{
			                    return false;
			                }}
		                int steps = max(abs(delta.x), abs(delta.y));
		                if (steps <= 1) {{
		                    return true;
		                }}
			                if (delta.y == 0) {{
			                    int x0 = min(start_cell.x, end_cell.x) + 1;
			                    int x1 = max(start_cell.x, end_cell.x);
			                    return row_masks_line_clear(start_cell.y, x0, x1);
			                }}
			                int y0 = min(start_cell.y, end_cell.y) + 1;
			                int y1 = max(start_cell.y, end_cell.y);
			                return column_masks_line_clear(start_cell.x, y0, y1);
			            }}

	            void solve_label_cells(ivec2 local_cell, ivec2 cell, bool local_valid) {{
	                int max_steps = max(1, tile_size * tile_size);
	                for (int step = 0; step < {FORMAL_CONNECTED_TILE_LOCAL_SIZE * FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++step) {{
	                    if (step >= max_steps) {{
	                        break;
	                    }}
	                    if (gl_LocalInvocationIndex == 0u) {{
	                        s_step_changed = 0u;
	                    }}
	                    barrier();
	                    int previous_label = s_label[local_cell.y][local_cell.x];
	                    int next_label = previous_label;
	                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && next_label > 0) {{
	                        if (local_cell.x > 0) {{
	                            int candidate = s_label[local_cell.y][local_cell.x - 1];
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }} else {{
	                            int candidate = label_global(cell + ivec2(-1, 0));
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }}
	                        if (local_cell.x + 1 < tile_size && local_cell.x + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
	                            int candidate = s_label[local_cell.y][local_cell.x + 1];
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }} else {{
	                            int candidate = label_global(cell + ivec2(1, 0));
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }}
	                        if (local_cell.y > 0) {{
	                            int candidate = s_label[local_cell.y - 1][local_cell.x];
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }} else {{
	                            int candidate = label_global(cell + ivec2(0, -1));
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }}
	                        if (local_cell.y + 1 < tile_size && local_cell.y + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
	                            int candidate = s_label[local_cell.y + 1][local_cell.x];
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }} else {{
	                            int candidate = label_global(cell + ivec2(0, 1));
	                            if (candidate > 0 && candidate < next_label) {{
	                                next_label = candidate;
	                            }}
	                        }}
	                    }}
		                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && next_label > 0 && next_label < previous_label) {{
		                        atomicOr(s_step_changed, 1u);
		                    }}
		                    s_next[local_cell.y][local_cell.x] = next_label;
		                    barrier();
		                    s_label[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
		                    barrier();
	                    if (s_step_changed == 0u) {{
	                        break;
	                    }}
	                    barrier();
	                }}
	            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid = !(
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    !in_grid(cell)
                );
                bool component = local_valid && texelFetch(component_mask_tex, cell, 0).x > 0.5;
                int current_label = component ? int(texelFetch(label_in_tex, cell, 0).x + 0.5) : 0;
	                s_component[local_cell.y][local_cell.x] = component ? 1 : 0;
	                s_label[local_cell.y][local_cell.x] = current_label;
	                s_next[local_cell.y][local_cell.x] = current_label;
	                barrier();

	                if (refine_local_labels) {{
		                    solve_label_cells(local_cell, cell, local_valid);
		                }}

	                if (jump > 1) {{
	                    if (gl_LocalInvocationIndex == 0u) {{
	                        s_jump_changed = 0u;
	                    }}
	                    barrier();

	                    int jump_label = s_label[local_cell.y][local_cell.x];
	                    int previous_jump_label = jump_label;
	                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && jump_label > 0) {{
	                        for (int axis = 0; axis < 4; ++axis) {{
	                            ivec2 offset = axis == 0 ? ivec2(-jump, 0) : axis == 1 ? ivec2(jump, 0) : axis == 2 ? ivec2(0, -jump) : ivec2(0, jump);
	                            ivec2 sample_cell = cell + offset;
	                            if (sample_in_current_tile(tile, sample_cell)) {{
	                                continue;
	                            }}
	                            int candidate = label_sample(sample_cell);
		                            if (candidate <= 0 || candidate >= jump_label) {{
		                                continue;
		                            }}
		                            if (label_line_clear(tile, sample_cell, cell)) {{
		                                jump_label = candidate;
		                            }}
		                        }}
		                    }}
		                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && jump_label > 0 && jump_label < previous_jump_label) {{
		                        atomicOr(s_jump_changed, 1u);
		                    }}
		                    s_next[local_cell.y][local_cell.x] = jump_label;
		                    barrier();
		                    s_label[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
		                    barrier();
	                    if (refine_local_labels && s_jump_changed != 0u) {{
		                        solve_label_cells(local_cell, cell, local_valid);
		                    }}
	                }}

                if (local_valid) {{
                    int final_label = s_component[local_cell.y][local_cell.x] != 0 ? s_label[local_cell.y][local_cell.x] : 0;
                    imageStore(label_out_img, cell, vec4(float(final_label), 0.0, 0.0, 0.0));
                }}
            }}
            """)
        )
        self.programs["component_label_propagate"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int jump;
            layout(binding=0) uniform sampler2D label_in_tex;
            layout(r32f, binding=1) writeonly uniform image2D label_out_img;
            layout(std430, binding=0) buffer ChangeFlagBuffer {{
                uint change_flag;
            }};

            bool label_line_clear(ivec2 start_cell, ivec2 end_cell) {{
                ivec2 delta = end_cell - start_cell;
                int steps = max(abs(delta.x), abs(delta.y));
                if (steps <= 1) {{
                    return true;
                }}
                for (int step_index = 1; step_index < steps; ++step_index) {{
                    float t = float(step_index) / float(steps);
                    ivec2 sample_cell = ivec2(round(mix(vec2(start_cell), vec2(end_cell), t)));
                    if (texelFetch(label_in_tex, sample_cell, 0).x < 0.5) {{
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
                float current = texelFetch(label_in_tex, cell, 0).x;
                float next_value = current;
                if (current > 0.5) {{
                    for (int axis = 0; axis < 4; ++axis) {{
                        ivec2 offset = axis == 0 ? ivec2(-jump, 0) : axis == 1 ? ivec2(jump, 0) : axis == 2 ? ivec2(0, -jump) : ivec2(0, jump);
                        ivec2 sample_cell = cell + offset;
                        if (sample_cell.x < 0 || sample_cell.y < 0 || sample_cell.x >= region_size.x || sample_cell.y >= region_size.y) {{
                            continue;
                        }}
                        if (!label_line_clear(cell, sample_cell)) {{
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
        self.programs["seed_formal_component_label_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D component_mask_tex;

            layout(std430, binding=0) buffer ComponentLabelsByCell {{
                int component_labels_by_cell[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(std430, binding=4) buffer FrontierTileFlags {{
                uint frontier_tile_flags[];
            }};
            layout(std430, binding=5) buffer FrontierTileList {{
                ivec2 frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer FrontierTileCount {{
                uint frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer FrontierTileDispatchArgs {{
                uint frontier_tile_dispatch_args[];
            }};

            shared uint s_has_component;
            shared uint s_touch_left;
            shared uint s_touch_right;
            shared uint s_touch_top;
            shared uint s_touch_bottom;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            void emit_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(frontier_tile_count[0], 1u);
                frontier_tile_list[int(slot)] = tile;
                atomicMax(frontier_tile_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_has_component = 0u;
                    s_touch_left = 0u;
                    s_touch_right = 0u;
                    s_touch_top = 0u;
                    s_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    in_grid(cell);
                bool component = local_valid && texelFetch(component_mask_tex, cell, 0).x > 0.5;
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                if (local_valid) {{
                    component_labels_by_cell[cell_index] = component ? (cell_index + 1) : 0;
                }}
                if (component) {{
                    atomicOr(s_has_component, 1u);
                    if (local_cell.x == 0) {{
                        atomicOr(s_touch_left, 1u);
                    }}
                    if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_touch_right, 1u);
                    }}
                    if (local_cell.y == 0) {{
                        atomicOr(s_touch_top, 1u);
                    }}
                    if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                        atomicOr(s_touch_bottom, 1u);
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u && s_has_component != 0u) {{
                    emit_frontier_tile(tile);
                    if (s_touch_left != 0u) {{
                        emit_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_touch_right != 0u) {{
                        emit_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_touch_top != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_touch_bottom != 0u) {{
                        emit_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["expand_formal_component_label_frontier"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D component_mask_tex;

            layout(std430, binding=0) buffer ComponentLabelsByCell {{
                int component_labels_by_cell[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer CurrentFrontierTileCount {{
                uint current_frontier_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer CurrentFrontierTileList {{
                ivec2 current_frontier_tile_list[];
            }};
            layout(std430, binding=4) buffer NextFrontierTileFlags {{
                uint next_frontier_tile_flags[];
            }};
            layout(std430, binding=5) buffer NextFrontierTileList {{
                ivec2 next_frontier_tile_list[];
            }};
            layout(std430, binding=6) buffer NextFrontierTileCount {{
                uint next_frontier_tile_count[];
            }};
            layout(std430, binding=7) buffer NextFrontierDispatchArgs {{
                uint next_frontier_dispatch_args[];
            }};

            shared int s_component[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_label[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared int s_next[{FORMAL_CONNECTED_TILE_LOCAL_SIZE}][{FORMAL_CONNECTED_TILE_LOCAL_SIZE}];
            shared uint s_changed;
            shared uint s_step_changed;
            shared uint s_touch_left;
            shared uint s_touch_right;
            shared uint s_touch_top;
            shared uint s_touch_bottom;

            bool in_grid(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_in_grid(ivec2 tile) {{
                return tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y;
            }}

            bool tile_connected_mask(ivec2 tile) {{
                if (!tile_in_grid(tile)) {{
                    return false;
                }}
                return connected_tiles[tile.y * tile_grid_size.x + tile.x] != 0;
            }}

            bool component_global(ivec2 cell) {{
                return
                    in_grid(cell) &&
                    tile_connected_mask(cell / tile_size) &&
                    texelFetch(component_mask_tex, cell, 0).x > 0.5;
            }}

            int label_global(ivec2 cell) {{
                if (!component_global(cell)) {{
                    return 0;
                }}
                return component_labels_by_cell[cell.y * cell_grid_size.x + cell.x];
            }}

            void emit_next_frontier_tile(ivec2 tile) {{
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(next_frontier_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(next_frontier_tile_count[0], 1u);
                next_frontier_tile_list[int(slot)] = tile;
                atomicMax(next_frontier_dispatch_args[0], slot + 1u);
            }}

            void main() {{
                uint frontier_index = gl_WorkGroupID.x;
                if (frontier_index >= current_frontier_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = current_frontier_tile_list[int(frontier_index)];
                if (!tile_connected_mask(tile)) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    s_changed = 0u;
                    s_touch_left = 0u;
                    s_touch_right = 0u;
                    s_touch_top = 0u;
                    s_touch_bottom = 0u;
                }}
                barrier();
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                bool local_valid =
                    local_cell.x < tile_size &&
                    local_cell.y < tile_size &&
                    in_grid(cell);
                int old_label = local_valid ? label_global(cell) : 0;
                bool component = local_valid && old_label > 0;
                s_component[local_cell.y][local_cell.x] = component ? 1 : 0;
                s_label[local_cell.y][local_cell.x] = old_label;
                s_next[local_cell.y][local_cell.x] = old_label;
                barrier();

                int max_steps = max(1, tile_size * tile_size);
                for (int step = 0; step < {FORMAL_CONNECTED_TILE_LOCAL_SIZE * FORMAL_CONNECTED_TILE_LOCAL_SIZE}; ++step) {{
                    if (step >= max_steps) {{
                        break;
                    }}
                    if (gl_LocalInvocationIndex == 0u) {{
                        s_step_changed = 0u;
                    }}
                    barrier();
                    int previous_label = s_label[local_cell.y][local_cell.x];
                    int next_label = previous_label;
                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && next_label > 0) {{
                        if (local_cell.x > 0) {{
                            int candidate = s_label[local_cell.y][local_cell.x - 1];
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }} else {{
                            int candidate = label_global(cell + ivec2(-1, 0));
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }}
                        if (local_cell.x + 1 < tile_size && local_cell.x + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            int candidate = s_label[local_cell.y][local_cell.x + 1];
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }} else {{
                            int candidate = label_global(cell + ivec2(1, 0));
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }}
                        if (local_cell.y > 0) {{
                            int candidate = s_label[local_cell.y - 1][local_cell.x];
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }} else {{
                            int candidate = label_global(cell + ivec2(0, -1));
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }}
                        if (local_cell.y + 1 < tile_size && local_cell.y + 1 < {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            int candidate = s_label[local_cell.y + 1][local_cell.x];
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }} else {{
                            int candidate = label_global(cell + ivec2(0, 1));
                            if (candidate > 0 && candidate < next_label) {{
                                next_label = candidate;
                            }}
                        }}
                    }}
                    if (local_valid && s_component[local_cell.y][local_cell.x] != 0 && next_label > 0 && next_label < previous_label) {{
                        atomicOr(s_step_changed, 1u);
                    }}
                    s_next[local_cell.y][local_cell.x] = next_label;
                    barrier();
                    s_label[local_cell.y][local_cell.x] = s_next[local_cell.y][local_cell.x];
                    barrier();
                    if (s_step_changed == 0u) {{
                        break;
                    }}
                    barrier();
                }}

                if (local_valid && component) {{
                    int final_label = s_label[local_cell.y][local_cell.x];
                    component_labels_by_cell[cell.y * cell_grid_size.x + cell.x] = final_label;
                    if (final_label > 0 && final_label < old_label) {{
                        atomicOr(s_changed, 1u);
                        if (local_cell.x == 0) {{
                            atomicOr(s_touch_left, 1u);
                        }}
                        if (local_cell.x + 1 >= tile_size || local_cell.x + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_right, 1u);
                        }}
                        if (local_cell.y == 0) {{
                            atomicOr(s_touch_top, 1u);
                        }}
                        if (local_cell.y + 1 >= tile_size || local_cell.y + 1 >= {FORMAL_CONNECTED_TILE_LOCAL_SIZE}) {{
                            atomicOr(s_touch_bottom, 1u);
                        }}
                    }}
                }}
                barrier();
                if (gl_LocalInvocationIndex == 0u) {{
                    if (s_changed != 0u) {{
                        emit_next_frontier_tile(tile);
                    }}
                    if (s_touch_left != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(-1, 0));
                    }}
                    if (s_touch_right != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(1, 0));
                    }}
                    if (s_touch_top != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, -1));
                    }}
                    if (s_touch_bottom != 0u) {{
                        emit_next_frontier_tile(tile + ivec2(0, 1));
                    }}
                }}
            }}
            """
        )
        self.programs["copy_formal_component_label_buffer_to_texture"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D component_mask_tex;
            layout(r32f, binding=1) writeonly uniform image2D label_out_img;

            layout(std430, binding=0) readonly buffer ComponentLabelsByCell {{
                int component_labels_by_cell[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                bool component = texelFetch(component_mask_tex, cell, 0).x > 0.5;
                int label = component ? component_labels_by_cell[cell_index] : 0;
                imageStore(label_out_img, cell, vec4(float(label), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["collect_component_labels"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 empty_min;

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
            layout(std430, binding=3) buffer ComponentMetadata {{
                int component_metadata[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int cell_index = cell.y * region_size.x + cell.x;
                int label = int(texelFetch(label_tex, cell, 0).x + 0.5);
                if (label != cell_index + 1) {{
                    return;
                }}
                uint output_index = atomicAdd(component_count, 1u);
                component_labels[output_index] = label;
                component_flags[uint(label - 1)] = output_index + 1u;
                int base = int(output_index) * 5;
                component_metadata[base + 0] = empty_min.x;
                component_metadata[base + 1] = empty_min.y;
                component_metadata[base + 2] = 0;
                component_metadata[base + 3] = 0;
                component_metadata[base + 4] = 0;
            }}
            """
        )
        self.programs["clear_component_label_flags_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int component_capacity;

            layout(binding=0) uniform sampler2D label_tex;

            layout(std430, binding=0) buffer ComponentFlags {{
                uint component_flags[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int label = int(texelFetch(label_tex, cell, 0).x + 0.5);
                if (label <= 0 || label > component_capacity) {{
                    return;
                }}
                component_flags[uint(label - 1)] = 0u;
            }}
            """
        )
        self.programs["collect_component_labels_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int component_capacity;
            uniform ivec2 empty_min;

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
            layout(std430, binding=3) buffer ComponentMetadata {{
                int component_metadata[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=6) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                int label = int(texelFetch(label_tex, cell, 0).x + 0.5);
                if (label <= 0 || label > component_capacity || label != cell_index + 1) {{
                    return;
                }}
                uint output_index = atomicAdd(component_count, 1u);
                component_labels[output_index] = label;
                component_flags[uint(label - 1)] = output_index + 1u;
                int base = int(output_index) * 5;
                component_metadata[base + 0] = empty_min.x;
                component_metadata[base + 1] = empty_min.y;
                component_metadata[base + 2] = 0;
                component_metadata[base + 3] = 0;
                component_metadata[base + 4] = 0;
            }}
            """
        )
        self.programs["build_component_dispatch_args"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform int component_capacity;
            uniform int invocations_per_group;

            layout(std430, binding=0) readonly buffer ComponentCount {
                uint component_count;
            };
            layout(std430, binding=1) buffer ComponentDispatchArgs {
                uint dispatch_args[];
            };

            void main() {
                uint count = min(component_count, uint(max(component_capacity, 0)));
                uint group_size = uint(max(invocations_per_group, 1));
                dispatch_args[0] = max(1u, (count + group_size - 1u) / group_size);
                dispatch_args[1] = 1u;
                dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["index_compact_component_labels"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int component_capacity;
            uniform ivec2 empty_min;

            layout(std430, binding=0) readonly buffer ComponentLabels {
                int component_labels[];
            };
            layout(std430, binding=1) buffer ComponentLabelIndices {
                uint component_label_indices[];
            };
            layout(std430, binding=2) buffer ComponentMetadata {
                int component_metadata[];
            };
            layout(std430, binding=3) readonly buffer ComponentCount {
                uint component_count;
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                uint count = min(component_count, uint(max(component_capacity, 0)));
                if (index >= count) {
                    return;
                }
                int label = component_labels[index];
                if (label <= 0 || label > component_capacity) {
                    return;
                }
                component_label_indices[uint(label - 1)] = index + 1u;
                int base = int(index) * 5;
                component_metadata[base + 0] = empty_min.x;
                component_metadata[base + 1] = empty_min.y;
                component_metadata[base + 2] = 0;
                component_metadata[base + 3] = 0;
                component_metadata[base + 4] = 0;
            }
            """
        )
        self.programs["summarize_compact_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform int component_capacity;

            layout(binding=0) uniform sampler2D component_label_tex;

            layout(std430, binding=0) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};
            layout(std430, binding=1) buffer ComponentMetadata {{
                int component_metadata[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label <= 0 || label > component_capacity) {{
                    return;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return;
                }}
                int base = int(slot_plus_one - 1u) * 5;
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
        self.programs["summarize_compact_components_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform ivec2 region_origin;
            uniform int tile_size;
            uniform int component_capacity;

            layout(binding=0) uniform sampler2D component_label_tex;

            layout(std430, binding=0) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};
            layout(std430, binding=1) buffer ComponentMetadata {{
                int component_metadata[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 local_cell = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + local_cell;
                if (
                    local_cell.x >= tile_size ||
                    local_cell.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label <= 0 || label > component_capacity) {{
                    return;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return;
                }}
                int base = int(slot_plus_one - 1u) * 5;
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
            uniform bool use_eligibility;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D supported_tex;
            layout(binding=2) uniform sampler2D behavior_tex;
            layout(binding=3) uniform sampler2D pending_tex;
            layout(binding=7) uniform sampler2D eligibility_tex;
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
                bool eligible = !use_eligibility || texelFetch(eligibility_tex, cell, 0).x > 0.5;
                bool unsupported = eligible && structural && !supported;
                int behavior = int(texelFetch(behavior_tex, cell, 0).x + 0.5);
                bool pending = texelFetch(pending_tex, cell, 0).x > 0.5;
                bool delayed_pending = (use_eligibility && !eligible)
                    ? pending
                    : (unsupported && behavior == behavior_delayed && !pending);
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
        self.programs["resolve_outcomes_from_supported_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int behavior_falling_island;
            uniform int behavior_delayed;
            uniform int behavior_immune;
            uniform bool use_eligibility;

            layout(binding=0) uniform sampler2D structural_tex;
            layout(binding=1) uniform sampler2D supported_tex;
            layout(binding=2) uniform sampler2D behavior_tex;
            layout(binding=3) uniform sampler2D pending_tex;
            layout(binding=7) uniform sampler2D eligibility_tex;
            layout(r32f, binding=4) writeonly uniform image2D delayed_pending_img;
            layout(r32f, binding=5) writeonly uniform image2D immune_unsupported_img;
            layout(r32f, binding=6) writeonly uniform image2D collapse_now_img;
            layout(std430, binding=0) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    cell.x >= region_size.x ||
                    cell.y >= region_size.y
                ) {{
                    return;
                }}
                bool structural = texelFetch(structural_tex, cell, 0).x > 0.5;
                bool supported = texelFetch(supported_tex, cell, 0).x > 0.5;
                bool eligible = !use_eligibility || texelFetch(eligibility_tex, cell, 0).x > 0.5;
                bool unsupported = eligible && structural && !supported;
                int behavior = int(texelFetch(behavior_tex, cell, 0).x + 0.5);
                bool pending = texelFetch(pending_tex, cell, 0).x > 0.5;
                bool delayed_pending = (use_eligibility && !eligible)
                    ? pending
                    : (unsupported && behavior == behavior_delayed && !pending);
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
        self.programs["materialize_compact_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int label_capacity;
            uniform int island_id_base;
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

            layout(std430, binding=0) buffer CollapseGenerationIds {{
                int collapse_generation_ids[];
            }};
            layout(std430, binding=1) buffer BaseIntegrityValues {{
                float base_integrity_values[];
            }};
            layout(std430, binding=2) buffer SpawnTemperatureValues {{
                float spawn_temperature_values[];
            }};
            layout(std430, binding=3) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return 0;
                }}
                return island_id_base + int(slot_plus_one) - 1;
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
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    int generated_material = collapse_generation_id(source_material);
                    int final_material = generated_material > 0 ? generated_material : source_material;
                    out_material = float(final_material);
                    out_phase = float(phase_falling_island);
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
        self.programs["materialize_compact_components_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int label_capacity;
            uniform int island_id_base;
            layout(binding=0) uniform sampler2D component_label_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            layout(std430, binding=3) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return 0;
                }}
                return island_id_base + int(slot_plus_one) - 1;
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
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    out_island_id = float(next_island_id);
                    out_entity_id = 0.0;
                    out_displaced = 0.0;
                }}
                imageStore(island_id_out_img, cell, vec4(out_island_id, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, cell, vec4(out_entity_id, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, cell, vec4(out_displaced, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["materialize_compact_components_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int label_capacity;
            uniform int island_id_base;
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

            layout(std430, binding=0) buffer CollapseGenerationIds {{
                int collapse_generation_ids[];
            }};
            layout(std430, binding=1) buffer BaseIntegrityValues {{
                float base_integrity_values[];
            }};
            layout(std430, binding=2) buffer SpawnTemperatureValues {{
                float spawn_temperature_values[];
            }};
            layout(std430, binding=3) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=6) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return 0;
                }}
                return island_id_base + int(slot_plus_one) - 1;
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
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int source_material = int(texelFetch(material_tex, cell, 0).x + 0.5);
                float out_material = float(source_material);
                float out_phase = texelFetch(phase_tex, cell, 0).x;
                float out_flags = texelFetch(cell_flags_tex, cell, 0).x;
                vec4 out_timer = texelFetch(timer_tex, cell, 0);
                float out_integrity = texelFetch(integrity_tex, cell, 0).x;
                float out_temperature = texelFetch(temp_tex, cell, 0).x;
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    int generated_material = collapse_generation_id(source_material);
                    int final_material = generated_material > 0 ? generated_material : source_material;
                    out_material = float(final_material);
                    out_phase = float(phase_falling_island);
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
        self.programs["materialize_compact_components_aux_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int label_capacity;
            uniform int island_id_base;
            layout(binding=0) uniform sampler2D component_label_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            layout(std430, binding=3) readonly buffer ComponentLabelIndices {{
                uint component_label_indices[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=6) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                uint slot_plus_one = component_label_indices[uint(label - 1)];
                if (slot_plus_one == 0u) {{
                    return 0;
                }}
                return island_id_base + int(slot_plus_one) - 1;
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    cell.x >= cell_grid_size.x ||
                    cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                float out_island_id = texelFetch(island_id_tex, cell, 0).x;
                float out_entity_id = texelFetch(entity_id_tex, cell, 0).x;
                float out_displaced = texelFetch(displaced_tex, cell, 0).x;
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    out_island_id = float(next_island_id);
                    out_entity_id = 0.0;
                    out_displaced = 0.0;
                }}
                imageStore(island_id_out_img, cell, vec4(out_island_id, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, cell, vec4(out_entity_id, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, cell, vec4(out_displaced, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["materialize_dense_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int label_capacity;
            uniform int island_id_base;
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

            layout(std430, binding=0) buffer CollapseGenerationIds {{
                int collapse_generation_ids[];
            }};
            layout(std430, binding=1) buffer BaseIntegrityValues {{
                float base_integrity_values[];
            }};
            layout(std430, binding=2) buffer SpawnTemperatureValues {{
                float spawn_temperature_values[];
            }};

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                return island_id_base + label - 1;
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

                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    int generated_material = collapse_generation_id(source_material);
                    int final_material = generated_material > 0 ? generated_material : source_material;
                    out_material = float(final_material);
                    out_phase = float(phase_falling_island);
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
        self.programs["materialize_dense_components_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform int label_capacity;
            uniform int island_id_base;
            layout(binding=0) uniform sampler2D component_label_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            int island_id_for_label(int label) {{
                if (label <= 0 || label > label_capacity) {{
                    return 0;
                }}
                return island_id_base + label - 1;
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
                int next_island_id = island_id_for_label(label);
                if (next_island_id > 0) {{
                    out_island_id = float(next_island_id);
                    out_entity_id = 0.0;
                    out_displaced = 0.0;
                }}
                imageStore(island_id_out_img, cell, vec4(out_island_id, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, cell, vec4(out_entity_id, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, cell, vec4(out_displaced, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["init_dense_component_metadata"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int component_capacity;
            uniform ivec2 empty_min;

            layout(std430, binding=0) buffer ComponentMetadata {
                int component_metadata[];
            };

            void main() {
                int index = int(gl_GlobalInvocationID.x);
                if (index >= component_capacity) {
                    return;
                }
                int base = index * 5;
                component_metadata[base + 0] = empty_min.x;
                component_metadata[base + 1] = empty_min.y;
                component_metadata[base + 2] = 0;
                component_metadata[base + 3] = 0;
                component_metadata[base + 4] = 0;
            }
            """
        )
        self.programs["summarize_dense_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform int component_capacity;

            layout(binding=0) uniform sampler2D component_label_tex;

            layout(std430, binding=0) buffer ComponentMetadata {{
                int component_metadata[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= region_size.x || cell.y >= region_size.y) {{
                    return;
                }}
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int index = label - 1;
                if (index < 0 || index >= component_capacity) {{
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
        self.programs["publish_dense_component_island_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int component_capacity;
            uniform int island_id_base;
            uniform ivec2 cell_grid_size;
            uniform ivec2 paging_origin;
            uniform ivec2 paging_buffer_origin;

            layout(std430, binding=0) readonly buffer ComponentMetadata {{
                int component_metadata[];
            }};
            layout(std430, binding=1) writeonly buffer IslandRuntimeWords {{
                int runtime_words[];
            }};
            layout(std430, binding=2) buffer IslandRuntimeCount {{
                int runtime_count;
            }};

            int positive_mod(int value, int divisor) {{
                int result = value % divisor;
                return result < 0 ? result + divisor : result;
            }}

            void store_runtime_word(int record_index, int word_offset, int value) {{
                runtime_words[record_index * {ISLAND_RUNTIME_DTYPE.itemsize // 4} + word_offset] = value;
            }}

            void clear_runtime_record(int record_index) {{
                for (int word_offset = 0; word_offset < {ISLAND_RUNTIME_DTYPE.itemsize // 4}; ++word_offset) {{
                    store_runtime_word(record_index, word_offset, 0);
                }}
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= component_capacity) {{
                    return;
                }}
                clear_runtime_record(index);
                int base = index * 5;
                int cell_count = component_metadata[base + 4];
                if (cell_count <= 0) {{
                    return;
                }}
                atomicAdd(runtime_count, 1);
                int x0 = component_metadata[base + 0];
                int y0 = component_metadata[base + 1];
                int x1 = component_metadata[base + 2];
                int y1 = component_metadata[base + 3];
                int width = max(0, x1 - x0);
                int height = max(0, y1 - y0);
                int world_x0 = paging_origin.x + positive_mod(x0 - paging_buffer_origin.x, cell_grid_size.x);
                int world_y0 = paging_origin.y + positive_mod(y0 - paging_buffer_origin.y, cell_grid_size.y);

                store_runtime_word(index, 0, island_id_base + index);
                store_runtime_word(index, 1, x0);
                store_runtime_word(index, 2, y0);
                store_runtime_word(index, 3, x1);
                store_runtime_word(index, 4, y1);
                store_runtime_word(index, 5, world_x0);
                store_runtime_word(index, 6, world_y0);
                store_runtime_word(index, 7, world_x0 + width);
                store_runtime_word(index, 8, world_y0 + height);
                store_runtime_word(index, 9, floatBitsToInt(0.0));
                store_runtime_word(index, 10, floatBitsToInt(0.0));
                store_runtime_word(index, 11, floatBitsToInt(0.0));
                store_runtime_word(index, 12, floatBitsToInt(0.0));
            }}
            """
        )
        self.programs["publish_compact_component_island_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int component_capacity;
            uniform int island_id_base;
            uniform ivec2 cell_grid_size;
            uniform ivec2 paging_origin;
            uniform ivec2 paging_buffer_origin;

            layout(std430, binding=0) readonly buffer ComponentMetadata {{
                int component_metadata[];
            }};
            layout(std430, binding=1) writeonly buffer IslandRuntimeWords {{
                int runtime_words[];
            }};
            layout(std430, binding=2) buffer IslandRuntimeCount {{
                int runtime_count;
            }};
            layout(std430, binding=3) readonly buffer ComponentCount {{
                uint component_count;
            }};

            int positive_mod(int value, int divisor) {{
                int result = value % divisor;
                return result < 0 ? result + divisor : result;
            }}

            void store_runtime_word(int record_index, int word_offset, int value) {{
                runtime_words[record_index * {ISLAND_RUNTIME_DTYPE.itemsize // 4} + word_offset] = value;
            }}

            void main() {{
                uint index = gl_GlobalInvocationID.x;
                uint count = min(component_count, uint(max(component_capacity, 0)));
                if (index >= count) {{
                    return;
                }}
                int base = int(index) * 5;
                int cell_count = component_metadata[base + 4];
                if (cell_count <= 0) {{
                    return;
                }}
                int out_index = atomicAdd(runtime_count, 1);
                int x0 = component_metadata[base + 0];
                int y0 = component_metadata[base + 1];
                int x1 = component_metadata[base + 2];
                int y1 = component_metadata[base + 3];
                int width = max(0, x1 - x0);
                int height = max(0, y1 - y0);
                int world_x0 = paging_origin.x + positive_mod(x0 - paging_buffer_origin.x, cell_grid_size.x);
                int world_y0 = paging_origin.y + positive_mod(y0 - paging_buffer_origin.y, cell_grid_size.y);

                store_runtime_word(out_index, 0, island_id_base + int(index));
                store_runtime_word(out_index, 1, x0);
                store_runtime_word(out_index, 2, y0);
                store_runtime_word(out_index, 3, x1);
                store_runtime_word(out_index, 4, y1);
                store_runtime_word(out_index, 5, world_x0);
                store_runtime_word(out_index, 6, world_y0);
                store_runtime_word(out_index, 7, world_x0 + width);
                store_runtime_word(out_index, 8, world_y0 + height);
                store_runtime_word(out_index, 9, floatBitsToInt(0.0));
                store_runtime_word(out_index, 10, floatBitsToInt(0.0));
                store_runtime_word(out_index, 11, floatBitsToInt(0.0));
                store_runtime_word(out_index, 12, floatBitsToInt(0.0));
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
        self.programs["load_bridge_connected_tile_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool copy_cell_core;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
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
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= cell_grid_size.x ||
                    local_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return;
                }}
                int cell_index = world_cell.y * world_grid_size.x + world_cell.x;
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
        self.programs["load_bridge_connected_tile_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;
            layout(std430, binding=0) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=1) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=2) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_img;
            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= cell_grid_size.x ||
                    local_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return;
                }}
                int cell_index = world_cell.y * world_grid_size.x + world_cell.x;
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
        self.programs["load_bridge_connected_tile_pending"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            layout(std430, binding=0) readonly buffer BridgePendingBuffer {{
                int bridge_pending[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D pending_img;
            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= cell_grid_size.x ||
                    local_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return;
                }}
                int cell_index = world_cell.y * world_grid_size.x + world_cell.x;
                imageStore(pending_img, local_cell, vec4(bridge_pending[cell_index] != 0 ? 1.0 : 0.0, 0.0, 0.0, 0.0));
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
        self.programs["publish_bridge_region_cell_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

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
            layout(std430, binding=4) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=5) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=6) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= cell_grid_size.x ||
                    local_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return;
                }}
                int cell_index = world_cell.y * world_grid_size.x + world_cell.x;
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
        self.programs["publish_bridge_region_pending_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D pending_tex;
            layout(std430, binding=0) writeonly buffer BridgePendingBuffer {{
                int bridge_pending[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= region_size.x ||
                    local_cell.y >= region_size.y
                ) {{
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
        self.programs["publish_bridge_region_mask_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int mode;

            layout(binding=0) uniform sampler2D value_tex;
            layout(binding=1) uniform sampler2D structural_tex;
            layout(std430, binding=0) writeonly buffer BridgeMaskBuffer {{
                int bridge_mask[];
            }};
            layout(std430, binding=1) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= region_size.x ||
                    local_cell.y >= region_size.y
                ) {{
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
        self.programs["publish_bridge_supported_unsupported_masks_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 region_size;
            uniform ivec2 region_origin;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D supported_tex;
            layout(binding=1) uniform sampler2D structural_tex;
            layout(std430, binding=0) writeonly buffer BridgeSupportedMaskBuffer {{
                int bridge_supported_mask[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeUnsupportedMaskBuffer {{
                int bridge_unsupported_mask[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= region_size.x ||
                    local_cell.y >= region_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (
                    world_cell.x < 0 ||
                    world_cell.y < 0 ||
                    world_cell.x >= cell_grid_size.x ||
                    world_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                int cell_index = world_cell.y * cell_grid_size.x + world_cell.x;
                bool supported = texelFetch(supported_tex, local_cell, 0).x > 0.5;
                bool unsupported = texelFetch(structural_tex, local_cell, 0).x > 0.5 && !supported;
                bridge_supported_mask[cell_index] = supported ? 1 : 0;
                bridge_unsupported_mask[cell_index] = unsupported ? 1 : 0;
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
        self.programs["publish_bridge_region_labels_connected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(
                local_size_x={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_y={FORMAL_CONNECTED_TILE_LOCAL_SIZE},
                local_size_z=1
            ) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 region_origin;
            uniform ivec2 world_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(binding=0) uniform sampler2D label_tex;
            layout(std430, binding=0) writeonly buffer BridgeLabelBuffer {{
                int bridge_labels[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeCollapsedMaskBuffer {{
                int bridge_collapsed_mask[];
            }};
            layout(std430, binding=2) readonly buffer ConnectedTileMask {{
                int connected_tiles[];
            }};
            layout(std430, binding=3) readonly buffer ConnectedTileCount {{
                uint connected_tile_count[];
            }};
            layout(std430, binding=4) readonly buffer ConnectedTileList {{
                ivec2 connected_tile_list[];
            }};

            void main() {{
                uint work_index = gl_WorkGroupID.x;
                if (work_index >= connected_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = connected_tile_list[int(work_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (connected_tiles[tile_index] == 0) {{
                    return;
                }}
                ivec2 tile_local = ivec2(gl_LocalInvocationID.xy);
                ivec2 local_cell = tile * tile_size + tile_local;
                if (
                    tile_local.x >= tile_size ||
                    tile_local.y >= tile_size ||
                    local_cell.x >= cell_grid_size.x ||
                    local_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 world_cell = region_origin + local_cell;
                if (world_cell.x < 0 || world_cell.y < 0 || world_cell.x >= world_grid_size.x || world_cell.y >= world_grid_size.y) {{
                    return;
                }}
                int cell_index = world_cell.y * world_grid_size.x + world_cell.x;
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

    # ``_formal_gpu_frame`` inherited from GPUPipelineBase.

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

        world._require_gpu_authoritative_resources(
            "collapse input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )
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

    def _load_authoritative_bridge_connected_tile_inputs(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
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

        self._ensure_programs(bridge.ctx)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected bridge input load requires tile_size <= 32")
        tile_grid_size = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )

        if copy_cell_core:
            program = self.programs["load_bridge_connected_tile_cell"]
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal connected bridge cell input load requires ComputeShader.run_indirect")
            program["cell_grid_size"].value = (int(width), int(height))
            program["region_origin"].value = (int(x0), int(y0))
            program["world_grid_size"].value = (int(world.width), int(world.height))
            program["tile_grid_size"].value = tile_grid_size
            program["tile_size"].value = int(tile_size)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.timer_tex.bind_to_image(3, read=False, write=True)
            resources.integrity_tex.bind_to_image(4, read=False, write=True)
            resources.temp_tex.bind_to_image(5, read=False, write=True)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
            self._sync_compute_writes(bridge.ctx)

        if copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_connected_tile_cell_aux"]
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal connected bridge aux input load requires ComputeShader.run_indirect")
            program["cell_grid_size"].value = (int(width), int(height))
            program["region_origin"].value = (int(x0), int(y0))
            program["world_grid_size"].value = (int(world.width), int(world.height))
            program["tile_grid_size"].value = tile_grid_size
            program["tile_size"].value = int(tile_size)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=0)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=3)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=4)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=5)
            resources.island_id_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_tex.bind_to_image(2, read=False, write=True)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
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

    def _load_authoritative_bridge_connected_tile_pending(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if "collapse_delay_pending" not in bridge.gpu_authoritative_resources:
            return
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending state")
        self._ensure_programs(bridge.ctx)
        tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
            raise RuntimeError("formal connected pending input load requires tile_size <= 32")
        program = self.programs["load_bridge_connected_tile_pending"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected pending input load requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(tile_size)
        bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        resources.phase_tex.bind_to_image(0, read=False, write=True)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
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
        *,
        tile_mask_name: str | None = None,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending output")
        connected_tiles = tile_mask_name is not None
        program = self.programs[
            "publish_bridge_region_pending_connected_tiles" if connected_tiles else "publish_bridge_region_pending"
        ]
        if connected_tiles and not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected pending publish requires ComputeShader.run_indirect")
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        if connected_tiles:
            program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
        pending_texture.use(location=0)
        bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
        if connected_tiles:
            assert tile_mask_name is not None
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
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
        tile_mask_name: str | None = None,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative runtime masks")
        connected_tiles = tile_mask_name is not None
        program = self.programs["publish_bridge_region_mask_connected_tiles" if connected_tiles else "publish_bridge_region_mask"]
        if connected_tiles and not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected mask publish requires ComputeShader.run_indirect")
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["mode"].value = int(mode)
        if connected_tiles:
            program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
        texture.use(location=0)
        resources.structural_tex.use(location=1)
        bridge.buffers[resource_name].bind_to_storage_buffer(binding=0)
        if connected_tiles:
            assert tile_mask_name is not None
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(resource_name)

    def _publish_bridge_supported_unsupported_masks_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        supported_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative runtime masks")
        program = self.programs["publish_bridge_supported_unsupported_masks_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected support mask publish requires ComputeShader.run_indirect")
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
        supported_texture.use(location=0)
        resources.structural_tex.use(location=1)
        bridge.buffers["collapse_supported_mask"].bind_to_storage_buffer(binding=0)
        bridge.buffers["collapse_unsupported_mask"].bind_to_storage_buffer(binding=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("collapse_supported_mask", "collapse_unsupported_mask")

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

    def _publish_bridge_region_labels_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        label_texture: Any,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative component labels")
        program = self.programs["publish_bridge_region_labels_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component label publish requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        label_texture.use(location=0)
        bridge.buffers["collapse_component_label"].bind_to_storage_buffer(binding=0)
        bridge.buffers["collapse_collapsed_cell_mask"].bind_to_storage_buffer(binding=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
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

    def _publish_bridge_region_outputs_connected_tiles(
        self,
        world: "WorldEngine",
        resources: GPUCollapseResources,
        x0: int,
        y0: int,
        width: int,
        height: int,
        tile_mask_name: str,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative output state")
        if bridge.ctx is not world.bridge.ctx:
            raise RuntimeError("GPU collapse pipeline cannot publish authoritative state from a separate GL context")
        if "cell_core" not in bridge.gpu_authoritative_resources:
            world._require_gpu_authoritative_resources("collapse output", "cell_core")
            bridge.sync_world(world)

        program = self.programs["publish_bridge_region_cell_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected output publish requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
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
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    def _barrier_bits(self) -> tuple[str, ...]:
        # collapse uses indirect dispatch in addition to the default
        # image/texture/storage sync.
        return (
            "SHADER_STORAGE_BARRIER_BIT",
            "SHADER_IMAGE_ACCESS_BARRIER_BIT",
            "TEXTURE_FETCH_BARRIER_BIT",
            "COMMAND_BARRIER_BIT",
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
