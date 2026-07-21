from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.sim.gpu_collapse_dirty import (
    clear_collapse_structure_dirty_tile_queue_on_gpu,
    ensure_collapse_structure_dirty_tile_queue,
)

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
    FORMAL_CONNECTED_FRONTIER_BUFFER,
    FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
    FORMAL_CONNECTED_PROCESSED_BUFFER,
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    FORMAL_CONNECTED_TILE_SCRATCH_BUFFER,
    FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_SEED_BUFFER,
    FORMAL_DEFERRED_REGION_REQUEST_BUFFER,
    FORMAL_DEFERRED_REGION_REQUEST_CAPACITY,
    FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER,
    LOCAL_SIZE
)



def prewarm_formal_connected_resources(pipeline, world: "WorldEngine") -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    pipeline._ensure_resources(ctx, int(world.width), int(world.height))
    pipeline._ensure_formal_connected_frontier_buffers_impl(world)
    ctx.finish()


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


def _formal_support_unit_pass_count(width: int, height: int) -> int:
    """Fixed jump=1 cleanup passes after JFA, not a region-diameter flood."""
    return 2


def _formal_label_unit_pass_count(width: int, height: int) -> int:
    """Fixed jump=1 cleanup passes after JFA, not a region-diameter flood."""
    return 2


def _formal_support_refine_round_count(width: int, height: int) -> int:
    """Single bounded cleanup stage after coarse JFA propagation."""
    return 1


def _formal_label_refine_round_count(width: int, height: int) -> int:
    """Single bounded cleanup stage after coarse JFA propagation."""
    return 1


def _run_formal_support_refine_passes(
    pipeline,
    ctx: Any,
    resources: GPUCollapseResources,
    current: Any,
    scratch: Any,
    width: int,
    height: int,
    jumps: tuple[int, ...],
) -> tuple[Any, Any]:
    unit_pass_count = pipeline._formal_support_unit_pass_count(width, height)
    refine_round_count = pipeline._formal_support_refine_round_count(width, height)
    for round_index in range(refine_round_count):
        for _ in range(unit_pass_count):
            current, scratch, _ = pipeline._run_pass(
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
            current, scratch, _ = pipeline._run_pass(
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


def seed_structural_region_texture(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    program = pipeline.programs["seed_structural_region"]
    program["region_size"].value = (int(width), int(height))
    program["seed_rect"].value = (int(seed_x0), int(seed_y0), int(seed_x1), int(seed_y1))
    resources.structural_tex.use(location=0)
    resources.support_ping.bind_to_image(1, read=False, write=True)
    resources.support_pong.bind_to_image(2, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def connected_structural_region_texture(
    pipeline,
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
    pipeline.seed_structural_region_texture(
        world,
        resources,
        width,
        height,
        max(0, int(seed_x0)),
        max(0, int(seed_y0)),
        min(int(width), int(seed_x1)),
        min(int(height), int(seed_y1)),
    )
    connected_texture = pipeline.solve_region_textures(
        world,
        resources,
        width,
        height,
        x0=0,
        y0=0,
        publish_masks=False,
    )
    pipeline.copy_mask_texture(world, resources, connected_texture, resources.cell_flags_out_tex, width, height)
    return resources.cell_flags_out_tex


def copy_mask_texture(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    program = pipeline.programs["copy_mask_texture"]
    program["region_size"].value = (int(width), int(height))
    source_texture.use(location=0)
    target_texture.bind_to_image(1, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _copy_mask_texture_connected_tiles(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["copy_mask_texture_connected_tiles"]
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
    pipeline._sync_compute_writes(ctx)


def _ensure_formal_connected_axis_mask_buffers(
    pipeline,
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
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    source_texture: Any,
    width: int,
    height: int,
    tile_mask_name: str,
    *,
    support_seed_texture: Any | None = None,
    support_seed_u8_texture: Any | None = None,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0:
        return
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected axis masks require tile_size <= 32")
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    pipeline._ensure_formal_connected_axis_mask_buffers(ctx, resources, tile_width * tile_height)
    convert_support_seed = support_seed_u8_texture is not None
    if convert_support_seed != (support_seed_texture is not None):
        raise ValueError("u8 support seed conversion requires both source and destination textures")
    program = pipeline.programs[
        "build_formal_connected_axis_masks_u8"
        if convert_support_seed
        else "build_formal_connected_axis_masks"
    ]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected axis mask build requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["tile_grid_size"].value = (tile_width, tile_height)
    program["tile_size"].value = int(tile_size)
    source_texture.use(location=0)
    if convert_support_seed:
        assert support_seed_texture is not None and support_seed_u8_texture is not None
        support_seed_texture.use(location=1)
        support_seed_u8_texture.bind_to_image(0, read=False, write=True)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
    resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
    resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)


def detect_connected_internal_boundary_flags(
    pipeline,
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

    pipeline._ensure_programs(ctx)
    resources.region_flags.write(np.zeros(1, dtype=np.uint32).tobytes())
    program = pipeline.programs["detect_connected_internal_boundary_flags"]
    program["region_size"].value = (int(width), int(height))
    program["internal_edges"].value = internal_edges
    eligibility_texture.use(location=0)
    resources.region_flags.bind_to_storage_buffer(binding=0)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    return 0


def _ensure_formal_deferred_region_request_buffers(pipeline, world: "WorldEngine") -> tuple[Any, Any, int]:
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


def _ensure_formal_connected_frontier_buffers(pipeline, world: "WorldEngine") -> tuple[str, str]:
    with pipeline._profile_pass(world, "connected_frontier_buffer_prepare"):
        return pipeline._ensure_formal_connected_frontier_buffers_impl(world)


def _invalidate_persistent_dense_tile_worklist(pipeline) -> None:
    if pipeline._persistent_dense_tile_worklist_signature is not None:
        pipeline.persistent_dense_tile_worklist_invalidations += 1
    pipeline._persistent_dense_tile_worklist_signature = None


def _persistent_dense_tile_worklist_signature(
    pipeline,
    world: "WorldEngine",
    width: int,
    height: int,
) -> tuple[int, ...] | None:
    bridge = world.bridge
    ctx = bridge.ctx
    buffer_names = (
        FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )
    if ctx is None or any(name not in bridge.buffers for name in buffer_names):
        return None
    return (
        id(ctx),
        int(world.width),
        int(world.height),
        int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)),
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
        0,
        0,
        int(width),
        int(height),
        *(id(bridge.buffers[name]) for name in buffer_names),
    )


def _ensure_formal_connected_frontier_buffers_impl(pipeline, world: "WorldEngine") -> tuple[str, str]:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected frontier expansion")
    pipeline._ensure_programs(bridge.ctx)
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
    pipeline._clear_formal_connected_tile_mask_buffers(world)
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
    pipeline,
    world: "WorldEngine",
    width: int,
    height: int,
) -> str | None:
    bridge = world.bridge
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected tile worklists")
    if pipeline._persistent_dense_tile_worklist_enabled:
        signature = pipeline._persistent_dense_tile_worklist_signature_for(
            world,
            width,
            height,
        )
        if signature is not None and signature == pipeline._persistent_dense_tile_worklist_signature:
            pipeline.persistent_dense_tile_worklist_hits += 1
            return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER

    pipeline._ensure_formal_connected_frontier_buffers(world)
    bridge = world.bridge
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

    pipeline._invalidate_persistent_dense_tile_worklist()
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
    if pipeline._persistent_dense_tile_worklist_enabled:
        pipeline._persistent_dense_tile_worklist_signature = (
            pipeline._persistent_dense_tile_worklist_signature_for(world, width, height)
        )
        pipeline.persistent_dense_tile_worklist_rebuilds += 1
    return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER


def _clear_formal_connected_cell_buffer_names(pipeline, world: "WorldEngine", buffer_names: tuple[str, ...]) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    bridge = world.bridge
    cell_count = max(1, int(world.width) * int(world.height))
    program = pipeline.programs["clear_formal_connected_cell_buffer"]
    program["cell_count"].value = int(cell_count)
    for name in buffer_names:
        bridge.buffers[name].bind_to_storage_buffer(binding=0)
        program.run((cell_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(*buffer_names)


def _clear_formal_connected_tile_mask_buffers(pipeline, world: "WorldEngine") -> None:
    pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    bridge = world.bridge
    program = pipeline.programs["clear_formal_connected_tile_masks_by_list"]
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
    pipeline._sync_compute_writes(ctx)
    pipeline._clear_formal_connected_tile_worklists(world)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_TILE_SEED_BUFFER,
        FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
        FORMAL_CONNECTED_TILE_SCRATCH_BUFFER,
    )


def _clear_formal_connected_tile_worklist(
    pipeline,
    world: "WorldEngine",
    count_name: str,
    dispatch_args_name: str,
) -> None:
    if (
        count_name == FORMAL_CONNECTED_TILE_COUNT_BUFFER
        or dispatch_args_name == FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER
    ):
        pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    bridge = world.bridge
    program = pipeline.programs["clear_formal_connected_tile_worklist"]
    bridge.buffers[count_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=1)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(count_name, dispatch_args_name)


def _clear_formal_connected_tile_worklists(pipeline, world: "WorldEngine") -> None:
    pipeline._clear_formal_connected_tile_worklist(
        world,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )
    pipeline._clear_formal_connected_tile_worklist(
        world,
        FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
    )
    pipeline._clear_formal_connected_tile_worklist(
        world,
        FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
    )


def _clear_formal_connected_cell_buffer_connected_tiles(
    pipeline,
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
    program = pipeline.programs["clear_formal_connected_cell_buffer_connected_tiles"]
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(buffer_name)


def reset_formal_connected_frontier(
    pipeline,
    world: "WorldEngine",
    seed_rect: tuple[int, int, int, int],
) -> tuple[str, str]:
    current_name, scratch_name = pipeline._ensure_formal_connected_frontier_buffers(world)
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    program = pipeline.programs["seed_formal_connected_frontier_rect"]
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["seed_rect"].value = tuple(int(value) for value in seed_rect)
    world.bridge.buffers[current_name].bind_to_storage_buffer(binding=0)
    group_x = (int(world.width) + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (int(world.height) + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    world.bridge.mark_gpu_authoritative(current_name, scratch_name)
    return current_name, scratch_name


def clear_formal_connected_frontier_buffer(pipeline, world: "WorldEngine", buffer_name: str) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for connected frontier expansion")
    if buffer_name not in bridge.buffers:
        pipeline._ensure_formal_connected_frontier_buffers(world)
    cell_count = max(1, int(world.width) * int(world.height))
    pipeline._ensure_programs(bridge.ctx)
    program = pipeline.programs["clear_formal_connected_cell_buffer"]
    program["cell_count"].value = int(cell_count)
    bridge.buffers[buffer_name].bind_to_storage_buffer(binding=0)
    program.run((cell_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(buffer_name)


def clear_formal_deferred_region_requests(pipeline, world: "WorldEngine") -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    count_buffer = bridge.buffers.get(FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER)
    if count_buffer is None:
        return
    count_buffer.write(np.zeros(1, dtype=np.uint32).tobytes())
    bridge.mark_gpu_authoritative(FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER)


def execute_formal_connected_expansion(
    pipeline,
    world: "WorldEngine",
    seed_rect: tuple[int, int, int, int],
    *,
    resource_region: tuple[int, int, int, int] | None = None,
) -> int:
    if not pipeline._formal_gpu_frame(world):
        raise RuntimeError("formal connected collapse requires an active formal GPU frame")
    width = int(world.width)
    height = int(world.height)
    if width <= 0 or height <= 0:
        raise ValueError("formal connected collapse requires a non-empty world")
    pipeline._ensure_formal_connected_frontier_buffers(world)
    if resource_region is None:
        resource_region = (0, 0, width, height)
    outcome_resources, outcome_x0, outcome_y0, outcome_width, outcome_height = pipeline._solve_formal_connected_tile_textures(
        world,
        seed_rect,
        resource_region=resource_region,
    )
    return pipeline.materialize_component_texture_formal(
        world,
        outcome_resources.phase_out_tex,
        outcome_width,
        outcome_height,
        outcome_x0,
        outcome_y0,
        tile_mask_name=pipeline._last_formal_connected_tile_mask_name,
    )


def execute_formal_connected_dirty_tile_queue(pipeline, world: "WorldEngine") -> int:
    if not pipeline._formal_gpu_frame(world):
        raise RuntimeError("formal dirty tile collapse requires an active formal GPU frame")
    width = int(world.width)
    height = int(world.height)
    if width <= 0 or height <= 0:
        raise ValueError("formal dirty tile collapse requires a non-empty world")
    ensure_collapse_structure_dirty_tile_queue(world)
    pipeline._ensure_formal_connected_frontier_buffers(world)
    outcome_resources, outcome_x0, outcome_y0, outcome_width, outcome_height = (
        pipeline._solve_formal_connected_dirty_tile_textures(world)
    )
    component_capacity = pipeline.materialize_component_texture_formal(
        world,
        outcome_resources.phase_out_tex,
        outcome_width,
        outcome_height,
        outcome_x0,
        outcome_y0,
        tile_mask_name=pipeline._last_formal_connected_tile_mask_name,
    )
    clear_collapse_structure_dirty_tile_queue_on_gpu(world)
    return component_capacity
