from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    LOCAL_SIZE
)



def _label_component_texture(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    if pipeline._formal_gpu_frame(world) and tile_mask_name is not None:
        return pipeline._label_component_texture_connected_tiles(
            world,
            resources,
            component_texture,
            width,
            height,
            x0=x0,
            y0=y0,
            tile_mask_name=tile_mask_name,
        )
    if pipeline._formal_gpu_frame(world):
        region_tile_mask_name = pipeline._seed_formal_texture_region_tile_worklist(world, width, height)
        if region_tile_mask_name is not None:
            return pipeline._label_component_texture_connected_tiles_from_texture_init(
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

    init_program = pipeline.programs["component_label_init"]
    init_program["region_size"].value = (width, height)
    component_texture.use(location=0)
    resources.support_ping.bind_to_image(1, read=False, write=True)
    init_program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    current = resources.support_ping
    scratch = resources.support_pong
    propagate = pipeline.programs["component_label_propagate"]
    propagate["region_size"].value = (width, height)
    if pipeline._formal_gpu_frame(world):
        jumps = pipeline._formal_jfa_jumps(width, height)
        for jump in jumps:
            propagate["jump"].value = int(jump)
            resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
            current.use(location=0)
            scratch.bind_to_image(1, read=False, write=True)
            resources.change_flag.bind_to_storage_buffer(binding=0)
            propagate.run(group_x, group_y, 1)
            pipeline._sync_compute_writes(ctx)
            current, scratch = scratch, current
        current, scratch = pipeline._run_formal_label_refine_passes(
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
        pipeline._publish_bridge_region_labels(world, resources, current, x0, y0, width, height)
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
    pipeline,
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
    with pipeline._profile_pass(world, "label_jfa.texture_init"):
        init_program = pipeline.programs["component_label_init"]
        init_program["region_size"].value = (width, height)
        component_texture.use(location=0)
        resources.support_ping.bind_to_image(1, read=False, write=True)
        init_program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    current = resources.support_ping
    scratch = resources.support_pong
    with pipeline._profile_pass(world, "label_jfa.axis_masks"):
        pipeline._build_formal_connected_axis_masks(
            world,
            resources,
            component_texture,
            width,
            height,
            tile_mask_name,
        )
    jumps = pipeline._formal_jfa_jumps(width, height)
    with pipeline._profile_pass(world, "label_jfa.jfa"):
        for band_name, band_jumps in pipeline._formal_jfa_profile_jump_bands(jumps):
            with pipeline._profile_pass(world, f"label_jfa.jfa.{band_name}"):
                for jump in band_jumps:
                    current, scratch = pipeline._run_formal_connected_component_label_pass(
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
    with pipeline._profile_pass(world, "label_jfa.refine"):
        current, scratch = pipeline._run_formal_connected_component_label_refine_passes(
            world,
            resources,
            component_texture,
            current,
            scratch,
            width,
            height,
            tile_mask_name,
        )
    with pipeline._profile_pass(world, "label_jfa.publish"):
        pipeline._publish_bridge_region_labels_connected_tiles(
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
    pipeline,
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
    current, scratch, schedule = _begin_formal_connected_component_labeling(
        pipeline,
        world,
        resources,
        component_texture,
        width,
        height,
        tile_mask_name,
    )
    current, scratch = _run_formal_connected_component_label_slice(
        pipeline,
        world,
        resources,
        component_texture,
        current,
        scratch,
        width,
        height,
        tile_mask_name,
        schedule,
        0,
        len(schedule),
    )
    _publish_formal_connected_component_labels(
        pipeline,
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


def _begin_formal_connected_component_labeling(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    component_texture: Any,
    width: int,
    height: int,
    tile_mask_name: str,
) -> tuple[Any, Any, tuple[int, ...]]:
    if pipeline._label_seed_materialize_axis_fusion_enabled:
        with pipeline._profile_pass(world, "label_jfa.seed_materialize_axis"):
            pipeline._seed_formal_component_labels_and_axis_masks(
                world,
                resources,
                component_texture,
                resources.support_ping,
                width,
                height,
                tile_mask_name,
            )
    else:
        seed_frontier = (
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
            FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
        )
        with pipeline._profile_pass(world, "label_jfa.seed"):
            pipeline._seed_formal_component_label_frontier(
                world,
                resources,
                component_texture,
                width,
                height,
                tile_mask_name,
                seed_frontier,
            )
        with pipeline._profile_pass(world, "label_jfa.materialize"):
            pipeline._copy_formal_component_label_buffer_to_texture(
                world,
                resources,
                component_texture,
                resources.support_ping,
                width,
                height,
                tile_mask_name,
            )
        with pipeline._profile_pass(world, "label_jfa.axis_masks"):
            pipeline._build_formal_connected_axis_masks(
                world,
                resources,
                component_texture,
                width,
                height,
                tile_mask_name,
            )
    current = resources.support_ping
    scratch = resources.support_pong
    schedule = (
        *pipeline._formal_jfa_jumps(width, height),
        *([1] * pipeline._formal_connected_tile_refine_pass_count(world)),
    )
    return current, scratch, tuple(int(jump) for jump in schedule)


def _run_formal_connected_component_label_slice(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    component_texture: Any,
    current: Any,
    scratch: Any,
    width: int,
    height: int,
    tile_mask_name: str,
    schedule: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[Any, Any]:
    schedule_length = len(schedule)
    slice_start = max(0, min(schedule_length, int(start)))
    slice_stop = max(slice_start, min(schedule_length, int(stop)))
    refine_pass_count = min(
        schedule_length,
        max(0, int(pipeline._formal_connected_tile_refine_pass_count(world))),
    )
    jfa_stop = schedule_length - refine_pass_count
    jfa_slice_start = min(slice_start, jfa_stop)
    jfa_slice_stop = min(slice_stop, jfa_stop)
    if jfa_slice_start < jfa_slice_stop:
        jumps = schedule[jfa_slice_start:jfa_slice_stop]
        with pipeline._profile_pass(world, "label_jfa.jfa"):
            for band_name, band_jumps in pipeline._formal_jfa_profile_jump_bands(jumps):
                with pipeline._profile_pass(world, f"label_jfa.jfa.{band_name}"):
                    for jump in band_jumps:
                        current, scratch = pipeline._run_formal_connected_component_label_pass(
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
    refine_slice_start = max(slice_start, jfa_stop)
    if refine_slice_start < slice_stop:
        with pipeline._profile_pass(world, "label_jfa.refine"):
            for jump in schedule[refine_slice_start:slice_stop]:
                current, scratch = pipeline._run_formal_connected_component_label_pass(
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
    return current, scratch


def _publish_formal_connected_component_labels(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    current: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    with pipeline._profile_pass(world, "label_jfa.publish"):
        pipeline._publish_bridge_region_labels_connected_tiles(
            world,
            resources,
            current,
            x0,
            y0,
            width,
            height,
            tile_mask_name,
        )


def _ensure_formal_connected_component_label_union_buffers(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    width: int,
    height: int,
) -> int:
    from oracle_game.sim.gpu_collapse_frontier import _ensure_support_tile_union_buffers

    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal component label tile union requires a valid GL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_width = max(1, int(world.active.tile_width))
    tile_height = max(1, int(world.active.tile_height))
    tile_size = max(1, int(world.active.tile_size))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal component label tile union requires tile_size <= 32")
    edge_capacity = max(
        1,
        max(0, tile_width - 1) * int(height)
        + max(0, tile_height - 1) * int(width),
    )
    _ensure_support_tile_union_buffers(
        pipeline,
        ctx,
        resources,
        width,
        height,
        edge_capacity,
    )
    roots = resources.support_tile_union_roots
    parents = resources.support_tile_union_parent
    edges = resources.support_tile_union_edges
    edge_count = resources.support_tile_union_edge_count
    if roots is None or parents is None or edges is None or edge_count is None:
        raise RuntimeError("formal component label tile union buffers were not allocated")
    return edge_capacity


def _begin_formal_connected_component_label_union(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    component_texture: Any,
    width: int,
    height: int,
    tile_mask_name: str,
    *,
    local_components_ready: bool = False,
) -> tuple[int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal component label tile union requires a valid GL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_width = max(1, int(world.active.tile_width))
    tile_height = max(1, int(world.active.tile_height))
    tile_size = max(1, int(world.active.tile_size))
    edge_capacity = pipeline._ensure_formal_connected_component_label_union_buffers(
        world,
        resources,
        width,
        height,
    )
    roots = resources.support_tile_union_roots
    parents = resources.support_tile_union_parent
    edges = resources.support_tile_union_edges
    edge_count = resources.support_tile_union_edge_count
    if roots is None or parents is None or edges is None or edge_count is None:
        raise RuntimeError("formal component label tile union buffers were not allocated")
    edge_count.write(np.zeros(1, dtype=np.uint32).tobytes())

    if not local_components_ready:
        with pipeline._profile_pass(world, "label_tile_union.local_components"):
            program = pipeline.programs["label_tile_union_local"]
            program["cell_grid_size"].value = (int(width), int(height))
            program["tile_grid_size"].value = (tile_width, tile_height)
            program["tile_size"].value = tile_size
            component_texture.use(location=0)
            bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
            bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
            bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
            roots.bind_to_storage_buffer(binding=3)
            parents.bind_to_storage_buffer(binding=4)
            program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
            pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "label_tile_union.boundary_edges"):
        program = pipeline.programs["support_tile_union_edges"]
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = tile_size
        program["edge_capacity"].value = edge_capacity
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        roots.bind_to_storage_buffer(binding=3)
        edges.bind_to_storage_buffer(binding=4)
        edge_count.bind_to_storage_buffer(binding=5)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(ctx)

    edge_groups = max(1, (edge_capacity + 63) // 64)
    resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
    resources.component_dispatch_args.write(
        np.asarray([edge_groups, 1, 1], dtype=np.uint32).tobytes()
    )
    # The fixed upper bound is retained for exact worst-case connectivity.
    # Adaptive dispatch only zeros later rounds after convergence is proven.
    union_round_count = max(1, (max(1, int(width) * int(height))).bit_length() + 2)
    return union_round_count, edge_capacity


def _run_formal_connected_component_label_union_slice(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    edge_capacity: int,
    start: int,
    stop: int,
) -> int:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal component label tile union requires a valid GL context")
    roots = resources.support_tile_union_roots
    parents = resources.support_tile_union_parent
    edges = resources.support_tile_union_edges
    edge_count = resources.support_tile_union_edge_count
    if roots is None or parents is None or edges is None or edge_count is None:
        raise RuntimeError("formal component label tile union buffers were not allocated")
    slice_start = max(0, int(start))
    slice_stop = max(slice_start, int(stop))
    edge_groups = max(1, (int(edge_capacity) + 63) // 64)
    with pipeline._profile_pass(world, "label_tile_union.union"):
        hook = pipeline.programs["label_tile_union_hook"]
        hook["edge_capacity"].value = int(edge_capacity)
        shortcut = pipeline.programs["support_tile_union_shortcut"]
        shortcut["edge_capacity"].value = int(edge_capacity)
        build_dispatch = pipeline.programs["label_tile_union_build_dispatch"]
        build_dispatch["edge_group_count"].value = edge_groups
        for _ in range(slice_start, slice_stop):
            edges.bind_to_storage_buffer(binding=0)
            edge_count.bind_to_storage_buffer(binding=1)
            parents.bind_to_storage_buffer(binding=2)
            resources.change_flag.bind_to_storage_buffer(binding=3)
            hook.run_indirect(resources.component_dispatch_args)
            pipeline._sync_compute_writes(ctx)
            edges.bind_to_storage_buffer(binding=0)
            edge_count.bind_to_storage_buffer(binding=1)
            parents.bind_to_storage_buffer(binding=2)
            shortcut.run_indirect(resources.component_dispatch_args)
            pipeline._sync_compute_writes(ctx)
            resources.change_flag.bind_to_storage_buffer(binding=0)
            resources.component_dispatch_args.bind_to_storage_buffer(binding=1)
            build_dispatch.run(1, 1, 1)
            pipeline._sync_compute_writes(ctx)
    return slice_stop


def _materialize_formal_connected_component_label_union(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    width: int,
    height: int,
    tile_mask_name: str,
) -> tuple[Any, Any]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal component label tile union requires a valid GL context")
    bridge = world.bridge
    roots = resources.support_tile_union_roots
    parents = resources.support_tile_union_parent
    if roots is None or parents is None:
        raise RuntimeError("formal component label tile union buffers were not allocated")
    tile_width = max(1, int(world.active.tile_width))
    tile_height = max(1, int(world.active.tile_height))
    tile_size = max(1, int(world.active.tile_size))
    with pipeline._profile_pass(world, "label_tile_union.materialize"):
        program = pipeline.programs["label_tile_union_materialize"]
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = tile_size
        resources.support_ping.bind_to_image(0, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        roots.bind_to_storage_buffer(binding=3)
        parents.bind_to_storage_buffer(binding=4)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(ctx)
    return resources.support_ping, resources.support_pong


def _seed_formal_component_labels_and_axis_masks(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected component labeling requires tile_size <= 32")
    program = pipeline.programs["seed_formal_component_labels_and_axis_masks"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected fused component label seed requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(tile_size)
    component_texture.use(location=0)
    target_texture.bind_to_image(1, read=False, write=True)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
    resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
    resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)


def _seed_formal_component_label_frontier(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    pipeline._clear_formal_connected_cell_frontier_tiles(world, frontier)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected component labeling requires tile_size <= 32")
    program = pipeline.programs["seed_formal_component_label_frontier"]
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_FRONTIER_BUFFER,
        flags_name,
        list_name,
        count_name,
        dispatch_args_name,
    )


def _expand_formal_component_label_frontier(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected component labeling requires tile_size <= 32")
    program = pipeline.programs["expand_formal_component_label_frontier"]
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected component labeling requires tile_size <= 32")
    program = pipeline.programs["propagate_formal_connected_component_labels"]
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
    current.use(location=1)
    scratch.bind_to_image(2, read=False, write=True)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
    resources.connected_tile_row_masks.bind_to_storage_buffer(binding=3)
    resources.connected_tile_column_masks.bind_to_storage_buffer(binding=4)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        tile_mask_name,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )
    return scratch, current


def _run_formal_connected_component_label_refine_passes(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    component_texture: Any,
    current: Any,
    scratch: Any,
    width: int,
    height: int,
    tile_mask_name: str,
) -> tuple[Any, Any]:
    for _ in range(pipeline._formal_connected_tile_refine_pass_count(world)):
        current, scratch = pipeline._run_formal_connected_component_label_pass(
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
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    program = pipeline.programs["copy_formal_component_label_buffer_to_texture"]
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
    pipeline._sync_compute_writes(ctx)


def _run_formal_label_refine_passes(
    pipeline,
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
    propagate = pipeline.programs["component_label_propagate"]
    unit_pass_count = pipeline._formal_label_unit_pass_count(width, height)
    refine_round_count = pipeline._formal_label_refine_round_count(width, height)
    for round_index in range(refine_round_count):
        for _ in range(unit_pass_count):
            propagate["jump"].value = 1
            resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
            current.use(location=0)
            scratch.bind_to_image(1, read=False, write=True)
            resources.change_flag.bind_to_storage_buffer(binding=0)
            propagate.run(group_x, group_y, 1)
            pipeline._sync_compute_writes(ctx)
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
            pipeline._sync_compute_writes(ctx)
            current, scratch = scratch, current
    return current, scratch


def materialize_labeled_component_texture(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    pipeline._upload_region_state(world, resources, x0, y0, width, height)
    collapse_generation, base_integrity, spawn_temperature = pipeline._materialize_material_params(world)
    pipeline._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
    pipeline._write_dynamic_buffer(ctx, resources, "component_island_ids", component_island_ids.astype(np.int32, copy=False))
    pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
    pipeline._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
    pipeline._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

    program = pipeline.programs["materialize_components"]
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
    pipeline._sync_compute_writes(ctx)
    aux_program = pipeline.programs["materialize_components_aux"]
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
    pipeline._sync_compute_writes(ctx)
    pipeline._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_region_state(world, resources, x0, y0, width, height)
