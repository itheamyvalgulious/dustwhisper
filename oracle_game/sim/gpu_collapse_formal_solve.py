from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.sim.gpu_collapse_dirty import get_collapse_structure_dirty_tile_bounds
from oracle_game.types import Phase

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER,
    FORMAL_CONNECTED_DIRTY_JUMP_ROUNDS,
    FORMAL_CONNECTED_FRONTIER_BUFFER,
    FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT,
    FORMAL_DEFERRED_REGION_REQUEST_BUFFER,
    FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER,
    LOCAL_SIZE
)



def solve_formal_connected_region_textures(
    pipeline,
    world: "WorldEngine",
    seed_rect: tuple[int, int, int, int],
) -> tuple[GPUCollapseResources, int, int, int, int]:
    return pipeline._solve_formal_connected_tile_textures(world, seed_rect)


def _solve_formal_connected_dirty_tile_textures(
    pipeline,
    world: "WorldEngine",
) -> tuple[GPUCollapseResources, int, int, int, int]:
    resource_region = (0, 0, int(world.width), int(world.height))
    resources, x0, y0, width, height = pipeline._prepare_formal_connected_tile_resources_without_input_upload(
        world,
        resource_region,
    )
    with pipeline._profile_pass(world, "tile_region_worklist"):
        tile_mask_name = pipeline._seed_formal_texture_region_tile_worklist(world, width, height)
        if tile_mask_name is None:
            raise RuntimeError("formal dirty tile collapse requires a non-empty connected tile worklist")
        pipeline._last_formal_connected_tile_mask_name = tile_mask_name
    with pipeline._profile_pass(world, "connected_bridge_input_load"):
        pipeline._load_authoritative_bridge_connected_tile_inputs(
            world,
            resources,
            x0,
            y0,
            width,
            height,
            tile_mask_name,
        )
    with pipeline._profile_pass(world, "classify_filter"):
        pipeline._classify_formal_connected_tile_textures(world, resources, tile_mask_name, x0, y0, width, height)
        pipeline._publish_bridge_region_mask(
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
        pipeline._publish_bridge_region_mask(
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

    with pipeline._profile_pass(world, "support_jfa"):
        supported_texture = pipeline._solve_formal_connected_tile_support_textures(
            world,
            resources,
            x0,
            y0,
            width,
            height,
            tile_mask_name,
        )
    outcome_resources, _, _ = pipeline.resolve_supported_outcome_textures(
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
    pipeline,
    world: "WorldEngine",
    seed_rect: tuple[int, int, int, int],
    *,
    resource_region: tuple[int, int, int, int] | None = None,
) -> tuple[GPUCollapseResources, int, int, int, int]:
    resources, x0, y0, width, height = pipeline._prepare_formal_connected_tile_resources(world, resource_region)
    with pipeline._profile_pass(world, "tile_region_worklist"):
        tile_mask_name = pipeline._seed_formal_texture_region_tile_worklist(world, width, height)
        if tile_mask_name is None:
            raise RuntimeError("formal connected collapse requires a non-empty connected tile worklist")
        pipeline._last_formal_connected_tile_mask_name = tile_mask_name
    with pipeline._profile_pass(world, "classify_filter"):
        pipeline._classify_formal_connected_tile_textures(world, resources, tile_mask_name, x0, y0, width, height)
        pipeline._publish_bridge_region_mask(
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
        pipeline._publish_bridge_region_mask(
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

    with pipeline._profile_pass(world, "support_jfa"):
        supported_texture = pipeline._solve_formal_connected_tile_support_textures(
            world,
            resources,
            x0,
            y0,
            width,
            height,
            tile_mask_name,
        )
    outcome_resources, _, _ = pipeline.resolve_supported_outcome_textures(
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
    pipeline,
    world: "WorldEngine",
    region: tuple[int, int, int, int] | None = None,
) -> tuple[GPUCollapseResources, int, int, int, int]:
    with pipeline._profile_pass(world, "tile_resource_prepare"):
        return pipeline._prepare_formal_connected_tile_resources_impl(world, region)


def _prepare_formal_connected_tile_resources_without_input_upload(
    pipeline,
    world: "WorldEngine",
    region: tuple[int, int, int, int] | None = None,
) -> tuple[GPUCollapseResources, int, int, int, int]:
    with pipeline._profile_pass(world, "tile_resource_prepare"):
        return pipeline._prepare_formal_connected_tile_resources_impl(
            world,
            region,
            upload_region_state=False,
        )


def _prepare_formal_connected_tile_resources_impl(
    pipeline,
    world: "WorldEngine",
    region: tuple[int, int, int, int] | None = None,
    *,
    upload_region_state: bool = True,
) -> tuple[GPUCollapseResources, int, int, int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    x0, y0, x1, y1 = pipeline._clamp_formal_connected_region(world, region)
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("formal connected world classification requires a non-empty world")
    with pipeline._profile_pass(world, "tile_resource_prepare.ensure_programs"):
        pipeline._ensure_programs(ctx)
    with pipeline._profile_pass(world, "tile_resource_prepare.ensure_resources"):
        resources = pipeline._ensure_resources(ctx, width, height)
    if upload_region_state:
        with pipeline._profile_pass(world, "tile_resource_prepare.upload_region_state"):
            pipeline._upload_region_state(world, resources, x0, y0, width, height)
    with pipeline._profile_pass(world, "tile_resource_prepare.material_params"):
        structural_params, support_params, behavior_params = pipeline._classification_material_params(world)
    with pipeline._profile_pass(world, "tile_resource_prepare.material_buffer_writes"):
        pipeline._write_dynamic_buffer(ctx, resources, "material_structural", structural_params)
        pipeline._write_dynamic_buffer(ctx, resources, "material_support_anchor", support_params)
        pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_behavior", behavior_params)
    return resources, x0, y0, width, height


def _clamp_formal_connected_region(
    pipeline,
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
    pipeline,
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
    return pipeline._formal_connected_dirty_tile_resource_region_from_tile_bounds(
        world,
        tile_x0,
        tile_y0,
        tile_x1,
        tile_y1,
    )


def _formal_connected_dirty_tile_resource_region_from_tile_bounds(
    pipeline,
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
    pipeline,
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
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    _, _, behavior_params = pipeline._classification_material_params(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected tile classification requires tile_size <= 32")
    program = pipeline.programs["classify_formal_connected_tiles"]
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
    pipeline,
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
    with pipeline._profile_pass(world, "cell_frontier_seed_list_build"):
        pipeline._seed_formal_connected_cell_frontier(world, resources, seed_rect, width, height, tile_mask_name)
    cell_frontier = (
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    )
    current_name = FORMAL_CONNECTED_FRONTIER_BUFFER
    scratch_name = FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER
    with pipeline._profile_pass(world, "cell_frontier_loop"):
        for jump in pipeline._formal_connected_cell_jump_schedule(width, height):
            pipeline._clear_formal_connected_cell_frontier_tiles(world, scratch_frontier)
            pipeline._expand_formal_connected_cell_frontier(
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
                jump_generation=pipeline._next_formal_connected_cell_frontier_generation(),
            )
            cell_frontier, scratch_frontier = scratch_frontier, cell_frontier
    with pipeline._profile_pass(world, "cell_frontier_final_copy"):
        pipeline._copy_formal_connected_buffer_to_texture(
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
    pipeline,
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
    with pipeline._profile_pass(world, "cell_frontier_seed_list_build"):
        pipeline._seed_formal_connected_cell_frontier_from_dirty_queue(
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
    with pipeline._profile_pass(world, "cell_frontier_loop"):
        for jump in pipeline._formal_connected_dirty_jump_schedule(width, height):
            pipeline._clear_formal_connected_cell_frontier_tiles(world, scratch_frontier)
            pipeline._expand_formal_connected_cell_frontier(
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
                jump_generation=pipeline._next_formal_connected_cell_frontier_generation(),
            )
            cell_frontier, scratch_frontier = scratch_frontier, cell_frontier
    with pipeline._profile_pass(world, "cell_frontier_final_copy"):
        pipeline._copy_formal_connected_buffer_to_texture(
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


def _formal_connected_expansion_pass_count(pipeline, world: "WorldEngine") -> int:
    return len(pipeline._formal_connected_tile_jump_schedule(world))


def _formal_connected_tile_jump_schedule(pipeline, world: "WorldEngine") -> tuple[int, ...]:
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    return pipeline._formal_connected_cell_jump_schedule(tile_width, tile_height)


def _formal_connected_tile_refine_pass_count(pipeline, world: "WorldEngine") -> int:
    return FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT


def _formal_connected_tile_support_frontier_pass_count(pipeline, world: "WorldEngine") -> int:
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    return max(1, tile_width + tile_height)


def _formal_connected_component_label_frontier_pass_count(pipeline, world: "WorldEngine") -> int:
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    return max(1, tile_width + tile_height)


def _formal_connected_dirty_tile_jump_schedule(pipeline, world: "WorldEngine") -> tuple[int, ...]:
    return pipeline._formal_connected_tile_jump_schedule(world)


def _formal_connected_dirty_jump_schedule(width: int, height: int) -> tuple[int, ...]:
    return GPUCollapsePipeline._formal_connected_cell_jump_schedule(width, height)


def _formal_connected_cell_jump_schedule(width: int, height: int) -> tuple[int, ...]:
    jumps = GPUCollapsePipeline._formal_jfa_jumps(width, height)
    cleanup = (1,) * FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT
    round_schedule = jumps + cleanup
    if not round_schedule:
        return cleanup or (1,)
    rounds = min(FORMAL_CONNECTED_DIRTY_JUMP_ROUNDS, max(1, len(jumps)))
    return tuple(jump for _ in range(rounds) for jump in round_schedule)


def _next_formal_connected_cell_frontier_generation(pipeline) -> int:
    pipeline._formal_connected_cell_frontier_generation += 1
    return pipeline._formal_connected_cell_frontier_generation


def _filter_formal_connected_eligibility(
    pipeline,
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
        pipeline._ensure_formal_connected_frontier_buffers(world)
    pipeline._ensure_programs(ctx)
    if tile_mask_name is not None:
        pipeline._clear_formal_connected_cell_buffer_connected_tiles(
            world,
            processed_buffer_name,
            width,
            height,
            tile_mask_name,
        )
    else:
        pipeline._clear_formal_connected_cell_buffer_names(world, (processed_buffer_name,))
    connected_tiles = tile_mask_name is not None
    program = pipeline.programs[
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(processed_buffer_name)
    if tile_mask_name is not None:
        pipeline._copy_mask_texture_connected_tiles(
            world,
            resources,
            resources.integrity_out_tex,
            resources.structural_tex,
            width,
            height,
            tile_mask_name,
        )
        pipeline._copy_mask_texture_connected_tiles(
            world,
            resources,
            resources.support_pong,
            resources.support_ping,
            width,
            height,
            tile_mask_name,
        )
    else:
        pipeline.copy_mask_texture(world, resources, resources.integrity_out_tex, resources.structural_tex, width, height)
        pipeline.copy_mask_texture(world, resources, resources.support_pong, resources.support_ping, width, height)


def connected_structural_frontier_texture(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    program = pipeline.programs["seed_structural_frontier_region"]
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
    pipeline._sync_compute_writes(ctx)
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


def drain_formal_deferred_region_requests(pipeline, world: "WorldEngine") -> list[tuple[int, int, int, int]]:
    """Legacy compatibility hook; formal connected expansion is GPU-frontier driven."""
    return []


def enqueue_connected_internal_boundary_deferred_regions(
    pipeline,
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

    pipeline._ensure_programs(ctx)
    request_count, request_buffer, request_capacity = pipeline._ensure_formal_deferred_region_request_buffers(world)
    resources.region_flags.write(np.zeros(1, dtype=np.uint32).tobytes())
    program = pipeline.programs["enqueue_connected_internal_boundary_deferred_regions"]
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
    pipeline._sync_compute_writes(ctx)
    world.bridge.mark_gpu_authoritative(
        FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER,
        FORMAL_DEFERRED_REGION_REQUEST_BUFFER,
    )


def exclude_internal_boundary_connected_texture_to_frontier(
    pipeline,
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
    pipeline.copy_mask_texture(world, resources, eligibility_texture, resources.structural_tex, width, height)
    pipeline._ensure_programs(ctx)
    seed_program = pipeline.programs["seed_internal_boundary_region"]
    seed_program["region_size"].value = (int(width), int(height))
    seed_program["internal_edges"].value = internal_edges
    resources.structural_tex.use(location=0)
    resources.support_ping.bind_to_image(1, read=False, write=True)
    resources.support_pong.bind_to_image(2, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    seed_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)

    boundary_connected_texture = pipeline.solve_region_textures(
        world,
        resources,
        width,
        height,
        x0=0,
        y0=0,
        publish_masks=False,
    )
    publish_program = pipeline.programs["publish_internal_boundary_frontier"]
    publish_program["region_size"].value = (int(width), int(height))
    publish_program["region_origin"].value = (int(x0), int(y0))
    publish_program["cell_grid_size"].value = (int(world.width), int(world.height))
    publish_program["internal_edges"].value = internal_edges
    boundary_connected_texture.use(location=0)
    bridge.buffers[frontier_buffer_name].bind_to_storage_buffer(binding=0)
    publish_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(frontier_buffer_name)

    exclude_program = pipeline.programs["exclude_boundary_connected_mask"]
    exclude_program["region_size"].value = (int(width), int(height))
    resources.structural_tex.use(location=0)
    boundary_connected_texture.use(location=1)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    exclude_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    return resources.cell_flags_out_tex


def exclude_internal_boundary_connected_texture(
    pipeline,
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

    pipeline.copy_mask_texture(world, resources, eligibility_texture, resources.structural_tex, width, height)
    pipeline._ensure_programs(ctx)
    seed_program = pipeline.programs["seed_internal_boundary_region"]
    seed_program["region_size"].value = (int(width), int(height))
    seed_program["internal_edges"].value = internal_edges
    resources.structural_tex.use(location=0)
    resources.support_ping.bind_to_image(1, read=False, write=True)
    resources.support_pong.bind_to_image(2, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    seed_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)

    boundary_connected_texture = pipeline.solve_region_textures(
        world,
        resources,
        width,
        height,
        x0=0,
        y0=0,
        publish_masks=False,
    )
    exclude_program = pipeline.programs["exclude_boundary_connected_mask"]
    exclude_program["region_size"].value = (int(width), int(height))
    resources.structural_tex.use(location=0)
    boundary_connected_texture.use(location=1)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    exclude_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    return resources.cell_flags_out_tex
