from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.sim.gpu_collapse_dirty import ensure_collapse_structure_dirty_tile_queue
from oracle_game.types import Phase

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
    FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_FRONTIER_BUFFER,
    FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER
)



def _solve_formal_connected_tile_frontier(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    seed_rect: tuple[int, int, int, int],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    pipeline._seed_formal_connected_tile_frontier(world, resources, seed_rect, x0, y0, width, height)
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
    for jump in pipeline._formal_connected_tile_jump_schedule(world):
        pipeline._clear_formal_connected_tile_worklist(world, scratch_frontier[1], scratch_frontier[2])
        pipeline._expand_formal_connected_tile_frontier(
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
    pipeline._last_formal_connected_tile_mask_name = FORMAL_CONNECTED_TILE_FRONTIER_BUFFER
    return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER


def _solve_formal_connected_dirty_tile_frontier(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    pipeline._seed_formal_connected_tile_frontier_from_dirty_queue(world, resources, x0, y0, width, height)
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
    for jump in pipeline._formal_connected_dirty_tile_jump_schedule(world):
        pipeline._clear_formal_connected_tile_worklist(world, scratch_frontier[1], scratch_frontier[2])
        pipeline._expand_formal_connected_tile_frontier(
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
    pipeline._last_formal_connected_tile_mask_name = FORMAL_CONNECTED_TILE_FRONTIER_BUFFER
    return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER


def _compact_formal_connected_tile_mask(pipeline, world: "WorldEngine", tile_mask_name: str) -> None:
    pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if tile_mask_name not in bridge.buffers:
        raise RuntimeError("formal connected tile mask is not allocated")
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    tile_count = tile_width * tile_height

    clear_program = pipeline.programs["clear_formal_connected_tile_worklist"]
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=0)
    bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    compact_program = pipeline.programs["compact_formal_connected_tile_mask"]
    compact_program["tile_grid_size"].value = (tile_width, tile_height)
    compact_program["tile_count"].value = int(tile_count)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
    bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].bind_to_storage_buffer(binding=3)
    compact_program.run((tile_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )


def _seed_formal_connected_tile_frontier(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    seed_rect: tuple[int, int, int, int],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    pipeline._clear_formal_connected_tile_mask_buffers(world)
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
    program = pipeline.programs["seed_formal_connected_tile_frontier"]
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (tile_width, tile_height)
    program["tile_size"].value = int(tile_size)
    program["seed_rect"].value = tuple(int(value) for value in seed_rect)
    program["seed_tile_origin"].value = (tile_x0, tile_y0)
    _, _, behavior_params = pipeline._classification_material_params(world)
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    pipeline._clear_formal_connected_tile_mask_buffers(world)
    dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
    if dirty_queue is None:
        raise RuntimeError("formal dirty tile expansion requires dirty tile queue buffers")
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    program = pipeline.programs["seed_formal_connected_tile_frontier_from_dirty_queue"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal dirty tile frontier requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["region_tile_origin"].value = (int(x0) // int(tile_size), int(y0) // int(tile_size))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (tile_width, tile_height)
    program["tile_size"].value = int(tile_size)
    _, _, behavior_params = pipeline._classification_material_params(world)
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
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
    pipeline._invalidate_persistent_dense_tile_worklist()
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    program = pipeline.programs["expand_formal_connected_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected tile frontier requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (tile_width, tile_height)
    program["tile_size"].value = int(tile_size)
    program["jump"].value = int(max(1, jump))
    _, _, behavior_params = pipeline._classification_material_params(world)
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
    world: "WorldEngine",
    frontier: tuple[str, str, str, str],
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    flags_name, list_name, count_name, dispatch_args_name = frontier
    clear_program = pipeline.programs["clear_formal_connected_cell_frontier_tile_flags_by_list"]
    if not hasattr(clear_program, "run_indirect"):
        raise RuntimeError("formal connected cell frontier flag clear requires ComputeShader.run_indirect")
    clear_program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    bridge.buffers[flags_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[list_name].bind_to_storage_buffer(binding=1)
    clear_program.run_indirect(bridge.buffers[dispatch_args_name])
    pipeline._sync_compute_writes(ctx)

    program = pipeline.programs["reset_formal_connected_cell_frontier_tiles"]
    bridge.buffers[flags_name].bind_to_storage_buffer(binding=0)
    bridge.buffers[count_name].bind_to_storage_buffer(binding=1)
    bridge.buffers[dispatch_args_name].bind_to_storage_buffer(binding=2)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(flags_name, count_name, dispatch_args_name)


def _accumulate_formal_connected_cell_frontier_tiles(
    pipeline,
    world: "WorldEngine",
    *,
    target_frontier: tuple[str, str, str, str],
    source_frontier: tuple[str, str, str, str],
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["accumulate_formal_connected_cell_frontier_tiles"]
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["seed_formal_connected_cell_frontier"]
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected tile expansion requires tile_size <= 32")
    current_frontier = (
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    )
    pipeline._clear_formal_connected_cell_frontier_tiles(world, current_frontier)
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_FRONTIER_BUFFER,
        FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    )


def _seed_formal_connected_cell_frontier_from_dirty_queue(
    pipeline,
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
    pipeline._ensure_programs(ctx)
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
    pipeline._clear_formal_connected_cell_frontier_tiles(world, current_frontier)
    pipeline._clear_formal_connected_cell_buffer_connected_tiles(
        world,
        FORMAL_CONNECTED_FRONTIER_BUFFER,
        width,
        height,
        tile_mask_name,
    )
    program = pipeline.programs["seed_formal_connected_cell_frontier_from_dirty_queue"]
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_FRONTIER_BUFFER,
        FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER,
    )


def _expand_formal_connected_cell_frontier(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["expand_formal_connected_cells_by_tile"]
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
    pipeline._sync_compute_writes(ctx)
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
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["copy_formal_connected_buffer_to_texture"]
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
    pipeline._sync_compute_writes(ctx)


def _solve_formal_connected_tile_support_textures(
    pipeline,
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
    current, scratch, schedule = _begin_formal_connected_tile_support(
        pipeline,
        world,
        resources,
        width,
        height,
        tile_mask_name,
    )
    current, scratch = _run_formal_connected_tile_support_slice(
        pipeline,
        world,
        resources,
        current,
        scratch,
        width,
        height,
        tile_mask_name,
        schedule,
        0,
        len(schedule),
    )
    if publish_masks:
        with pipeline._profile_pass(world, "support_jfa.publish"):
            pipeline._publish_bridge_supported_unsupported_masks_connected_tiles(
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


def _begin_formal_connected_tile_support(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    width: int,
    height: int,
    tile_mask_name: str,
    *,
    use_u8: bool = False,
    axis_masks_prebuilt: bool = False,
) -> tuple[Any, Any, tuple[int, ...]]:
    if pipeline._support_tile_union_enabled:
        # The graph candidate performs one tile-local closure and one boundary
        # union job instead of a texture ping-pong schedule.
        return resources.support_ping, resources.support_pong, (0,)
    if use_u8:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
        current, scratch = pipeline._ensure_formal_connected_u8_support_textures(ctx, resources)
    else:
        current, scratch = resources.support_ping, resources.support_pong
    if axis_masks_prebuilt and not use_u8:
        raise ValueError("prebuilt formal support axis masks require the U8 support path")
    if not axis_masks_prebuilt:
        with pipeline._profile_pass(world, "support_jfa.axis_masks"):
            pipeline._build_formal_connected_axis_masks(
                world,
                resources,
                resources.structural_tex,
                width,
                height,
                tile_mask_name,
                support_seed_texture=resources.support_ping if use_u8 else None,
                support_seed_u8_texture=current if use_u8 else None,
            )
    schedule = (
        *pipeline._formal_jfa_jumps(width, height),
        *([1] * pipeline._formal_connected_tile_refine_pass_count(world)),
    )
    return current, scratch, tuple(int(jump) for jump in schedule)


def _run_formal_connected_tile_support_slice(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
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
    if pipeline._support_tile_union_enabled:
        if slice_start == 0 and slice_stop > 0:
            return _run_formal_connected_tile_support_union(
                pipeline,
                world,
                resources,
                width,
                height,
                tile_mask_name,
            )
        return current, scratch
    refine_pass_count = min(
        schedule_length,
        max(0, int(pipeline._formal_connected_tile_refine_pass_count(world))),
    )
    jfa_stop = schedule_length - refine_pass_count
    jfa_slice_start = min(slice_start, jfa_stop)
    jfa_slice_stop = min(slice_stop, jfa_stop)
    if jfa_slice_start < jfa_slice_stop:
        jumps = schedule[jfa_slice_start:jfa_slice_stop]
        with pipeline._profile_pass(world, "support_jfa.jfa"):
            for band_name, band_jumps in pipeline._formal_jfa_profile_jump_bands(jumps):
                with pipeline._profile_pass(world, f"support_jfa.jfa.{band_name}"):
                    for jump in band_jumps:
                        with pipeline._profile_pass(
                            world,
                            f"support_jfa.jump_{int(jump)}",
                        ):
                            current, scratch = (
                                pipeline._run_formal_connected_tile_support_pass(
                                    world,
                                    resources,
                                    current,
                                    scratch,
                                    width,
                                    height,
                                    tile_mask_name,
                                    jump,
                                )
                            )
    refine_slice_start = max(slice_start, jfa_stop)
    if refine_slice_start < slice_stop:
        with pipeline._profile_pass(world, "support_jfa.refine"):
            for jump in schedule[refine_slice_start:slice_stop]:
                current, scratch = pipeline._run_formal_connected_tile_support_pass(
                    world,
                    resources,
                    current,
                    scratch,
                    width,
                    height,
                    tile_mask_name,
                    jump,
                )
    return current, scratch


def _ensure_support_tile_union_buffers(
    pipeline,
    ctx: Any,
    resources: GPUCollapseResources,
    width: int,
    height: int,
    edge_capacity: int,
) -> None:
    cell_bytes = max(4, int(width) * int(height) * np.dtype(np.uint32).itemsize)
    edge_bytes = max(8, int(edge_capacity) * 2 * np.dtype(np.uint32).itemsize)
    specs = (
        ("support_tile_union_roots", cell_bytes),
        ("support_tile_union_parent", cell_bytes),
        ("support_tile_union_seeded", cell_bytes),
        ("support_tile_union_edges", edge_bytes),
        ("support_tile_union_edge_count", 4),
    )
    for name, required_bytes in specs:
        buffer = getattr(resources, name)
        if buffer is not None and buffer.size >= required_bytes:
            continue
        if buffer is not None:
            buffer.release()
        setattr(resources, name, ctx.buffer(reserve=required_bytes, dynamic=True))


def _run_formal_connected_tile_support_union(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    width: int,
    height: int,
    tile_mask_name: str,
) -> tuple[Any, Any]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_width = max(1, int(world.active.tile_width))
    tile_height = max(1, int(world.active.tile_height))
    tile_size = max(1, int(world.active.tile_size))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected support tile union requires tile_size <= 32")
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
    seeded = resources.support_tile_union_seeded
    edges = resources.support_tile_union_edges
    edge_count = resources.support_tile_union_edge_count
    if roots is None or parents is None or seeded is None or edges is None or edge_count is None:
        raise RuntimeError("formal connected support tile union buffers were not allocated")
    edge_count.write(np.zeros(1, dtype=np.uint32).tobytes())

    with pipeline._profile_pass(world, "support_tile_union.local_components"):
        program = pipeline.programs["support_tile_union_local"]
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = tile_size
        resources.structural_tex.use(location=0)
        resources.support_ping.use(location=1)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        roots.bind_to_storage_buffer(binding=3)
        parents.bind_to_storage_buffer(binding=4)
        seeded.bind_to_storage_buffer(binding=5)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "support_tile_union.boundary_edges"):
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

    edge_groups = (edge_capacity + 63) // 64
    with pipeline._profile_pass(world, "support_tile_union.union"):
        if pipeline._support_tile_union_atomic_union_enabled:
            cell_count = max(1, int(width) * int(height))
            hook = pipeline.programs["support_tile_union_atomic_hook"]
            hook["edge_capacity"].value = edge_capacity
            hook["cell_count"].value = cell_count
            edges.bind_to_storage_buffer(binding=0)
            edge_count.bind_to_storage_buffer(binding=1)
            parents.bind_to_storage_buffer(binding=2)
            hook.run(edge_groups, 1, 1)
            pipeline._sync_compute_writes(ctx)

            # Union is complete after the lock-free hook. Compress every edge
            # endpoint once so the existing seed/materialize shaders retain
            # their shallow-parent fast path, including adversarial tile chains.
            shortcut = pipeline.programs["support_tile_union_atomic_shortcut"]
            shortcut["edge_capacity"].value = edge_capacity
            shortcut["cell_count"].value = cell_count
            edges.bind_to_storage_buffer(binding=0)
            edge_count.bind_to_storage_buffer(binding=1)
            parents.bind_to_storage_buffer(binding=2)
            shortcut.run(edge_groups, 1, 1)
            pipeline._sync_compute_writes(ctx)
        else:
            union_rounds = max(1, (max(1, int(width) * int(height))).bit_length() + 2)
            hook = pipeline.programs["support_tile_union_hook"]
            hook["edge_capacity"].value = edge_capacity
            shortcut = pipeline.programs["support_tile_union_shortcut"]
            shortcut["edge_capacity"].value = edge_capacity
            for _ in range(union_rounds):
                edges.bind_to_storage_buffer(binding=0)
                edge_count.bind_to_storage_buffer(binding=1)
                parents.bind_to_storage_buffer(binding=2)
                hook.run(edge_groups, 1, 1)
                pipeline._sync_compute_writes(ctx)
                edges.bind_to_storage_buffer(binding=0)
                edge_count.bind_to_storage_buffer(binding=1)
                parents.bind_to_storage_buffer(binding=2)
                shortcut.run(edge_groups, 1, 1)
                pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "support_tile_union.propagate_seeds"):
        program = pipeline.programs["support_tile_union_seed"]
        program["edge_capacity"].value = edge_capacity
        edges.bind_to_storage_buffer(binding=0)
        edge_count.bind_to_storage_buffer(binding=1)
        parents.bind_to_storage_buffer(binding=2)
        seeded.bind_to_storage_buffer(binding=3)
        program.run(edge_groups, 1, 1)
        pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "support_tile_union.materialize"):
        program = pipeline.programs["support_tile_union_materialize"]
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (tile_width, tile_height)
        program["tile_size"].value = tile_size
        resources.support_pong.bind_to_image(0, read=False, write=True)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        roots.bind_to_storage_buffer(binding=3)
        parents.bind_to_storage_buffer(binding=4)
        seeded.bind_to_storage_buffer(binding=5)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(ctx)
    return resources.support_pong, resources.support_ping


def _seed_formal_connected_tile_support_frontier(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    pipeline._clear_formal_connected_cell_frontier_tiles(world, frontier)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected support propagation requires tile_size <= 32")
    program = pipeline.programs["seed_formal_connected_tile_support_frontier"]
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
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_FRONTIER_BUFFER,
        flags_name,
        list_name,
        count_name,
        dispatch_args_name,
    )


def _expand_formal_connected_tile_support_frontier(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected support propagation requires tile_size <= 32")
    program = pipeline.programs["expand_formal_connected_tile_support_frontier"]
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


def _run_formal_connected_tile_support_pass(
    pipeline,
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
    pipeline._ensure_programs(ctx)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected support propagation requires tile_size <= 32")
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    use_u8 = current is resources.support_u8_ping or current is resources.support_u8_pong
    if use_u8:
        if (
            pipeline._support_jfa_u8_propagated_source_mask_elision_enabled
            and pipeline._support_jfa_row_major_output_enabled
            and pipeline._support_jfa_nv32_row_hydrate_enabled
            and pipeline._support_jfa_nv32_row_hydrate_supported
        ):
            program_name = (
                "propagate_formal_connected_tiles_u8_row_major_"
                "source_mask_elision_nv32"
            )
        elif pipeline._support_jfa_u8_propagated_source_mask_elision_enabled:
            program_name = (
                "propagate_formal_connected_tiles_u8_row_major_source_mask_elision"
                if pipeline._support_jfa_row_major_output_enabled
                else "propagate_formal_connected_tiles_u8_source_mask_elision"
            )
        else:
            program_name = (
                "propagate_formal_connected_tiles_u8_row_major"
                if pipeline._support_jfa_row_major_output_enabled
                else "propagate_formal_connected_tiles_u8"
            )
    else:
        program_name = (
            "propagate_formal_connected_tiles_row_major"
            if pipeline._support_jfa_row_major_output_enabled
            else "propagate_formal_connected_tiles"
        )
    program = pipeline.programs[program_name]
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
    if pipeline._support_jfa_image_barrier_elision_enabled:
        # The next JFA pass samples ``current`` as a texture and writes a
        # different ping-pong image.  Image-access ordering is therefore not
        # needed; texture-fetch + storage ordering covers the actual hazards.
        ctx.memory_barrier(
            ctx.TEXTURE_FETCH_BARRIER_BIT | ctx.SHADER_STORAGE_BARRIER_BIT
        )
    else:
        pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative(
        tile_mask_name,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )
    return scratch, current


def _run_formal_connected_tile_support_refine_passes(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    current: Any,
    scratch: Any,
    width: int,
    height: int,
    tile_mask_name: str,
) -> tuple[Any, Any]:
    for _ in range(pipeline._formal_connected_tile_refine_pass_count(world)):
        current, scratch = pipeline._run_formal_connected_tile_support_pass(
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
