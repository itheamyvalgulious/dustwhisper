from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE,
    ACTIVE_TILE_WORKGROUP_AXIS,
    ACTIVE_TILE_WORKGROUPS_PER_TILE,
    POWDER_RESERVATION_LOCAL_SIZE,
    FALLING_ISLAND_RESERVATION_DTYPE
)


def _active_tile_workgroups_per_tile(pipeline, world: "WorldEngine") -> int:
    if int(world.active.tile_size) == LOCAL_SIZE * ACTIVE_TILE_WORKGROUP_AXIS:
        return ACTIVE_TILE_WORKGROUPS_PER_TILE
    axis = max(1, (int(world.active.tile_size) + LOCAL_SIZE - 1) // LOCAL_SIZE)
    return axis * axis


def _active_scheduler_gpu_authoritative(pipeline, world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return (
        pipeline._formal_gpu_frame(world)
        and "active_tile_ttl" in authoritative
        and "active_chunk_mask" in authoritative
    )


def _compact_active_tiles(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_program = pipeline.programs["clear_active_tile_dispatch"]
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    workgroups_per_tile = int(pipeline._active_tile_workgroups_per_tile(world))
    if pipeline._active_scheduler_gpu_authoritative(world):
        bridge = world.bridge
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        compact_program = pipeline.programs["compact_active_tiles_from_chunks"]
        if not hasattr(compact_program, "run_indirect"):
            raise RuntimeError("GPU motion active chunk compaction requires ModernGL ComputeShader.run_indirect")
        compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
        compact_program["workgroups_per_tile"].value = workgroups_per_tile
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
        bridge.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
        compact_program.run_indirect(bridge.buffers["active_chunk_dispatch_args"])
    else:
        tile_count = int(world.active.tile_width * world.active.tile_height)
        compact_program = pipeline.programs["compact_active_tiles"]
        compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        compact_program["workgroups_per_tile"].value = workgroups_per_tile
        resources.active_tile_tex.use(location=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        compact_program.run((tile_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)


def _build_active_tile_count_dispatch_args(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["build_powder_reservation_dispatch"]
    program["invocations_per_group"].value = 1
    program["max_reservation_count"].value = int(world.active.tile_width * world.active.tile_height)
    resources.active_tile_count.bind_to_storage_buffer(binding=6)
    resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)


def _build_falling_island_materialization_candidate_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_program = pipeline.programs["clear_active_tile_dispatch"]
    resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=0)
    resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    pipeline._build_active_tile_count_dispatch_args(world, resources)
    program = pipeline.programs["build_falling_island_materialization_candidate_dispatch"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    resources.cell_state_tex.use(location=0)
    resources.island_id_tex.use(location=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=2)
    resources.island_materialization_candidate_tile_list.bind_to_storage_buffer(binding=3)
    resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=4)
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("GPU motion falling island materialization candidate dispatch requires indirect dispatch")
    program.run_indirect(resources.island_runtime_dispatch_args)
    pipeline._sync_compute_writes(ctx)


def _copy_scalar_texture(pipeline, ctx: Any, source_tex: Any, dest_tex: Any, width: int, height: int) -> None:
    program = pipeline.programs["copy_scalar_texture"]
    program["grid_size"].value = (int(width), int(height))
    source_tex.use(location=0)
    dest_tex.bind_to_image(1, read=False, write=True)
    group_x = (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def _swap_powder_apply_textures(pipeline, resources: GPUMotionResources) -> None:
    resources.cell_state_tex, resources.cell_state_out_tex = (
        resources.cell_state_out_tex,
        resources.cell_state_tex,
    )
    resources.velocity_tex, resources.velocity_out_tex = resources.velocity_out_tex, resources.velocity_tex
    resources.temp_tex, resources.temp_out_tex = resources.temp_out_tex, resources.temp_tex
    resources.timer_tex, resources.timer_out_tex = resources.timer_out_tex, resources.timer_tex
    resources.integrity_tex, resources.integrity_out_tex = resources.integrity_out_tex, resources.integrity_tex
    resources.island_id_tex, resources.island_id_out_tex = resources.island_id_out_tex, resources.island_id_tex
    resources.entity_id_tex, resources.entity_id_out_tex = resources.entity_id_out_tex, resources.entity_id_tex
    resources.displaced_tex, resources.displaced_out_tex = resources.displaced_out_tex, resources.displaced_tex


def _barrier_bits(pipeline) -> tuple[str, ...]:
    # motion uses indirect dispatch + buffer updates in addition to the
    # default image/texture/storage sync.
    return (
        "SHADER_STORAGE_BARRIER_BIT",
        "SHADER_IMAGE_ACCESS_BARRIER_BIT",
        "TEXTURE_FETCH_BARRIER_BIT",
        "COMMAND_BARRIER_BIT",
        "BUFFER_UPDATE_BARRIER_BIT",
    )


def _run_active_tile_indirect(
    pipeline,
    program: Any,
    resources: GPUMotionResources,
    pass_name: str,
    *,
    dispatch_args: Any | None = None,
) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
    program.run_indirect(resources.active_tile_dispatch_args if dispatch_args is None else dispatch_args)


def _refresh_authoritative_active_scheduler_after_apply(pipeline, world: "WorldEngine", pass_name: str) -> None:
    if not (pipeline._formal_gpu_frame(world) and "active_tile_ttl" in world.bridge.gpu_authoritative_resources):
        return
    with pipeline._profile_pass(world, pass_name):
        world.bridge._ensure_active_scheduler_programs()
        world.bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        world.bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")


def _build_powder_reservation_dispatch_args(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    invocations_per_group: int,
    count_buffer: Any | None = None,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["build_powder_reservation_dispatch"]
    program["invocations_per_group"].value = int(invocations_per_group)
    program["max_reservation_count"].value = int(world.width * world.height)
    (
        resources.powder_reservation_count
        if count_buffer is None
        else count_buffer
    ).bind_to_storage_buffer(binding=6)
    resources.powder_reservation_dispatch_args.bind_to_storage_buffer(binding=7)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)


def _run_powder_reservation_indirect(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    program: Any,
    pass_name: str,
    *,
    invocations_per_group: int = POWDER_RESERVATION_LOCAL_SIZE,
    before_run: Any | None = None,
    count_buffer: Any | None = None,
) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
    pipeline._build_powder_reservation_dispatch_args(
        world,
        resources,
        invocations_per_group=int(invocations_per_group),
        count_buffer=count_buffer,
    )
    if before_run is not None:
        before_run()
    program.run_indirect(resources.powder_reservation_dispatch_args)


def _build_island_reservation_dispatch_args(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    reservation_capacity: int,
    invocations_per_group: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["build_powder_reservation_dispatch"]
    program["invocations_per_group"].value = int(invocations_per_group)
    program["max_reservation_count"].value = int(reservation_capacity)
    resources.island_reservation_count.bind_to_storage_buffer(binding=6)
    resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)


def _run_island_reservation_indirect(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    program: Any,
    pass_name: str,
    *,
    reservation_capacity: int,
    invocations_per_group: int = LOCAL_SIZE,
) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
    pipeline._build_island_reservation_dispatch_args(
        world,
        resources,
        reservation_capacity=int(reservation_capacity),
        invocations_per_group=int(invocations_per_group),
    )
    program.run_indirect(resources.island_runtime_dispatch_args)


def _build_island_runtime_dispatch_args(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    runtime_capacity: int,
    invocations_per_group: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["build_island_runtime_dispatch"]
    program["invocations_per_group"].value = int(invocations_per_group)
    program["runtime_capacity"].value = int(runtime_capacity)
    world.bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=6)
    resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
    program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)


def _run_island_runtime_indirect(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    program: Any,
    pass_name: str,
    *,
    runtime_capacity: int,
    invocations_per_group: int = LOCAL_SIZE,
    before_run: Any | None = None,
) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
    pipeline._build_island_runtime_dispatch_args(
        world,
        resources,
        runtime_capacity=int(runtime_capacity),
        invocations_per_group=int(invocations_per_group),
    )
    if before_run is not None:
        before_run()
    program.run_indirect(resources.island_runtime_dispatch_args)


def _build_powder_apply_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    tile_count = int(world.active.tile_width * world.active.tile_height)
    clear_program = pipeline.programs["clear_powder_affected_tile_dispatch"]
    clear_program["tile_count"].value = tile_count
    resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
    clear_program.run((tile_count + 255) // 256, 1, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))

    build_program = pipeline.programs["build_powder_apply_dispatch"]
    build_program["cell_grid_size"].value = (world.width, world.height)
    build_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    build_program["tile_size"].value = int(world.active.tile_size)
    build_program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    resources.powder_reservations.bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=2)
    resources.active_tile_count.bind_to_storage_buffer(binding=3)
    resources.active_tile_list.bind_to_storage_buffer(binding=4)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=5)
    pipeline._run_powder_reservation_indirect(
        world,
        resources,
        build_program,
        "powder apply affected tile dispatch build",
    )
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))


def _build_falling_island_apply_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    reservation_count: int,
    operation: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    reservation_count = max(0, int(reservation_count))
    tile_count = int(world.active.tile_width * world.active.tile_height)
    clear_program = pipeline.programs["clear_powder_affected_tile_dispatch"]
    clear_program["tile_count"].value = tile_count
    resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
    clear_program.run((tile_count + 255) // 256, 1, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))
    if reservation_count <= 0:
        return

    build_program = pipeline.programs["build_falling_island_apply_dispatch"]
    build_program["cell_grid_size"].value = (world.width, world.height)
    build_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    build_program["tile_size"].value = int(world.active.tile_size)
    build_program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    build_program["operation"].value = int(operation)
    resources.island_reservations.bind_to_storage_buffer(binding=0)
    resources.island_reservation_count.bind_to_storage_buffer(binding=1)
    resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=2)
    resources.active_tile_count.bind_to_storage_buffer(binding=3)
    resources.active_tile_list.bind_to_storage_buffer(binding=4)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=5)
    pipeline._run_island_reservation_indirect(
        world,
        resources,
        build_program,
        "falling island apply affected tile dispatch build",
        reservation_capacity=reservation_count,
        invocations_per_group=POWDER_RESERVATION_LOCAL_SIZE,
    )
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))


def _ensure_falling_island_index_capacity(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_bytes = int(world.width * world.height * np.dtype(np.int32).itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_apply_incoming", cell_bytes)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_apply_outgoing", cell_bytes)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_materialization_index", cell_bytes)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_reservation_source_index", cell_bytes)


def _clear_falling_island_index(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    pass_name: str,
    clear_flags: int,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._ensure_falling_island_index_capacity(world, resources)
    with pipeline._profile_pass(world, pass_name):
        if pipeline._formal_gpu_frame(world):
            reservation_count = max(0, int(reservation_count))
            if reservation_count <= 0:
                return
            program = pipeline.programs["clear_falling_island_index_for_reservations"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["clear_flags"].value = int(clear_flags)
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            resources.island_reservation_count.bind_to_storage_buffer(binding=1)
            resources.island_apply_incoming.bind_to_storage_buffer(binding=2)
            resources.island_apply_outgoing.bind_to_storage_buffer(binding=3)
            resources.island_materialization_index.bind_to_storage_buffer(binding=4)
            resources.island_reservation_source_index.bind_to_storage_buffer(binding=5)
            pipeline._run_island_reservation_indirect(
                world,
                resources,
                program,
                "falling island index reservation-domain clear",
                reservation_capacity=reservation_count,
                invocations_per_group=1,
            )
        else:
            cell_count = int(world.width * world.height)
            program = pipeline.programs["clear_falling_island_index"]
            program["cell_count"].value = cell_count
            program["clear_flags"].value = int(clear_flags)
            resources.island_apply_incoming.bind_to_storage_buffer(binding=0)
            resources.island_apply_outgoing.bind_to_storage_buffer(binding=1)
            resources.island_materialization_index.bind_to_storage_buffer(binding=2)
            resources.island_reservation_source_index.bind_to_storage_buffer(binding=3)
            program.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _ensure_bridge_runtime_reservation_capacity(
    pipeline,
    ctx: Any,
    resources: GPUMotionResources,
    runtime_capacity: int,
) -> None:
    runtime_capacity = max(0, int(runtime_capacity))
    pipeline._ensure_dynamic_buffer_capacity(
        ctx,
        resources,
        "island_reservations",
        runtime_capacity * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
    )


def _ensure_bridge_runtime_planning_capacity(
    pipeline,
    ctx: Any,
    resources: GPUMotionResources,
    runtime_capacity: int,
) -> None:
    runtime_capacity = max(0, int(runtime_capacity))
    int_itemsize = np.dtype(np.int32).itemsize
    float_itemsize = np.dtype(np.float32).itemsize
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_ids", runtime_capacity * int_itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_bboxes", runtime_capacity * 4 * int_itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_motion", runtime_capacity * 4 * float_itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "island_shift_results", runtime_capacity * 4 * int_itemsize)
    pipeline._ensure_dynamic_buffer_capacity(
        ctx,
        resources,
        "island_reservations",
        runtime_capacity * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
    )
