from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.gpu import pack_island_runtime_upload
from oracle_game.types import Phase

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE,
    ISLAND_RESERVATION_LINEAR_LOCAL_SIZE,
    FALLING_ISLAND_INDEX_CLEAR_SOURCE,
    FALLING_ISLAND_INDEX_CLEAR_APPLY,
    FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION,
    FALLING_ISLAND_RESERVATION_DTYPE,
    ISLAND_RESOLVE_STALE
)


def _dispatch_index_falling_island_reservation_sources(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._clear_falling_island_index(
        world,
        resources,
        pass_name="island_reservation_source_index_clear",
        clear_flags=FALLING_ISLAND_INDEX_CLEAR_SOURCE,
        reservation_count=int(reservation_count),
    )
    with pipeline._profile_pass(world, "island_reservation_source_index_build"):
        program = pipeline.programs["fill_falling_island_reservation_source_index"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=1)
        resources.island_reservation_source_index.bind_to_storage_buffer(binding=2)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        pipeline._run_island_reservation_indirect(
            world,
            resources,
            program,
            "falling island reservation source index build",
            reservation_capacity=int(reservation_count),
            invocations_per_group=1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def _dispatch_index_falling_island_apply(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._clear_falling_island_index(
        world,
        resources,
        pass_name="island_apply_index_clear",
        clear_flags=FALLING_ISLAND_INDEX_CLEAR_APPLY,
        reservation_count=int(reservation_count),
    )
    with pipeline._profile_pass(world, "island_apply_index_build"):
        program = pipeline.programs["fill_falling_island_apply_index"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=1)
        resources.island_apply_incoming.bind_to_storage_buffer(binding=2)
        resources.island_apply_outgoing.bind_to_storage_buffer(binding=3)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=7)
        pipeline._run_island_reservation_indirect(
            world,
            resources,
            program,
            "falling island apply index build",
            reservation_capacity=int(reservation_count),
            invocations_per_group=1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def _dispatch_index_falling_island_materialization(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._clear_falling_island_index(
        world,
        resources,
        pass_name="island_materialization_index_clear",
        clear_flags=FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION,
        reservation_count=int(reservation_count),
    )
    with pipeline._profile_pass(world, "island_materialization_index_build"):
        program = pipeline.programs["fill_falling_island_materialization_index"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=1)
        resources.island_materialization_index.bind_to_storage_buffer(binding=2)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=7)
        pipeline._run_island_reservation_indirect(
            world,
            resources,
            program,
            "falling island materialization index build",
            reservation_capacity=int(reservation_count),
            invocations_per_group=1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def apply_falling_island_reservations(pipeline, world: "WorldEngine", reservations: np.ndarray) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or len(reservations) == 0:
        return False
    moving = np.any(reservations["resolved_shift"] != 0, axis=1)
    if not bool(np.any(moving)):
        pipeline.upload_falling_island_reservations(world, reservations)
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.upload_falling_island_reservations(world, reservations)
    pipeline._dispatch_apply_falling_island_reservations(world, resources, int(len(reservations)))
    return True


def apply_uploaded_falling_island_reservations(pipeline, world: "WorldEngine", reservation_count: int) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or int(reservation_count) <= 0:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    if not pipeline._formal_gpu_frame(world):
        resources.island_reservation_count.write(np.array([int(reservation_count)], dtype=np.int32).tobytes())
    pipeline._dispatch_apply_falling_island_reservations(world, resources, int(reservation_count))
    if pipeline._formal_gpu_frame(world):
        pipeline._dispatch_apply_falling_island_materialization(
            world,
            resources,
            reservation_count=int(reservation_count),
            mode=0,
            use_existing_active_tile_dispatch=True,
        )
    return True


def shed_falling_island_fragments(pipeline, world: "WorldEngine") -> bool:
    ctx = world.bridge.ctx
    if ctx is None:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.upload_falling_island_reservations(world, np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE))
    pipeline._dispatch_apply_falling_island_materialization(world, resources, reservation_count=0, mode=0)
    return True


def apply_falling_island_settlements(pipeline, world: "WorldEngine", reservations: np.ndarray) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or len(reservations) == 0:
        return False
    settling = (
        (reservations["resolve_state"] != ISLAND_RESOLVE_STALE)
        & np.any(reservations["target_shift"] != 0, axis=1)
        & ~np.any(reservations["resolved_shift"] != 0, axis=1)
    )
    if not bool(np.any(settling)):
        pipeline.upload_falling_island_reservations(world, reservations)
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.upload_falling_island_reservations(world, reservations)
    pipeline._dispatch_apply_falling_island_materialization(world, resources, int(len(reservations)), mode=1)
    return True


def apply_uploaded_falling_island_settlements(pipeline, world: "WorldEngine", reservation_count: int) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or int(reservation_count) <= 0:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    if not pipeline._formal_gpu_frame(world):
        resources.island_reservation_count.write(np.array([int(reservation_count)], dtype=np.int32).tobytes())
    pipeline._dispatch_apply_falling_island_materialization(world, resources, int(reservation_count), mode=1)
    return True


def _dispatch_apply_falling_island_materialization(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    reservation_count: int,
    *,
    mode: int,
    inputs_already_loaded: bool = False,
    use_existing_active_tile_dispatch: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    formal_frame = pipeline._formal_gpu_frame(world)
    if formal_frame and int(mode) == 0:
        pipeline._sync_compute_writes(ctx)
    pipeline._upload_powder_apply_state(world, resources)
    pipeline._upload_material_rule_params(world, resources)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    formal_mode_zero = formal_frame and int(mode) == 0
    if formal_frame:
        if formal_mode_zero and inputs_already_loaded:
            if not pipeline._active_scheduler_gpu_authoritative(world):
                pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
            pipeline._compact_active_tiles(world, resources)
        elif formal_mode_zero and not pipeline._active_scheduler_gpu_authoritative(world):
            pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
            pipeline._compact_active_tiles(world, resources)
        elif not formal_mode_zero:
            pipeline._build_falling_island_apply_dispatch(
                world,
                resources,
                reservation_count=int(reservation_count),
                operation=1,
            )
    if not inputs_already_loaded:
        reuse_active_dispatch = bool(
            formal_frame and (not formal_mode_zero or use_existing_active_tile_dispatch)
        )
        pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=reuse_active_dispatch,
        )
    materialization_tile_count_buffer = resources.active_tile_count
    materialization_tile_list_buffer = resources.active_tile_list
    materialization_dispatch_args = resources.active_tile_dispatch_args
    if formal_mode_zero:
        pipeline._build_falling_island_materialization_candidate_dispatch(world, resources)
        materialization_tile_count_buffer = resources.island_materialization_candidate_tile_count
        materialization_tile_list_buffer = resources.island_materialization_candidate_tile_list
        materialization_dispatch_args = resources.island_materialization_candidate_dispatch_args
    if int(mode) != 0:
        pipeline._dispatch_index_falling_island_materialization(
            world,
            resources,
            reservation_count=int(reservation_count),
        )
    program = pipeline.programs["apply_falling_island_materialization"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["reservation_count"].value = int(reservation_count)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["phase_powder"].value = int(Phase.POWDER)
    program["phase_static_solid"].value = int(Phase.STATIC_SOLID)
    program["mode"].value = int(mode)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["use_reservation_count_buffer"].value = bool(formal_frame)
    program["use_active_tile_dispatch"].value = bool(formal_frame)
    resources.island_reservations.bind_to_storage_buffer(binding=0)
    resources.material_falling_params.bind_to_storage_buffer(binding=1)
    resources.island_reservation_count.bind_to_storage_buffer(binding=2)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.island_materialization_index.bind_to_storage_buffer(binding=4)
    materialization_tile_count_buffer.bind_to_storage_buffer(binding=6)
    materialization_tile_list_buffer.bind_to_storage_buffer(binding=7)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.cell_flags_tex.use(location=2)
    resources.velocity_tex.use(location=3)
    resources.temp_tex.use(location=4)
    resources.timer_tex.use(location=5)
    resources.integrity_tex.use(location=6)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.material_out_tex.bind_to_image(0, read=False, write=True)
    resources.phase_out_tex.bind_to_image(1, read=False, write=True)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
    resources.temp_out_tex.bind_to_image(4, read=False, write=True)
    resources.timer_out_tex.bind_to_image(5, read=False, write=True)
    resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
    with pipeline._profile_pass(world, "island_materialization_main"):
        if formal_frame:
            pipeline._run_active_tile_indirect(
                program,
                resources,
                "falling island materialization",
                dispatch_args=materialization_dispatch_args,
            )
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    aux_program = pipeline.programs["apply_falling_island_materialization_aux"]
    aux_program["cell_grid_size"].value = (world.width, world.height)
    aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    aux_program["tile_size"].value = int(world.active.tile_size)
    aux_program["reservation_count"].value = int(reservation_count)
    aux_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    aux_program["mode"].value = int(mode)
    aux_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    aux_program["use_reservation_count_buffer"].value = bool(formal_frame)
    aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
    resources.island_reservations.bind_to_storage_buffer(binding=0)
    resources.material_falling_params.bind_to_storage_buffer(binding=1)
    resources.island_reservation_count.bind_to_storage_buffer(binding=2)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.island_materialization_index.bind_to_storage_buffer(binding=4)
    materialization_tile_count_buffer.bind_to_storage_buffer(binding=6)
    materialization_tile_list_buffer.bind_to_storage_buffer(binding=7)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
    resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
    resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
    with pipeline._profile_pass(world, "island_materialization_aux"):
        if formal_frame:
            pipeline._run_active_tile_indirect(
                aux_program,
                resources,
                "falling island materialization aux",
                dispatch_args=materialization_dispatch_args,
            )
        else:
            aux_program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "island_materialization_bridge_publish"):
        pipeline._publish_bridge_outputs(
            world,
            resources,
            output_textures=True,
            active_tile_indirect=formal_frame,
            active_tile_count_buffer=materialization_tile_count_buffer,
            active_tile_list_buffer=materialization_tile_list_buffer,
            active_tile_dispatch_args=materialization_dispatch_args,
        )
    pipeline._refresh_authoritative_active_scheduler_after_apply(
        world,
        "active_refresh_after_falling_island_materialization",
    )
    pipeline.last_cpu_mirror_downloaded = not formal_frame
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_powder_apply_state(world, resources)


def _dispatch_apply_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    formal_frame = pipeline._formal_gpu_frame(world)
    pipeline._upload_powder_apply_state(world, resources)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if formal_frame:
        pipeline._build_falling_island_apply_dispatch(
            world,
            resources,
            reservation_count=int(reservation_count),
            operation=0,
        )
    pipeline._load_authoritative_bridge_inputs(
        world,
        resources,
        group_x,
        group_y,
        use_existing_active_tile_dispatch=formal_frame,
    )
    pipeline._dispatch_index_falling_island_apply(
        world,
        resources,
        reservation_count=int(reservation_count),
    )
    program = pipeline.programs["apply_falling_island_reservations"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["gas_cell_size"].value = world.gas_cell_size
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["reservation_count"].value = int(reservation_count)
    program["use_reservation_count_buffer"].value = bool(formal_frame)
    program["use_active_tile_dispatch"].value = bool(formal_frame)
    resources.island_reservations.bind_to_storage_buffer(binding=0)
    resources.island_reservation_count.bind_to_storage_buffer(binding=2)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.island_apply_incoming.bind_to_storage_buffer(binding=4)
    resources.island_apply_outgoing.bind_to_storage_buffer(binding=5)
    resources.active_tile_count.bind_to_storage_buffer(binding=6)
    resources.active_tile_list.bind_to_storage_buffer(binding=7)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.cell_flags_tex.use(location=2)
    resources.velocity_tex.use(location=3)
    resources.temp_tex.use(location=4)
    resources.timer_tex.use(location=5)
    resources.integrity_tex.use(location=6)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.ambient_tex.use(location=20)
    resources.material_out_tex.bind_to_image(0, read=False, write=True)
    resources.phase_out_tex.bind_to_image(1, read=False, write=True)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
    resources.temp_out_tex.bind_to_image(4, read=False, write=True)
    resources.timer_out_tex.bind_to_image(5, read=False, write=True)
    resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
    with pipeline._profile_pass(world, "island_apply_main"):
        if formal_frame:
            pipeline._run_active_tile_indirect(program, resources, "falling island reservation apply")
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    aux_program = pipeline.programs["apply_falling_island_reservation_aux"]
    aux_program["cell_grid_size"].value = (world.width, world.height)
    aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    aux_program["tile_size"].value = int(world.active.tile_size)
    aux_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    aux_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    aux_program["reservation_count"].value = int(reservation_count)
    aux_program["use_reservation_count_buffer"].value = bool(formal_frame)
    aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
    resources.island_reservations.bind_to_storage_buffer(binding=0)
    resources.island_reservation_count.bind_to_storage_buffer(binding=2)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.island_apply_incoming.bind_to_storage_buffer(binding=4)
    resources.island_apply_outgoing.bind_to_storage_buffer(binding=5)
    resources.active_tile_count.bind_to_storage_buffer(binding=6)
    resources.active_tile_list.bind_to_storage_buffer(binding=7)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
    resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
    resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
    with pipeline._profile_pass(world, "island_apply_aux"):
        if formal_frame:
            pipeline._run_active_tile_indirect(aux_program, resources, "falling island reservation aux apply")
        else:
            aux_program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "island_apply_bridge_publish"):
        pipeline._publish_bridge_outputs(world, resources, output_textures=True, active_tile_indirect=formal_frame)
    pipeline._refresh_authoritative_active_scheduler_after_apply(
        world,
        "active_refresh_after_falling_island_reservation",
    )
    pipeline.last_cpu_mirror_downloaded = not formal_frame
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_powder_apply_state(world, resources)


def plan_uploaded_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    island_ids: list[int] | None = None,
    motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> int:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    runtime = pack_island_runtime_upload(world)
    if island_ids is not None:
        wanted = set(int(island_id) for island_id in island_ids)
        runtime = runtime[np.isin(runtime["island_id"], np.fromiter(wanted, dtype=np.int32, count=len(wanted)))]
    if motion_overrides:
        for island_id, (velocity_xy, subcell_offset) in motion_overrides.items():
            matches = np.nonzero(runtime["island_id"] == int(island_id))[0]
            if matches.size == 0:
                continue
            index = int(matches[0])
            runtime[index]["velocity_xy"] = np.asarray(velocity_xy, dtype=np.float32)
            runtime[index]["subcell_offset"] = np.asarray(subcell_offset, dtype=np.float32)
    if runtime.size == 0:
        reservations = np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
        pipeline.upload_falling_island_reservations(world, reservations)
        return 0
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["cell_core"]:
        resources.material_tex.write(world.material_id.astype("f4").tobytes())
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    pipeline._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
    pipeline._upload_material_rule_params(world, resources)
    packed_ids = np.ascontiguousarray(runtime["island_id"].astype(np.int32))
    packed_bboxes = np.ascontiguousarray(runtime["buffer_bbox"].astype(np.int32))
    packed_motion = np.zeros((runtime.shape[0], 4), dtype=np.float32)
    packed_motion[:, :2] = runtime["velocity_xy"]
    packed_motion[:, 2:] = runtime["subcell_offset"]
    packed_shifts = np.zeros((runtime.shape[0], 4), dtype=np.int32)
    empty_reservations = np.zeros((runtime.shape[0],), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
    pipeline._write_dynamic_buffer(ctx, resources, "island_ids", packed_ids)
    pipeline._write_dynamic_buffer(ctx, resources, "island_bboxes", packed_bboxes)
    pipeline._write_dynamic_buffer(ctx, resources, "island_motion", packed_motion)
    pipeline._write_dynamic_buffer(ctx, resources, "island_shift_results", packed_shifts)
    pipeline._write_dynamic_buffer(ctx, resources, "island_reservations", empty_reservations)
    resources.island_reservation_count.write(np.array([int(runtime.shape[0])], dtype=np.int32).tobytes())
    program = pipeline.programs["island_shifts"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["island_count"].value = int(runtime.shape[0])
    program["use_island_count_buffer"].value = False
    use_bridge_state = pipeline._bridge_authoritative_island_state(world)
    program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
    program["dt"].value = float(dt)
    resources.material_tex.use(location=0)
    resources.island_id_tex.use(location=1)
    resources.island_ids.bind_to_storage_buffer(binding=0)
    resources.island_bboxes.bind_to_storage_buffer(binding=1)
    resources.island_motion.bind_to_storage_buffer(binding=2)
    resources.island_shift_results.bind_to_storage_buffer(binding=3)
    resources.material_params.bind_to_storage_buffer(binding=5)
    if use_bridge_state:
        pipeline._bind_bridge_island_state(world, cell_binding=7)
    group_x = (runtime.shape[0] + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, 1, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    pack_program = pipeline.programs["pack_falling_island_reservations"]
    pack_program["island_count"].value = int(runtime.shape[0])
    pack_program["use_island_count_buffer"].value = False
    resources.island_ids.bind_to_storage_buffer(binding=0)
    resources.island_bboxes.bind_to_storage_buffer(binding=1)
    resources.island_motion.bind_to_storage_buffer(binding=2)
    resources.island_shift_results.bind_to_storage_buffer(binding=3)
    resources.island_reservations.bind_to_storage_buffer(binding=4)
    resources.island_reservation_count.bind_to_storage_buffer(binding=5)
    pack_group_x = (runtime.shape[0] + ISLAND_RESERVATION_LINEAR_LOCAL_SIZE - 1) // ISLAND_RESERVATION_LINEAR_LOCAL_SIZE
    pack_program.run(pack_group_x, 1, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    return int(runtime.shape[0])


def plan_uploaded_falling_island_reservations_from_bridge_runtime(
    pipeline,
    world: "WorldEngine",
    dt: float,
    runtime_capacity: int,
) -> int:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    runtime_capacity = int(runtime_capacity)
    if runtime_capacity <= 0:
        pipeline.upload_falling_island_reservations(
            world,
            np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE),
        )
        return 0
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime planning")
    if pipeline._formal_gpu_frame(world) and "island_runtime" not in bridge.gpu_authoritative_resources:
        raise RuntimeError("GPU motion pipeline requires GPU-authoritative island_runtime for bridge runtime planning")
    if not pipeline._bridge_context_active(world):
        raise RuntimeError("GPU motion pipeline cannot consume island runtime from a separate GL context")

    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    formal_frame = pipeline._formal_gpu_frame(world)
    if formal_frame:
        with pipeline._profile_pass(world, "island_shift_planning"):
            pipeline._ensure_bridge_runtime_reservation_capacity(ctx, resources, runtime_capacity)
            upload_plan = pipeline._cpu_upload_plan(world)
            pipeline._record_cpu_upload_plan(upload_plan)
            use_bridge_state = pipeline._bridge_authoritative_island_state(world)
            if upload_plan["cell_core"]:
                resources.material_tex.write(world.material_id.astype("f4").tobytes())
            if upload_plan["island_id"]:
                resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
            if not use_bridge_state:
                cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
                cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
                pipeline._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
            pipeline._upload_material_rule_params(world, resources)
            program = pipeline.programs["plan_bridge_runtime_falling_island_reservations"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["runtime_capacity"].value = runtime_capacity
            program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
            program["dt"].value = float(dt)
            resources.material_tex.use(location=0)
            resources.island_id_tex.use(location=1)
            bridge.buffers["island_runtime"].bind_to_storage_buffer(binding=0)
            resources.island_reservations.bind_to_storage_buffer(binding=1)
            resources.island_reservation_count.bind_to_storage_buffer(binding=2)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=3)
            resources.material_params.bind_to_storage_buffer(binding=4)
            before_plan_run = None
            if use_bridge_state:
                def rebind_bridge_island_state() -> None:
                    pipeline._bind_bridge_island_state(world, cell_binding=7)

                before_plan_run = rebind_bridge_island_state
            pipeline._run_island_runtime_indirect(
                world,
                resources,
                program,
                "bridge runtime falling island reservation planning",
                runtime_capacity=runtime_capacity,
                before_run=before_plan_run,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        # The returned value is only a buffer capacity upper bound. The actual
        # reservation count remains GPU-authoritative in island_reservation_count.
        return runtime_capacity

    with pipeline._profile_pass(world, "island_runtime_unpack"):
        pipeline._ensure_bridge_runtime_planning_capacity(ctx, resources, runtime_capacity)
        resources.island_reservation_count.write(np.array([0], dtype=np.int32).tobytes())

        unpack_program = pipeline.programs["unpack_bridge_island_runtime"]
        unpack_program["runtime_capacity"].value = runtime_capacity
        bridge.buffers["island_runtime"].bind_to_storage_buffer(binding=0)
        resources.island_ids.bind_to_storage_buffer(binding=1)
        resources.island_bboxes.bind_to_storage_buffer(binding=2)
        resources.island_motion.bind_to_storage_buffer(binding=3)
        bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=4)
        pipeline._run_island_runtime_indirect(
            world,
            resources,
            unpack_program,
            "bridge island runtime unpack",
            runtime_capacity=runtime_capacity,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    with pipeline._profile_pass(world, "island_shift_planning"):
        upload_plan = pipeline._cpu_upload_plan(world)
        pipeline._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        pipeline._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
        pipeline._upload_material_rule_params(world, resources)
        program = pipeline.programs["island_shifts"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["island_count"].value = runtime_capacity
        program["use_island_count_buffer"].value = True
        use_bridge_state = pipeline._bridge_authoritative_island_state(world)
        program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
        program["dt"].value = float(dt)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=4)
        resources.material_params.bind_to_storage_buffer(binding=5)
        before_shift_run = None
        if use_bridge_state:
            def rebind_bridge_island_state() -> None:
                pipeline._bind_bridge_island_state(world, cell_binding=7)

            before_shift_run = rebind_bridge_island_state
        pipeline._run_island_runtime_indirect(
            world,
            resources,
            program,
            "bridge island shift planning",
            runtime_capacity=runtime_capacity,
            before_run=before_shift_run,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    with pipeline._profile_pass(world, "island_reservation_packing"):
        pack_program = pipeline.programs["pack_falling_island_reservations"]
        pack_program["island_count"].value = runtime_capacity
        pack_program["use_island_count_buffer"].value = True
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        resources.island_reservations.bind_to_storage_buffer(binding=4)
        resources.island_reservation_count.bind_to_storage_buffer(binding=5)
        bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=6)
        pipeline._run_island_runtime_indirect(
            world,
            resources,
            pack_program,
            "bridge island reservation packing",
            runtime_capacity=runtime_capacity,
            invocations_per_group=ISLAND_RESERVATION_LINEAR_LOCAL_SIZE,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    # The returned value is only a buffer capacity upper bound. The actual
    # reservation count remains GPU-authoritative in island_reservation_count.
    return runtime_capacity


def plan_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    island_ids: list[int] | None = None,
    motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> np.ndarray:
    reservation_count = pipeline.plan_uploaded_falling_island_reservations(
        world,
        dt,
        island_ids=island_ids,
        motion_overrides=motion_overrides,
    )
    resources = pipeline._ensure_resources(world)
    return pipeline._read_falling_island_reservations(resources, reservation_count)


def upload_falling_island_reservations(pipeline, world: "WorldEngine", reservations: np.ndarray) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    resources = pipeline._ensure_resources(world)
    pipeline._write_dynamic_buffer(ctx, resources, "island_reservations", reservations)
    resources.island_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())


def resolve_falling_island_reservations(pipeline, world: "WorldEngine", reservations: np.ndarray) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    if len(reservations) == 0:
        pipeline.upload_falling_island_reservations(world, reservations)
        return reservations
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline._write_dynamic_buffer(ctx, resources, "island_reservations", reservations)
    resources.island_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())
    pipeline._dispatch_resolve_falling_island_reservations(world, resources, int(len(reservations)))
    pipeline.publish_bridge_falling_island_reservations(world, int(len(reservations)))
    pipeline.publish_bridge_falling_island_runtime_from_reservations(world, int(len(reservations)))
    resolved = pipeline._read_falling_island_reservations(resources, int(len(reservations)))
    resources.island_reservation_count.write(np.array([len(resolved)], dtype=np.int32).tobytes())
    return resolved


def resolve_uploaded_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    reservation_count: int,
) -> bool:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    reservation_count = int(reservation_count)
    if reservation_count <= 0:
        resources = pipeline._ensure_resources(world)
        resources.island_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    if not pipeline._formal_gpu_frame(world):
        resources.island_reservation_count.write(np.array([reservation_count], dtype=np.int32).tobytes())
    pipeline._dispatch_resolve_falling_island_reservations(world, resources, reservation_count)
    if not pipeline._formal_gpu_frame(world):
        pipeline.publish_bridge_falling_island_reservations(world, reservation_count)
    pipeline.publish_bridge_falling_island_runtime_from_reservations(world, reservation_count)
    return True


def _dispatch_resolve_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    reservation_count: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    formal_frame = pipeline._formal_gpu_frame(world)
    pipeline._upload_material_rule_params(world, resources)
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["cell_core"]:
        resources.material_tex.write(world.material_id.astype("f4").tobytes())
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if formal_frame:
        pipeline._build_falling_island_apply_dispatch(
            world,
            resources,
            reservation_count=int(reservation_count),
            operation=2,
        )
    pipeline._load_authoritative_bridge_inputs(
        world,
        resources,
        cell_group_x,
        cell_group_y,
        use_existing_active_tile_dispatch=formal_frame,
    )
    pipeline._dispatch_index_falling_island_reservation_sources(
        world,
        resources,
        reservation_count=int(reservation_count),
    )
    with pipeline._profile_pass(world, "island_reservation_resolve"):
        program = pipeline.programs["resolve_falling_island_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["reservation_count"].value = int(reservation_count)
        program["use_reservation_count_buffer"].value = bool(formal_frame)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.island_reservation_count.bind_to_storage_buffer(binding=2)
        resources.island_reservation_source_index.bind_to_storage_buffer(binding=3)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        if formal_frame:
            pipeline._run_island_reservation_indirect(
                world,
                resources,
                program,
                "falling island reservation resolve",
                reservation_capacity=int(reservation_count),
            )
        else:
            group_x = (int(reservation_count) + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if not formal_frame:
        ctx.finish()


def _read_falling_island_reservations(pipeline, resources: GPUMotionResources, reservation_count: int) -> np.ndarray:
    reservation_count = int(reservation_count)
    if reservation_count <= 0:
        return np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
    return np.frombuffer(
        resources.island_reservations.read(size=reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize),
        dtype=FALLING_ISLAND_RESERVATION_DTYPE,
        count=reservation_count,
    ).copy()


def resolve_falling_island_shifts(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    island_ids: list[int] | None = None,
    motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> dict[int, tuple[int, int]]:
    reservations = pipeline.plan_falling_island_reservations(
        world,
        dt,
        island_ids=island_ids,
        motion_overrides=motion_overrides,
    )
    return {
        int(record["island_id"]): (int(record["reserved_shift"][0]), int(record["reserved_shift"][1]))
        for record in reservations
    }
