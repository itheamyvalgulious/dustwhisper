from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE,
    POWDER_RESERVATION_DTYPE,
    POWDER_RESOLVE_BLOCKED
)


def _clear_powder_target_winners_for_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["clear_powder_target_winners_for_reservations"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.powder_reservations.bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.powder_target_winner.bind_to_storage_buffer(binding=2)
    pipeline._run_powder_reservation_indirect(
        world,
        resources,
        program,
        "powder target winner reservation clear",
    )
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _clear_powder_apply_index_for_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["clear_powder_apply_index_for_reservations"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.powder_reservations.bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.powder_target_winner.bind_to_storage_buffer(binding=2)
    resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
    resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
    pipeline._run_powder_reservation_indirect(
        world,
        resources,
        program,
        "powder apply index reservation clear",
    )
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _clear_powder_apply_index_for_active_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["clear_powder_apply_index_for_active_tiles"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.powder_target_winner.bind_to_storage_buffer(binding=2)
    resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
    resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
    pipeline._run_active_tile_indirect(program, resources, "powder apply affected tile index clear")
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _run_powder_targets(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    group_x: int,
    group_y: int,
    dt: float,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    del group_x, group_y
    program = pipeline.programs["powder_targets"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["phase_powder"].value = int(Phase.POWDER)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["dt"].value = float(dt)
    use_bridge_blockers = pipeline._bridge_authoritative_cell_blockers(world)
    program["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
    use_liquid_flow_intent = (
        pipeline._formal_gpu_frame(world)
        and "liquid_flow_intent" in world.bridge.gpu_authoritative_resources
        and pipeline._bridge_context_active(world)
    )
    program["use_liquid_flow_intent"].value = bool(use_liquid_flow_intent)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    if use_bridge_blockers:
        pipeline._bind_bridge_cell_blockers(world, cell_binding=8)
    resources.material_tex.use(location=1)
    resources.phase_tex.use(location=2)
    resources.velocity_tex.use(location=3)
    resources.active_tile_tex.use(location=4)
    resources.entity_id_tex.use(location=6)
    resources.displaced_tex.use(location=7)
    if use_liquid_flow_intent:
        world.bridge.textures["liquid_flow_intent"].use(location=8)
    else:
        resources.velocity_tex.use(location=8)
    resources.powder_target_tex.bind_to_image(5, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, "powder target generation")
    pipeline._sync_compute_writes(ctx)


def _run_generate_powder_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    dt: float,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    generate = pipeline.programs["generate_powder_reservations"]
    generate["cell_grid_size"].value = (world.width, world.height)
    generate["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    generate["tile_size"].value = world.active.tile_size
    generate["phase_powder"].value = int(Phase.POWDER)
    generate["phase_liquid"].value = int(Phase.LIQUID)
    generate["dt"].value = float(dt)
    resources.powder_reservations.bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.material_params.bind_to_storage_buffer(binding=2)
    resources.material_contact_params.bind_to_storage_buffer(binding=3)
    resources.active_tile_count.bind_to_storage_buffer(binding=4)
    resources.active_tile_list.bind_to_storage_buffer(binding=5)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.velocity_tex.use(location=2)
    resources.active_tile_tex.use(location=3)
    resources.powder_target_tex.use(location=4)
    pipeline._run_active_tile_indirect(generate, resources, "powder reservation generation")
    pipeline._sync_compute_writes(ctx)


def plan_powder_reservations(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    solve_tile_mask: np.ndarray,
    solve_cell_mask: np.ndarray,
) -> np.ndarray:
    powder_targets = pipeline.step(world, dt, solve_tile_mask=solve_tile_mask)
    reservations = pipeline._build_powder_reservations(world, solve_cell_mask, powder_targets, dt)
    pipeline.upload_powder_reservations(world, reservations)
    return reservations


def upload_powder_reservations(pipeline, world: "WorldEngine", reservations: np.ndarray) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    resources = pipeline._ensure_resources(world)
    pipeline._write_dynamic_buffer(ctx, resources, "powder_reservations", reservations)
    resources.powder_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())


def resolve_and_apply_powders(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    solve_tile_mask: np.ndarray,
) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "powder_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "powder_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "powder_targets"):
        pipeline._run_powder_targets(world, resources, group_x, group_y, dt)
    with pipeline._profile_pass(world, "powder_buffer_prepare"):
        cell_count = world.width * world.height
        pipeline._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "powder_reservations",
            cell_count * POWDER_RESERVATION_DTYPE.itemsize,
        )
        pipeline._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "powder_target_winner",
            cell_count * np.dtype(np.int32).itemsize,
        )
        pipeline._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "powder_apply_incoming",
            cell_count * np.dtype(np.int32).itemsize,
        )
        pipeline._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "powder_apply_outgoing",
            cell_count * np.dtype(np.int32).itemsize,
        )
        resources.powder_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
    with pipeline._profile_pass(world, "powder_generate"):
        pipeline._run_generate_powder_reservations(world, resources, dt)
    with pipeline._profile_pass(world, "powder_index_targets"):
        pipeline._clear_powder_target_winners_for_reservations(world, resources)

        index_winners = pipeline.programs["index_powder_target_winners"]
        index_winners["cell_grid_size"].value = (world.width, world.height)
        index_winners["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        index_winners["tile_size"].value = world.active.tile_size
        index_winners["phase_powder"].value = int(Phase.POWDER)
        index_winners["phase_liquid"].value = int(Phase.LIQUID)
        index_winners["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        use_bridge_blockers = pipeline._bridge_authoritative_cell_blockers(world)
        index_winners["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        if use_bridge_blockers:
            pipeline._bind_bridge_cell_blockers(world, cell_binding=8)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.active_tile_tex.use(location=3)
        resources.entity_id_tex.use(location=5)
        resources.displaced_tex.use(location=6)
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            index_winners,
            "powder target winner indexing",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    with pipeline._profile_pass(world, "powder_resolve"):
        resolve = pipeline.programs["resolve_powder_reservations"]
        resolve["cell_grid_size"].value = (world.width, world.height)
        resolve["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        resolve["tile_size"].value = world.active.tile_size
        resolve["phase_powder"].value = int(Phase.POWDER)
        resolve["phase_liquid"].value = int(Phase.LIQUID)
        resolve["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        use_bridge_blockers = pipeline._bridge_authoritative_cell_blockers(world)
        resolve["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.material_params.bind_to_storage_buffer(binding=2)
        resources.material_contact_params.bind_to_storage_buffer(binding=3)
        resources.powder_target_winner.bind_to_storage_buffer(binding=4)
        if use_bridge_blockers:
            pipeline._bind_bridge_cell_blockers(world, cell_binding=8)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.active_tile_tex.use(location=3)
        resources.entity_id_tex.use(location=5)
        resources.displaced_tex.use(location=6)
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            resolve,
            "powder reservation resolve",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if pipeline._formal_gpu_frame(world):
        pipeline.publish_bridge_powder_reservations(world, world.width * world.height)
        pipeline._dispatch_index_powder_apply(world, resources)
        pipeline._dispatch_apply_powder_reservations(
            world,
            resources,
            None,
            inputs_already_loaded=True,
        )
        return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
    reservation_count = int(np.frombuffer(resources.powder_reservation_count.read(size=4), dtype=np.int32, count=1)[0])
    reservation_count = max(0, min(reservation_count, world.width * world.height))
    pipeline._dispatch_index_powder_apply(world, resources)
    pipeline._dispatch_apply_powder_reservations(world, resources, reservation_count)
    return pipeline._read_powder_reservations(resources, reservation_count)


def _dispatch_apply_powder_fast_path(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    group_x: int,
    group_y: int,
    dt: float,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["apply_powder_fast_path"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["phase_powder"].value = int(Phase.POWDER)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["max_powder_step"].value = 3
    program["dt"].value = float(dt)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_contact_params.bind_to_storage_buffer(binding=1)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=11)
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
    resources.active_tile_tex.use(location=10)
    resources.material_out_tex.bind_to_image(0, read=False, write=True)
    resources.phase_out_tex.bind_to_image(1, read=False, write=True)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
    resources.temp_out_tex.bind_to_image(4, read=False, write=True)
    resources.timer_out_tex.bind_to_image(5, read=False, write=True)
    resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    pipeline._publish_bridge_outputs(world, resources, output_textures=True)
    world.bridge._ensure_active_scheduler_programs()
    world.bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    resources.powder_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
    pipeline.publish_bridge_powder_reservations(world, 0)
    pipeline.last_cpu_mirror_downloaded = False


def apply_powder_reservations(pipeline, world: "WorldEngine", reservations: np.ndarray) -> bool:
    ctx = world.bridge.ctx
    if ctx is None:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.upload_powder_reservations(world, reservations)
    pipeline._dispatch_index_powder_apply(world, resources)
    pipeline._dispatch_apply_powder_reservations(world, resources, int(len(reservations)))
    return True


def _dispatch_index_powder_apply(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_count = int(world.width * world.height)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_target_winner", cell_count * np.dtype(np.int32).itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_incoming", cell_count * np.dtype(np.int32).itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_outgoing", cell_count * np.dtype(np.int32).itemsize)
    with pipeline._profile_pass(world, "powder_index_apply"):
        if pipeline._formal_gpu_frame(world):
            pipeline._build_powder_apply_dispatch(world, resources)
            pipeline._clear_powder_apply_index_for_active_tiles(world, resources)
            pipeline._clear_powder_apply_index_for_reservations(world, resources)
        else:
            clear_apply = pipeline.programs["clear_powder_apply_index"]
            clear_apply["cell_count"].value = cell_count
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=0)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=1)
            clear_apply.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

            clear_winners = pipeline.programs["clear_powder_target_winners"]
            clear_winners["cell_count"].value = cell_count
            resources.powder_target_winner.bind_to_storage_buffer(binding=0)
            clear_winners.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        index_winners = pipeline.programs["index_powder_apply_winners"]
        index_winners["cell_grid_size"].value = (world.width, world.height)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            index_winners,
            "powder apply winner indexing",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        fill_index = pipeline.programs["fill_powder_apply_index"]
        fill_index["cell_grid_size"].value = (world.width, world.height)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            fill_index,
            "powder apply index fill",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def _dispatch_apply_powder_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    reservation_count: int | None,
    *,
    inputs_already_loaded: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    formal_frame = pipeline._formal_gpu_frame(world)
    pipeline._upload_powder_apply_state(world, resources)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if formal_frame:
        pipeline._build_powder_apply_dispatch(world, resources)
    if not (formal_frame and inputs_already_loaded):
        pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=formal_frame,
        )
    with pipeline._profile_pass(world, "powder_apply_main"):
        program = pipeline.programs["apply_powder_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program_members = getattr(program, "_members", {})
        if "reservation_count" in program_members:
            program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
        if "use_reservation_count_buffer" in program_members:
            program["use_reservation_count_buffer"].value = reservation_count is None
        if "use_active_tile_dispatch" in program_members:
            program["use_active_tile_dispatch"].value = bool(formal_frame)
        if "skip_untouched_original_stores" in program_members:
            program["skip_untouched_original_stores"].value = bool(formal_frame)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
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
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.timer_out_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        if formal_frame:
            pipeline._run_active_tile_indirect(program, resources, "powder reservation apply")
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "powder_apply_aux"):
        aux_program = pipeline.programs["apply_powder_reservation_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        aux_program["tile_size"].value = int(world.active.tile_size)
        aux_members = getattr(aux_program, "_members", {})
        if "reservation_count" in aux_members:
            aux_program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
        if "use_reservation_count_buffer" in aux_members:
            aux_program["use_reservation_count_buffer"].value = reservation_count is None
        if "use_active_tile_dispatch" in aux_members:
            aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
        resources.active_tile_count.bind_to_storage_buffer(binding=6)
        resources.active_tile_list.bind_to_storage_buffer(binding=7)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        if formal_frame:
            pipeline._run_active_tile_indirect(aux_program, resources, "powder reservation aux apply")
        else:
            aux_program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    pipeline._publish_bridge_outputs(
        world,
        resources,
        output_textures=True,
        active_tile_indirect=formal_frame,
        use_powder_apply_touch_sources=formal_frame,
    )
    pipeline._refresh_authoritative_active_scheduler_after_apply(world, "active_refresh_after_powder")
    pipeline.last_cpu_mirror_downloaded = not formal_frame
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_powder_apply_state(world, resources)


def _read_powder_reservations(pipeline, resources: GPUMotionResources, reservation_count: int) -> np.ndarray:
    if reservation_count <= 0:
        return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
    return np.frombuffer(
        resources.powder_reservations.read(size=reservation_count * POWDER_RESERVATION_DTYPE.itemsize),
        dtype=POWDER_RESERVATION_DTYPE,
        count=reservation_count,
    ).copy()


def _build_powder_reservations(
    pipeline,
    world: "WorldEngine",
    solve_cell_mask: np.ndarray,
    powder_targets: np.ndarray,
    dt: float,
) -> np.ndarray:
    material_table = world.bridge.shadow_typed_tables["material_table"]
    reservations: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[float, float], int]] = []
    for y in range(world.height - 2, -1, -1):
        active_xs = np.flatnonzero(solve_cell_mask[y])
        if active_xs.size == 0:
            continue
        for x in active_xs.tolist():
            material_id = int(world.material_id[y, x])
            phase_id = int(world.phase[y, x])
            if material_id <= 0 or phase_id not in (int(Phase.POWDER), int(Phase.LIQUID)):
                continue
            max_step = 0
            if material_id < material_table.shape[0]:
                max_step = int(material_table[material_id]["max_dda_step"])
            velocity = world.velocity[y, x]
            frame_delta_x = float(velocity[0]) * float(dt)
            frame_delta_y = float(velocity[1]) * float(dt)
            desired_dx = int(np.clip(np.rint(frame_delta_x), -max_step, max_step))
            desired_dy = int(np.clip(np.rint(frame_delta_y), -max_step, max_step))
            reservations.append(
                (
                    (int(x), int(y)),
                    (int(x) + desired_dx, int(y) + desired_dy),
                    (int(powder_targets[y, x, 0]), int(powder_targets[y, x, 1])),
                    (int(x), int(y)),
                    (float(velocity[0]), float(velocity[1])),
                    material_id,
                    POWDER_RESOLVE_BLOCKED,
                )
            )
    packed = np.zeros((len(reservations),), dtype=POWDER_RESERVATION_DTYPE)
    for index, (
        source_xy,
        desired_target_xy,
        reserved_target_xy,
        resolved_target_xy,
        velocity_xy,
        material_id,
        resolve_state,
    ) in enumerate(reservations):
        packed[index]["source_xy"] = np.asarray(source_xy, dtype=np.int32)
        packed[index]["desired_target_xy"] = np.asarray(desired_target_xy, dtype=np.int32)
        packed[index]["reserved_target_xy"] = np.asarray(reserved_target_xy, dtype=np.int32)
        packed[index]["resolved_target_xy"] = np.asarray(resolved_target_xy, dtype=np.int32)
        packed[index]["velocity_xy"] = np.asarray(velocity_xy, dtype=np.float32)
        packed[index]["material_id"] = int(material_id)
        packed[index]["resolve_state"] = int(resolve_state)
    return packed
