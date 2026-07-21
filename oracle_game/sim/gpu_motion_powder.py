from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE,
    POWDER_RESERVATION_LOCAL_SIZE,
    POWDER_RESERVATION_DTYPE,
    POWDER_RESOLVE_BLOCKED,
)


_COMPACT_POWDER_RESERVATION_ITEMSIZE = 24


def _source_indexed_powder_apply_enabled(pipeline, world: "WorldEngine") -> bool:
    return bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._powder_source_indexed_direct_apply_enabled
        and not pipeline._powder_direct_bridge_apply_enabled
        and pipeline._bridge_authoritative_powder_inputs(world)
        and pipeline._bridge_context_active(world)
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
    )


def _compact_powder_reservation_safe(pipeline, world: "WorldEngine") -> bool:
    if not (
        pipeline._powder_compact_reservation_enabled
        and _source_indexed_powder_apply_enabled(pipeline, world)
        and pipeline._bridge_authoritative_cell_blockers(world)
    ):
        return False
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is None or material_table.shape[0] == 0:
        return False
    steps = material_table["max_dda_step"].astype(np.int64, copy=False)
    min_step = int(steps.min())
    max_step = int(steps.max())
    if min_step < 0 or max_step > 32768:
        return False
    return (
        world.width > 0
        and world.height > 0
        and world.width - 1 + max_step <= 32767
        and world.height - 1 + max_step <= 32767
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
    resources.cell_state_tex.use(location=1)
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
    *,
    compact_reservations: bool = False,
    nontrivial_resolve_worklist: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    if nontrivial_resolve_worklist:
        generate_name = "generate_powder_reservations_compact_nontrivial_worklist"
    elif compact_reservations:
        generate_name = "generate_powder_reservations_compact"
    else:
        generate_name = "generate_powder_reservations"
    generate = pipeline.programs[generate_name]
    generate["cell_grid_size"].value = (world.width, world.height)
    generate["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    generate["tile_size"].value = world.active.tile_size
    generate["phase_powder"].value = int(Phase.POWDER)
    generate["phase_liquid"].value = int(Phase.LIQUID)
    generate["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    generate["dt"].value = float(dt)
    use_bridge_blockers = pipeline._bridge_authoritative_cell_blockers(world)
    generate["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
    use_liquid_flow_intent = (
        pipeline._formal_gpu_frame(world)
        and "liquid_flow_intent" in world.bridge.gpu_authoritative_resources
        and pipeline._bridge_context_active(world)
    )
    generate["use_liquid_flow_intent"].value = bool(use_liquid_flow_intent)
    generate["precompute_fallback_blockers"].value = bool(
        pipeline._powder_precomputed_fallback_blockers_enabled
    )
    if nontrivial_resolve_worklist:
        generate["apply_workgroups_per_tile"].value = int(
            pipeline._active_tile_workgroups_per_tile(world)
        )
    (
        resources.powder_compact_reservations
        if compact_reservations
        else resources.powder_reservations
    ).bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.material_params.bind_to_storage_buffer(binding=2)
    resources.material_contact_params.bind_to_storage_buffer(binding=3)
    resources.active_tile_count.bind_to_storage_buffer(binding=4)
    resources.active_tile_list.bind_to_storage_buffer(binding=5)
    resources.powder_target_winner.bind_to_storage_buffer(binding=6)
    if nontrivial_resolve_worklist:
        resources.powder_direct_apply_unsafe.bind_to_storage_buffer(binding=7)
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=12)
        resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=13)
        resources.island_materialization_candidate_tile_list.bind_to_storage_buffer(binding=14)
        resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=15)
    if use_bridge_blockers:
        pipeline._bind_bridge_cell_blockers(world, cell_binding=8)
    if nontrivial_resolve_worklist:
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=9)
    resources.cell_state_tex.use(location=0)
    resources.velocity_tex.use(location=2)
    resources.active_tile_tex.use(location=3)
    if use_liquid_flow_intent:
        world.bridge.textures["liquid_flow_intent"].use(location=4)
    else:
        resources.velocity_tex.use(location=4)
    resources.entity_id_tex.use(location=5)
    resources.displaced_tex.use(location=6)
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
        if not pipeline._bridge_authoritative_powder_inputs(world):
            pipeline._load_authoritative_bridge_inputs(
                world,
                resources,
                group_x,
                group_y,
                use_existing_active_tile_dispatch=True,
                load_gas_inputs=False,
            )
    with pipeline._profile_pass(world, "powder_buffer_prepare"):
        cell_count = world.width * world.height
        source_indexed_apply = _source_indexed_powder_apply_enabled(pipeline, world)
        compact_reservations = _compact_powder_reservation_safe(pipeline, world)
        lazy_compact_expand = bool(
            compact_reservations
            and pipeline._powder_compact_reservation_lazy_expand_enabled
        )
        provisional_moving_worklist = bool(
            pipeline._powder_provisional_moving_worklist_enabled
            and source_indexed_apply
            and compact_reservations
            and lazy_compact_expand
            and pipeline._powder_trivial_blocked_classification_enabled
            and pipeline._powder_apply_tile_workgroup_dedup_enabled
        )
        nontrivial_resolve_worklist = bool(
            pipeline._powder_nontrivial_resolve_worklist_enabled
            and provisional_moving_worklist
            and pipeline._powder_precomputed_fallback_blockers_enabled
        )
        if not lazy_compact_expand:
            pipeline._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_reservations",
                cell_count * POWDER_RESERVATION_DTYPE.itemsize,
            )
        if compact_reservations:
            pipeline._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_compact_reservations",
                cell_count * _COMPACT_POWDER_RESERVATION_ITEMSIZE,
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
        if provisional_moving_worklist:
            resources.powder_provisional_moving_count.write(
                np.array([0], dtype=np.uint32).tobytes()
            )
        if nontrivial_resolve_worklist:
            resources.powder_direct_apply_unsafe.write(
                np.array([0], dtype=np.uint32).tobytes()
            )
    if nontrivial_resolve_worklist:
        with pipeline._profile_pass(world, "powder_apply_dispatch_clear"):
            tile_count = int(world.active.tile_width * world.active.tile_height)
            clear_dispatch = pipeline.programs["clear_powder_affected_tile_dispatch"]
            clear_dispatch["tile_count"].value = tile_count
            resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
            resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=1)
            resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=2)
            clear_dispatch.run((tile_count + 255) // 256, 1, 1)
            ctx.memory_barrier(
                ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0)
            )
    with pipeline._profile_pass(world, "powder_target_clear"):
        clear_local_size = 64 if pipeline._powder_target_clear_local64_enabled else LOCAL_SIZE
        clear_winners = pipeline.programs[
            "clear_powder_target_winners_local64"
            if pipeline._powder_target_clear_local64_enabled
            else "clear_powder_target_winners"
        ]
        clear_winners["cell_count"].value = cell_count
        resources.powder_target_winner.bind_to_storage_buffer(binding=0)
        clear_winners.run((cell_count + clear_local_size - 1) // clear_local_size, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    with pipeline._profile_pass(world, "powder_generate"):
        pipeline._run_generate_powder_reservations(
            world,
            resources,
            dt,
            compact_reservations=compact_reservations,
            nontrivial_resolve_worklist=nontrivial_resolve_worklist,
        )
    if nontrivial_resolve_worklist:
        (
            resources.active_tile_count,
            resources.island_materialization_candidate_tile_count,
        ) = (
            resources.island_materialization_candidate_tile_count,
            resources.active_tile_count,
        )
        (
            resources.active_tile_list,
            resources.island_materialization_candidate_tile_list,
        ) = (
            resources.island_materialization_candidate_tile_list,
            resources.active_tile_list,
        )
        (
            resources.active_tile_dispatch_args,
            resources.island_materialization_candidate_dispatch_args,
        ) = (
            resources.island_materialization_candidate_dispatch_args,
            resources.active_tile_dispatch_args,
        )
    formal_frame = pipeline._formal_gpu_frame(world)
    build_apply_dispatch = bool(formal_frame)
    if build_apply_dispatch and not nontrivial_resolve_worklist:
        with pipeline._profile_pass(world, "powder_apply_dispatch_clear"):
            tile_count = int(world.active.tile_width * world.active.tile_height)
            clear_dispatch = pipeline.programs["clear_powder_affected_tile_dispatch"]
            clear_dispatch["tile_count"].value = tile_count
            resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=1)
            resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            clear_dispatch.run((tile_count + 255) // 256, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))
    with pipeline._profile_pass(world, "powder_resolve"):
        dedup_resolve = False
        if (
            build_apply_dispatch
            and pipeline._powder_apply_tile_workgroup_dedup_enabled
            and not nontrivial_resolve_worklist
        ):
            dedup_resolve = True
        if compact_reservations:
            resolve_name = (
                "resolve_powder_reservations_compact_tile_dedup"
                if dedup_resolve
                else "resolve_powder_reservations_compact"
            )
            if pipeline._powder_trivial_blocked_classification_enabled:
                resolve_name += "_trivial_blocked"
        else:
            resolve_name = (
                "resolve_powder_reservations_tile_dedup"
                if dedup_resolve
                else "resolve_powder_reservations"
            )
        if provisional_moving_worklist:
            resolve_name += "_moving_worklist"
        if nontrivial_resolve_worklist:
            resolve_name += "_nontrivial_worklist"
        resolve = pipeline.programs[resolve_name]
        resolve["cell_grid_size"].value = (world.width, world.height)
        resolve["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        resolve["tile_size"].value = world.active.tile_size
        resolve["phase_powder"].value = int(Phase.POWDER)
        resolve["phase_liquid"].value = int(Phase.LIQUID)
        resolve["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resolve["generated_sources_prevalidated"].value = True
        resolve["reuse_generated_reserved_path"].value = bool(
            pipeline._powder_generated_path_reuse_enabled
        )
        resolve["use_precomputed_fallback_blockers"].value = bool(
            pipeline._powder_precomputed_fallback_blockers_enabled
        )
        resolve["build_apply_dispatch"].value = build_apply_dispatch
        resolve["apply_workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
        use_bridge_blockers = pipeline._bridge_authoritative_cell_blockers(world)
        resolve["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
        (
            resources.powder_compact_reservations
            if compact_reservations
            else resources.powder_reservations
        ).bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.material_params.bind_to_storage_buffer(binding=2)
        resources.material_contact_params.bind_to_storage_buffer(binding=3)
        resources.powder_target_winner.bind_to_storage_buffer(binding=4)
        affected_tile_count = resources.active_tile_count
        affected_tile_list = resources.active_tile_list
        affected_dispatch_args = resources.active_tile_dispatch_args
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=5)
        affected_tile_count.bind_to_storage_buffer(binding=6)
        affected_tile_list.bind_to_storage_buffer(binding=7)
        affected_dispatch_args.bind_to_storage_buffer(binding=12)
        if provisional_moving_worklist:
            resources.powder_provisional_moving_count.bind_to_storage_buffer(binding=13)
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=14)
        if use_bridge_blockers:
            pipeline._bind_bridge_cell_blockers(world, cell_binding=8)
        if nontrivial_resolve_worklist:
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=9)
            resources.powder_direct_apply_unsafe.bind_to_storage_buffer(binding=15)
        resources.cell_state_tex.use(location=0)
        resources.active_tile_tex.use(location=3)
        resources.entity_id_tex.use(location=5)
        resources.displaced_tex.use(location=6)
        def bind_apply_dispatch_outputs() -> None:
            affected_tile_count.bind_to_storage_buffer(binding=6)
            affected_tile_list.bind_to_storage_buffer(binding=7)

        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            resolve,
            "powder reservation resolve",
            before_run=bind_apply_dispatch_outputs,
            count_buffer=(
                resources.powder_direct_apply_unsafe
                if nontrivial_resolve_worklist
                else None
            ),
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if formal_frame:
        if source_indexed_apply:
            _dispatch_source_indexed_direct_apply(
                pipeline,
                world,
                resources,
                compact_reservations=compact_reservations,
                lazy_compact_expand=lazy_compact_expand,
                provisional_moving_worklist=provisional_moving_worklist,
            )
        else:
            pipeline._dispatch_index_powder_apply(
                world,
                resources,
                apply_dispatch_already_built=True,
            )
            pipeline._dispatch_apply_powder_reservations(
                world,
                resources,
                None,
                inputs_already_loaded=True,
                allow_aux_index_scratch=True,
            )
        if lazy_compact_expand:
            pipeline.publish_bridge_compact_powder_reservations(
                world,
                world.width * world.height,
            )
        else:
            pipeline.publish_bridge_powder_reservations(world, world.width * world.height)
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
    resources.cell_state_tex.use(location=0)
    resources.velocity_tex.use(location=3)
    resources.temp_tex.use(location=4)
    resources.timer_tex.use(location=5)
    resources.integrity_tex.use(location=6)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.active_tile_tex.use(location=10)
    resources.cell_state_out_tex.bind_to_image(0, read=False, write=True)
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
    pipeline._dispatch_apply_powder_reservations(
        world,
        resources,
        int(len(reservations)),
        # Keep the historical texture+terminal path for normal uploaded
        # reservations. The packed indices are only needed by the opt-in
        # direct bridge candidate (and are otherwise an observable aux format
        # change for callers that use this API).
        allow_aux_index_scratch=bool(pipeline._powder_direct_bridge_apply_enabled),
    )
    return True


def _dispatch_index_powder_apply(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    apply_dispatch_already_built: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_count = int(world.width * world.height)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_target_winner", cell_count * np.dtype(np.int32).itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_incoming", cell_count * np.dtype(np.int32).itemsize)
    pipeline._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_outgoing", cell_count * np.dtype(np.int32).itemsize)
    epoch_enabled = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._powder_apply_index_epoch_enabled
    )
    if epoch_enabled:
        _prepare_powder_apply_epoch(pipeline, world, resources)
    with pipeline._profile_pass(world, "powder_index_apply"):
        if pipeline._formal_gpu_frame(world):
            if not apply_dispatch_already_built:
                pipeline._build_powder_apply_dispatch(world, resources)
            if not epoch_enabled:
                pipeline._clear_powder_apply_index_for_active_tiles(world, resources)
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

        if not pipeline._formal_gpu_frame(world):
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

        fill_index = pipeline.programs[
            "fill_powder_apply_index" if epoch_enabled else "fill_powder_apply_index_legacy"
        ]
        fill_index["cell_grid_size"].value = (world.width, world.height)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
        if epoch_enabled:
            fill_index["powder_apply_epoch"].value = int(pipeline._powder_apply_epoch)
        if epoch_enabled:
            resources.powder_apply_epoch.bind_to_storage_buffer(binding=5)
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            fill_index,
            "powder apply index fill",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)


def _prepare_powder_apply_epoch(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    """Advance the formal apply stamp without dispatching the full index pass."""
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_count = int(world.width * world.height)
    pipeline._ensure_dynamic_buffer_capacity(
        ctx,
        resources,
        "powder_apply_epoch",
        cell_count * 2 * np.dtype(np.uint32).itemsize,
    )
    if pipeline._powder_apply_epoch >= 0xFFFFFFFF:
        clear_apply = pipeline.programs["clear_powder_apply_index"]
        clear_apply["cell_count"].value = cell_count
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=0)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=1)
        clear_apply.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
        resources.powder_apply_epoch.write(bytes(cell_count * 2 * np.dtype(np.uint32).itemsize))
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        pipeline._powder_apply_epoch = 1
    else:
        pipeline._powder_apply_epoch += 1


def _dispatch_source_indexed_direct_apply(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    compact_reservations: bool = False,
    lazy_compact_expand: bool = False,
    provisional_moving_worklist: bool = False,
) -> None:
    """Apply generated formal reservations directly from disjoint source/target cells."""
    ctx = world.bridge.ctx
    assert ctx is not None
    with pipeline._profile_pass(world, "powder_source_indexed_direct_apply"):
        if provisional_moving_worklist:
            program_name = (
                "apply_powder_reservations_source_indexed_direct_"
                "compact_lazy_moving_worklist"
            )
        elif lazy_compact_expand:
            program_name = "apply_powder_reservations_source_indexed_direct_compact_lazy"
        elif compact_reservations:
            program_name = "apply_powder_reservations_source_indexed_direct_compact"
        else:
            program_name = "apply_powder_reservations_source_indexed_direct"
        program = pipeline.programs[program_name]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)

        def bind_direct_apply_state() -> None:
            # Indirect dispatch construction uses SSBO bindings 6 and 7.
            # Restore every direct-apply binding after it has completed.
            (
                resources.powder_compact_reservations
                if compact_reservations
                else resources.powder_reservations
            ).bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            resources.material_contact_params.bind_to_storage_buffer(binding=2)
            resources.powder_target_winner.bind_to_storage_buffer(binding=3)
            world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=4)
            world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=5)
            world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=6)
            world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=7)
            world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=8)
            if compact_reservations and not lazy_compact_expand:
                resources.powder_reservations.bind_to_storage_buffer(binding=9)
            if provisional_moving_worklist:
                resources.powder_provisional_moving_count.bind_to_storage_buffer(binding=10)
                resources.powder_apply_incoming.bind_to_storage_buffer(binding=11)
            world.bridge.textures["material"].bind_to_image(0, read=False, write=True)

        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            program,
            "powder source-indexed direct apply",
            before_run=bind_direct_apply_state,
            count_buffer=(
                resources.powder_provisional_moving_count
                if provisional_moving_worklist
                else None
            ),
        )
        pipeline._sync_compute_writes(ctx)
    world.bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )
    pipeline._refresh_authoritative_active_scheduler_after_apply(
        world,
        "active_refresh_after_powder",
    )


def _dispatch_apply_powder_reservations(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    reservation_count: int | None,
    *,
    inputs_already_loaded: bool = False,
    allow_aux_index_scratch: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    formal_frame = pipeline._formal_gpu_frame(world)
    epoch_enabled = bool(
        formal_frame and pipeline._powder_apply_index_epoch_enabled
    )
    use_bridge_inputs = pipeline._bridge_authoritative_powder_inputs(world)
    use_packed_powder_aux = bool(
        formal_frame
        and allow_aux_index_scratch
        and use_bridge_inputs
        and pipeline._powder_aux_index_scratch_fusion_enabled
    )
    pipeline._upload_powder_apply_state(world, resources)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    # Formal callers build a conservative source/target tile superset while
    # indexing. Conflict resolution can only remove targets from that set.
    if not (formal_frame and inputs_already_loaded):
        pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=formal_frame,
            load_gas_inputs=False,
        )
    direct_bridge_outputs = bool(
        formal_frame
        and use_packed_powder_aux
        and pipeline._powder_direct_bridge_apply_enabled
        and pipeline._bridge_context_active(world)
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
    )
    if direct_bridge_outputs:
        _snapshot_powder_direct_apply_sources(pipeline, world, resources)
    with pipeline._profile_pass(world, "powder_apply_main"):
        if direct_bridge_outputs:
            program_name = (
                "apply_powder_reservations_bridge"
                if epoch_enabled
                else "apply_powder_reservations_bridge_legacy"
            )
        else:
            program_name = (
                "apply_powder_reservations"
                if epoch_enabled
                else "apply_powder_reservations_legacy"
            )
        program = pipeline.programs[program_name]
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
        if "use_bridge_inputs" in program_members:
            program["use_bridge_inputs"].value = bool(use_bridge_inputs)
        if "pack_aux_in_apply_indices" in program_members:
            program["pack_aux_in_apply_indices"].value = bool(use_packed_powder_aux)
        if "write_cell_core" in program_members:
            program["write_cell_core"].value = not bool(
                getattr(world, "phase_c_defer_cell_publish", False)
            )
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
        if epoch_enabled:
            program["powder_apply_epoch"].value = int(pipeline._powder_apply_epoch)
        if epoch_enabled:
            resources.powder_apply_epoch.bind_to_storage_buffer(binding=15)
        resources.active_tile_count.bind_to_storage_buffer(binding=6)
        resources.active_tile_list.bind_to_storage_buffer(binding=7)
        if use_bridge_inputs:
            world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=8)
        if use_packed_powder_aux:
            world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=9)
            world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=10)
            world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=11)
            resources.powder_target_winner.bind_to_storage_buffer(binding=12)
        if direct_bridge_outputs:
            resources.powder_source_cell_core_snapshot.bind_to_storage_buffer(binding=13)
            resources.powder_source_aux_snapshot.bind_to_storage_buffer(binding=14)
        resources.cell_state_tex.use(location=0)
        resources.velocity_tex.use(location=3)
        resources.temp_tex.use(location=4)
        resources.timer_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        if direct_bridge_outputs:
            world.bridge.textures["material"].bind_to_image(1, read=False, write=True)
        if not direct_bridge_outputs:
            resources.cell_state_out_tex.bind_to_image(0, read=False, write=True)
            resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
            resources.temp_out_tex.bind_to_image(4, read=False, write=True)
            resources.timer_out_tex.bind_to_image(5, read=False, write=True)
            resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        if formal_frame:
            pipeline._run_active_tile_indirect(program, resources, "powder reservation apply")
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    if direct_bridge_outputs:
        world.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )
        pipeline._refresh_authoritative_active_scheduler_after_apply(
            world,
            "active_refresh_after_powder",
        )
        pipeline.last_cpu_mirror_downloaded = False
        return

    if not use_packed_powder_aux:
        with pipeline._profile_pass(world, "powder_apply_aux"):
            aux_program = pipeline.programs[
                "apply_powder_reservation_aux" if epoch_enabled else "apply_powder_reservation_aux_legacy"
            ]
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
            if "use_bridge_inputs" in aux_members:
                aux_program["use_bridge_inputs"].value = bool(use_bridge_inputs)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
            if epoch_enabled:
                aux_program["powder_apply_epoch"].value = int(pipeline._powder_apply_epoch)
            if epoch_enabled:
                resources.powder_apply_epoch.bind_to_storage_buffer(binding=11)
            resources.active_tile_count.bind_to_storage_buffer(binding=6)
            resources.active_tile_list.bind_to_storage_buffer(binding=7)
            if use_bridge_inputs:
                world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=8)
                world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=9)
                world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=10)
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
    sparse_powder_bridge_publish = bool(
        formal_frame
            and use_packed_powder_aux
            and pipeline._powder_sparse_bridge_publish_enabled
            and not bool(getattr(world, "phase_c_defer_cell_publish", False))
    )
    with pipeline._profile_pass(world, "powder_terminal_cell_publish"):
        pipeline._publish_bridge_outputs(
            world,
            resources,
            output_textures=True,
            active_tile_indirect=formal_frame,
            use_powder_apply_touch_sources=formal_frame,
            use_packed_powder_aux=use_packed_powder_aux,
            sparse_powder_bridge_publish=sparse_powder_bridge_publish,
        )
    pipeline._refresh_authoritative_active_scheduler_after_apply(world, "active_refresh_after_powder")
    pipeline.last_cpu_mirror_downloaded = not formal_frame
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_powder_apply_state(world, resources)


def _snapshot_powder_direct_apply_sources(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> None:
    """Freeze bridge payloads before the direct apply overwrites source cells."""
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_count = int(world.width * world.height)
    core_bytes = cell_count * 5 * np.dtype(np.uint32).itemsize
    aux_bytes = cell_count * np.dtype(np.int32).itemsize
    pipeline._ensure_dynamic_buffer_capacity(
        ctx,
        resources,
        "powder_source_cell_core_snapshot",
        core_bytes,
    )
    pipeline._ensure_dynamic_buffer_capacity(
        ctx,
        resources,
        "powder_source_aux_snapshot",
        aux_bytes * 3,
    )
    with pipeline._profile_pass(world, "powder_direct_source_snapshot"):
        ctx.copy_buffer(
            resources.powder_source_cell_core_snapshot,
            world.bridge.buffers["cell_core"],
            size=core_bytes,
        )
        for aux_index, resource_name in enumerate(
            ("island_id", "entity_id", "placeholder_displaced_material")
        ):
            ctx.copy_buffer(
                resources.powder_source_aux_snapshot,
                world.bridge.buffers[resource_name],
                size=aux_bytes,
                write_offset=aux_index * aux_bytes,
            )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _powder_direct_apply_is_safe(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
) -> bool:
    """Legacy detector for the pre-snapshot direct variant.

    The current opt-in direct path snapshots source payloads before applying,
    so it does not call this conservative detector. Keep the detector for
    callers/tests that probe the old race condition; the default path never
    pays its synchronization cost.
    """
    ctx = world.bridge.ctx
    assert ctx is not None
    unsafe = resources.powder_direct_apply_unsafe
    unsafe.write(np.zeros((1,), dtype=np.uint32).tobytes())
    program = pipeline.programs["detect_powder_direct_apply_unsafe"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.powder_reservations.bind_to_storage_buffer(binding=0)
    resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
    resources.powder_apply_incoming.bind_to_storage_buffer(binding=2)
    resources.powder_apply_outgoing.bind_to_storage_buffer(binding=3)
    unsafe.bind_to_storage_buffer(binding=4)
    with pipeline._profile_pass(world, "powder_direct_apply_safety"):
        pipeline._run_powder_reservation_indirect(
            world,
            resources,
            program,
            "powder direct apply safety",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        ctx.finish()
    value = np.frombuffer(unsafe.read(size=4), dtype=np.uint32, count=1)[0]
    return int(value) == 0


def _read_powder_reservations(pipeline, resources: GPUMotionResources, reservation_count: int) -> np.ndarray:
    if reservation_count <= 0:
        return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
    return np.frombuffer(
        resources.powder_reservations.read(size=reservation_count * POWDER_RESERVATION_DTYPE.itemsize),
        dtype=POWDER_RESERVATION_DTYPE,
        count=reservation_count,
    ).copy()


def materialize_compact_powder_reservations(
    pipeline,
    world: "WorldEngine",
    *,
    download: bool = False,
) -> np.ndarray | bool:
    """Expand the current compact bridge stream only for an explicit observer."""
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if "powder_reservation_compact" not in bridge.gpu_authoritative_resources:
        return False
    ctx = bridge.ctx
    if ctx is None or not pipeline._bridge_context_active(world):
        raise RuntimeError("compact powder reservation materialization requires the bridge GL context")
    pipeline._ensure_programs(ctx)
    cell_count = int(world.width * world.height)
    if "powder_reservation_standard" not in bridge.gpu_authoritative_resources:
        required_bytes = max(4, cell_count * POWDER_RESERVATION_DTYPE.itemsize)
        standard = bridge.buffers["powder_reservation"]
        if standard.size < required_bytes:
            standard.release()
            standard = ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["powder_reservation"] = standard
        program = pipeline.programs["expand_compact_powder_reservations"]
        program["reservation_capacity"].value = cell_count
        bridge.buffers["powder_reservation_compact"].bind_to_storage_buffer(binding=0)
        bridge.buffers["powder_reservation_count"].bind_to_storage_buffer(binding=1)
        standard.bind_to_storage_buffer(binding=2)
        with pipeline._profile_pass(world, "powder_reservation_lazy_expand"):
            program.run(
                (cell_count + POWDER_RESERVATION_LOCAL_SIZE - 1)
                // POWDER_RESERVATION_LOCAL_SIZE,
                1,
                1,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        pipeline.compact_powder_reservation_materialization_count += 1
        bridge.mark_gpu_authoritative("powder_reservation_standard")
    if not download:
        return True
    if "powder_reservation_cpu_mirror" in bridge.gpu_authoritative_resources:
        return world.motion_solver.last_powder_reservations.copy()
    count = int(
        np.frombuffer(
            bridge.buffers["powder_reservation_count"].read(size=4),
            dtype=np.int32,
            count=1,
        )[0]
    )
    count = max(0, min(count, cell_count))
    if count > 0:
        records = np.frombuffer(
            bridge.buffers["powder_reservation"].read(
                size=count * POWDER_RESERVATION_DTYPE.itemsize
            ),
            dtype=POWDER_RESERVATION_DTYPE,
            count=count,
        ).copy()
    else:
        records = np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
    world.motion_solver.last_powder_reservations = records.copy()
    world.motion_solver.last_public_powder_reservations = (
        world.motion_solver._capture_public_powder_reservations(world, records)
    )
    bridge.shadow_buffers["powder_reservation"] = records.copy()
    bridge.shadow_buffers["powder_reservation_count"] = np.array([count], dtype=np.int32)
    bridge.mark_gpu_authoritative("powder_reservation_cpu_mirror")
    return records


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
