from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _active_scheduler_gpu_authoritative,
    _ensure_material_flags_buffer,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
    mark_collapse_structure_dirty_tiles_from_bridge_cell_core,
)
from oracle_game.sim.gpu_heat import (
    FREEZE_COLD_NEIGHBOR_THRESHOLD,
    LOCAL_SIZE,
    MAX_GAS_SPECIES,
    GPUHeatResources,
    GPUHeatStageTargets,
)
from oracle_game.sim.gpu_timer_pack import unpack_cell_state, unpack_u8x4
from oracle_game.types import Phase


def step(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    solve_tile_mask: np.ndarray,
    ambient_iterations: int,
) -> GPUHeatStageTargets:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU heat pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline._deferred_cell_core_frame_id = None
    pipeline._motion_handoff_candidate = None
    pipeline._terminal_bridge_aux_dirty_frame_id = None
    pipeline._terminal_phase_fusion_frame_id = None
    pipeline._terminal_dirty_publish_frame_id = None
    pipeline._deferred_dirty_publish_handoff_frame_id = None
    pipeline._heat_gas_bridge_residency_frame_id = None
    pipeline.last_terminal_bridge_aux_residency_used = False
    pipeline.last_terminal_dirty_workgroup_aggregation_used = False
    pipeline.last_terminal4x6_workgroup16x8_used = False
    pipeline.last_terminal_sparse_resident_specialization_used = False
    pipeline.last_terminal_lazy_action_inputs_used = False
    pipeline.last_packed_phase_boil_targets_used = False
    pipeline.last_terminal_hierarchical_row_summary_used = False
    pipeline.last_terminal_nv32_ballot_gas_reduction_used = False
    pipeline.last_heat_sparse_bridge_residency_used = False
    pipeline.last_heat_gas_bridge_residency_used = False
    pipeline.last_terminal_bridge_gas_residency_used = False
    pipeline.last_cell_heat_bridge_diffuse4_fusion_used = False
    with pipeline._profile_pass(world, "upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_heat_bridge_diffuse4_candidate = bool(
        pipeline._cell_heat_bridge_diffuse4_fusion_enabled
        and pipeline._cell_heat_bridge_exchange_feedback4_fusion_enabled
        and pipeline._ambient_exchange_feedback4_enabled
        and pipeline._formal_gpu_frame(world)
        and int(world.gas_cell_size) == 4
        and int(ambient_iterations) == 4
        and {"cell_core", "ambient_temperature"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    heat_gas_bridge_residency = bool(
        (
            pipeline._heat_gas_bridge_residency_enabled
            or getattr(pipeline, "_terminal_bridge_gas_residency_enabled", False)
        )
        and pipeline._terminal4x6_fusion_enabled
        and pipeline._condense_apply_gas4x6_fusion_enabled
        and pipeline._formal_gpu_frame(world)
        and int(world.gas_cell_size) == 4
        and int(world.gas_concentration.shape[0]) == 6
        and int(world.gas_width) == (int(world.width) + 3) // 4
        and int(world.gas_height) == (int(world.height) + 3) // 4
        and int(ambient_iterations) > 0
        and int(ambient_iterations) % 2 == 0
        and {"ambient_temperature", "gas_concentration"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
        and not cell_heat_bridge_diffuse4_candidate
    )
    pipeline.last_heat_gas_bridge_residency_used = heat_gas_bridge_residency
    pipeline.last_terminal_bridge_gas_residency_used = heat_gas_bridge_residency
    sparse_bridge_aux_residency = bool(
        pipeline._heat_sparse_bridge_residency_enabled
        and not pipeline._terminal4x6_workgroup16x8_enabled
        and not pipeline._terminal_phase_fusion_enabled
        and pipeline._terminal_dirty_publish_fusion_enabled
        and pipeline._terminal_dirty_workgroup_aggregation_enabled
        and pipeline._terminal_inplace_sparse_write_enabled
    )
    bridge_aux_residency = bool(
        (
            pipeline._terminal_bridge_aux_residency_enabled
            or sparse_bridge_aux_residency
        )
        and pipeline._terminal4x6_fusion_enabled
        and pipeline._condense_apply_gas4x6_fusion_enabled
        and pipeline._terminal_bridge_aux_dirty_fusion_enabled
        and pipeline._formal_gpu_frame(world)
        and int(world.gas_cell_size) == 4
        and int(world.gas_concentration.shape[0]) == 6
        and int(world.gas_width) == (int(world.width) + 3) // 4
        and int(world.gas_height) == (int(world.height) + 3) // 4
        and bool(getattr(world, "heat_motion_handoff_active", False))
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and {"island_id", "entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
        and _active_scheduler_gpu_authoritative(world)
    )
    pipeline.last_heat_sparse_bridge_residency_used = bool(
        sparse_bridge_aux_residency
        and bridge_aux_residency
    )
    pipeline.last_terminal_bridge_aux_residency_used = bridge_aux_residency
    with pipeline._profile_pass(world, "load_bridge_inputs"):
        deferred_cell_core = pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            gas_group_x,
            gas_group_y,
            skip_cell_aux_hydration=bridge_aux_residency,
            skip_ambient_gas_hydration=heat_gas_bridge_residency,
            skip_ambient_hydration=cell_heat_bridge_diffuse4_candidate,
        )
    fuse_ambient_exchange_feedback = bool(
        pipeline._ambient_exchange_feedback4_enabled
        and pipeline._formal_gpu_frame(world)
        and int(world.gas_cell_size) == 4
    )
    fuse_cell_heat_exchange_feedback = bool(
        fuse_ambient_exchange_feedback
        and deferred_cell_core
        and pipeline._cell_heat_bridge_exchange_feedback4_fusion_enabled
    )
    fuse_cell_heat_bridge_diffuse4 = bool(
        cell_heat_bridge_diffuse4_candidate
        and fuse_cell_heat_exchange_feedback
    )
    pipeline.last_cell_heat_bridge_diffuse4_fusion_used = (
        fuse_cell_heat_bridge_diffuse4
    )
    if fuse_cell_heat_bridge_diffuse4:
        with pipeline._profile_pass(world, "ambient_diffuse_fused_into_cell_heat"):
            pass
    else:
        with pipeline._profile_pass(world, "ambient_diffuse"):
            pipeline._run_ambient_diffuse(
                world,
                resources,
                gas_group_x,
                gas_group_y,
                iterations=ambient_iterations,
                bridge_ambient_resident=heat_gas_bridge_residency,
            )
    with pipeline._profile_pass(world, "cell_heat"):
        pipeline._run_cell_heat(
            world,
            dt,
            resources,
            group_x,
            group_y,
            hydrate_cell_core=deferred_cell_core,
            fuse_ambient_exchange_feedback4=fuse_cell_heat_exchange_feedback,
            fuse_ambient_diffuse4=fuse_cell_heat_bridge_diffuse4,
        )
    if fuse_cell_heat_exchange_feedback:
        pass
    elif fuse_ambient_exchange_feedback:
        with pipeline._profile_pass(world, "ambient_exchange_feedback4"):
            pipeline._run_ambient_exchange_feedback4(world, dt, resources, group_x, group_y)
    else:
        with pipeline._profile_pass(world, "ambient_exchange"):
            pipeline._run_ambient_exchange(world, dt, resources, group_x, group_y)
        with pipeline._profile_pass(world, "ambient_feedback"):
            pipeline._run_ambient_feedback(world, dt, resources, gas_group_x, gas_group_y)
    fuse_condense_apply_gas = bool(
        pipeline._condense_apply_gas4x6_fusion_enabled
        and pipeline._formal_gpu_frame(world)
        and int(world.gas_cell_size) == 4
        and int(world.gas_concentration.shape[0]) == 6
    )
    fuse_terminal4x6 = bool(
        pipeline._terminal4x6_fusion_enabled
        and fuse_condense_apply_gas
        and int(world.gas_width) == (int(world.width) + 3) // 4
        and int(world.gas_height) == (int(world.height) + 3) // 4
    )
    terminal_prepared_dirty_resources = None
    if (
        fuse_terminal4x6
        and (
            pipeline._terminal_phase_fusion_enabled
            or pipeline._terminal_dirty_publish_fusion_enabled
        )
        and deferred_cell_core
        and bool(getattr(world, "heat_motion_handoff_active", False))
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and {
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
        and _active_scheduler_gpu_authoritative(world)
    ):
        dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_buffer is not None and dirty_queue is not None:
            material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
            dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
            terminal_prepared_dirty_resources = (
                dirty_buffer,
                dirty_count,
                dirty_list,
                dirty_dispatch_args,
                material_flags_buffer,
                material_count,
            )
    terminal_phase_fusion = bool(
        terminal_prepared_dirty_resources is not None
        and pipeline._terminal_phase_fusion_enabled
    )
    terminal_dirty_publish_fusion = bool(
        terminal_prepared_dirty_resources is not None
        and pipeline._terminal_dirty_publish_fusion_enabled
    )
    terminal_dirty_workgroup_aggregation = bool(
        terminal_dirty_publish_fusion
        and pipeline._terminal_dirty_workgroup_aggregation_enabled
        and int(world.active.tile_size) >= 8
        and int(world.active.tile_size) % 8 == 0
    )
    pipeline.last_terminal_dirty_workgroup_aggregation_used = (
        terminal_dirty_workgroup_aggregation
    )
    terminal_bridge_aux_dirty = bool(
        terminal_phase_fusion or terminal_dirty_publish_fusion
    )
    if (
        not terminal_bridge_aux_dirty
        and fuse_terminal4x6
        and pipeline._terminal_bridge_aux_dirty_fusion_enabled
        and bool(getattr(world, "heat_motion_handoff_active", False))
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and {"island_id", "entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
        and _active_scheduler_gpu_authoritative(world)
    ):
        terminal_bridge_aux_dirty = True
    terminal_lazy_action_inputs = bool(
        pipeline._terminal_lazy_action_inputs_enabled
        and pipeline._terminal_sparse_resident_specialization_enabled
        and pipeline._formal_gpu_frame(world)
        and fuse_terminal4x6
        and bridge_aux_residency
        and terminal_bridge_aux_dirty
        and terminal_dirty_publish_fusion
        and terminal_dirty_workgroup_aggregation
        and not terminal_phase_fusion
        and not heat_gas_bridge_residency
        and not pipeline._terminal4x6_workgroup16x8_enabled
        and pipeline._terminal_inplace_sparse_write_enabled
        and pipeline._terminal_split_target_active_reuse_enabled
        and pipeline._terminal_dead_condense_target_store_elision_enabled
    )
    current_table_signature = (
        int(world.bridge.table_generations.get("materials", 0)),
        int(world.bridge.table_generations.get("reactions", 0)),
    )
    if (
        pipeline._packed_phase_boil_targets_enabled
        and terminal_lazy_action_inputs
        and pipeline._packed_phase_boil_table_signature is None
    ):
        pipeline._packed_phase_boil_table_signature = current_table_signature
    packed_phase_boil_targets = bool(
        pipeline._packed_phase_boil_targets_enabled
        and terminal_lazy_action_inputs
        and pipeline._packed_phase_boil_table_signature == current_table_signature
    )
    pipeline.last_packed_phase_boil_targets_used = packed_phase_boil_targets
    # Formal motion consumes bridge aux. The private target/aux textures are
    # either overwritten from the bridge next frame or ignored by this mode.
    if not terminal_phase_fusion:
        with pipeline._profile_pass(world, "phase_boil_targets"):
            pipeline._run_phase_boil_targets(
                world,
                dt,
                resources,
                group_x,
                group_y,
                clear_terminal_aux=fuse_terminal4x6,
                publish_bridge_aux=terminal_bridge_aux_dirty,
                skip_private_aux_writes=(
                    bridge_aux_residency and terminal_bridge_aux_dirty
                ),
                lazy_action_inputs=terminal_lazy_action_inputs,
                packed_phase_boil_targets=packed_phase_boil_targets,
            )
    if fuse_terminal4x6:
        with pipeline._profile_pass(world, "apply_terminal4x6"):
            pipeline._run_apply_terminal4x6(
                world,
                dt,
                resources,
                bridge_aux_dirty=terminal_bridge_aux_dirty,
                fuse_phase_boil=terminal_phase_fusion,
                dirty_publish_resources=(
                    terminal_prepared_dirty_resources
                    if terminal_dirty_publish_fusion
                    else None
                ),
                bridge_aux_resident=(
                    bridge_aux_residency and terminal_bridge_aux_dirty
                ),
                aggregate_dirty_tiles=terminal_dirty_workgroup_aggregation,
                bridge_gas_resident=heat_gas_bridge_residency,
                packed_phase_boil_targets=packed_phase_boil_targets,
            )
    else:
        if not fuse_condense_apply_gas:
            with pipeline._profile_pass(world, "condense_targets"):
                pipeline._run_condense_targets(world, resources, gas_group_x, gas_group_y)
        with pipeline._profile_pass(world, "apply_cell_targets"):
            pipeline._run_apply_cell_targets(world, dt, resources, group_x, group_y)
        with pipeline._profile_pass(world, "apply_gas_targets"):
            pipeline._run_apply_gas_targets(
                world,
                dt,
                resources,
                gas_group_x,
                gas_group_y,
                fuse_condense_targets=fuse_condense_apply_gas,
            )
        with pipeline._profile_pass(world, "apply_condense_cells"):
            pipeline._run_apply_condense_cells(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "publish_bridge_outputs"):
        pipeline._publish_bridge_outputs(
            world,
            resources,
            group_x,
            group_y,
            gas_group_x,
            gas_group_y,
            skip_cell=terminal_dirty_publish_fusion,
            prepared_dirty_resources=(
                terminal_prepared_dirty_resources
                if terminal_phase_fusion and not terminal_dirty_publish_fusion
                else None
            ),
            skip_gas=heat_gas_bridge_residency,
        )
    pipeline._last_formal_output_frame_id = (
        int(getattr(world, "frame_id", 0))
        if pipeline._formal_gpu_frame(world)
        else None
    )
    pipeline.last_cpu_mirror_downloaded = not (
        getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        with pipeline._profile_pass(world, "download_outputs"):
            return pipeline._download_outputs(world, resources)
    return pipeline._empty_stage_targets(world)


def abort_deferred_cell_core(pipeline, world: "WorldEngine") -> bool:
    """Restore the post-heat checkpoint if a later stage aborts the frame."""
    if pipeline._deferred_cell_core_frame_id != int(getattr(world, "frame_id", 0)):
        return False
    resources = pipeline.resources
    if resources is None:
        raise RuntimeError("deferred heat cell core has no recovery resources")
    previous_handoff = bool(getattr(world, "heat_motion_handoff_active", False))
    previous_phase_c = bool(getattr(world, "phase_c_defer_cell_publish", False))
    world.heat_motion_handoff_active = False
    world.phase_c_defer_cell_publish = False
    try:
        pipeline._publish_bridge_outputs(
            world,
            resources,
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        )
        pipeline._deferred_cell_core_frame_id = None
        pipeline._motion_handoff_candidate = None
        return True
    finally:
        world.heat_motion_handoff_active = previous_handoff
        world.phase_c_defer_cell_publish = previous_phase_c


# ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.


def _load_authoritative_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    gas_group_x: int,
    gas_group_y: int,
    *,
    skip_cell_aux_hydration: bool = False,
    skip_ambient_gas_hydration: bool = False,
    skip_ambient_hydration: bool = False,
) -> bool:
    if not pipeline._formal_gpu_frame(world):
        return False
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    copy_ambient = "ambient_temperature" in authoritative
    copy_gas = "gas_concentration" in authoritative
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient or copy_gas):
        return False
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative input state")

    if (copy_island_id or copy_entity_id or copy_displaced) and not skip_cell_aux_hydration:
        program = pipeline.programs["load_bridge_cell_aux"]
        program["cell_grid_size"].value = (world.width, world.height)
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

    hydrate_ambient = bool(
        copy_ambient
        and not skip_ambient_gas_hydration
        and not skip_ambient_hydration
    )
    hydrate_gas = bool(copy_gas and not skip_ambient_gas_hydration)
    if hydrate_ambient or hydrate_gas:
        program = pipeline.programs["load_bridge_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["copy_ambient"].value = hydrate_ambient
        program["copy_gas"].value = hydrate_gas
        bridge.textures["ambient_temperature"].use(location=0)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
        resources.ambient_ping.bind_to_image(2, read=False, write=True)
        resources.gas_tex.bind_to_image(3, read=False, write=True)
        program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

    pipeline._sync_compute_writes(bridge.ctx)
    return bool(copy_cell_core)


def _run_cell_heat(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    *,
    hydrate_cell_core: bool = False,
    fuse_ambient_exchange_feedback4: bool = False,
    fuse_ambient_diffuse4: bool = False,
) -> None:
    if fuse_ambient_exchange_feedback4:
        if not hydrate_cell_core:
            raise RuntimeError("fused heat exchange requires bridge cell-core hydration")
        program = pipeline.programs[
            "cell_heat_bridge_diffuse4_exchange_feedback4"
            if fuse_ambient_diffuse4
            else "cell_heat_bridge_exchange_feedback4"
        ]
    elif fuse_ambient_diffuse4:
        raise RuntimeError("ambient diffuse4 fusion requires fused heat exchange")
    else:
        program = pipeline.programs["cell_heat_bridge" if hydrate_cell_core else "cell_heat"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    if hydrate_cell_core:
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=8)
        resources.cell_state_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_tex.bind_to_image(3, read=False, write=True)
        if fuse_ambient_exchange_feedback4:
            if fuse_ambient_diffuse4:
                world.bridge.textures["ambient_temperature"].use(location=4)
                resources.ambient_ping.bind_to_image(1, read=False, write=True)
            else:
                resources.ambient_ping.use(location=4)
            resources.temp_ping.bind_to_image(4, read=False, write=True)
            resources.ambient_pong.bind_to_image(7, read=False, write=True)
        else:
            resources.temp_pong.bind_to_image(4, read=False, write=True)
        resources.timer_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_tex.bind_to_image(6, read=False, write=True)
    else:
        resources.cell_state_tex.use(location=1)
        resources.temp_ping.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.temp_pong.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_ambient_exchange(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["ambient_exchange"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_pong.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.temp_ping.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_ambient_exchange_feedback4(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["ambient_exchange_feedback4"]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["gas_cell_size"].value = world.gas_cell_size
    program["tile_size"].value = world.active.tile_size
    program["dt"].value = dt
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_pong.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.temp_ping.bind_to_image(5, read=False, write=True)
    resources.ambient_pong.bind_to_image(6, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_ambient_diffuse(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
    *,
    iterations: int,
    bridge_ambient_resident: bool = False,
) -> None:
    if iterations <= 0:
        return
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["ambient_diffuse"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    for iteration in range(iterations):
        if bridge_ambient_resident and iteration == 0:
            world.bridge.textures["ambient_temperature"].use(location=3)
        else:
            resources.ambient_ping.use(location=3)
        resources.ambient_pong.bind_to_image(4, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        pipeline._sync_compute_writes(ctx)
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping


def _run_ambient_feedback(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    program = pipeline.programs["ambient_feedback"]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["gas_cell_size"].value = world.gas_cell_size
    program["tile_size"].value = world.active.tile_size
    program["dt"].value = dt
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_pong.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.ambient_pong.bind_to_image(5, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_phase_targets(pipeline, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
    program = pipeline.programs["phase_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
    pipeline._set_uniform_if_present(program, "freeze_cold_neighbor_threshold", FREEZE_COLD_NEIGHBOR_THRESHOLD)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.temp_ping.use(location=5)
    resources.phase_target_tex.bind_to_image(6, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_phase_boil_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    *,
    clear_terminal_aux: bool = False,
    publish_bridge_aux: bool = False,
    skip_private_aux_writes: bool = False,
    lazy_action_inputs: bool = False,
    packed_phase_boil_targets: bool = False,
) -> None:
    if packed_phase_boil_targets and not lazy_action_inputs:
        raise RuntimeError("packed heat targets require lazy action inputs")
    if packed_phase_boil_targets:
        program_name = "phase_boil_targets_packed_lazy_action_inputs"
    elif lazy_action_inputs:
        program_name = "phase_boil_targets_lazy_action_inputs"
    else:
        program_name = "phase_boil_targets"
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
    pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
    pipeline._set_uniform_if_present(program, "freeze_cold_neighbor_threshold", FREEZE_COLD_NEIGHBOR_THRESHOLD)
    pipeline._set_uniform_if_present(program, "dt", dt)
    pipeline._set_uniform_if_present(program, "clear_terminal_aux", clear_terminal_aux)
    pipeline._set_uniform_if_present(program, "publish_bridge_aux", publish_bridge_aux)
    pipeline._set_uniform_if_present(
        program,
        "skip_private_aux_writes",
        skip_private_aux_writes,
    )
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.temp_ping.use(location=4)
    resources.integrity_tex.use(location=7)
    resources.phase_target_tex.bind_to_image(5, read=False, write=True)
    if not packed_phase_boil_targets:
        resources.boil_target_tex.bind_to_image(6, read=False, write=True)
    resources.island_id_tex.bind_to_image(0, read=False, write=True)
    resources.entity_id_tex.bind_to_image(1, read=False, write=True)
    if publish_bridge_aux:
        world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_boil_targets(pipeline, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
    program = pipeline.programs["boil_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.temp_ping.use(location=4)
    resources.boil_target_tex.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_condense_targets(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    program = pipeline.programs["condense_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.gas_params.bind_to_storage_buffer(binding=3)
    resources.gas_tex.use(location=4)
    resources.ambient_pong.use(location=5)
    resources.condense_target_tex.bind_to_image(6, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_cell_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    with pipeline._profile_pass(world, "apply_cell_targets.main"):
        program = pipeline.programs["apply_cell_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        pipeline._set_uniform_if_present(program, "dt", dt)
        pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.cell_state_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.phase_target_tex.use(location=3)
        resources.timer_tex.use(location=6)
        resources.boil_target_tex.use(location=7)
        resources.temp_ping.use(location=8)
        resources.integrity_tex.use(location=9)
        resources.ambient_pong.use(location=22)
        resources.cell_state_out_tex.bind_to_image(0, read=False, write=True)
        resources.timer_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_pong.bind_to_image(4, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
        resources.island_id_tex.bind_to_image(6, read=False, write=True)
        resources.entity_id_tex.bind_to_image(7, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "apply_cell_targets.aux"):
        pipeline._run_apply_cell_aux_targets(world, dt, resources, group_x, group_y)


def _run_apply_cell_aux_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["apply_cell_aux_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
    pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.phase_target_tex.use(location=3)
    resources.boil_target_tex.use(location=5)
    resources.integrity_tex.use(location=6)
    resources.displaced_tex.use(location=7)
    resources.velocity_tex.use(location=8)
    resources.displaced_out_tex.bind_to_image(0, read=False, write=True)
    resources.velocity_out_tex.bind_to_image(1, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_gas_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
    *,
    fuse_condense_targets: bool = False,
) -> None:
    program = pipeline.programs[
        "apply_gas_targets4x6" if fuse_condense_targets else "apply_gas_targets"
    ]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.gas_params.bind_to_storage_buffer(binding=3)
    resources.gas_tex.use(location=4)
    resources.boil_target_tex.use(location=5)
    resources.condense_target_tex.use(location=6)
    if fuse_condense_targets:
        resources.ambient_pong.use(location=7)
        resources.condense_target_tex.bind_to_image(6, read=False, write=True)
    resources.cell_state_out_tex.use(location=8)
    resources.gas_out_tex.bind_to_image(0, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_terminal4x6(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    *,
    bridge_aux_dirty: bool = False,
    fuse_phase_boil: bool = False,
    dirty_publish_resources=None,
    bridge_aux_resident: bool = False,
    aggregate_dirty_tiles: bool = False,
    bridge_gas_resident: bool = False,
    packed_phase_boil_targets: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    use_workgroup16x8 = bool(pipeline._terminal4x6_workgroup16x8_enabled)
    if aggregate_dirty_tiles:
        tile_size = int(world.active.tile_size)
        use_workgroup16x8 = bool(
            use_workgroup16x8
            and tile_size >= 16
            and tile_size % 16 == 0
        )
    pipeline.last_terminal4x6_workgroup16x8_used = use_workgroup16x8
    sparse_resident_specialized = bool(
        pipeline._terminal_sparse_resident_specialization_enabled
        and pipeline._formal_gpu_frame(world)
        and bridge_aux_resident
        and bridge_aux_dirty
        and dirty_publish_resources is not None
        and aggregate_dirty_tiles
        and not fuse_phase_boil
        and not bridge_gas_resident
        and not use_workgroup16x8
        and pipeline._terminal_inplace_sparse_write_enabled
        and pipeline._terminal_split_target_active_reuse_enabled
        and pipeline._terminal_dead_condense_target_store_elision_enabled
    )
    pipeline.last_terminal_sparse_resident_specialization_used = (
        sparse_resident_specialized
    )
    lazy_action_inputs = bool(
        sparse_resident_specialized
        and pipeline._terminal_lazy_action_inputs_enabled
    )
    pipeline.last_terminal_lazy_action_inputs_used = lazy_action_inputs
    if packed_phase_boil_targets and not lazy_action_inputs:
        raise RuntimeError("packed heat targets require the sparse lazy terminal")
    hierarchical_row_summary = bool(
        packed_phase_boil_targets
        and lazy_action_inputs
        and pipeline._terminal_hierarchical_row_summary_enabled
    )
    nv32_ballot_gas_reduction = bool(
        hierarchical_row_summary
        and pipeline._terminal_nv32_ballot_gas_reduction_enabled
        and pipeline._terminal_nv32_ballot_supported
    )
    pipeline.last_terminal_hierarchical_row_summary_used = bool(
        hierarchical_row_summary and not nv32_ballot_gas_reduction
    )
    pipeline.last_terminal_nv32_ballot_gas_reduction_used = (
        nv32_ballot_gas_reduction
    )
    if nv32_ballot_gas_reduction:
        program_name = (
            "apply_terminal4x6_sparse_resident_packed_lazy_nv32_ballot"
        )
    elif hierarchical_row_summary:
        program_name = "apply_terminal4x6_sparse_resident_packed_lazy_row_summary"
    elif packed_phase_boil_targets:
        program_name = "apply_terminal4x6_sparse_resident_packed_lazy_action_inputs"
    elif lazy_action_inputs:
        program_name = "apply_terminal4x6_sparse_resident_lazy_action_inputs"
    elif sparse_resident_specialized:
        program_name = "apply_terminal4x6_sparse_resident_specialized"
    elif bridge_gas_resident:
        program_name = (
            "apply_terminal4x6_bridge_resident_dirty_workgroup"
            if aggregate_dirty_tiles
            else "apply_terminal4x6_bridge_resident"
        )
    else:
        program_name = (
            "apply_terminal4x6_dirty_workgroup"
            if aggregate_dirty_tiles
            else "apply_terminal4x6"
        )
    if use_workgroup16x8:
        program_name += "_workgroup16x8"
    program = pipeline.programs[program_name]
    inplace_sparse_write = bool(
        pipeline._terminal_inplace_sparse_write_enabled
        and pipeline._formal_gpu_frame(world)
        and not bridge_gas_resident
    )
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["dt"].value = dt
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    pipeline._set_uniform_if_present(
        program,
        "freeze_cold_neighbor_threshold",
        FREEZE_COLD_NEIGHBOR_THRESHOLD,
    )
    pipeline._set_uniform_if_present(program, "publish_bridge_aux", bridge_aux_dirty)
    pipeline._set_uniform_if_present(program, "fuse_phase_boil", fuse_phase_boil)
    pipeline._set_uniform_if_present(
        program, "fuse_dirty_publish", dirty_publish_resources is not None
    )
    pipeline._set_uniform_if_present(
        program, "use_bridge_displaced", bridge_aux_resident
    )
    pipeline._set_uniform_if_present(
        program, "write_private_displaced", not bridge_aux_resident
    )
    pipeline._set_uniform_if_present(
        program, "inplace_sparse_write", inplace_sparse_write
    )
    pipeline._set_uniform_if_present(
        program,
        "reuse_split_target_active_mask",
        bool(pipeline._terminal_split_target_active_reuse_enabled),
    )
    # Formal GPU consumers receive the empty stage-target sentinel and the
    # terminal shader has already applied each condensation decision in place.
    pipeline._set_uniform_if_present(
        program,
        "write_condense_targets",
        not bool(
            pipeline._terminal_dead_condense_target_store_elision_enabled
            and pipeline._formal_gpu_frame(world)
        ),
    )
    resources.material_phase_params.bind_to_storage_buffer(binding=0)
    resources.material_response_params.bind_to_storage_buffer(binding=1)
    resources.gas_params.bind_to_storage_buffer(binding=2)
    resources.material_params.bind_to_storage_buffer(binding=4)
    resources.cell_state_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.phase_target_tex.use(location=3)
    resources.gas_tex.use(location=4)
    if not packed_phase_boil_targets:
        resources.boil_target_tex.use(location=5)
    resources.timer_tex.use(location=6)
    resources.displaced_tex.use(location=7)
    resources.velocity_tex.use(location=8)
    resources.temp_ping.use(location=9)
    resources.integrity_tex.use(location=10)
    resources.ambient_pong.use(location=11)
    (resources.cell_state_tex if inplace_sparse_write else resources.cell_state_out_tex).bind_to_image(
        0, read=False, write=True
    )
    (resources.timer_tex if inplace_sparse_write else resources.timer_out_tex).bind_to_image(
        1, read=False, write=True
    )
    (resources.temp_ping if inplace_sparse_write else resources.temp_pong).bind_to_image(
        2, read=False, write=True
    )
    (resources.integrity_tex if inplace_sparse_write else resources.integrity_out_tex).bind_to_image(
        3, read=False, write=True
    )
    (resources.displaced_tex if inplace_sparse_write else resources.displaced_out_tex).bind_to_image(
        4, read=False, write=True
    )
    (resources.velocity_tex if inplace_sparse_write else resources.velocity_out_tex).bind_to_image(
        5, read=False, write=True
    )
    (resources.gas_tex if inplace_sparse_write else resources.gas_out_tex).bind_to_image(
        6, read=False, write=True
    )
    resources.condense_target_tex.bind_to_image(7, read=False, write=True)
    if bridge_gas_resident:
        world.bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=13)
    if bridge_aux_dirty:
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
    if fuse_phase_boil:
        world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=7)
    if dirty_publish_resources is not None:
        (
            dirty_buffer,
            dirty_count,
            dirty_list,
            dirty_dispatch_args,
            material_flags_buffer,
            material_count,
        ) = dirty_publish_resources
        program["material_count"].value = int(material_count)
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=5)
        material_flags_buffer.bind_to_storage_buffer(binding=8)
        dirty_buffer.bind_to_storage_buffer(binding=9)
        dirty_count.bind_to_storage_buffer(binding=10)
        dirty_list.bind_to_storage_buffer(binding=11)
        dirty_dispatch_args.bind_to_storage_buffer(binding=12)
    terminal_gas_group_width = 4 if use_workgroup16x8 else 2
    program.run(
        (world.gas_width + terminal_gas_group_width - 1) // terminal_gas_group_width,
        (world.gas_height + 1) // 2,
        1,
    )
    pipeline._sync_compute_writes(ctx)
    if inplace_sparse_write:
        # Publish always samples gas_out_tex. The in-place result currently
        # lives in gas_tex, so rotate only the gas roles after the dispatch.
        resources.gas_tex, resources.gas_out_tex = resources.gas_out_tex, resources.gas_tex
    else:
        resources.cell_state_tex, resources.cell_state_out_tex = (
            resources.cell_state_out_tex,
            resources.cell_state_tex,
        )
        resources.timer_tex, resources.timer_out_tex = resources.timer_out_tex, resources.timer_tex
        resources.temp_ping, resources.temp_pong = resources.temp_pong, resources.temp_ping
        resources.integrity_tex, resources.integrity_out_tex = (
            resources.integrity_out_tex,
            resources.integrity_tex,
        )
        resources.displaced_tex, resources.displaced_out_tex = (
            resources.displaced_out_tex,
            resources.displaced_tex,
        )
        resources.velocity_tex, resources.velocity_out_tex = (
            resources.velocity_out_tex,
            resources.velocity_tex,
        )
    if bridge_aux_dirty:
        pipeline._terminal_bridge_aux_dirty_frame_id = int(getattr(world, "frame_id", 0))
    if fuse_phase_boil:
        pipeline._terminal_phase_fusion_frame_id = int(getattr(world, "frame_id", 0))
    if dirty_publish_resources is not None:
        frame_id = int(getattr(world, "frame_id", 0))
        pipeline._terminal_dirty_publish_frame_id = frame_id
        setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
        world.bridge.mark_gpu_authoritative(
            COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
        )
    if bridge_gas_resident:
        pipeline._heat_gas_bridge_residency_frame_id = int(
            getattr(world, "frame_id", 0)
        )


def _run_apply_condense_cells(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    with pipeline._profile_pass(world, "apply_condense_cells.main"):
        program = pipeline.programs["apply_condense_cells"]
        ctx = world.bridge.ctx
        assert ctx is not None
        pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        pipeline._set_uniform_if_present(
            program,
            "gas_species_count",
            min(world.gas_concentration.shape[0], MAX_GAS_SPECIES),
        )
        pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=8)
        resources.cell_state_out_tex.use(location=4)
        resources.timer_out_tex.use(location=9)
        resources.temp_pong.use(location=10)
        resources.integrity_out_tex.use(location=11)
        resources.displaced_out_tex.use(location=22)
        resources.condense_target_tex.use(location=24)
        resources.cell_state_tex.bind_to_image(0, read=False, write=True)
        resources.timer_tex.bind_to_image(3, read=False, write=True)
        resources.temp_ping.bind_to_image(4, read=False, write=True)
        resources.integrity_tex.bind_to_image(5, read=False, write=True)
        resources.displaced_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
        resources.velocity_tex, resources.velocity_out_tex = resources.velocity_out_tex, resources.velocity_tex


def _run_apply_condense_cell_aux(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["apply_condense_cell_aux"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    resources.active_tile_tex.use(location=2)
    resources.gas_params.bind_to_storage_buffer(binding=8)
    resources.cell_state_out_tex.use(location=4)
    resources.island_id_out_tex.use(location=12)
    resources.displaced_out_tex.use(location=22)
    resources.velocity_out_tex.use(location=23)
    resources.condense_target_tex.use(location=24)
    resources.displaced_tex.bind_to_image(0, read=False, write=True)
    resources.velocity_tex.bind_to_image(1, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _download_outputs(pipeline, world: "WorldEngine", resources: GPUHeatResources) -> GPUHeatStageTargets:
    cell_state = np.frombuffer(resources.cell_state_tex.read(), dtype="u4").reshape(
        (world.height, world.width)
    )
    material, phase, flags = unpack_cell_state(cell_state)
    world.material_id[:] = material
    world.phase[:] = phase
    world.cell_flags[:] = flags
    world.timer_pack[:] = unpack_u8x4(
        np.frombuffer(resources.timer_tex.read(), dtype="u4").reshape((world.height, world.width))
    )
    world.cell_temperature[:] = np.frombuffer(resources.temp_ping.read(), dtype="f4").reshape((world.height, world.width))
    world.integrity[:] = np.frombuffer(resources.integrity_tex.read(), dtype="f4").reshape((world.height, world.width))
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_id_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.entity_id[:] = np.rint(
        np.frombuffer(resources.entity_id_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.placeholder_displaced_material[:] = np.rint(
        np.frombuffer(resources.displaced_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.velocity[:] = np.frombuffer(resources.velocity_tex.read(), dtype="f4").reshape((world.height, world.width, 2))
    world.ambient_temperature[:] = np.frombuffer(resources.ambient_pong.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
    world.gas_concentration[:] = np.frombuffer(resources.gas_out_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
    return GPUHeatStageTargets(
        phase_targets=np.rint(
            np.frombuffer(resources.phase_target_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32),
        boil_targets=np.rint(
            np.frombuffer(resources.boil_target_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32),
        condense_targets=(
            np.frombuffer(resources.condense_target_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
            > 0.5
        ),
    )


def _empty_stage_targets(pipeline, world: "WorldEngine") -> GPUHeatStageTargets:
    if pipeline._formal_gpu_frame(world):
        return GPUHeatStageTargets.empty_sentinel()
    return GPUHeatStageTargets(
        phase_targets=np.zeros((world.height, world.width), dtype=np.int32),
        boil_targets=np.zeros((world.height, world.width), dtype=np.int32),
        condense_targets=np.zeros(world.gas_concentration.shape, dtype=np.bool_),
    )


def _publish_bridge_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    gas_group_x: int,
    gas_group_y: int,
    *,
    skip_cell: bool = False,
    prepared_dirty_resources=None,
    skip_gas: bool = False,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative heat state")
    defer_cell_core = bool(
        pipeline._formal_gpu_frame(world)
        and getattr(world, "heat_motion_handoff_active", False)
        and not getattr(world, "phase_c_defer_cell_publish", False)
    )
    motion_pipeline = getattr(getattr(world, "motion_solver", None), "gpu_pipeline", None)
    can_consume_deferred = getattr(motion_pipeline, "can_consume_deferred_heat_core", None)
    defer_dirty_publish_to_handoff = bool(
        not skip_cell
        and pipeline._deferred_dirty_publish_handoff_enabled
        and defer_cell_core
        and callable(can_consume_deferred)
        and can_consume_deferred(world)
    )
    skip_cell_publish = bool(skip_cell or defer_dirty_publish_to_handoff)
    terminal_bridge_aux_dirty = bool(
        pipeline._terminal_bridge_aux_dirty_frame_id == int(getattr(world, "frame_id", 0))
    )
    terminal_dirty_publish = bool(
        pipeline._terminal_dirty_publish_frame_id == int(getattr(world, "frame_id", 0))
    )
    fuse_structure_dirty_mark = False
    dirty_buffer = None
    dirty_count = None
    dirty_list = None
    dirty_dispatch_args = None
    material_flags_buffer = None
    material_count = 0
    if (
        not skip_cell_publish
        and pipeline._formal_gpu_frame(world)
        and _active_scheduler_gpu_authoritative(world)
    ):
        if prepared_dirty_resources is not None:
            (
                dirty_buffer,
                dirty_count,
                dirty_list,
                dirty_dispatch_args,
                material_flags_buffer,
                material_count,
            ) = prepared_dirty_resources
            fuse_structure_dirty_mark = True
        else:
            dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
            dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
            if dirty_buffer is not None and dirty_queue is not None:
                dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
                material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
                fuse_structure_dirty_mark = True
    if not skip_cell_publish:
        with pipeline._profile_pass(world, "publish_bridge_outputs.collapse_dirty_mark"):
            if not fuse_structure_dirty_mark:
                mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
                    world,
                    None,
                    None,
                    cell_state_texture=resources.cell_state_tex,
                )
        with pipeline._profile_pass(world, "publish_bridge_outputs.cell"):
            cell_program = pipeline.programs[
                "publish_bridge_cell_aux_dirty" if defer_cell_core else "publish_bridge_cell"
            ]
            cell_program["cell_grid_size"].value = (world.width, world.height)
            cell_program["tile_grid_size"].value = (
                int(world.active.tile_width),
                int(world.active.tile_height),
            )
            cell_program["tile_size"].value = int(world.active.tile_size)
            cell_program["material_count"].value = int(material_count)
            cell_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            cell_program["mark_structure_dirty"].value = bool(fuse_structure_dirty_mark)
            pipeline._set_uniform_if_present(
                cell_program,
                "write_bridge_aux",
                not terminal_bridge_aux_dirty,
            )
            if not defer_cell_core:
                cell_program["write_cell_core"].value = not bool(
                    getattr(world, "phase_c_defer_cell_publish", False)
                )
            cell_program["cell_state_tex"].value = 0
            if defer_cell_core:
                cell_program["island_tex"].value = 2
                cell_program["entity_tex"].value = 3
                cell_program["displaced_tex"].value = 4
            else:
                cell_program["timer_tex"].value = 3
                cell_program["temp_tex"].value = 4
                cell_program["integrity_tex"].value = 5
                cell_program["island_tex"].value = 6
                cell_program["entity_tex"].value = 7
                cell_program["displaced_tex"].value = 8
                cell_program["velocity_tex"].value = 9
            resources.cell_state_tex.use(location=0)
            if defer_cell_core:
                resources.island_id_tex.use(location=2)
                resources.entity_id_tex.use(location=3)
                resources.displaced_tex.use(location=4)
            else:
                resources.timer_tex.use(location=3)
                resources.temp_ping.use(location=4)
                resources.integrity_tex.use(location=5)
                resources.island_id_tex.use(location=6)
                resources.entity_id_tex.use(location=7)
                resources.displaced_tex.use(location=8)
                resources.velocity_tex.use(location=9)
                bridge.textures["material"].bind_to_image(0, read=False, write=True)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            if fuse_structure_dirty_mark:
                assert dirty_buffer is not None
                assert dirty_count is not None
                assert dirty_list is not None
                assert dirty_dispatch_args is not None
                assert material_flags_buffer is not None
                material_flags_buffer.bind_to_storage_buffer(binding=4)
                dirty_buffer.bind_to_storage_buffer(binding=5)
                dirty_count.bind_to_storage_buffer(binding=6)
                dirty_list.bind_to_storage_buffer(binding=7)
                dirty_dispatch_args.bind_to_storage_buffer(binding=8)
            if defer_cell_core or not bool(getattr(world, "phase_c_defer_cell_publish", False)):
                cell_program.run(group_x, group_y, 1)
    elif defer_dirty_publish_to_handoff:
        with pipeline._profile_pass(world, "publish_bridge_outputs.cell_deferred_to_handoff"):
            pipeline._deferred_dirty_publish_handoff_frame_id = int(
                getattr(world, "frame_id", 0)
            )

    if not skip_gas:
        with pipeline._profile_pass(world, "publish_bridge_outputs.gas"):
            gas_program = pipeline.programs["publish_bridge_gas"]
            gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            gas_program["species_count"].value = int(world.gas_concentration.shape[0])
            gas_program["ambient_tex"].value = 0
            gas_program["gas_tex"].value = 2
            resources.ambient_pong.use(location=0)
            resources.gas_out_tex.use(location=2)
            bridge.textures["ambient_temperature"].bind_to_image(1, read=False, write=True)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=4)
            gas_program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))
    else:
        with pipeline._profile_pass(world, "publish_bridge_outputs.ambient_resident"):
            ambient_program = pipeline.programs["publish_bridge_ambient"]
            ambient_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            resources.ambient_pong.use(location=0)
            bridge.textures["ambient_temperature"].bind_to_image(
                1, read=False, write=True
            )
            ambient_program.run(gas_group_x, gas_group_y, 1)

    with pipeline._profile_pass(world, "publish_bridge_outputs.sync"):
        pipeline._sync_compute_writes(bridge.ctx)
        if fuse_structure_dirty_mark or terminal_dirty_publish:
            setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
            bridge.mark_gpu_authoritative(
                COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
            )
        authoritative_outputs = [
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "gas_concentration",
        ]
        if not defer_cell_core:
            authoritative_outputs.extend(("cell_core", "material"))
        bridge.mark_gpu_authoritative(*authoritative_outputs)
        if defer_cell_core:
            frame_id = int(getattr(world, "frame_id", 0))
            bridge.gpu_authoritative_resources.discard("cell_core")
            bridge.gpu_authoritative_resources.discard("material")
            pipeline._deferred_cell_core_frame_id = frame_id
            pipeline._motion_handoff_candidate = {
                "cell_state": resources.cell_state_tex,
                "temp": resources.temp_ping,
                "integrity": resources.integrity_tex,
                "velocity": resources.velocity_tex,
                "timer": resources.timer_tex,
                "base_flags": None,
                "meta": None,
                "frame_id": frame_id,
            }


# ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
# from :class:`GPUPipelineBase`; the heat pass uses the default barrier
# bits (image-access | texture-fetch | shader-storage).
