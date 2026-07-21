from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    FLOW_SOURCE_GENERATION_BINDING,
    GPUReactionResources,
    LOCAL_SIZE,
)


def _run_cell_gas_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    apply_action_side_effects: bool = True,
    material_gas_rule_count: int = 0,
    may_have_flow_sources: bool = True,
    modify_gas_layer_mask: int | None = None,
    direct_core_outputs: bool = False,
    timed_candidate_outputs: bool = False,
    timed_candidate_outputs_inplace: bool = False,
    light_dose_guard_buffer: Any | None = None,
    action_gas_delta_already_applied: bool = False,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
    deferred_hi_valid: bool = True,
    packed_local_deferred_outputs: bool = False,
    use_expanded_active_tile_mask: bool = False,
) -> None:
    if action_gas_delta_already_applied:
        if may_have_flow_sources:
            with pipeline._profile_pass(world, "cell_gas_action_delta_flow_source_scatter"):
                pipeline._run_cell_gas_action_delta_pass(
                    world,
                    resources,
                    modify_gas_layer_mask=(
                        int(modify_gas_layer_mask)
                        if modify_gas_layer_mask is not None
                        else (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
                    ),
                    may_have_flow_sources=may_have_flow_sources,
                    direct_core_outputs=direct_core_outputs,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                    gas_delta_already_applied=True,
                    flow_source_layers=flow_source_layers,
                    deferred_hi_valid=deferred_hi_valid,
                    packed_local_deferred_outputs=packed_local_deferred_outputs,
                    use_expanded_active_tile_mask=use_expanded_active_tile_mask,
                )
            with pipeline._profile_pass(world, "cell_gas_action_delta_flow_sources"):
                pipeline._append_flow_sources_from_gpu(
                    world,
                    resources,
                    may_have_flow_sources=may_have_flow_sources,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                    flow_source_layers=flow_source_layers,
                )
        return
    if timed_candidate_outputs:
        if int(getattr(pipeline, "_timed_candidate_count_cpu", 1)) <= 0:
            return
        if not apply_action_side_effects or material_gas_rule_count > 0:
            return
        if modify_gas_layer_mask is None:
            modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
        pipeline._run_timed_candidate_gas_side_effect_pass(
            world,
            resources,
            modify_gas_layer_mask=int(modify_gas_layer_mask),
            may_have_flow_sources=may_have_flow_sources,
            flow_source_layers=flow_source_layers,
        )
        return
    if apply_action_side_effects and material_gas_rule_count <= 0:
        if modify_gas_layer_mask is None:
            modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
        pipeline._run_cell_gas_action_delta_pass(
            world,
            resources,
            modify_gas_layer_mask=int(modify_gas_layer_mask),
            may_have_flow_sources=may_have_flow_sources,
            direct_core_outputs=direct_core_outputs,
            light_dose_guard_buffer=light_dose_guard_buffer,
            flow_source_layers=flow_source_layers,
            deferred_hi_valid=deferred_hi_valid,
            packed_local_deferred_outputs=packed_local_deferred_outputs,
            use_expanded_active_tile_mask=use_expanded_active_tile_mask,
        )
        return
    program = pipeline.programs["cell_gas_side_effects"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "apply_action_side_effects", int(apply_action_side_effects))
    pipeline._set_uniform_if_present(program, "material_gas_rule_count", int(material_gas_rule_count))
    pipeline._set_uniform_if_present(program, "use_local_deferred_outputs", bool(direct_core_outputs))
    pipeline._set_uniform_if_present(program, "deferred_hi_valid", bool(deferred_hi_valid))
    if material_gas_rule_count > 0 or modify_gas_layer_mask is None:
        modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
    pipeline._set_uniform_if_present(program, "modify_gas_layer_mask", int(modify_gas_layer_mask))
    cell_state_in, _phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = pipeline._current_cell_textures(resources)
    resources.gas_ping.use(location=0)
    resources.trigger_lo_tex.use(location=1)
    resources.trigger_hi_tex.use(location=2)
    resources.deferred_scale_lo_tex.use(location=3)
    resources.deferred_scale_hi_tex.use(location=4)
    cell_state_in.use(location=5)
    temp_in.use(location=7)
    resources.active_cell_tex.use(location=8)
    resources.local_deferred_lo_out.use(location=9)
    resources.local_deferred_hi_out.use(location=10)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.mg_rule_i.bind_to_storage_buffer(binding=2)
    resources.mg_rule_f.bind_to_storage_buffer(binding=3)
    resources.mg_rule_tags.bind_to_storage_buffer(binding=4)
    resources.material_tags.bind_to_storage_buffer(binding=5)
    resources.gas_tags.bind_to_storage_buffer(binding=6)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.gas_pong.bind_to_image(0, read=False, write=True)
    resources.flow_source_tex.bind_to_image(1, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "cell_gas_side_effects_gather"):
        if light_dose_guard_buffer is not None:
            pipeline._run_light_dose_guarded_dispatch(
                world,
                resources,
                program,
                light_dose_guard_buffer,
                group_x,
                group_y,
                world.gas_concentration.shape[0],
            )
        else:
            program.run(group_x, group_y, world.gas_concentration.shape[0])
        pipeline._sync_compute_writes(world.bridge.ctx)
    with pipeline._profile_pass(world, "cell_gas_side_effects_publish"):
        if light_dose_guard_buffer is not None:
            pipeline._publish_bridge_gas_state(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
        else:
            pipeline._download_gas_state(world, resources)
    with pipeline._profile_pass(world, "cell_gas_side_effects_flow_sources"):
        pipeline._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=may_have_flow_sources,
            light_dose_guard_buffer=light_dose_guard_buffer,
        )



def _run_cell_gas_action_delta_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    modify_gas_layer_mask: int,
    may_have_flow_sources: bool,
    direct_core_outputs: bool = False,
    light_dose_guard_buffer: Any | None = None,
    gas_delta_already_applied: bool = False,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
    deferred_hi_valid: bool = True,
    packed_local_deferred_outputs: bool = False,
    use_expanded_active_tile_mask: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    segment_key = pipeline._formal_segment_batch_key
    batch_formal_delta = (
        pipeline._formal_segment_batch_active()
        and direct_core_outputs
        and segment_key is not None
    )
    gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
    if gas_delta_already_applied:
        pass
    elif batch_formal_delta:
        pipeline._clear_formal_segment_gas_delta(world, resources, segment_key)
    else:
        clear_local_size = LOCAL_SIZE
        clear_program = pipeline.programs["clear_cell_gas_delta"]
        clear_program["delta_count"].value = gas_delta_count
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        with pipeline._profile_pass(world, "cell_gas_action_delta_clear"):
            clear_groups = (gas_delta_count + clear_local_size - 1) // clear_local_size
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    clear_program,
                    light_dose_guard_buffer,
                    clear_groups,
                    1,
                    1,
                )
            else:
                clear_program.run(clear_groups, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    scatter_program = pipeline.programs["scatter_cell_gas_action_delta"]
    scatter_program["cell_grid_size"].value = (world.width, world.height)
    scatter_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    scatter_program["gas_cell_size"].value = int(world.gas_cell_size)
    scatter_program["gas_count"].value = int(world.gas_concentration.shape[0])
    scatter_program["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
    scatter_program["use_local_deferred_outputs"].value = bool(direct_core_outputs)
    scatter_program["use_packed_local_deferred_outputs"].value = bool(packed_local_deferred_outputs)
    pipeline._set_uniform_if_present(scatter_program, "deferred_hi_valid", bool(deferred_hi_valid))
    scatter_program["gas_delta_already_applied"].value = bool(gas_delta_already_applied)
    pipeline._set_uniform_if_present(
        scatter_program,
        "use_expanded_active_tile_mask",
        bool(use_expanded_active_tile_mask),
    )
    resources.trigger_lo_tex.use(location=0)
    resources.trigger_hi_tex.use(location=1)
    resources.deferred_scale_lo_tex.use(location=2)
    resources.deferred_scale_hi_tex.use(location=3)
    resources.active_cell_tex.use(location=4)
    resources.local_deferred_lo_out.use(location=5)
    resources.local_deferred_hi_out.use(location=6)
    resources.local_deferred_packed_out.use(location=7)
    resources.expanded_active_tile_tex.use(location=8)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.flow_source_tex.bind_to_image(0, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        scatter_program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    with pipeline._profile_pass(world, "cell_gas_action_delta_scatter"):
        scatter_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        scatter_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            pipeline._run_light_dose_guarded_dispatch(
                world,
                resources,
                scatter_program,
                light_dose_guard_buffer,
                scatter_group_x,
                scatter_group_y,
                1,
            )
        else:
            scatter_program.run(scatter_group_x, scatter_group_y, 1)
        ctx.memory_barrier(
            ctx.SHADER_STORAGE_BARRIER_BIT
            | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
        )
    if gas_delta_already_applied:
        return
    if batch_formal_delta:
        if may_have_flow_sources:
            with pipeline._profile_pass(world, "cell_gas_action_delta_flow_sources"):
                pipeline._append_flow_sources_from_gpu(
                    world,
                    resources,
                    may_have_flow_sources=True,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                    flow_source_layers=flow_source_layers,
                )
        return

    apply_program = pipeline.programs["apply_cell_gas_delta"]
    apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
    resources.gas_ping.use(location=0)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
    resources.gas_pong.bind_to_image(0, read=False, write=True)
    with pipeline._profile_pass(world, "cell_gas_action_delta_apply"):
        apply_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        apply_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        apply_group_z = int(world.gas_concentration.shape[0])
        if light_dose_guard_buffer is not None:
            pipeline._run_light_dose_guarded_dispatch(
                world,
                resources,
                apply_program,
                light_dose_guard_buffer,
                apply_group_x,
                apply_group_y,
                apply_group_z,
            )
        else:
            apply_program.run(apply_group_x, apply_group_y, apply_group_z)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "cell_gas_action_delta_publish"):
        if light_dose_guard_buffer is not None:
            pipeline._publish_bridge_gas_state(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
        else:
            pipeline._download_gas_state(world, resources)
    with pipeline._profile_pass(world, "cell_gas_action_delta_flow_sources"):
        pipeline._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=may_have_flow_sources,
            light_dose_guard_buffer=light_dose_guard_buffer,
            flow_source_layers=flow_source_layers,
        )


def _run_self_candidate_gas_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    may_have_flow_sources: bool,
    modify_gas_layer_mask: int,
    flow_source_layers: int,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    segment_key = pipeline._formal_segment_batch_key
    if segment_key is None:
        raise RuntimeError("formal self gas candidate scatter requires an active segment batch")
    pipeline._clear_formal_segment_gas_delta(world, resources, segment_key)

    program = pipeline.programs["scatter_self_gas_action_delta_candidates"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["gas_cell_size"].value = int(world.gas_cell_size)
    program["gas_count"].value = int(world.gas_concentration.shape[0])
    program["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
    resources.local_deferred_packed_out.use(location=0)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
    resources.timed_material_target_list.bind_to_storage_buffer(binding=3)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=4)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.flow_source_tex.bind_to_image(0, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    with pipeline._profile_pass(world, "cell_gas_action_delta_scatter_candidates"):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal self gas candidate scatter requires indirect dispatch")
        program.run_indirect(resources.timed_candidate_dispatch_args)
        ctx.memory_barrier(
            ctx.SHADER_STORAGE_BARRIER_BIT
            | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
        )
    if may_have_flow_sources:
        with pipeline._profile_pass(world, "cell_gas_action_delta_flow_sources"):
            pipeline._append_flow_sources_from_gpu(
                world,
                resources,
                may_have_flow_sources=True,
                flow_source_layers=flow_source_layers,
            )


def _run_timed_candidate_gas_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    modify_gas_layer_mask: int,
    may_have_flow_sources: bool,
    flow_source_layers: int = 16,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    segment_key = pipeline._formal_segment_batch_key
    batch_formal_delta = (
        pipeline._formal_segment_batch_active()
        and not may_have_flow_sources
        and segment_key is not None
    )
    gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
    if batch_formal_delta:
        pipeline._clear_formal_segment_gas_delta(world, resources, segment_key)
    else:
        clear_local_size = LOCAL_SIZE
        clear_program = pipeline.programs["clear_cell_gas_delta"]
        clear_program["delta_count"].value = gas_delta_count
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        with pipeline._profile_pass(world, "cell_gas_action_delta_clear"):
            clear_program.run((gas_delta_count + clear_local_size - 1) // clear_local_size, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    scatter_program = pipeline.programs["scatter_cell_gas_action_delta_candidates"]
    scatter_program["cell_grid_size"].value = (world.width, world.height)
    scatter_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    scatter_program["gas_cell_size"].value = int(world.gas_cell_size)
    scatter_program["gas_count"].value = int(world.gas_concentration.shape[0])
    scatter_program["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
    resources.local_deferred_lo_out.use(location=0)
    resources.local_deferred_hi_out.use(location=1)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=3)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=4)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.flow_source_tex.bind_to_image(0, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        scatter_program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    with pipeline._profile_pass(world, "cell_gas_action_delta_scatter_candidates"):
        if not hasattr(scatter_program, "run_indirect"):
            raise RuntimeError("formal timed gas side-effect scatter requires indirect dispatch")
        scatter_program.run_indirect(resources.timed_candidate_dispatch_args)
        ctx.memory_barrier(
            ctx.SHADER_STORAGE_BARRIER_BIT
            | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
        )
    if batch_formal_delta:
        return

    apply_program = pipeline.programs["apply_cell_gas_delta"]
    apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
    resources.gas_ping.use(location=0)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
    resources.gas_pong.bind_to_image(0, read=False, write=True)
    with pipeline._profile_pass(world, "cell_gas_action_delta_apply"):
        apply_program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            int(world.gas_concentration.shape[0]),
        )
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "cell_gas_action_delta_publish"):
        pipeline._download_gas_state(world, resources)
    with pipeline._profile_pass(world, "cell_gas_action_delta_flow_sources"):
        pipeline._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=may_have_flow_sources,
            flow_source_layers=flow_source_layers,
        )



def _run_material_light_dose_consume_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    rule_count: int,
    *,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    cell_program = pipeline.programs["material_light_cell_dose_consume"]
    pipeline._set_uniform_if_present(cell_program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(cell_program, "light_count", world.cell_optical_dose.shape[0])
    pipeline._set_uniform_if_present(cell_program, "rule_count", int(rule_count))
    resources.cell_dose_tex.use(location=0)
    cell_state_in, _phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    cell_state_in.use(location=1)
    temp_in.use(location=3)
    resources.active_cell_tex.use(location=4)
    resources.ml_rule_i.bind_to_storage_buffer(binding=0)
    resources.ml_rule_f.bind_to_storage_buffer(binding=1)
    resources.ml_rule_tags.bind_to_storage_buffer(binding=2)
    resources.material_tags.bind_to_storage_buffer(binding=3)
    resources.cell_dose_pong.bind_to_image(5, read=False, write=True)
    cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_group_z = world.cell_optical_dose.shape[0]
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            cell_program,
            light_dose_guard_buffer,
            cell_group_x,
            cell_group_y,
            cell_group_z,
        )
    else:
        cell_program.run(cell_group_x, cell_group_y, cell_group_z)
    pipeline._sync_compute_writes(world.bridge.ctx)

    gas_program = pipeline.programs["material_light_gas_dose_consume"]
    pipeline._set_uniform_if_present(gas_program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(gas_program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(gas_program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(gas_program, "light_count", world.gas_optical_dose.shape[0])
    pipeline._set_uniform_if_present(gas_program, "rule_count", int(rule_count))
    resources.gas_dose_tex.use(location=0)
    resources.cell_dose_tex.use(location=1)
    cell_state_in.use(location=2)
    temp_in.use(location=4)
    resources.active_cell_tex.use(location=5)
    resources.ml_rule_i.bind_to_storage_buffer(binding=0)
    resources.ml_rule_f.bind_to_storage_buffer(binding=1)
    resources.ml_rule_tags.bind_to_storage_buffer(binding=2)
    resources.material_tags.bind_to_storage_buffer(binding=3)
    resources.gas_dose_pong.bind_to_image(6, read=False, write=True)
    gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_z = world.gas_optical_dose.shape[0]
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            gas_program,
            light_dose_guard_buffer,
            gas_group_x,
            gas_group_y,
            gas_group_z,
        )
    else:
        gas_program.run(gas_group_x, gas_group_y, gas_group_z)
    pipeline._sync_compute_writes(world.bridge.ctx)
    if light_dose_guard_buffer is not None:
        pipeline._publish_bridge_dose_state(
            world,
            resources,
            light_dose_guard_buffer=light_dose_guard_buffer,
        )
    else:
        pipeline._download_dose_state(world, resources)



def _run_cell_material_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    direct_core_outputs: bool = False,
    timed_candidate_outputs: bool = False,
    timed_candidate_outputs_inplace: bool = False,
    light_dose_guard_buffer: Any | None = None,
    copy_velocity_passthrough: bool = False,
    velocity_in_place: bool = False,
    inplace: bool = False,
    deferred_hi_valid: bool = True,
    packed_local_deferred_outputs: bool = False,
) -> None:
    if timed_candidate_outputs:
        if int(getattr(pipeline, "_timed_candidate_count_cpu", 1)) <= 0:
            return
        pipeline._run_timed_candidate_material_side_effect_pass(
            world, resources, inplace=timed_candidate_outputs_inplace
        )
        return
    program = pipeline.programs["cell_material_side_effects"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "use_local_deferred_outputs", bool(direct_core_outputs))
    pipeline._set_uniform_if_present(
        program,
        "use_packed_local_deferred_outputs",
        bool(packed_local_deferred_outputs),
    )
    pipeline._set_uniform_if_present(program, "deferred_hi_valid", bool(deferred_hi_valid))
    pipeline._set_uniform_if_present(program, "copy_velocity_passthrough", bool(copy_velocity_passthrough))
    pipeline._set_uniform_if_present(program, "velocity_in_place", bool(velocity_in_place))
    cell_state_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    if inplace:
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = (
            pipeline._current_cell_textures(resources)
        )
    else:
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = pipeline._next_cell_textures(resources)
    cell_state_in.use(location=0)
    (velocity_out if velocity_in_place else velocity_in).use(location=1)
    resources.trigger_lo_tex.use(location=2)
    resources.trigger_hi_tex.use(location=3)
    resources.deferred_scale_lo_tex.use(location=4)
    resources.deferred_scale_hi_tex.use(location=5)
    temp_in.use(location=6)
    resources.local_deferred_lo_out.use(location=7)
    resources.local_deferred_hi_out.use(location=8)
    resources.local_deferred_packed_out.use(location=9)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    velocity_out.bind_to_image(4, read=False, write=True)
    timer_out.bind_to_image(5, read=False, write=True)
    resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
    if velocity_in_place:
        velocity_in.bind_to_image(7, read=True, write=True)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            program,
            light_dose_guard_buffer,
            group_x,
            group_y,
            1,
        )
    else:
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(world.bridge.ctx)



def _run_timed_candidate_material_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    inplace: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_state_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )

    compact_program = pipeline.programs["compact_timed_material_targets"]
    compact_program["cell_grid_size"].value = (world.width, world.height)
    velocity_in.use(location=0)
    resources.local_deferred_lo_out.use(location=1)
    resources.local_deferred_hi_out.use(location=2)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=1)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=2)
    resources.timed_material_target_list.bind_to_storage_buffer(binding=3)
    resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=4)
    resources.timed_material_target_marks.bind_to_storage_buffer(binding=5)
    with pipeline._profile_pass(world, "timed_material_targets_compact"):
        if not hasattr(compact_program, "run_indirect"):
            raise RuntimeError("formal timed material side-effect target compaction requires indirect dispatch")
        compact_program.run_indirect(resources.timed_candidate_dispatch_args)
        pipeline._sync_storage_and_indirect_writes(ctx)

    program = pipeline.programs["cell_material_side_effects_candidates"]
    program["cell_grid_size"].value = (world.width, world.height)
    cell_state_in.use(location=0)
    velocity_in.use(location=1)
    temp_in.use(location=2)
    resources.local_deferred_lo_out.use(location=3)
    resources.local_deferred_hi_out.use(location=4)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
    resources.timed_candidate_marks.bind_to_storage_buffer(binding=4)
    resources.timed_material_target_list.bind_to_storage_buffer(binding=5)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    if inplace:
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = pipeline._current_cell_textures(resources)
    else:
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = pipeline._next_cell_textures(resources)
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    velocity_out.bind_to_image(4, read=False, write=True)
    timer_out.bind_to_image(5, read=False, write=True)
    resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
    with pipeline._profile_pass(world, "timed_material_targets_apply"):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal timed material side-effect apply requires indirect dispatch")
        program.run_indirect(resources.timed_material_target_dispatch_args)
        pipeline._sync_compute_writes(ctx)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)



def _run_packed_timed_material_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    copy_velocity_passthrough: bool = False,
    velocity_in_place: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if copy_velocity_passthrough:
        with pipeline._profile_pass(world, "packed_timed_material_velocity_copy"):
            pipeline._copy_current_velocity_to_next_role(
                world,
                resources,
                group_x,
                group_y,
            )

    pipeline._clear_packed_timed_material_target_worklist(world, resources)

    cell_state_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    compact_program = pipeline.programs["compact_packed_timed_material_targets"]
    compact_program["cell_grid_size"].value = (world.width, world.height)
    resources.local_deferred_packed_out.use(location=0)
    cell_state_in.use(location=1)
    velocity_in.use(location=2)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
    resources.timed_material_target_list.bind_to_storage_buffer(binding=2)
    resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=3)
    resources.timed_material_target_marks.bind_to_storage_buffer(binding=4)
    with pipeline._profile_pass(world, "packed_timed_material_targets_compact"):
        compact_program.run(group_x, group_y, 1)
        pipeline._sync_storage_and_indirect_writes(ctx)

    pipeline._run_packed_material_target_apply_pass(
        world,
        resources,
        velocity_in_place=velocity_in_place,
        deferred_hi_valid=False,
        profile_name="packed_timed_material_targets_apply",
    )



def _clear_packed_timed_material_target_worklist(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    preserve_timed_candidates: bool = False,
) -> None:
    clear_program = pipeline.programs[
        "clear_timed_material_target_worklist"
        if preserve_timed_candidates
        else "clear_timed_candidate_worklist"
    ]
    resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
    if preserve_timed_candidates:
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=1)
    else:
        resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=2)
    with pipeline._profile_pass(world, "packed_timed_material_targets_clear"):
        clear_program.run(1, 1, 1)
        pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)



def _run_produced_packed_timed_material_side_effect_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    copy_velocity_passthrough: bool,
    velocity_in_place: bool,
    core_in_place: bool = False,
) -> None:
    if copy_velocity_passthrough:
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with pipeline._profile_pass(world, "packed_timed_material_velocity_copy"):
            pipeline._copy_current_velocity_to_next_role(
                world,
                resources,
                group_x,
                group_y,
            )
    dispatch_program = pipeline.programs["build_packed_material_target_dispatch"]
    resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
    resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=1)
    resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=2)
    with pipeline._profile_pass(world, "packed_timed_material_targets_dispatch"):
        dispatch_program.run(1, 1, 1)
        pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)
    pipeline._run_packed_material_target_apply_pass(
        world,
        resources,
        velocity_in_place=velocity_in_place,
        deferred_hi_valid=False,
        profile_name="packed_timed_material_targets_apply",
        core_in_place=core_in_place,
    )



def _run_packed_material_target_apply_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    velocity_in_place: bool,
    deferred_hi_valid: bool,
    profile_name: str,
    core_in_place: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    cell_state_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    program = pipeline.programs["cell_material_side_effects_packed_targets"]
    program["cell_grid_size"].value = (world.width, world.height)
    pipeline._set_uniform_if_present(program, "velocity_in_place", bool(velocity_in_place))
    pipeline._set_uniform_if_present(program, "deferred_hi_valid", bool(deferred_hi_valid))
    cell_state_in.use(location=0)
    velocity_in.use(location=1)
    temp_in.use(location=2)
    resources.local_deferred_packed_out.use(location=3)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
    resources.timed_material_target_list.bind_to_storage_buffer(binding=4)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    output_textures = (
        pipeline._current_cell_textures(resources)
        if core_in_place
        else pipeline._next_cell_textures(resources)
    )
    cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = output_textures
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    velocity_out.bind_to_image(4, read=False, write=True)
    timer_out.bind_to_image(5, read=False, write=True)
    resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
    if velocity_in_place:
        velocity_in.bind_to_image(7, read=True, write=True)
    with pipeline._profile_pass(world, profile_name):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal packed material side effects require indirect dispatch")
        program.run_indirect(resources.timed_material_target_dispatch_args)
        pipeline._sync_compute_writes(ctx)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
