from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    GPUDeferredActionBatch,
    GPUReactionResources,
    LOCAL_SIZE,
)


def _run_cell_pass(
    pipeline,
    world: "WorldEngine",
    program_name: str,
    compiled_actions: tuple[np.ndarray, np.ndarray],
    rule_i: np.ndarray,
    rule_f: np.ndarray,
    rule_tags: np.ndarray,
    rule_count: int,
    solve_cell_mask: object | None,
    lhs_rule_candidate_masks: np.ndarray | None = None,
    light_dose_guard_buffer: Any | None = None,
) -> GPUDeferredActionBatch:
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    has_rhs_consume = pipeline._compiled_rules_include_rhs_consume(rule_tags)
    modifies_gas = pipeline._compiled_actions_include_modify_gas(compiled_actions)
    gas_side_effects_required = modifies_gas or (program_name == "material_gas" and has_rhs_consume)
    with pipeline._profile_pass(world, f"{program_name}_upload_state"):
        pipeline._upload_state(
            world,
            resources,
            reaction_group=program_name,
            compiled_actions=compiled_actions,
            publishes_gas=gas_side_effects_required,
            light_dose_guard_buffer=light_dose_guard_buffer,
        )
    upload_cell_mask, upload_gas_mask = pipeline._active_masks_for_cell_reaction_upload(
        world,
        solve_cell_mask,
        reaction_group=program_name,
    )
    with pipeline._profile_pass(world, f"{program_name}_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            upload_cell_mask,
            upload_gas_mask,
            reaction_group=program_name,
            light_dose_guard_buffer=light_dose_guard_buffer,
        )
    rule_i_buffer = getattr(resources, f"{program_name[:2]}_rule_i", None)
    rule_f_buffer = getattr(resources, f"{program_name[:2]}_rule_f", None)
    rule_tags_buffer = None
    if program_name == "material_material":
        rule_i_buffer = resources.mm_rule_i
        rule_f_buffer = resources.mm_rule_f
        rule_tags_buffer = resources.mm_rule_tags
    elif program_name == "material_gas":
        rule_i_buffer = resources.mg_rule_i
        rule_f_buffer = resources.mg_rule_f
        rule_tags_buffer = resources.mg_rule_tags
    elif program_name == "material_light":
        rule_i_buffer = resources.ml_rule_i
        rule_f_buffer = resources.ml_rule_f
        rule_tags_buffer = resources.ml_rule_tags
    assert rule_i_buffer is not None and rule_f_buffer is not None and rule_tags_buffer is not None
    with pipeline._profile_pass(world, f"{program_name}_upload_metadata"):
        pipeline._upload_local_metadata(world, resources)
        resources.action_i.write(compiled_actions[0].tobytes())
        resources.action_f.write(compiled_actions[1].tobytes())
        rule_i_buffer.write(rule_i.tobytes())
        rule_f_buffer.write(rule_f.tobytes())
        rule_tags_buffer.write(rule_tags.tobytes())
        if lhs_rule_candidate_masks is not None:
            resources.rule_lhs_candidate_masks.write(lhs_rule_candidate_masks.tobytes())
    program = pipeline.programs[program_name]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", rule_count)
    pipeline._set_uniform_if_present(program, "rule_candidate_word_count", pipeline._rule_candidate_word_count(rule_count))
    pipeline._set_uniform_if_present(program, "has_rhs_consume", has_rhs_consume)
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    material_in.use(location=0)
    phase_in.use(location=1)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    velocity_in.use(location=8)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    rule_i_buffer.bind_to_storage_buffer(binding=3)
    rule_f_buffer.bind_to_storage_buffer(binding=4)
    rule_tags_buffer.bind_to_storage_buffer(binding=5)
    resources.material_tags.bind_to_storage_buffer(binding=6)
    resources.gas_tags.bind_to_storage_buffer(binding=7)
    resources.material_slots_lo.bind_to_storage_buffer(binding=8)
    resources.material_slots_hi.bind_to_storage_buffer(binding=9)
    resources.action_meta.bind_to_storage_buffer(binding=10)
    resources.random_targets.bind_to_storage_buffer(binding=11)
    if program_name in {"material_material", "material_gas", "material_light"}:
        resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    direct_core_outputs = pipeline._formal_gpu_frame(world)
    may_have_flow_sources = (
        modifies_gas
        and pipeline._compiled_actions_include_flow_sources(compiled_actions)
    )
    direct_action_gas_delta = (
        direct_core_outputs
        and modifies_gas
        and not has_rhs_consume
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_segment_batch_key is not None
    )
    modify_gas_layer_mask = pipeline._compiled_modify_gas_layer_mask(
        compiled_actions,
        world.gas_concentration.shape[0],
    )
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", bool(direct_action_gas_delta))
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
    if direct_action_gas_delta:
        assert pipeline._formal_segment_batch_key is not None
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
    pipeline._bind_local_cell_action_output_images(resources, direct_core_outputs=direct_core_outputs)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, f"{program_name}_shader"):
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
    apply_material_side_effects = pipeline._compiled_actions_include_emit_material(compiled_actions)
    material_side_effect_copies_velocity = bool(direct_core_outputs and apply_material_side_effects)
    if direct_core_outputs:
        if not material_side_effect_copies_velocity:
            with pipeline._profile_pass(world, f"{program_name}_velocity_copy"):
                pipeline._copy_current_velocity_to_next_role(
                    world,
                    resources,
                    group_x,
                    group_y,
                )
    else:
        with pipeline._profile_pass(world, f"{program_name}_scatter"):
            pipeline._scatter_local_cell_action_outputs(
                world,
                resources,
                group_x,
                group_y,
            )
    if apply_material_side_effects:
        with pipeline._profile_pass(world, f"{program_name}_material_side_effects"):
            pipeline._run_cell_material_side_effect_pass(
                world,
                resources,
                direct_core_outputs=direct_core_outputs,
                copy_velocity_passthrough=material_side_effect_copies_velocity,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
    if gas_side_effects_required:
        with pipeline._profile_pass(world, f"{program_name}_gas_side_effects"):
            pipeline._run_cell_gas_side_effect_pass(
                world,
                resources,
                apply_action_side_effects=modifies_gas,
                material_gas_rule_count=rule_count if program_name == "material_gas" and has_rhs_consume else 0,
                may_have_flow_sources=may_have_flow_sources,
                modify_gas_layer_mask=modify_gas_layer_mask,
                direct_core_outputs=direct_core_outputs,
                action_gas_delta_already_applied=direct_action_gas_delta,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
    if program_name == "material_light" and has_rhs_consume:
        with pipeline._profile_pass(world, f"{program_name}_dose_consume"):
            pipeline._run_material_light_dose_consume_pass(
                world,
                resources,
                rule_count,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
    with pipeline._profile_pass(world, f"{program_name}_publish_cell_state"):
        pipeline._download_cell_state(world, resources, direct_core_outputs=direct_core_outputs)
    with pipeline._profile_pass(world, f"{program_name}_publish_deferred"):
        return pipeline._download_deferred_batch(world, resources)



def _run_local_cell_action_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    program_name: str,
    *,
    self_rule_count: int = 0,
    apply_material_side_effects: bool = False,
    apply_gas_side_effects: bool = False,
    modify_gas_layer_mask: int | None = None,
    may_have_flow_sources: bool = True,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
) -> None:
    program = pipeline.programs[program_name]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", 0)
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "self_rule_count", int(self_rule_count))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    direct_core_outputs = pipeline._formal_gpu_frame(world)
    direct_action_gas_delta = (
        direct_core_outputs
        and program_name == "timed_apply"
        and apply_gas_side_effects
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_segment_batch_key is not None
    )
    if modify_gas_layer_mask is None:
        modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", bool(direct_action_gas_delta))
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
    if direct_action_gas_delta:
        assert pipeline._formal_segment_batch_key is not None
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
    material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    material_in.use(location=0)
    phase_in.use(location=1)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    velocity_in.use(location=8)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.mm_rule_i.bind_to_storage_buffer(binding=3)
    resources.mm_rule_f.bind_to_storage_buffer(binding=4)
    resources.mm_rule_tags.bind_to_storage_buffer(binding=5)
    resources.material_tags.bind_to_storage_buffer(binding=6)
    resources.gas_tags.bind_to_storage_buffer(binding=7)
    resources.material_slots_lo.bind_to_storage_buffer(binding=8)
    resources.material_slots_hi.bind_to_storage_buffer(binding=9)
    resources.action_meta.bind_to_storage_buffer(binding=10)
    resources.random_targets.bind_to_storage_buffer(binding=11)
    resources.self_rule_i.bind_to_storage_buffer(binding=12)
    resources.self_rule_f.bind_to_storage_buffer(binding=13)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    if direct_action_gas_delta:
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
    pipeline._bind_local_cell_action_output_images(resources, direct_core_outputs=direct_core_outputs)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, f"{program_name}_shader"):
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(world.bridge.ctx)
    material_side_effect_copies_velocity = bool(direct_core_outputs and apply_material_side_effects)
    if direct_core_outputs:
        if not material_side_effect_copies_velocity:
            with pipeline._profile_pass(world, f"{program_name}_velocity_copy"):
                pipeline._copy_current_velocity_to_next_role(world, resources, group_x, group_y)
    else:
        with pipeline._profile_pass(world, f"{program_name}_scatter"):
            pipeline._scatter_local_cell_action_outputs(
                world,
                resources,
                group_x,
                group_y,
            )
    if apply_material_side_effects:
        with pipeline._profile_pass(world, f"{program_name}_material_side_effects"):
            pipeline._run_cell_material_side_effect_pass(
                world,
                resources,
                direct_core_outputs=direct_core_outputs,
                copy_velocity_passthrough=material_side_effect_copies_velocity,
            )
    if apply_gas_side_effects:
        with pipeline._profile_pass(world, f"{program_name}_gas_side_effects"):
            pipeline._run_cell_gas_side_effect_pass(
                world,
                resources,
                may_have_flow_sources=may_have_flow_sources,
                modify_gas_layer_mask=modify_gas_layer_mask,
                direct_core_outputs=direct_core_outputs,
                action_gas_delta_already_applied=direct_action_gas_delta,
                flow_source_layers=flow_source_layers,
            )



def _run_timed_candidate_action_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    apply_material_side_effects: bool = False,
    apply_gas_side_effects: bool = False,
    modify_gas_layer_mask: int | None = None,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._prepare_timed_candidate_worklist(world, resources)
    if not pipeline._formal_segment_batch_active():
        with pipeline._profile_pass(world, "timed_candidate_clear_local_meta"):
            pipeline._clear_timed_candidate_local_meta(world, resources)

    program = pipeline.programs["timed_apply_candidates"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", 0)
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    material_in.use(location=0)
    phase_in.use(location=1)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    velocity_in.use(location=8)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.mm_rule_i.bind_to_storage_buffer(binding=3)
    resources.mm_rule_f.bind_to_storage_buffer(binding=4)
    resources.mm_rule_tags.bind_to_storage_buffer(binding=5)
    resources.material_tags.bind_to_storage_buffer(binding=6)
    resources.gas_tags.bind_to_storage_buffer(binding=7)
    resources.material_slots_lo.bind_to_storage_buffer(binding=8)
    resources.material_slots_hi.bind_to_storage_buffer(binding=9)
    resources.action_meta.bind_to_storage_buffer(binding=10)
    resources.random_targets.bind_to_storage_buffer(binding=11)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=12)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=13)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    pipeline._bind_local_cell_action_output_images(resources, direct_core_outputs=True)
    with pipeline._profile_pass(world, "timed_apply_candidates_shader"):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal timed reaction candidate apply requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.timed_candidate_dispatch_args)
        pipeline._sync_compute_writes(ctx)
        pipeline._sync_storage_and_indirect_writes(ctx)

    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "timed_apply_candidates_velocity_copy"):
        pipeline._copy_current_velocity_to_next_role(world, resources, group_x, group_y)

    if apply_material_side_effects:
        with pipeline._profile_pass(world, "timed_apply_material_side_effects"):
            pipeline._run_cell_material_side_effect_pass(
                world,
                resources,
                direct_core_outputs=True,
                timed_candidate_outputs=True,
            )
    if apply_gas_side_effects:
        with pipeline._profile_pass(world, "timed_apply_gas_side_effects"):
            pipeline._run_cell_gas_side_effect_pass(
                world,
                resources,
                modify_gas_layer_mask=modify_gas_layer_mask,
                direct_core_outputs=True,
                timed_candidate_outputs=True,
            )



def _prepare_timed_candidate_worklist(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    setup_program = pipeline.programs["clear_timed_candidate_worklist"]
    resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
    resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=2)
    with pipeline._profile_pass(world, "timed_candidates_clear"):
        setup_program.run(1, 1, 1)
        pipeline._sync_storage_and_indirect_writes(ctx)

    compact_program = pipeline.programs["compact_timed_candidates"]
    compact_program["cell_grid_size"].value = (world.width, world.height)
    material_in, _phase_in, _temp_in, _integrity_in, _velocity_in, timer_in = pipeline._current_cell_textures(resources)
    material_in.use(location=0)
    timer_in.use(location=1)
    resources.active_cell_tex.use(location=2)
    resources.material_slots_lo.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=2)
    resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=3)
    resources.timed_candidate_marks.bind_to_storage_buffer(binding=4)
    with pipeline._profile_pass(world, "timed_candidates_compact"):
        compact_program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        pipeline._sync_storage_and_indirect_writes(ctx)



def _clear_timed_candidate_local_meta(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    program = pipeline.programs["clear_timed_candidate_local_meta"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.local_cell_meta_out.bind_to_image(0, read=False, write=True)
    program.run(
        (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(ctx)



def _publish_timed_candidate_cell_state(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_gpu_frame(world):
        pipeline._download_cell_state(world, resources)
        return
    rotate_formal_cell_roles = pipeline._formal_before_motion_cell_roles_active()
    if pipeline._formal_segment_batch_active():
        pipeline._accumulate_timed_candidate_segment_cell_transient_state(world, resources)
        if rotate_formal_cell_roles:
            pipeline._advance_formal_cell_read_role()
        else:
            pipeline._promote_cell_pong_to_ping(world, resources)
        pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
    else:
        if rotate_formal_cell_roles:
            source_role = pipeline._formal_cell_write_role()
            pipeline._publish_bridge_cell_state(
                world,
                resources,
                source_role=source_role,
                cell_meta_texture=resources.local_cell_meta_out,
            )
            pipeline._set_formal_cell_read_role(source_role)
        else:
            pipeline._publish_bridge_cell_state(
                world,
                resources,
                cell_meta_texture=resources.local_cell_meta_out,
            )
            pipeline._promote_cell_pong_to_ping(world, resources)
    pipeline.last_cpu_mirror_downloaded = False



def _accumulate_timed_candidate_segment_cell_transient_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    ctx = world.bridge.ctx
    if ctx is None:
        return
    program = pipeline.programs["accumulate_timed_candidate_segment_cell_transient_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.local_cell_meta_out.use(location=0)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
    resources.segment_cell_reset_tex.bind_to_image(0, read=True, write=True)
    resources.segment_reaction_latched_tex.bind_to_image(1, read=True, write=True)
    with pipeline._profile_pass(world, "accumulate_timed_candidate_segment_cell_transient_state"):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal timed candidate segment accumulation requires indirect dispatch")
        program.run_indirect(resources.timed_candidate_dispatch_args)
        pipeline._sync_compute_writes(ctx)



def _bind_local_cell_action_output_images(
    pipeline,
    resources: GPUReactionResources,
    *,
    direct_core_outputs: bool,
) -> None:
    if direct_core_outputs:
        material_out, phase_out, temp_out, integrity_out, _velocity_out, timer_out = pipeline._next_cell_textures(resources)
    else:
        material_out = resources.local_material_out
        phase_out = resources.local_phase_out
        temp_out = resources.local_temp_out
        integrity_out = resources.local_integrity_out
        timer_out = resources.local_timer_out
    material_out.bind_to_image(0, read=False, write=True)
    phase_out.bind_to_image(1, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    timer_out.bind_to_image(4, read=False, write=True)
    resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
    resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
    resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)



def _scatter_local_cell_action_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    group_x: int,
    group_y: int,
    *,
    core_outputs_direct: bool = False,
) -> None:
    if core_outputs_direct:
        pipeline._copy_current_velocity_to_next_role(world, resources, group_x, group_y)
        return

    program = pipeline.programs["scatter_local_action_outputs"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    resources.local_material_out.use(location=0)
    resources.local_phase_out.use(location=1)
    resources.local_temp_out.use(location=2)
    resources.local_integrity_out.use(location=3)
    resources.local_timer_out.use(location=4)
    resources.local_deferred_lo_out.use(location=5)
    resources.local_deferred_hi_out.use(location=6)
    resources.local_cell_meta_out.use(location=7)
    resources.material_pong.bind_to_image(0, read=False, write=True)
    resources.phase_pong.bind_to_image(1, read=False, write=True)
    resources.temp_pong.bind_to_image(2, read=False, write=True)
    resources.integrity_pong.bind_to_image(3, read=False, write=True)
    resources.timer_pong.bind_to_image(4, read=False, write=True)
    resources.trigger_lo_tex.bind_to_image(5, read=False, write=True)
    resources.trigger_hi_tex.bind_to_image(6, read=False, write=True)
    resources.deferred_scale_lo_tex.bind_to_image(7, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(world.bridge.ctx)

    tail_program = pipeline.programs["scatter_local_action_tail_outputs"]
    pipeline._set_uniform_if_present(tail_program, "cell_grid_size", (world.width, world.height))
    resources.local_deferred_hi_out.use(location=5)
    resources.local_cell_meta_out.use(location=7)
    resources.deferred_scale_hi_tex.bind_to_image(0, read=False, write=True)
    resources.cell_reset_tex.bind_to_image(1, read=False, write=True)
    resources.reaction_latched_tex.bind_to_image(2, read=False, write=True)
    tail_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(world.bridge.ctx)



def _copy_current_velocity_to_next_role(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    group_x: int,
    group_y: int,
    *,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    _material_in, _phase_in, _temp_in, _integrity_in, velocity_in, _timer_in = pipeline._current_cell_textures(resources)
    _material_out, _phase_out, _temp_out, _integrity_out, velocity_out, _timer_out = pipeline._next_cell_textures(resources)
    if velocity_in is velocity_out:
        return
    program = pipeline.programs["copy_reaction_velocity_state"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    velocity_in.use(location=0)
    velocity_out.bind_to_image(0, read=False, write=True)
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



def _scatter_local_emit_cell_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    program = pipeline.programs["scatter_local_emit_cell_outputs"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    resources.local_emit_cell_lo_out.use(location=0)
    resources.local_emit_cell_hi_out.use(location=1)
    resources.local_timer_out.use(location=2)
    resources.local_cell_meta_out.use(location=3)
    resources.material_pong.bind_to_image(0, read=False, write=True)
    resources.phase_pong.bind_to_image(1, read=False, write=True)
    resources.temp_pong.bind_to_image(2, read=False, write=True)
    resources.integrity_pong.bind_to_image(3, read=False, write=True)
    resources.velocity_pong.bind_to_image(4, read=False, write=True)
    resources.timer_pong.bind_to_image(5, read=False, write=True)
    resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
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

