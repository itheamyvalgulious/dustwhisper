from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    GPUDeferredActionBatch,
    GPUReactionMaterialPairPlan,
    GPUReactionResources,
    LOCAL_SIZE,
    MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
    MAX_ACTIONS,
    MAX_MATERIALS,
    MAX_RULES,
    RULE_CANDIDATE_VECS,
    SELF_FUSED_FLOW_SOURCE_BINDING,
    SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING,
    SELF_FUSED_GAS_DELTA_BINDING,
)
from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _material_participation_flags,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
)
from oracle_game.types import Phase, ReactionType


def _upload_material_pair_plan(resources: GPUReactionResources, plan: GPUReactionMaterialPairPlan) -> bool:
    if resources.material_pair_plan_upload_key == plan.cache_key:
        return False
    resources.material_pair_action_i.write(plan.compiled_actions[0].tobytes())
    resources.material_pair_action_f.write(plan.compiled_actions[1].tobytes())
    resources.material_pair_rule_i.write(plan.packed_rule_i.tobytes())
    resources.material_pair_rule_f.write(plan.packed_rule_f.tobytes())
    resources.material_pair_rule_tags.write(plan.packed_rule_tags.tobytes())
    resources.material_pair_lhs_candidate_masks.write(plan.packed_lhs_candidate_masks.tobytes())
    resources.material_pair_plan_upload_key = plan.cache_key
    return True


def _post_pair_gas_actions_preserve_terminal(world: "WorldEngine") -> bool:
    tables = world.bridge.shadow_typed_tables
    action_table = tables["reaction_action_table"]
    safe_types = {int(ReactionType.NONE.value), int(ReactionType.MODIFY_GAS.value)}
    for table_name in ("gas_gas_rule_table", "gas_light_rule_table"):
        for rule in tables[table_name]:
            if int(rule["trigger_slot_index"]) >= 0:
                return False
            action_index = int(rule["result_action"])
            if action_index < 0:
                continue
            if action_index >= int(action_table.shape[0]):
                return False
            action = action_table[action_index]
            if int(action["reaction_type_id"]) not in safe_types:
                return False
            if (
                int(action["reaction_type_id"]) == int(ReactionType.MODIFY_GAS.value)
                and float(action["strength"]) > 0.0
                and int(action["range_cells"]) > 0
            ):
                return False
    return True


def _can_run_material_pair_terminal(
    pipeline,
    world: "WorldEngine",
    plan: GPUReactionMaterialPairPlan,
    material_light_dose_guard: Any | None,
) -> bool:
    motion_pipeline = getattr(getattr(world, "motion_solver", None), "gpu_pipeline", None)
    authoritative = world.bridge.gpu_authoritative_resources
    terminal_dt = getattr(world, "_reaction_motion_terminal_dt", None)
    return bool(
        pipeline._material_triplet_motion_terminal_enabled
        and plan.material_light_rule_count > 0
        and material_light_dose_guard is not None
        and pipeline._formal_gpu_frame(world)
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_state_key_is_before_motion()
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and motion_pipeline is not None
        and motion_pipeline.can_consume_reaction_handoff(world)
        and terminal_dt is not None
        and np.isfinite(float(terminal_dt))
        and {"flow_velocity", "active_tile_ttl"}.issubset(authoritative)
        and world.bridge.buffers.get("cell_core") is not None
        and _post_pair_gas_actions_preserve_terminal(world)
    )


def _upload_material_pair_terminal_tables(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    plan: GPUReactionMaterialPairPlan,
) -> int:
    tables = world.bridge.shadow_typed_tables
    material_table = tables["material_table"]
    gas_table = tables["gas_table"]
    action_table = tables["reaction_action_table"]
    material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
    material_key = (
        int(world.bridge.table_generations.get("materials", 0)),
        int(world.bridge.table_generations.get("gases", 0)),
        material_count,
        int(gas_table.shape[0]),
    )
    if resources.material_pair_terminal_material_upload_key != material_key:
        reaction_params = np.zeros((MAX_MATERIALS, 4), dtype=np.float32)
        material_tags = np.zeros((MAX_MATERIALS, 4), dtype=np.uint32)
        gas_tags = np.zeros((MAX_MATERIALS, 4), dtype=np.uint32)
        slots_lo = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
        slots_hi = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
        motion_params = np.zeros((MAX_MATERIALS, 4), dtype=np.float32)
        participation = np.zeros((MAX_MATERIALS,), dtype=np.uint32)
        if material_count > 0:
            rows = material_table[:material_count]
            reaction_params[:material_count, 0] = rows["base_integrity"]
            reaction_params[:material_count, 1] = rows["default_phase"]
            reaction_params[:material_count, 2] = rows["spawn_temperature"]
            material_tags[:material_count, 0] = rows["material_tag_mask"]
            material_tags[:material_count, 1] = rows["gas_tag_mask"]
            material_tags[:material_count, 2] = rows["light_tag_mask"]
            slots_lo[:material_count] = rows["reaction_slots"][:, :4]
            slots_hi[:material_count] = rows["reaction_slots"][:, 4:8]
            motion_params[:material_count, 0] = rows["max_dda_step"]
            motion_params[:material_count, 1] = rows["gravity_scale"]
            motion_params[:material_count, 2] = rows["wind_coupling"]
            motion_params[:material_count, 3] = rows["drag_scale"]
        gas_count = min(MAX_MATERIALS, int(gas_table.shape[0]))
        if gas_count > 0:
            gas_tags[:gas_count, 0] = gas_table[:gas_count]["material_reaction_tag_mask"]
            gas_tags[:gas_count, 1] = gas_table[:gas_count]["light_reaction_tag_mask"]
        source_participation = _material_participation_flags(world)
        participation[: min(MAX_MATERIALS, int(source_participation.size))] = source_participation[:MAX_MATERIALS]
        resources.material_pair_terminal_material_tables.write(
            b"".join(
                array.tobytes()
                for array in (
                    reaction_params,
                    material_tags,
                    gas_tags,
                    slots_lo,
                    slots_hi,
                    motion_params,
                    participation,
                )
            )
        )
        resources.material_pair_terminal_material_upload_key = material_key

    action_key = (
        plan.cache_key,
        int(world.bridge.table_generations.get("reactions", 0)),
        resources.random_targets_signature,
    )
    if resources.material_pair_terminal_action_upload_key != action_key:
        action_meta = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
        action_count = min(MAX_ACTIONS, int(action_table.shape[0]))
        action_meta[:action_count, 0] = action_table[:action_count]["duration"]
        resources.material_pair_terminal_action_tables.write(
            b"".join(
                (
                    plan.compiled_actions[0].tobytes(),
                    plan.compiled_actions[1].tobytes(),
                    action_meta.tobytes(),
                    pipeline.random_targets.astype(np.int32, copy=False).tobytes(),
                )
            )
        )
        resources.material_pair_terminal_action_upload_key = action_key

    if resources.material_pair_terminal_rule_upload_key != plan.cache_key:
        resources.material_pair_terminal_rule_tables.write(
            b"".join(
                (
                    plan.packed_rule_i.tobytes(),
                    plan.packed_rule_f.tobytes(),
                    plan.packed_rule_tags.tobytes(),
                    plan.packed_lhs_candidate_masks.tobytes(),
                )
            )
        )
        resources.material_pair_terminal_rule_upload_key = plan.cache_key
    return material_count


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
    direct_bridge_cell_dose = bool(
        pipeline._formal_gpu_frame(world)
        and program_name == "material_light"
        and not has_rhs_consume
    )
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
            direct_bridge_cell_dose=direct_bridge_cell_dose,
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
    packed_light_dose_guard = bool(
        direct_bridge_cell_dose and light_dose_guard_buffer is not None
    )
    with pipeline._profile_pass(world, f"{program_name}_upload_metadata"):
        pipeline._upload_local_metadata(world, resources)
        resources.action_i.write(compiled_actions[0].tobytes())
        resources.action_f.write(compiled_actions[1].tobytes())
        rule_i_buffer.write(rule_i.tobytes())
        rule_f_buffer.write(rule_f.tobytes())
        rule_tags_buffer.write(rule_tags.tobytes())
        if lhs_rule_candidate_masks is not None:
            resources.rule_lhs_candidate_masks.write(lhs_rule_candidate_masks.tobytes())
        if packed_light_dose_guard:
            world.bridge.ctx.copy_buffer(
                rule_tags_buffer,
                light_dose_guard_buffer,
                size=np.dtype(np.uint32).itemsize,
                write_offset=MAX_RULES * 4 * np.dtype(np.uint32).itemsize,
            )
            pipeline._sync_compute_writes(world.bridge.ctx)
    program_key = (
        f"{program_name}_authoritative_lhs"
        if program_name in {"material_material", "material_gas", "material_light"}
        and pipeline._authoritative_lhs_candidate_masks_enabled
        else program_name
    )
    program = pipeline.programs[program_key]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", rule_count)
    pipeline._set_uniform_if_present(program, "rule_candidate_word_count", pipeline._rule_candidate_word_count(rule_count))
    pipeline._set_uniform_if_present(program, "has_rhs_consume", has_rhs_consume)
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "use_bridge_cell_dose", direct_bridge_cell_dose)
    pipeline._set_uniform_if_present(program, "use_packed_light_dose_guard", packed_light_dose_guard)
    pipeline._set_uniform_if_present(program, "light_dose_guard_rule_index", MAX_RULES)
    cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    cell_state_in.use(location=0)
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
    if direct_bridge_cell_dose:
        world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=16)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    direct_core_outputs = pipeline._formal_gpu_frame(world)
    accumulate_segment_cell_meta = bool(
        direct_core_outputs
        and pipeline._formal_segment_batch_active()
        and pipeline._pair_segment_meta_fusion_enabled
    )
    if accumulate_segment_cell_meta:
        pipeline._ensure_formal_segment_meta_physical_zero(world, resources)
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
    write_deferred_outputs = bool(
        not direct_core_outputs
        or pipeline._compiled_actions_require_deferred_outputs(
            compiled_actions,
            direct_modify_gas=bool(direct_action_gas_delta and not may_have_flow_sources),
        )
    )
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", bool(direct_action_gas_delta))
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
    pipeline._set_uniform_if_present(program, "write_deferred_outputs", write_deferred_outputs)
    if direct_action_gas_delta:
        assert pipeline._formal_segment_batch_key is not None
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
        # The clear pass temporarily owns SSBO binding 0. Restore material
        # params and bind the direct-delta storage explicitly before dispatch.
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
    pipeline._set_uniform_if_present(
        program,
        "accumulate_segment_cell_meta",
        accumulate_segment_cell_meta,
    )
    pipeline._bind_local_cell_action_output_images(
        resources,
        direct_core_outputs=direct_core_outputs,
        accumulate_segment_cell_meta=accumulate_segment_cell_meta,
    )
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, f"{program_name}_shader"):
        if light_dose_guard_buffer is not None and not packed_light_dose_guard:
            dispatch_args = pipeline._build_light_dose_guarded_dispatch_args(
                world,
                resources,
                light_dose_guard_buffer,
                group_x,
                group_y,
                1,
            )
            if lhs_rule_candidate_masks is not None:
                resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
            resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal light-dose guarded reactions require indirect dispatch")
            program.run_indirect(dispatch_args)
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(world.bridge.ctx)
    apply_material_side_effects = pipeline._compiled_actions_include_emit_material(compiled_actions)
    material_side_effect_copies_velocity = bool(direct_core_outputs and apply_material_side_effects)
    resident_velocity_role = bool(
        direct_core_outputs and pipeline._formal_before_motion_cell_roles_active()
    )
    sparse_velocity_side_effects = bool(
        material_side_effect_copies_velocity and resident_velocity_role
    )
    if direct_core_outputs:
        if not material_side_effect_copies_velocity and not resident_velocity_role:
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
                copy_velocity_passthrough=bool(
                    material_side_effect_copies_velocity and not sparse_velocity_side_effects
                ),
                velocity_in_place=sparse_velocity_side_effects,
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
        pipeline._download_cell_state(
            world,
            resources,
            direct_core_outputs=direct_core_outputs,
            segment_cell_meta_already_accumulated=accumulate_segment_cell_meta,
            advance_velocity_role=bool(
                material_side_effect_copies_velocity
                and resident_velocity_role
                and not sparse_velocity_side_effects
            ),
        )
    with pipeline._profile_pass(world, f"{program_name}_publish_deferred"):
        return pipeline._download_deferred_batch(world, resources)



def _run_material_pair_fused_pass(
    pipeline,
    world: "WorldEngine",
    plan: GPUReactionMaterialPairPlan,
    solve_cell_mask: object,
    *,
    material_light_dose_guard: Any | None = None,
) -> GPUDeferredActionBatch:
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    compiled_actions = plan.compiled_actions
    material_material_rule_count = plan.material_material_rule_count
    rule_count = plan.rule_count
    material_light_rule_count = plan.material_light_rule_count
    material_light_packed_descriptors = plan.material_light_packed_descriptors
    material_pair_packed_descriptors = plan.material_pair_packed_descriptors
    lhs_rule_candidate_masks = plan.packed_lhs_candidate_masks
    modifies_gas = plan.modifies_gas
    profile_prefix = "material_triplet_fused" if material_light_rule_count > 0 else "material_pair_fused"
    pipeline.last_material_pair_terminal_handoff = False
    with pipeline._profile_pass(world, f"{profile_prefix}_upload_state"):
        pipeline._upload_state(
            world,
            resources,
            reaction_group="material_pair_fused",
            compiled_actions=compiled_actions,
            publishes_gas=modifies_gas,
        )
    terminal_handoff = _can_run_material_pair_terminal(
        pipeline,
        world,
        plan,
        material_light_dose_guard,
    )
    use_expanded_active_tile_mask = bool(
        terminal_handoff
        and pipeline._can_use_expanded_active_tile_mask(world)
    )
    upload_cell_mask, upload_gas_mask = pipeline._active_masks_for_cell_reaction_upload(
        world,
        solve_cell_mask,
        reaction_group="material_pair_fused",
    )
    with pipeline._profile_pass(world, f"{profile_prefix}_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            upload_cell_mask,
            upload_gas_mask,
            reaction_group="material_pair_fused",
            use_expanded_tile_mask=use_expanded_active_tile_mask,
    )
    with pipeline._profile_pass(world, f"{profile_prefix}_upload_metadata"):
        pipeline._upload_local_metadata(world, resources)
        _upload_material_pair_plan(resources, plan)
        if material_light_rule_count > 0:
            assert material_light_dose_guard is not None
            world.bridge.ctx.copy_buffer(
                resources.material_pair_rule_tags,
                material_light_dose_guard,
                size=np.dtype(np.uint32).itemsize,
                write_offset=MAX_RULES * 2 * 4 * np.dtype(np.uint32).itemsize,
            )
            pipeline._sync_compute_writes(world.bridge.ctx)

    terminal_material_count = 0
    if terminal_handoff:
        terminal_material_count = _upload_material_pair_terminal_tables(
            pipeline,
            world,
            resources,
            plan,
        )
        assert material_light_dose_guard is not None
        terminal_guard_offset = (
            plan.packed_rule_i.nbytes
            + plan.packed_rule_f.nbytes
            + MAX_RULES * 2 * 4 * np.dtype(np.uint32).itemsize
        )
        world.bridge.ctx.copy_buffer(
            resources.material_pair_terminal_rule_tables,
            material_light_dose_guard,
            size=np.dtype(np.uint32).itemsize,
            write_offset=terminal_guard_offset,
        )
        pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)

    if not pipeline._formal_segment_batch_active() or pipeline._formal_segment_batch_key is None:
        raise RuntimeError("material pair state fusion requires an active formal segment batch")
    terminal_32x8 = bool(
        terminal_handoff
        and pipeline._material_triplet_terminal_32x8_enabled
        and pipeline._material_triplet_terminal_shared_transpose_enabled
        and pipeline._material_triplet_terminal_local16_enabled
        and pipeline._material_triplet_terminal_dirty_fast_equal_enabled
    )
    terminal_segment_meta_zero = bool(
        terminal_32x8
        and pipeline._can_use_terminal_segment_meta_zero()
    )
    pipeline.last_terminal_segment_meta_lazy_zero_used = terminal_segment_meta_zero
    if not terminal_segment_meta_zero:
        pipeline._ensure_formal_segment_meta_physical_zero(world, resources)
    program = pipeline.programs[
        "material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose_segment_zero"
        if terminal_segment_meta_zero
        else "material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose"
        if terminal_32x8
        else "material_pair_fused_terminal_local16_dirty_fast_shared_transpose"
        if (
            terminal_handoff
            and pipeline._material_triplet_terminal_shared_transpose_enabled
            and pipeline._material_triplet_terminal_local16_enabled
            and pipeline._material_triplet_terminal_dirty_fast_equal_enabled
        )
        else
        "material_pair_fused_terminal_local16_dirty_fast"
        if terminal_handoff
        and pipeline._material_triplet_terminal_local16_enabled
        and pipeline._material_triplet_terminal_dirty_fast_equal_enabled
        else "material_pair_fused_terminal_local16"
        if terminal_handoff and pipeline._material_triplet_terminal_local16_enabled
        else "material_pair_fused_terminal"
        if terminal_handoff
        else "material_pair_fused"
    ]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", rule_count)
    pipeline._set_uniform_if_present(program, "material_material_rule_count", material_material_rule_count)
    pipeline._set_uniform_if_present(program, "material_gas_rule_offset", material_material_rule_count)
    pipeline._set_uniform_if_present(program, "material_light_rule_count", material_light_rule_count)
    pipeline._set_uniform_if_present(
        program,
        "use_material_light_packed_descriptors",
        material_light_packed_descriptors is not None,
    )
    pipeline._set_uniform_if_present(
        program,
        "use_material_pair_packed_descriptors",
        material_pair_packed_descriptors is not None,
    )
    pipeline._set_uniform_if_present(program, "material_light_rule_offset", MAX_RULES)
    pipeline._set_uniform_if_present(
        program,
        "material_light_candidate_vec_offset",
        MAX_MATERIALS * lhs_rule_candidate_masks.shape[1],
    )
    pipeline._set_uniform_if_present(program, "material_light_guard_rule_index", MAX_RULES * 2)
    pipeline._set_uniform_if_present(
        program,
        "material_light_rule_candidate_word_count",
        pipeline._rule_candidate_word_count(material_light_rule_count),
    )
    pipeline._set_uniform_if_present(program, "rule_candidate_word_count", pipeline._rule_candidate_word_count(rule_count))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", modifies_gas)
    pipeline._set_uniform_if_present(
        program,
        "direct_modify_gas_layer_mask",
        plan.direct_modify_gas_layer_mask,
    )
    pipeline._set_uniform_if_present(program, "write_deferred_outputs", False)
    pipeline._set_uniform_if_present(
        program,
        "use_expanded_active_tile_mask",
        use_expanded_active_tile_mask,
    )

    if modifies_gas:
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
    cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    cell_state_in.use(location=0)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    resources.expanded_active_tile_tex.use(location=1)
    velocity_in.use(location=8)
    if terminal_handoff:
        dirty_mask = ensure_collapse_structure_dirty_tile_mask(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_mask is None or dirty_queue is None:
            raise RuntimeError("material triplet terminal requires collapse dirty queue resources")
        dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
        pipeline._set_uniform_if_present(
            program,
            "terminal_tile_grid_size",
            (world.active.tile_width, world.active.tile_height),
        )
        pipeline._set_uniform_if_present(program, "terminal_tile_size", int(world.active.tile_size))
        pipeline._set_uniform_if_present(program, "terminal_material_count", int(terminal_material_count))
        pipeline._set_uniform_if_present(
            program,
            "terminal_phase_falling_island",
            int(Phase.FALLING_ISLAND),
        )
        pipeline._set_uniform_if_present(
            program,
            "terminal_dt",
            float(world._reaction_motion_terminal_dt),
        )
        pipeline._set_uniform_if_present(
            program,
            "terminal_clear_reaction_latched",
            bool(getattr(world.motion_solver.gpu_pipeline, "_reaction_latch_handoff_clear_enabled", False)),
        )
        if not terminal_segment_meta_zero:
            resources.segment_cell_meta_tex.use(location=9)
        world.bridge.textures["flow_velocity"].use(location=10)
        resources.material_pair_terminal_material_tables.bind_to_storage_buffer(binding=0)
        resources.material_pair_terminal_action_tables.bind_to_storage_buffer(binding=1)
        resources.material_pair_terminal_rule_tables.bind_to_storage_buffer(binding=2)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=3)
        world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=4)
        resources.light_emitter_count.bind_to_storage_buffer(binding=5)
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=6)
        dirty_mask.bind_to_storage_buffer(binding=7)
        dirty_count.bind_to_storage_buffer(binding=8)
        dirty_list.bind_to_storage_buffer(binding=9)
        dirty_dispatch_args.bind_to_storage_buffer(binding=10)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=11)
        world.bridge.textures["material"].bind_to_image(0, read=False, write=True)
    else:
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_pair_action_i.bind_to_storage_buffer(binding=1)
        resources.material_pair_action_f.bind_to_storage_buffer(binding=2)
        resources.material_pair_rule_i.bind_to_storage_buffer(binding=3)
        resources.material_pair_rule_f.bind_to_storage_buffer(binding=4)
        resources.material_pair_rule_tags.bind_to_storage_buffer(binding=5)
        resources.material_tags.bind_to_storage_buffer(binding=6)
        resources.gas_tags.bind_to_storage_buffer(binding=7)
        resources.material_slots_lo.bind_to_storage_buffer(binding=8)
        resources.material_slots_hi.bind_to_storage_buffer(binding=9)
        resources.action_meta.bind_to_storage_buffer(binding=10)
        resources.random_targets.bind_to_storage_buffer(binding=11)
        resources.material_pair_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        if material_light_rule_count > 0:
            assert material_light_dose_guard is not None
            world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=14)
        else:
            resources.material_params.bind_to_storage_buffer(binding=14)
        pipeline._bind_local_cell_action_output_images(
            resources,
            direct_core_outputs=True,
            accumulate_segment_cell_meta=True,
        )
    terminal_local_size_x = (
        32
        if terminal_32x8
        else 16
        if terminal_handoff and pipeline._material_triplet_terminal_local16_enabled
        else LOCAL_SIZE
    )
    terminal_local_size_y = 8 if terminal_32x8 else (
        16
        if terminal_handoff and pipeline._material_triplet_terminal_local16_enabled
        else LOCAL_SIZE
    )
    group_x = (world.width + terminal_local_size_x - 1) // terminal_local_size_x
    group_y = (world.height + terminal_local_size_y - 1) // terminal_local_size_y
    with pipeline._profile_pass(world, f"{profile_prefix}_shader"):
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(world.bridge.ctx)
        if terminal_handoff:
            pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)
    if terminal_handoff:
        pipeline.last_material_pair_terminal_handoff = True
        pipeline._motion_handoff_candidate = {
            "terminal_integrated": True,
            "frame_id": int(getattr(world, "frame_id", 0)),
        }
        if bool(getattr(world.motion_solver.gpu_pipeline, "_reaction_latch_handoff_clear_enabled", False)):
            setattr(
                world,
                "_reaction_latches_handoff_cleared_frame_id",
                int(getattr(world, "frame_id", 0)),
            )
        setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
        world.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
        )
    else:
        with pipeline._profile_pass(world, f"{profile_prefix}_publish_cell_state"):
            pipeline._download_cell_state(
                world,
                resources,
                direct_core_outputs=True,
                segment_cell_meta_already_accumulated=True,
            )
    with pipeline._profile_pass(world, f"{profile_prefix}_publish_deferred"):
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
    write_deferred_hi_outputs: bool | None = None,
    fused_gas_output_safe: bool = False,
    inplace: bool = False,
    candidate_dispatch: bool = False,
    timed_sparse_inplace: bool = False,
    use_expanded_active_tile_mask: bool = False,
) -> bool:
    use_expanded_active_tile_mask = bool(
        use_expanded_active_tile_mask
        and not candidate_dispatch
        and not timed_sparse_inplace
        and pipeline._can_use_expanded_active_tile_mask(world)
    )
    direct_core_outputs = pipeline._formal_gpu_frame(world)
    packed_local_deferred_outputs = bool(
        direct_core_outputs
        and program_name in {"timed_apply", "self_apply"}
        and (not candidate_dispatch or timed_sparse_inplace)
    )
    use_packed_timed_emit_target_producer = bool(
        program_name == "timed_apply"
        and packed_local_deferred_outputs
        and apply_material_side_effects
        and pipeline._packed_timed_emit_target_worklist_enabled
        and pipeline._timed_emit_target_producer_enabled
    )
    cell_flag_meta = bool(
        pipeline._formal_segment_batch_active()
        and pipeline._timed_self_cell_flag_meta_enabled
    )
    pipeline.last_timed_self_cell_flag_meta_used = cell_flag_meta
    use_self_gas_fused = bool(
        program_name == "self_apply"
        and not candidate_dispatch
        and packed_local_deferred_outputs
        and apply_gas_side_effects
        and pipeline._self_apply_fused_gas_output_enabled
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_segment_batch_key is not None
    )
    use_self_gas_candidates = bool(
        program_name == "self_apply"
        and not candidate_dispatch
        and packed_local_deferred_outputs
        and apply_gas_side_effects
        and not use_self_gas_fused
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_segment_batch_key is not None
        and pipeline._self_gas_candidate_worklist_enabled
    )
    use_direct_self_rule_spans = bool(
        program_name == "self_apply"
        and packed_local_deferred_outputs
        and not use_self_gas_fused
        and getattr(pipeline, "_self_rule_direct_action_spans_enabled", False)
        and getattr(resources, "self_rule_span_direct_actions", False)
        and pipeline._self_rule_material_spans_enabled
    )
    use_self_cached_cell_state = bool(
        program_name == "self_apply"
        and packed_local_deferred_outputs
        and not candidate_dispatch
        and getattr(pipeline, "_self_apply_cached_cell_state_enabled", False)
    )
    program_key = f"{program_name}_packed" if packed_local_deferred_outputs else program_name
    if use_packed_timed_emit_target_producer:
        program_key = "timed_apply_packed_emit_targets"
    if use_direct_self_rule_spans:
        program_key = "self_apply_packed_direct_spans"
    if use_self_gas_fused:
        program_key = "self_apply_packed_fused_gas"
    if cell_flag_meta:
        program_key = f"{program_key}_cell_flag_meta"
    if use_self_cached_cell_state:
        program_key = f"{program_key}_cached_cell_state"
    if timed_sparse_inplace:
        program_key = "timed_apply_packed_sparse_inplace"
    if candidate_dispatch:
        sparse_suffix = "_sparse"
        if use_direct_self_rule_spans:
            sparse_suffix += "_direct_spans"
        if cell_flag_meta:
            sparse_suffix += "_cell_flag_meta"
        program_key = f"self_apply_packed{sparse_suffix}"
    pipeline.last_self_rule_direct_action_spans_used = use_direct_self_rule_spans
    pipeline.last_self_apply_cached_cell_state_used = use_self_cached_cell_state
    program = pipeline.programs[program_key]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", 0)
    pipeline._set_uniform_if_present(
        program,
        "use_expanded_active_tile_mask",
        use_expanded_active_tile_mask,
    )
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "self_rule_count", int(self_rule_count))
    pipeline._set_uniform_if_present(
        program,
        "use_self_rule_material_spans",
        bool(pipeline._self_rule_material_spans_enabled),
    )
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    direct_action_gas_delta = (
        direct_core_outputs
        and program_name == "timed_apply"
        and apply_gas_side_effects
        and pipeline._formal_segment_batch_active()
        and pipeline._formal_segment_batch_key is not None
    )
    if modify_gas_layer_mask is None:
        modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
    pipeline._set_uniform_if_present(program, "write_deferred_outputs", True)
    if write_deferred_hi_outputs is None:
        write_deferred_hi_outputs = bool(not (program_name == "timed_apply" and direct_core_outputs))
    else:
        write_deferred_hi_outputs = bool(write_deferred_hi_outputs)
    use_packed_self_emit_targets = bool(
        program_name == "self_apply"
        and not candidate_dispatch
        and direct_core_outputs
        and packed_local_deferred_outputs
        and apply_material_side_effects
        and pipeline._packed_self_emit_target_worklist_enabled
    )
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", direct_action_gas_delta)
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
    pipeline._set_uniform_if_present(program, "write_deferred_hi_outputs", write_deferred_hi_outputs)
    pipeline._set_uniform_if_present(program, "collect_self_emit_targets", use_packed_self_emit_targets)
    pipeline._set_uniform_if_present(program, "collect_self_gas_candidates", use_self_gas_candidates)
    pipeline._set_uniform_if_present(
        program,
        "self_fused_flow_sources_enabled",
        bool(use_self_gas_fused and may_have_flow_sources),
    )
    if direct_action_gas_delta:
        assert pipeline._formal_segment_batch_key is not None
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
    elif use_self_gas_fused:
        assert pipeline._formal_segment_batch_key is not None
        pipeline._clear_formal_segment_gas_delta(world, resources, pipeline._formal_segment_batch_key)
    if use_packed_timed_emit_target_producer:
        pipeline._clear_packed_timed_material_target_worklist(
            world,
            resources,
            preserve_timed_candidates=timed_sparse_inplace,
        )
    if use_packed_self_emit_targets or use_self_gas_candidates:
        clear_program = pipeline.programs["clear_timed_candidate_worklist"]
        resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
        resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=2)
        with pipeline._profile_pass(world, "packed_self_material_targets_clear"):
            clear_program.run(1, 1, 1)
            pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)
    cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    cell_state_in.use(location=0)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    resources.expanded_active_tile_tex.use(location=1)
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
    (
        resources.self_rule_span_i if use_direct_self_rule_spans else resources.self_rule_i
    ).bind_to_storage_buffer(binding=12)
    resources.self_rule_f.bind_to_storage_buffer(binding=13)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    if direct_action_gas_delta:
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
    if use_self_gas_fused:
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=SELF_FUSED_GAS_DELTA_BINDING)
    if use_packed_self_emit_targets or use_self_gas_candidates:
        resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
        resources.timed_material_target_list.bind_to_storage_buffer(binding=4)
        resources.timed_material_target_marks.bind_to_storage_buffer(binding=5)
    if use_packed_timed_emit_target_producer or timed_sparse_inplace:
        resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
        resources.timed_material_target_list.bind_to_storage_buffer(binding=4)
        resources.timed_material_target_marks.bind_to_storage_buffer(binding=5)
    if candidate_dispatch:
        resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=4)
        resources.timed_candidate_marks.bind_to_storage_buffer(binding=5)
    if timed_sparse_inplace:
        resources.timed_candidate_list.bind_to_storage_buffer(binding=12)
    if candidate_dispatch or timed_sparse_inplace:
        cell_state_out, _phase_out, temp_out, integrity_out, _velocity_out, timer_out = (
            pipeline._current_cell_textures(resources)
        )
        cell_state_out.bind_to_image(0, read=False, write=True)
        temp_out.bind_to_image(2, read=False, write=True)
        integrity_out.bind_to_image(3, read=False, write=True)
        timer_out.bind_to_image(4, read=False, write=True)
        if packed_local_deferred_outputs:
            resources.local_deferred_packed_out.bind_to_image(5, read=False, write=True)
        else:
            resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
            resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
            resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)
    else:
        pipeline._bind_local_cell_action_output_images(
            resources,
            direct_core_outputs=direct_core_outputs,
            packed_local_deferred_outputs=packed_local_deferred_outputs,
        )
    if use_self_gas_fused:
        resources.flow_source_tex.bind_to_image(
            SELF_FUSED_FLOW_SOURCE_BINDING,
            read=False,
            write=True,
        )
        pipeline._bind_flow_source_generation_output(
            world,
            resources,
            program,
            binding=SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING,
        )
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, f"{program_name}_shader"):
        if candidate_dispatch or timed_sparse_inplace:
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal sparse cell action requires indirect dispatch")
            program.run_indirect(resources.timed_candidate_dispatch_args)
        else:
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(world.bridge.ctx)
        if (
            use_packed_timed_emit_target_producer
            or use_packed_self_emit_targets
            or use_self_gas_candidates
        ):
            pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)
    if program_name in {"timed_apply", "self_apply"}:
        pipeline._record_formal_segment_cell_meta_in_flags(cell_flag_meta)
    if use_packed_self_emit_targets or use_self_gas_candidates:
        dispatch_program = pipeline.programs["build_packed_material_target_dispatch"]
        resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=1)
        resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=2)
        with pipeline._profile_pass(world, "packed_self_material_targets_dispatch"):
            dispatch_program.run(1, 1, 1)
            pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)
    material_side_effect_copies_velocity = bool(direct_core_outputs and apply_material_side_effects)
    resident_velocity_role = bool(
        direct_core_outputs and pipeline._formal_before_motion_cell_roles_active()
    )
    sparse_velocity_side_effects = bool(
        material_side_effect_copies_velocity and resident_velocity_role
    )
    if direct_core_outputs:
        if not material_side_effect_copies_velocity and not resident_velocity_role:
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
            if candidate_dispatch:
                pipeline._run_timed_candidate_material_side_effect_pass(
                    world, resources, inplace=True
                )
            else:
                copy_velocity_passthrough = bool(
                    material_side_effect_copies_velocity and not sparse_velocity_side_effects
                )
                use_packed_timed_emit_targets = bool(
                    program_name == "timed_apply"
                    and direct_core_outputs
                    and packed_local_deferred_outputs
                    and pipeline._packed_timed_emit_target_worklist_enabled
                )
            if not candidate_dispatch and use_packed_timed_emit_target_producer:
                pipeline._run_produced_packed_timed_material_side_effect_pass(
                    world,
                    resources,
                    copy_velocity_passthrough=copy_velocity_passthrough,
                    velocity_in_place=sparse_velocity_side_effects,
                    core_in_place=timed_sparse_inplace,
                )
            elif not candidate_dispatch and use_packed_timed_emit_targets:
                pipeline._run_packed_timed_material_side_effect_pass(
                    world,
                    resources,
                    copy_velocity_passthrough=copy_velocity_passthrough,
                    velocity_in_place=sparse_velocity_side_effects,
                )
            elif not candidate_dispatch and use_packed_self_emit_targets:
                if copy_velocity_passthrough:
                    with pipeline._profile_pass(world, "packed_self_material_velocity_copy"):
                        pipeline._copy_current_velocity_to_next_role(
                            world,
                            resources,
                            group_x,
                            group_y,
                        )
                pipeline._run_packed_material_target_apply_pass(
                    world,
                    resources,
                    velocity_in_place=sparse_velocity_side_effects,
                    deferred_hi_valid=write_deferred_hi_outputs,
                    profile_name="packed_self_material_targets_apply",
                )
            elif not candidate_dispatch:
                pipeline._run_cell_material_side_effect_pass(
                    world,
                    resources,
                    direct_core_outputs=direct_core_outputs,
                    copy_velocity_passthrough=copy_velocity_passthrough,
                    velocity_in_place=sparse_velocity_side_effects,
                    inplace=candidate_dispatch,
                    deferred_hi_valid=write_deferred_hi_outputs,
                    packed_local_deferred_outputs=packed_local_deferred_outputs,
                )
    if apply_gas_side_effects:
        with pipeline._profile_pass(world, f"{program_name}_gas_side_effects"):
            if candidate_dispatch:
                pipeline._run_timed_candidate_gas_side_effect_pass(
                    world,
                    resources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    may_have_flow_sources=may_have_flow_sources,
                )
            elif use_self_gas_fused:
                if may_have_flow_sources:
                    with pipeline._profile_pass(world, "self_fused_gas_flow_sources"):
                        pipeline._append_flow_sources_from_gpu(
                            world,
                            resources,
                            may_have_flow_sources=True,
                            flow_source_layers=flow_source_layers,
                        )
            elif use_self_gas_candidates:
                pipeline._run_self_candidate_gas_side_effect_pass(
                    world,
                    resources,
                    may_have_flow_sources=may_have_flow_sources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    flow_source_layers=flow_source_layers,
                )
            else:
                pipeline._run_cell_gas_side_effect_pass(
                    world,
                    resources,
                    may_have_flow_sources=may_have_flow_sources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    direct_core_outputs=direct_core_outputs,
                    action_gas_delta_already_applied=direct_action_gas_delta,
                    flow_source_layers=flow_source_layers,
                    deferred_hi_valid=write_deferred_hi_outputs,
                    packed_local_deferred_outputs=packed_local_deferred_outputs,
                    use_expanded_active_tile_mask=use_expanded_active_tile_mask,
                )
    return bool(
        material_side_effect_copies_velocity
        and resident_velocity_role
        and not sparse_velocity_side_effects
    )


def _run_timed_candidate_action_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    apply_material_side_effects: bool = False,
    apply_gas_side_effects: bool = False,
    modify_gas_layer_mask: int | None = None,
    may_have_flow_sources: bool = False,
    flow_source_layers: int = 16,
    inplace: bool = False,
) -> bool:
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._prepare_timed_candidate_worklist(world, resources)
    # The count is a tiny control read used only by the experimental sparse
    # path.  It lets us avoid gas/material side-effect dispatches when no
    # countdown is live (the common random-materials case).
    pipeline._timed_candidate_count_cpu = int(
        np.frombuffer(resources.timed_candidate_count.read(size=4), dtype=np.uint32, count=1)[0]
    )
    if (
        inplace
        and pipeline._timed_candidate_count_cpu > 0
        and not pipeline._timed_sparse_positive_count_enabled
    ):
        # Leave the canonical path untouched for frames with live timers; the
        # candidate side-effect pipelines have different output ownership.
        return False
    if not pipeline._formal_segment_batch_active():
        with pipeline._profile_pass(world, "timed_candidate_clear_local_meta"):
            pipeline._clear_timed_candidate_local_meta(world, resources)

    program = pipeline.programs["timed_apply_candidates"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", 0)
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "write_deferred_outputs", True)
    pipeline._set_uniform_if_present(program, "write_deferred_hi_outputs", True)
    cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = pipeline._current_cell_textures(resources)
    cell_state_in.use(location=0)
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
    if inplace:
        # Both formal roles are seeded from the same bridge snapshot.  The
        # sparse dispatch only writes positive-countdown cells, so keep the
        # current role in place and avoid a full-screen promote pass.
        cell_state_out, _phase_out, temp_out, integrity_out, _velocity_out, timer_out = (
            pipeline._current_cell_textures(resources)
        )
        cell_state_out.bind_to_image(0, read=False, write=True)
        temp_out.bind_to_image(2, read=False, write=True)
        integrity_out.bind_to_image(3, read=False, write=True)
        timer_out.bind_to_image(4, read=False, write=True)
        resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
        resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)
    else:
        pipeline._bind_local_cell_action_output_images(resources, direct_core_outputs=True)
    with pipeline._profile_pass(world, "timed_apply_candidates_shader"):
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal timed reaction candidate apply requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.timed_candidate_dispatch_args)
        pipeline._sync_compute_writes(ctx)
        pipeline._sync_storage_and_indirect_writes(ctx)

    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if not inplace:
        with pipeline._profile_pass(world, "timed_apply_candidates_velocity_copy"):
            pipeline._copy_current_velocity_to_next_role(world, resources, group_x, group_y)

    if apply_material_side_effects:
        with pipeline._profile_pass(world, "timed_apply_material_side_effects"):
            pipeline._run_cell_material_side_effect_pass(
                world,
                resources,
                direct_core_outputs=True,
                timed_candidate_outputs=True,
                timed_candidate_outputs_inplace=inplace,
            )
    if apply_gas_side_effects:
        with pipeline._profile_pass(world, "timed_apply_gas_side_effects"):
            pipeline._run_cell_gas_side_effect_pass(
                world,
                resources,
                modify_gas_layer_mask=modify_gas_layer_mask,
                may_have_flow_sources=may_have_flow_sources,
                flow_source_layers=flow_source_layers,
                direct_core_outputs=True,
                timed_candidate_outputs=True,
                timed_candidate_outputs_inplace=inplace,
            )
    return True



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
    if not pipeline._timed_sparse_positive_count_enabled:
        return
    compact_program = pipeline.programs["compact_timed_candidates"]
    compact_program["cell_grid_size"].value = (world.width, world.height)
    cell_state, _phase, _temp, _integrity, _velocity, timer = (
        pipeline._current_cell_textures(resources)
    )
    cell_state.use(location=0)
    timer.use(location=1)
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


def _prepare_self_candidate_worklist(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_program = pipeline.programs["clear_timed_candidate_worklist"]
    resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
    resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=2)
    clear_program.run(1, 1, 1)
    pipeline._sync_storage_and_indirect_writes(ctx)

    compact_program = pipeline.programs["compact_self_candidates"]
    compact_program["cell_grid_size"].value = (world.width, world.height)
    cell_state, _phase, _temp, _integrity, _velocity, _timer = pipeline._current_cell_textures(resources)
    cell_state.use(location=0)
    resources.active_cell_tex.use(location=2)
    resources.material_tags.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=2)
    resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=3)
    resources.timed_candidate_marks.bind_to_storage_buffer(binding=4)
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
    pipeline._ensure_formal_segment_meta_physical_zero(world, resources)
    program = pipeline.programs["accumulate_timed_candidate_segment_cell_transient_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.local_cell_meta_out.use(location=0)
    resources.timed_candidate_list.bind_to_storage_buffer(binding=0)
    resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
    resources.segment_cell_meta_tex.bind_to_image(0, read=True, write=True)
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
    packed_local_deferred_outputs: bool = False,
    accumulate_segment_cell_meta: bool = False,
) -> None:
    if direct_core_outputs:
        cell_state_out, _phase_out, temp_out, integrity_out, _velocity_out, timer_out = pipeline._next_cell_textures(resources)
    else:
        cell_state_out = resources.local_cell_state_out
        temp_out = resources.local_temp_out
        integrity_out = resources.local_integrity_out
        timer_out = resources.local_timer_out
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    timer_out.bind_to_image(4, read=False, write=True)
    if packed_local_deferred_outputs:
        resources.local_deferred_packed_out.bind_to_image(5, read=False, write=True)
    else:
        resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
        resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
    if not packed_local_deferred_outputs:
        if accumulate_segment_cell_meta:
            resources.segment_cell_meta_tex.bind_to_image(7, read=True, write=True)
        else:
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
    resources.local_cell_state_out.use(location=0)
    resources.local_temp_out.use(location=1)
    resources.local_integrity_out.use(location=2)
    resources.local_timer_out.use(location=3)
    resources.local_deferred_lo_out.use(location=4)
    resources.local_deferred_hi_out.use(location=5)
    resources.cell_state_pong.bind_to_image(0, read=False, write=True)
    resources.temp_pong.bind_to_image(1, read=False, write=True)
    resources.integrity_pong.bind_to_image(2, read=False, write=True)
    resources.timer_pong.bind_to_image(3, read=False, write=True)
    resources.trigger_lo_tex.bind_to_image(4, read=False, write=True)
    resources.trigger_hi_tex.bind_to_image(5, read=False, write=True)
    resources.deferred_scale_lo_tex.bind_to_image(6, read=False, write=True)
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
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if pipeline._formal_gpu_frame(world):
        current = pipeline._current_cell_textures(resources)
        next_outputs = pipeline._next_cell_textures(resources)
        copy_program = pipeline.programs["promote_reaction_cell_state"]
        copy_program["cell_grid_size"].value = (world.width, world.height)
        cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = current
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = next_outputs
        for location, texture in enumerate((cell_state_in, temp_in, integrity_in, velocity_in, timer_in)):
            texture.use(location=location)
        for binding, texture in enumerate((cell_state_out, temp_out, integrity_out, velocity_out, timer_out)):
            texture.bind_to_image(binding, read=False, write=True)
        copy_program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(world.bridge.ctx)
        cell_state_out, _phase_out, temp_out, integrity_out, velocity_out, timer_out = next_outputs
    else:
        cell_state_out = resources.cell_state_pong
        temp_out = resources.temp_pong
        integrity_out = resources.integrity_pong
        velocity_out = resources.velocity_pong
        timer_out = resources.timer_pong

    program = pipeline.programs["scatter_local_emit_cell_outputs"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    resources.local_emit_cell_lo_out.use(location=0)
    resources.local_emit_cell_hi_out.use(location=1)
    resources.local_timer_out.use(location=2)
    resources.local_cell_meta_out.use(location=3)
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(1, read=False, write=True)
    integrity_out.bind_to_image(2, read=False, write=True)
    velocity_out.bind_to_image(3, read=False, write=True)
    timer_out.bind_to_image(4, read=False, write=True)
    resources.emitted_material_mask_tex.bind_to_image(5, read=False, write=True)
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
