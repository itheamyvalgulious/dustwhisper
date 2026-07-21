from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.sim.gpu_reactions import (
    CONSUME_POLICY_BOTH,
    CONSUME_POLICY_NONE,
    CONSUME_POLICY_RHS,
    FORMAL_GPU_EMPTY_DEFERRED_BATCH,
    FLOW_SOURCE_GENERATION_BINDING,
    GPUDeferredActionBatch,
    GPUReactionMaterialPairPlan,
    LOCAL_SIZE,
    MATERIAL_LIGHT_PACKED_HEADER_OFFSET,
    MATERIAL_PAIR_PACKED_HEADER_OFFSET,
    MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
    MAX_MATERIALS,
    MAX_RULES,
    MAX_SELF_RULES,
    RULE_CANDIDATE_WORDS,
    TYPE_CONVERT_MATERIAL,
    TYPE_HARM,
    TYPE_MODIFY_GAS,
    TYPE_MODIFY_TEMPERATURE,
    TYPE_NONE,
    _SHADER_SUBS,
)
from oracle_game.sim.gpu_timer_pack import unpack_u8x4


def run_timed_actions(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    pipeline._timed_self_same_dispatch_pending = False
    pipeline.last_timed_self_same_dispatch_used = False
    pipeline.last_timed_sparse_inplace_used = False
    world.bridge.sync_rule_tables(world)
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    material_table = world.bridge.shadow_typed_tables["material_table"]
    with pipeline._profile_pass(world, "timed_compile_actions"):
        used_indices = pipeline._cached_used_action_indices_for_material_slots(world, material_table, slot_count=4)
        compiled = pipeline._compile_action_buffers_cached(world, action_table, used_indices)
    if compiled is None:
        return None
    combined_self_rule_count = 0
    combined_candidate = False
    if (
        pipeline._formal_gpu_frame(world)
        and pipeline._active_scheduler_gpu_authoritative(world)
        and getattr(pipeline, "_timed_self_same_dispatch_enabled", False)
    ):
        self_rule_table = world.bridge.shadow_typed_tables["self_rule_table"]
        combined_self_rule_count = min(MAX_SELF_RULES, int(self_rule_table.shape[0]))
        if combined_self_rule_count > 0:
            with pipeline._profile_pass(world, "timed_self_compile_actions"):
                self_used_indices = pipeline._cached_used_action_indices_for_self_rules(
                    world,
                    self_rule_table,
                    material_table,
                )
                combined_used_indices = set(used_indices or ()) | set(self_used_indices or ())
                combined_compiled = pipeline._compile_action_buffers_cached(
                    world,
                    action_table,
                    combined_used_indices,
                )
            combined_candidate = bool(
                combined_compiled is not None
                and not pipeline._compiled_actions_include_emit_material(compiled)
                and not pipeline._compiled_actions_include_emit_material(combined_compiled)
                and not pipeline._compiled_actions_include_flow_sources(compiled)
                and not pipeline._compiled_actions_include_flow_sources(combined_compiled)
                and not (
                    pipeline._compiled_actions_include_emit_light(compiled)
                    and pipeline._compiled_actions_include_emit_light(combined_compiled)
                )
            )
            if combined_candidate:
                assert combined_compiled is not None
                compiled = combined_compiled
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "timed_upload_state"):
        pipeline._upload_state(world, resources, reaction_group="timed", compiled_actions=compiled)
    use_expanded_active_tile_mask = bool(
        pipeline._can_use_expanded_active_tile_mask(world)
        and (
            combined_candidate
            or not getattr(pipeline, "_timed_sparse_inplace_enabled", False)
        )
    )
    upload_cell_mask, upload_gas_mask = pipeline._active_masks_for_cell_reaction_upload(
        world,
        solve_cell_mask,
        reaction_group="timed",
    )
    with pipeline._profile_pass(world, "timed_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            upload_cell_mask,
            upload_gas_mask,
            reaction_group="timed",
            load_gas_mask=False,
            use_expanded_tile_mask=use_expanded_active_tile_mask,
        )
    with pipeline._profile_pass(world, "timed_upload_metadata"):
        pipeline._upload_local_metadata(
            world,
            resources,
            include_self_rules=combined_candidate,
        )
        resources.action_i.write(compiled[0].tobytes())
        resources.action_f.write(compiled[1].tobytes())
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    emits_material = pipeline._compiled_actions_include_emit_material(compiled)
    modifies_gas = pipeline._compiled_actions_include_modify_gas(compiled)
    modify_gas_layer_mask = pipeline._compiled_modify_gas_layer_mask(
        compiled,
        world.gas_concentration.shape[0],
    )
    may_have_flow_sources = pipeline._compiled_actions_include_flow_sources(compiled)
    if combined_candidate:
        advance_velocity_role = pipeline._run_timed_self_combined_action_pass(
            world,
            resources,
            self_rule_count=combined_self_rule_count,
            modify_gas_layer_mask=modify_gas_layer_mask,
            modifies_gas=modifies_gas,
            use_expanded_active_tile_mask=use_expanded_active_tile_mask,
        )
        with pipeline._profile_pass(world, "timed_self_publish_cell_state"):
            pipeline._download_cell_state(
                world,
                resources,
                direct_core_outputs=True,
                advance_velocity_role=advance_velocity_role,
                packed_local_cell_meta=True,
            )
        pipeline._timed_self_same_dispatch_pending = True
        pipeline.last_timed_self_same_dispatch_used = True
        with pipeline._profile_pass(world, "timed_self_publish_deferred"):
            return pipeline._download_deferred_batch(world, resources)
    sparse_inplace = bool(
        formal_gpu_frame
        and getattr(pipeline, "_timed_sparse_inplace_enabled", False)
    )
    if sparse_inplace and pipeline._timed_sparse_positive_count_enabled:
        pipeline._prepare_timed_candidate_worklist(world, resources)
        pipeline._timed_candidate_count_cpu = 1
        pipeline._run_local_cell_action_pass(
            world,
            resources,
            "timed_apply",
            apply_material_side_effects=emits_material,
            apply_gas_side_effects=modifies_gas,
            modify_gas_layer_mask=modify_gas_layer_mask,
            may_have_flow_sources=may_have_flow_sources,
            flow_source_layers=16,
            timed_sparse_inplace=True,
        )
        if pipeline._formal_segment_batch_active():
            pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
        pipeline.last_timed_sparse_inplace_used = True
        with pipeline._profile_pass(world, "timed_publish_deferred"):
            return pipeline._download_deferred_batch(world, resources)
    if sparse_inplace:
        sparse_completed = pipeline._run_timed_candidate_action_pass(
            world,
            resources,
            apply_material_side_effects=emits_material,
            apply_gas_side_effects=modifies_gas,
            modify_gas_layer_mask=modify_gas_layer_mask,
            may_have_flow_sources=may_have_flow_sources,
            flow_source_layers=16,
            inplace=True,
        )
        if sparse_completed and pipeline._formal_segment_batch_active():
            pipeline._accumulate_timed_candidate_segment_cell_transient_state(world, resources)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
        if sparse_completed:
            pipeline.last_timed_sparse_inplace_used = True
            with pipeline._profile_pass(world, "timed_publish_deferred"):
                return pipeline._download_deferred_batch(world, resources)
    advance_velocity_role = pipeline._run_local_cell_action_pass(
        world,
        resources,
        "timed_apply",
        apply_material_side_effects=emits_material,
        apply_gas_side_effects=modifies_gas,
        modify_gas_layer_mask=modify_gas_layer_mask,
        may_have_flow_sources=may_have_flow_sources,
        flow_source_layers=16,
        use_expanded_active_tile_mask=use_expanded_active_tile_mask,
    )
    with pipeline._profile_pass(world, "timed_publish_cell_state"):
        pipeline._download_cell_state(
            world,
            resources,
            direct_core_outputs=formal_gpu_frame,
            advance_velocity_role=advance_velocity_role,
            packed_local_cell_meta=formal_gpu_frame,
            segment_cell_meta_already_accumulated=bool(
                pipeline.last_timed_self_cell_flag_meta_used
            ),
        )
    with pipeline._profile_pass(world, "timed_publish_deferred"):
        return pipeline._download_deferred_batch(world, resources)



def run_timed_triggers(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if not pipeline.available(world):
        return None
    if pipeline._formal_gpu_frame(world):
        raise RuntimeError("GPU reaction timed trigger readback is not allowed in formal GPU frames; CPU fallback is disabled")
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    pipeline._upload_state(world, resources)
    pipeline._upload_active_masks(
        world,
        resources,
        solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
        np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
    )
    pipeline._upload_local_metadata(world, resources)
    program = pipeline.programs["timed_trigger"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    resources.cell_state_ping.use(location=0)
    resources.timer_ping.use(location=1)
    resources.active_cell_tex.use(location=2)
    resources.material_slots_lo.bind_to_storage_buffer(binding=0)
    resources.trigger_lo_tex.bind_to_image(3, read=False, write=True)
    resources.timer_pong.bind_to_image(4, read=False, write=True)
    program.run((world.width + LOCAL_SIZE - 1) // LOCAL_SIZE, (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE, 1)
    world.bridge.ctx.finish()
    world.timer_pack[:] = unpack_u8x4(
        np.frombuffer(resources.timer_pong.read(), dtype="u4").reshape((world.height, world.width))
    )
    return np.rint(
        np.frombuffer(resources.trigger_lo_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.int32)



def run_self_triggers(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not pipeline.available(world):
        return None
    if pipeline._formal_gpu_frame(world):
        raise RuntimeError("GPU reaction self trigger readback is not allowed in formal GPU frames; CPU fallback is disabled")
    world.bridge.sync_rule_tables(world)
    self_rule_count = min(MAX_SELF_RULES, int(world.bridge.shadow_typed_tables["self_rule_table"].shape[0]))
    if self_rule_count <= 0:
        return None
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    pipeline._upload_state(world, resources)
    pipeline._upload_active_masks(
        world,
        resources,
        solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
        np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
    )
    pipeline._upload_local_metadata(world, resources, include_self_rules=True)
    program = pipeline.programs["self_trigger"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "self_rule_count", self_rule_count)
    resources.cell_state_ping.use(location=0)
    resources.temp_ping.use(location=2)
    resources.integrity_ping.use(location=3)
    resources.timer_ping.use(location=4)
    resources.active_cell_tex.use(location=5)
    resources.material_slots_lo.bind_to_storage_buffer(binding=0)
    resources.material_slots_hi.bind_to_storage_buffer(binding=1)
    resources.action_meta.bind_to_storage_buffer(binding=2)
    resources.self_rule_i.bind_to_storage_buffer(binding=3)
    resources.self_rule_f.bind_to_storage_buffer(binding=4)
    resources.timer_pong.bind_to_image(0, read=False, write=True)
    resources.trigger_lo_tex.bind_to_image(1, read=False, write=True)
    resources.trigger_hi_tex.bind_to_image(2, read=False, write=True)
    program.run((world.width + LOCAL_SIZE - 1) // LOCAL_SIZE, (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE, 1)
    world.bridge.ctx.finish()
    world.timer_pack[:] = unpack_u8x4(
        np.frombuffer(resources.timer_pong.read(), dtype="u4").reshape((world.height, world.width))
    )
    trigger_lo = np.rint(
        np.frombuffer(resources.trigger_lo_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.int32)
    trigger_hi = np.rint(
        np.frombuffer(resources.trigger_hi_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.int32)
    return (trigger_lo, trigger_hi)



def run_self_actions(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    if pipeline._timed_self_same_dispatch_pending:
        pipeline._timed_self_same_dispatch_pending = False
        return FORMAL_GPU_EMPTY_DEFERRED_BATCH
    pipeline.last_self_sparse_inplace_used = False
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["self_rule_table"]
    self_rule_count = min(MAX_SELF_RULES, int(rule_table.shape[0]))
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    material_table = world.bridge.shadow_typed_tables["material_table"]
    with pipeline._profile_pass(world, "self_compile_actions"):
        used_indices = pipeline._cached_used_action_indices_for_self_rules(world, rule_table, material_table)
        compiled = pipeline._compile_action_buffers_cached(world, action_table, used_indices)
    if compiled is None:
        return None
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    flow_source_layers = pipeline._compiled_self_rule_flow_source_layers(
        rule_table,
        material_table,
        compiled,
    )
    with pipeline._profile_pass(world, "self_upload_state"):
        pipeline._upload_state(
            world,
            resources,
            reaction_group="self",
            compiled_actions=compiled,
            flow_source_layers=flow_source_layers,
        )
    sparse_inplace = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._formal_segment_batch_active()
        and getattr(pipeline, "_self_sparse_inplace_enabled", False)
        and pipeline._active_scheduler_gpu_authoritative(world)
    )
    use_expanded_active_tile_mask = bool(
        pipeline._can_use_expanded_active_tile_mask(world)
        and not sparse_inplace
    )
    upload_cell_mask, upload_gas_mask = pipeline._active_masks_for_cell_reaction_upload(
        world,
        solve_cell_mask,
        reaction_group="self",
    )
    with pipeline._profile_pass(world, "self_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            upload_cell_mask,
            upload_gas_mask,
            reaction_group="self",
            use_expanded_tile_mask=use_expanded_active_tile_mask,
        )
    with pipeline._profile_pass(world, "self_upload_metadata"):
        pipeline._upload_local_metadata(world, resources, include_self_rules=True)
        resources.action_i.write(compiled[0].tobytes())
        resources.action_f.write(compiled[1].tobytes())
    emits_material = pipeline._compiled_actions_include_emit_material(compiled)
    write_deferred_hi_outputs = bool(
        not pipeline._formal_gpu_frame(world)
        or pipeline._self_rules_require_deferred_hi_outputs(rule_table, material_table, compiled)
    )
    if sparse_inplace:
        pipeline._prepare_self_candidate_worklist(world, resources)
        pipeline._run_local_cell_action_pass(
            world,
            resources,
            "self_apply",
            self_rule_count=self_rule_count,
            apply_material_side_effects=emits_material,
            apply_gas_side_effects=pipeline._compiled_actions_include_modify_gas(compiled),
            modify_gas_layer_mask=pipeline._compiled_modify_gas_layer_mask(
                compiled, world.gas_concentration.shape[0]
            ),
            may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(compiled),
            flow_source_layers=flow_source_layers,
            write_deferred_hi_outputs=write_deferred_hi_outputs,
            fused_gas_output_safe=False,
            inplace=True,
            candidate_dispatch=True,
        )
        if not pipeline.last_timed_self_cell_flag_meta_used:
            pipeline._accumulate_timed_candidate_segment_cell_transient_state(world, resources)
        pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
        pipeline.last_self_sparse_inplace_used = True
        with pipeline._profile_pass(world, "self_publish_deferred"):
            return pipeline._download_deferred_batch(world, resources)
    advance_velocity_role = pipeline._run_local_cell_action_pass(
        world,
        resources,
        "self_apply",
        self_rule_count=self_rule_count,
        apply_material_side_effects=emits_material,
        apply_gas_side_effects=pipeline._compiled_actions_include_modify_gas(compiled),
        modify_gas_layer_mask=pipeline._compiled_modify_gas_layer_mask(compiled, world.gas_concentration.shape[0]),
        may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(compiled),
        flow_source_layers=flow_source_layers,
        write_deferred_hi_outputs=write_deferred_hi_outputs,
        fused_gas_output_safe=not pipeline._compiled_actions_include_emit_light(compiled),
        use_expanded_active_tile_mask=use_expanded_active_tile_mask,
    )
    with pipeline._profile_pass(world, "self_publish_cell_state"):
        pipeline._download_cell_state(
            world,
            resources,
            direct_core_outputs=pipeline._formal_gpu_frame(world),
            advance_velocity_role=advance_velocity_role,
            packed_local_cell_meta=pipeline._formal_gpu_frame(world),
            segment_cell_meta_already_accumulated=bool(
                pipeline.last_timed_self_cell_flag_meta_used
            ),
        )
    with pipeline._profile_pass(world, "self_publish_deferred"):
        return pipeline._download_deferred_batch(world, resources)



def run_material_material(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["material_material_rule_table"]
    rule_count = int(rule_table.shape[0])
    if rule_count <= 0 or rule_count > MAX_RULES:
        return None
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    material_table = world.bridge.shadow_typed_tables["material_table"]
    used_indices = pipeline._cached_used_action_indices_for_pair_rules(
        world,
        rule_table,
        material_table,
        rule_kind="material_material",
        lhs_tag_field="material_tag_mask",
    )
    compiled = pipeline._compile_action_buffers_cached(world, action_table, used_indices)
    if compiled is None:
        return None
    rule_i, rule_f, rule_tags = pipeline._compile_material_material_rules(rule_table)
    lhs_candidate_masks = pipeline._compile_material_rule_candidate_masks(
        rule_table,
        material_table,
        selector_id_field="lhs_material_id",
        selector_tag_field="lhs_tag_mask",
        material_tag_field="material_tag_mask",
    )
    return pipeline._run_cell_pass(
        world,
        "material_material",
        compiled,
        rule_i,
        rule_f,
        rule_tags,
        rule_count,
        solve_cell_mask,
        lhs_rule_candidate_masks=lhs_candidate_masks,
    )



def run_material_gas(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["material_gas_rule_table"]
    rule_count = int(rule_table.shape[0])
    if rule_count <= 0 or rule_count > MAX_RULES:
        return None
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    material_table = world.bridge.shadow_typed_tables["material_table"]
    used_indices = pipeline._cached_used_action_indices_for_pair_rules(
        world,
        rule_table,
        material_table,
        rule_kind="material_gas",
        lhs_tag_field="gas_tag_mask",
    )
    compiled = pipeline._compile_action_buffers_cached(world, action_table, used_indices)
    if compiled is None:
        return None
    rule_i, rule_f, rule_tags = pipeline._compile_material_gas_rules(rule_table)
    lhs_candidate_masks = pipeline._compile_material_rule_candidate_masks(
        rule_table,
        material_table,
        selector_id_field="lhs_material_id",
        selector_tag_field="lhs_tag_mask",
        material_tag_field="gas_tag_mask",
    )
    return pipeline._run_cell_pass(
        world,
        "material_gas",
        compiled,
        rule_i,
        rule_f,
        rule_tags,
        rule_count,
        solve_cell_mask,
        lhs_rule_candidate_masks=lhs_candidate_masks,
    )



def _merge_material_pair_candidate_masks(
    material_material_masks: np.ndarray,
    material_gas_masks: np.ndarray,
    material_gas_rule_offset: int,
    material_gas_rule_count: int,
) -> np.ndarray:
    merged = np.zeros_like(material_material_masks)
    merged_words = merged.reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    mm_words = material_material_masks.reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    mg_words = material_gas_masks.reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    merged_words[:] = mm_words
    for rule_index in range(material_gas_rule_count):
        selected = (mg_words[:, rule_index // 32] & np.uint32(1 << (rule_index % 32))) != 0
        shifted_rule_index = material_gas_rule_offset + rule_index
        merged_words[selected, shifted_rule_index // 32] |= np.uint32(1 << (shifted_rule_index % 32))
    return merged


def _material_pair_plan_cache_key(
    pipeline,
    world: "WorldEngine",
    *,
    include_material_light: bool,
) -> tuple[object, ...]:
    world.bridge.sync_rule_tables(world)
    tables = world.bridge.shadow_typed_tables
    generations = world.bridge.table_generations
    return (
        "material_pair_plan_v1",
        id(world.bridge),
        int(generations.get("reactions", 0)),
        int(generations.get("materials", 0)),
        int(generations.get("gases", 0)),
        int(generations.get("lights", 0)),
        tuple(int(value) for value in tables["reaction_action_table"].shape),
        tuple(int(value) for value in tables["material_table"].shape),
        tuple(int(value) for value in tables["gas_table"].shape),
        tuple(int(value) for value in tables["light_table"].shape),
        tuple(int(value) for value in tables["material_material_rule_table"].shape),
        tuple(int(value) for value in tables["material_gas_rule_table"].shape),
        tuple(int(value) for value in tables["material_light_rule_table"].shape),
        int(world.gas_concentration.shape[0]),
        int(world.cell_optical_dose.shape[0]),
        bool(include_material_light),
        bool(pipeline._authoritative_lhs_candidate_masks_enabled),
        bool(pipeline._material_pair_packed_descriptors_enabled),
        bool(pipeline._material_triplet_ml_packed_descriptors_enabled),
    )


def _compile_material_pair_plan(
    pipeline,
    world: "WorldEngine",
    *,
    include_material_light: bool,
    cache_key: tuple[object, ...] | None = None,
) -> GPUReactionMaterialPairPlan | None:
    world.bridge.sync_rule_tables(world)
    tables = world.bridge.shadow_typed_tables
    mm_table = tables["material_material_rule_table"]
    mg_table = tables["material_gas_rule_table"]
    ml_table = tables["material_light_rule_table"]
    action_table = tables["reaction_action_table"]
    material_table = tables["material_table"]
    light_table = tables["light_table"]
    mm_count = int(mm_table.shape[0])
    mg_count = int(mg_table.shape[0])
    if (
        not pipeline._authoritative_lhs_candidate_masks_enabled
        or mm_count <= 0
        or mg_count <= 0
        or mm_count + mg_count > MAX_RULES
    ):
        return None

    mm_used = pipeline._cached_used_action_indices_for_pair_rules(
        world,
        mm_table,
        material_table,
        rule_kind="material_material",
        lhs_tag_field="material_tag_mask",
    )
    mg_used = pipeline._cached_used_action_indices_for_pair_rules(
        world,
        mg_table,
        material_table,
        rule_kind="material_gas",
        lhs_tag_field="gas_tag_mask",
    )
    if mm_used is None or mg_used is None:
        return None
    pair_used = set(mm_used) | set(mg_used)
    compiled = pipeline._compile_action_buffers(action_table, pair_used)
    if compiled is None:
        return None
    action_i = np.asarray(compiled[0], dtype=np.int32)
    allowed_types = np.asarray(
        (TYPE_NONE, TYPE_HARM, TYPE_MODIFY_TEMPERATURE, TYPE_CONVERT_MATERIAL, TYPE_MODIFY_GAS),
        dtype=np.int32,
    )
    if not bool(np.all(np.isin(action_i[:, 0], allowed_types))):
        return None
    if bool(np.any((action_i[:, 0] == TYPE_MODIFY_GAS) & (action_i[:, 3] != 0))):
        return None

    mm_i, mm_f, mm_tags = pipeline._compile_material_material_rules(mm_table)
    mg_i, mg_f, mg_tags = pipeline._compile_material_gas_rules(mg_table)
    if pipeline._compiled_rules_include_rhs_consume(mm_tags) or pipeline._compiled_rules_include_rhs_consume(mg_tags):
        return None
    material_pair_packed_descriptors = None
    if pipeline._material_pair_packed_descriptors_enabled:
        material_pair_packed_descriptors = pipeline._compile_material_pair_packed_descriptors_cached(
            world,
            mm_table,
            mg_table,
            material_table,
            int(world.gas_concentration.shape[0]),
        )

    material_light_rule_count = 0
    material_light_rule_i = None
    material_light_rule_f = None
    material_light_rule_tags = None
    material_light_lhs_candidate_masks = None
    material_light_packed_descriptors = None
    if include_material_light:
        ml_count = int(ml_table.shape[0])
        if 0 < ml_count <= MAX_RULES:
            ml_used = pipeline._cached_used_action_indices_for_pair_rules(
                world,
                ml_table,
                material_table,
                rule_kind="material_light",
                lhs_tag_field="light_tag_mask",
            )
            combined = (
                pipeline._compile_action_buffers(action_table, pair_used | set(ml_used))
                if ml_used is not None
                else None
            )
            if combined is not None and ml_used is not None:
                combined_action_i = np.asarray(combined[0], dtype=np.int32)
                ml_action_indices = np.asarray(sorted(ml_used), dtype=np.int32)
                ml_action_types = combined_action_i[ml_action_indices, 0]
                allowed_ml_types = np.asarray(
                    (TYPE_NONE, TYPE_HARM, TYPE_MODIFY_TEMPERATURE, TYPE_CONVERT_MATERIAL),
                    dtype=np.int32,
                )
                material_light_rule_i, material_light_rule_f, material_light_rule_tags = (
                    pipeline._compile_material_light_rules(ml_table, light_table)
                )
                can_fuse_material_light = bool(
                    np.all(np.isin(ml_action_types, allowed_ml_types))
                    and not pipeline._compiled_rules_include_rhs_consume(material_light_rule_tags)
                    and not pipeline._compiled_actions_require_deferred_outputs(
                        combined,
                        direct_modify_gas=True,
                    )
                )
                if can_fuse_material_light:
                    material_light_lhs_candidate_masks = pipeline._compile_material_rule_candidate_masks(
                        ml_table,
                        material_table,
                        selector_id_field="lhs_material_id",
                        selector_tag_field="lhs_tag_mask",
                        material_tag_field="light_tag_mask",
                    )
                    if pipeline._material_triplet_ml_packed_descriptors_enabled:
                        material_light_packed_descriptors = (
                            pipeline._compile_material_light_packed_descriptors_cached(
                                world,
                                ml_table,
                                material_table,
                                light_table,
                            )
                        )
                    compiled = combined
                    material_light_rule_count = ml_count

    rule_i = np.zeros_like(mm_i)
    rule_i[:, 3] = -1
    rule_f = np.zeros_like(mm_f)
    rule_tags = np.zeros_like(mm_tags)
    rule_i[:mm_count] = mm_i[:mm_count]
    rule_f[:mm_count] = mm_f[:mm_count]
    rule_tags[:mm_count] = mm_tags[:mm_count]
    rule_i[mm_count : mm_count + mg_count] = mg_i[:mg_count]
    rule_f[mm_count : mm_count + mg_count] = mg_f[:mg_count]
    rule_tags[mm_count : mm_count + mg_count] = mg_tags[:mg_count]

    mm_masks = pipeline._compile_material_rule_candidate_masks(
        mm_table,
        material_table,
        selector_id_field="lhs_material_id",
        selector_tag_field="lhs_tag_mask",
        material_tag_field="material_tag_mask",
    )
    mg_masks = pipeline._compile_material_rule_candidate_masks(
        mg_table,
        material_table,
        selector_id_field="lhs_material_id",
        selector_tag_field="lhs_tag_mask",
        material_tag_field="gas_tag_mask",
    )
    merged_masks = _merge_material_pair_candidate_masks(mm_masks, mg_masks, mm_count, mg_count)

    packed_rule_i = np.zeros((MATERIAL_PAIR_RULE_I_ENTRY_COUNT, 4), dtype=np.int32)
    packed_rule_f = np.zeros((MAX_RULES * 2 + 1, 4), dtype=np.float32)
    packed_rule_tags = np.zeros((MAX_RULES * 2 + 1, 4), dtype=np.uint32)
    packed_candidate_masks = np.zeros(
        (MAX_MATERIALS * 2, merged_masks.shape[1], 4),
        dtype=np.uint32,
    )
    packed_rule_i[:MAX_RULES] = rule_i
    packed_rule_f[:MAX_RULES] = rule_f
    packed_rule_tags[:MAX_RULES] = rule_tags
    packed_candidate_masks[:MAX_MATERIALS] = merged_masks
    if material_light_rule_count > 0:
        assert material_light_rule_i is not None
        assert material_light_rule_f is not None
        assert material_light_rule_tags is not None
        assert material_light_lhs_candidate_masks is not None
        packed_rule_i[MAX_RULES : MAX_RULES * 2] = material_light_rule_i
        packed_rule_f[MAX_RULES : MAX_RULES * 2] = material_light_rule_f
        packed_rule_tags[MAX_RULES : MAX_RULES * 2] = material_light_rule_tags
        packed_candidate_masks[MAX_MATERIALS:] = material_light_lhs_candidate_masks
        if material_light_packed_descriptors is not None:
            descriptor_count = int(material_light_packed_descriptors.shape[0])
            packed_rule_i.view(np.uint32)[
                MATERIAL_LIGHT_PACKED_HEADER_OFFSET :
                MATERIAL_LIGHT_PACKED_HEADER_OFFSET + descriptor_count
            ] = material_light_packed_descriptors
    if material_pair_packed_descriptors is not None:
        descriptor_count = int(material_pair_packed_descriptors.shape[0])
        packed_rule_i.view(np.uint32)[
            MATERIAL_PAIR_PACKED_HEADER_OFFSET :
            MATERIAL_PAIR_PACKED_HEADER_OFFSET + descriptor_count
        ] = material_pair_packed_descriptors

    compiled_actions = (
        np.asarray(compiled[0], dtype=np.int32),
        np.asarray(compiled[1], dtype=np.float32),
    )
    for array in (*compiled_actions, packed_rule_i, packed_rule_f, packed_rule_tags, packed_candidate_masks):
        array.flags.writeable = False
    if cache_key is None:
        cache_key = pipeline._material_pair_plan_cache_key(
            world,
            include_material_light=include_material_light,
        )
    return GPUReactionMaterialPairPlan(
        cache_key=cache_key,
        compiled_actions=compiled_actions,
        packed_rule_i=packed_rule_i,
        packed_rule_f=packed_rule_f,
        packed_rule_tags=packed_rule_tags,
        packed_lhs_candidate_masks=packed_candidate_masks,
        material_material_rule_count=mm_count,
        rule_count=mm_count + mg_count,
        material_light_rule_count=material_light_rule_count,
        material_light_packed_descriptors=material_light_packed_descriptors,
        material_pair_packed_descriptors=material_pair_packed_descriptors,
        modifies_gas=pipeline._compiled_actions_include_modify_gas(compiled_actions),
        direct_modify_gas_layer_mask=pipeline._compiled_modify_gas_layer_mask(
            compiled_actions,
            world.gas_concentration.shape[0],
        ),
    )


def _compile_material_pair_plan_cached(
    pipeline,
    world: "WorldEngine",
    *,
    include_material_light: bool,
) -> GPUReactionMaterialPairPlan | None:
    key = pipeline._material_pair_plan_cache_key(
        world,
        include_material_light=include_material_light,
    )
    if key not in pipeline._material_pair_plan_cache:
        if len(pipeline._material_pair_plan_cache) > 16:
            pipeline._material_pair_plan_cache.clear()
        pipeline._material_pair_plan_cache[key] = pipeline._compile_material_pair_plan(
            world,
            include_material_light=include_material_light,
            cache_key=key,
        )
    return pipeline._material_pair_plan_cache[key]


def run_material_pair_fused(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object,
    fuse_material_light: bool = False,
) -> GPUDeferredActionBatch | None:
    pipeline.last_material_pair_fused_light = False
    if (
        not pipeline._material_pair_state_fusion_enabled
        or not pipeline.available(world)
        or not pipeline._formal_gpu_frame(world)
        or not pipeline._active_scheduler_gpu_authoritative(world)
        or pipeline._formal_segment_batch_base_key is None
        or not pipeline._authoritative_lhs_candidate_masks_enabled
    ):
        return None
    world.bridge.sync_rule_tables(world)
    material_light_dose_guard = None
    include_material_light = False
    if fuse_material_light:
        bridge = world.bridge
        ml_count = int(bridge.shadow_typed_tables["material_light_rule_table"].shape[0])
        material_light_dose_guard = pipeline._formal_light_dose_guard_buffer(world)
        include_material_light = bool(
            0 < ml_count <= MAX_RULES
            and material_light_dose_guard is not None
            and "cell_optical_dose" in bridge.gpu_authoritative_resources
            and bridge.buffers.get("cell_optical_dose") is not None
        )
    plan = pipeline._compile_material_pair_plan_cached(
        world,
        include_material_light=include_material_light,
    )
    if plan is None:
        return None
    if plan.material_light_rule_count <= 0:
        material_light_dose_guard = None
    deferred = pipeline._run_material_pair_fused_pass(
        world,
        plan,
        solve_cell_mask,
        material_light_dose_guard=material_light_dose_guard,
    )
    pipeline.last_material_pair_fused_light = plan.material_light_rule_count > 0
    return deferred



def run_material_light(
    pipeline,
    world: "WorldEngine",
    *,
    solve_cell_mask: object | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["material_light_rule_table"]
    rule_count = int(rule_table.shape[0])
    if rule_count <= 0 or rule_count > MAX_RULES:
        return None
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    material_table = world.bridge.shadow_typed_tables["material_table"]
    used_indices = pipeline._cached_used_action_indices_for_pair_rules(
        world,
        rule_table,
        material_table,
        rule_kind="material_light",
        lhs_tag_field="light_tag_mask",
    )
    compiled = pipeline._compile_action_buffers_cached(world, action_table, used_indices)
    if compiled is None:
        return None
    light_table = world.bridge.shadow_typed_tables["light_table"]
    rule_i, rule_f, rule_tags = pipeline._compile_material_light_rules(rule_table, light_table)
    lhs_candidate_masks = pipeline._compile_material_rule_candidate_masks(
        rule_table,
        material_table,
        selector_id_field="lhs_material_id",
        selector_tag_field="lhs_tag_mask",
        material_tag_field="light_tag_mask",
    )
    light_dose_guard = pipeline._formal_light_dose_guard_buffer(world)
    if light_dose_guard is not None and pipeline._formal_segment_batch_base_key is None:
        return pipeline._run_formal_guarded_material_light(
            world,
            compiled,
            rule_i,
            rule_f,
            rule_tags,
            rule_count,
            solve_cell_mask,
            light_dose_guard,
            lhs_candidate_masks,
        )
    return pipeline._run_cell_pass(
        world,
        "material_light",
        compiled,
        rule_i,
        rule_f,
        rule_tags,
        rule_count,
        solve_cell_mask,
        lhs_rule_candidate_masks=lhs_candidate_masks,
        light_dose_guard_buffer=light_dose_guard,
    )



def _run_formal_guarded_material_light(
    pipeline,
    world: "WorldEngine",
    compiled_actions: tuple[np.ndarray, np.ndarray],
    rule_i: np.ndarray,
    rule_f: np.ndarray,
    rule_tags: np.ndarray,
    rule_count: int,
    solve_cell_mask: object | None,
    light_dose_guard: Any,
    lhs_rule_candidate_masks: np.ndarray,
) -> GPUDeferredActionBatch:
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    has_rhs_consume = pipeline._compiled_rules_include_rhs_consume(rule_tags)
    direct_bridge_cell_dose = not has_rhs_consume
    with pipeline._profile_pass(world, "material_light_upload_state"):
        pipeline._upload_state(
            world,
            resources,
            reaction_group="material_light",
            compiled_actions=compiled_actions,
            light_dose_guard_buffer=light_dose_guard,
            direct_bridge_cell_dose=direct_bridge_cell_dose,
        )
    active_authoritative = pipeline._active_scheduler_gpu_authoritative(world)
    with pipeline._profile_pass(world, "material_light_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            None
            if active_authoritative
            else solve_cell_mask
            if solve_cell_mask is not None
            else np.ones((world.height, world.width), dtype=np.bool_),
            None if active_authoritative else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
            reaction_group="material_light",
            light_dose_guard_buffer=light_dose_guard,
        )
    with pipeline._profile_pass(world, "material_light_upload_metadata"):
        pipeline._upload_local_metadata(world, resources)
        resources.action_i.write(compiled_actions[0].tobytes())
        resources.action_f.write(compiled_actions[1].tobytes())
        resources.ml_rule_i.write(rule_i.tobytes())
        resources.ml_rule_f.write(rule_f.tobytes())
        resources.ml_rule_tags.write(rule_tags.tobytes())
        resources.rule_lhs_candidate_masks.write(lhs_rule_candidate_masks.tobytes())

    program_key = (
        "material_light_authoritative_lhs"
        if pipeline._authoritative_lhs_candidate_masks_enabled
        else "material_light"
    )
    program = pipeline.programs[program_key]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", rule_count)
    pipeline._set_uniform_if_present(program, "rule_candidate_word_count", pipeline._rule_candidate_word_count(rule_count))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", False)
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", 0)
    pipeline._set_uniform_if_present(program, "use_bridge_cell_dose", direct_bridge_cell_dose)
    pipeline._set_uniform_if_present(
        program,
        "write_deferred_outputs",
        pipeline._compiled_actions_require_deferred_outputs(compiled_actions),
    )
    pipeline._set_uniform_if_present(program, "accumulate_segment_cell_meta", False)
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
    resources.ml_rule_i.bind_to_storage_buffer(binding=3)
    resources.ml_rule_f.bind_to_storage_buffer(binding=4)
    resources.ml_rule_tags.bind_to_storage_buffer(binding=5)
    resources.material_tags.bind_to_storage_buffer(binding=6)
    resources.gas_tags.bind_to_storage_buffer(binding=7)
    resources.material_slots_lo.bind_to_storage_buffer(binding=8)
    resources.material_slots_hi.bind_to_storage_buffer(binding=9)
    resources.action_meta.bind_to_storage_buffer(binding=10)
    resources.random_targets.bind_to_storage_buffer(binding=11)
    resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
    if direct_bridge_cell_dose:
        world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=16)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
    pipeline._bind_local_cell_action_output_images(resources, direct_core_outputs=True)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "material_light_shader"):
        dispatch_args = pipeline._build_light_dose_guarded_dispatch_args(
            world,
            resources,
            light_dose_guard,
            group_x,
            group_y,
            1,
        )
        resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal light-dose guarded reactions require ModernGL ComputeShader.run_indirect")
        program.run_indirect(dispatch_args)
        pipeline._sync_compute_writes(world.bridge.ctx)
    with pipeline._profile_pass(world, "material_light_velocity_copy"):
        pipeline._copy_current_velocity_to_next_role(
            world,
            resources,
            group_x,
            group_y,
            light_dose_guard_buffer=light_dose_guard,
        )

    if pipeline._compiled_actions_include_emit_material(compiled_actions):
        with pipeline._profile_pass(world, "material_light_material_side_effects"):
            pipeline._run_cell_material_side_effect_pass(
                world,
                resources,
                direct_core_outputs=True,
                light_dose_guard_buffer=light_dose_guard,
            )
    if pipeline._compiled_actions_include_modify_gas(compiled_actions):
        may_have_flow_sources = pipeline._compiled_actions_include_flow_sources(compiled_actions)
        with pipeline._profile_pass(world, "material_light_gas_side_effects"):
            pipeline._run_cell_gas_side_effect_pass(
                world,
                resources,
                apply_action_side_effects=True,
                may_have_flow_sources=may_have_flow_sources,
                modify_gas_layer_mask=pipeline._compiled_modify_gas_layer_mask(
                    compiled_actions,
                    world.gas_concentration.shape[0],
                ),
                direct_core_outputs=True,
                light_dose_guard_buffer=light_dose_guard,
            )
    if has_rhs_consume:
        with pipeline._profile_pass(world, "material_light_dose_consume"):
            pipeline._run_material_light_dose_consume_pass(
                world,
                resources,
                rule_count,
                light_dose_guard_buffer=light_dose_guard,
            )
    with pipeline._profile_pass(world, "material_light_publish_cell_state"):
        pipeline._publish_bridge_cell_state(
            world,
            resources,
            cell_meta_texture=resources.local_cell_meta_out,
            light_dose_guard_buffer=light_dose_guard,
            mark_structure_dirty=pipeline._compiled_actions_may_change_structure(compiled_actions),
        )
    with pipeline._profile_pass(world, "material_light_publish_deferred"):
        return pipeline._download_deferred_batch(world, resources)



def run_gas_gas(
    pipeline,
    world: "WorldEngine",
    *,
    solve_gas_mask: np.ndarray | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["gas_gas_rule_table"]
    rule_count = int(rule_table.shape[0])
    if rule_count <= 0 or rule_count > MAX_RULES:
        return None
    used_indices = pipeline._used_action_indices(rule_table)
    if used_indices is None:
        return None
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    compiled = pipeline._compile_gas_action_buffers(action_table, used_indices)
    if compiled is None:
        return None
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    pipeline._upload_state(world, resources, reaction_group="gas_gas", compiled_actions=compiled)
    pipeline._upload_local_metadata(world, resources)
    pipeline._upload_active_masks(
        world,
        resources,
        np.ones((world.height, world.width), dtype=np.bool_),
        solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
        reaction_group="gas_gas",
        load_cell_mask=False,
    )
    program = pipeline.programs["gas_gas"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    resources.active_gas_tex.use(location=2)
    cell_state_in, _phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    cell_state_in.use(location=3)
    temp_in.use(location=4)
    resources.flow_velocity_tex.use(location=5)
    resources.gas_tags.bind_to_storage_buffer(binding=5)
    resources.material_params.bind_to_storage_buffer(binding=6)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    resources.action_i.write(compiled[0].tobytes())
    resources.action_f.write(compiled[1].tobytes())
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.flow_source_tex.bind_to_image(2, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
    resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
    resources.local_timer_out.bind_to_image(5, read=False, write=True)
    resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
    ping_is_primary = True
    for rule_index in range(rule_count):
        rule_compiled = pipeline._compile_single_gas_gas_rule(rule_table[rule_index : rule_index + 1])
        resources.gg_rule_i.write(rule_compiled[0].tobytes())
        resources.gg_rule_f.write(rule_compiled[1].tobytes())
        resources.gg_rule_tags.write(rule_compiled[2].tobytes())
        resources.gg_rule_i.bind_to_storage_buffer(binding=2)
        resources.gg_rule_f.bind_to_storage_buffer(binding=3)
        resources.gg_rule_tags.bind_to_storage_buffer(binding=4)
        pipeline._set_uniform_if_present(program, "rule_count", 1)
        if ping_is_primary:
            resources.gas_ping.use(location=0)
            resources.ambient_ping.use(location=1)
            resources.gas_pong.bind_to_image(0, read=False, write=True)
            resources.ambient_pong.bind_to_image(1, read=False, write=True)
        else:
            resources.gas_pong.use(location=0)
            resources.ambient_pong.use(location=1)
            resources.gas_ping.bind_to_image(0, read=False, write=True)
            resources.ambient_ping.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, world.gas_concentration.shape[0])
        pipeline._sync_compute_writes(world.bridge.ctx)
        ping_is_primary = not ping_is_primary
    final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
    final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
    if pipeline._formal_gpu_frame(world):
        pipeline.last_cpu_mirror_downloaded = False
        if pipeline._formal_segment_batch_active():
            pipeline._promote_gas_result(world, resources, final_gas, final_ambient)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "gas")
        else:
            pipeline._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
            pipeline._promote_gas_result(world, resources, final_gas, final_ambient)
    else:
        pipeline.last_cpu_mirror_downloaded = True
        world.gas_concentration[:] = np.maximum(
            np.frombuffer(final_gas.read(), dtype="f4").reshape(world.gas_concentration.shape),
            0.0,
        )
        world.ambient_temperature[:] = np.frombuffer(final_ambient.read(), dtype="f4").reshape(world.ambient_temperature.shape)
    pipeline._append_flow_sources_from_gpu(
        world,
        resources,
        may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(compiled),
    )
    if pipeline._compiled_actions_include_emit_material(compiled):
        pipeline._scatter_local_emit_cell_outputs(world, resources)
        pipeline._download_cell_state(
            world,
            resources,
            direct_core_outputs=pipeline._formal_gpu_frame(world),
            advance_velocity_role=pipeline._formal_before_motion_cell_roles_active(),
        )
    return pipeline._download_deferred_batch(world, resources)



def run_gas_light(
    pipeline,
    world: "WorldEngine",
    *,
    solve_gas_mask: np.ndarray | None = None,
) -> GPUDeferredActionBatch | None:
    if not pipeline.available(world):
        return None
    world.bridge.sync_rule_tables(world)
    rule_table = world.bridge.shadow_typed_tables["gas_light_rule_table"]
    rule_count = int(rule_table.shape[0])
    if rule_count <= 0 or rule_count > MAX_RULES:
        return None
    if pipeline._has_unsupported_consume_policies(rule_table, {CONSUME_POLICY_NONE, CONSUME_POLICY_RHS, CONSUME_POLICY_BOTH}):
        return None
    used_indices = {int(value) for value in rule_table["result_action"].tolist() if int(value) >= 0}
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    action_compiled = pipeline._compile_gas_light_action_buffers(
        action_table,
        used_indices,
    )
    if action_compiled is None:
        return None
    light_dose_guard = pipeline._formal_light_dose_guard_buffer(world)
    if light_dose_guard is not None:
        return pipeline._run_formal_guarded_gas_light(
            world,
            rule_table,
            action_compiled,
            rule_count,
            solve_gas_mask,
            light_dose_guard,
        )
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    direct_bridge_gas_dose = pipeline._formal_gpu_frame(world)
    pipeline._upload_state(
        world,
        resources,
        reaction_group="gas_light",
        compiled_actions=action_compiled,
        direct_bridge_gas_dose=direct_bridge_gas_dose,
    )
    pipeline._upload_local_metadata(world, resources)
    pipeline._upload_active_masks(
        world,
        resources,
        np.ones((world.height, world.width), dtype=np.bool_),
        solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
        reaction_group="gas_light",
        load_cell_mask=False,
    )
    light_table = world.bridge.shadow_typed_tables["light_table"]
    resources.action_i.write(action_compiled[0].tobytes())
    resources.action_f.write(action_compiled[1].tobytes())
    program = pipeline.programs["gas_light"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "use_bridge_gas_dose", direct_bridge_gas_dose)
    resources.gas_dose_tex.use(location=1)
    resources.active_gas_tex.use(location=2)
    cell_state_in, _phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    cell_state_in.use(location=4)
    temp_in.use(location=5)
    resources.flow_velocity_tex.use(location=6)
    resources.action_i.bind_to_storage_buffer(binding=2)
    resources.action_f.bind_to_storage_buffer(binding=3)
    resources.gas_tags.bind_to_storage_buffer(binding=5)
    resources.material_params.bind_to_storage_buffer(binding=6)
    if direct_bridge_gas_dose:
        world.bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=16)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.flow_source_tex.bind_to_image(2, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
    resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
    resources.local_timer_out.bind_to_image(5, read=False, write=True)
    resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    ping_is_primary = True
    for rule_index in range(rule_count):
        rule_compiled = pipeline._compile_single_gas_light_rule(rule_table[rule_index : rule_index + 1], light_table)
        resources.gl_rule_i.write(rule_compiled[0].tobytes())
        resources.gl_rule_f.write(rule_compiled[1].tobytes())
        resources.gl_rule_tags.write(rule_compiled[2].tobytes())
        resources.gl_rule_i.bind_to_storage_buffer(binding=0)
        resources.gl_rule_f.bind_to_storage_buffer(binding=1)
        resources.gl_rule_tags.bind_to_storage_buffer(binding=4)
        pipeline._set_uniform_if_present(program, "rule_count", 1)
        if ping_is_primary:
            resources.gas_ping.use(location=0)
            resources.ambient_ping.use(location=3)
            resources.gas_pong.bind_to_image(0, read=False, write=True)
            resources.ambient_pong.bind_to_image(1, read=False, write=True)
        else:
            resources.gas_pong.use(location=0)
            resources.ambient_pong.use(location=3)
            resources.gas_ping.bind_to_image(0, read=False, write=True)
            resources.ambient_ping.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, world.gas_concentration.shape[0])
        pipeline._sync_compute_writes(world.bridge.ctx)
        ping_is_primary = not ping_is_primary
    final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
    final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
    if pipeline._formal_gpu_frame(world):
        pipeline.last_cpu_mirror_downloaded = False
        if pipeline._formal_segment_batch_active():
            pipeline._promote_gas_result(world, resources, final_gas, final_ambient)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "gas")
        else:
            pipeline._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
            pipeline._promote_gas_result(world, resources, final_gas, final_ambient)
    else:
        pipeline.last_cpu_mirror_downloaded = True
        world.gas_concentration[:] = np.maximum(
            np.frombuffer(final_gas.read(), dtype="f4").reshape(world.gas_concentration.shape),
            0.0,
        )
        world.ambient_temperature[:] = np.frombuffer(final_ambient.read(), dtype="f4").reshape(world.ambient_temperature.shape)
    pipeline._append_flow_sources_from_gpu(
        world,
        resources,
        may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(action_compiled),
    )
    if pipeline._compiled_actions_include_emit_material(action_compiled):
        pipeline._scatter_local_emit_cell_outputs(world, resources)
        pipeline._download_cell_state(
            world,
            resources,
            direct_core_outputs=pipeline._formal_gpu_frame(world),
            advance_velocity_role=pipeline._formal_before_motion_cell_roles_active(),
        )
    return pipeline._download_deferred_batch(world, resources)



def _run_formal_guarded_gas_light(
    pipeline,
    world: "WorldEngine",
    rule_table: np.ndarray,
    action_compiled: tuple[np.ndarray, np.ndarray],
    rule_count: int,
    solve_gas_mask: np.ndarray | None,
    light_dose_guard: Any,
) -> GPUDeferredActionBatch:
    pipeline._ensure_programs(world.bridge.ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "gas_light_upload_state"):
        pipeline._upload_state(
            world,
            resources,
            reaction_group="gas_light",
            compiled_actions=action_compiled,
            light_dose_guard_buffer=light_dose_guard,
            direct_bridge_gas_dose=True,
        )
    active_authoritative = pipeline._active_scheduler_gpu_authoritative(world)
    with pipeline._profile_pass(world, "gas_light_upload_active_masks"):
        pipeline._upload_active_masks(
            world,
            resources,
            None if active_authoritative else np.ones((world.height, world.width), dtype=np.bool_),
            None
            if active_authoritative
            else solve_gas_mask
            if solve_gas_mask is not None
            else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
            reaction_group="gas_light",
            light_dose_guard_buffer=light_dose_guard,
            load_cell_mask=False,
        )
    pipeline._upload_local_metadata(world, resources)
    light_table = world.bridge.shadow_typed_tables["light_table"]
    resources.action_i.write(action_compiled[0].tobytes())
    resources.action_f.write(action_compiled[1].tobytes())
    program = pipeline.programs["gas_light"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
    pipeline._set_uniform_if_present(program, "use_bridge_gas_dose", True)
    resources.gas_dose_tex.use(location=1)
    resources.active_gas_tex.use(location=2)
    cell_state_in, _phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = (
        pipeline._current_cell_textures(resources)
    )
    cell_state_in.use(location=4)
    temp_in.use(location=5)
    resources.flow_velocity_tex.use(location=6)
    resources.action_i.bind_to_storage_buffer(binding=2)
    resources.action_f.bind_to_storage_buffer(binding=3)
    resources.gas_tags.bind_to_storage_buffer(binding=5)
    resources.material_params.bind_to_storage_buffer(binding=6)
    world.bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=16)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    resources.flow_source_tex.bind_to_image(2, read=False, write=True)
    pipeline._bind_flow_source_generation_output(
        world,
        resources,
        program,
        binding=FLOW_SOURCE_GENERATION_BINDING,
    )
    resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
    resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
    resources.local_timer_out.bind_to_image(5, read=False, write=True)
    resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_z = int(world.gas_concentration.shape[0])
    ping_is_primary = True
    for rule_index in range(rule_count):
        rule_compiled = pipeline._compile_single_gas_light_rule(rule_table[rule_index : rule_index + 1], light_table)
        resources.gl_rule_i.write(rule_compiled[0].tobytes())
        resources.gl_rule_f.write(rule_compiled[1].tobytes())
        resources.gl_rule_tags.write(rule_compiled[2].tobytes())
        resources.gl_rule_i.bind_to_storage_buffer(binding=0)
        resources.gl_rule_f.bind_to_storage_buffer(binding=1)
        resources.gl_rule_tags.bind_to_storage_buffer(binding=4)
        pipeline._set_uniform_if_present(program, "rule_count", 1)
        if ping_is_primary:
            resources.gas_ping.use(location=0)
            resources.ambient_ping.use(location=3)
            resources.gas_pong.bind_to_image(0, read=False, write=True)
            resources.ambient_pong.bind_to_image(1, read=False, write=True)
        else:
            resources.gas_pong.use(location=0)
            resources.ambient_pong.use(location=3)
            resources.gas_ping.bind_to_image(0, read=False, write=True)
            resources.ambient_ping.bind_to_image(1, read=False, write=True)
        with pipeline._profile_pass(world, "gas_light_shader"):
            pipeline._run_light_dose_guarded_dispatch(
                world,
                resources,
                program,
                light_dose_guard,
                group_x,
                group_y,
                group_z,
            )
            pipeline._sync_compute_writes(world.bridge.ctx)
        ping_is_primary = not ping_is_primary
    final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
    final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
    if pipeline._formal_terminal_gas_publish_fusion_pending():
        # Gas-light is terminal in the before-motion formal order.  The
        # immediate segment flush applies the pending delta and publishes the
        # final gas state before any bridge consumer can run.
        with pipeline._profile_pass(world, "gas_light_publish_gas_state_deferred"):
            pass
    else:
        with pipeline._profile_pass(world, "gas_light_publish_gas_state"):
            pipeline._publish_bridge_gas_state(
                world,
                resources,
                gas_texture=final_gas,
                ambient_texture=final_ambient,
                light_dose_guard_buffer=light_dose_guard,
            )
    pipeline._append_flow_sources_from_gpu(
        world,
        resources,
        may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(action_compiled),
        light_dose_guard_buffer=light_dose_guard,
    )
    if pipeline._compiled_actions_include_emit_material(action_compiled):
        pipeline._scatter_local_emit_cell_outputs(
            world,
            resources,
            light_dose_guard_buffer=light_dose_guard,
        )
        if pipeline._formal_before_motion_cell_roles_active():
            source_role = pipeline._formal_cell_write_role()
            source_velocity_role = pipeline._formal_velocity_write_role()
            pipeline._accumulate_segment_cell_transient_state(
                world,
                resources,
                direct_core_outputs=True,
                light_dose_guard_buffer=light_dose_guard,
            )
            pipeline._set_formal_cell_read_role(source_role)
            pipeline._set_formal_velocity_read_role(source_velocity_role)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
        else:
            pipeline._publish_bridge_cell_state(
                world,
                resources,
                cell_meta_texture=resources.local_cell_meta_out,
                light_dose_guard_buffer=light_dose_guard,
            )
            pipeline._promote_cell_pong_to_ping(world, resources)
    return pipeline._download_deferred_batch(world, resources)



def clear_reaction_latches(pipeline, world: "WorldEngine") -> bool:
    if not pipeline.available(world):
        return False
    ctx = world.bridge.ctx
    if ctx is None:
        return False
    if pipeline._formal_gpu_frame(world):
        pipeline._clear_reaction_latches_on_bridge(world)
        pipeline.last_cpu_mirror_downloaded = False
        return True
    if pipeline._clear_latches_program is None:
        pipeline._clear_latches_program = build_compute_shader(
            ctx, "reactions/_clear_latches_program.comp", _SHADER_SUBS
        )
    flat_flags = np.asarray(world.cell_flags.reshape(-1), dtype=np.uint32)
    flag_buffer = ctx.buffer(flat_flags.tobytes())
    try:
        pipeline._clear_latches_program["cell_count"].value = int(flat_flags.size)
        flag_buffer.bind_to_storage_buffer(binding=0)
        pipeline._clear_latches_program.run((int(flat_flags.size) + 255) // 256, 1, 1)
        pipeline._sync_compute_writes(ctx)
        world.cell_flags[:] = np.frombuffer(flag_buffer.read(), dtype=np.uint32).reshape(world.cell_flags.shape).astype(np.uint8)
    finally:
        flag_buffer.release()
    pipeline.last_cpu_mirror_downloaded = True
    return True

# ``_formal_gpu_frame`` is inherited from GPUPipelineBase.
