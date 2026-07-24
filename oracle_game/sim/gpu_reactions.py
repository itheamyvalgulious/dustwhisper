from __future__ import annotations

import time  # noqa: F401  # facade re-export, grafting/monkeypatch contract
from contextlib import (
    contextmanager,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
)
from dataclasses import (
    dataclass,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    field,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
)
from typing import Any

import numpy as np

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.gpu import (  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CONSUME_POLICY_IDS,
    DIRECTION_IDS,
    typed_material_id,
)
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    _ensure_material_flags_buffer,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    ensure_collapse_structure_dirty_tile_mask,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    ensure_collapse_structure_dirty_tile_queue,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    mark_collapse_structure_dirty_tiles_from_bridge_cell_core,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
)
from oracle_game.sim.gpu_reactions_bridge import (
    _append_flow_sources_from_gpu,
    _apply_flow_sources_to_bridge_velocity,
    _download_cell_state,
    _download_deferred_batch,
    _download_dose_state,
    _download_gas_state,
    _load_authoritative_bridge_inputs,
    _publish_bridge_cell_state,
    _publish_bridge_dose_state,
    _publish_bridge_gas_state,
    _publish_bridge_light_emitters,
    _unsupported_deferred_action_indices,
)
from oracle_game.sim.gpu_reactions_cell_pass import (
    _accumulate_timed_candidate_segment_cell_transient_state,
    _bind_local_cell_action_output_images,
    _clear_timed_candidate_local_meta,
    _copy_current_velocity_to_next_role,
    _prepare_self_candidate_worklist,
    _prepare_timed_candidate_worklist,
    _publish_timed_candidate_cell_state,
    _run_cell_pass,
    _run_local_cell_action_pass,
    _run_material_pair_fused_pass,
    _run_timed_candidate_action_pass,
    _scatter_local_cell_action_outputs,
    _scatter_local_emit_cell_outputs,
)
from oracle_game.sim.gpu_reactions_pairings import (
    _compile_material_pair_plan,
    _compile_material_pair_plan_cached,
    _material_pair_plan_cache_key,
    _run_formal_guarded_gas_light,
    _run_formal_guarded_material_light,
    clear_reaction_latches,
    run_gas_gas,
    run_gas_light,
    run_material_gas,
    run_material_light,
    run_material_material,
    run_material_pair_fused,
    run_self_actions,
    run_self_triggers,
    run_timed_actions,
    run_timed_triggers,
)
from oracle_game.sim.gpu_reactions_resources import (
    _SHADER_SUBS,
    ACTION_FLAG_ALLOW_SUBUNIT_SCALE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    ACTION_FLAG_RANDOM_TARGET,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CONSUME_POLICY_BOTH,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CONSUME_POLICY_LHS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CONSUME_POLICY_NONE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CONSUME_POLICY_RHS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    DIRECT_CORE_OUTPUT_REACTION_GROUPS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    FLOW_SOURCE_GENERATION_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    FLOW_SOURCE_LAYERS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    FORMAL_GPU_EMPTY_DEFERRED_BATCH,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    GAS_DELTA_FIXED_SCALE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    LIGHT_DOSE_GUARD_BUFFER,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    LOCAL_SIZE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MATERIAL_LIGHT_PACKED_HEADER_OFFSET,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MATERIAL_PAIR_PACKED_HEADER_OFFSET,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
    MAX_ACTIONS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MAX_EMITTED_LIGHTS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MAX_MATERIAL_LIGHT_PACKED_RULES,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MAX_MATERIAL_PAIR_PACKED_RULES,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    MAX_MATERIALS,
    MAX_RULES,
    MAX_SELF_RULES,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    RULE_CANDIDATE_VECS,
    RULE_CANDIDATE_WORDS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    SELF_FUSED_FLOW_SOURCE_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    SELF_FUSED_GAS_DELTA_BINDING,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_CONVERT_MATERIAL,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_DEFERRED,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_EMIT_LIGHT,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_EMIT_MATERIAL,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_HARM,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_MODIFY_GAS,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_MODIFY_TEMPERATURE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    TYPE_NONE,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    GPUDeferredActionBatch,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    GPUReactionBridgeInputLoads,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    GPUReactionMaterialPairPlan,
    GPUReactionResources,
    _ensure_resources,
    _profile_scoped_pass,
    _record_profile_pass,
    _upload_state_profile_scope,
)
from oracle_game.sim.gpu_reactions_rules import (
    _cached_used_action_indices_for_material_slots,
    _cached_used_action_indices_for_pair_rules,
    _cached_used_action_indices_for_self_rules,
    _compile_action_buffers,
    _compile_action_buffers_cached,
    _compile_gas_action_buffers,
    _compile_gas_gas_rules,
    _compile_gas_light_action_buffers,
    _compile_gas_light_rules,
    _compile_material_gas_rules,
    _compile_material_light_packed_descriptors,
    _compile_material_light_packed_descriptors_cached,
    _compile_material_light_rules,
    _compile_material_material_rules,
    _compile_material_pair_packed_descriptors,
    _compile_material_pair_packed_descriptors_cached,
    _compile_material_rule_candidate_masks,
    _compile_single_gas_gas_rule,
    _compile_single_gas_light_rule,
    _compiled_actions_include_emit_material,
    _compiled_actions_include_flow_sources,
    _compiled_actions_include_modify_gas,
    _compiled_actions_may_change_structure,
    _compiled_actions_require_deferred_outputs,
    _compiled_modify_gas_layer_mask,
    _compiled_rules_include_rhs_consume,
    _compiled_self_rule_flow_source_layers,
    _empty_rule_candidate_masks,
    _has_unsupported_consume_policies,
    _modify_gas_action_requires_cpu_flow_side_effect,
    _rule_candidate_word_count,
    _self_rules_require_deferred_hi_outputs,
    _set_rule_candidate,
    _used_action_indices,
    _used_action_indices_for_material_slots,
    _used_action_indices_for_pair_rules,
    _used_action_indices_for_self_rules,
)
from oracle_game.sim.gpu_reactions_segments import (
    _active_masks_for_cell_reaction_upload,
    _active_scheduler_gpu_authoritative,
    _advance_formal_cell_read_role,
    _advance_formal_velocity_read_role,
    _bridge_cell_core_read_role_only_load,
    _build_light_dose_guarded_dispatch_args,
    _can_use_expanded_active_tile_mask,
    _cell_role_textures,
    _clear_formal_external_cell_state,
    _clear_formal_segment_gas_delta,
    _clear_reaction_latches_on_bridge,
    _current_cell_textures,
    _flush_formal_segment_gas_delta,
    _formal_before_motion_cell_roles_active,
    _formal_cell_read_role,
    _formal_cell_write_role,
    _formal_light_dose_guard_buffer,
    _formal_reaction_active_mask_cache_key,
    _formal_reaction_segment_base_key,
    _formal_reaction_segment_cache_key,
    _formal_reaction_state_cache_active,
    _formal_reaction_state_cache_key,
    _formal_segment_batch_active,
    _formal_state_key_is_before_motion,
    _formal_terminal_gas_publish_fusion_pending,
    _formal_velocity_read_role,
    _formal_velocity_write_role,
    _load_authoritative_active_masks,
    _mark_formal_bridge_publish_pending,
    _next_cell_textures,
    _reaction_state_segment,
    _reset_formal_cell_read_role,
    _reset_formal_velocity_read_role,
    _run_light_dose_guarded_dispatch,
    _set_formal_cell_read_role,
    _set_formal_velocity_read_role,
    _upload_active_masks,
    begin_formal_reaction_segment,
    end_formal_reaction_segment,
    flush_formal_reaction_segment,
)
from oracle_game.sim.gpu_reactions_side_effects import (
    _clear_packed_timed_material_target_worklist,
    _run_cell_gas_action_delta_pass,
    _run_cell_gas_side_effect_pass,
    _run_cell_material_side_effect_pass,
    _run_material_light_dose_consume_pass,
    _run_packed_material_target_apply_pass,
    _run_packed_timed_material_side_effect_pass,
    _run_produced_packed_timed_material_side_effect_pass,
    _run_self_candidate_gas_side_effect_pass,
    _run_timed_candidate_gas_side_effect_pass,
    _run_timed_candidate_material_side_effect_pass,
)
from oracle_game.sim.gpu_reactions_timed_self import _run_timed_self_combined_action_pass
from oracle_game.sim.gpu_reactions_transient import (
    _accumulate_segment_cell_transient_state,
    _advance_flow_source_generation,
    _begin_formal_segment_meta_lazy_zero,
    _bind_flow_source_generation_output,
    _bridge_input_load_requirements,
    _can_use_terminal_segment_meta_zero,
    _clear_segment_transient_state,
    _clear_transient_state,
    _copy_bridge_flow_velocity_to_reaction,
    _copy_gas_state,
    _ensure_formal_segment_meta_physical_zero,
    _flow_source_generation_validity_active,
    _missing_formal_bridge_input_loads,
    _promote_cell_pong_to_ping,
    _promote_dose_pong_to_ping,
    _promote_gas_pong_to_ping,
    _promote_gas_result,
    _record_formal_bridge_inputs_loaded,
    _record_formal_segment_cell_meta_in_flags,
    _reset_formal_segment_meta_lazy_zero,
    _sync_compute_writes,
    _sync_storage_and_indirect_writes,
    _transient_clear_requirements,
    _upload_local_metadata,
    _upload_random_targets,
    _upload_state,
    release,
)
from oracle_game.sim.shader_loader import (  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    build_compute_shader,
    shader_source,
)
from oracle_game.types import (  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    CellFlag,
    ForceSource,  # noqa: F401  # facade re-export, grafting/monkeypatch contract
    Phase,
    ReactionType,
)


class GPUReactionPipeline(GPUPipelineBase):
    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
        self.resources: GPUReactionResources | None = None
        self.programs: dict[str, Any] = {}
        self._clear_latches_program: Any | None = None
        self._clear_bridge_latches_program: Any | None = None
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_flow_velocity_upload_skipped = False
        self.last_cpu_cell_dose_upload_skipped = False
        self.last_cpu_gas_dose_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self._expanded_active_tile_mask_enabled = (
            self.engine_config.expanded_active_tile_mask_enabled
        )
        self.last_expanded_active_tile_mask_used = False
        self.expanded_active_tile_mask_build_count = 0
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self.random_targets = np.zeros((MAX_MATERIALS,), dtype=np.int32)
        self.random_target_count = 0
        self._used_action_indices_cache: dict[tuple[object, ...], set[int] | None] = {}
        self._compiled_action_cache: dict[
            tuple[object, ...], tuple[np.ndarray, np.ndarray] | None
        ] = {}
        self._material_light_packed_descriptor_cache_key: tuple[object, ...] | None = None
        self._material_light_packed_descriptor_cache: np.ndarray | None = None
        self._material_pair_packed_descriptor_cache_key: tuple[object, ...] | None = None
        self._material_pair_packed_descriptor_cache: np.ndarray | None = None
        self._material_pair_plan_cache: dict[
            tuple[object, ...], GPUReactionMaterialPairPlan | None
        ] = {}
        self._formal_state_cache_key: tuple[object, ...] | None = None
        self._formal_active_mask_cache_key: tuple[object, ...] | None = None
        self._formal_loaded_bridge_inputs_key: tuple[object, ...] | None = None
        self._formal_loaded_bridge_inputs: set[str] = set()
        self._formal_segment_batch_base_key: tuple[object, ...] | None = None
        self._formal_segment_batch_key: tuple[object, ...] | None = None
        self._formal_light_counters_cleared_key: tuple[object, ...] | None = None
        self._formal_pending_bridge_publish_key: tuple[object, ...] | None = None
        self._formal_pending_bridge_publish: set[str] = set()
        self._motion_handoff_candidate: dict[str, Any] | None = None
        self._formal_pending_gas_delta_key: tuple[object, ...] | None = None
        # Candidate: the terminal gas-delta apply can publish its exact result
        # to the bridge while preserving the resident ping/pong transition.
        self._terminal_gas_publish_fusion_enabled = (
            self.engine_config.terminal_gas_publish_fusion_enabled
        )
        self._formal_cell_state_role_key: tuple[object, ...] | None = None
        self._formal_cell_state_read_role: str = "ping"
        # Cell-core roles can advance without touching velocity.  Reaction
        # passes only write velocity for EMIT_MATERIAL side effects; keeping a
        # separate role avoids a full-screen velocity copy after every other
        # direct-core pass.
        self._formal_velocity_state_role_key: tuple[object, ...] | None = None
        self._formal_velocity_state_read_role: str = "ping"
        self._formal_external_cell_state_key: tuple[object, ...] | None = None
        self._formal_external_cell_state_textures: tuple[Any, Any, Any, Any, Any, Any] | None = None
        self._formal_external_cell_flags_texture: Any | None = None
        self._pair_segment_meta_fusion_enabled = self.engine_config.pair_segment_meta_fusion_enabled
        # Candidate: timed/self can carry reset/latch operations in packed
        # cell flags. A proven terminal pass may then start its local material
        # metadata at zero and avoid initializing the full-grid segment image.
        self._terminal_segment_meta_lazy_zero_enabled = (
            self.engine_config.terminal_segment_meta_lazy_zero_enabled
        )
        self._formal_segment_meta_lazy_key: tuple[object, ...] | None = None
        self._formal_segment_meta_logically_zero = False
        self._formal_segment_meta_physically_cleared = False
        self._formal_segment_all_prior_cell_meta_in_flags = False
        self.last_terminal_segment_meta_lazy_zero_used = False
        self.last_segment_meta_lazy_clear_skipped = False
        self.segment_meta_lazy_fallback_clear_count = 0
        # Experimental: the segment-meta initializer already covers every
        # cell.  Its first row can also reset the 16 reaction counters and
        # remove a separate one-workgroup dispatch.
        self._segment_meta_light_counter_clear_fusion_enabled = (
            self.engine_config.segment_meta_light_counter_clear_fusion_enabled
        )
        self._packed_timed_emit_target_worklist_enabled = (
            self.engine_config.packed_timed_emit_target_worklist_enabled
        )
        # Candidate: timed_apply can append its packed low-4 emit-material
        # targets while the action id and source cell are already resident.
        # Keep the established full-grid compactor as the default fallback.
        self._timed_emit_target_producer_enabled = (
            self.engine_config.timed_emit_target_producer_enabled
        )
        # Candidate: cache the packed input word once in dense self_apply and
        # reuse it for material, phase, and preserved output flags.
        self.last_self_apply_cached_cell_state_used = False
        self._packed_self_emit_target_worklist_enabled = (
            self.engine_config.packed_self_emit_target_worklist_enabled
        )
        self._authoritative_lhs_candidate_masks_enabled = (
            self.engine_config.authoritative_lhs_candidate_masks_enabled
        )
        self._self_gas_candidate_worklist_enabled = (
            self.engine_config.self_gas_candidate_worklist_enabled
        )
        # Formal batches invalidate flow sources by generation instead of
        # clearing the 32-layer payload texture before each pass.
        self._flow_source_generation_validity_enabled = (
            self.engine_config.flow_source_generation_validity_enabled
        )
        self._flow_source_generation_programs_enabled = False
        # Candidate: use an 8-bit generation token for the 32-layer validity
        # texture, clearing before token 1 is reused after 255.
        self._flow_source_generation_u8_token_enabled = (
            self.engine_config.flow_source_generation_u8_token_enabled
        )
        self._flow_source_generation_u8_programs_enabled = False
        # Experimental: timed/self already consume the GPU-authoritative TTL
        # mask directly.  Represent their CPU-side runtime bookkeeping with
        # the same non-materialized segment mask used by the fused pair pass,
        # rather than copying and OR-ing full-resolution bool grids twice.
        self._timed_self_authoritative_segment_masks_enabled = (
            self.engine_config.timed_self_authoritative_segment_masks_enabled
        )
        # Experimental: apply timed/self reset and latch metadata directly to
        # the packed cell-state flags, avoiding a later full-grid accumulation.
        self._timed_self_cell_flag_meta_enabled = (
            self.engine_config.timed_self_cell_flag_meta_enabled
        )
        self.last_timed_self_cell_flag_meta_used = False
        self._self_rule_material_spans_enabled = self.engine_config.self_rule_material_spans_enabled
        # Experimental: explicit self-rule spans carry their resolved action
        # index, avoiding a per-rule material-slot SSBO lookup.  Wildcard or
        # malformed rules stay on the canonical path.
        self._self_rule_direct_action_spans_enabled = (
            self.engine_config.self_rule_direct_action_spans_enabled
        )
        self._material_pair_state_fusion_enabled = (
            self.engine_config.material_pair_state_fusion_enabled
        )
        self._material_pair_light_state_fusion_enabled = (
            self.engine_config.material_pair_light_state_fusion_enabled
        )
        # Publish the material/light terminal state directly to motion.  The
        # handoff is raw-byte exact and avoids a second full-grid integration.
        self._material_triplet_motion_terminal_enabled = (
            self.engine_config.material_triplet_motion_terminal_enabled
        )
        # The terminal shader has no shared tile state; 16x16 groups reduce
        # dispatch overhead while preserving per-cell evaluation.
        self._material_triplet_terminal_local16_enabled = (
            self.engine_config.material_triplet_terminal_local16_enabled
        )
        self._material_triplet_terminal_dirty_fast_equal_enabled = (
            self.engine_config.material_triplet_terminal_dirty_fast_equal_enabled
        )
        # Experimental: stage terminal bridge words in shared memory and emit
        # row-contiguous stores.  Keep the canonical per-cell stores default.
        self._material_triplet_terminal_shared_transpose_enabled = (
            self.engine_config.material_triplet_terminal_shared_transpose_enabled
        )
        # Candidate: keep 256 invocations but make each NVIDIA warp cover one
        # complete row. The established 16x16 terminal remains the fallback.
        self._material_triplet_terminal_32x8_enabled = (
            self.engine_config.material_triplet_terminal_32x8_enabled
        )
        self.last_material_pair_terminal_handoff = False
        self._material_triplet_ml_packed_descriptors_enabled = (
            self.engine_config.material_triplet_ml_packed_descriptors_enabled
        )
        self._material_pair_packed_descriptors_enabled = (
            self.engine_config.material_pair_packed_descriptors_enabled
        )
        self.last_material_pair_fused_light = False

    # ``available`` and ``reset_pass_profile`` are inherited from GPUPipelineBase.

    def _ensure_programs(self, ctx: Any | None) -> None:
        if not ctx or self.programs:
            return
        self._flow_source_generation_programs_enabled = bool(
            self._flow_source_generation_validity_enabled
        )
        self._flow_source_generation_u8_programs_enabled = bool(
            self._flow_source_generation_programs_enabled
            and self._flow_source_generation_u8_token_enabled
        )
        flow_source_subs = {
            **_SHADER_SUBS,
            "FLOW_SOURCE_GENERATION_VALIDITY": int(self._flow_source_generation_programs_enabled),
            "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": (
                "r8ui" if self._flow_source_generation_u8_programs_enabled else "r32ui"
            ),
        }
        self.programs["load_active_cell"] = build_compute_shader(
            ctx,
            "reactions/load_active_cell.comp",
            _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_expanded_active_tiles"] = build_compute_shader(
            ctx,
            "reactions/load_expanded_active_tiles.comp",
            _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_active_gas"] = build_compute_shader(
            ctx,
            "reactions/load_active_gas.comp",
            _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_bridge_cell"] = build_compute_shader(
            ctx, "reactions/load_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_role"] = build_compute_shader(
            ctx, "reactions/load_bridge_cell_role.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_aux"] = build_compute_shader(
            ctx, "reactions/load_bridge_cell_aux.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_aux_role"] = build_compute_shader(
            ctx, "reactions/load_bridge_cell_aux_role.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_gas"] = build_compute_shader(
            ctx, "reactions/load_bridge_gas.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_dose"] = build_compute_shader(
            ctx, "reactions/load_bridge_dose.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_cell"] = build_compute_shader(
            ctx, "reactions/publish_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_gas"] = build_compute_shader(
            ctx, "reactions/publish_bridge_gas.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_cell_dose"] = build_compute_shader(
            ctx, "reactions/publish_bridge_cell_dose.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_gas_dose"] = build_compute_shader(
            ctx, "reactions/publish_bridge_gas_dose.comp", _SHADER_SUBS
        )
        self.programs["apply_bridge_flow_sources"] = build_compute_shader(
            ctx,
            "reactions/apply_bridge_flow_sources.comp",
            flow_source_subs,
        )
        self.programs["promote_reaction_cell_state"] = build_compute_shader(
            ctx, "reactions/promote_reaction_cell_state.comp", _SHADER_SUBS
        )
        self.programs["copy_reaction_velocity_state"] = build_compute_shader(
            ctx, "reactions/copy_reaction_velocity_state.comp", _SHADER_SUBS
        )
        self.programs["promote_reaction_gas_state"] = build_compute_shader(
            ctx, "reactions/promote_reaction_gas_state.comp", _SHADER_SUBS
        )
        self.programs["promote_reaction_dose_state"] = build_compute_shader(
            ctx, "reactions/promote_reaction_dose_state.comp", _SHADER_SUBS
        )
        self.programs["copy_bridge_flow_velocity_to_reaction"] = build_compute_shader(
            ctx, "reactions/copy_bridge_flow_velocity_to_reaction.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_light_emitters"] = build_compute_shader(
            ctx, "reactions/publish_bridge_light_emitters.comp", _SHADER_SUBS
        )
        self.programs["timed_trigger"] = build_compute_shader(
            ctx, "reactions/timed_trigger.comp", _SHADER_SUBS
        )
        self.programs["self_trigger"] = build_compute_shader(
            ctx, "reactions/self_trigger.comp", _SHADER_SUBS
        )
        self.programs["timed_apply"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["timed_apply_packed"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_local_action_output_packed.comp"],
        )
        timed_emit_target_subs = {**_SHADER_SUBS, "TIMED_EMIT_TARGET_PRODUCER": 1}
        self.programs["timed_apply_packed_emit_targets"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            timed_emit_target_subs,
            includes=[
                "reactions/_common.comp",
                "reactions/_timed_emit_target_output.comp",
                "reactions/_local_action_output_packed.comp",
            ],
        )
        packed_cell_meta_subs = {**_SHADER_SUBS, "PACK_CELL_META_IN_STATE": 1}
        self.programs["timed_apply_packed_cell_flag_meta"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            packed_cell_meta_subs,
            includes=["reactions/_common.comp", "reactions/_local_action_output_packed.comp"],
        )
        timed_emit_target_cell_meta_subs = {
            **timed_emit_target_subs,
            "PACK_CELL_META_IN_STATE": 1,
        }
        self.programs["timed_apply_packed_emit_targets_cell_flag_meta"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            timed_emit_target_cell_meta_subs,
            includes=[
                "reactions/_common.comp",
                "reactions/_timed_emit_target_output.comp",
                "reactions/_local_action_output_packed.comp",
            ],
        )
        self.programs["timed_apply_packed_sparse_inplace"] = build_compute_shader(
            ctx,
            "reactions/timed_apply.comp",
            {
                **timed_emit_target_cell_meta_subs,
                "TIMED_SPARSE_INPLACE": 1,
            },
            includes=[
                "reactions/_common.comp",
                "reactions/_timed_emit_target_output.comp",
                "reactions/_local_action_output_packed.comp",
            ],
        )
        self.programs["clear_timed_candidate_worklist"] = build_compute_shader(
            ctx, "reactions/clear_timed_candidate_worklist.comp", _SHADER_SUBS
        )
        self.programs["clear_timed_material_target_worklist"] = build_compute_shader(
            ctx,
            "reactions/clear_timed_material_target_worklist.comp",
            _SHADER_SUBS,
        )
        self.programs["build_light_dose_guarded_dispatch_args"] = build_compute_shader(
            ctx, "reactions/build_light_dose_guarded_dispatch_args.comp", _SHADER_SUBS
        )
        self.programs["compact_self_candidates"] = build_compute_shader(
            ctx, "reactions/compact_self_candidates.comp", _SHADER_SUBS
        )
        self.programs["timed_apply_candidates"] = build_compute_shader(
            ctx,
            "reactions/timed_apply_candidates.comp",
            _SHADER_SUBS,
            includes=["reactions/_common_no_direct.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["self_apply"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            _SHADER_SUBS,
            includes=["reactions/_common_self_apply.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["self_apply_packed"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            _SHADER_SUBS,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        self.programs["self_apply_packed_cell_flag_meta"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            packed_cell_meta_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        sparse_self_subs = {**_SHADER_SUBS, "SELF_SPARSE_INPLACE": 1}
        self.programs["self_apply_packed_sparse"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            sparse_self_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
        )
        sparse_self_direct_subs = {
            **sparse_self_subs,
            "SELF_RULE_DIRECT_ACTION_SPANS": 1,
        }
        self.programs["self_apply_packed_sparse_direct_spans"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            sparse_self_direct_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
        )
        self.programs["self_apply_packed_sparse_cell_flag_meta"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {**sparse_self_subs, "PACK_CELL_META_IN_STATE": 1},
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
        )
        self.programs["self_apply_packed_sparse_direct_spans_cell_flag_meta"] = (
            build_compute_shader(
                ctx,
                "reactions/self_apply.comp",
                {**sparse_self_direct_subs, "PACK_CELL_META_IN_STATE": 1},
                includes=[
                    "reactions/_common_self_apply.comp",
                    "reactions/_local_action_output.comp",
                    "reactions/_self_sparse_dispatch_io.comp",
                ],
            )
        )
        direct_self_span_subs = {
            **_SHADER_SUBS,
            "SELF_RULE_DIRECT_ACTION_SPANS": 1,
        }
        self.programs["self_apply_packed_direct_spans"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            direct_self_span_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        direct_self_span_cell_meta_subs = {
            **direct_self_span_subs,
            "PACK_CELL_META_IN_STATE": 1,
        }
        self.programs["self_apply_packed_direct_spans_cell_flag_meta"] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            direct_self_span_cell_meta_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        self.programs["timed_self_apply_combined"] = build_compute_shader(
            ctx,
            "reactions/timed_self_apply_combined.comp",
            _SHADER_SUBS,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
                "reactions/_timed_self_combined_output.comp",
            ],
        )
        self.programs["scatter_timed_self_gas_action_delta"] = build_compute_shader(
            ctx,
            "reactions/scatter_timed_self_gas_action_delta.comp",
            _SHADER_SUBS,
        )
        self.programs["scatter_self_gas_action_delta_candidates"] = build_compute_shader(
            ctx,
            "reactions/scatter_self_gas_action_delta_candidates.comp",
            flow_source_subs,
        )
        self.programs["scatter_local_action_outputs"] = build_compute_shader(
            ctx, "reactions/scatter_local_action_outputs.comp", _SHADER_SUBS
        )
        self.programs["scatter_local_action_deferred_meta_outputs"] = build_compute_shader(
            ctx, "reactions/scatter_local_action_deferred_meta_outputs.comp", _SHADER_SUBS
        )
        self.programs["scatter_local_action_tail_outputs"] = build_compute_shader(
            ctx, "reactions/scatter_local_action_tail_outputs.comp", _SHADER_SUBS
        )
        self.programs["scatter_local_emit_cell_outputs"] = build_compute_shader(
            ctx, "reactions/scatter_local_emit_cell_outputs.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_cell_state"] = build_compute_shader(
            ctx, "reactions/clear_transient_cell_state.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_aux_state"] = build_compute_shader(
            ctx, "reactions/clear_transient_aux_state.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_light_counters"] = build_compute_shader(
            ctx, "reactions/clear_transient_light_counters.comp", _SHADER_SUBS
        )
        self.programs["clear_timed_candidate_local_meta"] = build_compute_shader(
            ctx, "reactions/clear_timed_candidate_local_meta.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_emit_material_mask"] = build_compute_shader(
            ctx, "reactions/clear_transient_emit_material_mask.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_emit_material_buffers"] = build_compute_shader(
            ctx, "reactions/clear_transient_emit_material_buffers.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_flow_sources"] = build_compute_shader(
            ctx, "reactions/clear_transient_flow_sources.comp", _SHADER_SUBS
        )
        self.programs["clear_transient_flow_source_generations"] = build_compute_shader(
            ctx,
            "reactions/clear_transient_flow_source_generations.comp",
            flow_source_subs,
        )
        self.programs["clear_segment_cell_transient_state"] = build_compute_shader(
            ctx, "reactions/clear_segment_cell_transient_state.comp", _SHADER_SUBS
        )
        self.programs["clear_segment_cell_transient_state_light_counters"] = build_compute_shader(
            ctx,
            "reactions/clear_segment_cell_transient_state.comp",
            {**_SHADER_SUBS, "CLEAR_LIGHT_COUNTERS": 1},
        )
        self.programs["accumulate_segment_cell_transient_state"] = build_compute_shader(
            ctx, "reactions/accumulate_segment_cell_transient_state.comp", _SHADER_SUBS
        )
        self.programs["accumulate_timed_candidate_segment_cell_transient_state"] = (
            build_compute_shader(
                ctx,
                "reactions/accumulate_timed_candidate_segment_cell_transient_state.comp",
                _SHADER_SUBS,
            )
        )
        self.programs["cell_material_side_effects"] = build_compute_shader(
            ctx, "reactions/cell_material_side_effects.comp", _SHADER_SUBS
        )
        self.programs["compact_timed_material_targets"] = build_compute_shader(
            ctx, "reactions/compact_timed_material_targets.comp", _SHADER_SUBS
        )
        self.programs["cell_material_side_effects_candidates"] = build_compute_shader(
            ctx, "reactions/cell_material_side_effects_candidates.comp", _SHADER_SUBS
        )
        self.programs["compact_packed_timed_material_targets"] = build_compute_shader(
            ctx, "reactions/compact_packed_timed_material_targets.comp", _SHADER_SUBS
        )
        self.programs["cell_material_side_effects_packed_targets"] = build_compute_shader(
            ctx, "reactions/cell_material_side_effects_packed_targets.comp", _SHADER_SUBS
        )
        self.programs["build_packed_material_target_dispatch"] = build_compute_shader(
            ctx, "reactions/build_packed_material_target_dispatch.comp", _SHADER_SUBS
        )
        self.programs["clear_cell_gas_delta"] = build_compute_shader(
            ctx, "reactions/clear_cell_gas_delta.comp", _SHADER_SUBS
        )
        self.programs["scatter_cell_gas_action_delta"] = build_compute_shader(
            ctx,
            "reactions/scatter_cell_gas_action_delta.comp",
            flow_source_subs,
        )
        self.programs["scatter_cell_gas_action_delta_candidates"] = build_compute_shader(
            ctx,
            "reactions/scatter_cell_gas_action_delta_candidates.comp",
            flow_source_subs,
        )
        self.programs["apply_cell_gas_delta"] = build_compute_shader(
            ctx, "reactions/apply_cell_gas_delta.comp", _SHADER_SUBS
        )
        self.programs["apply_cell_gas_delta_publish_bridge"] = build_compute_shader(
            ctx,
            "reactions/apply_cell_gas_delta_publish_bridge.comp",
            _SHADER_SUBS,
        )
        self.programs["cell_gas_side_effects"] = build_compute_shader(
            ctx,
            "reactions/cell_gas_side_effects.comp",
            flow_source_subs,
        )
        self.programs["material_light_cell_dose_consume"] = build_compute_shader(
            ctx, "reactions/material_light_cell_dose_consume.comp", _SHADER_SUBS
        )
        self.programs["material_light_gas_dose_consume"] = build_compute_shader(
            ctx, "reactions/material_light_gas_dose_consume.comp", _SHADER_SUBS
        )
        self.programs["material_material"] = build_compute_shader(
            ctx,
            "reactions/material_material.comp",
            _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_material_authoritative_lhs"] = build_compute_shader(
            ctx,
            "reactions/material_material.comp",
            _SHADER_SUBS,
            includes=[
                "reactions/_common.comp",
                "reactions/_lhs_candidate.comp",
                "reactions/_authoritative_lhs_candidate.comp",
            ],
        )
        self.programs["material_gas"] = build_compute_shader(
            ctx,
            "reactions/material_gas.comp",
            _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_gas_authoritative_lhs"] = build_compute_shader(
            ctx,
            "reactions/material_gas.comp",
            _SHADER_SUBS,
            includes=[
                "reactions/_common.comp",
                "reactions/_lhs_candidate.comp",
                "reactions/_authoritative_lhs_candidate.comp",
            ],
        )
        material_pair_subs = {
            **_SHADER_SUBS,
            "ENABLE_LIGHT_EMITTER_OUTPUT": 0,
            "MAX_RULES": MAX_RULES * 2 + 1,
            "RULE_I_CAPACITY": MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
            "MAX_MATERIALS_TIMES_RULE_CANDIDATE_VECS": (MAX_MATERIALS * RULE_CANDIDATE_VECS * 2),
        }
        self.programs["material_pair_fused"] = build_compute_shader(
            ctx,
            "reactions/material_pair_fused.comp",
            material_pair_subs,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        material_pair_terminal_subs = {
            **material_pair_subs,
            "MATERIAL_PAIR_TERMINAL_HANDOFF": 1,
            "DIRECT_GAS_DELTA_BINDING": 3,
            "BRIDGE_CELL_DOSE_BINDING": 4,
            "REACTION_COUNTER_BINDING": 5,
        }
        self.programs["material_pair_fused_terminal"] = build_compute_shader(
            ctx,
            "reactions/material_pair_fused.comp",
            material_pair_terminal_subs,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_pair_fused_terminal_local16"] = build_compute_shader(
            ctx,
            "reactions/material_pair_fused.comp",
            {
                **material_pair_terminal_subs,
                "LOCAL_SIZE": 16,
                "LOCAL_SIZE_X": 16,
                "LOCAL_SIZE_Y": 16,
            },
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_pair_fused_terminal_local16_dirty_fast"] = build_compute_shader(
            ctx,
            "reactions/material_pair_fused.comp",
            {
                **material_pair_terminal_subs,
                "LOCAL_SIZE": 16,
                "LOCAL_SIZE_X": 16,
                "LOCAL_SIZE_Y": 16,
                "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 1,
            },
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_pair_fused_terminal_local16_dirty_fast_shared_transpose"] = (
            build_compute_shader(
                ctx,
                "reactions/material_pair_fused.comp",
                {
                    **material_pair_terminal_subs,
                    "LOCAL_SIZE": 16,
                    "LOCAL_SIZE_X": 16,
                    "LOCAL_SIZE_Y": 16,
                    "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 1,
                    "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 1,
                },
                includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
            )
        )
        self.programs["material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose"] = (
            build_compute_shader(
                ctx,
                "reactions/material_pair_fused.comp",
                {
                    **material_pair_terminal_subs,
                    "LOCAL_SIZE": 16,
                    "LOCAL_SIZE_X": 32,
                    "LOCAL_SIZE_Y": 8,
                    "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 1,
                    "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 1,
                },
                includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
            )
        )
        self.programs[
            "material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose_segment_zero"
        ] = build_compute_shader(
            ctx,
            "reactions/material_pair_fused.comp",
            {
                **material_pair_terminal_subs,
                "LOCAL_SIZE": 16,
                "LOCAL_SIZE_X": 32,
                "LOCAL_SIZE_Y": 8,
                "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 1,
                "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 1,
                "MATERIAL_PAIR_TERMINAL_SEGMENT_META_ZERO": 1,
            },
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        material_light_subs = {
            **_SHADER_SUBS,
            "MAX_RULES": MAX_RULES + 1,
            "RULE_I_CAPACITY": MAX_RULES + 1,
        }
        self.programs["material_light"] = build_compute_shader(
            ctx,
            "reactions/material_light.comp",
            material_light_subs,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_light_authoritative_lhs"] = build_compute_shader(
            ctx,
            "reactions/material_light.comp",
            material_light_subs,
            includes=[
                "reactions/_common.comp",
                "reactions/_lhs_candidate.comp",
                "reactions/_authoritative_lhs_candidate.comp",
            ],
        )
        self.programs["gas_gas"] = build_compute_shader(
            ctx,
            "reactions/gas_gas.comp",
            flow_source_subs,
        )
        self.programs["gas_light"] = build_compute_shader(
            ctx,
            "reactions/gas_light.comp",
            flow_source_subs,
        )

    # ------------------------------------------------------------------
    # Satellite method delegates (W3: retired the `_x = _x` class grafts).
    #
    # Each body resolves the bare function name through this module's global
    # namespace -- method bodies never see class scope -- i.e. the satellite
    # function imported at the top of this file, bound at import time exactly
    # like the historical grafts.  Monkeypatch semantics are unchanged:
    # patching the attribute on the class or on an instance shadows/replaces
    # the delegate, while patching the satellite module's attribute does NOT
    # affect calls through the pipeline.
    # ------------------------------------------------------------------

    def _record_profile_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _record_profile_pass(self, *args, **kwargs)

    def _upload_state_profile_scope(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_state_profile_scope(self, *args, **kwargs)

    def _profile_scoped_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _profile_scoped_pass(self, *args, **kwargs)

    def _ensure_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_resources(self, *args, **kwargs)

    def run_timed_actions(self, *args: Any, **kwargs: Any) -> Any:
        return run_timed_actions(self, *args, **kwargs)

    def run_timed_triggers(self, *args: Any, **kwargs: Any) -> Any:
        return run_timed_triggers(self, *args, **kwargs)

    def run_self_triggers(self, *args: Any, **kwargs: Any) -> Any:
        return run_self_triggers(self, *args, **kwargs)

    def run_self_actions(self, *args: Any, **kwargs: Any) -> Any:
        return run_self_actions(self, *args, **kwargs)

    def run_material_material(self, *args: Any, **kwargs: Any) -> Any:
        return run_material_material(self, *args, **kwargs)

    def run_material_gas(self, *args: Any, **kwargs: Any) -> Any:
        return run_material_gas(self, *args, **kwargs)

    def run_material_pair_fused(self, *args: Any, **kwargs: Any) -> Any:
        return run_material_pair_fused(self, *args, **kwargs)

    def _compile_material_pair_plan(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_pair_plan(self, *args, **kwargs)

    def _compile_material_pair_plan_cached(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_pair_plan_cached(self, *args, **kwargs)

    def _material_pair_plan_cache_key(self, *args: Any, **kwargs: Any) -> Any:
        return _material_pair_plan_cache_key(self, *args, **kwargs)

    def run_material_light(self, *args: Any, **kwargs: Any) -> Any:
        return run_material_light(self, *args, **kwargs)

    def _run_formal_guarded_material_light(self, *args: Any, **kwargs: Any) -> Any:
        return _run_formal_guarded_material_light(self, *args, **kwargs)

    def run_gas_gas(self, *args: Any, **kwargs: Any) -> Any:
        return run_gas_gas(self, *args, **kwargs)

    def run_gas_light(self, *args: Any, **kwargs: Any) -> Any:
        return run_gas_light(self, *args, **kwargs)

    def _run_formal_guarded_gas_light(self, *args: Any, **kwargs: Any) -> Any:
        return _run_formal_guarded_gas_light(self, *args, **kwargs)

    def clear_reaction_latches(self, *args: Any, **kwargs: Any) -> Any:
        return clear_reaction_latches(self, *args, **kwargs)

    def _active_scheduler_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return _active_scheduler_gpu_authoritative(self, *args, **kwargs)

    def _formal_light_dose_guard_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_light_dose_guard_buffer(self, *args, **kwargs)

    def _build_light_dose_guarded_dispatch_args(self, *args: Any, **kwargs: Any) -> Any:
        return _build_light_dose_guarded_dispatch_args(self, *args, **kwargs)

    def _run_light_dose_guarded_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return _run_light_dose_guarded_dispatch(self, *args, **kwargs)

    def _active_masks_for_cell_reaction_upload(self, *args: Any, **kwargs: Any) -> Any:
        return _active_masks_for_cell_reaction_upload(self, *args, **kwargs)

    def _reaction_state_segment(self, *args: Any, **kwargs: Any) -> Any:
        return _reaction_state_segment(self, *args, **kwargs)

    def _bridge_cell_core_read_role_only_load(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_cell_core_read_role_only_load(self, *args, **kwargs)

    def _formal_reaction_segment_base_key(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_reaction_segment_base_key(self, *args, **kwargs)

    def _formal_reaction_segment_cache_key(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_reaction_segment_cache_key(self, *args, **kwargs)

    def _formal_reaction_state_cache_key(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_reaction_state_cache_key(self, *args, **kwargs)

    def _formal_reaction_active_mask_cache_key(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_reaction_active_mask_cache_key(self, *args, **kwargs)

    def _can_use_expanded_active_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _can_use_expanded_active_tile_mask(self, *args, **kwargs)

    def _formal_reaction_state_cache_active(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_reaction_state_cache_active(self, *args, **kwargs)

    def _formal_segment_batch_active(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_segment_batch_active(self, *args, **kwargs)

    def _formal_terminal_gas_publish_fusion_pending(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_terminal_gas_publish_fusion_pending(self, *args, **kwargs)

    def _formal_state_key_is_before_motion(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_state_key_is_before_motion(self, *args, **kwargs)

    def _formal_before_motion_cell_roles_active(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_before_motion_cell_roles_active(self, *args, **kwargs)

    def _formal_cell_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_cell_read_role(self, *args, **kwargs)

    def _formal_cell_write_role(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_cell_write_role(self, *args, **kwargs)

    def _set_formal_cell_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _set_formal_cell_read_role(self, *args, **kwargs)

    def _advance_formal_cell_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _advance_formal_cell_read_role(self, *args, **kwargs)

    def _reset_formal_cell_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _reset_formal_cell_read_role(self, *args, **kwargs)

    def _formal_velocity_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_velocity_read_role(self, *args, **kwargs)

    def _formal_velocity_write_role(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_velocity_write_role(self, *args, **kwargs)

    def _set_formal_velocity_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _set_formal_velocity_read_role(self, *args, **kwargs)

    def _advance_formal_velocity_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _advance_formal_velocity_read_role(self, *args, **kwargs)

    def _reset_formal_velocity_read_role(self, *args: Any, **kwargs: Any) -> Any:
        return _reset_formal_velocity_read_role(self, *args, **kwargs)

    def _clear_formal_external_cell_state(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_formal_external_cell_state(self, *args, **kwargs)

    def _cell_role_textures(self, *args: Any, **kwargs: Any) -> Any:
        return _cell_role_textures(self, *args, **kwargs)

    def _current_cell_textures(self, *args: Any, **kwargs: Any) -> Any:
        return _current_cell_textures(self, *args, **kwargs)

    def _next_cell_textures(self, *args: Any, **kwargs: Any) -> Any:
        return _next_cell_textures(self, *args, **kwargs)

    def begin_formal_reaction_segment(self, *args: Any, **kwargs: Any) -> Any:
        return begin_formal_reaction_segment(self, *args, **kwargs)

    def end_formal_reaction_segment(self, *args: Any, **kwargs: Any) -> Any:
        return end_formal_reaction_segment(self, *args, **kwargs)

    def _mark_formal_bridge_publish_pending(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_formal_bridge_publish_pending(self, *args, **kwargs)

    def flush_formal_reaction_segment(self, *args: Any, **kwargs: Any) -> Any:
        return flush_formal_reaction_segment(self, *args, **kwargs)

    def _clear_formal_segment_gas_delta(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_formal_segment_gas_delta(self, *args, **kwargs)

    def _flush_formal_segment_gas_delta(self, *args: Any, **kwargs: Any) -> Any:
        return _flush_formal_segment_gas_delta(self, *args, **kwargs)

    def _clear_reaction_latches_on_bridge(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_reaction_latches_on_bridge(self, *args, **kwargs)

    def _upload_active_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_active_masks(self, *args, **kwargs)

    def _load_authoritative_active_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_active_masks(self, *args, **kwargs)

    def _run_cell_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_cell_pass(self, *args, **kwargs)

    def _run_local_cell_action_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_local_cell_action_pass(self, *args, **kwargs)

    def _run_timed_candidate_action_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_timed_candidate_action_pass(self, *args, **kwargs)

    def _prepare_timed_candidate_worklist(self, *args: Any, **kwargs: Any) -> Any:
        return _prepare_timed_candidate_worklist(self, *args, **kwargs)

    def _prepare_self_candidate_worklist(self, *args: Any, **kwargs: Any) -> Any:
        return _prepare_self_candidate_worklist(self, *args, **kwargs)

    def _clear_timed_candidate_local_meta(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_timed_candidate_local_meta(self, *args, **kwargs)

    def _publish_timed_candidate_cell_state(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_timed_candidate_cell_state(self, *args, **kwargs)

    def _accumulate_timed_candidate_segment_cell_transient_state(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return _accumulate_timed_candidate_segment_cell_transient_state(self, *args, **kwargs)

    def _bind_local_cell_action_output_images(self, *args: Any, **kwargs: Any) -> Any:
        return _bind_local_cell_action_output_images(self, *args, **kwargs)

    def _scatter_local_cell_action_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _scatter_local_cell_action_outputs(self, *args, **kwargs)

    def _copy_current_velocity_to_next_role(self, *args: Any, **kwargs: Any) -> Any:
        return _copy_current_velocity_to_next_role(self, *args, **kwargs)

    def _scatter_local_emit_cell_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _scatter_local_emit_cell_outputs(self, *args, **kwargs)

    def _run_cell_gas_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_cell_gas_side_effect_pass(self, *args, **kwargs)

    def _run_cell_gas_action_delta_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_cell_gas_action_delta_pass(self, *args, **kwargs)

    def _run_self_candidate_gas_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_self_candidate_gas_side_effect_pass(self, *args, **kwargs)

    def _run_timed_candidate_gas_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_timed_candidate_gas_side_effect_pass(self, *args, **kwargs)

    def _run_material_light_dose_consume_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_material_light_dose_consume_pass(self, *args, **kwargs)

    def _run_cell_material_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_cell_material_side_effect_pass(self, *args, **kwargs)

    def _run_timed_candidate_material_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_timed_candidate_material_side_effect_pass(self, *args, **kwargs)

    def _clear_packed_timed_material_target_worklist(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_packed_timed_material_target_worklist(self, *args, **kwargs)

    def _run_packed_timed_material_side_effect_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_packed_timed_material_side_effect_pass(self, *args, **kwargs)

    def _run_produced_packed_timed_material_side_effect_pass(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return _run_produced_packed_timed_material_side_effect_pass(self, *args, **kwargs)

    def _run_packed_material_target_apply_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_packed_material_target_apply_pass(self, *args, **kwargs)

    def _run_timed_self_combined_action_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_timed_self_combined_action_pass(self, *args, **kwargs)

    def _load_authoritative_bridge_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_bridge_inputs(self, *args, **kwargs)

    def _publish_bridge_cell_state(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_cell_state(self, *args, **kwargs)

    def _publish_bridge_gas_state(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_gas_state(self, *args, **kwargs)

    def _publish_bridge_dose_state(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_dose_state(self, *args, **kwargs)

    def _apply_flow_sources_to_bridge_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_flow_sources_to_bridge_velocity(self, *args, **kwargs)

    def _publish_bridge_light_emitters(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_light_emitters(self, *args, **kwargs)

    def _download_cell_state(self, *args: Any, **kwargs: Any) -> Any:
        return _download_cell_state(self, *args, **kwargs)

    def _download_gas_state(self, *args: Any, **kwargs: Any) -> Any:
        return _download_gas_state(self, *args, **kwargs)

    def _download_dose_state(self, *args: Any, **kwargs: Any) -> Any:
        return _download_dose_state(self, *args, **kwargs)

    def _download_deferred_batch(self, *args: Any, **kwargs: Any) -> Any:
        return _download_deferred_batch(self, *args, **kwargs)

    def _unsupported_deferred_action_indices(self, *args: Any, **kwargs: Any) -> Any:
        return _unsupported_deferred_action_indices(self, *args, **kwargs)

    def _append_flow_sources_from_gpu(self, *args: Any, **kwargs: Any) -> Any:
        return _append_flow_sources_from_gpu(self, *args, **kwargs)

    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def _upload_state(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_state(self, *args, **kwargs)

    def _bridge_input_load_requirements(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_input_load_requirements(self, *args, **kwargs)

    def _missing_formal_bridge_input_loads(self, *args: Any, **kwargs: Any) -> Any:
        return _missing_formal_bridge_input_loads(self, *args, **kwargs)

    def _record_formal_bridge_inputs_loaded(self, *args: Any, **kwargs: Any) -> Any:
        return _record_formal_bridge_inputs_loaded(self, *args, **kwargs)

    def _transient_clear_requirements(self, *args: Any, **kwargs: Any) -> Any:
        return _transient_clear_requirements(self, *args, **kwargs)

    def _upload_random_targets(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_random_targets(self, *args, **kwargs)

    def _clear_transient_state(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_transient_state(self, *args, **kwargs)

    def _flow_source_generation_validity_active(self, *args: Any, **kwargs: Any) -> Any:
        return _flow_source_generation_validity_active(self, *args, **kwargs)

    def _advance_flow_source_generation(self, *args: Any, **kwargs: Any) -> Any:
        return _advance_flow_source_generation(self, *args, **kwargs)

    def _bind_flow_source_generation_output(self, *args: Any, **kwargs: Any) -> Any:
        return _bind_flow_source_generation_output(self, *args, **kwargs)

    def _clear_segment_transient_state(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_segment_transient_state(self, *args, **kwargs)

    def _begin_formal_segment_meta_lazy_zero(self, *args: Any, **kwargs: Any) -> Any:
        return _begin_formal_segment_meta_lazy_zero(self, *args, **kwargs)

    def _reset_formal_segment_meta_lazy_zero(self, *args: Any, **kwargs: Any) -> Any:
        return _reset_formal_segment_meta_lazy_zero(self, *args, **kwargs)

    def _record_formal_segment_cell_meta_in_flags(self, *args: Any, **kwargs: Any) -> Any:
        return _record_formal_segment_cell_meta_in_flags(self, *args, **kwargs)

    def _ensure_formal_segment_meta_physical_zero(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_formal_segment_meta_physical_zero(self, *args, **kwargs)

    def _can_use_terminal_segment_meta_zero(self, *args: Any, **kwargs: Any) -> Any:
        return _can_use_terminal_segment_meta_zero(self, *args, **kwargs)

    def _accumulate_segment_cell_transient_state(self, *args: Any, **kwargs: Any) -> Any:
        return _accumulate_segment_cell_transient_state(self, *args, **kwargs)

    def _upload_local_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_local_metadata(self, *args, **kwargs)

    def _promote_cell_pong_to_ping(self, *args: Any, **kwargs: Any) -> Any:
        return _promote_cell_pong_to_ping(self, *args, **kwargs)

    def _copy_gas_state(self, *args: Any, **kwargs: Any) -> Any:
        return _copy_gas_state(self, *args, **kwargs)

    def _promote_gas_pong_to_ping(self, *args: Any, **kwargs: Any) -> Any:
        return _promote_gas_pong_to_ping(self, *args, **kwargs)

    def _promote_gas_result(self, *args: Any, **kwargs: Any) -> Any:
        return _promote_gas_result(self, *args, **kwargs)

    def _promote_dose_pong_to_ping(self, *args: Any, **kwargs: Any) -> Any:
        return _promote_dose_pong_to_ping(self, *args, **kwargs)

    def _copy_bridge_flow_velocity_to_reaction(self, *args: Any, **kwargs: Any) -> Any:
        return _copy_bridge_flow_velocity_to_reaction(self, *args, **kwargs)

    def _sync_storage_and_indirect_writes(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_storage_and_indirect_writes(self, *args, **kwargs)

    def _sync_compute_writes(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_compute_writes(self, *args, **kwargs)

    def _compile_action_buffers(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_action_buffers(self, *args, **kwargs)

    def _compile_action_buffers_cached(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_action_buffers_cached(self, *args, **kwargs)

    def _compiled_actions_include_modify_gas(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_actions_include_modify_gas(self, *args, **kwargs)

    def _compiled_actions_include_flow_sources(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_actions_include_flow_sources(self, *args, **kwargs)

    @staticmethod
    def _compiled_self_rule_flow_source_layers(*args: Any, **kwargs: Any) -> Any:
        return _compiled_self_rule_flow_source_layers(*args, **kwargs)

    @staticmethod
    def _compiled_modify_gas_layer_mask(*args: Any, **kwargs: Any) -> Any:
        return _compiled_modify_gas_layer_mask(*args, **kwargs)

    def _compiled_actions_include_emit_material(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_actions_include_emit_material(self, *args, **kwargs)

    def _compiled_actions_require_deferred_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_actions_require_deferred_outputs(self, *args, **kwargs)

    def _self_rules_require_deferred_hi_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _self_rules_require_deferred_hi_outputs(self, *args, **kwargs)

    def _compiled_actions_may_change_structure(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_actions_may_change_structure(self, *args, **kwargs)

    def _compiled_rules_include_rhs_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_rules_include_rhs_consume(self, *args, **kwargs)

    def _compile_gas_action_buffers(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_gas_action_buffers(self, *args, **kwargs)

    def _compile_gas_light_action_buffers(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_gas_light_action_buffers(self, *args, **kwargs)

    @staticmethod
    def _modify_gas_action_requires_cpu_flow_side_effect(*args: Any, **kwargs: Any) -> Any:
        return _modify_gas_action_requires_cpu_flow_side_effect(*args, **kwargs)

    @staticmethod
    def _rule_candidate_word_count(*args: Any, **kwargs: Any) -> Any:
        return _rule_candidate_word_count(*args, **kwargs)

    @staticmethod
    def _empty_rule_candidate_masks(*args: Any, **kwargs: Any) -> Any:
        return _empty_rule_candidate_masks(*args, **kwargs)

    @staticmethod
    def _set_rule_candidate(*args: Any, **kwargs: Any) -> Any:
        return _set_rule_candidate(*args, **kwargs)

    def _compile_material_rule_candidate_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_rule_candidate_masks(self, *args, **kwargs)

    def _compile_material_material_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_material_rules(self, *args, **kwargs)

    def _compile_material_gas_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_gas_rules(self, *args, **kwargs)

    def _compile_material_light_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_light_rules(self, *args, **kwargs)

    @staticmethod
    def _compile_material_light_packed_descriptors(*args: Any, **kwargs: Any) -> Any:
        return _compile_material_light_packed_descriptors(*args, **kwargs)

    def _compile_material_light_packed_descriptors_cached(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_light_packed_descriptors_cached(self, *args, **kwargs)

    @staticmethod
    def _compile_material_pair_packed_descriptors(*args: Any, **kwargs: Any) -> Any:
        return _compile_material_pair_packed_descriptors(*args, **kwargs)

    def _compile_material_pair_packed_descriptors_cached(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_material_pair_packed_descriptors_cached(self, *args, **kwargs)

    def _compile_gas_gas_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_gas_gas_rules(self, *args, **kwargs)

    def _compile_single_gas_gas_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_single_gas_gas_rule(self, *args, **kwargs)

    def _compile_gas_light_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_gas_light_rules(self, *args, **kwargs)

    def _compile_single_gas_light_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _compile_single_gas_light_rule(self, *args, **kwargs)

    def _used_action_indices(self, *args: Any, **kwargs: Any) -> Any:
        return _used_action_indices(self, *args, **kwargs)

    def _used_action_indices_for_material_slots(self, *args: Any, **kwargs: Any) -> Any:
        return _used_action_indices_for_material_slots(self, *args, **kwargs)

    def _cached_used_action_indices_for_material_slots(self, *args: Any, **kwargs: Any) -> Any:
        return _cached_used_action_indices_for_material_slots(self, *args, **kwargs)

    def _used_action_indices_for_self_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _used_action_indices_for_self_rules(self, *args, **kwargs)

    def _cached_used_action_indices_for_self_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _cached_used_action_indices_for_self_rules(self, *args, **kwargs)

    def _used_action_indices_for_pair_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _used_action_indices_for_pair_rules(self, *args, **kwargs)

    def _cached_used_action_indices_for_pair_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _cached_used_action_indices_for_pair_rules(self, *args, **kwargs)

    @staticmethod
    def _has_unsupported_consume_policies(*args: Any, **kwargs: Any) -> Any:
        return _has_unsupported_consume_policies(*args, **kwargs)

    def _run_material_pair_fused_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_material_pair_fused_pass(self, *args, **kwargs)
