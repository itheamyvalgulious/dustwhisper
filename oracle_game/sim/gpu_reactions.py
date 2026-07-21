from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from oracle_game.gpu import CONSUME_POLICY_IDS, DIRECTION_IDS, typed_material_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader, shader_source
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
from oracle_game.types import ForceSource
from oracle_game.types import CellFlag, Phase, ReactionType


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_ACTIONS = 128
MAX_RULES = 256
RULE_CANDIDATE_WORDS = (MAX_RULES + 31) // 32
RULE_CANDIDATE_VECS = (RULE_CANDIDATE_WORDS + 3) // 4
MAX_SELF_RULES = 256
MAX_MATERIAL_LIGHT_PACKED_RULES = 8
MAX_MATERIAL_PAIR_PACKED_RULES = 8
MATERIAL_LIGHT_PACKED_HEADER_OFFSET = MAX_RULES * 2 + 1
MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET = MATERIAL_LIGHT_PACKED_HEADER_OFFSET + MAX_MATERIALS
MATERIAL_PAIR_PACKED_HEADER_OFFSET = MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET = MATERIAL_PAIR_PACKED_HEADER_OFFSET + MAX_MATERIALS
MATERIAL_PAIR_RULE_I_ENTRY_COUNT = MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
FLOW_SOURCE_LAYERS = 32
FLOW_SOURCE_GENERATION_BINDING = 7
# The self shader never reads gas_tags. Fused variants reuse its binding 7 for
# gas deltas while retaining the light-emitter SSBO at binding 14.
SELF_FUSED_GAS_DELTA_BINDING = 7
SELF_FUSED_FLOW_SOURCE_BINDING = 6
SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING = 1
MAX_EMITTED_LIGHTS = 256
GAS_DELTA_FIXED_SCALE = 1000000
LIGHT_DOSE_GUARD_BUFFER = "optics_light_dose_guard"
LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING = 12
LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING = 13

TYPE_NONE = 0
TYPE_HARM = 1
TYPE_MODIFY_TEMPERATURE = 2
TYPE_CONVERT_MATERIAL = 3
TYPE_DEFERRED = 4
TYPE_MODIFY_GAS = 5
TYPE_EMIT_LIGHT = 6
TYPE_EMIT_MATERIAL = 7
ACTION_FLAG_RANDOM_TARGET = 1
ACTION_FLAG_ALLOW_SUBUNIT_SCALE = 2
CONSUME_POLICY_NONE = int(CONSUME_POLICY_IDS["none"])
CONSUME_POLICY_LHS = int(CONSUME_POLICY_IDS["lhs"])
CONSUME_POLICY_RHS = int(CONSUME_POLICY_IDS["rhs"])
CONSUME_POLICY_BOTH = int(CONSUME_POLICY_IDS["both"])
DIRECT_CORE_OUTPUT_REACTION_GROUPS = frozenset(
    (
        "timed",
        "self",
        "material_material",
        "material_gas",
        "material_pair_fused",
        "material_light",
    )
)

# Superset of every {{NAME}} marker referenced by any reaction shader; the
# loader ignores unused keys, so one shared dict suffices for all passes.
_SHADER_SUBS = {
    "ACTION_FLAG_ALLOW_SUBUNIT_SCALE": ACTION_FLAG_ALLOW_SUBUNIT_SCALE,
    "CONSUME_POLICY_BOTH": CONSUME_POLICY_BOTH,
    "CONSUME_POLICY_LHS": CONSUME_POLICY_LHS,
    "CONSUME_POLICY_RHS": CONSUME_POLICY_RHS,
    "DIRECTION_ALL": DIRECTION_IDS["all"],
    "DIRECTION_DOWN": DIRECTION_IDS["down"],
    "DIRECTION_LEFT": DIRECTION_IDS["left"],
    "DIRECTION_RANDOM": DIRECTION_IDS["random"],
    "DIRECTION_RIGHT": DIRECTION_IDS["right"],
    "DIRECTION_SPEED": DIRECTION_IDS["speed"],
    "DIRECTION_UP": DIRECTION_IDS["up"],
    "ENABLE_LIGHT_EMITTER_OUTPUT": 1,
    "MATERIAL_PAIR_TERMINAL_HANDOFF": 0,
    "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 0,
    "MATERIAL_PAIR_TERMINAL_SEGMENT_META_ZERO": 0,
    "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 0,
    "PACK_CELL_META_IN_STATE": 0,
    "SELF_CACHE_CELL_STATE": 0,
    "SELF_RULE_DIRECT_ACTION_SPANS": 0,
    "SELF_SPARSE_INPLACE": 0,
    "DIRECT_GAS_DELTA_BINDING": 13,
    "REACTION_COUNTER_BINDING": 15,
    "BRIDGE_CELL_DOSE_BINDING": 14,
    "CLEAR_LIGHT_COUNTERS": 0,
    "FLOW_SOURCE_LAYERS": FLOW_SOURCE_LAYERS,
    "FLOW_SOURCE_GENERATION_VALIDITY": 0,
    "FLOW_SOURCE_GENERATION_BINDING": FLOW_SOURCE_GENERATION_BINDING,
    "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": "r32ui",
    "GAS_DELTA_FIXED_SCALE": GAS_DELTA_FIXED_SCALE,
    "LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING": LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING,
    "LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING": LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING,
    "LOCAL_SIZE": LOCAL_SIZE,
    "LOCAL_SIZE_X": LOCAL_SIZE,
    "LOCAL_SIZE_Y": LOCAL_SIZE,
    "MAX_ACTIONS": MAX_ACTIONS,
    "MAX_EMITTED_LIGHTS": MAX_EMITTED_LIGHTS,
    "MAX_EMITTED_LIGHTS_TIMES_2": MAX_EMITTED_LIGHTS * 2,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
    "MAX_MATERIAL_LIGHT_PACKED_RULES": MAX_MATERIAL_LIGHT_PACKED_RULES,
    "MAX_MATERIAL_PAIR_PACKED_RULES": MAX_MATERIAL_PAIR_PACKED_RULES,
    "MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET": MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET,
    "MATERIAL_LIGHT_PACKED_HEADER_OFFSET": MATERIAL_LIGHT_PACKED_HEADER_OFFSET,
    "MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET": MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET,
    "MATERIAL_PAIR_PACKED_HEADER_OFFSET": MATERIAL_PAIR_PACKED_HEADER_OFFSET,
    "MAX_MATERIALS_TIMES_RULE_CANDIDATE_VECS": MAX_MATERIALS * RULE_CANDIDATE_VECS,
    "MAX_RULES": MAX_RULES,
    "RULE_I_CAPACITY": MAX_RULES,
    "MAX_SELF_RULES": MAX_SELF_RULES,
    "PHASE_POWDER": int(Phase.POWDER),
    "REACTION_LATCHED_FLAG": int(CellFlag.REACTION_LATCHED),
    "REACTION_LATCHED_FLAG_SHIFTED_24": int(CellFlag.REACTION_LATCHED) << 24,
    "RULE_CANDIDATE_VECS": RULE_CANDIDATE_VECS,
    "RULE_CANDIDATE_WORDS": RULE_CANDIDATE_WORDS,
    "SELF_FUSED_FLOW_SOURCE_BINDING": SELF_FUSED_FLOW_SOURCE_BINDING,
    "SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING": SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING,
    "SELF_FUSED_GAS_DELTA_BINDING": SELF_FUSED_GAS_DELTA_BINDING,
    "SELF_FUSED_GAS_OUTPUT": 0,
    "TIMED_EMIT_TARGET_PRODUCER": 0,
    "TIMED_SPARSE_INPLACE": 0,
    "TYPE_CONVERT_MATERIAL": TYPE_CONVERT_MATERIAL,
    "TYPE_DEFERRED": TYPE_DEFERRED,
    "TYPE_EMIT_LIGHT": TYPE_EMIT_LIGHT,
    "TYPE_EMIT_MATERIAL": TYPE_EMIT_MATERIAL,
    "TYPE_HARM": TYPE_HARM,
    "TYPE_MODIFY_GAS": TYPE_MODIFY_GAS,
    "TYPE_MODIFY_TEMPERATURE": TYPE_MODIFY_TEMPERATURE,
}


@dataclass(slots=True)
class GPUDeferredActionBatch:
    action_lo: np.ndarray
    action_hi: np.ndarray
    scale_lo: np.ndarray
    scale_hi: np.ndarray
    emitted_lights: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    emitted_material_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.bool_))
    gpu_local_action_counts: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint32))
    formal_gpu_empty: bool = False


FORMAL_GPU_EMPTY_DEFERRED_BATCH = GPUDeferredActionBatch(
    action_lo=np.zeros((0, 0, 4), dtype=np.int32),
    action_hi=np.zeros((0, 0, 4), dtype=np.int32),
    scale_lo=np.zeros((0, 0, 4), dtype=np.float32),
    scale_hi=np.zeros((0, 0, 4), dtype=np.float32),
    emitted_lights=np.zeros((0, 8), dtype=np.float32),
    emitted_material_mask=np.zeros((0, 0), dtype=np.bool_),
    gpu_local_action_counts=np.zeros((8,), dtype=np.uint32),
    formal_gpu_empty=True,
)


@dataclass(frozen=True, slots=True)
class GPUReactionBridgeInputLoads:
    cell_core: bool = True
    gas: bool = True
    ambient: bool = True
    flow_velocity: bool = True
    cell_dose: bool = True
    gas_dose: bool = True

    def any(self) -> bool:
        return any(
            (
                self.cell_core,
                self.gas,
                self.ambient,
                self.flow_velocity,
                self.cell_dose,
                self.gas_dose,
            )
        )

    def resource_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.cell_core:
            names.append("cell_core")
        if self.gas:
            names.append("gas_concentration")
        if self.ambient:
            names.append("ambient_temperature")
        if self.flow_velocity:
            names.append("flow_velocity")
        if self.cell_dose:
            names.append("cell_optical_dose")
        if self.gas_dose:
            names.append("gas_optical_dose")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class GPUReactionMaterialPairPlan:
    cache_key: tuple[object, ...]
    compiled_actions: tuple[np.ndarray, np.ndarray]
    packed_rule_i: np.ndarray
    packed_rule_f: np.ndarray
    packed_rule_tags: np.ndarray
    packed_lhs_candidate_masks: np.ndarray
    material_material_rule_count: int
    rule_count: int
    material_light_rule_count: int
    material_light_packed_descriptors: np.ndarray | None
    material_pair_packed_descriptors: np.ndarray | None
    modifies_gas: bool
    direct_modify_gas_layer_mask: int


@dataclass(slots=True)
class GPUReactionResources:
    signature: tuple[int, int, int, int, int, int]
    cell_state_ping: Any
    cell_state_pong: Any
    temp_ping: Any
    temp_pong: Any
    integrity_ping: Any
    integrity_pong: Any
    velocity_ping: Any
    velocity_pong: Any
    timer_ping: Any
    timer_pong: Any
    ambient_ping: Any
    ambient_pong: Any
    gas_ping: Any
    gas_pong: Any
    flow_velocity_tex: Any
    active_cell_tex: Any
    expanded_active_tile_tex: Any
    active_gas_tex: Any
    cell_dose_tex: Any
    cell_dose_pong: Any
    gas_dose_tex: Any
    gas_dose_pong: Any
    flow_source_tex: Any
    flow_source_generation_tex: Any
    gas_delta_buffer: Any
    timed_candidate_count: Any
    timed_candidate_list: Any
    timed_candidate_dispatch_args: Any
    light_dose_guarded_dispatch_args: Any
    timed_candidate_marks: Any
    timed_material_target_list: Any
    timed_material_target_dispatch_args: Any
    timed_material_target_marks: Any
    trigger_lo_tex: Any
    trigger_hi_tex: Any
    deferred_scale_lo_tex: Any
    deferred_scale_hi_tex: Any
    cell_reset_tex: Any
    reaction_latched_tex: Any
    segment_cell_meta_tex: Any
    emitted_material_mask_tex: Any
    local_cell_state_out: Any
    handoff_material_tex: Any
    handoff_phase_tex: Any
    handoff_flags_tex: Any
    local_temp_out: Any
    local_integrity_out: Any
    local_timer_out: Any
    local_deferred_lo_out: Any
    local_deferred_hi_out: Any
    local_deferred_packed_out: Any
    local_cell_meta_out: Any
    local_emit_cell_lo_out: Any
    local_emit_cell_hi_out: Any
    material_params: Any
    material_tags: Any
    gas_tags: Any
    material_slots_lo: Any
    material_slots_hi: Any
    action_meta: Any
    light_emitter_buffer: Any
    light_emitter_count: Any
    random_targets: Any
    action_i: Any
    action_f: Any
    material_pair_action_i: Any
    material_pair_action_f: Any
    mm_rule_i: Any
    mm_rule_f: Any
    mm_rule_tags: Any
    mg_rule_i: Any
    mg_rule_f: Any
    mg_rule_tags: Any
    material_pair_rule_i: Any
    material_pair_rule_f: Any
    material_pair_rule_tags: Any
    material_pair_lhs_candidate_masks: Any
    material_pair_terminal_material_tables: Any
    material_pair_terminal_action_tables: Any
    material_pair_terminal_rule_tables: Any
    rule_lhs_candidate_masks: Any
    ml_rule_i: Any
    ml_rule_f: Any
    ml_rule_tags: Any
    gg_rule_i: Any
    gg_rule_f: Any
    gg_rule_tags: Any
    gl_rule_i: Any
    gl_rule_f: Any
    gl_rule_tags: Any
    self_rule_i: Any
    self_rule_f: Any
    self_rule_span_i: Any
    self_rule_span_direct_actions: bool = False
    flow_source_generation: int = 0
    material_params_signature: tuple[int, int] | None = None
    material_slots_signature: tuple[int, int] | None = None
    gas_tags_signature: tuple[int, int] | None = None
    action_meta_signature: tuple[int, int] | None = None
    self_rule_signature: tuple[int, int] | None = None
    random_targets_signature: tuple[int, int, int] | None = None
    material_pair_plan_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_material_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_action_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_rule_upload_key: tuple[object, ...] | None = None


from oracle_game.sim.gpu_reactions_resources import (
    _record_profile_pass,
    _upload_state_profile_scope,
    _profile_scoped_pass,
    _ensure_resources,
)

from oracle_game.sim.gpu_reactions_pairings import (
    run_timed_actions,
    run_timed_triggers,
    run_self_triggers,
    run_self_actions,
    run_material_material,
    run_material_gas,
    run_material_pair_fused,
    _compile_material_pair_plan,
    _compile_material_pair_plan_cached,
    _material_pair_plan_cache_key,
    run_material_light,
    _run_formal_guarded_material_light,
    run_gas_gas,
    run_gas_light,
    _run_formal_guarded_gas_light,
    clear_reaction_latches,
)

from oracle_game.sim.gpu_reactions_segments import (
    _active_scheduler_gpu_authoritative,
    _formal_light_dose_guard_buffer,
    _build_light_dose_guarded_dispatch_args,
    _run_light_dose_guarded_dispatch,
    _active_masks_for_cell_reaction_upload,
    _reaction_state_segment,
    _bridge_cell_core_read_role_only_load,
    _formal_reaction_segment_base_key,
    _formal_reaction_segment_cache_key,
    _formal_reaction_state_cache_key,
    _formal_reaction_active_mask_cache_key,
    _can_use_expanded_active_tile_mask,
    _formal_reaction_state_cache_active,
    _formal_segment_batch_active,
    _formal_terminal_gas_publish_fusion_pending,
    _formal_state_key_is_before_motion,
    _formal_before_motion_cell_roles_active,
    _formal_cell_read_role,
    _formal_cell_write_role,
    _set_formal_cell_read_role,
    _advance_formal_cell_read_role,
    _reset_formal_cell_read_role,
    _formal_velocity_read_role,
    _formal_velocity_write_role,
    _set_formal_velocity_read_role,
    _advance_formal_velocity_read_role,
    _reset_formal_velocity_read_role,
    _clear_formal_external_cell_state,
    _cell_role_textures,
    _current_cell_textures,
    _next_cell_textures,
    begin_formal_reaction_segment,
    end_formal_reaction_segment,
    _mark_formal_bridge_publish_pending,
    flush_formal_reaction_segment,
    _clear_formal_segment_gas_delta,
    _flush_formal_segment_gas_delta,
    _clear_reaction_latches_on_bridge,
    _upload_active_masks,
    _load_authoritative_active_masks,
)

from oracle_game.sim.gpu_reactions_cell_pass import (
    _run_cell_pass,
    _run_material_pair_fused_pass,
    _run_local_cell_action_pass,
    _run_timed_candidate_action_pass,
    _prepare_timed_candidate_worklist,
    _prepare_self_candidate_worklist,
    _clear_timed_candidate_local_meta,
    _publish_timed_candidate_cell_state,
    _accumulate_timed_candidate_segment_cell_transient_state,
    _bind_local_cell_action_output_images,
    _scatter_local_cell_action_outputs,
    _copy_current_velocity_to_next_role,
    _scatter_local_emit_cell_outputs,
)

from oracle_game.sim.gpu_reactions_side_effects import (
    _run_cell_gas_side_effect_pass,
    _run_cell_gas_action_delta_pass,
    _run_self_candidate_gas_side_effect_pass,
    _run_timed_candidate_gas_side_effect_pass,
    _run_material_light_dose_consume_pass,
    _run_cell_material_side_effect_pass,
    _run_timed_candidate_material_side_effect_pass,
    _clear_packed_timed_material_target_worklist,
    _run_packed_timed_material_side_effect_pass,
    _run_produced_packed_timed_material_side_effect_pass,
    _run_packed_material_target_apply_pass,
)
from oracle_game.sim.gpu_reactions_timed_self import _run_timed_self_combined_action_pass

from oracle_game.sim.gpu_reactions_bridge import (
    _load_authoritative_bridge_inputs,
    _publish_bridge_cell_state,
    _publish_bridge_gas_state,
    _publish_bridge_dose_state,
    _apply_flow_sources_to_bridge_velocity,
    _publish_bridge_light_emitters,
    _download_cell_state,
    _download_gas_state,
    _download_dose_state,
    _download_deferred_batch,
    _unsupported_deferred_action_indices,
    _append_flow_sources_from_gpu,
)

from oracle_game.sim.gpu_reactions_transient import (
    release,
    _upload_state,
    _bridge_input_load_requirements,
    _missing_formal_bridge_input_loads,
    _record_formal_bridge_inputs_loaded,
    _transient_clear_requirements,
    _upload_random_targets,
    _clear_transient_state,
    _flow_source_generation_validity_active,
    _advance_flow_source_generation,
    _bind_flow_source_generation_output,
    _clear_segment_transient_state,
    _begin_formal_segment_meta_lazy_zero,
    _reset_formal_segment_meta_lazy_zero,
    _record_formal_segment_cell_meta_in_flags,
    _ensure_formal_segment_meta_physical_zero,
    _can_use_terminal_segment_meta_zero,
    _accumulate_segment_cell_transient_state,
    _upload_local_metadata,
    _promote_cell_pong_to_ping,
    _copy_gas_state,
    _promote_gas_pong_to_ping,
    _promote_gas_result,
    _promote_dose_pong_to_ping,
    _copy_bridge_flow_velocity_to_reaction,
    _sync_storage_and_indirect_writes,
    _sync_compute_writes,
)

from oracle_game.sim.gpu_reactions_rules import (
    _compile_action_buffers,
    _compile_action_buffers_cached,
    _compiled_actions_include_modify_gas,
    _compiled_actions_include_flow_sources,
    _compiled_self_rule_flow_source_layers,
    _compiled_modify_gas_layer_mask,
    _compiled_actions_include_emit_material,
    _compiled_actions_include_emit_light,
    _compiled_actions_require_deferred_outputs,
    _self_rules_require_deferred_hi_outputs,
    _compiled_actions_may_change_structure,
    _compiled_rules_include_rhs_consume,
    _compile_gas_action_buffers,
    _compile_gas_light_action_buffers,
    _modify_gas_action_requires_cpu_flow_side_effect,
    _rule_candidate_word_count,
    _empty_rule_candidate_masks,
    _set_rule_candidate,
    _compile_material_rule_candidate_masks,
    _compile_material_material_rules,
    _compile_material_gas_rules,
    _compile_material_light_rules,
    _compile_material_light_packed_descriptors,
    _compile_material_light_packed_descriptors_cached,
    _compile_material_pair_packed_descriptors,
    _compile_material_pair_packed_descriptors_cached,
    _compile_gas_gas_rules,
    _compile_single_gas_gas_rule,
    _compile_gas_light_rules,
    _compile_single_gas_light_rule,
    _used_action_indices,
    _used_action_indices_for_material_slots,
    _cached_used_action_indices_for_material_slots,
    _used_action_indices_for_self_rules,
    _cached_used_action_indices_for_self_rules,
    _used_action_indices_for_pair_rules,
    _cached_used_action_indices_for_pair_rules,
    _has_unsupported_consume_policies,
)



class GPUReactionPipeline(GPUPipelineBase):
    def __init__(self) -> None:
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
        self._expanded_active_tile_mask_enabled = True
        self.last_expanded_active_tile_mask_used = False
        self.expanded_active_tile_mask_build_count = 0
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self.random_targets = np.zeros((MAX_MATERIALS,), dtype=np.int32)
        self.random_target_count = 0
        self._used_action_indices_cache: dict[tuple[object, ...], set[int] | None] = {}
        self._compiled_action_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray] | None] = {}
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
        self._terminal_gas_publish_fusion_enabled = True
        self.last_terminal_gas_publish_fusion_used = False
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
        self._pair_segment_meta_fusion_enabled = True
        # Candidate: timed/self can carry reset/latch operations in packed
        # cell flags. A proven terminal pass may then start its local material
        # metadata at zero and avoid initializing the full-grid segment image.
        self._terminal_segment_meta_lazy_zero_enabled = True
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
        self._segment_meta_light_counter_clear_fusion_enabled = True
        self._packed_timed_emit_target_worklist_enabled = True
        # Candidate: timed_apply can append its packed low-4 emit-material
        # targets while the action id and source cell are already resident.
        # Keep the established full-grid compactor as the default fallback.
        self._timed_emit_target_producer_enabled = True
        # Sparse in-place timed dispatch is exact but experimental; keep the
        # canonical full-grid path as the production default until frame-level
        # A/B confirms a stable win.
        self._timed_sparse_inplace_enabled = False
        self._timed_sparse_positive_count_enabled = False
        self.last_timed_sparse_inplace_used = False
        # Experimental: compact only cells whose material has at least one
        # self rule, then apply those outputs in the current role in-place.
        self._self_sparse_inplace_enabled = False
        self.last_self_sparse_inplace_used = False
        # Candidate: cache the packed input word once in dense self_apply and
        # reuse it for material, phase, and preserved output flags.
        self._self_apply_cached_cell_state_enabled = False
        self.last_self_apply_cached_cell_state_used = False
        self._packed_self_emit_target_worklist_enabled = True
        self._authoritative_lhs_candidate_masks_enabled = True
        self._self_gas_candidate_worklist_enabled = True
        # Experimental until whole-frame timings justify enabling it.
        self._self_apply_fused_gas_output_enabled = False
        # Formal batches invalidate flow sources by generation instead of
        # clearing the 32-layer payload texture before each pass.
        self._flow_source_generation_validity_enabled = True
        self._flow_source_generation_programs_enabled = False
        # Candidate: use an 8-bit generation token for the 32-layer validity
        # texture, clearing before token 1 is reused after 255.
        self._flow_source_generation_u8_token_enabled = True
        self._flow_source_generation_u8_programs_enabled = False
        # Experimental candidate: one cell dispatch evaluates timed actions
        # followed by self rules.  Unsafe side-effect classes fall back.
        self._timed_self_same_dispatch_enabled = False
        self._timed_self_same_dispatch_pending = False
        self.last_timed_self_same_dispatch_used = False
        # Experimental: timed/self already consume the GPU-authoritative TTL
        # mask directly.  Represent their CPU-side runtime bookkeeping with
        # the same non-materialized segment mask used by the fused pair pass,
        # rather than copying and OR-ing full-resolution bool grids twice.
        self._timed_self_authoritative_segment_masks_enabled = True
        # Experimental: apply timed/self reset and latch metadata directly to
        # the packed cell-state flags, avoiding a later full-grid accumulation.
        self._timed_self_cell_flag_meta_enabled = True
        self.last_timed_self_cell_flag_meta_used = False
        self._self_rule_material_spans_enabled = True
        # Experimental: explicit self-rule spans carry their resolved action
        # index, avoiding a per-rule material-slot SSBO lookup.  Wildcard or
        # malformed rules stay on the canonical path.
        self._self_rule_direct_action_spans_enabled = True
        self.last_self_rule_direct_action_spans_used = False
        self._material_pair_state_fusion_enabled = True
        self._material_pair_light_state_fusion_enabled = True
        # Publish the material/light terminal state directly to motion.  The
        # handoff is raw-byte exact and avoids a second full-grid integration.
        self._material_triplet_motion_terminal_enabled = True
        # The terminal shader has no shared tile state; 16x16 groups reduce
        # dispatch overhead while preserving per-cell evaluation.
        self._material_triplet_terminal_local16_enabled = True
        self._material_triplet_terminal_dirty_fast_equal_enabled = True
        # Experimental: stage terminal bridge words in shared memory and emit
        # row-contiguous stores.  Keep the canonical per-cell stores default.
        self._material_triplet_terminal_shared_transpose_enabled = True
        # Candidate: keep 256 invocations but make each NVIDIA warp cover one
        # complete row. The established 16x16 terminal remains the fallback.
        self._material_triplet_terminal_32x8_enabled = True
        self.last_material_pair_terminal_handoff = False
        self._material_triplet_ml_packed_descriptors_enabled = True
        self._material_pair_packed_descriptors_enabled = True
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
            "FLOW_SOURCE_GENERATION_VALIDITY": int(
                self._flow_source_generation_programs_enabled
            ),
            "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": (
                "r8ui"
                if self._flow_source_generation_u8_programs_enabled
                else "r32ui"
            ),
        }
        self.programs["load_active_cell"] = build_compute_shader(
            ctx, "reactions/load_active_cell.comp", _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_expanded_active_tiles"] = build_compute_shader(
            ctx,
            "reactions/load_expanded_active_tiles.comp",
            _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_active_gas"] = build_compute_shader(
            ctx, "reactions/load_active_gas.comp", _SHADER_SUBS,
            includes=["reactions/_active_helper.comp"],
        )
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "reactions/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_role"] = build_compute_shader(ctx, "reactions/load_bridge_cell_role.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux"] = build_compute_shader(ctx, "reactions/load_bridge_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux_role"] = build_compute_shader(ctx, "reactions/load_bridge_cell_aux_role.comp", _SHADER_SUBS)
        self.programs["load_bridge_gas"] = build_compute_shader(ctx, "reactions/load_bridge_gas.comp", _SHADER_SUBS)
        self.programs["load_bridge_dose"] = build_compute_shader(ctx, "reactions/load_bridge_dose.comp", _SHADER_SUBS)
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "reactions/publish_bridge_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_gas"] = build_compute_shader(ctx, "reactions/publish_bridge_gas.comp", _SHADER_SUBS)
        self.programs["publish_bridge_cell_dose"] = build_compute_shader(ctx, "reactions/publish_bridge_cell_dose.comp", _SHADER_SUBS)
        self.programs["publish_bridge_gas_dose"] = build_compute_shader(ctx, "reactions/publish_bridge_gas_dose.comp", _SHADER_SUBS)
        self.programs["apply_bridge_flow_sources"] = build_compute_shader(
            ctx,
            "reactions/apply_bridge_flow_sources.comp",
            flow_source_subs,
        )
        self.programs["promote_reaction_cell_state"] = build_compute_shader(ctx, "reactions/promote_reaction_cell_state.comp", _SHADER_SUBS)
        self.programs["copy_reaction_velocity_state"] = build_compute_shader(ctx, "reactions/copy_reaction_velocity_state.comp", _SHADER_SUBS)
        self.programs["promote_reaction_gas_state"] = build_compute_shader(ctx, "reactions/promote_reaction_gas_state.comp", _SHADER_SUBS)
        self.programs["promote_reaction_dose_state"] = build_compute_shader(ctx, "reactions/promote_reaction_dose_state.comp", _SHADER_SUBS)
        self.programs["copy_bridge_flow_velocity_to_reaction"] = build_compute_shader(ctx, "reactions/copy_bridge_flow_velocity_to_reaction.comp", _SHADER_SUBS)
        self.programs["publish_bridge_light_emitters"] = build_compute_shader(ctx, "reactions/publish_bridge_light_emitters.comp", _SHADER_SUBS)
        self.programs["timed_trigger"] = build_compute_shader(ctx, "reactions/timed_trigger.comp", _SHADER_SUBS)
        self.programs["self_trigger"] = build_compute_shader(ctx, "reactions/self_trigger.comp", _SHADER_SUBS)
        self.programs["timed_apply"] = build_compute_shader(
            ctx, "reactions/timed_apply.comp", _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["timed_apply_packed"] = build_compute_shader(
            ctx, "reactions/timed_apply.comp", _SHADER_SUBS,
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
            ctx, "reactions/timed_apply.comp", packed_cell_meta_subs,
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
        self.programs["clear_timed_candidate_worklist"] = build_compute_shader(ctx, "reactions/clear_timed_candidate_worklist.comp", _SHADER_SUBS)
        self.programs["clear_timed_material_target_worklist"] = build_compute_shader(
            ctx,
            "reactions/clear_timed_material_target_worklist.comp",
            _SHADER_SUBS,
        )
        self.programs["build_light_dose_guarded_dispatch_args"] = build_compute_shader(ctx, "reactions/build_light_dose_guarded_dispatch_args.comp", _SHADER_SUBS)
        self.programs["compact_timed_candidates"] = build_compute_shader(ctx, "reactions/compact_timed_candidates.comp", _SHADER_SUBS)
        self.programs["compact_self_candidates"] = build_compute_shader(
            ctx, "reactions/compact_self_candidates.comp", _SHADER_SUBS
        )
        self.programs["timed_apply_candidates"] = build_compute_shader(
            ctx, "reactions/timed_apply_candidates.comp", _SHADER_SUBS,
            includes=["reactions/_common_no_direct.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["self_apply"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", _SHADER_SUBS,
            includes=["reactions/_common_self_apply.comp", "reactions/_local_action_output.comp"],
        )
        self.programs["self_apply_packed"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", _SHADER_SUBS,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        cached_self_state_subs = {**_SHADER_SUBS, "SELF_CACHE_CELL_STATE": 1}
        self.programs["self_apply_packed_cached_cell_state"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", cached_self_state_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        self.programs["self_apply_packed_cell_flag_meta"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", packed_cell_meta_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        self.programs[
            "self_apply_packed_cell_flag_meta_cached_cell_state"
        ] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {
                **cached_self_state_subs,
                "PACK_CELL_META_IN_STATE": 1,
            },
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        sparse_self_subs = {**_SHADER_SUBS, "SELF_SPARSE_INPLACE": 1}
        self.programs["self_apply_packed_sparse"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", sparse_self_subs,
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
            ctx, "reactions/self_apply.comp", sparse_self_direct_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
        )
        self.programs["self_apply_packed_sparse_cell_flag_meta"] = build_compute_shader(
            ctx, "reactions/self_apply.comp",
            {**sparse_self_subs, "PACK_CELL_META_IN_STATE": 1},
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
        )
        self.programs["self_apply_packed_sparse_direct_spans_cell_flag_meta"] = build_compute_shader(
            ctx, "reactions/self_apply.comp",
            {**sparse_self_direct_subs, "PACK_CELL_META_IN_STATE": 1},
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output.comp",
                "reactions/_self_sparse_dispatch_io.comp",
            ],
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
        self.programs[
            "self_apply_packed_direct_spans_cached_cell_state"
        ] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {
                **direct_self_span_subs,
                "SELF_CACHE_CELL_STATE": 1,
            },
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
        self.programs[
            "self_apply_packed_direct_spans_cell_flag_meta_cached_cell_state"
        ] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {
                **direct_self_span_cell_meta_subs,
                "SELF_CACHE_CELL_STATE": 1,
            },
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
            ],
        )
        fused_self_subs = {**flow_source_subs, "SELF_FUSED_GAS_OUTPUT": 1}
        self.programs["self_apply_packed_fused_gas"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", fused_self_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
                "reactions/_self_fused_gas_output.comp",
            ],
        )
        self.programs[
            "self_apply_packed_fused_gas_cached_cell_state"
        ] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {
                **fused_self_subs,
                "SELF_CACHE_CELL_STATE": 1,
            },
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
                "reactions/_self_fused_gas_output.comp",
            ],
        )
        fused_self_cell_meta_subs = {
            **fused_self_subs,
            "PACK_CELL_META_IN_STATE": 1,
        }
        self.programs["self_apply_packed_fused_gas_cell_flag_meta"] = build_compute_shader(
            ctx, "reactions/self_apply.comp", fused_self_cell_meta_subs,
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
                "reactions/_self_fused_gas_output.comp",
            ],
        )
        self.programs[
            "self_apply_packed_fused_gas_cell_flag_meta_cached_cell_state"
        ] = build_compute_shader(
            ctx,
            "reactions/self_apply.comp",
            {
                **fused_self_cell_meta_subs,
                "SELF_CACHE_CELL_STATE": 1,
            },
            includes=[
                "reactions/_common_self_apply.comp",
                "reactions/_local_action_output_packed.comp",
                "reactions/_self_emit_target_output.comp",
                "reactions/_self_fused_gas_output.comp",
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
        self.programs["scatter_local_action_outputs"] = build_compute_shader(ctx, "reactions/scatter_local_action_outputs.comp", _SHADER_SUBS)
        self.programs["scatter_local_action_deferred_meta_outputs"] = build_compute_shader(ctx, "reactions/scatter_local_action_deferred_meta_outputs.comp", _SHADER_SUBS)
        self.programs["scatter_local_action_tail_outputs"] = build_compute_shader(ctx, "reactions/scatter_local_action_tail_outputs.comp", _SHADER_SUBS)
        self.programs["scatter_local_emit_cell_outputs"] = build_compute_shader(ctx, "reactions/scatter_local_emit_cell_outputs.comp", _SHADER_SUBS)
        self.programs["clear_transient_cell_state"] = build_compute_shader(ctx, "reactions/clear_transient_cell_state.comp", _SHADER_SUBS)
        self.programs["clear_transient_aux_state"] = build_compute_shader(ctx, "reactions/clear_transient_aux_state.comp", _SHADER_SUBS)
        self.programs["clear_transient_light_counters"] = build_compute_shader(ctx, "reactions/clear_transient_light_counters.comp", _SHADER_SUBS)
        self.programs["clear_timed_candidate_local_meta"] = build_compute_shader(ctx, "reactions/clear_timed_candidate_local_meta.comp", _SHADER_SUBS)
        self.programs["clear_transient_emit_material_mask"] = build_compute_shader(ctx, "reactions/clear_transient_emit_material_mask.comp", _SHADER_SUBS)
        self.programs["clear_transient_emit_material_buffers"] = build_compute_shader(ctx, "reactions/clear_transient_emit_material_buffers.comp", _SHADER_SUBS)
        self.programs["clear_transient_flow_sources"] = build_compute_shader(ctx, "reactions/clear_transient_flow_sources.comp", _SHADER_SUBS)
        self.programs["clear_transient_flow_source_generations"] = build_compute_shader(
            ctx,
            "reactions/clear_transient_flow_source_generations.comp",
            flow_source_subs,
        )
        self.programs["clear_segment_cell_transient_state"] = build_compute_shader(ctx, "reactions/clear_segment_cell_transient_state.comp", _SHADER_SUBS)
        self.programs["clear_segment_cell_transient_state_light_counters"] = build_compute_shader(
            ctx,
            "reactions/clear_segment_cell_transient_state.comp",
            {**_SHADER_SUBS, "CLEAR_LIGHT_COUNTERS": 1},
        )
        self.programs["accumulate_segment_cell_transient_state"] = build_compute_shader(ctx, "reactions/accumulate_segment_cell_transient_state.comp", _SHADER_SUBS)
        self.programs["accumulate_timed_candidate_segment_cell_transient_state"] = build_compute_shader(ctx, "reactions/accumulate_timed_candidate_segment_cell_transient_state.comp", _SHADER_SUBS)
        self.programs["cell_material_side_effects"] = build_compute_shader(ctx, "reactions/cell_material_side_effects.comp", _SHADER_SUBS)
        self.programs["compact_timed_material_targets"] = build_compute_shader(ctx, "reactions/compact_timed_material_targets.comp", _SHADER_SUBS)
        self.programs["cell_material_side_effects_candidates"] = build_compute_shader(ctx, "reactions/cell_material_side_effects_candidates.comp", _SHADER_SUBS)
        self.programs["compact_packed_timed_material_targets"] = build_compute_shader(ctx, "reactions/compact_packed_timed_material_targets.comp", _SHADER_SUBS)
        self.programs["cell_material_side_effects_packed_targets"] = build_compute_shader(ctx, "reactions/cell_material_side_effects_packed_targets.comp", _SHADER_SUBS)
        self.programs["build_packed_material_target_dispatch"] = build_compute_shader(ctx, "reactions/build_packed_material_target_dispatch.comp", _SHADER_SUBS)
        self.programs["clear_cell_gas_delta"] = build_compute_shader(ctx, "reactions/clear_cell_gas_delta.comp", _SHADER_SUBS)
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
        self.programs["apply_cell_gas_delta"] = build_compute_shader(ctx, "reactions/apply_cell_gas_delta.comp", _SHADER_SUBS)
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
        self.programs["material_light_cell_dose_consume"] = build_compute_shader(ctx, "reactions/material_light_cell_dose_consume.comp", _SHADER_SUBS)
        self.programs["material_light_gas_dose_consume"] = build_compute_shader(ctx, "reactions/material_light_gas_dose_consume.comp", _SHADER_SUBS)
        self.programs["material_material"] = build_compute_shader(
            ctx, "reactions/material_material.comp", _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_material_authoritative_lhs"] = build_compute_shader(
            ctx, "reactions/material_material.comp", _SHADER_SUBS,
            includes=[
                "reactions/_common.comp",
                "reactions/_lhs_candidate.comp",
                "reactions/_authoritative_lhs_candidate.comp",
            ],
        )
        self.programs["material_gas"] = build_compute_shader(
            ctx, "reactions/material_gas.comp", _SHADER_SUBS,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_gas_authoritative_lhs"] = build_compute_shader(
            ctx, "reactions/material_gas.comp", _SHADER_SUBS,
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
            "MAX_MATERIALS_TIMES_RULE_CANDIDATE_VECS": (
                MAX_MATERIALS * RULE_CANDIDATE_VECS * 2
            ),
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
        self.programs["material_pair_fused_terminal_local16_dirty_fast_shared_transpose"] = build_compute_shader(
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
        self.programs["material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose"] = build_compute_shader(
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
        self.programs["material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose_segment_zero"] = build_compute_shader(
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
            ctx, "reactions/material_light.comp", material_light_subs,
            includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
        )
        self.programs["material_light_authoritative_lhs"] = build_compute_shader(
            ctx, "reactions/material_light.comp", material_light_subs,
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

    _record_profile_pass = _record_profile_pass
    _upload_state_profile_scope = _upload_state_profile_scope
    _profile_scoped_pass = _profile_scoped_pass
    _ensure_resources = _ensure_resources

    run_timed_actions = run_timed_actions
    run_timed_triggers = run_timed_triggers
    run_self_triggers = run_self_triggers
    run_self_actions = run_self_actions
    run_material_material = run_material_material
    run_material_gas = run_material_gas
    run_material_pair_fused = run_material_pair_fused
    _compile_material_pair_plan = _compile_material_pair_plan
    _compile_material_pair_plan_cached = _compile_material_pair_plan_cached
    _material_pair_plan_cache_key = _material_pair_plan_cache_key
    run_material_light = run_material_light
    _run_formal_guarded_material_light = _run_formal_guarded_material_light
    run_gas_gas = run_gas_gas
    run_gas_light = run_gas_light
    _run_formal_guarded_gas_light = _run_formal_guarded_gas_light
    clear_reaction_latches = clear_reaction_latches

    _active_scheduler_gpu_authoritative = _active_scheduler_gpu_authoritative
    _formal_light_dose_guard_buffer = _formal_light_dose_guard_buffer
    _build_light_dose_guarded_dispatch_args = _build_light_dose_guarded_dispatch_args
    _run_light_dose_guarded_dispatch = _run_light_dose_guarded_dispatch
    _active_masks_for_cell_reaction_upload = _active_masks_for_cell_reaction_upload
    _reaction_state_segment = _reaction_state_segment
    _bridge_cell_core_read_role_only_load = _bridge_cell_core_read_role_only_load
    _formal_reaction_segment_base_key = _formal_reaction_segment_base_key
    _formal_reaction_segment_cache_key = _formal_reaction_segment_cache_key
    _formal_reaction_state_cache_key = _formal_reaction_state_cache_key
    _formal_reaction_active_mask_cache_key = _formal_reaction_active_mask_cache_key
    _can_use_expanded_active_tile_mask = _can_use_expanded_active_tile_mask
    _formal_reaction_state_cache_active = _formal_reaction_state_cache_active
    _formal_segment_batch_active = _formal_segment_batch_active
    _formal_terminal_gas_publish_fusion_pending = _formal_terminal_gas_publish_fusion_pending
    _formal_state_key_is_before_motion = _formal_state_key_is_before_motion
    _formal_before_motion_cell_roles_active = _formal_before_motion_cell_roles_active
    _formal_cell_read_role = _formal_cell_read_role
    _formal_cell_write_role = _formal_cell_write_role
    _set_formal_cell_read_role = _set_formal_cell_read_role
    _advance_formal_cell_read_role = _advance_formal_cell_read_role
    _reset_formal_cell_read_role = _reset_formal_cell_read_role
    _formal_velocity_read_role = _formal_velocity_read_role
    _formal_velocity_write_role = _formal_velocity_write_role
    _set_formal_velocity_read_role = _set_formal_velocity_read_role
    _advance_formal_velocity_read_role = _advance_formal_velocity_read_role
    _reset_formal_velocity_read_role = _reset_formal_velocity_read_role
    _clear_formal_external_cell_state = _clear_formal_external_cell_state
    _cell_role_textures = _cell_role_textures
    _current_cell_textures = _current_cell_textures
    _next_cell_textures = _next_cell_textures
    begin_formal_reaction_segment = begin_formal_reaction_segment
    end_formal_reaction_segment = end_formal_reaction_segment
    _mark_formal_bridge_publish_pending = _mark_formal_bridge_publish_pending
    flush_formal_reaction_segment = flush_formal_reaction_segment
    _clear_formal_segment_gas_delta = _clear_formal_segment_gas_delta
    _flush_formal_segment_gas_delta = _flush_formal_segment_gas_delta
    _clear_reaction_latches_on_bridge = _clear_reaction_latches_on_bridge
    _upload_active_masks = _upload_active_masks
    _load_authoritative_active_masks = _load_authoritative_active_masks

    _run_cell_pass = _run_cell_pass
    _run_local_cell_action_pass = _run_local_cell_action_pass
    _run_timed_candidate_action_pass = _run_timed_candidate_action_pass
    _prepare_timed_candidate_worklist = _prepare_timed_candidate_worklist
    _prepare_self_candidate_worklist = _prepare_self_candidate_worklist
    _clear_timed_candidate_local_meta = _clear_timed_candidate_local_meta
    _publish_timed_candidate_cell_state = _publish_timed_candidate_cell_state
    _accumulate_timed_candidate_segment_cell_transient_state = _accumulate_timed_candidate_segment_cell_transient_state
    _bind_local_cell_action_output_images = _bind_local_cell_action_output_images
    _scatter_local_cell_action_outputs = _scatter_local_cell_action_outputs
    _copy_current_velocity_to_next_role = _copy_current_velocity_to_next_role
    _scatter_local_emit_cell_outputs = _scatter_local_emit_cell_outputs

    _run_cell_gas_side_effect_pass = _run_cell_gas_side_effect_pass
    _run_cell_gas_action_delta_pass = _run_cell_gas_action_delta_pass
    _run_self_candidate_gas_side_effect_pass = _run_self_candidate_gas_side_effect_pass
    _run_timed_candidate_gas_side_effect_pass = _run_timed_candidate_gas_side_effect_pass
    _run_material_light_dose_consume_pass = _run_material_light_dose_consume_pass
    _run_cell_material_side_effect_pass = _run_cell_material_side_effect_pass
    _run_timed_candidate_material_side_effect_pass = _run_timed_candidate_material_side_effect_pass
    _clear_packed_timed_material_target_worklist = _clear_packed_timed_material_target_worklist
    _run_packed_timed_material_side_effect_pass = _run_packed_timed_material_side_effect_pass
    _run_produced_packed_timed_material_side_effect_pass = _run_produced_packed_timed_material_side_effect_pass
    _run_packed_material_target_apply_pass = _run_packed_material_target_apply_pass
    _run_timed_self_combined_action_pass = _run_timed_self_combined_action_pass

    _load_authoritative_bridge_inputs = _load_authoritative_bridge_inputs
    _publish_bridge_cell_state = _publish_bridge_cell_state
    _publish_bridge_gas_state = _publish_bridge_gas_state
    _publish_bridge_dose_state = _publish_bridge_dose_state
    _apply_flow_sources_to_bridge_velocity = _apply_flow_sources_to_bridge_velocity
    _publish_bridge_light_emitters = _publish_bridge_light_emitters
    _download_cell_state = _download_cell_state
    _download_gas_state = _download_gas_state
    _download_dose_state = _download_dose_state
    _download_deferred_batch = _download_deferred_batch
    _unsupported_deferred_action_indices = _unsupported_deferred_action_indices
    _append_flow_sources_from_gpu = _append_flow_sources_from_gpu

    release = release
    _upload_state = _upload_state
    _bridge_input_load_requirements = _bridge_input_load_requirements
    _missing_formal_bridge_input_loads = _missing_formal_bridge_input_loads
    _record_formal_bridge_inputs_loaded = _record_formal_bridge_inputs_loaded
    _transient_clear_requirements = _transient_clear_requirements
    _upload_random_targets = _upload_random_targets
    _clear_transient_state = _clear_transient_state
    _flow_source_generation_validity_active = _flow_source_generation_validity_active
    _advance_flow_source_generation = _advance_flow_source_generation
    _bind_flow_source_generation_output = _bind_flow_source_generation_output
    _clear_segment_transient_state = _clear_segment_transient_state
    _begin_formal_segment_meta_lazy_zero = _begin_formal_segment_meta_lazy_zero
    _reset_formal_segment_meta_lazy_zero = _reset_formal_segment_meta_lazy_zero
    _record_formal_segment_cell_meta_in_flags = _record_formal_segment_cell_meta_in_flags
    _ensure_formal_segment_meta_physical_zero = _ensure_formal_segment_meta_physical_zero
    _can_use_terminal_segment_meta_zero = _can_use_terminal_segment_meta_zero
    _accumulate_segment_cell_transient_state = _accumulate_segment_cell_transient_state
    _upload_local_metadata = _upload_local_metadata
    _promote_cell_pong_to_ping = _promote_cell_pong_to_ping
    _copy_gas_state = _copy_gas_state
    _promote_gas_pong_to_ping = _promote_gas_pong_to_ping
    _promote_gas_result = _promote_gas_result
    _promote_dose_pong_to_ping = _promote_dose_pong_to_ping
    _copy_bridge_flow_velocity_to_reaction = _copy_bridge_flow_velocity_to_reaction
    _sync_storage_and_indirect_writes = _sync_storage_and_indirect_writes
    _sync_compute_writes = _sync_compute_writes

    _compile_action_buffers = _compile_action_buffers
    _compile_action_buffers_cached = _compile_action_buffers_cached
    _compiled_actions_include_modify_gas = _compiled_actions_include_modify_gas
    _compiled_actions_include_flow_sources = _compiled_actions_include_flow_sources
    _compiled_self_rule_flow_source_layers = staticmethod(_compiled_self_rule_flow_source_layers)
    _compiled_modify_gas_layer_mask = staticmethod(_compiled_modify_gas_layer_mask)
    _compiled_actions_include_emit_material = _compiled_actions_include_emit_material
    _compiled_actions_include_emit_light = _compiled_actions_include_emit_light
    _compiled_actions_require_deferred_outputs = _compiled_actions_require_deferred_outputs
    _self_rules_require_deferred_hi_outputs = _self_rules_require_deferred_hi_outputs
    _compiled_actions_may_change_structure = _compiled_actions_may_change_structure
    _compiled_rules_include_rhs_consume = _compiled_rules_include_rhs_consume
    _compile_gas_action_buffers = _compile_gas_action_buffers
    _compile_gas_light_action_buffers = _compile_gas_light_action_buffers
    _modify_gas_action_requires_cpu_flow_side_effect = staticmethod(_modify_gas_action_requires_cpu_flow_side_effect)
    _rule_candidate_word_count = staticmethod(_rule_candidate_word_count)
    _empty_rule_candidate_masks = staticmethod(_empty_rule_candidate_masks)
    _set_rule_candidate = staticmethod(_set_rule_candidate)
    _compile_material_rule_candidate_masks = _compile_material_rule_candidate_masks
    _compile_material_material_rules = _compile_material_material_rules
    _compile_material_gas_rules = _compile_material_gas_rules
    _compile_material_light_rules = _compile_material_light_rules
    _compile_material_light_packed_descriptors = staticmethod(
        _compile_material_light_packed_descriptors
    )
    _compile_material_light_packed_descriptors_cached = (
        _compile_material_light_packed_descriptors_cached
    )
    _compile_material_pair_packed_descriptors = staticmethod(
        _compile_material_pair_packed_descriptors
    )
    _compile_material_pair_packed_descriptors_cached = (
        _compile_material_pair_packed_descriptors_cached
    )
    _compile_gas_gas_rules = _compile_gas_gas_rules
    _compile_single_gas_gas_rule = _compile_single_gas_gas_rule
    _compile_gas_light_rules = _compile_gas_light_rules
    _compile_single_gas_light_rule = _compile_single_gas_light_rule
    _used_action_indices = _used_action_indices
    _used_action_indices_for_material_slots = _used_action_indices_for_material_slots
    _cached_used_action_indices_for_material_slots = _cached_used_action_indices_for_material_slots
    _used_action_indices_for_self_rules = _used_action_indices_for_self_rules
    _cached_used_action_indices_for_self_rules = _cached_used_action_indices_for_self_rules
    _used_action_indices_for_pair_rules = _used_action_indices_for_pair_rules
    _cached_used_action_indices_for_pair_rules = _cached_used_action_indices_for_pair_rules
    _has_unsupported_consume_policies = staticmethod(_has_unsupported_consume_policies)

    _run_material_pair_fused_pass = _run_material_pair_fused_pass
