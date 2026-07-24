"""Central engine configuration.

Every ``self._<name>_enabled`` experimental switch that used to be a
hard-coded literal in the GPU pipeline facades now lives here as a typed
field.  Pipelines copy their switches out of the config at construction
time into ordinary writable instance attributes, so existing tests that
patch ``pipeline._<name>_enabled`` directly keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """Central configuration for WorldEngine solver/pipeline switches."""

    # GPU heat pipeline (gpu_heat.py)
    ambient_exchange_feedback4_enabled: bool = True
    cell_heat_bridge_exchange_feedback4_fusion_enabled: bool = True
    condense_apply_gas4x6_fusion_enabled: bool = True
    terminal4x6_fusion_enabled: bool = True
    terminal4x6_workgroup16x8_enabled: bool = False
    terminal_bridge_aux_dirty_fusion_enabled: bool = True
    terminal_phase_fusion_enabled: bool = False
    terminal_dirty_publish_fusion_enabled: bool = True
    terminal_dirty_workgroup_aggregation_enabled: bool = True
    terminal_split_target_active_reuse_enabled: bool = True
    terminal_dead_condense_target_store_elision_enabled: bool = True
    terminal_inplace_sparse_write_enabled: bool = True
    terminal_sparse_resident_specialization_enabled: bool = True
    terminal_lazy_action_inputs_enabled: bool = True
    packed_phase_boil_targets_enabled: bool = True
    terminal_hierarchical_row_summary_enabled: bool = True
    terminal_nv32_ballot_gas_reduction_enabled: bool = True
    heat_sparse_bridge_residency_enabled: bool = True

    # GPU liquid pipeline (gpu_liquid.py)
    tile_warp_direct_vertical_mapping_enabled: bool = True
    tile_warp_provenance_row_stream_enabled: bool = True
    tile_warp_lane_change_vote_enabled: bool = True
    tile_solve_bridge_hydration_fusion_enabled: bool = True
    placeholder_lazy_roles_enabled: bool = True
    tile_solve_snapshot_output_fusion_enabled: bool = True
    compact_tile_solve_snapshot_enabled: bool = True
    tile_snapshot_state_elision_enabled: bool = True
    tile_packed_pre_state_blocker_enabled: bool = True
    buoyancy_pass_fusion_enabled: bool = True
    seam_x_row_leader_enabled: bool = True
    seam_prefetch_zero_full_active_enabled: bool = True
    seam_y_shared_snapshot_enabled: bool = True
    bridge_aux_cleanup_fusion_enabled: bool = True
    cleanup_local16_enabled: bool = True
    flow_intent_shared_halo_enabled: bool = True
    flow_intent_provenance_shared_meta_cache_enabled: bool = True
    flow_intent_provenance_lazy_aux_enabled: bool = True
    provenance_terminal_enabled: bool = True
    provenance_init_fusion_enabled: bool = True
    buoyancy_cleanup_split_fusion_enabled: bool = True
    buoyancy_snapshot_pre_state_enabled: bool = True
    buoyancy_shared_sink_cache_enabled: bool = True
    buoyancy_blocker_displaced_hydration_enabled: bool = True

    # GPU reaction pipeline (gpu_reactions.py)
    expanded_active_tile_mask_enabled: bool = True
    terminal_gas_publish_fusion_enabled: bool = True
    pair_segment_meta_fusion_enabled: bool = True
    terminal_segment_meta_lazy_zero_enabled: bool = True
    segment_meta_light_counter_clear_fusion_enabled: bool = True
    packed_timed_emit_target_worklist_enabled: bool = True
    timed_emit_target_producer_enabled: bool = True
    packed_self_emit_target_worklist_enabled: bool = True
    authoritative_lhs_candidate_masks_enabled: bool = True
    self_gas_candidate_worklist_enabled: bool = True
    flow_source_generation_validity_enabled: bool = True
    flow_source_generation_u8_token_enabled: bool = True
    timed_self_authoritative_segment_masks_enabled: bool = True
    timed_self_cell_flag_meta_enabled: bool = True
    self_rule_material_spans_enabled: bool = True
    self_rule_direct_action_spans_enabled: bool = True
    material_pair_state_fusion_enabled: bool = True
    material_pair_light_state_fusion_enabled: bool = True
    material_triplet_motion_terminal_enabled: bool = True
    material_triplet_terminal_local16_enabled: bool = True
    material_triplet_terminal_dirty_fast_equal_enabled: bool = True
    material_triplet_terminal_shared_transpose_enabled: bool = True
    material_triplet_terminal_32x8_enabled: bool = True
    material_triplet_ml_packed_descriptors_enabled: bool = True
    material_pair_packed_descriptors_enabled: bool = True

    # GPU collapse pipeline (gpu_collapse.py)
    persistent_dense_tile_worklist_enabled: bool = True
    support_outcome_publish_fusion_enabled: bool = True
    classification_mask_publish_fusion_enabled: bool = True
    classification_bridge_hydration_fusion_enabled: bool = True
    label_seed_materialize_axis_fusion_enabled: bool = True
    support_tile_union_enabled: bool = False
    support_tile_union_atomic_union_enabled: bool = False
    support_jfa_row_major_output_enabled: bool = True
    incremental_support_jfa_u8_enabled: bool = True
    support_jfa_u8_propagated_source_mask_elision_enabled: bool = True
    support_jfa_nv32_row_hydrate_enabled: bool = True
    incremental_classification_support_axis_u8_fusion_enabled: bool = True
    outcome_label_tile_union_enabled: bool = True
    incremental_collapse_pipeline_enabled: bool = True
    incremental_jfa_four_frame_balance_enabled: bool = True
    incremental_phase_peak_v3_balance_enabled: bool = True
    incremental_support_outcome_publish_fusion_enabled: bool = False
    incremental_direct_immune_publish_enabled: bool = True
    incremental_direct_delayed_publish_enabled: bool = True
    incremental_materialize_metadata_fusion_enabled: bool = True
    incremental_materialize_filter_fusion_enabled: bool = True
    incremental_label_union_materialize_validation_fusion_enabled: bool = True
    incremental_component_invalid_generation_enabled: bool = True
    incremental_component_flag_generation_enabled: bool = True
    incremental_outcome_label_local_fusion_enabled: bool = True
    runtime_admission_stride_dispatch_enabled: bool = True

    # GPU motion pipeline (gpu_motion.py)
    falling_island_materialization_bridge_fusion_enabled: bool = True
    falling_island_apply_bridge_fusion_enabled: bool = True
    powder_aux_index_scratch_fusion_enabled: bool = True
    powder_sparse_bridge_publish_enabled: bool = True
    powder_apply_index_epoch_enabled: bool = True
    powder_target_clear_local64_enabled: bool = True
    powder_precomputed_fallback_blockers_enabled: bool = True
    powder_trivial_blocked_classification_enabled: bool = True
    powder_apply_tile_workgroup_dedup_enabled: bool = True
    powder_source_indexed_direct_apply_enabled: bool = True
    powder_compact_reservation_enabled: bool = True
    powder_compact_reservation_lazy_expand_enabled: bool = True
    powder_provisional_moving_worklist_enabled: bool = True
    powder_nontrivial_resolve_worklist_enabled: bool = True
    falling_island_materialization_minimal_hydration_enabled: bool = True
    falling_island_resolve_minimal_hydration_enabled: bool = True
    falling_island_materialization_changed_only_enabled: bool = True
    falling_island_apply_changed_only_enabled: bool = True
    reaction_latch_handoff_clear_enabled: bool = True

    # GPU optics pipeline (gpu_optics.py)
    direct_bridge_visible_publish_enabled: bool = True
    direct_visual_accumulator_compose_enabled: bool = True
    sparse_optics_worklists_enabled: bool = True
    sparse_gas_visible_scan_fusion_enabled: bool = True
    sparse_tile_seeded_build_enabled: bool = True
    sparse_tile_local_atomic_max_enabled: bool = False
    sparse_gas_owner_warp_compaction_enabled: bool = True
    bounded_trace_stack_enabled: bool = True
    full_active_mask_hydration_elision_enabled: bool = True

    # GPU gas pipeline (gpu_gas.py)
    divergence_pressure_seed_enabled: bool = True
    species_terminal_cooperative_enabled: bool = True
    pressure_jacobi_pair_enabled: bool = True
    density_tree_reduction_enabled: bool = True

    # CPU optics solver (optics.py)
    formal_full_active_mask_reuse_enabled: bool = True
    formal_changed_mask_alias_enabled: bool = True


# Shared defaults for call sites that construct pipelines/solvers directly
# (tests included); WorldEngine builds its own instance when none is given.
DEFAULT_ENGINE_CONFIG = EngineConfig()
