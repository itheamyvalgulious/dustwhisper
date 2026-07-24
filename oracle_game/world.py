from __future__ import annotations

import threading  # noqa: F401  # facade re-export on central module
import time
from collections import deque  # noqa: F401  # facade re-export on central module
from contextlib import contextmanager
from enum import Enum  # noqa: F401  # facade re-export on central module
from typing import Any

from oracle_game.engine_config import EngineConfig
from oracle_game.gpu import (
    GPUBufferReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUCellCoreWindowReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUGasWindowReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUReadbackSegment,  # noqa: F401  # facade re-export on central module
    GPUSegmentedBufferReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUSegmentedCellCoreWindowReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUSegmentedTextureReadbackSource,  # noqa: F401  # facade re-export on central module
    GPUTextureReadbackSource,  # noqa: F401  # facade re-export on central module
)
from oracle_game.page_store import (  # noqa: F401  # facade re-export on central module
    PageStore,
    StoredStripeKey,
)
from oracle_game.readback_contract import (
    READBACK_ALLOWED_CHANNELS,  # noqa: F401  # facade re-export on central module
)
from oracle_game.types import (
    CarrierIntent,  # noqa: F401  # facade re-export on central module
    ChangeIntent,  # noqa: F401  # facade re-export on central module
    DebugView,  # noqa: F401  # facade re-export on central module
    EntityCellFeedback,  # noqa: F401  # facade re-export on central module
    EntityFeedback,  # noqa: F401  # facade re-export on central module
    EntityObservationSpec,  # noqa: F401  # facade re-export on central module
    EntityPlaceholder,  # noqa: F401  # facade re-export on central module
    EntityState,  # noqa: F401  # facade re-export on central module
    EntityStatePatch,  # noqa: F401  # facade re-export on central module
    ForceSource,  # noqa: F401  # facade re-export on central module
    GasSpeciesDef,  # noqa: F401  # facade re-export on central module
    LightTypeDef,  # noqa: F401  # facade re-export on central module
    MaterialDef,  # noqa: F401  # facade re-export on central module
    MaterialOpticsDef,  # noqa: F401  # facade re-export on central module
    ObservationTarget,  # noqa: F401  # facade re-export on central module
    PageStripeUpdate,  # noqa: F401  # facade re-export on central module
    PairReactionRule,  # noqa: F401  # facade re-export on central module
    Phase,  # noqa: F401  # facade re-export on central module
    ReactionAction,  # noqa: F401  # facade re-export on central module
    ReadbackRequest,  # noqa: F401  # facade re-export on central module
    ReadbackResult,  # noqa: F401  # facade re-export on central module
    ResolvedCarrierIntent,  # noqa: F401  # facade re-export on central module
    ResolvedChangeIntent,  # noqa: F401  # facade re-export on central module
    ResolvedTarget,  # noqa: F401  # facade re-export on central module
    SelfReactionRule,  # noqa: F401  # facade re-export on central module
    TargetQuery,  # noqa: F401  # facade re-export on central module
    WorldCommand,
    WorldFrameInput,
    WorldFrameOutput,  # noqa: F401  # facade re-export on central module
    WorldFramePreview,  # noqa: F401  # facade re-export on central module
)
from oracle_game.world_backend_gating import (
    _gpu_active_tile_count,
    _gpu_context_available,
    _gpu_pipeline_available,
    _gpu_realtime_budget_active,
    _gpu_world_simulation_required,
    _invalidate_gpu_authoritative_cell_resources,
    _invalidate_gpu_authoritative_resources,
    _require_cpu_oracle_backend,
    _require_gpu_authoritative_resources,
    _require_gpu_stage,
    _should_run_formal_collapse_this_frame,
    _skip_budgeted_gpu_stage,
    prewarm_formal_connected_collapse,
    require_gpu_world_backend,
    use_cpu_oracle_backend,
)
from oracle_game.world_bridge_serializers import (
    _bridge_row_count,
    _clamped_gas_window,
    _decode_bridge_uploaded_command,
    _decode_bridge_uploaded_label,
    _decode_bridge_uploaded_page_stripe_section,
    _normalize_bridge_slice_bounds,
    _normalize_bridge_window_bounds,
    _record_bridge_page_stripe,
    _serialize_bridge_index_stages,
    _serialize_bridge_ndarray,
    _serialize_bridge_ndarray_slice,
    _serialize_bridge_ndarray_window,
    _serialize_bridge_readback_request_stages,
    _serialize_bridge_resource_summary,
    _serialize_bridge_spatial_window_payload,
    serialize_bridge_frame_snapshot,
    serialize_bridge_resources,
    serialize_bridge_runtime,
    serialize_bridge_shadow_buffer,
    serialize_bridge_shadow_buffer_gas_window,
    serialize_bridge_shadow_buffer_slice,
    serialize_bridge_shadow_buffer_window,
    serialize_bridge_shadow_buffer_world_window,
    serialize_bridge_typed_table,
    serialize_bridge_typed_table_slice,
    serialize_bridge_upload_snapshot,
)
from oracle_game.world_capabilities import serialize_engine_capabilities
from oracle_game.world_cell_mutators import (
    _gas_field_count,
    _gas_window_for_cell_rect,
    _inject_gas_immediate,
    _inject_temperature_immediate,
    _inject_velocity_immediate,
    add_gas_from_cells,
    allocate_island_id,
    ambient_temperature_at_cell,
    ambient_temperature_region,
    cell_to_gas,
    cell_xy_to_gas,
    clear_cell,
    clear_cell_region,
    clear_cells,
    in_bounds,
    material_by_id,
    sample_ambient_to_cells,
    sample_flow_to_cells,
    set_cell,
    set_cell_by_id,
    set_material_by_mask,
    swap_cells,
)
from oracle_game.world_command_application import (
    _apply_commands,
    _apply_grid_world_command_cpu,
    _apply_grid_world_commands,
    _grid_world_command_runtime_regions,
    _queue_loaded_collapse_pending_regions,
    _queue_loaded_collapse_pending_regions_from_payload,
    _resolve_targeted_commands,
    _subtract_page_stripe_range_from_region,
)
from oracle_game.world_command_queue import (
    _public_resolved_carrier_intent,
    _public_world_command,
    _resolve_direct_targeted_coords,
    _resolve_public_world_command,
    inject_force,
    inject_gas,
    inject_light,
    inject_material,
    inject_temperature,
    inject_velocity,
    preview_carrier_intent,
    preview_change_intent,
    preview_observation,
    preview_readback,
    preview_target_queries,
    preview_world_command,
    queue_command,
    request_carrier_intent,
    request_change_intent,
    request_observation,
    request_readback,
    request_world_command,
    write_material_region,
)
from oracle_game.world_constants import (
    BASE_MATERIAL_RUNTIME_ALIASES,  # noqa: F401  # facade re-export on central module
    ENTITY_STATE_PATCH_METADATA_FIELDS,  # noqa: F401  # facade re-export on central module
    TARGET_QUERY_DISTANCE_HINT_CELLS,  # noqa: F401  # facade re-export on central module
    UNSET_CONTROLLER_STATE,  # noqa: F401  # facade re-export on central module
)
from oracle_game.world_controller_turn import (
    _build_preview_controller_turn_entities,
    controller_turn_to_frame_input,
    preview_entity_controller_turn,
    request_entity_controller_cycle,
    request_entity_controller_turn,
    run_entity_controller_cycle,
    run_entity_controller_turn,
    set_controller_state,
)
from oracle_game.world_debug_frame import (
    _accumulate_debug_point,
    _active_frame,
    _collapse_frame,
    _draw_debug_bbox_outline,
    _gas_frame,
    _heat_frame,
    _liquid_frame,
    _material_frame,
    _motion_frame,
    _optics_dose_frame,
    _optics_frame,
    _pressure_frame,
    _reaction_frame,
    _temperature_frame,
    _vector_field_frame,
    debug_frame,
)
from oracle_game.world_demo_scene import (
    _build_demo_scene,
    _fill_rect,
    _paint_material,
    _world_engine_del,
    _write_material_region_immediate,
    close,
)
from oracle_game.world_engine_init import _init_world_engine
from oracle_game.world_entity_sync import (
    _append_force_source_immediate,
    _append_transient_light_emitter_immediate,
    _build_entity_feedback,
    _build_entity_feedback_from_state,
    _build_observation_request,
    _build_preview_entity_placeholders,
    _collect_entity_feedback,
    _collect_observations,
    _frame_entities_to_placeholders_and_observations,
    _mirror_occupy_entity_placeholder_cell,
    _occupy_entity_placeholder_cell,
    _preview_consume_entity_observation_results,
    _release_entity_placeholder_cell,
    _resolve_readback_request,
    _sync_entity_observation_specs,
    _sync_entity_placeholders,
    _sync_entity_states,
    _sync_force_sources,
    _sync_persistent_emitters,
    _sync_pre_simulation_bridge_without_debug_upload,
    consume_entity_observation_results,
    patch_entity_states,
    set_emitters,
    set_force_sources,
    sync_entity_observation_specs,
    sync_entity_placeholders,
    sync_entity_states,
)
from oracle_game.world_frame_io import (
    _apply_frame_input,
    _clear_bridge_frame_inputs,
    _needs_pre_simulation_bridge_sync,
    _prepare_bridge_frame_inputs,
    _prepare_preview_frame_context,
    cancel_frame_submission,
    cancel_readback_request,
    frame_submission_status,
    pending_frame_submission_ids,
    poll_all_frame_outputs,
    poll_frame_output,
    preview_frame_input,
    request_frame_cycle,
    request_frame_input,
    submit_frame_input,
)
from oracle_game.world_frame_pipeline import (
    _collect_ready_readbacks,
    _finish_readbacks,
    _mark_active_rect_runtime,
    _mark_active_rects_runtime,
    _merge_phase_c,
    _queue_persistent_entity_observations,
    _restore_preview_runtime_state,
    _snapshot_preview_runtime_state,
    _step_once,
    _step_once_impl,
    _store_entity_observation_consume_snapshot,
    run_cpu_frame,
    step,
)
from oracle_game.world_geometry import (
    _apply_change_stability_drift,
    _bounded_material_state_for_position,
    _buffer_bbox_to_world_bbox,
    _buffer_cell_bounds,
    _buffer_gas_to_world_position,
    _buffer_to_world_float_position,
    _buffer_to_world_position,
    _capsule_world_cells,
    _capsule_world_cells_raw,
    _centered_world_window,
    _clamped_world_window,
    _direction_vector,
    _disk_world_cells,
    _disk_world_cells_raw,
    _extract_world_window,
    _find_nearest_empty_world_position,
    _force_source_buffer_position,
    _force_source_world_position,
    _line_world_cells,
    _line_world_cells_raw,
    _matches_direction_filter,
    _pack_cell_core_world_window,
    _query_direction_vector,
    _resolve_entity_anchor,
    _resolve_legal_world_position,
    _resolve_terrain_anchor,
    _terrain_cell_matches,
    _world_axis_indices,
    _world_axis_spans,
    _world_cell_is_empty,
    _world_cell_is_empty_local,
    _world_cell_is_solid_local,
    _world_to_buffer_clamped,
    _world_to_buffer_float_position,
)
from oracle_game.world_input_coercion import (
    _assign_preview_readback_request_ids,
    _assign_readback_request_id,
    _canonical_material_input_name,
    _coerce_carrier_intent,
    _coerce_change_intent,
    _coerce_emitter,
    _coerce_entity_observation_spec,
    _coerce_entity_placeholder,
    _coerce_entity_state,
    _coerce_entity_state_patch,
    _coerce_enum,
    _coerce_force_source,
    _coerce_gas_species_def,
    _coerce_json_value,
    _coerce_light_type_def,
    _coerce_material_def,
    _coerce_material_optics_def,
    _coerce_observation_target,
    _coerce_pair_reaction_rule,
    _coerce_reaction_action,
    _coerce_reaction_rules,
    _coerce_readback_request,
    _coerce_self_reaction_rule,
    _coerce_target_query,
    _coerce_world_command,
    _coerce_world_frame_input,
    _controller_turn_entity_input,
    _frame_emitter_input,
    _frame_entity_placeholder_input,
    _frame_entity_state_input,
    _frame_entity_state_patch_input,
    _frame_force_source_input,
    _normalize_entity_state_patch_fields,
    _normalize_gas_patch_fields,
    _normalize_json_payload_value,
    _normalize_material_optics_patch_fields,
    _normalize_material_patch_fields,
    _normalize_reaction_action_patch_fields,
    _normalize_reaction_rule_patch_fields,
    _normalize_readback_channels,
    _normalize_readback_request,
    _public_entity_placeholder_input,
    _public_entity_state_input,
    _public_entity_state_patch_input,
    _public_force_source_input,
)
from oracle_game.world_intent_helpers import (
    _build_entity_feedback_from_current_state,
    _build_entity_feedback_from_world,
    _build_observation_requests,
    _clamp_world_position,
    _combine_resolution_notes,
    _default_target_source_position,
    _distance_meters_to_cells,
    _entity_center_buffer_position,
    _entity_center_world_position,
    _entity_matches_anchor_filters,
    _entity_placeholder_bbox,
    _intent_resolution_status,
    _material_state_for_position,
    _normalize_runtime_force_source,
    _normalized_world_direction,
    _patch_entity_states,
    _preview_can_occupy_placeholder_cell,
    _public_resolved_target,
    _resolve_carrier_intents,
    _resolve_change_intents,
    _resolve_intent_source_positions,
    _resolve_intent_world_position,
    _resolve_query_source_position,
    _resolve_readback_requests,
    _source_facing_vector,
    _terrain_hill_cell_matches,
    _terrain_tree_cell_matches,
    _world_cell_material_has_tag,
    _world_distance_sq,
    _world_gas_window_for_cell_world_rect,
)
from oracle_game.world_intent_resolver import (
    _resolve_carrier_intent,
    _resolve_change_intent,
    _resolve_change_intent_world_position,
    _resolve_target_queries,
    _resolve_target_query,
    _resolve_target_query_distance_cells,
)
from oracle_game.world_internal_helpers import (
    _advance_paging,
    _build_observation_request_pairs,
    _frame_readback_request_ids,
    _light_field_count,
    _mark_grid_world_command_runtime_regions,
    _mirror_release_entity_placeholder_cell,
    _page_store_key_lookup_update,
    _page_stripe_island_bboxes_from_payload,
    _pending_frame_input,
    _public_resolved_change_intent,
    _queued_command_xy,
    _refresh_island_records_for_ids,
    _resolve_anchor_target,
    _set_nested_payload_value,
    bootstrap_defaults,
    cancel_all_pending_frame_submissions,
    downsample_cells_to_gas,
    readback_request_status,
    submit_entity_controller_turn,
)
from oracle_game.world_paging import (
    _apply_page_stripe,
    _apply_page_stripe_dense_cpu,
    _capture_page_stripe_cpu_snapshot,
    _capture_stripe_array,
    _clear_saved_page_stripe_runtime_state,
    _coerce_page_store_key,
    _coerce_page_stripe_payload,
    _contextualize_page_stripe_update,
    _default_page_stripe_payload,
    _mark_loaded_page_stripe_active,
    _page_store_key,
    _preview_apply_paging_updates,
    _prune_page_stripe_regions,
    _stripe_buffer_ranges,
    _sync_loaded_page_stripe_cpu_mirror,
    _write_stripe_array,
    advance_paging,
    apply_page_stripe,
    apply_stored_page_stripe,
    capture_page_stripe,
    capture_page_stripe_to_store,
    clear_page_store,
    export_page_store_entries,
    focus_paging,
    import_page_store_entries,
    list_page_store_stripe_keys,
    load_page_stripe,
    page_store_has_stripe,
    poll_all_readbacks,
    poll_readbacks,
    store_page_stripe,
)
from oracle_game.world_payload_serializers import (
    _infer_readback_payload_coord_space,
    _serialize_cpu_visible_entity_placeholders,
    _serialize_emitter_record,
    _serialize_force_source_record,
    _serialize_observation_plan_for_target_request,
    _serialize_preview_bridge_frame_snapshot,
    _serialize_readback_payload,
    _serialize_readback_plan_for_request,
    _serialize_readback_plans_for_requests,
    _serialize_readback_source_descriptor,
    serialize_carrier_intent_input,
    serialize_change_intent_input,
    serialize_consumed_entity_feedback_snapshot,
    serialize_controller_state,
    serialize_debug_frame,
    serialize_emitters,
    serialize_entity_feedback,
    serialize_entity_feedback_snapshot,
    serialize_entity_observation_consume_state,
    serialize_entity_observation_spec,
    serialize_entity_observation_state,
    serialize_entity_placeholder_index_snapshot,
    serialize_entity_placeholder_input,
    serialize_entity_placeholders,
    serialize_entity_state,
    serialize_entity_state_input,
    serialize_entity_state_patch,
    serialize_entity_states,
    serialize_force_sources,
    serialize_frame_input,
    serialize_frame_output,
    serialize_frame_preview,
    serialize_frame_state,
    serialize_gas,
    serialize_gas_species_table,
    serialize_light_type_table,
    serialize_local_cells,
    serialize_material_optics_table,
    serialize_material_table,
    serialize_observation_plan,
    serialize_observation_result,
    serialize_observation_target,
    serialize_optics,
    serialize_page_store_key,
    serialize_page_store_state,
    serialize_page_stripe_payload,
    serialize_page_stripe_update,
    serialize_pending_commands,
    serialize_pending_frame_detail,
    serialize_pending_frame_inputs,
    serialize_pressure,
    serialize_reaction_table,
    serialize_readback_plan,
    serialize_readback_request,
    serialize_readback_result,
    serialize_readback_state,
    serialize_ready_frame_outputs,
    serialize_ready_readbacks,
    serialize_resolved_carrier_intent,
    serialize_resolved_change_intent,
    serialize_resolved_target,
    serialize_target_query_input,
    serialize_temperature_window,
    serialize_velocity,
    serialize_visible_illumination,
    serialize_world_command,
)
from oracle_game.world_readback_payload import make_readback_payload as _make_readback_payload
from oracle_game.world_runtime_rebuild import (
    _apply_page_stripe_entity_placeholder_runtime,
    _capture_page_stripe_entity_placeholder_runtime,
    _capture_page_stripe_island_runtime,
    _cell_participates_in_collapse,
    _drain_gpu_collapse_structure_dirty_tiles,
    _mark_collapse_dirty_rect,
    _merge_island_runtime_payload,
    _normalize_cell_runtime_arrays,
    _normalize_page_stripe_cell_runtime,
    _rebuild_entity_placeholder_index,
    _rebuild_gas_property_arrays,
    _rebuild_island_records,
    _rebuild_light_property_arrays,
    _rebuild_material_property_arrays,
    _rebuild_sparse_runtime_indexes,
)
from oracle_game.world_runtime_serializers import (
    serialize_active_runtime,
    serialize_collapse_runtime,
    serialize_gas_runtime,
    serialize_heat_runtime,
    serialize_liquid_runtime,
    serialize_motion_runtime,
    serialize_optics_runtime,
    serialize_paging_state,
    serialize_reaction_runtime,
)
from oracle_game.world_shadow_tables import (
    _reaction_rule_list,
    _resolve_sanctioned_gas_id,
    _resolve_sanctioned_light_id,
    _resolve_sanctioned_material_id,
    _resolve_sanctioned_placeholder_material_id,
    _shadow_condense_target_material_id,
    _shadow_gas_name,
    _shadow_gas_row_valid,
    _shadow_gas_species_def,
    _shadow_light_color,
    _shadow_light_default_range,
    _shadow_light_dose_channel,
    _shadow_light_name,
    _shadow_light_name_and_range,
    _shadow_light_row_valid,
    _shadow_light_type_def,
    _shadow_material_base_integrity,
    _shadow_material_def,
    _shadow_material_default_phase,
    _shadow_material_id_by_name,
    _shadow_material_is_placeholder,
    _shadow_material_is_plant,
    _shadow_material_name,
    _shadow_material_optics_def,
    _shadow_material_row_valid,
    _shadow_material_spawn_temperature,
    _shadow_reaction_action,
    _shadow_reaction_rule,
)
from oracle_game.world_state_snapshots import (
    _bridge_shadow_buffer_coord_space,
    _current_cell_state_snapshot,
    _current_entity_runtime_snapshot,
    _entity_placeholder_state_gpu_authoritative,
    _material_optics_snapshot_map,
    _preview_bridge_placeholder_dirty_rects,
    _runtime_entities_to_immediate_observation_targets,
    simulation_backend_report,
)
from oracle_game.world_table_api import (
    _reset_world_state,
    delete_reaction_action,
    delete_reaction_rule,
    patch_gas,
    patch_light,
    patch_material,
    patch_material_optics,
    patch_reaction_action,
    patch_reaction_rule,
    replace_reaction_table,
    reset_world,
    update_gas_species_table,
    update_light_type_table,
    update_material_optics_table,
    update_material_table,
    update_reaction_table,
)
from oracle_game.world_table_validation import (
    _clamp_material_payload_reaction_slots,
    _gas_species_table_snapshot_payload,
    _light_type_table_snapshot_payload,
    _material_optics_table_snapshot_payload,
    _material_placeholder_mask,
    _material_table_snapshot_payload,
    _merged_gas_species_table_payload,
    _merged_light_type_table_payload,
    _merged_material_optics_table_payload,
    _merged_material_table_payload,
    _merged_reaction_table_payload,
    _payload_name_set,
    _reaction_table_snapshot_payload,
    _remap_material_payload_reaction_slots,
    _remap_reaction_payload_result_actions,
    _set_reaction_rule_list,
    _set_reaction_rules_payload,
    _set_stable_shadow_payload,
    _shadow_gas_species_payload,
    _shadow_has_table_payload,
    _shadow_light_type_payload,
    _shadow_material_payload,
    _shadow_reaction_payload,
    _stable_shadow_payload,
    _validate_gas_species_payload,
    _validate_light_type_payload,
    _validate_material_optics_payload,
    _validate_material_table_payload,
    _validate_named_reference,
    _validate_reaction_payload,
    _validate_unique_identity_fields,
)


class WorldEngine:
    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 192,
        active_width: int | None = None,
        active_height: int | None = None,
        gas_cell_size: int = 4,
        gpu_context: Any | None = None,
        page_store: PageStore | None = None,
        simulation_backend: str = "gpu",
        engine_config: EngineConfig | None = None,
    ) -> None:
        _init_world_engine(
            self,
            width=width,
            height=height,
            active_width=active_width,
            active_height=active_height,
            gas_cell_size=gas_cell_size,
            gpu_context=gpu_context,
            page_store=page_store,
            simulation_backend=simulation_backend,
            engine_config=engine_config,
        )

    # ------------------------------------------------------------------
    # Satellite method delegates (W5/B6: retired the `_x = _x` class grafts).
    #
    # Each body resolves the bare function name through this module's global
    # namespace -- method bodies never see class scope -- i.e. the satellite
    # function imported at the top of this file, bound at import time exactly
    # like the historical grafts.  Monkeypatch semantics are unchanged:
    # patching the attribute on the class or on an instance shadows/replaces
    # the delegate, while patching the satellite module's attribute does NOT
    # affect calls through the engine.
    # ------------------------------------------------------------------

    def use_cpu_oracle_backend(self, *args: Any, **kwargs: Any) -> Any:
        return use_cpu_oracle_backend(self, *args, **kwargs)

    def require_gpu_world_backend(self, *args: Any, **kwargs: Any) -> Any:
        return require_gpu_world_backend(self, *args, **kwargs)

    def prewarm_formal_connected_collapse(self, *args: Any, **kwargs: Any) -> Any:
        return prewarm_formal_connected_collapse(self, *args, **kwargs)

    def _gpu_context_available(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_context_available(self, *args, **kwargs)

    def _gpu_world_simulation_required(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_world_simulation_required(self, *args, **kwargs)

    def _gpu_realtime_budget_active(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_realtime_budget_active(self, *args, **kwargs)

    def _gpu_active_tile_count(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_active_tile_count(self, *args, **kwargs)

    def _skip_budgeted_gpu_stage(self, *args: Any, **kwargs: Any) -> Any:
        return _skip_budgeted_gpu_stage(self, *args, **kwargs)

    def _should_run_formal_collapse_this_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _should_run_formal_collapse_this_frame(self, *args, **kwargs)

    @contextmanager
    def _profile_pass(self, name: str):
        profile = self.last_pass_profile if self.profile_passes_enabled else None
        ctx = self.bridge.ctx if bool(getattr(self, "profile_passes_sync", False)) else None
        if profile is not None and ctx is not None:
            ctx.finish()
        start = time.perf_counter() if profile is not None else 0.0
        try:
            yield
        finally:
            # Returning from finally would suppress failures in the stage.
            if profile is not None:
                if ctx is not None:
                    ctx.finish()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                entry = {
                    "name": str(name),
                    "cpu_ms": elapsed_ms,
                    "gpu_ms": elapsed_ms if ctx is not None else None,
                }
                profile["passes"].append(entry)
                summary = profile["summary"].setdefault(
                    str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None}
                )
                summary["count"] += 1
                summary["cpu_ms"] += elapsed_ms
                if ctx is not None:
                    summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

    def _gpu_pipeline_available(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_pipeline_available(self, *args, **kwargs)

    def _require_gpu_stage(self, *args: Any, **kwargs: Any) -> Any:
        return _require_gpu_stage(self, *args, **kwargs)

    def _require_gpu_authoritative_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _require_gpu_authoritative_resources(self, *args, **kwargs)

    def _require_cpu_oracle_backend(self, *args: Any, **kwargs: Any) -> Any:
        return _require_cpu_oracle_backend(self, *args, **kwargs)

    def _invalidate_gpu_authoritative_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _invalidate_gpu_authoritative_resources(self, *args, **kwargs)

    def _invalidate_gpu_authoritative_cell_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _invalidate_gpu_authoritative_cell_resources(self, *args, **kwargs)

    def bootstrap_defaults(self) -> None:
        return bootstrap_defaults(self)

    def _material_table_snapshot_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _material_table_snapshot_payload(self, *args, **kwargs)

    def _gas_species_table_snapshot_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_species_table_snapshot_payload(self, *args, **kwargs)

    def _light_type_table_snapshot_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _light_type_table_snapshot_payload(self, *args, **kwargs)

    def _material_optics_table_snapshot_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _material_optics_table_snapshot_payload(self, *args, **kwargs)

    def _material_optics_snapshot_map(self, *args: Any, **kwargs: Any) -> Any:
        return _material_optics_snapshot_map(self, *args, **kwargs)

    def _reaction_table_snapshot_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _reaction_table_snapshot_payload(self, *args, **kwargs)

    def _stable_shadow_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _stable_shadow_payload(self, *args, **kwargs)

    def _set_stable_shadow_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _set_stable_shadow_payload(self, *args, **kwargs)

    def _shadow_has_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_has_table_payload(self, *args, **kwargs)

    def _merged_reaction_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merged_reaction_table_payload(self, *args, **kwargs)

    def _merged_material_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merged_material_table_payload(self, *args, **kwargs)

    def _merged_gas_species_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merged_gas_species_table_payload(self, *args, **kwargs)

    def _merged_light_type_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merged_light_type_table_payload(self, *args, **kwargs)

    def _merged_material_optics_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merged_material_optics_table_payload(self, *args, **kwargs)

    @staticmethod
    def _coerce_enum(*args: Any, **kwargs: Any) -> Any:
        return _coerce_enum(*args, **kwargs)

    def _coerce_material_def(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_material_def(self, *args, **kwargs)

    def _coerce_gas_species_def(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_gas_species_def(self, *args, **kwargs)

    def _coerce_light_type_def(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_light_type_def(self, *args, **kwargs)

    @staticmethod
    def _canonical_material_input_name(*args: Any, **kwargs: Any) -> Any:
        return _canonical_material_input_name(*args, **kwargs)

    def _coerce_material_optics_def(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_material_optics_def(self, *args, **kwargs)

    def _coerce_reaction_action(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_reaction_action(self, *args, **kwargs)

    def _coerce_pair_reaction_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_pair_reaction_rule(self, *args, **kwargs)

    def _coerce_self_reaction_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_self_reaction_rule(self, *args, **kwargs)

    def _coerce_reaction_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_reaction_rules(self, *args, **kwargs)

    def _shadow_material_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_payload(self, *args, **kwargs)

    def _shadow_gas_species_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_gas_species_payload(self, *args, **kwargs)

    def _shadow_light_type_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_type_payload(self, *args, **kwargs)

    def _shadow_reaction_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_reaction_payload(self, *args, **kwargs)

    @staticmethod
    def _payload_name_set(*args: Any, **kwargs: Any) -> Any:
        return _payload_name_set(*args, **kwargs)

    @staticmethod
    def _validate_named_reference(*args: Any, **kwargs: Any) -> Any:
        return _validate_named_reference(*args, **kwargs)

    @staticmethod
    def _validate_unique_identity_fields(*args: Any, **kwargs: Any) -> Any:
        return _validate_unique_identity_fields(*args, **kwargs)

    def _validate_material_table_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _validate_material_table_payload(self, *args, **kwargs)

    def _validate_gas_species_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _validate_gas_species_payload(self, *args, **kwargs)

    def _validate_light_type_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _validate_light_type_payload(self, *args, **kwargs)

    def _validate_material_optics_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _validate_material_optics_payload(self, *args, **kwargs)

    def _validate_reaction_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _validate_reaction_payload(self, *args, **kwargs)

    def _normalize_material_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_material_patch_fields(self, *args, **kwargs)

    def _normalize_gas_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_gas_patch_fields(self, *args, **kwargs)

    def _normalize_material_optics_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_material_optics_patch_fields(self, *args, **kwargs)

    def _normalize_reaction_action_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_reaction_action_patch_fields(self, *args, **kwargs)

    def _normalize_reaction_rule_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_reaction_rule_patch_fields(self, *args, **kwargs)

    def _coerce_force_source(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_force_source(self, *args, **kwargs)

    def _public_force_source_input(self, *args: Any, **kwargs: Any) -> Any:
        return _public_force_source_input(self, *args, **kwargs)

    def _frame_force_source_input(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_force_source_input(self, *args, **kwargs)

    def _coerce_emitter(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_emitter(self, *args, **kwargs)

    def _frame_emitter_input(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_emitter_input(self, *args, **kwargs)

    def _coerce_entity_placeholder(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_entity_placeholder(self, *args, **kwargs)

    def _public_entity_placeholder_input(self, *args: Any, **kwargs: Any) -> Any:
        return _public_entity_placeholder_input(self, *args, **kwargs)

    def _frame_entity_placeholder_input(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_entity_placeholder_input(self, *args, **kwargs)

    def _coerce_entity_state(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_entity_state(self, *args, **kwargs)

    def _public_entity_state_input(self, *args: Any, **kwargs: Any) -> Any:
        return _public_entity_state_input(self, *args, **kwargs)

    def _frame_entity_state_input(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_entity_state_input(self, *args, **kwargs)

    def _coerce_entity_observation_spec(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_entity_observation_spec(self, *args, **kwargs)

    def _normalize_entity_state_patch_fields(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_entity_state_patch_fields(self, *args, **kwargs)

    def _public_entity_state_patch_input(self, *args: Any, **kwargs: Any) -> Any:
        return _public_entity_state_patch_input(self, *args, **kwargs)

    def _controller_turn_entity_input(self, *args: Any, **kwargs: Any) -> Any:
        return _controller_turn_entity_input(self, *args, **kwargs)

    def _frame_entity_state_patch_input(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_entity_state_patch_input(self, *args, **kwargs)

    def _coerce_entity_state_patch(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_entity_state_patch(self, *args, **kwargs)

    def _coerce_observation_target(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_observation_target(self, *args, **kwargs)

    def _coerce_target_query(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_target_query(self, *args, **kwargs)

    def _coerce_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_change_intent(self, *args, **kwargs)

    def _coerce_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_carrier_intent(self, *args, **kwargs)

    def _coerce_readback_request(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_readback_request(self, *args, **kwargs)

    @staticmethod
    def _normalize_readback_channels(*args: Any, **kwargs: Any) -> Any:
        return _normalize_readback_channels(*args, **kwargs)

    def _normalize_readback_request(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_readback_request(self, *args, **kwargs)

    def _assign_readback_request_id(self, *args: Any, **kwargs: Any) -> Any:
        return _assign_readback_request_id(self, *args, **kwargs)

    def _assign_preview_readback_request_ids(self, *args: Any, **kwargs: Any) -> Any:
        return _assign_preview_readback_request_ids(self, *args, **kwargs)

    def _coerce_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_world_command(self, *args, **kwargs)

    @staticmethod
    def _coerce_json_value(*args: Any, **kwargs: Any) -> Any:
        return _coerce_json_value(*args, **kwargs)

    @staticmethod
    def _normalize_json_payload_value(*args: Any, **kwargs: Any) -> Any:
        return _normalize_json_payload_value(*args, **kwargs)

    def _coerce_world_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_world_frame_input(self, *args, **kwargs)

    def _gas_field_count(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_field_count(self, *args, **kwargs)

    def _light_field_count(self, *args: Any, **kwargs: Any) -> Any:
        return _light_field_count(self, *args, **kwargs)

    def update_material_table(self, *args: Any, **kwargs: Any) -> Any:
        return update_material_table(self, *args, **kwargs)

    def update_gas_species_table(self, *args: Any, **kwargs: Any) -> Any:
        return update_gas_species_table(self, *args, **kwargs)

    def update_light_type_table(self, *args: Any, **kwargs: Any) -> Any:
        return update_light_type_table(self, *args, **kwargs)

    def update_material_optics_table(self, *args: Any, **kwargs: Any) -> Any:
        return update_material_optics_table(self, *args, **kwargs)

    def update_reaction_table(self, *args: Any, **kwargs: Any) -> Any:
        return update_reaction_table(self, *args, **kwargs)

    def replace_reaction_table(self, *args: Any, **kwargs: Any) -> Any:
        return replace_reaction_table(self, *args, **kwargs)

    def reset_world(self, *args: Any, **kwargs: Any) -> Any:
        return reset_world(self, *args, **kwargs)

    def _reset_world_state(self, *args: Any, **kwargs: Any) -> Any:
        return _reset_world_state(self, *args, **kwargs)

    def queue_command(self, *args: Any, **kwargs: Any) -> Any:
        return queue_command(self, *args, **kwargs)

    def _resolve_direct_targeted_coords(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_direct_targeted_coords(self, *args, **kwargs)

    def inject_material(self, *args: Any, **kwargs: Any) -> Any:
        return inject_material(self, *args, **kwargs)

    def write_material_region(self, *args: Any, **kwargs: Any) -> Any:
        return write_material_region(self, *args, **kwargs)

    def inject_temperature(self, *args: Any, **kwargs: Any) -> Any:
        return inject_temperature(self, *args, **kwargs)

    def inject_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return inject_velocity(self, *args, **kwargs)

    def inject_force(self, *args: Any, **kwargs: Any) -> Any:
        return inject_force(self, *args, **kwargs)

    def inject_gas(self, *args: Any, **kwargs: Any) -> Any:
        return inject_gas(self, *args, **kwargs)

    def request_readback(self, *args: Any, **kwargs: Any) -> Any:
        return request_readback(self, *args, **kwargs)

    def preview_readback(self, *args: Any, **kwargs: Any) -> Any:
        return preview_readback(self, *args, **kwargs)

    def request_observation(self, *args: Any, **kwargs: Any) -> Any:
        return request_observation(self, *args, **kwargs)

    def preview_observation(self, *args: Any, **kwargs: Any) -> Any:
        return preview_observation(self, *args, **kwargs)

    def _resolve_public_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_public_world_command(self, *args, **kwargs)

    def _public_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return _public_world_command(self, *args, **kwargs)

    def preview_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return preview_world_command(self, *args, **kwargs)

    def preview_target_queries(self, *args: Any, **kwargs: Any) -> Any:
        return preview_target_queries(self, *args, **kwargs)

    def request_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return request_world_command(self, *args, **kwargs)

    def preview_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return preview_change_intent(self, *args, **kwargs)

    def request_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return request_change_intent(self, *args, **kwargs)

    def preview_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return preview_carrier_intent(self, *args, **kwargs)

    def request_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return request_carrier_intent(self, *args, **kwargs)

    def preview_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return preview_frame_input(self, *args, **kwargs)

    def submit_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return submit_frame_input(self, *args, **kwargs)

    def request_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return request_frame_input(self, *args, **kwargs)

    def request_frame_cycle(self, *args: Any, **kwargs: Any) -> Any:
        return request_frame_cycle(self, *args, **kwargs)

    def pending_frame_submission_ids(self, *args: Any, **kwargs: Any) -> Any:
        return pending_frame_submission_ids(self, *args, **kwargs)

    def _pending_frame_input(self, submission_id: int) -> WorldFrameInput:
        return _pending_frame_input(self, submission_id)

    @staticmethod
    def _frame_readback_request_ids(*args: Any, **kwargs: Any) -> Any:
        return _frame_readback_request_ids(*args, **kwargs)

    def cancel_frame_submission(self, *args: Any, **kwargs: Any) -> Any:
        return cancel_frame_submission(self, *args, **kwargs)

    def cancel_all_pending_frame_submissions(self) -> list[int]:
        return cancel_all_pending_frame_submissions(self)

    def cancel_readback_request(self, *args: Any, **kwargs: Any) -> Any:
        return cancel_readback_request(self, *args, **kwargs)

    def poll_frame_output(self, *args: Any, **kwargs: Any) -> Any:
        return poll_frame_output(self, *args, **kwargs)

    def poll_all_frame_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return poll_all_frame_outputs(self, *args, **kwargs)

    def serialize_pending_commands(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_pending_commands(self, *args, **kwargs)

    def serialize_readback_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_readback_state(self, *args, **kwargs)

    def serialize_bridge_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_runtime(self, *args, **kwargs)

    @staticmethod
    def _serialize_bridge_resource_summary(*args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_resource_summary(*args, **kwargs)

    def serialize_bridge_resources(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_resources(self, *args, **kwargs)

    def serialize_ready_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_ready_readbacks(self, *args, **kwargs)

    def readback_request_status(self, *args: Any, **kwargs: Any) -> Any:
        return readback_request_status(self, *args, **kwargs)

    def serialize_frame_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_frame_state(self, *args, **kwargs)

    def serialize_pending_frame_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_pending_frame_inputs(self, *args, **kwargs)

    def serialize_pending_frame_detail(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_pending_frame_detail(self, *args, **kwargs)

    def serialize_ready_frame_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_ready_frame_outputs(self, *args, **kwargs)

    def frame_submission_status(self, *args: Any, **kwargs: Any) -> Any:
        return frame_submission_status(self, *args, **kwargs)

    def inject_light(self, *args: Any, **kwargs: Any) -> Any:
        return inject_light(self, *args, **kwargs)

    def focus_paging(self, *args: Any, **kwargs: Any) -> Any:
        return focus_paging(self, *args, **kwargs)

    def advance_paging(self, *args: Any, **kwargs: Any) -> Any:
        return advance_paging(self, *args, **kwargs)

    def capture_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return capture_page_stripe(self, *args, **kwargs)

    def _capture_page_stripe_cpu_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_page_stripe_cpu_snapshot(self, *args, **kwargs)

    def apply_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return apply_page_stripe(self, *args, **kwargs)

    def store_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return store_page_stripe(self, *args, **kwargs)

    def capture_page_stripe_to_store(self, *args: Any, **kwargs: Any) -> Any:
        return capture_page_stripe_to_store(self, *args, **kwargs)

    def load_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return load_page_stripe(self, *args, **kwargs)

    def apply_stored_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return apply_stored_page_stripe(self, *args, **kwargs)

    def page_store_has_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return page_store_has_stripe(self, *args, **kwargs)

    def list_page_store_stripe_keys(self, *args: Any, **kwargs: Any) -> Any:
        return list_page_store_stripe_keys(self, *args, **kwargs)

    def export_page_store_entries(self, *args: Any, **kwargs: Any) -> Any:
        return export_page_store_entries(self, *args, **kwargs)

    def import_page_store_entries(self, *args: Any, **kwargs: Any) -> Any:
        return import_page_store_entries(self, *args, **kwargs)

    def clear_page_store(self, *args: Any, **kwargs: Any) -> Any:
        return clear_page_store(self, *args, **kwargs)

    def serialize_page_store_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_page_store_state(self, *args, **kwargs)

    def _coerce_page_store_key(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_page_store_key(self, *args, **kwargs)

    @staticmethod
    def _page_store_key_lookup_update(*args: Any, **kwargs: Any) -> Any:
        return _page_store_key_lookup_update(*args, **kwargs)

    def _coerce_page_stripe_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _coerce_page_stripe_payload(self, *args, **kwargs)

    def sync_entity_placeholders(self, *args: Any, **kwargs: Any) -> Any:
        return sync_entity_placeholders(self, *args, **kwargs)

    def sync_entity_states(self, *args: Any, **kwargs: Any) -> Any:
        return sync_entity_states(self, *args, **kwargs)

    def patch_entity_states(self, *args: Any, **kwargs: Any) -> Any:
        return patch_entity_states(self, *args, **kwargs)

    def sync_entity_observation_specs(self, *args: Any, **kwargs: Any) -> Any:
        return sync_entity_observation_specs(self, *args, **kwargs)

    def set_force_sources(self, *args: Any, **kwargs: Any) -> Any:
        return set_force_sources(self, *args, **kwargs)

    def set_emitters(self, *args: Any, **kwargs: Any) -> Any:
        return set_emitters(self, *args, **kwargs)

    def patch_material(self, *args: Any, **kwargs: Any) -> Any:
        return patch_material(self, *args, **kwargs)

    def patch_light(self, *args: Any, **kwargs: Any) -> Any:
        return patch_light(self, *args, **kwargs)

    def patch_gas(self, *args: Any, **kwargs: Any) -> Any:
        return patch_gas(self, *args, **kwargs)

    def patch_material_optics(self, *args: Any, **kwargs: Any) -> Any:
        return patch_material_optics(self, *args, **kwargs)

    def patch_reaction_action(self, *args: Any, **kwargs: Any) -> Any:
        return patch_reaction_action(self, *args, **kwargs)

    def patch_reaction_rule(self, *args: Any, **kwargs: Any) -> Any:
        return patch_reaction_rule(self, *args, **kwargs)

    def delete_reaction_action(self, *args: Any, **kwargs: Any) -> Any:
        return delete_reaction_action(self, *args, **kwargs)

    def delete_reaction_rule(self, *args: Any, **kwargs: Any) -> Any:
        return delete_reaction_rule(self, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def simulation_backend_report(self, *args: Any, **kwargs: Any) -> Any:
        return simulation_backend_report(self, *args, **kwargs)

    def poll_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return poll_readbacks(self, *args, **kwargs)

    def poll_all_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return poll_all_readbacks(self, *args, **kwargs)

    def consume_entity_observation_results(self, *args: Any, **kwargs: Any) -> Any:
        return consume_entity_observation_results(self, *args, **kwargs)

    def run_entity_controller_turn(self, *args: Any, **kwargs: Any) -> Any:
        return run_entity_controller_turn(self, *args, **kwargs)

    def set_controller_state(self, *args: Any, **kwargs: Any) -> Any:
        return set_controller_state(self, *args, **kwargs)

    def serialize_controller_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_controller_state(self, *args, **kwargs)

    def _build_preview_controller_turn_entities(self, *args: Any, **kwargs: Any) -> Any:
        return _build_preview_controller_turn_entities(self, *args, **kwargs)

    def _preview_consume_entity_observation_results(self, *args: Any, **kwargs: Any) -> Any:
        return _preview_consume_entity_observation_results(self, *args, **kwargs)

    def controller_turn_to_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return controller_turn_to_frame_input(self, *args, **kwargs)

    def preview_entity_controller_turn(self, *args: Any, **kwargs: Any) -> Any:
        return preview_entity_controller_turn(self, *args, **kwargs)

    def submit_entity_controller_turn(self, *args: Any, **kwargs: Any) -> Any:
        return submit_entity_controller_turn(self, *args, **kwargs)

    def request_entity_controller_turn(self, *args: Any, **kwargs: Any) -> Any:
        return request_entity_controller_turn(self, *args, **kwargs)

    def request_entity_controller_cycle(self, *args: Any, **kwargs: Any) -> Any:
        return request_entity_controller_cycle(self, *args, **kwargs)

    def run_entity_controller_cycle(self, *args: Any, **kwargs: Any) -> Any:
        return run_entity_controller_cycle(self, *args, **kwargs)

    def run_cpu_frame(self, *args: Any, **kwargs: Any) -> Any:
        return run_cpu_frame(self, *args, **kwargs)

    def _step_once(self, *args: Any, **kwargs: Any) -> Any:
        return _step_once(self, *args, **kwargs)

    def _merge_phase_c(self, *args: Any, **kwargs: Any) -> Any:
        return _merge_phase_c(self, *args, **kwargs)

    def _step_once_impl(self, *args: Any, **kwargs: Any) -> Any:
        return _step_once_impl(self, *args, **kwargs)

    def _queue_persistent_entity_observations(self, *args: Any, **kwargs: Any) -> Any:
        return _queue_persistent_entity_observations(self, *args, **kwargs)

    def _apply_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_frame_input(self, *args, **kwargs)

    def _prepare_preview_frame_context(self, *args: Any, **kwargs: Any) -> Any:
        return _prepare_preview_frame_context(self, *args, **kwargs)

    def _snapshot_preview_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return _snapshot_preview_runtime_state(self, *args, **kwargs)

    def _restore_preview_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return _restore_preview_runtime_state(self, *args, **kwargs)

    def _contextualize_page_stripe_update(self, *args: Any, **kwargs: Any) -> Any:
        return _contextualize_page_stripe_update(self, *args, **kwargs)

    @staticmethod
    def _page_store_key(*args: Any, **kwargs: Any) -> Any:
        return _page_store_key(*args, **kwargs)

    def _preview_apply_paging_updates(self, *args: Any, **kwargs: Any) -> Any:
        return _preview_apply_paging_updates(self, *args, **kwargs)

    def _preview_bridge_placeholder_dirty_rects(self, *args: Any, **kwargs: Any) -> Any:
        return _preview_bridge_placeholder_dirty_rects(self, *args, **kwargs)

    @staticmethod
    def _serialize_bridge_readback_request_stages(*args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_readback_request_stages(*args, **kwargs)

    @staticmethod
    def _serialize_bridge_index_stages(*args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_index_stages(*args, **kwargs)

    def _serialize_preview_bridge_frame_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_preview_bridge_frame_snapshot(self, *args, **kwargs)

    def _queue_loaded_collapse_pending_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _queue_loaded_collapse_pending_regions(self, *args, **kwargs)

    def _clear_saved_page_stripe_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_saved_page_stripe_runtime_state(self, *args, **kwargs)

    def _prune_page_stripe_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _prune_page_stripe_regions(self, *args, **kwargs)

    @staticmethod
    def _subtract_page_stripe_range_from_region(*args: Any, **kwargs: Any) -> Any:
        return _subtract_page_stripe_range_from_region(*args, **kwargs)

    def close(self, *args: Any, **kwargs: Any) -> Any:
        return close(self, *args, **kwargs)

    def __del__(self) -> None:  # pragma: no cover
        _world_engine_del(self)

    def material_by_id(self, *args: Any, **kwargs: Any) -> Any:
        return material_by_id(self, *args, **kwargs)

    def allocate_island_id(self, *args: Any, **kwargs: Any) -> Any:
        return allocate_island_id(self, *args, **kwargs)

    def _refresh_island_records_for_ids(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_island_records_for_ids(self, *args, **kwargs)

    def in_bounds(self, *args: Any, **kwargs: Any) -> Any:
        return in_bounds(self, *args, **kwargs)

    def cell_xy_to_gas(self, *args: Any, **kwargs: Any) -> Any:
        return cell_xy_to_gas(self, *args, **kwargs)

    def cell_to_gas(self, *args: Any, **kwargs: Any) -> Any:
        return cell_to_gas(self, *args, **kwargs)

    def sample_ambient_to_cells(self, *args: Any, **kwargs: Any) -> Any:
        return sample_ambient_to_cells(self, *args, **kwargs)

    def ambient_temperature_at_cell(self, *args: Any, **kwargs: Any) -> Any:
        return ambient_temperature_at_cell(self, *args, **kwargs)

    def ambient_temperature_region(self, *args: Any, **kwargs: Any) -> Any:
        return ambient_temperature_region(self, *args, **kwargs)

    def sample_flow_to_cells(self, *args: Any, **kwargs: Any) -> Any:
        return sample_flow_to_cells(self, *args, **kwargs)

    def downsample_cells_to_gas(self, *args: Any, **kwargs: Any) -> Any:
        return downsample_cells_to_gas(self, *args, **kwargs)

    def add_gas_from_cells(self, *args: Any, **kwargs: Any) -> Any:
        return add_gas_from_cells(self, *args, **kwargs)

    def set_cell_by_id(self, *args: Any, **kwargs: Any) -> Any:
        return set_cell_by_id(self, *args, **kwargs)

    def _inject_velocity_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _inject_velocity_immediate(self, *args, **kwargs)

    def _inject_temperature_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _inject_temperature_immediate(self, *args, **kwargs)

    def _inject_gas_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _inject_gas_immediate(self, *args, **kwargs)

    def set_cell(self, *args: Any, **kwargs: Any) -> Any:
        return set_cell(self, *args, **kwargs)

    def clear_cell(self, *args: Any, **kwargs: Any) -> Any:
        return clear_cell(self, *args, **kwargs)

    def clear_cells(self, *args: Any, **kwargs: Any) -> Any:
        return clear_cells(self, *args, **kwargs)

    def set_material_by_mask(self, *args: Any, **kwargs: Any) -> Any:
        return set_material_by_mask(self, *args, **kwargs)

    def swap_cells(self, *args: Any, **kwargs: Any) -> Any:
        return swap_cells(self, *args, **kwargs)

    def clear_cell_region(self, *args: Any, **kwargs: Any) -> Any:
        return clear_cell_region(self, *args, **kwargs)

    def serialize_local_cells(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_local_cells(self, *args, **kwargs)

    def serialize_temperature_window(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_temperature_window(self, *args, **kwargs)

    def serialize_gas(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_gas(self, *args, **kwargs)

    def serialize_pressure(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_pressure(self, *args, **kwargs)

    def serialize_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_velocity(self, *args, **kwargs)

    def serialize_visible_illumination(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_visible_illumination(self, *args, **kwargs)

    def serialize_gas_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_gas_runtime(self, *args, **kwargs)

    def serialize_heat_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_heat_runtime(self, *args, **kwargs)

    def serialize_liquid_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_liquid_runtime(self, *args, **kwargs)

    def serialize_reaction_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_reaction_runtime(self, *args, **kwargs)

    def serialize_collapse_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_collapse_runtime(self, *args, **kwargs)

    def serialize_optics_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_optics_runtime(self, *args, **kwargs)

    def serialize_active_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_active_runtime(self, *args, **kwargs)

    def serialize_motion_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_motion_runtime(self, *args, **kwargs)

    def serialize_paging_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_paging_state(self, *args, **kwargs)

    def serialize_engine_capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_engine_capabilities(self, *args, **kwargs)

    def serialize_material_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_material_table(self, *args, **kwargs)

    def _serialize_bridge_ndarray(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_ndarray(self, *args, **kwargs)

    @staticmethod
    def _bridge_row_count(*args: Any, **kwargs: Any) -> Any:
        return _bridge_row_count(*args, **kwargs)

    @staticmethod
    def _normalize_bridge_slice_bounds(*args: Any, **kwargs: Any) -> Any:
        return _normalize_bridge_slice_bounds(*args, **kwargs)

    @staticmethod
    def _normalize_bridge_window_bounds(*args: Any, **kwargs: Any) -> Any:
        return _normalize_bridge_window_bounds(*args, **kwargs)

    def _clamped_gas_window(self, *args: Any, **kwargs: Any) -> Any:
        return _clamped_gas_window(self, *args, **kwargs)

    def _bridge_shadow_buffer_coord_space(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_shadow_buffer_coord_space(self, *args, **kwargs)

    def _serialize_bridge_ndarray_slice(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_ndarray_slice(self, *args, **kwargs)

    def _serialize_bridge_ndarray_window(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_ndarray_window(self, *args, **kwargs)

    def _serialize_bridge_spatial_window_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_bridge_spatial_window_payload(self, *args, **kwargs)

    def serialize_bridge_typed_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_typed_table(self, *args, **kwargs)

    def serialize_bridge_typed_table_slice(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_typed_table_slice(self, *args, **kwargs)

    def serialize_bridge_shadow_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_shadow_buffer(self, *args, **kwargs)

    def serialize_bridge_shadow_buffer_slice(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_shadow_buffer_slice(self, *args, **kwargs)

    def serialize_bridge_shadow_buffer_window(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_shadow_buffer_window(self, *args, **kwargs)

    def serialize_bridge_shadow_buffer_world_window(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_shadow_buffer_world_window(self, *args, **kwargs)

    def serialize_bridge_shadow_buffer_gas_window(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_shadow_buffer_gas_window(self, *args, **kwargs)

    @staticmethod
    def _decode_bridge_uploaded_command(*args: Any, **kwargs: Any) -> Any:
        return _decode_bridge_uploaded_command(*args, **kwargs)

    @staticmethod
    def _decode_bridge_uploaded_label(*args: Any, **kwargs: Any) -> Any:
        return _decode_bridge_uploaded_label(*args, **kwargs)

    @staticmethod
    def _decode_bridge_uploaded_page_stripe_section(*args: Any, **kwargs: Any) -> Any:
        return _decode_bridge_uploaded_page_stripe_section(*args, **kwargs)

    @staticmethod
    def _set_nested_payload_value(*args: Any, **kwargs: Any) -> Any:
        return _set_nested_payload_value(*args, **kwargs)

    def serialize_bridge_upload_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_upload_snapshot(self, *args, **kwargs)

    def serialize_bridge_frame_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_bridge_frame_snapshot(self, *args, **kwargs)

    def _serialize_force_source_record(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_force_source_record(self, *args, **kwargs)

    def serialize_force_sources(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_force_sources(self, *args, **kwargs)

    def _serialize_emitter_record(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_emitter_record(self, *args, **kwargs)

    def serialize_emitters(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_emitters(self, *args, **kwargs)

    def serialize_gas_species_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_gas_species_table(self, *args, **kwargs)

    def serialize_light_type_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_light_type_table(self, *args, **kwargs)

    def serialize_material_optics_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_material_optics_table(self, *args, **kwargs)

    def serialize_reaction_table(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_reaction_table(self, *args, **kwargs)

    def serialize_optics(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_optics(self, *args, **kwargs)

    def serialize_readback_request(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_readback_request(self, *args, **kwargs)

    def _infer_readback_payload_coord_space(self, *args: Any, **kwargs: Any) -> Any:
        return _infer_readback_payload_coord_space(self, *args, **kwargs)

    def _serialize_readback_source_descriptor(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_source_descriptor(self, *args, **kwargs)

    def _serialize_readback_plan_for_request(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_plan_for_request(self, *args, **kwargs)

    def _serialize_readback_plans_for_requests(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_plans_for_requests(self, *args, **kwargs)

    def _serialize_observation_plan_for_target_request(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_observation_plan_for_target_request(self, *args, **kwargs)

    def _build_observation_request_pairs(
        self,
        targets: list[ObservationTarget],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[tuple[ObservationTarget, ReadbackRequest]]:
        return _build_observation_request_pairs(self, targets, resolved_targets)

    def serialize_readback_plan(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_readback_plan(self, *args, **kwargs)

    def serialize_observation_plan(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_observation_plan(self, *args, **kwargs)

    def serialize_world_command(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_world_command(self, *args, **kwargs)

    @staticmethod
    def serialize_entity_placeholder_input(*args: Any, **kwargs: Any) -> Any:
        return serialize_entity_placeholder_input(*args, **kwargs)

    @staticmethod
    def serialize_target_query_input(*args: Any, **kwargs: Any) -> Any:
        return serialize_target_query_input(*args, **kwargs)

    @staticmethod
    def serialize_page_stripe_update(*args: Any, **kwargs: Any) -> Any:
        return serialize_page_stripe_update(*args, **kwargs)

    @staticmethod
    def serialize_page_store_key(*args: Any, **kwargs: Any) -> Any:
        return serialize_page_store_key(*args, **kwargs)

    @classmethod
    def serialize_page_stripe_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return serialize_page_stripe_payload(cls, payload)

    @staticmethod
    def serialize_change_intent_input(*args: Any, **kwargs: Any) -> Any:
        return serialize_change_intent_input(*args, **kwargs)

    @staticmethod
    def serialize_carrier_intent_input(*args: Any, **kwargs: Any) -> Any:
        return serialize_carrier_intent_input(*args, **kwargs)

    def serialize_frame_input(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_frame_input(self, *args, **kwargs)

    def _serialize_readback_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_payload(self, *args, **kwargs)

    def serialize_readback_result(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_readback_result(self, *args, **kwargs)

    def serialize_resolved_target(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_resolved_target(self, *args, **kwargs)

    def serialize_resolved_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_resolved_change_intent(self, *args, **kwargs)

    def serialize_resolved_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_resolved_carrier_intent(self, *args, **kwargs)

    def serialize_observation_result(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_observation_result(self, *args, **kwargs)

    @staticmethod
    def serialize_entity_observation_spec(*args: Any, **kwargs: Any) -> Any:
        return serialize_entity_observation_spec(*args, **kwargs)

    def serialize_entity_state_patch(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_state_patch(self, *args, **kwargs)

    @staticmethod
    def serialize_observation_target(*args: Any, **kwargs: Any) -> Any:
        return serialize_observation_target(*args, **kwargs)

    @staticmethod
    def serialize_entity_state_input(*args: Any, **kwargs: Any) -> Any:
        return serialize_entity_state_input(*args, **kwargs)

    def serialize_entity_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_state(self, *args, **kwargs)

    def serialize_entity_states(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_states(self, *args, **kwargs)

    def serialize_entity_observation_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_observation_state(self, *args, **kwargs)

    def _current_cell_state_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return _current_cell_state_snapshot(self, *args, **kwargs)

    def _current_entity_runtime_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return _current_entity_runtime_snapshot(self, *args, **kwargs)

    def _entity_placeholder_state_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return _entity_placeholder_state_gpu_authoritative(self, *args, **kwargs)

    def serialize_entity_placeholders(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_placeholders(self, *args, **kwargs)

    def serialize_entity_placeholder_index_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_placeholder_index_snapshot(self, *args, **kwargs)

    def serialize_entity_feedback_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_feedback_snapshot(self, *args, **kwargs)

    def serialize_consumed_entity_feedback_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_consumed_entity_feedback_snapshot(self, *args, **kwargs)

    def _serialize_cpu_visible_entity_placeholders(self, *args: Any, **kwargs: Any) -> Any:
        return _serialize_cpu_visible_entity_placeholders(self, *args, **kwargs)

    def serialize_entity_feedback(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_feedback(self, *args, **kwargs)

    def _store_entity_observation_consume_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return _store_entity_observation_consume_snapshot(self, *args, **kwargs)

    def serialize_entity_observation_consume_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_entity_observation_consume_state(self, *args, **kwargs)

    def serialize_frame_output(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_frame_output(self, *args, **kwargs)

    def serialize_frame_preview(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_frame_preview(self, *args, **kwargs)

    def serialize_debug_frame(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_debug_frame(self, *args, **kwargs)

    def debug_frame(self, *args: Any, **kwargs: Any) -> Any:
        return debug_frame(self, *args, **kwargs)

    def _material_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _material_frame(self, *args, **kwargs)

    def _temperature_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _temperature_frame(self, *args, **kwargs)

    def _pressure_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _pressure_frame(self, *args, **kwargs)

    def _vector_field_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _vector_field_frame(self, *args, **kwargs)

    def _active_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _active_frame(self, *args, **kwargs)

    def _motion_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _motion_frame(self, *args, **kwargs)

    def _heat_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _heat_frame(self, *args, **kwargs)

    def _liquid_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _liquid_frame(self, *args, **kwargs)

    def _reaction_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _reaction_frame(self, *args, **kwargs)

    def _collapse_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _collapse_frame(self, *args, **kwargs)

    def _optics_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _optics_frame(self, *args, **kwargs)

    def _optics_dose_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _optics_dose_frame(self, *args, **kwargs)

    def _gas_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_frame(self, *args, **kwargs)

    def _accumulate_debug_point(self, *args: Any, **kwargs: Any) -> Any:
        return _accumulate_debug_point(self, *args, **kwargs)

    def _draw_debug_bbox_outline(self, *args: Any, **kwargs: Any) -> Any:
        return _draw_debug_bbox_outline(self, *args, **kwargs)

    def _apply_grid_world_commands(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_grid_world_commands(self, *args, **kwargs)

    def _apply_grid_world_command_cpu(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_grid_world_command_cpu(self, *args, **kwargs)

    def _grid_world_command_runtime_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _grid_world_command_runtime_regions(self, *args, **kwargs)

    def _mark_grid_world_command_runtime_regions(self, command: WorldCommand) -> None:
        return _mark_grid_world_command_runtime_regions(self, command)

    def _apply_commands(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_commands(self, *args, **kwargs)

    def _finish_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return _finish_readbacks(self, *args, **kwargs)

    def _collect_ready_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return _collect_ready_readbacks(self, *args, **kwargs)

    def _queued_command_xy(self, command: WorldCommand) -> tuple[int, int]:
        return _queued_command_xy(self, command)

    def _make_readback_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _make_readback_payload(self, *args, **kwargs)

    def _gas_window_for_cell_rect(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_window_for_cell_rect(self, *args, **kwargs)

    def _apply_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_page_stripe(self, *args, **kwargs)

    def _apply_page_stripe_dense_cpu(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_page_stripe_dense_cpu(self, *args, **kwargs)

    def _sync_loaded_page_stripe_cpu_mirror(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_loaded_page_stripe_cpu_mirror(self, *args, **kwargs)

    def _queue_loaded_collapse_pending_regions_from_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _queue_loaded_collapse_pending_regions_from_payload(self, *args, **kwargs)

    def _advance_paging(self, *args: Any, **kwargs: Any) -> Any:
        return _advance_paging(self, *args, **kwargs)

    def _prepare_bridge_frame_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _prepare_bridge_frame_inputs(self, *args, **kwargs)

    def _needs_pre_simulation_bridge_sync(self, *args: Any, **kwargs: Any) -> Any:
        return _needs_pre_simulation_bridge_sync(self, *args, **kwargs)

    def _sync_pre_simulation_bridge_without_debug_upload(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_pre_simulation_bridge_without_debug_upload(self, *args, **kwargs)

    def _clear_bridge_frame_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_bridge_frame_inputs(self, *args, **kwargs)

    def _mark_active_rect_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_active_rect_runtime(self, *args, **kwargs)

    def _mark_active_rects_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_active_rects_runtime(self, *args, **kwargs)

    def _sync_entity_placeholders(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_entity_placeholders(self, *args, **kwargs)

    def _sync_force_sources(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_force_sources(self, *args, **kwargs)

    def _append_force_source_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _append_force_source_immediate(self, *args, **kwargs)

    def _sync_persistent_emitters(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_persistent_emitters(self, *args, **kwargs)

    def _append_transient_light_emitter_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _append_transient_light_emitter_immediate(self, *args, **kwargs)

    def _record_bridge_page_stripe(self, *args: Any, **kwargs: Any) -> Any:
        return _record_bridge_page_stripe(self, *args, **kwargs)

    def _release_entity_placeholder_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _release_entity_placeholder_cell(self, *args, **kwargs)

    def _mirror_release_entity_placeholder_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _mirror_release_entity_placeholder_cell(self, *args, **kwargs)

    def _resolve_sanctioned_material_id(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_sanctioned_material_id(self, *args, **kwargs)

    def _shadow_material_id_by_name(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_id_by_name(self, *args, **kwargs)

    def _resolve_sanctioned_placeholder_material_id(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_sanctioned_placeholder_material_id(self, *args, **kwargs)

    def _resolve_sanctioned_light_id(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_sanctioned_light_id(self, *args, **kwargs)

    def _resolve_sanctioned_gas_id(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_sanctioned_gas_id(self, *args, **kwargs)

    def _shadow_material_row_valid(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_row_valid(self, *args, **kwargs)

    def _shadow_gas_row_valid(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_gas_row_valid(self, *args, **kwargs)

    def _shadow_light_row_valid(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_row_valid(self, *args, **kwargs)

    def _shadow_material_def(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_def(self, *args, **kwargs)

    def _shadow_light_type_def(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_type_def(self, *args, **kwargs)

    def _shadow_gas_species_def(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_gas_species_def(self, *args, **kwargs)

    def _shadow_material_optics_def(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_optics_def(self, *args, **kwargs)

    def _shadow_material_name(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_name(self, *args, **kwargs)

    def _shadow_gas_name(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_gas_name(self, *args, **kwargs)

    def _shadow_light_name(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_name(self, *args, **kwargs)

    def _shadow_light_default_range(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_default_range(self, *args, **kwargs)

    def _shadow_light_dose_channel(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_dose_channel(self, *args, **kwargs)

    def _shadow_light_color(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_color(self, *args, **kwargs)

    def _shadow_light_name_and_range(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_light_name_and_range(self, *args, **kwargs)

    def _shadow_material_default_phase(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_default_phase(self, *args, **kwargs)

    def _shadow_material_base_integrity(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_base_integrity(self, *args, **kwargs)

    def _shadow_material_spawn_temperature(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_spawn_temperature(self, *args, **kwargs)

    def _shadow_condense_target_material_id(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_condense_target_material_id(self, *args, **kwargs)

    def _shadow_material_is_placeholder(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_is_placeholder(self, *args, **kwargs)

    def _material_placeholder_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _material_placeholder_mask(self, *args, **kwargs)

    def _shadow_material_is_plant(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_material_is_plant(self, *args, **kwargs)

    def _shadow_reaction_action(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_reaction_action(self, *args, **kwargs)

    def _reaction_rule_list(self, *args: Any, **kwargs: Any) -> Any:
        return _reaction_rule_list(self, *args, **kwargs)

    def _set_reaction_rule_list(self, *args: Any, **kwargs: Any) -> Any:
        return _set_reaction_rule_list(self, *args, **kwargs)

    def _set_reaction_rules_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _set_reaction_rules_payload(self, *args, **kwargs)

    @staticmethod
    def _remap_reaction_payload_result_actions(*args: Any, **kwargs: Any) -> Any:
        return _remap_reaction_payload_result_actions(*args, **kwargs)

    @staticmethod
    def _remap_material_payload_reaction_slots(*args: Any, **kwargs: Any) -> Any:
        return _remap_material_payload_reaction_slots(*args, **kwargs)

    @staticmethod
    def _clamp_material_payload_reaction_slots(*args: Any, **kwargs: Any) -> Any:
        return _clamp_material_payload_reaction_slots(*args, **kwargs)

    def _shadow_reaction_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_reaction_rule(self, *args, **kwargs)

    def _occupy_entity_placeholder_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _occupy_entity_placeholder_cell(self, *args, **kwargs)

    def _mirror_occupy_entity_placeholder_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _mirror_occupy_entity_placeholder_cell(self, *args, **kwargs)

    def _frame_entities_to_placeholders_and_observations(self, *args: Any, **kwargs: Any) -> Any:
        return _frame_entities_to_placeholders_and_observations(self, *args, **kwargs)

    def _runtime_entities_to_immediate_observation_targets(self, *args: Any, **kwargs: Any) -> Any:
        return _runtime_entities_to_immediate_observation_targets(self, *args, **kwargs)

    def _sync_entity_states(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_entity_states(self, *args, **kwargs)

    def _sync_entity_observation_specs(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_entity_observation_specs(self, *args, **kwargs)

    def _patch_entity_states(self, *args: Any, **kwargs: Any) -> Any:
        return _patch_entity_states(self, *args, **kwargs)

    def _build_preview_entity_placeholders(self, *args: Any, **kwargs: Any) -> Any:
        return _build_preview_entity_placeholders(self, *args, **kwargs)

    def _preview_can_occupy_placeholder_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _preview_can_occupy_placeholder_cell(self, *args, **kwargs)

    def _material_state_for_position(self, *args: Any, **kwargs: Any) -> Any:
        return _material_state_for_position(self, *args, **kwargs)

    def _build_observation_requests(self, *args: Any, **kwargs: Any) -> Any:
        return _build_observation_requests(self, *args, **kwargs)

    def _build_observation_request(self, *args: Any, **kwargs: Any) -> Any:
        return _build_observation_request(self, *args, **kwargs)

    def _resolve_readback_requests(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_readback_requests(self, *args, **kwargs)

    def _resolve_readback_request(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_readback_request(self, *args, **kwargs)

    def _resolve_target_queries(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_target_queries(self, *args, **kwargs)

    def _resolve_change_intents(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_change_intents(self, *args, **kwargs)

    def _public_resolved_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _public_resolved_change_intent(self, *args, **kwargs)

    def _public_resolved_target(self, *args: Any, **kwargs: Any) -> Any:
        return _public_resolved_target(self, *args, **kwargs)

    def _resolve_carrier_intents(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_carrier_intents(self, *args, **kwargs)

    def _public_resolved_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _public_resolved_carrier_intent(self, *args, **kwargs)

    def _resolve_carrier_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_carrier_intent(self, *args, **kwargs)

    def _resolve_change_intent(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_change_intent(self, *args, **kwargs)

    def _resolve_change_intent_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_change_intent_world_position(self, *args, **kwargs)

    def _resolve_intent_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_intent_world_position(self, *args, **kwargs)

    def _resolve_intent_source_positions(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_intent_source_positions(self, *args, **kwargs)

    @staticmethod
    def _normalized_world_direction(*args: Any, **kwargs: Any) -> Any:
        return _normalized_world_direction(*args, **kwargs)

    def _disk_world_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _disk_world_cells(self, *args, **kwargs)

    @staticmethod
    def _disk_world_cells_raw(*args: Any, **kwargs: Any) -> Any:
        return _disk_world_cells_raw(*args, **kwargs)

    def _line_world_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _line_world_cells(self, *args, **kwargs)

    @staticmethod
    def _line_world_cells_raw(*args: Any, **kwargs: Any) -> Any:
        return _line_world_cells_raw(*args, **kwargs)

    def _capsule_world_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _capsule_world_cells(self, *args, **kwargs)

    def _capsule_world_cells_raw(self, *args: Any, **kwargs: Any) -> Any:
        return _capsule_world_cells_raw(self, *args, **kwargs)

    @staticmethod
    def _buffer_cell_bounds(*args: Any, **kwargs: Any) -> Any:
        return _buffer_cell_bounds(*args, **kwargs)

    def _apply_change_stability_drift(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_change_stability_drift(self, *args, **kwargs)

    def _resolve_legal_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_legal_world_position(self, *args, **kwargs)

    @staticmethod
    def _intent_resolution_status(*args: Any, **kwargs: Any) -> Any:
        return _intent_resolution_status(*args, **kwargs)

    @staticmethod
    def _combine_resolution_notes(*args: Any, **kwargs: Any) -> Any:
        return _combine_resolution_notes(*args, **kwargs)

    def _resolve_targeted_commands(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_targeted_commands(self, *args, **kwargs)

    def _resolve_target_query(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_target_query(self, *args, **kwargs)

    def _resolve_target_query_distance_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_target_query_distance_cells(self, *args, **kwargs)

    @staticmethod
    def _distance_meters_to_cells(*args: Any, **kwargs: Any) -> Any:
        return _distance_meters_to_cells(*args, **kwargs)

    def _resolve_query_source_position(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_query_source_position(self, *args, **kwargs)

    def _default_target_source_position(self, *args: Any, **kwargs: Any) -> Any:
        return _default_target_source_position(self, *args, **kwargs)

    def _resolve_anchor_target(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_anchor_target(self, *args, **kwargs)

    def _resolve_entity_anchor(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_entity_anchor(self, *args, **kwargs)

    def _resolve_terrain_anchor(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_terrain_anchor(self, *args, **kwargs)

    def _entity_matches_anchor_filters(self, *args: Any, **kwargs: Any) -> Any:
        return _entity_matches_anchor_filters(self, *args, **kwargs)

    def _terrain_cell_matches(self, *args: Any, **kwargs: Any) -> Any:
        return _terrain_cell_matches(self, *args, **kwargs)

    def _terrain_tree_cell_matches(self, *args: Any, **kwargs: Any) -> Any:
        return _terrain_tree_cell_matches(self, *args, **kwargs)

    def _terrain_hill_cell_matches(self, *args: Any, **kwargs: Any) -> Any:
        return _terrain_hill_cell_matches(self, *args, **kwargs)

    def _world_cell_is_solid_local(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_is_solid_local(self, *args, **kwargs)

    def _world_cell_is_empty_local(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_is_empty_local(self, *args, **kwargs)

    def _world_cell_material_has_tag(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_material_has_tag(self, *args, **kwargs)

    def _bounded_material_state_for_position(self, *args: Any, **kwargs: Any) -> Any:
        return _bounded_material_state_for_position(self, *args, **kwargs)

    def _matches_direction_filter(self, *args: Any, **kwargs: Any) -> Any:
        return _matches_direction_filter(self, *args, **kwargs)

    def _query_direction_vector(self, *args: Any, **kwargs: Any) -> Any:
        return _query_direction_vector(self, *args, **kwargs)

    def _direction_vector(self, *args: Any, **kwargs: Any) -> Any:
        return _direction_vector(self, *args, **kwargs)

    def _source_facing_vector(self, *args: Any, **kwargs: Any) -> Any:
        return _source_facing_vector(self, *args, **kwargs)

    def _entity_center_buffer_position(self, *args: Any, **kwargs: Any) -> Any:
        return _entity_center_buffer_position(self, *args, **kwargs)

    def _entity_center_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _entity_center_world_position(self, *args, **kwargs)

    def _buffer_to_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _buffer_to_world_position(self, *args, **kwargs)

    def _buffer_to_world_float_position(self, *args: Any, **kwargs: Any) -> Any:
        return _buffer_to_world_float_position(self, *args, **kwargs)

    def _world_to_buffer_float_position(self, *args: Any, **kwargs: Any) -> Any:
        return _world_to_buffer_float_position(self, *args, **kwargs)

    def _force_source_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _force_source_world_position(self, *args, **kwargs)

    def _force_source_buffer_position(self, *args: Any, **kwargs: Any) -> Any:
        return _force_source_buffer_position(self, *args, **kwargs)

    def _normalize_runtime_force_source(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_runtime_force_source(self, *args, **kwargs)

    def _buffer_gas_to_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _buffer_gas_to_world_position(self, *args, **kwargs)

    def _buffer_bbox_to_world_bbox(self, *args: Any, **kwargs: Any) -> Any:
        return _buffer_bbox_to_world_bbox(self, *args, **kwargs)

    def _clamped_world_window(self, *args: Any, **kwargs: Any) -> Any:
        return _clamped_world_window(self, *args, **kwargs)

    def _centered_world_window(self, *args: Any, **kwargs: Any) -> Any:
        return _centered_world_window(self, *args, **kwargs)

    def _world_axis_spans(self, *args: Any, **kwargs: Any) -> Any:
        return _world_axis_spans(self, *args, **kwargs)

    def _world_axis_indices(self, *args: Any, **kwargs: Any) -> Any:
        return _world_axis_indices(self, *args, **kwargs)

    def _extract_world_window(self, *args: Any, **kwargs: Any) -> Any:
        return _extract_world_window(self, *args, **kwargs)

    def _world_gas_window_for_cell_world_rect(self, *args: Any, **kwargs: Any) -> Any:
        return _world_gas_window_for_cell_world_rect(self, *args, **kwargs)

    def _pack_cell_core_world_window(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_cell_core_world_window(self, *args, **kwargs)

    def _world_to_buffer_clamped(self, *args: Any, **kwargs: Any) -> Any:
        return _world_to_buffer_clamped(self, *args, **kwargs)

    def _clamp_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _clamp_world_position(self, *args, **kwargs)

    def _find_nearest_empty_world_position(self, *args: Any, **kwargs: Any) -> Any:
        return _find_nearest_empty_world_position(self, *args, **kwargs)

    def _world_cell_is_empty(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_is_empty(self, *args, **kwargs)

    @staticmethod
    def _world_distance_sq(*args: Any, **kwargs: Any) -> Any:
        return _world_distance_sq(*args, **kwargs)

    def _entity_placeholder_bbox(self, *args: Any, **kwargs: Any) -> Any:
        return _entity_placeholder_bbox(self, *args, **kwargs)

    def _collect_observations(self, *args: Any, **kwargs: Any) -> Any:
        return _collect_observations(self, *args, **kwargs)

    def _collect_entity_feedback(self, *args: Any, **kwargs: Any) -> Any:
        return _collect_entity_feedback(self, *args, **kwargs)

    def _build_entity_feedback(self, *args: Any, **kwargs: Any) -> Any:
        return _build_entity_feedback(self, *args, **kwargs)

    def _build_entity_feedback_from_world(self, *args: Any, **kwargs: Any) -> Any:
        return _build_entity_feedback_from_world(self, *args, **kwargs)

    def _build_entity_feedback_from_current_state(self, *args: Any, **kwargs: Any) -> Any:
        return _build_entity_feedback_from_current_state(self, *args, **kwargs)

    def _build_entity_feedback_from_state(self, *args: Any, **kwargs: Any) -> Any:
        return _build_entity_feedback_from_state(self, *args, **kwargs)

    def _capture_stripe_array(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_stripe_array(self, *args, **kwargs)

    def _write_stripe_array(self, *args: Any, **kwargs: Any) -> Any:
        return _write_stripe_array(self, *args, **kwargs)

    def _default_page_stripe_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _default_page_stripe_payload(self, *args, **kwargs)

    def _stripe_buffer_ranges(self, *args: Any, **kwargs: Any) -> Any:
        return _stripe_buffer_ranges(self, *args, **kwargs)

    def _mark_loaded_page_stripe_active(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_loaded_page_stripe_active(self, *args, **kwargs)

    def _rebuild_sparse_runtime_indexes(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_sparse_runtime_indexes(self, *args, **kwargs)

    def _rebuild_entity_placeholder_index(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_entity_placeholder_index(self, *args, **kwargs)

    def _normalize_cell_runtime_arrays(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_cell_runtime_arrays(self, *args, **kwargs)

    def _normalize_page_stripe_cell_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_page_stripe_cell_runtime(self, *args, **kwargs)

    def _capture_page_stripe_entity_placeholder_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_page_stripe_entity_placeholder_runtime(self, *args, **kwargs)

    def _apply_page_stripe_entity_placeholder_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_page_stripe_entity_placeholder_runtime(self, *args, **kwargs)

    def _rebuild_island_records(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_island_records(self, *args, **kwargs)

    def _capture_page_stripe_island_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_page_stripe_island_runtime(self, *args, **kwargs)

    def _page_stripe_island_bboxes_from_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _page_stripe_island_bboxes_from_payload(self, *args, **kwargs)

    def _merge_island_runtime_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _merge_island_runtime_payload(self, *args, **kwargs)

    def _rebuild_material_property_arrays(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_material_property_arrays(self, *args, **kwargs)

    def _rebuild_gas_property_arrays(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_gas_property_arrays(self, *args, **kwargs)

    def _rebuild_light_property_arrays(self, *args: Any, **kwargs: Any) -> Any:
        return _rebuild_light_property_arrays(self, *args, **kwargs)

    def _cell_participates_in_collapse(self, *args: Any, **kwargs: Any) -> Any:
        return _cell_participates_in_collapse(self, *args, **kwargs)

    def _mark_collapse_dirty_rect(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_collapse_dirty_rect(self, *args, **kwargs)

    def _drain_gpu_collapse_structure_dirty_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _drain_gpu_collapse_structure_dirty_tiles(self, *args, **kwargs)

    def _paint_material(self, *args: Any, **kwargs: Any) -> Any:
        return _paint_material(self, *args, **kwargs)

    def _write_material_region_immediate(self, *args: Any, **kwargs: Any) -> Any:
        return _write_material_region_immediate(self, *args, **kwargs)

    def _build_demo_scene(self, *args: Any, **kwargs: Any) -> Any:
        return _build_demo_scene(self, *args, **kwargs)

    def _fill_rect(self, *args: Any, **kwargs: Any) -> Any:
        return _fill_rect(self, *args, **kwargs)
