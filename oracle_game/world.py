from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from enum import Enum
import threading
import time
from typing import Any

from oracle_game.gpu import (
    GPUBufferReadbackSource, GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource, GPUReadbackSegment, GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource, GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource
)
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.page_store import PageStore, StoredStripeKey
from oracle_game.types import (
    CarrierIntent, CarrierIntent, ChangeIntent, DebugView, EntityCellFeedback,
    EntityFeedback, EntityObservationSpec, EntityPlaceholder, EntityStatePatch, EntityState,
    ForceSource, GasSpeciesDef, LightTypeDef, MaterialDef, MaterialOpticsDef,
    ObservationTarget, PageStripeUpdate, PairReactionRule, Phase, ReactionAction,
    ReadbackRequest, ReadbackResult, ResolvedCarrierIntent, ResolvedChangeIntent,
    ResolvedTarget, SelfReactionRule, TargetQuery, WorldFrameInput, WorldFrameOutput,
    WorldFramePreview, WorldCommand
)

from oracle_game.world_constants import (
    BASE_MATERIAL_RUNTIME_ALIASES, ENTITY_STATE_PATCH_METADATA_FIELDS,
    TARGET_QUERY_DISTANCE_HINT_CELLS, UNSET_CONTROLLER_STATE
)
from oracle_game.world_engine_init import _init_world_engine
from oracle_game.world_capabilities import serialize_engine_capabilities
from oracle_game.world_payload_serializers import (
    _infer_readback_payload_coord_space, _serialize_cpu_visible_entity_placeholders,
    _serialize_emitter_record, _serialize_force_source_record,
    _serialize_observation_plan_for_target_request, _serialize_preview_bridge_frame_snapshot,
    _serialize_readback_payload, _serialize_readback_plan_for_request,
    _serialize_readback_plans_for_requests, _serialize_readback_source_descriptor,
    serialize_carrier_intent_input, serialize_change_intent_input,
    serialize_consumed_entity_feedback_snapshot, serialize_controller_state,
    serialize_debug_frame, serialize_emitters, serialize_entity_feedback,
    serialize_entity_feedback_snapshot, serialize_entity_observation_consume_state,
    serialize_entity_observation_spec, serialize_entity_observation_state,
    serialize_entity_placeholder_index_snapshot, serialize_entity_placeholder_input,
    serialize_entity_placeholders, serialize_entity_state, serialize_entity_state_input,
    serialize_entity_state_patch, serialize_entity_states, serialize_force_sources,
    serialize_frame_input, serialize_frame_output, serialize_frame_preview,
    serialize_frame_state, serialize_gas, serialize_gas_species_table,
    serialize_light_type_table, serialize_local_cells, serialize_material_optics_table,
    serialize_material_table, serialize_observation_plan, serialize_observation_result,
    serialize_observation_target, serialize_optics, serialize_page_store_key,
    serialize_page_store_state, serialize_page_stripe_payload, serialize_page_stripe_update,
    serialize_pending_commands, serialize_pending_frame_detail,
    serialize_pending_frame_inputs, serialize_pressure, serialize_reaction_table,
    serialize_readback_plan, serialize_readback_request, serialize_readback_result,
    serialize_readback_state, serialize_ready_frame_outputs, serialize_ready_readbacks,
    serialize_resolved_carrier_intent, serialize_resolved_change_intent,
    serialize_resolved_target, serialize_target_query_input, serialize_temperature_window,
    serialize_velocity, serialize_visible_illumination, serialize_world_command
)
from oracle_game.world_debug_frame import (
    _accumulate_debug_point, _active_frame, _collapse_frame, _draw_debug_bbox_outline,
    _gas_frame, _heat_frame, _liquid_frame, _material_frame, _motion_frame,
    _optics_dose_frame, _optics_frame, _pressure_frame, _reaction_frame, _temperature_frame,
    _vector_field_frame, debug_frame
)
from oracle_game.world_readback_payload import make_readback_payload as _make_readback_payload
from oracle_game.world_runtime_serializers import (
    serialize_active_runtime, serialize_collapse_runtime, serialize_gas_runtime,
    serialize_heat_runtime, serialize_liquid_runtime, serialize_motion_runtime,
    serialize_optics_runtime, serialize_paging_state, serialize_reaction_runtime
)
from oracle_game.world_geometry import (
    _apply_change_stability_drift, _bounded_material_state_for_position,
    _buffer_bbox_to_world_bbox, _buffer_cell_bounds, _buffer_gas_to_world_position,
    _buffer_to_world_float_position, _buffer_to_world_position, _capsule_world_cells,
    _capsule_world_cells_raw, _centered_world_window, _clamped_world_window,
    _direction_vector, _disk_world_cells, _disk_world_cells_raw, _extract_world_window,
    _find_nearest_empty_world_position, _force_source_buffer_position,
    _force_source_world_position, _line_world_cells, _line_world_cells_raw,
    _matches_direction_filter, _pack_cell_core_world_window, _query_direction_vector,
    _resolve_entity_anchor, _resolve_legal_world_position, _resolve_terrain_anchor,
    _terrain_cell_matches, _world_axis_indices, _world_axis_spans, _world_cell_is_empty,
    _world_cell_is_empty_local, _world_cell_is_solid_local, _world_to_buffer_clamped,
    _world_to_buffer_float_position
)
from oracle_game.world_command_queue import (
    _public_resolved_carrier_intent, _public_world_command, _resolve_direct_targeted_coords,
    _resolve_public_world_command, inject_force, inject_gas, inject_light, inject_material,
    inject_temperature, inject_velocity, preview_carrier_intent, preview_change_intent,
    preview_observation, preview_readback, preview_target_queries, preview_world_command,
    queue_command, request_carrier_intent, request_change_intent, request_observation,
    request_readback, request_world_command, write_material_region
)
from oracle_game.world_entity_sync import (
    sync_entity_placeholders, sync_entity_states, patch_entity_states,
    sync_entity_observation_specs, set_force_sources, set_emitters,
    consume_entity_observation_results, _append_force_source_immediate,
    _append_transient_light_emitter_immediate, _mirror_occupy_entity_placeholder_cell,
    _preview_consume_entity_observation_results, _sync_entity_placeholders,
    _sync_force_sources, _sync_persistent_emitters,
    _sync_pre_simulation_bridge_without_debug_upload, _release_entity_placeholder_cell,
    _occupy_entity_placeholder_cell, _frame_entities_to_placeholders_and_observations,
    _sync_entity_states, _sync_entity_observation_specs, _build_preview_entity_placeholders,
    _build_observation_request, _resolve_readback_request, _collect_observations,
    _collect_entity_feedback, _build_entity_feedback, _build_entity_feedback_from_state
)
from oracle_game.world_frame_io import (
    preview_frame_input, submit_frame_input, request_frame_input, request_frame_cycle,
    pending_frame_submission_ids, cancel_frame_submission, cancel_readback_request,
    poll_frame_output, poll_all_frame_outputs, frame_submission_status, _apply_frame_input,
    _prepare_preview_frame_context, _prepare_bridge_frame_inputs,
    _needs_pre_simulation_bridge_sync, _clear_bridge_frame_inputs
)
from oracle_game.world_table_api import (
    delete_reaction_action, delete_reaction_rule, patch_gas, patch_light, patch_material,
    patch_material_optics, patch_reaction_action, patch_reaction_rule,
    replace_reaction_table, reset_world, update_gas_species_table, update_light_type_table,
    update_material_optics_table, update_material_table, update_reaction_table,
    _reset_world_state
)
from oracle_game.world_paging import (
    _apply_page_stripe, _apply_page_stripe_dense_cpu, _capture_page_stripe_cpu_snapshot,
    _capture_stripe_array, _clear_saved_page_stripe_runtime_state, _coerce_page_store_key,
    _coerce_page_stripe_payload, _contextualize_page_stripe_update,
    _default_page_stripe_payload, _mark_loaded_page_stripe_active, _page_store_key,
    _preview_apply_paging_updates, _prune_page_stripe_regions, _stripe_buffer_ranges,
    _write_stripe_array, advance_paging, apply_page_stripe, apply_stored_page_stripe,
    capture_page_stripe, capture_page_stripe_to_store, clear_page_store,
    export_page_store_entries, focus_paging, import_page_store_entries,
    list_page_store_stripe_keys, load_page_stripe, page_store_has_stripe, poll_all_readbacks,
    poll_readbacks, store_page_stripe
)
from oracle_game.world_intent_resolver import (
    _resolve_carrier_intent, _resolve_change_intent, _resolve_change_intent_world_position,
    _resolve_target_query, _resolve_target_query_distance_cells, _resolve_target_queries
)
from oracle_game.world_cell_mutators import (
    _gas_field_count, _gas_window_for_cell_rect, _inject_gas_immediate,
    _inject_temperature_immediate, _inject_velocity_immediate, add_gas_from_cells,
    allocate_island_id, ambient_temperature_at_cell, ambient_temperature_region, cell_to_gas,
    cell_xy_to_gas, clear_cell, clear_cell_region, clear_cells, in_bounds, material_by_id,
    sample_ambient_to_cells, sample_flow_to_cells, set_cell, set_cell_by_id,
    set_material_by_mask, swap_cells
)
from oracle_game.world_input_coercion import (
    _assign_preview_readback_request_ids, _assign_readback_request_id,
    _canonical_material_input_name, _coerce_carrier_intent, _coerce_change_intent,
    _coerce_entity_observation_spec, _coerce_entity_placeholder, _coerce_entity_state,
    _coerce_entity_state_patch, _coerce_emitter, _coerce_enum, _coerce_force_source, _coerce_gas_species_def,
    _coerce_json_value, _coerce_light_type_def, _coerce_material_def,
    _coerce_material_optics_def, _coerce_observation_target, _coerce_pair_reaction_rule,
    _coerce_readback_request, _coerce_reaction_action, _coerce_reaction_rules,
    _coerce_self_reaction_rule, _coerce_target_query, _coerce_world_command,
    _coerce_world_frame_input, _controller_turn_entity_input, _frame_emitter_input,
    _frame_entity_placeholder_input, _frame_entity_state_input,
    _frame_entity_state_patch_input, _frame_force_source_input,
    _normalize_entity_state_patch_fields, _normalize_gas_patch_fields,
    _normalize_json_payload_value, _normalize_material_optics_patch_fields,
    _normalize_material_patch_fields, _normalize_readback_channels,
    _normalize_readback_request, _normalize_reaction_action_patch_fields,
    _normalize_reaction_rule_patch_fields, _public_entity_placeholder_input,
    _public_entity_state_input, _public_entity_state_patch_input, _public_force_source_input
)
from oracle_game.world_table_validation import (
    _clamp_material_payload_reaction_slots, _gas_species_table_snapshot_payload,
    _light_type_table_snapshot_payload, _material_optics_table_snapshot_payload,
    _material_placeholder_mask, _material_table_snapshot_payload,
    _merged_gas_species_table_payload, _merged_light_type_table_payload,
    _merged_material_optics_table_payload, _merged_material_table_payload,
    _merged_reaction_table_payload, _payload_name_set, _reaction_table_snapshot_payload,
    _remap_material_payload_reaction_slots, _remap_reaction_payload_result_actions,
    _set_reaction_rule_list, _set_reaction_rules_payload, _set_stable_shadow_payload,
    _shadow_gas_species_payload, _shadow_has_table_payload, _shadow_light_type_payload,
    _shadow_material_payload, _shadow_reaction_payload, _stable_shadow_payload,
    _validate_gas_species_payload, _validate_light_type_payload,
    _validate_material_optics_payload, _validate_material_table_payload,
    _validate_named_reference, _validate_reaction_payload, _validate_unique_identity_fields
)
from oracle_game.world_shadow_tables import (
    _reaction_rule_list, _resolve_sanctioned_gas_id, _resolve_sanctioned_light_id,
    _resolve_sanctioned_material_id, _resolve_sanctioned_placeholder_material_id,
    _shadow_condense_target_material_id, _shadow_gas_name, _shadow_gas_row_valid,
    _shadow_gas_species_def, _shadow_light_color, _shadow_light_default_range,
    _shadow_light_dose_channel, _shadow_light_name, _shadow_light_name_and_range,
    _shadow_light_row_valid, _shadow_light_type_def, _shadow_material_base_integrity,
    _shadow_material_default_phase, _shadow_material_def, _shadow_material_id_by_name,
    _shadow_material_is_placeholder, _shadow_material_is_plant, _shadow_material_name,
    _shadow_material_optics_def, _shadow_material_row_valid,
    _shadow_material_spawn_temperature, _shadow_reaction_action, _shadow_reaction_rule
)
from oracle_game.world_runtime_rebuild import (
    _apply_page_stripe_entity_placeholder_runtime,
    _capture_page_stripe_entity_placeholder_runtime, _capture_page_stripe_island_runtime,
    _cell_participates_in_collapse, _drain_gpu_collapse_structure_dirty_tiles,
    _mark_collapse_dirty_rect, _merge_island_runtime_payload, _normalize_cell_runtime_arrays,
    _normalize_page_stripe_cell_runtime, _rebuild_entity_placeholder_index,
    _rebuild_gas_property_arrays, _rebuild_island_records, _rebuild_light_property_arrays,
    _rebuild_material_property_arrays, _rebuild_sparse_runtime_indexes
)
from oracle_game.world_command_application import (
    _apply_commands, _apply_grid_world_command_cpu, _apply_grid_world_commands,
    _grid_world_command_runtime_regions, _queue_loaded_collapse_pending_regions,
    _queue_loaded_collapse_pending_regions_from_payload, _resolve_targeted_commands,
    _subtract_page_stripe_range_from_region
)
from oracle_game.world_frame_pipeline import (
    _collect_ready_readbacks, _finish_readbacks, _mark_active_rect_runtime,
    _mark_active_rects_runtime, _merge_phase_c, _queue_persistent_entity_observations,
    _restore_preview_runtime_state, _snapshot_preview_runtime_state, _step_once,
    _step_once_impl, _store_entity_observation_consume_snapshot, run_cpu_frame, step
)
from oracle_game.world_demo_scene import (
    _build_demo_scene, _fill_rect, _paint_material, _write_material_region_immediate,
    _world_engine_del, close
)
from oracle_game.world_state_snapshots import (
    _bridge_shadow_buffer_coord_space, _current_cell_state_snapshot,
    _current_entity_runtime_snapshot, _entity_placeholder_state_gpu_authoritative,
    _material_optics_snapshot_map, _preview_bridge_placeholder_dirty_rects,
    _runtime_entities_to_immediate_observation_targets, simulation_backend_report
)
from oracle_game.world_bridge_serializers import (
    _bridge_row_count, _clamped_gas_window, _decode_bridge_uploaded_command,
    _decode_bridge_uploaded_label, _decode_bridge_uploaded_page_stripe_section,
    _normalize_bridge_slice_bounds, _normalize_bridge_window_bounds,
    _record_bridge_page_stripe, _serialize_bridge_index_stages, _serialize_bridge_ndarray,
    _serialize_bridge_ndarray_slice, _serialize_bridge_ndarray_window,
    _serialize_bridge_readback_request_stages, _serialize_bridge_resource_summary,
    _serialize_bridge_spatial_window_payload, serialize_bridge_frame_snapshot,
    serialize_bridge_resources, serialize_bridge_runtime, serialize_bridge_shadow_buffer,
    serialize_bridge_shadow_buffer_gas_window, serialize_bridge_shadow_buffer_slice,
    serialize_bridge_shadow_buffer_window, serialize_bridge_shadow_buffer_world_window,
    serialize_bridge_typed_table, serialize_bridge_typed_table_slice,
    serialize_bridge_upload_snapshot
)
from oracle_game.world_controller_turn import (
    _build_preview_controller_turn_entities, controller_turn_to_frame_input,
    preview_entity_controller_turn, request_entity_controller_cycle,
    request_entity_controller_turn, run_entity_controller_cycle, run_entity_controller_turn,
    set_controller_state
)
from oracle_game.world_internal_helpers import (
    _advance_paging, _build_observation_request_pairs, _frame_readback_request_ids,
    _light_field_count, _mark_grid_world_command_runtime_regions, _mirror_release_entity_placeholder_cell,
    _page_store_key_lookup_update, _page_stripe_island_bboxes_from_payload,
    _pending_frame_input, _public_resolved_change_intent, _queued_command_xy,
    _refresh_island_records_for_ids, _resolve_anchor_target, _set_nested_payload_value,
    bootstrap_defaults, cancel_all_pending_frame_submissions, downsample_cells_to_gas,
    readback_request_status, submit_entity_controller_turn
)
from oracle_game.world_backend_gating import (
    _gpu_active_tile_count, _gpu_context_available, _gpu_pipeline_available,
    _gpu_realtime_budget_active, _gpu_world_simulation_required,
    _invalidate_gpu_authoritative_cell_resources, _invalidate_gpu_authoritative_resources,
    _require_cpu_oracle_backend, _require_gpu_authoritative_resources, _require_gpu_stage,
    _should_run_formal_collapse_this_frame, _skip_budgeted_gpu_stage,
    prewarm_formal_connected_collapse, require_gpu_world_backend, use_cpu_oracle_backend
)
from oracle_game.world_intent_helpers import (
    _build_entity_feedback_from_current_state, _build_entity_feedback_from_world,
    _build_observation_requests, _clamp_world_position, _combine_resolution_notes,
    _default_target_source_position, _distance_meters_to_cells,
    _entity_center_buffer_position, _entity_center_world_position,
    _entity_matches_anchor_filters, _entity_placeholder_bbox, _intent_resolution_status,
    _material_state_for_position, _normalized_world_direction,
    _normalize_runtime_force_source, _patch_entity_states,
    _preview_can_occupy_placeholder_cell, _public_resolved_target, _resolve_carrier_intents,
    _resolve_change_intents, _resolve_intent_source_positions,
    _resolve_intent_world_position, _resolve_query_source_position,
    _resolve_readback_requests, _source_facing_vector, _terrain_hill_cell_matches,
    _terrain_tree_cell_matches, _world_cell_material_has_tag, _world_distance_sq,
    _world_gas_window_for_cell_world_rect
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
        )

    use_cpu_oracle_backend = use_cpu_oracle_backend
    require_gpu_world_backend = require_gpu_world_backend
    prewarm_formal_connected_collapse = prewarm_formal_connected_collapse
    _gpu_context_available = _gpu_context_available
    _gpu_world_simulation_required = _gpu_world_simulation_required
    _gpu_realtime_budget_active = _gpu_realtime_budget_active
    _gpu_active_tile_count = _gpu_active_tile_count
    _skip_budgeted_gpu_stage = _skip_budgeted_gpu_stage
    _should_run_formal_collapse_this_frame = _should_run_formal_collapse_this_frame

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
                summary = profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
                summary["count"] += 1
                summary["cpu_ms"] += elapsed_ms
                if ctx is not None:
                    summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

    _gpu_pipeline_available = _gpu_pipeline_available
    _require_gpu_stage = _require_gpu_stage
    _require_gpu_authoritative_resources = _require_gpu_authoritative_resources
    _require_cpu_oracle_backend = _require_cpu_oracle_backend
    _invalidate_gpu_authoritative_resources = _invalidate_gpu_authoritative_resources
    _invalidate_gpu_authoritative_cell_resources = _invalidate_gpu_authoritative_cell_resources

    def bootstrap_defaults(self) -> None:
        return bootstrap_defaults(self)

    _material_table_snapshot_payload = _material_table_snapshot_payload
    _gas_species_table_snapshot_payload = _gas_species_table_snapshot_payload
    _light_type_table_snapshot_payload = _light_type_table_snapshot_payload
    _material_optics_table_snapshot_payload = _material_optics_table_snapshot_payload
    _material_optics_snapshot_map = _material_optics_snapshot_map
    _reaction_table_snapshot_payload = _reaction_table_snapshot_payload
    _stable_shadow_payload = _stable_shadow_payload
    _set_stable_shadow_payload = _set_stable_shadow_payload
    _shadow_has_table_payload = _shadow_has_table_payload
    _merged_reaction_table_payload = _merged_reaction_table_payload
    _merged_material_table_payload = _merged_material_table_payload
    _merged_gas_species_table_payload = _merged_gas_species_table_payload
    _merged_light_type_table_payload = _merged_light_type_table_payload
    _merged_material_optics_table_payload = _merged_material_optics_table_payload
    _coerce_enum = staticmethod(_coerce_enum)
    _coerce_material_def = _coerce_material_def
    _coerce_gas_species_def = _coerce_gas_species_def
    _coerce_light_type_def = _coerce_light_type_def
    _canonical_material_input_name = staticmethod(_canonical_material_input_name)
    _coerce_material_optics_def = _coerce_material_optics_def
    _coerce_reaction_action = _coerce_reaction_action
    _coerce_pair_reaction_rule = _coerce_pair_reaction_rule
    _coerce_self_reaction_rule = _coerce_self_reaction_rule
    _coerce_reaction_rules = _coerce_reaction_rules
    _shadow_material_payload = _shadow_material_payload
    _shadow_gas_species_payload = _shadow_gas_species_payload
    _shadow_light_type_payload = _shadow_light_type_payload
    _shadow_reaction_payload = _shadow_reaction_payload
    _payload_name_set = staticmethod(_payload_name_set)
    _validate_named_reference = staticmethod(_validate_named_reference)
    _validate_unique_identity_fields = staticmethod(_validate_unique_identity_fields)
    _validate_material_table_payload = _validate_material_table_payload
    _validate_gas_species_payload = _validate_gas_species_payload
    _validate_light_type_payload = _validate_light_type_payload
    _validate_material_optics_payload = _validate_material_optics_payload
    _validate_reaction_payload = _validate_reaction_payload
    _normalize_material_patch_fields = _normalize_material_patch_fields
    _normalize_gas_patch_fields = _normalize_gas_patch_fields
    _normalize_material_optics_patch_fields = _normalize_material_optics_patch_fields
    _normalize_reaction_action_patch_fields = _normalize_reaction_action_patch_fields
    _normalize_reaction_rule_patch_fields = _normalize_reaction_rule_patch_fields
    _coerce_force_source = _coerce_force_source
    _public_force_source_input = _public_force_source_input
    _frame_force_source_input = _frame_force_source_input
    _coerce_emitter = _coerce_emitter
    _frame_emitter_input = _frame_emitter_input
    _coerce_entity_placeholder = _coerce_entity_placeholder
    _public_entity_placeholder_input = _public_entity_placeholder_input
    _frame_entity_placeholder_input = _frame_entity_placeholder_input
    _coerce_entity_state = _coerce_entity_state
    _public_entity_state_input = _public_entity_state_input
    _frame_entity_state_input = _frame_entity_state_input
    _coerce_entity_observation_spec = _coerce_entity_observation_spec
    _normalize_entity_state_patch_fields = _normalize_entity_state_patch_fields
    _public_entity_state_patch_input = _public_entity_state_patch_input
    _controller_turn_entity_input = _controller_turn_entity_input
    _frame_entity_state_patch_input = _frame_entity_state_patch_input
    _coerce_entity_state_patch = _coerce_entity_state_patch
    _coerce_observation_target = _coerce_observation_target
    _coerce_target_query = _coerce_target_query
    _coerce_change_intent = _coerce_change_intent
    _coerce_carrier_intent = _coerce_carrier_intent
    _coerce_readback_request = _coerce_readback_request
    _normalize_readback_channels = staticmethod(_normalize_readback_channels)
    _normalize_readback_request = _normalize_readback_request
    _assign_readback_request_id = _assign_readback_request_id
    _assign_preview_readback_request_ids = _assign_preview_readback_request_ids
    _coerce_world_command = _coerce_world_command
    _coerce_json_value = staticmethod(_coerce_json_value)
    _normalize_json_payload_value = staticmethod(_normalize_json_payload_value)
    _coerce_world_frame_input = _coerce_world_frame_input
    _gas_field_count = _gas_field_count
    _light_field_count = _light_field_count
    update_material_table = update_material_table
    update_gas_species_table = update_gas_species_table
    update_light_type_table = update_light_type_table
    update_material_optics_table = update_material_optics_table
    update_reaction_table = update_reaction_table
    replace_reaction_table = replace_reaction_table
    reset_world = reset_world
    _reset_world_state = _reset_world_state
    queue_command = queue_command
    _resolve_direct_targeted_coords = _resolve_direct_targeted_coords
    inject_material = inject_material
    write_material_region = write_material_region
    inject_temperature = inject_temperature
    inject_velocity = inject_velocity
    inject_force = inject_force
    inject_gas = inject_gas
    request_readback = request_readback
    preview_readback = preview_readback
    request_observation = request_observation
    preview_observation = preview_observation
    _resolve_public_world_command = _resolve_public_world_command
    _public_world_command = _public_world_command
    preview_world_command = preview_world_command
    preview_target_queries = preview_target_queries
    request_world_command = request_world_command
    preview_change_intent = preview_change_intent
    request_change_intent = request_change_intent
    preview_carrier_intent = preview_carrier_intent
    request_carrier_intent = request_carrier_intent
    preview_frame_input = preview_frame_input
    submit_frame_input = submit_frame_input
    request_frame_input = request_frame_input
    request_frame_cycle = request_frame_cycle
    pending_frame_submission_ids = pending_frame_submission_ids

    def _pending_frame_input(self, submission_id: int) -> WorldFrameInput:
        return _pending_frame_input(self, submission_id)

    _frame_readback_request_ids = staticmethod(_frame_readback_request_ids)
    cancel_frame_submission = cancel_frame_submission

    def cancel_all_pending_frame_submissions(self) -> list[int]:
        return cancel_all_pending_frame_submissions(self)

    cancel_readback_request = cancel_readback_request
    poll_frame_output = poll_frame_output
    poll_all_frame_outputs = poll_all_frame_outputs
    serialize_pending_commands = serialize_pending_commands
    serialize_readback_state = serialize_readback_state
    serialize_bridge_runtime = serialize_bridge_runtime
    _serialize_bridge_resource_summary = staticmethod(_serialize_bridge_resource_summary)
    serialize_bridge_resources = serialize_bridge_resources
    serialize_ready_readbacks = serialize_ready_readbacks
    readback_request_status = readback_request_status
    serialize_frame_state = serialize_frame_state
    serialize_pending_frame_inputs = serialize_pending_frame_inputs
    serialize_pending_frame_detail = serialize_pending_frame_detail
    serialize_ready_frame_outputs = serialize_ready_frame_outputs
    frame_submission_status = frame_submission_status
    inject_light = inject_light
    focus_paging = focus_paging
    advance_paging = advance_paging
    capture_page_stripe = capture_page_stripe
    _capture_page_stripe_cpu_snapshot = _capture_page_stripe_cpu_snapshot
    apply_page_stripe = apply_page_stripe
    store_page_stripe = store_page_stripe
    capture_page_stripe_to_store = capture_page_stripe_to_store
    load_page_stripe = load_page_stripe
    apply_stored_page_stripe = apply_stored_page_stripe
    page_store_has_stripe = page_store_has_stripe
    list_page_store_stripe_keys = list_page_store_stripe_keys
    export_page_store_entries = export_page_store_entries
    import_page_store_entries = import_page_store_entries
    clear_page_store = clear_page_store
    serialize_page_store_state = serialize_page_store_state
    _coerce_page_store_key = _coerce_page_store_key
    _page_store_key_lookup_update = staticmethod(_page_store_key_lookup_update)
    _coerce_page_stripe_payload = _coerce_page_stripe_payload
    sync_entity_placeholders = sync_entity_placeholders
    sync_entity_states = sync_entity_states
    patch_entity_states = patch_entity_states
    sync_entity_observation_specs = sync_entity_observation_specs
    set_force_sources = set_force_sources
    set_emitters = set_emitters
    patch_material = patch_material
    patch_light = patch_light
    patch_gas = patch_gas
    patch_material_optics = patch_material_optics
    patch_reaction_action = patch_reaction_action
    patch_reaction_rule = patch_reaction_rule
    delete_reaction_action = delete_reaction_action
    delete_reaction_rule = delete_reaction_rule
    step = step
    simulation_backend_report = simulation_backend_report
    poll_readbacks = poll_readbacks
    poll_all_readbacks = poll_all_readbacks
    consume_entity_observation_results = consume_entity_observation_results
    run_entity_controller_turn = run_entity_controller_turn
    set_controller_state = set_controller_state
    serialize_controller_state = serialize_controller_state
    _build_preview_controller_turn_entities = _build_preview_controller_turn_entities
    _preview_consume_entity_observation_results = _preview_consume_entity_observation_results
    controller_turn_to_frame_input = controller_turn_to_frame_input
    preview_entity_controller_turn = preview_entity_controller_turn
    submit_entity_controller_turn = submit_entity_controller_turn
    request_entity_controller_turn = request_entity_controller_turn
    request_entity_controller_cycle = request_entity_controller_cycle
    run_entity_controller_cycle = run_entity_controller_cycle

    run_cpu_frame = run_cpu_frame
    _step_once = _step_once
    _merge_phase_c = _merge_phase_c
    _step_once_impl = _step_once_impl
    _queue_persistent_entity_observations = _queue_persistent_entity_observations
    _apply_frame_input = _apply_frame_input
    _prepare_preview_frame_context = _prepare_preview_frame_context
    _snapshot_preview_runtime_state = _snapshot_preview_runtime_state
    _restore_preview_runtime_state = _restore_preview_runtime_state
    _contextualize_page_stripe_update = _contextualize_page_stripe_update
    _page_store_key = staticmethod(_page_store_key)
    _preview_apply_paging_updates = _preview_apply_paging_updates
    _preview_bridge_placeholder_dirty_rects = _preview_bridge_placeholder_dirty_rects
    _serialize_bridge_readback_request_stages = staticmethod(_serialize_bridge_readback_request_stages)
    _serialize_bridge_index_stages = staticmethod(_serialize_bridge_index_stages)
    _serialize_preview_bridge_frame_snapshot = _serialize_preview_bridge_frame_snapshot
    _queue_loaded_collapse_pending_regions = _queue_loaded_collapse_pending_regions
    _clear_saved_page_stripe_runtime_state = _clear_saved_page_stripe_runtime_state
    _prune_page_stripe_regions = _prune_page_stripe_regions
    _subtract_page_stripe_range_from_region = staticmethod(_subtract_page_stripe_range_from_region)
    close = close

    def __del__(self) -> None:  # pragma: no cover
        _world_engine_del(self)

    material_by_id = material_by_id
    allocate_island_id = allocate_island_id
    _refresh_island_records_for_ids = _refresh_island_records_for_ids
    in_bounds = in_bounds
    cell_xy_to_gas = cell_xy_to_gas
    cell_to_gas = cell_to_gas
    sample_ambient_to_cells = sample_ambient_to_cells
    ambient_temperature_at_cell = ambient_temperature_at_cell
    ambient_temperature_region = ambient_temperature_region
    sample_flow_to_cells = sample_flow_to_cells
    downsample_cells_to_gas = downsample_cells_to_gas
    add_gas_from_cells = add_gas_from_cells
    set_cell_by_id = set_cell_by_id
    _inject_velocity_immediate = _inject_velocity_immediate
    _inject_temperature_immediate = _inject_temperature_immediate
    _inject_gas_immediate = _inject_gas_immediate
    set_cell = set_cell
    clear_cell = clear_cell
    clear_cells = clear_cells
    set_material_by_mask = set_material_by_mask
    swap_cells = swap_cells
    clear_cell_region = clear_cell_region
    serialize_local_cells = serialize_local_cells
    serialize_temperature_window = serialize_temperature_window
    serialize_gas = serialize_gas
    serialize_pressure = serialize_pressure
    serialize_velocity = serialize_velocity
    serialize_visible_illumination = serialize_visible_illumination
    serialize_gas_runtime = serialize_gas_runtime
    serialize_heat_runtime = serialize_heat_runtime
    serialize_liquid_runtime = serialize_liquid_runtime
    serialize_reaction_runtime = serialize_reaction_runtime
    serialize_collapse_runtime = serialize_collapse_runtime
    serialize_optics_runtime = serialize_optics_runtime
    serialize_active_runtime = serialize_active_runtime
    serialize_motion_runtime = serialize_motion_runtime
    serialize_paging_state = serialize_paging_state
    serialize_engine_capabilities = serialize_engine_capabilities
    serialize_material_table = serialize_material_table
    _serialize_bridge_ndarray = _serialize_bridge_ndarray
    _bridge_row_count = staticmethod(_bridge_row_count)
    _normalize_bridge_slice_bounds = staticmethod(_normalize_bridge_slice_bounds)
    _normalize_bridge_window_bounds = staticmethod(_normalize_bridge_window_bounds)
    _clamped_gas_window = _clamped_gas_window
    _bridge_shadow_buffer_coord_space = _bridge_shadow_buffer_coord_space
    _serialize_bridge_ndarray_slice = _serialize_bridge_ndarray_slice
    _serialize_bridge_ndarray_window = _serialize_bridge_ndarray_window
    _serialize_bridge_spatial_window_payload = _serialize_bridge_spatial_window_payload
    serialize_bridge_typed_table = serialize_bridge_typed_table
    serialize_bridge_typed_table_slice = serialize_bridge_typed_table_slice
    serialize_bridge_shadow_buffer = serialize_bridge_shadow_buffer
    serialize_bridge_shadow_buffer_slice = serialize_bridge_shadow_buffer_slice
    serialize_bridge_shadow_buffer_window = serialize_bridge_shadow_buffer_window
    serialize_bridge_shadow_buffer_world_window = serialize_bridge_shadow_buffer_world_window
    serialize_bridge_shadow_buffer_gas_window = serialize_bridge_shadow_buffer_gas_window
    _decode_bridge_uploaded_command = staticmethod(_decode_bridge_uploaded_command)
    _decode_bridge_uploaded_label = staticmethod(_decode_bridge_uploaded_label)
    _decode_bridge_uploaded_page_stripe_section = staticmethod(_decode_bridge_uploaded_page_stripe_section)
    _set_nested_payload_value = staticmethod(_set_nested_payload_value)
    serialize_bridge_upload_snapshot = serialize_bridge_upload_snapshot
    serialize_bridge_frame_snapshot = serialize_bridge_frame_snapshot
    _serialize_force_source_record = _serialize_force_source_record
    serialize_force_sources = serialize_force_sources
    _serialize_emitter_record = _serialize_emitter_record
    serialize_emitters = serialize_emitters
    serialize_gas_species_table = serialize_gas_species_table
    serialize_light_type_table = serialize_light_type_table
    serialize_material_optics_table = serialize_material_optics_table
    serialize_reaction_table = serialize_reaction_table
    serialize_optics = serialize_optics
    serialize_readback_request = serialize_readback_request
    _infer_readback_payload_coord_space = _infer_readback_payload_coord_space
    _serialize_readback_source_descriptor = _serialize_readback_source_descriptor
    _serialize_readback_plan_for_request = _serialize_readback_plan_for_request
    _serialize_readback_plans_for_requests = _serialize_readback_plans_for_requests
    _serialize_observation_plan_for_target_request = _serialize_observation_plan_for_target_request

    def _build_observation_request_pairs(
        self,
        targets: list[ObservationTarget],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[tuple[ObservationTarget, ReadbackRequest]]:
        return _build_observation_request_pairs(self, targets, resolved_targets)

    serialize_readback_plan = serialize_readback_plan
    serialize_observation_plan = serialize_observation_plan
    serialize_world_command = serialize_world_command
    serialize_entity_placeholder_input = staticmethod(serialize_entity_placeholder_input)
    serialize_target_query_input = staticmethod(serialize_target_query_input)
    serialize_page_stripe_update = staticmethod(serialize_page_stripe_update)
    serialize_page_store_key = staticmethod(serialize_page_store_key)

    @classmethod
    def serialize_page_stripe_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return serialize_page_stripe_payload(cls, payload)

    serialize_change_intent_input = staticmethod(serialize_change_intent_input)
    serialize_carrier_intent_input = staticmethod(serialize_carrier_intent_input)
    serialize_frame_input = serialize_frame_input
    _serialize_readback_payload = _serialize_readback_payload
    serialize_readback_result = serialize_readback_result
    serialize_resolved_target = serialize_resolved_target
    serialize_resolved_change_intent = serialize_resolved_change_intent
    serialize_resolved_carrier_intent = serialize_resolved_carrier_intent
    serialize_observation_result = serialize_observation_result
    serialize_entity_observation_spec = staticmethod(serialize_entity_observation_spec)
    serialize_entity_state_patch = serialize_entity_state_patch
    serialize_observation_target = staticmethod(serialize_observation_target)
    serialize_entity_state_input = staticmethod(serialize_entity_state_input)
    serialize_entity_state = serialize_entity_state
    serialize_entity_states = serialize_entity_states
    serialize_entity_observation_state = serialize_entity_observation_state
    _current_cell_state_snapshot = _current_cell_state_snapshot
    _current_entity_runtime_snapshot = _current_entity_runtime_snapshot
    _entity_placeholder_state_gpu_authoritative = _entity_placeholder_state_gpu_authoritative
    serialize_entity_placeholders = serialize_entity_placeholders
    serialize_entity_placeholder_index_snapshot = serialize_entity_placeholder_index_snapshot
    serialize_entity_feedback_snapshot = serialize_entity_feedback_snapshot
    serialize_consumed_entity_feedback_snapshot = serialize_consumed_entity_feedback_snapshot
    _serialize_cpu_visible_entity_placeholders = _serialize_cpu_visible_entity_placeholders
    serialize_entity_feedback = serialize_entity_feedback
    _store_entity_observation_consume_snapshot = _store_entity_observation_consume_snapshot
    serialize_entity_observation_consume_state = serialize_entity_observation_consume_state
    serialize_frame_output = serialize_frame_output
    serialize_frame_preview = serialize_frame_preview
    serialize_debug_frame = serialize_debug_frame
    debug_frame = debug_frame
    _material_frame = _material_frame
    _temperature_frame = _temperature_frame
    _pressure_frame = _pressure_frame
    _vector_field_frame = _vector_field_frame
    _active_frame = _active_frame
    _motion_frame = _motion_frame
    _heat_frame = _heat_frame
    _liquid_frame = _liquid_frame
    _reaction_frame = _reaction_frame
    _collapse_frame = _collapse_frame
    _optics_frame = _optics_frame
    _optics_dose_frame = _optics_dose_frame
    _gas_frame = _gas_frame
    _accumulate_debug_point = _accumulate_debug_point
    _draw_debug_bbox_outline = _draw_debug_bbox_outline

    _apply_grid_world_commands = _apply_grid_world_commands
    _apply_grid_world_command_cpu = _apply_grid_world_command_cpu
    _grid_world_command_runtime_regions = _grid_world_command_runtime_regions

    def _mark_grid_world_command_runtime_regions(self, command: WorldCommand) -> None:
        return _mark_grid_world_command_runtime_regions(self, command)

    _apply_commands = _apply_commands
    _finish_readbacks = _finish_readbacks
    _collect_ready_readbacks = _collect_ready_readbacks

    def _queued_command_xy(self, command: WorldCommand) -> tuple[int, int]:
        return _queued_command_xy(self, command)

    _make_readback_payload = _make_readback_payload
    _gas_window_for_cell_rect = _gas_window_for_cell_rect
    _apply_page_stripe = _apply_page_stripe
    _apply_page_stripe_dense_cpu = _apply_page_stripe_dense_cpu
    _queue_loaded_collapse_pending_regions_from_payload = _queue_loaded_collapse_pending_regions_from_payload
    _advance_paging = _advance_paging
    _prepare_bridge_frame_inputs = _prepare_bridge_frame_inputs
    _needs_pre_simulation_bridge_sync = _needs_pre_simulation_bridge_sync
    _sync_pre_simulation_bridge_without_debug_upload = _sync_pre_simulation_bridge_without_debug_upload
    _clear_bridge_frame_inputs = _clear_bridge_frame_inputs
    _mark_active_rect_runtime = _mark_active_rect_runtime
    _mark_active_rects_runtime = _mark_active_rects_runtime
    _sync_entity_placeholders = _sync_entity_placeholders
    _sync_force_sources = _sync_force_sources
    _append_force_source_immediate = _append_force_source_immediate
    _sync_persistent_emitters = _sync_persistent_emitters
    _append_transient_light_emitter_immediate = _append_transient_light_emitter_immediate
    _record_bridge_page_stripe = _record_bridge_page_stripe
    _release_entity_placeholder_cell = _release_entity_placeholder_cell
    _mirror_release_entity_placeholder_cell = _mirror_release_entity_placeholder_cell
    _resolve_sanctioned_material_id = _resolve_sanctioned_material_id
    _shadow_material_id_by_name = _shadow_material_id_by_name
    _resolve_sanctioned_placeholder_material_id = _resolve_sanctioned_placeholder_material_id
    _resolve_sanctioned_light_id = _resolve_sanctioned_light_id
    _resolve_sanctioned_gas_id = _resolve_sanctioned_gas_id
    _shadow_material_row_valid = _shadow_material_row_valid
    _shadow_gas_row_valid = _shadow_gas_row_valid
    _shadow_light_row_valid = _shadow_light_row_valid
    _shadow_material_def = _shadow_material_def
    _shadow_light_type_def = _shadow_light_type_def
    _shadow_gas_species_def = _shadow_gas_species_def
    _shadow_material_optics_def = _shadow_material_optics_def
    _shadow_material_name = _shadow_material_name
    _shadow_gas_name = _shadow_gas_name
    _shadow_light_name = _shadow_light_name
    _shadow_light_default_range = _shadow_light_default_range
    _shadow_light_dose_channel = _shadow_light_dose_channel
    _shadow_light_color = _shadow_light_color
    _shadow_light_name_and_range = _shadow_light_name_and_range
    _shadow_material_default_phase = _shadow_material_default_phase
    _shadow_material_base_integrity = _shadow_material_base_integrity
    _shadow_material_spawn_temperature = _shadow_material_spawn_temperature
    _shadow_condense_target_material_id = _shadow_condense_target_material_id
    _shadow_material_is_placeholder = _shadow_material_is_placeholder
    _material_placeholder_mask = _material_placeholder_mask
    _shadow_material_is_plant = _shadow_material_is_plant
    _shadow_reaction_action = _shadow_reaction_action
    _reaction_rule_list = _reaction_rule_list
    _set_reaction_rule_list = _set_reaction_rule_list
    _set_reaction_rules_payload = _set_reaction_rules_payload
    _remap_reaction_payload_result_actions = staticmethod(_remap_reaction_payload_result_actions)
    _remap_material_payload_reaction_slots = staticmethod(_remap_material_payload_reaction_slots)
    _clamp_material_payload_reaction_slots = staticmethod(_clamp_material_payload_reaction_slots)
    _shadow_reaction_rule = _shadow_reaction_rule
    _occupy_entity_placeholder_cell = _occupy_entity_placeholder_cell
    _mirror_occupy_entity_placeholder_cell = _mirror_occupy_entity_placeholder_cell
    _frame_entities_to_placeholders_and_observations = _frame_entities_to_placeholders_and_observations
    _runtime_entities_to_immediate_observation_targets = _runtime_entities_to_immediate_observation_targets
    _sync_entity_states = _sync_entity_states
    _sync_entity_observation_specs = _sync_entity_observation_specs
    _patch_entity_states = _patch_entity_states
    _build_preview_entity_placeholders = _build_preview_entity_placeholders
    _preview_can_occupy_placeholder_cell = _preview_can_occupy_placeholder_cell
    _material_state_for_position = _material_state_for_position
    _build_observation_requests = _build_observation_requests
    _build_observation_request = _build_observation_request
    _resolve_readback_requests = _resolve_readback_requests
    _resolve_readback_request = _resolve_readback_request
    _resolve_target_queries = _resolve_target_queries
    _resolve_change_intents = _resolve_change_intents
    _public_resolved_change_intent = _public_resolved_change_intent
    _public_resolved_target = _public_resolved_target
    _resolve_carrier_intents = _resolve_carrier_intents
    _public_resolved_carrier_intent = _public_resolved_carrier_intent
    _resolve_carrier_intent = _resolve_carrier_intent
    _resolve_change_intent = _resolve_change_intent
    _resolve_change_intent_world_position = _resolve_change_intent_world_position
    _resolve_intent_world_position = _resolve_intent_world_position
    _resolve_intent_source_positions = _resolve_intent_source_positions
    _normalized_world_direction = staticmethod(_normalized_world_direction)
    _disk_world_cells = _disk_world_cells
    _disk_world_cells_raw = staticmethod(_disk_world_cells_raw)
    _line_world_cells = _line_world_cells
    _line_world_cells_raw = staticmethod(_line_world_cells_raw)
    _capsule_world_cells = _capsule_world_cells
    _capsule_world_cells_raw = _capsule_world_cells_raw
    _buffer_cell_bounds = staticmethod(_buffer_cell_bounds)
    _apply_change_stability_drift = _apply_change_stability_drift
    _resolve_legal_world_position = _resolve_legal_world_position
    _intent_resolution_status = staticmethod(_intent_resolution_status)
    _combine_resolution_notes = staticmethod(_combine_resolution_notes)
    _resolve_targeted_commands = _resolve_targeted_commands
    _resolve_target_query = _resolve_target_query
    _resolve_target_query_distance_cells = _resolve_target_query_distance_cells
    _distance_meters_to_cells = staticmethod(_distance_meters_to_cells)
    _resolve_query_source_position = _resolve_query_source_position
    _default_target_source_position = _default_target_source_position
    _resolve_anchor_target = _resolve_anchor_target
    _resolve_entity_anchor = _resolve_entity_anchor
    _resolve_terrain_anchor = _resolve_terrain_anchor
    _entity_matches_anchor_filters = _entity_matches_anchor_filters
    _terrain_cell_matches = _terrain_cell_matches
    _terrain_tree_cell_matches = _terrain_tree_cell_matches
    _terrain_hill_cell_matches = _terrain_hill_cell_matches
    _world_cell_is_solid_local = _world_cell_is_solid_local
    _world_cell_is_empty_local = _world_cell_is_empty_local
    _world_cell_material_has_tag = _world_cell_material_has_tag
    _bounded_material_state_for_position = _bounded_material_state_for_position
    _matches_direction_filter = _matches_direction_filter
    _query_direction_vector = _query_direction_vector
    _direction_vector = _direction_vector
    _source_facing_vector = _source_facing_vector
    _entity_center_buffer_position = _entity_center_buffer_position
    _entity_center_world_position = _entity_center_world_position
    _buffer_to_world_position = _buffer_to_world_position
    _buffer_to_world_float_position = _buffer_to_world_float_position
    _world_to_buffer_float_position = _world_to_buffer_float_position
    _force_source_world_position = _force_source_world_position
    _force_source_buffer_position = _force_source_buffer_position
    _normalize_runtime_force_source = _normalize_runtime_force_source
    _buffer_gas_to_world_position = _buffer_gas_to_world_position
    _buffer_bbox_to_world_bbox = _buffer_bbox_to_world_bbox
    _clamped_world_window = _clamped_world_window
    _centered_world_window = _centered_world_window
    _world_axis_spans = _world_axis_spans
    _world_axis_indices = _world_axis_indices
    _extract_world_window = _extract_world_window
    _world_gas_window_for_cell_world_rect = _world_gas_window_for_cell_world_rect
    _pack_cell_core_world_window = _pack_cell_core_world_window
    _world_to_buffer_clamped = _world_to_buffer_clamped
    _clamp_world_position = _clamp_world_position
    _find_nearest_empty_world_position = _find_nearest_empty_world_position
    _world_cell_is_empty = _world_cell_is_empty
    _world_distance_sq = staticmethod(_world_distance_sq)
    _entity_placeholder_bbox = _entity_placeholder_bbox
    _collect_observations = _collect_observations
    _collect_entity_feedback = _collect_entity_feedback
    _build_entity_feedback = _build_entity_feedback
    _build_entity_feedback_from_world = _build_entity_feedback_from_world
    _build_entity_feedback_from_current_state = _build_entity_feedback_from_current_state
    _build_entity_feedback_from_state = _build_entity_feedback_from_state
    _capture_stripe_array = _capture_stripe_array
    _write_stripe_array = _write_stripe_array
    _default_page_stripe_payload = _default_page_stripe_payload
    _stripe_buffer_ranges = _stripe_buffer_ranges
    _mark_loaded_page_stripe_active = _mark_loaded_page_stripe_active
    _rebuild_sparse_runtime_indexes = _rebuild_sparse_runtime_indexes
    _rebuild_entity_placeholder_index = _rebuild_entity_placeholder_index
    _normalize_cell_runtime_arrays = _normalize_cell_runtime_arrays
    _normalize_page_stripe_cell_runtime = _normalize_page_stripe_cell_runtime
    _capture_page_stripe_entity_placeholder_runtime = _capture_page_stripe_entity_placeholder_runtime
    _apply_page_stripe_entity_placeholder_runtime = _apply_page_stripe_entity_placeholder_runtime
    _rebuild_island_records = _rebuild_island_records
    _capture_page_stripe_island_runtime = _capture_page_stripe_island_runtime
    _page_stripe_island_bboxes_from_payload = _page_stripe_island_bboxes_from_payload
    _merge_island_runtime_payload = _merge_island_runtime_payload
    _rebuild_material_property_arrays = _rebuild_material_property_arrays
    _rebuild_gas_property_arrays = _rebuild_gas_property_arrays
    _rebuild_light_property_arrays = _rebuild_light_property_arrays
    _cell_participates_in_collapse = _cell_participates_in_collapse
    _mark_collapse_dirty_rect = _mark_collapse_dirty_rect
    _drain_gpu_collapse_structure_dirty_tiles = _drain_gpu_collapse_structure_dirty_tiles
    _paint_material = _paint_material
    _write_material_region_immediate = _write_material_region_immediate
    _build_demo_scene = _build_demo_scene
    _fill_rect = _fill_rect
