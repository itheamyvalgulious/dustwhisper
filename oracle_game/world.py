from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from enum import Enum
import threading
import time
from typing import Any, Iterable, Sequence
import numpy as np

from oracle_game.active import ActiveRegionTracker
from oracle_game.gpu import (
    GPUBridge,
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUReadbackSegment,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
)
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.page_store import InMemoryPageStore, PageStore, StoredStripeKey
from oracle_game.paging import RingPagingWindow
from oracle_game.rules import RuleBook, build_default_payloads
from oracle_game.sim.collapse import CollapseSolver
from oracle_game.sim.gas import GasSolver
from oracle_game.sim.heat import HeatSolver
from oracle_game.sim.liquid import LiquidSolver
from oracle_game.sim.motion import MotionSolver
from oracle_game.sim.optics import OpticsSolver
from oracle_game.sim.gpu_placeholders import GPUPlaceholderPipeline
from oracle_game.sim.gpu_page_stripes import GPUPageStripePipeline
from oracle_game.sim.gpu_world_commands import GPUWorldCommandPipeline
from oracle_game.sim.reactions import ReactionSolver
from oracle_game.sim.gpu_merge import GPUMergePipeline
from oracle_game.types import (
    CarrierIntent,
    CarrierIntent,
    ChangeIntent,
    DebugView,
    EntityCellFeedback,
    EntityFeedback,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityStatePatch,
    EntityState,
    ForceSource,
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    ObservationTarget,
    PageStripeUpdate,
    PairReactionRule,
    Phase,
    ReactionAction,
    ReadbackRequest,
    ReadbackResult,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    SelfReactionRule,
    TargetQuery,
    WorldFrameInput,
    WorldFrameOutput,
    WorldFramePreview,
    WorldCommand,
)

from oracle_game.world_constants import (
    BASE_MATERIAL_RUNTIME_ALIASES,
    CARDINAL_DIRECTION_VECTORS,
    ENTITY_STATE_PATCH_METADATA_FIELDS,
    GPU_REALTIME_BUDGET_CELL_THRESHOLD,
    IGNORED_ANCHOR_FILTERS,
    PAIR_REACTION_RULE_SET_NAMES,
    REACTION_RULE_SET_NAMES,
    TARGET_QUERY_CELLS_PER_METER,
    TARGET_QUERY_DISTANCE_HINT_CELLS,
    TERRAIN_ANCHOR_FILTERS,
    UNSET_CONTROLLER_STATE,
)
from oracle_game.world_capabilities import serialize_engine_capabilities
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
from oracle_game.world_readback_payload import make_readback_payload
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
from oracle_game.world_entity_sync import (
    sync_entity_placeholders,
    sync_entity_states,
    patch_entity_states,
    sync_entity_observation_specs,
    set_force_sources,
    set_emitters,
    consume_entity_observation_results,
    _preview_consume_entity_observation_results,
    _sync_entity_placeholders,
    _release_entity_placeholder_cell,
    _occupy_entity_placeholder_cell,
    _frame_entities_to_placeholders_and_observations,
    _sync_entity_states,
    _sync_entity_observation_specs,
    _build_preview_entity_placeholders,
    _build_observation_request,
    _resolve_readback_request,
    _collect_observations,
    _collect_entity_feedback,
    _build_entity_feedback,
    _build_entity_feedback_from_state,
)
from oracle_game.world_frame_io import (
    preview_frame_input,
    submit_frame_input,
    request_frame_input,
    request_frame_cycle,
    pending_frame_submission_ids,
    cancel_frame_submission,
    cancel_readback_request,
    poll_frame_output,
    poll_all_frame_outputs,
    frame_submission_status,
    _apply_frame_input,
    _prepare_preview_frame_context,
    _prepare_bridge_frame_inputs,
    _needs_pre_simulation_bridge_sync,
    _clear_bridge_frame_inputs,
)
from oracle_game.world_table_api import (
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
    _reset_world_state,
)
from oracle_game.world_paging import (
    _apply_page_stripe,
    _apply_page_stripe_dense_cpu,
    _capture_page_stripe_cpu_snapshot,
    _coerce_page_store_key,
    _coerce_page_stripe_payload,
    _default_page_stripe_payload,
    _stripe_buffer_ranges,
    advance_paging,
    apply_page_stripe,
    apply_stored_page_stripe,
    capture_page_stripe,
    clear_page_store,
    export_page_store_entries,
    focus_paging,
    import_page_store_entries,
    list_page_store_stripe_keys,
    load_page_stripe,
    page_store_has_stripe,
    store_page_stripe,
)
from oracle_game.world_intent_resolver import (
    _resolve_carrier_intent,
    _resolve_change_intent,
    _resolve_change_intent_world_position,
    _resolve_target_query,
    _resolve_target_query_distance_cells,
    _resolve_target_queries,
)
from oracle_game.world_cell_mutators import (
    _inject_gas_immediate,
    _inject_temperature_immediate,
    _inject_velocity_immediate,
    add_gas_from_cells,
    allocate_island_id,
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
from oracle_game.world_input_coercion import (
    _assign_preview_readback_request_ids,
    _assign_readback_request_id,
    _canonical_material_input_name,
    _coerce_carrier_intent,
    _coerce_change_intent,
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
    _coerce_readback_request,
    _coerce_reaction_action,
    _coerce_reaction_rules,
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
    _normalize_readback_channels,
    _normalize_readback_request,
    _normalize_reaction_action_patch_fields,
    _normalize_reaction_rule_patch_fields,
    _public_entity_placeholder_input,
    _public_entity_state_input,
    _public_entity_state_patch_input,
    _public_force_source_input,
)
from oracle_game.world_table_validation import (
    _gas_species_table_snapshot_payload,
    _light_type_table_snapshot_payload,
    _material_optics_table_snapshot_payload,
    _material_table_snapshot_payload,
    _merged_gas_species_table_payload,
    _merged_light_type_table_payload,
    _merged_material_optics_table_payload,
    _merged_material_table_payload,
    _merged_reaction_table_payload,
    _payload_name_set,
    _reaction_table_snapshot_payload,
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
    _shadow_material_default_phase,
    _shadow_material_def,
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
from oracle_game.world_demo_scene import (
    _build_demo_scene,
    _fill_rect,
    _paint_material,
    _write_material_region_immediate,
    _world_engine_del,
    close,
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
from oracle_game.world_bridge_serializers import (
    _decode_bridge_uploaded_command,
    _decode_bridge_uploaded_label,
    _decode_bridge_uploaded_page_stripe_section,
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
from oracle_game.world_internal_helpers import (
    _advance_paging,
    _light_field_count,
    _mirror_release_entity_placeholder_cell,
    _page_store_key_lookup_update,
    _page_stripe_island_bboxes_from_payload,
    _public_resolved_change_intent,
    _refresh_island_records_for_ids,
    _resolve_anchor_target,
    _set_nested_payload_value,
    downsample_cells_to_gas,
    readback_request_status,
    submit_entity_controller_turn,
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
        simulation_backend = str(simulation_backend).lower()
        if simulation_backend not in {"gpu", "cpu"}:
            raise ValueError("simulation_backend must be one of: gpu, cpu")
        self.simulation_backend = simulation_backend
        self._world_simulation_frame_active = False
        self.width = width
        self.height = height
        self.gas_cell_size = gas_cell_size
        self.gas_width = max(1, (width + gas_cell_size - 1) // gas_cell_size)
        self.gas_height = max(1, (height + gas_cell_size - 1) // gas_cell_size)
        self.paging = RingPagingWindow(width, height, active_width or width // 2, active_height or height // 2)
        self.active = ActiveRegionTracker(width, height)
        self.rulebook = RuleBook()
        self.bridge = (
            GPUBridge(create_standalone=False)
            if simulation_backend == "cpu" and gpu_context is None
            else GPUBridge(ctx=gpu_context)
        )
        if simulation_backend == "gpu" and not self._gpu_context_available():
            raise RuntimeError("GPU world simulation requires a ModernGL 4.3+ context; CPU fallback is disabled")
        self.page_store = InMemoryPageStore() if page_store is None else page_store
        self.frame_id = 0
        self.state_lock = threading.RLock()
        self.command_queue: deque[WorldCommand] = deque()
        self.pending_frame_inputs: deque[WorldFrameInput] = deque()
        self.completed_frame_outputs: deque[WorldFrameOutput] = deque()
        self.canceled_frame_submission_ids: set[int] = set()
        self.next_frame_submission_id = 1
        self.next_readback_request_id = 1
        self.pending_readbacks: list[ReadbackRequest] = []
        self.inflight_readbacks: list[ReadbackRequest] = []
        self.completed_readbacks: deque[ReadbackResult] = deque()
        self.canceled_readback_request_ids: set[int] = set()
        self.last_entity_observation_consume_snapshot: dict[str, Any] = {
            "frame_id": 0,
            "consumed": 0,
            "consumed_readbacks": [],
            "observations": {},
            "entity_feedback": {},
        }
        self.controller_state_snapshot: Any = None
        self.bootstrap_log: list[str] = []
        self.bridge_frame_commands: list[WorldCommand] = []
        self.bridge_frame_readback_requests: list[ReadbackRequest] = []
        self.bridge_frame_placeholders: list[EntityPlaceholder] = []
        self.bridge_frame_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
        self._pending_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
        self.bridge_frame_paging_updates: list[PageStripeUpdate] = []
        self.bridge_frame_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        self._bridge_inputs_prepared = False
        self._gpu_cpu_dirty_resources: set[str] = set()
        self._resolver_blocked_cells: set[tuple[int, int]] | None = None
        self._resolver_released_cells: set[tuple[int, int]] | None = None
        self.force_sources: list[ForceSource] = []
        self.persistent_emitters: list[dict[str, object]] = []
        self.emitters: list[dict[str, object]] = []
        self._formal_gpu_frame_has_light_dose: bool | None = None
        self._gpu_optics_outputs_clear = False
        self.gpu_realtime_budget_enabled = True
        self.gpu_realtime_budget_cell_threshold = GPU_REALTIME_BUDGET_CELL_THRESHOLD
        self.profile_passes_enabled = False
        self.profile_passes_sync = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}, "skipped_stages": []}
        self.last_skipped_gpu_stages: list[str] = []
        self.formal_collapse_interval_frames = 4
        self.collapse_dirty_regions: list[tuple[int, int, int, int]] = []
        self.collapse_deferred_regions: list[tuple[int, int, int, int]] = []
        self._gpu_collapse_structure_dirty_tiles_pending = False
        self._gpu_collapse_structure_dirty_tiles_deferred = False
        self.islands: dict[int, object] = {}
        self.entity_states: dict[int, EntityState] = {}
        self.entity_placeholders: dict[int, set[tuple[int, int]]] = {}
        self.next_island_id = 1

        self.material_id = np.zeros((height, width), dtype=np.int32)
        self.phase = np.zeros((height, width), dtype=np.uint8)
        self.cell_flags = np.zeros((height, width), dtype=np.uint8)
        self.velocity = np.zeros((height, width, 2), dtype=np.float32)
        self.cell_temperature = np.full((height, width), 20.0, dtype=np.float32)
        self.timer_pack = np.zeros((height, width, 4), dtype=np.uint8)
        self.integrity = np.zeros((height, width), dtype=np.float32)
        self.island_id = np.zeros((height, width), dtype=np.int32)
        self.entity_id = np.zeros((height, width), dtype=np.int32)
        self.placeholder_displaced_material = np.zeros((height, width), dtype=np.int32)
        self.collapse_delay_pending = np.zeros((height, width), dtype=np.bool_)

        self.flow_velocity = np.zeros((self.gas_height, self.gas_width, 2), dtype=np.float32)
        self.ambient_temperature = np.full((self.gas_height, self.gas_width), 20.0, dtype=np.float32)
        self.pressure_ping = np.zeros((self.gas_height, self.gas_width), dtype=np.float32)
        self.gas_concentration = np.zeros((1, self.gas_height, self.gas_width), dtype=np.float32)
        self.visible_illumination = np.zeros((height, width, 3), dtype=np.float32)
        self.cell_optical_dose = np.zeros((1, height, width), dtype=np.float32)
        self.gas_optical_dose = np.zeros((1, self.gas_height, self.gas_width), dtype=np.float32)
        self.default_debug_view = DebugView.MATERIAL

        self.material_density = np.zeros(1, dtype=np.float32)
        self.material_base_color = np.zeros((1, 3), dtype=np.float32)
        self.material_gravity = np.zeros(1, dtype=np.float32)
        self.material_wind = np.zeros(1, dtype=np.float32)
        self.material_drag = np.zeros(1, dtype=np.float32)
        self.material_friction = np.zeros(1, dtype=np.float32)
        self.material_elasticity = np.zeros(1, dtype=np.float32)
        self.material_max_dda_step = np.zeros(1, dtype=np.int32)
        self.material_default_phase = np.zeros(1, dtype=np.uint8)
        self.material_base_integrity = np.zeros(1, dtype=np.float32)
        self.material_spawn_temperature = np.full(1, np.nan, dtype=np.float32)
        self.material_reaction_slots = np.full((1, 8), -1, dtype=np.int32)
        self.material_material_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_gas_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_light_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_powder_solver_kind = np.zeros(1, dtype=np.uint8)
        self.material_liquid_solver_kind = np.zeros(1, dtype=np.uint8)
        self.material_falling_island_break_kind = np.zeros(1, dtype=np.uint8)
        self.material_heat_capacity = np.zeros(1, dtype=np.float32)
        self.material_conductivity = np.zeros(1, dtype=np.float32)
        self.material_ambient_exchange = np.zeros(1, dtype=np.float32)
        self.material_is_structural = np.zeros(1, dtype=np.bool_)
        self.material_is_support_anchor = np.zeros(1, dtype=np.bool_)
        self.material_is_plant = np.zeros(1, dtype=np.bool_)
        self.material_is_placeholder = np.zeros(1, dtype=np.bool_)
        self.material_collapse_behavior = np.zeros(1, dtype=np.uint8)
        self.material_collapse_generation_id = np.zeros(1, dtype=np.int32)
        self.material_powder_generation_id = np.zeros(1, dtype=np.int32)
        self.material_name_by_id: list[str] = [""]
        self.tag_bits_by_name: dict[str, int] = {}
        self.random_convert_material_ids: list[int] = []
        self.placeholder_material_id = 0

        self.gas_material_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
        self.gas_light_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
        self.gas_density_factor = np.zeros(1, dtype=np.float32)
        self.gas_condense_material_id = np.zeros(1, dtype=np.int32)
        self.gas_name_by_id: list[str] = []
        self.air_gas_species_id = -1

        self.light_default_range = np.zeros(1, dtype=np.int32)
        self.light_dose_channel = np.zeros(1, dtype=np.int32)
        self.light_color = np.zeros((1, 3), dtype=np.float32)
        self.light_name_by_id: list[str] = [""]
        self._stable_shadow_payloads: dict[str, Any] = {}

        self.gas_solver = GasSolver()
        self.heat_solver = HeatSolver()
        self.collapse_solver = CollapseSolver()
        self.motion_solver = MotionSolver()
        self.liquid_solver = LiquidSolver()
        self.optics_solver = OpticsSolver()
        self.reaction_solver = ReactionSolver()
        self.merge_pipeline = GPUMergePipeline()
        self.placeholder_pipeline = GPUPlaceholderPipeline()
        self.page_stripe_pipeline = GPUPageStripePipeline()
        self.grid_command_pipeline = GPUWorldCommandPipeline()

        self.bootstrap_defaults()
        self.reset_world()
        self.bridge.sync_world(self)
        if self.simulation_backend == "gpu":
            self.bridge.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
                "collapse_delay_pending",
                "gas_concentration",
                "ambient_temperature",
                "flow_velocity",
                "pressure_ping",
                "visible_illumination",
                "cell_optical_dose",
                "gas_optical_dose",
                "active_meta",
                "active_tile_ttl",
                "active_chunk_mask",
            )
            self._gpu_cpu_dirty_resources.clear()
        self._closed = False

    def use_cpu_oracle_backend(self) -> None:
        self.simulation_backend = "cpu"

    def require_gpu_world_backend(self) -> None:
        self.simulation_backend = "gpu"

    def prewarm_formal_connected_collapse(self) -> bool:
        if self.simulation_backend != "gpu":
            return False
        pipeline = self.collapse_solver.gpu_pipeline
        if not pipeline.available(self):
            return False
        pipeline.prewarm_formal_connected_resources(self)
        return True

    def _gpu_context_available(self) -> bool:
        ctx = self.bridge.ctx
        return bool(self.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)

    def _gpu_world_simulation_required(self) -> bool:
        return self.simulation_backend == "gpu"

    def _gpu_realtime_budget_active(self) -> bool:
        if not (self.gpu_realtime_budget_enabled and self.simulation_backend == "gpu"):
            return False
        active_tile_count = self._gpu_active_tile_count()
        if active_tile_count <= 0:
            return False
        estimated_active_cells = active_tile_count * int(self.active.tile_size) * int(self.active.tile_size)
        return estimated_active_cells >= int(self.gpu_realtime_budget_cell_threshold)

    def _gpu_active_tile_count(self) -> int:
        if "active_tile_ttl" in self.bridge.gpu_authoritative_resources:
            active_meta = self.bridge.shadow_buffers.get("active_meta")
            if isinstance(active_meta, np.ndarray) and active_meta.size > 0:
                return int(active_meta[0]["active_tile_count"])
            return 0
        active_tile_ttl = np.asarray(self.active.active_tile_ttl, dtype=np.int32)
        if active_tile_ttl.size <= 0:
            return 0
        return int(np.count_nonzero(active_tile_ttl > 0))

    def _skip_budgeted_gpu_stage(self, stage: str) -> bool:
        return False

    def _should_run_formal_collapse_this_frame(self) -> bool:
        if self.simulation_backend != "gpu":
            return True
        interval = max(1, int(getattr(self, "formal_collapse_interval_frames", 1)))
        if interval <= 1:
            return True
        frame_id = max(1, int(getattr(self, "frame_id", 1)))
        return (frame_id - 1) % interval == 0

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
            if profile is None:
                return
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

    def _gpu_pipeline_available(self, pipeline: Any, name: str, *, require: bool | None = None) -> bool:
        if self.simulation_backend == "cpu":
            return False
        available = bool(pipeline.available(self))
        required = self._gpu_world_simulation_required() if require is None else bool(require)
        if required and not available:
            raise RuntimeError(f"GPU world simulation requires the {name} GPU pipeline; CPU fallback is disabled")
        return available

    def _require_gpu_stage(self, name: str) -> None:
        if self._gpu_world_simulation_required():
            raise RuntimeError(f"GPU world simulation requires GPU support for {name}; CPU fallback is disabled")

    def _require_gpu_authoritative_resources(self, stage: str, *resource_names: str) -> None:
        if not (self.simulation_backend == "gpu" and self._world_simulation_frame_active):
            return
        missing = [str(name) for name in resource_names if str(name) not in self.bridge.gpu_authoritative_resources]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"GPU world simulation requires GPU-authoritative {stage} resources: {joined}; "
                "CPU fallback is disabled"
            )

    def _require_cpu_oracle_backend(self, name: str) -> None:
        if self.simulation_backend != "cpu":
            raise RuntimeError(
                f"{name} CPU oracle path requires simulation_backend='cpu'; CPU fallback is disabled"
            )

    def _invalidate_gpu_authoritative_resources(self, *resource_names: str) -> None:
        if self.simulation_backend == "gpu":
            self.bridge.clear_gpu_authoritative(*resource_names)
            self._bridge_inputs_prepared = False
            if not self._world_simulation_frame_active:
                self._gpu_cpu_dirty_resources.update(str(name) for name in resource_names)

    def _invalidate_gpu_authoritative_cell_resources(self) -> None:
        self._invalidate_gpu_authoritative_resources(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
            "liquid_flow_intent",
        )

    def bootstrap_defaults(self) -> None:
        payloads = build_default_payloads()
        self.update_material_table(payloads["materials"])
        self.update_gas_species_table(payloads["gases"])
        self.update_light_type_table(payloads["lights"])
        self.update_material_optics_table(payloads["optics"])
        self.update_reaction_table(payloads["actions"], payloads["rules"])

    _material_table_snapshot_payload = _material_table_snapshot_payload

    _gas_species_table_snapshot_payload = _gas_species_table_snapshot_payload

    _light_type_table_snapshot_payload = _light_type_table_snapshot_payload

    _material_optics_table_snapshot_payload = _material_optics_table_snapshot_payload

    def _material_optics_snapshot_map(self) -> dict[tuple[str, str], MaterialOpticsDef]:
        return _material_optics_snapshot_map(self)

    _reaction_table_snapshot_payload = _reaction_table_snapshot_payload

    _stable_shadow_payload = _stable_shadow_payload

    _set_stable_shadow_payload = _set_stable_shadow_payload

    _shadow_has_table_payload = _shadow_has_table_payload

    _merged_reaction_table_payload = _merged_reaction_table_payload

    _merged_material_table_payload = _merged_material_table_payload

    _merged_gas_species_table_payload = _merged_gas_species_table_payload

    _merged_light_type_table_payload = _merged_light_type_table_payload

    _merged_material_optics_table_payload = _merged_material_optics_table_payload

    @staticmethod
    def _coerce_enum(enum_type: type[Any], value: Any) -> Any:
        return _coerce_enum(enum_type, value)

    _coerce_material_def = _coerce_material_def

    _coerce_gas_species_def = _coerce_gas_species_def

    _coerce_light_type_def = _coerce_light_type_def

    @staticmethod
    def _canonical_material_input_name(name: str | None) -> str | None:
        return _canonical_material_input_name(name)

    _coerce_material_optics_def = _coerce_material_optics_def

    _coerce_reaction_action = _coerce_reaction_action

    _coerce_pair_reaction_rule = _coerce_pair_reaction_rule

    _coerce_self_reaction_rule = _coerce_self_reaction_rule

    _coerce_reaction_rules = _coerce_reaction_rules

    _shadow_material_payload = _shadow_material_payload

    _shadow_gas_species_payload = _shadow_gas_species_payload

    _shadow_light_type_payload = _shadow_light_type_payload

    _shadow_reaction_payload = _shadow_reaction_payload

    @staticmethod
    def _payload_name_set(payload: Iterable[dict[str, Any]], field: str = "name") -> set[str]:
        return _payload_name_set(payload, field=field)

    @staticmethod
    def _validate_named_reference(valid_names: set[str], reference: str | None) -> None:
        return _validate_named_reference(valid_names, reference)

    @staticmethod
    def _validate_unique_identity_fields(
        payload: Iterable[dict[str, Any]],
        *,
        id_field: str,
        name_field: str = "name",
        allow_zero_id: bool,
    ) -> None:
        return _validate_unique_identity_fields(
            payload,
            id_field=id_field,
            name_field=name_field,
            allow_zero_id=allow_zero_id,
        )

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

    def _coerce_emitter(self, emitter: dict[str, Any]) -> dict[str, object]:
        return _coerce_emitter(self, emitter)

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

    @staticmethod
    def _normalize_readback_channels(channels: Any) -> tuple[str, ...]:
        return _normalize_readback_channels(channels)

    _normalize_readback_request = _normalize_readback_request

    _assign_readback_request_id = _assign_readback_request_id

    _assign_preview_readback_request_ids = _assign_preview_readback_request_ids

    _coerce_world_command = _coerce_world_command

    @classmethod
    def _coerce_json_value(cls, value: Any) -> Any:
        return _coerce_json_value(value)

    @classmethod
    def _normalize_json_payload_value(cls, value: Any) -> Any:
        return _normalize_json_payload_value(value)

    _coerce_world_frame_input = _coerce_world_frame_input

    def _gas_field_count(self) -> int:
        return max(self.rulebook.gases_by_id, default=-1) + 1

    def _light_field_count(self) -> int:
        return _light_field_count(self)

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
        for frame_input in reversed(self.pending_frame_inputs):
            if frame_input.submission_id == int(submission_id):
                return frame_input
        raise KeyError(f"missing pending frame submission_id={submission_id}")

    @staticmethod
    def _frame_readback_request_ids(frame_input: WorldFrameInput) -> list[int]:
        return [
            int(request.request_id)
            for request in frame_input.readback_requests
            if request.request_id is not None
        ]

    cancel_frame_submission = cancel_frame_submission

    def cancel_all_pending_frame_submissions(self) -> list[int]:
        canceled = self.pending_frame_submission_ids()
        canceled_readback_ids: list[int] = []
        for frame_input in self.pending_frame_inputs:
            canceled_readback_ids.extend(self._frame_readback_request_ids(frame_input))
        self.pending_frame_inputs.clear()
        self.canceled_frame_submission_ids.update(canceled)
        self.canceled_readback_request_ids.update(canceled_readback_ids)
        return canceled

    cancel_readback_request = cancel_readback_request

    poll_frame_output = poll_frame_output

    poll_all_frame_outputs = poll_all_frame_outputs

    serialize_pending_commands = serialize_pending_commands

    serialize_readback_state = serialize_readback_state

    serialize_bridge_runtime = serialize_bridge_runtime

    @staticmethod
    def _serialize_bridge_resource_summary(name: str, array: np.ndarray) -> dict[str, Any]:
        return _serialize_bridge_resource_summary(name, array)

    serialize_bridge_resources = serialize_bridge_resources

    serialize_ready_readbacks = serialize_ready_readbacks

    def readback_request_status(self, request_id: int) -> str:
        return readback_request_status(self, request_id)

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

    def capture_page_stripe_to_store(self, update: PageStripeUpdate) -> dict[str, Any]:
        return self.store_page_stripe(update, self.capture_page_stripe(update))

    load_page_stripe = load_page_stripe

    apply_stored_page_stripe = apply_stored_page_stripe

    page_store_has_stripe = page_store_has_stripe

    list_page_store_stripe_keys = list_page_store_stripe_keys

    export_page_store_entries = export_page_store_entries

    import_page_store_entries = import_page_store_entries

    clear_page_store = clear_page_store

    serialize_page_store_state = serialize_page_store_state

    _coerce_page_store_key = _coerce_page_store_key

    @staticmethod
    def _page_store_key_lookup_update(key: StoredStripeKey) -> PageStripeUpdate:
        return _page_store_key_lookup_update(key)

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

    def simulation_backend_report(self) -> dict[str, Any]:
        return simulation_backend_report(self)

    def poll_readbacks(self, request_id: int | None = None) -> ReadbackResult | None:
        if request_id is None:
            if not self.completed_readbacks:
                return None
            return self.completed_readbacks.popleft()
        for index, result in enumerate(self.completed_readbacks):
            if result.request.request_id == int(request_id):
                del self.completed_readbacks[index]
                return result
        return None

    def poll_all_readbacks(self, *, current_frame_id: int | None = None) -> list[ReadbackResult]:
        results: list[ReadbackResult] = []
        if current_frame_id is not None:
            self._collect_ready_readbacks(current_frame_id)
        while self.completed_readbacks:
            results.append(self.completed_readbacks.popleft())
        return results

    consume_entity_observation_results = consume_entity_observation_results

    run_entity_controller_turn = run_entity_controller_turn

    set_controller_state = set_controller_state

    serialize_controller_state = serialize_controller_state

    _build_preview_controller_turn_entities = _build_preview_controller_turn_entities

    _preview_consume_entity_observation_results = _preview_consume_entity_observation_results

    controller_turn_to_frame_input = controller_turn_to_frame_input

    preview_entity_controller_turn = preview_entity_controller_turn

    def submit_entity_controller_turn(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> int:
        return submit_entity_controller_turn(
            self,
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )

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

    def _contextualize_page_stripe_update(self, update: PageStripeUpdate) -> PageStripeUpdate:
        if update.axis == "x":
            default_cross_start = int(self.paging.origin_y)
            default_cross_end = int(self.paging.origin_y + self.height)
        else:
            default_cross_start = int(self.paging.origin_x)
            default_cross_end = int(self.paging.origin_x + self.width)
        cross_world_start = default_cross_start if update.cross_world_start is None else int(update.cross_world_start)
        cross_world_end = default_cross_end if update.cross_world_end is None else int(update.cross_world_end)
        if (
            update.cross_world_start == cross_world_start
            and update.cross_world_end == cross_world_end
        ):
            return update
        return replace(
            update,
            cross_world_start=cross_world_start,
            cross_world_end=cross_world_end,
        )

    @staticmethod
    def _page_store_key(update: PageStripeUpdate) -> tuple[str, int, int, int, int]:
        return (
            str(update.axis),
            int(update.world_start),
            int(update.world_end),
            0 if update.cross_world_start is None else int(update.cross_world_start),
            0 if update.cross_world_end is None else int(update.cross_world_end),
        )

    def _preview_apply_paging_updates(
        self,
        updates: list[PageStripeUpdate],
    ) -> list[tuple[PageStripeUpdate, dict[str, Any]]]:
        preview_saved_payloads: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
        preview_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        for update in updates:
            if update.kind != "save":
                continue
            preview_saved_payloads[self._page_store_key(update)] = self.capture_page_stripe(update)
            self._clear_saved_page_stripe_runtime_state(update)
        for update in updates:
            if update.kind != "load":
                continue
            payload = preview_saved_payloads.get(self._page_store_key(update))
            if payload is None:
                payload = self.page_store.load(update)
            if payload is None:
                payload = self._default_page_stripe_payload(update)
            self._apply_page_stripe(update, payload)
            preview_page_stripes.append((PageStripeUpdate(**asdict(update)), deepcopy(payload)))
        return preview_page_stripes

    def _preview_bridge_placeholder_dirty_rects(
        self,
        current_entity_placeholders: dict[int, set[tuple[int, int]]],
        placeholders: list[EntityPlaceholder],
    ) -> list[dict[str, Any]]:
        return _preview_bridge_placeholder_dirty_rects(self, current_entity_placeholders, placeholders)

    @staticmethod
    def _serialize_bridge_readback_request_stages(
        requests: list[ReadbackRequest],
        *,
        stage: str | None = None,
        reserved_request_ids: set[int] | None = None,
        observation_request_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        return _serialize_bridge_readback_request_stages(
            requests,
            stage=stage,
            reserved_request_ids=reserved_request_ids,
            observation_request_ids=observation_request_ids,
        )

    @staticmethod
    def _serialize_bridge_index_stages(
        values: Sequence[Any],
        *,
        stage: str,
    ) -> list[dict[str, Any]]:
        return _serialize_bridge_index_stages(values, stage=stage)

    _serialize_preview_bridge_frame_snapshot = _serialize_preview_bridge_frame_snapshot

    _queue_loaded_collapse_pending_regions = _queue_loaded_collapse_pending_regions

    def _clear_saved_page_stripe_runtime_state(self, update: PageStripeUpdate) -> None:
        if update.kind != "save":
            return
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                self.active.clear_rect(start, 0, end, self.height)
            else:
                self.active.clear_rect(0, start, self.width, end)
        self.collapse_dirty_regions = self._prune_page_stripe_regions(self.collapse_dirty_regions, update)
        self.collapse_deferred_regions = self._prune_page_stripe_regions(self.collapse_deferred_regions, update)

    def _prune_page_stripe_regions(
        self,
        regions: Iterable[tuple[int, int, int, int]],
        update: PageStripeUpdate,
    ) -> list[tuple[int, int, int, int]]:
        next_regions = [tuple(int(value) for value in region) for region in regions]
        if update.kind != "save":
            return next_regions
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            pruned: list[tuple[int, int, int, int]] = []
            for region in next_regions:
                pruned.extend(self._subtract_page_stripe_range_from_region(region, axis=update.axis, start=start, end=end))
            next_regions = pruned
        return next_regions

    @staticmethod
    def _subtract_page_stripe_range_from_region(
        region: tuple[int, int, int, int],
        *,
        axis: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int, int, int]]:
        return _subtract_page_stripe_range_from_region(region, axis=axis, start=start, end=end)

    def close(self) -> None:
        return close(self)

    def __del__(self) -> None:  # pragma: no cover
        _world_engine_del(self)

    material_by_id = material_by_id

    allocate_island_id = allocate_island_id

    def _refresh_island_records_for_ids(self, island_ids: Iterable[int]) -> None:
        return _refresh_island_records_for_ids(self, island_ids)

    in_bounds = in_bounds

    def cell_xy_to_gas(self, x: int, y: int) -> tuple[int, int]:
        """Map a cell-space (x, y) pair onto the lower-resolution gas grid."""
        return cell_xy_to_gas(self, x, y)

    def cell_to_gas(self, y: int, x: int) -> tuple[int, int]:
        """Map a cell-space (y, x) pair onto the lower-resolution gas grid."""
        return self.cell_xy_to_gas(x, y)

    sample_ambient_to_cells = sample_ambient_to_cells

    def ambient_temperature_at_cell(self, x: int, y: int) -> float:
        gy, gx = self.cell_xy_to_gas(x, y)
        return float(self.ambient_temperature[gy, gx])

    def ambient_temperature_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        if x0 >= x1 or y0 >= y1:
            return np.zeros((0, 0), dtype=np.float32)
        gx0 = min(self.gas_width, max(0, x0 // self.gas_cell_size))
        gy0 = min(self.gas_height, max(0, y0 // self.gas_cell_size))
        gx1 = min(self.gas_width, max(gx0 + 1, (x1 + self.gas_cell_size - 1) // self.gas_cell_size))
        gy1 = min(self.gas_height, max(gy0 + 1, (y1 + self.gas_cell_size - 1) // self.gas_cell_size))
        repeated = np.repeat(
            np.repeat(self.ambient_temperature[gy0:gy1, gx0:gx1], self.gas_cell_size, axis=0),
            self.gas_cell_size,
            axis=1,
        )
        local_y0 = y0 - gy0 * self.gas_cell_size
        local_x0 = x0 - gx0 * self.gas_cell_size
        return np.ascontiguousarray(repeated[local_y0 : local_y0 + (y1 - y0), local_x0 : local_x0 + (x1 - x0)])

    sample_flow_to_cells = sample_flow_to_cells

    def downsample_cells_to_gas(self, field: np.ndarray) -> np.ndarray:
        return downsample_cells_to_gas(self, field)

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

    @staticmethod
    def _bridge_row_count(array: np.ndarray) -> int:
        return int(array.shape[0]) if array.ndim > 0 else 0

    @classmethod
    def _normalize_bridge_slice_bounds(cls, array: np.ndarray, *, offset: int = 0, limit: int | None = None) -> tuple[int, int]:
        row_count = cls._bridge_row_count(array)
        start = min(max(0, int(offset)), row_count)
        if limit is None:
            end = row_count
        else:
            end = min(row_count, start + max(0, int(limit)))
        return start, end

    @staticmethod
    def _normalize_bridge_window_bounds(
        array: np.ndarray,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> tuple[int, int, int, int]:
        if array.ndim < 2:
            raise ValueError("bridge shadow buffer window requires at least 2 dimensions")
        width = int(array.shape[-1])
        height = int(array.shape[-2])
        x0 = min(max(0, int(x)), width)
        y0 = min(max(0, int(y)), height)
        if w is None:
            x1 = width
        else:
            x1 = min(width, x0 + max(0, int(w)))
        if h is None:
            y1 = height
        else:
            y1 = min(height, y0 + max(0, int(h)))
        return x0, y0, x1, y1

    def _clamped_gas_window(self, gas_x: int, gas_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        min_gas_x = int(self.paging.origin_x) // int(self.gas_cell_size)
        min_gas_y = int(self.paging.origin_y) // int(self.gas_cell_size)
        max_gas_x = min_gas_x + int(self.gas_width)
        max_gas_y = min_gas_y + int(self.gas_height)
        clamped_gas_x = min_gas_x if self.gas_width <= 0 else max(min_gas_x, min(max_gas_x - 1, int(gas_x)))
        clamped_gas_y = min_gas_y if self.gas_height <= 0 else max(min_gas_y, min(max_gas_y - 1, int(gas_y)))
        span_x = max(0, int(width))
        span_y = max(0, int(height))
        return (
            int(clamped_gas_x),
            int(clamped_gas_y),
            int(min(max_gas_x, clamped_gas_x + span_x)),
            int(min(max_gas_y, clamped_gas_y + span_y)),
        )

    def _bridge_shadow_buffer_coord_space(self, array: np.ndarray) -> str | None:
        return _bridge_shadow_buffer_coord_space(self, array)

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

    @staticmethod
    def _decode_bridge_uploaded_command(meta_record: np.ndarray, payload_bytes: np.ndarray) -> dict[str, Any]:
        return _decode_bridge_uploaded_command(meta_record, payload_bytes)

    @staticmethod
    def _decode_bridge_uploaded_label(meta_record: np.ndarray, label_bytes: np.ndarray) -> str:
        return _decode_bridge_uploaded_label(meta_record, label_bytes)

    @staticmethod
    def _decode_bridge_uploaded_page_stripe_section(section_record: np.ndarray, payload_bytes: np.ndarray) -> np.ndarray:
        return _decode_bridge_uploaded_page_stripe_section(section_record, payload_bytes)

    @staticmethod
    def _set_nested_payload_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        return _set_nested_payload_value(payload, path, value)

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
        pairs: list[tuple[ObservationTarget, ReadbackRequest]] = []
        for target in targets:
            request = self._build_observation_request(target, resolved_targets)
            if request is not None:
                pairs.append((target, request))
        return pairs

    serialize_readback_plan = serialize_readback_plan

    serialize_observation_plan = serialize_observation_plan

    serialize_world_command = serialize_world_command

    @staticmethod
    def serialize_entity_placeholder_input(placeholder: EntityPlaceholder) -> dict[str, Any]:
        return serialize_entity_placeholder_input(placeholder)

    @staticmethod
    def serialize_target_query_input(query: TargetQuery) -> dict[str, Any]:
        return serialize_target_query_input(query)

    @staticmethod
    def serialize_page_stripe_update(update: PageStripeUpdate) -> dict[str, Any]:
        return serialize_page_stripe_update(update)

    @staticmethod
    def serialize_page_store_key(key: StoredStripeKey) -> dict[str, Any]:
        return serialize_page_store_key(key)

    @classmethod
    def serialize_page_stripe_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return serialize_page_stripe_payload(cls, payload)

    @staticmethod
    def serialize_change_intent_input(intent: ChangeIntent) -> dict[str, Any]:
        return serialize_change_intent_input(intent)

    @staticmethod
    def serialize_carrier_intent_input(intent: CarrierIntent) -> dict[str, Any]:
        return serialize_carrier_intent_input(intent)

    serialize_frame_input = serialize_frame_input

    _serialize_readback_payload = _serialize_readback_payload

    serialize_readback_result = serialize_readback_result

    serialize_resolved_target = serialize_resolved_target

    serialize_resolved_change_intent = serialize_resolved_change_intent

    serialize_resolved_carrier_intent = serialize_resolved_carrier_intent

    serialize_observation_result = serialize_observation_result

    @staticmethod
    def serialize_entity_observation_spec(spec: EntityObservationSpec) -> dict[str, Any]:
        return serialize_entity_observation_spec(spec)

    serialize_entity_state_patch = serialize_entity_state_patch

    @staticmethod
    def serialize_observation_target(target: ObservationTarget) -> dict[str, Any]:
        return serialize_observation_target(target)

    @staticmethod
    def serialize_entity_state_input(entity: EntityState) -> dict[str, Any]:
        return serialize_entity_state_input(entity)

    serialize_entity_state = serialize_entity_state

    serialize_entity_states = serialize_entity_states

    serialize_entity_observation_state = serialize_entity_observation_state

    def _current_cell_state_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        return _current_cell_state_snapshot(self, allow_gpu_sync_readback=allow_gpu_sync_readback)

    def _current_entity_runtime_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        return _current_entity_runtime_snapshot(self, allow_gpu_sync_readback=allow_gpu_sync_readback)

    def _entity_placeholder_state_gpu_authoritative(self) -> bool:
        return _entity_placeholder_state_gpu_authoritative(self)

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
        active_rect, collapse_rect = self._grid_world_command_runtime_regions(command)
        if active_rect is not None:
            x0, y0, x1, y1, tile_padding = active_rect
            self._mark_active_rect_runtime(x0, y0, x1, y1, tile_padding=tile_padding)
        if collapse_rect is not None:
            self._mark_collapse_dirty_rect(*collapse_rect)

    _apply_commands = _apply_commands

    _finish_readbacks = _finish_readbacks

    _collect_ready_readbacks = _collect_ready_readbacks

    def _queued_command_xy(self, command: WorldCommand) -> tuple[int, int]:
        x = int(command.payload["x"])
        y = int(command.payload["y"])
        if any(
            key in command.payload
            for key in ("resolved_target_query_id", "resolved_change_intent_id", "resolved_carrier_intent_id")
        ):
            return self._world_to_buffer_clamped(x, y)
        return self._world_to_buffer_clamped(x, y)

    def _make_readback_payload(self, request: ReadbackRequest) -> dict[str, Any]:
        return make_readback_payload(self, request)

    def _gas_window_for_cell_rect(self, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        gx0 = min(self.gas_width, max(0, x0 // self.gas_cell_size))
        gy0 = min(self.gas_height, max(0, y0 // self.gas_cell_size))
        gx1 = min(self.gas_width, max(gx0, (x1 + self.gas_cell_size - 1) // self.gas_cell_size))
        gy1 = min(self.gas_height, max(gy0, (y1 + self.gas_cell_size - 1) // self.gas_cell_size))
        return (gx0, gy0, gx1, gy1)

    _apply_page_stripe = _apply_page_stripe

    _apply_page_stripe_dense_cpu = _apply_page_stripe_dense_cpu

    _queue_loaded_collapse_pending_regions_from_payload = _queue_loaded_collapse_pending_regions_from_payload

    def _advance_paging(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        return _advance_paging(self, center_x, center_y)

    _prepare_bridge_frame_inputs = _prepare_bridge_frame_inputs

    _needs_pre_simulation_bridge_sync = _needs_pre_simulation_bridge_sync

    def _sync_pre_simulation_bridge_without_debug_upload(self) -> None:
        try:
            self.bridge.sync_world(self, upload_debug_texture=False)
        except TypeError as exc:
            if "upload_debug_texture" not in str(exc):
                raise
            self.bridge.sync_world(self)

    _clear_bridge_frame_inputs = _clear_bridge_frame_inputs

    _mark_active_rect_runtime = _mark_active_rect_runtime

    _mark_active_rects_runtime = _mark_active_rects_runtime

    _sync_entity_placeholders = _sync_entity_placeholders

    def _sync_force_sources(self, force_sources: list[ForceSource]) -> None:
        self.force_sources = [
            self._normalize_runtime_force_source(
                force_source
                if isinstance(force_source, ForceSource)
                else ForceSource(**force_source)
            )
            for force_source in force_sources
        ]
        for force_source in self.force_sources:
            radius = int(np.ceil(force_source.radius))
            x = int(round(force_source.x))
            y = int(round(force_source.y))
            self._mark_active_rect_runtime(
                max(0, x - radius),
                max(0, y - radius),
                min(self.width, x + radius + 1),
                min(self.height, y + radius + 1),
            )

    def _append_force_source_immediate(self, force_source: ForceSource) -> None:
        self.force_sources.append(self._normalize_runtime_force_source(force_source))
        radius = int(np.ceil(force_source.radius))
        x = int(round(self.force_sources[-1].x))
        y = int(round(self.force_sources[-1].y))
        self._mark_active_rect_runtime(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )

    def _sync_persistent_emitters(self, emitters: list[dict[str, object]]) -> None:
        self.persistent_emitters = [dict(emitter) for emitter in emitters]
        for emitter in self.persistent_emitters:
            radius = int(max(0, round(float(emitter["range_cells"]))))
            x = int(emitter["origin"][0])
            y = int(emitter["origin"][1])
            self._mark_active_rect_runtime(
                max(0, x - radius),
                max(0, y - radius),
                min(self.width, x + radius + 1),
                min(self.height, y + radius + 1),
            )

    def _append_transient_light_emitter_immediate(self, emitter: dict[str, object]) -> None:
        record = dict(emitter)
        self.emitters.append(record)
        radius = int(max(0, round(float(record["range_cells"]))))
        x = int(record["origin"][0])
        y = int(record["origin"][1])
        self._mark_active_rect_runtime(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )

    def _record_bridge_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        self.bridge_frame_page_stripes.append((PageStripeUpdate(**asdict(update)), deepcopy(payload)))

    _release_entity_placeholder_cell = _release_entity_placeholder_cell

    def _mirror_release_entity_placeholder_cell(self, x: int, y: int, entity_id: int) -> None:
        return _mirror_release_entity_placeholder_cell(self, x, y, entity_id)

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

    def _material_placeholder_mask(self, material_id: np.ndarray) -> np.ndarray:
        ids = np.asarray(material_id, dtype=np.int64)
        mask = np.zeros(ids.shape, dtype=np.bool_)
        valid = (ids >= 0) & (ids < int(self.material_is_placeholder.shape[0]))
        if np.any(valid):
            mask[valid] = self.material_is_placeholder[ids[valid]]
        return mask

    _shadow_material_is_plant = _shadow_material_is_plant

    _shadow_reaction_action = _shadow_reaction_action

    _reaction_rule_list = _reaction_rule_list

    def _set_reaction_rule_list(self, rule_set: str, entries: list[dict[str, Any]] | list[PairReactionRule] | list[SelfReactionRule]) -> None:
        normalized = str(rule_set)
        if normalized == "self_rules":
            normalized_entries = [self._coerce_self_reaction_rule(entry) for entry in entries]
            self.rulebook.self_rules = normalized_entries
            return
        normalized_entries = [self._coerce_pair_reaction_rule(entry) for entry in entries]
        if normalized == "material_material":
            self.rulebook.material_material_rules = normalized_entries
            return
        if normalized == "material_gas":
            self.rulebook.material_gas_rules = normalized_entries
            return
        if normalized == "material_light":
            self.rulebook.material_light_rules = normalized_entries
            return
        if normalized == "gas_gas":
            self.rulebook.gas_gas_rules = normalized_entries
            return
        if normalized == "gas_light":
            self.rulebook.gas_light_rules = normalized_entries
            return
        raise KeyError(rule_set)

    def _set_reaction_rules_payload(self, rules_payload: dict[str, list[dict[str, Any]]]) -> None:
        for rule_set in REACTION_RULE_SET_NAMES:
            self._set_reaction_rule_list(str(rule_set), list(rules_payload.get(str(rule_set), [])))

    @staticmethod
    def _remap_reaction_payload_result_actions(
        rules_payload: dict[str, list[dict[str, Any]]],
        *,
        deleted_action_index: int,
    ) -> None:
        for rule_set in PAIR_REACTION_RULE_SET_NAMES:
            for rule in rules_payload.get(str(rule_set), []):
                result_action = int(rule.get("result_action", -1))
                if result_action == deleted_action_index:
                    rule["result_action"] = 0
                elif result_action > deleted_action_index:
                    rule["result_action"] = result_action - 1

    @staticmethod
    def _remap_material_payload_reaction_slots(
        materials_payload: list[dict[str, Any]],
        *,
        deleted_action_index: int,
    ) -> None:
        for material in materials_payload:
            slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
            remapped_slots: list[int] = []
            for slot in slots[:8]:
                action_index = int(slot)
                if action_index == deleted_action_index:
                    remapped_slots.append(-1)
                elif action_index > deleted_action_index:
                    remapped_slots.append(action_index - 1)
                else:
                    remapped_slots.append(action_index)
            if len(remapped_slots) < 8:
                remapped_slots.extend([-1] * (8 - len(remapped_slots)))
            material["reaction_slots"] = tuple(remapped_slots)

    @staticmethod
    def _clamp_material_payload_reaction_slots(
        materials_payload: list[dict[str, Any]],
        *,
        action_count: int,
    ) -> None:
        for material in materials_payload:
            slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
            clamped_slots: list[int] = []
            for slot in slots[:8]:
                action_index = int(slot)
                if action_index < -1 or action_index >= action_count:
                    clamped_slots.append(-1)
                else:
                    clamped_slots.append(action_index)
            if len(clamped_slots) < 8:
                clamped_slots.extend([-1] * (8 - len(clamped_slots)))
            material["reaction_slots"] = tuple(clamped_slots)

    _shadow_reaction_rule = _shadow_reaction_rule

    _occupy_entity_placeholder_cell = _occupy_entity_placeholder_cell

    def _mirror_occupy_entity_placeholder_cell(self, x: int, y: int, placeholder: EntityPlaceholder) -> bool:
        if not self.in_bounds(x, y):
            return False
        placeholder_material_id = self._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
        if placeholder_material_id <= 0:
            return False
        material_id = int(self.material_id[y, x])
        if material_id != 0 and int(self.phase[y, x]) != int(Phase.LIQUID):
            return False
        self.set_cell_by_id(x, y, placeholder_material_id, mark_dirty=False)
        self.entity_id[y, x] = placeholder.entity_id
        self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
        return True

    _frame_entities_to_placeholders_and_observations = _frame_entities_to_placeholders_and_observations

    def _runtime_entities_to_immediate_observation_targets(
        self,
        entities: list[EntityState],
    ) -> list[ObservationTarget]:
        return _runtime_entities_to_immediate_observation_targets(self, entities)

    _sync_entity_states = _sync_entity_states

    _sync_entity_observation_specs = _sync_entity_observation_specs

    def _patch_entity_states(self, patches: list[EntityStatePatch]) -> None:
        next_entity_states = dict(self.entity_states)
        for patch in patches:
            entity = next_entity_states.get(patch.entity_id)
            if entity is None:
                raise KeyError(patch.entity_id)
            patch_fields = {name: value for name, value in patch.fields.items() if not name.startswith("_")}
            world_x = patch.fields.get(
                "_world_x",
                patch_fields.get("x", entity.world_x if entity.world_x is not None else entity.x),
            )
            world_y = patch.fields.get(
                "_world_y",
                patch_fields.get("y", entity.world_y if entity.world_y is not None else entity.y),
            )
            next_entity_states[patch.entity_id] = self._coerce_entity_state(
                replace(entity, **dict(patch_fields), world_x=int(world_x), world_y=int(world_y))
            )
        self.entity_states = next_entity_states
        placeholders, _ = self._frame_entities_to_placeholders_and_observations(list(self.entity_states.values()))
        self._sync_entity_placeholders(placeholders)

    _build_preview_entity_placeholders = _build_preview_entity_placeholders

    def _preview_can_occupy_placeholder_cell(
        self,
        x: int,
        y: int,
        placeholder: EntityPlaceholder,
        current_cells: dict[tuple[int, int], int],
        released_cells: set[tuple[int, int]],
    ) -> bool:
        if not self.in_bounds(x, y):
            return False
        placeholder_material_id = self._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
        if placeholder_material_id <= 0:
            return False
        material_id, phase = self._material_state_for_position(x, y, released_cells=released_cells)
        if current_cells.get((x, y)) == placeholder.entity_id and material_id > 0 and self._shadow_material_is_placeholder(material_id):
            return True
        return material_id == 0 or phase == int(Phase.LIQUID)

    def _material_state_for_position(
        self,
        x: int,
        y: int,
        *,
        blocked_cells: set[tuple[int, int]] | None = None,
        released_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        material_id = int(self.material_id[y, x])
        phase = int(self.phase[y, x])
        cell = (x, y)
        if released_cells is not None and cell in released_cells and material_id > 0 and self._shadow_material_is_placeholder(material_id):
            displaced_material = int(self.placeholder_displaced_material[y, x])
            if displaced_material > 0:
                return displaced_material, int(Phase.LIQUID)
            return 0, 0
        if blocked_cells is not None and cell in blocked_cells:
            return int(self.placeholder_material_id), int(Phase.STATIC_SOLID)
        return material_id, phase

    def _build_observation_requests(
        self,
        targets: list[ObservationTarget],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[ReadbackRequest]:
        return [request for _, request in self._build_observation_request_pairs(targets, resolved_targets)]

    _build_observation_request = _build_observation_request

    def _resolve_readback_requests(
        self,
        requests: list[ReadbackRequest],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[ReadbackRequest]:
        resolved: list[ReadbackRequest] = []
        for request in requests:
            concrete = self._resolve_readback_request(request, resolved_targets)
            if concrete is not None:
                resolved.append(concrete)
        return resolved

    _resolve_readback_request = _resolve_readback_request

    _resolve_target_queries = _resolve_target_queries

    def _resolve_change_intents(
        self,
        intents: list[ChangeIntent],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[dict[str, ResolvedChangeIntent], list[WorldCommand]]:
        resolved: dict[str, ResolvedChangeIntent] = {}
        commands: list[WorldCommand] = []
        for intent in intents:
            resolved_intent = self._resolve_change_intent(intent, resolved_targets)
            resolved[intent.intent_id] = resolved_intent
            commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
        return resolved, commands

    def _public_resolved_change_intent(self, intent: ResolvedChangeIntent) -> ResolvedChangeIntent:
        return _public_resolved_change_intent(self, intent)

    def _public_resolved_target(self, target: ResolvedTarget) -> ResolvedTarget:
        return replace(
            target,
            source_position=(
                None
                if target.source_world_position is None
                else tuple(int(value) for value in target.source_world_position)
            ),
            anchor_position=(
                None
                if target.anchor_world_position is None
                else tuple(int(value) for value in target.anchor_world_position)
            ),
            resolved_position=(
                None
                if target.resolved_world_position is None
                else tuple(int(value) for value in target.resolved_world_position)
            ),
        )

    def _resolve_carrier_intents(
        self,
        intents: list[CarrierIntent],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[dict[str, ResolvedCarrierIntent], list[WorldCommand]]:
        resolved: dict[str, ResolvedCarrierIntent] = {}
        commands: list[WorldCommand] = []
        for intent in intents:
            resolved_intent = self._resolve_carrier_intent(intent, resolved_targets)
            resolved[intent.intent_id] = resolved_intent
            commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
        return resolved, commands

    _public_resolved_carrier_intent = _public_resolved_carrier_intent

    _resolve_carrier_intent = _resolve_carrier_intent

    _resolve_change_intent = _resolve_change_intent

    _resolve_change_intent_world_position = _resolve_change_intent_world_position

    def _resolve_intent_world_position(
        self,
        *,
        target_query_id: str | None,
        center_x: int | None,
        center_y: int | None,
        target_dx: int,
        target_dy: int,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[int, int] | None:
        if target_query_id is not None:
            target = resolved_targets.get(target_query_id)
            if target is None or target.status not in {"resolved", "drifted"} or target.resolved_world_position is None:
                return None
            return (
                int(target.resolved_world_position[0]) + int(target_dx),
                int(target.resolved_world_position[1]) + int(target_dy),
            )
        if center_x is None or center_y is None:
            return None
        return (
            int(center_x) + int(target_dx),
            int(center_y) + int(target_dy),
        )

    def _resolve_intent_source_positions(
        self,
        *,
        source_entity_id: int | None,
        source_x: int | None,
        source_y: int | None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if source_entity_id is not None:
            entity = self.entity_states.get(int(source_entity_id))
            if entity is not None:
                world_position = self._entity_center_world_position(entity)
                return self._world_to_buffer_clamped(*world_position), world_position
        if source_x is not None and source_y is not None:
            world_position = (int(source_x), int(source_y))
            buffer_position = self._world_to_buffer_clamped(*world_position)
            return buffer_position, world_position
        buffer_position = self._default_target_source_position()
        return buffer_position, self._buffer_to_world_position(buffer_position)

    @staticmethod
    def _normalized_world_direction(
        source_world_position: tuple[int, int],
        target_world_position: tuple[int, int],
    ) -> tuple[float, float] | None:
        delta = np.asarray(target_world_position, dtype=np.float32) - np.asarray(source_world_position, dtype=np.float32)
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            return None
        direction = delta / length
        return (float(direction[0]), float(direction[1]))

    _disk_world_cells = _disk_world_cells

    @staticmethod
    def _disk_world_cells_raw(center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        return _disk_world_cells_raw(center_world_position, radius)

    _line_world_cells = _line_world_cells

    @staticmethod
    def _line_world_cells_raw(
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        return _line_world_cells_raw(start_world_position, end_world_position)

    _capsule_world_cells = _capsule_world_cells

    _capsule_world_cells_raw = _capsule_world_cells_raw

    @staticmethod
    def _buffer_cell_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        return _buffer_cell_bounds(cells)

    _apply_change_stability_drift = _apply_change_stability_drift

    _resolve_legal_world_position = _resolve_legal_world_position

    @staticmethod
    def _intent_resolution_status(*, drifted: bool, fallback_applied: bool) -> str:
        if fallback_applied:
            return "fallback"
        if drifted:
            return "drifted"
        return "resolved"

    @staticmethod
    def _combine_resolution_notes(*notes: str | None) -> str | None:
        filtered = [note for note in notes if note]
        if not filtered:
            return None
        return "; ".join(filtered)

    _resolve_targeted_commands = _resolve_targeted_commands

    _resolve_target_query = _resolve_target_query

    _resolve_target_query_distance_cells = _resolve_target_query_distance_cells

    @staticmethod
    def _distance_meters_to_cells(distance_meters: float) -> int:
        cells = int(round(float(distance_meters) * TARGET_QUERY_CELLS_PER_METER))
        if cells == 0 and abs(float(distance_meters)) > 1e-6:
            return 1 if distance_meters > 0.0 else -1
        return cells

    def _resolve_query_source_position(self, query: TargetQuery) -> tuple[int, int] | None:
        if query.source_entity_id is not None:
            entity = self.entity_states.get(int(query.source_entity_id))
            if entity is None:
                return None
            return self._world_to_buffer_clamped(*self._entity_center_world_position(entity))
        if query.source_x is not None and query.source_y is not None:
            return self._world_to_buffer_clamped(int(query.source_x), int(query.source_y))
        if query.source_x is None and query.source_y is None:
            return self._default_target_source_position()
        return None

    def _default_target_source_position(self) -> tuple[int, int]:
        return (
            (int(self.paging.buffer_origin_x) + int(self.paging.active_width) // 2) % self.width,
            (int(self.paging.buffer_origin_y) + int(self.paging.active_height) // 2) % self.height,
        )

    def _resolve_anchor_target(
        self,
        query: TargetQuery,
        source_world_position: tuple[int, int],
    ) -> dict[str, Any] | None:
        return _resolve_anchor_target(self, query, source_world_position)

    _resolve_entity_anchor = _resolve_entity_anchor

    _resolve_terrain_anchor = _resolve_terrain_anchor

    def _entity_matches_anchor_filters(self, entity: EntityState, filters: tuple[str, ...]) -> bool:
        area = max(1, int(entity.width) * int(entity.height))
        entity_tags = set(entity.tags)
        for item in filters:
            if item in TERRAIN_ANCHOR_FILTERS or item in IGNORED_ANCHOR_FILTERS:
                continue
            if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}:
                continue
            if item == "big":
                if area < 4:
                    return False
                continue
            if item == "small":
                if area > 1:
                    return False
                continue
            if item not in entity_tags:
                return False
        return True

    _terrain_cell_matches = _terrain_cell_matches

    def _terrain_tree_cell_matches(self, x: int, y: int, material_id: int, phase: int) -> bool:
        if material_id == 0 or phase in {int(Phase.LIQUID), int(Phase.POWDER)}:
            return False
        if not self._world_cell_material_has_tag(x, y, "plant"):
            return False
        if not (
            self._world_cell_material_has_tag(x, y - 1, "plant")
            or self._world_cell_material_has_tag(x, y + 1, "plant")
        ):
            return False
        plant_neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self._world_cell_material_has_tag(x + dx, y + dy, "plant"):
                    plant_neighbors += 1
        return plant_neighbors >= 2

    def _terrain_hill_cell_matches(self, x: int, y: int, material_id: int, phase: int) -> bool:
        if material_id == 0 or phase == int(Phase.LIQUID):
            return False
        if self._world_cell_material_has_tag(x, y, "plant") or self._world_cell_material_has_tag(x, y, "placeholder"):
            return False
        if not self._world_cell_is_empty_local(x, y - 1):
            return False
        if self._world_cell_is_solid_local(x - 1, y) or self._world_cell_is_solid_local(x + 1, y):
            return False
        left_support = self._world_cell_is_solid_local(x - 1, y + 1) or self._world_cell_is_solid_local(x - 1, y + 2)
        right_support = self._world_cell_is_solid_local(x + 1, y + 1) or self._world_cell_is_solid_local(x + 1, y + 2)
        return left_support and right_support

    _world_cell_is_solid_local = _world_cell_is_solid_local

    _world_cell_is_empty_local = _world_cell_is_empty_local

    def _world_cell_material_has_tag(self, x: int, y: int, tag: str) -> bool:
        material_id, _ = self._bounded_material_state_for_position(x, y)
        if material_id == 0:
            return False
        material_id = int(material_id)
        if tag == "plant":
            return self._shadow_material_is_plant(material_id)
        if tag == "placeholder":
            return self._shadow_material_is_placeholder(material_id)
        material = self._shadow_material_def(material_id)
        return material is not None and tag in material.tags

    _bounded_material_state_for_position = _bounded_material_state_for_position

    _matches_direction_filter = _matches_direction_filter

    _query_direction_vector = _query_direction_vector

    _direction_vector = _direction_vector

    def _source_facing_vector(self, source_entity_id: int | None) -> tuple[float, float]:
        if source_entity_id is not None:
            entity = self.entity_states.get(int(source_entity_id))
            if entity is not None:
                if entity.facing_xy is not None:
                    return (float(entity.facing_xy[0]), float(entity.facing_xy[1]))
                if entity.velocity_xy != (0.0, 0.0):
                    return (float(entity.velocity_xy[0]), float(entity.velocity_xy[1]))
        return (1.0, 0.0)

    def _entity_center_buffer_position(self, entity: EntityState) -> tuple[int, int]:
        return (
            int(entity.x) + max(0, int(entity.width) - 1) // 2,
            int(entity.y) + max(0, int(entity.height) - 1) // 2,
        )

    def _entity_center_world_position(self, entity: EntityState) -> tuple[int, int]:
        if entity.world_x is not None and entity.world_y is not None:
            return (
                int(entity.world_x) + max(0, int(entity.width) - 1) // 2,
                int(entity.world_y) + max(0, int(entity.height) - 1) // 2,
            )
        return self._buffer_to_world_position(self._entity_center_buffer_position(entity))

    _buffer_to_world_position = _buffer_to_world_position

    _buffer_to_world_float_position = _buffer_to_world_float_position

    _world_to_buffer_float_position = _world_to_buffer_float_position

    _force_source_world_position = _force_source_world_position

    _force_source_buffer_position = _force_source_buffer_position

    def _normalize_runtime_force_source(self, force_source: ForceSource) -> ForceSource:
        world_x, world_y = self._force_source_world_position(force_source)
        buffer_x, buffer_y = self._world_to_buffer_float_position((world_x, world_y))
        return replace(
            force_source,
            x=float(buffer_x),
            y=float(buffer_y),
            world_x=float(world_x),
            world_y=float(world_y),
        )

    _buffer_gas_to_world_position = _buffer_gas_to_world_position

    _buffer_bbox_to_world_bbox = _buffer_bbox_to_world_bbox

    _clamped_world_window = _clamped_world_window

    _centered_world_window = _centered_world_window

    _world_axis_spans = _world_axis_spans

    _world_axis_indices = _world_axis_indices

    _extract_world_window = _extract_world_window

    def _world_gas_window_for_cell_world_rect(
        self,
        world_x0: int,
        world_y0: int,
        world_x1: int,
        world_y1: int,
    ) -> tuple[int, int, int, int]:
        if world_x1 <= world_x0 or world_y1 <= world_y0:
            gas_world_x0 = int(world_x0) // int(self.gas_cell_size)
            gas_world_y0 = int(world_y0) // int(self.gas_cell_size)
            return (gas_world_x0, gas_world_y0, gas_world_x0, gas_world_y0)
        return (
            int(world_x0) // int(self.gas_cell_size),
            int(world_y0) // int(self.gas_cell_size),
            int((int(world_x1) + int(self.gas_cell_size) - 1) // int(self.gas_cell_size)),
            int((int(world_y1) + int(self.gas_cell_size) - 1) // int(self.gas_cell_size)),
        )

    _pack_cell_core_world_window = _pack_cell_core_world_window

    _world_to_buffer_clamped = _world_to_buffer_clamped

    def _clamp_world_position(self, world_x: int, world_y: int) -> tuple[int, int]:
        min_world_x = int(self.paging.origin_x)
        min_world_y = int(self.paging.origin_y)
        max_world_x = min_world_x + self.width - 1
        max_world_y = min_world_y + self.height - 1
        return (
            max(min_world_x, min(max_world_x, int(world_x))),
            max(min_world_y, min(max_world_y, int(world_y))),
        )

    _find_nearest_empty_world_position = _find_nearest_empty_world_position

    _world_cell_is_empty = _world_cell_is_empty

    @staticmethod
    def _world_distance_sq(left: tuple[int, int], right: tuple[int, int]) -> float:
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        return dx * dx + dy * dy

    def _entity_placeholder_bbox(self, entity_id: int) -> tuple[int, int, int, int] | None:
        cells = self.entity_placeholders.get(entity_id)
        if not cells:
            return None
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    _collect_observations = _collect_observations

    _collect_entity_feedback = _collect_entity_feedback

    _build_entity_feedback = _build_entity_feedback

    def _build_entity_feedback_from_world(self, entity: EntityState) -> EntityFeedback | None:
        cell_state = {
            "material_id": self.material_id,
            "phase": self.phase,
            "integrity": self.integrity,
        }
        entity_runtime = {
            "entity_id": self.entity_id,
            "placeholder_displaced_material": self.placeholder_displaced_material,
        }
        return self._build_entity_feedback_from_state(entity, cell_state=cell_state, entity_runtime=entity_runtime)

    def _build_entity_feedback_from_current_state(
        self,
        entity: EntityState,
        *,
        allow_gpu_sync_readback: bool = False,
    ) -> EntityFeedback | None:
        return self._build_entity_feedback_from_state(
            entity,
            cell_state=self._current_cell_state_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
            entity_runtime=self._current_entity_runtime_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
        )

    _build_entity_feedback_from_state = _build_entity_feedback_from_state

    def _capture_stripe_array(
        self,
        array: np.ndarray,
        update: PageStripeUpdate,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        spans = ranges if ranges is not None else self._stripe_buffer_ranges(update, gas_grid=False)
        parts: list[np.ndarray] = []
        for start, end in spans:
            slices = [slice(None)] * array.ndim
            slices[stripe_axis] = slice(start, end)
            parts.append(np.ascontiguousarray(array[tuple(slices)]))
        if not parts:
            return np.empty((0,), dtype=array.dtype)
        if len(parts) == 1:
            return parts[0].copy()
        return np.concatenate(parts, axis=stripe_axis)

    def _write_stripe_array(
        self,
        array: np.ndarray,
        update: PageStripeUpdate,
        values: np.ndarray,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        spans = ranges if ranges is not None else self._stripe_buffer_ranges(update, gas_grid=False)
        offset = 0
        for start, end in spans:
            span = end - start
            slices = [slice(None)] * array.ndim
            slices[stripe_axis] = slice(start, end)
            source_slices = [slice(None)] * values.ndim
            source_slices[stripe_axis] = slice(offset, offset + span)
            array[tuple(slices)] = values[tuple(source_slices)]
            offset += span

    _default_page_stripe_payload = _default_page_stripe_payload

    _stripe_buffer_ranges = _stripe_buffer_ranges

    def _mark_loaded_page_stripe_active(self, update: PageStripeUpdate) -> None:
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                self._mark_active_rect_runtime(start, 0, end, self.height, tile_padding=1)
            else:
                self._mark_active_rect_runtime(0, start, self.width, end, tile_padding=1)

    _rebuild_sparse_runtime_indexes = _rebuild_sparse_runtime_indexes

    _rebuild_entity_placeholder_index = _rebuild_entity_placeholder_index

    _normalize_cell_runtime_arrays = _normalize_cell_runtime_arrays

    _normalize_page_stripe_cell_runtime = _normalize_page_stripe_cell_runtime

    _capture_page_stripe_entity_placeholder_runtime = _capture_page_stripe_entity_placeholder_runtime

    _apply_page_stripe_entity_placeholder_runtime = _apply_page_stripe_entity_placeholder_runtime

    _rebuild_island_records = _rebuild_island_records

    _capture_page_stripe_island_runtime = _capture_page_stripe_island_runtime

    def _page_stripe_island_bboxes_from_payload(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
    ) -> dict[int, tuple[int, int, int, int]] | None:
        return _page_stripe_island_bboxes_from_payload(self, update, payload)

    _merge_island_runtime_payload = _merge_island_runtime_payload

    _rebuild_material_property_arrays = _rebuild_material_property_arrays

    _rebuild_gas_property_arrays = _rebuild_gas_property_arrays

    _rebuild_light_property_arrays = _rebuild_light_property_arrays

    _cell_participates_in_collapse = _cell_participates_in_collapse

    _mark_collapse_dirty_rect = _mark_collapse_dirty_rect

    _drain_gpu_collapse_structure_dirty_tiles = _drain_gpu_collapse_structure_dirty_tiles

    def _paint_material(self, x: int, y: int, material: str, radius: int) -> None:
        return _paint_material(self, x, y, material, radius)

    def _write_material_region_immediate(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        material: str,
    ) -> None:
        return _write_material_region_immediate(self, x, y, width, height, material)

    def _build_demo_scene(self) -> None:
        return _build_demo_scene(self)

    def _fill_rect(self, x: int, y: int, width: int, height: int, material: str) -> None:
        return _fill_rect(self, x, y, width, height, material)
