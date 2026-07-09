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
    unpack_cell_core,
)
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.page_store import InMemoryPageStore, PageStore, StoredStripeKey
from oracle_game.paging import RingPagingWindow
from oracle_game.rules import RuleBook, build_default_optics_entries, build_default_payloads
from oracle_game.sim.collapse import CollapseSolver
from oracle_game.sim.gas import GasSolver
from oracle_game.sim.gpu_collapse_dirty import clear_collapse_structure_dirty_tile_mask
from oracle_game.sim.heat import HeatSolver
from oracle_game.sim.liquid import LiquidSolver
from oracle_game.sim.motion import MotionSolver
from oracle_game.sim.optics import OpticsSolver
from oracle_game.sim.gpu_placeholders import GPUPlaceholderPipeline
from oracle_game.sim.gpu_page_stripes import GPUPageStripePipeline
from oracle_game.sim.gpu_world_commands import COMMAND_KIND_IDS as GPU_WORLD_COMMAND_KIND_IDS
from oracle_game.sim.gpu_world_commands import GPUWorldCommandPipeline
from oracle_game.sim.reactions import ReactionSolver
from oracle_game.sim.gpu_merge import GPUMergePipeline, MergeCandidates
from oracle_game.types import (
    CellFlag,
    CarrierIntent,
    ChangeIntent,
    DebugView,
    EntityCellFeedback,
    EntityFeedback,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityStatePatch,
    EntityState,
    FallingIslandRecord,
    ForceSource,
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    ObservationResult,
    ObservationTarget,
    PageStripeUpdate,
    PairReactionRule,
    Phase,
    ReactionAction,
    ReactionType,
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
    PUBLIC_WORLD_COMMAND_KINDS,
    REACTION_RULE_SET_NAMES,
    TARGET_QUERY_CELLS_PER_METER,
    TARGET_QUERY_DISTANCE_HINT_CELLS,
    TARGETED_COMMAND_COORD_FIELDS,
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

    def _material_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return _material_table_snapshot_payload(self)

    def _gas_species_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return _gas_species_table_snapshot_payload(self)

    def _light_type_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return _light_type_table_snapshot_payload(self)

    def _material_optics_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return _material_optics_table_snapshot_payload(self)

    def _material_optics_snapshot_map(self) -> dict[tuple[str, str], MaterialOpticsDef]:
        payload = self._stable_shadow_payload("optics", self._material_optics_table_snapshot_payload)
        snapshot: dict[tuple[str, str], MaterialOpticsDef] = {}
        for item in payload:
            entry = self._coerce_material_optics_def(item)
            snapshot[(entry.material_name, entry.light_type)] = entry
        return snapshot

    def _reaction_table_snapshot_payload(self) -> dict[str, object]:
        return _reaction_table_snapshot_payload(self)

    def _stable_shadow_payload(self, name: str, snapshot_factory: Callable[[], Any]) -> Any:
        return _stable_shadow_payload(self, name, snapshot_factory)

    def _set_stable_shadow_payload(self, name: str, payload: Any) -> None:
        return _set_stable_shadow_payload(self, name, payload)

    def _shadow_has_table_payload(self, name: str) -> bool:
        return _shadow_has_table_payload(self, name)

    def _merged_reaction_table_payload(
        self,
        actions: list[ReactionAction],
        rules: dict[str, list[object]],
    ) -> dict[str, object]:
        return _merged_reaction_table_payload(self, actions, rules)

    def _merged_material_table_payload(self, materials: list[MaterialDef]) -> list[dict[str, Any]]:
        return _merged_material_table_payload(self, materials)

    def _merged_gas_species_table_payload(self, gases: list[GasSpeciesDef]) -> list[dict[str, Any]]:
        return _merged_gas_species_table_payload(self, gases)

    def _merged_light_type_table_payload(self, lights: list[LightTypeDef]) -> list[dict[str, Any]]:
        return _merged_light_type_table_payload(self, lights)

    def _merged_material_optics_table_payload(self, optics: list[MaterialOpticsDef]) -> list[dict[str, Any]]:
        return _merged_material_optics_table_payload(self, optics)

    @staticmethod
    def _coerce_enum(enum_type: type[Any], value: Any) -> Any:
        return _coerce_enum(enum_type, value)

    def _coerce_material_def(self, material: MaterialDef | dict[str, Any]) -> MaterialDef:
        return _coerce_material_def(self, material)

    def _coerce_gas_species_def(self, gas: GasSpeciesDef | dict[str, Any]) -> GasSpeciesDef:
        return _coerce_gas_species_def(self, gas)

    def _coerce_light_type_def(self, light: LightTypeDef | dict[str, Any]) -> LightTypeDef:
        return _coerce_light_type_def(self, light)

    @staticmethod
    def _canonical_material_input_name(name: str | None) -> str | None:
        return _canonical_material_input_name(name)

    def _coerce_material_optics_def(self, optics: MaterialOpticsDef | dict[str, Any]) -> MaterialOpticsDef:
        return _coerce_material_optics_def(self, optics)

    def _coerce_reaction_action(self, action: ReactionAction | dict[str, Any]) -> ReactionAction:
        return _coerce_reaction_action(self, action)

    def _coerce_pair_reaction_rule(self, rule: PairReactionRule | dict[str, Any]) -> PairReactionRule:
        return _coerce_pair_reaction_rule(self, rule)

    def _coerce_self_reaction_rule(self, rule: SelfReactionRule | dict[str, Any]) -> SelfReactionRule:
        return _coerce_self_reaction_rule(self, rule)

    def _coerce_reaction_rules(self, rules: dict[str, object]) -> dict[str, list[object]]:
        return _coerce_reaction_rules(self, rules)

    def _shadow_material_payload(self) -> list[dict[str, Any]]:
        return _shadow_material_payload(self)

    def _shadow_gas_species_payload(self) -> list[dict[str, Any]]:
        return _shadow_gas_species_payload(self)

    def _shadow_light_type_payload(self) -> list[dict[str, Any]]:
        return _shadow_light_type_payload(self)

    def _shadow_reaction_payload(self) -> dict[str, Any]:
        return _shadow_reaction_payload(self)

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

    def _validate_material_table_payload(self, materials_payload: list[dict[str, Any]]) -> None:
        return _validate_material_table_payload(self, materials_payload)

    def _validate_gas_species_payload(self, gases_payload: list[dict[str, Any]]) -> None:
        return _validate_gas_species_payload(self, gases_payload)

    def _validate_light_type_payload(self, lights_payload: list[dict[str, Any]]) -> None:
        return _validate_light_type_payload(self, lights_payload)

    def _validate_material_optics_payload(self, optics_payload: list[dict[str, Any]]) -> None:
        return _validate_material_optics_payload(self, optics_payload)

    def _validate_reaction_payload(self, reactions_payload: dict[str, Any]) -> None:
        return _validate_reaction_payload(self, reactions_payload)

    def _normalize_material_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_material_patch_fields(self, fields)

    def _normalize_gas_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_gas_patch_fields(self, fields)

    def _normalize_material_optics_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_material_optics_patch_fields(self, fields)

    def _normalize_reaction_action_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_reaction_action_patch_fields(self, fields)

    def _normalize_reaction_rule_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_reaction_rule_patch_fields(self, fields)

    def _coerce_force_source(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        return _coerce_force_source(self, force_source)

    def _public_force_source_input(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        return _public_force_source_input(self, force_source)

    def _frame_force_source_input(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        return _frame_force_source_input(self, force_source)

    def _coerce_emitter(self, emitter: dict[str, Any]) -> dict[str, object]:
        return _coerce_emitter(self, emitter)

    def _frame_emitter_input(self, emitter: dict[str, Any]) -> dict[str, object]:
        return _frame_emitter_input(self, emitter)

    def _coerce_entity_placeholder(self, placeholder: EntityPlaceholder | dict[str, Any]) -> EntityPlaceholder:
        return _coerce_entity_placeholder(self, placeholder)

    def _public_entity_placeholder_input(
        self,
        placeholder: EntityPlaceholder | dict[str, Any],
    ) -> EntityPlaceholder:
        return _public_entity_placeholder_input(self, placeholder)

    def _frame_entity_placeholder_input(
        self,
        placeholder: EntityPlaceholder | dict[str, Any],
    ) -> EntityPlaceholder:
        return _frame_entity_placeholder_input(self, placeholder)

    def _coerce_entity_state(self, entity: EntityState | dict[str, Any]) -> EntityState:
        return _coerce_entity_state(self, entity)

    def _public_entity_state_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        return _public_entity_state_input(self, entity)

    def _frame_entity_state_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        return _frame_entity_state_input(self, entity)

    def _coerce_entity_observation_spec(
        self,
        spec: EntityObservationSpec | dict[str, Any],
    ) -> EntityObservationSpec:
        return _coerce_entity_observation_spec(self, spec)

    def _normalize_entity_state_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return _normalize_entity_state_patch_fields(self, fields)

    def _public_entity_state_patch_input(
        self,
        patch: EntityStatePatch | dict[str, Any],
    ) -> EntityStatePatch:
        return _public_entity_state_patch_input(self, patch)

    def _controller_turn_entity_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        return _controller_turn_entity_input(self, entity)

    def _frame_entity_state_patch_input(
        self,
        patch: EntityStatePatch | dict[str, Any],
    ) -> EntityStatePatch:
        return _frame_entity_state_patch_input(self, patch)

    def _coerce_entity_state_patch(self, patch: EntityStatePatch | dict[str, Any]) -> EntityStatePatch:
        return _coerce_entity_state_patch(self, patch)

    def _coerce_observation_target(self, target: ObservationTarget | dict[str, Any]) -> ObservationTarget:
        return _coerce_observation_target(self, target)

    def _coerce_target_query(self, query: TargetQuery | dict[str, Any]) -> TargetQuery:
        return _coerce_target_query(self, query)

    def _coerce_change_intent(self, intent: ChangeIntent | dict[str, Any]) -> ChangeIntent:
        return _coerce_change_intent(self, intent)

    def _coerce_carrier_intent(self, intent: CarrierIntent | dict[str, Any]) -> CarrierIntent:
        return _coerce_carrier_intent(self, intent)

    def _coerce_readback_request(self, request: ReadbackRequest | dict[str, Any]) -> ReadbackRequest:
        return _coerce_readback_request(self, request)

    @staticmethod
    def _normalize_readback_channels(channels: Any) -> tuple[str, ...]:
        return _normalize_readback_channels(channels)

    def _normalize_readback_request(self, request: ReadbackRequest) -> ReadbackRequest:
        return _normalize_readback_request(self, request)

    def _assign_readback_request_id(self, request: ReadbackRequest) -> ReadbackRequest:
        return _assign_readback_request_id(self, request)

    def _assign_preview_readback_request_ids(
        self,
        requests: list[ReadbackRequest],
        *,
        next_request_id: int | None = None,
    ) -> tuple[list[ReadbackRequest], int]:
        return _assign_preview_readback_request_ids(self, requests, next_request_id=next_request_id)

    def _coerce_world_command(self, command: WorldCommand | dict[str, Any]) -> WorldCommand:
        return _coerce_world_command(self, command)

    @classmethod
    def _coerce_json_value(cls, value: Any) -> Any:
        return _coerce_json_value(value)

    @classmethod
    def _normalize_json_payload_value(cls, value: Any) -> Any:
        return _normalize_json_payload_value(value)

    def _coerce_world_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> WorldFrameInput:
        return _coerce_world_frame_input(self, frame_input)

    def _gas_field_count(self) -> int:
        return max(self.rulebook.gases_by_id, default=-1) + 1

    def _light_field_count(self) -> int:
        max_light_id = max(self.rulebook.lights_by_id, default=-1)
        max_dose_channel = max((int(light.dose_channel_id) for light in self.rulebook.lights_by_id.values()), default=-1)
        return max(max_light_id, max_dose_channel) + 1

    def update_material_table(self, materials: list[MaterialDef | dict[str, Any]], *, immediate: bool = True) -> None:
        materials = [self._coerce_material_def(material) for material in materials]
        if not immediate:
            self.queue_command("update_material_table", materials=[asdict(material) for material in materials])
            return
        merged_payload = self._merged_material_table_payload(materials)
        self._validate_material_table_payload(merged_payload)
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(
            build_default_optics_entries(
                self.rulebook.materials_by_id.values(),
                self.rulebook.lights_by_id.values(),
                existing=self._material_optics_snapshot_map(),
            )
        )
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        optics_payload = self._material_optics_table_snapshot_payload()
        self._set_stable_shadow_payload("materials", merged_payload)
        self._set_stable_shadow_payload("optics", optics_payload)
        self.bridge.upload_table("materials", merged_payload)
        self.bridge.upload_table("optics", optics_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_material_table")
        self.bridge.ensure_world_resources(self)

    def update_gas_species_table(self, gases: list[GasSpeciesDef | dict[str, Any]], *, immediate: bool = True) -> None:
        gases = [self._coerce_gas_species_def(gas) for gas in gases]
        if not immediate:
            self.queue_command("update_gas_species_table", gases=[asdict(gas) for gas in gases])
            return
        merged_payload = self._merged_gas_species_table_payload(gases)
        self._validate_gas_species_payload(merged_payload)
        self.rulebook.gases_by_name.clear()
        self.rulebook.gases_by_id.clear()
        self.rulebook.update_gases(self._coerce_gas_species_def(item) for item in merged_payload)
        self._rebuild_gas_property_arrays()
        previous = self.gas_concentration
        gas_count = self._gas_field_count()
        self.gas_concentration = np.zeros((gas_count, self.gas_height, self.gas_width), dtype=np.float32)
        count = min(previous.shape[0], self.gas_concentration.shape[0])
        self.gas_concentration[:count] = previous[:count]
        if 0 <= self.air_gas_species_id < self.gas_concentration.shape[0]:
            self.gas_concentration[self.air_gas_species_id] = np.maximum(
                self.gas_concentration[self.air_gas_species_id], 1.0
            )
        self._set_stable_shadow_payload("gases", merged_payload)
        self.bridge.upload_table("gases", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_gas_species_table")
        self.bridge.ensure_world_resources(self)

    def update_light_type_table(self, lights: list[LightTypeDef | dict[str, Any]], *, immediate: bool = True) -> None:
        lights = [self._coerce_light_type_def(light) for light in lights]
        if not immediate:
            self.queue_command("update_light_type_table", lights=[asdict(light) for light in lights])
            return
        merged_payload = self._merged_light_type_table_payload(lights)
        self._validate_light_type_payload(merged_payload)
        self.rulebook.lights_by_name.clear()
        self.rulebook.lights_by_id.clear()
        self.rulebook.update_lights(self._coerce_light_type_def(item) for item in merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(
            build_default_optics_entries(
                self.rulebook.materials_by_id.values(),
                self.rulebook.lights_by_id.values(),
                existing=self._material_optics_snapshot_map(),
            )
        )
        self._rebuild_light_property_arrays()
        previous_cell_dose = self.cell_optical_dose
        previous_gas_dose = self.gas_optical_dose
        light_count = self._light_field_count()
        self.cell_optical_dose = np.zeros((light_count, self.height, self.width), dtype=np.float32)
        self.gas_optical_dose = np.zeros((light_count, self.gas_height, self.gas_width), dtype=np.float32)
        cell_count = min(previous_cell_dose.shape[0], self.cell_optical_dose.shape[0])
        gas_count = min(previous_gas_dose.shape[0], self.gas_optical_dose.shape[0])
        self.cell_optical_dose[:cell_count] = previous_cell_dose[:cell_count]
        self.gas_optical_dose[:gas_count] = previous_gas_dose[:gas_count]
        optics_payload = self._material_optics_table_snapshot_payload()
        self._set_stable_shadow_payload("lights", merged_payload)
        self._set_stable_shadow_payload("optics", optics_payload)
        self.bridge.upload_table("lights", merged_payload)
        self.bridge.upload_table("optics", optics_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_light_type_table")
        self.bridge.ensure_world_resources(self)

    def update_material_optics_table(self, optics: list[MaterialOpticsDef | dict[str, Any]], *, immediate: bool = True) -> None:
        optics = [self._coerce_material_optics_def(entry) for entry in optics]
        if not immediate:
            self.queue_command("update_material_optics_table", optics=[asdict(entry) for entry in optics])
            return
        merged_payload = self._merged_material_optics_table_payload(optics)
        self._validate_material_optics_payload(merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(self._coerce_material_optics_def(item) for item in merged_payload)
        self._set_stable_shadow_payload("optics", merged_payload)
        self.bridge.upload_table("optics", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_material_optics_table")

    def update_reaction_table(
        self,
        actions: list[ReactionAction | dict[str, Any]],
        rules: dict[str, object],
        *,
        immediate: bool = True,
    ) -> None:
        actions = [self._coerce_reaction_action(action) for action in actions]
        rules = self._coerce_reaction_rules(rules)
        if not immediate:
            self.queue_command(
                "update_reaction_table",
                actions=[asdict(action) for action in actions],
                rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
            )
            return
        merged_payload = self._merged_reaction_table_payload(actions, rules)
        self._validate_reaction_payload(merged_payload)
        self.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + [
            self._coerce_reaction_action(action) for action in merged_payload["actions"]
        ]
        merged_rules = merged_payload["rules"]
        self.rulebook.material_material_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_material"]
        ]
        self.rulebook.material_gas_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_gas"]
        ]
        self.rulebook.material_light_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_light"]
        ]
        self.rulebook.gas_gas_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_gas"]
        ]
        self.rulebook.gas_light_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_light"]
        ]
        self.rulebook.self_rules = [
            self._coerce_self_reaction_rule(rule) for rule in merged_rules["self_rules"]
        ]
        self._set_stable_shadow_payload("reactions", merged_payload)
        self.bridge.upload_table("reactions", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_reaction_table")

    def replace_reaction_table(
        self,
        actions: list[ReactionAction | dict[str, Any]],
        rules: dict[str, object],
        *,
        immediate: bool = True,
    ) -> None:
        actions = [self._coerce_reaction_action(action) for action in actions]
        rules = self._coerce_reaction_rules(rules)
        if not immediate:
            self.queue_command(
                "replace_reaction_table",
                actions=[asdict(action) for action in actions],
                rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
            )
            return
        replacement_payload = {
            "actions": [asdict(action) for action in actions],
            "rules": {name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
        }
        self._validate_reaction_payload(replacement_payload)
        materials_payload = self._shadow_material_payload()
        self._clamp_material_payload_reaction_slots(
            materials_payload,
            action_count=len(replacement_payload["actions"]) + 1,
        )
        self.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + list(actions)
        self.rulebook.material_material_rules = list(rules["material_material"])
        self.rulebook.material_gas_rules = list(rules["material_gas"])
        self.rulebook.material_light_rules = list(rules["material_light"])
        self.rulebook.gas_gas_rules = list(rules["gas_gas"])
        self.rulebook.gas_light_rules = list(rules["gas_light"])
        self.rulebook.self_rules = list(rules["self_rules"])
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in materials_payload)
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        reaction_payload = self._reaction_table_snapshot_payload()
        self._set_stable_shadow_payload("materials", materials_payload)
        self._set_stable_shadow_payload("reactions", reaction_payload)
        self.bridge.upload_table("materials", materials_payload)
        self.bridge.upload_table("reactions", reaction_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("replace_reaction_table")

    def reset_world(self, *, immediate: bool = True) -> None:
        if not immediate:
            self.queue_command("reset_world")
            return
        self._reset_world_state(reset_bridge_frame_inputs=True, keep_command_log=False)

    def _reset_world_state(
        self,
        *,
        reset_bridge_frame_inputs: bool,
        keep_command_log: bool,
    ) -> None:
        self.material_id.fill(0)
        self.phase.fill(0)
        self.cell_flags.fill(0)
        self.velocity.fill(0.0)
        self.cell_temperature.fill(20.0)
        self.timer_pack.fill(0)
        self.integrity.fill(0.0)
        self.island_id.fill(0)
        self.entity_id.fill(0)
        self.placeholder_displaced_material.fill(0)
        self.collapse_delay_pending.fill(False)
        self.flow_velocity.fill(0.0)
        self.ambient_temperature.fill(20.0)
        self.pressure_ping.fill(0.0)
        self.gas_concentration.fill(0.0)
        if 0 <= self.air_gas_species_id < self.gas_concentration.shape[0]:
            self.gas_concentration[self.air_gas_species_id] = 1.0
        self.visible_illumination.fill(0.0)
        self.cell_optical_dose.fill(0.0)
        self.gas_optical_dose.fill(0.0)
        self.force_sources.clear()
        self.persistent_emitters.clear()
        self.emitters.clear()
        self.collapse_dirty_regions.clear()
        self.collapse_deferred_regions.clear()
        clear_collapse_structure_dirty_tile_mask(self)
        self.islands.clear()
        self.entity_states.clear()
        self.entity_placeholders.clear()
        self.pending_frame_inputs.clear()
        self.completed_frame_outputs.clear()
        self.canceled_frame_submission_ids.clear()
        self.next_frame_submission_id = 1
        self.next_readback_request_id = 1
        self.pending_readbacks.clear()
        self.inflight_readbacks.clear()
        self.completed_readbacks.clear()
        self.canceled_readback_request_ids.clear()
        self.last_entity_observation_consume_snapshot = {
            "frame_id": int(self.frame_id),
            "consumed": 0,
            "consumed_readbacks": [],
            "observations": {},
            "entity_feedback": {},
        }
        self.controller_state_snapshot = None
        self.gas_solver.reset_runtime_state(self)
        self.heat_solver.reset_runtime_state(self)
        self.liquid_solver.reset_runtime_state(self)
        self.reaction_solver.reset_runtime_state(self)
        self.collapse_solver.reset_runtime_state(self)
        self.optics_solver.reset_runtime_state(self)
        self.motion_solver.reset_runtime_state()
        if reset_bridge_frame_inputs:
            self._clear_bridge_frame_inputs(keep_commands=keep_command_log, prepared=False)
        self.page_store.clear()
        self.next_island_id = 1
        self._build_demo_scene()

    def queue_command(self, kind: str, **payload: Any) -> None:
        self.command_queue.append(WorldCommand(kind=kind, payload=deepcopy(payload)))

    def _resolve_direct_targeted_coords(
        self,
        kind: str,
        x: int | None,
        y: int | None,
        *,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> tuple[int, int, str | None]:
        fields = TARGETED_COMMAND_COORD_FIELDS.get(kind)
        if fields is None:
            raise ValueError(f"unsupported direct target query kind '{kind}'")
        x_field, y_field = fields
        if target_queries is None:
            if x is None or y is None:
                raise ValueError(f"{x_field} and {y_field} are required unless target_queries resolve target_query_id")
            return int(x), int(y), None
        if target_query_id is None:
            raise ValueError("target_query_id is required when target_queries are provided")
        resolved_targets = self._resolve_target_queries(
            [self._coerce_target_query(query) for query in target_queries]
        )
        resolved_commands = self._resolve_targeted_commands(
            [
                WorldCommand(
                    kind=kind,
                    payload={
                        "target_query_id": str(target_query_id),
                        "target_dx": int(target_dx),
                        "target_dy": int(target_dy),
                    },
                )
            ],
            resolved_targets,
        )
        if not resolved_commands:
            raise ValueError(f"unable to resolve {kind} target query")
        payload = resolved_commands[0].payload
        return int(payload[x_field]), int(payload[y_field]), str(payload.get("resolved_target_query_id", target_query_id))

    def inject_material(
        self,
        x: int | None,
        y: int | None,
        material: str,
        radius: int = 2,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_material",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [WorldCommand(kind="inject_material", payload={"x": int(x), "y": int(y), "material": material, "radius": radius})]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "material": material, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_material", **payload)

    def write_material_region(
        self,
        x: int | None,
        y: int | None,
        width: int,
        height: int,
        material: str,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "write_material_region",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="write_material_region",
                        payload={"x": int(x), "y": int(y), "width": width, "height": height, "material": material},
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "material": material,
            }
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("write_material_region", **payload)

    def inject_temperature(
        self,
        x: int | None,
        y: int | None,
        delta: float,
        radius: int = 2,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_temperature",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [WorldCommand(kind="inject_temperature", payload={"x": int(x), "y": int(y), "delta": delta, "radius": radius})]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "delta": delta, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_temperature", **payload)

    def inject_velocity(
        self,
        x: int | None,
        y: int | None,
        velocity: tuple[float, float],
        radius: int = 2,
        *,
        carrier: str = "cell",
        mode: str = "add",
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_velocity",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="inject_velocity",
                        payload={
                            "x": int(x),
                            "y": int(y),
                            "velocity": velocity,
                            "radius": radius,
                            "carrier": carrier,
                            "mode": mode,
                        },
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {
                "x": x,
                "y": y,
                "velocity": velocity,
                "radius": radius,
                "carrier": carrier,
                "mode": mode,
            }
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_velocity", **payload)

    def inject_force(
        self,
        x: int | None,
        y: int | None,
        direction: tuple[float, float],
        radius: float,
        strength: float,
        lifetime: float = 0.5,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_force",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            world_x = float(x)
            world_y = float(y)
            x, y = self._world_to_buffer_clamped(int(x), int(y))
            self._append_force_source_immediate(
                ForceSource(
                    x=float(x),
                    y=float(y),
                    direction=(float(direction[0]), float(direction[1])),
                    radius=float(radius),
                    strength=float(strength),
                    lifetime=float(lifetime),
                    world_x=world_x,
                    world_y=world_y,
                )
            )
            return
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "direction": direction,
            "radius": radius,
            "strength": strength,
            "lifetime": lifetime,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("inject_force", **payload)

    def inject_gas(
        self,
        x: int | None,
        y: int | None,
        species: str,
        amount: float,
        radius: int = 1,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_gas",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="inject_gas",
                        payload={"x": int(x), "y": int(y), "species": species, "amount": amount, "radius": radius},
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "species": species, "amount": amount, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_gas", **payload)

    def request_readback(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> int:
        request = ReadbackRequest(
            request_id=request_id,
            center_x=None if center_x is None else int(center_x),
            center_y=None if center_y is None else int(center_y),
            width=int(width),
            height=int(height),
            channels=tuple(channels),
            observer_id=observer_id,
            label=label,
            target_query_id=None if target_query_id is None else str(target_query_id),
            target_dx=int(target_dx),
            target_dy=int(target_dy),
        )
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
            resolved_request = self._resolve_readback_request(request, resolved_targets)
            if resolved_request is None:
                raise ValueError("unable to resolve readback request target query")
            request = resolved_request
        request = self._normalize_readback_request(request)
        if request.center_x is None or request.center_y is None:
            raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
        request = self._assign_readback_request_id(request)
        self.queue_command(
            "request_readback",
            request_id=request.request_id,
            center_x=request.center_x,
            center_y=request.center_y,
            width=request.width,
            height=request.height,
            channels=request.channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )
        assert request.request_id is not None
        return int(request.request_id)

    def preview_readback(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ReadbackRequest:
        request = ReadbackRequest(
            request_id=request_id,
            center_x=None if center_x is None else int(center_x),
            center_y=None if center_y is None else int(center_y),
            width=int(width),
            height=int(height),
            channels=tuple(channels),
            observer_id=observer_id,
            label=label,
            target_query_id=None if target_query_id is None else str(target_query_id),
            target_dx=int(target_dx),
            target_dy=int(target_dy),
        )
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
            resolved_request = self._resolve_readback_request(request, resolved_targets)
            if resolved_request is None:
                raise ValueError("unable to resolve readback request target query")
            request = resolved_request
        request = self._normalize_readback_request(request)
        if request.center_x is None or request.center_y is None:
            raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
        return request

    def request_observation(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> int:
        target = self._coerce_observation_target(target)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        request = self._build_observation_request(target, resolved_targets)
        if request is None:
            if target.target_query_id is not None and target_queries is not None:
                raise ValueError("unable to resolve observation target query")
            raise ValueError("unable to resolve observation target")
        if request_id is not None:
            request = replace(request, request_id=int(request_id))
        request = self._assign_readback_request_id(request)
        self.queue_command(
            "request_readback",
            request_id=request.request_id,
            center_x=request.center_x,
            center_y=request.center_y,
            width=request.width,
            height=request.height,
            channels=request.channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )
        assert request.request_id is not None
        return int(request.request_id)

    def preview_observation(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ReadbackRequest:
        target = self._coerce_observation_target(target)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        request = self._build_observation_request(target, resolved_targets)
        if request is None:
            if target.target_query_id is not None and target_queries is not None:
                raise ValueError("unable to resolve observation target query")
            raise ValueError("unable to resolve observation target")
        if request_id is not None:
            request = replace(request, request_id=int(request_id))
        return request

    def _resolve_public_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None,
        assign_readback_request_id: bool,
    ) -> WorldCommand:
        command = self._coerce_world_command(command)
        if command.kind not in PUBLIC_WORLD_COMMAND_KINDS:
            raise ValueError(f"unsupported public world command kind '{command.kind}'")

        if command.kind == "request_readback":
            request = self._coerce_readback_request(command.payload)
            if target_queries is not None:
                resolved_targets = self._resolve_target_queries(
                    [self._coerce_target_query(query) for query in target_queries]
                )
                resolved_request = self._resolve_readback_request(request, resolved_targets)
                if resolved_request is None:
                    raise ValueError("unable to resolve world command target query")
                request = resolved_request
            elif request.target_query_id is not None and (request.center_x is None or request.center_y is None):
                raise ValueError("target_queries are required to resolve world command target_query_id")
            request = self._normalize_readback_request(request)
            if request.center_x is None or request.center_y is None:
                raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
            if assign_readback_request_id:
                request = self._assign_readback_request_id(request)
            return WorldCommand(
                kind="request_readback",
                payload={
                    "request_id": request.request_id,
                    "center_x": request.center_x,
                    "center_y": request.center_y,
                    "width": request.width,
                    "height": request.height,
                    "channels": request.channels,
                    "observer_id": request.observer_id,
                    "label": request.label,
                    "target_query_id": request.target_query_id,
                    "target_dx": int(request.target_dx),
                    "target_dy": int(request.target_dy),
                },
            )

        if target_queries is None:
            if command.payload.get("target_query_id") is not None:
                raise ValueError("target_queries are required to resolve world command target_query_id")
            return command
        resolved_targets = self._resolve_target_queries(
            [self._coerce_target_query(query) for query in target_queries]
        )
        resolved_commands = self._resolve_targeted_commands([command], resolved_targets)
        if not resolved_commands:
            raise ValueError("unable to resolve world command target query")
        return resolved_commands[0]

    def _public_world_command(self, command: WorldCommand) -> WorldCommand:
        payload = deepcopy(command.payload)
        if command.kind == "sync_entity_states" and isinstance(payload, dict):
            entities = payload.get("entities")
            if isinstance(entities, list):
                payload["entities"] = [
                    self.serialize_entity_state_input(
                        entity if isinstance(entity, EntityState) else self._coerce_entity_state(entity)
                    )
                    for entity in entities
                ]
        elif command.kind == "patch_entity_states" and isinstance(payload, dict):
            patches = payload.get("patches")
            if isinstance(patches, list):
                payload["patches"] = [
                    self.serialize_entity_state_patch(
                        patch if isinstance(patch, EntityStatePatch) else self._coerce_entity_state_patch(patch)
                    )
                    for patch in patches
                ]
        elif command.kind == "sync_entity_placeholders" and isinstance(payload, dict):
            placeholders = payload.get("placeholders")
            if isinstance(placeholders, list):
                payload["placeholders"] = [
                    self.serialize_entity_placeholder_input(
                        placeholder
                        if isinstance(placeholder, EntityPlaceholder)
                        else self._coerce_entity_placeholder(placeholder)
                    )
                    for placeholder in placeholders
                ]
        return WorldCommand(kind=command.kind, payload=payload)

    def preview_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> WorldCommand:
        command = self._coerce_world_command(command)
        resolved_command = self._resolve_public_world_command(
            command,
            target_queries=target_queries,
            assign_readback_request_id=False,
        )
        return self._public_world_command(resolved_command)

    def preview_target_queries(
        self,
        target_queries: list[TargetQuery | dict[str, Any]],
    ) -> dict[str, ResolvedTarget]:
        return {
            query_id: self._public_resolved_target(target)
            for query_id, target in self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            ).items()
        }

    def request_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> WorldCommand:
        resolved_command = self._resolve_public_world_command(
            command,
            target_queries=target_queries,
            assign_readback_request_id=True,
        )
        self.queue_command(resolved_command.kind, **resolved_command.payload)
        return self._public_world_command(resolved_command)

    def preview_change_intent(
        self,
        intent: ChangeIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedChangeIntent:
        intent = self._coerce_change_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        return self._public_resolved_change_intent(self._resolve_change_intent(intent, resolved_targets))

    def request_change_intent(
        self,
        intent: ChangeIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedChangeIntent:
        intent = self._coerce_change_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        resolved_intent = self._resolve_change_intent(intent, resolved_targets)
        for command in resolved_intent.generated_commands:
            self.queue_command(command.kind, **command.payload)
        return self._public_resolved_change_intent(resolved_intent)

    def preview_carrier_intent(
        self,
        intent: CarrierIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedCarrierIntent:
        intent = self._coerce_carrier_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        return self._public_resolved_carrier_intent(self._resolve_carrier_intent(intent, resolved_targets))

    def request_carrier_intent(
        self,
        intent: CarrierIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedCarrierIntent:
        intent = self._coerce_carrier_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        resolved_intent = self._resolve_carrier_intent(intent, resolved_targets)
        for command in resolved_intent.generated_commands:
            self.queue_command(command.kind, **command.payload)
        return self._public_resolved_carrier_intent(resolved_intent)

    def preview_frame_input(
        self,
        frame_input: WorldFrameInput | dict[str, Any],
        *,
        reserved_readback_request_ids: set[int] | None = None,
    ) -> WorldFramePreview:
        frame_input = self._coerce_world_frame_input(frame_input)
        saved_paging = deepcopy(self.paging)
        saved_preview_runtime = self._snapshot_preview_runtime_state()
        saved_entity_states = dict(self.entity_states)
        saved_entity_placeholders = {entity_id: set(cells) for entity_id, cells in self.entity_placeholders.items()}
        saved_controller_state = deepcopy(self.controller_state_snapshot)
        saved_blocked_cells = None if self._resolver_blocked_cells is None else set(self._resolver_blocked_cells)
        saved_released_cells = None if self._resolver_released_cells is None else set(self._resolver_released_cells)
        try:
            preview_controller_state = deepcopy(self.controller_state_snapshot)
            if frame_input.controller_state_provided:
                self.controller_state_snapshot = deepcopy(frame_input.controller_state)
                preview_controller_state = deepcopy(self.controller_state_snapshot)
            (
                paging_updates,
                preview_page_stripes,
                entity_observation_targets,
                placeholder_inputs,
                placeholder_count,
            ) = self._prepare_preview_frame_context(frame_input)
            resolved_targets = self._resolve_target_queries(frame_input.target_queries)
            resolved_change_intents, generated_commands = self._resolve_change_intents(frame_input.change_intents, resolved_targets)
            resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
                frame_input.carrier_intents,
                resolved_targets,
            )
            observation_pairs = self._build_observation_request_pairs(
                entity_observation_targets + frame_input.observation_targets,
                resolved_targets,
            )
            observation_requests, next_preview_request_id = self._assign_preview_readback_request_ids(
                [request for _, request in observation_pairs]
            )
            observation_pairs = [
                (target, request)
                for (target, _), request in zip(observation_pairs, observation_requests, strict=False)
            ]
            resolved_commands = (
                generated_commands
                + generated_carrier_commands
                + self._resolve_targeted_commands(frame_input.commands, resolved_targets)
            )
            readback_requests, _ = self._assign_preview_readback_request_ids(
                self._resolve_readback_requests(frame_input.readback_requests, resolved_targets),
                next_request_id=next_preview_request_id,
            )
            bridge_frame_snapshot = self._serialize_preview_bridge_frame_snapshot(
                current_entity_placeholders=saved_entity_placeholders,
                resolved_commands=resolved_commands,
                observation_requests=observation_requests,
                readback_requests=readback_requests,
                placeholder_inputs=placeholder_inputs,
                paging_updates=paging_updates,
                page_stripes=preview_page_stripes,
                reserved_readback_request_ids=reserved_readback_request_ids,
            )
            return WorldFramePreview(
                controller_state=preview_controller_state,
                resolved_targets={
                    query_id: self._public_resolved_target(target)
                    for query_id, target in resolved_targets.items()
                },
                resolved_change_intents={
                    intent_id: self._public_resolved_change_intent(intent)
                    for intent_id, intent in resolved_change_intents.items()
                },
                resolved_carrier_intents={
                    intent_id: self._public_resolved_carrier_intent(intent)
                    for intent_id, intent in resolved_carrier_intents.items()
                },
                resolved_commands=[self._public_world_command(command) for command in resolved_commands],
                observation_requests=observation_requests,
                observation_plans=[
                    self._serialize_observation_plan_for_target_request(target, request)
                    for target, request in observation_pairs
                ],
                readback_requests=readback_requests,
                readback_plans=self._serialize_readback_plans_for_requests(readback_requests),
                bridge_frame_snapshot=bridge_frame_snapshot,
                paging_updates=paging_updates,
                placeholder_count=placeholder_count,
            )
        finally:
            self._restore_preview_runtime_state(saved_preview_runtime)
            self.paging = saved_paging
            self.entity_states = saved_entity_states
            self.entity_placeholders = saved_entity_placeholders
            self.controller_state_snapshot = saved_controller_state
            self._resolver_blocked_cells = saved_blocked_cells
            self._resolver_released_cells = saved_released_cells

    def submit_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> int:
        frame_input = self._coerce_world_frame_input(frame_input)
        submission_id = frame_input.submission_id
        if submission_id is None:
            submission_id = self.next_frame_submission_id
        frame_input = replace(
            frame_input,
            submission_id=submission_id,
            readback_requests=[self._assign_readback_request_id(request) for request in frame_input.readback_requests],
        )
        self.next_frame_submission_id = max(self.next_frame_submission_id, int(submission_id) + 1)
        self.canceled_frame_submission_ids.discard(int(submission_id))
        self.pending_frame_inputs.append(frame_input)
        return int(submission_id)

    def request_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> dict[str, Any]:
        submission_id = self.submit_frame_input(frame_input)
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_frame_input(
            pending_frame_input,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
        }

    def request_frame_cycle(
        self,
        frame_input: WorldFrameInput | dict[str, Any] | None = None,
        *,
        apply_frame: bool = True,
    ) -> dict[str, Any]:
        normalized_frame_input = {} if frame_input is None else frame_input
        preview = self.preview_frame_input(normalized_frame_input)
        if not apply_frame:
            return {
                "applied": False,
                "queued": False,
                "pending_frames": len(self.pending_frame_inputs),
                "submission_id": None,
                "preview": preview,
                "result": None,
            }
        submission_id = self.submit_frame_input(normalized_frame_input)
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_frame_input(
            pending_frame_input,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "applied": True,
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
            "result": None,
        }

    def pending_frame_submission_ids(self) -> list[int]:
        return [int(frame_input.submission_id) for frame_input in self.pending_frame_inputs if frame_input.submission_id is not None]

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

    def cancel_frame_submission(self, submission_id: int) -> bool:
        for index, frame_input in enumerate(self.pending_frame_inputs):
            if frame_input.submission_id == submission_id:
                del self.pending_frame_inputs[index]
                self.canceled_frame_submission_ids.add(int(submission_id))
                self.canceled_readback_request_ids.update(self._frame_readback_request_ids(frame_input))
                return True
        return False

    def cancel_all_pending_frame_submissions(self) -> list[int]:
        canceled = self.pending_frame_submission_ids()
        canceled_readback_ids: list[int] = []
        for frame_input in self.pending_frame_inputs:
            canceled_readback_ids.extend(self._frame_readback_request_ids(frame_input))
        self.pending_frame_inputs.clear()
        self.canceled_frame_submission_ids.update(canceled)
        self.canceled_readback_request_ids.update(canceled_readback_ids)
        return canceled

    def cancel_readback_request(self, request_id: int) -> bool:
        request_id = int(request_id)
        canceled = False

        remaining_commands: deque[WorldCommand] = deque()
        for command in self.command_queue:
            if command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == request_id:
                canceled = True
                continue
            remaining_commands.append(command)
        self.command_queue = remaining_commands

        remaining_frames: deque[WorldFrameInput] = deque()
        for frame_input in self.pending_frame_inputs:
            remaining_readbacks = [request for request in frame_input.readback_requests if request.request_id != request_id]
            if len(remaining_readbacks) != len(frame_input.readback_requests):
                frame_input = replace(frame_input, readback_requests=remaining_readbacks)
                canceled = True
            remaining_frames.append(frame_input)
        self.pending_frame_inputs = remaining_frames

        next_pending = [request for request in self.pending_readbacks if request.request_id != request_id]
        if len(next_pending) != len(self.pending_readbacks):
            canceled = True
        self.pending_readbacks = next_pending

        next_inflight = [request for request in self.inflight_readbacks if request.request_id != request_id]
        if len(next_inflight) != len(self.inflight_readbacks):
            canceled = True
        self.inflight_readbacks = next_inflight

        next_completed = deque(
            result for result in self.completed_readbacks if result.request.request_id != request_id
        )
        if len(next_completed) != len(self.completed_readbacks):
            canceled = True
        self.completed_readbacks = next_completed

        if canceled:
            self.canceled_readback_request_ids.add(request_id)
        return canceled

    def poll_frame_output(self, submission_id: int | None = None) -> WorldFrameOutput | None:
        if submission_id is None:
            if not self.completed_frame_outputs:
                return None
            return self.completed_frame_outputs.popleft()
        for index, output in enumerate(self.completed_frame_outputs):
            if output.submission_id == submission_id:
                del self.completed_frame_outputs[index]
                return output
        return None

    def poll_all_frame_outputs(self) -> list[WorldFrameOutput]:
        outputs: list[WorldFrameOutput] = []
        while self.completed_frame_outputs:
            outputs.append(self.completed_frame_outputs.popleft())
        return outputs

    def serialize_pending_commands(self) -> dict[str, Any]:
        return serialize_pending_commands(self)

    def serialize_readback_state(self) -> dict[str, Any]:
        return serialize_readback_state(self)

    def serialize_bridge_runtime(self) -> dict[str, Any]:
        return serialize_bridge_runtime(self)

    @staticmethod
    def _serialize_bridge_resource_summary(name: str, array: np.ndarray) -> dict[str, Any]:
        return _serialize_bridge_resource_summary(name, array)

    def serialize_bridge_resources(self) -> dict[str, Any]:
        return serialize_bridge_resources(self)

    def serialize_ready_readbacks(self) -> dict[str, Any]:
        return serialize_ready_readbacks(self)

    def readback_request_status(self, request_id: int) -> str:
        if any(
            command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == int(request_id)
            for command in self.command_queue
        ):
            return "queued"
        if any(
            any(request.request_id == int(request_id) for request in frame_input.readback_requests)
            for frame_input in self.pending_frame_inputs
        ):
            return "pending_frame"
        if any(request.request_id == int(request_id) for request in self.pending_readbacks):
            return "pending"
        if any(request.request_id == int(request_id) for request in self.inflight_readbacks):
            return "inflight"
        if any(result.request.request_id == int(request_id) for result in self.completed_readbacks):
            return "ready"
        if int(request_id) in self.canceled_readback_request_ids:
            return "canceled"
        return "missing"

    def serialize_frame_state(self) -> dict[str, Any]:
        return serialize_frame_state(self)

    def serialize_pending_frame_inputs(self) -> dict[str, Any]:
        return serialize_pending_frame_inputs(self)

    def serialize_pending_frame_detail(self, frame_input: WorldFrameInput) -> dict[str, Any]:
        return serialize_pending_frame_detail(self, frame_input)

    def serialize_ready_frame_outputs(self) -> dict[str, Any]:
        return serialize_ready_frame_outputs(self)

    def frame_submission_status(self, submission_id: int) -> str:
        if any(frame_input.submission_id == submission_id for frame_input in self.pending_frame_inputs):
            return "pending"
        if any(output.submission_id == submission_id for output in self.completed_frame_outputs):
            return "ready"
        if submission_id in self.canceled_frame_submission_ids:
            return "canceled"
        return "missing"

    def inject_light(
        self,
        x: int | None,
        y: int | None,
        light_type: str,
        strength: float,
        radius: int | None = None,
        *,
        direction: tuple[float, float] = (0.0, 0.0),
        spread: float = 0.25,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_light",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        light_id = self._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            raise KeyError(light_type)
        if radius is not None:
            resolved_radius = int(radius)
        else:
            shadow_default_range = self._shadow_light_default_range(light_id)
            if shadow_default_range is None:
                raise KeyError(light_type)
            resolved_radius = int(shadow_default_range)
        if immediate:
            world_origin = (int(x), int(y))
            x, y = self._world_to_buffer_clamped(int(x), int(y))
            shadow_light = self._shadow_light_name(light_id)
            if shadow_light is None:
                raise KeyError(light_type)
            self._append_transient_light_emitter_immediate(
                {
                    "light_type": shadow_light,
                    "origin": (int(x), int(y)),
                    "world_origin": world_origin,
                    "direction": (float(direction[0]), float(direction[1])),
                    "spread": float(spread),
                    "strength": float(strength),
                    "range_cells": int(resolved_radius),
                }
            )
            return
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "light_type": self._shadow_light_name(light_id),
            "strength": strength,
            "radius": resolved_radius,
            "direction": direction,
            "spread": spread,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("inject_light", **payload)

    def focus_paging(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        return self.paging.focus_on(center_x, center_y)

    def advance_paging(
        self,
        center_x: int | None,
        center_y: int | None,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> list[PageStripeUpdate]:
        center_x, center_y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "advance_paging",
            center_x,
            center_y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            return self._advance_paging(center_x, center_y)
        payload: dict[str, Any] = {"center_x": center_x, "center_y": center_y}
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("advance_paging", **payload)
        return []

    def capture_page_stripe(self, update: PageStripeUpdate) -> dict[str, Any]:
        update = self._contextualize_page_stripe_update(update)
        if self.simulation_backend == "gpu":
            if self._gpu_cpu_dirty_resources:
                self.bridge.sync_world(self)
                self._gpu_cpu_dirty_resources.clear()
            return self.page_stripe_pipeline.capture(self, update)
        return self._capture_page_stripe_cpu_snapshot(update)

    def _capture_page_stripe_cpu_snapshot(self, update: PageStripeUpdate) -> dict[str, Any]:
        gas_ranges = self._stripe_buffer_ranges(update, gas_grid=True)
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1
        material_id = self._capture_stripe_array(self.material_id, update, stripe_axis=cell_axis)
        phase = self._capture_stripe_array(self.phase, update, stripe_axis=cell_axis)
        island_id = self._capture_stripe_array(self.island_id, update, stripe_axis=cell_axis)
        entity_id = self._capture_stripe_array(self.entity_id, update, stripe_axis=cell_axis)
        placeholder_displaced_material = self._capture_stripe_array(
            self.placeholder_displaced_material,
            update,
            stripe_axis=cell_axis,
        )
        phase, island_id, entity_id, placeholder_displaced_material = self._normalize_cell_runtime_arrays(
            material_id,
            phase,
            island_id,
            entity_id,
            placeholder_displaced_material,
        )
        runtime_payload = self._capture_page_stripe_island_runtime(
            island_id
        )
        runtime_payload["entity_placeholder_entity_id"] = self._capture_page_stripe_entity_placeholder_runtime(
            update,
            stripe_axis=cell_axis,
        )
        payload = {
            "meta": {
                "axis": update.axis,
                "world_start": update.world_start,
                "world_end": update.world_end,
                "buffer_start": update.buffer_start,
                "buffer_end": update.buffer_end,
                "kind": update.kind,
                "cross_world_start": update.cross_world_start,
                "cross_world_end": update.cross_world_end,
            },
            "cell": {
                "material_id": material_id,
                "phase": phase,
                "cell_flags": self._capture_stripe_array(self.cell_flags, update, stripe_axis=cell_axis),
                "velocity": self._capture_stripe_array(self.velocity, update, stripe_axis=cell_axis),
                "cell_temperature": self._capture_stripe_array(self.cell_temperature, update, stripe_axis=cell_axis),
                "timer_pack": self._capture_stripe_array(self.timer_pack, update, stripe_axis=cell_axis),
                "integrity": self._capture_stripe_array(self.integrity, update, stripe_axis=cell_axis),
                "island_id": island_id,
                "entity_id": entity_id,
                "placeholder_displaced_material": placeholder_displaced_material,
                "collapse_delay_pending": self._capture_stripe_array(
                    self.collapse_delay_pending.astype(np.uint8),
                    update,
                    stripe_axis=cell_axis,
                ),
                "visible_illumination": self._capture_stripe_array(
                    self.visible_illumination,
                    update,
                    stripe_axis=cell_axis,
                ),
                "cell_optical_dose": self._capture_stripe_array(
                    self.cell_optical_dose,
                    update,
                    stripe_axis=cell_dose_axis,
                ),
            },
            "runtime": runtime_payload,
            "gas": {
                "ambient_temperature": self._capture_stripe_array(
                    self.ambient_temperature,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "flow_velocity": self._capture_stripe_array(
                    self.flow_velocity,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "pressure_ping": self._capture_stripe_array(
                    self.pressure_ping,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "gas_concentration": self._capture_stripe_array(
                    self.gas_concentration,
                    update,
                    stripe_axis=2 if update.axis == "x" else 1,
                    ranges=gas_ranges,
                ),
                "gas_optical_dose": self._capture_stripe_array(
                    self.gas_optical_dose,
                    update,
                    stripe_axis=2 if update.axis == "x" else 1,
                    ranges=gas_ranges,
                ),
            },
        }
        return payload

    def apply_page_stripe(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
        *,
        immediate: bool = False,
    ) -> None:
        update = self._contextualize_page_stripe_update(update)
        payload = self._coerce_page_stripe_payload(payload)
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
            self._apply_page_stripe(update, payload)
            self._record_bridge_page_stripe(update, payload)
            return
        self.queue_command(
            "apply_page_stripe",
            update=asdict(update),
            payload=payload,
        )

    def store_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> dict[str, Any]:
        update = self._contextualize_page_stripe_update(update)
        normalized_payload = self._coerce_page_stripe_payload(payload)
        self.page_store.save(update, normalized_payload)
        stored_payload = self.page_store.load(update)
        assert stored_payload is not None
        return self._coerce_page_stripe_payload(stored_payload)

    def capture_page_stripe_to_store(self, update: PageStripeUpdate) -> dict[str, Any]:
        return self.store_page_stripe(update, self.capture_page_stripe(update))

    def load_page_stripe(self, update: PageStripeUpdate) -> dict[str, Any] | None:
        update = self._contextualize_page_stripe_update(update)
        payload = self.page_store.load(update)
        if payload is None:
            return None
        return self._coerce_page_stripe_payload(payload)

    def apply_stored_page_stripe(
        self,
        update: PageStripeUpdate,
        *,
        immediate: bool = False,
    ) -> dict[str, Any] | None:
        payload = self.load_page_stripe(update)
        if payload is None:
            return None
        self.apply_page_stripe(update, payload, immediate=immediate)
        return payload

    def page_store_has_stripe(self, update: PageStripeUpdate) -> bool:
        update = self._contextualize_page_stripe_update(update)
        return bool(self.page_store.has(update))

    def list_page_store_stripe_keys(self) -> list[StoredStripeKey] | None:
        list_keys = getattr(self.page_store, "keys", None)
        if not callable(list_keys):
            return None
        return [self._coerce_page_store_key(key) for key in list_keys()]

    def export_page_store_entries(self) -> dict[str, Any]:
        keys = self.list_page_store_stripe_keys()
        entries: list[dict[str, Any]] = []
        if keys is not None:
            for key in keys:
                payload = self.page_store.load(self._page_store_key_lookup_update(key))
                if payload is None:
                    continue
                entries.append(
                    {
                        "key": self.serialize_page_store_key(key),
                        "payload": self.serialize_page_stripe_payload(self._coerce_page_stripe_payload(payload)),
                    }
                )
        return {
            "stored_stripes": int(self.page_store.stored_count()),
            "key_listing_supported": keys is not None,
            "entries": entries,
        }

    def import_page_store_entries(self, entries: Iterable[dict[str, Any]], *, clear: bool = False) -> dict[str, int]:
        cleared = 0
        if clear:
            cleared = self.clear_page_store()
        imported = 0
        for entry in entries:
            key = self._coerce_page_store_key(entry["key"])
            payload = self._coerce_page_stripe_payload(dict(entry["payload"]))
            self.page_store.save(self._page_store_key_lookup_update(key), payload)
            imported += 1
        return {
            "cleared": int(cleared),
            "imported": int(imported),
            "stored_stripes": int(self.page_store.stored_count()),
        }

    def clear_page_store(self) -> int:
        cleared = int(self.page_store.stored_count())
        self.page_store.clear()
        return cleared

    def serialize_page_store_state(self) -> dict[str, Any]:
        return serialize_page_store_state(self)

    def _coerce_page_store_key(
        self,
        key: StoredStripeKey | PageStripeUpdate | dict[str, Any],
    ) -> StoredStripeKey:
        if isinstance(key, StoredStripeKey):
            return StoredStripeKey(
                axis=str(key.axis),
                world_start=int(key.world_start),
                world_end=int(key.world_end),
                cross_world_start=int(getattr(key, "cross_world_start", 0)),
                cross_world_end=int(getattr(key, "cross_world_end", 0)),
            )
        if isinstance(key, PageStripeUpdate):
            return StoredStripeKey(
                axis=str(key.axis),
                world_start=int(key.world_start),
                world_end=int(key.world_end),
                cross_world_start=0 if key.cross_world_start is None else int(key.cross_world_start),
                cross_world_end=0 if key.cross_world_end is None else int(key.cross_world_end),
            )
        payload = dict(key)
        return StoredStripeKey(
            axis=str(payload["axis"]),
            world_start=int(payload["world_start"]),
            world_end=int(payload["world_end"]),
            cross_world_start=int(payload.get("cross_world_start", 0)),
            cross_world_end=int(payload.get("cross_world_end", 0)),
        )

    @staticmethod
    def _page_store_key_lookup_update(key: StoredStripeKey) -> PageStripeUpdate:
        return PageStripeUpdate(
            axis=str(key.axis),
            world_start=int(key.world_start),
            world_end=int(key.world_end),
            buffer_start=0,
            buffer_end=max(1, int(key.world_end) - int(key.world_start)),
            kind="load",
            cross_world_start=int(getattr(key, "cross_world_start", 0)),
            cross_world_end=int(getattr(key, "cross_world_end", 0)),
        )

    def _coerce_page_stripe_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        cell_payload = dict(payload["cell"])
        gas_payload = dict(payload["gas"])
        runtime_payload = None if payload.get("runtime") is None else dict(payload["runtime"])
        gas_concentration = np.asarray(
            gas_payload["gas_concentration"],
            dtype=self.gas_concentration.dtype,
        ).copy()
        np.maximum(gas_concentration, 0.0, out=gas_concentration)
        if runtime_payload is not None and "entity_placeholder_entity_id" in runtime_payload:
            runtime_payload["entity_placeholder_entity_id"] = np.asarray(
                runtime_payload["entity_placeholder_entity_id"],
                dtype=np.int32,
            )
        if runtime_payload is not None:
            if "island_ids" in runtime_payload:
                runtime_payload["island_ids"] = np.asarray(runtime_payload["island_ids"], dtype=np.int32)
            if "island_velocity" in runtime_payload:
                runtime_payload["island_velocity"] = np.asarray(runtime_payload["island_velocity"], dtype=np.float32)
            if "island_subcell_offset" in runtime_payload:
                runtime_payload["island_subcell_offset"] = np.asarray(
                    runtime_payload["island_subcell_offset"],
                    dtype=np.float32,
                )
        return {
            "meta": dict(payload["meta"]),
            "cell": {
                "material_id": np.asarray(cell_payload["material_id"], dtype=self.material_id.dtype),
                "phase": np.asarray(cell_payload["phase"], dtype=self.phase.dtype),
                "cell_flags": np.asarray(cell_payload["cell_flags"], dtype=self.cell_flags.dtype),
                "velocity": np.asarray(cell_payload["velocity"], dtype=self.velocity.dtype),
                "cell_temperature": np.asarray(cell_payload["cell_temperature"], dtype=self.cell_temperature.dtype),
                "timer_pack": np.asarray(cell_payload["timer_pack"], dtype=self.timer_pack.dtype),
                "integrity": np.asarray(cell_payload["integrity"], dtype=self.integrity.dtype),
                "island_id": np.asarray(cell_payload["island_id"], dtype=self.island_id.dtype),
                "entity_id": np.asarray(cell_payload["entity_id"], dtype=self.entity_id.dtype),
                "placeholder_displaced_material": np.asarray(
                    cell_payload["placeholder_displaced_material"],
                    dtype=self.placeholder_displaced_material.dtype,
                ),
                "collapse_delay_pending": np.asarray(
                    cell_payload["collapse_delay_pending"],
                    dtype=np.uint8,
                ),
                "visible_illumination": np.asarray(
                    cell_payload["visible_illumination"],
                    dtype=self.visible_illumination.dtype,
                ),
                "cell_optical_dose": np.asarray(
                    cell_payload["cell_optical_dose"],
                    dtype=self.cell_optical_dose.dtype,
                ),
            },
            "runtime": runtime_payload,
            "gas": {
                "ambient_temperature": np.asarray(
                    gas_payload["ambient_temperature"],
                    dtype=self.ambient_temperature.dtype,
                ),
                "flow_velocity": np.asarray(gas_payload["flow_velocity"], dtype=self.flow_velocity.dtype),
                "pressure_ping": np.asarray(gas_payload["pressure_ping"], dtype=self.pressure_ping.dtype),
                "gas_concentration": gas_concentration,
                "gas_optical_dose": np.asarray(
                    gas_payload["gas_optical_dose"],
                    dtype=self.gas_optical_dose.dtype,
                ),
            },
        }

    def sync_entity_placeholders(
        self,
        placeholders: list[EntityPlaceholder],
        *,
        immediate: bool = False,
    ) -> None:
        placeholders = [self._public_entity_placeholder_input(placeholder) for placeholder in placeholders]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self._sync_entity_placeholders(
                [self._frame_entity_placeholder_input(placeholder) for placeholder in placeholders]
            )
            return
        self.queue_command(
            "sync_entity_placeholders",
            placeholders=[asdict(placeholder) for placeholder in placeholders],
        )

    def sync_entity_states(
        self,
        entities: list[EntityState | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        entities = [self._public_entity_state_input(entity) for entity in entities]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            placeholders, _ = self._sync_entity_states(
                [self._frame_entity_state_input(entity) for entity in entities]
            )
            self._sync_entity_placeholders(placeholders)
            return
        self.queue_command(
            "sync_entity_states",
            entities=[asdict(entity) for entity in entities],
        )

    def patch_entity_states(
        self,
        patches: list[EntityStatePatch | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        patches = [self._public_entity_state_patch_input(patch) for patch in patches]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self._patch_entity_states(
                [self._frame_entity_state_patch_input(patch) for patch in patches]
            )
            return
        self.queue_command(
            "patch_entity_states",
            patches=[asdict(patch) for patch in patches],
        )

    def sync_entity_observation_specs(
        self,
        observations: list[EntityObservationSpec | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        observations = [self._coerce_entity_observation_spec(observation) for observation in observations]
        if immediate:
            self._sync_entity_observation_specs(observations)
            return
        self.queue_command(
            "sync_entity_observation_specs",
            observations=[asdict(observation) for observation in observations],
        )

    def set_force_sources(
        self,
        force_sources: list[ForceSource | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        force_sources = [self._public_force_source_input(force_source) for force_source in force_sources]
        if immediate:
            self._sync_force_sources(
                [self._normalize_runtime_force_source(force_source) for force_source in force_sources]
            )
            return
        self.queue_command(
            "set_force_sources",
            force_sources=[
                {
                    "x": float(force_source.world_x),
                    "y": float(force_source.world_y),
                    "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
                    "radius": float(force_source.radius),
                    "strength": float(force_source.strength),
                    "lifetime": float(force_source.lifetime),
                }
                for force_source in force_sources
            ],
        )

    def set_emitters(
        self,
        emitters: list[dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        emitters = [self._coerce_emitter(emitter) for emitter in emitters]
        if immediate:
            normalized_emitters = [
                {
                    **dict(emitter),
                    "origin": self._world_to_buffer_clamped(
                        int(emitter["world_origin"][0]),
                        int(emitter["world_origin"][1]),
                    ),
                }
                for emitter in emitters
            ]
            self._sync_persistent_emitters(normalized_emitters)
            return
        self.queue_command(
            "set_emitters",
            emitters=[
                {
                    "x": int(emitter["origin"][0]),
                    "y": int(emitter["origin"][1]),
                    "light_type": str(emitter["light_type"]),
                    "direction": list(emitter["direction"]),
                    "spread": float(emitter["spread"]),
                    "strength": float(emitter["strength"]),
                    "radius": int(emitter["range_cells"]),
                }
                for emitter in emitters
            ],
        )

    def patch_material(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_material",
                name=str(self._canonical_material_input_name(name)),
                fields=self._normalize_material_patch_fields(fields),
            )
            return
        material_id = self._resolve_sanctioned_material_id(name)
        if material_id <= 0:
            raise KeyError(name)
        material = self._shadow_material_def(material_id)
        if material is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_material_name(material_id))
        patch_fields.setdefault("material_id", int(material_id))
        updated = self._coerce_material_def(asdict(replace(material, **patch_fields)))
        self.update_material_table([updated])

    def patch_light(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command("patch_light", name=name, fields=fields)
            return
        light_id = self._resolve_sanctioned_light_id(name)
        if light_id < 0:
            raise KeyError(name)
        light = self._shadow_light_type_def(light_id)
        if light is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_light_name(light_id))
        patch_fields.setdefault("light_type_id", int(light_id))
        updated = self._coerce_light_type_def(asdict(replace(light, **patch_fields)))
        self.update_light_type_table([updated])

    def patch_gas(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_gas",
                name=name,
                fields=self._normalize_gas_patch_fields(fields),
            )
            return
        species_id = self._resolve_sanctioned_gas_id(name)
        if species_id < 0:
            raise KeyError(name)
        gas = self._shadow_gas_species_def(species_id)
        if gas is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_gas_name(species_id))
        patch_fields.setdefault("species_id", int(species_id))
        updated = self._coerce_gas_species_def(asdict(replace(gas, **patch_fields)))
        self.update_gas_species_table([updated])

    def patch_material_optics(
        self,
        material_name: str,
        light_type: str,
        *,
        immediate: bool = True,
        **fields: Any,
    ) -> None:
        if not immediate:
            self.queue_command(
                "patch_material_optics",
                material_name=str(self._canonical_material_input_name(material_name)),
                light_type=light_type,
                fields=self._normalize_material_optics_patch_fields(fields),
            )
            return
        material_id = self._resolve_sanctioned_material_id(material_name)
        if material_id <= 0:
            raise KeyError(material_name)
        light_id = self._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            raise KeyError(light_type)
        canonical_material_name = self._shadow_material_name(material_id)
        canonical_light_type = self._shadow_light_name(light_id)
        if canonical_material_name is None or canonical_light_type is None:
            raise KeyError((material_name, light_type))
        optics = self._shadow_material_optics_def(canonical_material_name, canonical_light_type)
        if optics is None:
            raise KeyError((material_name, light_type))
        patch_fields = dict(fields)
        patch_fields.setdefault("material_name", canonical_material_name)
        patch_fields.setdefault("light_type", canonical_light_type)
        updated = self._coerce_material_optics_def(asdict(replace(optics, **patch_fields)))
        self.update_material_optics_table([updated])

    def patch_reaction_action(self, index: int, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_reaction_action",
                index=index,
                fields=self._normalize_reaction_action_patch_fields(fields),
            )
            return
        if index <= 0:
            raise ValueError("reaction action 0 is reserved")
        if index >= len(self.rulebook.reaction_actions):
            raise IndexError(index)
        action = self._shadow_reaction_action(index)
        if action is None:
            raise IndexError(index)
        updated = self._coerce_reaction_action(asdict(replace(action, **fields)))
        reactions_payload = self._shadow_reaction_payload()
        if index == 0:
            self.rulebook.reaction_actions = [updated] + [
                self._coerce_reaction_action(item) for item in reactions_payload["actions"]
            ]
            self._validate_reaction_payload(reactions_payload)
            self._set_stable_shadow_payload("reactions", reactions_payload)
            self.bridge.upload_table("reactions", reactions_payload)
            self.bridge.sync_rule_tables(self)
            return
        reactions_payload["actions"][index - 1] = asdict(updated)
        self._validate_reaction_payload(reactions_payload)
        self.rulebook.reaction_actions = [self.rulebook.reaction_actions[0]] + [
            self._coerce_reaction_action(item) for item in reactions_payload["actions"]
        ]
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def patch_reaction_rule(
        self,
        rule_set: str,
        index: int,
        *,
        immediate: bool = True,
        **fields: Any,
    ) -> None:
        rule_set = str(rule_set)
        if rule_set not in REACTION_RULE_SET_NAMES:
            raise KeyError(rule_set)
        if not immediate:
            self.queue_command(
                "patch_reaction_rule",
                rule_set=rule_set,
                index=index,
                fields=self._normalize_reaction_rule_patch_fields(fields),
            )
            return
        rule = self._shadow_reaction_rule(rule_set, index)
        if rule is None:
            raise IndexError(index)
        if rule_set == "self_rules":
            updated = self._coerce_self_reaction_rule(asdict(replace(rule, **fields)))
        else:
            updated = self._coerce_pair_reaction_rule(asdict(replace(rule, **fields)))
        reactions_payload = self._shadow_reaction_payload()
        reactions_payload["rules"][rule_set][index] = asdict(updated)
        self._validate_reaction_payload(reactions_payload)
        self._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def delete_reaction_action(self, index: int, *, immediate: bool = True) -> None:
        if not immediate:
            self.queue_command("delete_reaction_action", index=index)
            return
        if index <= 0:
            raise ValueError("reaction action 0 is reserved")
        if index >= len(self.rulebook.reaction_actions):
            raise IndexError(index)
        reactions_payload = self._shadow_reaction_payload()
        actions_payload = reactions_payload["actions"]
        if index > len(actions_payload):
            raise IndexError(index)
        materials_payload = self._shadow_material_payload()
        del actions_payload[index - 1]
        self._remap_reaction_payload_result_actions(reactions_payload["rules"], deleted_action_index=index)
        self._remap_material_payload_reaction_slots(materials_payload, deleted_action_index=index)
        self.rulebook.reaction_actions = [self.rulebook.reaction_actions[0]] + [
            self._coerce_reaction_action(item) for item in actions_payload
        ]
        self._set_reaction_rules_payload(reactions_payload["rules"])
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in materials_payload)
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        self._set_stable_shadow_payload("materials", materials_payload)
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("materials", materials_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)
        self.bridge.ensure_world_resources(self)

    def delete_reaction_rule(self, rule_set: str, index: int, *, immediate: bool = True) -> None:
        rule_set = str(rule_set)
        if rule_set not in REACTION_RULE_SET_NAMES:
            raise KeyError(rule_set)
        if not immediate:
            self.queue_command("delete_reaction_rule", rule_set=rule_set, index=index)
            return
        rules_list = self._reaction_rule_list(rule_set)
        if index < 0 or index >= len(rules_list):
            raise IndexError(index)
        reactions_payload = self._shadow_reaction_payload()
        del reactions_payload["rules"][rule_set][index]
        self._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def step(self, dt: float = 1.0 / 60.0, substeps: int = 1) -> None:
        for _ in range(max(1, substeps)):
            frame_input = self.pending_frame_inputs.popleft() if self.pending_frame_inputs else None
            output = self._step_once(dt, frame_input=frame_input, capture_output=frame_input is not None)
            if output is not None:
                self.completed_frame_outputs.append(output)

    def simulation_backend_report(self) -> dict[str, Any]:
        ctx = self.bridge.ctx
        gpu_available = bool(self.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)
        ctx_info = getattr(ctx, "info", {}) if ctx is not None else {}
        backends = {
            "collapse": str(self.collapse_solver.last_backend),
            "gas": str(self.gas_solver.last_backend),
            "heat": str(self.heat_solver.last_backend),
            "reactions": str(self.reaction_solver.last_runtime_backend),
            "motion": str(self.motion_solver.last_backend),
            "liquid": str(self.liquid_solver.last_backend),
            "placeholder": str(self.placeholder_pipeline.last_backend),
            "page_stripe": str(self.page_stripe_pipeline.last_backend),
            "world_commands": str(self.grid_command_pipeline.last_backend),
            "optics": str(self.optics_solver.last_backend),
        }
        non_gpu = {name: backend for name, backend in backends.items() if backend not in {"gpu", "idle"}}
        return {
            "simulation_backend": self.simulation_backend,
            "gpu_available": gpu_available,
            "renderer": str(ctx_info.get("GL_RENDERER", "")),
            "vendor": str(ctx_info.get("GL_VENDOR", "")),
            "opengl_version": str(ctx_info.get("GL_VERSION", "")),
            "gpu_realtime_budget": {
                "enabled": bool(self.gpu_realtime_budget_enabled),
                "active": bool(self._gpu_realtime_budget_active()),
                "cell_threshold": int(self.gpu_realtime_budget_cell_threshold),
                "skipped_stages": list(self.last_skipped_gpu_stages),
            },
            "backends": backends,
            "non_gpu_backends": non_gpu,
            "strict_gpu_ready": gpu_available and not non_gpu,
        }

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

    def consume_entity_observation_results(
        self,
        *,
        current_frame_id: int | None = None,
    ) -> dict[str, Any]:
        consumed_readbacks = self.poll_all_readbacks(current_frame_id=current_frame_id)
        observations = self._collect_observations(consumed_readbacks)
        entity_feedback = self._collect_entity_feedback(consumed_readbacks)
        frame_id = self.frame_id if current_frame_id is None else int(current_frame_id)
        return self._store_entity_observation_consume_snapshot(
            frame_id=frame_id,
            consumed_readbacks=consumed_readbacks,
            observations=observations,
            entity_feedback=entity_feedback,
        )

    def run_entity_controller_turn(
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
    ) -> dict[str, Any]:
        return run_entity_controller_turn(self, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands)

    def set_controller_state(self, controller_state: Any = None) -> dict[str, Any]:
        return set_controller_state(self, controller_state=controller_state)

    def serialize_controller_state(self) -> dict[str, Any]:
        return serialize_controller_state(self)

    def _build_preview_controller_turn_entities(
        self,
        *,
        entities: list[EntityState | dict[str, Any]] | None,
        patches: list[EntityStatePatch | dict[str, Any]] | None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None,
    ) -> list[EntityState] | None:
        return _build_preview_controller_turn_entities(self, entities=entities, patches=patches, observation_specs=observation_specs)

    def _preview_consume_entity_observation_results(self) -> dict[str, Any]:
        saved_completed_readbacks = deepcopy(self.completed_readbacks)
        saved_last_snapshot = deepcopy(self.last_entity_observation_consume_snapshot)
        try:
            return self.consume_entity_observation_results()
        finally:
            self.completed_readbacks = saved_completed_readbacks
            self.last_entity_observation_consume_snapshot = saved_last_snapshot

    def controller_turn_to_frame_input(
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
    ) -> WorldFrameInput:
        return controller_turn_to_frame_input(self, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands)

    def preview_entity_controller_turn(
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
        reserved_readback_request_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        return preview_entity_controller_turn(self, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands, reserved_readback_request_ids=reserved_readback_request_ids)

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
        frame_input = self.controller_turn_to_frame_input(
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
        return self.submit_frame_input(frame_input)

    def request_entity_controller_turn(
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
    ) -> dict[str, Any]:
        return request_entity_controller_turn(self, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands)

    def request_entity_controller_cycle(
        self,
        *,
        apply_turn: bool = True,
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
    ) -> dict[str, Any]:
        return request_entity_controller_cycle(self, apply_turn=apply_turn, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands)

    def run_entity_controller_cycle(
        self,
        *,
        apply_turn: bool = True,
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
    ) -> dict[str, Any]:
        return run_entity_controller_cycle(self, apply_turn=apply_turn, controller_state=controller_state, controller_state_provided=controller_state_provided, focus_center=focus_center, entities=entities, entity_placeholders=entity_placeholders, patches=patches, observation_specs=observation_specs, force_sources=force_sources, emitters=emitters, target_queries=target_queries, change_intents=change_intents, carrier_intents=carrier_intents, observation_targets=observation_targets, readback_requests=readback_requests, commands=commands)


    def run_cpu_frame(
        self,
        frame_input: WorldFrameInput | None = None,
        *,
        dt: float = 1.0 / 60.0,
        substeps: int = 1,
    ) -> WorldFrameOutput:
        output = self._step_once(dt, frame_input=frame_input, capture_output=True)
        assert output is not None
        for _ in range(max(1, substeps) - 1):
            self._step_once(dt, frame_input=None, capture_output=False)
            output.frame_id = self.frame_id
        return output

    def _step_once(
        self,
        dt: float,
        *,
        frame_input: WorldFrameInput | None,
        capture_output: bool,
    ) -> WorldFrameOutput | None:
        previous_frame_active = self._world_simulation_frame_active
        self._world_simulation_frame_active = True
        try:
            return self._step_once_impl(dt, frame_input=frame_input, capture_output=capture_output)
        finally:
            self._world_simulation_frame_active = previous_frame_active

    def _merge_phase_c(self) -> None:
        rxn_pipe = self.reaction_solver.gpu_pipeline
        rxn_cand = getattr(rxn_pipe, "_phase_c_rxn_candidate", None)
        if rxn_cand is None:
            return
        hr = self.heat_solver.gpu_pipeline.resources
        mr = self.motion_solver.gpu_pipeline.resources
        lr = self.liquid_solver.gpu_pipeline.resources
        candidates = MergeCandidates(
            heat={
                "material": hr.material_tex,
                "phase": hr.phase_tex,
                "temp": hr.temp_ping,
                "integrity": hr.integrity_tex,
                "flags": hr.cell_flags_tex,
            },
            reactions=rxn_cand,
            motion={
                "material": mr.material_tex,
                "phase": mr.phase_tex,
                "velocity": mr.velocity_tex,
            },
            liquid={
                "material": lr.material_in,
                "phase": lr.phase_in,
                "flags": lr.flags_in,
                "timer": lr.timer_in,
                "velocity": lr.velocity_in,
            },
        )
        self.merge_pipeline.merge_cell_core(self, candidates)

    def _step_once_impl(
        self,
        dt: float,
        *,
        frame_input: WorldFrameInput | None,
        capture_output: bool,
    ) -> WorldFrameOutput | None:
        self.last_skipped_gpu_stages = []
        self.last_pass_profile = {"passes": [], "summary": {}, "skipped_stages": self.last_skipped_gpu_stages}
        if not self._bridge_inputs_prepared:
            self._prepare_bridge_frame_inputs()
        consumed_readbacks: list[ReadbackResult] = []
        resolved_targets: dict[str, ResolvedTarget] = {}
        resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
        resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
        observations: dict[int, ObservationResult] = {}
        entity_feedback: dict[int, EntityFeedback] = {}
        paging_updates: list[PageStripeUpdate] = []
        observation_plans: list[dict[str, Any]] = []
        readback_plans: list[dict[str, Any]] = []
        bridge_upload_snapshot: dict[str, Any] = {}
        bridge_frame_snapshot: dict[str, Any] = {}
        output_controller_state = deepcopy(self.controller_state_snapshot)
        queued_observations = 0
        queued_readbacks = 0
        queued_commands = 0
        placeholder_count = 0
        with self._profile_pass("readback"):
            self._collect_ready_readbacks(self.frame_id + 1)
            if capture_output:
                consumed_readbacks = self.poll_all_readbacks()
                observations = self._collect_observations(consumed_readbacks)
                entity_feedback = self._collect_entity_feedback(consumed_readbacks)
        with self._profile_pass("commands"):
            if frame_input is not None:
                (
                    output_controller_state,
                    paging_updates,
                    resolved_targets,
                    resolved_change_intents,
                    resolved_carrier_intents,
                    observation_plans,
                    readback_plans,
                    queued_observations,
                    queued_readbacks,
                    queued_commands,
                    placeholder_count,
                ) = self._apply_frame_input(frame_input)
            else:
                output_controller_state = deepcopy(self.controller_state_snapshot)

        self.frame_id += 1
        if capture_output:
            self._store_entity_observation_consume_snapshot(
                frame_id=self.frame_id,
                consumed_readbacks=consumed_readbacks,
                observations=observations,
                entity_feedback=entity_feedback,
            )
        with self._profile_pass("commands"):
            self._apply_commands()
        with self._profile_pass("pre_sync"):
            if self._needs_pre_simulation_bridge_sync(frame_input=frame_input):
                self._sync_pre_simulation_bridge_without_debug_upload()
                self._gpu_cpu_dirty_resources.clear()
        with self._profile_pass("commands"):
            persistent_observation_plans = self._queue_persistent_entity_observations()
            observation_plans.extend(persistent_observation_plans)
            queued_observations += len(persistent_observation_plans)
        if self.profile_passes_enabled:
            self.collapse_solver.gpu_pipeline.reset_pass_profile()
        collapse_pipeline = self.collapse_solver.gpu_pipeline
        with self._profile_pass("collapse"):
            if self._should_run_formal_collapse_this_frame():
                with collapse_pipeline._profile_pass(self, "dirty_tile_drain"):
                    self._drain_gpu_collapse_structure_dirty_tiles()
                self.collapse_solver.step(self)
            else:
                with collapse_pipeline._profile_pass(self, "scheduled_defer"):
                    pass
        collapse_profile = getattr(getattr(self.collapse_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(collapse_profile, dict):
            self.last_pass_profile["collapse"] = collapse_profile
        if self.profile_passes_enabled:
            self.gas_solver.gpu_pipeline.reset_pass_profile()
        with self._profile_pass("gas"):
            self.gas_solver.step(self, dt)
        gas_profile = getattr(getattr(self.gas_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(gas_profile, dict):
            self.last_pass_profile["gas"] = gas_profile
        if self.profile_passes_enabled:
            self.heat_solver.gpu_pipeline.reset_pass_profile()
        # Phase C measured ~0.8ms win (A/B 83.56 vs 84.37) but its 1-frame
        # cross-system latency shifts condensation position by 1 frame
        # (test_world_step_condenses_water_gas_into_water_liquid CPU!=GPU).
        # Condensation still occurs (no solver skip), but the 1-frame behavior
        # shift conflicts with the "不丢质量" goal. Disabled; infrastructure kept.
        phase_c_active = False and (
            self.simulation_backend == "gpu"
            and bool(self._world_simulation_frame_active)
            and self.merge_pipeline.available(self)
        )
        self.phase_c_defer_cell_publish = phase_c_active
        with self._profile_pass("heat"):
            self.heat_solver.step(self, dt)
        heat_profile = getattr(getattr(self.heat_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(heat_profile, dict):
            self.last_pass_profile["heat"] = heat_profile
        self.reaction_solver.reset_runtime_state(self)
        if self.profile_passes_enabled:
            self.reaction_solver.gpu_pipeline.reset_pass_profile()
        with self._profile_pass("reactions before motion"):
            self.reaction_solver.gpu_pipeline.begin_formal_reaction_segment(self, "before_motion")
            try:
                with self._profile_pass("reaction_timed"):
                    self.reaction_solver._advance_timed_slots(self)
                with self._profile_pass("reaction_self"):
                    self.reaction_solver._run_self_rules(self)
                with self._profile_pass("reaction_material_material"):
                    self.reaction_solver._run_material_material(self)
                with self._profile_pass("reaction_material_gas"):
                    self.reaction_solver._run_material_gas(self)
                with self._profile_pass("reaction_material_light"):
                    self.reaction_solver._run_material_light(self)
                with self._profile_pass("reaction_gas_gas"):
                    self.reaction_solver._run_gas_gas(self)
                with self._profile_pass("reaction_gas_light"):
                    self.reaction_solver._run_gas_light(self)
                self.reaction_solver.gpu_pipeline.flush_formal_reaction_segment(self, "before_motion")
            finally:
                self.reaction_solver.gpu_pipeline.end_formal_reaction_segment(self, "before_motion")
        with self._profile_pass("motion"):
            self.motion_solver.step(self, dt)
        motion_profile = getattr(getattr(self.motion_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(motion_profile, dict):
            self.last_pass_profile["motion"] = motion_profile
        with self._profile_pass("liquid"):
            self.liquid_solver.step(self)
        liquid_profile = getattr(getattr(self.liquid_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(liquid_profile, dict):
            self.last_pass_profile["liquid"] = liquid_profile
        if phase_c_active:
            with self._profile_pass("merge_cell_core"):
                self._merge_phase_c()
            self.phase_c_defer_cell_publish = False
        with self._profile_pass("optics"):
            self.optics_solver.step(self)
        optics_profile = getattr(self.optics_solver, "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(optics_profile, dict):
            self.last_pass_profile["optics"] = optics_profile
        with self._profile_pass("latch_clear"):
            if self.reaction_solver.gpu_pipeline.clear_reaction_latches(self):
                self.reaction_solver._note_runtime_backend("gpu")
            else:
                self._require_gpu_stage("reaction latch clearing")
                self.cell_flags &= np.uint8(~int(CellFlag.REACTION_LATCHED) & 0xFF)
                self.reaction_solver._note_runtime_backend("cpu")
        reaction_profile = getattr(getattr(self.reaction_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(reaction_profile, dict):
            self.last_pass_profile["reactions"] = reaction_profile
        with self._profile_pass("active_decay"):
            active_scheduler_gpu_authoritative = (
                self.simulation_backend == "gpu"
                and "active_tile_ttl" in self.bridge.gpu_authoritative_resources
            )
            if active_scheduler_gpu_authoritative:
                if not self.bridge.decay_active_scheduler(self):
                    self._require_gpu_stage("active scheduler decay")
                    raise RuntimeError("GPU active scheduler decay failed; CPU fallback is disabled")
            elif self.simulation_backend == "gpu":
                self._require_gpu_stage("active scheduler decay")
            else:
                self.active.decay()
        bridge_world_synced = False
        if capture_output:
            if self.simulation_backend != "gpu":
                self.bridge.sync_world(self)
                bridge_world_synced = True
            else:
                self.bridge.sync_force_sources(self)
            bridge_upload_snapshot = self.serialize_bridge_upload_snapshot()
            bridge_frame_snapshot = self.serialize_bridge_frame_snapshot()
        with self._profile_pass("readback"):
            self._finish_readbacks(world_synced=bridge_world_synced)
            self._collect_ready_readbacks(self.frame_id)
        self._bridge_inputs_prepared = False

        if not capture_output:
            return None
        return WorldFrameOutput(
            frame_id=self.frame_id,
            submission_id=frame_input.submission_id if frame_input is not None else None,
            controller_state=output_controller_state,
            consumed_readbacks=consumed_readbacks,
            resolved_targets=resolved_targets,
            resolved_change_intents=resolved_change_intents,
            resolved_carrier_intents=resolved_carrier_intents,
            observations=observations,
            entity_feedback=entity_feedback,
            paging_updates=paging_updates,
            observation_plans=observation_plans,
            readback_plans=readback_plans,
            bridge_upload_snapshot=bridge_upload_snapshot,
            bridge_frame_snapshot=bridge_frame_snapshot,
            queued_observations=queued_observations,
            queued_readbacks=queued_readbacks,
            queued_commands=queued_commands,
            placeholder_count=placeholder_count,
        )

    def _queue_persistent_entity_observations(self) -> list[dict[str, Any]]:
        if not self.entity_states:
            return []
        _, observation_targets = self._frame_entities_to_placeholders_and_observations(list(self.entity_states.values()))
        observation_pairs = self._build_observation_request_pairs(observation_targets, {})
        observation_pairs = [
            (target, self._assign_readback_request_id(request))
            for target, request in observation_pairs
        ]
        observation_requests = [request for _, request in observation_pairs]
        self.pending_readbacks.extend(observation_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
        return [
            self._serialize_observation_plan_for_target_request(target, request)
            for target, request in observation_pairs
        ]

    def _apply_frame_input(
        self,
        frame_input: WorldFrameInput,
    ) -> tuple[
        Any,
        list[PageStripeUpdate],
        dict[str, ResolvedTarget],
        dict[str, ResolvedChangeIntent],
        dict[str, ResolvedCarrierIntent],
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        int,
        int,
        int,
    ]:
        paging_updates: list[PageStripeUpdate] = []
        resolved_targets: dict[str, ResolvedTarget] = {}
        resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
        resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
        observation_plans: list[dict[str, Any]] = []
        readback_plans: list[dict[str, Any]] = []
        queued_observations = 0
        queued_readbacks = 0
        queued_commands = 0
        placeholder_count = 0
        if frame_input.controller_state_provided:
            self.controller_state_snapshot = deepcopy(frame_input.controller_state)
        output_controller_state = deepcopy(self.controller_state_snapshot)
        if frame_input.focus_center is not None:
            paging_updates = self.advance_paging(
                frame_input.focus_center[0],
                frame_input.focus_center[1],
                immediate=True,
            )
        placeholder_inputs = [
            self._frame_entity_placeholder_input(placeholder)
            for placeholder in (frame_input.entity_placeholders or [])
        ]
        if frame_input.entities is not None:
            entity_placeholders, _ = self._sync_entity_states(
                [self._frame_entity_state_input(entity) for entity in frame_input.entities]
            )
            placeholder_inputs = entity_placeholders + placeholder_inputs
        if frame_input.entity_placeholders is not None or frame_input.entities is not None:
            self._sync_entity_placeholders(placeholder_inputs)
            placeholder_count = len(placeholder_inputs)
        if frame_input.force_sources is not None:
            self._sync_force_sources(
                [self._frame_force_source_input(force_source) for force_source in frame_input.force_sources]
            )
        if frame_input.emitters is not None:
            self._sync_persistent_emitters(
                [self._frame_emitter_input(emitter) for emitter in frame_input.emitters]
            )
        resolved_targets = self._resolve_target_queries(frame_input.target_queries)
        resolved_change_intents, generated_commands = self._resolve_change_intents(frame_input.change_intents, resolved_targets)
        resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
            frame_input.carrier_intents,
            resolved_targets,
        )
        observation_pairs = self._build_observation_request_pairs(frame_input.observation_targets, resolved_targets)
        observation_pairs = [
            (target, self._assign_readback_request_id(request))
            for target, request in observation_pairs
        ]
        observation_requests = [request for _, request in observation_pairs]
        self.pending_readbacks.extend(observation_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
        observation_plans = [
            self._serialize_observation_plan_for_target_request(target, request)
            for target, request in observation_pairs
        ]
        queued_observations = len(observation_requests)
        for command in (
            generated_commands
            + generated_carrier_commands
            + self._resolve_targeted_commands(frame_input.commands, resolved_targets)
        ):
            self.queue_command(command.kind, **command.payload)
            queued_commands += 1
        readback_requests = self._resolve_readback_requests(frame_input.readback_requests, resolved_targets)
        readback_requests = [self._assign_readback_request_id(request) for request in readback_requests]
        self.pending_readbacks.extend(readback_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in readback_requests)
        readback_plans = self._serialize_readback_plans_for_requests(readback_requests)
        queued_readbacks = len(readback_requests)
        public_resolved_targets = {
            query_id: self._public_resolved_target(target)
            for query_id, target in resolved_targets.items()
        }
        public_resolved_change_intents = {
            intent_id: self._public_resolved_change_intent(intent)
            for intent_id, intent in resolved_change_intents.items()
        }
        public_resolved_carrier_intents = {
            intent_id: self._public_resolved_carrier_intent(intent)
            for intent_id, intent in resolved_carrier_intents.items()
        }
        return (
            output_controller_state,
            paging_updates,
            public_resolved_targets,
            public_resolved_change_intents,
            public_resolved_carrier_intents,
            observation_plans,
            readback_plans,
            queued_observations,
            queued_readbacks,
            queued_commands,
            placeholder_count,
        )

    def _prepare_preview_frame_context(
        self,
        frame_input: WorldFrameInput,
    ) -> tuple[
        list[PageStripeUpdate],
        list[tuple[PageStripeUpdate, dict[str, Any]]],
        list[ObservationTarget],
        list[EntityPlaceholder],
        int,
    ]:
        paging = deepcopy(self.paging)
        paging_updates: list[PageStripeUpdate] = []
        preview_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        if frame_input.focus_center is not None:
            paging_updates = paging.focus_on(frame_input.focus_center[0], frame_input.focus_center[1])
        self.paging = paging
        if paging_updates:
            preview_page_stripes = self._preview_apply_paging_updates(paging_updates)

        entity_observation_targets: list[ObservationTarget] = []
        if frame_input.entities is None:
            preview_entity_states = dict(self.entity_states)
            derived_placeholders: list[EntityPlaceholder] = []
            _, entity_observation_targets = self._frame_entities_to_placeholders_and_observations(
                list(preview_entity_states.values())
            )
        else:
            frame_entities = [self._frame_entity_state_input(entity) for entity in frame_input.entities]
            preview_entity_states = {entity.entity_id: entity for entity in frame_entities}
            derived_placeholders, entity_observation_targets = self._frame_entities_to_placeholders_and_observations(frame_entities)
        self.entity_states = preview_entity_states

        placeholder_inputs = [
            self._frame_entity_placeholder_input(placeholder)
            for placeholder in (frame_input.entity_placeholders or [])
        ]
        placeholder_count = 0
        if frame_input.entities is not None:
            placeholder_inputs = derived_placeholders + placeholder_inputs
        if frame_input.entities is not None or frame_input.entity_placeholders is not None:
            preview_placeholders, blocked_cells, released_cells = self._build_preview_entity_placeholders(placeholder_inputs)
            self.entity_placeholders = preview_placeholders
            self._resolver_blocked_cells = blocked_cells
            self._resolver_released_cells = released_cells
            placeholder_count = len(placeholder_inputs)
        else:
            self._resolver_blocked_cells = None
            self._resolver_released_cells = None
        return (
            paging_updates,
            preview_page_stripes,
            entity_observation_targets,
            placeholder_inputs,
            placeholder_count,
        )

    def _snapshot_preview_runtime_state(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id.copy(),
            "phase": self.phase.copy(),
            "cell_flags": self.cell_flags.copy(),
            "velocity": self.velocity.copy(),
            "cell_temperature": self.cell_temperature.copy(),
            "timer_pack": self.timer_pack.copy(),
            "integrity": self.integrity.copy(),
            "island_id": self.island_id.copy(),
            "entity_id": self.entity_id.copy(),
            "placeholder_displaced_material": self.placeholder_displaced_material.copy(),
            "collapse_delay_pending": self.collapse_delay_pending.copy(),
            "flow_velocity": self.flow_velocity.copy(),
            "ambient_temperature": self.ambient_temperature.copy(),
            "pressure_ping": self.pressure_ping.copy(),
            "gas_concentration": self.gas_concentration.copy(),
            "visible_illumination": self.visible_illumination.copy(),
            "cell_optical_dose": self.cell_optical_dose.copy(),
            "gas_optical_dose": self.gas_optical_dose.copy(),
            "active": deepcopy(self.active),
            "islands": deepcopy(self.islands),
            "next_island_id": int(self.next_island_id),
            "collapse_dirty_regions": list(self.collapse_dirty_regions),
            "collapse_deferred_regions": list(self.collapse_deferred_regions),
        }

    def _restore_preview_runtime_state(self, snapshot: dict[str, Any]) -> None:
        self.material_id = snapshot["material_id"]
        self.phase = snapshot["phase"]
        self.cell_flags = snapshot["cell_flags"]
        self.velocity = snapshot["velocity"]
        self.cell_temperature = snapshot["cell_temperature"]
        self.timer_pack = snapshot["timer_pack"]
        self.integrity = snapshot["integrity"]
        self.island_id = snapshot["island_id"]
        self.entity_id = snapshot["entity_id"]
        self.placeholder_displaced_material = snapshot["placeholder_displaced_material"]
        self.collapse_delay_pending = snapshot["collapse_delay_pending"]
        self.flow_velocity = snapshot["flow_velocity"]
        self.ambient_temperature = snapshot["ambient_temperature"]
        self.pressure_ping = snapshot["pressure_ping"]
        self.gas_concentration = snapshot["gas_concentration"]
        self.visible_illumination = snapshot["visible_illumination"]
        self.cell_optical_dose = snapshot["cell_optical_dose"]
        self.gas_optical_dose = snapshot["gas_optical_dose"]
        self.active = snapshot["active"]
        self.islands = deepcopy(snapshot["islands"])
        self.next_island_id = int(snapshot["next_island_id"])
        self.collapse_dirty_regions = snapshot["collapse_dirty_regions"]
        self.collapse_deferred_regions = snapshot["collapse_deferred_regions"]

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
        current_cells = {
            cell: entity_id
            for entity_id, cells in current_entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        changed_cells: set[tuple[int, int]] = set()
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            if next_placeholder is None or next_placeholder.entity_id != entity_id:
                changed_cells.add(cell)
        for cell, placeholder in next_cells.items():
            x, y = cell
            material_id = int(self.material_id[y, x])
            entity_id = int(self.entity_id[y, x])
            has_matching_placeholder_cell = (
                material_id > 0
                and self._shadow_material_is_placeholder(material_id)
                and entity_id == int(placeholder.entity_id)
            )
            if current_cells.get(cell) != placeholder.entity_id or not has_matching_placeholder_cell:
                changed_cells.add(cell)

        payload: list[dict[str, Any]] = []
        for x, y in sorted(changed_cells):
            world_rect = self._buffer_bbox_to_world_bbox((int(x), int(y), int(x) + 1, int(y) + 1))
            payload.append(
                {
                    "buffer_rect": [int(x), int(y), int(x) + 1, int(y) + 1],
                    "world_rect": [int(world_rect[0]), int(world_rect[1]), int(world_rect[2]), int(world_rect[3])],
                }
            )
        return payload

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

    def _serialize_preview_bridge_frame_snapshot(
        self,
        *,
        current_entity_placeholders: dict[int, set[tuple[int, int]]],
        resolved_commands: list[WorldCommand],
        observation_requests: list[ReadbackRequest],
        readback_requests: list[ReadbackRequest],
        placeholder_inputs: list[EntityPlaceholder],
        paging_updates: list[PageStripeUpdate],
        page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]],
        reserved_readback_request_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        return _serialize_preview_bridge_frame_snapshot(self, current_entity_placeholders=current_entity_placeholders, resolved_commands=resolved_commands, observation_requests=observation_requests, readback_requests=readback_requests, placeholder_inputs=placeholder_inputs, paging_updates=paging_updates, page_stripes=page_stripes, reserved_readback_request_ids=reserved_readback_request_ids)

    def _queue_loaded_collapse_pending_regions(self, update: PageStripeUpdate) -> None:
        if update.kind != "load":
            return
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                pending = self.collapse_delay_pending[:, start:end]
                ys, xs = np.nonzero(pending)
                if ys.size == 0:
                    continue
                self.collapse_deferred_regions.append(
                    (
                        max(0, start + int(xs.min()) - 1),
                        max(0, int(ys.min()) - 1),
                        min(self.width, start + int(xs.max()) + 2),
                        min(self.height, int(ys.max()) + 2),
                    )
                )
                continue
            pending = self.collapse_delay_pending[start:end, :]
            ys, xs = np.nonzero(pending)
            if ys.size == 0:
                continue
            self.collapse_deferred_regions.append(
                (
                    max(0, int(xs.min()) - 1),
                    max(0, start + int(ys.min()) - 1),
                    min(self.width, int(xs.max()) + 2),
                    min(self.height, start + int(ys.max()) + 2),
                )
            )

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
        x0, y0, x1, y1 = (int(value) for value in region)
        if x1 <= x0 or y1 <= y0:
            return []
        if axis == "x":
            overlap0 = max(x0, int(start))
            overlap1 = min(x1, int(end))
            if overlap0 >= overlap1:
                return [(x0, y0, x1, y1)]
            remaining: list[tuple[int, int, int, int]] = []
            if x0 < overlap0:
                remaining.append((x0, y0, overlap0, y1))
            if overlap1 < x1:
                remaining.append((overlap1, y0, x1, y1))
            return remaining
        overlap0 = max(y0, int(start))
        overlap1 = min(y1, int(end))
        if overlap0 >= overlap1:
            return [(x0, y0, x1, y1)]
        remaining = []
        if y0 < overlap0:
            remaining.append((x0, y0, x1, overlap0))
        if overlap1 < y1:
            remaining.append((x0, overlap1, x1, y1))
        return remaining

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.gas_solver.release()
        self.heat_solver.release()
        self.collapse_solver.release()
        self.motion_solver.release()
        self.liquid_solver.release()
        self.optics_solver.release()
        self.reaction_solver.release()
        self.placeholder_pipeline.release()
        self.page_stripe_pipeline.release()
        self.grid_command_pipeline.release()
        self.bridge.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def material_by_id(self, material_id: int) -> MaterialDef:
        return material_by_id(self, material_id)

    def allocate_island_id(self) -> int:
        return allocate_island_id(self)

    def _refresh_island_records_for_ids(self, island_ids: Iterable[int]) -> None:
        touched = {int(island_id) for island_id in island_ids if int(island_id) > 0}
        if not touched:
            return
        for island_id in touched:
            invalid_mask = (self.island_id == island_id) & (
                (self.phase != int(Phase.FALLING_ISLAND)) | (self.material_id <= 0)
            )
            if np.any(invalid_mask):
                self.island_id[invalid_mask] = 0
            coords = np.argwhere(
                (self.island_id == island_id)
                & (self.phase == int(Phase.FALLING_ISLAND))
                & (self.material_id > 0)
            )
            if coords.size == 0:
                self.islands.pop(island_id, None)
                continue
            min_y, min_x = coords.min(axis=0).tolist()
            max_y, max_x = coords.max(axis=0).tolist()
            previous = self.islands.get(island_id)
            if previous is None:
                velocity_xy = tuple(np.mean(self.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32).tolist())
                subcell_offset = (0.0, 0.0)
            else:
                velocity_xy = (float(previous.velocity_xy[0]), float(previous.velocity_xy[1]))
                subcell_offset = (float(previous.subcell_offset[0]), float(previous.subcell_offset[1]))
            self.islands[island_id] = FallingIslandRecord(
                island_id=island_id,
                bbox=(int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1),
                velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
                subcell_offset=subcell_offset,
            )
        self.next_island_id = max(int(self.next_island_id), max(self.islands, default=0) + 1, 1)

    def in_bounds(self, x: int, y: int) -> bool:
        return in_bounds(self, x, y)

    def cell_xy_to_gas(self, x: int, y: int) -> tuple[int, int]:
        """Map a cell-space (x, y) pair onto the lower-resolution gas grid."""
        return cell_xy_to_gas(self, x, y)

    def cell_to_gas(self, y: int, x: int) -> tuple[int, int]:
        """Map a cell-space (y, x) pair onto the lower-resolution gas grid."""
        return self.cell_xy_to_gas(x, y)

    def sample_ambient_to_cells(self) -> np.ndarray:
        return sample_ambient_to_cells(self)

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

    def sample_flow_to_cells(self) -> np.ndarray:
        return sample_flow_to_cells(self)

    def downsample_cells_to_gas(self, field: np.ndarray) -> np.ndarray:
        result = np.zeros((self.gas_height, self.gas_width), dtype=np.float32)
        for gy in range(self.gas_height):
            for gx in range(self.gas_width):
                x0 = gx * self.gas_cell_size
                y0 = gy * self.gas_cell_size
                block = field[y0 : min(self.height, y0 + self.gas_cell_size), x0 : min(self.width, x0 + self.gas_cell_size)]
                result[gy, gx] = float(block.mean()) if block.size else 0.0
        return result

    def add_gas_from_cells(self, mask: np.ndarray, species: str, amount: float) -> None:
        return add_gas_from_cells(self, mask, species, amount)

    def set_cell_by_id(self, x: int, y: int, material_id: int, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        return set_cell_by_id(self, x, y, material_id, phase=phase, mark_dirty=mark_dirty)

    def _inject_velocity_immediate(
        self,
        x: int,
        y: int,
        velocity: tuple[float, float],
        radius: int,
        *,
        carrier: str,
        mode: str,
    ) -> None:
        return _inject_velocity_immediate(self, x, y, velocity, radius, carrier=carrier, mode=mode)

    def _inject_temperature_immediate(self, x: int, y: int, delta: float, radius: int) -> None:
        return _inject_temperature_immediate(self, x, y, delta, radius)

    def _inject_gas_immediate(self, x: int, y: int, species: str, amount: float, radius: int) -> None:
        return _inject_gas_immediate(self, x, y, species, amount, radius)

    def set_cell(self, x: int, y: int, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        return set_cell(self, x, y, material_name, phase=phase, mark_dirty=mark_dirty)

    def clear_cell(self, x: int, y: int, *, mark_dirty: bool = True) -> None:
        return clear_cell(self, x, y, mark_dirty=mark_dirty)

    def clear_cells(self, mask: np.ndarray, *, mark_dirty: bool = True) -> None:
        return clear_cells(self, mask, mark_dirty=mark_dirty)

    def set_material_by_mask(self, mask: np.ndarray, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        return set_material_by_mask(self, mask, material_name, phase=phase, mark_dirty=mark_dirty)

    def swap_cells(self, x0: int, y0: int, x1: int, y1: int) -> None:
        return swap_cells(self, x0, y0, x1, y1)

    def clear_cell_region(self, x0: int, y0: int, x1: int, y1: int, *, mark_dirty: bool = True) -> None:
        return clear_cell_region(self, x0, y0, x1, y1, mark_dirty=mark_dirty)

    def serialize_local_cells(self, x: int, y: int, width: int, height: int) -> dict[str, Any]:
        return serialize_local_cells(self, x, y, width, height)

    def serialize_temperature_window(self, x: int, y: int, width: int, height: int) -> dict[str, Any]:
        return serialize_temperature_window(self, x, y, width, height)

    def serialize_gas(self, species: str) -> list[list[float]]:
        return serialize_gas(self, species)

    def serialize_pressure(self) -> list[list[float]]:
        return serialize_pressure(self)

    def serialize_velocity(self) -> list[list[list[float]]]:
        return serialize_velocity(self)

    def serialize_visible_illumination(self) -> list[list[list[float]]]:
        return serialize_visible_illumination(self)

    def serialize_gas_runtime(self) -> dict[str, Any]:
        return serialize_gas_runtime(self)

    def serialize_heat_runtime(self) -> dict[str, Any]:
        return serialize_heat_runtime(self)

    def serialize_liquid_runtime(self) -> dict[str, Any]:
        return serialize_liquid_runtime(self)

    def serialize_reaction_runtime(self) -> dict[str, Any]:
        return serialize_reaction_runtime(self)

    def serialize_collapse_runtime(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        return serialize_collapse_runtime(self, allow_gpu_sync_readback=allow_gpu_sync_readback)

    def serialize_optics_runtime(self) -> dict[str, Any]:
        return serialize_optics_runtime(self)

    def serialize_active_runtime(self) -> dict[str, Any]:
        return serialize_active_runtime(self)

    def serialize_motion_runtime(self) -> dict[str, Any]:
        return serialize_motion_runtime(self)

    def serialize_paging_state(self) -> dict[str, Any]:
        return serialize_paging_state(self)

    def serialize_engine_capabilities(self) -> dict[str, Any]:
        return serialize_engine_capabilities(self)

    def serialize_material_table(self) -> list[dict[str, Any]]:
        return serialize_material_table(self)

    def _serialize_bridge_ndarray(self, name: str, array: np.ndarray) -> dict[str, Any]:
        return _serialize_bridge_ndarray(self, name, array)

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
        if array.ndim < 2:
            return None
        trailing_shape = tuple(int(value) for value in array.shape[-2:])
        if trailing_shape == (int(self.height), int(self.width)):
            return "world"
        if trailing_shape == (int(self.gas_height), int(self.gas_width)):
            return "gas"
        return None

    def _serialize_bridge_ndarray_slice(
        self,
        name: str,
        array: np.ndarray,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return _serialize_bridge_ndarray_slice(self, name, array, offset=offset, limit=limit)

    def _serialize_bridge_ndarray_window(
        self,
        name: str,
        array: np.ndarray,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        return _serialize_bridge_ndarray_window(self, name, array, x=x, y=y, w=w, h=h)

    def _serialize_bridge_spatial_window_payload(
        self,
        name: str,
        array: np.ndarray,
        *,
        coord_space: str,
        window_origin: tuple[int, int],
        requested_size: tuple[int, int],
        window_size: tuple[int, int],
        window: np.ndarray,
    ) -> dict[str, Any]:
        return _serialize_bridge_spatial_window_payload(
            self,
            name,
            array,
            coord_space=coord_space,
            window_origin=window_origin,
            requested_size=requested_size,
            window_size=window_size,
            window=window,
        )

    def serialize_bridge_typed_table(self, name: str) -> dict[str, Any]:
        return serialize_bridge_typed_table(self, name)

    def serialize_bridge_typed_table_slice(self, name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        return serialize_bridge_typed_table_slice(self, name, offset=offset, limit=limit)

    def serialize_bridge_shadow_buffer(self, name: str) -> dict[str, Any]:
        return serialize_bridge_shadow_buffer(self, name)

    def serialize_bridge_shadow_buffer_slice(self, name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        return serialize_bridge_shadow_buffer_slice(self, name, offset=offset, limit=limit)

    def serialize_bridge_shadow_buffer_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        return serialize_bridge_shadow_buffer_window(self, name, x=x, y=y, w=w, h=h)

    def serialize_bridge_shadow_buffer_world_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        return serialize_bridge_shadow_buffer_world_window(self, name, x=x, y=y, w=w, h=h)

    def serialize_bridge_shadow_buffer_gas_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        return serialize_bridge_shadow_buffer_gas_window(self, name, x=x, y=y, w=w, h=h)

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
        cursor = payload
        for key in path[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[path[-1]] = value

    def serialize_bridge_upload_snapshot(self) -> dict[str, Any]:
        return serialize_bridge_upload_snapshot(self)

    def serialize_bridge_frame_snapshot(self) -> dict[str, Any]:
        return serialize_bridge_frame_snapshot(self)

    def _serialize_force_source_record(self, force_source: ForceSource) -> dict[str, Any]:
        return _serialize_force_source_record(self, force_source)

    def serialize_force_sources(self) -> list[dict[str, Any]]:
        return serialize_force_sources(self)

    def _serialize_emitter_record(self, emitter: dict[str, object]) -> dict[str, object]:
        return _serialize_emitter_record(self, emitter)

    def serialize_emitters(self) -> dict[str, list[dict[str, object]]]:
        return serialize_emitters(self)

    def serialize_gas_species_table(self) -> list[dict[str, Any]]:
        return serialize_gas_species_table(self)

    def serialize_light_type_table(self) -> list[dict[str, Any]]:
        return serialize_light_type_table(self)

    def serialize_material_optics_table(self) -> list[dict[str, Any]]:
        return serialize_material_optics_table(self)

    def serialize_reaction_table(self) -> dict[str, object]:
        return serialize_reaction_table(self)

    def serialize_optics(
        self,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
        *,
        light_type: str | None = None,
    ) -> dict[str, Any]:
        return serialize_optics(self, x, y, width, height, light_type=light_type)

    def serialize_readback_request(self, request: ReadbackRequest) -> dict[str, Any]:
        return serialize_readback_request(self, request)

    def _infer_readback_payload_coord_space(
        self,
        path: tuple[str, ...],
        *,
        resource_name: str | None = None,
    ) -> str | None:
        return _infer_readback_payload_coord_space(self, path, resource_name=resource_name)

    def _serialize_readback_source_descriptor(self, path: tuple[str, ...], value: Any) -> Any:
        return _serialize_readback_source_descriptor(self, path, value)

    def _serialize_readback_plan_for_request(self, request: ReadbackRequest) -> dict[str, Any]:
        return _serialize_readback_plan_for_request(self, request)

    def _serialize_readback_plans_for_requests(self, requests: list[ReadbackRequest]) -> list[dict[str, Any]]:
        return _serialize_readback_plans_for_requests(self, requests)

    def _serialize_observation_plan_for_target_request(
        self,
        target: ObservationTarget,
        request: ReadbackRequest,
    ) -> dict[str, Any]:
        return _serialize_observation_plan_for_target_request(self, target, request)

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

    def serialize_readback_plan(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return serialize_readback_plan(self, center_x, center_y, width, height, channels, request_id=request_id, observer_id=observer_id, label=label, target_query_id=target_query_id, target_dx=target_dx, target_dy=target_dy, target_queries=target_queries)

    def serialize_observation_plan(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return serialize_observation_plan(self, target, request_id=request_id, target_queries=target_queries)

    def serialize_world_command(self, command: WorldCommand) -> dict[str, Any]:
        return serialize_world_command(self, command)

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

    def serialize_frame_input(self, frame_input: WorldFrameInput) -> dict[str, Any]:
        return serialize_frame_input(self, frame_input)

    def _serialize_readback_payload(self, payload: Any) -> Any:
        return _serialize_readback_payload(self, payload)

    def serialize_readback_result(self, result: ReadbackResult) -> dict[str, Any]:
        return serialize_readback_result(self, result)

    def serialize_resolved_target(self, target: ResolvedTarget) -> dict[str, Any]:
        return serialize_resolved_target(self, target)

    def serialize_resolved_change_intent(self, intent: ResolvedChangeIntent) -> dict[str, Any]:
        return serialize_resolved_change_intent(self, intent)

    def serialize_resolved_carrier_intent(self, intent: ResolvedCarrierIntent) -> dict[str, Any]:
        return serialize_resolved_carrier_intent(self, intent)

    def serialize_observation_result(self, result: ObservationResult) -> dict[str, Any]:
        return serialize_observation_result(self, result)

    @staticmethod
    def serialize_entity_observation_spec(spec: EntityObservationSpec) -> dict[str, Any]:
        return serialize_entity_observation_spec(spec)

    def serialize_entity_state_patch(self, patch: EntityStatePatch) -> dict[str, Any]:
        return serialize_entity_state_patch(self, patch)

    @staticmethod
    def serialize_observation_target(target: ObservationTarget) -> dict[str, Any]:
        return serialize_observation_target(target)

    @staticmethod
    def serialize_entity_state_input(entity: EntityState) -> dict[str, Any]:
        return serialize_entity_state_input(entity)

    def serialize_entity_state(self, entity: EntityState) -> dict[str, Any]:
        return serialize_entity_state(self, entity)

    def serialize_entity_states(self) -> dict[str, Any]:
        return serialize_entity_states(self)

    def serialize_entity_observation_state(self) -> dict[str, Any]:
        return serialize_entity_observation_state(self)

    def _current_cell_state_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        if (
            self.simulation_backend == "gpu"
            and "cell_core" in self.bridge.gpu_authoritative_resources
            and self.bridge.enabled
            and self.bridge.ctx is not None
            and "cell_core" in self.bridge.buffers
        ):
            if not allow_gpu_sync_readback:
                return {
                    "material_id": self.material_id,
                    "phase": self.phase,
                    "integrity": self.integrity,
                }
            try:
                core = np.frombuffer(
                    self.bridge.buffers["cell_core"].read(size=self.width * self.height * 5 * np.dtype(np.uint32).itemsize),
                    dtype=np.uint32,
                ).reshape((self.height, self.width, 5))
                unpacked = unpack_cell_core(core)
                return {
                    "material_id": unpacked["material_id"].astype(np.int32, copy=False),
                    "phase": unpacked["phase"].astype(np.uint8, copy=False),
                    "integrity": unpacked["integrity"].astype(np.float32, copy=False),
                }
            except Exception as exc:
                raise RuntimeError(
                    "GPU-authoritative cell state is not directly readable from this thread; "
                    "use async readback for CPU-visible world snapshots"
                ) from exc
        return {
            "material_id": self.material_id,
            "phase": self.phase,
            "integrity": self.integrity,
        }

    def _current_entity_runtime_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        if (
            self.simulation_backend == "gpu"
            and self.bridge.enabled
            and self.bridge.ctx is not None
            and "entity_id" in self.bridge.gpu_authoritative_resources
            and "placeholder_displaced_material" in self.bridge.gpu_authoritative_resources
            and "entity_id" in self.bridge.buffers
            and "placeholder_displaced_material" in self.bridge.buffers
        ):
            if not allow_gpu_sync_readback:
                return {
                    "entity_id": self.entity_id,
                    "placeholder_displaced_material": self.placeholder_displaced_material,
                }
            try:
                return {
                    "entity_id": np.frombuffer(
                        self.bridge.buffers["entity_id"].read(size=self.entity_id.nbytes),
                        dtype=np.int32,
                    ).reshape(self.entity_id.shape),
                    "placeholder_displaced_material": np.frombuffer(
                        self.bridge.buffers["placeholder_displaced_material"].read(size=self.placeholder_displaced_material.nbytes),
                        dtype=np.int32,
                    ).reshape(self.placeholder_displaced_material.shape),
                }
            except Exception as exc:
                raise RuntimeError(
                    "GPU-authoritative entity runtime state is not directly readable from this thread; "
                    "use async readback for CPU-visible world snapshots"
                ) from exc
        return {
            "entity_id": self.entity_id,
            "placeholder_displaced_material": self.placeholder_displaced_material,
        }

    def _entity_placeholder_state_gpu_authoritative(self) -> bool:
        if self.simulation_backend != "gpu":
            return False
        authoritative = self.bridge.gpu_authoritative_resources
        return bool(
            "cell_core" in authoritative
            or "entity_id" in authoritative
            or "placeholder_displaced_material" in authoritative
        )

    def serialize_entity_placeholders(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        return serialize_entity_placeholders(self, allow_gpu_sync_readback=allow_gpu_sync_readback)

    def serialize_entity_placeholder_index_snapshot(self) -> dict[str, Any]:
        return serialize_entity_placeholder_index_snapshot(self)

    def serialize_entity_feedback_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        return serialize_entity_feedback_snapshot(self, allow_gpu_sync_readback=allow_gpu_sync_readback)

    def serialize_consumed_entity_feedback_snapshot(self) -> dict[str, Any]:
        return serialize_consumed_entity_feedback_snapshot(self)

    def _serialize_cpu_visible_entity_placeholders(self) -> dict[str, Any]:
        return _serialize_cpu_visible_entity_placeholders(self)

    def serialize_entity_feedback(self, feedback: EntityFeedback) -> dict[str, Any]:
        return serialize_entity_feedback(self, feedback)

    def _store_entity_observation_consume_snapshot(
        self,
        *,
        frame_id: int,
        consumed_readbacks: list[ReadbackResult],
        observations: dict[int, ObservationResult],
        entity_feedback: dict[int, EntityFeedback],
    ) -> dict[str, Any]:
        snapshot = {
            "frame_id": int(frame_id),
            "consumed": len(consumed_readbacks),
            "consumed_readbacks": [self.serialize_readback_result(result) for result in consumed_readbacks],
            "observations": {
                str(observer_id): self.serialize_observation_result(result)
                for observer_id, result in observations.items()
            },
            "entity_feedback": {
                str(entity_id): self.serialize_entity_feedback(feedback)
                for entity_id, feedback in entity_feedback.items()
            },
        }
        self.last_entity_observation_consume_snapshot = snapshot
        return deepcopy(snapshot)

    def serialize_entity_observation_consume_state(self) -> dict[str, Any]:
        return serialize_entity_observation_consume_state(self)

    def serialize_frame_output(self, output: WorldFrameOutput) -> dict[str, Any]:
        return serialize_frame_output(self, output)

    def serialize_frame_preview(self, preview: WorldFramePreview) -> dict[str, Any]:
        return serialize_frame_preview(self, preview)

    def serialize_debug_frame(
        self,
        view: DebugView | str,
        *,
        gas_species: str | None = None,
        light_type: str | None = None,
    ) -> dict[str, Any]:
        return serialize_debug_frame(self, view, gas_species=gas_species, light_type=light_type)

    def debug_frame(
        self,
        view: DebugView,
        *,
        gas_species: str | None = None,
        light_type: str | None = None,
    ) -> np.ndarray:
        return debug_frame(self, view, gas_species=gas_species, light_type=light_type)

    def _material_frame(self) -> np.ndarray:
        return _material_frame(self)

    def _temperature_frame(self) -> np.ndarray:
        return _temperature_frame(self)

    def _pressure_frame(self) -> np.ndarray:
        return _pressure_frame(self)

    def _vector_field_frame(self, vectors: np.ndarray) -> np.ndarray:
        return _vector_field_frame(self, vectors)

    def _active_frame(self) -> np.ndarray:
        return _active_frame(self)

    def _motion_frame(self) -> np.ndarray:
        return _motion_frame(self)

    def _heat_frame(self) -> np.ndarray:
        return _heat_frame(self)

    def _liquid_frame(self) -> np.ndarray:
        return _liquid_frame(self)

    def _reaction_frame(self) -> np.ndarray:
        return _reaction_frame(self)

    def _collapse_frame(self) -> np.ndarray:
        return _collapse_frame(self)

    def _optics_frame(self, *, light_type: str | None = None) -> np.ndarray:
        return _optics_frame(self, light_type=light_type)

    def _optics_dose_frame(self, *, light_type: str | None = None) -> np.ndarray:
        return _optics_dose_frame(self, light_type=light_type)

    def _gas_frame(self, gas_species: str) -> np.ndarray:
        return _gas_frame(self, gas_species)

    def _accumulate_debug_point(self, frame: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
        return _accumulate_debug_point(self, frame, x, y, color)

    def _draw_debug_bbox_outline(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: np.ndarray,
    ) -> None:
        return _draw_debug_bbox_outline(self, frame, bbox, color)


    def _apply_grid_world_commands(self, commands: list[WorldCommand]) -> None:
        if not commands:
            self.grid_command_pipeline.last_backend = "idle"
            return
        if self._gpu_pipeline_available(
            self.grid_command_pipeline,
            "world command",
            require=self.simulation_backend == "gpu",
        ):
            self.grid_command_pipeline.apply(self, commands)
            if self.simulation_backend == "gpu" and self._world_simulation_frame_active:
                active_rects: list[tuple[int, int, int, int, int]] = []
                for command in commands:
                    active_rect, collapse_rect = self._grid_world_command_runtime_regions(command)
                    if active_rect is not None:
                        active_rects.append(active_rect)
                    if collapse_rect is not None:
                        self._mark_collapse_dirty_rect(*collapse_rect)
                if active_rects and not self.bridge.mark_active_rects(self, active_rects):
                    self._require_gpu_stage("active scheduler command marking")
            else:
                for command in commands:
                    self._mark_grid_world_command_runtime_regions(command)
            if self.grid_command_pipeline.last_cpu_mirror_downloaded:
                self._rebuild_island_records()
            return
        self._require_cpu_oracle_backend("world command")
        self.grid_command_pipeline.last_backend = "cpu"
        for command in commands:
            self._apply_grid_world_command_cpu(command)

    def _apply_grid_world_command_cpu(self, command: WorldCommand) -> None:
        if command.kind == "inject_material":
            x, y = self._queued_command_xy(command)
            self._paint_material(x, y, command.payload["material"], command.payload["radius"])
        elif command.kind == "write_material_region":
            x, y = self._queued_command_xy(command)
            self._write_material_region_immediate(
                x,
                y,
                command.payload["width"],
                command.payload["height"],
                command.payload["material"],
            )
        elif command.kind == "inject_temperature":
            x, y = self._queued_command_xy(command)
            self._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
        elif command.kind == "inject_velocity":
            x, y = self._queued_command_xy(command)
            self._inject_velocity_immediate(
                x,
                y,
                tuple(command.payload["velocity"]),
                command.payload["radius"],
                carrier=command.payload.get("carrier", "cell"),
                mode=command.payload.get("mode", "add"),
            )
        elif command.kind == "inject_gas":
            x, y = self._queued_command_xy(command)
            self._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])

    def _grid_world_command_runtime_regions(
        self,
        command: WorldCommand,
    ) -> tuple[tuple[int, int, int, int, int] | None, tuple[int, int, int, int] | None]:
        x, y = self._queued_command_xy(command)
        if command.kind == "write_material_region":
            width = max(0, int(command.payload["width"]))
            height = max(0, int(command.payload["height"]))
            x0 = max(0, int(x) - 1)
            y0 = max(0, int(y) - 1)
            x1 = min(self.width, int(x) + width + 1)
            y1 = min(self.height, int(y) + height + 1)
            if x0 < x1 and y0 < y1:
                return (
                    (x0, y0, x1, y1, 0),
                    (
                        max(0, int(x)),
                        max(0, int(y)),
                        min(self.width, int(x) + width),
                        min(self.height, int(y) + height),
                    ),
                )
            return None, None
        radius = max(0, int(command.payload.get("radius", 0)))
        pad = 1 if command.kind == "inject_material" else 0
        active_rect = (
            max(0, int(x) - radius - pad),
            max(0, int(y) - radius - pad),
            min(self.width, int(x) + radius + pad + 1),
            min(self.height, int(y) + radius + pad + 1),
        )
        collapse_rect = None
        if command.kind == "inject_material":
            collapse_rect = (
                max(0, int(x) - radius),
                max(0, int(y) - radius),
                min(self.width, int(x) + radius + 1),
                min(self.height, int(y) + radius + 1),
            )
        return (*active_rect, 0), collapse_rect

    def _mark_grid_world_command_runtime_regions(self, command: WorldCommand) -> None:
        active_rect, collapse_rect = self._grid_world_command_runtime_regions(command)
        if active_rect is not None:
            x0, y0, x1, y1, tile_padding = active_rect
            self._mark_active_rect_runtime(x0, y0, x1, y1, tile_padding=tile_padding)
        if collapse_rect is not None:
            self._mark_collapse_dirty_rect(*collapse_rect)

    def _apply_commands(self) -> None:
        pending_grid_commands: list[WorldCommand] = []
        def flush_pending_grid_commands() -> None:
            if not pending_grid_commands:
                return
            if self.simulation_backend == "gpu" and not self._world_simulation_frame_active:
                self.bridge.sync_world(self)
            self._apply_grid_world_commands(pending_grid_commands)
            pending_grid_commands.clear()

        while self.command_queue:
            command = self.command_queue.popleft()
            self.bridge_frame_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
            if command.kind in GPU_WORLD_COMMAND_KIND_IDS:
                if self.simulation_backend == "gpu":
                    self._gpu_pipeline_available(self.grid_command_pipeline, "world command")
                pending_grid_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
                continue
            flush_pending_grid_commands()
            if command.kind == "inject_material":
                x, y = self._queued_command_xy(command)
                self._paint_material(x, y, command.payload["material"], command.payload["radius"])
            elif command.kind == "write_material_region":
                x, y = self._queued_command_xy(command)
                self._write_material_region_immediate(
                    x,
                    y,
                    command.payload["width"],
                    command.payload["height"],
                    command.payload["material"],
                )
            elif command.kind == "inject_temperature":
                x, y = self._queued_command_xy(command)
                self._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
            elif command.kind == "inject_velocity":
                x, y = self._queued_command_xy(command)
                self._inject_velocity_immediate(
                    x,
                    y,
                    tuple(command.payload["velocity"]),
                    command.payload["radius"],
                    carrier=command.payload.get("carrier", "cell"),
                    mode=command.payload.get("mode", "add"),
                )
            elif command.kind == "inject_force":
                x, y = self._queued_command_xy(command)
                self._append_force_source_immediate(
                    ForceSource(
                        x=float(x),
                        y=float(y),
                        direction=tuple(command.payload["direction"]),
                        radius=float(command.payload["radius"]),
                        strength=float(command.payload["strength"]),
                        lifetime=float(command.payload.get("lifetime", 0.5)),
                        world_x=float(command.payload["x"]),
                        world_y=float(command.payload["y"]),
                    )
                )
            elif command.kind == "inject_gas":
                x, y = self._queued_command_xy(command)
                self._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])
            elif command.kind == "inject_light":
                light_type = command.payload["light_type"]
                light_id = self._resolve_sanctioned_light_id(str(light_type))
                if light_id < 0:
                    continue
                if "radius" in command.payload:
                    range_cells = int(command.payload["radius"])
                else:
                    shadow_default_range = self._shadow_light_default_range(light_id)
                    if shadow_default_range is None:
                        continue
                    range_cells = int(shadow_default_range)
                x, y = self._queued_command_xy(command)
                shadow_light = self._shadow_light_name(light_id)
                if shadow_light is None:
                    continue
                self._append_transient_light_emitter_immediate(
                    {
                        "light_type": shadow_light,
                        "origin": (x, y),
                        "world_origin": (int(command.payload["x"]), int(command.payload["y"])),
                        "direction": tuple(command.payload.get("direction", (0.0, 0.0))),
                        "spread": float(command.payload.get("spread", 0.25)),
                        "strength": command.payload["strength"],
                        "range_cells": range_cells,
                    }
                )
            elif command.kind == "sync_entity_placeholders":
                payload = command.payload.get("placeholders", [])
                self._sync_entity_placeholders(
                    [
                        self._frame_entity_placeholder_input(placeholder)
                        if isinstance(placeholder, EntityPlaceholder)
                        else self._frame_entity_placeholder_input(EntityPlaceholder(**placeholder))
                        for placeholder in payload
                    ]
                )
            elif command.kind == "sync_entity_states":
                payload = command.payload.get("entities", [])
                entities = [
                    self._frame_entity_state_input(entity)
                    if isinstance(entity, EntityState)
                    else self._frame_entity_state_input(self._coerce_entity_state(entity))
                    for entity in payload
                ]
                placeholders, _ = self._sync_entity_states(entities)
                self._sync_entity_placeholders(placeholders)
            elif command.kind == "patch_entity_states":
                payload = command.payload.get("patches", [])
                self._patch_entity_states(
                    [
                        self._frame_entity_state_patch_input(patch)
                        if isinstance(patch, EntityStatePatch)
                        else self._frame_entity_state_patch_input(self._coerce_entity_state_patch(patch))
                        for patch in payload
                    ]
                )
            elif command.kind == "sync_entity_observation_specs":
                payload = command.payload.get("observations", [])
                self._sync_entity_observation_specs(
                    [
                        observation
                        if isinstance(observation, EntityObservationSpec)
                        else self._coerce_entity_observation_spec(observation)
                        for observation in payload
                    ]
                )
            elif command.kind == "set_force_sources":
                payload = command.payload.get("force_sources", [])
                self._sync_force_sources(
                    [
                        self._public_force_source_input(force_source)
                        for force_source in payload
                    ]
                )
            elif command.kind == "set_emitters":
                payload = command.payload.get("emitters", [])
                normalized_emitters = []
                for emitter in payload:
                    record = self._coerce_emitter(emitter)
                    normalized_emitters.append(
                        {
                            **dict(record),
                            "origin": self._world_to_buffer_clamped(
                                int(record["world_origin"][0]),
                                int(record["world_origin"][1]),
                            ),
                        }
                    )
                self._sync_persistent_emitters(normalized_emitters)
            elif command.kind == "advance_paging":
                self._advance_paging(command.payload["center_x"], command.payload["center_y"])
            elif command.kind == "apply_page_stripe":
                update_payload = command.payload["update"]
                update = update_payload if isinstance(update_payload, PageStripeUpdate) else PageStripeUpdate(**update_payload)
                self.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
                self._apply_page_stripe(update, command.payload["payload"])
                self._record_bridge_page_stripe(update, command.payload["payload"])
            elif command.kind == "reset_world":
                self._reset_world_state(reset_bridge_frame_inputs=True, keep_command_log=True)
            elif command.kind == "request_readback":
                request = self._assign_readback_request_id(self._normalize_readback_request(ReadbackRequest(**command.payload)))
                self.pending_readbacks.append(request)
                self.bridge_frame_readback_requests.append(replace(request))
            elif command.kind == "update_material_table":
                self.update_material_table(command.payload["materials"], immediate=True)
            elif command.kind == "update_gas_species_table":
                self.update_gas_species_table(command.payload["gases"], immediate=True)
            elif command.kind == "update_light_type_table":
                self.update_light_type_table(command.payload["lights"], immediate=True)
            elif command.kind == "update_material_optics_table":
                self.update_material_optics_table(command.payload["optics"], immediate=True)
            elif command.kind == "update_reaction_table":
                self.update_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
            elif command.kind == "replace_reaction_table":
                self.replace_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
            elif command.kind == "patch_material":
                self.patch_material(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_light":
                self.patch_light(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_gas":
                self.patch_gas(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_material_optics":
                self.patch_material_optics(
                    command.payload["material_name"],
                    command.payload["light_type"],
                    immediate=True,
                    **command.payload["fields"],
                )
            elif command.kind == "patch_reaction_action":
                self.patch_reaction_action(command.payload["index"], immediate=True, **command.payload["fields"])
            elif command.kind == "delete_reaction_action":
                self.delete_reaction_action(command.payload["index"], immediate=True)
            elif command.kind == "patch_reaction_rule":
                self.patch_reaction_rule(
                    command.payload["rule_set"],
                    command.payload["index"],
                    immediate=True,
                    **command.payload["fields"],
                )
            elif command.kind == "delete_reaction_rule":
                self.delete_reaction_rule(
                    command.payload["rule_set"],
                    command.payload["index"],
                    immediate=True,
                )
        flush_pending_grid_commands()

    def _finish_readbacks(self, *, world_synced: bool = False) -> None:
        normalized_requests = [self._assign_readback_request_id(self._normalize_readback_request(request)) for request in self.pending_readbacks]
        self.pending_readbacks[:] = normalized_requests
        if self.pending_readbacks and not world_synced and self.simulation_backend != "gpu":
            self.bridge.sync_world(self)
        remaining_pending: list[ReadbackRequest] = []
        readback_upload_dirty = False
        for request in self.pending_readbacks:
            payload = self._make_readback_payload(request)
            if not self.bridge.queue_readback(
                self.frame_id,
                request,
                payload,
                require_gpu_sources=self.simulation_backend == "gpu",
            ):
                remaining_pending.append(request)
                continue
            readback_upload_dirty = True
            if request not in self.bridge_frame_readback_requests:
                self.bridge_frame_readback_requests.append(replace(request))
            if not any(existing.request_id == request.request_id for existing in self.inflight_readbacks):
                self.inflight_readbacks.append(replace(request))
        self.pending_readbacks[:] = remaining_pending
        if readback_upload_dirty:
            self.bridge.sync_readback_requests(self)

    def _collect_ready_readbacks(self, current_frame_id: int) -> None:
        while True:
            result = self.bridge.poll_readback(current_frame_id)
            if result is None:
                return
            if result.request.request_id is not None:
                self.inflight_readbacks = [
                    request for request in self.inflight_readbacks if request.request_id != result.request.request_id
                ]
                if int(result.request.request_id) in self.canceled_readback_request_ids:
                    continue
            self.completed_readbacks.append(result)

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

    def _apply_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        if self._gpu_pipeline_available(
            self.page_stripe_pipeline,
            "page stripe",
            require=self.simulation_backend == "gpu",
        ):
            self.page_stripe_pipeline.apply(self, update, payload)
        else:
            self._require_cpu_oracle_backend("page stripe")
            self.page_stripe_pipeline.last_backend = "cpu"
            self.page_stripe_pipeline.last_cpu_mirror_downloaded = True
            self._apply_page_stripe_dense_cpu(update, payload)

        self._normalize_page_stripe_cell_runtime(update)

        runtime_payload = payload.get("runtime")
        self._merge_island_runtime_payload(runtime_payload, update=update, payload=payload)
        if self.page_stripe_pipeline.last_cpu_mirror_downloaded:
            self._queue_loaded_collapse_pending_regions(update)
        else:
            self._queue_loaded_collapse_pending_regions_from_payload(update, payload)
        self._mark_loaded_page_stripe_active(update)
        if self.page_stripe_pipeline.last_cpu_mirror_downloaded:
            self._rebuild_island_records()
        self._apply_page_stripe_entity_placeholder_runtime(
            update,
            None
            if runtime_payload is None
            else runtime_payload.get("entity_placeholder_entity_id"),
        )
        self._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _apply_page_stripe_dense_cpu(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        gas_ranges = self._stripe_buffer_ranges(update, gas_grid=True)
        cell_payload = payload["cell"]
        gas_payload = payload["gas"]
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1

        self._write_stripe_array(self.material_id, update, cell_payload["material_id"], stripe_axis=cell_axis)
        self._write_stripe_array(self.phase, update, cell_payload["phase"], stripe_axis=cell_axis)
        self._write_stripe_array(self.cell_flags, update, cell_payload["cell_flags"], stripe_axis=cell_axis)
        self._write_stripe_array(self.velocity, update, cell_payload["velocity"], stripe_axis=cell_axis)
        self._write_stripe_array(self.cell_temperature, update, cell_payload["cell_temperature"], stripe_axis=cell_axis)
        self._write_stripe_array(self.timer_pack, update, cell_payload["timer_pack"], stripe_axis=cell_axis)
        self._write_stripe_array(self.integrity, update, cell_payload["integrity"], stripe_axis=cell_axis)
        self._write_stripe_array(self.island_id, update, cell_payload["island_id"], stripe_axis=cell_axis)
        self._write_stripe_array(self.entity_id, update, cell_payload["entity_id"], stripe_axis=cell_axis)
        self._write_stripe_array(
            self.placeholder_displaced_material,
            update,
            cell_payload["placeholder_displaced_material"],
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.collapse_delay_pending,
            update,
            np.asarray(cell_payload["collapse_delay_pending"], dtype=np.bool_),
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.visible_illumination,
            update,
            cell_payload["visible_illumination"],
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.cell_optical_dose,
            update,
            cell_payload["cell_optical_dose"],
            stripe_axis=cell_dose_axis,
        )

        self._write_stripe_array(
            self.ambient_temperature,
            update,
            gas_payload["ambient_temperature"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.flow_velocity,
            update,
            gas_payload["flow_velocity"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.pressure_ping,
            update,
            gas_payload["pressure_ping"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.gas_concentration,
            update,
            gas_payload["gas_concentration"],
            stripe_axis=2 if update.axis == "x" else 1,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.gas_optical_dose,
            update,
            gas_payload["gas_optical_dose"],
            stripe_axis=2 if update.axis == "x" else 1,
            ranges=gas_ranges,
        )

    def _queue_loaded_collapse_pending_regions_from_payload(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
    ) -> None:
        if update.kind != "load":
            return
        pending_payload = np.asarray(payload["cell"]["collapse_delay_pending"], dtype=np.bool_)
        offset = 0
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            span = int(end) - int(start)
            if span <= 0:
                continue
            if update.axis == "x":
                pending = pending_payload[:, offset : offset + span]
                ys, xs = np.nonzero(pending)
                if ys.size != 0:
                    self.collapse_deferred_regions.append(
                        (
                            max(0, start + int(xs.min()) - 1),
                            max(0, int(ys.min()) - 1),
                            min(self.width, start + int(xs.max()) + 2),
                            min(self.height, int(ys.max()) + 2),
                        )
                    )
            else:
                pending = pending_payload[offset : offset + span, :]
                ys, xs = np.nonzero(pending)
                if ys.size != 0:
                    self.collapse_deferred_regions.append(
                        (
                            max(0, int(xs.min()) - 1),
                            max(0, start + int(ys.min()) - 1),
                            min(self.width, int(xs.max()) + 2),
                            min(self.height, start + int(ys.max()) + 2),
                        )
                    )
            offset += span

    def _advance_paging(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        force_sources = [
            replace(force_source, world_x=self._force_source_world_position(force_source)[0], world_y=self._force_source_world_position(force_source)[1])
            for force_source in self.force_sources
        ]
        updates = self.focus_paging(center_x, center_y)
        if not updates:
            return []
        self.bridge_frame_paging_updates.extend(
            PageStripeUpdate(**asdict(update)) for update in updates
        )
        for update in updates:
            if update.kind == "save":
                self.page_store.save(update, self.capture_page_stripe(update))
                self._clear_saved_page_stripe_runtime_state(update)
        for update in updates:
            if update.kind != "load":
                continue
            payload = self.page_store.load(update)
            if payload is None:
                payload = self._default_page_stripe_payload(update)
            self._apply_page_stripe(update, payload)
            self._record_bridge_page_stripe(update, payload)
        if force_sources:
            self._sync_force_sources(force_sources)
        return updates

    def _prepare_bridge_frame_inputs(self) -> None:
        pending_placeholder_dirty_rects = list(self._pending_placeholder_dirty_rects)
        self._clear_bridge_frame_inputs(keep_commands=False, prepared=True)
        if pending_placeholder_dirty_rects:
            self.bridge_frame_placeholder_dirty_rects.extend(pending_placeholder_dirty_rects)
            self._pending_placeholder_dirty_rects.clear()

    def _needs_pre_simulation_bridge_sync(self, *, frame_input: WorldFrameInput | None) -> bool:
        if self.simulation_backend != "gpu":
            return False
        return bool(
            frame_input is not None
            or self.bridge_frame_placeholders
            or self.bridge_frame_placeholder_dirty_rects
            or self.bridge_frame_paging_updates
            or self.bridge_frame_page_stripes
            or self._gpu_cpu_dirty_resources
        )

    def _sync_pre_simulation_bridge_without_debug_upload(self) -> None:
        try:
            self.bridge.sync_world(self, upload_debug_texture=False)
        except TypeError as exc:
            if "upload_debug_texture" not in str(exc):
                raise
            self.bridge.sync_world(self)

    def _clear_bridge_frame_inputs(self, *, keep_commands: bool, prepared: bool) -> None:
        if not keep_commands:
            self.bridge_frame_commands.clear()
        self.bridge_frame_readback_requests.clear()
        self.bridge_frame_placeholders.clear()
        self.bridge_frame_placeholder_dirty_rects.clear()
        self.bridge_frame_paging_updates.clear()
        self.bridge_frame_page_stripes.clear()
        self._bridge_inputs_prepared = prepared

    def _mark_active_rect_runtime(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        tile_padding: int = 0,
    ) -> None:
        self._mark_active_rects_runtime([(x0, y0, x1, y1, tile_padding)])

    def _mark_active_rects_runtime(
        self,
        rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
    ) -> None:
        if not rects:
            return
        if self.simulation_backend == "gpu" and self._world_simulation_frame_active:
            if not self.bridge.mark_active_rects(self, rects):
                self._require_gpu_stage("active scheduler region marking")
            return
        for rect in rects:
            if len(rect) == 4:
                x0, y0, x1, y1 = rect
                tile_padding = 0
            else:
                x0, y0, x1, y1, tile_padding = rect
            self.active.mark_rect(int(x0), int(y0), int(x1), int(y1), tile_padding=int(tile_padding))
        if self.simulation_backend == "gpu":
            self._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _sync_entity_placeholders(self, placeholders: list[EntityPlaceholder]) -> None:
        self.bridge_frame_placeholders.extend(replace(placeholder) for placeholder in placeholders)
        current_cells = {
            cell: entity_id
            for entity_id, cells in self.entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        changed_cells: set[tuple[int, int]] = set()
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            if next_placeholder is None or next_placeholder.entity_id != entity_id:
                changed_cells.add(cell)
        for cell, placeholder in next_cells.items():
            if current_cells.get(cell) != placeholder.entity_id:
                changed_cells.add(cell)

        if self._gpu_pipeline_available(
            self.placeholder_pipeline,
            "placeholder",
            require=self.simulation_backend == "gpu",
        ):
            if current_cells or next_cells:
                if self.simulation_backend == "gpu" and self._world_simulation_frame_active and (
                    not self._bridge_inputs_prepared or self._gpu_cpu_dirty_resources
                ):
                    self._sync_pre_simulation_bridge_without_debug_upload()
                    self._gpu_cpu_dirty_resources.clear()
                    self._bridge_inputs_prepared = True
                self.placeholder_pipeline.apply(self, placeholders)
                if self.placeholder_pipeline.last_cpu_mirror_downloaded:
                    self._rebuild_entity_placeholder_index()
                else:
                    next_entity_cells: dict[int, set[tuple[int, int]]] = {}
                    for cell, placeholder in next_cells.items():
                        next_entity_cells.setdefault(int(placeholder.entity_id), set()).add(cell)
                    self.entity_placeholders = next_entity_cells
            else:
                self.entity_placeholders.clear()
                self.placeholder_pipeline.last_backend = "idle"
            for x, y in sorted(changed_cells):
                self._mark_active_rect_runtime(
                    max(0, x - 1),
                    max(0, y - 1),
                    min(self.width, x + 2),
                    min(self.height, y + 2),
                )
            self.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))
            return

        self._require_cpu_oracle_backend("placeholder")
        self.placeholder_pipeline.last_backend = "cpu" if (current_cells or next_cells) else "idle"
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            x, y = cell
            material_id = int(self.material_id[y, x])
            if (
                next_placeholder is not None
                and next_placeholder.entity_id == entity_id
                and material_id > 0
                and self._shadow_material_is_placeholder(material_id)
            ):
                continue
            self._release_entity_placeholder_cell(x, y, entity_id)

        next_entity_cells: dict[int, set[tuple[int, int]]] = {}
        for cell, placeholder in next_cells.items():
            x, y = cell
            material_id = int(self.material_id[y, x])
            if (
                current_cells.get(cell) == placeholder.entity_id
                and material_id > 0
                and self._shadow_material_is_placeholder(material_id)
            ):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
                self.entity_id[y, x] = placeholder.entity_id
                continue
            if self._occupy_entity_placeholder_cell(x, y, placeholder):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
        self.entity_placeholders = next_entity_cells
        self.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))

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

    def _release_entity_placeholder_cell(self, x: int, y: int, entity_id: int) -> None:
        if not self.in_bounds(x, y):
            return
        if int(self.entity_id[y, x]) != entity_id:
            return
        self.entity_id[y, x] = 0
        material_id = int(self.material_id[y, x])
        if material_id <= 0 or not self._shadow_material_is_placeholder(material_id):
            return
        displaced_material = int(self.placeholder_displaced_material[y, x])
        if displaced_material > 0:
            self.material_id[y, x] = displaced_material
            self.phase[y, x] = int(Phase.LIQUID)
            self.cell_flags[y, x] = 0
            self.timer_pack[y, x] = 0
            shadow_integrity = self._shadow_material_base_integrity(displaced_material)
            self.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
            self.placeholder_displaced_material[y, x] = 0
            self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
            return
        self.clear_cell(x, y, mark_dirty=False)

    def _mirror_release_entity_placeholder_cell(self, x: int, y: int, entity_id: int) -> None:
        if not self.in_bounds(x, y):
            return
        if int(self.entity_id[y, x]) != entity_id:
            return
        self._invalidate_gpu_authoritative_cell_resources()
        self.entity_id[y, x] = 0
        material_id = int(self.material_id[y, x])
        if material_id <= 0 or not self._shadow_material_is_placeholder(material_id):
            return
        displaced_material = int(self.placeholder_displaced_material[y, x])
        if displaced_material > 0:
            self.material_id[y, x] = displaced_material
            self.phase[y, x] = int(Phase.LIQUID)
            self.cell_flags[y, x] = 0
            self.timer_pack[y, x] = 0
            shadow_integrity = self._shadow_material_base_integrity(displaced_material)
            self.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
            self.placeholder_displaced_material[y, x] = 0
            self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
            return
        self.clear_cell(x, y, mark_dirty=False)

    def _resolve_sanctioned_material_id(self, name: str) -> int:
        return _resolve_sanctioned_material_id(self, name)

    def _shadow_material_id_by_name(self, name: str | None) -> int:
        return _shadow_material_id_by_name(self, name)

    def _resolve_sanctioned_placeholder_material_id(self, name: str) -> int:
        return _resolve_sanctioned_placeholder_material_id(self, name)

    def _resolve_sanctioned_light_id(self, name: str) -> int:
        return _resolve_sanctioned_light_id(self, name)

    def _resolve_sanctioned_gas_id(self, name: str) -> int:
        return _resolve_sanctioned_gas_id(self, name)

    def _shadow_material_row_valid(self, material_id: int) -> bool:
        return _shadow_material_row_valid(self, material_id)

    def _shadow_gas_row_valid(self, species_id: int) -> bool:
        return _shadow_gas_row_valid(self, species_id)

    def _shadow_light_row_valid(self, light_id: int) -> bool:
        return _shadow_light_row_valid(self, light_id)

    def _shadow_material_def(self, material_id: int) -> MaterialDef | None:
        return _shadow_material_def(self, material_id)

    def _shadow_light_type_def(self, light_id: int) -> LightTypeDef | None:
        return _shadow_light_type_def(self, light_id)

    def _shadow_gas_species_def(self, species_id: int) -> GasSpeciesDef | None:
        return _shadow_gas_species_def(self, species_id)

    def _shadow_material_optics_def(self, material_name: str, light_type: str) -> MaterialOpticsDef | None:
        return _shadow_material_optics_def(self, material_name, light_type)

    def _shadow_material_name(self, material_id: int) -> str | None:
        return _shadow_material_name(self, material_id)

    def _shadow_gas_name(self, species_id: int) -> str | None:
        return _shadow_gas_name(self, species_id)

    def _shadow_light_name(self, light_id: int) -> str | None:
        return _shadow_light_name(self, light_id)

    def _shadow_light_default_range(self, light_id: int) -> int | None:
        return _shadow_light_default_range(self, light_id)

    def _shadow_light_dose_channel(self, light_id: int) -> int | None:
        return _shadow_light_dose_channel(self, light_id)

    def _shadow_light_color(self, light_id: int) -> np.ndarray | None:
        return _shadow_light_color(self, light_id)

    def _shadow_light_name_and_range(self, light_id: int) -> tuple[str, int] | None:
        return _shadow_light_name_and_range(self, light_id)

    def _shadow_material_default_phase(self, material_id: int) -> int | None:
        return _shadow_material_default_phase(self, material_id)

    def _shadow_material_base_integrity(self, material_id: int) -> float | None:
        return _shadow_material_base_integrity(self, material_id)

    def _shadow_material_spawn_temperature(self, material_id: int) -> float | None:
        return _shadow_material_spawn_temperature(self, material_id)

    def _shadow_condense_target_material_id(self, species_id: int) -> int:
        return _shadow_condense_target_material_id(self, species_id)

    def _shadow_material_is_placeholder(self, material_id: int) -> bool:
        return _shadow_material_is_placeholder(self, material_id)

    def _material_placeholder_mask(self, material_id: np.ndarray) -> np.ndarray:
        ids = np.asarray(material_id, dtype=np.int64)
        mask = np.zeros(ids.shape, dtype=np.bool_)
        valid = (ids >= 0) & (ids < int(self.material_is_placeholder.shape[0]))
        if np.any(valid):
            mask[valid] = self.material_is_placeholder[ids[valid]]
        return mask

    def _shadow_material_is_plant(self, material_id: int) -> bool:
        return _shadow_material_is_plant(self, material_id)

    def _shadow_reaction_action(self, index: int) -> ReactionAction | None:
        return _shadow_reaction_action(self, index)

    def _reaction_rule_list(self, rule_set: str) -> list[PairReactionRule] | list[SelfReactionRule]:
        return _reaction_rule_list(self, rule_set)

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

    def _shadow_reaction_rule(self, rule_set: str, index: int) -> PairReactionRule | SelfReactionRule | None:
        return _shadow_reaction_rule(self, rule_set, index)

    def _occupy_entity_placeholder_cell(self, x: int, y: int, placeholder: EntityPlaceholder) -> bool:
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

    def _frame_entities_to_placeholders_and_observations(
        self,
        entities: list[EntityState],
    ) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
        placeholders = [
            EntityPlaceholder(
                entity_id=entity.entity_id,
                x=entity.x,
                y=entity.y,
                width=entity.width,
                height=entity.height,
                material=entity.placeholder_material,
                world_x=entity.world_x,
                world_y=entity.world_y,
            )
            for entity in entities
        ]
        observation_targets = [
            ObservationTarget(
                observer_id=entity.entity_id,
                entity_id=entity.entity_id,
                channels=entity.observe_channels,
                pad_cells=entity.observe_pad_cells,
                width=entity.observe_width,
                height=entity.observe_height,
                label=entity.observe_label,
            )
            for entity in entities
            if entity.observe_channels
        ]
        return placeholders, observation_targets

    def _runtime_entities_to_immediate_observation_targets(
        self,
        entities: list[EntityState],
    ) -> list[ObservationTarget]:
        targets: list[ObservationTarget] = []
        for entity in entities:
            if not entity.observe_channels:
                continue
            if entity.world_x is not None and entity.world_y is not None:
                world_x = int(entity.world_x)
                world_y = int(entity.world_y)
            else:
                world_x, world_y = self._buffer_to_world_position((int(entity.x), int(entity.y)))
            entity_width = max(1, int(entity.width))
            entity_height = max(1, int(entity.height))
            center_x = int((world_x + world_x + entity_width - 1) // 2)
            center_y = int((world_y + world_y + entity_height - 1) // 2)
            width = int(entity.observe_width) if entity.observe_width is not None else entity_width + int(entity.observe_pad_cells) * 2
            height = int(entity.observe_height) if entity.observe_height is not None else entity_height + int(entity.observe_pad_cells) * 2
            targets.append(
                ObservationTarget(
                    observer_id=int(entity.entity_id),
                    center_x=int(center_x),
                    center_y=int(center_y),
                    width=max(1, int(width)),
                    height=max(1, int(height)),
                    channels=entity.observe_channels,
                    pad_cells=int(entity.observe_pad_cells),
                    label=entity.observe_label,
                )
            )
        return targets

    def _sync_entity_states(self, entities: list[EntityState]) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
        self.entity_states = {entity.entity_id: entity for entity in entities}
        placeholders, observation_targets = self._frame_entities_to_placeholders_and_observations(entities)
        return placeholders, observation_targets

    def _sync_entity_observation_specs(self, observations: list[EntityObservationSpec]) -> None:
        observation_by_entity_id = {observation.entity_id: observation for observation in observations}
        self.entity_states = {
            entity_id: replace(
                entity,
                observe_channels=observation.observe_channels if observation is not None else (),
                observe_pad_cells=int(observation.observe_pad_cells) if observation is not None else 0,
                observe_width=None if observation is None else observation.observe_width,
                observe_height=None if observation is None else observation.observe_height,
                observe_label=None if observation is None else observation.observe_label,
            )
            for entity_id, entity in self.entity_states.items()
            for observation in [observation_by_entity_id.get(entity_id)]
        }

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

    def _build_preview_entity_placeholders(
        self,
        placeholders: list[EntityPlaceholder],
    ) -> tuple[dict[int, set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
        current_cells = {
            cell: entity_id
            for entity_id, cells in self.entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        released_cells = {
            cell
            for cell, entity_id in current_cells.items()
            if next_cells.get(cell) is None or next_cells[cell].entity_id != entity_id
        }
        next_entity_cells: dict[int, set[tuple[int, int]]] = {}
        for cell, placeholder in next_cells.items():
            if self._preview_can_occupy_placeholder_cell(cell[0], cell[1], placeholder, current_cells, released_cells):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
        blocked_cells = {
            cell
            for cells in next_entity_cells.values()
            for cell in cells
        }
        return next_entity_cells, blocked_cells, released_cells

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

    def _build_observation_request(
        self,
        target: ObservationTarget,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ReadbackRequest | None:
        center_x = target.center_x
        center_y = target.center_y
        width = target.width
        height = target.height
        if target.target_query_id is not None and (center_x is None or center_y is None or width is None or height is None):
            resolved_target = resolved_targets.get(target.target_query_id)
            if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
                return None
            if center_x is None:
                center_x = int(resolved_target.resolved_world_position[0]) + int(target.target_dx)
            if center_y is None:
                center_y = int(resolved_target.resolved_world_position[1]) + int(target.target_dy)
            if width is None:
                width = 1 + int(target.pad_cells) * 2
            if height is None:
                height = 1 + int(target.pad_cells) * 2
        if target.entity_id is not None and (center_x is None or center_y is None or width is None or height is None):
            bbox = self._entity_placeholder_bbox(target.entity_id)
            if bbox is None:
                return None
            x0, y0, x1, y1 = self._buffer_bbox_to_world_bbox(bbox)
            if center_x is None:
                center_x = (x0 + x1 - 1) // 2
            if center_y is None:
                center_y = (y0 + y1 - 1) // 2
            if width is None:
                width = (x1 - x0) + target.pad_cells * 2
            if height is None:
                height = (y1 - y0) + target.pad_cells * 2
        if center_x is None or center_y is None:
            return None
        return self._normalize_readback_request(
            ReadbackRequest(
                center_x=center_x,
                center_y=center_y,
                width=max(1, width if width is not None else 1),
                height=max(1, height if height is not None else 1),
                channels=target.channels,
                observer_id=target.observer_id,
                label=target.label,
                target_query_id=target.target_query_id,
                target_dx=int(target.target_dx),
                target_dy=int(target.target_dy),
            )
        )

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

    def _resolve_readback_request(
        self,
        request: ReadbackRequest,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ReadbackRequest | None:
        center_x = request.center_x
        center_y = request.center_y
        if request.target_query_id is not None and (center_x is None or center_y is None):
            resolved_target = resolved_targets.get(request.target_query_id)
            if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
                return None
            if center_x is None:
                center_x = int(resolved_target.resolved_world_position[0]) + int(request.target_dx)
            if center_y is None:
                center_y = int(resolved_target.resolved_world_position[1]) + int(request.target_dy)
        if center_x is None or center_y is None:
            return None
        return self._normalize_readback_request(
            ReadbackRequest(
                request_id=request.request_id,
                center_x=int(center_x),
                center_y=int(center_y),
                width=max(1, int(request.width)),
                height=max(1, int(request.height)),
                channels=request.channels,
                observer_id=request.observer_id,
                label=request.label,
                target_query_id=request.target_query_id,
                target_dx=int(request.target_dx),
                target_dy=int(request.target_dy),
            )
        )

    def _resolve_target_queries(self, queries: list[TargetQuery]) -> dict[str, ResolvedTarget]:
        return _resolve_target_queries(self, queries)

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
        effect_cells = (
            []
            if intent.center_world_position is None
            else self._disk_world_cells_raw(
                tuple(int(value) for value in intent.center_world_position),
                int(intent.effective_radius),
            )
        )
        effect_bounds = self._buffer_cell_bounds(effect_cells)
        generated_commands = [self._public_world_command(command) for command in intent.generated_commands]
        if intent.center_world_position is not None:
            center_world_x = int(intent.center_world_position[0])
            center_world_y = int(intent.center_world_position[1])
            for command in generated_commands:
                x_field, y_field = TARGETED_COMMAND_COORD_FIELDS.get(command.kind, (None, None))
                if x_field is not None and y_field is not None and x_field in command.payload and y_field in command.payload:
                    command.payload[x_field] = center_world_x
                    command.payload[y_field] = center_world_y
        return replace(
            intent,
            center_position=(
                None
                if intent.center_world_position is None
                else tuple(int(value) for value in intent.center_world_position)
            ),
            effect_cells=effect_cells,
            effect_bounds=effect_bounds,
            generated_commands=generated_commands,
        )

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

    def _public_resolved_carrier_intent(self, intent: ResolvedCarrierIntent) -> ResolvedCarrierIntent:
        effect_cells: list[tuple[int, int]]
        if intent.effect_shape == "beam" and intent.source_world_position is not None and intent.impact_world_position is not None:
            effect_cells = self._capsule_world_cells_raw(
                tuple(int(value) for value in intent.source_world_position),
                tuple(int(value) for value in intent.impact_world_position),
                int(intent.effective_radius),
            )
        elif intent.impact_world_position is not None:
            effect_cells = self._disk_world_cells_raw(
                tuple(int(value) for value in intent.impact_world_position),
                int(intent.effective_radius),
            )
        else:
            effect_cells = [self._buffer_to_world_position(cell) for cell in intent.effect_cells]
        effect_bounds = self._buffer_cell_bounds(effect_cells)
        generated_commands = [self._public_world_command(command) for command in intent.generated_commands]
        if intent.kind == "light":
            origin_world_position = (
                tuple(int(value) for value in intent.source_world_position)
                if intent.effect_shape == "beam" and intent.source_world_position is not None
                else None
                if intent.impact_world_position is None
                else tuple(int(value) for value in intent.impact_world_position)
            )
            if origin_world_position is not None:
                for command in generated_commands:
                    if command.kind == "inject_light":
                        command.payload["x"] = int(origin_world_position[0])
                        command.payload["y"] = int(origin_world_position[1])
        elif intent.kind == "force" and intent.source_world_position is not None:
            origin_world_position = tuple(int(value) for value in intent.source_world_position)
            for command in generated_commands:
                if command.kind == "inject_force":
                    command.payload["x"] = int(origin_world_position[0])
                    command.payload["y"] = int(origin_world_position[1])
        elif intent.kind in {"material", "gas"}:
            world_cells = (
                effect_cells
                if intent.effect_shape == "beam"
                else []
                if intent.impact_world_position is None
                else [tuple(int(value) for value in intent.impact_world_position)]
            )
            target_kind = "inject_material" if intent.kind == "material" else "inject_gas"
            rewritten = iter(world_cells)
            for command in generated_commands:
                if command.kind != target_kind:
                    continue
                try:
                    world_cell = next(rewritten)
                except StopIteration:
                    break
                command.payload["x"] = int(world_cell[0])
                command.payload["y"] = int(world_cell[1])
        return replace(
            intent,
            source_position=(
                None
                if intent.source_world_position is None
                else tuple(int(value) for value in intent.source_world_position)
            ),
            impact_position=(
                None
                if intent.impact_world_position is None
                else tuple(int(value) for value in intent.impact_world_position)
            ),
            effect_cells=effect_cells,
            effect_bounds=effect_bounds,
            generated_commands=generated_commands,
        )

    def _resolve_carrier_intent(
        self,
        intent: CarrierIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ResolvedCarrierIntent:
        return _resolve_carrier_intent(self, intent, resolved_targets)

    def _resolve_change_intent(
        self,
        intent: ChangeIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ResolvedChangeIntent:
        return _resolve_change_intent(self, intent, resolved_targets)

    def _resolve_change_intent_world_position(
        self,
        intent: ChangeIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[int, int] | None:
        return _resolve_change_intent_world_position(self, intent, resolved_targets)

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

    def _disk_world_cells(self, center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        return _disk_world_cells(self, center_world_position, radius)

    @staticmethod
    def _disk_world_cells_raw(center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        return _disk_world_cells_raw(center_world_position, radius)

    def _line_world_cells(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        return _line_world_cells(self, start_world_position, end_world_position)

    @staticmethod
    def _line_world_cells_raw(
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        return _line_world_cells_raw(start_world_position, end_world_position)

    def _capsule_world_cells(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
        radius: int,
    ) -> list[tuple[int, int]]:
        return _capsule_world_cells(self, start_world_position, end_world_position, radius)

    def _capsule_world_cells_raw(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
        radius: int,
    ) -> list[tuple[int, int]]:
        return _capsule_world_cells_raw(self, start_world_position, end_world_position, radius)

    @staticmethod
    def _buffer_cell_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        return _buffer_cell_bounds(cells)

    def _apply_change_stability_drift(
        self,
        intent_id: str,
        world_position: tuple[int, int],
        *,
        effective_radius: int,
        stability: float,
    ) -> tuple[int, int]:
        return _apply_change_stability_drift(self, intent_id, world_position, effective_radius=effective_radius, stability=stability)

    def _resolve_legal_world_position(
        self,
        world_position: tuple[int, int],
        *,
        require_empty: bool,
        fallback_mode: str,
        fallback_radius: int,
        effective_radius: int,
        source_world_position: tuple[int, int] | None,
    ) -> tuple[tuple[int, int] | None, bool, str | None]:
        return _resolve_legal_world_position(self, world_position, require_empty=require_empty, fallback_mode=fallback_mode, fallback_radius=fallback_radius, effective_radius=effective_radius, source_world_position=source_world_position)

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

    def _resolve_targeted_commands(
        self,
        commands: list[WorldCommand],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[WorldCommand]:
        resolved_commands: list[WorldCommand] = []
        for command in commands:
            payload = deepcopy(command.payload)
            target_query_id = payload.pop("target_query_id", None)
            target_dx = int(payload.pop("target_dx", 0))
            target_dy = int(payload.pop("target_dy", 0))
            if target_query_id is None:
                resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
                continue
            target = resolved_targets.get(str(target_query_id))
            fields = TARGETED_COMMAND_COORD_FIELDS.get(command.kind)
            if target is None or target.status != "resolved" or target.resolved_world_position is None or fields is None:
                continue
            world_x = int(target.resolved_world_position[0]) + target_dx
            world_y = int(target.resolved_world_position[1]) + target_dy
            x_field, y_field = fields
            if command.kind in {"request_readback", "advance_paging"}:
                payload[x_field] = int(world_x)
                payload[y_field] = int(world_y)
                payload["target_query_id"] = str(target_query_id)
                payload["target_dx"] = int(target_dx)
                payload["target_dy"] = int(target_dy)
            else:
                payload[x_field] = int(world_x)
                payload[y_field] = int(world_y)
                payload["resolved_target_query_id"] = str(target_query_id)
            resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
        return resolved_commands

    def _resolve_target_query(self, query: TargetQuery) -> ResolvedTarget:
        return _resolve_target_query(self, query)

    def _resolve_target_query_distance_cells(self, query: TargetQuery) -> int:
        return _resolve_target_query_distance_cells(self, query)

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
        directional_filters = [item for item in query.anchor_filters if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}]
        terrain_filters = [item for item in query.anchor_filters if item in TERRAIN_ANCHOR_FILTERS]
        entity_filters = [
            item
            for item in query.anchor_filters
            if item not in TERRAIN_ANCHOR_FILTERS
            and item not in IGNORED_ANCHOR_FILTERS
            and item not in CARDINAL_DIRECTION_VECTORS
            and item not in {"forward", "backward"}
        ]
        entity_anchor = None
        terrain_anchor = None
        if query.anchor_entity_id is not None or entity_filters or (directional_filters and not terrain_filters):
            entity_anchor = self._resolve_entity_anchor(
                query,
                source_world_position,
                direction_filter=directional_filters[0] if directional_filters else None,
            )
        if entity_anchor is not None:
            return entity_anchor
        if terrain_filters:
            terrain_anchor = self._resolve_terrain_anchor(
                source_world_position,
                terrain_filters,
                direction_filter=directional_filters[0] if directional_filters else None,
            )
        if terrain_anchor is not None:
            return terrain_anchor
        if query.anchor_entity_id is None and not query.anchor_filters:
            return {
                "kind": "source",
                "entity_id": query.source_entity_id,
                "buffer_position": self._world_to_buffer_clamped(*source_world_position),
                "world_position": source_world_position,
            }
        return None

    def _resolve_entity_anchor(
        self,
        query: TargetQuery,
        source_world_position: tuple[int, int],
        *,
        direction_filter: str | None,
    ) -> dict[str, Any] | None:
        return _resolve_entity_anchor(self, query, source_world_position, direction_filter=direction_filter)

    def _resolve_terrain_anchor(
        self,
        source_world_position: tuple[int, int],
        terrain_filters: list[str],
        *,
        direction_filter: str | None,
    ) -> dict[str, Any] | None:
        return _resolve_terrain_anchor(self, source_world_position, terrain_filters, direction_filter=direction_filter)

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

    def _terrain_cell_matches(self, x: int, y: int, terrain_filter: str) -> bool:
        return _terrain_cell_matches(self, x, y, terrain_filter)

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

    def _world_cell_is_solid_local(self, x: int, y: int) -> bool:
        return _world_cell_is_solid_local(self, x, y)

    def _world_cell_is_empty_local(self, x: int, y: int) -> bool:
        return _world_cell_is_empty_local(self, x, y)

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

    def _bounded_material_state_for_position(self, x: int, y: int) -> tuple[int, int]:
        return _bounded_material_state_for_position(self, x, y)

    def _matches_direction_filter(
        self,
        source_world_position: tuple[int, int],
        candidate_world_position: tuple[int, int],
        direction_name: str,
        *,
        source_entity_id: int | None,
    ) -> bool:
        return _matches_direction_filter(self, source_world_position, candidate_world_position, direction_name, source_entity_id=source_entity_id)

    def _query_direction_vector(
        self,
        query: TargetQuery,
        *,
        source_entity_id: int | None,
    ) -> tuple[int, int] | None:
        return _query_direction_vector(self, query, source_entity_id=source_entity_id)

    def _direction_vector(
        self,
        direction_name: str,
        *,
        source_entity_id: int | None,
    ) -> tuple[int, int] | None:
        return _direction_vector(self, direction_name, source_entity_id=source_entity_id)

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

    def _buffer_to_world_position(self, position: tuple[int, int]) -> tuple[int, int]:
        return _buffer_to_world_position(self, position)

    def _buffer_to_world_float_position(self, position: tuple[float, float]) -> tuple[float, float]:
        return _buffer_to_world_float_position(self, position)

    def _world_to_buffer_float_position(self, position: tuple[float, float]) -> tuple[float, float]:
        return _world_to_buffer_float_position(self, position)

    def _force_source_world_position(self, force_source: ForceSource) -> tuple[float, float]:
        return _force_source_world_position(self, force_source)

    def _force_source_buffer_position(self, force_source: ForceSource) -> tuple[float, float]:
        return _force_source_buffer_position(self, force_source)

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

    def _buffer_gas_to_world_position(self, position: tuple[int, int]) -> tuple[int, int]:
        return _buffer_gas_to_world_position(self, position)

    def _buffer_bbox_to_world_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return _buffer_bbox_to_world_bbox(self, bbox)

    def _clamped_world_window(self, world_x: int, world_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        return _clamped_world_window(self, world_x, world_y, width, height)

    def _centered_world_window(self, center_x: int, center_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        return _centered_world_window(self, center_x, center_y, width, height)

    def _world_axis_spans(
        self,
        world_start: int,
        world_end: int,
        *,
        axis: str,
        gas_grid: bool = False,
    ) -> list[tuple[int, int]]:
        return _world_axis_spans(self, world_start, world_end, axis=axis, gas_grid=gas_grid)

    def _world_axis_indices(self, world_start: int, world_end: int, *, axis: str, gas_grid: bool = False) -> np.ndarray:
        return _world_axis_indices(self, world_start, world_end, axis=axis, gas_grid=gas_grid)

    def _extract_world_window(
        self,
        array: np.ndarray,
        world_x0: int,
        world_y0: int,
        world_x1: int,
        world_y1: int,
        *,
        x_axis: int,
        y_axis: int,
        gas_grid: bool = False,
    ) -> np.ndarray:
        return _extract_world_window(self, array, world_x0, world_y0, world_x1, world_y1, x_axis=x_axis, y_axis=y_axis, gas_grid=gas_grid)

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

    def _pack_cell_core_world_window(self, world_x0: int, world_y0: int, world_x1: int, world_y1: int) -> np.ndarray:
        return _pack_cell_core_world_window(self, world_x0, world_y0, world_x1, world_y1)

    def _world_to_buffer_clamped(self, world_x: int, world_y: int) -> tuple[int, int]:
        return _world_to_buffer_clamped(self, world_x, world_y)

    def _clamp_world_position(self, world_x: int, world_y: int) -> tuple[int, int]:
        min_world_x = int(self.paging.origin_x)
        min_world_y = int(self.paging.origin_y)
        max_world_x = min_world_x + self.width - 1
        max_world_y = min_world_y + self.height - 1
        return (
            max(min_world_x, min(max_world_x, int(world_x))),
            max(min_world_y, min(max_world_y, int(world_y))),
        )

    def _find_nearest_empty_world_position(
        self,
        start_world_position: tuple[int, int],
        *,
        radius: int,
    ) -> tuple[int, int] | None:
        return _find_nearest_empty_world_position(self, start_world_position, radius=radius)

    def _world_cell_is_empty(self, world_x: int, world_y: int) -> bool:
        return _world_cell_is_empty(self, world_x, world_y)

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

    def _collect_observations(self, results: list[ReadbackResult]) -> dict[int, ObservationResult]:
        observations: dict[int, ObservationResult] = {}
        for result in results:
            observer_id = result.request.observer_id
            if observer_id is None:
                continue
            observations[observer_id] = ObservationResult(
                observer_id=observer_id,
                frame_id=result.frame_id,
                request=result.request,
                payload=result.payload,
            )
        return observations

    def _collect_entity_feedback(self, results: list[ReadbackResult]) -> dict[int, EntityFeedback]:
        feedback: dict[int, EntityFeedback] = {}
        for result in results:
            observer_id = result.request.observer_id
            if observer_id is None or observer_id in feedback:
                continue
            entity = self.entity_states.get(observer_id)
            if entity is None:
                continue
            entity_feedback = self._build_entity_feedback(result, entity)
            if entity_feedback is not None:
                feedback[observer_id] = entity_feedback
        return feedback

    def _build_entity_feedback(
        self,
        result: ReadbackResult,
        entity: EntityState,
    ) -> EntityFeedback | None:
        cell_payload = result.payload.get("cell")
        if cell_payload is None:
            return None
        core_words = cell_payload.get("core_words")
        entity_ids = cell_payload.get("entity_id")
        if core_words is None or entity_ids is None:
            return None

        origin_x, origin_y = (int(value) for value in cell_payload["origin"])
        width, height = (int(value) for value in cell_payload["size"])

        unpacked = unpack_cell_core(core_words)
        cells: list[EntityCellFeedback] = []
        base_world_x = int(entity.world_x) if entity.world_x is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
        base_world_y = int(entity.world_y) if entity.world_y is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
        for local_y in range(max(0, int(entity.height))):
            for local_x in range(max(0, int(entity.width))):
                world_x = base_world_x + local_x
                world_y = base_world_y + local_y
                lx = int(world_x) - origin_x
                ly = int(world_y) - origin_y
                if lx < 0 or ly < 0 or lx >= width or ly >= height:
                    continue
                material_id = int(unpacked["material_id"][ly, lx])
                phase = int(unpacked["phase"][ly, lx])
                integrity = float(unpacked["integrity"][ly, lx])
                occupant_entity_id = int(entity_ids[ly, lx])
                present = (
                    occupant_entity_id == entity.entity_id
                    and material_id > 0
                    and self._shadow_material_is_placeholder(material_id)
                )
                cells.append(
                    EntityCellFeedback(
                        x=int(world_x),
                        y=int(world_y),
                        present=present,
                        material_id=material_id,
                        phase=phase,
                        integrity=integrity,
                        entity_id=occupant_entity_id,
                    )
                )
        if not cells:
            return None
        bbox_xs = [cell.x for cell in cells]
        bbox_ys = [cell.y for cell in cells]
        return EntityFeedback(
            entity_id=entity.entity_id,
            bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
            cells=cells,
        )

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

    def _build_entity_feedback_from_state(
        self,
        entity: EntityState,
        *,
        cell_state: dict[str, np.ndarray],
        entity_runtime: dict[str, np.ndarray],
    ) -> EntityFeedback | None:
        cells: list[EntityCellFeedback] = []
        base_world_x = int(entity.world_x) if entity.world_x is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
        base_world_y = int(entity.world_y) if entity.world_y is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
        material_grid = cell_state["material_id"]
        phase_grid = cell_state["phase"]
        integrity_grid = cell_state["integrity"]
        entity_id_grid = entity_runtime["entity_id"]
        for local_y in range(max(0, int(entity.height))):
            for local_x in range(max(0, int(entity.width))):
                world_x = base_world_x + local_x
                world_y = base_world_y + local_y
                buffer_x, buffer_y = self._world_to_buffer_clamped(world_x, world_y)
                material_id = int(material_grid[buffer_y, buffer_x])
                phase = int(phase_grid[buffer_y, buffer_x])
                integrity = float(integrity_grid[buffer_y, buffer_x])
                occupant_entity_id = int(entity_id_grid[buffer_y, buffer_x])
                present = (
                    occupant_entity_id == entity.entity_id
                    and material_id > 0
                    and self._shadow_material_is_placeholder(material_id)
                )
                cells.append(
                    EntityCellFeedback(
                        x=int(world_x),
                        y=int(world_y),
                        present=present,
                        material_id=material_id,
                        phase=phase,
                        integrity=integrity,
                        entity_id=occupant_entity_id,
                    )
                )
        if not cells:
            return None
        bbox_xs = [cell.x for cell in cells]
        bbox_ys = [cell.y for cell in cells]
        return EntityFeedback(
            entity_id=int(entity.entity_id),
            bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
            cells=cells,
        )

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

    def _default_page_stripe_payload(self, update: PageStripeUpdate) -> dict[str, Any]:
        cell_span = self.paging.stripe_span(update)
        gas_span = cell_span // self.gas_cell_size
        cell_width = cell_span if update.axis == "x" else self.width
        cell_height = self.height if update.axis == "x" else cell_span
        gas_width = gas_span if update.axis == "x" else self.gas_width
        gas_height = self.gas_height if update.axis == "x" else gas_span
        light_count = self.cell_optical_dose.shape[0]
        gas_count = self.gas_concentration.shape[0]
        payload = {
            "meta": {
                "axis": update.axis,
                "world_start": update.world_start,
                "world_end": update.world_end,
                "buffer_start": update.buffer_start,
                "buffer_end": update.buffer_end,
                "kind": "generated",
            },
            "cell": {
                "material_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "phase": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "cell_flags": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "velocity": np.zeros((cell_height, cell_width, 2), dtype=np.float32),
                "cell_temperature": np.full((cell_height, cell_width), 20.0, dtype=np.float32),
                "timer_pack": np.zeros((cell_height, cell_width, 4), dtype=np.uint8),
                "integrity": np.zeros((cell_height, cell_width), dtype=np.float32),
                "island_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "placeholder_displaced_material": np.zeros((cell_height, cell_width), dtype=np.int32),
                "collapse_delay_pending": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "visible_illumination": np.zeros((cell_height, cell_width, 3), dtype=np.float32),
                "cell_optical_dose": np.zeros((light_count, cell_height, cell_width), dtype=np.float32),
            },
            "runtime": {
                "island_ids": np.zeros((0,), dtype=np.int32),
                "island_velocity": np.zeros((0, 2), dtype=np.float32),
                "island_subcell_offset": np.zeros((0, 2), dtype=np.float32),
                "entity_placeholder_entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
            },
            "gas": {
                "ambient_temperature": np.full((gas_height, gas_width), 20.0, dtype=np.float32),
                "flow_velocity": np.zeros((gas_height, gas_width, 2), dtype=np.float32),
                "pressure_ping": np.zeros((gas_height, gas_width), dtype=np.float32),
                "gas_concentration": np.zeros((gas_count, gas_height, gas_width), dtype=np.float32),
                "gas_optical_dose": np.zeros((light_count, gas_height, gas_width), dtype=np.float32),
            },
        }
        if 0 <= self.air_gas_species_id < gas_count:
            payload["gas"]["gas_concentration"][self.air_gas_species_id] = 1.0
        return payload

    def _stripe_buffer_ranges(self, update: PageStripeUpdate, *, gas_grid: bool) -> list[tuple[int, int]]:
        if not gas_grid:
            return self.paging.stripe_buffer_ranges(update)
        cell_span = self.paging.stripe_span(update)
        if cell_span % self.gas_cell_size != 0 or update.buffer_start % self.gas_cell_size != 0:
            raise ValueError("page stripe is not aligned to the gas grid")
        size = self.gas_width if update.axis == "x" else self.gas_height
        span = min(size, cell_span // self.gas_cell_size)
        if span <= 0:
            return []
        start = (update.buffer_start // self.gas_cell_size) % size
        if span >= size:
            return [(0, size)]
        end = (start + span) % size
        if start < end:
            return [(start, end)]
        ranges = [(start, size)]
        if end > 0:
            ranges.append((0, end))
        return ranges

    def _mark_loaded_page_stripe_active(self, update: PageStripeUpdate) -> None:
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                self._mark_active_rect_runtime(start, 0, end, self.height, tile_padding=1)
            else:
                self._mark_active_rect_runtime(0, start, self.width, end, tile_padding=1)

    def _rebuild_sparse_runtime_indexes(self) -> None:
        _rebuild_sparse_runtime_indexes(self)

    def _rebuild_entity_placeholder_index(self) -> None:
        _rebuild_entity_placeholder_index(self)

    def _normalize_cell_runtime_arrays(
        self,
        material_id: np.ndarray,
        phase: np.ndarray,
        island_id: np.ndarray,
        entity_id: np.ndarray,
        placeholder_displaced_material: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _normalize_cell_runtime_arrays(self, material_id, phase, island_id, entity_id, placeholder_displaced_material)

    def _normalize_page_stripe_cell_runtime(self, update: PageStripeUpdate) -> None:
        _normalize_page_stripe_cell_runtime(self, update)

    def _capture_page_stripe_entity_placeholder_runtime(
        self,
        update: PageStripeUpdate,
        *,
        stripe_axis: int,
    ) -> np.ndarray:
        return _capture_page_stripe_entity_placeholder_runtime(self, update, stripe_axis=stripe_axis)

    def _apply_page_stripe_entity_placeholder_runtime(
        self,
        update: PageStripeUpdate,
        entity_placeholder_entity_id: np.ndarray | None,
    ) -> None:
        _apply_page_stripe_entity_placeholder_runtime(self, update, entity_placeholder_entity_id)

    def _rebuild_island_records(self) -> None:
        _rebuild_island_records(self)

    def _capture_page_stripe_island_runtime(self, stripe_island_ids: np.ndarray) -> dict[str, np.ndarray]:
        return _capture_page_stripe_island_runtime(self, stripe_island_ids)

    def _page_stripe_island_bboxes_from_payload(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
    ) -> dict[int, tuple[int, int, int, int]] | None:
        cell_payload = payload.get("cell", {})
        try:
            material_id = np.asarray(cell_payload["material_id"], dtype=np.int32)
            phase = np.asarray(cell_payload["phase"], dtype=np.uint8)
            island_id = np.asarray(cell_payload["island_id"], dtype=np.int32)
        except KeyError:
            return None
        if material_id.shape != phase.shape or material_id.shape != island_id.shape:
            return None
        valid = (island_id > 0) & (material_id > 0) & (phase == int(Phase.FALLING_ISLAND))
        if not np.any(valid):
            return {}
        boxes: dict[int, list[int]] = {}
        offset = 0
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            span = int(end) - int(start)
            if span <= 0:
                continue
            if update.axis == "x":
                stripe = valid[:, offset : offset + span]
                stripe_ids = island_id[:, offset : offset + span]
                ys, xs = np.nonzero(stripe)
                for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                    current_id = int(stripe_ids[local_y, local_x])
                    x = int(start) + int(local_x)
                    y = int(local_y)
                    box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)
            else:
                stripe = valid[offset : offset + span, :]
                stripe_ids = island_id[offset : offset + span, :]
                ys, xs = np.nonzero(stripe)
                for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                    current_id = int(stripe_ids[local_y, local_x])
                    x = int(local_x)
                    y = int(start) + int(local_y)
                    box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)
            offset += span
        return {island_id: tuple(box) for island_id, box in boxes.items()}

    def _merge_island_runtime_payload(
        self,
        runtime_payload: dict[str, Any] | None,
        *,
        update: PageStripeUpdate | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        _merge_island_runtime_payload(self, runtime_payload, update=update, payload=payload)

    def _rebuild_material_property_arrays(self) -> None:
        _rebuild_material_property_arrays(self)

    def _rebuild_gas_property_arrays(self) -> None:
        _rebuild_gas_property_arrays(self)

    def _rebuild_light_property_arrays(self) -> None:
        _rebuild_light_property_arrays(self)

    def _cell_participates_in_collapse(self, material_id: int, phase: int) -> bool:
        return _cell_participates_in_collapse(self, material_id, phase)

    def _mark_collapse_dirty_rect(self, x0: int, y0: int, x1: int, y1: int, *, pad: int = 8) -> None:
        _mark_collapse_dirty_rect(self, x0, y0, x1, y1, pad=pad)

    def _drain_gpu_collapse_structure_dirty_tiles(self) -> None:
        _drain_gpu_collapse_structure_dirty_tiles(self)

    def _paint_material(self, x: int, y: int, material: str, radius: int) -> None:
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        self.set_material_by_mask(mask, material)

    def _write_material_region_immediate(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        material: str,
    ) -> None:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, int(x) + max(0, int(width)))
        y1 = min(self.height, int(y) + max(0, int(height)))
        if x0 >= x1 or y0 >= y1:
            return
        for write_y in range(y0, y1):
            for write_x in range(x0, x1):
                self.set_cell(write_x, write_y, material)

    def _build_demo_scene(self) -> None:
        active_w = int(self.paging.active_width)
        active_h = int(self.paging.active_height)
        floor_y = max(0, active_h - 28)
        self._fill_rect(0, floor_y, active_w, 28, "raw_stone_solid")
        self._fill_rect(32, floor_y - 58, 160, 14, "sandstone_solid")
        self._fill_rect(60, floor_y - 112, 118, 54, "sand_powder")
        self._fill_rect(230, floor_y - 24, 112, 24, "water_liquid")
        self._fill_rect(374, floor_y - 18, 78, 18, "oil_liquid")
        self._fill_rect(500, floor_y - 86, 12, 86, "raw_stone_solid")
        self._fill_rect(520, floor_y - 46, 130, 18, "sandstone_solid")
        self._fill_rect(550, floor_y - 86, 76, 40, "gravel_powder")
        self._fill_rect(690, floor_y - 140, 72, 16, "log_solid")
        self._fill_rect(700, floor_y - 188, 12, 48, "root_solid")

    def _fill_rect(self, x: int, y: int, width: int, height: int, material: str) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        material_id = self._resolve_sanctioned_material_id(material)
        if material_id <= 0:
            raise KeyError(material)
        phase = int(self.material_default_phase[material_id]) if material_id < self.material_default_phase.shape[0] else 0
        integrity = (
            float(self.material_base_integrity[material_id])
            if material_id < self.material_base_integrity.shape[0]
            else 0.0
        )
        self.material_id[y0:y1, x0:x1] = int(material_id)
        self.phase[y0:y1, x0:x1] = phase
        self.cell_flags[y0:y1, x0:x1] = 0
        self.velocity[y0:y1, x0:x1] = 0.0
        self.timer_pack[y0:y1, x0:x1] = 0
        self.integrity[y0:y1, x0:x1] = integrity
        self.island_id[y0:y1, x0:x1] = 0
        self.entity_id[y0:y1, x0:x1] = 0
        self.placeholder_displaced_material[y0:y1, x0:x1] = 0
        if material_id < self.material_spawn_temperature.shape[0]:
            spawn_temperature = float(self.material_spawn_temperature[material_id])
            if np.isfinite(spawn_temperature):
                self.cell_temperature[y0:y1, x0:x1] = np.maximum(
                    self.cell_temperature[y0:y1, x0:x1],
                    spawn_temperature,
                )
