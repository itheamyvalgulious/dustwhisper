"""Shared setup context for engine-capabilities serialization.

Computes every local value the section builders reference so the section
builders can stay verbatim-faithful to the original monolithic function.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from copy import deepcopy
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.world_constants import (
    BASE_MATERIAL_RUNTIME_ALIASES,
    PUBLIC_WORLD_COMMAND_KINDS,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def build_capabilities_context(engine: "WorldEngine") -> "SimpleNamespace":
    """Compute the shared local values used by every capabilities section builder."""

    material_payload = engine._shadow_material_payload()
    gas_payload = engine._shadow_gas_species_payload()
    light_payload = engine._shadow_light_type_payload()
    material_name_aliases = deepcopy(BASE_MATERIAL_RUNTIME_ALIASES)
    force_source_fields = ["x", "y", "direction", "radius", "strength", "lifetime"]
    emitter_fields = ["x", "y", "light_type", "strength", "radius", "direction", "spread"]
    entity_state_fields = [
        "entity_id",
        "x",
        "y",
        "width",
        "height",
        "velocity_xy",
        "facing_xy",
        "placeholder_material",
        "tags",
        "observe_channels",
        "observe_pad_cells",
        "observe_width",
        "observe_height",
        "observe_label",
    ]
    entity_state_patch_fields = ["entity_id", "fields"]
    entity_observation_spec_fields = [
        "entity_id",
        "observe_channels",
        "observe_pad_cells",
        "observe_width",
        "observe_height",
        "observe_label",
    ]
    entity_placeholder_fields = ["entity_id", "x", "y", "width", "height", "material"]
    target_query_fields = [
        "query_id",
        "anchor_filters",
        "source_entity_id",
        "source_x",
        "source_y",
        "anchor_entity_id",
        "direction",
        "distance_cells",
        "distance_meters",
        "distance_hint",
        "require_empty",
        "search_radius",
        "label",
    ]
    target_query_overlay_fields = ["target_query_id", "target_dx", "target_dy"]
    inline_target_query_optional_fields = [*target_query_overlay_fields, "target_queries"]
    readback_request_fields = [
        "request_id",
        "center_x",
        "center_y",
        "width",
        "height",
        "channels",
        "observer_id",
        "label",
        "target_query_id",
        "target_dx",
        "target_dy",
    ]
    observation_target_fields = [
        "observer_id",
        "channels",
        "center_x",
        "center_y",
        "width",
        "height",
        "entity_id",
        "pad_cells",
        "label",
        "target_query_id",
        "target_dx",
        "target_dy",
    ]
    change_intent_fields = [
        "intent_id",
        "target_query_id",
        "center_x",
        "center_y",
        "target_dx",
        "target_dy",
        "radius",
        "material",
        "temperature_delta",
        "velocity",
        "velocity_carrier",
        "velocity_mode",
        "require_empty",
        "fallback_mode",
        "fallback_radius",
        "potency",
        "stability",
        "label",
    ]
    carrier_intent_fields = [
        "intent_id",
        "kind",
        "target_query_id",
        "center_x",
        "center_y",
        "source_entity_id",
        "source_x",
        "source_y",
        "target_dx",
        "target_dy",
        "radius",
        "material",
        "gas_species",
        "gas_amount",
        "light_type",
        "light_strength",
        "light_spread",
        "force_radius",
        "force_strength",
        "force_lifetime",
        "release_mode",
        "require_empty",
        "fallback_mode",
        "fallback_radius",
        "potency",
        "stability",
        "label",
    ]
    material_fields = [
        "material_id",
        "name",
        "display_name",
        "default_phase",
        "render_group",
        "base_color",
        "density",
        "gravity_scale",
        "wind_coupling",
        "drag_scale",
        "friction",
        "elasticity",
        "max_dda_step",
        "powder_solver_kind",
        "liquid_solver_kind",
        "falling_island_break_kind",
        "is_structural",
        "is_support_anchor",
        "collapse_behavior",
        "collapse_generation",
        "powder_generation",
        "base_integrity",
        "heat_capacity",
        "conductivity",
        "ambient_exchange_rate",
        "melt_point",
        "boil_point",
        "melt_to_material",
        "freeze_to_material",
        "boil_to_gas_species",
        "spawn_temperature",
        "material_tag_mask",
        "gas_tag_mask",
        "light_tag_mask",
        "reaction_slots",
        "tags",
    ]
    gas_fields = [
        "species_id",
        "name",
        "display_name",
        "color",
        "diffusion_rate",
        "buoyancy",
        "decay_rate",
        "temperature_coupling",
        "condense_point",
        "condense_to_material",
        "pressure_factor",
        "density_factor",
        "material_reaction_tag_mask",
        "light_reaction_tag_mask",
    ]
    light_fields = [
        "light_type_id",
        "name",
        "display_name",
        "color",
        "visual_channel",
        "default_range",
        "max_bounce",
        "dose_channel_id",
        "render_style",
    ]
    optics_fields = ["material_name", "light_type", "absorption", "scattering", "refraction"]
    reaction_action_fields = [
        "reaction_type",
        "target_material",
        "emit_material",
        "light_type",
        "gas_species",
        "duration",
        "speed",
        "velocity",
        "direction",
        "strength",
        "beam_width",
        "range_cells",
        "delta",
        "value",
        "generation",
        "harm_per_frame",
        "integrity_threshold",
        "allow_subunit_scale",
    ]
    reaction_table_rule_sets = [
        "material_material",
        "material_gas",
        "material_light",
        "gas_gas",
        "gas_light",
        "self_rules",
    ]
    pair_reaction_rule_fields = [
        "lhs_material",
        "lhs_gas",
        "rhs_material",
        "rhs_gas",
        "rhs_light",
        "lhs_tag_mask",
        "rhs_tag_mask",
        "phases",
        "min_temperature",
        "max_temperature",
        "threshold",
        "rate",
        "consume_policy",
        "result_action",
        "trigger_slot_index",
    ]
    self_reaction_rule_fields = [
        "material",
        "trigger_slot_index",
        "phases",
        "min_temperature",
        "max_temperature",
        "timer_index",
        "integrity_at_most",
        "integrity_at_least",
    ]
    cell_window_fields = [
        "origin",
        "size",
        "material_id",
        "phase",
        "cell_flags",
        "velocity",
        "cell_temperature",
        "temperature",
        "timer_pack",
        "integrity",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "collapse_delay_pending",
    ]
    temperature_window_fields = ["temperature"]
    gas_window_fields = ["species", "concentration"]
    pressure_window_fields = ["pressure"]
    velocity_window_fields = ["velocity"]
    optics_window_fields = ["origin", "size", "gas_origin", "gas_size", "visible_illumination", "cell_dose", "gas_dose"]
    visible_illumination_fields = ["illumination"]
    debug_frame_fields = ["view", "origin", "size", "gas_species", "light_type", "frame"]
    bridge_runtime_fields = [
        "frame_id",
        "bridge",
        "pending_readbacks",
        "inflight_readbacks",
        "ready_readbacks",
        "pending_commands",
    ]
    bridge_resource_catalog_fields = ["typed_tables", "shadow_buffers", "snapshots"]
    bridge_typed_table_fields = ["name", "shape", "dtype", "structured", "field_names", "row_count", "rows"]
    bridge_typed_table_slice_fields = [
        "name",
        "shape",
        "dtype",
        "structured",
        "field_names",
        "row_count",
        "offset",
        "limit",
        "returned_count",
        "slice_shape",
        "rows",
    ]
    bridge_shadow_buffer_fields = ["name", "shape", "dtype", "structured", "field_names", "row_count", "rows", "values", "utf8"]
    bridge_shadow_buffer_slice_fields = [
        "name",
        "shape",
        "dtype",
        "structured",
        "field_names",
        "row_count",
        "offset",
        "limit",
        "returned_count",
        "slice_shape",
        "rows",
        "values",
        "utf8",
    ]
    bridge_shadow_buffer_window_fields = [
        "name",
        "shape",
        "dtype",
        "structured",
        "field_names",
        "row_count",
        "window_origin",
        "requested_size",
        "window_size",
        "window_axes",
        "returned_shape",
        "rows",
        "values",
        "utf8",
    ]
    bridge_shadow_buffer_spatial_window_fields = [
        "name",
        "shape",
        "dtype",
        "structured",
        "field_names",
        "row_count",
        "coord_space",
        "window_origin",
        "requested_size",
        "window_size",
        "window_axes",
        "returned_shape",
        "rows",
        "values",
        "utf8",
    ]
    bridge_upload_snapshot_fields = ["frame_meta", "world_commands", "readback_requests", "page_stripes"]
    bridge_readback_stage_fields = ["request_id", "stage"]
    bridge_index_stage_fields = ["index", "stage"]
    world_command_material_alias_fields_by_kind = {
        "inject_material": ["material"],
        "write_material_region": ["material"],
        "sync_entity_states": ["entities[].placeholder_material"],
        "patch_entity_states": ["patches[].fields.placeholder_material"],
    }
    world_command_collection_item_types_by_kind = {
        "sync_entity_states": {"entities": "entity_state"},
        "patch_entity_states": {"patches": "entity_state_patch"},
        "sync_entity_observation_specs": {"observations": "entity_observation_spec"},
    }
    bridge_frame_snapshot_fields = [
        "prepared",
        "commands",
        "command_stages",
        "readback_requests",
        "readback_request_stages",
        "placeholders",
        "placeholder_stages",
        "placeholder_dirty_rects",
        "paging_updates",
        "paging_update_stages",
        "page_stripes",
        "page_stripe_stages",
    ]
    readback_state_fields = [
        "queued",
        "queued_commands",
        "pending",
        "pending_requests",
        "inflight",
        "inflight_requests",
        "ready",
    ]
    readback_result_fields = ["frame_id", "request", "payload"]
    readback_poll_fields = ["ready", "status", "result"]
    readback_plan_fields = ["request", "layout", "nbytes", "gpu_source_count", "cpu_chunk_count", "payload"]
    observation_plan_fields = ["target", *readback_plan_fields]
    readback_channel_payload_types = {
        "cell": "readback_cell_payload",
        "ambient_temperature": "readback_scalar_window",
        "pressure": "readback_scalar_window",
        "velocity": "readback_vector_window",
        "optics": "readback_optics_payload",
        "gas": "readback_gas_payload",
    }
    readback_payload_fields = list(READBACK_ALLOWED_CHANNELS)
    readback_cell_payload_fields = [
        "origin",
        "size",
        "core_words",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "collapse_delay_pending",
    ]
    readback_scalar_window_fields = ["origin", "size", "grid", "values"]
    readback_vector_window_fields = ["origin", "size", "grid", "values"]
    readback_gas_payload_fields = ["origin", "size", "grid", "species"]
    readback_optics_payload_fields = [
        "origin",
        "size",
        "gas_origin",
        "gas_size",
        "visible_illumination",
        "cell_dose",
        "gas_dose",
    ]
    resolved_target_fields = [
        "query_id",
        "status",
        "anchor_filters",
        "direction",
        "distance_cells",
        "distance_meters",
        "distance_hint",
        "label",
        "source_position",
        "source_world_position",
        "anchor_kind",
        "anchor_entity_id",
        "anchor_position",
        "anchor_world_position",
        "resolved_position",
        "resolved_world_position",
        "note",
    ]
    resolved_change_intent_fields = [
        "intent_id",
        "status",
        "target_query_id",
        "label",
        "potency",
        "stability",
        "center_position",
        "center_world_position",
        "effective_radius",
        "material",
        "temperature_delta",
        "velocity",
        "velocity_carrier",
        "velocity_mode",
        "require_empty",
        "fallback_mode",
        "fallback_applied",
        "effect_shape",
        "effect_cells",
        "effect_bounds",
        "generated_commands",
        "note",
    ]
    resolved_carrier_intent_fields = [
        "intent_id",
        "status",
        "kind",
        "target_query_id",
        "label",
        "release_mode",
        "potency",
        "stability",
        "source_position",
        "source_world_position",
        "impact_position",
        "impact_world_position",
        "effective_radius",
        "material",
        "gas_species",
        "gas_amount",
        "light_type",
        "light_strength",
        "light_spread",
        "force_radius",
        "force_strength",
        "force_lifetime",
        "direction",
        "require_empty",
        "fallback_mode",
        "fallback_applied",
        "effect_shape",
        "effect_cells",
        "effect_bounds",
        "generated_commands",
        "note",
    ]
    observation_result_fields = ["observer_id", "frame_id", "request", "payload"]
    emitter_runtime_fields = ["persistent_emitters", "queued_emitters"]
    paging_state_fields = ["origin", "buffer_origin", "active_bounds", "buffer_size", "active_size", "stored_stripes"]
    pending_commands_fields = ["pending", "commands"]
    entity_observation_runtime_fields = ["observations", "targets", "requests"]
    entity_observation_consume_result_fields = [
        "frame_id",
        "consumed",
        "consumed_readbacks",
        "observations",
        "entity_feedback",
    ]
    control_state_fields = ["paused", "speed", "single_step"]
    readback_status_fields = ["request_id", "status"]
    readback_cancel_request_fields = ["request_id"]
    readback_cancel_result_fields = ["ok", "request_id", "status"]
    frame_pending_state_fields = ["pending", "submission_ids"]
    frame_state_fields = [
        "pending",
        "pending_submission_ids",
        "ready",
        "ready_submission_ids",
        "canceled_submission_ids",
    ]
    frame_submission_status_fields = ["submission_id", "status"]
    frame_cancel_request_fields = ["submission_id"]
    frame_cancel_result_fields = ["ok", "submission_id", "status", "pending_frames"]
    frame_cancel_all_result_fields = ["ok", "canceled_submission_ids", "pending_frames"]
    deferred_readback_request_result_fields = ["ok", "queued", "pending_commands", "request_id"]
    deferred_observation_request_result_fields = ["ok", "queued", "pending_commands", "request_id"]
    deferred_frame_submit_ack_fields = ["ok", "queued", "pending_frames", "submission_id"]
    deferred_controller_state_result_fields = ["ok", "queued", "pending_frames", "submission_id"]
    material_fill_request_fields = [
        "x",
        "y",
        "width",
        "height",
        "material",
        "immediate",
        *inline_target_query_optional_fields,
    ]
    paging_focus_request_fields = ["x", "y", *inline_target_query_optional_fields]
    paging_focus_result_fields = ["ok", "queued", "pending_commands", "target_center"]
    page_store_has_result_fields = ["stored"]
    page_store_apply_result_fields = ["ok", "stored", "queued", "pending_commands"]
    page_store_clear_result_fields = ["ok", "cleared", "stored_stripes"]
    page_stripe_apply_result_fields = ["ok", "queued", "pending_commands"]
    speed_request_fields = ["speed"]
    pause_result_fields = ["paused"]
    resume_result_fields = ["paused"]
    step_result_fields = ["single_step"]
    speed_result_fields = ["speed"]
    queued_mutation_result_fields = ["ok", "queued", "pending_commands"]
    control_reset_result_fields = ["ok", "queued", "pending_commands"]
    material_injector_fields = ["x", "y", "material", "radius", "immediate", *inline_target_query_optional_fields]
    temperature_injector_fields = ["x", "y", "delta", "radius", "immediate", *inline_target_query_optional_fields]
    velocity_injector_fields = [
        "x",
        "y",
        "velocity",
        "radius",
        "carrier",
        "mode",
        "immediate",
        *inline_target_query_optional_fields,
    ]
    gas_injector_fields = ["x", "y", "species", "amount", "radius", "immediate", *inline_target_query_optional_fields]
    force_injector_fields = [*force_source_fields, "immediate", *inline_target_query_optional_fields]
    light_injector_fields = [*emitter_fields, "immediate", *inline_target_query_optional_fields]
    material_patch_fields = ["name", "fields"]
    gas_patch_fields = ["name", "fields"]
    light_patch_fields = ["name", "fields"]
    optics_patch_fields = ["material_name", "light_type", "fields"]
    reaction_action_patch_fields = ["index", "fields"]
    reaction_action_delete_request_fields = ["index"]
    reaction_rule_patch_fields = ["rule_set", "index", "fields"]
    reaction_rule_delete_request_fields = ["rule_set", "index"]
    materials_table_fields = ["materials"]
    gases_table_fields = ["gases"]
    lights_table_fields = ["lights"]
    optics_table_fields = ["optics"]
    reactions_table_fields = ["actions", "rules"]
    gas_species_runtime_fields = ["species_id", "species", "total_concentration", "active_concentration"]
    heat_phase_target_fields = ["x", "y", "target_material_id", "target_material"]
    heat_boil_target_fields = ["x", "y", "target_species_id", "target_species"]
    heat_condense_target_fields = [
        "gas_x",
        "gas_y",
        "species_id",
        "species",
        "target_material_id",
        "target_material",
    ]
    collapse_component_fields = ["island_id", "bbox", "cell_count"]
    pending_displaced_cell_fields = ["x", "y", "material_id"]
    powder_reservation_fields = [
        "source_xy",
        "desired_target_xy",
        "reserved_target_xy",
        "resolved_target_xy",
        "velocity_xy",
        "material_id",
        "resolve_state",
    ]
    island_reservation_fields = [
        "island_id",
        "world_bbox",
        "velocity_xy",
        "subcell_offset",
        "target_shift",
        "reserved_shift",
        "resolved_shift",
        "resolve_state",
    ]
    gas_runtime_fields = [
        "backend",
        "pressure_iterations",
        "tile_grid_size",
        "gas_grid_size",
        "solve_tile_count",
        "solve_gas_count",
        "solve_tile_mask",
        "solve_gas_mask",
        "force_source_count_before",
        "force_source_count_after",
        "velocity_changed",
        "ambient_changed",
        "gas_changed",
        "pressure_range",
        "ambient_range",
        "flow_speed_range",
        "species_runtime",
    ]
    heat_runtime_fields = [
        "backend",
        "ambient_iterations",
        "tile_grid_size",
        "cell_grid_size",
        "gas_grid_size",
        "solve_tile_count",
        "solve_cell_count",
        "solve_gas_count",
        "phase_target_count",
        "boil_target_count",
        "condense_target_count",
        "solve_tile_mask",
        "solve_cell_mask",
        "solve_gas_mask",
        "cell_changed",
        "ambient_changed",
        "material_changed",
        "phase_changed",
        "integrity_changed",
        "gas_changed",
        "cell_temperature_range",
        "ambient_temperature_range",
        "integrity_range",
        "phase_targets",
        "boil_targets",
        "condense_targets",
    ]
    liquid_runtime_fields = [
        "backend",
        "tile_grid_size",
        "cell_grid_size",
        "solve_tile_count",
        "post_tile_count",
        "post_cell_count",
        "vertical_seam_cell_count",
        "horizontal_seam_cell_count",
        "buoyancy_candidate_count",
        "changed_cell_count",
        "material_changed",
        "phase_changed",
        "velocity_changed",
        "temperature_changed",
        "integrity_changed",
        "placeholder_changed",
        "pending_placeholder_count_before",
        "pending_placeholder_count_after",
        "liquid_cell_count_before",
        "liquid_cell_count_after",
        "solve_tile_mask",
        "post_tile_mask",
        "post_cell_mask",
        "vertical_seam_mask",
        "horizontal_seam_mask",
        "buoyancy_mask",
        "changed_cell_mask",
    ]
    reaction_runtime_fields = [
        "backend",
        "tile_grid_size",
        "cell_grid_size",
        "gas_grid_size",
        "solve_tile_count",
        "solve_cell_count",
        "solve_gas_count",
        "changed_cell_count",
        "changed_gas_count",
        "ambient_changed_count",
        "timer_changed_count",
        "emitted_light_count",
        "emitted_material_count",
        "executed_action_count",
        "emit_light_action_count",
        "emit_material_action_count",
        "modify_gas_action_count",
        "convert_material_action_count",
        "modify_temperature_action_count",
        "harm_action_count",
        "stage_action_counts",
        "stage_tile_masks",
        "solve_cell_mask",
        "solve_gas_mask",
        "changed_cell_mask",
        "changed_gas_mask",
        "ambient_changed_mask",
        "timer_changed_mask",
        "emitted_light_mask",
        "emitted_material_mask",
    ]
    collapse_runtime_fields = [
        "backend",
        "cell_grid_size",
        "dirty_region_count_before",
        "solve_region_count",
        "solve_region_cell_count",
        "structural_cell_count",
        "support_seed_count",
        "supported_cell_count",
        "unsupported_cell_count",
        "delayed_pending_count",
        "immune_unsupported_count",
        "collapsed_cell_count",
        "collapsed_component_count",
        "solve_region_mask",
        "structural_mask",
        "support_seed_mask",
        "supported_mask",
        "unsupported_mask",
        "delayed_pending_mask",
        "immune_unsupported_mask",
        "collapsed_cell_mask",
        "collapsed_components",
    ]
    optics_runtime_fields = [
        "backend",
        "tile_grid_size",
        "cell_grid_size",
        "gas_grid_size",
        "emitter_count",
        "secondary_branch_count",
        "solve_tile_count",
        "solve_cell_count",
        "solve_gas_count",
        "visible_changed_count",
        "cell_dose_changed_count",
        "gas_dose_changed_count",
        "visible_energy_total",
        "cell_dose_total",
        "gas_dose_total",
        "emitters",
        "solve_tile_mask",
        "solve_cell_mask",
        "solve_gas_mask",
        "visible_changed_mask",
        "cell_dose_changed_mask",
        "gas_dose_changed_mask",
        "emitter_origin_mask",
    ]
    active_runtime_fields = [
        "tile_size",
        "chunk_tiles",
        "active_ttl_reset",
        "tile_grid_size",
        "chunk_grid_size",
        "active_tile_count",
        "active_chunk_count",
        "active_tile_ttl",
        "active_chunk_mask",
        "pending_displaced_count",
        "pending_displaced_cells",
    ]
    motion_runtime_fields = ["backend", "powder_reservation_count", "island_reservation_count", "powder_reservations", "island_reservations"]
    cell_core_layout_fields = ["word0", "word1", "word2", "word3", "word4"]
    cell_core_unpack_fields = [
        "material_id",
        "phase",
        "cell_flags",
        "velocity",
        "cell_temperature",
        "timer_pack",
        "integrity",
    ]
    public_world_command_kinds = [*PUBLIC_WORLD_COMMAND_KINDS]
    frame_input_field_order = [
        "submission_id",
        "focus_center",
        "controller_state",
        "controller_state_provided",
        "entities",
        "entity_placeholders",
        "force_sources",
        "emitters",
        "target_queries",
        "change_intents",
        "carrier_intents",
        "observation_targets",
        "readback_requests",
        "commands",
    ]
    pending_frame_detail_field_order = [*frame_input_field_order, "preview"]
    frame_preview_field_order = [
        "controller_state",
        "resolved_targets",
        "resolved_change_intents",
        "resolved_carrier_intents",
        "resolved_commands",
        "observation_requests",
        "observation_plans",
        "readback_requests",
        "readback_plans",
        "bridge_frame_snapshot",
        "paging_updates",
        "placeholder_count",
    ]
    frame_output_field_order = [
        "frame_id",
        "submission_id",
        "controller_state",
        "consumed_readbacks",
        "resolved_targets",
        "resolved_change_intents",
        "resolved_carrier_intents",
        "observations",
        "entity_feedback",
        "paging_updates",
        "observation_plans",
        "readback_plans",
        "bridge_upload_snapshot",
        "bridge_frame_snapshot",
        "queued_observations",
        "queued_readbacks",
        "queued_commands",
        "placeholder_count",
    ]
    allowed_gas_species = [
        str(item["name"])
        for item in gas_payload
        if item.get("name")
    ]
    allowed_lights = [
        str(item["name"])
        for item in light_payload
        if item.get("name")
    ]
    debug_view_parameterized_views = {
        "gas": {
            "query_fields": ["gas_species"],
            "default_gas_species": "water_gas",
            "allowed_gas_species": allowed_gas_species,
        },
        "optics": {
            "query_fields": ["light"],
            "light_query_maps_to": "light_type",
            "light_optional": True,
            "omitted_light_means": "all_lights",
            "allowed_lights": allowed_lights,
        },
        "light": {
            "query_fields": ["light"],
            "light_query_maps_to": "light_type",
            "light_optional": True,
            "omitted_light_means": "all_lights",
            "allowed_lights": allowed_lights,
        },
    }
    debug_view_options = {
        "default": engine.default_debug_view.value,
        "frame_endpoint": "/api/read/debug_frame",
        "parameterized_views": debug_view_parameterized_views,
    }
    gas_read_options = {
        "species_query_field": "species",
        "default_species": "water_gas",
        "allowed_species": allowed_gas_species,
    }
    optics_read_options = {
        "light_query_field": "light",
        "light_query_maps_to": "light_type",
        "light_optional": True,
        "omitted_light_means": "all_lights",
        "allowed_lights": allowed_lights,
    }
    # Capture every local computed above (minus the engine parameter) into a
    # namespace so section builders can unpack the names they need verbatim.
    _locals_snapshot = dict(locals())
    _ns = SimpleNamespace()
    for _name, _value in _locals_snapshot.items():
        if _name != "engine":
            setattr(_ns, _name, _value)
    return _ns
