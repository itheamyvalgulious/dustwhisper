from __future__ import annotations

import numpy as np



ENTITY_STATE_DTYPE = np.dtype(
    [
        ("entity_id", "<i4"),
        ("buffer_x", "<i4"),
        ("buffer_y", "<i4"),
        ("world_x", "<i4"),
        ("world_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("placeholder_material_id", "<i4"),
        ("velocity_xy", "<f4", (2,)),
    ]
)


def entity_state_dtype() -> np.dtype:
    return ENTITY_STATE_DTYPE


FORCE_SOURCE_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("world_x", "<f4"),
        ("world_y", "<f4"),
        ("direction_xy", "<f4", (2,)),
        ("radius", "<f4"),
        ("strength", "<f4"),
        ("lifetime", "<f4"),
    ]
)


def force_source_dtype() -> np.dtype:
    return FORCE_SOURCE_DTYPE


ISLAND_RUNTIME_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("buffer_bbox", "<i4", (4,)),
        ("world_bbox", "<i4", (4,)),
        ("velocity_xy", "<f4", (2,)),
        ("subcell_offset", "<f4", (2,)),
    ]
)


def island_runtime_dtype() -> np.dtype:
    return ISLAND_RUNTIME_DTYPE


FRAME_META_DTYPE = np.dtype(
    [
        ("frame_id", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("gas_width", "<i4"),
        ("gas_height", "<i4"),
        ("origin_x", "<i4"),
        ("origin_y", "<i4"),
        ("buffer_origin_x", "<i4"),
        ("buffer_origin_y", "<i4"),
        ("active_width", "<i4"),
        ("active_height", "<i4"),
        ("entity_count", "<i4"),
        ("force_source_count", "<i4"),
        ("world_command_count", "<i4"),
        ("readback_request_count", "<i4"),
        ("placeholder_count", "<i4"),
        ("placeholder_dirty_rect_count", "<i4"),
        ("active_tile_count", "<i4"),
        ("active_chunk_count", "<i4"),
        ("page_update_count", "<i4"),
        ("page_stripe_section_count", "<i4"),
    ]
)


def frame_meta_dtype() -> np.dtype:
    return FRAME_META_DTYPE


WORLD_COMMAND_DTYPE = np.dtype(
    [
        ("kind_id", "<i4"),
        ("payload_offset", "<i4"),
        ("payload_length", "<i4"),
    ]
)


def world_command_dtype() -> np.dtype:
    return WORLD_COMMAND_DTYPE


READBACK_REQUEST_DTYPE = np.dtype(
    [
        ("request_id", "<i4"),
        ("center_x", "<i4"),
        ("center_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("channels_mask", "<i4"),
        ("observer_id", "<i4"),
        ("label_offset", "<i4"),
        ("label_length", "<i4"),
    ]
)


def readback_request_dtype() -> np.dtype:
    return READBACK_REQUEST_DTYPE


PLACEHOLDER_DTYPE = np.dtype(
    [
        ("entity_id", "<i4"),
        ("buffer_x", "<i4"),
        ("buffer_y", "<i4"),
        ("world_x", "<i4"),
        ("world_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("material_id", "<i4"),
    ]
)

PLACEHOLDER_DIRTY_RECT_DTYPE = np.dtype(
    [
        ("buffer_x0", "<i4"),
        ("buffer_y0", "<i4"),
        ("buffer_x1", "<i4"),
        ("buffer_y1", "<i4"),
        ("world_x0", "<i4"),
        ("world_y0", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
    ]
)

ACTIVE_META_DTYPE = np.dtype(
    [
        ("tile_size", "<i4"),
        ("chunk_tiles", "<i4"),
        ("active_ttl_reset", "<i4"),
        ("tile_width", "<i4"),
        ("tile_height", "<i4"),
        ("chunk_width", "<i4"),
        ("chunk_height", "<i4"),
        ("active_tile_count", "<i4"),
        ("active_chunk_count", "<i4"),
    ]
)

ACTIVE_RECT_DTYPE = np.dtype(
    [
        ("x0", "<i4"),
        ("y0", "<i4"),
        ("x1", "<i4"),
        ("y1", "<i4"),
        ("tile_padding", "<i4"),
    ]
)

GAS_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("pressure_iterations", "<i4"),
        ("force_source_count_before", "<i4"),
        ("force_source_count_after", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("velocity_changed", "<i4"),
        ("ambient_changed", "<i4"),
        ("gas_changed", "<i4"),
        ("pressure_min", "<f4"),
        ("pressure_max", "<f4"),
        ("ambient_min", "<f4"),
        ("ambient_max", "<f4"),
        ("flow_speed_min", "<f4"),
        ("flow_speed_max", "<f4"),
    ]
)

GAS_SPECIES_RUNTIME_DTYPE = np.dtype(
    [
        ("species_id", "<i4"),
        ("total_concentration", "<f4"),
        ("active_concentration", "<f4"),
    ]
)

HEAT_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("ambient_iterations", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("phase_target_count", "<i4"),
        ("boil_target_count", "<i4"),
        ("condense_target_count", "<i4"),
        ("cell_changed", "<i4"),
        ("ambient_changed", "<i4"),
        ("material_changed", "<i4"),
        ("phase_changed", "<i4"),
        ("integrity_changed", "<i4"),
        ("gas_changed", "<i4"),
        ("cell_temperature_min", "<f4"),
        ("cell_temperature_max", "<f4"),
        ("ambient_temperature_min", "<f4"),
        ("ambient_temperature_max", "<f4"),
        ("integrity_min", "<f4"),
        ("integrity_max", "<f4"),
    ]
)

LIQUID_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("solve_tile_count", "<i4"),
        ("post_tile_count", "<i4"),
        ("post_cell_count", "<i4"),
        ("vertical_seam_cell_count", "<i4"),
        ("horizontal_seam_cell_count", "<i4"),
        ("buoyancy_candidate_count", "<i4"),
        ("changed_cell_count", "<i4"),
        ("material_changed", "<i4"),
        ("phase_changed", "<i4"),
        ("velocity_changed", "<i4"),
        ("temperature_changed", "<i4"),
        ("integrity_changed", "<i4"),
        ("placeholder_changed", "<i4"),
        ("pending_placeholder_count_before", "<i4"),
        ("pending_placeholder_count_after", "<i4"),
        ("liquid_cell_count_before", "<i4"),
        ("liquid_cell_count_after", "<i4"),
    ]
)

REACTION_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("changed_cell_count", "<i4"),
        ("changed_gas_count", "<i4"),
        ("ambient_changed_count", "<i4"),
        ("timer_changed_count", "<i4"),
        ("executed_action_count", "<i4"),
        ("emitted_light_count", "<i4"),
        ("emitted_material_count", "<i4"),
        ("emit_light_action_count", "<i4"),
        ("emit_material_action_count", "<i4"),
        ("modify_gas_action_count", "<i4"),
        ("convert_material_action_count", "<i4"),
        ("modify_temperature_action_count", "<i4"),
        ("harm_action_count", "<i4"),
        ("timed_action_count", "<i4"),
        ("self_action_count", "<i4"),
        ("material_material_action_count", "<i4"),
        ("material_gas_action_count", "<i4"),
        ("material_light_action_count", "<i4"),
        ("gas_gas_action_count", "<i4"),
        ("gas_light_action_count", "<i4"),
    ]
)

COLLAPSE_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("dirty_region_count_before", "<i4"),
        ("solve_region_count", "<i4"),
        ("solve_region_cell_count", "<i4"),
        ("structural_cell_count", "<i4"),
        ("support_seed_count", "<i4"),
        ("supported_cell_count", "<i4"),
        ("unsupported_cell_count", "<i4"),
        ("delayed_pending_count", "<i4"),
        ("immune_unsupported_count", "<i4"),
        ("collapsed_cell_count", "<i4"),
        ("collapsed_component_count", "<i4"),
    ]
)

COLLAPSE_COMPONENT_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("bbox", "<i4", (4,)),
        ("cell_count", "<i4"),
    ]
)

OPTICS_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("emitter_count", "<i4"),
        ("secondary_branch_count", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("visible_changed_count", "<i4"),
        ("cell_dose_changed_count", "<i4"),
        ("gas_dose_changed_count", "<i4"),
        ("visible_energy_total", "<f4"),
        ("cell_dose_total", "<f4"),
        ("gas_dose_total", "<f4"),
    ]
)


def placeholder_dtype() -> np.dtype:
    return PLACEHOLDER_DTYPE


def placeholder_dirty_rect_dtype() -> np.dtype:
    return PLACEHOLDER_DIRTY_RECT_DTYPE


def active_meta_dtype() -> np.dtype:
    return ACTIVE_META_DTYPE


def active_rect_dtype() -> np.dtype:
    return ACTIVE_RECT_DTYPE


def gas_runtime_meta_dtype() -> np.dtype:
    return GAS_RUNTIME_META_DTYPE


def gas_species_runtime_dtype() -> np.dtype:
    return GAS_SPECIES_RUNTIME_DTYPE


def heat_runtime_meta_dtype() -> np.dtype:
    return HEAT_RUNTIME_META_DTYPE


def liquid_runtime_meta_dtype() -> np.dtype:
    return LIQUID_RUNTIME_META_DTYPE


def reaction_runtime_meta_dtype() -> np.dtype:
    return REACTION_RUNTIME_META_DTYPE


def collapse_runtime_meta_dtype() -> np.dtype:
    return COLLAPSE_RUNTIME_META_DTYPE


def collapse_component_dtype() -> np.dtype:
    return COLLAPSE_COMPONENT_DTYPE


def optics_runtime_meta_dtype() -> np.dtype:
    return OPTICS_RUNTIME_META_DTYPE


PAGE_STRIPE_META_DTYPE = np.dtype(
    [
        ("axis_id", "<i4"),
        ("kind_id", "<i4"),
        ("world_start", "<i4"),
        ("world_end", "<i4"),
        ("buffer_start", "<i4"),
        ("buffer_end", "<i4"),
        ("section_offset", "<i4"),
        ("section_count", "<i4"),
    ]
)


PAGE_STRIPE_SECTION_DTYPE = np.dtype(
    [
        ("stripe_index", "<i4"),
        ("field_id", "<i4"),
        ("dtype_code", "<i4"),
        ("ndim", "<i4"),
        ("dim0", "<i4"),
        ("dim1", "<i4"),
        ("dim2", "<i4"),
        ("byte_offset", "<i8"),
        ("byte_length", "<i8"),
    ]
)


def page_stripe_meta_dtype() -> np.dtype:
    return PAGE_STRIPE_META_DTYPE


def page_stripe_section_dtype() -> np.dtype:
    return PAGE_STRIPE_SECTION_DTYPE


RULE_TABLE_META_DTYPE = np.dtype(
    [
        ("material_count", "<i4"),
        ("gas_count", "<i4"),
        ("light_count", "<i4"),
        ("optics_count", "<i4"),
        ("reaction_action_count", "<i4"),
        ("material_material_rule_count", "<i4"),
        ("material_gas_rule_count", "<i4"),
        ("material_light_rule_count", "<i4"),
        ("gas_gas_rule_count", "<i4"),
        ("gas_light_rule_count", "<i4"),
        ("self_rule_count", "<i4"),
        ("material_generation", "<i4"),
        ("gas_generation", "<i4"),
        ("light_generation", "<i4"),
        ("optics_generation", "<i4"),
        ("reaction_generation", "<i4"),
    ]
)
MATERIAL_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("name_hash", "<u8"),
        ("default_phase", "<i4"),
        ("render_group_id", "<i4"),
        ("base_color", "<f4", (3,)),
        ("density", "<f4"),
        ("gravity_scale", "<f4"),
        ("wind_coupling", "<f4"),
        ("drag_scale", "<f4"),
        ("friction", "<f4"),
        ("elasticity", "<f4"),
        ("max_dda_step", "<i4"),
        ("powder_solver_kind_id", "<i4"),
        ("liquid_solver_kind_id", "<i4"),
        ("falling_island_break_kind_id", "<i4"),
        ("is_structural", "<i4"),
        ("is_support_anchor", "<i4"),
        ("collapse_behavior_id", "<i4"),
        ("collapse_generation_id", "<i4"),
        ("powder_generation_id", "<i4"),
        ("base_integrity", "<f4"),
        ("spawn_temperature", "<f4"),
        ("heat_capacity", "<f4"),
        ("conductivity", "<f4"),
        ("ambient_exchange_rate", "<f4"),
        ("melt_point", "<f4"),
        ("boil_point", "<f4"),
        ("melt_to_material_id", "<i4"),
        ("freeze_to_material_id", "<i4"),
        ("boil_to_gas_species_id", "<i4"),
        ("material_tag_mask", "<u4"),
        ("gas_tag_mask", "<u4"),
        ("light_tag_mask", "<u4"),
        ("reaction_slots", "<i4", (8,)),
    ]
)
GAS_TABLE_DTYPE = np.dtype(
    [
        ("species_id", "<i4"),
        ("name_hash", "<u8"),
        ("color", "<f4", (3,)),
        ("diffusion_rate", "<f4"),
        ("buoyancy", "<f4"),
        ("decay_rate", "<f4"),
        ("temperature_coupling", "<f4"),
        ("condense_point", "<f4"),
        ("condense_to_material_id", "<i4"),
        ("pressure_factor", "<f4"),
        ("density_factor", "<f4"),
        ("material_reaction_tag_mask", "<u4"),
        ("light_reaction_tag_mask", "<u4"),
    ]
)
LIGHT_TABLE_DTYPE = np.dtype(
    [
        ("light_type_id", "<i4"),
        ("name_hash", "<u8"),
        ("color", "<f4", (3,)),
        ("visual_channel", "<i4"),
        ("default_range", "<i4"),
        ("max_bounce", "<i4"),
        ("dose_channel_id", "<i4"),
        ("render_style_id", "<i4"),
    ]
)
OPTICS_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("light_type_id", "<i4"),
        ("absorption", "<f4"),
        ("scattering", "<f4"),
        ("refraction", "<f4"),
    ]
)
REACTION_ACTION_FLAG_RANDOM_TARGET = 1 << 0
REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE = 1 << 1
REACTION_ACTION_TABLE_DTYPE = np.dtype(
    [
        ("reaction_type_id", "<i4"),
        ("target_material_id", "<i4"),
        ("emit_material_id", "<i4"),
        ("light_type_id", "<i4"),
        ("gas_species_id", "<i4"),
        ("duration", "<i4"),
        ("velocity", "<f4", (2,)),
        ("direction_id", "<i4"),
        ("range_cells", "<i4"),
        ("generation", "<i4"),
        ("flags", "<i4"),
        ("speed", "<f4"),
        ("strength", "<f4"),
        ("beam_width", "<f4"),
        ("delta", "<f4"),
        ("value", "<f4"),
        ("harm_per_frame", "<f4"),
        ("integrity_threshold", "<f4"),
    ]
)
PAIR_REACTION_RULE_TABLE_DTYPE = np.dtype(
    [
        ("lhs_material_id", "<i4"),
        ("lhs_gas_id", "<i4"),
        ("rhs_material_id", "<i4"),
        ("rhs_gas_id", "<i4"),
        ("rhs_light_id", "<i4"),
        ("lhs_tag_mask", "<u4"),
        ("rhs_tag_mask", "<u4"),
        ("phase_mask", "<u4"),
        ("consume_policy_id", "<i4"),
        ("result_action", "<i4"),
        ("trigger_slot_index", "<i4"),
        ("min_temperature", "<f4"),
        ("max_temperature", "<f4"),
        ("threshold", "<f4"),
        ("rate", "<f4"),
    ]
)
SELF_REACTION_RULE_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("trigger_slot_index", "<i4"),
        ("phase_mask", "<u4"),
        ("timer_index", "<i4"),
        ("min_temperature", "<f4"),
        ("max_temperature", "<f4"),
        ("integrity_at_most", "<f4"),
        ("integrity_at_least", "<f4"),
    ]
)


def rule_table_meta_dtype() -> np.dtype:
    return RULE_TABLE_META_DTYPE


def material_table_dtype() -> np.dtype:
    return MATERIAL_TABLE_DTYPE


def gas_table_dtype() -> np.dtype:
    return GAS_TABLE_DTYPE


def light_table_dtype() -> np.dtype:
    return LIGHT_TABLE_DTYPE


def optics_table_dtype() -> np.dtype:
    return OPTICS_TABLE_DTYPE


def reaction_action_table_dtype() -> np.dtype:
    return REACTION_ACTION_TABLE_DTYPE


def pair_reaction_rule_table_dtype() -> np.dtype:
    return PAIR_REACTION_RULE_TABLE_DTYPE


def self_reaction_rule_table_dtype() -> np.dtype:
    return SELF_REACTION_RULE_TABLE_DTYPE
