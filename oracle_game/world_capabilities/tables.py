"""Material/gas/light/optic table, patch, reaction, and runtime schema keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from oracle_game.gpu import (FALLING_ISLAND_BREAK_KIND_IDS, LIQUID_SOLVER_KIND_IDS, POWDER_SOLVER_KIND_IDS)
from oracle_game.types import COLLAPSE_BEHAVIOR_IDS, Direction, Phase, ReactionType


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_tables_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the tables portion of the engine capabilities schema."""

    material_fields = ctx.material_fields
    gas_fields = ctx.gas_fields
    light_fields = ctx.light_fields
    optics_fields = ctx.optics_fields
    reaction_action_fields = ctx.reaction_action_fields
    reaction_table_rule_sets = ctx.reaction_table_rule_sets
    pair_reaction_rule_fields = ctx.pair_reaction_rule_fields
    self_reaction_rule_fields = ctx.self_reaction_rule_fields
    material_patch_fields = ctx.material_patch_fields
    gas_patch_fields = ctx.gas_patch_fields
    light_patch_fields = ctx.light_patch_fields
    optics_patch_fields = ctx.optics_patch_fields
    reaction_action_patch_fields = ctx.reaction_action_patch_fields
    reaction_action_delete_request_fields = ctx.reaction_action_delete_request_fields
    reaction_rule_patch_fields = ctx.reaction_rule_patch_fields
    reaction_rule_delete_request_fields = ctx.reaction_rule_delete_request_fields
    materials_table_fields = ctx.materials_table_fields
    gases_table_fields = ctx.gases_table_fields
    lights_table_fields = ctx.lights_table_fields
    optics_table_fields = ctx.optics_table_fields
    reactions_table_fields = ctx.reactions_table_fields
    gas_species_runtime_fields = ctx.gas_species_runtime_fields
    heat_phase_target_fields = ctx.heat_phase_target_fields
    heat_boil_target_fields = ctx.heat_boil_target_fields
    heat_condense_target_fields = ctx.heat_condense_target_fields
    collapse_component_fields = ctx.collapse_component_fields
    pending_displaced_cell_fields = ctx.pending_displaced_cell_fields
    powder_reservation_fields = ctx.powder_reservation_fields
    island_reservation_fields = ctx.island_reservation_fields
    gas_runtime_fields = ctx.gas_runtime_fields
    heat_runtime_fields = ctx.heat_runtime_fields
    liquid_runtime_fields = ctx.liquid_runtime_fields
    reaction_runtime_fields = ctx.reaction_runtime_fields
    collapse_runtime_fields = ctx.collapse_runtime_fields
    optics_runtime_fields = ctx.optics_runtime_fields
    active_runtime_fields = ctx.active_runtime_fields
    motion_runtime_fields = ctx.motion_runtime_fields

    return {
        "material": {
            "fields": material_fields,
            "identity_fields": ["material_id", "name"],
            "enums": {
                "default_phase": [phase.name.lower() for phase in Phase],
                "collapse_behavior": sorted(COLLAPSE_BEHAVIOR_IDS),
                "powder_solver_kind": sorted(POWDER_SOLVER_KIND_IDS),
                "liquid_solver_kind": sorted(LIQUID_SOLVER_KIND_IDS),
                "falling_island_break_kind": sorted(FALLING_ISLAND_BREAK_KIND_IDS),
            },
        },
        "gas": {
            "fields": gas_fields,
            "identity_fields": ["species_id", "name"],
        },
        "light": {
            "fields": light_fields,
            "identity_fields": ["light_type_id", "name"],
        },
        "optics": {
            "fields": optics_fields,
            "identity_fields": ["material_name", "light_type"],
        },
        "materials_table": {
            "fields": materials_table_fields,
            "record_type": "material",
        },
        "gases_table": {
            "fields": gases_table_fields,
            "record_type": "gas",
        },
        "lights_table": {
            "fields": lights_table_fields,
            "record_type": "light",
        },
        "optics_table": {
            "fields": optics_table_fields,
            "record_type": "optics",
        },
        "reaction_action": {
            "fields": reaction_action_fields,
            "enum_domains": {
                "reaction_type": [reaction_type.name.lower() for reaction_type in ReactionType],
                "direction": [direction.value for direction in Direction],
            },
        },
        "pair_reaction_rule": {
            "fields": pair_reaction_rule_fields,
            "phase_domain": [phase.name.lower() for phase in Phase],
        },
        "self_reaction_rule": {
            "fields": self_reaction_rule_fields,
            "phase_domain": [phase.name.lower() for phase in Phase],
        },
        "reactions_table": {
            "fields": reactions_table_fields,
            "action_type": "reaction_action",
            "pair_rule_type": "pair_reaction_rule",
            "self_rule_type": "self_reaction_rule",
            "rule_sets": reaction_table_rule_sets,
        },
        "material_patch": {
            "fields": material_patch_fields,
            "patchable_fields": [field for field in material_fields if field not in {"material_id"}],
        },
        "gas_patch": {
            "fields": gas_patch_fields,
            "patchable_fields": [field for field in gas_fields if field not in {"species_id"}],
        },
        "light_patch": {
            "fields": light_patch_fields,
            "patchable_fields": [field for field in light_fields if field not in {"light_type_id"}],
        },
        "optics_patch": {
            "fields": optics_patch_fields,
            "patchable_fields": ["absorption", "scattering", "refraction"],
        },
        "reaction_action_patch": {
            "fields": reaction_action_patch_fields,
            "patchable_fields": reaction_action_fields,
        },
        "reaction_action_delete_request": {
            "fields": reaction_action_delete_request_fields,
        },
        "reaction_rule_patch": {
            "fields": reaction_rule_patch_fields,
            "rule_sets": reaction_table_rule_sets,
            "pair_rule_type": "pair_reaction_rule",
            "self_rule_type": "self_reaction_rule",
        },
        "reaction_rule_delete_request": {
            "fields": reaction_rule_delete_request_fields,
            "rule_sets": reaction_table_rule_sets,
        },
        "reaction_table_append": {
            "fields": reactions_table_fields,
            "action_type": "reaction_action",
            "pair_rule_type": "pair_reaction_rule",
            "self_rule_type": "self_reaction_rule",
            "rule_sets": reaction_table_rule_sets,
        },
        "reaction_table_replace": {
            "fields": reactions_table_fields,
            "action_type": "reaction_action",
            "pair_rule_type": "pair_reaction_rule",
            "self_rule_type": "self_reaction_rule",
            "rule_sets": reaction_table_rule_sets,
            "replace_semantics": "replace_all",
        },
        "gas_species_runtime": {
            "fields": gas_species_runtime_fields,
        },
        "heat_phase_target": {
            "fields": heat_phase_target_fields,
        },
        "heat_boil_target": {
            "fields": heat_boil_target_fields,
        },
        "heat_condense_target": {
            "fields": heat_condense_target_fields,
        },
        "collapse_component": {
            "fields": collapse_component_fields,
            "bbox_type": "cell_rect",
        },
        "pending_displaced_cell": {
            "fields": pending_displaced_cell_fields,
        },
        "powder_reservation": {
            "fields": powder_reservation_fields,
        },
        "island_reservation": {
            "fields": island_reservation_fields,
        },
        "gas_runtime": {
            "fields": gas_runtime_fields,
            "species_runtime_type": "gas_species_runtime",
        },
        "heat_runtime": {
            "fields": heat_runtime_fields,
            "phase_target_type": "heat_phase_target",
            "boil_target_type": "heat_boil_target",
            "condense_target_type": "heat_condense_target",
        },
        "liquid_runtime": {
            "fields": liquid_runtime_fields,
        },
        "reaction_runtime": {
            "fields": reaction_runtime_fields,
            "stage_names": reaction_table_rule_sets,
        },
        "collapse_runtime": {
            "fields": collapse_runtime_fields,
            "collapsed_component_type": "collapse_component",
        },
        "optics_runtime": {
            "fields": optics_runtime_fields,
            "emitter_type": "emitter",
        },
        "active_runtime": {
            "fields": active_runtime_fields,
            "pending_displaced_cell_type": "pending_displaced_cell",
        },
        "motion_runtime": {
            "fields": motion_runtime_fields,
            "powder_reservation_type": "powder_reservation",
            "island_reservation_type": "island_reservation",
        },
    }
