"""Paging stripe/store, pending-commands, and table payload keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from oracle_game.gpu import (FALLING_ISLAND_BREAK_KIND_IDS, LIQUID_SOLVER_KIND_IDS, POWDER_SOLVER_KIND_IDS)
from oracle_game.types import COLLAPSE_BEHAVIOR_IDS, Direction, Phase, ReactionType


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_paging_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the paging portion of the engine capabilities schema."""

    material_payload = ctx.material_payload
    gas_payload = ctx.gas_payload
    light_payload = ctx.light_payload
    material_fields = ctx.material_fields
    gas_fields = ctx.gas_fields
    light_fields = ctx.light_fields
    optics_fields = ctx.optics_fields
    reaction_action_fields = ctx.reaction_action_fields
    pair_reaction_rule_fields = ctx.pair_reaction_rule_fields
    self_reaction_rule_fields = ctx.self_reaction_rule_fields
    paging_state_fields = ctx.paging_state_fields
    pending_commands_fields = ctx.pending_commands_fields

    return {
        "page_stripe_update": {
            "fields": [
                "axis",
                "world_start",
                "world_end",
                "buffer_start",
                "buffer_end",
                "kind",
                "cross_world_start",
                "cross_world_end",
            ],
            "axes": ["x", "y"],
            "kinds": ["save", "load"],
            "field_types": {
                "axis": {"type": "str"},
                "world_start": {"type": "int"},
                "world_end": {"type": "int"},
                "buffer_start": {"type": "int"},
                "buffer_end": {"type": "int"},
                "kind": {"type": "str"},
                "cross_world_start": {"type": "int"},
                "cross_world_end": {"type": "int"},
            },
        },
        "paging_state": {
            "fields": paging_state_fields,
            "origin_type": "cell_xy",
            "buffer_origin_type": "cell_xy",
            "active_bounds_type": "cell_rect",
            "buffer_size_type": "cell_wh",
            "active_size_type": "cell_wh",
        },
        "page_stripe_payload": {
            "fields": ["meta", "cell", "runtime", "gas"],
        },
        "page_stripe_apply": {
            "fields": ["update", "payload", "immediate"],
            "update_type": "page_stripe_update",
            "payload_type": "page_stripe_payload",
            "immediate_optional": True,
            "field_types": {
                "update": {"type": "page_stripe_update"},
                "payload": {"type": "page_stripe_payload"},
                "immediate": {"type": "bool", "optional": True},
            },
        },
        "page_store_state": {
            "fields": ["stored_stripes", "key_listing_supported", "stripe_keys"],
            "stripe_key_type": "page_store_key",
            "field_types": {
                "stored_stripes": {"type": "int"},
                "key_listing_supported": {"type": "bool"},
                "stripe_keys": {"type": "page_store_key[]"},
            },
        },
        "page_store_key": {
            "fields": ["axis", "world_start", "world_end", "cross_world_start", "cross_world_end"],
            "axes": ["x", "y"],
            "field_types": {
                "axis": {"type": "str"},
                "world_start": {"type": "int"},
                "world_end": {"type": "int"},
                "cross_world_start": {"type": "int"},
                "cross_world_end": {"type": "int"},
            },
        },
        "page_store_entry": {
            "fields": ["key", "payload"],
            "key_type": "page_store_key",
            "payload_type": "page_stripe_payload",
            "field_types": {
                "key": {"type": "page_store_key"},
                "payload": {"type": "page_stripe_payload"},
            },
        },
        "page_store_export": {
            "fields": ["stored_stripes", "key_listing_supported", "entries"],
            "entry_type": "page_store_entry",
            "field_types": {
                "stored_stripes": {"type": "int"},
                "key_listing_supported": {"type": "bool"},
                "entries": {"type": "page_store_entry[]"},
            },
        },
        "page_store_import": {
            "fields": ["entries", "clear"],
            "entry_type": "page_store_entry",
            "clear_optional": True,
            "field_types": {
                "entries": {"type": "page_store_entry[]"},
                "clear": {"type": "bool", "optional": True},
            },
        },
        "page_store_capture_result": {
            "fields": ["ok", "stored_stripes", "payload"],
            "field_types": {
                "ok": {"type": "bool"},
                "stored_stripes": {"type": "int"},
                "payload": {"type": "page_stripe_payload"},
            },
        },
        "page_store_load_result": {
            "fields": ["ok", "stored", "payload"],
            "field_types": {
                "ok": {"type": "bool"},
                "stored": {"type": "bool"},
                "payload": {"type": "page_stripe_payload", "optional": True},
            },
        },
        "page_store_save_result": {
            "fields": ["ok", "stored_stripes"],
            "field_types": {
                "ok": {"type": "bool"},
                "stored_stripes": {"type": "int"},
            },
        },
        "page_stripe_capture_result": {
            "fields": ["ok", "payload"],
            "field_types": {
                "ok": {"type": "bool"},
                "payload": {"type": "page_stripe_payload"},
            },
        },
        "page_store_import_result": {
            "fields": ["cleared", "imported", "stored_stripes"],
            "field_types": {
                "cleared": {"type": "int"},
                "imported": {"type": "int"},
                "stored_stripes": {"type": "int"},
            },
        },
        "pending_commands": {
            "fields": pending_commands_fields,
            "field_types": {
                "pending": {"type": "int"},
                "commands": {"type": "world_command[]"},
            },
        },
        "tables": {
            "materials": {
                "record_fields": material_fields,
                "identity_fields": ["material_id", "name"],
                "update_semantics": "merge_by_material_id",
                "patch_identity_field": "name",
                "patchable_fields": [field for field in material_fields if field not in {"material_id"}],
                "material_alias_fields": [
                    "name",
                    "collapse_generation",
                    "powder_generation",
                    "melt_to_material",
                    "freeze_to_material",
                ],
                "enums": {
                    "default_phase": [phase.name.lower() for phase in Phase],
                    "collapse_behavior": sorted(COLLAPSE_BEHAVIOR_IDS),
                    "powder_solver_kind": sorted(POWDER_SOLVER_KIND_IDS),
                    "liquid_solver_kind": sorted(LIQUID_SOLVER_KIND_IDS),
                    "falling_island_break_kind": sorted(FALLING_ISLAND_BREAK_KIND_IDS),
                },
            },
            "gases": {
                "record_fields": gas_fields,
                "identity_fields": ["species_id", "name"],
                "update_semantics": "merge_by_species_id",
                "patch_identity_field": "name",
                "patchable_fields": [field for field in gas_fields if field not in {"species_id"}],
                "material_alias_fields": ["condense_to_material"],
            },
            "lights": {
                "record_fields": light_fields,
                "identity_fields": ["light_type_id", "name"],
                "update_semantics": "merge_by_light_type_id",
                "patch_identity_field": "name",
                "patchable_fields": [field for field in light_fields if field not in {"light_type_id"}],
            },
            "optics": {
                "record_fields": optics_fields,
                "identity_fields": ["material_name", "light_type"],
                "update_semantics": "merge_by_material_name_and_light_type",
                "patch_identity_fields": ["material_name", "light_type"],
                "patchable_fields": ["absorption", "scattering", "refraction"],
                "material_alias_fields": ["material_name"],
            },
            "reactions": {
                "action_fields": reaction_action_fields,
                "action_patch_identity_field": "index",
                "action_patchable_fields": reaction_action_fields,
                "action_material_alias_fields": ["target_material", "emit_material"],
                "append_rule_sets": [
                    "material_material",
                    "material_gas",
                    "material_light",
                    "gas_gas",
                    "gas_light",
                    "self_rules",
                ],
                "replace_semantics": "replace_all",
                "rule_fields": {
                    "pair_rule": pair_reaction_rule_fields,
                    "self_rule": self_reaction_rule_fields,
                },
                "rule_material_alias_fields": {
                    "pair_rule": ["lhs_material", "rhs_material"],
                    "self_rule": ["material"],
                },
                "enum_domains": {
                    "reaction_type": [reaction_type.name.lower() for reaction_type in ReactionType],
                    "direction": [direction.value for direction in Direction],
                    "phase": [phase.name.lower() for phase in Phase],
                },
            },
        },
        "materials": [
            {
                "material_id": int(item["material_id"]),
                "name": str(item["name"]),
                "display_name": str(item["display_name"]),
                "default_phase": int(engine._coerce_enum(Phase, item["default_phase"])),
                "render_group": str(item["render_group"]),
                "tags": list(item.get("tags", ())),
            }
            for item in material_payload
        ],
        "gases": [
            {
                "species_id": int(item["species_id"]),
                "name": str(item["name"]),
                "display_name": str(item["display_name"]),
            }
            for item in gas_payload
        ],
        "lights": [
            {
                "light_type_id": int(item["light_type_id"]),
                "name": str(item["name"]),
                "display_name": str(item["display_name"]),
                "default_range": int(item["default_range"]),
                "dose_channel_id": int(item["dose_channel_id"]),
                "visual_channel": int(item["visual_channel"]),
            }
            for item in light_payload
        ],
    }
