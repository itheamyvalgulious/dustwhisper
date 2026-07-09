"""Request/command/injector schema definition keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.world_constants import (
    CARDINAL_DIRECTION_VECTORS,
    IGNORED_ANCHOR_FILTERS,
    MAX_ASYNC_READBACK_HEIGHT,
    MAX_ASYNC_READBACK_WIDTH,
    TARGETED_COMMAND_COORD_FIELDS,
    TARGET_QUERY_DISTANCE_HINT_CELLS,
    TERRAIN_ANCHOR_FILTERS,
)


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_schemas_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the schemas portion of the engine capabilities schema."""

    force_source_fields = ctx.force_source_fields
    emitter_fields = ctx.emitter_fields
    target_query_fields = ctx.target_query_fields
    target_query_overlay_fields = ctx.target_query_overlay_fields
    inline_target_query_optional_fields = ctx.inline_target_query_optional_fields
    readback_request_fields = ctx.readback_request_fields
    observation_target_fields = ctx.observation_target_fields
    change_intent_fields = ctx.change_intent_fields
    carrier_intent_fields = ctx.carrier_intent_fields
    world_command_material_alias_fields_by_kind = ctx.world_command_material_alias_fields_by_kind
    world_command_collection_item_types_by_kind = ctx.world_command_collection_item_types_by_kind
    material_injector_fields = ctx.material_injector_fields
    temperature_injector_fields = ctx.temperature_injector_fields
    velocity_injector_fields = ctx.velocity_injector_fields
    gas_injector_fields = ctx.gas_injector_fields
    force_injector_fields = ctx.force_injector_fields
    light_injector_fields = ctx.light_injector_fields
    public_world_command_kinds = ctx.public_world_command_kinds

    return {
        "world_command": {
            "fields": ["kind", "payload"],
            "public_kinds": public_world_command_kinds,
            "target_query_overlay_fields": ["target_query_id", "target_dx", "target_dy"],
            "material_alias_fields_by_kind": world_command_material_alias_fields_by_kind,
            "collection_item_types_by_kind": world_command_collection_item_types_by_kind,
            "target_query_supported_kinds": {
                kind: {"x_field": fields[0], "y_field": fields[1]}
                for kind, fields in sorted(TARGETED_COMMAND_COORD_FIELDS.items())
            },
            "field_types": {
                "kind": {"type": "str"},
                "payload": {"type": "json"},
            },
        },
        "force_source": {
            "fields": force_source_fields,
            "field_types": {
                "x": {"type": "float"},
                "y": {"type": "float"},
                "direction": {"type": "float2"},
                "radius": {"type": "float"},
                "strength": {"type": "float"},
                "lifetime": {"type": "float"},
            },
        },
        "emitter": {
            "fields": emitter_fields,
            "field_types": {
                "x": {"type": "int"},
                "y": {"type": "int"},
                "light_type": {"type": "str"},
                "strength": {"type": "float"},
                "radius": {"type": "int"},
                "direction": {"type": "float2"},
                "spread": {"type": "float"},
            },
        },
        "target_query": {
            "fields": target_query_fields,
            "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
            "distance_hints": sorted(TARGET_QUERY_DISTANCE_HINT_CELLS),
            "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
            "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
            "field_types": {
                "query_id": {"type": "str"},
                "anchor_filters": {"type": "str[]"},
                "source_entity_id": {"type": "int", "optional": True},
                "source_x": {"type": "int", "optional": True},
                "source_y": {"type": "int", "optional": True},
                "anchor_entity_id": {"type": "int", "optional": True},
                "direction": {"type": "str", "optional": True},
                "distance_cells": {"type": "int"},
                "distance_meters": {"type": "float", "optional": True},
                "distance_hint": {"type": "str", "optional": True},
                "require_empty": {"type": "bool"},
                "search_radius": {"type": "int"},
                "label": {"type": "str", "optional": True},
            },
        },
        "change_intent": {
            "fields": change_intent_fields,
            "velocity_carriers": ["cell", "flow", "both"],
            "velocity_modes": ["add", "set"],
            "fallback_modes": ["nearest_empty", "source"],
            "material_alias_fields": ["material"],
            "field_types": {
                "intent_id": {"type": "str"},
                "target_query_id": {"type": "str", "optional": True},
                "center_x": {"type": "int", "optional": True},
                "center_y": {"type": "int", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "radius": {"type": "int"},
                "material": {"type": "str", "optional": True},
                "temperature_delta": {"type": "float"},
                "velocity": {"type": "float2", "optional": True},
                "velocity_carrier": {"type": "str"},
                "velocity_mode": {"type": "str"},
                "require_empty": {"type": "bool"},
                "fallback_mode": {"type": "str"},
                "fallback_radius": {"type": "int"},
                "potency": {"type": "float"},
                "stability": {"type": "float"},
                "label": {"type": "str", "optional": True},
            },
        },
        "carrier_intent": {
            "fields": carrier_intent_fields,
            "kinds": ["material", "gas", "light", "force"],
            "release_modes": ["impact", "beam", "projectile"],
            "fallback_modes": ["nearest_empty", "source"],
            "material_alias_fields": ["material"],
            "field_types": {
                "intent_id": {"type": "str"},
                "kind": {"type": "str"},
                "target_query_id": {"type": "str", "optional": True},
                "center_x": {"type": "int", "optional": True},
                "center_y": {"type": "int", "optional": True},
                "source_entity_id": {"type": "int", "optional": True},
                "source_x": {"type": "int", "optional": True},
                "source_y": {"type": "int", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "radius": {"type": "int"},
                "material": {"type": "str", "optional": True},
                "gas_species": {"type": "str", "optional": True},
                "gas_amount": {"type": "float"},
                "light_type": {"type": "str", "optional": True},
                "light_strength": {"type": "float"},
                "light_spread": {"type": "float"},
                "force_radius": {"type": "float"},
                "force_strength": {"type": "float"},
                "force_lifetime": {"type": "float"},
                "release_mode": {"type": "str"},
                "require_empty": {"type": "bool"},
                "fallback_mode": {"type": "str"},
                "fallback_radius": {"type": "int"},
                "potency": {"type": "float"},
                "stability": {"type": "float"},
                "label": {"type": "str", "optional": True},
            },
        },
        "observation_target": {
            "fields": observation_target_fields,
            "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
            "field_types": {
                "observer_id": {"type": "int"},
                "channels": {"type": "str[]"},
                "center_x": {"type": "int", "optional": True},
                "center_y": {"type": "int", "optional": True},
                "width": {"type": "int", "optional": True},
                "height": {"type": "int", "optional": True},
                "entity_id": {"type": "int", "optional": True},
                "pad_cells": {"type": "int"},
                "label": {"type": "str", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
            },
        },
        "readback_request": {
            "fields": readback_request_fields,
            "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
            "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
            "field_types": {
                "request_id": {"type": "int", "optional": True},
                "center_x": {"type": "int", "optional": True},
                "center_y": {"type": "int", "optional": True},
                "width": {"type": "int"},
                "height": {"type": "int"},
                "channels": {"type": "str[]"},
                "observer_id": {"type": "int", "optional": True},
                "label": {"type": "str", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
            },
        },
        "material_injector": {
            "fields": material_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "material_alias_fields": ["material"],
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "material": {"type": "str"},
                "radius": {"type": "int"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "temperature_injector": {
            "fields": temperature_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "delta": {"type": "float"},
                "radius": {"type": "int"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "velocity_injector": {
            "fields": velocity_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "carriers": ["cell", "flow", "both"],
            "modes": ["add", "set"],
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "velocity": {"type": "float2"},
                "radius": {"type": "int"},
                "carrier": {"type": "str"},
                "mode": {"type": "str"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "gas_injector": {
            "fields": gas_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "species": {"type": "str"},
                "amount": {"type": "float"},
                "radius": {"type": "int"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "force_injector": {
            "fields": force_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "field_types": {
                "x": {"type": "float", "optional": True},
                "y": {"type": "float", "optional": True},
                "direction": {"type": "float2"},
                "radius": {"type": "float"},
                "strength": {"type": "float"},
                "lifetime": {"type": "float"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
        "light_injector": {
            "fields": light_injector_fields,
            "optional_fields": ["immediate", *inline_target_query_optional_fields],
            "target_query_fields": target_query_fields,
            "supports_inline_target_queries": True,
            "field_types": {
                "x": {"type": "int", "optional": True},
                "y": {"type": "int", "optional": True},
                "light_type": {"type": "str"},
                "strength": {"type": "float"},
                "radius": {"type": "int", "optional": True},
                "direction": {"type": "float2"},
                "spread": {"type": "float"},
                "immediate": {"type": "bool", "optional": True},
                "target_query_id": {"type": "str", "optional": True},
                "target_dx": {"type": "int"},
                "target_dy": {"type": "int"},
                "target_queries": {"type": "target_query[]", "optional": True},
            },
        },
    }
