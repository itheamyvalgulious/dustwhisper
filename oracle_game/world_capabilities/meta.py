"""Meta, HTTP API, readback, target-query and injector capability keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from copy import deepcopy
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS
from oracle_game.types import COLLAPSE_BEHAVIOR_IDS, DebugView, Phase, ReactionType
from oracle_game.world_constants import (
    CARDINAL_DIRECTION_VECTORS,
    IGNORED_ANCHOR_FILTERS,
    MAX_ASYNC_READBACK_HEIGHT,
    MAX_ASYNC_READBACK_WIDTH,
    TARGET_QUERY_CELLS_PER_METER,
    TARGET_QUERY_DISTANCE_HINT_CELLS,
    TERRAIN_ANCHOR_FILTERS,
)

from ._http_read_endpoints import _build_http_read_endpoints
from ._http_mutate_endpoints import _build_http_mutate_endpoints

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_meta_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the meta/http-api/readback/target-query/injectors capability keys."""

    material_name_aliases = ctx.material_name_aliases
    emitter_fields = ctx.emitter_fields
    target_query_fields = ctx.target_query_fields
    inline_target_query_optional_fields = ctx.inline_target_query_optional_fields
    readback_request_fields = ctx.readback_request_fields
    readback_channel_payload_types = ctx.readback_channel_payload_types
    debug_view_options = ctx.debug_view_options
    http_api_endpoints = {
        **_build_http_read_endpoints(engine, ctx),
        **_build_http_mutate_endpoints(engine, ctx),
    }

    return {
        "world": {
            "buffer_size": [int(engine.width), int(engine.height)],
            "active_size": [int(engine.paging.active_width), int(engine.paging.active_height)],
            "gas_cell_size": int(engine.gas_cell_size),
            "gas_grid_size": [int(engine.gas_width), int(engine.gas_height)],
            "tile_size": int(engine.active.tile_size),
            "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
            "chunk_tiles": int(engine.active.chunk_tiles),
            "chunk_grid_size": [int(engine.active.chunk_width), int(engine.active.chunk_height)],
            "default_debug_view": engine.default_debug_view.value,
        },
        "material_name_aliases": {
            "accepts_runtime_names": True,
            "accepts_base_aliases": True,
            "base_to_runtime": material_name_aliases,
            "fields": {
                "material": ["material"],
                "material_def": [
                    "name",
                    "collapse_generation",
                    "powder_generation",
                    "melt_to_material",
                    "freeze_to_material",
                ],
                "gas_def": ["condense_to_material"],
                "material_optics": ["material_name"],
                "reaction_action": ["target_material", "emit_material"],
                "pair_reaction_rule": ["lhs_material", "rhs_material"],
                "self_reaction_rule": ["material"],
                "entity_state": ["placeholder_material"],
                "entity_placeholder": ["material"],
            },
        },
        "debug_views": [view.value for view in DebugView],
        "debug_view_options": deepcopy(debug_view_options),
        "http_api": {
            "base_path": "/api",
            "transport": "json_over_http",
            "endpoints": http_api_endpoints,
        },
        "readback_channels": list(READBACK_ALLOWED_CHANNELS),
        "readback": {
            "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
            "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
            "request_fields": readback_request_fields,
            "result_fields": ["frame_id", "request", "payload"],
            "channel_payload_types": dict(readback_channel_payload_types),
        },
        "target_query": {
            "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
            "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
            "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
            "distance_hint_cells": {
                hint: int(distance)
                for hint, distance in TARGET_QUERY_DISTANCE_HINT_CELLS.items()
            },
            "cells_per_meter": float(TARGET_QUERY_CELLS_PER_METER),
        },
        "phases": {
            phase.name.lower(): int(phase)
            for phase in Phase
        },
        "reaction_types": [reaction_type.name.lower() for reaction_type in ReactionType],
        "collapse_behaviors": sorted(COLLAPSE_BEHAVIOR_IDS),
        "tag_bits": deepcopy(engine.tag_bits_by_name),
        "injectors": {
            "material": {
                "fields": ["x", "y", "material", "radius"],
                "material_aliases": material_name_aliases,
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "temperature": {
                "fields": ["x", "y", "delta", "radius"],
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "velocity": {
                "fields": ["x", "y", "velocity", "radius", "carrier", "mode"],
                "carriers": ["cell", "flow", "both"],
                "modes": ["add", "set"],
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "gas": {
                "fields": ["x", "y", "species", "amount", "radius"],
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "force": {
                "fields": ["x", "y", "direction", "radius", "strength", "lifetime"],
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "light": {
                "fields": emitter_fields,
                "supports_direction": True,
                "supports_spread": True,
                "default_spread": 0.25,
                "supports_immediate": True,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
        },
    }
