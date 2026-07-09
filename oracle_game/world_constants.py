"""Module-level constants for the world engine.

These were previously inlined at the top of :mod:`oracle_game.world`.  They are
factored out here so that focused engine submodules (capabilities schema,
intent resolution, geometry, …) can import them without creating a circular
dependency on :mod:`oracle_game.world`, and so that configuration/lookup tables
are separated from engine logic.

Nothing here is mutated at runtime; lookups use ``deepcopy`` or fresh containers
at the call site (e.g. ``BASE_MATERIAL_RUNTIME_ALIASES`` is deep-copied when the
capabilities schema is built).
"""
from __future__ import annotations

from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS

# Cardinal unit vectors keyed by human-facing direction name.
CARDINAL_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

# Anchor-filter vocabularies used by target/intent resolution.
TERRAIN_ANCHOR_FILTERS = {"empty", "hill", "hole", "liquid", "pool", "solid", "tree", "wall"}
IGNORED_ANCHOR_FILTERS = {"nearest"}

# Target-query spatial hints.
TARGET_QUERY_CELLS_PER_METER = 4.0
TARGET_QUERY_DISTANCE_HINT_CELLS: dict[str, int] = {
    "near": 4,
    "far": 12,
}

# Sentinel for "no controller state has been set yet".
UNSET_CONTROLLER_STATE = object()

# Per world-command-kind, the payload fields that carry a target coordinate.
TARGETED_COMMAND_COORD_FIELDS: dict[str, tuple[str, str]] = {
    "advance_paging": ("center_x", "center_y"),
    "inject_force": ("x", "y"),
    "inject_gas": ("x", "y"),
    "inject_light": ("x", "y"),
    "inject_material": ("x", "y"),
    "inject_temperature": ("x", "y"),
    "inject_velocity": ("x", "y"),
    "request_readback": ("center_x", "center_y"),
    "write_material_region": ("x", "y"),
}

# Command kinds exposed through the public frame-input API.
PUBLIC_WORLD_COMMAND_KINDS = (
    "advance_paging",
    "inject_force",
    "inject_gas",
    "inject_light",
    "inject_material",
    "inject_temperature",
    "inject_velocity",
    "patch_entity_states",
    "request_readback",
    "sync_entity_observation_specs",
    "sync_entity_states",
    "write_material_region",
)

# Reaction-rule set names.
PAIR_REACTION_RULE_SET_NAMES = frozenset(
    {
        "material_material",
        "material_gas",
        "material_light",
        "gas_gas",
        "gas_light",
    }
)
REACTION_RULE_SET_NAMES = frozenset({*PAIR_REACTION_RULE_SET_NAMES, "self_rules"})

READBACK_ALLOWED_CHANNEL_SET = frozenset(READBACK_ALLOWED_CHANNELS)

# Async readback cap (per-edge).
MAX_ASYNC_READBACK_WIDTH = 64
MAX_ASYNC_READBACK_HEIGHT = 64

# Cell count above which the realtime budget gates GPU stages.
GPU_REALTIME_BUDGET_CELL_THRESHOLD = 1080 * 720

# Friendly aliases -> canonical material names (used by the capabilities API
# and by sanctioned-name resolution).
BASE_MATERIAL_RUNTIME_ALIASES: dict[str, str] = {
    "sand": "sand_powder",
    "gravel": "gravel_powder",
    "soil": "soil_powder",
    "sandstone": "sandstone_solid",
    "raw_stone": "raw_stone_solid",
    "obsidian": "obsidian_solid",
    "root": "root_solid",
    "log": "log_solid",
    "leaf": "leaf_powder",
    "vine": "vine_solid",
    "moss": "moss_powder",
    "iron": "iron_solid",
    "gold": "gold_solid",
    "water": "water_liquid",
    "poison": "poison_liquid",
    "acid": "acid_liquid",
    "oil": "oil_liquid",
    "explosive": "explosive_solid",
    "phosphor_visible": "phosphor_visible_powder",
    "phosphor_holy": "phosphor_holy_powder",
    "phosphor_chaos": "phosphor_chaos_powder",
    "phosphor_magic": "phosphor_magic_powder",
    "pollution": "pollution_powder",
    "vortex_heart": "vortex_heart_solid",
    "fire": "fire_powder",
    "placeholder": "placeholder_solid",
}

# Entity-state fields a patch may set.
ENTITY_STATE_PATCHABLE_FIELDS = frozenset(
    {
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
    }
)
ENTITY_STATE_PATCH_METADATA_FIELDS = frozenset({"_world_x", "_world_y"})
