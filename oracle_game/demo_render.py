from __future__ import annotations

from typing import Any

from oracle_game.types import DebugView
from oracle_game.world import WorldEngine


DEMO_PATTERN_SCALE = 8.0
DEMO_AMBIENT_TOP_LIGHT = (0.48, 0.50, 0.46)
DEMO_AMBIENT_BOTTOM_LIGHT = (0.24, 0.26, 0.28)


def build_demo_render_uniforms(engine: WorldEngine, *, debug_view: DebugView) -> dict[str, object]:
    visible_width = int(getattr(engine, "demo_visible_width", engine.paging.active_width))
    visible_height = int(getattr(engine, "demo_visible_height", engine.paging.active_height))
    return {
        "buffer_size": (engine.width, engine.height),
        "active_size": (visible_width, visible_height),
        "buffer_origin": (engine.paging.buffer_origin_x, engine.paging.buffer_origin_y),
        "world_origin": (engine.paging.origin_x, engine.paging.origin_y),
        "atlas_grid": tuple(engine.bridge.atlas_grid),
        "view_mode": 0 if debug_view == DebugView.MATERIAL else 1,
        "force_debug_texture": debug_view != DebugView.MATERIAL,
        "pattern_scale": DEMO_PATTERN_SCALE,
        "ambient_top_light": DEMO_AMBIENT_TOP_LIGHT,
        "ambient_bottom_light": DEMO_AMBIENT_BOTTOM_LIGHT,
    }


def apply_demo_render_uniforms(program: Any, uniforms: dict[str, object]) -> None:
    for name, value in uniforms.items():
        program[name].value = value
