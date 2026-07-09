from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from oracle_game.types import DebugView
from oracle_game.world import WorldEngine


MATERIAL_KEYS = [
    "sand",
    "gravel",
    "soil",
    "raw_stone",
    "water",
    "poison",
    "acid",
    "oil",
    "fire",
    "phosphor_visible",
]
BRUSH_MODES = ("material", "gas", "light", "temperature", "velocity", "force")


def _alpha_keys(letter: str) -> tuple[int, int]:
    return (ord(letter.upper()), ord(letter.lower()))


DEBUG_VIEW_KEYMAP: dict[int, DebugView] = {
    key: view
    for letter, view in (
        ("M", DebugView.MATERIAL),
        ("C", DebugView.ACTIVE),
        ("T", DebugView.TEMPERATURE),
        ("Q", DebugView.PRESSURE),
        ("H", DebugView.HEAT),
        ("F", DebugView.LIQUID),
        ("R", DebugView.REACTION),
        ("B", DebugView.COLLAPSE),
        ("I", DebugView.OPTICS),
        ("V", DebugView.VELOCITY),
        ("L", DebugView.LIGHT),
        ("G", DebugView.GAS),
        ("O", DebugView.MOTION),
    )
    for key in _alpha_keys(letter)
}
RESET_WORLD_KEYS = frozenset(_alpha_keys("X"))
OPTICS_LIGHT_CYCLE_KEYS = frozenset(_alpha_keys("P"))
GAS_SPECIES_CYCLE_KEYS = frozenset(_alpha_keys("Y"))
BRUSH_MODE_CYCLE_KEYS = frozenset(_alpha_keys("K"))
CONTROLLER_TOGGLE_KEYS = frozenset(_alpha_keys("U"))


def demo_debug_view_for_key(key: int) -> DebugView | None:
    return DEBUG_VIEW_KEYMAP.get(key)


def demo_material_for_key(key: int) -> str | None:
    if ord("1") <= key <= ord("9"):
        index = key - ord("1")
        return MATERIAL_KEYS[index] if index < len(MATERIAL_KEYS) else None
    if key == ord("0") and len(MATERIAL_KEYS) >= 10:
        return MATERIAL_KEYS[9]
    return None


def is_demo_reset_key(key: int) -> bool:
    return key in RESET_WORLD_KEYS


def is_demo_optics_cycle_key(key: int) -> bool:
    return key in OPTICS_LIGHT_CYCLE_KEYS


def is_demo_gas_cycle_key(key: int) -> bool:
    return key in GAS_SPECIES_CYCLE_KEYS


def is_demo_brush_cycle_key(key: int) -> bool:
    return key in BRUSH_MODE_CYCLE_KEYS


def is_demo_controller_toggle_key(key: int) -> bool:
    return key in CONTROLLER_TOGGLE_KEYS


def cycle_demo_named_choice(current: str | None, values: Sequence[str | None]) -> str | None:
    choices = tuple(values)
    if not choices:
        return current
    try:
        index = choices.index(current)
    except ValueError:
        return choices[0]
    return choices[(index + 1) % len(choices)]


def cycle_demo_brush_mode(current: str) -> str:
    next_mode = cycle_demo_named_choice(current, BRUSH_MODES)
    return str(next_mode if next_mode is not None else BRUSH_MODES[0])


def clamp_demo_brush_radius(radius: int) -> int:
    return max(0, min(16, int(radius)))


def demo_velocity_from_drag(dx: int, dy: int) -> tuple[float, float]:
    if dx == 0 and dy == 0:
        return (0.0, -1.5)
    return (float(dx) * 0.12, float(-dy) * 0.12)


def demo_light_direction_and_spread_from_drag(dx: int, dy: int) -> tuple[tuple[float, float], float]:
    if dx == 0 and dy == 0:
        return ((0.0, 0.0), 0.25)
    direction_x = float(dx)
    direction_y = float(-dy)
    length = math.hypot(direction_x, direction_y)
    if length <= 1e-6:
        return ((0.0, 0.0), 0.25)
    return ((direction_x / length, direction_y / length), 0.05)


def demo_force_direction_from_drag(dx: int, dy: int) -> tuple[float, float]:
    if dx == 0 and dy == 0:
        return (0.0, -1.0)
    direction_x = float(dx)
    direction_y = float(-dy)
    length = math.hypot(direction_x, direction_y)
    if length <= 1e-6:
        return (0.0, -1.0)
    return (direction_x / length, direction_y / length)


def demo_screen_to_buffer_cell(
    screen_x: int,
    screen_y: int,
    *,
    screen_width: int,
    screen_height: int,
    buffer_width: int,
    buffer_height: int,
    active_width: int,
    active_height: int,
    buffer_origin_x: int,
    buffer_origin_y: int,
) -> tuple[int, int]:
    max_screen_x = max(1, int(screen_width) - 1)
    max_screen_y = max(1, int(screen_height) - 1)
    display_x = int(
        np.clip(
            screen_x / max_screen_x * max(0, int(active_width) - 1),
            0,
            max(0, int(active_width) - 1),
        )
    )
    display_y = int(
        np.clip(
            screen_y / max_screen_y * max(0, int(active_height) - 1),
            0,
            max(0, int(active_height) - 1),
        )
    )
    return (
        (display_x + int(buffer_origin_x)) % int(buffer_width),
        (display_y + int(buffer_origin_y)) % int(buffer_height),
    )


def demo_screen_to_world_cell(
    screen_x: int,
    screen_y: int,
    *,
    screen_width: int,
    screen_height: int,
    active_width: int,
    active_height: int,
    world_origin_x: int,
    world_origin_y: int,
) -> tuple[int, int]:
    max_screen_x = max(1, int(screen_width) - 1)
    max_screen_y = max(1, int(screen_height) - 1)
    display_x = int(
        np.clip(
            screen_x / max_screen_x * max(0, int(active_width) - 1),
            0,
            max(0, int(active_width) - 1),
        )
    )
    display_y = int(
        np.clip(
            screen_y / max_screen_y * max(0, int(active_height) - 1),
            0,
            max(0, int(active_height) - 1),
        )
    )
    return (int(world_origin_x) + display_x, int(world_origin_y) + display_y)


def resolve_demo_paint_command(
    brush_mode: str,
    *,
    selected_material: str,
    gas_species: str,
    light_type: str | None,
    radius: int,
    dx: int = 0,
    dy: int = 0,
) -> tuple[str, dict[str, object]]:
    if brush_mode == "gas":
        return (
            "inject_gas",
            {
                "species": gas_species,
                "amount": 0.75,
                "radius": max(1, int(radius)),
            },
        )
    if brush_mode == "light":
        direction, spread = demo_light_direction_and_spread_from_drag(dx, dy)
        return (
            "inject_light",
            {
                "light_type": light_type or "visible_light",
                "strength": 1.25,
                "direction": direction,
                "spread": spread,
            },
        )
    if brush_mode == "temperature":
        return (
            "inject_temperature",
            {
                "delta": 40.0,
                "radius": max(1, int(radius)),
            },
        )
    if brush_mode == "velocity":
        return (
            "inject_velocity",
            {
                "velocity": demo_velocity_from_drag(dx, dy),
                "radius": max(1, int(radius)),
                "carrier": "cell",
                "mode": "add",
            },
        )
    if brush_mode == "force":
        return (
            "inject_force",
            {
                "direction": demo_force_direction_from_drag(dx, dy),
                "radius": float(max(2, int(radius) * 2)),
                "strength": 2.0,
                "lifetime": 0.4,
            },
        )
    return (
        "inject_material",
        {
            "material": selected_material,
            "radius": max(0, int(radius)),
        },
    )


def apply_demo_paint(
    engine: WorldEngine,
    brush_mode: str,
    *,
    x: int,
    y: int,
    selected_material: str,
    gas_species: str,
    light_type: str | None,
    radius: int,
    dx: int = 0,
    dy: int = 0,
) -> tuple[str, dict[str, object]]:
    kind, payload = resolve_demo_paint_command(
        brush_mode,
        selected_material=selected_material,
        gas_species=gas_species,
        light_type=light_type,
        radius=radius,
        dx=dx,
        dy=dy,
    )
    if kind == "inject_gas":
        engine.inject_gas(
            int(x),
            int(y),
            str(payload["species"]),
            float(payload["amount"]),
            int(payload["radius"]),
            immediate=True,
        )
    elif kind == "inject_light":
        engine.inject_light(
            int(x),
            int(y),
            str(payload["light_type"]),
            float(payload["strength"]),
            direction=tuple(payload["direction"]),
            spread=float(payload["spread"]),
            immediate=True,
        )
    elif kind == "inject_temperature":
        engine.inject_temperature(
            int(x),
            int(y),
            float(payload["delta"]),
            int(payload["radius"]),
            immediate=True,
        )
    elif kind == "inject_velocity":
        engine.inject_velocity(
            int(x),
            int(y),
            tuple(payload["velocity"]),
            int(payload["radius"]),
            carrier=str(payload["carrier"]),
            mode=str(payload["mode"]),
            immediate=True,
        )
    elif kind == "inject_force":
        engine.inject_force(
            int(x),
            int(y),
            tuple(payload["direction"]),
            float(payload["radius"]),
            float(payload["strength"]),
            float(payload["lifetime"]),
            immediate=True,
        )
    else:
        engine.inject_material(
            int(x),
            int(y),
            str(payload["material"]),
            int(payload["radius"]),
            immediate=True,
        )
    return kind, payload


def queue_demo_paint(
    engine: WorldEngine,
    brush_mode: str,
    *,
    x: int,
    y: int,
    selected_material: str,
    gas_species: str,
    light_type: str | None,
    radius: int,
    dx: int = 0,
    dy: int = 0,
) -> tuple[str, dict[str, object]]:
    kind, payload = resolve_demo_paint_command(
        brush_mode,
        selected_material=selected_material,
        gas_species=gas_species,
        light_type=light_type,
        radius=radius,
        dx=dx,
        dy=dy,
    )
    command_payload = dict(payload)
    command_payload["x"] = int(x)
    command_payload["y"] = int(y)
    engine.queue_command(kind, **command_payload)
    return kind, payload
