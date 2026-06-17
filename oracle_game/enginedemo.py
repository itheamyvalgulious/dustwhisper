from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math
import threading
import time as _time
from typing import Any

import numpy as np

from oracle_game.http_console import EngineHTTPConsole, EngineRunState
from oracle_game.paging import RingPagingWindow
from oracle_game.types import DebugView, EntityState
from oracle_game.gpu import unpack_cell_core
from oracle_game.world import BASE_MATERIAL_RUNTIME_ALIASES, WorldEngine


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
ENGINE_DEMO_TITLE = "Dustwhisper Engine Demo"
DEMO_CONTROLLER_ENTITY_ID = 2_147_483_000
DEMO_PATTERN_SCALE = 8.0
DEMO_VERTEX_SHADER_SOURCE = """
    #version 330
    in vec2 in_pos;
    in vec2 in_uv;
    out vec2 v_uv;
    void main() {
        v_uv = in_uv;
        gl_Position = vec4(in_pos, 0.0, 1.0);
    }
"""
DEMO_FRAGMENT_SHADER_SOURCE = """
    #version 330
    uniform sampler2D material_tex;
    uniform sampler2D light_tex;
    uniform sampler2D debug_tex;
    uniform sampler2D atlas_tex;
    uniform ivec2 buffer_size;
    uniform ivec2 active_size;
    uniform ivec2 buffer_origin;
    uniform ivec2 world_origin;
    uniform ivec2 atlas_grid;
    uniform int view_mode;
    uniform float pattern_scale;
    in vec2 v_uv;
    out vec4 fragColor;
    void main() {
        ivec2 display_cell = ivec2(clamp(floor(v_uv * vec2(active_size)), vec2(0.0), vec2(active_size) - 1.0));
        ivec2 cell = ivec2(
            (display_cell.x + buffer_origin.x) % buffer_size.x,
            (display_cell.y + buffer_origin.y) % buffer_size.y
        );
        ivec2 logical_cell = world_origin + display_cell;
        vec3 light_rgb = clamp(texelFetch(light_tex, cell, 0).rgb, 0.0, 2.0);
        if (view_mode != 0) {
            fragColor = vec4(texelFetch(debug_tex, cell, 0).rgb, 1.0);
            return;
        }
        float material_id = texelFetch(material_tex, cell, 0).r;
        if (material_id < 0.5) {
            vec3 sky = mix(vec3(0.03, 0.04, 0.06), vec3(0.08, 0.11, 0.16), v_uv.y);
            fragColor = vec4(clamp(sky + light_rgb * 0.45, 0.0, 1.0), 1.0);
            return;
        }
        int mid = int(material_id + 0.5);
        int atlas_x = mid % atlas_grid.x;
        int atlas_y = mid / atlas_grid.x;
        vec2 repeat_uv = fract(vec2(logical_cell) / pattern_scale);
        vec2 atlas_uv = (vec2(atlas_x, atlas_y) + repeat_uv) / vec2(atlas_grid);
        vec3 base = texture(atlas_tex, atlas_uv).rgb;
        vec3 color = base * (0.35 + clamp(length(light_rgb), 0.0, 1.5)) + light_rgb * 0.85;
        fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
    }
"""


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


def _demo_shadow_material_name(engine: Any, material_id: int) -> str:
    shadow_lookup = getattr(engine, "_shadow_material_name", None)
    if callable(shadow_lookup):
        resolved = shadow_lookup(int(material_id))
        if resolved:
            return str(resolved)
        return "empty"
    material_name_by_id = getattr(engine, "material_name_by_id", ())
    if 0 <= int(material_id) < len(material_name_by_id) and material_name_by_id[int(material_id)]:
        return str(material_name_by_id[int(material_id)])
    return "empty"


def demo_material_for_key(key: int) -> str | None:
    if ord("1") <= key <= ord("9"):
        index = key - ord("1")
        return MATERIAL_KEYS[index] if index < len(MATERIAL_KEYS) else None
    if key == ord("0") and len(MATERIAL_KEYS) >= 10:
        return MATERIAL_KEYS[9]
    return None


def demo_display_material_name(material_name: str) -> str:
    runtime_name = str(material_name)
    for base_name, candidate in BASE_MATERIAL_RUNTIME_ALIASES.items():
        if candidate == runtime_name:
            return str(base_name)
    return runtime_name


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
            (1.0 - screen_y / max_screen_y) * max(0, int(active_height) - 1),
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
            (1.0 - screen_y / max_screen_y) * max(0, int(active_height) - 1),
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


def demo_view_focus_label(debug_view: DebugView, *, gas_species: str, light_type: str | None) -> str | None:
    if debug_view == DebugView.GAS:
        return f"gas={gas_species}"
    if debug_view in {DebugView.OPTICS, DebugView.LIGHT}:
        return f"light={light_type or 'all'}"
    return None


def demo_brush_selection_label(
    brush_mode: str,
    *,
    selected_material: str,
    gas_species: str,
    light_type: str | None,
) -> str:
    if brush_mode == "gas":
        return gas_species
    if brush_mode == "light":
        return light_type or "visible_light"
    if brush_mode == "temperature":
        return "delta=40"
    if brush_mode == "velocity":
        return "cell/add"
    if brush_mode == "force":
        return "sparse"
    return demo_display_material_name(selected_material)


def _format_demo_probe_vector(label: str, vector: Sequence[float]) -> str:
    return f"{label}=({float(vector[0]):.2f},{float(vector[1]):.2f})"


def _format_demo_probe_rgb(label: str, rgb: Sequence[float]) -> str:
    return f"{label}=({float(rgb[0]):.2f},{float(rgb[1]):.2f},{float(rgb[2]):.2f})"


def format_demo_focus_probe(
    engine: WorldEngine,
    *,
    focus_x: int,
    focus_y: int,
    debug_view: DebugView,
    gas_species: str,
    light_type: str | None,
) -> str:
    paging = engine.paging
    world_to_buffer = getattr(paging, "world_to_buffer", None)
    if callable(world_to_buffer):
        buffer_x, buffer_y = world_to_buffer(int(focus_x), int(focus_y))
    else:
        buffer_x = int(focus_x) - int(getattr(paging, "origin_x", 0)) + int(getattr(paging, "buffer_origin_x", 0))
        buffer_y = int(focus_y) - int(getattr(paging, "origin_y", 0)) + int(getattr(paging, "buffer_origin_y", 0))
    if not all(
        hasattr(engine, attr)
        for attr in (
            "material_id",
            "material_name_by_id",
            "cell_temperature",
            "ambient_temperature",
            "pressure_ping",
            "velocity",
            "flow_velocity",
            "visible_illumination",
            "rulebook",
        )
    ):
        return f"probe=({int(focus_x)},{int(focus_y)})"
    material_id = int(engine.material_id[buffer_y, buffer_x])
    material_name = _demo_shadow_material_name(engine, material_id)
    gas_y, gas_x = engine.cell_to_gas(buffer_y, buffer_x)
    cell_temperature = float(engine.cell_temperature[buffer_y, buffer_x])
    ambient_temperature = float(engine.ambient_temperature[gas_y, gas_x])
    pressure = float(engine.pressure_ping[gas_y, gas_x])
    cell_velocity = engine.velocity[buffer_y, buffer_x]
    flow_velocity = engine.flow_velocity[gas_y, gas_x]
    visible_rgb = engine.visible_illumination[buffer_y, buffer_x]
    parts = [f"probe={demo_display_material_name(material_name)}"]
    if debug_view in {DebugView.TEMPERATURE, DebugView.HEAT}:
        parts.extend(
            [
                f"cellT={cell_temperature:.1f}",
                f"ambientT={ambient_temperature:.1f}",
            ]
        )
    elif debug_view == DebugView.PRESSURE:
        parts.extend(
            [
                f"pressure={pressure:.2f}",
                _format_demo_probe_vector("flowV", flow_velocity),
            ]
        )
    elif debug_view in {DebugView.VELOCITY, DebugView.MOTION}:
        parts.extend(
            [
                _format_demo_probe_vector("cellV", cell_velocity),
                _format_demo_probe_vector("flowV", flow_velocity),
            ]
        )
    else:
        parts.extend(
            [
                f"cellT={cell_temperature:.1f}",
                _format_demo_probe_vector("cellV", cell_velocity),
            ]
        )
    if debug_view == DebugView.GAS:
        species_id = engine.rulebook.gas_id(gas_species)
        parts.extend(
            [
                f"gas={gas_species}:{float(engine.gas_concentration[species_id, gas_y, gas_x]):.2f}",
                f"ambientT={ambient_temperature:.1f}",
            ]
        )
    elif debug_view in {DebugView.LIGHT, DebugView.OPTICS}:
        if light_type is not None:
            resolve_light = getattr(engine, "_resolve_sanctioned_light_id", None)
            shadow_dose_channel = getattr(engine, "_shadow_light_dose_channel", None)
            light_id = int(resolve_light(light_type)) if callable(resolve_light) else engine.rulebook.light_id(light_type)
            dose_channel = shadow_dose_channel(light_id) if callable(shadow_dose_channel) else None
            if dose_channel is None and not callable(shadow_dose_channel) and 0 <= light_id < engine.light_dose_channel.shape[0]:
                dose_channel = int(engine.light_dose_channel[light_id])
            if dose_channel is not None and 0 <= dose_channel < engine.cell_optical_dose.shape[0]:
                parts.append(f"dose={light_type}:{float(engine.cell_optical_dose[dose_channel, buffer_y, buffer_x]):.2f}")
            if dose_channel is not None and 0 <= dose_channel < engine.gas_optical_dose.shape[0]:
                parts.append(f"gasDose={light_type}:{float(engine.gas_optical_dose[dose_channel, gas_y, gas_x]):.2f}")
        else:
            parts.append(_format_demo_probe_rgb("lit", visible_rgb))
    return " ".join(parts)


def build_demo_controller_state(*, focus_x: int, focus_y: int, cycle: int) -> dict[str, object]:
    return {
        "mode": "demo_probe",
        "cycle": int(cycle),
        "focus": [int(focus_x), int(focus_y)],
        "entity_id": int(DEMO_CONTROLLER_ENTITY_ID),
    }


def build_demo_controller_probe_entity(*, focus_x: int, focus_y: int) -> EntityState:
    return EntityState(
        entity_id=DEMO_CONTROLLER_ENTITY_ID,
        x=int(focus_x),
        y=int(focus_y),
        width=1,
        height=1,
        tags=("demo", "controller"),
        facing_xy=(0.0, -1.0),
        observe_channels=("cell", "ambient_temperature", "pressure", "velocity", "gas", "optics"),
        observe_width=7,
        observe_height=7,
        observe_label="demo_probe",
    )


def build_demo_controller_entities(
    existing_entities: Sequence[EntityState],
    *,
    focus_x: int,
    focus_y: int,
) -> list[EntityState]:
    merged = {
        int(entity.entity_id): replace(entity)
        for entity in existing_entities
        if int(entity.entity_id) != DEMO_CONTROLLER_ENTITY_ID
    }
    merged[DEMO_CONTROLLER_ENTITY_ID] = build_demo_controller_probe_entity(focus_x=focus_x, focus_y=focus_y)
    return [merged[entity_id] for entity_id in sorted(merged)]


def build_demo_controller_entities_for_world_focus(
    existing_entities: Sequence[EntityState],
    *,
    focus_x: int,
    focus_y: int,
    paging: RingPagingWindow,
) -> list[EntityState]:
    buffer_x, buffer_y = paging.world_to_buffer(int(focus_x), int(focus_y))
    return [
        replace(
            entity,
            world_x=int(focus_x) if int(entity.entity_id) == DEMO_CONTROLLER_ENTITY_ID else entity.world_x,
            world_y=int(focus_y) if int(entity.entity_id) == DEMO_CONTROLLER_ENTITY_ID else entity.world_y,
        )
        for entity in build_demo_controller_entities(
            existing_entities,
            focus_x=int(buffer_x),
            focus_y=int(buffer_y),
        )
    ]


def demo_default_focus_world(paging: RingPagingWindow) -> tuple[int, int]:
    return (
        int(paging.origin_x) + int(paging.active_width) // 2,
        int(paging.origin_y) + int(paging.active_height) // 2,
    )


def format_demo_controller_status(
    *,
    preview: dict[str, Any] | None,
    turn: dict[str, Any] | None,
    applied: bool | None = None,
) -> str | None:
    if preview is None:
        return None
    runtime_payload = preview if turn is None else turn
    controller_state = runtime_payload.get("controller_state")
    cycle = 0
    if isinstance(controller_state, dict):
        cycle = int(controller_state.get("cycle", 0))
    consumed_payload = runtime_payload.get("consumed", {})
    consumed = int(consumed_payload.get("consumed", 0)) if isinstance(consumed_payload, dict) else 0
    predicted = int(preview.get("queued_observations", 0))
    if turn is None:
        queued = int(preview.get("queued_readbacks", 0))
        return f"ctl=preview@{cycle} obs={consumed}/{predicted} queued={queued}"
    readback_payload = turn.get("readback_state", {})
    pending = int(readback_payload.get("pending", 0)) if isinstance(readback_payload, dict) else 0
    return f"ctl=demo@{cycle} obs={consumed}/{predicted} pending={pending}"


def format_demo_controller_observation_summary(
    observations: dict[str, Any] | None,
    *,
    gas_species: str,
    light_type: str | None,
    material_name_by_id: Any,
) -> str | None:
    if not isinstance(observations, dict) or not observations:
        return None
    first_key = sorted(observations)[0]
    observation = observations.get(first_key)
    if not isinstance(observation, dict):
        return None
    payload = observation.get("payload")
    if not isinstance(payload, dict):
        return None

    parts = [f"obs#{first_key}"]
    cell_payload = payload.get("cell")
    if isinstance(cell_payload, dict) and "core_words" in cell_payload:
        core_words = np.asarray(cell_payload["core_words"], dtype=np.uint32)
        unpacked = unpack_cell_core(core_words)
        center_y = unpacked["material_id"].shape[0] // 2
        center_x = unpacked["material_id"].shape[1] // 2
        material_id = int(unpacked["material_id"][center_y, center_x])
        material_name = _demo_shadow_material_name(material_name_by_id, material_id)
        parts.append(f"mat={demo_display_material_name(material_name)}")
        parts.append(f"cellT={float(unpacked['cell_temperature'][center_y, center_x]):.1f}")
        parts.append(_format_demo_probe_vector("cellV", unpacked["velocity"][center_y, center_x]))

    ambient_payload = payload.get("ambient_temperature")
    if isinstance(ambient_payload, dict) and "values" in ambient_payload:
        values = np.asarray(ambient_payload["values"], dtype=np.float32)
        parts.append(f"ambientT={float(values[values.shape[0] // 2, values.shape[1] // 2]):.1f}")

    pressure_payload = payload.get("pressure")
    if isinstance(pressure_payload, dict) and "values" in pressure_payload:
        values = np.asarray(pressure_payload["values"], dtype=np.float32)
        parts.append(f"pressure={float(values[values.shape[0] // 2, values.shape[1] // 2]):.2f}")

    velocity_payload = payload.get("velocity")
    if isinstance(velocity_payload, dict) and "values" in velocity_payload:
        values = np.asarray(velocity_payload["values"], dtype=np.float32)
        parts.append(_format_demo_probe_vector("flowV", values[values.shape[0] // 2, values.shape[1] // 2]))

    gas_payload = payload.get("gas")
    if isinstance(gas_payload, dict):
        species = gas_payload.get("species")
        if isinstance(species, dict) and gas_species in species:
            values = np.asarray(species[gas_species], dtype=np.float32)
            parts.append(f"gas={gas_species}:{float(values[values.shape[0] // 2, values.shape[1] // 2]):.2f}")

    optics_payload = payload.get("optics")
    if isinstance(optics_payload, dict):
        if light_type is not None:
            cell_dose = optics_payload.get("cell_dose")
            if isinstance(cell_dose, dict) and light_type in cell_dose:
                values = np.asarray(cell_dose[light_type], dtype=np.float32)
                parts.append(f"dose={light_type}:{float(values[values.shape[0] // 2, values.shape[1] // 2]):.2f}")
        else:
            visible = optics_payload.get("visible_illumination")
            if visible is not None:
                values = np.asarray(visible, dtype=np.float32)
                parts.append(_format_demo_probe_rgb("lit", values[values.shape[0] // 2, values.shape[1] // 2]))

    return " ".join(parts) if len(parts) > 1 else None


def build_demo_render_uniforms(engine: WorldEngine, *, debug_view: DebugView) -> dict[str, object]:
    return {
        "buffer_size": (engine.width, engine.height),
        "active_size": (engine.paging.active_width, engine.paging.active_height),
        "buffer_origin": (engine.paging.buffer_origin_x, engine.paging.buffer_origin_y),
        "world_origin": (engine.paging.origin_x, engine.paging.origin_y),
        "atlas_grid": tuple(engine.bridge.atlas_grid),
        "view_mode": 0 if debug_view == DebugView.MATERIAL else 1,
        "pattern_scale": DEMO_PATTERN_SCALE,
    }


def apply_demo_render_uniforms(program: Any, uniforms: dict[str, object]) -> None:
    for name, value in uniforms.items():
        program[name].value = value


def format_demo_status_title(
    *,
    debug_view: DebugView,
    brush_mode: str,
    brush_radius: int,
    selected_material: str,
    gas_species: str,
    light_type: str | None,
    paused: bool,
    speed: float,
    http_port: int,
    focus_x: int,
    focus_y: int,
    world_origin_x: int,
    world_origin_y: int,
    focus_probe_label: str | None = None,
    controller_label: str | None = None,
    cpu_fps: float | None = None,
    gpu_fps: float | None = None,
) -> str:
    parts = [
        ENGINE_DEMO_TITLE,
        f"view={debug_view.value}",
        f"focus=({int(focus_x)},{int(focus_y)})",
        f"origin=({int(world_origin_x)},{int(world_origin_y)})",
        f"brush={brush_mode}:{demo_brush_selection_label(brush_mode, selected_material=selected_material, gas_species=gas_species, light_type=light_type)}",
        f"r={max(0, int(brush_radius))}",
    ]
    if focus_probe_label is not None:
        parts.append(focus_probe_label)
    if controller_label is not None:
        parts.append(controller_label)
    if cpu_fps is not None:
        parts.append(f"cpu_fps={max(0.0, float(cpu_fps)):.1f}")
    if gpu_fps is not None:
        parts.append(f"gpu_fps={max(0.0, float(gpu_fps)):.1f}")
    parts.extend(
        [
            "paused" if paused else f"{float(speed):.2f}x",
            f"http={int(http_port)}",
        ]
    )
    view_focus = demo_view_focus_label(debug_view, gas_species=gas_species, light_type=light_type)
    if view_focus is not None:
        parts.insert(2, view_focus)
    return " | ".join(parts)


def main() -> None:
    try:
        import moderngl  # noqa: F401
        import moderngl_window as mglw
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install moderngl and moderngl-window inside the venv before running enginedemo.") from exc

    class EngineDemo(mglw.WindowConfig):
        gl_version = (4, 3)
        title = ENGINE_DEMO_TITLE
        window_size = (1440, 900)
        resizable = True
        aspect_ratio = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.engine = WorldEngine(width=256, height=192, gpu_context=self.ctx)
            self._bind_gui_gpu_context()
            self.state = EngineRunState()
            try:
                self.http = EngineHTTPConsole(self.engine, self.state, own_gpu_context=False)
            except TypeError as exc:
                if "own_gpu_context" not in str(exc):
                    raise
                self.http = EngineHTTPConsole(self.engine, self.state)
                setattr(self.http, "own_gpu_context", False)
            self.http.start()
            self.brush_radius = 3
            self.selected_material = MATERIAL_KEYS[0]
            self.brush_mode = BRUSH_MODES[0]
            self.debug_view = DebugView.MATERIAL
            self.gas_view_species = "water_gas"
            self.optics_view_light: str | None = None
            self.focus_x, self.focus_y = demo_default_focus_world(self.engine.paging)
            self.accumulator = 0.0
            self._status_title = ""
            self.controller_debug_enabled = False
            self.controller_debug_cycle = 0
            self.controller_debug_dirty = False
            self.controller_debug_label: str | None = None
            self.controller_debug_saved_state: Any = None
            self.cpu_fps = 0.0
            self.gpu_fps = 0.0
            self._cpu_fps_sample_time = _time.perf_counter()
            self._gpu_fps_sample_time = self._cpu_fps_sample_time
            self._cpu_frame_count = 0
            self._gpu_step_count = 0
            self._build_render_resources()
            self._refresh_status_title()

        def _bind_gui_gpu_context(self) -> None:
            bridge = self.engine.bridge
            if not hasattr(bridge, "ctx"):
                return
            if bridge.ctx is not self.ctx:
                bridge.attach_context(self.ctx)
                return
            bridge.enabled = True
            bridge.owner_thread_id = threading.get_ident()

        def _build_render_resources(self) -> None:
            self._bind_gui_gpu_context()
            self.program = self.ctx.program(
                vertex_shader=DEMO_VERTEX_SHADER_SOURCE,
                fragment_shader=DEMO_FRAGMENT_SHADER_SOURCE,
            )
            vertices = np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype="f4",
            )
            self.vbo = self.ctx.buffer(vertices.tobytes())
            self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "in_pos", "in_uv")])
            with self.engine.state_lock:
                self.engine.bridge.ensure_world_resources(self.engine)
                self.program["material_tex"] = 0
                self.program["light_tex"] = 1
                self.program["debug_tex"] = 2
                self.program["atlas_tex"] = 3
                apply_demo_render_uniforms(
                    self.program,
                    build_demo_render_uniforms(self.engine, debug_view=self.debug_view),
                )

        def on_render(self, time: float, frame_time: float) -> None:
            self._bind_gui_gpu_context()
            now = _time.perf_counter()
            self._record_cpu_frame(now)
            with self.engine.state_lock:
                stepped = False
                steps = 0
                if not self.state.paused:
                    self.accumulator += frame_time * max(0.1, self.state.speed)
                    while self.accumulator >= 1.0 / 60.0 and steps < 4:
                        self.engine.step(1.0 / 60.0)
                        self.accumulator -= 1.0 / 60.0
                        steps += 1
                        stepped = True
                elif self.state.single_step:
                    self.engine.step(1.0 / 60.0)
                    self.state.single_step = False
                    steps = 1
                    stepped = True
                self._record_gpu_steps(steps, _time.perf_counter())

                if self.controller_debug_enabled and (self.controller_debug_dirty or stepped):
                    self._run_demo_controller_cycle(apply_turn=stepped)

                debug = self.engine.debug_frame(
                    self.debug_view,
                    gas_species=self.gas_view_species,
                    light_type=self.optics_view_light,
                )
                self.engine.default_debug_view = self.debug_view
                self.engine.bridge.sync_world(self.engine, debug_frame=debug)

                apply_demo_render_uniforms(
                    self.program,
                    build_demo_render_uniforms(self.engine, debug_view=self.debug_view),
                )
                self.engine.bridge.texture("material").use(0)
                self.engine.bridge.texture("light").use(1)
                self.engine.bridge.texture("debug").use(2)
                self.engine.bridge.atlas_texture().use(3)
                self.ctx.clear(0.02, 0.03, 0.05)
                self.vao.render(mode=self.ctx.TRIANGLE_STRIP)
                self._refresh_status_title()

        def _record_cpu_frame(self, now: float) -> None:
            sample_time = float(getattr(self, "_cpu_fps_sample_time", now))
            frame_count = int(getattr(self, "_cpu_frame_count", 0)) + 1
            elapsed = max(0.0, float(now) - sample_time)
            if elapsed >= 0.5:
                self.cpu_fps = frame_count / elapsed
                self._cpu_frame_count = 0
                self._cpu_fps_sample_time = float(now)
                return
            self._cpu_frame_count = frame_count
            self._cpu_fps_sample_time = sample_time

        def _record_gpu_steps(self, steps: int, now: float) -> None:
            sample_time = float(getattr(self, "_gpu_fps_sample_time", now))
            step_count = int(getattr(self, "_gpu_step_count", 0)) + max(0, int(steps))
            elapsed = max(0.0, float(now) - sample_time)
            if elapsed >= 0.5:
                self.gpu_fps = step_count / elapsed
                self._gpu_step_count = 0
                self._gpu_fps_sample_time = float(now)
                return
            self._gpu_step_count = step_count
            self._gpu_fps_sample_time = sample_time

        def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
            self._paint_from_screen(x, y, dx=dx, dy=dy)

        def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
            if int(button) == 1:
                self._paint_from_screen(x, y, dx=0, dy=0)
                return
            self._focus_from_screen(x, y)

        def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
            with self.engine.state_lock:
                self.brush_radius = clamp_demo_brush_radius(self.brush_radius + int(y_offset))

        def on_key_event(self, key: int, action: int, modifiers: object) -> None:
            if action == 0:
                return
            with self.engine.state_lock:
                if (material := demo_material_for_key(key)) is not None:
                    self.selected_material = material
                elif (debug_view := demo_debug_view_for_key(key)) is not None:
                    self.debug_view = debug_view
                elif key == ord("["):
                    self.brush_radius = clamp_demo_brush_radius(self.brush_radius - 1)
                elif key == ord("]"):
                    self.brush_radius = clamp_demo_brush_radius(self.brush_radius + 1)
                elif key == ord("-"):
                    self.state.speed = max(0.1, self.state.speed * 0.8)
                elif key == ord("="):
                    self.state.speed = min(8.0, self.state.speed * 1.25)
                elif key == ord(" "):
                    self.state.paused = not self.state.paused
                elif key in (ord("N"), ord("n")):
                    self.state.single_step = True
                elif is_demo_optics_cycle_key(key):
                    self.optics_view_light = cycle_demo_named_choice(
                        self.optics_view_light,
                        (None, *self.engine.rulebook.lights_by_name.keys()),
                    )
                elif is_demo_gas_cycle_key(key):
                    gas_choices = tuple(name for name in self.engine.gas_name_by_id if name)
                    next_species = cycle_demo_named_choice(self.gas_view_species, gas_choices)
                    if next_species is not None:
                        self.gas_view_species = next_species
                elif is_demo_brush_cycle_key(key):
                    self.brush_mode = cycle_demo_brush_mode(self.brush_mode)
                elif is_demo_controller_toggle_key(key):
                    self._set_controller_debug_enabled(not self.controller_debug_enabled)
                elif is_demo_reset_key(key):
                    self.engine.reset_world()
                    self.focus_x, self.focus_y = demo_default_focus_world(self.engine.paging)
                    self.controller_debug_cycle = 0
                    self.controller_debug_label = None
                    self.controller_debug_saved_state = self.engine.serialize_controller_state()["controller_state"]
                    self.controller_debug_dirty = self.controller_debug_enabled
                elif key in (ord("W"), ord("w")):
                    self._move_focus(0, -self.engine.paging.tile_size)
                elif key in (ord("S"), ord("s")):
                    self._move_focus(0, self.engine.paging.tile_size)
                elif key in (ord("A"), ord("a")):
                    self._move_focus(-self.engine.paging.tile_size, 0)
                elif key in (ord("D"), ord("d")):
                    self._move_focus(self.engine.paging.tile_size, 0)

        def close(self) -> None:  # pragma: no cover
            self.http.stop()
            with self.engine.state_lock:
                self.engine.close()
            close = getattr(super(), "close", None)
            if callable(close):
                close()

        def _refresh_status_title(self) -> None:
            focus_probe_label = format_demo_focus_probe(
                self.engine,
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                debug_view=self.debug_view,
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
            )
            title = format_demo_status_title(
                debug_view=self.debug_view,
                brush_mode=self.brush_mode,
                brush_radius=self.brush_radius,
                selected_material=self.selected_material,
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
                paused=self.state.paused,
                speed=self.state.speed,
                http_port=self.http.port,
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                world_origin_x=self.engine.paging.origin_x,
                world_origin_y=self.engine.paging.origin_y,
                focus_probe_label=focus_probe_label,
                controller_label=self.controller_debug_label,
                cpu_fps=getattr(self, "cpu_fps", 0.0),
                gpu_fps=getattr(self, "gpu_fps", 0.0),
            )
            if title == self._status_title:
                return
            self._status_title = title
            try:
                self.wnd.title = title
            except Exception:
                pass

        def _paint_from_screen(self, x: int, y: int, *, dx: int, dy: int) -> None:
            with self.engine.state_lock:
                self._bind_gui_gpu_context()
                world_x, world_y = demo_screen_to_world_cell(
                    x,
                    y,
                    screen_width=self.wnd.width,
                    screen_height=self.wnd.height,
                    active_width=self.engine.paging.active_width,
                    active_height=self.engine.paging.active_height,
                    world_origin_x=self.engine.paging.origin_x,
                    world_origin_y=self.engine.paging.origin_y,
                )
                apply_demo_paint(
                    self.engine,
                    self.brush_mode,
                    x=world_x,
                    y=world_y,
                    selected_material=self.selected_material,
                    gas_species=self.gas_view_species,
                    light_type=self.optics_view_light,
                    radius=self.brush_radius,
                    dx=dx,
                    dy=dy,
                )

        def _focus_from_screen(self, x: int, y: int) -> None:
            with self.engine.state_lock:
                self._bind_gui_gpu_context()
                world_x, world_y = demo_screen_to_world_cell(
                    x,
                    y,
                    screen_width=self.wnd.width,
                    screen_height=self.wnd.height,
                    active_width=self.engine.paging.active_width,
                    active_height=self.engine.paging.active_height,
                    world_origin_x=self.engine.paging.origin_x,
                    world_origin_y=self.engine.paging.origin_y,
                )
                self.focus_x = int(world_x)
                self.focus_y = int(world_y)
                self.engine.advance_paging(self.focus_x, self.focus_y, immediate=True)
                self.controller_debug_dirty = self.controller_debug_enabled

        def _set_controller_debug_enabled(self, enabled: bool) -> None:
            enabled = bool(enabled)
            if enabled == self.controller_debug_enabled:
                return
            self.controller_debug_enabled = enabled
            self.controller_debug_label = None
            self.controller_debug_dirty = enabled
            self.controller_debug_cycle = 0
            if enabled:
                self.controller_debug_saved_state = self.engine.serialize_controller_state()["controller_state"]
                return

            surviving_entities = [
                replace(entity)
                for entity_id, entity in sorted(self.engine.entity_states.items())
                if entity_id != DEMO_CONTROLLER_ENTITY_ID
            ]
            self.engine.sync_entity_states(surviving_entities, immediate=True)
            current_state = self.engine.serialize_controller_state()["controller_state"]
            if isinstance(current_state, dict) and current_state.get("mode") == "demo_probe":
                self.engine.set_controller_state(self.controller_debug_saved_state)
            self.controller_debug_saved_state = None

        def _run_demo_controller_cycle(self, *, apply_turn: bool) -> None:
            entities = build_demo_controller_entities_for_world_focus(
                list(self.engine.entity_states.values()),
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                paging=self.engine.paging,
            )
            controller_state = build_demo_controller_state(
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                cycle=self.controller_debug_cycle,
            )
            cycle = self.engine.run_entity_controller_cycle(
                apply_turn=apply_turn,
                controller_state=controller_state,
                controller_state_provided=True,
                focus_center=(self.focus_x, self.focus_y),
                entities=entities,
            )
            status_label = format_demo_controller_status(
                preview=cycle["preview"],
                turn=cycle["result"],
                applied=cycle["applied"],
            )
            observation_summary = format_demo_controller_observation_summary(
                (cycle["result"] or cycle["preview"]).get("consumed", {}).get("observations"),
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
                material_name_by_id=self.engine,
            )
            self.controller_debug_label = (
                f"{status_label} {observation_summary}"
                if status_label is not None and observation_summary is not None
                else status_label or observation_summary
            )
            if cycle["applied"]:
                self.controller_debug_cycle += 1
            self.controller_debug_dirty = False

        def _move_focus(self, dx: int, dy: int) -> None:
            with self.engine.state_lock:
                self.focus_x += dx
                self.focus_y += dy
                self.engine.advance_paging(self.focus_x, self.focus_y, immediate=True)
                self.controller_debug_dirty = self.controller_debug_enabled

    mglw.run_window_config(EngineDemo)


if __name__ == "__main__":  # pragma: no cover
    main()
