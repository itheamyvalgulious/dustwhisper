from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from oracle_game.gpu import unpack_cell_core
from oracle_game.paging import RingPagingWindow
from oracle_game.types import DebugView, EntityState
from oracle_game.world import BASE_MATERIAL_RUNTIME_ALIASES, WorldEngine


ENGINE_DEMO_TITLE = "Dustwhisper Engine Demo"
DEMO_CONTROLLER_ENTITY_ID = 2_147_483_000
DEMO_TARGET_CELL_PIXELS = 1
DEMO_ACTIVE_SCALE = 1.5
DEMO_LOGICAL_WORLD_SCALE = 20


def compute_demo_grid_sizing(screen_width: int, screen_height: int) -> dict[str, int]:
    visible_width = max(64, int(screen_width) // DEMO_TARGET_CELL_PIXELS)
    visible_height = max(48, int(screen_height) // DEMO_TARGET_CELL_PIXELS)
    active_width = max(visible_width, int(round(visible_width * DEMO_ACTIVE_SCALE)))
    active_height = max(visible_height, int(round(visible_height * DEMO_ACTIVE_SCALE)))
    buffer_width = active_width
    buffer_height = active_height
    logical_world_width = visible_width * DEMO_LOGICAL_WORLD_SCALE
    logical_world_height = visible_height * DEMO_LOGICAL_WORLD_SCALE
    return {
        "visible_width": visible_width,
        "visible_height": visible_height,
        "active_width": active_width,
        "active_height": active_height,
        "buffer_width": buffer_width,
        "buffer_height": buffer_height,
        "logical_world_width": logical_world_width,
        "logical_world_height": logical_world_height,
    }


def demo_view_focus_label(debug_view: DebugView, *, gas_species: str, light_type: str | None) -> str | None:
    if debug_view == DebugView.GAS:
        return f"gas={gas_species}"
    if debug_view in {DebugView.OPTICS, DebugView.LIGHT}:
        return f"light={light_type or 'all'}"
    return None


def demo_display_material_name(material_name: str) -> str:
    runtime_name = str(material_name)
    for base_name, candidate in BASE_MATERIAL_RUNTIME_ALIASES.items():
        if candidate == runtime_name:
            return str(base_name)
    return runtime_name


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


def demo_backend_report(engine: WorldEngine) -> dict[str, object]:
    report_fn = getattr(engine, "simulation_backend_report", None)
    if callable(report_fn):
        try:
            return dict(report_fn())
        except AttributeError:
            pass
    return {
        "simulation_backend": str(getattr(engine, "simulation_backend", "")),
        "gpu_realtime_budget": {
            "enabled": bool(getattr(engine, "gpu_realtime_budget_enabled", False)),
            "active": False,
            "cell_threshold": int(getattr(engine, "gpu_realtime_budget_cell_threshold", 0)),
        },
    }


def request_demo_redraw(window: Any) -> None:
    raw_window = getattr(window, "_window", None)
    if raw_window is None:
        raw_window = window
    dispatch_events = getattr(raw_window, "dispatch_events", None)
    if callable(dispatch_events):
        try:
            dispatch_events()
        except Exception:
            pass
    invalidate = getattr(raw_window, "invalidate", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception:
            pass


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
    frame_ms: float | None = None,
    sim_ms: float | None = None,
    sync_ms: float | None = None,
    render_ms: float | None = None,
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
    if frame_ms is not None:
        parts.append(f"frame_ms={max(0.0, float(frame_ms)):.2f}")
    if sim_ms is not None:
        parts.append(f"sim_ms={max(0.0, float(sim_ms)):.2f}")
    if sync_ms is not None:
        parts.append(f"sync_ms={max(0.0, float(sync_ms)):.2f}")
    if render_ms is not None:
        parts.append(f"render_ms={max(0.0, float(render_ms)):.2f}")
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
