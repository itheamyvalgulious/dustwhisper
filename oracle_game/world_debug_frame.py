from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.types import DebugView

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def debug_frame(
    engine: "WorldEngine",
    view: DebugView,
    *,
    gas_species: str | None = None,
    light_type: str | None = None,
) -> np.ndarray:
    if view == DebugView.MATERIAL:
        return _material_frame(engine)
    if view == DebugView.ACTIVE:
        return _active_frame(engine)
    if view == DebugView.TEMPERATURE:
        return _temperature_frame(engine)
    if view == DebugView.PRESSURE:
        return _pressure_frame(engine)
    if view == DebugView.HEAT:
        return _heat_frame(engine)
    if view == DebugView.LIQUID:
        return _liquid_frame(engine)
    if view == DebugView.REACTION:
        return _reaction_frame(engine)
    if view == DebugView.COLLAPSE:
        return _collapse_frame(engine)
    if view == DebugView.OPTICS:
        return _optics_frame(engine, light_type=light_type)
    if view == DebugView.VELOCITY:
        flow = engine.sample_flow_to_cells()
        vectors = flow.copy()
        cell_speed = np.linalg.norm(engine.velocity, axis=-1)
        use_cell_velocity = (engine.material_id > 0) & (cell_speed > 1.0e-6)
        vectors[use_cell_velocity] = engine.velocity[use_cell_velocity]
        return _vector_field_frame(engine, vectors)
    if view == DebugView.LIGHT:
        if light_type is not None:
            return _optics_dose_frame(engine, light_type=light_type)
        return np.clip(engine.visible_illumination, 0.0, 1.0)
    if view == DebugView.MOTION:
        return _motion_frame(engine)
    return _gas_frame(engine, gas_species or "water_gas")


def _material_frame(engine: "WorldEngine") -> np.ndarray:
    frame = engine.material_base_color[engine.material_id]
    frame = frame * 0.35 + np.clip(engine.visible_illumination, 0.0, 2.5)
    return np.clip(frame, 0.0, 1.0)


def _temperature_frame(engine: "WorldEngine") -> np.ndarray:
    temp = engine.cell_temperature
    normalized = np.clip((temp - temp.min()) / max(1e-5, temp.max() - temp.min()), 0.0, 1.0)
    return np.stack([normalized, np.zeros_like(normalized), 1.0 - normalized], axis=-1)


def _pressure_frame(engine: "WorldEngine") -> np.ndarray:
    pressure = engine.pressure_ping.astype(np.float32, copy=False)
    max_abs = float(
        max(
            1e-5,
            abs(float(pressure.min(initial=0.0))),
            abs(float(pressure.max(initial=0.0))),
        )
    )
    normalized = pressure / max_abs
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    magnitude = np.clip(np.abs(normalized), 0.0, 1.0)

    gas_frame = np.zeros((engine.gas_height, engine.gas_width, 3), dtype=np.float32)
    gas_frame[..., 0] = positive
    gas_frame[..., 1] = (1.0 - magnitude) * 0.18
    gas_frame[..., 2] = negative

    snapshot = engine.gas_solver.runtime_snapshot()
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
    if solve_gas_mask.size > 0:
        gas_frame += solve_gas_mask[..., None] * np.array([0.03, 0.12, 0.03], dtype=np.float32)

    frame = np.repeat(
        np.repeat(gas_frame, engine.gas_cell_size, axis=0),
        engine.gas_cell_size,
        axis=1,
    )[: engine.height, : engine.width]
    return np.clip(frame, 0.0, 1.0)


def _vector_field_frame(engine: "WorldEngine", vectors: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vectors, axis=-1)
    value = np.clip(magnitude / max(1e-5, float(magnitude.max(initial=0.0))), 0.0, 1.0)
    hue = (np.arctan2(vectors[..., 1], vectors[..., 0]) / (2.0 * np.pi) + 1.0) % 1.0
    hue6 = hue * 6.0
    sector = np.floor(hue6).astype(np.int32) % 6
    fraction = hue6 - np.floor(hue6)
    q = value * (1.0 - fraction)
    t = value * fraction
    rgb = np.zeros(vectors.shape[:-1] + (3,), dtype=np.float32)
    for index, components in (
        (0, (value, t, 0.0)),
        (1, (q, value, 0.0)),
        (2, (0.0, value, t)),
        (3, (0.0, q, value)),
        (4, (t, 0.0, value)),
        (5, (value, 0.0, q)),
    ):
        mask = sector == index
        if not np.any(mask):
            continue
        rgb[..., 0][mask] = components[0][mask] if isinstance(components[0], np.ndarray) else components[0]
        rgb[..., 1][mask] = components[1][mask] if isinstance(components[1], np.ndarray) else components[1]
        rgb[..., 2][mask] = components[2][mask] if isinstance(components[2], np.ndarray) else components[2]
    return rgb


def _active_frame(engine: "WorldEngine") -> np.ndarray:
    tile_ttl = np.asarray(engine.active.active_tile_ttl, dtype=np.float32)
    active_chunk_mask = np.asarray(engine.active.active_chunk_mask, dtype=np.float32)
    ttl_scale = max(1.0, float(engine.active.active_ttl_reset))
    ttl_cells = np.repeat(
        np.repeat(tile_ttl / ttl_scale, engine.active.tile_size, axis=0),
        engine.active.tile_size,
        axis=1,
    )[: engine.height, : engine.width]
    chunk_span = engine.active.tile_size * engine.active.chunk_tiles
    chunk_cells = np.repeat(
        np.repeat(active_chunk_mask, chunk_span, axis=0),
        chunk_span,
        axis=1,
    )[: engine.height, : engine.width]
    frame = np.zeros((engine.height, engine.width, 3), dtype=np.float32)
    frame[..., 0] = ttl_cells * 0.10
    frame[..., 1] = ttl_cells * 0.95
    frame[..., 2] = chunk_cells * 0.35
    pending_mask = engine.placeholder_displaced_material > 0
    if np.any(pending_mask):
        frame[pending_mask] = np.maximum(frame[pending_mask], np.array([0.95, 0.10, 0.95], dtype=np.float32))
    return np.clip(frame, 0.0, 1.0)


def _motion_frame(engine: "WorldEngine") -> np.ndarray:
    from oracle_game.sim.gpu_motion import (
        ISLAND_RESOLVE_BLOCKED,
        ISLAND_RESOLVE_DIRECT,
        ISLAND_RESOLVE_RERESOLVED,
        ISLAND_RESOLVE_STALE,
        POWDER_RESOLVE_BLOCKED,
        POWDER_RESOLVE_DDA,
        POWDER_RESOLVE_FALLBACK,
        POWDER_RESOLVE_STALE,
    )

    frame = np.clip(_material_frame(engine) * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
    snapshot = engine.motion_solver.runtime_snapshot()

    powder_state_colors = {
        POWDER_RESOLVE_BLOCKED: np.array([1.0, 0.15, 0.15], dtype=np.float32),
        POWDER_RESOLVE_DDA: np.array([0.15, 1.0, 0.2], dtype=np.float32),
        POWDER_RESOLVE_FALLBACK: np.array([0.20, 0.85, 1.0], dtype=np.float32),
        POWDER_RESOLVE_STALE: np.array([1.0, 0.2, 0.8], dtype=np.float32),
    }
    island_state_colors = {
        ISLAND_RESOLVE_BLOCKED: np.array([1.0, 0.15, 0.15], dtype=np.float32),
        ISLAND_RESOLVE_DIRECT: np.array([0.15, 1.0, 0.2], dtype=np.float32),
        ISLAND_RESOLVE_RERESOLVED: np.array([0.20, 0.85, 1.0], dtype=np.float32),
        ISLAND_RESOLVE_STALE: np.array([1.0, 0.2, 0.8], dtype=np.float32),
    }

    for record in snapshot["powder_reservations"]:
        source_x, source_y = (int(value) for value in record["source_xy"])
        reserved_x, reserved_y = (int(value) for value in record["reserved_target_xy"])
        resolved_x, resolved_y = (int(value) for value in record["resolved_target_xy"])
        resolve_state = int(record["resolve_state"])
        _accumulate_debug_point(engine, frame, source_x, source_y, np.array([0.95, 0.35, 0.10], dtype=np.float32))
        if (reserved_x, reserved_y) != (source_x, source_y):
            _accumulate_debug_point(engine, frame, reserved_x, reserved_y, np.array([0.95, 0.80, 0.15], dtype=np.float32))
        _accumulate_debug_point(
            engine,
            frame,
            resolved_x,
            resolved_y,
            powder_state_colors.get(resolve_state, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        )

    for record in snapshot["island_reservations"]:
        x0, y0, x1, y1 = (int(value) for value in record["buffer_bbox"])
        target_dx, target_dy = (int(value) for value in record["target_shift"])
        resolved_dx, resolved_dy = (int(value) for value in record["resolved_shift"])
        resolve_state = int(record["resolve_state"])
        _draw_debug_bbox_outline(engine, frame, (x0, y0, x1, y1), np.array([0.85, 0.20, 0.95], dtype=np.float32))
        if target_dx != 0 or target_dy != 0:
            _draw_debug_bbox_outline(
                engine,
                frame,
                (x0 + target_dx, y0 + target_dy, x1 + target_dx, y1 + target_dy),
                np.array([0.95, 0.80, 0.15], dtype=np.float32),
            )
        _draw_debug_bbox_outline(
            engine,
            frame,
            (x0 + resolved_dx, y0 + resolved_dy, x1 + resolved_dx, y1 + resolved_dy),
            island_state_colors.get(resolve_state, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        )
    return np.clip(frame, 0.0, 1.0)


def _heat_frame(engine: "WorldEngine") -> np.ndarray:
    frame = np.clip(_temperature_frame(engine) * 0.45, 0.0, 1.0).astype(np.float32, copy=False)
    snapshot = engine.heat_solver.runtime_snapshot()
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
    phase_targets = np.asarray(snapshot["phase_targets"], dtype=np.int32)
    boil_targets = np.asarray(snapshot["boil_targets"], dtype=np.int32)
    condense_targets = np.asarray(snapshot["condense_targets"], dtype=np.bool_)

    if solve_cell_mask.size > 0:
        frame += solve_cell_mask[..., None] * np.array([0.02, 0.18, 0.02], dtype=np.float32)
    if solve_gas_mask.size > 0:
        solve_gas_cells = np.repeat(
            np.repeat(solve_gas_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += solve_gas_cells[..., None] * np.array([0.08, 0.04, 0.18], dtype=np.float32)

    phase_ys, phase_xs = np.nonzero(phase_targets > 0)
    for y, x in zip(phase_ys.tolist(), phase_xs.tolist()):
        _accumulate_debug_point(engine, frame, int(x), int(y), np.array([0.95, 0.15, 0.95], dtype=np.float32))

    boil_ys, boil_xs = np.nonzero(boil_targets > 0)
    for y, x in zip(boil_ys.tolist(), boil_xs.tolist()):
        _accumulate_debug_point(engine, frame, int(x), int(y), np.array([1.0, 0.65, 0.05], dtype=np.float32))

    if condense_targets.size > 0:
        condense_any = np.any(condense_targets, axis=0).astype(np.float32, copy=False)
        condense_cells = np.repeat(
            np.repeat(condense_any, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += condense_cells[..., None] * np.array([0.14, 0.48, 0.62], dtype=np.float32)
    return np.clip(frame, 0.0, 1.0)


def _liquid_frame(engine: "WorldEngine") -> np.ndarray:
    frame = np.clip(_material_frame(engine) * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
    snapshot = engine.liquid_solver.runtime_snapshot()
    post_cell_mask = np.asarray(snapshot["post_cell_mask"], dtype=np.float32)
    vertical_seam_mask = np.asarray(snapshot["vertical_seam_mask"], dtype=np.bool_)
    horizontal_seam_mask = np.asarray(snapshot["horizontal_seam_mask"], dtype=np.bool_)
    buoyancy_mask = np.asarray(snapshot["buoyancy_mask"], dtype=np.bool_)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.bool_)

    if post_cell_mask.size > 0:
        frame += post_cell_mask[..., None] * np.array([0.02, 0.16, 0.06], dtype=np.float32)
    if np.any(changed_cell_mask):
        frame[changed_cell_mask] = np.maximum(frame[changed_cell_mask], np.array([0.18, 0.85, 1.0], dtype=np.float32))
    if np.any(vertical_seam_mask):
        frame[vertical_seam_mask] = np.maximum(frame[vertical_seam_mask], np.array([0.95, 0.18, 0.95], dtype=np.float32))
    if np.any(horizontal_seam_mask):
        frame[horizontal_seam_mask] = np.maximum(frame[horizontal_seam_mask], np.array([0.10, 0.92, 1.0], dtype=np.float32))
    if np.any(buoyancy_mask):
        frame[buoyancy_mask] = np.maximum(frame[buoyancy_mask], np.array([1.0, 0.78, 0.10], dtype=np.float32))
    pending_mask = engine.placeholder_displaced_material > 0
    if np.any(pending_mask):
        frame[pending_mask] = np.maximum(frame[pending_mask], np.array([1.0, 0.18, 0.18], dtype=np.float32))
    return np.clip(frame, 0.0, 1.0)


def _reaction_frame(engine: "WorldEngine") -> np.ndarray:
    frame = np.clip(_material_frame(engine) * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
    snapshot = engine.reaction_solver.runtime_snapshot()
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.bool_)
    changed_gas_mask = np.asarray(snapshot["changed_gas_mask"], dtype=np.float32)
    ambient_changed_mask = np.asarray(snapshot["ambient_changed_mask"], dtype=np.float32)
    timer_changed_mask = np.asarray(snapshot["timer_changed_mask"], dtype=np.bool_)
    emitted_light_mask = np.asarray(snapshot["emitted_light_mask"], dtype=np.bool_)
    emitted_material_mask = np.asarray(snapshot["emitted_material_mask"], dtype=np.bool_)

    if solve_cell_mask.size > 0:
        frame += solve_cell_mask[..., None] * np.array([0.02, 0.14, 0.02], dtype=np.float32)
    if solve_gas_mask.size > 0:
        solve_gas_cells = np.repeat(
            np.repeat(solve_gas_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += solve_gas_cells[..., None] * np.array([0.04, 0.08, 0.18], dtype=np.float32)
    if np.any(changed_cell_mask):
        frame[changed_cell_mask] = np.maximum(frame[changed_cell_mask], np.array([0.18, 0.88, 1.0], dtype=np.float32))
    if ambient_changed_mask.size > 0:
        ambient_cells = np.repeat(
            np.repeat(ambient_changed_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += ambient_cells[..., None] * np.array([0.92, 0.20, 0.92], dtype=np.float32)
    if changed_gas_mask.size > 0:
        gas_changed_cells = np.repeat(
            np.repeat(changed_gas_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += gas_changed_cells[..., None] * np.array([0.12, 0.48, 0.92], dtype=np.float32)
    if np.any(timer_changed_mask):
        frame[timer_changed_mask] = np.maximum(frame[timer_changed_mask], np.array([1.0, 0.82, 0.16], dtype=np.float32))
    if np.any(emitted_light_mask):
        frame[emitted_light_mask] = np.maximum(frame[emitted_light_mask], np.array([1.0, 0.42, 0.08], dtype=np.float32))
    if np.any(emitted_material_mask):
        frame[emitted_material_mask] = np.maximum(frame[emitted_material_mask], np.array([1.0, 0.18, 0.85], dtype=np.float32))
    return np.clip(frame, 0.0, 1.0)


def _collapse_frame(engine: "WorldEngine") -> np.ndarray:
    frame = np.clip(_material_frame(engine) * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
    snapshot = engine.collapse_solver.runtime_snapshot(engine)
    solve_region_mask = np.asarray(snapshot["solve_region_mask"], dtype=np.float32)
    support_seed_mask = np.asarray(snapshot["support_seed_mask"], dtype=np.bool_)
    supported_mask = np.asarray(snapshot["supported_mask"], dtype=np.bool_)
    unsupported_mask = np.asarray(snapshot["unsupported_mask"], dtype=np.bool_)
    delayed_pending_mask = np.asarray(snapshot["delayed_pending_mask"], dtype=np.bool_)
    immune_unsupported_mask = np.asarray(snapshot["immune_unsupported_mask"], dtype=np.bool_)
    collapsed_cell_mask = np.asarray(snapshot["collapsed_cell_mask"], dtype=np.bool_)

    if solve_region_mask.size > 0:
        frame += solve_region_mask[..., None] * np.array([0.04, 0.08, 0.18], dtype=np.float32)
    if np.any(supported_mask):
        frame[supported_mask] = np.maximum(frame[supported_mask], np.array([0.18, 0.82, 0.18], dtype=np.float32))
    if np.any(unsupported_mask):
        frame[unsupported_mask] = np.maximum(frame[unsupported_mask], np.array([1.0, 0.78, 0.10], dtype=np.float32))
    if np.any(delayed_pending_mask):
        frame[delayed_pending_mask] = np.maximum(frame[delayed_pending_mask], np.array([1.0, 0.46, 0.10], dtype=np.float32))
    if np.any(immune_unsupported_mask):
        frame[immune_unsupported_mask] = np.maximum(frame[immune_unsupported_mask], np.array([0.22, 0.78, 1.0], dtype=np.float32))
    if np.any(collapsed_cell_mask):
        frame[collapsed_cell_mask] = np.maximum(frame[collapsed_cell_mask], np.array([1.0, 0.20, 0.92], dtype=np.float32))
    if np.any(support_seed_mask):
        frame[support_seed_mask] = np.maximum(frame[support_seed_mask], np.array([0.16, 1.0, 1.0], dtype=np.float32))
    return np.clip(frame, 0.0, 1.0)


def _optics_frame(engine: "WorldEngine", *, light_type: str | None = None) -> np.ndarray:
    frame = np.zeros((engine.height, engine.width, 3), dtype=np.float32)
    snapshot = engine.optics_solver.runtime_snapshot()
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
    visible_changed_mask = np.asarray(snapshot["visible_changed_mask"], dtype=np.bool_)
    cell_dose_changed_mask = np.asarray(snapshot["cell_dose_changed_mask"], dtype=np.bool_)
    gas_dose_changed_mask = np.asarray(snapshot["gas_dose_changed_mask"], dtype=np.float32)
    emitter_origin_mask = np.asarray(snapshot["emitter_origin_mask"], dtype=np.bool_)
    dose_frame = _optics_dose_frame(engine, light_type=light_type)

    if solve_cell_mask.size > 0:
        frame += solve_cell_mask[..., None] * np.array([0.02, 0.12, 0.02], dtype=np.float32)
    if solve_gas_mask.size > 0:
        solve_gas_cells = np.repeat(
            np.repeat(solve_gas_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += solve_gas_cells[..., None] * np.array([0.03, 0.05, 0.16], dtype=np.float32)
    frame = np.maximum(frame, dose_frame)
    if light_type is None and np.any(visible_changed_mask):
        frame[visible_changed_mask] = np.maximum(
            frame[visible_changed_mask],
            np.clip(engine.visible_illumination[visible_changed_mask], 0.0, 1.0),
        )
    if light_type is None and np.any(cell_dose_changed_mask):
        frame[cell_dose_changed_mask] = np.maximum(frame[cell_dose_changed_mask], np.array([0.16, 0.95, 1.0], dtype=np.float32))
    if light_type is None and gas_dose_changed_mask.size > 0:
        gas_changed_cells = np.repeat(
            np.repeat(gas_dose_changed_mask, engine.gas_cell_size, axis=0),
            engine.gas_cell_size,
            axis=1,
        )[: engine.height, : engine.width]
        frame += gas_changed_cells[..., None] * np.array([0.14, 0.08, 0.42], dtype=np.float32)
    emitters = snapshot.get("emitters", [])
    if emitters:
        fallback_color = np.array([1.0, 0.42, 0.10], dtype=np.float32)
        for emitter in emitters:
            emitter_light = emitter.get("light_type")
            if light_type is not None and emitter_light != light_type:
                continue
            origin = emitter.get("origin")
            if not isinstance(origin, tuple | list) or len(origin) != 2:
                continue
            ox, oy = int(origin[0]), int(origin[1])
            light_id = engine._resolve_sanctioned_light_id(str(emitter_light))
            if light_id < 0:
                _accumulate_debug_point(engine, frame, ox, oy, fallback_color)
                continue
            shadow_color = engine._shadow_light_color(light_id)
            if shadow_color is None:
                _accumulate_debug_point(engine, frame, ox, oy, fallback_color)
                continue
            color = np.clip(shadow_color * 1.15, 0.0, 1.0)
            _accumulate_debug_point(engine, frame, ox, oy, color)
    elif np.any(emitter_origin_mask):
        frame[emitter_origin_mask] = np.maximum(frame[emitter_origin_mask], np.array([1.0, 0.42, 0.10], dtype=np.float32))
    return np.clip(frame, 0.0, 1.0)


def _optics_dose_frame(engine: "WorldEngine", *, light_type: str | None = None) -> np.ndarray:
    frame = np.zeros((engine.height, engine.width, 3), dtype=np.float32)
    if light_type is None:
        light_ids = [
            light_id
            for light_id in range(len(engine.light_name_by_id))
            if engine._shadow_light_name(light_id) is not None
        ]
    else:
        light_id = engine._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            return frame
        light_ids = [light_id]
    for light_id in light_ids:
        dose_channel = engine._shadow_light_dose_channel(light_id)
        color = engine._shadow_light_color(light_id)
        if (
            dose_channel is None
            or color is None
            or dose_channel < 0
            or dose_channel >= engine.cell_optical_dose.shape[0]
            or dose_channel >= engine.gas_optical_dose.shape[0]
        ):
            continue
        cell_strength = 1.0 - np.exp(-np.clip(engine.cell_optical_dose[dose_channel], 0.0, None))
        gas_strength = 1.0 - np.exp(
            -np.clip(
                np.repeat(
                    np.repeat(engine.gas_optical_dose[dose_channel], engine.gas_cell_size, axis=0),
                    engine.gas_cell_size,
                    axis=1,
                )[: engine.height, : engine.width],
                0.0,
                None,
            )
            * 1.25
        )
        frame += color * cell_strength[..., None]
        frame += color * gas_strength[..., None] * 0.65
    return np.clip(frame, 0.0, 1.0)


def _gas_frame(engine: "WorldEngine", gas_species: str) -> np.ndarray:
    species_id = engine._resolve_sanctioned_gas_id(gas_species)
    if species_id < 0:
        raise KeyError(gas_species)
    gas_field = engine.gas_concentration[species_id]
    gas_cells = np.repeat(np.repeat(gas_field, engine.gas_cell_size, axis=0), engine.gas_cell_size, axis=1)[: engine.height, : engine.width]
    normalized = np.clip(gas_cells / max(1e-5, gas_cells.max(initial=1.0)), 0.0, 1.0)
    frame = np.stack([normalized * 0.3, normalized, normalized * 0.6], axis=-1).astype(np.float32, copy=False)
    snapshot = engine.gas_solver.runtime_snapshot()
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
    if solve_gas_mask.size > 0:
        solve_cells = np.repeat(np.repeat(solve_gas_mask, engine.gas_cell_size, axis=0), engine.gas_cell_size, axis=1)[: engine.height, : engine.width]
        frame += solve_cells[..., None] * np.array([0.25, 0.05, 0.35], dtype=np.float32)
    for force in engine.force_sources:
        force_color = np.array([1.0, 0.45, 0.15], dtype=np.float32)
        center_x = int(round(force.x))
        center_y = int(round(force.y))
        _accumulate_debug_point(engine, frame, center_x, center_y, force_color)
        _accumulate_debug_point(engine, frame, center_x - 1, center_y, force_color)
        _accumulate_debug_point(engine, frame, center_x + 1, center_y, force_color)
        _accumulate_debug_point(engine, frame, center_x, center_y - 1, force_color)
        _accumulate_debug_point(engine, frame, center_x, center_y + 1, force_color)
    return np.clip(frame, 0.0, 1.0)


def _accumulate_debug_point(engine: "WorldEngine", frame: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
    if 0 <= x < engine.width and 0 <= y < engine.height:
        frame[y, x] = np.maximum(frame[y, x], color)


def _draw_debug_bbox_outline(
    engine: "WorldEngine",
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: np.ndarray,
) -> None:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(engine.width, x0))
    y0 = max(0, min(engine.height, y0))
    x1 = max(0, min(engine.width, x1))
    y1 = max(0, min(engine.height, y1))
    if x0 >= x1 or y0 >= y1:
        return
    frame[y0, x0:x1] = np.maximum(frame[y0, x0:x1], color)
    frame[y1 - 1, x0:x1] = np.maximum(frame[y1 - 1, x0:x1], color)
    frame[y0:y1, x0] = np.maximum(frame[y0:y1, x0], color)
    frame[y0:y1, x1 - 1] = np.maximum(frame[y0:y1, x1 - 1], color)
