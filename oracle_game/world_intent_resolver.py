from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    TargetQuery,
    WorldCommand,
)
from oracle_game.world_constants import TARGET_QUERY_DISTANCE_HINT_CELLS

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _resolve_target_query_distance_cells(engine: "WorldEngine", query: TargetQuery) -> int:
    if int(query.distance_cells) != 0:
        return int(query.distance_cells)
    if query.distance_meters is not None:
        return engine._distance_meters_to_cells(float(query.distance_meters))
    if query.distance_hint is not None:
        return int(TARGET_QUERY_DISTANCE_HINT_CELLS.get(str(query.distance_hint).lower(), 0))
    return 0


def _resolve_target_query(engine: "WorldEngine", query: TargetQuery) -> ResolvedTarget:
    resolved_distance_cells = _resolve_target_query_distance_cells(engine, query)
    source_position = engine._resolve_query_source_position(query)
    if source_position is None:
        return ResolvedTarget(
            query_id=query.query_id,
            status="missing_source",
            anchor_filters=query.anchor_filters,
            direction=query.direction,
            distance_cells=resolved_distance_cells,
            distance_meters=query.distance_meters,
            distance_hint=query.distance_hint,
            label=query.label,
            note="source position could not be resolved",
        )

    source_world_position = engine._buffer_to_world_position(source_position)
    if query.source_entity_id is not None:
        entity = engine.entity_states.get(int(query.source_entity_id))
        if entity is not None:
            source_world_position = engine._entity_center_world_position(entity)
    anchor = engine._resolve_anchor_target(query, source_world_position)
    if anchor is None:
        return ResolvedTarget(
            query_id=query.query_id,
            status="missing_anchor",
            anchor_filters=query.anchor_filters,
            direction=query.direction,
            distance_cells=resolved_distance_cells,
            distance_meters=query.distance_meters,
            distance_hint=query.distance_hint,
            label=query.label,
            source_position=source_position,
            source_world_position=source_world_position,
            note="no matching anchor was found in the loaded world window",
        )

    resolved_world_position = anchor["world_position"]
    direction_vector = engine._query_direction_vector(query, source_entity_id=query.source_entity_id)
    if direction_vector is not None and resolved_distance_cells:
        resolved_world_position = engine._clamp_world_position(
            resolved_world_position[0] + direction_vector[0] * resolved_distance_cells,
            resolved_world_position[1] + direction_vector[1] * resolved_distance_cells,
        )

    if query.require_empty:
        empty_world_position = engine._find_nearest_empty_world_position(
            resolved_world_position,
            radius=max(0, int(query.search_radius)),
        )
        if empty_world_position is None:
            return ResolvedTarget(
                query_id=query.query_id,
                status="blocked",
                anchor_filters=query.anchor_filters,
                direction=query.direction,
                distance_cells=resolved_distance_cells,
                distance_meters=query.distance_meters,
                distance_hint=query.distance_hint,
                label=query.label,
                source_position=source_position,
                source_world_position=source_world_position,
                anchor_kind=str(anchor["kind"]),
                anchor_entity_id=anchor["entity_id"],
                anchor_position=anchor["buffer_position"],
                anchor_world_position=anchor["world_position"],
                note="no empty landing cell was found near the resolved target",
            )
        resolved_world_position = empty_world_position

    resolved_position = engine._world_to_buffer_clamped(*resolved_world_position)
    return ResolvedTarget(
        query_id=query.query_id,
        status="resolved",
        anchor_filters=query.anchor_filters,
        direction=query.direction,
        distance_cells=resolved_distance_cells,
        distance_meters=query.distance_meters,
        distance_hint=query.distance_hint,
        label=query.label,
        source_position=source_position,
        source_world_position=source_world_position,
        anchor_kind=str(anchor["kind"]),
        anchor_entity_id=anchor["entity_id"],
        anchor_position=anchor["buffer_position"],
        anchor_world_position=anchor["world_position"],
        resolved_position=resolved_position,
        resolved_world_position=resolved_world_position,
    )


def _resolve_target_queries(engine: "WorldEngine", queries: list[TargetQuery]) -> dict[str, ResolvedTarget]:
    resolved: dict[str, ResolvedTarget] = {}
    for query in queries:
        resolved[query.query_id] = _resolve_target_query(engine, query)
    return resolved


def _resolve_change_intent_world_position(
    engine: "WorldEngine",
    intent: ChangeIntent,
    resolved_targets: dict[str, ResolvedTarget],
) -> tuple[int, int] | None:
    return engine._resolve_intent_world_position(
        target_query_id=intent.target_query_id,
        center_x=intent.center_x,
        center_y=intent.center_y,
        target_dx=intent.target_dx,
        target_dy=intent.target_dy,
        resolved_targets=resolved_targets,
    )


def _resolve_change_intent(
    engine: "WorldEngine",
    intent: ChangeIntent,
    resolved_targets: dict[str, ResolvedTarget],
) -> ResolvedChangeIntent:
    fallback_mode = str(intent.fallback_mode or "nearest_empty").lower()
    if intent.material is None and not intent.temperature_delta and intent.velocity is None:
        return ResolvedChangeIntent(
            intent_id=intent.intent_id,
            status="empty",
            target_query_id=intent.target_query_id,
            label=intent.label,
            potency=float(intent.potency),
            stability=float(intent.stability),
            velocity_carrier=intent.velocity_carrier,
            velocity_mode=intent.velocity_mode,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            note="change intent had no material, temperature, or velocity edit",
        )

    world_position = _resolve_change_intent_world_position(engine, intent, resolved_targets)
    if world_position is None:
        return ResolvedChangeIntent(
            intent_id=intent.intent_id,
            status="missing_target",
            target_query_id=intent.target_query_id,
            label=intent.label,
            potency=float(intent.potency),
            stability=float(intent.stability),
            velocity_carrier=intent.velocity_carrier,
            velocity_mode=intent.velocity_mode,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            note="change intent target could not be resolved",
        )

    potency = max(0.0, float(intent.potency))
    stability = min(1.0, max(0.0, float(intent.stability)))
    effective_radius = max(0, int(round(int(intent.radius) * potency)))
    attempted_world_position = engine._apply_change_stability_drift(
        intent.intent_id,
        world_position,
        effective_radius=effective_radius,
        stability=stability,
    )
    scaled_temperature_delta = float(intent.temperature_delta) * potency
    scaled_velocity = None
    if intent.velocity is not None:
        scaled_velocity = (
            float(intent.velocity[0]) * potency,
            float(intent.velocity[1]) * potency,
        )

    drifted = attempted_world_position != world_position
    drift_note = "low stability introduced deterministic target drift" if drifted else None
    final_world_position, fallback_applied, fallback_note = engine._resolve_legal_world_position(
        attempted_world_position,
        require_empty=bool(intent.require_empty),
        fallback_mode=fallback_mode,
        fallback_radius=int(intent.fallback_radius),
        effective_radius=effective_radius,
        source_world_position=None,
    )
    attempted_position = engine._world_to_buffer_clamped(*attempted_world_position)
    if final_world_position is None:
        return ResolvedChangeIntent(
            intent_id=intent.intent_id,
            status="blocked",
            target_query_id=intent.target_query_id,
            label=intent.label,
            potency=potency,
            stability=stability,
            center_position=attempted_position,
            center_world_position=attempted_world_position,
            effective_radius=effective_radius,
            material=intent.material,
            temperature_delta=scaled_temperature_delta,
            velocity=scaled_velocity,
            velocity_carrier=intent.velocity_carrier,
            velocity_mode=intent.velocity_mode,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_applied=False,
            note=engine._combine_resolution_notes(drift_note, fallback_note),
        )

    center_position = engine._world_to_buffer_clamped(*final_world_position)
    effect_world_cells = engine._disk_world_cells(final_world_position, effective_radius)
    effect_cells = [engine._world_to_buffer_clamped(world_x, world_y) for world_x, world_y in effect_world_cells]

    generated_commands: list[WorldCommand] = []
    base_meta: dict[str, Any] = {"resolved_change_intent_id": intent.intent_id}
    if intent.target_query_id is not None:
        base_meta["resolved_target_query_id"] = intent.target_query_id
    if intent.material is not None:
        generated_commands.append(
            WorldCommand(
                kind="inject_material",
                payload={
                    "x": int(final_world_position[0]),
                    "y": int(final_world_position[1]),
                    "material": intent.material,
                    "radius": effective_radius,
                    **base_meta,
                },
            )
        )
    if scaled_temperature_delta != 0.0:
        generated_commands.append(
            WorldCommand(
                kind="inject_temperature",
                payload={
                    "x": int(final_world_position[0]),
                    "y": int(final_world_position[1]),
                    "delta": scaled_temperature_delta,
                    "radius": effective_radius,
                    **base_meta,
                },
            )
        )
    if scaled_velocity is not None:
        generated_commands.append(
            WorldCommand(
                kind="inject_velocity",
                payload={
                    "x": int(final_world_position[0]),
                    "y": int(final_world_position[1]),
                    "velocity": [float(scaled_velocity[0]), float(scaled_velocity[1])],
                    "radius": effective_radius,
                    "carrier": intent.velocity_carrier,
                    "mode": intent.velocity_mode,
                    **base_meta,
                },
            )
        )

    status = engine._intent_resolution_status(drifted=drifted, fallback_applied=fallback_applied)
    note = engine._combine_resolution_notes(drift_note, fallback_note)

    return ResolvedChangeIntent(
        intent_id=intent.intent_id,
        status=status,
        target_query_id=intent.target_query_id,
        label=intent.label,
        potency=potency,
        stability=stability,
        center_position=center_position,
        center_world_position=final_world_position,
        effective_radius=effective_radius,
        material=intent.material,
        temperature_delta=scaled_temperature_delta,
        velocity=scaled_velocity,
        velocity_carrier=intent.velocity_carrier,
        velocity_mode=intent.velocity_mode,
        require_empty=bool(intent.require_empty),
        fallback_mode=fallback_mode,
        fallback_applied=fallback_applied,
        effect_shape="burst",
        effect_cells=effect_cells,
        effect_bounds=engine._buffer_cell_bounds(effect_cells),
        generated_commands=generated_commands,
        note=note,
    )


def _resolve_carrier_intent(
    engine: "WorldEngine",
    intent: CarrierIntent,
    resolved_targets: dict[str, ResolvedTarget],
) -> ResolvedCarrierIntent:
    fallback_mode = str(intent.fallback_mode or "nearest_empty").lower()
    kind = intent.kind.lower()
    if kind not in {"material", "gas", "light", "force"}:
        return ResolvedCarrierIntent(
            intent_id=intent.intent_id,
            status="invalid_kind",
            kind=kind,
            target_query_id=intent.target_query_id,
            label=intent.label,
            release_mode=intent.release_mode,
            potency=float(intent.potency),
            stability=float(intent.stability),
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            note="unsupported carrier intent kind",
        )

    impact_world_position = engine._resolve_intent_world_position(
        target_query_id=intent.target_query_id,
        center_x=intent.center_x,
        center_y=intent.center_y,
        target_dx=intent.target_dx,
        target_dy=intent.target_dy,
        resolved_targets=resolved_targets,
    )
    if impact_world_position is None:
        return ResolvedCarrierIntent(
            intent_id=intent.intent_id,
            status="missing_target",
            kind=kind,
            target_query_id=intent.target_query_id,
            label=intent.label,
            release_mode=intent.release_mode,
            potency=float(intent.potency),
            stability=float(intent.stability),
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            note="carrier intent target could not be resolved",
        )

    potency = max(0.0, float(intent.potency))
    stability = min(1.0, max(0.0, float(intent.stability)))
    effective_radius = max(0, int(round(int(intent.radius) * potency)))
    attempted_world_position = engine._apply_change_stability_drift(
        intent.intent_id,
        impact_world_position,
        effective_radius=effective_radius,
        stability=stability,
    )
    source_position: tuple[int, int] | None = None
    source_world_position: tuple[int, int] | None = None
    if (
        intent.source_entity_id is not None
        or intent.source_x is not None
        or intent.source_y is not None
        or kind in {"light", "force"}
        or intent.release_mode != "impact"
        or (intent.require_empty and fallback_mode == "source")
    ):
        source_position, source_world_position = engine._resolve_intent_source_positions(
            source_entity_id=intent.source_entity_id,
            source_x=intent.source_x,
            source_y=intent.source_y,
        )

    drifted = attempted_world_position != impact_world_position
    drift_note = "low stability introduced deterministic impact drift" if drifted else None
    final_world_position, fallback_applied, fallback_note = engine._resolve_legal_world_position(
        attempted_world_position,
        require_empty=bool(intent.require_empty),
        fallback_mode=fallback_mode,
        fallback_radius=int(intent.fallback_radius),
        effective_radius=effective_radius,
        source_world_position=source_world_position,
    )
    attempted_position = engine._world_to_buffer_clamped(*attempted_world_position)
    if final_world_position is None:
        return ResolvedCarrierIntent(
            intent_id=intent.intent_id,
            status="blocked",
            kind=kind,
            target_query_id=intent.target_query_id,
            label=intent.label,
            release_mode=intent.release_mode,
            potency=potency,
            stability=stability,
            source_position=source_position,
            source_world_position=source_world_position,
            impact_position=attempted_position,
            impact_world_position=attempted_world_position,
            effective_radius=effective_radius,
            material=intent.material,
            gas_species=intent.gas_species,
            gas_amount=float(intent.gas_amount) * potency if kind == "gas" else 0.0,
            light_type=intent.light_type,
            light_strength=float(intent.light_strength) * potency if kind == "light" else 0.0,
            light_spread=float(intent.light_spread),
            force_radius=float(intent.force_radius) * potency if kind == "force" else 0.0,
            force_strength=float(intent.force_strength) * potency if kind == "force" else 0.0,
            force_lifetime=float(intent.force_lifetime),
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_applied=False,
            note=engine._combine_resolution_notes(drift_note, fallback_note),
        )

    impact_position = engine._world_to_buffer_clamped(*final_world_position)

    direction = None
    if source_world_position is not None:
        direction = engine._normalized_world_direction(source_world_position, final_world_position)

    effect_shape = "impact"
    if intent.release_mode in {"beam", "projectile"} and source_world_position is not None:
        effect_shape = "beam"
        effect_world_cells = engine._capsule_world_cells(source_world_position, final_world_position, effective_radius)
        raw_effect_world_cells = engine._capsule_world_cells_raw(source_world_position, final_world_position, effective_radius)
    else:
        effect_world_cells = engine._disk_world_cells(final_world_position, effective_radius)
        raw_effect_world_cells = engine._disk_world_cells_raw(final_world_position, effective_radius)
    effect_cells = [engine._world_to_buffer_clamped(world_x, world_y) for world_x, world_y in effect_world_cells]

    generated_commands: list[WorldCommand] = []
    base_meta: dict[str, Any] = {"resolved_carrier_intent_id": intent.intent_id}
    if intent.target_query_id is not None:
        base_meta["resolved_target_query_id"] = intent.target_query_id

    note = engine._combine_resolution_notes(drift_note, fallback_note)
    status = engine._intent_resolution_status(drifted=drifted, fallback_applied=fallback_applied)

    resolved_force_radius = float(intent.force_radius) * potency if intent.force_radius > 0.0 else max(1.0, float(effective_radius))

    if kind == "material":
        if intent.material is None:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="invalid_payload",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=potency,
                stability=stability,
                source_position=source_position,
                source_world_position=source_world_position,
                impact_position=impact_position,
                impact_world_position=final_world_position,
                effective_radius=effective_radius,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=fallback_applied,
                note="material carrier intent requires material",
            )
        generated_commands.extend(
            WorldCommand(
                kind="inject_material",
                payload={
                    "x": int(cell[0]),
                    "y": int(cell[1]),
                    "material": intent.material,
                    "radius": 0 if effect_shape == "beam" else effective_radius,
                    **base_meta,
                },
            )
            for cell in (raw_effect_world_cells if effect_shape == "beam" else [final_world_position])
        )
    elif kind == "gas":
        if intent.gas_species is None or intent.gas_amount == 0.0:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="invalid_payload",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=potency,
                stability=stability,
                source_position=source_position,
                source_world_position=source_world_position,
                impact_position=impact_position,
                impact_world_position=final_world_position,
                effective_radius=effective_radius,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=fallback_applied,
                note="gas carrier intent requires gas_species and non-zero gas_amount",
            )
        scaled_amount = float(intent.gas_amount) * potency
        gas_cells = raw_effect_world_cells if effect_shape == "beam" else [final_world_position]
        per_cell_amount = scaled_amount / max(1, len(gas_cells))
        generated_commands.extend(
            WorldCommand(
                kind="inject_gas",
                payload={
                    "x": int(cell[0]),
                    "y": int(cell[1]),
                    "species": intent.gas_species,
                    "amount": per_cell_amount,
                    "radius": 0 if effect_shape == "beam" else effective_radius,
                    **base_meta,
                },
            )
            for cell in gas_cells
        )
    elif kind == "light":
        if intent.light_type is None:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="invalid_payload",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=potency,
                stability=stability,
                source_position=source_position,
                source_world_position=source_world_position,
                impact_position=impact_position,
                impact_world_position=final_world_position,
                effective_radius=effective_radius,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=fallback_applied,
                note="light carrier intent requires light_type",
            )
        light_strength = float(intent.light_strength) * potency
        origin_position = impact_position
        light_direction = (0.0, 0.0)
        light_range = max(1, effective_radius if effective_radius > 0 else int(intent.radius) or 1)
        if intent.release_mode in {"beam", "projectile"} and source_position is not None and source_world_position is not None:
            origin_position = source_position
            light_direction = (0.0, 0.0) if direction is None else direction
            if int(intent.radius) <= 0:
                light_range = max(
                    1,
                    int(round(np.linalg.norm(np.asarray(final_world_position, dtype=np.float32) - np.asarray(source_world_position, dtype=np.float32)))),
                )
        generated_commands.append(
            WorldCommand(
                kind="inject_light",
                payload={
                    "x": int(source_world_position[0] if intent.release_mode in {"beam", "projectile"} and source_world_position is not None else final_world_position[0]),
                    "y": int(source_world_position[1] if intent.release_mode in {"beam", "projectile"} and source_world_position is not None else final_world_position[1]),
                    "light_type": intent.light_type,
                    "strength": light_strength,
                    "radius": light_range,
                    "direction": light_direction,
                    "spread": float(intent.light_spread),
                    **base_meta,
                },
            )
        )
    else:
        if intent.force_strength == 0.0 and intent.force_radius == 0.0:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="invalid_payload",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=potency,
                stability=stability,
                source_position=source_position,
                source_world_position=source_world_position,
                impact_position=impact_position,
                impact_world_position=final_world_position,
                effective_radius=effective_radius,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=fallback_applied,
                note="force carrier intent requires non-zero force radius or strength",
            )
        if source_position is None or source_world_position is None:
            source_position, source_world_position = engine._resolve_intent_source_positions(
                source_entity_id=None,
                source_x=None,
                source_y=None,
            )
            direction = engine._normalized_world_direction(source_world_position, final_world_position)
        force_direction = (1.0, 0.0) if direction is None else direction
        generated_commands.append(
            WorldCommand(
                kind="inject_force",
                payload={
                    "x": int(source_world_position[0]),
                    "y": int(source_world_position[1]),
                    "direction": force_direction,
                    "radius": resolved_force_radius,
                    "strength": float(intent.force_strength) * potency,
                    "lifetime": float(intent.force_lifetime),
                    **base_meta,
                },
            )
        )

    return ResolvedCarrierIntent(
        intent_id=intent.intent_id,
        status=status,
        kind=kind,
        target_query_id=intent.target_query_id,
        label=intent.label,
        release_mode=intent.release_mode,
        potency=potency,
        stability=stability,
        source_position=source_position,
        source_world_position=source_world_position,
        impact_position=impact_position,
        impact_world_position=final_world_position,
        effective_radius=effective_radius,
        material=intent.material,
        gas_species=intent.gas_species,
        gas_amount=float(intent.gas_amount) * potency if kind == "gas" else 0.0,
        light_type=intent.light_type,
        light_strength=float(intent.light_strength) * potency if kind == "light" else 0.0,
        light_spread=float(intent.light_spread),
        force_radius=resolved_force_radius if kind == "force" else 0.0,
        force_strength=float(intent.force_strength) * potency if kind == "force" else 0.0,
        force_lifetime=float(intent.force_lifetime),
        direction=direction,
        require_empty=bool(intent.require_empty),
        fallback_mode=fallback_mode,
        fallback_applied=fallback_applied,
        effect_shape=effect_shape,
        effect_cells=effect_cells,
        effect_bounds=engine._buffer_cell_bounds(effect_cells),
        generated_commands=generated_commands,
        note=note,
    )
