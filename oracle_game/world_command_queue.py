from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ForceSource,
    ObservationTarget,
    ReadbackRequest,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    TargetQuery,
    WorldCommand,
)
from oracle_game.world_constants import (
    PUBLIC_WORLD_COMMAND_KINDS,
    TARGETED_COMMAND_COORD_FIELDS,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def queue_command(engine: "WorldEngine", kind: str, **payload: Any) -> None:
    engine.command_queue.append(WorldCommand(kind=kind, payload=deepcopy(payload)))


def _resolve_direct_targeted_coords(
    engine: "WorldEngine",
    kind: str,
    x: int | None,
    y: int | None,
    *,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> tuple[int, int, str | None]:
    fields = TARGETED_COMMAND_COORD_FIELDS.get(kind)
    if fields is None:
        raise ValueError(f"unsupported direct target query kind '{kind}'")
    x_field, y_field = fields
    if target_queries is None:
        if x is None or y is None:
            raise ValueError(f"{x_field} and {y_field} are required unless target_queries resolve target_query_id")
        return int(x), int(y), None
    if target_query_id is None:
        raise ValueError("target_query_id is required when target_queries are provided")
    resolved_targets = engine._resolve_target_queries(
        [engine._coerce_target_query(query) for query in target_queries]
    )
    resolved_commands = engine._resolve_targeted_commands(
        [
            WorldCommand(
                kind=kind,
                payload={
                    "target_query_id": str(target_query_id),
                    "target_dx": int(target_dx),
                    "target_dy": int(target_dy),
                },
            )
        ],
        resolved_targets,
    )
    if not resolved_commands:
        raise ValueError(f"unable to resolve {kind} target query")
    payload = resolved_commands[0].payload
    return int(payload[x_field]), int(payload[y_field]), str(payload.get("resolved_target_query_id", target_query_id))


def inject_material(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    material: str,
    radius: int = 2,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_material",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        engine._apply_grid_world_commands(
            [WorldCommand(kind="inject_material", payload={"x": int(x), "y": int(y), "material": material, "radius": radius})]
        )
    else:
        payload: dict[str, Any] = {"x": x, "y": y, "material": material, "radius": radius}
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        queue_command(engine, "inject_material", **payload)


def write_material_region(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    width: int,
    height: int,
    material: str,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "write_material_region",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        engine._apply_grid_world_commands(
            [
                WorldCommand(
                    kind="write_material_region",
                    payload={"x": int(x), "y": int(y), "width": width, "height": height, "material": material},
                )
            ]
        )
    else:
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "material": material,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        queue_command(engine, "write_material_region", **payload)


def inject_temperature(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    delta: float,
    radius: int = 2,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_temperature",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        engine._apply_grid_world_commands(
            [WorldCommand(kind="inject_temperature", payload={"x": int(x), "y": int(y), "delta": delta, "radius": radius})]
        )
    else:
        payload: dict[str, Any] = {"x": x, "y": y, "delta": delta, "radius": radius}
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        queue_command(engine, "inject_temperature", **payload)


def inject_velocity(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    velocity: tuple[float, float],
    radius: int = 2,
    *,
    carrier: str = "cell",
    mode: str = "add",
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_velocity",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        engine._apply_grid_world_commands(
            [
                WorldCommand(
                    kind="inject_velocity",
                    payload={
                        "x": int(x),
                        "y": int(y),
                        "velocity": velocity,
                        "radius": radius,
                        "carrier": carrier,
                        "mode": mode,
                    },
                )
            ]
        )
    else:
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "velocity": velocity,
            "radius": radius,
            "carrier": carrier,
            "mode": mode,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        queue_command(engine, "inject_velocity", **payload)


def inject_force(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    direction: tuple[float, float],
    radius: float,
    strength: float,
    lifetime: float = 0.5,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_force",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        world_x = float(x)
        world_y = float(y)
        x, y = engine._world_to_buffer_clamped(int(x), int(y))
        engine._append_force_source_immediate(
            ForceSource(
                x=float(x),
                y=float(y),
                direction=(float(direction[0]), float(direction[1])),
                radius=float(radius),
                strength=float(strength),
                lifetime=float(lifetime),
                world_x=world_x,
                world_y=world_y,
            )
        )
        return
    payload: dict[str, Any] = {
        "x": x,
        "y": y,
        "direction": direction,
        "radius": radius,
        "strength": strength,
        "lifetime": lifetime,
    }
    if resolved_target_query_id is not None:
        payload["resolved_target_query_id"] = resolved_target_query_id
    queue_command(engine, "inject_force", **payload)


def inject_gas(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    species: str,
    amount: float,
    radius: int = 1,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_gas",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        engine._apply_grid_world_commands(
            [
                WorldCommand(
                    kind="inject_gas",
                    payload={"x": int(x), "y": int(y), "species": species, "amount": amount, "radius": radius},
                )
            ]
        )
    else:
        payload: dict[str, Any] = {"x": x, "y": y, "species": species, "amount": amount, "radius": radius}
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        queue_command(engine, "inject_gas", **payload)


def request_readback(
    engine: "WorldEngine",
    center_x: int | None,
    center_y: int | None,
    width: int,
    height: int,
    channels: tuple[str, ...],
    *,
    request_id: int | None = None,
    observer_id: int | None = None,
    label: str | None = None,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> int:
    request = ReadbackRequest(
        request_id=request_id,
        center_x=None if center_x is None else int(center_x),
        center_y=None if center_y is None else int(center_y),
        width=int(width),
        height=int(height),
        channels=tuple(channels),
        observer_id=observer_id,
        label=label,
        target_query_id=None if target_query_id is None else str(target_query_id),
        target_dx=int(target_dx),
        target_dy=int(target_dy),
    )
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
        resolved_request = engine._resolve_readback_request(request, resolved_targets)
        if resolved_request is None:
            raise ValueError("unable to resolve readback request target query")
        request = resolved_request
    request = engine._normalize_readback_request(request)
    if request.center_x is None or request.center_y is None:
        raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
    request = engine._assign_readback_request_id(request)
    queue_command(
        engine,
        "request_readback",
        request_id=request.request_id,
        center_x=request.center_x,
        center_y=request.center_y,
        width=request.width,
        height=request.height,
        channels=request.channels,
        observer_id=request.observer_id,
        label=request.label,
        target_query_id=request.target_query_id,
        target_dx=int(request.target_dx),
        target_dy=int(request.target_dy),
    )
    assert request.request_id is not None
    return int(request.request_id)


def preview_readback(
    engine: "WorldEngine",
    center_x: int | None,
    center_y: int | None,
    width: int,
    height: int,
    channels: tuple[str, ...],
    *,
    request_id: int | None = None,
    observer_id: int | None = None,
    label: str | None = None,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ReadbackRequest:
    request = ReadbackRequest(
        request_id=request_id,
        center_x=None if center_x is None else int(center_x),
        center_y=None if center_y is None else int(center_y),
        width=int(width),
        height=int(height),
        channels=tuple(channels),
        observer_id=observer_id,
        label=label,
        target_query_id=None if target_query_id is None else str(target_query_id),
        target_dx=int(target_dx),
        target_dy=int(target_dy),
    )
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
        resolved_request = engine._resolve_readback_request(request, resolved_targets)
        if resolved_request is None:
            raise ValueError("unable to resolve readback request target query")
        request = resolved_request
    request = engine._normalize_readback_request(request)
    if request.center_x is None or request.center_y is None:
        raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
    return request


def request_observation(
    engine: "WorldEngine",
    target: ObservationTarget | dict[str, Any],
    *,
    request_id: int | None = None,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> int:
    target = engine._coerce_observation_target(target)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    request = engine._build_observation_request(target, resolved_targets)
    if request is None:
        if target.target_query_id is not None and target_queries is not None:
            raise ValueError("unable to resolve observation target query")
        raise ValueError("unable to resolve observation target")
    if request_id is not None:
        request = replace(request, request_id=int(request_id))
    request = engine._assign_readback_request_id(request)
    queue_command(
        engine,
        "request_readback",
        request_id=request.request_id,
        center_x=request.center_x,
        center_y=request.center_y,
        width=request.width,
        height=request.height,
        channels=request.channels,
        observer_id=request.observer_id,
        label=request.label,
        target_query_id=request.target_query_id,
        target_dx=int(request.target_dx),
        target_dy=int(request.target_dy),
    )
    assert request.request_id is not None
    return int(request.request_id)


def preview_observation(
    engine: "WorldEngine",
    target: ObservationTarget | dict[str, Any],
    *,
    request_id: int | None = None,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ReadbackRequest:
    target = engine._coerce_observation_target(target)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    request = engine._build_observation_request(target, resolved_targets)
    if request is None:
        if target.target_query_id is not None and target_queries is not None:
            raise ValueError("unable to resolve observation target query")
        raise ValueError("unable to resolve observation target")
    if request_id is not None:
        request = replace(request, request_id=int(request_id))
    return request


def _resolve_public_world_command(
    engine: "WorldEngine",
    command: WorldCommand | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None,
    assign_readback_request_id: bool,
) -> WorldCommand:
    command = engine._coerce_world_command(command)
    if command.kind not in PUBLIC_WORLD_COMMAND_KINDS:
        raise ValueError(f"unsupported public world command kind '{command.kind}'")

    if command.kind == "request_readback":
        request = engine._coerce_readback_request(command.payload)
        if target_queries is not None:
            resolved_targets = engine._resolve_target_queries(
                [engine._coerce_target_query(query) for query in target_queries]
            )
            resolved_request = engine._resolve_readback_request(request, resolved_targets)
            if resolved_request is None:
                raise ValueError("unable to resolve world command target query")
            request = resolved_request
        elif request.target_query_id is not None and (request.center_x is None or request.center_y is None):
            raise ValueError("target_queries are required to resolve world command target_query_id")
        request = engine._normalize_readback_request(request)
        if request.center_x is None or request.center_y is None:
            raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
        if assign_readback_request_id:
            request = engine._assign_readback_request_id(request)
        return WorldCommand(
            kind="request_readback",
            payload={
                "request_id": request.request_id,
                "center_x": request.center_x,
                "center_y": request.center_y,
                "width": request.width,
                "height": request.height,
                "channels": request.channels,
                "observer_id": request.observer_id,
                "label": request.label,
                "target_query_id": request.target_query_id,
                "target_dx": int(request.target_dx),
                "target_dy": int(request.target_dy),
            },
        )

    if target_queries is None:
        if command.payload.get("target_query_id") is not None:
            raise ValueError("target_queries are required to resolve world command target_query_id")
        return command
    resolved_targets = engine._resolve_target_queries(
        [engine._coerce_target_query(query) for query in target_queries]
    )
    resolved_commands = engine._resolve_targeted_commands([command], resolved_targets)
    if not resolved_commands:
        raise ValueError("unable to resolve world command target query")
    return resolved_commands[0]


def _public_world_command(engine: "WorldEngine", command: WorldCommand) -> WorldCommand:
    payload = deepcopy(command.payload)
    if command.kind == "sync_entity_states" and isinstance(payload, dict):
        entities = payload.get("entities")
        if isinstance(entities, list):
            payload["entities"] = [
                engine.serialize_entity_state_input(
                    entity if isinstance(entity, EntityState) else engine._coerce_entity_state(entity)
                )
                for entity in entities
            ]
    elif command.kind == "patch_entity_states" and isinstance(payload, dict):
        patches = payload.get("patches")
        if isinstance(patches, list):
            payload["patches"] = [
                engine.serialize_entity_state_patch(
                    patch if isinstance(patch, EntityStatePatch) else engine._coerce_entity_state_patch(patch)
                )
                for patch in patches
            ]
    elif command.kind == "sync_entity_placeholders" and isinstance(payload, dict):
        placeholders = payload.get("placeholders")
        if isinstance(placeholders, list):
            payload["placeholders"] = [
                engine.serialize_entity_placeholder_input(
                    placeholder
                    if isinstance(placeholder, EntityPlaceholder)
                    else engine._coerce_entity_placeholder(placeholder)
                )
                for placeholder in placeholders
            ]
    return WorldCommand(kind=command.kind, payload=payload)


def preview_world_command(
    engine: "WorldEngine",
    command: WorldCommand | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> WorldCommand:
    command = engine._coerce_world_command(command)
    resolved_command = _resolve_public_world_command(
        engine,
        command,
        target_queries=target_queries,
        assign_readback_request_id=False,
    )
    return _public_world_command(engine, resolved_command)


def preview_target_queries(
    engine: "WorldEngine",
    target_queries: list[TargetQuery | dict[str, Any]],
) -> dict[str, ResolvedTarget]:
    return {
        query_id: engine._public_resolved_target(target)
        for query_id, target in engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        ).items()
    }


def request_world_command(
    engine: "WorldEngine",
    command: WorldCommand | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> WorldCommand:
    resolved_command = _resolve_public_world_command(
        engine,
        command,
        target_queries=target_queries,
        assign_readback_request_id=True,
    )
    queue_command(engine, resolved_command.kind, **resolved_command.payload)
    return _public_world_command(engine, resolved_command)


def preview_change_intent(
    engine: "WorldEngine",
    intent: ChangeIntent | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ResolvedChangeIntent:
    intent = engine._coerce_change_intent(intent)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    return _public_resolved_change_intent(engine, engine._resolve_change_intent(intent, resolved_targets))


def request_change_intent(
    engine: "WorldEngine",
    intent: ChangeIntent | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ResolvedChangeIntent:
    intent = engine._coerce_change_intent(intent)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    resolved_intent = engine._resolve_change_intent(intent, resolved_targets)
    for command in resolved_intent.generated_commands:
        queue_command(engine, command.kind, **command.payload)
    return _public_resolved_change_intent(engine, resolved_intent)


def preview_carrier_intent(
    engine: "WorldEngine",
    intent: CarrierIntent | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ResolvedCarrierIntent:
    intent = engine._coerce_carrier_intent(intent)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    return _public_resolved_carrier_intent(engine, engine._resolve_carrier_intent(intent, resolved_targets))


def request_carrier_intent(
    engine: "WorldEngine",
    intent: CarrierIntent | dict[str, Any],
    *,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> ResolvedCarrierIntent:
    intent = engine._coerce_carrier_intent(intent)
    resolved_targets: dict[str, ResolvedTarget] = {}
    if target_queries is not None:
        resolved_targets = engine._resolve_target_queries(
            [engine._coerce_target_query(query) for query in target_queries]
        )
    resolved_intent = engine._resolve_carrier_intent(intent, resolved_targets)
    for command in resolved_intent.generated_commands:
        queue_command(engine, command.kind, **command.payload)
    return _public_resolved_carrier_intent(engine, resolved_intent)


def inject_light(
    engine: "WorldEngine",
    x: int | None,
    y: int | None,
    light_type: str,
    strength: float,
    radius: int | None = None,
    *,
    direction: tuple[float, float] = (0.0, 0.0),
    spread: float = 0.25,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> None:
    x, y, resolved_target_query_id = _resolve_direct_targeted_coords(
        engine,
        "inject_light",
        x,
        y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    light_id = engine._resolve_sanctioned_light_id(light_type)
    if light_id < 0:
        raise KeyError(light_type)
    if radius is not None:
        resolved_radius = int(radius)
    else:
        shadow_default_range = engine._shadow_light_default_range(light_id)
        if shadow_default_range is None:
            raise KeyError(light_type)
        resolved_radius = int(shadow_default_range)
    if immediate:
        world_origin = (int(x), int(y))
        x, y = engine._world_to_buffer_clamped(int(x), int(y))
        shadow_light = engine._shadow_light_name(light_id)
        if shadow_light is None:
            raise KeyError(light_type)
        engine._append_transient_light_emitter_immediate(
            {
                "light_type": shadow_light,
                "origin": (int(x), int(y)),
                "world_origin": world_origin,
                "direction": (float(direction[0]), float(direction[1])),
                "spread": float(spread),
                "strength": float(strength),
                "range_cells": int(resolved_radius),
            }
        )
        return
    payload: dict[str, Any] = {
        "x": x,
        "y": y,
        "light_type": engine._shadow_light_name(light_id),
        "strength": strength,
        "radius": resolved_radius,
        "direction": direction,
        "spread": spread,
    }
    if resolved_target_query_id is not None:
        payload["resolved_target_query_id"] = resolved_target_query_id
    queue_command(engine, "inject_light", **payload)


def _public_resolved_carrier_intent(
    engine: "WorldEngine",
    intent: ResolvedCarrierIntent,
) -> ResolvedCarrierIntent:
    effect_cells: list[tuple[int, int]]
    if intent.effect_shape == "beam" and intent.source_world_position is not None and intent.impact_world_position is not None:
        effect_cells = engine._capsule_world_cells_raw(
            tuple(int(value) for value in intent.source_world_position),
            tuple(int(value) for value in intent.impact_world_position),
            int(intent.effective_radius),
        )
    elif intent.impact_world_position is not None:
        effect_cells = engine._disk_world_cells_raw(
            tuple(int(value) for value in intent.impact_world_position),
            int(intent.effective_radius),
        )
    else:
        effect_cells = [engine._buffer_to_world_position(cell) for cell in intent.effect_cells]
    effect_bounds = engine._buffer_cell_bounds(effect_cells)
    generated_commands = [_public_world_command(engine, command) for command in intent.generated_commands]
    if intent.kind == "light":
        origin_world_position = (
            tuple(int(value) for value in intent.source_world_position)
            if intent.effect_shape == "beam" and intent.source_world_position is not None
            else None
            if intent.impact_world_position is None
            else tuple(int(value) for value in intent.impact_world_position)
        )
        if origin_world_position is not None:
            for command in generated_commands:
                if command.kind == "inject_light":
                    command.payload["x"] = int(origin_world_position[0])
                    command.payload["y"] = int(origin_world_position[1])
    elif intent.kind == "force" and intent.source_world_position is not None:
        origin_world_position = tuple(int(value) for value in intent.source_world_position)
        for command in generated_commands:
            if command.kind == "inject_force":
                command.payload["x"] = int(origin_world_position[0])
                command.payload["y"] = int(origin_world_position[1])
    elif intent.kind in {"material", "gas"}:
        world_cells = (
            effect_cells
            if intent.effect_shape == "beam"
            else []
            if intent.impact_world_position is None
            else [tuple(int(value) for value in intent.impact_world_position)]
        )
        target_kind = "inject_material" if intent.kind == "material" else "inject_gas"
        rewritten = iter(world_cells)
        for command in generated_commands:
            if command.kind != target_kind:
                continue
            try:
                world_cell = next(rewritten)
            except StopIteration:
                break
            command.payload["x"] = int(world_cell[0])
            command.payload["y"] = int(world_cell[1])
    return replace(
        intent,
        source_position=(
            None
            if intent.source_world_position is None
            else tuple(int(value) for value in intent.source_world_position)
        ),
        impact_position=(
            None
            if intent.impact_world_position is None
            else tuple(int(value) for value in intent.impact_world_position)
        ),
        effect_cells=effect_cells,
        effect_bounds=effect_bounds,
        generated_commands=generated_commands,
    )
