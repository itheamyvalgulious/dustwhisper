from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    Direction,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ForceSource,
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    ObservationTarget,
    PairReactionRule,
    Phase,
    ReactionAction,
    ReactionType,
    ReadbackRequest,
    SelfReactionRule,
    TargetQuery,
    WorldCommand,
    WorldFrameInput,
)
from oracle_game.world_constants import (
    BASE_MATERIAL_RUNTIME_ALIASES,
    ENTITY_STATE_PATCH_METADATA_FIELDS,
    ENTITY_STATE_PATCHABLE_FIELDS,
    MAX_ASYNC_READBACK_HEIGHT,
    MAX_ASYNC_READBACK_WIDTH,
    READBACK_ALLOWED_CHANNEL_SET,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _coerce_enum(enum_type: type[Any], value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        if value in enum_type.__members__:
            return enum_type.__members__[value]
        return enum_type(value)
    return enum_type(value)


def _coerce_material_def(engine, material: MaterialDef | dict[str, Any]) -> MaterialDef:
    payload = asdict(material) if isinstance(material, MaterialDef) else dict(material)
    payload["default_phase"] = _coerce_enum(Phase, payload["default_phase"])
    payload["base_color"] = tuple(payload["base_color"])
    payload["reaction_slots"] = tuple(payload.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
    payload["tags"] = tuple(payload.get("tags", ()))
    payload["collapse_generation"] = _canonical_material_input_name(payload.get("collapse_generation"))
    payload["powder_generation"] = _canonical_material_input_name(payload.get("powder_generation"))
    payload["melt_to_material"] = _canonical_material_input_name(payload.get("melt_to_material"))
    payload["freeze_to_material"] = _canonical_material_input_name(payload.get("freeze_to_material"))
    return MaterialDef(**payload)


def _coerce_gas_species_def(engine, gas: GasSpeciesDef | dict[str, Any]) -> GasSpeciesDef:
    payload = asdict(gas) if isinstance(gas, GasSpeciesDef) else dict(gas)
    payload["color"] = tuple(payload["color"])
    payload["condense_to_material"] = _canonical_material_input_name(payload.get("condense_to_material"))
    return GasSpeciesDef(**payload)


def _coerce_light_type_def(engine, light: LightTypeDef | dict[str, Any]) -> LightTypeDef:
    if isinstance(light, LightTypeDef):
        return light
    payload = dict(light)
    payload["color"] = tuple(payload["color"])
    return LightTypeDef(**payload)


def _canonical_material_input_name(name: str | None) -> str | None:
    if name is None:
        return None
    if name == "__random__":
        return name
    return BASE_MATERIAL_RUNTIME_ALIASES.get(str(name), str(name))


def _coerce_material_optics_def(engine, optics: MaterialOpticsDef | dict[str, Any]) -> MaterialOpticsDef:
    payload = asdict(optics) if isinstance(optics, MaterialOpticsDef) else dict(optics)
    payload["material_name"] = _canonical_material_input_name(payload.get("material_name"))
    return MaterialOpticsDef(**payload)


def _coerce_reaction_action(engine, action: ReactionAction | dict[str, Any]) -> ReactionAction:
    payload = asdict(action) if isinstance(action, ReactionAction) else dict(action)
    payload["reaction_type"] = _coerce_enum(ReactionType, payload["reaction_type"])
    payload["direction"] = _coerce_enum(Direction, payload.get("direction", Direction.ALL))
    payload["velocity"] = tuple(payload.get("velocity", (0.0, 0.0)))
    payload["target_material"] = _canonical_material_input_name(payload.get("target_material"))
    payload["emit_material"] = _canonical_material_input_name(payload.get("emit_material"))
    return ReactionAction(**payload)


def _coerce_pair_reaction_rule(engine, rule: PairReactionRule | dict[str, Any]) -> PairReactionRule:
    payload = asdict(rule) if isinstance(rule, PairReactionRule) else dict(rule)
    payload["phases"] = tuple(_coerce_enum(Phase, phase) for phase in payload.get("phases", ()))
    payload["lhs_material"] = _canonical_material_input_name(payload.get("lhs_material"))
    payload["rhs_material"] = _canonical_material_input_name(payload.get("rhs_material"))
    return PairReactionRule(**payload)


def _coerce_self_reaction_rule(engine, rule: SelfReactionRule | dict[str, Any]) -> SelfReactionRule:
    payload = asdict(rule) if isinstance(rule, SelfReactionRule) else dict(rule)
    payload["phases"] = tuple(_coerce_enum(Phase, phase) for phase in payload.get("phases", ()))
    payload["material"] = _canonical_material_input_name(payload.get("material"))
    return SelfReactionRule(**payload)


def _coerce_reaction_rules(engine, rules: dict[str, object]) -> dict[str, list[object]]:
    return {
        "material_material": [_coerce_pair_reaction_rule(engine, rule) for rule in rules.get("material_material", [])],
        "material_gas": [_coerce_pair_reaction_rule(engine, rule) for rule in rules.get("material_gas", [])],
        "material_light": [_coerce_pair_reaction_rule(engine, rule) for rule in rules.get("material_light", [])],
        "gas_gas": [_coerce_pair_reaction_rule(engine, rule) for rule in rules.get("gas_gas", [])],
        "gas_light": [_coerce_pair_reaction_rule(engine, rule) for rule in rules.get("gas_light", [])],
        "self_rules": [_coerce_self_reaction_rule(engine, rule) for rule in rules.get("self_rules", [])],
    }


def _normalize_material_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_json_payload_value(
            str(_canonical_material_input_name(value))
            if key in {"collapse_generation", "powder_generation", "melt_to_material", "freeze_to_material"}
            else value
        )
        for key, value in fields.items()
    }


def _normalize_gas_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_json_payload_value(
            str(_canonical_material_input_name(value)) if key == "condense_to_material" else value
        )
        for key, value in fields.items()
    }


def _normalize_material_optics_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_json_payload_value(
            str(_canonical_material_input_name(value)) if key == "material_name" else value
        )
        for key, value in fields.items()
    }


def _normalize_reaction_action_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_json_payload_value(
            str(_canonical_material_input_name(value))
            if key in {"target_material", "emit_material"}
            else value
        )
        for key, value in fields.items()
    }


def _normalize_reaction_rule_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_json_payload_value(
            str(_canonical_material_input_name(value))
            if key in {"lhs_material", "rhs_material", "material"}
            else value
        )
        for key, value in fields.items()
    }


def _coerce_force_source(engine, force_source: ForceSource | dict[str, Any]) -> ForceSource:
    if isinstance(force_source, ForceSource):
        return force_source
    payload = dict(force_source)
    payload["direction"] = tuple(payload.get("direction", (0.0, 0.0)))
    return ForceSource(**payload)


def _public_force_source_input(engine, force_source: ForceSource | dict[str, Any]) -> ForceSource:
    force_source = _coerce_force_source(engine, force_source)
    world_x = float(force_source.x) if force_source.world_x is None else float(force_source.world_x)
    world_y = float(force_source.y) if force_source.world_y is None else float(force_source.world_y)
    return replace(force_source, world_x=world_x, world_y=world_y)


def _frame_force_source_input(engine, force_source: ForceSource | dict[str, Any]) -> ForceSource:
    force_source = _coerce_force_source(engine, force_source)
    if force_source.world_x is not None and force_source.world_y is not None:
        return replace(force_source, world_x=float(force_source.world_x), world_y=float(force_source.world_y))
    world_x, world_y = engine._buffer_to_world_float_position((float(force_source.x), float(force_source.y)))
    return replace(force_source, world_x=float(world_x), world_y=float(world_y))


def _coerce_emitter(engine, emitter: dict[str, Any]) -> dict[str, object]:
    payload = dict(emitter)
    light_id = engine._resolve_sanctioned_light_id(str(payload["light_type"]))
    if light_id < 0:
        raise KeyError(payload["light_type"])
    x = int(payload["x"]) if "x" in payload else int(payload["origin"][0])
    y = int(payload["y"]) if "y" in payload else int(payload["origin"][1])
    radius = payload.get("radius", payload.get("range_cells"))
    if radius is None:
        resolved_range = engine._shadow_light_default_range(light_id)
        if resolved_range is None:
            raise KeyError(payload["light_type"])
        radius = int(resolved_range)
    direction = tuple(float(value) for value in payload.get("direction", (0.0, 0.0)))
    shadow_light = engine._shadow_light_name(light_id)
    if shadow_light is None:
        raise KeyError(payload["light_type"])
    return {
        "light_type": shadow_light,
        "origin": (x, y),
        "world_origin": (
            (int(payload["world_origin"][0]), int(payload["world_origin"][1]))
            if "world_origin" in payload
            else (x, y)
        ),
        "direction": direction,
        "spread": float(payload.get("spread", 0.25)),
        "strength": float(payload.get("strength", 1.0)),
        "range_cells": int(radius),
    }


def _frame_emitter_input(engine, emitter: dict[str, Any]) -> dict[str, object]:
    record = _coerce_emitter(engine, emitter)
    if "world_origin" in emitter:
        return {
            **dict(record),
            "origin": engine._world_to_buffer_clamped(
                int(record["world_origin"][0]),
                int(record["world_origin"][1]),
            ),
        }
    origin = (int(record["origin"][0]), int(record["origin"][1]))
    return {
        **dict(record),
        "origin": origin,
        "world_origin": engine._buffer_to_world_position(origin),
    }


def _coerce_entity_placeholder(engine, placeholder: EntityPlaceholder | dict[str, Any]) -> EntityPlaceholder:
    if isinstance(placeholder, EntityPlaceholder):
        return replace(
            placeholder,
            entity_id=int(placeholder.entity_id),
            x=int(placeholder.x),
            y=int(placeholder.y),
            width=int(placeholder.width),
            height=int(placeholder.height),
            material=str(_canonical_material_input_name(placeholder.material)),
            world_x=None if placeholder.world_x is None else int(placeholder.world_x),
            world_y=None if placeholder.world_y is None else int(placeholder.world_y),
        )
    payload = dict(placeholder)
    return EntityPlaceholder(
        entity_id=int(payload["entity_id"]),
        x=int(payload["x"]),
        y=int(payload["y"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        material=str(_canonical_material_input_name(payload.get("material", "placeholder_solid"))),
        world_x=None if payload.get("world_x") is None else int(payload["world_x"]),
        world_y=None if payload.get("world_y") is None else int(payload["world_y"]),
    )


def _public_entity_placeholder_input(
    engine,
    placeholder: EntityPlaceholder | dict[str, Any],
) -> EntityPlaceholder:
    placeholder = _coerce_entity_placeholder(engine, placeholder)
    if placeholder.world_x is not None and placeholder.world_y is not None:
        world_x = int(placeholder.world_x)
        world_y = int(placeholder.world_y)
    elif 0 <= int(placeholder.x) < engine.width and 0 <= int(placeholder.y) < engine.height:
        world_x, world_y = engine._buffer_to_world_position((int(placeholder.x), int(placeholder.y)))
    else:
        world_x = int(placeholder.x)
        world_y = int(placeholder.y)
    return replace(placeholder, world_x=world_x, world_y=world_y)


def _frame_entity_placeholder_input(
    engine,
    placeholder: EntityPlaceholder | dict[str, Any],
) -> EntityPlaceholder:
    placeholder = _coerce_entity_placeholder(engine, placeholder)
    if placeholder.world_x is not None and placeholder.world_y is not None:
        buffer_x, buffer_y = engine._world_to_buffer_clamped(int(placeholder.world_x), int(placeholder.world_y))
        world_x = int(placeholder.world_x)
        world_y = int(placeholder.world_y)
    else:
        world_x = int(placeholder.x)
        world_y = int(placeholder.y)
        buffer_x, buffer_y = engine._world_to_buffer_clamped(world_x, world_y)
    return replace(
        placeholder,
        x=int(buffer_x),
        y=int(buffer_y),
        world_x=int(world_x),
        world_y=int(world_y),
    )


def _coerce_entity_state(engine, entity: EntityState | dict[str, Any]) -> EntityState:
    if isinstance(entity, EntityState):
        return replace(
            entity,
            entity_id=int(entity.entity_id),
            x=int(entity.x),
            y=int(entity.y),
            width=int(entity.width),
            height=int(entity.height),
            velocity_xy=(float(entity.velocity_xy[0]), float(entity.velocity_xy[1])),
            facing_xy=None if entity.facing_xy is None else (float(entity.facing_xy[0]), float(entity.facing_xy[1])),
            placeholder_material=str(_canonical_material_input_name(entity.placeholder_material)),
            tags=tuple(str(item) for item in entity.tags),
            observe_channels=_normalize_readback_channels(entity.observe_channels),
            observe_pad_cells=int(entity.observe_pad_cells),
            observe_width=None if entity.observe_width is None else int(entity.observe_width),
            observe_height=None if entity.observe_height is None else int(entity.observe_height),
            observe_label=None if entity.observe_label is None else str(entity.observe_label),
            world_x=None if entity.world_x is None else int(entity.world_x),
            world_y=None if entity.world_y is None else int(entity.world_y),
        )
    payload = dict(entity)
    payload["velocity_xy"] = tuple(payload.get("velocity_xy", (0.0, 0.0)))
    payload["facing_xy"] = None if payload.get("facing_xy") is None else tuple(payload["facing_xy"])
    payload["placeholder_material"] = str(
        _canonical_material_input_name(payload.get("placeholder_material", "placeholder_solid"))
    )
    payload["tags"] = tuple(payload.get("tags", ()))
    payload["observe_channels"] = _normalize_readback_channels(payload.get("observe_channels", ()))
    payload["world_x"] = None if payload.get("world_x") is None else int(payload["world_x"])
    payload["world_y"] = None if payload.get("world_y") is None else int(payload["world_y"])
    return EntityState(**payload)


def _public_entity_state_input(engine, entity: EntityState | dict[str, Any]) -> EntityState:
    entity = _coerce_entity_state(engine, entity)
    if entity.world_x is not None and entity.world_y is not None:
        world_x = int(entity.world_x)
        world_y = int(entity.world_y)
    elif 0 <= int(entity.x) < engine.width and 0 <= int(entity.y) < engine.height:
        world_x, world_y = engine._buffer_to_world_position((int(entity.x), int(entity.y)))
    else:
        world_x = int(entity.x)
        world_y = int(entity.y)
    return replace(entity, world_x=world_x, world_y=world_y)


def _frame_entity_state_input(engine, entity: EntityState | dict[str, Any]) -> EntityState:
    entity = _coerce_entity_state(engine, entity)
    if entity.world_x is not None and entity.world_y is not None:
        buffer_x, buffer_y = engine._world_to_buffer_clamped(int(entity.world_x), int(entity.world_y))
        world_x = int(entity.world_x)
        world_y = int(entity.world_y)
    else:
        buffer_x = int(entity.x)
        buffer_y = int(entity.y)
        world_x, world_y = engine._buffer_to_world_position((buffer_x, buffer_y))
    return replace(
        entity,
        x=int(buffer_x),
        y=int(buffer_y),
        world_x=int(world_x),
        world_y=int(world_y),
    )


def _coerce_entity_observation_spec(
    engine,
    spec: EntityObservationSpec | dict[str, Any],
) -> EntityObservationSpec:
    if isinstance(spec, EntityObservationSpec):
        return replace(
            spec,
            entity_id=int(spec.entity_id),
            observe_channels=_normalize_readback_channels(spec.observe_channels),
            observe_pad_cells=int(spec.observe_pad_cells),
            observe_width=None if spec.observe_width is None else int(spec.observe_width),
            observe_height=None if spec.observe_height is None else int(spec.observe_height),
            observe_label=None if spec.observe_label is None else str(spec.observe_label),
        )
    payload = dict(spec)
    return EntityObservationSpec(
        entity_id=int(payload["entity_id"]),
        observe_channels=_normalize_readback_channels(payload.get("observe_channels", ())),
        observe_pad_cells=int(payload.get("observe_pad_cells", 0)),
        observe_width=None if payload.get("observe_width") is None else int(payload["observe_width"]),
        observe_height=None if payload.get("observe_height") is None else int(payload["observe_height"]),
        observe_label=None if payload.get("observe_label") is None else str(payload["observe_label"]),
    )


def _normalize_entity_state_patch_fields(engine, fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in ENTITY_STATE_PATCHABLE_FIELDS and name not in ENTITY_STATE_PATCH_METADATA_FIELDS:
            raise KeyError(name)
        if name in {"x", "y", "width", "height", "observe_pad_cells", "_world_x", "_world_y"}:
            normalized[name] = int(value)
        elif name == "velocity_xy":
            normalized[name] = tuple(float(item) for item in value)
        elif name == "facing_xy":
            normalized[name] = None if value is None else tuple(float(item) for item in value)
        elif name == "placeholder_material":
            normalized[name] = str(_canonical_material_input_name(value))
        elif name == "tags":
            normalized[name] = tuple(str(item) for item in value)
        elif name == "observe_channels":
            normalized[name] = _normalize_readback_channels(value)
        elif name in {"observe_width", "observe_height"}:
            normalized[name] = None if value is None else int(value)
        elif name == "observe_label":
            normalized[name] = None if value is None else str(value)
    return normalized


def _public_entity_state_patch_input(
    engine,
    patch: EntityStatePatch | dict[str, Any],
) -> EntityStatePatch:
    patch = _coerce_entity_state_patch(engine, patch)
    fields = dict(patch.fields)
    if "x" in fields and "y" in fields:
        if 0 <= int(fields["x"]) < engine.width and 0 <= int(fields["y"]) < engine.height:
            world_x, world_y = engine._buffer_to_world_position((int(fields["x"]), int(fields["y"])))
            fields["_world_x"] = int(world_x)
            fields["_world_y"] = int(world_y)
        else:
            fields["_world_x"] = int(fields["x"])
            fields["_world_y"] = int(fields["y"])
    else:
        if "x" in fields:
            fields["_world_x"] = int(fields["x"])
        if "y" in fields:
            fields["_world_y"] = int(fields["y"])
    return EntityStatePatch(entity_id=int(patch.entity_id), fields=_normalize_entity_state_patch_fields(engine, fields))


def _controller_turn_entity_input(engine, entity: EntityState | dict[str, Any]) -> EntityState:
    entity = _coerce_entity_state(engine, entity)
    return replace(
        entity,
        world_x=None if entity.world_x is None else int(entity.world_x),
        world_y=None if entity.world_y is None else int(entity.world_y),
    )


def _frame_entity_state_patch_input(
    engine,
    patch: EntityStatePatch | dict[str, Any],
) -> EntityStatePatch:
    patch = _coerce_entity_state_patch(engine, patch)
    fields = dict(patch.fields)
    if "_world_x" in fields:
        buffer_x, _ = engine._world_to_buffer_clamped(int(fields["_world_x"]), int(fields.get("_world_y", 0)))
        fields["x"] = int(buffer_x)
    if "_world_y" in fields:
        _, buffer_y = engine._world_to_buffer_clamped(int(fields.get("_world_x", 0)), int(fields["_world_y"]))
        fields["y"] = int(buffer_y)
    return EntityStatePatch(entity_id=int(patch.entity_id), fields=_normalize_entity_state_patch_fields(engine, fields))


def _coerce_entity_state_patch(engine, patch: EntityStatePatch | dict[str, Any]) -> EntityStatePatch:
    if isinstance(patch, EntityStatePatch):
        return EntityStatePatch(
            entity_id=int(patch.entity_id),
            fields=_normalize_entity_state_patch_fields(engine, dict(patch.fields)),
        )
    payload = dict(patch)
    return EntityStatePatch(
        entity_id=int(payload["entity_id"]),
        fields=_normalize_entity_state_patch_fields(engine, dict(payload.get("fields", {}))),
    )


def _coerce_observation_target(engine, target: ObservationTarget | dict[str, Any]) -> ObservationTarget:
    if isinstance(target, ObservationTarget):
        return target
    payload = dict(target)
    return ObservationTarget(
        observer_id=int(payload["observer_id"]),
        channels=tuple(payload.get("channels", ())),
        center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
        center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
        width=None if payload.get("width") is None else int(payload["width"]),
        height=None if payload.get("height") is None else int(payload["height"]),
        entity_id=None if payload.get("entity_id") is None else int(payload["entity_id"]),
        pad_cells=int(payload.get("pad_cells", 0)),
        label=None if payload.get("label") is None else str(payload["label"]),
        target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
        target_dx=int(payload.get("target_dx", 0)),
        target_dy=int(payload.get("target_dy", 0)),
    )


def _coerce_target_query(engine, query: TargetQuery | dict[str, Any]) -> TargetQuery:
    if isinstance(query, TargetQuery):
        return query
    payload = dict(query)
    return TargetQuery(
        query_id=str(payload["query_id"]),
        anchor_filters=tuple(str(item) for item in payload.get("anchor_filters", ())),
        source_entity_id=None if payload.get("source_entity_id") is None else int(payload["source_entity_id"]),
        source_x=None if payload.get("source_x") is None else int(payload["source_x"]),
        source_y=None if payload.get("source_y") is None else int(payload["source_y"]),
        anchor_entity_id=None if payload.get("anchor_entity_id") is None else int(payload["anchor_entity_id"]),
        direction=None if payload.get("direction") is None else str(payload["direction"]),
        distance_cells=int(payload.get("distance_cells", 0)),
        distance_meters=None if payload.get("distance_meters") is None else float(payload["distance_meters"]),
        distance_hint=None if payload.get("distance_hint") is None else str(payload["distance_hint"]),
        require_empty=bool(payload.get("require_empty", False)),
        search_radius=int(payload.get("search_radius", 0)),
        label=None if payload.get("label") is None else str(payload["label"]),
    )


def _coerce_change_intent(engine, intent: ChangeIntent | dict[str, Any]) -> ChangeIntent:
    if isinstance(intent, ChangeIntent):
        return intent
    payload = dict(intent)
    return ChangeIntent(
        intent_id=str(payload["intent_id"]),
        target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
        center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
        center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
        target_dx=int(payload.get("target_dx", 0)),
        target_dy=int(payload.get("target_dy", 0)),
        radius=int(payload.get("radius", 0)),
        material=None if payload.get("material") is None else str(_canonical_material_input_name(payload["material"])),
        temperature_delta=float(payload.get("temperature_delta", 0.0)),
        velocity=None if payload.get("velocity") is None else tuple(float(value) for value in payload["velocity"]),
        velocity_carrier=str(payload.get("velocity_carrier", "cell")),
        velocity_mode=str(payload.get("velocity_mode", "add")),
        require_empty=bool(payload.get("require_empty", False)),
        fallback_mode=str(payload.get("fallback_mode", "nearest_empty")),
        fallback_radius=int(payload.get("fallback_radius", 0)),
        potency=float(payload.get("potency", 1.0)),
        stability=float(payload.get("stability", 1.0)),
        label=None if payload.get("label") is None else str(payload["label"]),
    )


def _coerce_carrier_intent(engine, intent: CarrierIntent | dict[str, Any]) -> CarrierIntent:
    if isinstance(intent, CarrierIntent):
        return intent
    payload = dict(intent)
    return CarrierIntent(
        intent_id=str(payload["intent_id"]),
        kind=str(payload["kind"]),
        target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
        center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
        center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
        source_entity_id=None if payload.get("source_entity_id") is None else int(payload["source_entity_id"]),
        source_x=None if payload.get("source_x") is None else int(payload["source_x"]),
        source_y=None if payload.get("source_y") is None else int(payload["source_y"]),
        target_dx=int(payload.get("target_dx", 0)),
        target_dy=int(payload.get("target_dy", 0)),
        radius=int(payload.get("radius", 0)),
        material=None if payload.get("material") is None else str(_canonical_material_input_name(payload["material"])),
        gas_species=None if payload.get("gas_species") is None else str(payload["gas_species"]),
        gas_amount=float(payload.get("gas_amount", 0.0)),
        light_type=None if payload.get("light_type") is None else str(payload["light_type"]),
        light_strength=float(payload.get("light_strength", 1.0)),
        light_spread=float(payload.get("light_spread", 0.25)),
        force_radius=float(payload.get("force_radius", 0.0)),
        force_strength=float(payload.get("force_strength", 0.0)),
        force_lifetime=float(payload.get("force_lifetime", 0.5)),
        release_mode=str(payload.get("release_mode", "impact")),
        require_empty=bool(payload.get("require_empty", False)),
        fallback_mode=str(payload.get("fallback_mode", "nearest_empty")),
        fallback_radius=int(payload.get("fallback_radius", 0)),
        potency=float(payload.get("potency", 1.0)),
        stability=float(payload.get("stability", 1.0)),
        label=None if payload.get("label") is None else str(payload["label"]),
    )


def _coerce_readback_request(engine, request: ReadbackRequest | dict[str, Any]) -> ReadbackRequest:
    if isinstance(request, ReadbackRequest):
        return _normalize_readback_request(engine, replace(request))
    payload = dict(request)
    return _normalize_readback_request(engine,
        ReadbackRequest(
            request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
            center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
            center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
            width=int(payload.get("width", 1)),
            height=int(payload.get("height", 1)),
            channels=tuple(payload.get("channels", ())),
            observer_id=None if payload.get("observer_id") is None else int(payload["observer_id"]),
            label=None if payload.get("label") is None else str(payload["label"]),
            target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
            target_dx=int(payload.get("target_dx", 0)),
            target_dy=int(payload.get("target_dy", 0)),
        )
    )


def _normalize_readback_channels(channels: Any) -> tuple[str, ...]:
    return tuple(
        channel
        for channel in dict.fromkeys(str(channel) for channel in channels)
        if channel in READBACK_ALLOWED_CHANNEL_SET
    )


def _normalize_readback_request(engine, request: ReadbackRequest) -> ReadbackRequest:
    width = max(1, min(MAX_ASYNC_READBACK_WIDTH, int(request.width)))
    height = max(1, min(MAX_ASYNC_READBACK_HEIGHT, int(request.height)))
    channels = _normalize_readback_channels(request.channels)
    return ReadbackRequest(
        request_id=None if request.request_id is None else int(request.request_id),
        center_x=None if request.center_x is None else int(request.center_x),
        center_y=None if request.center_y is None else int(request.center_y),
        width=width,
        height=height,
        channels=channels,
        observer_id=request.observer_id,
        label=request.label,
        target_query_id=request.target_query_id,
        target_dx=int(request.target_dx),
        target_dy=int(request.target_dy),
    )


def _assign_readback_request_id(engine, request: ReadbackRequest) -> ReadbackRequest:
    if request.request_id is None:
        request_id = engine.next_readback_request_id
        engine.next_readback_request_id += 1
        engine.canceled_readback_request_ids.discard(int(request_id))
        return replace(request, request_id=int(request_id))
    request_id = int(request.request_id)
    engine.next_readback_request_id = max(engine.next_readback_request_id, request_id + 1)
    engine.canceled_readback_request_ids.discard(request_id)
    return replace(request, request_id=request_id)


def _assign_preview_readback_request_ids(
    engine,
    requests: list[ReadbackRequest],
    *,
    next_request_id: int | None = None,
) -> tuple[list[ReadbackRequest], int]:
    predicted_next = int(engine.next_readback_request_id if next_request_id is None else next_request_id)
    assigned_requests: list[ReadbackRequest] = []
    for request in requests:
        normalized_request = _normalize_readback_request(engine, request)
        if normalized_request.request_id is None:
            normalized_request = replace(normalized_request, request_id=int(predicted_next))
            predicted_next += 1
        else:
            request_id = int(normalized_request.request_id)
            normalized_request = replace(normalized_request, request_id=request_id)
            predicted_next = max(predicted_next, request_id + 1)
        assigned_requests.append(normalized_request)
    return assigned_requests, predicted_next


def _coerce_world_command(engine, command: WorldCommand | dict[str, Any]) -> WorldCommand:
    if isinstance(command, WorldCommand):
        kind = str(command.kind)
        payload = deepcopy(command.payload)
    else:
        raw = dict(command)
        kind = str(raw["kind"])
        payload = deepcopy(dict(raw.get("payload", {})))
    if kind in {"inject_material", "write_material_region"} and payload.get("material") is not None:
        payload["material"] = str(_canonical_material_input_name(payload["material"]))
    elif kind == "sync_entity_states" and isinstance(payload.get("entities"), list):
        payload["entities"] = [
            asdict(_public_entity_state_input(engine, entity))
            for entity in payload.get("entities", [])
        ]
    elif kind == "sync_entity_placeholders" and isinstance(payload.get("placeholders"), list):
        payload["placeholders"] = [
            asdict(_public_entity_placeholder_input(engine, placeholder))
            for placeholder in payload.get("placeholders", [])
        ]
    elif kind == "patch_entity_states" and isinstance(payload.get("patches"), list):
        payload["patches"] = [
            asdict(_public_entity_state_patch_input(engine, patch))
            for patch in payload.get("patches", [])
        ]
    return WorldCommand(kind=kind, payload=payload)


def _coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("controller_state keys must be strings")
            normalized[key] = _coerce_json_value(item)
        return normalized
    raise TypeError(f"controller_state must be JSON-serializable, got {type(value).__name__}")


def _normalize_json_payload_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [_normalize_json_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_payload_value(item)
            for key, item in value.items()
        }
    return deepcopy(value)


def _coerce_world_frame_input(engine, frame_input: WorldFrameInput | dict[str, Any]) -> WorldFrameInput:
    if isinstance(frame_input, WorldFrameInput):
        controller_state_provided = bool(
            frame_input.controller_state_provided or frame_input.controller_state is not None
        )
        return WorldFrameInput(
            submission_id=None if frame_input.submission_id is None else int(frame_input.submission_id),
            focus_center=(
                None
                if frame_input.focus_center is None
                else (int(frame_input.focus_center[0]), int(frame_input.focus_center[1]))
            ),
            controller_state=(
                _coerce_json_value(frame_input.controller_state)
                if controller_state_provided
                else None
            ),
            controller_state_provided=controller_state_provided,
            entities=None
            if frame_input.entities is None
            else [_public_entity_state_input(engine, entity) for entity in frame_input.entities],
            entity_placeholders=None
            if frame_input.entity_placeholders is None
            else [_public_entity_placeholder_input(engine, item) for item in frame_input.entity_placeholders],
            force_sources=None
            if frame_input.force_sources is None
            else [_public_force_source_input(engine, source) for source in frame_input.force_sources],
            emitters=None
            if frame_input.emitters is None
            else [_coerce_emitter(engine, emitter) for emitter in frame_input.emitters],
            target_queries=[_coerce_target_query(engine, query) for query in frame_input.target_queries],
            change_intents=[_coerce_change_intent(engine, intent) for intent in frame_input.change_intents],
            carrier_intents=[_coerce_carrier_intent(engine, intent) for intent in frame_input.carrier_intents],
            observation_targets=[
                _coerce_observation_target(engine, target) for target in frame_input.observation_targets
            ],
            readback_requests=[
                _coerce_readback_request(engine, request) for request in frame_input.readback_requests
            ],
            commands=[_coerce_world_command(engine, command) for command in frame_input.commands],
        )
    payload = dict(frame_input)
    focus_center = payload.get("focus_center")
    controller_state_provided = bool(payload.get("controller_state_provided", False)) or "controller_state" in payload
    return WorldFrameInput(
        submission_id=None if payload.get("submission_id") is None else int(payload["submission_id"]),
        focus_center=None if focus_center is None else (int(focus_center[0]), int(focus_center[1])),
        controller_state=(
            _coerce_json_value(payload.get("controller_state"))
            if controller_state_provided
            else None
        ),
        controller_state_provided=controller_state_provided,
        entities=None
        if payload.get("entities") is None
        else [_public_entity_state_input(engine, entity) for entity in payload.get("entities", [])],
        entity_placeholders=None
        if payload.get("entity_placeholders") is None
        else [_public_entity_placeholder_input(engine, item) for item in payload.get("entity_placeholders", [])],
        force_sources=None
        if payload.get("force_sources") is None
        else [_public_force_source_input(engine, source) for source in payload.get("force_sources", [])],
        emitters=None
        if payload.get("emitters") is None
        else [_coerce_emitter(engine, emitter) for emitter in payload.get("emitters", [])],
        target_queries=[_coerce_target_query(engine, query) for query in payload.get("target_queries", [])],
        change_intents=[_coerce_change_intent(engine, intent) for intent in payload.get("change_intents", [])],
        carrier_intents=[_coerce_carrier_intent(engine, intent) for intent in payload.get("carrier_intents", [])],
        observation_targets=[_coerce_observation_target(engine, target) for target in payload.get("observation_targets", [])],
        readback_requests=[_coerce_readback_request(engine, request) for request in payload.get("readback_requests", [])],
        commands=[_coerce_world_command(engine, command) for command in payload.get("commands", [])],
    )
