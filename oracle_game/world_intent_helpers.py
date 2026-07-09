from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    EntityFeedback,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ForceSource,
    ObservationTarget,
    Phase,
    ReadbackRequest,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    TargetQuery,
    WorldCommand,
)
from oracle_game.world_constants import (
    CARDINAL_DIRECTION_VECTORS,
    IGNORED_ANCHOR_FILTERS,
    TARGET_QUERY_CELLS_PER_METER,
    TERRAIN_ANCHOR_FILTERS,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _patch_entity_states(engine: "WorldEngine", patches: list[EntityStatePatch]) -> None:
    next_entity_states = dict(engine.entity_states)
    for patch in patches:
        entity = next_entity_states.get(patch.entity_id)
        if entity is None:
            raise KeyError(patch.entity_id)
        patch_fields = {name: value for name, value in patch.fields.items() if not name.startswith("_")}
        world_x = patch.fields.get(
            "_world_x",
            patch_fields.get("x", entity.world_x if entity.world_x is not None else entity.x),
        )
        world_y = patch.fields.get(
            "_world_y",
            patch_fields.get("y", entity.world_y if entity.world_y is not None else entity.y),
        )
        next_entity_states[patch.entity_id] = engine._coerce_entity_state(
            replace(entity, **dict(patch_fields), world_x=int(world_x), world_y=int(world_y))
        )
    engine.entity_states = next_entity_states
    placeholders, _ = engine._frame_entities_to_placeholders_and_observations(list(engine.entity_states.values()))
    engine._sync_entity_placeholders(placeholders)


def _preview_can_occupy_placeholder_cell(
    engine: "WorldEngine",
    x: int,
    y: int,
    placeholder: EntityPlaceholder,
    current_cells: dict[tuple[int, int], int],
    released_cells: set[tuple[int, int]],
) -> bool:
    if not engine.in_bounds(x, y):
        return False
    placeholder_material_id = engine._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
    if placeholder_material_id <= 0:
        return False
    material_id, phase = _material_state_for_position(engine, x, y, released_cells=released_cells)
    if current_cells.get((x, y)) == placeholder.entity_id and material_id > 0 and engine._shadow_material_is_placeholder(material_id):
        return True
    return material_id == 0 or phase == int(Phase.LIQUID)


def _material_state_for_position(
    engine: "WorldEngine",
    x: int,
    y: int,
    *,
    blocked_cells: set[tuple[int, int]] | None = None,
    released_cells: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    material_id = int(engine.material_id[y, x])
    phase = int(engine.phase[y, x])
    cell = (x, y)
    if released_cells is not None and cell in released_cells and material_id > 0 and engine._shadow_material_is_placeholder(material_id):
        displaced_material = int(engine.placeholder_displaced_material[y, x])
        if displaced_material > 0:
            return displaced_material, int(Phase.LIQUID)
        return 0, 0
    if blocked_cells is not None and cell in blocked_cells:
        return int(engine.placeholder_material_id), int(Phase.STATIC_SOLID)
    return material_id, phase


def _build_observation_requests(
    engine: "WorldEngine",
    targets: list[ObservationTarget],
    resolved_targets: dict[str, ResolvedTarget],
) -> list[ReadbackRequest]:
    return [request for _, request in engine._build_observation_request_pairs(targets, resolved_targets)]


def _resolve_readback_requests(
    engine: "WorldEngine",
    requests: list[ReadbackRequest],
    resolved_targets: dict[str, ResolvedTarget],
) -> list[ReadbackRequest]:
    resolved: list[ReadbackRequest] = []
    for request in requests:
        concrete = engine._resolve_readback_request(request, resolved_targets)
        if concrete is not None:
            resolved.append(concrete)
    return resolved


def _resolve_change_intents(
    engine: "WorldEngine",
    intents: list[ChangeIntent],
    resolved_targets: dict[str, ResolvedTarget],
) -> tuple[dict[str, ResolvedChangeIntent], list[WorldCommand]]:
    resolved: dict[str, ResolvedChangeIntent] = {}
    commands: list[WorldCommand] = []
    for intent in intents:
        resolved_intent = engine._resolve_change_intent(intent, resolved_targets)
        resolved[intent.intent_id] = resolved_intent
        commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
    return resolved, commands


def _public_resolved_target(engine: "WorldEngine", target: ResolvedTarget) -> ResolvedTarget:
    return replace(
        target,
        source_position=(
            None
            if target.source_world_position is None
            else tuple(int(value) for value in target.source_world_position)
        ),
        anchor_position=(
            None
            if target.anchor_world_position is None
            else tuple(int(value) for value in target.anchor_world_position)
        ),
        resolved_position=(
            None
            if target.resolved_world_position is None
            else tuple(int(value) for value in target.resolved_world_position)
        ),
    )


def _resolve_carrier_intents(
    engine: "WorldEngine",
    intents: list[CarrierIntent],
    resolved_targets: dict[str, ResolvedTarget],
) -> tuple[dict[str, ResolvedCarrierIntent], list[WorldCommand]]:
    resolved: dict[str, ResolvedCarrierIntent] = {}
    commands: list[WorldCommand] = []
    for intent in intents:
        resolved_intent = engine._resolve_carrier_intent(intent, resolved_targets)
        resolved[intent.intent_id] = resolved_intent
        commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
    return resolved, commands


def _resolve_intent_world_position(
    engine: "WorldEngine",
    *,
    target_query_id: str | None,
    center_x: int | None,
    center_y: int | None,
    target_dx: int,
    target_dy: int,
    resolved_targets: dict[str, ResolvedTarget],
) -> tuple[int, int] | None:
    if target_query_id is not None:
        target = resolved_targets.get(target_query_id)
        if target is None or target.status not in {"resolved", "drifted"} or target.resolved_world_position is None:
            return None
        return (
            int(target.resolved_world_position[0]) + int(target_dx),
            int(target.resolved_world_position[1]) + int(target_dy),
        )
    if center_x is None or center_y is None:
        return None
    return (
        int(center_x) + int(target_dx),
        int(center_y) + int(target_dy),
    )


def _resolve_intent_source_positions(
    engine: "WorldEngine",
    *,
    source_entity_id: int | None,
    source_x: int | None,
    source_y: int | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if source_entity_id is not None:
        entity = engine.entity_states.get(int(source_entity_id))
        if entity is not None:
            world_position = _entity_center_world_position(engine, entity)
            return engine._world_to_buffer_clamped(*world_position), world_position
    if source_x is not None and source_y is not None:
        world_position = (int(source_x), int(source_y))
        buffer_position = engine._world_to_buffer_clamped(*world_position)
        return buffer_position, world_position
    buffer_position = _default_target_source_position(engine)
    return buffer_position, engine._buffer_to_world_position(buffer_position)


def _normalized_world_direction(
    source_world_position: tuple[int, int],
    target_world_position: tuple[int, int],
) -> tuple[float, float] | None:
    delta = np.asarray(target_world_position, dtype=np.float32) - np.asarray(source_world_position, dtype=np.float32)
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        return None
    direction = delta / length
    return (float(direction[0]), float(direction[1]))


def _intent_resolution_status(*, drifted: bool, fallback_applied: bool) -> str:
    if fallback_applied:
        return "fallback"
    if drifted:
        return "drifted"
    return "resolved"


def _combine_resolution_notes(*notes: str | None) -> str | None:
    filtered = [note for note in notes if note]
    if not filtered:
        return None
    return "; ".join(filtered)


def _distance_meters_to_cells(distance_meters: float) -> int:
    cells = int(round(float(distance_meters) * TARGET_QUERY_CELLS_PER_METER))
    if cells == 0 and abs(float(distance_meters)) > 1e-6:
        return 1 if distance_meters > 0.0 else -1
    return cells


def _resolve_query_source_position(engine: "WorldEngine", query: TargetQuery) -> tuple[int, int] | None:
    if query.source_entity_id is not None:
        entity = engine.entity_states.get(int(query.source_entity_id))
        if entity is None:
            return None
        return engine._world_to_buffer_clamped(*_entity_center_world_position(engine, entity))
    if query.source_x is not None and query.source_y is not None:
        return engine._world_to_buffer_clamped(int(query.source_x), int(query.source_y))
    if query.source_x is None and query.source_y is None:
        return _default_target_source_position(engine)
    return None


def _default_target_source_position(engine: "WorldEngine") -> tuple[int, int]:
    return (
        (int(engine.paging.buffer_origin_x) + int(engine.paging.active_width) // 2) % engine.width,
        (int(engine.paging.buffer_origin_y) + int(engine.paging.active_height) // 2) % engine.height,
    )


def _entity_matches_anchor_filters(engine: "WorldEngine", entity: EntityState, filters: tuple[str, ...]) -> bool:
    area = max(1, int(entity.width) * int(entity.height))
    entity_tags = set(entity.tags)
    for item in filters:
        if item in TERRAIN_ANCHOR_FILTERS or item in IGNORED_ANCHOR_FILTERS:
            continue
        if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}:
            continue
        if item == "big":
            if area < 4:
                return False
            continue
        if item == "small":
            if area > 1:
                return False
            continue
        if item not in entity_tags:
            return False
    return True


def _terrain_tree_cell_matches(engine: "WorldEngine", x: int, y: int, material_id: int, phase: int) -> bool:
    if material_id == 0 or phase in {int(Phase.LIQUID), int(Phase.POWDER)}:
        return False
    if not _world_cell_material_has_tag(engine, x, y, "plant"):
        return False
    if not (
        _world_cell_material_has_tag(engine, x, y - 1, "plant")
        or _world_cell_material_has_tag(engine, x, y + 1, "plant")
    ):
        return False
    plant_neighbors = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            if _world_cell_material_has_tag(engine, x + dx, y + dy, "plant"):
                plant_neighbors += 1
    return plant_neighbors >= 2


def _terrain_hill_cell_matches(engine: "WorldEngine", x: int, y: int, material_id: int, phase: int) -> bool:
    if material_id == 0 or phase == int(Phase.LIQUID):
        return False
    if _world_cell_material_has_tag(engine, x, y, "plant") or _world_cell_material_has_tag(engine, x, y, "placeholder"):
        return False
    if not engine._world_cell_is_empty_local(x, y - 1):
        return False
    if engine._world_cell_is_solid_local(x - 1, y) or engine._world_cell_is_solid_local(x + 1, y):
        return False
    left_support = engine._world_cell_is_solid_local(x - 1, y + 1) or engine._world_cell_is_solid_local(x - 1, y + 2)
    right_support = engine._world_cell_is_solid_local(x + 1, y + 1) or engine._world_cell_is_solid_local(x + 1, y + 2)
    return left_support and right_support


def _world_cell_material_has_tag(engine: "WorldEngine", x: int, y: int, tag: str) -> bool:
    material_id, _ = engine._bounded_material_state_for_position(x, y)
    if material_id == 0:
        return False
    material_id = int(material_id)
    if tag == "plant":
        return engine._shadow_material_is_plant(material_id)
    if tag == "placeholder":
        return engine._shadow_material_is_placeholder(material_id)
    material = engine._shadow_material_def(material_id)
    return material is not None and tag in material.tags


def _source_facing_vector(engine: "WorldEngine", source_entity_id: int | None) -> tuple[float, float]:
    if source_entity_id is not None:
        entity = engine.entity_states.get(int(source_entity_id))
        if entity is not None:
            if entity.facing_xy is not None:
                return (float(entity.facing_xy[0]), float(entity.facing_xy[1]))
            if entity.velocity_xy != (0.0, 0.0):
                return (float(entity.velocity_xy[0]), float(entity.velocity_xy[1]))
    return (1.0, 0.0)


def _entity_center_buffer_position(engine: "WorldEngine", entity: EntityState) -> tuple[int, int]:
    return (
        int(entity.x) + max(0, int(entity.width) - 1) // 2,
        int(entity.y) + max(0, int(entity.height) - 1) // 2,
    )


def _entity_center_world_position(engine: "WorldEngine", entity: EntityState) -> tuple[int, int]:
    if entity.world_x is not None and entity.world_y is not None:
        return (
            int(entity.world_x) + max(0, int(entity.width) - 1) // 2,
            int(entity.world_y) + max(0, int(entity.height) - 1) // 2,
        )
    return engine._buffer_to_world_position(_entity_center_buffer_position(engine, entity))


def _normalize_runtime_force_source(engine: "WorldEngine", force_source: ForceSource) -> ForceSource:
    world_x, world_y = engine._force_source_world_position(force_source)
    buffer_x, buffer_y = engine._world_to_buffer_float_position((world_x, world_y))
    return replace(
        force_source,
        x=float(buffer_x),
        y=float(buffer_y),
        world_x=float(world_x),
        world_y=float(world_y),
    )


def _world_gas_window_for_cell_world_rect(
    engine: "WorldEngine",
    world_x0: int,
    world_y0: int,
    world_x1: int,
    world_y1: int,
) -> tuple[int, int, int, int]:
    if world_x1 <= world_x0 or world_y1 <= world_y0:
        gas_world_x0 = int(world_x0) // int(engine.gas_cell_size)
        gas_world_y0 = int(world_y0) // int(engine.gas_cell_size)
        return (gas_world_x0, gas_world_y0, gas_world_x0, gas_world_y0)
    return (
        int(world_x0) // int(engine.gas_cell_size),
        int(world_y0) // int(engine.gas_cell_size),
        int((int(world_x1) + int(engine.gas_cell_size) - 1) // int(engine.gas_cell_size)),
        int((int(world_y1) + int(engine.gas_cell_size) - 1) // int(engine.gas_cell_size)),
    )


def _clamp_world_position(engine: "WorldEngine", world_x: int, world_y: int) -> tuple[int, int]:
    min_world_x = int(engine.paging.origin_x)
    min_world_y = int(engine.paging.origin_y)
    max_world_x = min_world_x + engine.width - 1
    max_world_y = min_world_y + engine.height - 1
    return (
        max(min_world_x, min(max_world_x, int(world_x))),
        max(min_world_y, min(max_world_y, int(world_y))),
    )


def _world_distance_sq(left: tuple[int, int], right: tuple[int, int]) -> float:
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    return dx * dx + dy * dy


def _entity_placeholder_bbox(engine: "WorldEngine", entity_id: int) -> tuple[int, int, int, int] | None:
    cells = engine.entity_placeholders.get(entity_id)
    if not cells:
        return None
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _build_entity_feedback_from_world(engine: "WorldEngine", entity: EntityState) -> EntityFeedback | None:
    cell_state = {
        "material_id": engine.material_id,
        "phase": engine.phase,
        "integrity": engine.integrity,
    }
    entity_runtime = {
        "entity_id": engine.entity_id,
        "placeholder_displaced_material": engine.placeholder_displaced_material,
    }
    return engine._build_entity_feedback_from_state(entity, cell_state=cell_state, entity_runtime=entity_runtime)


def _build_entity_feedback_from_current_state(
    engine: "WorldEngine",
    entity: EntityState,
    *,
    allow_gpu_sync_readback: bool = False,
) -> EntityFeedback | None:
    return engine._build_entity_feedback_from_state(
        entity,
        cell_state=engine._current_cell_state_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
        entity_runtime=engine._current_entity_runtime_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
    )
