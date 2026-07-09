from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np

from oracle_game.page_store import StoredStripeKey
from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    FallingIslandRecord,
    ForceSource,
    ObservationTarget,
    PageStripeUpdate,
    Phase,
    ReadbackRequest,
    ResolvedChangeIntent,
    TargetQuery,
    WorldCommand,
)
from oracle_game.world_constants import (
    CARDINAL_DIRECTION_VECTORS,
    IGNORED_ANCHOR_FILTERS,
    TARGETED_COMMAND_COORD_FIELDS,
    TERRAIN_ANCHOR_FILTERS,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _light_field_count(engine: "WorldEngine") -> int:
    max_light_id = max(engine.rulebook.lights_by_id, default=-1)
    max_dose_channel = max((int(light.dose_channel_id) for light in engine.rulebook.lights_by_id.values()), default=-1)
    return max(max_light_id, max_dose_channel) + 1


def readback_request_status(engine: "WorldEngine", request_id: int) -> str:
    if any(
        command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == int(request_id)
        for command in engine.command_queue
    ):
        return "queued"
    if any(
        any(request.request_id == int(request_id) for request in frame_input.readback_requests)
        for frame_input in engine.pending_frame_inputs
    ):
        return "pending_frame"
    if any(request.request_id == int(request_id) for request in engine.pending_readbacks):
        return "pending"
    if any(request.request_id == int(request_id) for request in engine.inflight_readbacks):
        return "inflight"
    if any(result.request.request_id == int(request_id) for result in engine.completed_readbacks):
        return "ready"
    if int(request_id) in engine.canceled_readback_request_ids:
        return "canceled"
    return "missing"


def _page_store_key_lookup_update(key: StoredStripeKey) -> PageStripeUpdate:
    return PageStripeUpdate(
        axis=str(key.axis),
        world_start=int(key.world_start),
        world_end=int(key.world_end),
        buffer_start=0,
        buffer_end=max(1, int(key.world_end) - int(key.world_start)),
        kind="load",
        cross_world_start=int(getattr(key, "cross_world_start", 0)),
        cross_world_end=int(getattr(key, "cross_world_end", 0)),
    )


def submit_entity_controller_turn(
    engine: "WorldEngine",
    *,
    controller_state: Any = None,
    controller_state_provided: bool = False,
    focus_center: tuple[int, int] | None = None,
    entities: list[EntityState | dict[str, Any]] | None = None,
    entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
    patches: list[EntityStatePatch | dict[str, Any]] | None = None,
    observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
    force_sources: list[ForceSource | dict[str, Any]] | None = None,
    emitters: list[dict[str, Any]] | None = None,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
    carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
    observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
    readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
    commands: list[WorldCommand | dict[str, Any]] | None = None,
) -> int:
    frame_input = engine.controller_turn_to_frame_input(
        controller_state=controller_state,
        controller_state_provided=controller_state_provided,
        focus_center=focus_center,
        entities=entities,
        entity_placeholders=entity_placeholders,
        patches=patches,
        observation_specs=observation_specs,
        force_sources=force_sources,
        emitters=emitters,
        target_queries=target_queries,
        change_intents=change_intents,
        carrier_intents=carrier_intents,
        observation_targets=observation_targets,
        readback_requests=readback_requests,
        commands=commands,
    )
    return engine.submit_frame_input(frame_input)


def _refresh_island_records_for_ids(engine: "WorldEngine", island_ids: Iterable[int]) -> None:
    touched = {int(island_id) for island_id in island_ids if int(island_id) > 0}
    if not touched:
        return
    for island_id in touched:
        invalid_mask = (engine.island_id == island_id) & (
            (engine.phase != int(Phase.FALLING_ISLAND)) | (engine.material_id <= 0)
        )
        if np.any(invalid_mask):
            engine.island_id[invalid_mask] = 0
        coords = np.argwhere(
            (engine.island_id == island_id)
            & (engine.phase == int(Phase.FALLING_ISLAND))
            & (engine.material_id > 0)
        )
        if coords.size == 0:
            engine.islands.pop(island_id, None)
            continue
        min_y, min_x = coords.min(axis=0).tolist()
        max_y, max_x = coords.max(axis=0).tolist()
        previous = engine.islands.get(island_id)
        if previous is None:
            velocity_xy = tuple(np.mean(engine.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32).tolist())
            subcell_offset = (0.0, 0.0)
        else:
            velocity_xy = (float(previous.velocity_xy[0]), float(previous.velocity_xy[1]))
            subcell_offset = (float(previous.subcell_offset[0]), float(previous.subcell_offset[1]))
        engine.islands[island_id] = FallingIslandRecord(
            island_id=island_id,
            bbox=(int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1),
            velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
            subcell_offset=subcell_offset,
        )
    engine.next_island_id = max(int(engine.next_island_id), max(engine.islands, default=0) + 1, 1)


def downsample_cells_to_gas(engine: "WorldEngine", field: np.ndarray) -> np.ndarray:
    result = np.zeros((engine.gas_height, engine.gas_width), dtype=np.float32)
    for gy in range(engine.gas_height):
        for gx in range(engine.gas_width):
            x0 = gx * engine.gas_cell_size
            y0 = gy * engine.gas_cell_size
            block = field[y0 : min(engine.height, y0 + engine.gas_cell_size), x0 : min(engine.width, x0 + engine.gas_cell_size)]
            result[gy, gx] = float(block.mean()) if block.size else 0.0
    return result


def _set_nested_payload_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = payload
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value


def _advance_paging(engine: "WorldEngine", center_x: int, center_y: int) -> list[PageStripeUpdate]:
    force_sources = [
        replace(force_source, world_x=engine._force_source_world_position(force_source)[0], world_y=engine._force_source_world_position(force_source)[1])
        for force_source in engine.force_sources
    ]
    updates = engine.focus_paging(center_x, center_y)
    if not updates:
        return []
    engine.bridge_frame_paging_updates.extend(
        PageStripeUpdate(**asdict(update)) for update in updates
    )
    for update in updates:
        if update.kind == "save":
            engine.page_store.save(update, engine.capture_page_stripe(update))
            engine._clear_saved_page_stripe_runtime_state(update)
    for update in updates:
        if update.kind != "load":
            continue
        payload = engine.page_store.load(update)
        if payload is None:
            payload = engine._default_page_stripe_payload(update)
        engine._apply_page_stripe(update, payload)
        engine._record_bridge_page_stripe(update, payload)
    if force_sources:
        engine._sync_force_sources(force_sources)
    return updates


def _mirror_release_entity_placeholder_cell(engine: "WorldEngine", x: int, y: int, entity_id: int) -> None:
    if not engine.in_bounds(x, y):
        return
    if int(engine.entity_id[y, x]) != entity_id:
        return
    engine._invalidate_gpu_authoritative_cell_resources()
    engine.entity_id[y, x] = 0
    material_id = int(engine.material_id[y, x])
    if material_id <= 0 or not engine._shadow_material_is_placeholder(material_id):
        return
    displaced_material = int(engine.placeholder_displaced_material[y, x])
    if displaced_material > 0:
        engine.material_id[y, x] = displaced_material
        engine.phase[y, x] = int(Phase.LIQUID)
        engine.cell_flags[y, x] = 0
        engine.timer_pack[y, x] = 0
        shadow_integrity = engine._shadow_material_base_integrity(displaced_material)
        engine.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
        engine.placeholder_displaced_material[y, x] = 0
        engine._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(engine.width, x + 2), min(engine.height, y + 2))
        return
    engine.clear_cell(x, y, mark_dirty=False)


def _public_resolved_change_intent(engine: "WorldEngine", intent: ResolvedChangeIntent) -> ResolvedChangeIntent:
    effect_cells = (
        []
        if intent.center_world_position is None
        else engine._disk_world_cells_raw(
            tuple(int(value) for value in intent.center_world_position),
            int(intent.effective_radius),
        )
    )
    effect_bounds = engine._buffer_cell_bounds(effect_cells)
    generated_commands = [engine._public_world_command(command) for command in intent.generated_commands]
    if intent.center_world_position is not None:
        center_world_x = int(intent.center_world_position[0])
        center_world_y = int(intent.center_world_position[1])
        for command in generated_commands:
            x_field, y_field = TARGETED_COMMAND_COORD_FIELDS.get(command.kind, (None, None))
            if x_field is not None and y_field is not None and x_field in command.payload and y_field in command.payload:
                command.payload[x_field] = center_world_x
                command.payload[y_field] = center_world_y
    return replace(
        intent,
        center_position=(
            None
            if intent.center_world_position is None
            else tuple(int(value) for value in intent.center_world_position)
        ),
        effect_cells=effect_cells,
        effect_bounds=effect_bounds,
        generated_commands=generated_commands,
    )


def _resolve_anchor_target(
    engine: "WorldEngine",
    query: TargetQuery,
    source_world_position: tuple[int, int],
) -> dict[str, Any] | None:
    directional_filters = [item for item in query.anchor_filters if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}]
    terrain_filters = [item for item in query.anchor_filters if item in TERRAIN_ANCHOR_FILTERS]
    entity_filters = [
        item
        for item in query.anchor_filters
        if item not in TERRAIN_ANCHOR_FILTERS
        and item not in IGNORED_ANCHOR_FILTERS
        and item not in CARDINAL_DIRECTION_VECTORS
        and item not in {"forward", "backward"}
    ]
    entity_anchor = None
    terrain_anchor = None
    if query.anchor_entity_id is not None or entity_filters or (directional_filters and not terrain_filters):
        entity_anchor = engine._resolve_entity_anchor(
            query,
            source_world_position,
            direction_filter=directional_filters[0] if directional_filters else None,
        )
    if entity_anchor is not None:
        return entity_anchor
    if terrain_filters:
        terrain_anchor = engine._resolve_terrain_anchor(
            source_world_position,
            terrain_filters,
            direction_filter=directional_filters[0] if directional_filters else None,
        )
    if terrain_anchor is not None:
        return terrain_anchor
    if query.anchor_entity_id is None and not query.anchor_filters:
        return {
            "kind": "source",
            "entity_id": query.source_entity_id,
            "buffer_position": engine._world_to_buffer_clamped(*source_world_position),
            "world_position": source_world_position,
        }
    return None


def _page_stripe_island_bboxes_from_payload(
    engine: "WorldEngine",
    update: PageStripeUpdate,
    payload: dict[str, Any],
) -> dict[int, tuple[int, int, int, int]] | None:
    cell_payload = payload.get("cell", {})
    try:
        material_id = np.asarray(cell_payload["material_id"], dtype=np.int32)
        phase = np.asarray(cell_payload["phase"], dtype=np.uint8)
        island_id = np.asarray(cell_payload["island_id"], dtype=np.int32)
    except KeyError:
        return None
    if material_id.shape != phase.shape or material_id.shape != island_id.shape:
        return None
    valid = (island_id > 0) & (material_id > 0) & (phase == int(Phase.FALLING_ISLAND))
    if not np.any(valid):
        return {}
    boxes: dict[int, list[int]] = {}
    offset = 0
    for start, end in engine._stripe_buffer_ranges(update, gas_grid=False):
        span = int(end) - int(start)
        if span <= 0:
            continue
        if update.axis == "x":
            stripe = valid[:, offset : offset + span]
            stripe_ids = island_id[:, offset : offset + span]
            ys, xs = np.nonzero(stripe)
            for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                current_id = int(stripe_ids[local_y, local_x])
                x = int(start) + int(local_x)
                y = int(local_y)
                box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x + 1)
                box[3] = max(box[3], y + 1)
        else:
            stripe = valid[offset : offset + span, :]
            stripe_ids = island_id[offset : offset + span, :]
            ys, xs = np.nonzero(stripe)
            for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                current_id = int(stripe_ids[local_y, local_x])
                x = int(local_x)
                y = int(start) + int(local_y)
                box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x + 1)
                box[3] = max(box[3], y + 1)
        offset += span
    return {island_id: tuple(box) for island_id, box in boxes.items()}
