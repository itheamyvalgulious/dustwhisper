from __future__ import annotations

from typing import Any, TYPE_CHECKING

from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np
from oracle_game.types import (
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    EntityObservationSpec,
    ObservationTarget,
    ObservationResult,
    EntityFeedback,
    ReadbackRequest,
    ReadbackResult,
    ForceSource,
    ResolvedTarget,
    Phase,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine



def sync_entity_placeholders(
    engine,
    placeholders: list[EntityPlaceholder],
    *,
    immediate: bool = False,
) -> None:
    placeholders = [engine._public_entity_placeholder_input(placeholder) for placeholder in placeholders]
    if immediate:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        _sync_entity_placeholders(engine, 
            [engine._frame_entity_placeholder_input(placeholder) for placeholder in placeholders]
        )
        return
    engine.queue_command(
        "sync_entity_placeholders",
        placeholders=[asdict(placeholder) for placeholder in placeholders],
    )


def sync_entity_states(
    engine,
    entities: list[EntityState | dict[str, Any]],
    *,
    immediate: bool = False,
) -> None:
    entities = [engine._public_entity_state_input(entity) for entity in entities]
    if immediate:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        placeholders, _ = _sync_entity_states(engine, 
            [engine._frame_entity_state_input(entity) for entity in entities]
        )
        _sync_entity_placeholders(engine, placeholders)
        return
    engine.queue_command(
        "sync_entity_states",
        entities=[asdict(entity) for entity in entities],
    )


def patch_entity_states(
    engine,
    patches: list[EntityStatePatch | dict[str, Any]],
    *,
    immediate: bool = False,
) -> None:
    patches = [engine._public_entity_state_patch_input(patch) for patch in patches]
    if immediate:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        engine._patch_entity_states(
            [engine._frame_entity_state_patch_input(patch) for patch in patches]
        )
        return
    engine.queue_command(
        "patch_entity_states",
        patches=[asdict(patch) for patch in patches],
    )


def sync_entity_observation_specs(
    engine,
    observations: list[EntityObservationSpec | dict[str, Any]],
    *,
    immediate: bool = False,
) -> None:
    observations = [engine._coerce_entity_observation_spec(observation) for observation in observations]
    if immediate:
        _sync_entity_observation_specs(engine, observations)
        return
    engine.queue_command(
        "sync_entity_observation_specs",
        observations=[asdict(observation) for observation in observations],
    )


def set_force_sources(
    engine,
    force_sources: list[ForceSource | dict[str, Any]],
    *,
    immediate: bool = False,
) -> None:
    force_sources = [engine._public_force_source_input(force_source) for force_source in force_sources]
    if immediate:
        engine._sync_force_sources(
            [engine._normalize_runtime_force_source(force_source) for force_source in force_sources]
        )
        return
    engine.queue_command(
        "set_force_sources",
        force_sources=[
            {
                "x": float(force_source.world_x),
                "y": float(force_source.world_y),
                "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
                "radius": float(force_source.radius),
                "strength": float(force_source.strength),
                "lifetime": float(force_source.lifetime),
            }
            for force_source in force_sources
        ],
    )


def set_emitters(
    engine,
    emitters: list[dict[str, Any]],
    *,
    immediate: bool = False,
) -> None:
    emitters = [engine._coerce_emitter(emitter) for emitter in emitters]
    if immediate:
        normalized_emitters = [
            {
                **dict(emitter),
                "origin": engine._world_to_buffer_clamped(
                    int(emitter["world_origin"][0]),
                    int(emitter["world_origin"][1]),
                ),
            }
            for emitter in emitters
        ]
        engine._sync_persistent_emitters(normalized_emitters)
        return
    engine.queue_command(
        "set_emitters",
        emitters=[
            {
                "x": int(emitter["origin"][0]),
                "y": int(emitter["origin"][1]),
                "light_type": str(emitter["light_type"]),
                "direction": list(emitter["direction"]),
                "spread": float(emitter["spread"]),
                "strength": float(emitter["strength"]),
                "radius": int(emitter["range_cells"]),
            }
            for emitter in emitters
        ],
    )


def consume_entity_observation_results(
    engine,
    *,
    current_frame_id: int | None = None,
) -> dict[str, Any]:
    consumed_readbacks = engine.poll_all_readbacks(current_frame_id=current_frame_id)
    observations = _collect_observations(engine, consumed_readbacks)
    entity_feedback = _collect_entity_feedback(engine, consumed_readbacks)
    frame_id = engine.frame_id if current_frame_id is None else int(current_frame_id)
    return engine._store_entity_observation_consume_snapshot(
        frame_id=frame_id,
        consumed_readbacks=consumed_readbacks,
        observations=observations,
        entity_feedback=entity_feedback,
    )


def _preview_consume_entity_observation_results(engine) -> dict[str, Any]:
    saved_completed_readbacks = deepcopy(engine.completed_readbacks)
    saved_last_snapshot = deepcopy(engine.last_entity_observation_consume_snapshot)
    try:
        return consume_entity_observation_results(engine, )
    finally:
        engine.completed_readbacks = saved_completed_readbacks
        engine.last_entity_observation_consume_snapshot = saved_last_snapshot


def _sync_entity_placeholders(engine, placeholders: list[EntityPlaceholder]) -> None:
    engine.bridge_frame_placeholders.extend(replace(placeholder) for placeholder in placeholders)
    current_cells = {
        cell: entity_id
        for entity_id, cells in engine.entity_placeholders.items()
        for cell in cells
    }
    next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
    for placeholder in placeholders:
        for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
            for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                if not engine.in_bounds(x, y):
                    continue
                next_cells[(x, y)] = placeholder

    changed_cells: set[tuple[int, int]] = set()
    for cell, entity_id in current_cells.items():
        next_placeholder = next_cells.get(cell)
        if next_placeholder is None or next_placeholder.entity_id != entity_id:
            changed_cells.add(cell)
    for cell, placeholder in next_cells.items():
        if current_cells.get(cell) != placeholder.entity_id:
            changed_cells.add(cell)

    if engine._gpu_pipeline_available(
        engine.placeholder_pipeline,
        "placeholder",
        require=engine.simulation_backend == "gpu",
    ):
        if current_cells or next_cells:
            if engine.simulation_backend == "gpu" and engine._world_simulation_frame_active and (
                not engine._bridge_inputs_prepared or engine._gpu_cpu_dirty_resources
            ):
                engine._sync_pre_simulation_bridge_without_debug_upload()
                engine._gpu_cpu_dirty_resources.clear()
                engine._bridge_inputs_prepared = True
            engine.placeholder_pipeline.apply(engine, placeholders)
            if engine.placeholder_pipeline.last_cpu_mirror_downloaded:
                engine._rebuild_entity_placeholder_index()
            else:
                next_entity_cells: dict[int, set[tuple[int, int]]] = {}
                for cell, placeholder in next_cells.items():
                    next_entity_cells.setdefault(int(placeholder.entity_id), set()).add(cell)
                engine.entity_placeholders = next_entity_cells
        else:
            engine.entity_placeholders.clear()
            engine.placeholder_pipeline.last_backend = "idle"
        for x, y in sorted(changed_cells):
            engine._mark_active_rect_runtime(
                max(0, x - 1),
                max(0, y - 1),
                min(engine.width, x + 2),
                min(engine.height, y + 2),
            )
        engine.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))
        return

    engine._require_cpu_oracle_backend("placeholder")
    engine.placeholder_pipeline.last_backend = "cpu" if (current_cells or next_cells) else "idle"
    for cell, entity_id in current_cells.items():
        next_placeholder = next_cells.get(cell)
        x, y = cell
        material_id = int(engine.material_id[y, x])
        if (
            next_placeholder is not None
            and next_placeholder.entity_id == entity_id
            and material_id > 0
            and engine._shadow_material_is_placeholder(material_id)
        ):
            continue
        _release_entity_placeholder_cell(engine, x, y, entity_id)

    next_entity_cells: dict[int, set[tuple[int, int]]] = {}
    for cell, placeholder in next_cells.items():
        x, y = cell
        material_id = int(engine.material_id[y, x])
        if (
            current_cells.get(cell) == placeholder.entity_id
            and material_id > 0
            and engine._shadow_material_is_placeholder(material_id)
        ):
            next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
            engine.entity_id[y, x] = placeholder.entity_id
            continue
        if _occupy_entity_placeholder_cell(engine, x, y, placeholder):
            next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
    engine.entity_placeholders = next_entity_cells
    engine.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))


def _release_entity_placeholder_cell(engine, x: int, y: int, entity_id: int) -> None:
    if not engine.in_bounds(x, y):
        return
    if int(engine.entity_id[y, x]) != entity_id:
        return
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


def _occupy_entity_placeholder_cell(engine, x: int, y: int, placeholder: EntityPlaceholder) -> bool:
    if not engine.in_bounds(x, y):
        return False
    placeholder_material_id = engine._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
    if placeholder_material_id <= 0:
        return False
    material_id = int(engine.material_id[y, x])
    if material_id != 0 and int(engine.phase[y, x]) != int(Phase.LIQUID):
        return False
    engine.set_cell_by_id(x, y, placeholder_material_id, mark_dirty=False)
    engine.entity_id[y, x] = placeholder.entity_id
    engine._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(engine.width, x + 2), min(engine.height, y + 2))
    return True


def _frame_entities_to_placeholders_and_observations(
    engine,
    entities: list[EntityState],
) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
    placeholders = [
        EntityPlaceholder(
            entity_id=entity.entity_id,
            x=entity.x,
            y=entity.y,
            width=entity.width,
            height=entity.height,
            material=entity.placeholder_material,
            world_x=entity.world_x,
            world_y=entity.world_y,
        )
        for entity in entities
    ]
    observation_targets = [
        ObservationTarget(
            observer_id=entity.entity_id,
            entity_id=entity.entity_id,
            channels=entity.observe_channels,
            pad_cells=entity.observe_pad_cells,
            width=entity.observe_width,
            height=entity.observe_height,
            label=entity.observe_label,
        )
        for entity in entities
        if entity.observe_channels
    ]
    return placeholders, observation_targets


def _sync_entity_states(engine, entities: list[EntityState]) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
    engine.entity_states = {entity.entity_id: entity for entity in entities}
    placeholders, observation_targets = _frame_entities_to_placeholders_and_observations(engine, entities)
    return placeholders, observation_targets


def _sync_entity_observation_specs(engine, observations: list[EntityObservationSpec]) -> None:
    observation_by_entity_id = {observation.entity_id: observation for observation in observations}
    engine.entity_states = {
        entity_id: replace(
            entity,
            observe_channels=observation.observe_channels if observation is not None else (),
            observe_pad_cells=int(observation.observe_pad_cells) if observation is not None else 0,
            observe_width=None if observation is None else observation.observe_width,
            observe_height=None if observation is None else observation.observe_height,
            observe_label=None if observation is None else observation.observe_label,
        )
        for entity_id, entity in engine.entity_states.items()
        for observation in [observation_by_entity_id.get(entity_id)]
    }


def _build_preview_entity_placeholders(
    engine,
    placeholders: list[EntityPlaceholder],
) -> tuple[dict[int, set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    current_cells = {
        cell: entity_id
        for entity_id, cells in engine.entity_placeholders.items()
        for cell in cells
    }
    next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
    for placeholder in placeholders:
        for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
            for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                if not engine.in_bounds(x, y):
                    continue
                next_cells[(x, y)] = placeholder

    released_cells = {
        cell
        for cell, entity_id in current_cells.items()
        if next_cells.get(cell) is None or next_cells[cell].entity_id != entity_id
    }
    next_entity_cells: dict[int, set[tuple[int, int]]] = {}
    for cell, placeholder in next_cells.items():
        if engine._preview_can_occupy_placeholder_cell(cell[0], cell[1], placeholder, current_cells, released_cells):
            next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
    blocked_cells = {
        cell
        for cells in next_entity_cells.values()
        for cell in cells
    }
    return next_entity_cells, blocked_cells, released_cells


def _build_observation_request(
    engine,
    target: ObservationTarget,
    resolved_targets: dict[str, ResolvedTarget],
) -> ReadbackRequest | None:
    center_x = target.center_x
    center_y = target.center_y
    width = target.width
    height = target.height
    if target.target_query_id is not None and (center_x is None or center_y is None or width is None or height is None):
        resolved_target = resolved_targets.get(target.target_query_id)
        if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
            return None
        if center_x is None:
            center_x = int(resolved_target.resolved_world_position[0]) + int(target.target_dx)
        if center_y is None:
            center_y = int(resolved_target.resolved_world_position[1]) + int(target.target_dy)
        if width is None:
            width = 1 + int(target.pad_cells) * 2
        if height is None:
            height = 1 + int(target.pad_cells) * 2
    if target.entity_id is not None and (center_x is None or center_y is None or width is None or height is None):
        bbox = engine._entity_placeholder_bbox(target.entity_id)
        if bbox is None:
            return None
        x0, y0, x1, y1 = engine._buffer_bbox_to_world_bbox(bbox)
        if center_x is None:
            center_x = (x0 + x1 - 1) // 2
        if center_y is None:
            center_y = (y0 + y1 - 1) // 2
        if width is None:
            width = (x1 - x0) + target.pad_cells * 2
        if height is None:
            height = (y1 - y0) + target.pad_cells * 2
    if center_x is None or center_y is None:
        return None
    return engine._normalize_readback_request(
        ReadbackRequest(
            center_x=center_x,
            center_y=center_y,
            width=max(1, width if width is not None else 1),
            height=max(1, height if height is not None else 1),
            channels=target.channels,
            observer_id=target.observer_id,
            label=target.label,
            target_query_id=target.target_query_id,
            target_dx=int(target.target_dx),
            target_dy=int(target.target_dy),
        )
    )


def _resolve_readback_request(
    engine,
    request: ReadbackRequest,
    resolved_targets: dict[str, ResolvedTarget],
) -> ReadbackRequest | None:
    center_x = request.center_x
    center_y = request.center_y
    if request.target_query_id is not None and (center_x is None or center_y is None):
        resolved_target = resolved_targets.get(request.target_query_id)
        if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
            return None
        if center_x is None:
            center_x = int(resolved_target.resolved_world_position[0]) + int(request.target_dx)
        if center_y is None:
            center_y = int(resolved_target.resolved_world_position[1]) + int(request.target_dy)
    if center_x is None or center_y is None:
        return None
    return engine._normalize_readback_request(
        ReadbackRequest(
            request_id=request.request_id,
            center_x=int(center_x),
            center_y=int(center_y),
            width=max(1, int(request.width)),
            height=max(1, int(request.height)),
            channels=request.channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )
    )


def _collect_observations(engine, results: list[ReadbackResult]) -> dict[int, ObservationResult]:
    observations: dict[int, ObservationResult] = {}
    for result in results:
        observer_id = result.request.observer_id
        if observer_id is None:
            continue
        observations[observer_id] = ObservationResult(
            observer_id=observer_id,
            frame_id=result.frame_id,
            request=result.request,
            payload=result.payload,
        )
    return observations


def _collect_entity_feedback(engine, results: list[ReadbackResult]) -> dict[int, EntityFeedback]:
    feedback: dict[int, EntityFeedback] = {}
    for result in results:
        observer_id = result.request.observer_id
        if observer_id is None or observer_id in feedback:
            continue
        entity = engine.entity_states.get(observer_id)
        if entity is None:
            continue
        entity_feedback = _build_entity_feedback(engine, result, entity)
        if entity_feedback is not None:
            feedback[observer_id] = entity_feedback
    return feedback


def _build_entity_feedback(
    engine,
    result: ReadbackResult,
    entity: EntityState,
) -> EntityFeedback | None:
    cell_payload = result.payload.get("cell")
    if cell_payload is None:
        return None
    core_words = cell_payload.get("core_words")
    entity_ids = cell_payload.get("entity_id")
    if core_words is None or entity_ids is None:
        return None

    origin_x, origin_y = (int(value) for value in cell_payload["origin"])
    width, height = (int(value) for value in cell_payload["size"])

    unpacked = unpack_cell_core(core_words)
    cells: list[EntityCellFeedback] = []
    base_world_x = int(entity.world_x) if entity.world_x is not None else int(engine._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
    base_world_y = int(entity.world_y) if entity.world_y is not None else int(engine._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
    for local_y in range(max(0, int(entity.height))):
        for local_x in range(max(0, int(entity.width))):
            world_x = base_world_x + local_x
            world_y = base_world_y + local_y
            lx = int(world_x) - origin_x
            ly = int(world_y) - origin_y
            if lx < 0 or ly < 0 or lx >= width or ly >= height:
                continue
            material_id = int(unpacked["material_id"][ly, lx])
            phase = int(unpacked["phase"][ly, lx])
            integrity = float(unpacked["integrity"][ly, lx])
            occupant_entity_id = int(entity_ids[ly, lx])
            present = (
                occupant_entity_id == entity.entity_id
                and material_id > 0
                and engine._shadow_material_is_placeholder(material_id)
            )
            cells.append(
                EntityCellFeedback(
                    x=int(world_x),
                    y=int(world_y),
                    present=present,
                    material_id=material_id,
                    phase=phase,
                    integrity=integrity,
                    entity_id=occupant_entity_id,
                )
            )
    if not cells:
        return None
    bbox_xs = [cell.x for cell in cells]
    bbox_ys = [cell.y for cell in cells]
    return EntityFeedback(
        entity_id=entity.entity_id,
        bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
        cells=cells,
    )


def _build_entity_feedback_from_state(
    engine,
    entity: EntityState,
    *,
    cell_state: dict[str, np.ndarray],
    entity_runtime: dict[str, np.ndarray],
) -> EntityFeedback | None:
    cells: list[EntityCellFeedback] = []
    base_world_x = int(entity.world_x) if entity.world_x is not None else int(engine._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
    base_world_y = int(entity.world_y) if entity.world_y is not None else int(engine._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
    material_grid = cell_state["material_id"]
    phase_grid = cell_state["phase"]
    integrity_grid = cell_state["integrity"]
    entity_id_grid = entity_runtime["entity_id"]
    for local_y in range(max(0, int(entity.height))):
        for local_x in range(max(0, int(entity.width))):
            world_x = base_world_x + local_x
            world_y = base_world_y + local_y
            buffer_x, buffer_y = engine._world_to_buffer_clamped(world_x, world_y)
            material_id = int(material_grid[buffer_y, buffer_x])
            phase = int(phase_grid[buffer_y, buffer_x])
            integrity = float(integrity_grid[buffer_y, buffer_x])
            occupant_entity_id = int(entity_id_grid[buffer_y, buffer_x])
            present = (
                occupant_entity_id == entity.entity_id
                and material_id > 0
                and engine._shadow_material_is_placeholder(material_id)
            )
            cells.append(
                EntityCellFeedback(
                    x=int(world_x),
                    y=int(world_y),
                    present=present,
                    material_id=material_id,
                    phase=phase,
                    integrity=integrity,
                    entity_id=occupant_entity_id,
                )
            )
    if not cells:
        return None
    bbox_xs = [cell.x for cell in cells]
    bbox_ys = [cell.y for cell in cells]
    return EntityFeedback(
        entity_id=int(entity.entity_id),
        bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
        cells=cells,
    )
