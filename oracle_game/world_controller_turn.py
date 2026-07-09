from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ForceSource,
    ObservationTarget,
    PageStripeUpdate,
    ReadbackRequest,
    TargetQuery,
    WorldCommand,
    WorldFrameInput,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def run_entity_controller_turn(
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
) -> dict[str, Any]:
    if controller_state_provided or controller_state is not None:
        engine.controller_state_snapshot = engine._coerce_json_value(controller_state)
    normalized_controller_state = deepcopy(engine.controller_state_snapshot)
    consumed = engine.consume_entity_observation_results()
    paging_updates: list[PageStripeUpdate] = []
    if focus_center is not None:
        paging_updates = engine.advance_paging(int(focus_center[0]), int(focus_center[1]), immediate=True)
    if entities is not None:
        engine.sync_entity_states(entities, immediate=True)
    if patches is not None:
        engine.patch_entity_states(patches, immediate=True)
    if entity_placeholders is not None:
        placeholder_inputs = [
            engine._coerce_entity_placeholder(placeholder)
            for placeholder in entity_placeholders
        ]
        if entities is not None:
            entity_placeholder_inputs, _ = engine._frame_entities_to_placeholders_and_observations(list(engine.entity_states.values()))
            placeholder_inputs = entity_placeholder_inputs + placeholder_inputs
        engine.sync_entity_placeholders(placeholder_inputs, immediate=True)
    if observation_specs is not None:
        engine.sync_entity_observation_specs(observation_specs, immediate=True)
    if force_sources is not None:
        engine.set_force_sources(force_sources, immediate=True)
    if emitters is not None:
        engine.set_emitters(emitters, immediate=True)
    resolved_targets = engine._resolve_target_queries(
        [engine._coerce_target_query(query) for query in (target_queries or [])]
    )
    resolved_change_intents, generated_commands = engine._resolve_change_intents(
        [engine._coerce_change_intent(intent) for intent in (change_intents or [])],
        resolved_targets,
    )
    resolved_carrier_intents, generated_carrier_commands = engine._resolve_carrier_intents(
        [engine._coerce_carrier_intent(intent) for intent in (carrier_intents or [])],
        resolved_targets,
    )
    entity_observation_targets = engine._runtime_entities_to_immediate_observation_targets(
        list(engine.entity_states.values())
    )
    queued_observation_requests: list[ReadbackRequest] = []
    all_observation_targets = entity_observation_targets + [
        engine._coerce_observation_target(target) for target in (observation_targets or [])
    ]
    if all_observation_targets:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        queued_observation_requests = engine._build_observation_requests(
            all_observation_targets,
            resolved_targets,
        )
        queued_observation_requests = [
            engine._assign_readback_request_id(request) for request in queued_observation_requests
        ]
        engine.pending_readbacks.extend(queued_observation_requests)
        engine.bridge_frame_readback_requests.extend(replace(request) for request in queued_observation_requests)
    queued_readback_requests: list[ReadbackRequest] = []
    if readback_requests:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        queued_readback_requests = engine._resolve_readback_requests(
            [engine._coerce_readback_request(request) for request in readback_requests],
            resolved_targets,
        )
        queued_readback_requests = [
            engine._assign_readback_request_id(request) for request in queued_readback_requests
        ]
        engine.pending_readbacks.extend(queued_readback_requests)
        engine.bridge_frame_readback_requests.extend(replace(request) for request in queued_readback_requests)
    resolved_commands = (
        generated_commands
        + generated_carrier_commands
        + engine._resolve_targeted_commands(
            [engine._coerce_world_command(command) for command in (commands or [])],
            resolved_targets,
        )
    )
    for command in resolved_commands:
        engine.queue_command(command.kind, **command.payload)
    return {
        "frame_id": int(engine.frame_id),
        "controller_state": normalized_controller_state,
        "consumed": consumed,
        "paging_updates": [asdict(update) for update in paging_updates],
        "resolved_targets": {
            query_id: engine.serialize_resolved_target(target)
            for query_id, target in resolved_targets.items()
        },
        "resolved_change_intents": {
            intent_id: engine.serialize_resolved_change_intent(engine._public_resolved_change_intent(intent))
            for intent_id, intent in resolved_change_intents.items()
        },
        "resolved_carrier_intents": {
            intent_id: engine.serialize_resolved_carrier_intent(engine._public_resolved_carrier_intent(intent))
            for intent_id, intent in resolved_carrier_intents.items()
        },
        "resolved_commands": [engine.serialize_world_command(command) for command in resolved_commands],
        "observation_requests": [
            engine.serialize_readback_request(request) for request in queued_observation_requests
        ],
        "readback_requests": [
            engine.serialize_readback_request(request) for request in queued_readback_requests
        ],
        "queued_observations": len(queued_observation_requests),
        "queued_readbacks": len(queued_readback_requests),
        "queued_commands": len(resolved_commands),
        "entities": engine.serialize_entity_states()["entities"],
        "placeholders": engine._serialize_cpu_visible_entity_placeholders()["placeholders"],
        "observation_state": engine.serialize_entity_observation_state(),
        "paging_state": engine.serialize_paging_state(),
        "readback_state": engine.serialize_readback_state(),
        "force_sources": engine.serialize_force_sources(),
        "emitters": engine.serialize_emitters(),
        "pending_commands": engine.serialize_pending_commands(),
    }

def set_controller_state(engine: "WorldEngine", controller_state: Any = None) -> dict[str, Any]:
    engine.controller_state_snapshot = engine._coerce_json_value(controller_state)
    return engine.serialize_controller_state()
def _build_preview_controller_turn_entities(
    engine: "WorldEngine",
    *,
    entities: list[EntityState | dict[str, Any]] | None,
    patches: list[EntityStatePatch | dict[str, Any]] | None,
    observation_specs: list[EntityObservationSpec | dict[str, Any]] | None,
) -> list[EntityState] | None:
    if entities is None and patches is None and observation_specs is None:
        return None
    next_entities = {
        entity.entity_id: entity
        for entity in (
            [engine._controller_turn_entity_input(entity) for entity in entities]
            if entities is not None
            else [
                replace(entity, world_x=None, world_y=None)
                for _, entity in sorted(engine.entity_states.items())
            ]
        )
    }
    if patches is not None:
        for patch in [engine._coerce_entity_state_patch(patch) for patch in patches]:
            entity = next_entities.get(patch.entity_id)
            if entity is None:
                raise KeyError(patch.entity_id)
            patch_fields = {name: value for name, value in patch.fields.items() if not name.startswith("_")}
            next_entity = replace(entity, **dict(patch_fields))
            if "_world_x" in patch.fields or "_world_y" in patch.fields:
                next_entity = replace(
                    next_entity,
                    world_x=int(patch.fields.get("_world_x", entity.world_x if entity.world_x is not None else entity.x)),
                    world_y=int(patch.fields.get("_world_y", entity.world_y if entity.world_y is not None else entity.y)),
                )
            elif "x" in patch_fields or "y" in patch_fields:
                next_entity = replace(next_entity, world_x=None, world_y=None)
            next_entities[patch.entity_id] = engine._coerce_entity_state(next_entity)
    if observation_specs is not None:
        observation_by_entity_id = {
            observation.entity_id: observation
            for observation in [engine._coerce_entity_observation_spec(spec) for spec in observation_specs]
        }
        next_entities = {
            entity_id: replace(
                entity,
                observe_channels=observation.observe_channels if observation is not None else (),
                observe_pad_cells=int(observation.observe_pad_cells) if observation is not None else 0,
                observe_width=None if observation is None else observation.observe_width,
                observe_height=None if observation is None else observation.observe_height,
                observe_label=None if observation is None else observation.observe_label,
            )
            for entity_id, entity in next_entities.items()
            for observation in [observation_by_entity_id.get(entity_id)]
        }
    return [next_entities[entity_id] for entity_id in sorted(next_entities)]
def controller_turn_to_frame_input(
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
) -> WorldFrameInput:
    preview_entities = _build_preview_controller_turn_entities(engine, 
        entities=entities,
        patches=patches,
        observation_specs=observation_specs,
    )
    normalized_controller_state_provided = bool(controller_state_provided or controller_state is not None)
    return WorldFrameInput(
        focus_center=focus_center,
        controller_state=(
            engine._coerce_json_value(controller_state)
            if normalized_controller_state_provided
            else None
        ),
        controller_state_provided=normalized_controller_state_provided,
        entities=preview_entities,
        entity_placeholders=None
        if entity_placeholders is None
        else [engine._public_entity_placeholder_input(placeholder) for placeholder in entity_placeholders],
        force_sources=[]
        if force_sources == []
        else None
        if force_sources is None
        else [engine._public_force_source_input(force_source) for force_source in force_sources],
        emitters=[]
        if emitters == []
        else None
        if emitters is None
        else [engine._coerce_emitter(emitter) for emitter in emitters],
        target_queries=[engine._coerce_target_query(query) for query in (target_queries or [])],
        change_intents=[engine._coerce_change_intent(intent) for intent in (change_intents or [])],
        carrier_intents=[engine._coerce_carrier_intent(intent) for intent in (carrier_intents or [])],
        observation_targets=[engine._coerce_observation_target(target) for target in (observation_targets or [])],
        readback_requests=[engine._coerce_readback_request(request) for request in (readback_requests or [])],
        commands=[engine._coerce_world_command(command) for command in (commands or [])],
    )

def preview_entity_controller_turn(
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
    reserved_readback_request_ids: set[int] | None = None,
) -> dict[str, Any]:
    frame_input = controller_turn_to_frame_input(engine, 
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
    normalized_controller_state = (
        deepcopy(frame_input.controller_state)
        if frame_input.controller_state_provided
        else deepcopy(engine.controller_state_snapshot)
    )
    consumed = engine._preview_consume_entity_observation_results()
    force_sources_payload = (
        engine.serialize_force_sources()
        if force_sources is None
        else [engine._serialize_force_source_record(force_source) for force_source in frame_input.force_sources or []]
    )
    emitters_payload = (
        engine.serialize_emitters()
        if emitters is None
        else {
            "persistent_emitters": [engine._serialize_emitter_record(emitter) for emitter in frame_input.emitters or []],
            "queued_emitters": [],
        }
    )
    pending_commands_payload = engine.serialize_pending_commands()
    pending_commands = {
        "pending": int(pending_commands_payload["pending"]),
        "commands": list(pending_commands_payload["commands"]),
    }

    saved_paging = deepcopy(engine.paging)
    saved_preview_runtime = engine._snapshot_preview_runtime_state()
    saved_entity_states = dict(engine.entity_states)
    saved_entity_placeholders = {entity_id: set(cells) for entity_id, cells in engine.entity_placeholders.items()}
    saved_blocked_cells = None if engine._resolver_blocked_cells is None else set(engine._resolver_blocked_cells)
    saved_released_cells = None if engine._resolver_released_cells is None else set(engine._resolver_released_cells)
    try:
        (
            paging_updates,
            preview_page_stripes,
            entity_observation_targets,
            placeholder_inputs,
            placeholder_count,
        ) = engine._prepare_preview_frame_context(frame_input)
        resolved_targets = engine._resolve_target_queries(frame_input.target_queries)
        resolved_change_intents, generated_commands = engine._resolve_change_intents(frame_input.change_intents, resolved_targets)
        resolved_carrier_intents, generated_carrier_commands = engine._resolve_carrier_intents(
            frame_input.carrier_intents,
            resolved_targets,
        )
        observation_pairs = engine._build_observation_request_pairs(
            entity_observation_targets + frame_input.observation_targets,
            resolved_targets,
        )
        observation_requests, next_preview_request_id = engine._assign_preview_readback_request_ids(
            [request for _, request in observation_pairs]
        )
        observation_pairs = [
            (target, request)
            for (target, _), request in zip(observation_pairs, observation_requests, strict=False)
        ]
        resolved_commands = (
            generated_commands
            + generated_carrier_commands
            + engine._resolve_targeted_commands(frame_input.commands, resolved_targets)
        )
        readback_request_plan, _ = engine._assign_preview_readback_request_ids(
            engine._resolve_readback_requests(frame_input.readback_requests, resolved_targets),
            next_request_id=next_preview_request_id,
        )
        pending_commands["pending"] += len(resolved_commands)
        pending_commands["commands"].extend(
            engine.serialize_world_command(command)
            for command in resolved_commands
        )
        bridge_frame_snapshot = engine._serialize_preview_bridge_frame_snapshot(
            current_entity_placeholders=saved_entity_placeholders,
            resolved_commands=resolved_commands,
            observation_requests=observation_requests,
            readback_requests=readback_request_plan,
            placeholder_inputs=placeholder_inputs,
            paging_updates=paging_updates,
            page_stripes=preview_page_stripes,
            reserved_readback_request_ids=reserved_readback_request_ids,
        )
        return {
            "frame_id": int(engine.frame_id),
            "controller_state": normalized_controller_state,
            "consumed": consumed,
            "paging_updates": [asdict(update) for update in paging_updates],
            "resolved_targets": {
                query_id: engine.serialize_resolved_target(target)
                for query_id, target in resolved_targets.items()
            },
            "resolved_change_intents": {
                intent_id: engine.serialize_resolved_change_intent(engine._public_resolved_change_intent(intent))
                for intent_id, intent in resolved_change_intents.items()
            },
            "resolved_carrier_intents": {
                intent_id: engine.serialize_resolved_carrier_intent(engine._public_resolved_carrier_intent(intent))
                for intent_id, intent in resolved_carrier_intents.items()
            },
            "resolved_commands": [engine.serialize_world_command(command) for command in resolved_commands],
            "observation_requests": [
                engine.serialize_readback_request(request) for request in observation_requests
            ],
            "observation_plans": [
                engine._serialize_observation_plan_for_target_request(target, request)
                for target, request in observation_pairs
            ],
            "readback_requests": [
                engine.serialize_readback_request(request) for request in readback_request_plan
            ],
            "readback_plans": engine._serialize_readback_plans_for_requests(readback_request_plan),
            "bridge_frame_snapshot": bridge_frame_snapshot,
            "queued_observations": len(observation_requests),
            "queued_readbacks": len(readback_request_plan),
            "queued_commands": len(resolved_commands),
            "placeholder_count": int(placeholder_count),
            "entities": engine.serialize_entity_states()["entities"],
            "placeholders": engine._serialize_cpu_visible_entity_placeholders()["placeholders"],
            "observation_state": engine.serialize_entity_observation_state(),
            "paging_state": engine.serialize_paging_state(),
            "force_sources": force_sources_payload,
            "emitters": emitters_payload,
            "pending_commands": pending_commands,
        }
    finally:
        engine._restore_preview_runtime_state(saved_preview_runtime)
        engine.paging = saved_paging
        engine.entity_states = saved_entity_states
        engine.entity_placeholders = saved_entity_placeholders
        engine._resolver_blocked_cells = saved_blocked_cells
        engine._resolver_released_cells = saved_released_cells
def request_entity_controller_turn(
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
) -> dict[str, Any]:
    submission_id = engine.submit_entity_controller_turn(
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
    pending_frame_input = engine._pending_frame_input(submission_id)
    preview = preview_entity_controller_turn(engine, 
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
        readback_requests=[
            replace(request)
            for request in pending_frame_input.readback_requests
        ],
        commands=commands,
        reserved_readback_request_ids=set(engine._frame_readback_request_ids(pending_frame_input)),
    )
    return {
        "queued": True,
        "pending_frames": len(engine.pending_frame_inputs),
        "submission_id": submission_id,
        "preview": preview,
    }

def request_entity_controller_cycle(
    engine: "WorldEngine",
    *,
    apply_turn: bool = True,
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
) -> dict[str, Any]:
    preview = preview_entity_controller_turn(engine, 
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
    if not apply_turn:
        return {
            "applied": False,
            "queued": False,
            "pending_frames": len(engine.pending_frame_inputs),
            "submission_id": None,
            "preview": preview,
            "result": None,
        }
    submission_id = engine.submit_entity_controller_turn(
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
    pending_frame_input = engine._pending_frame_input(submission_id)
    preview = preview_entity_controller_turn(engine, 
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
        readback_requests=[
            replace(request)
            for request in pending_frame_input.readback_requests
        ],
        commands=commands,
        reserved_readback_request_ids=set(engine._frame_readback_request_ids(pending_frame_input)),
    )
    return {
        "applied": True,
        "queued": True,
        "pending_frames": len(engine.pending_frame_inputs),
        "submission_id": submission_id,
        "preview": preview,
        "result": None,
    }

def run_entity_controller_cycle(
    engine: "WorldEngine",
    *,
    apply_turn: bool = True,
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
) -> dict[str, Any]:
    preview = preview_entity_controller_turn(engine, 
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
    if not apply_turn:
        return {
            "applied": False,
            "preview": preview,
            "result": None,
        }
    result = run_entity_controller_turn(engine, 
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
    return {
        "applied": True,
        "preview": preview,
        "result": result,
    }

