from __future__ import annotations

from typing import Any, TYPE_CHECKING

from collections import deque
from copy import deepcopy
from dataclasses import replace
from oracle_game.types import (
    EntityPlaceholder,
    ObservationTarget,
    PageStripeUpdate,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    WorldCommand,
    WorldFrameInput,
    WorldFrameOutput,
    WorldFramePreview,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def preview_frame_input(
    engine,
    frame_input: WorldFrameInput | dict[str, Any],
    *,
    reserved_readback_request_ids: set[int] | None = None,
) -> WorldFramePreview:
    frame_input = engine._coerce_world_frame_input(frame_input)
    saved_paging = deepcopy(engine.paging)
    saved_preview_runtime = engine._snapshot_preview_runtime_state()
    saved_entity_states = dict(engine.entity_states)
    saved_entity_placeholders = {entity_id: set(cells) for entity_id, cells in engine.entity_placeholders.items()}
    saved_controller_state = deepcopy(engine.controller_state_snapshot)
    saved_blocked_cells = None if engine._resolver_blocked_cells is None else set(engine._resolver_blocked_cells)
    saved_released_cells = None if engine._resolver_released_cells is None else set(engine._resolver_released_cells)
    try:
        preview_controller_state = deepcopy(engine.controller_state_snapshot)
        if frame_input.controller_state_provided:
            engine.controller_state_snapshot = deepcopy(frame_input.controller_state)
            preview_controller_state = deepcopy(engine.controller_state_snapshot)
        (
            paging_updates,
            preview_page_stripes,
            entity_observation_targets,
            placeholder_inputs,
            placeholder_count,
        ) = _prepare_preview_frame_context(engine, frame_input)
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
        readback_requests, _ = engine._assign_preview_readback_request_ids(
            engine._resolve_readback_requests(frame_input.readback_requests, resolved_targets),
            next_request_id=next_preview_request_id,
        )
        bridge_frame_snapshot = engine._serialize_preview_bridge_frame_snapshot(
            current_entity_placeholders=saved_entity_placeholders,
            resolved_commands=resolved_commands,
            observation_requests=observation_requests,
            readback_requests=readback_requests,
            placeholder_inputs=placeholder_inputs,
            paging_updates=paging_updates,
            page_stripes=preview_page_stripes,
            reserved_readback_request_ids=reserved_readback_request_ids,
        )
        return WorldFramePreview(
            controller_state=preview_controller_state,
            resolved_targets={
                query_id: engine._public_resolved_target(target)
                for query_id, target in resolved_targets.items()
            },
            resolved_change_intents={
                intent_id: engine._public_resolved_change_intent(intent)
                for intent_id, intent in resolved_change_intents.items()
            },
            resolved_carrier_intents={
                intent_id: engine._public_resolved_carrier_intent(intent)
                for intent_id, intent in resolved_carrier_intents.items()
            },
            resolved_commands=[engine._public_world_command(command) for command in resolved_commands],
            observation_requests=observation_requests,
            observation_plans=[
                engine._serialize_observation_plan_for_target_request(target, request)
                for target, request in observation_pairs
            ],
            readback_requests=readback_requests,
            readback_plans=engine._serialize_readback_plans_for_requests(readback_requests),
            bridge_frame_snapshot=bridge_frame_snapshot,
            paging_updates=paging_updates,
            placeholder_count=placeholder_count,
        )
    finally:
        engine._restore_preview_runtime_state(saved_preview_runtime)
        engine.paging = saved_paging
        engine.entity_states = saved_entity_states
        engine.entity_placeholders = saved_entity_placeholders
        engine.controller_state_snapshot = saved_controller_state
        engine._resolver_blocked_cells = saved_blocked_cells
        engine._resolver_released_cells = saved_released_cells


def submit_frame_input(engine, frame_input: WorldFrameInput | dict[str, Any]) -> int:
    frame_input = engine._coerce_world_frame_input(frame_input)
    submission_id = frame_input.submission_id
    if submission_id is None:
        submission_id = engine.next_frame_submission_id
    frame_input = replace(
        frame_input,
        submission_id=submission_id,
        readback_requests=[engine._assign_readback_request_id(request) for request in frame_input.readback_requests],
    )
    engine.next_frame_submission_id = max(engine.next_frame_submission_id, int(submission_id) + 1)
    engine.canceled_frame_submission_ids.discard(int(submission_id))
    engine.pending_frame_inputs.append(frame_input)
    return int(submission_id)


def request_frame_input(engine, frame_input: WorldFrameInput | dict[str, Any]) -> dict[str, Any]:
    submission_id = submit_frame_input(engine, frame_input)
    pending_frame_input = engine._pending_frame_input(submission_id)
    preview = preview_frame_input(
        engine,
        pending_frame_input,
        reserved_readback_request_ids=set(engine._frame_readback_request_ids(pending_frame_input)),
    )
    return {
        "queued": True,
        "pending_frames": len(engine.pending_frame_inputs),
        "submission_id": submission_id,
        "preview": preview,
    }


def request_frame_cycle(
    engine,
    frame_input: WorldFrameInput | dict[str, Any] | None = None,
    *,
    apply_frame: bool = True,
) -> dict[str, Any]:
    normalized_frame_input = {} if frame_input is None else frame_input
    preview = preview_frame_input(engine, normalized_frame_input)
    if not apply_frame:
        return {
            "applied": False,
            "queued": False,
            "pending_frames": len(engine.pending_frame_inputs),
            "submission_id": None,
            "preview": preview,
            "result": None,
        }
    submission_id = submit_frame_input(engine, normalized_frame_input)
    pending_frame_input = engine._pending_frame_input(submission_id)
    preview = preview_frame_input(
        engine,
        pending_frame_input,
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


def pending_frame_submission_ids(engine) -> list[int]:
    return [int(frame_input.submission_id) for frame_input in engine.pending_frame_inputs if frame_input.submission_id is not None]


def cancel_frame_submission(engine, submission_id: int) -> bool:
    for index, frame_input in enumerate(engine.pending_frame_inputs):
        if frame_input.submission_id == submission_id:
            del engine.pending_frame_inputs[index]
            engine.canceled_frame_submission_ids.add(int(submission_id))
            engine.canceled_readback_request_ids.update(engine._frame_readback_request_ids(frame_input))
            return True
    return False


def cancel_readback_request(engine, request_id: int) -> bool:
    request_id = int(request_id)
    canceled = False

    remaining_commands: deque[WorldCommand] = deque()
    for command in engine.command_queue:
        if command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == request_id:
            canceled = True
            continue
        remaining_commands.append(command)
    engine.command_queue = remaining_commands

    remaining_frames: deque[WorldFrameInput] = deque()
    for frame_input in engine.pending_frame_inputs:
        remaining_readbacks = [request for request in frame_input.readback_requests if request.request_id != request_id]
        if len(remaining_readbacks) != len(frame_input.readback_requests):
            frame_input = replace(frame_input, readback_requests=remaining_readbacks)
            canceled = True
        remaining_frames.append(frame_input)
    engine.pending_frame_inputs = remaining_frames

    next_pending = [request for request in engine.pending_readbacks if request.request_id != request_id]
    if len(next_pending) != len(engine.pending_readbacks):
        canceled = True
    engine.pending_readbacks = next_pending

    next_inflight = [request for request in engine.inflight_readbacks if request.request_id != request_id]
    if len(next_inflight) != len(engine.inflight_readbacks):
        canceled = True
    engine.inflight_readbacks = next_inflight

    next_completed = deque(
        result for result in engine.completed_readbacks if result.request.request_id != request_id
    )
    if len(next_completed) != len(engine.completed_readbacks):
        canceled = True
    engine.completed_readbacks = next_completed

    if canceled:
        engine.canceled_readback_request_ids.add(request_id)
    return canceled


def poll_frame_output(engine, submission_id: int | None = None) -> WorldFrameOutput | None:
    if submission_id is None:
        if not engine.completed_frame_outputs:
            return None
        return engine.completed_frame_outputs.popleft()
    for index, output in enumerate(engine.completed_frame_outputs):
        if output.submission_id == submission_id:
            del engine.completed_frame_outputs[index]
            return output
    return None


def poll_all_frame_outputs(engine) -> list[WorldFrameOutput]:
    outputs: list[WorldFrameOutput] = []
    while engine.completed_frame_outputs:
        outputs.append(engine.completed_frame_outputs.popleft())
    return outputs


def frame_submission_status(engine, submission_id: int) -> str:
    if any(frame_input.submission_id == submission_id for frame_input in engine.pending_frame_inputs):
        return "pending"
    if any(output.submission_id == submission_id for output in engine.completed_frame_outputs):
        return "ready"
    if submission_id in engine.canceled_frame_submission_ids:
        return "canceled"
    return "missing"


def _apply_frame_input(
    engine,
    frame_input: WorldFrameInput,
) -> tuple[
    Any,
    list[PageStripeUpdate],
    dict[str, ResolvedTarget],
    dict[str, ResolvedChangeIntent],
    dict[str, ResolvedCarrierIntent],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    int,
    int,
]:
    paging_updates: list[PageStripeUpdate] = []
    resolved_targets: dict[str, ResolvedTarget] = {}
    resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
    resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
    observation_plans: list[dict[str, Any]] = []
    readback_plans: list[dict[str, Any]] = []
    queued_observations = 0
    queued_readbacks = 0
    queued_commands = 0
    placeholder_count = 0
    if frame_input.controller_state_provided:
        engine.controller_state_snapshot = deepcopy(frame_input.controller_state)
    output_controller_state = deepcopy(engine.controller_state_snapshot)
    if frame_input.focus_center is not None:
        paging_updates = engine.advance_paging(
            frame_input.focus_center[0],
            frame_input.focus_center[1],
            immediate=True,
        )
    placeholder_inputs = [
        engine._frame_entity_placeholder_input(placeholder)
        for placeholder in (frame_input.entity_placeholders or [])
    ]
    if frame_input.entities is not None:
        entity_placeholders, _ = engine._sync_entity_states(
            [engine._frame_entity_state_input(entity) for entity in frame_input.entities]
        )
        placeholder_inputs = entity_placeholders + placeholder_inputs
    if frame_input.entity_placeholders is not None or frame_input.entities is not None:
        engine._sync_entity_placeholders(placeholder_inputs)
        placeholder_count = len(placeholder_inputs)
    if frame_input.force_sources is not None:
        engine._sync_force_sources(
            [engine._frame_force_source_input(force_source) for force_source in frame_input.force_sources]
        )
    if frame_input.emitters is not None:
        engine._sync_persistent_emitters(
            [engine._frame_emitter_input(emitter) for emitter in frame_input.emitters]
        )
    resolved_targets = engine._resolve_target_queries(frame_input.target_queries)
    resolved_change_intents, generated_commands = engine._resolve_change_intents(frame_input.change_intents, resolved_targets)
    resolved_carrier_intents, generated_carrier_commands = engine._resolve_carrier_intents(
        frame_input.carrier_intents,
        resolved_targets,
    )
    observation_pairs = engine._build_observation_request_pairs(frame_input.observation_targets, resolved_targets)
    observation_pairs = [
        (target, engine._assign_readback_request_id(request))
        for target, request in observation_pairs
    ]
    observation_requests = [request for _, request in observation_pairs]
    engine.pending_readbacks.extend(observation_requests)
    engine.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
    observation_plans = [
        engine._serialize_observation_plan_for_target_request(target, request)
        for target, request in observation_pairs
    ]
    queued_observations = len(observation_requests)
    for command in (
        generated_commands
        + generated_carrier_commands
        + engine._resolve_targeted_commands(frame_input.commands, resolved_targets)
    ):
        engine.queue_command(command.kind, **command.payload)
        queued_commands += 1
    readback_requests = engine._resolve_readback_requests(frame_input.readback_requests, resolved_targets)
    readback_requests = [engine._assign_readback_request_id(request) for request in readback_requests]
    engine.pending_readbacks.extend(readback_requests)
    engine.bridge_frame_readback_requests.extend(replace(request) for request in readback_requests)
    readback_plans = engine._serialize_readback_plans_for_requests(readback_requests)
    queued_readbacks = len(readback_requests)
    public_resolved_targets = {
        query_id: engine._public_resolved_target(target)
        for query_id, target in resolved_targets.items()
    }
    public_resolved_change_intents = {
        intent_id: engine._public_resolved_change_intent(intent)
        for intent_id, intent in resolved_change_intents.items()
    }
    public_resolved_carrier_intents = {
        intent_id: engine._public_resolved_carrier_intent(intent)
        for intent_id, intent in resolved_carrier_intents.items()
    }
    return (
        output_controller_state,
        paging_updates,
        public_resolved_targets,
        public_resolved_change_intents,
        public_resolved_carrier_intents,
        observation_plans,
        readback_plans,
        queued_observations,
        queued_readbacks,
        queued_commands,
        placeholder_count,
    )


def _prepare_preview_frame_context(
    engine,
    frame_input: WorldFrameInput,
) -> tuple[
    list[PageStripeUpdate],
    list[tuple[PageStripeUpdate, dict[str, Any]]],
    list[ObservationTarget],
    list[EntityPlaceholder],
    int,
]:
    paging = deepcopy(engine.paging)
    paging_updates: list[PageStripeUpdate] = []
    preview_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
    if frame_input.focus_center is not None:
        paging_updates = paging.focus_on(frame_input.focus_center[0], frame_input.focus_center[1])
    engine.paging = paging
    if paging_updates:
        preview_page_stripes = engine._preview_apply_paging_updates(paging_updates)

    entity_observation_targets: list[ObservationTarget] = []
    if frame_input.entities is None:
        preview_entity_states = dict(engine.entity_states)
        derived_placeholders: list[EntityPlaceholder] = []
        _, entity_observation_targets = engine._frame_entities_to_placeholders_and_observations(
            list(preview_entity_states.values())
        )
    else:
        frame_entities = [engine._frame_entity_state_input(entity) for entity in frame_input.entities]
        preview_entity_states = {entity.entity_id: entity for entity in frame_entities}
        derived_placeholders, entity_observation_targets = engine._frame_entities_to_placeholders_and_observations(frame_entities)
    engine.entity_states = preview_entity_states

    placeholder_inputs = [
        engine._frame_entity_placeholder_input(placeholder)
        for placeholder in (frame_input.entity_placeholders or [])
    ]
    placeholder_count = 0
    if frame_input.entities is not None:
        placeholder_inputs = derived_placeholders + placeholder_inputs
    if frame_input.entities is not None or frame_input.entity_placeholders is not None:
        preview_placeholders, blocked_cells, released_cells = engine._build_preview_entity_placeholders(placeholder_inputs)
        engine.entity_placeholders = preview_placeholders
        engine._resolver_blocked_cells = blocked_cells
        engine._resolver_released_cells = released_cells
        placeholder_count = len(placeholder_inputs)
    else:
        engine._resolver_blocked_cells = None
        engine._resolver_released_cells = None
    return (
        paging_updates,
        preview_page_stripes,
        entity_observation_targets,
        placeholder_inputs,
        placeholder_count,
    )


def _prepare_bridge_frame_inputs(engine) -> None:
    pending_placeholder_dirty_rects = list(engine._pending_placeholder_dirty_rects)
    _clear_bridge_frame_inputs(engine, keep_commands=False, prepared=True)
    if pending_placeholder_dirty_rects:
        engine.bridge_frame_placeholder_dirty_rects.extend(pending_placeholder_dirty_rects)
        engine._pending_placeholder_dirty_rects.clear()


def _needs_pre_simulation_bridge_sync(engine, *, frame_input: WorldFrameInput | None) -> bool:
    if engine.simulation_backend != "gpu":
        return False
    return bool(
        frame_input is not None
        or engine.bridge_frame_placeholders
        or engine.bridge_frame_placeholder_dirty_rects
        or engine.bridge_frame_paging_updates
        or engine.bridge_frame_page_stripes
        or engine._gpu_cpu_dirty_resources
    )


def _clear_bridge_frame_inputs(engine, *, keep_commands: bool, prepared: bool) -> None:
    if not keep_commands:
        engine.bridge_frame_commands.clear()
    engine.bridge_frame_readback_requests.clear()
    engine.bridge_frame_placeholders.clear()
    engine.bridge_frame_placeholder_dirty_rects.clear()
    engine.bridge_frame_paging_updates.clear()
    engine.bridge_frame_page_stripes.clear()
    engine._bridge_inputs_prepared = prepared
