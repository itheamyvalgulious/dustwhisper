from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.sim.gpu_merge import MergeCandidates
from oracle_game.types import (
    CellFlag,
    EntityFeedback,
    ObservationResult,
    PageStripeUpdate,
    ReadbackRequest,
    ReadbackResult,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    WorldFrameInput,
    WorldFrameOutput,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def step(engine: "WorldEngine", dt: float = 1.0 / 60.0, substeps: int = 1) -> None:
    for _ in range(max(1, substeps)):
        frame_input = engine.pending_frame_inputs.popleft() if engine.pending_frame_inputs else None
        output = _step_once(engine, dt, frame_input=frame_input, capture_output=frame_input is not None)
        if output is not None:
            engine.completed_frame_outputs.append(output)


def run_cpu_frame(
    engine: "WorldEngine",
    frame_input: WorldFrameInput | None = None,
    *,
    dt: float = 1.0 / 60.0,
    substeps: int = 1,
) -> WorldFrameOutput:
    output = _step_once(engine, dt, frame_input=frame_input, capture_output=True)
    assert output is not None
    for _ in range(max(1, substeps) - 1):
        _step_once(engine, dt, frame_input=None, capture_output=False)
        output.frame_id = engine.frame_id
    return output


def _step_once(
    engine: "WorldEngine",
    dt: float,
    *,
    frame_input: WorldFrameInput | None,
    capture_output: bool,
) -> WorldFrameOutput | None:
    previous_frame_active = engine._world_simulation_frame_active
    engine._world_simulation_frame_active = True
    try:
        return _step_once_impl(engine, dt, frame_input=frame_input, capture_output=capture_output)
    finally:
        engine._world_simulation_frame_active = previous_frame_active


def _merge_phase_c(engine: "WorldEngine") -> None:
    rxn_pipe = engine.reaction_solver.gpu_pipeline
    rxn_cand = getattr(rxn_pipe, "_phase_c_rxn_candidate", None)
    if rxn_cand is None:
        return
    hr = engine.heat_solver.gpu_pipeline.resources
    mr = engine.motion_solver.gpu_pipeline.resources
    lr = engine.liquid_solver.gpu_pipeline.resources
    candidates = MergeCandidates(
        heat={
            "material": hr.material_tex,
            "phase": hr.phase_tex,
            "temp": hr.temp_ping,
            "integrity": hr.integrity_tex,
            "flags": hr.cell_flags_tex,
        },
        reactions=rxn_cand,
        motion={
            "material": mr.material_tex,
            "phase": mr.phase_tex,
            "velocity": mr.velocity_tex,
        },
        liquid={
            "material": lr.material_in,
            "phase": lr.phase_in,
            "flags": lr.flags_in,
            "timer": lr.timer_in,
            "velocity": lr.velocity_in,
        },
    )
    engine.merge_pipeline.merge_cell_core(engine, candidates)


def _step_once_impl(
    engine: "WorldEngine",
    dt: float,
    *,
    frame_input: WorldFrameInput | None,
    capture_output: bool,
) -> WorldFrameOutput | None:
    engine.last_skipped_gpu_stages = []
    engine.last_pass_profile = {"passes": [], "summary": {}, "skipped_stages": engine.last_skipped_gpu_stages}
    if not engine._bridge_inputs_prepared:
        engine._prepare_bridge_frame_inputs()
    consumed_readbacks: list[ReadbackResult] = []
    resolved_targets: dict[str, ResolvedTarget] = {}
    resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
    resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
    observations: dict[int, ObservationResult] = {}
    entity_feedback: dict[int, EntityFeedback] = {}
    paging_updates: list[PageStripeUpdate] = []
    observation_plans: list[dict[str, Any]] = []
    readback_plans: list[dict[str, Any]] = []
    bridge_upload_snapshot: dict[str, Any] = {}
    bridge_frame_snapshot: dict[str, Any] = {}
    output_controller_state = deepcopy(engine.controller_state_snapshot)
    queued_observations = 0
    queued_readbacks = 0
    queued_commands = 0
    placeholder_count = 0
    with engine._profile_pass("readback"):
        _collect_ready_readbacks(engine, engine.frame_id + 1)
        if capture_output:
            consumed_readbacks = engine.poll_all_readbacks()
            observations = engine._collect_observations(consumed_readbacks)
            entity_feedback = engine._collect_entity_feedback(consumed_readbacks)
    with engine._profile_pass("commands"):
        if frame_input is not None:
            (
                output_controller_state,
                paging_updates,
                resolved_targets,
                resolved_change_intents,
                resolved_carrier_intents,
                observation_plans,
                readback_plans,
                queued_observations,
                queued_readbacks,
                queued_commands,
                placeholder_count,
            ) = engine._apply_frame_input(frame_input)
        else:
            output_controller_state = deepcopy(engine.controller_state_snapshot)

    engine.frame_id += 1
    if capture_output:
        _store_entity_observation_consume_snapshot(engine, 
            frame_id=engine.frame_id,
            consumed_readbacks=consumed_readbacks,
            observations=observations,
            entity_feedback=entity_feedback,
        )
    with engine._profile_pass("commands"):
        engine._apply_commands()
    with engine._profile_pass("pre_sync"):
        if engine._needs_pre_simulation_bridge_sync(frame_input=frame_input):
            engine._sync_pre_simulation_bridge_without_debug_upload()
            engine._gpu_cpu_dirty_resources.clear()
    with engine._profile_pass("commands"):
        persistent_observation_plans = _queue_persistent_entity_observations(engine)
        observation_plans.extend(persistent_observation_plans)
        queued_observations += len(persistent_observation_plans)
    if engine.profile_passes_enabled:
        engine.collapse_solver.gpu_pipeline.reset_pass_profile()
    collapse_pipeline = engine.collapse_solver.gpu_pipeline
    with engine._profile_pass("collapse"):
        if engine._should_run_formal_collapse_this_frame():
            with collapse_pipeline._profile_pass(engine, "dirty_tile_drain"):
                engine._drain_gpu_collapse_structure_dirty_tiles()
            engine.collapse_solver.step(engine)
        else:
            with collapse_pipeline._profile_pass(engine, "scheduled_defer"):
                pass
    collapse_profile = getattr(getattr(engine.collapse_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(collapse_profile, dict):
        engine.last_pass_profile["collapse"] = collapse_profile
    if engine.profile_passes_enabled:
        engine.gas_solver.gpu_pipeline.reset_pass_profile()
    with engine._profile_pass("gas"):
        engine.gas_solver.step(engine, dt)
    gas_profile = getattr(getattr(engine.gas_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(gas_profile, dict):
        engine.last_pass_profile["gas"] = gas_profile
    if engine.profile_passes_enabled:
        engine.heat_solver.gpu_pipeline.reset_pass_profile()
    # Phase C measured ~0.8ms win (A/B 83.56 vs 84.37) but its 1-frame
    # cross-system latency shifts condensation position by 1 frame
    # (test_world_step_condenses_water_gas_into_water_liquid CPU!=GPU).
    # Condensation still occurs (no solver skip), but the 1-frame behavior
    # shift conflicts with the "不丢质量" goal. Disabled; infrastructure kept.
    phase_c_active = False and (
        engine.simulation_backend == "gpu"
        and bool(engine._world_simulation_frame_active)
        and engine.merge_pipeline.available(engine)
    )
    engine.phase_c_defer_cell_publish = phase_c_active
    with engine._profile_pass("heat"):
        engine.heat_solver.step(engine, dt)
    heat_profile = getattr(getattr(engine.heat_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(heat_profile, dict):
        engine.last_pass_profile["heat"] = heat_profile
    engine.reaction_solver.reset_runtime_state(engine)
    if engine.profile_passes_enabled:
        engine.reaction_solver.gpu_pipeline.reset_pass_profile()
    with engine._profile_pass("reactions before motion"):
        engine.reaction_solver.gpu_pipeline.begin_formal_reaction_segment(engine, "before_motion")
        try:
            with engine._profile_pass("reaction_timed"):
                engine.reaction_solver._advance_timed_slots(engine)
            with engine._profile_pass("reaction_self"):
                engine.reaction_solver._run_self_rules(engine)
            with engine._profile_pass("reaction_material_material"):
                engine.reaction_solver._run_material_material(engine)
            with engine._profile_pass("reaction_material_gas"):
                engine.reaction_solver._run_material_gas(engine)
            with engine._profile_pass("reaction_material_light"):
                engine.reaction_solver._run_material_light(engine)
            with engine._profile_pass("reaction_gas_gas"):
                engine.reaction_solver._run_gas_gas(engine)
            with engine._profile_pass("reaction_gas_light"):
                engine.reaction_solver._run_gas_light(engine)
            engine.reaction_solver.gpu_pipeline.flush_formal_reaction_segment(engine, "before_motion")
        finally:
            engine.reaction_solver.gpu_pipeline.end_formal_reaction_segment(engine, "before_motion")
    with engine._profile_pass("motion"):
        engine.motion_solver.step(engine, dt)
    motion_profile = getattr(getattr(engine.motion_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(motion_profile, dict):
        engine.last_pass_profile["motion"] = motion_profile
    with engine._profile_pass("liquid"):
        engine.liquid_solver.step(engine)
    liquid_profile = getattr(getattr(engine.liquid_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(liquid_profile, dict):
        engine.last_pass_profile["liquid"] = liquid_profile
    if phase_c_active:
        with engine._profile_pass("merge_cell_core"):
            _merge_phase_c(engine)
        engine.phase_c_defer_cell_publish = False
    with engine._profile_pass("optics"):
        engine.optics_solver.step(engine)
    optics_profile = getattr(engine.optics_solver, "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(optics_profile, dict):
        engine.last_pass_profile["optics"] = optics_profile
    with engine._profile_pass("latch_clear"):
        if engine.reaction_solver.gpu_pipeline.clear_reaction_latches(engine):
            engine.reaction_solver._note_runtime_backend("gpu")
        else:
            engine._require_gpu_stage("reaction latch clearing")
            engine.cell_flags &= np.uint8(~int(CellFlag.REACTION_LATCHED) & 0xFF)
            engine.reaction_solver._note_runtime_backend("cpu")
    reaction_profile = getattr(getattr(engine.reaction_solver, "gpu_pipeline", None), "last_pass_profile", None)
    if engine.profile_passes_enabled and isinstance(reaction_profile, dict):
        engine.last_pass_profile["reactions"] = reaction_profile
    with engine._profile_pass("active_decay"):
        active_scheduler_gpu_authoritative = (
            engine.simulation_backend == "gpu"
            and "active_tile_ttl" in engine.bridge.gpu_authoritative_resources
        )
        if active_scheduler_gpu_authoritative:
            if not engine.bridge.decay_active_scheduler(engine):
                engine._require_gpu_stage("active scheduler decay")
                raise RuntimeError("GPU active scheduler decay failed; CPU fallback is disabled")
        elif engine.simulation_backend == "gpu":
            engine._require_gpu_stage("active scheduler decay")
        else:
            engine.active.decay()
    bridge_world_synced = False
    if capture_output:
        if engine.simulation_backend != "gpu":
            engine.bridge.sync_world(engine)
            bridge_world_synced = True
        else:
            engine.bridge.sync_force_sources(engine)
        bridge_upload_snapshot = engine.serialize_bridge_upload_snapshot()
        bridge_frame_snapshot = engine.serialize_bridge_frame_snapshot()
    with engine._profile_pass("readback"):
        _finish_readbacks(engine, world_synced=bridge_world_synced)
        _collect_ready_readbacks(engine, engine.frame_id)
    engine._bridge_inputs_prepared = False

    if not capture_output:
        return None
    return WorldFrameOutput(
        frame_id=engine.frame_id,
        submission_id=frame_input.submission_id if frame_input is not None else None,
        controller_state=output_controller_state,
        consumed_readbacks=consumed_readbacks,
        resolved_targets=resolved_targets,
        resolved_change_intents=resolved_change_intents,
        resolved_carrier_intents=resolved_carrier_intents,
        observations=observations,
        entity_feedback=entity_feedback,
        paging_updates=paging_updates,
        observation_plans=observation_plans,
        readback_plans=readback_plans,
        bridge_upload_snapshot=bridge_upload_snapshot,
        bridge_frame_snapshot=bridge_frame_snapshot,
        queued_observations=queued_observations,
        queued_readbacks=queued_readbacks,
        queued_commands=queued_commands,
        placeholder_count=placeholder_count,
    )


def _queue_persistent_entity_observations(engine: "WorldEngine") -> list[dict[str, Any]]:
    if not engine.entity_states:
        return []
    _, observation_targets = engine._frame_entities_to_placeholders_and_observations(list(engine.entity_states.values()))
    observation_pairs = engine._build_observation_request_pairs(observation_targets, {})
    observation_pairs = [
        (target, engine._assign_readback_request_id(request))
        for target, request in observation_pairs
    ]
    observation_requests = [request for _, request in observation_pairs]
    engine.pending_readbacks.extend(observation_requests)
    engine.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
    return [
        engine._serialize_observation_plan_for_target_request(target, request)
        for target, request in observation_pairs
    ]


def _snapshot_preview_runtime_state(engine: "WorldEngine") -> dict[str, Any]:
    return {
        "material_id": engine.material_id.copy(),
        "phase": engine.phase.copy(),
        "cell_flags": engine.cell_flags.copy(),
        "velocity": engine.velocity.copy(),
        "cell_temperature": engine.cell_temperature.copy(),
        "timer_pack": engine.timer_pack.copy(),
        "integrity": engine.integrity.copy(),
        "island_id": engine.island_id.copy(),
        "entity_id": engine.entity_id.copy(),
        "placeholder_displaced_material": engine.placeholder_displaced_material.copy(),
        "collapse_delay_pending": engine.collapse_delay_pending.copy(),
        "flow_velocity": engine.flow_velocity.copy(),
        "ambient_temperature": engine.ambient_temperature.copy(),
        "pressure_ping": engine.pressure_ping.copy(),
        "gas_concentration": engine.gas_concentration.copy(),
        "visible_illumination": engine.visible_illumination.copy(),
        "cell_optical_dose": engine.cell_optical_dose.copy(),
        "gas_optical_dose": engine.gas_optical_dose.copy(),
        "active": deepcopy(engine.active),
        "islands": deepcopy(engine.islands),
        "next_island_id": int(engine.next_island_id),
        "collapse_dirty_regions": list(engine.collapse_dirty_regions),
        "collapse_deferred_regions": list(engine.collapse_deferred_regions),
    }


def _restore_preview_runtime_state(engine: "WorldEngine", snapshot: dict[str, Any]) -> None:
    engine.material_id = snapshot["material_id"]
    engine.phase = snapshot["phase"]
    engine.cell_flags = snapshot["cell_flags"]
    engine.velocity = snapshot["velocity"]
    engine.cell_temperature = snapshot["cell_temperature"]
    engine.timer_pack = snapshot["timer_pack"]
    engine.integrity = snapshot["integrity"]
    engine.island_id = snapshot["island_id"]
    engine.entity_id = snapshot["entity_id"]
    engine.placeholder_displaced_material = snapshot["placeholder_displaced_material"]
    engine.collapse_delay_pending = snapshot["collapse_delay_pending"]
    engine.flow_velocity = snapshot["flow_velocity"]
    engine.ambient_temperature = snapshot["ambient_temperature"]
    engine.pressure_ping = snapshot["pressure_ping"]
    engine.gas_concentration = snapshot["gas_concentration"]
    engine.visible_illumination = snapshot["visible_illumination"]
    engine.cell_optical_dose = snapshot["cell_optical_dose"]
    engine.gas_optical_dose = snapshot["gas_optical_dose"]
    engine.active = snapshot["active"]
    engine.islands = deepcopy(snapshot["islands"])
    engine.next_island_id = int(snapshot["next_island_id"])
    engine.collapse_dirty_regions = snapshot["collapse_dirty_regions"]
    engine.collapse_deferred_regions = snapshot["collapse_deferred_regions"]


def _store_entity_observation_consume_snapshot(
    engine: "WorldEngine",
    *,
    frame_id: int,
    consumed_readbacks: list[ReadbackResult],
    observations: dict[int, ObservationResult],
    entity_feedback: dict[int, EntityFeedback],
) -> dict[str, Any]:
    snapshot = {
        "frame_id": int(frame_id),
        "consumed": len(consumed_readbacks),
        "consumed_readbacks": [engine.serialize_readback_result(result) for result in consumed_readbacks],
        "observations": {
            str(observer_id): engine.serialize_observation_result(result)
            for observer_id, result in observations.items()
        },
        "entity_feedback": {
            str(entity_id): engine.serialize_entity_feedback(feedback)
            for entity_id, feedback in entity_feedback.items()
        },
    }
    engine.last_entity_observation_consume_snapshot = snapshot
    return deepcopy(snapshot)


def _finish_readbacks(engine: "WorldEngine", *, world_synced: bool = False) -> None:
    normalized_requests = [engine._assign_readback_request_id(engine._normalize_readback_request(request)) for request in engine.pending_readbacks]
    engine.pending_readbacks[:] = normalized_requests
    if engine.pending_readbacks and not world_synced and engine.simulation_backend != "gpu":
        engine.bridge.sync_world(engine)
    remaining_pending: list[ReadbackRequest] = []
    readback_upload_dirty = False
    for request in engine.pending_readbacks:
        payload = engine._make_readback_payload(request)
        if not engine.bridge.queue_readback(
            engine.frame_id,
            request,
            payload,
            require_gpu_sources=engine.simulation_backend == "gpu",
        ):
            remaining_pending.append(request)
            continue
        readback_upload_dirty = True
        if request not in engine.bridge_frame_readback_requests:
            engine.bridge_frame_readback_requests.append(replace(request))
        if not any(existing.request_id == request.request_id for existing in engine.inflight_readbacks):
            engine.inflight_readbacks.append(replace(request))
    engine.pending_readbacks[:] = remaining_pending
    if readback_upload_dirty:
        engine.bridge.sync_readback_requests(engine)


def _collect_ready_readbacks(engine: "WorldEngine", current_frame_id: int) -> None:
    while True:
        result = engine.bridge.poll_readback(current_frame_id)
        if result is None:
            return
        if result.request.request_id is not None:
            engine.inflight_readbacks = [
                request for request in engine.inflight_readbacks if request.request_id != result.request.request_id
            ]
            if int(result.request.request_id) in engine.canceled_readback_request_ids:
                continue
        engine.completed_readbacks.append(result)


def _mark_active_rect_runtime(
    engine: "WorldEngine",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    tile_padding: int = 0,
) -> None:
    _mark_active_rects_runtime(engine, [(x0, y0, x1, y1, tile_padding)])


def _mark_active_rects_runtime(
    engine: "WorldEngine",
    rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
) -> None:
    if not rects:
        return
    if engine.simulation_backend == "gpu" and engine._world_simulation_frame_active:
        if not engine.bridge.mark_active_rects(engine, rects):
            engine._require_gpu_stage("active scheduler region marking")
        return
    for rect in rects:
        if len(rect) == 4:
            x0, y0, x1, y1 = rect
            tile_padding = 0
        else:
            x0, y0, x1, y1, tile_padding = rect
        engine.active.mark_rect(int(x0), int(y0), int(x1), int(y1), tile_padding=int(tile_padding))
    if engine.simulation_backend == "gpu":
        engine._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")


