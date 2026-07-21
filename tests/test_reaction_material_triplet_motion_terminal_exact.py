from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    clear_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
)
from oracle_game.sim.gpu_reactions import GPUReactionPipeline
from oracle_game.types import CellFlag, PairReactionRule, Phase, ReactionAction, ReactionType
from oracle_game.world import WorldEngine


_BRIDGE_BUFFER_NAMES = (
    "cell_core",
    "gas_concentration",
    "island_id",
    "entity_id",
    "placeholder_displaced_material",
    "cell_optical_dose",
    "active_meta",
    "active_tile_ttl",
    "active_chunk_mask",
    "optics_light_dose_guard",
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
)
_BRIDGE_TEXTURE_NAMES = (
    "material",
    "flow_velocity",
    "ambient_temperature",
)
_REACTION_STATE_ATTRS = (
    "_formal_state_cache_key",
    "_formal_active_mask_cache_key",
    "_formal_loaded_bridge_inputs_key",
    "_formal_loaded_bridge_inputs",
    "_formal_segment_batch_base_key",
    "_formal_segment_batch_key",
    "_formal_segment_meta_lazy_key",
    "_formal_segment_meta_logically_zero",
    "_formal_segment_meta_physically_cleared",
    "_formal_segment_all_prior_cell_meta_in_flags",
    "_formal_light_counters_cleared_key",
    "_formal_pending_bridge_publish_key",
    "_formal_pending_bridge_publish",
    "_formal_pending_gas_delta_key",
    "_formal_cell_state_role_key",
    "_formal_cell_state_read_role",
    "_formal_velocity_state_role_key",
    "_formal_velocity_state_read_role",
    "_formal_external_cell_state_key",
    "_formal_external_cell_state_textures",
    "_formal_external_cell_flags_texture",
    "_phase_c_rxn_candidate",
    "_motion_handoff_candidate",
    "last_material_pair_fused_light",
    "last_material_pair_terminal_handoff",
    "last_terminal_segment_meta_lazy_zero_used",
    "last_segment_meta_lazy_clear_skipped",
    "segment_meta_lazy_fallback_clear_count",
)
_WORLD_STATE_ATTRS = (
    "frame_id",
    "_gpu_collapse_structure_dirty_tiles_pending",
    "_reaction_latches_handoff_cleared_frame_id",
    "phase_c_defer_cell_publish",
)


def test_material_triplet_terminal_dirty_fast_equal_is_default_enabled() -> None:
    assert GPUReactionPipeline()._material_triplet_terminal_dirty_fast_equal_enabled is True


def test_material_triplet_terminal_shared_transpose_is_default_enabled() -> None:
    assert GPUReactionPipeline()._material_triplet_terminal_shared_transpose_enabled is True


def _prepare_terminal_world(*, width: int = 67, height: int = 67) -> WorldEngine:
    actions = [
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="sand_powder",
            harm_per_frame=1.0,
            integrity_threshold=100.0,
        ),
        ReactionAction(
            ReactionType.MODIFY_GAS,
            gas_species="water_gas",
            speed=0.25,
            duration=4,
        ),
        ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=10.0),
    ]
    rules = {
        "material_material": [
            PairReactionRule(
                lhs_material="gold_solid",
                rhs_material="iron_solid",
                result_action=1,
            )
        ],
        "material_gas": [
            PairReactionRule(
                lhs_material="iron_solid",
                rhs_gas="poison_gas",
                threshold=0.1,
                trigger_slot_index=0,
            )
        ],
        "material_light": [
            PairReactionRule(
                lhs_material="iron_solid",
                rhs_light="magic_light",
                threshold=0.1,
                result_action=3,
            )
        ],
        "gas_gas": [],
        "gas_light": [],
        "self_rules": [],
    }
    engine = WorldEngine(width=width, height=height, gas_cell_size=4)
    pipeline = engine.reaction_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU reaction pipeline is not available")

    engine.replace_reaction_table(actions, rules)
    iron = engine.rulebook.materials_by_name["iron_solid"]
    engine.patch_material("iron_solid", reaction_slots=(2, *iron.reaction_slots[1:]))

    gold_id = engine.rulebook.material_id("gold_solid")
    iron_id = engine.rulebook.material_id("iron_solid")
    pattern = np.asarray((gold_id, iron_id, iron_id, gold_id), dtype=np.int32)
    engine.material_id[:] = np.resize(pattern, engine.material_id.shape)
    for material_id in (gold_id, iron_id):
        engine.phase[engine.material_id == material_id] = int(
            engine.rulebook.materials_by_id[material_id].default_phase
        )

    rng = np.random.default_rng(7731)
    engine.cell_flags.fill(int(CellFlag.PHASE_LOCKED | CellFlag.REACTION_LATCHED))
    engine.cell_temperature[:] = rng.uniform(-20.0, 140.0, size=engine.cell_temperature.shape).astype("f4")
    engine.integrity.fill(50.0)
    engine.timer_pack[:] = rng.integers(0, 5, size=engine.timer_pack.shape, dtype=np.uint8)
    engine.velocity[:] = rng.normal(0.0, 0.75, size=engine.velocity.shape).astype("f4")
    engine.flow_velocity[:] = rng.normal(0.0, 0.1, size=engine.flow_velocity.shape).astype("f4")
    engine.ambient_temperature[:] = rng.uniform(-30.0, 80.0, size=engine.ambient_temperature.shape).astype("f4")
    engine.gas_concentration.fill(0.0)
    engine.gas_concentration[engine.rulebook.gas_id("poison_gas")].fill(0.6)
    magic_light = engine.rulebook.light_id("magic_light")
    magic_dose_channel = engine.rulebook.lights_by_id[magic_light].dose_channel_id
    engine.cell_optical_dose[magic_dose_channel].fill(0.4)

    engine.active.mark_rect(0, 0, engine.width, engine.height)
    for tile_y in range(engine.active.tile_height):
        for tile_x in range(engine.active.tile_width):
            engine.active.active_tile_ttl[tile_y][tile_x] = (
                0 if (tile_x, tile_y) == (1, 1) else 3
            )
    engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
    guard = engine.optics_solver.gpu_pipeline._ensure_light_dose_guard(engine)
    guard.write(np.asarray([1, 0, 0, 0], dtype=np.uint32).tobytes())
    ensure_collapse_structure_dirty_tile_mask(engine, clear=True)
    ensure_collapse_structure_dirty_tile_queue(engine, clear=True)
    engine.bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "gas_concentration",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "ambient_temperature",
        "flow_velocity",
        "cell_optical_dose",
        "optics_light_dose_guard",
        "active_meta",
        "active_tile_ttl",
        "active_chunk_mask",
    )
    return engine


def _read_writable_resource(resource: Any) -> bytes | None:
    if not callable(getattr(resource, "read", None)) or not callable(getattr(resource, "write", None)):
        return None
    try:
        return resource.read()
    except Exception:
        return None


def _snapshot_resource_fields(resource_owner: Any) -> tuple[dict[str, tuple[Any, bytes]], dict[str, Any]]:
    gpu_payloads: dict[str, tuple[Any, bytes]] = {}
    metadata: dict[str, Any] = {}
    for field in fields(resource_owner):
        value = getattr(resource_owner, field.name)
        payload = _read_writable_resource(value)
        if payload is None:
            metadata[field.name] = deepcopy(value)
        else:
            gpu_payloads[field.name] = (value, payload)
    return gpu_payloads, metadata


def _restore_resource_fields(
    resource_owner: Any,
    snapshot: tuple[dict[str, tuple[Any, bytes]], dict[str, Any]],
) -> None:
    gpu_payloads, metadata = snapshot
    for name, (resource, payload) in gpu_payloads.items():
        assert getattr(resource_owner, name) is resource, f"checkpointed resource {name} was replaced"
        resource.write(payload)
    for name, value in metadata.items():
        setattr(resource_owner, name, deepcopy(value))


def _snapshot_mapping(mapping: dict[str, Any], names: tuple[str, ...]) -> dict[str, tuple[Any, bytes]]:
    snapshot: dict[str, tuple[Any, bytes]] = {}
    for name in names:
        resource = mapping.get(name)
        assert resource is not None, f"missing checkpoint resource {name}"
        payload = _read_writable_resource(resource)
        assert payload is not None, f"resource {name} cannot be checkpointed"
        snapshot[name] = (resource, payload)
    return snapshot


def _restore_mapping(mapping: dict[str, Any], snapshot: dict[str, tuple[Any, bytes]]) -> None:
    for name, (resource, payload) in snapshot.items():
        assert mapping.get(name) is resource, f"checkpointed bridge resource {name} was replaced"
        resource.write(payload)


def _snapshot_attrs(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: deepcopy(getattr(owner, name, None)) for name in names}


def _restore_attrs(owner: Any, snapshot: dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(owner, name, deepcopy(value))


def _take_checkpoint(engine: WorldEngine) -> dict[str, Any]:
    reaction = engine.reaction_solver.gpu_pipeline
    motion = engine.motion_solver.gpu_pipeline
    assert reaction.resources is not None and motion.resources is not None
    ctx = engine.bridge.ctx
    assert ctx is not None
    ctx.finish()
    return {
        "bridge_buffers": _snapshot_mapping(engine.bridge.buffers, _BRIDGE_BUFFER_NAMES),
        "bridge_textures": _snapshot_mapping(engine.bridge.textures, _BRIDGE_TEXTURE_NAMES),
        "reaction_resources": _snapshot_resource_fields(reaction.resources),
        "motion_resources": _snapshot_resource_fields(motion.resources),
        "reaction_attrs": _snapshot_attrs(reaction, _REACTION_STATE_ATTRS),
        "motion_attrs": _snapshot_attrs(motion, ("last_cpu_mirror_downloaded",)),
        "heat_attrs": _snapshot_attrs(
            engine.heat_solver.gpu_pipeline,
            ("_motion_handoff_candidate", "_deferred_cell_core_frame_id"),
        ),
        "world_attrs": _snapshot_attrs(engine, _WORLD_STATE_ATTRS),
        "authoritative": set(engine.bridge.gpu_authoritative_resources),
    }


def _restore_checkpoint(engine: WorldEngine, checkpoint: dict[str, Any]) -> None:
    reaction = engine.reaction_solver.gpu_pipeline
    motion = engine.motion_solver.gpu_pipeline
    assert reaction.resources is not None and motion.resources is not None
    _restore_mapping(engine.bridge.buffers, checkpoint["bridge_buffers"])
    _restore_mapping(engine.bridge.textures, checkpoint["bridge_textures"])
    _restore_resource_fields(reaction.resources, checkpoint["reaction_resources"])
    _restore_resource_fields(motion.resources, checkpoint["motion_resources"])
    _restore_attrs(reaction, checkpoint["reaction_attrs"])
    _restore_attrs(motion, checkpoint["motion_attrs"])
    _restore_attrs(engine.heat_solver.gpu_pipeline, checkpoint["heat_attrs"])
    _restore_attrs(engine, checkpoint["world_attrs"])
    engine.bridge.gpu_authoritative_resources.clear()
    engine.bridge.gpu_authoritative_resources.update(checkpoint["authoritative"])


def _sorted_live_pairs(raw: bytes, count: int) -> bytes:
    pairs = np.frombuffer(raw, dtype=np.int32).reshape((-1, 2))[:count].copy()
    if pairs.size:
        pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
    return pairs.tobytes()


def _capture_terminal_state(engine: WorldEngine) -> dict[str, object]:
    motion = engine.motion_solver.gpu_pipeline
    resources = motion.resources
    ctx = engine.bridge.ctx
    assert resources is not None and ctx is not None
    ctx.finish()
    dirty_count_raw = engine.bridge.buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER].read()
    dirty_count = int(np.frombuffer(dirty_count_raw, dtype=np.uint32, count=1)[0])
    active_count_raw = resources.active_tile_count.read()
    active_count = int(np.frombuffer(active_count_raw, dtype=np.uint32, count=1)[0])
    return {
        "bridge.cell_core": engine.bridge.buffers["cell_core"].read(),
        "bridge.material": engine.bridge.textures["material"].read(),
        "bridge.gas": engine.bridge.buffers["gas_concentration"].read(),
        "bridge.flow": engine.bridge.textures["flow_velocity"].read(),
        "bridge.ambient": engine.bridge.textures["ambient_temperature"].read(),
        "bridge.active_meta": engine.bridge.buffers["active_meta"].read(),
        "bridge.active_ttl": engine.bridge.buffers["active_tile_ttl"].read(),
        "bridge.active_chunk": engine.bridge.buffers["active_chunk_mask"].read(),
        "bridge.dirty_mask": engine.bridge.buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER].read(),
        "bridge.dirty_count": dirty_count_raw,
        "bridge.dirty_dispatch": engine.bridge.buffers[
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER
        ].read(),
        "bridge.dirty_list_live_sorted": _sorted_live_pairs(
            engine.bridge.buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER].read(),
            dirty_count,
        ),
        "motion.active_tile": resources.active_tile_tex.read(),
        "motion.active_count": active_count_raw,
        "motion.active_dispatch": resources.active_tile_dispatch_args.read(),
        "motion.active_list_live_sorted": _sorted_live_pairs(
            resources.active_tile_list.read(),
            active_count,
        ),
        "world.dirty_pending": bool(engine._gpu_collapse_structure_dirty_tiles_pending),
        "world.latch_clear_frame": engine._reaction_latches_handoff_cleared_frame_id,
        "reaction.handoff_consumed": engine.reaction_solver.gpu_pipeline._motion_handoff_candidate is None,
        "motion.cpu_mirror": bool(motion.last_cpu_mirror_downloaded),
    }


def _run_terminal_frame(engine: WorldEngine, *, terminal: bool, dt: float) -> bool:
    reaction = engine.reaction_solver.gpu_pipeline
    reaction._material_triplet_motion_terminal_enabled = terminal
    engine._reaction_motion_terminal_dt = dt
    engine.reaction_solver.reset_runtime_state(engine)
    assert reaction.begin_formal_reaction_segment(engine, "before_motion")
    try:
        engine.reaction_solver._advance_timed_slots(engine)
        engine.reaction_solver._run_self_rules(engine)
        assert engine.reaction_solver._try_run_material_pair_fused(engine)
        used_terminal = bool(reaction.last_material_pair_terminal_handoff)
        engine.reaction_solver._run_gas_gas(engine)
        engine.reaction_solver._run_gas_light(engine)
        reaction.flush_formal_reaction_segment(engine, "before_motion")
    finally:
        reaction.end_formal_reaction_segment(engine, "before_motion")
    engine.motion_solver.gpu_pipeline.integrate_velocity(
        engine,
        dt,
        solve_tile_mask=np.zeros(
            (engine.active.tile_height, engine.active.tile_width),
            dtype=np.bool_,
        ),
    )
    return used_terminal


def _run_terminal_sequence(
    engine: WorldEngine,
    terminal_frames: tuple[bool, bool],
) -> tuple[list[dict[str, object]], tuple[bool, bool]]:
    snapshots: list[dict[str, object]] = []
    used_terminal: list[bool] = []
    for terminal in terminal_frames:
        used_terminal.append(_run_terminal_frame(engine, terminal=terminal, dt=1.0 / 60.0))
        snapshots.append(_capture_terminal_state(engine))
        engine.frame_id += 1
    return snapshots, (used_terminal[0], used_terminal[1])


def _run_lazy_terminal_sequence(
    engine: WorldEngine,
    terminal_frames: tuple[bool, bool],
) -> tuple[list[dict[str, object]], tuple[bool, bool], tuple[bool, bool]]:
    snapshots: list[dict[str, object]] = []
    used_terminal: list[bool] = []
    used_lazy_zero: list[bool] = []
    reaction = engine.reaction_solver.gpu_pipeline
    for terminal in terminal_frames:
        used_terminal.append(_run_terminal_frame(engine, terminal=terminal, dt=1.0 / 60.0))
        used_lazy_zero.append(bool(reaction.last_terminal_segment_meta_lazy_zero_used))
        snapshots.append(_capture_terminal_state(engine))
        engine.frame_id += 1
    return (
        snapshots,
        (used_terminal[0], used_terminal[1]),
        (used_lazy_zero[0], used_lazy_zero[1]),
    )


def _capture_phase_c_candidate(engine: WorldEngine) -> dict[str, bytes]:
    candidate = engine.reaction_solver.gpu_pipeline._phase_c_rxn_candidate
    assert isinstance(candidate, dict)
    assert engine.bridge.ctx is not None
    engine.bridge.ctx.finish()
    return {
        name: candidate[name].read()
        for name in ("cell_state", "temp", "integrity", "velocity", "timer", "meta")
    }


def _run_phase_c_sequence(engine: WorldEngine) -> tuple[list[dict[str, bytes]], tuple[bool, bool]]:
    reaction = engine.reaction_solver.gpu_pipeline
    snapshots: list[dict[str, bytes]] = []
    lazy_zero_used: list[bool] = []
    engine.phase_c_defer_cell_publish = True
    for _frame_index in range(2):
        engine._reaction_motion_terminal_dt = 1.0 / 60.0
        engine.reaction_solver.reset_runtime_state(engine)
        assert reaction.begin_formal_reaction_segment(engine, "before_motion")
        try:
            engine.reaction_solver._advance_timed_slots(engine)
            engine.reaction_solver._run_self_rules(engine)
            assert engine.reaction_solver._try_run_material_pair_fused(engine)
            assert not reaction.last_material_pair_terminal_handoff
            engine.reaction_solver._run_gas_gas(engine)
            engine.reaction_solver._run_gas_light(engine)
            assert reaction.flush_formal_reaction_segment(engine, "before_motion")
            snapshots.append(_capture_phase_c_candidate(engine))
            lazy_zero_used.append(bool(reaction.last_terminal_segment_meta_lazy_zero_used))
        finally:
            reaction.end_formal_reaction_segment(engine, "before_motion")
        engine.frame_id += 1
    return snapshots, (lazy_zero_used[0], lazy_zero_used[1])


def test_material_triplet_motion_terminal_matches_handoff_with_checkpoint_restore() -> None:
    engine = _prepare_terminal_world()
    previous_frame_active = engine._world_simulation_frame_active
    previous_reaction_handoff = bool(getattr(engine, "reaction_motion_handoff_active", False))
    engine._world_simulation_frame_active = True
    engine.reaction_motion_handoff_active = True
    try:
        reaction = engine.reaction_solver.gpu_pipeline
        motion = engine.motion_solver.gpu_pipeline
        assert engine.bridge.ctx is not None
        reaction._ensure_programs(engine.bridge.ctx)
        reaction._ensure_resources(engine)
        motion._ensure_programs(engine.bridge.ctx)
        motion._ensure_resources(engine)
        clear_collapse_structure_dirty_tile_mask(engine)
        checkpoint = _take_checkpoint(engine)

        control, control_used = _run_terminal_sequence(engine, (False, False))
        _restore_checkpoint(engine, checkpoint)
        consecutive, consecutive_used = _run_terminal_sequence(engine, (True, True))
        _restore_checkpoint(engine, checkpoint)
        candidate_to_fallback, mixed_used = _run_terminal_sequence(engine, (True, False))

        assert control_used == (False, False)
        assert consecutive_used == (True, True)
        assert mixed_used == (True, False)
        for scenario, snapshots in (
            ("consecutive", consecutive),
            ("candidate-to-fallback", candidate_to_fallback),
        ):
            for frame_index, (expected, actual) in enumerate(
                zip(control, snapshots, strict=True),
                start=1,
            ):
                assert actual.keys() == expected.keys()
                for resource_name in expected:
                    assert actual[resource_name] == expected[resource_name], (
                        f"{scenario} frame {frame_index} differs in {resource_name}"
                    )

        dirty_count = int(
            np.frombuffer(consecutive[0]["bridge.dirty_count"], dtype=np.uint32, count=1)[0]
        )
        assert dirty_count > 0
        core = np.frombuffer(consecutive[0]["bridge.cell_core"], dtype=np.uint32).reshape(
            (engine.height, engine.width, 5)
        )
        flags = ((core[..., 0] >> np.uint32(24)) & np.uint32(0xFF)).astype(np.uint8)
        assert not np.any(flags & np.uint8(int(CellFlag.REACTION_LATCHED)))
        assert np.any(flags == 0)
        assert np.any(flags == np.uint8(int(CellFlag.PHASE_LOCKED)))
    finally:
        engine._world_simulation_frame_active = previous_frame_active
        engine.reaction_motion_handoff_active = previous_reaction_handoff
        engine.close()


def test_terminal_segment_meta_lazy_zero_matches_clear_then_fallback_two_frames() -> None:
    engine = _prepare_terminal_world()
    previous_frame_active = engine._world_simulation_frame_active
    previous_reaction_handoff = bool(getattr(engine, "reaction_motion_handoff_active", False))
    engine._world_simulation_frame_active = True
    engine.reaction_motion_handoff_active = True
    try:
        reaction = engine.reaction_solver.gpu_pipeline
        motion = engine.motion_solver.gpu_pipeline
        assert engine.bridge.ctx is not None
        reaction._ensure_programs(engine.bridge.ctx)
        reaction._ensure_resources(engine)
        motion._ensure_programs(engine.bridge.ctx)
        motion._ensure_resources(engine)
        checkpoint = _take_checkpoint(engine)

        reaction._terminal_segment_meta_lazy_zero_enabled = False
        control, control_terminal, control_lazy = _run_lazy_terminal_sequence(
            engine,
            (True, False),
        )
        _restore_checkpoint(engine, checkpoint)
        reaction._terminal_segment_meta_lazy_zero_enabled = True
        candidate, candidate_terminal, candidate_lazy = _run_lazy_terminal_sequence(
            engine,
            (True, False),
        )

        assert control_terminal == candidate_terminal == (True, False)
        assert control_lazy == (False, False)
        assert candidate_lazy == (True, False)
        assert candidate == control
    finally:
        engine._world_simulation_frame_active = previous_frame_active
        engine.reaction_motion_handoff_active = previous_reaction_handoff
        engine.close()


def test_terminal_segment_meta_lazy_zero_phase_c_handoff_is_raw_exact_two_frames() -> None:
    engine = _prepare_terminal_world()
    previous_frame_active = engine._world_simulation_frame_active
    engine._world_simulation_frame_active = True
    try:
        reaction = engine.reaction_solver.gpu_pipeline
        motion = engine.motion_solver.gpu_pipeline
        assert engine.bridge.ctx is not None
        reaction._ensure_programs(engine.bridge.ctx)
        reaction._ensure_resources(engine)
        motion._ensure_programs(engine.bridge.ctx)
        motion._ensure_resources(engine)
        checkpoint = _take_checkpoint(engine)

        reaction._terminal_segment_meta_lazy_zero_enabled = False
        control, control_lazy = _run_phase_c_sequence(engine)
        _restore_checkpoint(engine, checkpoint)
        reaction._terminal_segment_meta_lazy_zero_enabled = True
        candidate, candidate_lazy = _run_phase_c_sequence(engine)

        assert control_lazy == candidate_lazy == (False, False)
        assert candidate == control
        assert reaction.segment_meta_lazy_fallback_clear_count >= 2
    finally:
        engine._world_simulation_frame_active = previous_frame_active
        engine.close()
