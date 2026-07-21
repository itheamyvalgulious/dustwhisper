from __future__ import annotations

import numpy as np
import pytest

from oracle_game.sim.gpu_reactions import (
    GPUReactionPipeline,
    MAX_MATERIALS,
    RULE_CANDIDATE_VECS,
    RULE_CANDIDATE_WORDS,
)
from oracle_game.sim.gpu_reactions_pairings import _merge_material_pair_candidate_masks
from oracle_game.types import Direction, PairReactionRule, ReactionAction, ReactionType, SelfReactionRule
from oracle_game.world import WorldEngine


def _prepare_pair_world(
    *,
    fused: bool,
    packed_descriptors: bool | None = None,
    fallback_kind: str | None = None,
    prefix_gas_actions: bool = False,
    material_light: bool = False,
    unsafe_material_light: bool = False,
    light_dose_guard: int = 1,
) -> WorldEngine:
    gas_action = ReactionAction(
        ReactionType.MODIFY_GAS,
        gas_species="water_gas",
        speed=0.25,
        duration=4,
        strength=2.0 if fallback_kind == "flow" else 0.0,
        range_cells=3 if fallback_kind == "flow" else 0,
        direction=Direction.RIGHT if fallback_kind == "flow" else Direction.ALL,
    )
    actions = [
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="iron_solid",
            harm_per_frame=1.0,
            integrity_threshold=100.0,
        ),
        gas_action,
    ]
    if prefix_gas_actions:
        actions.extend(
            (
                ReactionAction(
                    ReactionType.MODIFY_GAS,
                    gas_species="water_gas",
                    speed=0.125,
                    duration=2,
                ),
                ReactionAction(
                    ReactionType.MODIFY_GAS,
                    gas_species="water_gas",
                    speed=0.375,
                ),
            )
        )
    if material_light:
        actions.append(
            ReactionAction(
                ReactionType.EMIT_LIGHT,
                light_type="visible_light",
                direction=Direction.ALL,
                strength=1.0,
                range_cells=3,
            )
            if unsafe_material_light
            else ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=10.0)
        )
    mm_rule = PairReactionRule(
        lhs_material="gold_solid",
        rhs_material="iron_solid",
        result_action=1,
    )
    if fallback_kind == "rhs":
        mm_rule = PairReactionRule(
            lhs_material="gold_solid",
            rhs_material="iron_solid",
            result_action=1,
            consume_policy="rhs",
        )
    elif fallback_kind == "emit_light":
        actions.append(
            ReactionAction(
                ReactionType.EMIT_LIGHT,
                light_type="visible_light",
                direction=Direction.ALL,
                strength=1.0,
                range_cells=3,
            )
        )
        mm_rule = PairReactionRule(
            lhs_material="gold_solid",
            rhs_material="iron_solid",
            result_action=3,
        )
    elif fallback_kind == "packed_rate":
        mm_rule = PairReactionRule(
            lhs_material="gold_solid",
            rhs_material="iron_solid",
            rate=0.5,
            result_action=1,
        )
    rules = {
        "material_material": [mm_rule],
        "material_gas": [
            PairReactionRule(
                lhs_material="iron_solid",
                rhs_gas="poison_gas",
                threshold=0.1,
                trigger_slot_index=0,
            )
        ],
        "material_light": (
            [
                PairReactionRule(
                    lhs_material="iron_solid",
                    rhs_light="magic_light",
                    threshold=0.1,
                    result_action=len(actions),
                )
            ]
            if material_light
            else []
        ),
        "gas_gas": [],
        "gas_light": [],
        "self_rules": (
            [SelfReactionRule(material="raw_stone_solid", trigger_slot_index=1)]
            if prefix_gas_actions
            else []
        ),
    }
    engine = WorldEngine(width=32, height=16, gas_cell_size=4)
    pipeline = engine.reaction_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU reaction pipeline is not available")
    pipeline._material_pair_state_fusion_enabled = fused
    pipeline._material_pair_light_state_fusion_enabled = bool(fused and material_light)
    if packed_descriptors is not None:
        pipeline._material_pair_packed_descriptors_enabled = packed_descriptors
    engine.replace_reaction_table(actions, rules)
    iron = engine.rulebook.materials_by_name["iron_solid"]
    engine.patch_material(
        "iron_solid",
        reaction_slots=(2, *iron.reaction_slots[1:]),
    )
    if prefix_gas_actions:
        stone = engine.rulebook.materials_by_name["raw_stone_solid"]
        engine.patch_material(
            "raw_stone_solid",
            reaction_slots=(3, 4, *stone.reaction_slots[2:]),
        )

    gold_id = engine.rulebook.material_id("gold_solid")
    iron_id = engine.rulebook.material_id("iron_solid")
    stone_id = engine.rulebook.material_id("raw_stone_solid")
    pattern = np.asarray((gold_id, gold_id, iron_id, stone_id), dtype=np.int32)
    engine.material_id[:] = np.resize(pattern, engine.material_id.shape)
    for material_id in (gold_id, iron_id, stone_id):
        engine.phase[engine.material_id == material_id] = int(
            engine.rulebook.materials_by_id[material_id].default_phase
        )
    rng = np.random.default_rng(9241)
    engine.cell_flags[:] = rng.integers(0, 4, size=engine.cell_flags.shape, dtype=np.uint8)
    engine.cell_temperature[:] = rng.uniform(-20.0, 140.0, size=engine.cell_temperature.shape).astype(np.float32)
    engine.integrity[:] = np.float32(50.0)
    engine.timer_pack[:] = rng.integers(0, 4, size=engine.timer_pack.shape, dtype=np.uint8)
    engine.timer_pack[..., 0] = 0
    if prefix_gas_actions:
        stone_cells = engine.material_id == stone_id
        engine.timer_pack[stone_cells, 0] = 1
        engine.timer_pack[stone_cells, 1] = 0
    poison_id = engine.rulebook.gas_id("poison_gas")
    engine.gas_concentration.fill(0.0)
    engine.gas_concentration[poison_id] = np.float32(0.6)
    engine.flow_velocity[:] = rng.normal(0.0, 0.1, size=engine.flow_velocity.shape).astype(np.float32)
    engine.active.mark_rect(0, 0, engine.width, engine.height)
    if material_light:
        magic_light = engine.rulebook.light_id("magic_light")
        dose_channel = engine.rulebook.lights_by_id[magic_light].dose_channel_id
        engine.cell_optical_dose[dose_channel].fill(np.float32(0.4))
    engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
    if material_light:
        guard = engine.optics_solver.gpu_pipeline._ensure_light_dose_guard(engine)
        guard.write(np.asarray([light_dose_guard, 0, 0, 0], dtype=np.uint32).tobytes())
    engine.bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "gas_concentration",
        "ambient_temperature",
        "flow_velocity",
        "active_meta",
        "active_tile_ttl",
        "active_chunk_mask",
        *(
            ("cell_optical_dose", "optics_light_dose_guard")
            if material_light
            else ()
        ),
    )
    return engine


def _run_pair_frames(
    *,
    fused: bool,
    packed_descriptors: bool | None = None,
    frame_count: int,
    fallback_kind: str | None = None,
    prefix_gas_actions: bool = False,
    material_light: bool = False,
    unsafe_material_light: bool = False,
    light_dose_guard: int = 1,
    light_fusion_used_out: list[bool] | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[bool, ...]]:
    engine = _prepare_pair_world(
        fused=fused,
        packed_descriptors=packed_descriptors,
        fallback_kind=fallback_kind,
        prefix_gas_actions=prefix_gas_actions,
        material_light=material_light,
        unsafe_material_light=unsafe_material_light,
        light_dose_guard=light_dose_guard,
    )
    pipeline = engine.reaction_solver.gpu_pipeline
    used_fusion: list[bool] = []
    frame_meta: list[np.ndarray] = []
    previous_frame_active = engine._world_simulation_frame_active
    engine._world_simulation_frame_active = True
    try:
        for _ in range(frame_count):
            engine.reaction_solver.reset_runtime_state(engine)
            assert pipeline.begin_formal_reaction_segment(engine, "before_motion")
            try:
                # Match formal frame ordering and retain any timed-stage direct
                # gas delta in the same segment accumulator as the pair stages.
                engine.reaction_solver._advance_timed_slots(engine)
                engine.reaction_solver._run_self_rules(engine)
                did_fuse = engine.reaction_solver._try_run_material_pair_fused(engine)
                if not did_fuse:
                    engine.reaction_solver._run_material_material(engine)
                    engine.reaction_solver._run_material_gas(engine)
                light_fused = bool(did_fuse and pipeline.last_material_pair_fused_light)
                if material_light and not light_fused:
                    engine.reaction_solver._run_material_light(engine)
                used_fusion.append(did_fuse)
                if light_fusion_used_out is not None:
                    light_fusion_used_out.append(light_fused)
                assert pipeline.resources is not None
                frame_meta.append(
                    np.frombuffer(pipeline.resources.segment_cell_meta_tex.read(), dtype=np.float32).copy()
                )
                pipeline.flush_formal_reaction_segment(engine, "before_motion")
            finally:
                pipeline.end_formal_reaction_segment(engine, "before_motion")
            engine.frame_id += 1

        assert pipeline.resources is not None
        state = (
            np.frombuffer(engine.bridge.buffers["cell_core"].read(), dtype=np.uint32).copy(),
            np.frombuffer(engine.bridge.buffers["gas_concentration"].read(), dtype=np.float32).copy(),
            np.frombuffer(engine.bridge.textures["flow_velocity"].read(), dtype=np.float32).copy(),
            np.frombuffer(pipeline.resources.light_emitter_count.read(), dtype=np.uint32).copy(),
            *frame_meta,
        )
        return state, tuple(used_fusion)
    finally:
        engine._world_simulation_frame_active = previous_frame_active
        engine.close()


def test_material_pair_state_fusion_matches_two_passes_for_two_frames() -> None:
    control, control_used = _run_pair_frames(fused=False, frame_count=2)
    candidate, candidate_used = _run_pair_frames(fused=True, frame_count=2)

    assert control_used == (False, False)
    assert candidate_used == (True, True)
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)

    cell_core = candidate[0].reshape((16, 32, 5))
    iron_id = int(cell_core[0, 2, 0] & np.uint32(0xFFFF))
    assert np.array_equal(cell_core[0, :3, 0] & np.uint32(0xFFFF), np.full(3, iron_id, dtype=np.uint32))
    assert np.array_equal(
        cell_core[0, :3, 3] & np.uint32(0xFF),
        np.asarray((4, 3, 3), dtype=np.uint32),
    )
    assert np.any(candidate[1] > 0.0)
    first_meta = candidate[4].reshape((16, 32, 2))
    second_meta = candidate[5].reshape((16, 32, 2))
    assert np.array_equal(first_meta[0, 1], np.asarray((1.0, 1.0), dtype=np.float32))
    assert np.array_equal(second_meta[0, 0], np.asarray((1.0, 1.0), dtype=np.float32))


@pytest.mark.parametrize("fallback_kind", ("rhs", "emit_light", "flow"))
def test_material_pair_state_fusion_falls_back_for_unsafe_rules(fallback_kind: str) -> None:
    control, _ = _run_pair_frames(fused=False, frame_count=1, fallback_kind=fallback_kind)
    candidate, candidate_used = _run_pair_frames(fused=True, frame_count=1, fallback_kind=fallback_kind)

    assert candidate_used == (False,)
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_material_pair_state_fusion_is_enabled_by_default() -> None:
    assert GPUReactionPipeline()._material_pair_state_fusion_enabled is True
    assert GPUReactionPipeline()._material_pair_light_state_fusion_enabled is True
    assert GPUReactionPipeline()._material_pair_packed_descriptors_enabled is True


def test_material_pair_packed_descriptors_match_mask_path_for_two_frames() -> None:
    control, control_used = _run_pair_frames(
        fused=True,
        packed_descriptors=False,
        frame_count=2,
        material_light=True,
    )
    candidate, candidate_used = _run_pair_frames(
        fused=True,
        packed_descriptors=True,
        frame_count=2,
        material_light=True,
    )

    assert control_used == candidate_used == (True, True)
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_material_pair_packed_descriptors_fall_back_to_fused_mask_path() -> None:
    control, control_used = _run_pair_frames(
        fused=True,
        packed_descriptors=False,
        frame_count=2,
        fallback_kind="packed_rate",
    )
    candidate, candidate_used = _run_pair_frames(
        fused=True,
        packed_descriptors=True,
        frame_count=2,
        fallback_kind="packed_rate",
    )

    assert control_used == candidate_used == (True, True)
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_material_pair_state_fusion_accumulates_prefix_gas_delta_exactly() -> None:
    control, _ = _run_pair_frames(
        fused=False,
        frame_count=1,
        prefix_gas_actions=True,
    )
    candidate, candidate_used = _run_pair_frames(
        fused=True,
        frame_count=1,
        prefix_gas_actions=True,
    )
    without_prefix, _ = _run_pair_frames(fused=False, frame_count=1)

    assert candidate_used == (True,)
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)
    assert np.any(control[1] > without_prefix[1])


@pytest.mark.parametrize("light_dose_guard", (0, 1))
def test_material_triplet_state_fusion_matches_three_passes_for_two_frames(
    light_dose_guard: int,
) -> None:
    control_light_fusion: list[bool] = []
    candidate_light_fusion: list[bool] = []
    control, control_pair_fusion = _run_pair_frames(
        fused=False,
        frame_count=2,
        material_light=True,
        light_dose_guard=light_dose_guard,
        light_fusion_used_out=control_light_fusion,
    )
    candidate, candidate_pair_fusion = _run_pair_frames(
        fused=True,
        frame_count=2,
        material_light=True,
        light_dose_guard=light_dose_guard,
        light_fusion_used_out=candidate_light_fusion,
    )

    assert control_pair_fusion == (False, False)
    assert candidate_pair_fusion == (True, True)
    assert control_light_fusion == [False, False]
    assert candidate_light_fusion == [True, True]
    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_material_triplet_state_fusion_falls_back_for_unsafe_light_action() -> None:
    light_fusion_used: list[bool] = []
    _, pair_fusion_used = _run_pair_frames(
        fused=True,
        frame_count=1,
        material_light=True,
        unsafe_material_light=True,
        light_fusion_used_out=light_fusion_used,
    )

    assert pair_fusion_used == (True,)
    assert light_fusion_used == [False]


@pytest.mark.parametrize("material_material_rule_count", (31, 32, 33))
def test_material_pair_candidate_mask_merge_crosses_words_exactly(
    material_material_rule_count: int,
) -> None:
    shape = (MAX_MATERIALS, RULE_CANDIDATE_VECS, 4)
    mm_masks = np.zeros(shape, dtype=np.uint32)
    mg_masks = np.zeros(shape, dtype=np.uint32)
    mm_words = mm_masks.reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    mg_words = mg_masks.reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    mm_indices = (0, material_material_rule_count - 1)
    mg_indices = (0, 1, 2)
    for rule_index in mm_indices:
        mm_words[7, rule_index // 32] |= np.uint32(1 << (rule_index % 32))
    for rule_index in mg_indices:
        mg_words[7, rule_index // 32] |= np.uint32(1 << (rule_index % 32))
    mg_words[8, 1] |= np.uint32(1 << 5)

    merged = _merge_material_pair_candidate_masks(
        mm_masks,
        mg_masks,
        material_material_rule_count,
        38,
    ).reshape((MAX_MATERIALS, RULE_CANDIDATE_WORDS))
    expected = np.zeros_like(merged)
    for rule_index in mm_indices:
        expected[7, rule_index // 32] |= np.uint32(1 << (rule_index % 32))
    for rule_index in mg_indices:
        shifted = material_material_rule_count + rule_index
        expected[7, shifted // 32] |= np.uint32(1 << (shifted % 32))
    shifted = material_material_rule_count + 37
    expected[8, shifted // 32] |= np.uint32(1 << (shifted % 32))

    assert np.array_equal(merged, expected)
