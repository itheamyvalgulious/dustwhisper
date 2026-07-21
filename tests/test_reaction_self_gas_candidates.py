from __future__ import annotations

import numpy as np
import pytest

from oracle_game.types import Direction, ReactionAction, ReactionType, SelfReactionRule
from oracle_game.world import WorldEngine


MATERIALS = (
    "gold_solid",
    "raw_stone_solid",
    "root_solid",
    "water_liquid",
    "acid_liquid",
    "sand_powder",
)


def _run_dense_self_gas_case(
    candidate_enabled: bool,
    *,
    material_spans_enabled: bool = True,
) -> tuple[np.ndarray, ...]:
    actions = [
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="water_gas", speed=0.25, strength=2.0, range_cells=3, direction=Direction.RIGHT, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="poison_gas", speed=0.5, strength=3.0, range_cells=4, direction=Direction.LEFT, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="oil_gas", speed=-0.125, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="water_gas", speed=0.75, strength=4.0, range_cells=5, direction=Direction.ALL, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="poison_gas", speed=-0.25, strength=2.5, range_cells=2, direction=Direction.UP, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="oil_gas", speed=0.375, strength=1.5, range_cells=6, direction=Direction.DOWN, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="water_gas", speed=-0.5, duration=0),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="poison_gas", speed=0.125, strength=5.0, range_cells=3, direction=Direction.RIGHT, duration=0),
    ]
    rules = [
        SelfReactionRule(material=material, trigger_slot_index=slot)
        for slot in range(8)
        for material in MATERIALS
    ]
    engine = WorldEngine(width=48, height=32, gas_cell_size=4)
    try:
        pipeline = engine.reaction_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU reaction pipeline is not available")
        pipeline._self_gas_candidate_worklist_enabled = candidate_enabled
        pipeline._self_rule_material_spans_enabled = material_spans_enabled
        engine.replace_reaction_table(
            actions,
            {
                "material_material": [],
                "material_gas": [],
                "material_light": [],
                "gas_gas": [],
                "gas_light": [],
                "self_rules": rules,
            },
        )
        for material in MATERIALS:
            engine.patch_material(material, reaction_slots=tuple(range(1, 9)))

        rng = np.random.default_rng(7719)
        material_ids = np.asarray(
            [engine.rulebook.material_id(material) for material in MATERIALS],
            dtype=np.int32,
        )
        phase_lookup = np.zeros((int(material_ids.max()) + 1,), dtype=np.uint8)
        for material_id in material_ids:
            phase_lookup[material_id] = int(
                engine.rulebook.materials_by_id[int(material_id)].default_phase
            )
        engine.material_id[:] = rng.choice(material_ids, size=engine.material_id.shape)
        engine.phase[:] = phase_lookup[engine.material_id]
        engine.cell_flags[:] = rng.integers(0, 4, size=engine.cell_flags.shape, dtype=np.uint8)
        engine.cell_temperature[:] = rng.normal(25.0, 30.0, size=engine.cell_temperature.shape).astype(np.float32)
        engine.integrity[:] = rng.uniform(2.0, 100.0, size=engine.integrity.shape).astype(np.float32)
        engine.timer_pack.fill(0)
        engine.gas_concentration[:] = rng.uniform(0.0, 0.3, size=engine.gas_concentration.shape).astype(np.float32)
        engine.flow_velocity[:] = rng.normal(0.0, 0.1, size=engine.flow_velocity.shape).astype(np.float32)
        engine.active.mark_rect(0, 0, engine.width, engine.height)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "gas_concentration",
            "ambient_temperature",
            "flow_velocity",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )

        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            engine.reaction_solver.reset_runtime_state(engine)
            assert pipeline.begin_formal_reaction_segment(engine, "before_motion")
            engine.reaction_solver._run_self_rules(engine)
            assert pipeline.resources is not None
            counters = np.frombuffer(
                pipeline.resources.light_emitter_count.read(), dtype=np.uint32
            ).copy()
            assert pipeline.flush_formal_reaction_segment(engine, "before_motion")
        finally:
            pipeline.end_formal_reaction_segment(engine, "before_motion")
            engine._world_simulation_frame_active = previous_frame_active

        return (
            np.frombuffer(engine.bridge.buffers["cell_core"].read(), dtype=np.uint32).copy(),
            np.frombuffer(engine.bridge.buffers["gas_concentration"].read(), dtype=np.float32).copy(),
            np.frombuffer(engine.bridge.textures["flow_velocity"].read(), dtype=np.float32).copy(),
            counters,
        )
    finally:
        engine.close()


def test_self_gas_candidate_worklist_matches_full_scatter_with_dense_collisions() -> None:
    control = _run_dense_self_gas_case(False)
    candidate = _run_dense_self_gas_case(True)

    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_self_rule_material_spans_match_full_rule_scan_with_interleaved_rules() -> None:
    control = _run_dense_self_gas_case(True, material_spans_enabled=False)
    candidate = _run_dense_self_gas_case(True, material_spans_enabled=True)

    for control_state, candidate_state in zip(control, candidate, strict=True):
        assert np.array_equal(candidate_state, control_state)


def test_self_rule_material_span_metadata_is_grouped_and_cache_invalidates() -> None:
    rules = [
        SelfReactionRule(material="root_solid", trigger_slot_index=0),
        SelfReactionRule(material="gold_solid", trigger_slot_index=1),
        SelfReactionRule(material="water_liquid", trigger_slot_index=2),
        SelfReactionRule(material="root_solid", trigger_slot_index=3),
        SelfReactionRule(material="gold_solid", trigger_slot_index=4),
        SelfReactionRule(material="acid_liquid", trigger_slot_index=5),
        SelfReactionRule(material="water_liquid", trigger_slot_index=6),
    ]
    empty_pair_rules = {
        "material_material": [],
        "material_gas": [],
        "material_light": [],
        "gas_gas": [],
        "gas_light": [],
    }
    engine = WorldEngine(width=16, height=16, gas_cell_size=4)
    try:
        pipeline = engine.reaction_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU reaction pipeline is not available")
        engine.replace_reaction_table(
            [],
            {**empty_pair_rules, "self_rules": rules},
        )
        resources = pipeline._ensure_resources(engine)

        def read_metadata() -> tuple[np.ndarray, np.ndarray]:
            pipeline._upload_local_metadata(
                engine,
                resources,
                include_self_rules=True,
            )
            compiled = (
                np.frombuffer(resources.self_rule_i.read(), dtype=np.int32)
                .reshape((-1, 4))
                .copy()
            )
            material_tags = (
                np.frombuffer(resources.material_tags.read(), dtype=np.uint32)
                .reshape((-1, 4))
                .copy()
            )
            return compiled, material_tags

        def assert_grouped_metadata(
            expected_rules: list[SelfReactionRule],
            compiled: np.ndarray,
            material_tags: np.ndarray,
        ) -> None:
            ordered = sorted(
                enumerate(expected_rules),
                key=lambda entry: (
                    engine.rulebook.material_id(entry[1].material),
                    entry[0],
                ),
            )
            expected_rows = np.asarray(
                [
                    (
                        engine.rulebook.material_id(rule.material),
                        rule.trigger_slot_index,
                    )
                    for _, rule in ordered
                ],
                dtype=np.int32,
            )
            assert np.array_equal(compiled[: len(expected_rules), :2], expected_rows)

            expected_spans: dict[int, tuple[int, int]] = {}
            for row_index, (material_id, _) in enumerate(expected_rows):
                start, count = expected_spans.get(int(material_id), (row_index, 0))
                expected_spans[int(material_id)] = (start, count + 1)
            for material_id, (start, count) in expected_spans.items():
                encoded = int(material_tags[material_id, 3])
                assert encoded & 0xFFFF == start
                assert encoded >> 16 == count
                assert np.all(compiled[start : start + count, 0] == material_id)

            unused_material_id = engine.rulebook.material_id("raw_stone_solid")
            assert unused_material_id not in expected_spans
            assert int(material_tags[unused_material_id, 3]) == 0

        first_compiled, first_tags = read_metadata()
        assert_grouped_metadata(rules, first_compiled, first_tags)
        first_signature = resources.self_rule_signature
        assert first_signature is not None

        sentinel_compiled = np.full(first_compiled.shape, -777, dtype=np.int32)
        sentinel_tags = first_tags.copy()
        sentinel_tags[:, 3] = np.uint32(0xDEADBEEF)
        resources.self_rule_i.write(sentinel_compiled.tobytes())
        resources.material_tags.write(sentinel_tags.tobytes())
        cached_compiled, cached_tags = read_metadata()
        assert np.array_equal(cached_compiled, sentinel_compiled)
        assert np.array_equal(cached_tags, sentinel_tags)
        assert resources.self_rule_signature == first_signature

        engine.patch_reaction_rule(
            "self_rules",
            0,
            material="acid_liquid",
            trigger_slot_index=7,
        )
        updated_rules = list(rules)
        updated_rules[0] = SelfReactionRule(
            material="acid_liquid",
            trigger_slot_index=7,
        )
        updated_compiled, updated_tags = read_metadata()
        assert resources.self_rule_signature != first_signature
        assert not np.array_equal(updated_compiled, sentinel_compiled)
        assert not np.array_equal(updated_tags, sentinel_tags)
        assert_grouped_metadata(updated_rules, updated_compiled, updated_tags)
    finally:
        engine.close()
