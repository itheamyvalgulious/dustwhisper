from __future__ import annotations

import inspect

import numpy as np

from oracle_game.sim.gpu_reactions import (
    GPUReactionPipeline,
    MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET,
    MATERIAL_LIGHT_PACKED_HEADER_OFFSET,
    MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET,
    MATERIAL_PAIR_PACKED_HEADER_OFFSET,
    MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
    MAX_MATERIALS,
    MAX_MATERIAL_LIGHT_PACKED_RULES,
    MAX_RULES,
)
from oracle_game.world import WorldEngine


def _tables() -> tuple[WorldEngine, np.ndarray, np.ndarray, np.ndarray]:
    world = WorldEngine(width=8, height=8, simulation_backend="cpu")
    return (
        world,
        world.bridge.shadow_typed_tables["material_light_rule_table"],
        world.bridge.shadow_typed_tables["material_table"],
        world.bridge.shadow_typed_tables["light_table"],
    )


def test_material_triplet_packed_descriptors_are_dense_and_source_ordered() -> None:
    world, rules, materials, lights = _tables()
    try:
        pipeline = GPUReactionPipeline()
        packed = pipeline._compile_material_light_packed_descriptors(rules, materials, lights)
        assert pipeline._material_triplet_ml_packed_descriptors_enabled is True
        assert packed is not None
        assert packed.shape == (MAX_MATERIALS + MAX_RULES, 4)
        headers = packed[:MAX_MATERIALS]
        assert int(headers[:, 0].sum()) == int(rules.shape[0]) == 191
        assert int(headers[:, 0].max()) <= MAX_MATERIAL_LIGHT_PACKED_RULES

        for material_id in range(1, int(materials.shape[0])):
            material_rules = rules[rules["lhs_material_id"] == material_id]
            count = int(headers[material_id, 0])
            assert count == int(material_rules.shape[0])
            if count == 0:
                continue
            start = MAX_MATERIALS + int(headers[material_id, 2])
            descriptors = packed[start : start + count]
            for descriptor, rule in zip(descriptors, material_rules, strict=True):
                operation = int(descriptor[0])
                direct_action = int(rule["result_action"]) >= 0
                expected_index = (
                    int(rule["result_action"])
                    if direct_action
                    else int(rule["trigger_slot_index"])
                )
                assert operation & 0xFF == expected_index
                assert bool(operation & 0x100) is direct_action
                light_id = int(rule["rhs_light_id"])
                expected_channel = int(lights[light_id]["dose_channel_id"])
                assert (operation >> 9) & 0x3 == expected_channel
    finally:
        world.close()


def test_material_triplet_packed_descriptor_cache_tracks_table_generations() -> None:
    world, rules, materials, lights = _tables()
    try:
        pipeline = GPUReactionPipeline()
        first = pipeline._compile_material_light_packed_descriptors_cached(
            world, rules, materials, lights
        )
        second = pipeline._compile_material_light_packed_descriptors_cached(
            world, rules, materials, lights
        )
        assert first is second

        world.bridge.table_generations["lights"] = (
            world.bridge.table_generations.get("lights", 0) + 1
        )
        third = pipeline._compile_material_light_packed_descriptors_cached(
            world, rules, materials, lights
        )
        assert third is not second
        assert third is not None
        assert np.array_equal(third, second)
    finally:
        world.close()


def test_material_triplet_packed_descriptors_keep_strict_fallback() -> None:
    world, rules, materials, lights = _tables()
    try:
        pipeline = GPUReactionPipeline()
        for field, value in (
            ("lhs_tag_mask", 1),
            ("phase_mask", 1),
            ("consume_policy_id", 1),
            ("rate", np.float32(0.5)),
            ("rhs_light_id", -1),
        ):
            candidate = rules.copy()
            candidate[0][field] = value
            assert pipeline._compile_material_light_packed_descriptors(
                candidate, materials, lights
            ) is None, field

        source = inspect.getsource(GPUReactionPipeline._compile_material_pair_plan)
        shader = inspect.getsource(GPUReactionPipeline._run_material_pair_fused_pass)
        assert "_compile_material_light_packed_descriptors_cached" in source
        assert "material_light_packed_descriptors is not None" in shader
        assert MATERIAL_LIGHT_PACKED_HEADER_OFFSET == MAX_RULES * 2 + 1
        assert MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET == (
            MATERIAL_LIGHT_PACKED_HEADER_OFFSET + MAX_MATERIALS
        )
        assert MATERIAL_PAIR_PACKED_HEADER_OFFSET == (
            MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
        )
        assert MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET == (
            MATERIAL_PAIR_PACKED_HEADER_OFFSET + MAX_MATERIALS
        )
        assert MATERIAL_PAIR_RULE_I_ENTRY_COUNT == (
            MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
        )
    finally:
        world.close()
