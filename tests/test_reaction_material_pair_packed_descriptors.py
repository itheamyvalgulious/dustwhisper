from __future__ import annotations

import numpy as np

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, MAX_MATERIALS, MAX_RULES
from oracle_game.world import WorldEngine


def _default_tables() -> tuple[WorldEngine, np.ndarray, np.ndarray, np.ndarray]:
    world = WorldEngine(width=8, height=8, simulation_backend="cpu")
    world.bridge.sync_rule_tables(world)
    return (
        world,
        world.bridge.shadow_typed_tables["material_material_rule_table"],
        world.bridge.shadow_typed_tables["material_gas_rule_table"],
        world.bridge.shadow_typed_tables["material_table"],
    )


def test_material_pair_packed_descriptors_are_dense_and_source_ordered() -> None:
    world, mm_rules, mg_rules, materials = _default_tables()
    try:
        pipeline = GPUReactionPipeline()
        packed = pipeline._compile_material_pair_packed_descriptors(
            mm_rules,
            mg_rules,
            materials,
            world.gas_concentration.shape[0],
        )
        assert packed is not None
        assert packed.shape == (MAX_MATERIALS + MAX_RULES, 4)

        headers = packed[:MAX_MATERIALS]
        descriptor_total = 0
        for material_id in range(1, int(materials.shape[0])):
            mm_span = int(headers[material_id, 0])
            mg_span = int(headers[material_id, 1])
            mm_offset, mm_count = mm_span & 0xFFFF, mm_span >> 16
            mg_offset, mg_count = mg_span & 0xFFFF, mg_span >> 16
            expected_mm = mm_rules[mm_rules["lhs_material_id"] == material_id]
            expected_mg = mg_rules[mg_rules["lhs_material_id"] == material_id]
            assert mm_count == int(expected_mm.shape[0])
            assert mg_count == int(expected_mg.shape[0])
            assert mm_count + mg_count <= 8
            descriptor_total += mm_count + mg_count

            mm_descriptors = packed[
                MAX_MATERIALS + mm_offset : MAX_MATERIALS + mm_offset + mm_count
            ]
            for descriptor, rule in zip(mm_descriptors, expected_mm, strict=True):
                operation = int(descriptor[0])
                assert (operation >> 9) & 0xFF == int(rule["rhs_material_id"])
                assert int(descriptor[1]) == int(np.float32(rule["min_temperature"]).view(np.uint32))
                assert int(descriptor[2]) == int(np.float32(rule["max_temperature"]).view(np.uint32))

            mg_descriptors = packed[
                MAX_MATERIALS + mg_offset : MAX_MATERIALS + mg_offset + mg_count
            ]
            gas_ids = [
                (int(headers[material_id, 2]) >> (slot * 8)) & 0xFF
                for slot in range(int(headers[material_id, 3]))
            ]
            for descriptor, rule in zip(mg_descriptors, expected_mg, strict=True):
                operation = int(descriptor[0])
                gas_slot = (operation >> 17) & 0x3
                assert gas_ids[gas_slot] == int(rule["rhs_gas_id"])
                assert int(descriptor[3]) == int(np.float32(rule["threshold"]).view(np.uint32))

        assert descriptor_total == int(mm_rules.shape[0] + mg_rules.shape[0]) == 74
    finally:
        world.close()

def test_material_pair_packed_descriptors_keep_strict_fallback() -> None:
    world, mm_rules, mg_rules, materials = _default_tables()
    try:
        pipeline = GPUReactionPipeline()
        gas_count = int(world.gas_concentration.shape[0])
        for field, value in (
            ("lhs_tag_mask", 1),
            ("rhs_tag_mask", 1),
            ("phase_mask", 1),
            ("consume_policy_id", 1),
            ("rate", np.float32(0.5)),
            ("rhs_material_id", 0),
        ):
            candidate = mm_rules.copy()
            candidate[0][field] = value
            assert pipeline._compile_material_pair_packed_descriptors(
                candidate, mg_rules, materials, gas_count
            ) is None, field

        candidate = mg_rules.copy()
        candidate[0]["rhs_gas_id"] = -1
        assert pipeline._compile_material_pair_packed_descriptors(
            mm_rules, candidate, materials, gas_count
        ) is None
        assert pipeline._compile_material_pair_packed_descriptors(
            mm_rules, mg_rules, materials, 9
        ) is None
    finally:
        world.close()


def test_material_pair_packed_descriptor_cache_tracks_table_generations() -> None:
    world, mm_rules, mg_rules, materials = _default_tables()
    try:
        pipeline = GPUReactionPipeline()
        gas_count = int(world.gas_concentration.shape[0])
        first = pipeline._compile_material_pair_packed_descriptors_cached(
            world, mm_rules, mg_rules, materials, gas_count
        )
        second = pipeline._compile_material_pair_packed_descriptors_cached(
            world, mm_rules, mg_rules, materials, gas_count
        )
        assert first is second
        assert first is not None

        world.bridge.table_generations["gases"] = (
            world.bridge.table_generations.get("gases", 0) + 1
        )
        third = pipeline._compile_material_pair_packed_descriptors_cached(
            world, mm_rules, mg_rules, materials, gas_count
        )
        assert third is not second
        assert np.array_equal(third, second)
    finally:
        world.close()
