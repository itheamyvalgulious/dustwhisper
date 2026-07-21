from __future__ import annotations

from types import SimpleNamespace

from oracle_game.sim.gpu_reactions import GPUReactionPipeline
from oracle_game.sim.gpu_reactions_cell_pass import _upload_material_pair_plan
from oracle_game.world import WorldEngine


class _WriteRecorder:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def write(self, payload: bytes) -> None:
        self.payloads.append(payload)


def _fake_resources() -> SimpleNamespace:
    return SimpleNamespace(
        material_pair_action_i=_WriteRecorder(),
        material_pair_action_f=_WriteRecorder(),
        material_pair_rule_i=_WriteRecorder(),
        material_pair_rule_f=_WriteRecorder(),
        material_pair_rule_tags=_WriteRecorder(),
        material_pair_lhs_candidate_masks=_WriteRecorder(),
        material_pair_plan_upload_key=None,
    )


def test_material_triplet_plan_cache_tracks_generations_flags_and_bridge_owner() -> None:
    world = WorldEngine(width=8, height=8, simulation_backend="cpu")
    other_world = WorldEngine(width=8, height=8, simulation_backend="cpu")
    pipeline = GPUReactionPipeline()
    try:
        plan = pipeline._compile_material_pair_plan_cached(world, include_material_light=True)
        assert plan is not None
        assert plan.material_light_rule_count > 0
        assert pipeline._compile_material_pair_plan_cached(
            world, include_material_light=True
        ) is plan
        assert all(
            not array.flags.writeable
            for array in (
                *plan.compiled_actions,
                plan.packed_rule_i,
                plan.packed_rule_f,
                plan.packed_rule_tags,
                plan.packed_lhs_candidate_masks,
            )
        )

        pair_plan = pipeline._compile_material_pair_plan_cached(
            world, include_material_light=False
        )
        assert pair_plan is not None
        assert pair_plan is not plan
        assert pair_plan.material_light_rule_count == 0

        for generation in ("reactions", "materials", "gases", "lights"):
            previous = plan
            world.bridge.table_generations[generation] = int(
                world.bridge.table_generations.get(generation, 0)
            ) + 1
            plan = pipeline._compile_material_pair_plan_cached(
                world, include_material_light=True
            )
            assert plan is not None
            assert plan is not previous

        pipeline._material_pair_packed_descriptors_enabled = False
        no_pair_descriptors = pipeline._compile_material_pair_plan_cached(
            world, include_material_light=True
        )
        assert no_pair_descriptors is not None
        assert no_pair_descriptors is not plan
        assert no_pair_descriptors.material_pair_packed_descriptors is None

        other_plan = pipeline._compile_material_pair_plan_cached(
            other_world, include_material_light=True
        )
        assert other_plan is not None
        assert other_plan.cache_key != no_pair_descriptors.cache_key
    finally:
        other_world.close()
        world.close()


def test_material_triplet_plan_upload_is_resident_and_new_resources_reupload() -> None:
    world = WorldEngine(width=8, height=8, simulation_backend="cpu")
    try:
        plan = GPUReactionPipeline()._compile_material_pair_plan_cached(
            world, include_material_light=True
        )
        assert plan is not None
        resources = _fake_resources()
        assert _upload_material_pair_plan(resources, plan) is True
        assert _upload_material_pair_plan(resources, plan) is False
        for name in (
            "material_pair_action_i",
            "material_pair_action_f",
            "material_pair_rule_i",
            "material_pair_rule_f",
            "material_pair_rule_tags",
            "material_pair_lhs_candidate_masks",
        ):
            recorder = getattr(resources, name)
            assert len(recorder.payloads) == 1
            assert len(recorder.payloads[0]) > 0

        rebuilt_resources = _fake_resources()
        assert _upload_material_pair_plan(rebuilt_resources, plan) is True
    finally:
        world.close()
