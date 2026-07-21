from __future__ import annotations

import inspect

import numpy as np
import pytest

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source
from oracle_game.types import CellFlag, ReactionAction, ReactionType, SelfReactionRule
from oracle_game.world import WorldEngine


def test_timed_self_packed_metadata_is_enabled_and_compiled_into_cell_flags() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._timed_self_cell_flag_meta_enabled is True

    source = shader_source(
        "reactions/timed_apply.comp",
        {**_SHADER_SUBS, "PACK_CELL_META_IN_STATE": 1},
        includes=[
            "reactions/_common.comp",
            "reactions/_local_action_output_packed.comp",
        ],
    )
    reset = source.index("packed_cell_state &= 0x00FFFFFFu")
    latch = source.index("packed_cell_state |= uint(")
    store = source.index("imageStore(cell_state_out_img")
    assert reset < latch < store

    local_pass = inspect.getsource(GPUReactionPipeline._run_local_cell_action_pass)
    assert "pipeline._timed_self_cell_flag_meta_enabled" in local_pass
    assert 'program_key = f"{program_key}_cell_flag_meta"' in local_pass


@pytest.mark.parametrize("reverse_action_order", (False, True))
def test_formal_timed_self_metadata_skips_segment_accumulation(
    monkeypatch: pytest.MonkeyPatch,
    reverse_action_order: bool,
) -> None:
    engine = WorldEngine(width=32, height=16, gas_cell_size=4)
    try:
        pipeline = engine.reaction_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU reaction pipeline is not available")
        assert pipeline._timed_self_cell_flag_meta_enabled is True

        engine.replace_reaction_table(
            [
                ReactionAction(ReactionType.HARM, value=200.0),
                ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=3.0),
            ],
            {
                "material_material": [],
                "material_gas": [],
                "material_light": [],
                "gas_gas": [],
                "gas_light": [],
                "self_rules": [
                    SelfReactionRule(material="root_solid", trigger_slot_index=4)
                ],
            },
        )
        engine.patch_material(
            "root_solid",
            reaction_slots=(
                (2, 0, 0, 0, 1, 0, 0, 0)
                if reverse_action_order
                else (1, 0, 0, 0, 2, 0, 0, 0)
            ),
        )
        root_id = engine.rulebook.material_id("root_solid")
        root_phase = int(engine.rulebook.materials_by_id[root_id].default_phase)
        engine.material_id.fill(root_id)
        engine.phase.fill(root_phase)
        engine.cell_flags.fill(int(CellFlag.RECENTLY_CONVERTED))
        engine.integrity.fill(100.0)
        engine.timer_pack.fill(0)
        engine.timer_pack[::2, ::2, 0] = 1
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

        def forbidden_accumulation(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("timed/self metadata must not run the segment accumulate pass")

        monkeypatch.setattr(
            pipeline,
            "_accumulate_segment_cell_transient_state",
            forbidden_accumulation,
        )
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        engine.reaction_motion_handoff_active = False
        try:
            engine.reaction_solver.reset_runtime_state(engine)
            assert pipeline.begin_formal_reaction_segment(engine, "before_motion")
            engine.reaction_solver._advance_timed_slots(engine)
            assert pipeline.last_timed_self_cell_flag_meta_used
            engine.reaction_solver._run_self_rules(engine)
            assert pipeline.last_timed_self_cell_flag_meta_used

            resources = pipeline.resources
            assert resources is not None
            cell_state, *_rest = pipeline._current_cell_textures(resources)
            packed = np.frombuffer(cell_state.read(), dtype=np.uint32).reshape(
                engine.height, engine.width
            )
            flags = ((packed >> 24) & 0xFF).astype(np.uint8)
            killed = np.zeros(flags.shape, dtype=np.bool_)
            killed[::2, ::2] = True
            if reverse_action_order:
                # Timed latches first; the later self reset clears every flag.
                assert np.all(flags == 0)
            else:
                expected_latched = int(
                    CellFlag.RECENTLY_CONVERTED | CellFlag.REACTION_LATCHED
                )
                assert np.all(flags[killed] == 0)
                assert np.all(flags[~killed] == expected_latched)

            assert pipeline.flush_formal_reaction_segment(engine, "before_motion")
            assert engine.bridge.ctx is not None
            engine.bridge.ctx.finish()
            bridge_words = np.frombuffer(
                engine.bridge.buffers["cell_core"].read(), dtype=np.uint32
            ).reshape(engine.height, engine.width, 5)[..., 0]
            bridge_flags = ((bridge_words >> 24) & 0xFF).astype(np.uint8)
            assert np.array_equal(bridge_flags, flags)
        finally:
            pipeline.end_formal_reaction_segment(engine, "before_motion")
            engine._world_simulation_frame_active = previous_frame_active
    finally:
        engine.close()
