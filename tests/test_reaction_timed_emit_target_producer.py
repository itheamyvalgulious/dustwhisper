from __future__ import annotations

import inspect

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_timed_emit_target_producer_is_bounded_and_default_on() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._timed_emit_target_producer_enabled is True

    source = shader_source(
        "reactions/timed_apply.comp",
        {**_SHADER_SUBS, "TIMED_EMIT_TARGET_PRODUCER": 1},
        includes=[
            "reactions/_common.comp",
            "reactions/_timed_emit_target_output.comp",
            "reactions/_local_action_output_packed.comp",
        ],
    )
    assert "timed_emit_deferred_slot < 4" in source
    assert "deferred_count == timed_emit_deferred_slot + 1" in source
    assert "collect_timed_emit_material_target(gid, ai);" in source
    assert "texelFetch(velocity_tex, source, 0)" in source
    assert "material_value_at(target)" in source
    assert "if (action.y <= 0)" in source
    assert "timed_emit_target_count[1]" in source
    assert "atomicExchange(timed_emit_target_marks[target_index], generation)" in source
    assert "atomicAdd(timed_emit_target_count[2], 1u)" in source
    assert "binding=3" in source
    assert "binding=4" in source
    assert "binding=5" in source
    assert "#if !1\n            layout(std430, binding=3) buffer RuleI" in source


def test_timed_emit_target_producer_reuses_canonical_apply_and_args_builder() -> None:
    program_source = inspect.getsource(GPUReactionPipeline._ensure_programs)
    assert 'self.programs["timed_apply_packed_emit_targets"]' in program_source
    assert 'self.programs["timed_apply_packed_emit_targets_cell_flag_meta"]' in program_source

    pass_source = inspect.getsource(GPUReactionPipeline._run_local_cell_action_pass)
    assert "pipeline._timed_emit_target_producer_enabled" in pass_source
    assert "pipeline._clear_packed_timed_material_target_worklist(" in pass_source
    assert "resources.timed_candidate_count.bind_to_storage_buffer(binding=3)" in pass_source
    assert "pipeline._sync_storage_and_indirect_writes(world.bridge.ctx)" in pass_source
    assert "pipeline._run_produced_packed_timed_material_side_effect_pass(" in pass_source
    assert "pipeline._run_packed_timed_material_side_effect_pass(" in pass_source

    producer_apply = inspect.getsource(
        GPUReactionPipeline._run_produced_packed_timed_material_side_effect_pass
    )
    assert 'pipeline.programs["build_packed_material_target_dispatch"]' in producer_apply
    assert "resources.timed_candidate_count.bind_to_storage_buffer(binding=0)" in producer_apply
    assert "pipeline._run_packed_material_target_apply_pass(" in producer_apply
    assert 'profile_name="packed_timed_material_targets_apply"' in producer_apply
