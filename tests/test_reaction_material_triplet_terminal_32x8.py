from __future__ import annotations

import inspect

from oracle_game.sim import gpu_reactions_cell_pass
from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_material_triplet_terminal_32x8_is_default_enabled() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._material_triplet_terminal_32x8_enabled is True
    assert _SHADER_SUBS["LOCAL_SIZE_X"] == _SHADER_SUBS["LOCAL_SIZE"]
    assert _SHADER_SUBS["LOCAL_SIZE_Y"] == _SHADER_SUBS["LOCAL_SIZE"]


def test_material_triplet_terminal_32x8_preserves_256_thread_shared_transpose() -> None:
    substitutions = {
        **_SHADER_SUBS,
        "ENABLE_LIGHT_EMITTER_OUTPUT": 0,
        "MAX_RULES": 513,
        "RULE_I_CAPACITY": 513,
        "MAX_MATERIALS_TIMES_RULE_CANDIDATE_VECS": 1024,
        "MATERIAL_PAIR_TERMINAL_HANDOFF": 1,
        "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 1,
        "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 1,
        "LOCAL_SIZE": 16,
        "LOCAL_SIZE_X": 32,
        "LOCAL_SIZE_Y": 8,
    }
    source = shader_source(
        "reactions/material_pair_fused.comp",
        substitutions,
        includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
    )

    assert "layout(local_size_x=32, local_size_y=8, local_size_z=1) in;" in source
    assert "shared uint terminal_bridge_words[16 * 16 * 5];" in source
    assert "int row_word_count = 32 * 5;" in source
    assert "int total_invocation_count = 32 * 8;" in source
    assert "int(gl_WorkGroupID.x) * 32" in source
    assert "int(gl_WorkGroupID.y) * 8" in source


def test_material_triplet_terminal_32x8_keeps_full_fallback_and_xy_dispatch() -> None:
    source = inspect.getsource(gpu_reactions_cell_pass._run_material_pair_fused_pass)

    fallbacks = (
        "material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose",
        "material_pair_fused_terminal_local16_dirty_fast_shared_transpose",
        "material_pair_fused_terminal_local16_dirty_fast",
        "material_pair_fused_terminal_local16",
        "material_pair_fused_terminal",
        "material_pair_fused",
    )
    for name in fallbacks:
        assert f'"{name}"' in source
    positions = tuple(source.index(f'"{name}"') for name in fallbacks[:-1])
    assert positions == tuple(sorted(positions))
    assert source.rindex('"material_pair_fused"') > positions[-1]
    assert "terminal_local_size_x = (" in source
    assert "terminal_local_size_y = 8 if terminal_32x8" in source
    assert "world.width + terminal_local_size_x - 1" in source
    assert "world.height + terminal_local_size_y - 1" in source
