from __future__ import annotations

import inspect

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_reaction_self_cached_cell_state_candidate_is_bounded_and_default_off() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._self_apply_cached_cell_state_enabled is False
    assert pipeline.last_self_apply_cached_cell_state_used is False
    assert _SHADER_SUBS["SELF_CACHE_CELL_STATE"] == 0

    ensure_programs = inspect.getsource(GPUReactionPipeline._ensure_programs)
    for program_key in (
        "self_apply_packed_cached_cell_state",
        "self_apply_packed_cell_flag_meta_cached_cell_state",
        "self_apply_packed_direct_spans_cached_cell_state",
        "self_apply_packed_direct_spans_cell_flag_meta_cached_cell_state",
        "self_apply_packed_fused_gas_cached_cell_state",
        "self_apply_packed_fused_gas_cell_flag_meta_cached_cell_state",
    ):
        assert program_key in ensure_programs

    local_pass = inspect.getsource(GPUReactionPipeline._run_local_cell_action_pass)
    assert 'program_name == "self_apply"' in local_pass
    assert "and packed_local_deferred_outputs" in local_pass
    assert "and not candidate_dispatch" in local_pass
    assert 'program_key = f"{program_key}_cached_cell_state"' in local_pass


def test_reaction_self_cached_cell_state_preserves_packed_flags_and_meta_order() -> None:
    self_apply = shader_source("reactions/self_apply.comp")
    packed_output = shader_source("reactions/_local_action_output_packed.comp")

    cached_fetch = "uint source_cell_state = texelFetch(cell_state_tex, gid, 0).x;"
    assert self_apply.count(cached_fetch) == 1
    assert "float material_value = float(source_cell_state & 0xFFFFu);" in self_apply
    assert "float phase_value = float((source_cell_state >> 16u) & 0xFFu);" in self_apply
    assert self_apply.count("source_cell_state,") == 3

    preserve = packed_output.index("source_cell_state & 0xFF000000u")
    reset = packed_output.index("packed_cell_state &= 0x00FFFFFFu")
    latch = packed_output.index("packed_cell_state |= uint(")
    store = packed_output.index("imageStore(cell_state_out_img")
    assert preserve < reset < latch < store

    material = 0xBEEF
    phase = 0x7A
    latch_flag = int(_SHADER_SUBS["REACTION_LATCHED_FLAG_SHIFTED_24"])
    for flags in range(256):
        source_word = material | (phase << 16) | (flags << 24)
        canonical = material | (phase << 16) | (source_word & 0xFF000000)
        for reset_requested in (False, True):
            for latch_requested in (False, True):
                expected = canonical
                cached = canonical
                if reset_requested:
                    expected &= 0x00FFFFFF
                    cached &= 0x00FFFFFF
                if latch_requested:
                    expected |= latch_flag
                    cached |= latch_flag
                assert cached == expected
