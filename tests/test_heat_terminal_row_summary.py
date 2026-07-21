from __future__ import annotations

import inspect

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_terminal_hierarchical_row_summary_is_default_on_and_isolated() -> None:
    pipeline = GPUHeatPipeline()
    assert pipeline._terminal_hierarchical_row_summary_enabled is True

    canonical = shader_source("heat/apply_terminal4x6.comp", _SHADER_SUBS)
    candidate = shader_source(
        "heat/apply_terminal4x6.comp",
        {**_SHADER_SUBS, "TERMINAL_HIERARCHICAL_ROW_SUMMARY": 1},
    )
    assert "#if 0\nshared uint gas_row_empty_nibbles" in canonical
    assert "#if 1\nshared uint gas_row_empty_nibbles" in candidate
    assert "for (int row_x = 0; row_x < 4; ++row_x)" in candidate
    assert "boil_counts_packed += 1u << uint((species_plus_one - 1) * 5);" in candidate
    assert "empty_count = bitCount(empty_mask[gas_lane]);" in candidate
    assert "for (int cell_lane = 0; cell_lane < 16; ++cell_lane)" in canonical

    ensure_source = inspect.getsource(GPUHeatPipeline._ensure_programs)
    run_source = inspect.getsource(GPUHeatPipeline._run_apply_terminal4x6)
    assert '"apply_terminal4x6_sparse_resident_packed_lazy_row_summary"' in ensure_source
    assert "and pipeline._terminal_hierarchical_row_summary_enabled" in run_source
    assert 'program_name = "apply_terminal4x6_sparse_resident_packed_lazy_row_summary"' in run_source
    assert 'elif packed_phase_boil_targets:' in run_source
