from __future__ import annotations

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_jfa_row_major_output_is_default_enabled_and_keeps_canonical_shader() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._support_jfa_row_major_output_enabled is True

    canonical = shader_source(
        "collapse/propagate_formal_connected_tile_rows.comp",
        _SHADER_SUBS,
    )
    row_major = shader_source(
        "collapse/propagate_formal_connected_tile_rows.comp",
        {**_SHADER_SUBS, "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1},
    )

    assert "#if 0" in canonical
    assert "for (int local_x = 0; local_x < 32; ++local_x)" in canonical
    assert "#if 1" in row_major
    assert "int linear = int(gl_LocalInvocationIndex)" in row_major
    assert "if (!in_grid(cell))" in row_major
