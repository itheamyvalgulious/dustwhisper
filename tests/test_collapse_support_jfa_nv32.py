from __future__ import annotations

import inspect

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_support_jfa_nv32_row_hydrate_is_specialized_and_default_on() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._support_jfa_nv32_row_hydrate_enabled is True
    assert pipeline._support_jfa_nv32_row_hydrate_supported is False

    source = shader_source(
        "collapse/propagate_formal_connected_tile_rows.comp",
        {
            **_SHADER_SUBS,
            "SUPPORT_JFA_U8": 1,
            "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1,
            "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 1,
            "SUPPORT_JFA_NV32_ROW_HYDRATE": 1,
            "SUPPORT_JFA_EXTENSIONS": (
                "#extension GL_NV_gpu_shader5 : require\n"
                "#extension GL_NV_shader_thread_group : require"
            ),
        },
    )
    assert "#define SUPPORT_JFA_NV32_ROW_HYDRATE 1" in source
    assert "int local_x = int(gl_ThreadInWarpNV);" in source
    assert "s_structural_rows[local_y] = structural_row;" in source
    assert "uint input_structural_row = s_structural_rows[input_y];" in source
    assert "&& in_grid(cell)" in source
    assert "uint supported_mask = ballotThreadNV(supported);" in source
    assert "s_supported_rows[input_y] = close_row(" in source

    from oracle_game.sim import gpu_collapse_frontier

    dispatch_source = inspect.getsource(
        gpu_collapse_frontier._run_formal_connected_tile_support_pass
    )
    assert "pipeline._support_jfa_nv32_row_hydrate_enabled" in dispatch_source
    assert "pipeline._support_jfa_nv32_row_hydrate_supported" in dispatch_source
    assert "source_mask_elision_nv32" in dispatch_source
