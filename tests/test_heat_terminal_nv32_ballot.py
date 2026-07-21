from __future__ import annotations

import inspect

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_heat_terminal_nv32_ballot_is_default_on_and_strictly_gated() -> None:
    pipeline = GPUHeatPipeline()
    assert pipeline._terminal_nv32_ballot_gas_reduction_enabled is True
    assert pipeline._terminal_nv32_ballot_supported is False
    assert pipeline.last_terminal_nv32_ballot_gas_reduction_used is False

    ensure_source = inspect.getsource(GPUHeatPipeline._ensure_programs)
    run_source = inspect.getsource(GPUHeatPipeline._run_apply_terminal4x6)
    assert '"GL_NV_gpu_shader5"' in ensure_source
    assert '"GL_NV_shader_thread_group"' in ensure_source
    assert "terminal_warp_size == 32" in ensure_source
    assert "if self._terminal_nv32_ballot_supported:" in ensure_source
    assert (
        '"apply_terminal4x6_sparse_resident_packed_lazy_nv32_ballot"'
        in ensure_source
    )
    assert "hierarchical_row_summary" in run_source
    assert "and pipeline._terminal_nv32_ballot_gas_reduction_enabled" in run_source
    assert "and pipeline._terminal_nv32_ballot_supported" in run_source
    assert "elif hierarchical_row_summary:" in run_source
    assert (
        'program_name = "apply_terminal4x6_sparse_resident_packed_lazy_row_summary"'
        in run_source
    )


def test_heat_terminal_nv32_ballot_reconstructs_each_gas_quad() -> None:
    candidate = shader_source(
        "heat/apply_terminal4x6.comp",
        {
            **_SHADER_SUBS,
            "DIRTY_WORKGROUP_AGGREGATE": 1,
            "TERMINAL_SPARSE_RESIDENT_SPECIALIZED": 1,
            "HEAT_LAZY_ACTION_INPUTS": 1,
            "HEAT_PACKED_PHASE_BOIL_TARGETS": 1,
            "TERMINAL_HIERARCHICAL_ROW_SUMMARY": 0,
            "TERMINAL_NV32_GAS_BALLOT": 1,
            "TERMINAL_NV32_EXTENSIONS": "\n".join(
                (
                    "#extension GL_NV_gpu_shader5 : require",
                    "#extension GL_NV_shader_thread_group : require",
                )
            ),
        },
    )

    assert "layout(local_size_x=8, local_size_y=8, local_size_z=1)" in candidate
    assert candidate.count("ballotThreadNV(") == 7
    assert "0x0F0F0F0Fu" in candidate
    assert "0xF0F0F0F0u" in candidate
    assert "#if !1\nshared uint after_cell_state" in candidate
    assert "#if 0\nshared uint gas_row_empty_nibbles" in candidate
    assert "#if 1\n    // Every invocation executes every ballot" in candidate
    assert "#elif 0\n        uint boil_counts_packed" in candidate
    assert "barrier();\n#endif\n\n#if 0" in candidate
    assert "if (lane == 0) {\n        flush_workgroup_dirty_tiles();" in candidate

    # The two alternating half-warp masks compact each 4x4 gas cell back to
    # the canonical row-major 16-bit empty mask without reordering cells.
    for gas_x, mask in ((0, 0x0F0F0F0F), (1, 0xF0F0F0F0)):
        for expected in range(1 << 16):
            ballot = 0
            for row in range(4):
                nibble = (expected >> (row * 4)) & 0xF
                ballot |= nibble << (row * 8 + gas_x * 4)
            selected = ballot & mask
            shift = gas_x * 4
            compacted = (
                ((selected >> shift) & 0x000F)
                | ((selected >> (shift + 4)) & 0x00F0)
                | ((selected >> (shift + 8)) & 0x0F00)
                | ((selected >> (shift + 12)) & 0xF000)
            )
            assert compacted == expected
