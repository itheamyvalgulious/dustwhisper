from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_tile_snapshot_state_elision_is_default_on() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._tile_snapshot_state_elision_enabled is True
    assert pipeline._tile_snapshot_state_elision_frame_enabled is False
    assert pipeline.last_tile_snapshot_state_elision_used is False


def test_tile_snapshot_state_elision_writes_one_metadata_word() -> None:
    source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_INPUTS": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_COMPACT_SNAPSHOT": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "TILE_SNAPSHOT_STATE_ELISION": 1,
            "LIQUID_PROVENANCE": 1,
        },
    )
    assert "int snapshot_word = cell.y * cell_grid_size.x + cell.x;" in source
    assert "tile_solve_snapshot[snapshot_word] = (pre_state << 16u) | packed_source;" in source
    assert "#if !TILE_SNAPSHOT_STATE_ELISION" in source


def test_state_elided_seam_reads_active_state_texture_and_inactive_bridge() -> None:
    source = shader_source(
        "liquid/seam_x.comp",
        {
            **_SHADER_SUBS,
            "SEAM_SNAPSHOT_INPUT": 1,
            "SEAM_COMPACT_SNAPSHOT": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "TILE_SNAPSHOT_STATE_ELISION": 1,
            "LIQUID_PROVENANCE": 1,
        },
    )
    assert "uint source_word = tile_solve_snapshot[snapshot_word_index(cell)];" in source
    assert "? texelFetch(cell_state_in_tex, cell, 0).x" in source
    assert ": bridge_cell_core[bridge_word_index(cell)];" in source


def test_state_elided_buoyancy_decodes_dense_metadata_and_keeps_fallback() -> None:
    source = shader_source(
        "liquid/buoyancy_fused.comp",
        {
            **_SHADER_SUBS,
            "LIQUID_PROVENANCE": 1,
            "FUSE_CLEANUP": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "TILE_SNAPSHOT_STATE_ELISION": 1,
        },
    )
    assert "uint packed_pre = tile_solve_snapshot[cell_index] >> 16u;" in source
    assert "return bridge_cell_core[cell_index * 5];" in source
    assert "all(equal(validated_token, token))" in source


def test_state_elision_program_selection_is_frame_gated() -> None:
    step_source = inspect.getsource(GPULiquidPipeline.step)
    tile_source = inspect.getsource(GPULiquidPipeline._run_tile_solve)
    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    assert "pipeline._buoyancy_snapshot_pre_state_frame_enabled" in step_source
    assert "pipeline._tile_snapshot_state_elision_frame_enabled" in step_source
    assert 'program_name += "_state_elided"' in tile_source
    assert 'self.programs[f"{name}_state_elided"]' in ensure_source
