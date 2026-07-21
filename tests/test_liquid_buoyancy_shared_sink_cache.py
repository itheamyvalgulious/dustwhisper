from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_buoyancy_shared_sink_cache_is_default_on() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._buoyancy_shared_sink_cache_enabled is True
    assert pipeline._buoyancy_shared_sink_cache_frame_enabled is False
    assert pipeline.last_buoyancy_shared_sink_cache_used is False


def test_buoyancy_shared_sink_cache_covers_exact_vertical_dependency() -> None:
    source = shader_source(
        "liquid/buoyancy_fused.comp",
        {
            **_SHADER_SUBS,
            "LIQUID_PROVENANCE": 1,
            "FUSE_CLEANUP": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "BUOYANCY_SHARED_SINK_CACHE": 1,
        },
    )
    assert "const int SINK_INPUT_ROWS = 8 + 4;" in source
    assert "const int SINK_QUERY_ROWS = 8 + 2;" in source
    assert "group_origin + ivec2(x, y - 2)" in source
    assert "shared_sink_input[input_y + 1][x]" in source
    assert "shared_sink_input[input_y - 1][x]" in source
    assert "source_code == 1u" in source
    assert "source_code == 2u" in source


def test_buoyancy_shared_sink_cache_keeps_partial_canonical_fallback() -> None:
    source = shader_source(
        "liquid/buoyancy_fused.comp",
        {**_SHADER_SUBS, "BUOYANCY_SHARED_SINK_CACHE": 1},
    )
    assert "active_tile_count[0]" in source
    assert ">= uint(tile_grid_size.x * tile_grid_size.y)" in source
    assert "if (shared_sink_cache_enabled != 0u)" in source
    assert "return sink_state(cell, changed, source, source_uses_fallback);" in source


def test_buoyancy_shared_sink_cache_program_is_strictly_gated() -> None:
    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    step_source = inspect.getsource(GPULiquidPipeline.step)
    assert "buoyancy_fused_provenance_cleanup_snapshot_pre_shared_sink" in ensure_source
    assert "pipeline._buoyancy_snapshot_pre_state_frame_enabled" in step_source
    assert "pipeline._provenance_terminal_frame_enabled" in step_source
    assert '"_shared_sink"' in step_source
