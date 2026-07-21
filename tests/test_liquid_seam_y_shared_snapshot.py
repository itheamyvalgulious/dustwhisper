from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_seam_y_shared_snapshot_is_default_enabled() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._seam_y_shared_snapshot_enabled is True
    assert pipeline._buoyancy_pass_fusion_enabled is True
    assert pipeline._seam_workgroups_per_boundary("y") == 1


def test_liquid_seam_y_shared_snapshot_loads_both_rows_before_writes() -> None:
    source = shader_source("liquid/seam_y_shared_snapshot.comp", _SHADER_SUBS)
    barrier_offset = source.index("memoryBarrierShared();")
    assert "layout(local_size_x=32, local_size_y=1" in source
    for shared_value in (
        "s_cell_state[2][TILE_SIZE]",
        "s_timer[2][TILE_SIZE]",
        "s_temp[2][TILE_SIZE]",
        "s_integrity[2][TILE_SIZE]",
        "s_velocity[2][TILE_SIZE]",
        "s_entity[2][TILE_SIZE]",
        "s_displaced[2][TILE_SIZE]",
    ):
        assert shared_value in source
    assert barrier_offset < source.index("store_cleared(top_cell);")
    assert "if (!valid)" in source[barrier_offset:]
    assert "texelFetch(cell_state_in_tex" not in source[barrier_offset:]


def test_liquid_step_keeps_shared_snapshot_control_fallback_and_role_handoff() -> None:
    source = inspect.getsource(GPULiquidPipeline.step)
    buoyancy_source = inspect.getsource(GPULiquidPipeline._run_buoyancy_pass)
    assert "seam_y_shared_snapshot" in source
    assert "if not seam_y_shared_snapshot" in source
    assert "liquid_copy_seam_x" in source
    assert "sink_fallback_resources=(" in source
    assert "fallback_resources" in buoyancy_source
