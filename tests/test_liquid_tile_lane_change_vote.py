from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_tile_lane_change_vote_is_default_enabled() -> None:
    assert GPULiquidPipeline()._tile_warp_lane_change_vote_enabled is True


def test_liquid_tile_lane_change_vote_is_warp_gated() -> None:
    source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "TILE_WARP_EXTENSIONS": "#extension GL_NV_shader_thread_group : require",
            "TILE_WARP_FAST_PATH": 1,
            "TILE_WARP_LANE_CHANGE_VOTE": 1,
        },
    )
    programs_source = inspect.getsource(GPULiquidPipeline._ensure_programs)

    assert "#define TILE_WARP_LANE_CHANGE_VOTE 1" in source
    assert "bool lane_changed = false;" in source
    assert "bool tile_changed = anyThreadNV(lane_changed);" in source
    assert "TILE_WARP_FAST_PATH && TILE_WARP_LANE_CHANGE_VOTE" in source
    assert 'tile_solve_subs["TILE_WARP_LANE_CHANGE_VOTE"]' in programs_source
    assert "self._tile_warp_lane_change_vote_enabled" in programs_source
