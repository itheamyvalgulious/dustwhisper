from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_tile_packed_pre_state_blocker_is_default_enabled() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._tile_packed_pre_state_blocker_enabled is True


def test_liquid_tile_packed_pre_state_blocker_is_warp_snapshot_pre_only() -> None:
    source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "TILE_WARP_FAST_PATH": 1,
            "TILE_WARP_PROVENANCE_ROW_STREAM": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_COMPACT_SNAPSHOT": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "TILE_PACKED_PRE_STATE_BLOCKER": 1,
            "LIQUID_PROVENANCE": 1,
        },
    )
    gate = "TILE_WARP_FAST_PATH && TILE_SNAPSHOT_PRE_STATE && TILE_PACKED_PRE_STATE_BLOCKER"
    assert source.count(gate) >= 3
    assert "shared uint s_pre_state_packed[TILE_SIZE][TILE_SIZE / 2];" in source
    assert "shared uint s_blocker_rows[TILE_SIZE];" in source
    assert "atomicAnd(s_blocker_rows[y], ~(1u << uint(x)))" in source
    assert "uint adjacent_pre_state = shuffleDownNV(pre_state, 1u, TILE_SIZE);" in source
    assert "uint blocker_ballot = ballotThreadNV(blocker != 0);" in source
    assert "uint pre_state = shared_pre_state(y, x);" in source
    assert "uint pre_state = shared_pre_state(y, local.x);" in source

    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    assert 'tile_solve_subs["TILE_PACKED_PRE_STATE_BLOCKER"]' in ensure_source
    assert "self._tile_packed_pre_state_blocker_enabled" in ensure_source
