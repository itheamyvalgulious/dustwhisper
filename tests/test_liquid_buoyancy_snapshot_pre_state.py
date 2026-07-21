from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_buoyancy_snapshot_pre_state_candidate_is_default_on() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._buoyancy_snapshot_pre_state_enabled is True
    assert pipeline._buoyancy_snapshot_pre_state_frame_enabled is False
    assert pipeline.last_buoyancy_snapshot_pre_state_used is False


def test_candidate_packs_source_low16_and_destination_pre_state_high16() -> None:
    source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_INPUTS": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_COMPACT_SNAPSHOT": 1,
            "LIQUID_PROVENANCE": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
        },
    )
    assert "s_blocked[y][x] &= ~1" in source
    assert "uint pre_material = cell_state & 0xFFu" in source
    assert "uint pre_phase = (cell_state >> 16u) & 0xFFu" in source
    assert "(pre_state << 16u) | packed_source" in source
    assert "packed_source = has_source ? uint(source_id) : 0xFFFFu" in source
    assert "tile_snapshot_tile_tokens[tile_index] = uvec2(" in source
    blocker_uses = [line.strip() for line in source.splitlines() if "s_blocked[" in line]
    assert all(
        "shared int s_blocked" in line
        or "& 1" in line
        or "&= ~1" in line
        or "pre_state = uint(s_blocked" in line
        or "s_blocked[y][local.x] = blocker" in line
        or "s_blocked[y][x] = 0" in line
        for line in blocker_uses
    )


def test_runtime_blocker_mask_is_initialized_and_only_writes_boolean_values() -> None:
    resource_source = inspect.getsource(__import__(
        "oracle_game.sim.gpu_liquid_resources",
        fromlist=["_ensure_resources"],
    )._ensure_resources)
    blocker_source = shader_source("liquid/load_bridge_blocker_displaced.comp", _SHADER_SUBS)
    assert "blocker_mask.write(np.zeros" in resource_source
    assert "uint blocked = entity > 0 || displaced > 0 ? 1u : 0u" in blocker_source
    assert "imageStore(blocker_mask_img, gid, uvec4(blocked" in blocker_source


def test_candidate_seam_masks_only_snapshot_source_low16() -> None:
    source = shader_source(
        "liquid/seam_x.comp",
        {
            **_SHADER_SUBS,
            "SEAM_SNAPSHOT_INPUT": 1,
            "SEAM_COMPACT_SNAPSHOT": 1,
            "LIQUID_PROVENANCE": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
        },
    )
    assert "uint packed_source = source_word & 0xFFFFu" in source
    assert "packed_source == 0xFFFFu ? -1 : int(packed_source)" in source
    assert source.count("compact_snapshot_source_id(cell)") >= 2


def test_candidate_buoyancy_requires_matching_gpu_tile_token() -> None:
    source = shader_source(
        "liquid/buoyancy_fused.comp",
        {
            **_SHADER_SUBS,
            "LIQUID_PROVENANCE": 1,
            "FUSE_CLEANUP": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
        },
    )
    assert "all(equal(validated_token, token))" in source
    assert "tile_snapshot_state[4] == 0u" in source
    assert "tile_solve_snapshot[cell_index * 2 + 1] >> 16u" in source
    assert "return bridge_cell_core[cell_index * 5]" in source


def test_candidate_token_is_gpu_generated_and_specializations_are_isolated() -> None:
    token_source = shader_source("liquid/advance_tile_snapshot_token.comp", _SHADER_SUBS)
    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    run_source = inspect.getsource(GPULiquidPipeline._run_tile_solve)
    validation_source = shader_source("liquid/validate_tile_snapshot_coverage.comp", _SHADER_SUBS)
    assert "token.x += 1u" in token_source
    assert "tile_snapshot_state[4] = 1u" in token_source
    assert "tile_snapshot_tile_tokens[tile_index]" in validation_source
    assert "tile_snapshot_state[4] = coverage_invalid" in validation_source
    assert "if self._buoyancy_snapshot_pre_state_enabled" in ensure_source
    assert 'program_name += "_snapshot_pre"' in run_source
    assert 'pipeline.programs["advance_tile_snapshot_token"]' in run_source
