from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_hotpass_candidate_defaults() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._flow_intent_active_mask_cache_enabled is False
    assert pipeline._flow_intent_provenance_shared_meta_cache_enabled is True
    assert pipeline._flow_intent_provenance_lazy_aux_enabled is True
    assert pipeline._tile_solve_liquid_kind_cache_enabled is False


def test_liquid_provenance_active_mask_cache_is_specialized() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_AUX_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "PROVENANCE_TERMINAL": 1,
            "PROVENANCE_ACTIVE_MASK_CACHE": 1,
        },
    )
    assert "shared uint shared_all_tiles_active;" in source
    assert "active_tile_count[0] >= uint(tile_grid_size.x * tile_grid_size.y)" in source
    assert "if (shared_all_tiles_active != 0u)" in source


def test_liquid_tile_solve_kind_cache_tracks_state_writes() -> None:
    source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_INPUTS": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_COMPACT_SNAPSHOT": 1,
            "TILE_WARP_FAST_PATH": 1,
            "TILE_WARP_PROVENANCE_ROW_STREAM": 1,
            "LIQUID_PROVENANCE": 1,
            "TILE_LIQUID_KIND_CACHE": 1,
        },
    )
    assert "shared uint s_liquid_kind[TILE_SIZE][TILE_SIZE];" in source
    assert "s_liquid_kind[y][x] = uint(liquid_kind_for(float(cell_state & 0xFFFFu)));" in source
    assert "s_liquid_kind[y][x] = 0u;" in source
    assert "shared_liquid_kind_at(y, x) == 1" in source


def test_liquid_terminal_provenance_shared_meta_cache_reuses_halo_resolution() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_AUX_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "PROVENANCE_TERMINAL": 1,
            "PROVENANCE_ACTIVE_MASK_CACHE": 1,
            "PROVENANCE_SHARED_META_CACHE": 1,
        },
    )
    assert "shared uint shared_terminal_provenance[SHARED_COUNT];" in source
    assert "bool tile_is_active = tile_active(cell);" in source
    assert "shared_terminal_provenance[index] = provenance;" in source
    assert "terminal_velocity_at(gid, terminal_provenance, primary_role)" in source
    assert "store_terminal_core_resolved(" in source
    assert "terminal_provenance" in source
    dispatch_source = inspect.getsource(GPULiquidPipeline._run_liquid_intent_pass)
    assert '"liquid_flow_intent_shared_halo_provenance_shared_meta"' in dispatch_source
    assert "_flow_intent_provenance_shared_meta_cache_enabled" in dispatch_source


def test_liquid_terminal_forward_halo_omits_only_unused_top_row() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_AUX_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "PROVENANCE_TERMINAL": 1,
            "PROVENANCE_ACTIVE_MASK_CACHE": 1,
            "PROVENANCE_SHARED_META_CACHE": 1,
        },
    )
    assert "const int SHARED_ROWS = 8 + 1;" in source
    assert "group_origin + local_cell - ivec2(1, 0)" in source
    assert "cell - group_origin + ivec2(1, 0)" in source


def test_liquid_terminal_lazy_aux_reads_only_reachable_empty_candidates() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_AUX_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "PROVENANCE_TERMINAL": 1,
            "PROVENANCE_ACTIVE_MASK_CACHE": 1,
            "PROVENANCE_SHARED_META_CACHE": 1,
            "PROVENANCE_LAZY_AUX": 1,
        },
    )
    assert "#define PROVENANCE_LAZY_AUX 1" in source
    assert "#if !PROVENANCE_LAZY_AUX\nshared float shared_entity" in source
    assert "float(bridge_entity_id[cell_index]) < 0.5" in source
    assert "float(bridge_displaced[cell_index]) < 0.5" in source
    dispatch_source = inspect.getsource(GPULiquidPipeline._run_liquid_intent_pass)
    assert '"liquid_flow_intent_shared_halo_provenance_shared_meta_lazy_aux"' in dispatch_source


def test_liquid_forward_halo_covers_every_flow_probe() -> None:
    pass_size = 8
    loaded_offsets = {
        (x - 1, y)
        for y in range(pass_size + 1)
        for x in range(pass_size + 2)
    }
    for y in range(pass_size):
        for x in range(pass_size):
            required_offsets = {
                (x, y),
                (x, y + 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y + 1),
                (x + 1, y + 1),
            }
            assert required_offsets <= loaded_offsets
    assert all(y >= 0 for _, y in loaded_offsets)
