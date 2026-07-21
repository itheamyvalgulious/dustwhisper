from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_provenance_cleanup_terminal_fusion_is_default_off() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._provenance_cleanup_terminal_fusion_enabled is False
    assert pipeline._provenance_cleanup_terminal_fusion_frame_enabled is False
    assert pipeline.last_provenance_cleanup_terminal_fusion_used is False


def test_provenance_cleanup_terminal_reuses_final_state_and_bridge_aux() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_AUX_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "PROVENANCE_TERMINAL": 1,
            "FUSE_CLEANUP": 1,
        },
    )
    assert "cleaned_entity_at(cell, shared_cell_state[index])" in source
    assert "cleaned_displaced_at(cell, shared_cell_state[index])" in source
    assert "cleaned_island_at(gid, cell_state)" in source
    assert "bridge_island_id[cell_index] = cleaned_island" in source
    assert "bridge_entity_id[cell_index] = cleaned_entity" in source
    assert "bridge_displaced[cell_index] = cleaned_displaced" in source
    assert "#if !PROVENANCE_TERMINAL" in source


def test_terminal_blocker_mask_preserves_tile_blocking_predicate() -> None:
    load_source = shader_source(
        "liquid/load_bridge_blocker_displaced.comp",
        _SHADER_SUBS,
    )
    tile_source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_INPUTS": 1,
            "LIQUID_PROVENANCE": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_BLOCKER_MASK_INPUT": 1,
        },
    )
    assert "entity > 0 || displaced > 0" in load_source
    assert "layout(r8ui, binding=0) writeonly uniform uimage2D blocker_mask_img" in load_source
    assert "texelFetch(blocker_mask_tex, cell, 0).x != 0u" in tile_source
    assert "|| aux_blocked" in tile_source


def test_provenance_cleanup_terminal_has_strict_frame_gate() -> None:
    step_source = inspect.getsource(GPULiquidPipeline.step)
    load_source = inspect.getsource(GPULiquidPipeline._load_authoritative_bridge_inputs)
    flow_source = inspect.getsource(GPULiquidPipeline._run_liquid_intent_pass)
    assert "_provenance_cleanup_terminal_fusion_frame_enabled" in step_source
    assert "_provenance_terminal_frame_enabled" in step_source
    assert "_blocker_displaced_hydration_frame_enabled" in load_source
    assert "_provenance_cleanup_terminal_fusion_frame_enabled" in step_source
    assert "liquid_flow_intent_shared_halo_provenance_cleanup_bridge_aux" in flow_source
    assert "load_bridge_blocker_displaced" in load_source
