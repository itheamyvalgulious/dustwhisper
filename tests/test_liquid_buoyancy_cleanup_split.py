from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_buoyancy_cleanup_split_fusion_is_default_on() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._buoyancy_cleanup_split_fusion_enabled is True
    assert pipeline._buoyancy_cleanup_split_fusion_frame_enabled is False
    assert pipeline.last_buoyancy_cleanup_split_fusion_used is False


def test_buoyancy_cleanup_uses_final_cell_and_immutable_pre_core() -> None:
    source = shader_source(
        "liquid/buoyancy_fused.comp",
        {**_SHADER_SUBS, "LIQUID_PROVENANCE": 1, "FUSE_CLEANUP": 1},
    )
    assert "uint pre_state = bridge_cell_core[cell_index * 5]" in source
    assert "cleanup_bridge_aux(cell, value.cell_state)" in source
    assert "bridge_entity_id[cell_index] = 0" in source
    assert "bridge_displaced[cell_index] = 0" in source
    assert "bridge_island_id[cell_index] = 0" in source


def test_placeholder_cleanup_restores_only_affected_active_tiles() -> None:
    source = shader_source(
        "liquid/cleanup_runtime.comp",
        {
            **_SHADER_SUBS,
            "PASS_LOCAL_SIZE": 16,
            "PASS_LOCAL_SIZE_MINUS_1": 15,
            "DIRECT_BRIDGE_INPUTS": 1,
            "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
            "RESTORE_BRIDGE_AUX_OUTPUTS": 1,
        },
    )
    assert "restrict_to_active_tiles && texelFetch(active_tile_tex, tile, 0).x <= 0.5" in source
    assert "bridge_island_id[cell_index] = int(round(island_value))" in source
    assert "bridge_entity_id[cell_index] = int(round(entity_value))" in source
    assert "bridge_displaced[cell_index] = int(round(displaced_value))" in source


def test_split_schedule_keeps_cleanup_out_of_terminal_flow() -> None:
    step_source = inspect.getsource(GPULiquidPipeline.step)
    cleanup_source = inspect.getsource(GPULiquidPipeline._run_cleanup_runtime)
    assert '"buoyancy_fused_provenance_cleanup"' in step_source
    assert '"liquid_cleanup_placeholder_affected"' in step_source
    assert "affected_tile_dispatch=True" in step_source
    assert "restore_bridge_aux_values=True" in step_source
    assert '"cleanup_runtime_bridge_aux_restore16"' in cleanup_source
