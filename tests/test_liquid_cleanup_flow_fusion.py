from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_cleanup_flow_fusion_is_default_off() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._cleanup_flow_fusion_enabled is False
    assert pipeline._cleanup_flow_fusion_frame_enabled is False
    assert pipeline.last_cleanup_flow_fusion_used is False


def test_cleanup_flow_fusion_replays_cleanup_in_shared_halo() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {**_SHADER_SUBS, "FUSE_CLEANUP": 1},
    )
    assert "cleaned_entity_at(cell, shared_cell_state[index])" in source
    assert "cleaned_displaced_at(cell, shared_cell_state[index])" in source
    assert "cleaned_island_at(gid, cell_state)" in source
    assert "bridge_island_id[cell_index] = cleaned_island" in source
    assert "bridge_entity_id[cell_index] = cleaned_entity" in source
    assert "bridge_displaced[cell_index]" in source


def test_cleanup_flow_fusion_keeps_strict_formal_fallback_gate() -> None:
    step_source = inspect.getsource(GPULiquidPipeline.step)
    flow_source = inspect.getsource(GPULiquidPipeline._run_liquid_intent_pass)
    assert "_cleanup_flow_fusion_frame_enabled" in step_source
    assert "phase_c_defer_cell_publish" in step_source
    assert "if not pipeline._cleanup_flow_fusion_frame_enabled" in step_source
    assert '"liquid_flow_intent_shared_halo_cleanup"' in flow_source
    assert "publish_bridge_outputs and pipeline._cleanup_flow_fusion_frame_enabled" in flow_source
