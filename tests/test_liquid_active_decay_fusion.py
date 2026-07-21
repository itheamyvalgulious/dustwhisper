from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.gpu_optics import GPUOpticsPipeline
from oracle_game.sim.shader_loader import shader_source
from oracle_game.world_frame_pipeline import _step_once_impl


def test_liquid_active_decay_fusion_is_default_off() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._flow_active_decay_fusion_enabled is False
    assert pipeline._flow_active_decay_fusion_frame_enabled is False
    assert pipeline.last_flow_active_decay_fusion_used is False


def test_flow_active_decay_is_one_tile_leader_after_terminal_publish() -> None:
    source = shader_source(
        "liquid/liquid_flow_intent_shared_halo.comp",
        {**_SHADER_SUBS, "FUSE_ACTIVE_DECAY": 1},
    )
    assert "layout(std430, binding=11) buffer ActiveTileTTLBuffer" in source
    assert "any(notEqual(cell, group_origin))" in source
    assert "group_origin.x % tile_size != 0" in source
    assert "active_tile_ttl[tile_index] = max(0, active_tile_ttl[tile_index] - 1)" in source
    assert source.index("decay_active_tile_once(gid, group_origin)") > source.index("if (publish_bridge_aux_outputs)")


def test_optics_uses_current_frame_liquid_snapshot_only_after_fusion() -> None:
    source = shader_source(
        "optics/_active_common.comp",
        {"ACTIVE_TILE_SNAPSHOT": 1, "LOCAL_SIZE": 8, "LIGHT_PARAM_COUNT": 16, "MAX_EMITTERS": 256, "MAX_LIGHTS": 8},
    )
    assert "layout(binding=5) uniform sampler2D active_tile_snapshot_tex" in source
    assert "texelFetch(active_tile_snapshot_tex, tile, 0).x > 0.5" in source
    optics_source = inspect.getsource(GPUOpticsPipeline._load_authoritative_active_masks)
    assert "last_flow_active_decay_fusion_used" in optics_source
    assert "_flow_active_decay_fusion_frame_id" in optics_source
    assert "active_tile_snapshot.use(location=5)" in optics_source


def test_world_active_decay_keeps_standalone_fallback() -> None:
    source = inspect.getsource(_step_once_impl)
    assert "if not active_decay_fused and not engine.bridge.decay_active_scheduler(engine)" in source
    assert "elif engine.simulation_backend == \"gpu\":" in source
    assert "engine.active.decay()" in source
