from __future__ import annotations

import inspect

from oracle_game.sim import gpu_reactions_pairings, gpu_reactions_segments
from oracle_game.sim.gpu_reactions import GPUReactionPipeline
from oracle_game.sim.shader_loader import shader_source


def test_terminal_gas_publish_fusion_is_default_on() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._terminal_gas_publish_fusion_enabled is True


def test_terminal_gas_publish_fusion_requires_matching_formal_pending_delta() -> None:
    pipeline = GPUReactionPipeline()
    key = (object(), 7, "before_motion", object(), (67, 45, 17, 12, 6))
    pipeline._terminal_gas_publish_fusion_enabled = True
    pipeline._formal_state_cache_key = key
    pipeline._formal_segment_batch_key = key
    pipeline._formal_pending_gas_delta_key = key
    assert pipeline._formal_terminal_gas_publish_fusion_pending()

    pipeline._formal_pending_gas_delta_key = (*key[:-1], object())
    assert not pipeline._formal_terminal_gas_publish_fusion_pending()
    pipeline._formal_pending_gas_delta_key = key
    pipeline._formal_segment_batch_key = None
    assert not pipeline._formal_terminal_gas_publish_fusion_pending()
    after_optics_key = (object(), 7, "after_optics", object(), (67, 45, 17, 12, 6))
    pipeline._formal_state_cache_key = after_optics_key
    pipeline._formal_segment_batch_key = after_optics_key
    pipeline._formal_pending_gas_delta_key = after_optics_key
    assert not pipeline._formal_terminal_gas_publish_fusion_pending()


def test_terminal_gas_publish_fusion_keeps_canonical_fallback_source() -> None:
    segment_source = inspect.getsource(gpu_reactions_segments._flush_formal_segment_gas_delta)
    gas_light_source = inspect.getsource(gpu_reactions_pairings._run_formal_guarded_gas_light)
    canonical_source = shader_source("reactions/apply_cell_gas_delta.comp")
    fused_source = shader_source("reactions/apply_cell_gas_delta_publish_bridge.comp")

    assert 'else "apply_cell_gas_delta"' in segment_source
    assert "pipeline._formal_terminal_gas_publish_fusion_pending()" in gas_light_source
    assert '"gas_light_publish_gas_state_deferred"' in gas_light_source
    assert "BridgeGasBuffer" not in canonical_source
    assert "BridgeGasBuffer" in fused_source
    assert "float result = max(0.0, gas_value + delta);" in fused_source
    assert "imageStore(gas_out_img, gid, vec4(result" in fused_source
    assert "bridge_gas[linear_index] = max(result, 0.0);" in fused_source
    assert "if (gid.z == 0)" in fused_source
    assert "texelFetch(ambient_tex, gid.xy, 0).x" in fused_source
