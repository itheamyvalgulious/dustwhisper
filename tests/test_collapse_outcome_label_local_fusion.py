from __future__ import annotations

import inspect

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_outcome_label_local_fusion_is_independent_and_default_on() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_outcome_label_local_fusion_enabled is True

    source = shader_source(
        "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BEHAVIOR_INPUTS": 1,
            "PUBLISH_IMMUNE_DIRECT": 1,
            "PUBLISH_DELAYED_DIRECT": 1,
            "SUPPORT_TEXTURE_U8": 1,
            "INITIALIZE_LABEL_TILE_UNION": 1,
        },
    )
    assert "shared uint s_parent[" in source
    assert "bool component = cell_valid && collapse_now;" in source
    assert "label_union_local_roots[cell_index] = root_label;" in source
    assert "imageStore(collapse_now_img, cell" in source


def test_outcome_label_local_fusion_keeps_a_strict_runtime_gate() -> None:
    from oracle_game.sim import gpu_collapse_incremental, gpu_collapse_stages

    finish = inspect.getsource(gpu_collapse_incremental._finish_support_and_resolve_outcome)
    resolve = inspect.getsource(gpu_collapse_stages.resolve_supported_outcome_textures)
    assert "not fuse_support_publish" in finish
    assert "publish_immune_direct" in finish
    assert "publish_delayed_direct" in finish
    assert "not epoch.packed_cell_snapshot" in finish
    assert "fused outcome/label-local initialization requires" in resolve
    assert "binding=10" in resolve
    assert "binding=11" in resolve
