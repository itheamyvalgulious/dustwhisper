from __future__ import annotations

import inspect
from pathlib import Path

from oracle_game.sim import gpu_collapse_incremental as incremental
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline


ROOT = Path(__file__).parents[1]


def test_label_union_materialize_validation_fusion_defaults_on_with_fallback() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_label_union_materialize_validation_fusion_enabled is True

    phase2_source = inspect.getsource(incremental._advance_phase2)
    commit_source = inspect.getsource(incremental._commit_formal_dirty_epoch)
    assert "label_union_materialize_validation_deferred = True" in phase2_source
    assert "_materialize_formal_connected_component_label_union" in phase2_source
    assert "not fuse_label_union_materialize_validation" in commit_source
    assert "materialize_label_union" in commit_source


def test_union_materialize_validation_shader_preserves_root_walk_and_validation_order() -> None:
    source = (
        ROOT
        / "oracle_game/shaders/collapse/validate_incremental_component_labels_union_materialize.comp"
    ).read_text(encoding="ascii")

    assert "for (int step = 0; step < 64; ++step)" in source
    assert "uint parent = union_parents[label - 1u];" in source
    root_load = "uint local_root = local_roots[cell.y * cell_grid_size.x + cell.x];"
    label_store = "imageStore(component_label_img, cell, vec4(float(root), 0.0, 0.0, 0.0));"
    invalid_write = "atomicOr(invalid_components[uint(label - 1)], 1u);"
    assert root_load in source
    assert label_store in source
    assert invalid_write in source
    assert source.index(root_load) < source.index(label_store) < source.index(invalid_write)


def test_union_materialize_validation_has_dedicated_packed_and_unpacked_programs() -> None:
    source = inspect.getsource(GPUCollapsePipeline._ensure_programs)
    assert '"validate_incremental_component_labels_union_materialize"' in source
    assert '"validate_incremental_component_labels_union_materialize_packed"' in source
    assert '"collapse/validate_incremental_component_labels_union_materialize.comp"' in source
