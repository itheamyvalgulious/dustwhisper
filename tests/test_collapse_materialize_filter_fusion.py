from __future__ import annotations

import inspect
from pathlib import Path

from oracle_game.sim import gpu_collapse_incremental as incremental
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline


ROOT = Path(__file__).parents[1]


def test_incremental_materialize_filter_fusion_is_default_with_fallback() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_materialize_filter_fusion_enabled is True

    validate_source = inspect.getsource(
        incremental._validate_and_collect_formal_dirty_epoch_labels
    )
    fused_return = "return label_texture, component_capacity"
    fallback_lookup = 'pipeline.programs["filter_incremental_component_labels"]'
    assert fused_return in validate_source
    assert fallback_lookup in validate_source
    assert validate_source.index(fused_return) < validate_source.index(fallback_lookup)


def test_incremental_materialize_filter_fusion_preserves_filtered_texture_output() -> None:
    materialize_source = inspect.getsource(
        incremental._materialize_formal_dirty_epoch_direct
    )
    assert 'program_name += "_filter"' in materialize_source
    assert "epoch.label_scratch.bind_to_image(1, read=False, write=True)" in materialize_source
    assert (
        "epoch.label_texture, epoch.label_scratch = epoch.label_scratch, epoch.label_texture"
        in materialize_source
    )

    pipeline_source = inspect.getsource(GPUCollapsePipeline._ensure_programs)
    assert '"materialize_incremental_components_bridge_filter"' in pipeline_source
    assert '"materialize_incremental_components_bridge_metadata_filter"' in pipeline_source
    assert '"WRITE_FILTERED_COMPONENT_LABELS": 1' in pipeline_source

    shader_source = (
        ROOT
        / "oracle_game/shaders/collapse/materialize_incremental_components_bridge.comp"
    ).read_text(encoding="ascii")
    assert "const bool WRITE_FILTERED_COMPONENT_LABELS" in shader_source
    assert "int filtered_label = slot_plus_one != 0u ? label : 0;" in shader_source
    assert "imageStore(\n            filtered_component_label_img," in shader_source
    assert "bridge_component_label[cell_index] = filtered_label;" in shader_source
    assert "bridge_collapsed_cell_mask[cell_index] = filtered_label > 0 ? 1 : 0;" in shader_source
