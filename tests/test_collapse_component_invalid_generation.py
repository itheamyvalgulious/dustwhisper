from __future__ import annotations

import inspect

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_component_invalid_generation_is_specialized_and_default_on() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_component_invalid_generation_enabled is True
    assert pipeline._component_invalid_generation == 0

    validate = shader_source(
        "collapse/validate_incremental_component_labels_union_materialize.comp",
        {**_SHADER_SUBS, "INVALID_COMPONENT_GENERATION_VALIDITY": 1},
    )
    collect = shader_source(
        "collapse/collect_component_labels_connected_tiles.comp",
        {**_SHADER_SUBS, "INVALID_COMPONENT_GENERATION_VALIDITY": 1},
    )
    assert "atomicExchange(invalid_components[uint(label - 1)], invalid_generation)" in validate
    assert "bool invalid_component = invalid_value == invalid_generation" in collect

    from oracle_game.sim import gpu_collapse_incremental, gpu_collapse_labeling

    validation_source = inspect.getsource(
        gpu_collapse_incremental._validate_and_collect_formal_dirty_epoch_labels
    )
    collect_source = inspect.getsource(gpu_collapse_labeling._collect_component_labels_gpu)
    assert "previous_generation >= 0xFFFFFFFF" in validation_source
    assert "and materialize_label_union" in validation_source
    assert "and not packed_cell_snapshot" in validation_source
    assert '"collect_component_labels_connected_tiles_generation"' in collect_source


def test_component_flag_generation_is_specialized_and_default_on() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_component_flag_generation_enabled is True
    assert pipeline._active_component_flag_generation == 0

    materialize = shader_source(
        "collapse/materialize_incremental_components_bridge.comp",
        {
            **_SHADER_SUBS,
            "SUMMARIZE_COMPONENT_METADATA": 1,
            "WRITE_FILTERED_COMPONENT_LABELS": 1,
            "COMPONENT_FLAG_GENERATION_VALIDITY": 1,
        },
    )
    assert "invalid_components[uint(label - 1)] == component_flag_generation" in materialize

    from oracle_game.sim import gpu_collapse_incremental, gpu_collapse_labeling

    validation_source = inspect.getsource(
        gpu_collapse_incremental._validate_and_collect_formal_dirty_epoch_labels
    )
    materialize_source = inspect.getsource(
        gpu_collapse_incremental._materialize_formal_dirty_epoch_direct
    )
    collect_source = inspect.getsource(gpu_collapse_labeling._collect_component_labels_gpu)
    assert "pipeline._incremental_component_flag_generation_enabled" in validation_source
    assert "component_flag_generation is None" in collect_source
    assert 'program_name += "_generation"' in materialize_source
