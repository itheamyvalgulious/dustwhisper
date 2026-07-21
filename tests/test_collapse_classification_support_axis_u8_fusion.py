from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from oracle_game.sim import gpu_collapse_frontier as frontier
from oracle_game.sim import gpu_collapse_incremental as incremental
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline


ROOT = Path(__file__).parents[1]


def test_classification_support_axis_u8_fusion_defaults_on_with_fallback() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._incremental_classification_support_axis_u8_fusion_enabled is True

    begin_source = inspect.getsource(frontier._begin_formal_connected_tile_support)
    incremental_source = inspect.getsource(incremental._begin_formal_dirty_epoch)
    assert "if not axis_masks_prebuilt:" in begin_source
    assert "pipeline._build_formal_connected_axis_masks(" in begin_source
    assert 'support_begin_kwargs["axis_masks_prebuilt"] = True' in incremental_source
    assert "pipeline._ensure_formal_connected_u8_support_textures(" in incremental_source


def test_classification_support_axis_u8_fusion_has_independent_programs() -> None:
    source = inspect.getsource(GPUCollapsePipeline._ensure_programs)
    assert '"classify_formal_connected_tiles_bridge_publish_incremental_axis_u8"' in source
    assert '"classify_formal_connected_tiles_bridge_publish_incremental_packed_axis_u8"' in source
    assert '"collapse/classify_formal_connected_tiles_support_axis_u8.comp"' in source


def test_fused_shader_clears_connected_masks_before_bounds_and_preserves_inactive_stale() -> None:
    source = (
        ROOT
        / "oracle_game/shaders/collapse/classify_formal_connected_tiles_support_axis_u8.comp"
    ).read_text(encoding="ascii")

    row_clear = "if (tile_connected && local_cell.x == 0)"
    column_clear = "if (tile_connected && local_cell.y == 0)"
    barrier = "memoryBarrierBuffer();\n    barrier();"
    bounds = "local_cell.x >= tile_size"
    u8_write_guard = "if (tile_connected) {\n        imageStore(support_seed_u8_img"
    assert row_clear in source
    assert column_clear in source
    assert barrier in source
    assert u8_write_guard in source
    assert source.index(row_clear) < source.index(barrier) < source.index(bounds)
    assert source.index(column_clear) < source.index(barrier)
    assert "int mask_base = tile_index * {{FORMAL_CONNECTED_TILE_LOCAL_SIZE}};" in source


def test_fused_shader_keeps_r32_outputs_and_builds_masks_from_same_classification() -> None:
    source = (
        ROOT
        / "oracle_game/shaders/collapse/classify_formal_connected_tiles_support_axis_u8.comp"
    ).read_text(encoding="ascii")

    structural = "bool structural = material_id > 0"
    r32_structural = "imageStore(structural_img, cell, vec4(structural ? 1.0 : 0.0"
    r32_seed = "imageStore(support_seed_img, cell, vec4(support_seed ? 1.0 : 0.0"
    r8_seed = "imageStore(support_seed_u8_img, cell, uvec4(support_seed ? 1u : 0u"
    row_or = "atomicOr(connected_tile_row_masks"
    column_or = "atomicOr(connected_tile_column_masks"
    for fragment in (structural, r32_structural, r32_seed, r8_seed, row_or, column_or):
        assert fragment in source
    assert source.index(structural) < source.index(r32_structural) < source.index(r8_seed)


def test_prebuilt_axis_masks_skip_the_original_builder_and_keep_the_schedule() -> None:
    class Pipeline:
        _support_tile_union_enabled = False

        @staticmethod
        def _ensure_formal_connected_u8_support_textures(*_args: object) -> tuple[str, str]:
            return "u8_ping", "u8_pong"

        @staticmethod
        def _formal_jfa_jumps(*_args: object) -> tuple[int, ...]:
            return (32, 16, 8, 4, 2, 1)

        @staticmethod
        def _formal_connected_tile_refine_pass_count(*_args: object) -> int:
            return 2

        @staticmethod
        def _build_formal_connected_axis_masks(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("prebuilt axis masks invoked the fallback builder")

    world = SimpleNamespace(bridge=SimpleNamespace(ctx=object()))
    resources = SimpleNamespace(support_ping="r32_ping", support_pong="r32_pong")

    current, scratch, schedule = frontier._begin_formal_connected_tile_support(
        Pipeline(),
        world,
        resources,
        70,
        45,
        "tiles",
        use_u8=True,
        axis_masks_prebuilt=True,
    )

    assert (current, scratch) == ("u8_ping", "u8_pong")
    assert schedule == (32, 16, 8, 4, 2, 1, 1, 1)
