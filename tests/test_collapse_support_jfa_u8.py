from __future__ import annotations

import inspect

import numpy as np
import pytest

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    GPUCollapsePipeline,
    _SHADER_SUBS,
)
from oracle_game.sim import gpu_collapse_frontier
from oracle_game.sim.shader_loader import shader_source
from oracle_game.world import WorldEngine


def _seed_all_tiles(engine: WorldEngine) -> str:
    pipeline = engine.collapse_solver.gpu_pipeline
    pipeline._ensure_formal_connected_frontier_buffers(engine)
    tiles = np.asarray(
        [
            (tile_x, tile_y)
            for tile_y in range(int(engine.active.tile_height))
            for tile_x in range(int(engine.active.tile_width))
        ],
        dtype=np.int32,
    )
    tile_count = int(tiles.shape[0])
    bridge = engine.bridge
    bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].write(
        np.ones(tile_count, dtype=np.int32).tobytes()
    )
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].write(tiles.tobytes())
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].write(
        np.asarray([tile_count], dtype=np.uint32).tobytes()
    )
    bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER].write(
        np.asarray([tile_count, 1, 1], dtype=np.uint32).tobytes()
    )
    bridge.mark_gpu_authoritative(
        FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    )
    return FORMAL_CONNECTED_TILE_FRONTIER_BUFFER


def _run_support(
    pipeline,
    engine: WorldEngine,
    resources,
    tile_mask_name: str,
    width: int,
    height: int,
    *,
    use_u8: bool,
):
    current, scratch, schedule = pipeline._begin_formal_connected_tile_support(
        engine,
        resources,
        width,
        height,
        tile_mask_name,
        use_u8=use_u8,
    )
    current, _ = pipeline._run_formal_connected_tile_support_slice(
        engine,
        resources,
        current,
        scratch,
        width,
        height,
        tile_mask_name,
        schedule,
        0,
        len(schedule),
    )
    return current


def test_incremental_support_jfa_u8_is_default_on_and_lazy() -> None:
    pipeline = GPUCollapsePipeline()

    assert pipeline._incremental_support_jfa_u8_enabled is True


def test_u8_propagated_source_mask_elision_is_default_on_and_isolated() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._support_jfa_u8_propagated_source_mask_elision_enabled is True

    canonical = shader_source(
        "collapse/propagate_formal_connected_tile_rows.comp",
        {**_SHADER_SUBS, "SUPPORT_JFA_U8": 1},
    )
    candidate = shader_source(
        "collapse/propagate_formal_connected_tile_rows.comp",
        {
            **_SHADER_SUBS,
            "SUPPORT_JFA_U8": 1,
            "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 1,
        },
    )
    assert "#if 0\n    // U8 support seeds" in canonical
    assert "#if 1\n    // U8 support seeds" in candidate
    assert "return cell_connected(cell) && texelFetch(support_in_tex, cell, 0).x != 0u;" in candidate
    assert "return structural_connected(cell) && texelFetch(support_in_tex, cell, 0).x > 0.5;" in candidate

    dispatch_source = inspect.getsource(
        gpu_collapse_frontier._run_formal_connected_tile_support_pass
    )
    assert "if pipeline._support_jfa_u8_propagated_source_mask_elision_enabled:" in dispatch_source
    assert '"propagate_formal_connected_tiles_u8_row_major_source_mask_elision"' in dispatch_source
    assert 'else "propagate_formal_connected_tiles_u8_source_mask_elision"' in dispatch_source


@pytest.mark.parametrize("width,height", [(64, 64), (67, 45)])
def test_incremental_support_jfa_u8_matches_f32_raw_outputs(
    width: int,
    height: int,
) -> None:
    engine = WorldEngine(width=width, height=height, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        ctx = engine.bridge.ctx
        assert ctx is not None
        pipeline._ensure_programs(ctx)
        resources = pipeline._ensure_resources(ctx, width, height)
        assert resources.support_u8_ping is None
        assert resources.support_u8_pong is None
        tile_mask_name = _seed_all_tiles(engine)

        rng = np.random.default_rng(20260720 + width + height)
        structural = rng.random((height, width)) < 0.74
        # Force long paths across tile and partial-tile boundaries.
        structural[height // 2, :] = True
        structural[:, width // 2] = True
        seeds = structural & (rng.random((height, width)) < 0.035)
        seeds[height // 2, 0] = True
        if (width, height) == (67, 45):
            structural[height - 1, :64] = True
            seeds[height - 1, 0] = True
        resources.structural_tex.write(structural.astype("f4", copy=False).tobytes())
        resources.support_ping.write(seeds.astype("f4", copy=False).tobytes())

        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            f32_supported = _run_support(
                pipeline,
                engine,
                resources,
                tile_mask_name,
                width,
                height,
                use_u8=False,
            )
            f32_support_raw = f32_supported.read(alignment=1)
            f32_support_mask = np.frombuffer(f32_support_raw, dtype=np.float32).reshape(
                (height, width)
            ) > 0.5
            f32_support_bytes = f32_support_mask.astype(np.uint8).tobytes()

            resources.support_ping.write(seeds.astype("f4", copy=False).tobytes())
            u8_supported = _run_support(
                pipeline,
                engine,
                resources,
                tile_mask_name,
                width,
                height,
                use_u8=True,
            )
            assert resources.support_u8_ping is not None
            assert resources.support_u8_pong is not None
            u8_support_raw = u8_supported.read(alignment=1)
            assert u8_support_raw == f32_support_bytes
            if (width, height) == (67, 45):
                u8_support = np.frombuffer(u8_support_raw, dtype=np.uint8).reshape(
                    (height, width)
                )
                assert np.all(u8_support[height - 1, 24:32] == 1)
                assert np.all(u8_support[height - 1, 56:64] == 1)

            # Restore an f32 control texture in case support_ping was the final
            # control ping-pong target and had to be reset for u8 seed conversion.
            resources.support_ping.write(f32_support_raw)
            f32_supported = resources.support_ping

            pipeline._publish_bridge_supported_unsupported_masks_connected_tiles(
                engine,
                resources,
                f32_supported,
                0,
                0,
                width,
                height,
                tile_mask_name,
            )
            f32_published = (
                engine.bridge.buffers["collapse_supported_mask"].read(),
                engine.bridge.buffers["collapse_unsupported_mask"].read(),
            )
            pipeline._publish_bridge_supported_unsupported_masks_connected_tiles(
                engine,
                resources,
                u8_supported,
                0,
                0,
                width,
                height,
                tile_mask_name,
            )
            assert engine.bridge.buffers["collapse_supported_mask"].read() == f32_published[0]
            assert engine.bridge.buffers["collapse_unsupported_mask"].read() == f32_published[1]

            behaviors = rng.integers(1, 4, size=(height, width), dtype=np.int32)
            resources.material_out_tex.write(behaviors.astype("f4").tobytes())
            engine.collapse_delay_pending[:, :] = rng.random((height, width)) < 0.2

            pipeline.resolve_supported_outcome_textures(
                engine,
                resources,
                f32_supported,
                0,
                0,
                width,
                height,
                eligibility_texture=resources.structural_tex,
                tile_mask_name=tile_mask_name,
                publish_runtime_masks=True,
                publish_outputs=False,
            )
            f32_outcomes = (
                resources.temp_out_tex.read(alignment=1),
                resources.integrity_out_tex.read(alignment=1),
                resources.phase_out_tex.read(alignment=1),
                engine.bridge.buffers["collapse_supported_mask"].read(),
                engine.bridge.buffers["collapse_unsupported_mask"].read(),
            )
            pipeline.resolve_supported_outcome_textures(
                engine,
                resources,
                u8_supported,
                0,
                0,
                width,
                height,
                eligibility_texture=resources.structural_tex,
                tile_mask_name=tile_mask_name,
                publish_runtime_masks=True,
                publish_outputs=False,
            )
            u8_outcomes = (
                resources.temp_out_tex.read(alignment=1),
                resources.integrity_out_tex.read(alignment=1),
                resources.phase_out_tex.read(alignment=1),
                engine.bridge.buffers["collapse_supported_mask"].read(),
                engine.bridge.buffers["collapse_unsupported_mask"].read(),
            )
            assert u8_outcomes == f32_outcomes
        finally:
            engine._world_simulation_frame_active = previous_frame_active
    finally:
        engine.close()
