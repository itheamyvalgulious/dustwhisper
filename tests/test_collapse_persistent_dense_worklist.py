from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from oracle_game.sim import gpu_collapse_dirty as collapse_dirty
from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    GPUCollapsePipeline,
)
from oracle_game.world import WorldEngine


_DENSE_WORKLIST_BUFFERS = (
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
)


def _seed_all_dirty_tiles(engine: WorldEngine) -> None:
    dirty_mask = collapse_dirty.ensure_collapse_structure_dirty_tile_mask(engine)
    dirty_queue = collapse_dirty.ensure_collapse_structure_dirty_tile_queue(engine)
    assert dirty_mask is not None and dirty_queue is not None
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    tiles = np.asarray(
        [
            (tile_x, tile_y)
            for tile_y in range(int(engine.active.tile_height))
            for tile_x in range(int(engine.active.tile_width))
        ],
        dtype=np.int32,
    )
    tile_count = int(tiles.shape[0])
    dirty_mask.write(np.ones(tile_count, dtype=np.uint32).tobytes())
    dirty_count.write(np.asarray([tile_count], dtype=np.uint32).tobytes())
    dirty_list.write(tiles.tobytes())
    dirty_dispatch_args.write(np.asarray([tile_count, 1, 1], dtype=np.uint32).tobytes())
    engine._gpu_collapse_structure_dirty_tiles_pending = True


def _run_two_incremental_epochs(
    *, cache_enabled: bool
) -> tuple[dict[str, bytes], tuple[int, int], tuple[int, ...]]:
    engine = WorldEngine(width=64, height=45, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        for x in range(4, 60):
            engine.set_cell(x, 8, "log_solid", mark_dirty=False)
        for y in range(8, 42):
            engine.set_cell(4, y, "log_solid", mark_dirty=False)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.prewarm_formal_connected_collapse()
        pipeline._persistent_dense_tile_worklist_enabled = cache_enabled
        support_jumps: list[int] = []
        original_support_pass = pipeline._run_formal_connected_tile_support_pass

        def record_support_pass(*args: object, **kwargs: object) -> tuple[object, object]:
            support_jumps.append(int(args[-1]))
            return original_support_pass(*args, **kwargs)

        pipeline._run_formal_connected_tile_support_pass = record_support_pass

        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            frame_id = 1
            for _epoch in range(2):
                _seed_all_dirty_tiles(engine)
                for _phase in range(4):
                    engine.frame_id = frame_id
                    engine.collapse_solver.advance_formal_gpu_dirty_epoch(engine)
                    frame_id += 1
        finally:
            engine._world_simulation_frame_active = previous_frame_active
        ctx = engine.bridge.ctx
        assert ctx is not None
        ctx.finish()
        outputs = {
            name: engine.bridge.buffers[name].read()
            for name in (
                "cell_core",
                "island_id",
                "entity_id",
                "collapse_component_label",
                "collapse_collapsed_cell_mask",
                "collapse_structural_mask",
                "collapse_support_seed_mask",
                "collapse_supported_mask",
                "collapse_unsupported_mask",
                "collapse_delay_pending",
                "collapse_delayed_pending_mask",
                "collapse_immune_unsupported_mask",
            )
        }
        return outputs, (
            pipeline.persistent_dense_tile_worklist_rebuilds,
            pipeline.persistent_dense_tile_worklist_hits,
        ), tuple(support_jumps)
    finally:
        engine.close()


def _run_incremental_support_publish_case(
    *,
    fused: bool,
    direct_immune_publish: bool = False,
    direct_delayed_publish: bool = False,
    jfa_four_frame_balance: bool = False,
) -> tuple[dict[str, bytes], int]:
    engine = WorldEngine(width=64, height=45, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        for x in range(4, 60):
            engine.set_cell(x, 8, "log_solid", mark_dirty=False)
            engine.set_cell(x, 28, "log_solid", mark_dirty=False)
        for y in range(8, engine.height):
            engine.set_cell(4, y, "log_solid", mark_dirty=False)
        engine.collapse_delay_pending[28, 20:28] = True
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.prewarm_formal_connected_collapse()
        assert pipeline._incremental_support_outcome_publish_fusion_enabled is False
        assert pipeline._incremental_direct_immune_publish_enabled is True
        assert pipeline._incremental_direct_delayed_publish_enabled is True
        pipeline._incremental_support_outcome_publish_fusion_enabled = fused
        pipeline._incremental_direct_immune_publish_enabled = direct_immune_publish
        pipeline._incremental_direct_delayed_publish_enabled = direct_delayed_publish
        pipeline._incremental_jfa_four_frame_balance_enabled = jfa_four_frame_balance

        publish_calls = 0
        original_publish = pipeline._publish_bridge_supported_unsupported_masks_connected_tiles

        def record_publish(*args: object, **kwargs: object) -> None:
            nonlocal publish_calls
            publish_calls += 1
            original_publish(*args, **kwargs)

        pipeline._publish_bridge_supported_unsupported_masks_connected_tiles = record_publish
        _seed_all_dirty_tiles(engine)
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        snapshots: dict[str, bytes] = {}
        try:
            for frame_id in range(1, 5):
                engine.frame_id = frame_id
                engine.collapse_solver.advance_formal_gpu_dirty_epoch(engine)
                if frame_id == 2:
                    ctx = engine.bridge.ctx
                    assert ctx is not None
                    ctx.finish()
                    for name in (
                        "collapse_supported_mask",
                        "collapse_unsupported_mask",
                        "collapse_delay_pending",
                        "collapse_delayed_pending_mask",
                        "collapse_immune_unsupported_mask",
                    ):
                        snapshots[f"phase1.{name}"] = engine.bridge.buffers[name].read()
        finally:
            engine._world_simulation_frame_active = previous_frame_active

        ctx = engine.bridge.ctx
        assert ctx is not None
        ctx.finish()
        for name in (
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_supported_mask",
            "collapse_unsupported_mask",
            "collapse_delay_pending",
            "collapse_delayed_pending_mask",
            "collapse_immune_unsupported_mask",
            "collapse_collapsed_cell_mask",
            "collapse_component_label",
        ):
            resource = engine.bridge.textures[name] if name == "material" else engine.bridge.buffers[name]
            snapshots[f"final.{name}"] = resource.read()
        return snapshots, publish_calls
    finally:
        engine.close()


def test_persistent_dense_tile_worklist_is_default_on() -> None:
    pipeline = GPUCollapsePipeline()
    assert pipeline._persistent_dense_tile_worklist_enabled is True
    assert pipeline._persistent_dense_tile_worklist_signature is None


def test_persistent_dense_tile_worklist_matches_two_uncached_epochs() -> None:
    control, control_cache_counts, control_jumps = _run_two_incremental_epochs(cache_enabled=False)
    candidate, candidate_cache_counts, candidate_jumps = _run_two_incremental_epochs(cache_enabled=True)

    assert candidate == control
    assert control_cache_counts == (0, 0)
    assert candidate_cache_counts == (1, 1)
    expected_jumps = (*GPUCollapsePipeline._formal_jfa_jumps(64, 45), 1, 1)
    assert control_jumps == expected_jumps * 2
    assert candidate_jumps == expected_jumps * 2


def test_incremental_support_publish_fusion_matches_separate_publish_at_phase_boundary() -> None:
    control, control_publish_calls = _run_incremental_support_publish_case(fused=False)
    candidate, candidate_publish_calls = _run_incremental_support_publish_case(fused=True)

    assert candidate == control
    assert control_publish_calls == 1
    assert candidate_publish_calls == 0


def test_incremental_direct_immune_publish_matches_texture_publish_at_phase_boundary() -> None:
    control, control_publish_calls = _run_incremental_support_publish_case(
        fused=False,
        direct_immune_publish=False,
    )
    candidate, candidate_publish_calls = _run_incremental_support_publish_case(
        fused=False,
        direct_immune_publish=True,
    )

    assert candidate == control
    assert candidate_publish_calls == control_publish_calls == 1


def test_incremental_phase_peak_candidate_matches_control_after_full_epoch() -> None:
    control, _ = _run_incremental_support_publish_case(
        fused=False,
        direct_immune_publish=False,
        jfa_four_frame_balance=False,
    )
    candidate, _ = _run_incremental_support_publish_case(
        fused=False,
        direct_immune_publish=True,
        direct_delayed_publish=True,
        jfa_four_frame_balance=True,
    )

    control_final = {name: data for name, data in control.items() if name.startswith("final.")}
    candidate_final = {name: data for name, data in candidate.items() if name.startswith("final.")}
    assert candidate_final == control_final


def test_persistent_dense_tile_worklist_rebuilds_after_frontier_pollution_and_release() -> None:
    engine = WorldEngine(width=64, height=45, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        engine.set_cell(8, 8, "log_solid", mark_dirty=False)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        resources, x0, y0, width, height = (
            pipeline._prepare_formal_connected_tile_resources_without_input_upload(
                engine,
                (0, 0, engine.width, engine.height),
            )
        )
        pipeline._persistent_dense_tile_worklist_enabled = True
        pipeline._seed_formal_texture_region_tile_worklist(engine, width, height)
        expected = {name: engine.bridge.buffers[name].read() for name in _DENSE_WORKLIST_BUFFERS}
        assert pipeline.persistent_dense_tile_worklist_rebuilds == 1

        pipeline._seed_formal_connected_tile_frontier(
            engine,
            resources,
            (0, 0, 32, 32),
            x0,
            y0,
            width,
            height,
        )
        assert pipeline._persistent_dense_tile_worklist_signature is None
        polluted_count = int(
            np.frombuffer(
                engine.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].read(size=4),
                dtype=np.uint32,
                count=1,
            )[0]
        )
        assert polluted_count == 1

        pipeline._seed_formal_texture_region_tile_worklist(engine, width, height)
        rebuilt = {name: engine.bridge.buffers[name].read() for name in _DENSE_WORKLIST_BUFFERS}
        assert rebuilt == expected
        assert pipeline.persistent_dense_tile_worklist_rebuilds == 2

        pipeline.release()
        assert pipeline._persistent_dense_tile_worklist_signature is None
        pipeline._seed_formal_texture_region_tile_worklist(engine, width, height)
        assert pipeline.persistent_dense_tile_worklist_rebuilds == 3
        assert {name: engine.bridge.buffers[name].read() for name in _DENSE_WORKLIST_BUFFERS} == expected
    finally:
        engine.close()


def test_persistent_dense_tile_worklist_signature_covers_context_world_region_and_tiles() -> None:
    engine = WorldEngine(width=64, height=45, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        pipeline._ensure_formal_connected_frontier_buffers(engine)
        signature = pipeline._persistent_dense_tile_worklist_signature_for(engine, 64, 45)
        assert signature is not None

        changed_world = SimpleNamespace(
            width=65,
            height=engine.height,
            active=engine.active,
            bridge=engine.bridge,
        )
        changed_tiles = SimpleNamespace(
            width=engine.width,
            height=engine.height,
            active=SimpleNamespace(
                tile_size=engine.active.tile_size,
                tile_width=engine.active.tile_width + 1,
                tile_height=engine.active.tile_height,
            ),
            bridge=engine.bridge,
        )
        changed_context = SimpleNamespace(
            width=engine.width,
            height=engine.height,
            active=engine.active,
            bridge=SimpleNamespace(ctx=object(), buffers=engine.bridge.buffers),
        )
        assert pipeline._persistent_dense_tile_worklist_signature_for(changed_world, 64, 45) != signature
        assert pipeline._persistent_dense_tile_worklist_signature_for(engine, 63, 45) != signature
        assert pipeline._persistent_dense_tile_worklist_signature_for(changed_tiles, 64, 45) != signature
        assert pipeline._persistent_dense_tile_worklist_signature_for(changed_context, 64, 45) != signature
    finally:
        engine.close()
