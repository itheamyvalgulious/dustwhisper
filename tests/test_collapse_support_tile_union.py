from __future__ import annotations

import numpy as np
import pytest

from oracle_game.sim import gpu_collapse_dirty as collapse_dirty
from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_FRONTIER_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
)
from oracle_game.world import WorldEngine


def _reachable(structural: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    reached = np.zeros_like(structural, dtype=np.bool_)
    queue = [tuple(int(value) for value in cell) for cell in np.argwhere(structural & seeds)]
    for y, x in queue:
        reached[y, x] = True
    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                ny < 0
                or nx < 0
                or ny >= structural.shape[0]
                or nx >= structural.shape[1]
                or reached[ny, nx]
                or not structural[ny, nx]
            ):
                continue
            reached[ny, nx] = True
            queue.append((ny, nx))
    return reached


def _single_tile_snake(tile_size: int = 32) -> np.ndarray:
    mask = np.zeros((tile_size, tile_size), dtype=np.bool_)
    for lane, y in enumerate(range(0, tile_size, 2)):
        mask[y, :] = True
        if y + 1 < tile_size:
            mask[y + 1, tile_size - 1 if lane % 2 == 0 else 0] = True
    return mask


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
    tile_mask = np.ones(tile_count, dtype=np.int32)
    bridge = engine.bridge
    bridge.buffers[FORMAL_CONNECTED_TILE_FRONTIER_BUFFER].write(tile_mask.tobytes())
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


def _solve_candidate(
    structural: np.ndarray,
    seeds: np.ndarray,
    *,
    capture_union: bool = False,
    atomic_union: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = structural.shape
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
        resources.structural_tex.write(structural.astype("f4", copy=False).tobytes())
        resources.support_ping.write(seeds.astype("f4", copy=False).tobytes())
        tile_mask_name = _seed_all_tiles(engine)
        pipeline._support_tile_union_enabled = True
        pipeline._support_tile_union_atomic_union_enabled = bool(atomic_union)
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            supported_texture = pipeline._solve_formal_connected_tile_support_textures(
                engine,
                resources,
                0,
                0,
                width,
                height,
                tile_mask_name,
                publish_masks=False,
            )
        finally:
            engine._world_simulation_frame_active = previous_frame_active
        supported = np.frombuffer(supported_texture.read(), dtype="f4").reshape((height, width)) > 0.5
        if not capture_union:
            return supported
        roots_buffer = resources.support_tile_union_roots
        parents_buffer = resources.support_tile_union_parent
        assert roots_buffer is not None and parents_buffer is not None
        roots = np.frombuffer(roots_buffer.read(), dtype=np.uint32).copy()
        parents = np.frombuffer(parents_buffer.read(), dtype=np.uint32).copy()
        return supported, roots, parents
    finally:
        engine.close()


def test_support_tile_union_is_default_off_and_lazy() -> None:
    engine = WorldEngine(width=33, height=35, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        resources = pipeline._ensure_resources(engine.bridge.ctx, engine.width, engine.height)
        assert pipeline._support_tile_union_enabled is False
        assert pipeline._support_tile_union_atomic_union_enabled is False
        assert pipeline._support_jfa_image_barrier_elision_enabled is False
        assert resources.support_tile_union_roots is None
        assert resources.support_tile_union_parent is None
        assert resources.support_tile_union_seeded is None
        assert resources.support_tile_union_edges is None
        assert resources.support_tile_union_edge_count is None
    finally:
        engine.close()


def test_support_tile_union_matches_orthogonal_connectivity_fixtures() -> None:
    snake = np.concatenate((_single_tile_snake(), _single_tile_snake()), axis=1)
    snake_seeds = np.zeros_like(snake)
    snake_seeds[0, 0] = True

    diagonal = np.zeros((32, 64), dtype=np.bool_)
    diagonal[8, 2:32] = True
    diagonal[9, 32:55] = True
    diagonal_seeds = np.zeros_like(diagonal)
    diagonal_seeds[8, 2] = True

    multi_anchor = np.zeros((32, 64), dtype=np.bool_)
    multi_anchor[3, 1:63] = True
    multi_anchor[12, 2:24] = True
    multi_anchor[20, 36:62] = True
    multi_anchor_seeds = np.zeros_like(multi_anchor)
    multi_anchor_seeds[3, 1] = True
    multi_anchor_seeds[3, 62] = True
    multi_anchor_seeds[20, 48] = True

    partial_bottom = np.zeros((45, 64), dtype=np.bool_)
    partial_bottom[2:44, 31] = True
    partial_bottom[43, 31:60] = True
    partial_bottom[5:40, 48] = True
    partial_bottom_seeds = np.zeros_like(partial_bottom)
    partial_bottom_seeds[43, 59] = True

    for structural, seeds in (
        (snake, snake_seeds),
        (diagonal, diagonal_seeds),
        (multi_anchor, multi_anchor_seeds),
        (partial_bottom, partial_bottom_seeds),
    ):
        candidate = _solve_candidate(structural, seeds)
        assert isinstance(candidate, np.ndarray)
        assert np.array_equal(candidate, _reachable(structural, seeds))


def test_support_tile_union_atomic_matches_original_union_output() -> None:
    rng = np.random.default_rng(20260719)
    for density in (0.08, 0.35, 0.8, 1.0):
        structural = rng.random((45, 64)) < density
        seeds = structural & (rng.random(structural.shape) < 0.04)
        original = _solve_candidate(structural, seeds)
        atomic = _solve_candidate(structural, seeds, atomic_union=True)
        assert isinstance(original, np.ndarray)
        assert isinstance(atomic, np.ndarray)
        assert np.array_equal(atomic, original)


def test_support_tile_union_converges_across_sixteen_tile_zigzag() -> None:
    tile_size = 32
    tile_count = 16
    width = tile_size * tile_count
    structural = np.zeros((tile_size, width), dtype=np.bool_)
    for lane, y in enumerate(range(0, tile_size, 2)):
        structural[y, :] = True
        if y + 1 < tile_size:
            structural[y + 1, width - 1 if lane % 2 == 0 else 0] = True
    seeds = np.zeros_like(structural)
    seeds[0, 0] = True

    captured = _solve_candidate(structural, seeds, capture_union=True)
    assert isinstance(captured, tuple)
    supported, local_roots, parents = captured

    def final_root(label: int) -> int:
        for _ in range(64):
            parent = int(parents[label - 1])
            if parent == label:
                return label
            label = parent
        raise AssertionError("support tile union parent chain did not converge")

    assert np.array_equal(supported, structural)
    assert {final_root(int(label)) for label in local_roots[structural.ravel()]} == {1}


def test_support_tile_union_matches_random_partial_tile_cases_without_stale_roots() -> None:
    width, height = 64, 45
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
        tile_mask_name = _seed_all_tiles(engine)
        pipeline._support_tile_union_enabled = True
        rng = np.random.default_rng(20260718)
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            for density in (0.08, 0.25, 0.55, 0.9):
                for _ in range(3):
                    structural = rng.random((height, width)) < density
                    seeds = structural & (rng.random((height, width)) < 0.025)
                    resources.structural_tex.write(structural.astype("f4", copy=False).tobytes())
                    resources.support_ping.write(seeds.astype("f4", copy=False).tobytes())
                    supported_texture = pipeline._solve_formal_connected_tile_support_textures(
                        engine,
                        resources,
                        0,
                        0,
                        width,
                        height,
                        tile_mask_name,
                        publish_masks=False,
                    )
                    supported = np.frombuffer(
                        supported_texture.read(),
                        dtype="f4",
                    ).reshape((height, width)) > 0.5
                    assert np.array_equal(supported, _reachable(structural, seeds))
        finally:
            engine._world_simulation_frame_active = previous_frame_active
    finally:
        engine.close()


def test_support_tile_union_preserves_incremental_live_mutation_validation() -> None:
    engine = WorldEngine(width=64, height=32, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        for x in range(4, 60):
            engine.set_cell(x, 8, "log_solid", mark_dirty=False)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.prewarm_formal_connected_collapse()
        pipeline._support_tile_union_enabled = True

        dirty_mask = collapse_dirty.ensure_collapse_structure_dirty_tile_mask(engine)
        dirty_queue = collapse_dirty.ensure_collapse_structure_dirty_tile_queue(engine)
        assert dirty_mask is not None and dirty_queue is not None
        dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
        tile_count = int(engine.active.tile_width) * int(engine.active.tile_height)
        tiles = np.asarray(
            [
                (tile_x, tile_y)
                for tile_y in range(int(engine.active.tile_height))
                for tile_x in range(int(engine.active.tile_width))
            ],
            dtype=np.int32,
        )
        dirty_mask.write(np.ones(tile_count, dtype=np.uint32).tobytes())
        dirty_count.write(np.asarray([tile_count], dtype=np.uint32).tobytes())
        dirty_list.write(tiles.tobytes())
        dirty_dispatch_args.write(np.asarray([tile_count, 1, 1], dtype=np.uint32).tobytes())
        engine._gpu_collapse_structure_dirty_tiles_pending = True

        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            engine.frame_id = 1
            engine.collapse_solver.advance_formal_gpu_dirty_epoch(engine)

            # The epoch snapshot still contains log_solid. Mutating one live
            # word after phase 0 must invalidate the entire eventual component.
            mutated_cell_index = 8 * engine.width + 20
            word_offset = mutated_cell_index * 5 * np.dtype(np.uint32).itemsize
            live_word = np.frombuffer(
                engine.bridge.buffers["cell_core"].read(size=4, offset=word_offset),
                dtype=np.uint32,
                count=1,
            ).copy()
            live_word[0] &= np.uint32(0xFFFF0000)
            engine.bridge.buffers["cell_core"].write(live_word.tobytes(), offset=word_offset)
            engine.bridge.mark_gpu_authoritative("cell_core")

            for frame_id in range(2, 5):
                engine.frame_id = frame_id
                engine.collapse_solver.advance_formal_gpu_dirty_epoch(engine)
        finally:
            engine._world_simulation_frame_active = previous_frame_active

        collapsed = np.frombuffer(
            engine.bridge.buffers["collapse_collapsed_cell_mask"].read(),
            dtype=np.int32,
        ).reshape((engine.height, engine.width))
        island_ids = np.frombuffer(
            engine.bridge.buffers["island_id"].read(),
            dtype=np.int32,
        ).reshape((engine.height, engine.width))
        assert not np.any(collapsed[8, 4:60])
        assert not np.any(island_ids[8, 4:60])
        assert pipeline._formal_dirty_epoch is None
    finally:
        engine.close()
