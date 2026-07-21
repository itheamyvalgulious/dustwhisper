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


def _canonical_labels(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros_like(mask, dtype=np.bool_)
    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        cells = [(y, x)]
        visited[y, x] = True
        cursor = 0
        while cursor < len(cells):
            cell_y, cell_x = cells[cursor]
            cursor += 1
            for next_y, next_x in (
                (cell_y - 1, cell_x),
                (cell_y + 1, cell_x),
                (cell_y, cell_x - 1),
                (cell_y, cell_x + 1),
            ):
                if (
                    next_y < 0
                    or next_x < 0
                    or next_y >= height
                    or next_x >= width
                    or visited[next_y, next_x]
                    or not mask[next_y, next_x]
                ):
                    continue
                visited[next_y, next_x] = True
                cells.append((next_y, next_x))
        root = min(cell_y * width + cell_x + 1 for cell_y, cell_x in cells)
        for cell_y, cell_x in cells:
            labels[cell_y, cell_x] = root
    return labels


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


def _solve_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    height, width = masks[0].shape
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
        pipeline._outcome_label_tile_union_enabled = True
        results: list[np.ndarray] = []
        for mask in masks:
            resources.phase_out_tex.write(mask.astype("f4", copy=False).tobytes())
            round_count, edge_capacity = pipeline._begin_formal_connected_component_label_union(
                engine,
                resources,
                resources.phase_out_tex,
                width,
                height,
                tile_mask_name,
            )
            first_stop = min(4, round_count)
            second_stop = min(14, round_count)
            cursor = pipeline._run_formal_connected_component_label_union_slice(
                engine, resources, edge_capacity, 0, first_stop
            )
            cursor = pipeline._run_formal_connected_component_label_union_slice(
                engine, resources, edge_capacity, cursor, second_stop
            )
            pipeline._run_formal_connected_component_label_union_slice(
                engine, resources, edge_capacity, cursor, round_count
            )
            label_texture, _ = pipeline._materialize_formal_connected_component_label_union(
                engine,
                resources,
                width,
                height,
                tile_mask_name,
            )
            labels = np.rint(
                np.frombuffer(label_texture.read(), dtype="f4").reshape((height, width))
            ).astype(np.int32)
            results.append(labels)
            dispatch = np.frombuffer(
                resources.component_dispatch_args.read(), dtype=np.uint32, count=3
            )
            assert int(dispatch[0]) == 0
        return results
    finally:
        engine.close()


def _single_tile_snake(tile_size: int = 32) -> np.ndarray:
    mask = np.zeros((tile_size, tile_size), dtype=np.bool_)
    for lane, y in enumerate(range(0, tile_size, 2)):
        mask[y, :] = True
        if y + 1 < tile_size:
            mask[y + 1, tile_size - 1 if lane % 2 == 0 else 0] = True
    return mask


def test_outcome_label_tile_union_is_default_on() -> None:
    engine = WorldEngine(width=33, height=35, simulation_backend="gpu")
    try:
        assert engine.collapse_solver.gpu_pipeline._outcome_label_tile_union_enabled is True
    finally:
        engine.close()


def test_outcome_label_tile_union_matches_cross_tile_components() -> None:
    # Exercise both the 1024-cell local snake and a global parent graph much
    # longer than the materializer's bounded root walk.
    snake = np.concatenate([_single_tile_snake() for _ in range(256)], axis=1)

    diagonal_gap = np.zeros((64, 96), dtype=np.bool_)
    diagonal_gap[8, 2:32] = True
    diagonal_gap[9, 32:70] = True
    diagonal_gap[40, 4:92] = True

    partial = np.zeros((45, 96), dtype=np.bool_)
    partial[2:44, 31] = True
    partial[43, 31:95] = True
    partial[5:40, 64] = True
    partial[20, 3:25] = True

    for mask in (snake, diagonal_gap, partial):
        candidate = _solve_masks([mask])[0]
        assert np.array_equal(candidate, _canonical_labels(mask))


def test_outcome_label_tile_union_clears_stale_roots_and_matches_random_masks() -> None:
    width, height = 96, 45
    rng = np.random.default_rng(20260718)
    masks = [np.ones((height, width), dtype=np.bool_)]
    masks.extend(rng.random((height, width)) < density for density in (0.08, 0.25, 0.55, 0.9))
    candidates = _solve_masks(masks)
    for candidate, mask in zip(candidates, masks, strict=True):
        assert np.array_equal(candidate, _canonical_labels(mask))


def test_outcome_label_tile_union_preserves_live_mutation_validation() -> None:
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
        pipeline._outcome_label_tile_union_enabled = True

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
            engine.bridge.buffers["collapse_collapsed_cell_mask"].read(), dtype=np.int32
        ).reshape((engine.height, engine.width))
        island_ids = np.frombuffer(
            engine.bridge.buffers["island_id"].read(), dtype=np.int32
        ).reshape((engine.height, engine.width))
        assert not np.any(collapsed[8, 4:60])
        assert not np.any(island_ids[8, 4:60])
        assert pipeline._formal_dirty_epoch is None
    finally:
        engine.close()
