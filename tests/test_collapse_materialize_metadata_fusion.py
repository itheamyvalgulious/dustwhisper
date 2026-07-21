from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE
from oracle_game.sim import gpu_collapse_labeling as labeling
from oracle_game.sim import gpu_collapse_dirty as collapse_dirty
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline
from oracle_game.world import WorldEngine


@contextmanager
def _profile_pass(*_args: object, **_kwargs: object):
    yield


class _PreparePipeline:
    def __init__(self) -> None:
        self.summarize_calls = 0

    _profile_pass = staticmethod(_profile_pass)

    def _collect_component_labels_gpu(self, *_args: object, **_kwargs: object) -> int:
        return 128

    def _summarize_formal_component_metadata(self, *_args: object, **_kwargs: object) -> None:
        self.summarize_calls += 1


@pytest.mark.parametrize(("defer", "expected_calls"), ((False, 1), (True, 0)))
def test_prepare_can_defer_metadata_summary_to_materialization(
    defer: bool,
    expected_calls: int,
) -> None:
    pipeline = _PreparePipeline()
    world = SimpleNamespace(bridge=SimpleNamespace(ctx=object()))

    capacity = labeling._prepare_formal_component_list_and_metadata(
        pipeline,
        world,
        object(),
        4,
        7,
        16,
        12,
        tile_mask_name="tiles",
        reject_invalid_components=True,
        defer_metadata_summary=defer,
    )

    assert capacity == 128
    assert pipeline.summarize_calls == expected_calls


def test_materialize_metadata_fusion_is_opt_in_and_reuses_materialize_slot() -> None:
    assert GPUCollapsePipeline()._incremental_materialize_metadata_fusion_enabled is True
    shader_path = (
        Path(__file__).parents[1]
        / "oracle_game/shaders/collapse/materialize_incremental_components_bridge.comp"
    )
    source = shader_path.read_text(encoding="ascii")

    assert "SUMMARIZE_COMPONENT_METADATA" in source
    assert "uint slot_plus_one = component_slot_for_label(label);" in source
    assert "atomicMin(component_metadata[metadata_base + 0], world_cell.x);" in source
    assert "atomicAdd(component_metadata[metadata_base + 4], 1);" in source


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


def _run_metadata_fusion_epoch(*, fused: bool) -> tuple[dict[str, bytes], int, int]:
    # Both axes end in partial 32-cell tiles. Three separated shapes make the
    # metadata reduction exercise independent component slots and bboxes.
    engine = WorldEngine(width=70, height=45, simulation_backend="gpu")
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU collapse pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        for x in range(2, 13):
            engine.set_cell(x, 5, "log_solid", mark_dirty=False)
        for y in range(5, 11):
            engine.set_cell(2, y, "log_solid", mark_dirty=False)
        for y in range(17, 20):
            for x in range(35, 49):
                engine.set_cell(x, y, "log_solid", mark_dirty=False)
        for x in range(64, 69):
            engine.set_cell(x, 37, "log_solid", mark_dirty=False)
        for y in range(37, 43):
            engine.set_cell(68, y, "log_solid", mark_dirty=False)

        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.prewarm_formal_connected_collapse()
        pipeline._incremental_materialize_metadata_fusion_enabled = fused

        phase_resources = (
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
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
        # Separate GL contexts do not guarantee initial buffer contents. These
        # outputs are epoch-owned but are not all published before phase 3, so
        # give every compared byte the same defined initial state.
        for name in phase_resources[4:]:
            buffer = engine.bridge.buffers[name]
            buffer.write(bytes(buffer.size))
        resources = pipeline.resources
        assert resources is not None
        for buffer in (
            resources.component_count,
            resources.component_labels,
            resources.component_flags,
            resources.component_metadata,
            resources.component_invalid,
        ):
            buffer.write(bytes(buffer.size))
        engine.bridge.buffers["island_runtime_count"].write(bytes(4))
        _seed_all_dirty_tiles(engine)

        snapshots: dict[str, bytes] = {}
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            for frame_id in range(1, 5):
                engine.frame_id = frame_id
                engine.collapse_solver.advance_formal_gpu_dirty_epoch(engine)
                ctx = engine.bridge.ctx
                assert ctx is not None
                ctx.finish()
                for name in phase_resources:
                    snapshots[f"phase{frame_id - 1}.{name}"] = engine.bridge.buffers[name].read()
        finally:
            engine._world_simulation_frame_active = previous_frame_active

        component_count_raw = resources.component_count.read(size=4)
        component_count = int(np.frombuffer(component_count_raw, dtype=np.uint32, count=1)[0])
        snapshots["component_count"] = component_count_raw
        snapshots["component_labels"] = resources.component_labels.read(
            size=component_count * np.dtype(np.int32).itemsize
        )
        snapshots["component_metadata"] = resources.component_metadata.read(
            size=component_count * 5 * np.dtype(np.int32).itemsize
        )

        runtime_count_raw = engine.bridge.buffers["island_runtime_count"].read(size=4)
        runtime_count = int(np.frombuffer(runtime_count_raw, dtype=np.int32, count=1)[0])
        snapshots["island_runtime_count"] = runtime_count_raw
        snapshots["island_runtime"] = engine.bridge.buffers["island_runtime"].read(
            size=runtime_count * ISLAND_RUNTIME_DTYPE.itemsize
        )
        snapshots["material"] = engine.bridge.textures["material"].read(alignment=1)

        for name in (
            collapse_dirty.COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
            collapse_dirty.COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
            collapse_dirty.COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
            collapse_dirty.COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
        ):
            snapshots[f"dirty_queue.{name}"] = engine.bridge.buffers[name].read()
        snapshots["next_island_id"] = int(engine.next_island_id).to_bytes(8, "little")
        return snapshots, runtime_count, component_count
    finally:
        engine.close()


def test_materialize_metadata_fusion_matches_full_four_frame_epoch_raw_bytes() -> None:
    control, control_runtime_count, control_component_count = _run_metadata_fusion_epoch(
        fused=False
    )
    candidate, candidate_runtime_count, candidate_component_count = _run_metadata_fusion_epoch(
        fused=True
    )

    differences: list[str] = []
    for name in sorted(control.keys() | candidate.keys()):
        control_raw = control.get(name)
        candidate_raw = candidate.get(name)
        if control_raw == candidate_raw:
            continue
        if control_raw is None or candidate_raw is None:
            differences.append(f"{name}: missing from one variant")
            continue
        common_size = min(len(control_raw), len(candidate_raw))
        first_offset = next(
            (
                offset
                for offset in range(common_size)
                if control_raw[offset] != candidate_raw[offset]
            ),
            common_size,
        )
        different_bytes = sum(
            lhs != rhs
            for lhs, rhs in zip(control_raw, candidate_raw, strict=False)
        ) + abs(len(control_raw) - len(candidate_raw))
        differences.append(
            f"{name}: first_offset={first_offset}, differing_bytes={different_bytes}, "
            f"sizes={len(control_raw)}/{len(candidate_raw)}"
        )
    assert not differences, "raw-byte differences:\n" + "\n".join(differences)
    assert candidate_runtime_count == control_runtime_count
    assert candidate_component_count == control_component_count
    assert candidate_component_count >= 2
    metadata = np.frombuffer(candidate["component_metadata"], dtype=np.int32).reshape((-1, 5))
    nonempty_metadata = metadata[metadata[:, 4] > 0]
    assert nonempty_metadata.shape[0] >= 2
    assert np.unique(nonempty_metadata[:, :4], axis=0).shape[0] >= 2
