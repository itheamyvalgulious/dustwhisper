from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oracle_game.sim.gpu_motion import GPUMotionPipeline, powder_reservation_dtype
from oracle_game.types import Phase
from oracle_game.world import WorldEngine


ROOT = Path(__file__).resolve().parents[1]


def _run_powder_case(*, dedup: bool) -> list[tuple[dict[str, bytes], set[tuple[int, int]]]]:
    engine = WorldEngine(width=96, height=64, gas_cell_size=4)
    if not engine.motion_solver.gpu_pipeline.available(engine):
        engine.close()
        pytest.skip("GPU motion pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        rng = np.random.default_rng(8128)
        for y in range(3, engine.height - 3, 3):
            for x in range(1, engine.width - 1, 5):
                material = "water_liquid" if (x + y) % 4 == 0 else "sand_powder"
                phase = Phase.LIQUID if material == "water_liquid" else Phase.POWDER
                engine.set_cell(x, y, material, phase=phase, mark_dirty=False)
                velocities = np.asarray(
                    ((-120.0, 0.0), (120.0, 0.0), (0.0, 120.0), (60.0, 60.0)),
                    dtype=np.float32,
                )
                engine.velocity[y, x] = velocities[int(rng.integers(0, len(velocities)))]
                engine.cell_temperature[y, x] = np.float32(17.0 + (x % 11))
                engine.integrity[y, x] = np.uint16(20 + (y % 13))
                engine.island_id[y, x] = 20_000_003 + x + y * engine.width
                engine.entity_id[y, x] = 700 + x
                engine.placeholder_displaced_material[y, x] = 11 + (y % 7)

        # Exercise cross-tile target publication and a blocked DDA prefix.
        engine.set_cell(31, 8, "sand_powder", phase=Phase.POWDER, mark_dirty=False)
        engine.velocity[8, 31] = np.asarray((120.0, 0.0), dtype=np.float32)
        engine.set_cell(34, 8, "raw_stone_solid", mark_dirty=False)
        engine.set_cell(63, 24, "sand_powder", phase=Phase.POWDER, mark_dirty=False)
        engine.velocity[24, 63] = np.asarray((120.0, 0.0), dtype=np.float32)
        engine.active.mark_rect(0, 0, engine.width, engine.height)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "flow_velocity",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )

        pipeline = engine.motion_solver.gpu_pipeline
        pipeline._powder_apply_tile_workgroup_dedup_enabled = dedup
        ctx = engine.bridge.ctx
        assert ctx is not None
        snapshots: list[tuple[dict[str, bytes], set[tuple[int, int]]]] = []
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            for _ in range(2):
                engine.motion_solver.step(engine, 1.0 / 60.0)
                ctx.finish()
                pipeline.materialize_compact_powder_reservations(engine, download=False)
                ctx.finish()
                resources = pipeline.resources
                assert resources is not None
                reservation_count = int(
                    np.frombuffer(
                        engine.bridge.buffers["powder_reservation_count"].read(size=4),
                        dtype=np.int32,
                        count=1,
                    )[0]
                )
                active_count = int(
                    np.frombuffer(resources.active_tile_count.read(size=4), dtype=np.uint32, count=1)[0]
                )
                active_tiles = np.frombuffer(
                    resources.active_tile_list.read(size=active_count * 8),
                    dtype=np.int32,
                    count=active_count * 2,
                ).reshape((active_count, 2))
                reservations = np.frombuffer(
                    engine.bridge.buffers["powder_reservation"].read(
                        size=reservation_count * powder_reservation_dtype().itemsize
                    ),
                    dtype=powder_reservation_dtype(),
                    count=reservation_count,
                ).copy()
                if reservation_count:
                    source_xy = reservations["source_xy"]
                    reservations = reservations[
                        np.lexsort((source_xy[:, 0], source_xy[:, 1]))
                    ]
                snapshots.append(
                    (
                        {
                            "cell_core": engine.bridge.buffers["cell_core"].read(),
                            "material": engine.bridge.textures["material"].read(),
                            "island": engine.bridge.buffers["island_id"].read(),
                            "entity": engine.bridge.buffers["entity_id"].read(),
                            "displaced": engine.bridge.buffers[
                                "placeholder_displaced_material"
                            ].read(),
                            "active_ttl": engine.bridge.buffers["active_tile_ttl"].read(),
                            "active_meta": engine.bridge.buffers["active_meta"].read(),
                            "active_chunk_mask": engine.bridge.buffers["active_chunk_mask"].read(),
                            "reservation_count": engine.bridge.buffers[
                                "powder_reservation_count"
                            ].read(size=4),
                            "reservations_by_source": reservations.tobytes(),
                        },
                        {tuple(map(int, tile)) for tile in active_tiles},
                    )
                )
        finally:
            engine._world_simulation_frame_active = previous_frame_active
        return snapshots
    finally:
        engine.close()


def test_powder_apply_tile_workgroup_dedup_is_default_on_and_barrier_safe() -> None:
    source = (ROOT / "oracle_game/shaders/motion/resolve_powder_reservations.comp").read_text()

    assert GPUMotionPipeline()._powder_apply_tile_workgroup_dedup_enabled is True
    assert "shared int apply_tile_hash[APPLY_TILE_HASH_SLOTS];" in source
    assert "initialize_apply_tile_hash();" in source
    assert "publish_staged_apply_tiles();" in source
    assert source.index("initialize_apply_tile_hash();") < source.index("if (work_index < count)")
    assert source.index("if (work_index < count)") < source.index("publish_staged_apply_tiles();")


def test_powder_apply_tile_workgroup_dedup_is_two_frame_raw_byte_exact() -> None:
    control = _run_powder_case(dedup=False)
    candidate = _run_powder_case(dedup=True)

    assert len(control) == len(candidate) == 2
    for (control_raw, control_tiles), (candidate_raw, candidate_tiles) in zip(
        control, candidate, strict=True
    ):
        assert candidate_raw == control_raw
        assert candidate_tiles == control_tiles
