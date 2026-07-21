from __future__ import annotations

import numpy as np
import pytest

from oracle_game.sim.gpu_optics import GPUOpticsPipeline, _SHADER_SUBS
from oracle_game.sim.gpu_motion import GPUMotionPipeline
from oracle_game.sim.shader_loader import shader_source
from oracle_game.types import CellFlag
from oracle_game.world import WorldEngine


def _capture_tile_seeded_optics(
    *,
    candidate: bool,
    partial: bool = False,
    local_atomic_max: bool = False,
    capture_runtime: bool = False,
) -> list[dict[str, bytes]]:
    engine = WorldEngine(width=40, height=33, gas_cell_size=4)
    pipeline = engine.optics_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU optics pipeline is not available")
    try:
        engine.emitters.append(
            {
                "light_type": "visible_light",
                "origin": (2, 7),
                "direction": (1.0, 0.2),
                "spread": 0.13,
                "strength": 1.0,
                "range_cells": 37,
            }
        )
        engine.cell_flags[::3, 1::4] |= np.uint8(int(CellFlag.REACTION_LATCHED))
        if partial:
            engine.active.mark_rect(0, 0, 20, 20)
        else:
            engine.active.mark_rect(0, 0, engine.width, engine.height)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()

        reaction_emitters = np.zeros((512, 4), dtype=np.float32)
        reaction_emitters[0] = (38.0, 30.0, -1.0, -0.15)
        reaction_emitters[1] = (1.1, 35.0, 0.12, 0.0)
        reaction_counts = np.zeros((16,), dtype=np.uint32)
        reaction_counts[0] = 1
        if not partial:
            engine.bridge.buffers["reaction_light_emitter"].write(reaction_emitters.tobytes())
            engine.bridge.buffers["reaction_light_emitter_count"].write(reaction_counts.tobytes())
            engine.bridge.mark_gpu_authoritative("reaction_light_emitter", "reaction_light_emitter_count")

        pipeline._sparse_tile_seeded_build_enabled = candidate
        pipeline._sparse_tile_local_atomic_max_enabled = local_atomic_max
        snapshots: list[dict[str, bytes]] = []
        for _ in range(2):
            engine._world_simulation_frame_active = True
            try:
                solve_cell_mask = np.zeros((engine.height, engine.width), dtype=np.bool_)
                solve_gas_mask = np.zeros((engine.gas_height, engine.gas_width), dtype=np.bool_)
                if partial:
                    solve_cell_mask[:20, :20] = True
                    solve_gas_mask[:5, :5] = True
                else:
                    solve_cell_mask[:] = True
                    solve_gas_mask[:] = True
                pipeline.step(
                    engine,
                    list(engine.emitters),
                    solve_cell_mask=solve_cell_mask,
                    solve_gas_mask=solve_gas_mask,
                )
            finally:
                engine._world_simulation_frame_active = False
            engine.bridge.ctx.finish()
            resources = pipeline.resources
            assert resources is not None
            snapshot = {
                    "light": engine.bridge.textures["light"].read(),
                    "visible": engine.bridge.textures["visible_illumination"].read(),
                    "cell_dose": engine.bridge.buffers["cell_optical_dose"].read(),
                    "gas_dose": engine.bridge.buffers["gas_optical_dose"].read(),
                    "cell_core": engine.bridge.buffers["cell_core"].read(),
                    "guard": engine.bridge.buffers["optics_light_dose_guard"].read(),
                    "cell_accum": resources.cell_dose_accum.read(),
                    "gas_accum": resources.gas_dose_accum.read(),
                    "illum_accum": resources.illum_accum.read(),
            }
            if capture_runtime:
                snapshot["sparse_runtime"] = resources.sparse_runtime.read()
            snapshots.append(snapshot)
            assert pipeline.last_reaction_latch_clear_fused
        return snapshots
    finally:
        engine.close()


def test_gpu_optics_sparse_tile_seeded_build_is_default_on_and_shader_bounded() -> None:
    assert GPUOpticsPipeline()._sparse_tile_seeded_build_enabled is True
    assert GPUOpticsPipeline()._sparse_tile_local_atomic_max_enabled is False
    assert GPUMotionPipeline()._reaction_latch_handoff_clear_enabled is True
    trace = shader_source("optics/trace_body.comp", {**_SHADER_SUBS, "TILE_SEEDED_BUILD": 1})
    tile_build = shader_source(
        "optics/sparse_build_tile_worklists.comp",
        _SHADER_SUBS,
    )
    assert "atomicExchange(sparse_cell_tile_marks[tile_index], sparse_generation)" in trace
    assert "atomicExchange(sparse_gas_tile_marks[tile_index], sparse_generation)" in trace
    assert "tile_slot >= sparse_runtime[16]" in tile_build
    assert "tile_slot >= sparse_runtime[17]" in tile_build


@pytest.mark.parametrize("partial", (False, True))
def test_gpu_optics_sparse_tile_local_atomic_max_is_two_frame_raw_byte_exact(
    partial: bool,
) -> None:
    control = _capture_tile_seeded_optics(
        candidate=True,
        partial=partial,
        local_atomic_max=False,
        capture_runtime=True,
    )
    candidate = _capture_tile_seeded_optics(
        candidate=True,
        partial=partial,
        local_atomic_max=True,
        capture_runtime=True,
    )
    assert candidate == control


def test_gpu_optics_sparse_tile_seeded_build_is_two_frame_raw_byte_exact() -> None:
    control = _capture_tile_seeded_optics(candidate=False)
    candidate = _capture_tile_seeded_optics(candidate=True)
    assert candidate == control


def test_gpu_optics_sparse_tile_seeded_build_partial_masks_are_two_frame_raw_byte_exact() -> None:
    control = _capture_tile_seeded_optics(candidate=False, partial=True)
    candidate = _capture_tile_seeded_optics(candidate=True, partial=True)
    assert candidate == control
