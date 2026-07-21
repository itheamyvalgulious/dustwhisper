from __future__ import annotations

import numpy as np
import pytest

from oracle_game.types import CellFlag, Phase
from oracle_game.world import WorldEngine


def _run_sparse_case(*, candidate: bool) -> list[dict[str, bytes]]:
    engine = WorldEngine(width=35, height=29, gas_cell_size=3)
    pipeline = engine.optics_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU optics pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        engine.set_cell(17, 14, "gold_solid", phase=Phase.STATIC_SOLID, mark_dirty=False)
        engine.cell_flags[::3, 1::4] |= np.uint8(int(CellFlag.REACTION_LATCHED))
        engine.active.mark_rect(0, 0, engine.width, engine.height)
        emitters = [
            {
                "light_type": light_name,
                "origin": (3 + index * 5, 4 + index * 3),
                "direction": (1.0, 0.1 * (index - 1)),
                "spread": 0.1 + index * 0.03,
                "strength": 0.8 + index * 0.1,
                "range_cells": 28,
            }
            for index, light_name in enumerate(
                ("visible_light", "holy_light", "chaos_light", "magic_light")
            )
        ]
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        pipeline._sparse_optics_worklists_enabled = candidate
        snapshots: list[dict[str, bytes]] = []
        for _ in range(3):
            engine._world_simulation_frame_active = True
            try:
                pipeline.step(
                    engine,
                    emitters,
                    solve_cell_mask=np.ones((engine.height, engine.width), dtype=np.bool_),
                    solve_gas_mask=np.ones((engine.gas_height, engine.gas_width), dtype=np.bool_),
                )
            finally:
                engine._world_simulation_frame_active = False
            assert engine.bridge.ctx is not None
            engine.bridge.ctx.finish()
            resources = pipeline.resources
            assert resources is not None
            snapshots.append(
                {
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
            )
        return snapshots
    finally:
        engine.close()


def test_gpu_optics_sparse_worklists_are_three_frame_byte_exact() -> None:
    assert _run_sparse_case(candidate=True) == _run_sparse_case(candidate=False)


def test_gpu_optics_sparse_worklists_default_enabled_and_bridge_upload_invalidates() -> None:
    engine = WorldEngine(width=12, height=9, gas_cell_size=3)
    pipeline = engine.optics_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU optics pipeline is not available")
    try:
        assert pipeline._sparse_optics_worklists_enabled is True
        resources = pipeline._ensure_resources(engine)
        resources.sparse_initialized = True
        engine.bridge.clear_gpu_authoritative("visible_illumination")
        engine.bridge.sync_world(engine)
        assert resources.sparse_initialized is False
    finally:
        engine.close()
