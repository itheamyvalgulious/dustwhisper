from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np
import pytest

from oracle_game.sim.gpu_optics import GPUOpticsPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source
from oracle_game.types import CellFlag, Phase
from oracle_game.world import WorldEngine


def _write_zero_sentinel(resource: object) -> None:
    size = resource.size
    if isinstance(size, tuple):
        texel_count = int(np.prod(size, dtype=np.int64))
        byte_count = texel_count * int(resource.components) * int(str(resource.dtype)[1:])
    else:
        byte_count = int(size)
    resource.write(bytes(byte_count))


def _initialize_optics_resources(pipeline: GPUOpticsPipeline, engine: WorldEngine) -> None:
    resources = pipeline._ensure_resources(engine)
    for field in fields(resources):
        resource = getattr(resources, field.name)
        if callable(getattr(resource, "write", None)) and hasattr(resource, "size"):
            _write_zero_sentinel(resource)


def _seed_reaction_emitter(engine: WorldEngine) -> None:
    emitters = np.zeros((512, 4), dtype=np.float32)
    emitters[0] = (engine.width - 2.0, engine.height - 3.0, -1.0, -0.2)
    emitters[1] = (1.1, 31.0, 0.14, 0.0)
    counts = np.zeros((16,), dtype=np.uint32)
    counts[0] = 1
    engine.bridge.buffers["reaction_light_emitter"].write(emitters.tobytes())
    engine.bridge.buffers["reaction_light_emitter_count"].write(counts.tobytes())
    engine.bridge.mark_gpu_authoritative(
        "reaction_light_emitter",
        "reaction_light_emitter_count",
    )


def _capture_authoritative(engine: WorldEngine) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name in sorted(engine.bridge.gpu_authoritative_resources):
        if name in engine.bridge.buffers:
            snapshot[f"authoritative.buffer.{name}"] = engine.bridge.buffers[name].read()
        if name in engine.bridge.textures:
            snapshot[f"authoritative.texture.{name}"] = engine.bridge.textures[name].read()
    return snapshot


def _sorted_live_u32(resource: object, count: int) -> bytes:
    values = np.frombuffer(resource.read(), dtype=np.uint32)[:count].copy()
    values.sort()
    return values.tobytes()


def _capture_case(*, candidate: bool) -> tuple[list[dict[str, bytes]], list[bool]]:
    engine = WorldEngine(width=41, height=35, gas_cell_size=4)
    pipeline = engine.optics_solver.gpu_pipeline
    if not pipeline.available(engine):
        engine.close()
        pytest.skip("GPU optics pipeline is not available")
    try:
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        for y in range(2, engine.height, 5):
            for x in range(3, engine.width, 7):
                engine.set_cell(x, y, "gold_solid", phase=Phase.STATIC_SOLID, mark_dirty=False)
        engine.cell_flags[::3, 1::4] |= np.uint8(int(CellFlag.REACTION_LATCHED))
        for row in engine.active.active_tile_ttl:
            row[:] = [3] * len(row)
        for row in engine.active.active_chunk_mask:
            row[:] = [True] * len(row)

        _initialize_optics_resources(pipeline, engine)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )
        _seed_reaction_emitter(engine)
        pipeline._full_active_mask_hydration_elision_enabled = candidate

        emitters = [
            {
                "light_type": "visible_light",
                "origin": (2, 4),
                "direction": (1.0, 0.15),
                "spread": 0.12,
                "strength": 1.0,
                "range_cells": 37,
            },
            {
                "light_type": "magic_light",
                "origin": (20, 33),
                "direction": (0.1, -1.0),
                "spread": 0.09,
                "strength": 0.8,
                "range_cells": 29,
            },
        ]
        snapshots: list[dict[str, bytes]] = []
        used: list[bool] = []
        for frame_id in (1, 2):
            if frame_id == 2:
                engine.bridge.clear_gpu_authoritative(
                    "reaction_light_emitter",
                    "reaction_light_emitter_count",
                )
                partial_ttl = np.zeros(
                    (engine.active.tile_height, engine.active.tile_width),
                    dtype=np.int32,
                )
                partial_ttl[0, 0] = 3
                partial_ttl[-1, -1] = 3
                engine.bridge.buffers["active_tile_ttl"].write(partial_ttl.tobytes())

            solve_cell_mask = np.ones((engine.height, engine.width), dtype=np.bool_)
            solve_gas_mask = np.ones((engine.gas_height, engine.gas_width), dtype=np.bool_)
            if frame_id == 2:
                solve_cell_mask.fill(False)
                solve_cell_mask[: engine.active.tile_size, : engine.active.tile_size] = True
                solve_gas_mask.fill(False)
                solve_gas_mask[:4, :4] = True

            engine.frame_id = frame_id
            engine._world_simulation_frame_active = True
            try:
                pipeline.step(
                    engine,
                    emitters,
                    solve_cell_mask=solve_cell_mask,
                    solve_gas_mask=solve_gas_mask,
                )
            finally:
                engine._world_simulation_frame_active = False
            assert engine.bridge.ctx is not None
            engine.bridge.ctx.finish()
            resources = pipeline.resources
            assert resources is not None
            used.append(pipeline.last_full_active_mask_hydration_elision_used)
            snapshot = _capture_authoritative(engine)
            sparse_runtime = resources.sparse_runtime.read()
            sparse_counts = np.frombuffer(sparse_runtime, dtype=np.uint32)
            snapshot.update(
                {
                    "trace.cell_accum": resources.cell_dose_accum.read(),
                    "trace.gas_accum": resources.gas_dose_accum.read(),
                    "trace.illum_accum": resources.illum_accum.read(),
                    "trace.cell_tile_marks": resources.sparse_cell_tile_marks.read(),
                    "trace.gas_tile_marks": resources.sparse_gas_tile_marks.read(),
                    "trace.cell_tile_list": resources.sparse_cell_tile_list.read(),
                    "trace.gas_tile_list": resources.sparse_gas_tile_list.read(),
                    "sparse.runtime": sparse_runtime,
                    "sparse.cell_list.live_sorted": _sorted_live_u32(
                        resources.sparse_cell_list, int(sparse_counts[0])
                    ),
                    "sparse.gas_list.live_sorted": _sorted_live_u32(
                        resources.sparse_gas_list, int(sparse_counts[1])
                    ),
                    "sparse.visible_list.live_sorted": _sorted_live_u32(
                        resources.sparse_visible_list, int(sparse_counts[2])
                    ),
                    "sparse.visible_marks": resources.sparse_visible_marks.read(),
                    "resident.cell_dose": resources.cell_dose.read(),
                    "resident.gas_dose": resources.gas_dose.read(),
                    "resident.illum_layers": resources.illum_layers.read(),
                    "resident.visible": resources.visible_tex.read(),
                }
            )
            if frame_id == 2:
                snapshot["partial.active_cell"] = resources.active_cell_tex.read()
                snapshot["partial.active_gas"] = resources.active_gas_tex.read()
            snapshots.append(snapshot)
        return snapshots, used
    finally:
        engine.close()


def test_full_active_mask_hydration_elision_is_default_on_and_strictly_gated() -> None:
    pipeline = GPUOpticsPipeline()
    assert pipeline._full_active_mask_hydration_elision_enabled is True
    assert pipeline.last_full_active_mask_hydration_elision_used is False
    upload_source = inspect.getsource(GPUOpticsPipeline._upload_inputs)
    assert "and active_authoritative" in upload_source
    assert "and force_all_active" in upload_source
    assert "if not skip_active_mask_hydration:" in upload_source
    for common_shader in (
        "optics/_trace_common_full_active.comp",
        "optics/_trace_common_full_active_shift.comp",
    ):
        source = shader_source(common_shader, _SHADER_SUBS)
        assert "uniform sampler2D active_cell_tex" not in source
        assert "uniform sampler2D active_gas_tex" not in source
        assert source.count("return true;") == 2


def test_full_active_mask_hydration_elision_is_two_frame_full_to_partial_exact() -> None:
    control, control_used = _capture_case(candidate=False)
    candidate, candidate_used = _capture_case(candidate=True)
    assert control_used == [False, False]
    assert candidate_used == [True, False]
    assert candidate == control
