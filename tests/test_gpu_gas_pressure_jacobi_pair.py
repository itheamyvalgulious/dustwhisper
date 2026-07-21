from __future__ import annotations

import inspect

import numpy as np
import pytest

from oracle_game.sim.gpu_gas import GPUGasPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source
from oracle_game.world import WorldEngine

from test_gpu_gas_exact_candidates import _capture_gas_state, _seed_gas_world


def _capture_pair_case(*, candidate: bool, full_active: bool) -> tuple[list[dict[str, bytes]], list[bool]]:
    # 67x49 maps to a 17x13 gas grid, exercising partial pair cores on both axes.
    engine = WorldEngine(width=67, height=49, gas_cell_size=4)
    try:
        pipeline = engine.gas_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU gas pipeline is not available")
        solve_mask = _seed_gas_world(
            engine,
            seed=0x2A11 + int(full_active),
        )
        for tile_y in range(engine.active.tile_height):
            for tile_x in range(engine.active.tile_width):
                engine.active.active_tile_ttl[tile_y][tile_x] = (
                    3
                    if full_active or (tile_x + 2 * tile_y) % 3 != 0
                    else 0
                )
        for row in engine.active.active_chunk_mask:
            row[:] = [True] * len(row)

        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine.bridge.mark_gpu_authoritative(
            "flow_velocity",
            "ambient_temperature",
            "gas_concentration",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )
        pipeline._pressure_jacobi_pair_enabled = candidate
        pipeline._divergence_pressure_seed_enabled = True
        assert pipeline.pressure_iterations == 12

        ctx = engine.bridge.ctx
        assert ctx is not None
        pipeline._ensure_programs(ctx)
        resources = pipeline._ensure_resources(engine)
        ping_poison = np.full(
            (engine.gas_height, engine.gas_width),
            np.float32(12345.75),
            dtype=np.float32,
        ).tobytes()
        pong_poison = np.full(
            (engine.gas_height, engine.gas_width),
            np.float32(-9876.5),
            dtype=np.float32,
        ).tobytes()
        shadow_poison = np.full(
            (engine.gas_height, engine.gas_width),
            np.float32(2468.25),
            dtype=np.float32,
        ).tobytes()
        resources.pressure_ping.write(ping_poison)
        resources.pressure_pong.write(pong_poison)
        resources.pressure_cone_shadow.write(shadow_poison)

        snapshots: list[dict[str, bytes]] = []
        used: list[bool] = []
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            for frame_id, dt in ((1, 1.0 / 60.0), (2, 1.0 / 55.0)):
                engine.frame_id = frame_id
                pipeline.step(
                    engine,
                    dt,
                    solve_gas_mask=(
                        np.ones_like(solve_mask, dtype=np.bool_)
                        if full_active
                        else np.roll(solve_mask, shift=(1, -2), axis=(0, 1))
                    ),
                )
                snapshots.append(_capture_gas_state(engine))
                used.append(pipeline.last_pressure_jacobi_pair_used)
        finally:
            engine._world_simulation_frame_active = previous_frame_active
        return snapshots, used
    finally:
        engine.close()


def test_pressure_jacobi_pair_is_default_on_and_preserves_odd_tail() -> None:
    pipeline = GPUGasPipeline()
    assert pipeline._pressure_jacobi_pair_enabled is True
    assert pipeline.last_pressure_jacobi_pair_used is False
    source = inspect.getsource(GPUGasPipeline._run_pressure_jacobi)
    assert "pair_count = remaining_iterations // 2" in source
    assert "remaining_iterations - pair_count * 2" in source
    assert "resources.pressure_cone_shadow" in source
    assert "and self._formal_gpu_frame(world)" in inspect.getsource(
        GPUGasPipeline._can_run_pressure_jacobi_pair
    )

    shader = shader_source("gas/jacobi_pair.comp", _SHADER_SUBS)
    assert "local_size_x=16" in shader
    assert "const int CORE_SIZE = 12;" in shader
    canonical_expression = (
        "pressure_zero_at(left, tile_origin)\n"
        "                + pressure_zero_at(right, tile_origin)\n"
        "                + pressure_zero_at(down, tile_origin)\n"
        "                + pressure_zero_at(up, tile_origin)\n"
        "                - divergence_values[canonical_index]"
    )
    assert canonical_expression in shader


@pytest.mark.parametrize("full_active", (False, True), ids=("partial", "full"))
def test_pressure_jacobi_pair_is_two_frame_formal_raw_byte_exact(
    full_active: bool,
) -> None:
    control, control_used = _capture_pair_case(
        candidate=False,
        full_active=full_active,
    )
    candidate, candidate_used = _capture_pair_case(
        candidate=True,
        full_active=full_active,
    )
    assert control_used == [False, False]
    assert candidate_used == [True, True]
    assert candidate == control
