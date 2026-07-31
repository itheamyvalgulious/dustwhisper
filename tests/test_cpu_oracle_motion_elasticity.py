"""CPU-oracle vs GPU-mainline motion collision semantics at elasticity=0.

The GPU formal-frame mainline (shaders/motion/apply_powder_reservations_
source_indexed_direct.comp) leaves non-moving (fully blocked) powder
reservations untouched: the source cell keeps its velocity as-is.  The CPU
oracle (sim/motion_powder.py::_apply_powder_reservations) mirrors that
contract; collision response (normal-component bounce scaled by ``elasticity``,
tangential damping scaled by ``friction``) applies only to reservations that
actually moved but did not reach their desired target (partial moves).
"""

from __future__ import annotations

import numpy as np

from oracle_game.gpu import unpack_cell_core
from oracle_game.sim.gpu_motion_constants import GRAVITY_CELLS_PER_SECOND_SQ
from oracle_game.types import Phase
from oracle_game.world import WorldEngine

GRAVITY_TICK = GRAVITY_CELLS_PER_SECOND_SQ / 60.0  # gravity_scale=1.0 * dt * g for one 1/60s step


def _build_blocked_powder_case(engine: WorldEngine, *, elasticity: float) -> None:
    engine.patch_material(
        "sand_powder",
        friction=0.0,
        elasticity=elasticity,
        wind_coupling=0.0,
        drag_scale=0.0,
    )
    engine.clear_cell_region(0, 0, engine.width, engine.height)
    engine.set_cell(8, 8, "sand_powder", phase=Phase.POWDER, mark_dirty=False)
    engine.set_cell(7, 9, "raw_stone_solid", mark_dirty=False)
    engine.set_cell(8, 9, "raw_stone_solid", mark_dirty=False)
    engine.set_cell(9, 9, "raw_stone_solid", mark_dirty=False)
    engine.velocity[8, 8] = np.array([0.0, 60.0], dtype=np.float32)


def _gpu_cell_velocity(engine: WorldEngine, x: int, y: int) -> np.ndarray:
    core = np.frombuffer(
        engine.bridge.buffers["cell_core"].read(
            size=engine.width * engine.height * 5 * np.dtype(np.uint32).itemsize
        ),
        dtype=np.uint32,
    ).reshape((engine.height, engine.width, 5))
    unpacked = unpack_cell_core(core)
    return np.asarray(unpacked["velocity"][y, x], dtype=np.float32)


def test_cpu_oracle_blocked_powder_keeps_velocity_at_zero_elasticity() -> None:
    engine = WorldEngine(width=24, height=24, simulation_backend="cpu")
    sand_id = engine.rulebook.material_id("sand_powder")
    _build_blocked_powder_case(engine, elasticity=0.0)

    engine.step(1.0 / 60.0)

    assert engine.motion_solver.last_backend == "cpu"
    assert int(engine.material_id[8, 8]) == sand_id
    # Fully blocked: the GPU mainline keeps the source velocity untouched, so
    # the oracle must too (one gravity tick was added during integration).
    assert np.allclose(engine.velocity[8, 8], (0.0, 60.0 + GRAVITY_TICK), atol=1.0e-4)


def test_cpu_oracle_blocked_powder_keeps_velocity_at_nonzero_elasticity() -> None:
    engine = WorldEngine(width=24, height=24, simulation_backend="cpu")
    sand_id = engine.rulebook.material_id("sand_powder")
    _build_blocked_powder_case(engine, elasticity=0.5)

    engine.step(1.0 / 60.0)

    assert engine.motion_solver.last_backend == "cpu"
    assert int(engine.material_id[8, 8]) == sand_id
    # No movement happened, so no collision response fires — not even a bounce.
    assert np.allclose(engine.velocity[8, 8], (0.0, 60.0 + GRAVITY_TICK), atol=1.0e-4)


def test_cpu_oracle_partial_move_applies_elasticity_collision_response() -> None:
    def run_case(*, elasticity: float) -> tuple[int, np.ndarray]:
        engine = WorldEngine(width=24, height=24, simulation_backend="cpu")
        engine.patch_material(
            "sand_powder",
            friction=0.0,
            elasticity=elasticity,
            wind_coupling=0.0,
            drag_scale=0.0,
        )
        engine.clear_cell_region(0, 0, engine.width, engine.height)
        engine.set_cell(5, 8, "sand_powder", phase=Phase.POWDER, mark_dirty=False)
        engine.set_cell(7, 8, "raw_stone_solid", mark_dirty=False)
        engine.velocity[8, 5] = np.array([120.0, 0.0], dtype=np.float32)
        engine.step(1.0 / 60.0)
        return int(engine.material_id[8, 6]), engine.velocity[8, 6].copy()

    sand_id = WorldEngine(width=4, height=4, simulation_backend="cpu").rulebook.material_id(
        "sand_powder"
    )
    moved_flat, velocity_flat = run_case(elasticity=0.0)
    moved_bounce, velocity_bounce = run_case(elasticity=0.5)

    # Desired delta is (2, 0); the stone at (7, 8) clips the move to (1, 0), a
    # partial collision, so the normal component responds with -v * elasticity.
    assert moved_flat == sand_id
    assert moved_bounce == sand_id
    assert np.allclose(velocity_flat, (0.0, GRAVITY_TICK), atol=1.0e-4)
    assert np.allclose(velocity_bounce, (-60.0, GRAVITY_TICK), atol=1.0e-4)


def test_gpu_formal_frame_matches_cpu_oracle_for_blocked_powder(require_gpu) -> None:
    gpu_engine = WorldEngine(width=24, height=24, gpu_context=require_gpu)
    _build_blocked_powder_case(gpu_engine, elasticity=0.0)
    cpu_engine = WorldEngine(width=24, height=24, simulation_backend="cpu")
    _build_blocked_powder_case(cpu_engine, elasticity=0.0)

    gpu_engine.step(1.0 / 60.0)
    cpu_engine.step(1.0 / 60.0)

    try:
        gpu_velocity = _gpu_cell_velocity(gpu_engine, 8, 8)
    finally:
        gpu_engine.close()
    cpu_velocity = cpu_engine.velocity[8, 8]
    # The GPU bridge stores velocity as half floats; at magnitude ~60 the half
    # quantum is 0.03125, so allow one quantum of slack.
    assert np.allclose(gpu_velocity, cpu_velocity, atol=0.04)
    assert gpu_velocity[1] > 59.0  # kept, not zeroed
