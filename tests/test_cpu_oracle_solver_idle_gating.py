"""CPU-oracle idle honesty and explicit gating for gas/heat/liquid solvers.

The CPU solver paths are explicit oracles (``_require_cpu_oracle_backend``):
they only run when ``simulation_backend == "cpu"`` and must fail loudly when
invoked as a non-oracle direct call.  When a step has no work (no active
tiles), the solver must report ``last_backend == "idle"`` honestly instead of
retaining a stale ``"cpu"``/``"gpu"`` from a previous step (which made a
silent no-op look like a completed solve).
"""

from __future__ import annotations

import numpy as np
import pytest

from oracle_game.world import WorldEngine

SOLVER_CASES = {
    "gas": lambda engine: engine.gas_solver.step(engine, 1.0 / 60.0),
    "heat": lambda engine: engine.heat_solver.step(engine, 1.0 / 60.0),
    "liquid": lambda engine: engine.liquid_solver.step(engine),
}


def _solver(engine: WorldEngine, name: str):
    return getattr(engine, f"{name}_solver")


def _clear_active_tiles(engine: WorldEngine) -> None:
    for row in engine.active.active_tile_ttl:
        for index in range(len(row)):
            row[index] = 0


@pytest.mark.parametrize("solver_name", sorted(SOLVER_CASES))
def test_cpu_oracle_idle_step_reports_idle_not_stale_backend(solver_name: str) -> None:
    engine = WorldEngine(width=32, height=24, simulation_backend="cpu")
    solver = _solver(engine, solver_name)
    step = SOLVER_CASES[solver_name]

    engine.active.mark_rect(0, 0, engine.width, engine.height)
    step(engine)
    assert solver.last_backend == "cpu"

    _clear_active_tiles(engine)
    step(engine)
    assert solver.last_backend == "idle"


@pytest.mark.parametrize("solver_name", sorted(SOLVER_CASES))
def test_cpu_oracle_engine_step_without_active_tiles_does_not_fake_compute(
    solver_name: str,
) -> None:
    engine = WorldEngine(width=32, height=24, simulation_backend="cpu")
    _clear_active_tiles(engine)

    engine.step(1.0 / 60.0)

    solver = _solver(engine, solver_name)
    assert solver.last_backend == "idle"
    snapshot = solver.runtime_snapshot()
    assert int(np.count_nonzero(snapshot["solve_tile_mask"])) == 0


@pytest.mark.parametrize("solver_name", sorted(SOLVER_CASES))
@pytest.mark.parametrize("with_active_tiles", [False, True])
def test_non_oracle_direct_call_raises_even_when_idle(
    solver_name: str, with_active_tiles: bool
) -> None:
    engine = WorldEngine(width=32, height=24, simulation_backend="cpu")
    solver = _solver(engine, solver_name)
    solver.gpu_pipeline.available = lambda world: False
    engine.simulation_backend = "hybrid"  # neither "cpu" (oracle) nor "gpu" (strict)
    if with_active_tiles:
        engine.active.mark_rect(0, 0, engine.width, engine.height)
    else:
        _clear_active_tiles(engine)

    with pytest.raises(RuntimeError, match="CPU oracle path requires simulation_backend='cpu'"):
        SOLVER_CASES[solver_name](engine)
