from __future__ import annotations

import numpy as np

from oracle_game.world import WorldEngine


_RUNTIME_MASK_ATTRIBUTES = (
    "last_structural_mask",
    "last_support_seed_mask",
    "last_supported_mask",
    "last_unsupported_mask",
    "last_delayed_pending_mask",
    "last_immune_unsupported_mask",
    "last_collapsed_cell_mask",
)


def test_formal_gpu_collapse_reset_defers_cpu_runtime_masks() -> None:
    engine = WorldEngine(width=17, height=13, simulation_backend="gpu")
    try:
        solver = engine.collapse_solver
        engine._world_simulation_frame_active = True

        solver.reset_runtime_state(engine)

        assert solver.last_solve_region_mask.shape == (13, 17)
        for attribute in _RUNTIME_MASK_ATTRIBUTES:
            assert getattr(solver, attribute).shape == (0, 0)

        snapshot = solver.runtime_snapshot(engine)
        assert snapshot["solve_region_mask"].shape == (13, 17)
        for key in (
            "structural_mask",
            "support_seed_mask",
            "supported_mask",
            "unsupported_mask",
            "delayed_pending_mask",
            "immune_unsupported_mask",
            "collapsed_cell_mask",
        ):
            assert snapshot[key].shape == (13, 17)
            assert not bool(np.any(snapshot[key]))
    finally:
        engine.close()


def test_collapse_reset_restores_independent_cpu_runtime_masks() -> None:
    engine = WorldEngine(width=9, height=7, simulation_backend="gpu")
    try:
        solver = engine.collapse_solver
        engine._world_simulation_frame_active = True
        solver.reset_runtime_state(engine)

        engine._world_simulation_frame_active = False
        solver.reset_runtime_state(engine)

        masks = [getattr(solver, attribute) for attribute in _RUNTIME_MASK_ATTRIBUTES]
        assert all(mask.shape == (7, 9) for mask in masks)
        masks[0][2, 3] = True
        assert all(not bool(mask[2, 3]) for mask in masks[1:])
    finally:
        engine.close()
