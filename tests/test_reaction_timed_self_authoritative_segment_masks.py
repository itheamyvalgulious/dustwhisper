from __future__ import annotations

from types import SimpleNamespace

from oracle_game.sim.reactions import ReactionSolver


def _formal_gpu_world() -> SimpleNamespace:
    return SimpleNamespace(
        width=96,
        height=64,
        gas_width=24,
        gas_height=16,
        active=SimpleNamespace(tile_width=6, tile_height=4),
        simulation_backend="gpu",
        _world_simulation_frame_active=True,
        bridge=SimpleNamespace(
            gpu_authoritative_resources={"active_tile_ttl"},
        ),
    )


def test_timed_self_authoritative_segment_masks_are_default_on() -> None:
    solver = ReactionSolver()
    assert solver.gpu_pipeline._timed_self_authoritative_segment_masks_enabled


def test_timed_self_authoritative_segment_masks_do_not_materialize_cpu_grids() -> None:
    solver = ReactionSolver()
    solver.gpu_pipeline._timed_self_authoritative_segment_masks_enabled = True
    world = _formal_gpu_world()

    def forbidden_materialization(_world: object) -> object:
        raise AssertionError("timed/self segment masks must remain GPU-authoritative")

    solver._full_solve_masks = forbidden_materialization  # type: ignore[method-assign]
    for stage in ("timed", "self"):
        masks = solver._solve_masks(
            world,
            seed_timer_cells=True,
            stage=stage,
        )
        assert solver._all_full_gpu_authoritative_masks(*masks)
