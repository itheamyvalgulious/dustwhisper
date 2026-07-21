from __future__ import annotations

import inspect

from oracle_game.sim.gpu_motion import GPUMotionPipeline


def test_falling_island_resolve_minimal_hydration_is_default_enabled() -> None:
    assert GPUMotionPipeline()._falling_island_resolve_minimal_hydration_enabled is True


def test_falling_island_resolve_minimal_hydration_has_strict_gate_and_fallback() -> None:
    source = inspect.getsource(GPUMotionPipeline._dispatch_resolve_falling_island_reservations)

    assert "formal_frame" in source
    assert "pipeline._falling_island_resolve_minimal_hydration_enabled" in source
    assert "pipeline._bridge_context_active(world)" in source
    assert "pipeline._bridge_authoritative_powder_inputs(world)" in source
    assert "pipeline._active_scheduler_gpu_authoritative(world)" in source
    assert "pipeline._load_authoritative_materialization_inputs(" in source
    assert "use_existing_active_tile_dispatch=True" in source
    assert "pipeline._load_authoritative_bridge_inputs(" in source
