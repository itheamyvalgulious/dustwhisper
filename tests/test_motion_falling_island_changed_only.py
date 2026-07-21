from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_island
from oracle_game.sim.gpu_motion import GPUMotionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_falling_island_changed_only_materialization_is_default_enabled() -> None:
    assert GPUMotionPipeline()._falling_island_materialization_changed_only_enabled is True


def test_falling_island_changed_only_specialization_uses_existing_shader_marker() -> None:
    shader = shader_source(
        "motion/apply_falling_island_materialization.comp",
        {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 2},
    )
    dispatch_source = inspect.getsource(
        gpu_motion_island._dispatch_apply_falling_island_materialization
    )

    assert "#if 2 == 2" in shader
    assert '"apply_falling_island_materialization_bridge_changed_only"' in dispatch_source
