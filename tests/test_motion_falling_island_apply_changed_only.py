from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_island
from oracle_game.sim.gpu_motion import GPUMotionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_falling_island_apply_changed_only_is_default_on_and_bridge_gated() -> None:
    pipeline = GPUMotionPipeline()
    assert pipeline._falling_island_apply_changed_only_enabled is True

    host = inspect.getsource(gpu_motion_island._dispatch_apply_falling_island_reservations)
    assert "direct_bridge_outputs" in host
    assert "pipeline._falling_island_apply_changed_only_enabled" in host
    assert '"apply_falling_island_reservations_bridge_changed_only"' in host


def test_falling_island_apply_changed_only_preserves_changed_writes() -> None:
    source = shader_source(
        "motion/apply_falling_island_reservations.comp",
        {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1, "ISLAND_APPLY_CHANGED_ONLY": 1},
    )
    incoming = source.index("store_incoming(gid, reservations[incoming_index], source_cell);")
    outgoing = source.index("store_clear(gid);")
    unchanged = source.index("if (!DIRECT_BRIDGE_OUTPUTS || !ISLAND_APPLY_CHANGED_ONLY)")
    assert incoming < outgoing < unchanged
    assert "refresh_changed_cell(gid);" in source
    assert "store_original(gid);" in source[unchanged:]
