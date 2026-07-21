from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_powder
from oracle_game.sim.gpu_motion import GPUMotionPipeline
from oracle_game.sim.shader_loader import shader_source


def test_powder_precomputed_fallback_blockers_is_default_enabled() -> None:
    assert GPUMotionPipeline()._powder_precomputed_fallback_blockers_enabled is True


def test_powder_precomputed_fallback_blockers_preserves_winner_arbitration() -> None:
    host = inspect.getsource(gpu_motion_powder.resolve_and_apply_powders)
    generate = shader_source("motion/generate_powder_reservations.comp")
    resolve = shader_source("motion/resolve_powder_reservations.comp")

    assert 'generate["precompute_fallback_blockers"].value' in inspect.getsource(
        gpu_motion_powder._run_generate_powder_reservations
    )
    assert 'resolve["use_precomputed_fallback_blockers"].value' in host
    assert "int fallback_blocker_mask(" in generate
    assert "reservations[index].resolve_state = precomputed_fallback_mask << 8;" in generate
    assert "int fallback_blocker_mask = (reservation.resolve_state >> 8) & 7;" in resolve
    assert "fallback_blocked = use_precomputed_fallback_blockers" in resolve
    assert "&& target_winner_rank(fallback) < 0" in resolve
    assert "atomicMin(target_winners[resolved.y * cell_grid_size.x + resolved.x]" in resolve
