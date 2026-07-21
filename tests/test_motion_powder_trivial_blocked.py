from __future__ import annotations

import inspect

from oracle_game.sim.gpu_motion import GPUMotionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_powder_trivial_blocked_classification_is_default_on_and_compact_only() -> None:
    pipeline = GPUMotionPipeline()
    assert pipeline._powder_trivial_blocked_classification_enabled is True

    source = shader_source(
        "motion/resolve_powder_reservations.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
        },
    )
    assert "#if 1 && 1" in source
    assert "generated_sources_prevalidated" in source
    assert "use_precomputed_fallback_blockers" in source
    assert "same_cell(reserved, source)" in source
    assert "fallback_blocker_mask == 7" in source
    assert "if (!trivial_blocked)" in source
    assert source.index("store_resolution(index, resolved, resolve_state);") > source.index(
        "if (!trivial_blocked)"
    )
    assert "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION" not in shader_source(
        "motion/generate_powder_reservations.comp",
        _SHADER_SUBS,
    )


def test_powder_trivial_blocked_classification_keeps_resolve_fallbacks() -> None:
    program_source = inspect.getsource(GPUMotionPipeline._ensure_programs)
    assert 'self.programs["resolve_powder_reservations_compact"]' in program_source
    assert 'self.programs["resolve_powder_reservations_compact_trivial_blocked"]' in program_source
    assert '"resolve_powder_reservations_compact_tile_dedup_trivial_blocked"' in program_source

    host_source = inspect.getsource(GPUMotionPipeline.resolve_and_apply_powders)
    assert "if compact_reservations:" in host_source
    assert "pipeline._powder_trivial_blocked_classification_enabled" in host_source
    assert 'resolve_name += "_trivial_blocked"' in host_source
