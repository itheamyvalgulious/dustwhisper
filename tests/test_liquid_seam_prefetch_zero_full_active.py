from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline
from oracle_game.sim.shader_loader import shader_source


def test_liquid_seam_prefetch_zero_full_active_is_default_enabled() -> None:
    assert GPULiquidPipeline()._seam_prefetch_zero_full_active_enabled is True


def test_liquid_seam_prefetch_retarget_preserves_partial_active_work() -> None:
    source = shader_source("liquid/retarget_seam_prefetch_dispatch.comp")
    build_source = inspect.getsource(GPULiquidPipeline._build_seam_boundary_dispatch)
    prefetch_source = inspect.getsource(
        GPULiquidPipeline._prefetch_seam_boundary_bridge_inputs
    )

    assert "boundary_count[0] * max(workgroups_per_boundary, 1u)" in source
    assert "active_tile_count[0] >= total_tile_count" in source
    assert "? 0u" in source
    assert 'pipeline.programs["retarget_seam_prefetch_dispatch"]' in build_source
    assert "resources.affected_tile_prefetch_dispatch_args" in build_source
    assert "pipeline._seam_prefetch_zero_full_active_enabled" in prefetch_source
