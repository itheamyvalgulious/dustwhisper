from __future__ import annotations

import inspect

from oracle_game.sim import gpu_collapse_labeling
from oracle_game.sim.gpu_collapse import GPUCollapsePipeline


def test_runtime_admission_stride_dispatch_is_default_enabled() -> None:
    assert GPUCollapsePipeline()._runtime_admission_stride_dispatch_enabled is True


def test_runtime_admission_dispatch_scales_group_coverage_by_stride() -> None:
    source = inspect.getsource(gpu_collapse_labeling._publish_compact_component_island_runtime)

    assert "256 * dispatch_stride" in source
    assert "invocations_per_group=dispatch_invocations" in source
