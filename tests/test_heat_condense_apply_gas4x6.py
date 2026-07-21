from __future__ import annotations

import inspect

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_heat_condense_apply_gas4x6_is_default_enabled() -> None:
    assert GPUHeatPipeline()._condense_apply_gas4x6_fusion_enabled is True


def test_heat_condense_apply_gas4x6_keeps_strict_dispatch_gate() -> None:
    source = inspect.getsource(GPUHeatPipeline.step)
    assert "_condense_apply_gas4x6_fusion_enabled" in source
    assert "int(world.gas_cell_size) == 4" in source
    assert "int(world.gas_concentration.shape[0]) == 6" in source
    assert "if not fuse_condense_apply_gas:" in source
    assert "fuse_condense_targets=fuse_condense_apply_gas" in source


def test_heat_condense_apply_gas4x6_preserves_species_order_and_original_gas() -> None:
    source = shader_source(
        "heat/apply_gas_targets4x6.comp",
        _SHADER_SUBS,
        includes=["heat/_common.comp"],
    )
    assert "for (int local_y = 0; local_y < 4; ++local_y)" in source
    assert "for (int species = 0; species < 6; ++species)" in source
    assert "float original_value = texelFetch(gas_tex" in source
    assert "&& original_value > 0.7" in source
    assert "value += float(boil_count) * 0.6 * dt;" in source
    assert "condense_rank += 1;" in source
    assert "empty_count >= condense_rank" in source
