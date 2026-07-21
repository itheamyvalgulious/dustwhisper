from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_cleanup_local16_is_default_enabled() -> None:
    assert GPULiquidPipeline()._cleanup_local16_enabled is True


def test_liquid_cleanup_local16_specialization_covers_sixteen_by_sixteen_cells() -> None:
    source = shader_source(
        "liquid/cleanup_runtime.comp",
        {
            **_SHADER_SUBS,
            "PASS_LOCAL_SIZE": 16,
            "PASS_LOCAL_SIZE_MINUS_1": 15,
            "MAX_MATERIALS": 256,
            "MAX_MATERIALS_MINUS_1": 255,
            "DIRECT_BRIDGE_INPUTS": 1,
            "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
            "DIRECT_BRIDGE_AUX_INPUTS": 0,
            "DIRECT_BRIDGE_ISLAND_INPUTS": 0,
        },
    )
    assert "local_size_x=16" in source
    assert "local_size_y=16" in source


def test_liquid_cleanup_local16_retargets_to_four_groups_per_tile() -> None:
    source = inspect.getsource(GPULiquidPipeline._run_cleanup_runtime)
    assert '"cleanup_runtime_bridge_aux16"' in source
    assert 'retarget_program["workgroups_per_tile"].value = 4' in source
