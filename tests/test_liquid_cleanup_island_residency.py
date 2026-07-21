from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.gpu_liquid_bridge import _load_authoritative_bridge_inputs
from oracle_game.sim.shader_loader import shader_source


def test_cleanup_island_residency_is_default_off() -> None:
    assert GPULiquidPipeline()._cleanup_bridge_island_residency_enabled is False


def test_cleanup_island_residency_has_a_separate_shader_input_gate() -> None:
    source = shader_source(
        "liquid/cleanup_runtime.comp",
        {**_SHADER_SUBS, "DIRECT_BRIDGE_ISLAND_INPUTS": 1},
    )
    assert "const bool DIRECT_BRIDGE_ISLAND_INPUTS = 1 != 0;" in source
    assert "DIRECT_BRIDGE_ISLAND_INPUTS" in source


def test_cleanup_island_residency_selects_local16_variant() -> None:
    source = inspect.getsource(GPULiquidPipeline._run_cleanup_runtime)
    assert '"cleanup_runtime_bridge_aux_island16"' in source


def test_cleanup_island_residency_requires_tile_bridge_fusion() -> None:
    source = inspect.getsource(_load_authoritative_bridge_inputs)
    assert "_tile_solve_bridge_hydration_fusion_enabled" in source
