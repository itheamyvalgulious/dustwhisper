from pathlib import Path

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_heat_sparse_bridge_residency_is_default_enabled() -> None:
    pipeline = GPUHeatPipeline()

    assert pipeline._heat_sparse_bridge_residency_enabled is True


def test_heat_sparse_bridge_residency_gates_aux_domain() -> None:
    source = Path("oracle_game/sim/gpu_heat_stages.py").read_text()

    assert "sparse_bridge_aux_residency = bool(" in source
    assert "or sparse_bridge_aux_residency" in source
    assert "and pipeline._terminal_inplace_sparse_write_enabled" in source
    assert "and bridge_aux_residency" in source


def test_heat_bridge_resident_aux_reads_displaced_only_on_transition() -> None:
    source = shader_source("heat/apply_terminal4x6.comp", _SHADER_SUBS)

    assert "if (!TERMINAL_USE_BRIDGE_DISPLACED)" in source
    assert "displaced_value = texelFetch(displaced_tex, cell, 0).x" in source
    assert "if (is_placeholder_material(previous_material))" in source
    assert "displaced_value = displaced_at(cell)" in source


def test_heat_aux_cache_is_rehydrated_when_residency_gate_exits() -> None:
    source = Path("oracle_game/sim/gpu_heat_stages.py").read_text()

    assert "skip_cell_aux_hydration=bridge_aux_residency" in source
    assert "and not skip_cell_aux_hydration" in source
