from __future__ import annotations

import inspect

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_heat_lazy_action_inputs_is_default_disabled() -> None:
    pipeline = GPUHeatPipeline()

    assert pipeline._terminal_lazy_action_inputs_enabled is True


def test_heat_lazy_action_inputs_has_strict_sparse_resident_gate() -> None:
    source = inspect.getsource(GPUHeatPipeline.step)
    terminal_source = inspect.getsource(GPUHeatPipeline._run_apply_terminal4x6)

    assert "terminal_lazy_action_inputs = bool(" in source
    assert "pipeline._terminal_sparse_resident_specialization_enabled" in source
    assert "and bridge_aux_residency" in source
    assert "and terminal_dirty_publish_fusion" in source
    assert "and terminal_dirty_workgroup_aggregation" in source
    assert "and not terminal_phase_fusion" in source
    assert "and not heat_gas_bridge_residency" in source
    assert "lazy_action_inputs=terminal_lazy_action_inputs" in source
    assert "sparse_resident_specialized" in terminal_source
    assert "and pipeline._terminal_lazy_action_inputs_enabled" in terminal_source
    assert '"apply_terminal4x6_sparse_resident_lazy_action_inputs"' in terminal_source


def test_heat_lazy_action_shaders_defer_unchanged_field_reads() -> None:
    substitutions = {
        **_SHADER_SUBS,
        "DIRTY_WORKGROUP_AGGREGATE": 1,
        "TERMINAL_SPARSE_RESIDENT_SPECIALIZED": 1,
        "HEAT_LAZY_ACTION_INPUTS": 1,
    }
    terminal = shader_source("heat/apply_terminal4x6.comp", substitutions)
    targets = shader_source(
        "heat/phase_boil_targets.comp",
        substitutions,
        includes=["heat/_common.comp"],
    )

    assert "#if 1" in terminal
    assert "bool timer_reset = false;" in terminal
    assert "bool temperature_loaded = false;" in terminal
    assert "original_integrity = texelFetch(integrity_tex, cell, 0).x;" in terminal
    assert "if (timer_reset)" in terminal
    assert "if (temperature_changed)" in terminal
    assert "if (integrity_changed)" in terminal
    assert "if (velocity_changed)" in terminal
    assert "if (cell_state_value != original_cell_state) {\n        mark_dirty" in terminal
    assert "if (phase_target <= 0)" in targets
    assert "integrity_value = texelFetch(integrity_tex, gid, 0).x;" in targets
