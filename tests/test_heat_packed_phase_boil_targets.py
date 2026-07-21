from __future__ import annotations

import inspect

from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_heat_packed_phase_boil_targets_is_default_enabled() -> None:
    pipeline = GPUHeatPipeline()

    assert pipeline._packed_phase_boil_targets_enabled is True
    assert pipeline.last_packed_phase_boil_targets_used is False


def test_heat_packed_phase_boil_targets_has_strict_lazy_static_table_gate() -> None:
    step_source = inspect.getsource(GPUHeatPipeline.step)
    target_source = inspect.getsource(GPUHeatPipeline._run_phase_boil_targets)
    terminal_source = inspect.getsource(GPUHeatPipeline._run_apply_terminal4x6)

    assert "and terminal_lazy_action_inputs" in step_source
    assert 'table_generations.get("materials", 0)' in step_source
    assert 'table_generations.get("reactions", 0)' in step_source
    assert "pipeline._packed_phase_boil_table_signature == current_table_signature" in step_source
    assert '"phase_boil_targets_packed_lazy_action_inputs"' in target_source
    assert "packed heat targets require lazy action inputs" in target_source
    assert '"apply_terminal4x6_sparse_resident_packed_lazy_action_inputs"' in terminal_source
    assert "packed heat targets require the sparse lazy terminal" in terminal_source


def test_heat_packed_phase_boil_target_shaders_exchange_one_word_per_cell() -> None:
    substitutions = {
        **_SHADER_SUBS,
        "DIRTY_WORKGROUP_AGGREGATE": 1,
        "TERMINAL_SPARSE_RESIDENT_SPECIALIZED": 1,
        "HEAT_LAZY_ACTION_INPUTS": 1,
        "HEAT_PACKED_PHASE_BOIL_TARGETS": 1,
    }
    targets = shader_source(
        "heat/phase_boil_targets.comp",
        substitutions,
        includes=["heat/_common.comp"],
    )
    terminal = shader_source("heat/apply_terminal4x6.comp", substitutions)

    assert "int packed_targets = phase_target | (boil_target << 8);" in targets
    assert "#if !1\nlayout(r32f, binding=6) writeonly uniform image2D boil_target_img;" in targets
    assert "int packed_targets = int(texelFetch(phase_target_tex, cell, 0).x + 0.5);" in terminal
    assert "target_material = is_active ? (packed_targets & 0xFF) : 0;" in terminal
    assert "gas_boil_target = (packed_targets >> 8) & 0xFF;" in terminal
    assert "#if !1\nlayout(binding=5) uniform sampler2D boil_target_tex;" in terminal
