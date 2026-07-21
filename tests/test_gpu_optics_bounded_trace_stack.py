from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from oracle_game.sim.gpu_optics import (
    BOUNDED_RAY_STACK,
    BOUNDED_RAY_STACK_MAX_BOUNCE,
    GPUOpticsPipeline,
    _SHADER_SUBS,
)
from oracle_game.sim.shader_loader import shader_source


def _world(max_bounces: tuple[int, ...], *, gas_cell_size: int = 4) -> SimpleNamespace:
    light_table = np.zeros(
        (len(max_bounces),),
        dtype=[("max_bounce", np.int32)],
    )
    light_table["max_bounce"] = max_bounces
    return SimpleNamespace(
        gas_cell_size=gas_cell_size,
        bridge=SimpleNamespace(shadow_typed_tables={"light_table": light_table}),
    )


def test_bounded_trace_stack_is_default_on_for_the_formal_specialization() -> None:
    pipeline = GPUOpticsPipeline()
    assert pipeline._bounded_trace_stack_enabled is True
    assert (
        pipeline._trace_emitter_program_name(
            _world((0, 1, 2)),
            force_all_active=True,
            tile_seeded_build=True,
        )
        == "trace_emitters_full_active_shift_tile_seeded_stack4"
    )


def test_bounded_trace_stack_selection_falls_back_for_every_other_path() -> None:
    pipeline = GPUOpticsPipeline()

    assert not pipeline._trace_max_bounce_at_most(
        _world((BOUNDED_RAY_STACK_MAX_BOUNCE + 1,)),
        BOUNDED_RAY_STACK_MAX_BOUNCE,
    )
    assert (
        pipeline._trace_emitter_program_name(
            _world((0, 1, BOUNDED_RAY_STACK_MAX_BOUNCE + 1)),
            force_all_active=True,
            tile_seeded_build=True,
        )
        == "trace_emitters_full_active_shift_tile_seeded"
    )
    assert (
        pipeline._trace_emitter_program_name(
            _world((0, 1, 2), gas_cell_size=3),
            force_all_active=True,
            tile_seeded_build=True,
        )
        == "trace_emitters_full_active_tile_seeded"
    )
    assert (
        pipeline._trace_emitter_program_name(
            _world((0, 1, 2)),
            force_all_active=False,
            tile_seeded_build=True,
        )
        == "trace_emitters_tile_seeded"
    )
    assert (
        pipeline._trace_emitter_program_name(
            _world((0, 1, 2)),
            force_all_active=True,
            tile_seeded_build=False,
        )
        == "trace_emitters_full_active_shift"
    )


def test_bounded_trace_stack_shader_only_changes_stack_capacity() -> None:
    control = shader_source("optics/trace_body.comp", _SHADER_SUBS)
    candidate = shader_source(
        "optics/trace_body.comp",
        {**_SHADER_SUBS, "MAX_RAY_STACK": BOUNDED_RAY_STACK},
    )

    assert f"vec4 stack_pos_dir[{BOUNDED_RAY_STACK}];" in candidate
    assert f"vec2 stack_energy_bounce[{BOUNDED_RAY_STACK}];" in candidate
    assert f"stack_count < {BOUNDED_RAY_STACK}" in candidate

    def normalized(source: str) -> str:
        lines = []
        for line in source.splitlines():
            if any(
                marker in line
                for marker in (
                    "stack_pos_dir[",
                    "stack_energy_bounce[",
                    "stack_count < ",
                )
            ):
                line = line.replace(str(_SHADER_SUBS["MAX_RAY_STACK"]), "STACK_CAPACITY")
                line = line.replace(str(BOUNDED_RAY_STACK), "STACK_CAPACITY")
            lines.append(line)
        return "\n".join(lines)

    assert normalized(candidate) == normalized(control)
