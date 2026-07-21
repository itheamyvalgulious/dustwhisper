from __future__ import annotations

import inspect
from types import SimpleNamespace

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_terminal_segment_meta_lazy_zero_is_default_on_and_has_independent_program() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._terminal_segment_meta_lazy_zero_enabled is True

    programs = inspect.getsource(GPUReactionPipeline._ensure_programs)
    assert (
        '"material_pair_fused_terminal_local32x8_dirty_fast_shared_transpose_segment_zero"'
        in programs
    )
    source = shader_source(
        "reactions/material_pair_fused.comp",
        {
            **_SHADER_SUBS,
            "MATERIAL_PAIR_TERMINAL_HANDOFF": 1,
            "MATERIAL_PAIR_TERMINAL_SEGMENT_META_ZERO": 1,
        },
        includes=["reactions/_common.comp", "reactions/_lhs_candidate.comp"],
    )
    assert "#if 1 && 1" in source
    zero = source.index("segment_meta = vec2(0.0);")
    material_material = source.index("apply_material_material_rules(", zero)
    material_gas = source.index("apply_material_gas_rules(", material_material)
    material_light = source.index("apply_material_light_rules(", material_gas)
    terminal_publish = source.index("terminal_publish_cell(", material_light)
    assert zero < material_material < material_gas < material_light < terminal_publish


def test_terminal_segment_meta_lazy_zero_requires_every_prior_producer() -> None:
    pipeline = GPUReactionPipeline()
    pipeline._terminal_segment_meta_lazy_zero_enabled = True
    pipeline._formal_segment_batch_key = ("segment", 1)
    pipeline._begin_formal_segment_meta_lazy_zero()

    pipeline._record_formal_segment_cell_meta_in_flags(True)
    assert pipeline._can_use_terminal_segment_meta_zero()
    pipeline._record_formal_segment_cell_meta_in_flags(False)
    assert not pipeline._can_use_terminal_segment_meta_zero()


def test_terminal_segment_meta_lazy_zero_physically_clears_once_on_legacy_touch() -> None:
    pipeline = GPUReactionPipeline()
    pipeline._terminal_segment_meta_lazy_zero_enabled = True
    pipeline._formal_segment_batch_key = ("segment", 2)
    pipeline._begin_formal_segment_meta_lazy_zero()
    calls: list[bool] = []

    def clear_segment(
        _world: object,
        _resources: object,
        *,
        clear_light_counters: bool = False,
    ) -> None:
        calls.append(clear_light_counters)
        pipeline._formal_segment_meta_physically_cleared = True
        pipeline._formal_segment_meta_logically_zero = False
        pipeline._formal_segment_all_prior_cell_meta_in_flags = False

    pipeline._clear_segment_transient_state = clear_segment
    world = SimpleNamespace(profile_passes_enabled=False, profile_passes_sync=False)
    resources = object()
    pipeline._ensure_formal_segment_meta_physical_zero(world, resources)
    pipeline._ensure_formal_segment_meta_physical_zero(world, resources)

    assert calls == [False]
    assert pipeline.segment_meta_lazy_fallback_clear_count == 1
    assert not pipeline._can_use_terminal_segment_meta_zero()


def test_terminal_segment_meta_lazy_zero_hooks_all_legacy_consumers() -> None:
    from oracle_game.sim import gpu_reactions_cell_pass, gpu_reactions_segments
    from oracle_game.sim import gpu_reactions_transient

    cell_pass = inspect.getsource(gpu_reactions_cell_pass)
    segments = inspect.getsource(gpu_reactions_segments.flush_formal_reaction_segment)
    transient = inspect.getsource(gpu_reactions_transient)
    ensure = "_ensure_formal_segment_meta_physical_zero"
    assert cell_pass.count(ensure) >= 3
    assert ensure in segments
    assert ensure in transient
