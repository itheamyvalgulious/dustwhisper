from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_powder
from oracle_game.sim.gpu_motion import GPUMotionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_powder_nontrivial_resolve_worklist_is_default_on_and_strictly_gated() -> None:
    pipeline = GPUMotionPipeline()
    assert pipeline._powder_nontrivial_resolve_worklist_enabled is True

    host = inspect.getsource(gpu_motion_powder.resolve_and_apply_powders)
    gate = host[host.index("nontrivial_resolve_worklist = bool(") :]
    assert "provisional_moving_worklist" in gate
    assert "pipeline._powder_precomputed_fallback_blockers_enabled" in gate
    assert "and not nontrivial_resolve_worklist" in gate
    assert 'resolve_name += "_nontrivial_worklist"' in gate
    assert "count_buffer=(" in gate
    assert "resources.powder_direct_apply_unsafe" in gate


def test_powder_nontrivial_worklist_scratch_roles_are_source_indexed_only() -> None:
    source_gate = inspect.getsource(gpu_motion_powder._source_indexed_powder_apply_enabled)
    direct_safety = inspect.getsource(gpu_motion_powder._powder_direct_apply_is_safe)
    generate = inspect.getsource(gpu_motion_powder._run_generate_powder_reservations)

    assert "not pipeline._powder_direct_bridge_apply_enabled" in source_gate
    assert "resources.powder_direct_apply_unsafe" in direct_safety
    assert "resources.powder_apply_outgoing" in direct_safety
    assert "resources.powder_direct_apply_unsafe.bind_to_storage_buffer(binding=7)" in generate
    assert "resources.powder_apply_outgoing.bind_to_storage_buffer(binding=9)" in generate


def test_powder_generate_finalizes_terminal_blocked_and_emits_canonical_indices() -> None:
    source = shader_source(
        "motion/generate_powder_reservations.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
            "POWDER_SOURCE_TILE_PRODUCER": 1,
        },
    )
    terminal = "terminal_blocked = same_cell(reserved, source)"
    append = "int reservation_index = append_reservation("
    resolve_append = "nontrivial_resolve_indices[resolve_slot] = reservation_index;"

    assert "phase_id == phase_liquid" in source
    assert "solver_kind == 2" in source
    assert "precomputed_fallback_mask == 7" in source
    assert "terminal_blocked ? 0 : precomputed_fallback_mask" in source
    assert source.index(terminal) < source.index(append) < source.index(resolve_append)


def test_powder_nontrivial_resolve_variant_has_no_shared_hash_and_only_stages_targets() -> None:
    source = shader_source(
        "motion/resolve_powder_reservations.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
            "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
            "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
            "POWDER_SOURCE_TILE_PRODUCER": 1,
            "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 0,
        },
    )
    canonical_index = "int index = nontrivial_resolve_indices[work_index];"
    load = "PowderReservation reservation = load_reservation(index);"
    source_stage = "stage_apply_tile(source);"

    assert source.index(canonical_index) < source.index(load)
    assert "#if !1" in source
    assert source.index("#if !1") < source.index(source_stage) < source.index(
        "#endif", source.index(source_stage)
    )
    shared_hash = "shared int apply_tile_hash[APPLY_TILE_HASH_SLOTS];"
    hash_index = source.index(shared_hash)
    assert source.rfind("#if 0", 0, hash_index) >= 0
    assert "append_apply_tile_index(tile_index);" in source


def test_powder_nontrivial_worklist_source_tiles_use_workgroup_producer() -> None:
    source = shader_source(
        "motion/generate_powder_reservations.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
            "POWDER_SOURCE_TILE_PRODUCER": 1,
        },
    )
    reduce = "atomicOr(generated_reservation_in_workgroup, 1u);"
    publish = "append_source_tile(source_tile);"

    assert source.index(reduce) < source.rindex("barrier();") < source.index(publish)
    assert "atomicExchange(affected_tile_flags[tile_index], 1u)" in source
