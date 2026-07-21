from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_dispatch, gpu_motion_powder, gpu_motion_resources
from oracle_game.sim.gpu_motion import GPUMotionPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_powder_provisional_moving_worklist_is_default_on_and_strictly_gated() -> None:
    pipeline = GPUMotionPipeline()
    assert pipeline._powder_provisional_moving_worklist_enabled is True

    host = inspect.getsource(gpu_motion_powder.resolve_and_apply_powders)
    for condition in (
        "source_indexed_apply",
        "compact_reservations",
        "lazy_compact_expand",
        "pipeline._powder_trivial_blocked_classification_enabled",
        "pipeline._powder_apply_tile_workgroup_dedup_enabled",
    ):
        assert condition in host
    assert 'resolve_name += "_moving_worklist"' in host
    assert "resources.powder_apply_incoming.bind_to_storage_buffer(binding=14)" in host


def test_powder_resolve_appends_every_provisionally_moving_reservation_index() -> None:
    source = shader_source(
        "motion/resolve_powder_reservations.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1,
            "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
            "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
        },
    )
    append = "uint moving_slot = atomicAdd(provisional_moving_count[0], 1u);"
    store = "provisional_moving_indices[moving_slot] = index;"
    resolution = "store_resolution(index, resolved, resolve_state);"

    assert "resolve_state == 1 || resolve_state == 2" in source
    assert "&& !same_cell(source, resolved)" in source
    assert source.index(resolution) < source.index(append) < source.index(store)


def test_powder_source_indexed_apply_dispatches_only_worklist_indices() -> None:
    source = shader_source(
        "motion/apply_powder_reservations_source_indexed_direct.comp",
        {
            **_SHADER_SUBS,
            "POWDER_COMPACT_RESERVATION": 1,
            "POWDER_COMPACT_LAZY_EXPAND": 1,
            "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
        },
    )
    moving_index = "int moving_index = int(gl_GlobalInvocationID.x);"
    reservation_index = "int reservation_index = provisional_moving_indices[moving_index];"
    load = "PowderReservation reservation = load_reservation(reservation_index);"

    assert source.index(moving_index) < source.index(reservation_index) < source.index(load)
    apply_host = inspect.getsource(gpu_motion_powder._dispatch_source_indexed_direct_apply)
    assert "count_buffer=(" in apply_host
    assert "resources.powder_provisional_moving_count" in apply_host
    assert "resources.powder_apply_incoming.bind_to_storage_buffer(binding=11)" in apply_host


def test_powder_moving_count_is_a_separate_four_byte_resource() -> None:
    resources_source = inspect.getsource(gpu_motion_resources._ensure_resources)
    release_source = inspect.getsource(gpu_motion_resources.release)
    dispatch_source = inspect.getsource(
        gpu_motion_dispatch._build_powder_reservation_dispatch_args
    )

    assert "powder_provisional_moving_count=ctx.buffer(reserve=4" in resources_source
    assert "pipeline.resources.powder_provisional_moving_count" in release_source
    assert "count_buffer: Any | None = None" in dispatch_source
