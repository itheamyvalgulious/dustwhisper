from __future__ import annotations

import inspect

from oracle_game.sim import gpu_motion_powder
from oracle_game.sim.gpu_motion import GPUMotionPipeline
from oracle_game.sim.shader_loader import shader_source


def test_powder_compact_reservations_are_default_enabled_and_private() -> None:
    pipeline = GPUMotionPipeline()

    assert pipeline._powder_compact_reservation_enabled is True
    assert pipeline._powder_compact_reservation_lazy_expand_enabled is True
    assert gpu_motion_powder._COMPACT_POWDER_RESERVATION_ITEMSIZE == 24
    assert "powder_compact_reservations" in inspect.getsource(
        gpu_motion_powder.resolve_and_apply_powders
    )


def test_powder_compact_reservation_gate_checks_abi_ranges_and_formal_path() -> None:
    gate = inspect.getsource(gpu_motion_powder._compact_powder_reservation_safe)

    assert "_source_indexed_powder_apply_enabled" in gate
    assert "_bridge_authoritative_cell_blockers" in gate
    assert "min_step < 0 or max_step > 32768" in gate
    assert "world.width - 1 + max_step <= 32767" in gate
    assert "world.height - 1 + max_step <= 32767" in gate


def test_powder_compact_shaders_preserve_signed_coordinates_and_public_abi() -> None:
    generate = shader_source("motion/generate_powder_reservations.comp")
    resolve = shader_source("motion/resolve_powder_reservations.comp")
    apply = shader_source("motion/apply_powder_reservations_source_indexed_direct.comp")

    for source in (generate, resolve, apply):
        assert "#if {{POWDER_COMPACT_RESERVATION}}" in source
        assert "struct CompactPowderReservation" in source
    assert "bridge_cell_core[cell_index(source) * 5 + 1]" in generate
    assert "int(word << 16u) >> 16" in resolve
    assert "int(word) >> 16" in resolve
    assert "compact_reservations[index].material_resolve & 0xFFFFu" in resolve
    assert "writeonly buffer ExpandedPowderReservations" in apply
    assert "expanded_reservations[index] = reservation" in apply


def test_powder_compact_lazy_expand_keeps_observable_abi_on_demand() -> None:
    host = inspect.getsource(gpu_motion_powder.resolve_and_apply_powders)
    materialize = inspect.getsource(
        gpu_motion_powder.materialize_compact_powder_reservations
    )
    apply = shader_source("motion/apply_powder_reservations_source_indexed_direct.comp")
    expand = shader_source("motion/expand_compact_powder_reservations.comp")

    assert "lazy_compact_expand" in host
    assert "publish_bridge_compact_powder_reservations" in host
    assert "#if !{{POWDER_COMPACT_LAZY_EXPAND}}" in apply
    assert 'pipeline.programs["expand_compact_powder_reservations"]' in materialize
    assert 'bridge.mark_gpu_authoritative("powder_reservation_standard")' in materialize
    assert "struct PowderReservation" in expand
    assert "reservations[index] = reservation" in expand
