from __future__ import annotations

import inspect

from oracle_game.sim.gpu_optics import GPUOpticsPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_gpu_optics_gas_owner_warp_compaction_is_default_on_and_strictly_gated() -> None:
    pipeline = GPUOpticsPipeline()
    assert pipeline._sparse_gas_owner_warp_compaction_enabled is True
    assert pipeline._sparse_gas_owner_warp_compaction_supported is False

    host = inspect.getsource(GPUOpticsPipeline._build_tile_seeded_optics_worklists)
    assert "self._sparse_gas_owner_warp_compaction_enabled" in host
    assert "self._sparse_gas_owner_warp_compaction_supported" in host
    assert "int(world.gas_cell_size) == 4" in host
    assert "self._sparse_gas_visible_scan_fusion_enabled" in host
    assert 'self.programs["sparse_build_tile_worklists_gas_owner_nv32"]' in host


def test_gpu_optics_gas_owner_warp_compaction_batches_unique_gas_outputs() -> None:
    source = shader_source(
        "optics/sparse_build_tile_worklists.comp",
        {
            **_SHADER_SUBS,
            "SPARSE_GAS_OWNER_WARP_COMPACTION": 1,
            "SPARSE_WARP_EXTENSIONS": "\n".join(
                (
                    "#extension GL_NV_gpu_shader5 : require",
                    "#extension GL_NV_shader_thread_group : require",
                    "#extension GL_NV_shader_thread_shuffle : require",
                )
            ),
        },
    )

    assert "uint warp_prefix_sum(uint value, out uint warp_total)" in source
    assert "uint warp_reserve(uint runtime_index, uint count, out uint warp_total)" in source
    assert "uint active_mask = ballotThreadNV(has_dose);" in source
    assert "uint owned_visible_mask = 0u;" in source
    assert "uint visible_output = warp_reserve(2u, owned_count, warp_total);" in source
    assert "sparse_visible_marks[cell_index] = sparse_generation;" in source
    owner_body = source.split("void build_gas_entries_owner_warp()", 1)[1]
    owner_body = owner_body.split("#endif", 1)[0]
    assert "atomicExchange(sparse_visible_marks" not in owner_body
    assert "gas_cell * 4" in owner_body


def test_gpu_optics_gas_owner_warp_compaction_probes_nv32_before_compiling() -> None:
    ensure = inspect.getsource(GPUOpticsPipeline._ensure_programs)
    for extension in (
        "GL_NV_gpu_shader5",
        "GL_NV_shader_thread_group",
        "GL_NV_shader_thread_shuffle",
    ):
        assert extension in ensure
    assert '"optics/query_nv_warp_size.comp"' in ensure
    assert "owner_warp_size == 32" in ensure
