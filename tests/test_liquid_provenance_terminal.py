from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def _terminal_subs() -> dict[str, object]:
    return {
        **_SHADER_SUBS,
        "DIRECT_BRIDGE_INPUTS": 1,
        "DIRECT_BRIDGE_AUX_INPUTS": 1,
        "TILE_SNAPSHOT_OUTPUT": 1,
        "LIQUID_PROVENANCE": 1,
        "PROVENANCE_TERMINAL": 1,
    }


def test_liquid_provenance_terminal_is_default_enabled() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._provenance_terminal_enabled is True
    assert pipeline._provenance_terminal_frame_enabled is False
    assert pipeline._provenance_init_fusion_enabled is True
    assert pipeline._provenance_init_fusion_frame_enabled is False
    assert pipeline.last_provenance_init_fusion_used is False
    assert pipeline.last_provenance_terminal_used is False


def test_flow_terminal_writes_private_five_word_payload() -> None:
    source = shader_source("liquid/liquid_flow_intent_shared_halo.comp", _terminal_subs())
    assert "layout(std430, binding=10) writeonly buffer BridgeCellCoreSpareBuffer" in source
    assert "store_terminal_core(gid, cell_state, velocity, primary_role)" in source
    for word in ("bridge_cell_core_spare[out_word]", "bridge_cell_core_spare[out_word + 1]", "bridge_cell_core_spare[out_word + 2]", "bridge_cell_core_spare[out_word + 3]", "bridge_cell_core_spare[out_word + 4]"):
        assert word in source


def test_provenance_variants_carry_source_and_texture_markers() -> None:
    substitutions = _terminal_subs()
    buoyancy = shader_source("liquid/buoyancy_fused.comp", substitutions)
    copy = shader_source("liquid/copy_with_pending.comp", substitutions)
    placeholder = shader_source("liquid/placeholder_displace.comp", substitutions)
    assert "liquid_provenance_in[]" in buoyancy
    assert "liquid_provenance_out[]" in buoyancy
    assert "liquid_provenance_out[provenance_index]" in copy
    assert "const uint PROVENANCE_TEXTURE = 0xFFFFFFFEu" in placeholder
    assert "liquid_provenance_out[target.y * cell_grid_size.x + target.x] = PROVENANCE_TEXTURE" in placeholder
    flow = shader_source("liquid/liquid_flow_intent_shared_halo.comp", substitutions)
    assert "copy_original_core(gid)" in flow
    assert "provenance == PROVENANCE_EMPTY" in flow


def test_liquid_step_swaps_spare_only_after_terminal_flow() -> None:
    source = inspect.getsource(GPULiquidPipeline.step)
    bridge_source = inspect.getsource(GPULiquidPipeline._run_liquid_intent_pass)
    solve = inspect.getsource(GPULiquidPipeline._run_tile_solve)
    buoyancy = inspect.getsource(GPULiquidPipeline._run_buoyancy_pass)
    assert "_provenance_terminal_frame_enabled" in source
    assert "phase_c_defer_cell_publish" in source
    assert "liquid_flow_intent_shared_halo_provenance" in bridge_source
    assert "cell_core_spare" in bridge_source
    assert "resources.provenance_out.bind_to_storage_buffer(binding=8)" in bridge_source
    assert "resources.provenance_in.bind_to_storage_buffer(binding=9)" in bridge_source
    assert "resources.provenance_in, resources.provenance_out" in source
    assert "program_name.startswith(\"buoyancy_fused\")" in buoyancy
    assert "init_liquid_provenance" in inspect.getsource(GPULiquidPipeline._run_provenance_init)
    assert "ensure_cell_core_spare" not in solve


def test_provenance_init_fusion_is_gated_and_skips_only_identity_pass() -> None:
    source = inspect.getsource(GPULiquidPipeline.step)
    init_source = inspect.getsource(GPULiquidPipeline._run_provenance_init)
    assert "_provenance_init_fusion_frame_enabled" in source
    assert "skip_when_all_tiles_active=pipeline._provenance_init_fusion_frame_enabled" in source
    assert "last_provenance_init_fusion_used" in source
    assert 'pipeline.programs["retarget_provenance_init_dispatch"]' in init_source
    assert "program.run_indirect(resources.affected_tile_prefetch_dispatch_args)" in init_source
