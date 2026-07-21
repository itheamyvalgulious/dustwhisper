from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_buoyancy_blocker_displaced_hydration_is_default_enabled() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._buoyancy_blocker_displaced_hydration_enabled is True
    assert pipeline._buoyancy_blocker_displaced_hydration_frame_enabled is False
    assert pipeline.last_buoyancy_blocker_displaced_hydration_used is False
    assert pipeline._blocker_displaced_hydration_frame_enabled is False


def test_candidate_has_strict_snapshot_cleanup_gate_and_canonical_fallback() -> None:
    step_source = inspect.getsource(GPULiquidPipeline.step)
    required_gates = (
        "_buoyancy_cleanup_split_fusion_frame_enabled",
        "_buoyancy_snapshot_pre_state_frame_enabled",
        "_provenance_terminal_frame_enabled",
        "_tile_solve_bridge_hydration_fusion_enabled",
        "_tile_solve_snapshot_output_fusion_enabled",
        "_compact_tile_solve_snapshot_enabled",
        "_bridge_aux_cleanup_fusion_enabled",
        "_placeholder_lazy_roles_enabled",
        "not pipeline._bridge_aux_residency_enabled",
    )
    assert all(gate in step_source for gate in required_gates)
    assert '"cell_core"' in step_source
    assert '"island_id"' in step_source
    assert '"entity_id"' in step_source
    assert '"placeholder_displaced_material"' in step_source
    assert "or pipeline._buoyancy_blocker_displaced_hydration_frame_enabled" in step_source


def test_loader_and_tile_solve_share_blocker_hydration_gate() -> None:
    load_source = inspect.getsource(GPULiquidPipeline._load_authoritative_bridge_inputs)
    tile_source = inspect.getsource(GPULiquidPipeline._run_tile_solve)
    assert "pipeline._blocker_displaced_hydration_frame_enabled" in load_source
    assert "pipeline._blocker_displaced_hydration_frame_enabled" in tile_source
    assert "load_bridge_blocker_displaced" in load_source
    assert "copy_island_id and not (skip_island_hydration or blocker_mask_inputs)" in load_source
    assert "copy_entity_id and not (direct_bridge_aux_inputs or blocker_mask_inputs)" in load_source
    assert "copy_displaced and not (direct_bridge_aux_inputs or blocker_mask_inputs)" in load_source
    assert "tile_solve_bridge_snapshot_provenance_blocker" in tile_source
    assert "_provenance_cleanup_terminal_fusion_frame_enabled" not in load_source
    assert "_provenance_cleanup_terminal_fusion_frame_enabled" not in tile_source


def test_snapshot_pre_blocker_specializations_preserve_pre_state_encoding() -> None:
    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    tile_source = shader_source(
        "liquid/tile_solve.comp",
        {
            **_SHADER_SUBS,
            "DIRECT_BRIDGE_INPUTS": 1,
            "TILE_SNAPSHOT_OUTPUT": 1,
            "TILE_COMPACT_SNAPSHOT": 1,
            "LIQUID_PROVENANCE": 1,
            "TILE_SNAPSHOT_PRE_STATE": 1,
            "TILE_BLOCKER_MASK_INPUT": 1,
        },
    )
    assert "tile_solve_bridge_snapshot_provenance_blocker_snapshot_pre" in ensure_source
    assert "tile_solve_bridge_snapshot_row_stream_provenance_blocker_snapshot_pre" in ensure_source
    assert "texelFetch(blocker_mask_tex, cell, 0).x != 0u" in tile_source
    assert "uint pre_material = cell_state & 0xFFu" in tile_source
    assert "(pre_state << 16u) | packed_source" in tile_source


def test_restore_reads_bridge_identities_but_keeps_displaced_texture() -> None:
    ensure_source = inspect.getsource(GPULiquidPipeline._ensure_programs)
    cleanup_source = inspect.getsource(GPULiquidPipeline._run_cleanup_runtime)
    shader = shader_source(
        "liquid/cleanup_runtime.comp",
        {
            **_SHADER_SUBS,
            "PASS_LOCAL_SIZE": 16,
            "PASS_LOCAL_SIZE_MINUS_1": 15,
            "DIRECT_BRIDGE_INPUTS": 1,
            "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
            "DIRECT_BRIDGE_ISLAND_INPUTS": 1,
            "DIRECT_BRIDGE_ENTITY_INPUTS": 1,
            "RESTORE_BRIDGE_AUX_OUTPUTS": 1,
        },
    )
    assert "cleanup_runtime_bridge_aux_restore_bridge_ids16" in ensure_source
    assert "if pipeline._buoyancy_blocker_displaced_hydration_frame_enabled" in cleanup_source
    assert "float(bridge_island_id[cell_index])" in shader
    assert "DIRECT_BRIDGE_AUX_INPUTS || DIRECT_BRIDGE_ENTITY_INPUTS" in shader
    assert "float(bridge_entity_id[cell_index])" in shader
    assert "texelFetch(displaced_in_tex, gid, 0).x" in shader
    assert "DIRECT_BRIDGE_AUX_INPUTS" in shader
