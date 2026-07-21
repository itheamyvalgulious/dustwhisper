from __future__ import annotations

import inspect

from oracle_game.sim.gpu_liquid import GPULiquidPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_liquid_seam_x_row_leader_is_default_enabled() -> None:
    pipeline = GPULiquidPipeline()
    assert pipeline._seam_x_row_leader_enabled is True
    assert pipeline._seam_x_multirow_rows == 4
    assert pipeline._seam_x_multirow_frame_rows == 0


def test_liquid_seam_x_row_leader_specialization_is_single_warp_per_row() -> None:
    source = shader_source(
        "liquid/seam_x.comp",
        {
            **_SHADER_SUBS,
            "SEAM_SNAPSHOT_INPUT": 1,
            "SEAM_ROW_LEADER": 1,
        },
    )
    assert "layout(local_size_x=32, local_size_y=1" in source
    assert "shared int row_source_base;" in source
    assert "gl_WorkGroupID.x % uint(TILE_SIZE)" in source
    assert "left_to_right_move(boundary_x, y" in source
    assert "right_to_left_move(boundary_x, y" in source
    assert "apply_row_leader_move(left_cell);" in source
    assert "apply_row_leader_move(right_cell);" in source


def test_liquid_seam_x_multirow_specialization_packs_four_independent_row_warps() -> None:
    rows_per_group = 4
    source = shader_source(
        "liquid/seam_x.comp",
        {
            **_SHADER_SUBS,
            "SEAM_SNAPSHOT_INPUT": 1,
            "SEAM_ROW_LEADER": 1,
            "SEAM_ROWS_PER_GROUP": rows_per_group,
        },
    )
    assert f"#define SEAM_ROWS_PER_GROUP {rows_per_group}" in source
    assert f"layout(local_size_x=32, local_size_y={rows_per_group}" in source
    assert "int row_slot = int(gl_LocalInvocationID.y);" in source
    assert "uint boundary_index = gl_WorkGroupID.x;" in source
    assert "int row_group = int(gl_WorkGroupID.y);" in source
    assert "#define ROW_MOVE_DIRECTION(slot) row_move_direction[slot]" in source
    assert "ROW_MOVE_DIRECTION(row_slot)" in source
    assert "bool row_valid = row < TILE_SIZE && y < cell_grid_size.y;" in source
    assert (
        "#if SEAM_ROWS_PER_GROUP > 1\n"
        "    bool row_valid = row < TILE_SIZE && y < cell_grid_size.y;\n"
        "#else\n"
        "    if (y >= cell_grid_size.y)"
    ) in source

    pipeline = GPULiquidPipeline()
    pipeline._seam_x_multirow_frame_rows = rows_per_group
    assert pipeline._seam_workgroups_per_boundary("x") == 8
    assert pipeline._seam_workgroups_per_boundary("x", canonical=True) == 32


def test_liquid_step_keeps_control_fallback() -> None:
    source = inspect.getsource(GPULiquidPipeline.step)
    assert "_seam_x_multirow_frame_rows" in source
    assert "requested_seam_rows == 4" in source
    assert '"seam_x_snapshot_row_leader"' in source
    assert 'seam_x_program = "seam_x_snapshot"' in source


def test_liquid_seam_x_multirow_keeps_prefetch_on_canonical_flat_dispatch() -> None:
    build_source = inspect.getsource(GPULiquidPipeline._build_seam_boundary_dispatch)
    prefetch_source = inspect.getsource(GPULiquidPipeline._prefetch_seam_boundary_bridge_inputs)
    compact_source = shader_source(
        "liquid/compact_seam_x_boundaries_from_active_tiles.comp",
        _SHADER_SUBS,
    )
    assert 'program["boundary_row_groups"]' in build_source
    assert "canonical=True" in build_source
    assert "affected_tile_prefetch_dispatch_args" in prefetch_source
    assert "atomicMax(affected_tile_dispatch_args[0], slot + 1u);" in compact_source
    assert "atomicMax(affected_tile_dispatch_args[1], boundary_row_groups);" in compact_source
