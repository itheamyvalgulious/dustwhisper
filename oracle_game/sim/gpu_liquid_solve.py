from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_liquid import GPULiquidResources

from oracle_game.gpu import typed_material_id
from oracle_game.types import Phase

from oracle_game.sim.gpu_liquid import (
    PASS_LOCAL_SIZE,
)


def _build_seam_boundary_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    axis: str,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_program = pipeline.programs["clear_active_tile_dispatch"]
    resources.affected_tile_count.bind_to_storage_buffer(binding=0)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    program = pipeline.programs[f"compact_seam_{axis}_boundaries_from_active_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU liquid seam {axis} boundary compaction requires ModernGL ComputeShader.run_indirect")
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["source_workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    program["workgroups_per_boundary"].value = int(pipeline._seam_workgroups_per_boundary(axis))
    if axis == "x":
        program["boundary_row_groups"].value = int(
            pipeline._seam_workgroups_per_boundary(axis)
            if pipeline._seam_x_multirow_frame_rows == 4
            else 0
        )
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
    program.run_indirect(resources.active_tile_dispatch_args)
    pipeline._sync_compute_writes(ctx)
    seam_y_shared = bool(
        axis == "y"
        and pipeline._seam_y_shared_snapshot_enabled
        and pipeline._buoyancy_pass_fusion_enabled
    )
    multirow_seam_x = bool(
        axis == "x" and pipeline._seam_x_multirow_frame_rows == 4
    )
    if pipeline._seam_prefetch_zero_full_active_enabled or multirow_seam_x:
        prefetch_groups = (
            max(
                1,
                (int(world.active.tile_size) + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            )
            if seam_y_shared
            else pipeline._seam_workgroups_per_boundary(axis, canonical=True)
        )
        retarget_program = pipeline.programs["retarget_seam_prefetch_dispatch"]
        retarget_program["workgroups_per_boundary"].value = int(prefetch_groups)
        retarget_program["total_tile_count"].value = int(
            world.active.tile_width * world.active.tile_height
        )
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.affected_tile_prefetch_dispatch_args.bind_to_storage_buffer(binding=2)
        retarget_program.run(1, 1, 1)
        pipeline._sync_compute_writes(ctx)
    elif seam_y_shared:
        prefetch_groups = max(
            1,
            (int(world.active.tile_size) + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
        )
        retarget_program = pipeline.programs["retarget_active_tile_dispatch"]
        retarget_program["workgroups_per_tile"].value = prefetch_groups
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_prefetch_dispatch_args.bind_to_storage_buffer(binding=1)
        retarget_program.run(1, 1, 1)
        pipeline._sync_compute_writes(ctx)



def _prefetch_seam_boundary_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    axis: str,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    if axis not in ("x", "y"):
        raise ValueError(f"unknown liquid seam axis: {axis}")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid seam prefetch requires bridge GPU resources")
    world._require_gpu_authoritative_resources(
        "liquid seam boundary prefetch",
        "cell_core",
        "entity_id",
        "placeholder_displaced_material",
        "active_tile_ttl",
    )
    program = pipeline.programs["prefetch_seam_boundary_bridge_inputs"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("GPU liquid seam prefetch requires ModernGL ComputeShader.run_indirect")
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["seam_axis"].value = 0 if axis == "x" else 1
    program["total_tile_count"].value = int(world.active.tile_width * world.active.tile_height)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.affected_tile_count.bind_to_storage_buffer(binding=4)
    resources.affected_tile_list.bind_to_storage_buffer(binding=5)
    resources.active_tile_count.bind_to_storage_buffer(binding=6)
    if axis == "x":
        cell_state_tex = resources.cell_state_out
        timer_tex = resources.timer_out
        temp_tex = resources.temp_out
        integrity_tex = resources.integrity_out
        velocity_tex = resources.velocity_out
    else:
        cell_state_tex = resources.cell_state_in
        timer_tex = resources.timer_in
        temp_tex = resources.temp_in
        integrity_tex = resources.integrity_in
        velocity_tex = resources.velocity_in
    cell_state_tex.bind_to_image(0, read=False, write=True)
    timer_tex.bind_to_image(3, read=False, write=True)
    temp_tex.bind_to_image(4, read=False, write=True)
    integrity_tex.bind_to_image(5, read=False, write=True)
    velocity_tex.bind_to_image(6, read=False, write=True)
    prefetch_dispatch_args = (
        resources.affected_tile_prefetch_dispatch_args
        if pipeline._seam_prefetch_zero_full_active_enabled
        or (axis == "x" and pipeline._seam_x_multirow_frame_rows == 4)
        or (
            axis == "y"
            and pipeline._seam_y_shared_snapshot_enabled
            and pipeline._buoyancy_pass_fusion_enabled
        )
        else resources.affected_tile_dispatch_args
    )
    program.run_indirect(prefetch_dispatch_args)
    pipeline._sync_compute_writes(bridge.ctx)

    if pipeline._bridge_aux_residency_enabled:
        return
    aux_program = pipeline.programs["prefetch_seam_boundary_bridge_aux_inputs"]
    if not hasattr(aux_program, "run_indirect"):
        raise RuntimeError("GPU liquid seam aux prefetch requires ModernGL ComputeShader.run_indirect")
    aux_program["cell_grid_size"].value = (world.width, world.height)
    aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    aux_program["tile_size"].value = int(world.active.tile_size)
    aux_program["seam_axis"].value = 0 if axis == "x" else 1
    aux_program["total_tile_count"].value = int(world.active.tile_width * world.active.tile_height)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=0)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=1)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=2)
    resources.affected_tile_count.bind_to_storage_buffer(binding=3)
    resources.affected_tile_list.bind_to_storage_buffer(binding=4)
    resources.active_tile_count.bind_to_storage_buffer(binding=5)
    resources.entity_in.bind_to_image(0, read=False, write=True)
    resources.displaced_in.bind_to_image(1, read=False, write=True)
    aux_program.run_indirect(prefetch_dispatch_args)
    pipeline._sync_compute_writes(bridge.ctx)



def _build_placeholder_dirty_affected_tile_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    cell_state_tex: Any,
    displaced_tex: Any,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_flags = pipeline.programs["clear_affected_tile_flags"]
    tile_count = int(world.active.tile_width * world.active.tile_height)
    clear_flags["tile_count"].value = tile_count
    resources.affected_tile_flags.bind_to_storage_buffer(binding=0)
    clear_flags.run((tile_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)

    clear_program = pipeline.programs["clear_active_tile_dispatch"]
    resources.affected_tile_count.bind_to_storage_buffer(binding=0)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    dirty_rect_count = int(len(world.bridge_frame_placeholder_dirty_rects))
    if dirty_rect_count > 0:
        world.bridge.ensure_world_resources(world)
        if "placeholder_dirty_rect" not in world.bridge.buffers:
            raise RuntimeError("GPU liquid placeholder displacement requires placeholder dirty rect buffer")
        program = pipeline.programs["compact_placeholder_dirty_affected_tiles"]
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["dirty_rect_count"].value = dirty_rect_count
        program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_list.bind_to_storage_buffer(binding=1)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        resources.affected_tile_flags.bind_to_storage_buffer(binding=3)
        world.bridge.buffers["placeholder_dirty_rect"].bind_to_storage_buffer(binding=4)
        program.run((dirty_rect_count + 63) // 64, 1, 1)
        pipeline._sync_compute_writes(ctx)

    direct_bridge_aux_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    pending_program = pipeline.programs[
        "compact_placeholder_active_pending_affected_tiles_bridge_aux"
        if direct_bridge_aux_inputs
        else "compact_placeholder_active_pending_affected_tiles"
    ]
    if not hasattr(pending_program, "run_indirect"):
        raise RuntimeError("GPU liquid placeholder pending compaction requires ModernGL ComputeShader.run_indirect")
    material_table = world.bridge.shadow_typed_tables["material_table"]
    pending_program["cell_grid_size"].value = (world.width, world.height)
    pending_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    pending_program["tile_size"].value = int(world.active.tile_size)
    pending_program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
    pending_program["source_workgroups_per_tile"].value = int(
        pipeline._active_tile_workgroups_per_tile(world)
    )
    pending_program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    cell_state_tex.use(location=0)
    displaced_tex.use(location=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
    resources.affected_tile_flags.bind_to_storage_buffer(binding=5)
    if direct_bridge_aux_inputs:
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
    pending_program.run_indirect(resources.active_tile_dispatch_args)
    pipeline._sync_compute_writes(ctx)



def _compact_active_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    workgroups_per_tile: int = 1,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    clear_program = pipeline.programs["clear_active_tile_dispatch"]
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
    clear_program.run(1, 1, 1)
    pipeline._sync_compute_writes(ctx)

    compact_workgroups_per_tile = int(max(1, workgroups_per_tile))
    if pipeline._active_scheduler_gpu_authoritative(world):
        bridge = world.bridge
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        compact_program = pipeline.programs["compact_active_tiles_from_chunks"]
        if not hasattr(compact_program, "run_indirect"):
            raise RuntimeError("GPU liquid active chunk compaction requires ModernGL ComputeShader.run_indirect")
        compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
        compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
        bridge.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
        compact_program.run_indirect(bridge.buffers["active_chunk_dispatch_args"])
    else:
        tile_count = int(world.active.tile_width * world.active.tile_height)
        compact_program = pipeline.programs["compact_active_tiles"]
        compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
        resources.active_tile_tex.use(location=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        compact_program.run((tile_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)



def _run_tile_solve(pipeline, world: "WorldEngine", resources: GPULiquidResources) -> None:
    direct_bridge_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and "cell_core" in world.bridge.gpu_authoritative_resources
    )
    direct_bridge_aux_inputs = bool(
        direct_bridge_inputs
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    snapshot_output = bool(
        direct_bridge_inputs
        and pipeline._tile_solve_snapshot_output_fusion_enabled
        and pipeline._active_scheduler_gpu_authoritative(world)
        and {
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    provenance_output = bool(
        snapshot_output
        and pipeline._provenance_terminal_frame_enabled
    )
    blocker_mask_input = bool(
        provenance_output
        and pipeline._blocker_displaced_hydration_frame_enabled
    )
    row_stream = bool(
        snapshot_output
        and pipeline.tile_solve_warp_fast_path
        and pipeline._tile_warp_provenance_row_stream_enabled
    )
    if row_stream:
        program_name = (
            "tile_solve_bridge_snapshot_row_stream_provenance_blocker"
            if blocker_mask_input
            else "tile_solve_bridge_snapshot_row_stream_provenance_kind_cache_aux"
            if provenance_output
            and direct_bridge_aux_inputs
            and pipeline._tile_solve_liquid_kind_cache_enabled
            else "tile_solve_bridge_snapshot_row_stream_provenance_kind_cache"
            if provenance_output
            and pipeline._tile_solve_liquid_kind_cache_enabled
            else "tile_solve_bridge_snapshot_row_stream_provenance_aux"
            if provenance_output and direct_bridge_aux_inputs
            else "tile_solve_bridge_snapshot_row_stream_provenance"
            if provenance_output
            else "tile_solve_bridge_snapshot_row_stream_aux"
            if direct_bridge_aux_inputs
            else "tile_solve_bridge_snapshot_row_stream"
        )
    elif snapshot_output:
        program_name = (
            "tile_solve_bridge_snapshot_provenance_blocker"
            if blocker_mask_input
            else "tile_solve_bridge_snapshot_provenance_aux"
            if provenance_output and direct_bridge_aux_inputs
            else "tile_solve_bridge_snapshot_provenance"
            if provenance_output
            else "tile_solve_bridge_snapshot_aux"
            if direct_bridge_aux_inputs
            else "tile_solve_bridge_snapshot"
        )
    else:
        if direct_bridge_aux_inputs:
            program_name = "tile_solve_bridge_aux"
        else:
            program_name = "tile_solve_bridge" if direct_bridge_inputs else "tile_solve"
    snapshot_pre_state = bool(pipeline._buoyancy_snapshot_pre_state_frame_enabled)
    if snapshot_pre_state:
        candidate_names = {
            "tile_solve_bridge_snapshot_provenance",
            "tile_solve_bridge_snapshot_provenance_aux",
            "tile_solve_bridge_snapshot_provenance_blocker",
            "tile_solve_bridge_snapshot_row_stream_provenance",
            "tile_solve_bridge_snapshot_row_stream_provenance_aux",
            "tile_solve_bridge_snapshot_row_stream_provenance_blocker",
        }
        if program_name not in candidate_names:
            raise RuntimeError(f"liquid snapshot pre-state unsupported tile program: {program_name}")
        program_name += "_snapshot_pre"
        if pipeline._tile_snapshot_state_elision_frame_enabled:
            program_name += "_state_elided"
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("GPU liquid active tile solve requires ModernGL ComputeShader.run_indirect")
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    resources.cell_state_in.use(location=0)
    resources.timer_in.use(location=3)
    resources.temp_in.use(location=4)
    resources.integrity_in.use(location=5)
    resources.velocity_in.use(location=6)
    resources.entity_in.use(location=8)
    resources.displaced_in.use(location=9)
    resources.blocker_mask.use(location=10)
    resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=2)
    resources.active_tile_list.bind_to_storage_buffer(binding=3)
    if direct_bridge_inputs:
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=4)
    if direct_bridge_aux_inputs:
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=7)
    if snapshot_output:
        resources.tile_solve_snapshot.bind_to_storage_buffer(binding=5)
    if provenance_output:
        resources.provenance_in.bind_to_storage_buffer(binding=8)
    if snapshot_pre_state:
        token_program = pipeline.programs["advance_tile_snapshot_token"]
        resources.tile_snapshot_token.bind_to_storage_buffer(binding=0)
        token_program.run(1, 1, 1)
        pipeline._sync_compute_writes(ctx)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.tile_snapshot_token.bind_to_storage_buffer(binding=9)
        resources.tile_snapshot_tile_tokens.bind_to_storage_buffer(binding=10)
    cell_state_out = resources.cell_state_in if snapshot_output else resources.cell_state_out
    timer_out = resources.timer_in if snapshot_output else resources.timer_out
    temp_out = resources.temp_in if snapshot_output else resources.temp_out
    integrity_out = resources.integrity_in if snapshot_output else resources.integrity_out
    velocity_out = resources.velocity_in if snapshot_output else resources.velocity_out
    cell_state_out.bind_to_image(0, read=False, write=True)
    timer_out.bind_to_image(3, read=False, write=True)
    temp_out.bind_to_image(4, read=False, write=True)
    integrity_out.bind_to_image(5, read=False, write=True)
    velocity_out.bind_to_image(6, read=False, write=True)
    program.run_indirect(resources.active_tile_dispatch_args)
    pipeline._sync_compute_writes(ctx)


def _run_provenance_init(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    skip_when_all_tiles_active: bool = False,
) -> None:
    program = pipeline.programs["init_liquid_provenance"]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    resources.provenance_in.bind_to_storage_buffer(binding=0)
    groups_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    groups_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    if skip_when_all_tiles_active:
        retarget = pipeline.programs["retarget_provenance_init_dispatch"]
        retarget["full_grid_groups"].value = (groups_x, groups_y)
        retarget["total_tile_count"].value = int(
            world.active.tile_width * world.active.tile_height
        )
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_prefetch_dispatch_args.bind_to_storage_buffer(binding=1)
        retarget.run(1, 1, 1)
        pipeline._sync_compute_writes(ctx)
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid provenance init requires indirect dispatch")
        resources.provenance_in.bind_to_storage_buffer(binding=0)
        program.run_indirect(resources.affected_tile_prefetch_dispatch_args)
    else:
        program.run(groups_x, groups_y, 1)
    pipeline._sync_compute_writes(ctx)



def _run_seam_pass(
    pipeline,
    program_name: str,
    world: "WorldEngine",
    read_resources: tuple[Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any],
    active_tile_tex: Any,
    *,
    boundary_dispatch: bool = False,
) -> None:
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["use_boundary_dispatch"].value = bool(boundary_dispatch)
    read_resources[0].use(location=0)
    read_resources[1].use(location=3)
    read_resources[2].use(location=4)
    read_resources[3].use(location=5)
    read_resources[4].use(location=6)
    active_tile_tex.use(location=7)
    pipeline.resources.entity_in.use(location=8)
    pipeline.resources.displaced_in.use(location=9)
    pipeline.resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources = pipeline.resources
    assert resources is not None
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    if program_name.startswith("seam_x_snapshot"):
        resources.tile_solve_snapshot.bind_to_storage_buffer(binding=4)
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=5)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=7)
        if "_provenance" in program_name:
            resources.provenance_in.bind_to_storage_buffer(binding=8)
    elif program_name in {
        "seam_x_bridge_aux",
        "seam_y_bridge_aux",
        "seam_y_shared_snapshot_aux",
        "seam_y_shared_snapshot_provenance_aux",
    }:
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=4)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=5)
    if program_name.startswith("seam_y_shared_snapshot") and "_provenance" in program_name:
        resources.provenance_in.bind_to_storage_buffer(binding=8)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(3, read=False, write=True)
    write_resources[2].bind_to_image(4, read=False, write=True)
    write_resources[3].bind_to_image(5, read=False, write=True)
    write_resources[4].bind_to_image(6, read=False, write=True)
    if boundary_dispatch:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU liquid {program_name} requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.affected_tile_dispatch_args)
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(ctx)



def _run_buoyancy_pass(
    pipeline,
    program_name: str,
    world: "WorldEngine",
    resources: GPULiquidResources,
    read_resources: tuple[Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any],
    *,
    sink_fallback_resources: tuple[Any, Any, Any, Any, Any] | None = None,
) -> None:
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    snapshot_pre_state = program_name.startswith(
        "buoyancy_fused_provenance_cleanup_snapshot_pre"
    )
    if snapshot_pre_state:
        validation_program = pipeline.programs["validate_tile_snapshot_coverage"]
        validation_program["tile_grid_size"].value = (
            world.active.tile_width,
            world.active.tile_height,
        )
        resources.tile_snapshot_token.bind_to_storage_buffer(binding=0)
        resources.tile_snapshot_tile_tokens.bind_to_storage_buffer(binding=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
        validation_program.run(1, 1, 1)
        pipeline._sync_compute_writes(ctx)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_powder"].value = int(Phase.POWDER)
    cleanup_fused = program_name.startswith("buoyancy_fused_provenance_cleanup")
    if cleanup_fused:
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    read_resources[0].use(location=0)
    read_resources[1].use(location=3)
    read_resources[2].use(location=4)
    read_resources[3].use(location=5)
    read_resources[4].use(location=6)
    resources.active_tile_tex.use(location=7)
    if program_name.startswith("buoyancy_fused"):
        fallback_resources = (
            write_resources
            if sink_fallback_resources is None
            else sink_fallback_resources
        )
        # Sink only writes active cells. Float can sample an inactive neighbor,
        # so preserve the old sink destination as the exact fallback value.
        fallback_resources[0].use(location=8)
        fallback_resources[1].use(location=9)
        fallback_resources[2].use(location=10)
        fallback_resources[3].use(location=11)
        fallback_resources[4].use(location=12)
    resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=2)
    resources.active_tile_list.bind_to_storage_buffer(binding=3)
    if program_name.startswith("buoyancy_fused_provenance"):
        resources.provenance_in.bind_to_storage_buffer(binding=8)
        resources.provenance_out.bind_to_storage_buffer(binding=9)
    if cleanup_fused:
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=4)
        world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=5)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=7)
        if snapshot_pre_state:
            resources.tile_solve_snapshot.bind_to_storage_buffer(binding=10)
            resources.tile_snapshot_token.bind_to_storage_buffer(binding=11)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(3, read=False, write=True)
    write_resources[2].bind_to_image(4, read=False, write=True)
    write_resources[3].bind_to_image(5, read=False, write=True)
    write_resources[4].bind_to_image(6, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, program_name)
    pipeline._sync_compute_writes(ctx)



def _run_copy_core_state(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    read_resources: tuple[Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any],
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program = pipeline.programs["copy_core_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    read_resources[0].use(location=0)
    read_resources[1].use(location=3)
    read_resources[2].use(location=4)
    read_resources[3].use(location=5)
    read_resources[4].use(location=6)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(3, read=False, write=True)
    write_resources[2].bind_to_image(4, read=False, write=True)
    write_resources[3].bind_to_image(5, read=False, write=True)
    write_resources[4].bind_to_image(6, read=False, write=True)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    if active_tile_indirect:
        pipeline._run_active_tile_indirect(program, resources, "copy core state")
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(ctx)



def _run_copy_for_placeholder(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    read_resources: tuple[Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any],
    displaced_in: Any,
    displaced_out: Any,
    *,
    affected_tile_dispatch: bool = False,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    direct_bridge_aux_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    provenance_output = bool(
        active_tile_indirect
        and pipeline._provenance_terminal_frame_enabled
    )
    program = pipeline.programs[
        "copy_with_pending_provenance_bridge_aux"
        if provenance_output and direct_bridge_aux_inputs
        else "copy_with_pending_provenance"
        if provenance_output
        else "copy_with_pending_bridge_aux"
        if direct_bridge_aux_inputs
        else "copy_with_pending"
    ]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    program["restrict_to_active_tiles"].value = bool(affected_tile_dispatch)
    read_resources[0].use(location=0)
    read_resources[1].use(location=3)
    read_resources[2].use(location=4)
    read_resources[3].use(location=5)
    read_resources[4].use(location=6)
    displaced_in.use(location=7)
    resources.active_tile_tex.use(location=8)
    if direct_bridge_aux_inputs:
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
    if provenance_output:
        resources.provenance_in.bind_to_storage_buffer(binding=8)
        resources.provenance_out.bind_to_storage_buffer(binding=9)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(3, read=False, write=True)
    write_resources[2].bind_to_image(4, read=False, write=True)
    write_resources[3].bind_to_image(5, read=False, write=True)
    write_resources[4].bind_to_image(6, read=False, write=True)
    displaced_out.bind_to_image(7, read=False, write=True)
    tile_count_buffer = resources.affected_tile_count if affected_tile_dispatch else resources.active_tile_count
    tile_list_buffer = resources.affected_tile_list if affected_tile_dispatch else resources.active_tile_list
    dispatch_args = (
        resources.affected_tile_dispatch_args
        if affected_tile_dispatch
        else resources.active_tile_dispatch_args
    )
    tile_count_buffer.bind_to_storage_buffer(binding=0)
    tile_list_buffer.bind_to_storage_buffer(binding=1)
    if active_tile_indirect:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid copy with pending requires indirect dispatch")
        program.run_indirect(dispatch_args)
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(ctx)



def _run_placeholder_displacement(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    read_resources: tuple[Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any],
    displaced_in: Any,
    displaced_out: Any,
    *,
    affected_dispatch_prepared: bool = False,
) -> None:
    direct_bridge_aux_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    provenance_output = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._provenance_terminal_frame_enabled
    )
    program = pipeline.programs[
        "placeholder_displace_provenance_bridge_aux"
        if provenance_output and direct_bridge_aux_inputs
        else "placeholder_displace_provenance"
        if provenance_output
        else "placeholder_displace_bridge_aux"
        if direct_bridge_aux_inputs
        else "placeholder_displace"
    ]
    ctx = world.bridge.ctx
    assert ctx is not None
    dirty_affected_dispatch = pipeline._formal_gpu_frame(world)
    if dirty_affected_dispatch and not affected_dispatch_prepared:
        pipeline._build_placeholder_dirty_affected_tile_dispatch(
            world,
            resources,
            cell_state_tex=read_resources[0],
            displaced_tex=displaced_in,
        )
    material_table = world.bridge.shadow_typed_tables["material_table"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
    program["placeholder_claim_epoch"].value = int(pipeline._next_placeholder_claim_epoch(resources, world))
    read_resources[0].use(location=0)
    read_resources[1].use(location=3)
    read_resources[2].use(location=4)
    read_resources[3].use(location=5)
    read_resources[4].use(location=6)
    resources.active_tile_tex.use(location=7)
    displaced_in.use(location=8)
    if direct_bridge_aux_inputs:
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
    if provenance_output:
        resources.provenance_in.bind_to_storage_buffer(binding=8)
        resources.provenance_out.bind_to_storage_buffer(binding=9)
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=10)
    resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    if dirty_affected_dispatch:
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    else:
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
    resources.affected_tile_flags.bind_to_storage_buffer(binding=4)
    resources.placeholder_target_claims.bind_to_storage_buffer(binding=5)
    write_resources[0].bind_to_image(0)
    write_resources[1].bind_to_image(3)
    write_resources[2].bind_to_image(4)
    write_resources[3].bind_to_image(5)
    write_resources[4].bind_to_image(6)
    displaced_out.bind_to_image(7)
    if dirty_affected_dispatch:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid placeholder displacement requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.affected_tile_dispatch_args)
    else:
        pipeline._run_active_tile_indirect(program, resources, "placeholder displacement")
    pipeline._sync_compute_writes(ctx)



def _run_cleanup_runtime(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    affected_tile_dispatch: bool = False,
    restore_bridge_aux_values: bool = False,
) -> None:
    direct_bridge_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and "cell_core" in world.bridge.gpu_authoritative_resources
    )
    direct_bridge_aux_outputs = bool(
        direct_bridge_inputs and pipeline._bridge_aux_cleanup_fusion_enabled
    )
    direct_bridge_aux_inputs = bool(
        direct_bridge_inputs
        and pipeline._bridge_aux_residency_enabled
        and {"island_id", "entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    direct_bridge_island_input = bool(
        direct_bridge_inputs
        and pipeline._cleanup_bridge_island_residency_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and not direct_bridge_aux_inputs
        and "island_id" in world.bridge.gpu_authoritative_resources
    )
    program_name = (
        "cleanup_runtime_bridge_aux_resident"
        if direct_bridge_aux_inputs and direct_bridge_aux_outputs
        else "cleanup_runtime_bridge_aux_island"
        if direct_bridge_island_input and direct_bridge_aux_outputs
        else "cleanup_runtime_bridge_aux"
        if direct_bridge_aux_outputs
        else "cleanup_runtime_bridge" if direct_bridge_inputs else "cleanup_runtime"
    )
    if restore_bridge_aux_values:
        if not (affected_tile_dispatch and direct_bridge_aux_outputs):
            raise RuntimeError("liquid aux restore requires affected bridge-output cleanup")
        program_name = (
            "cleanup_runtime_bridge_aux_restore_bridge_ids16"
            if pipeline._buoyancy_blocker_displaced_hydration_frame_enabled
            else "cleanup_runtime_bridge_aux_restore16"
        )
    cleanup_local16 = bool(
        (pipeline._cleanup_local16_enabled or restore_bridge_aux_values)
        and program_name in {
            "cleanup_runtime_bridge_aux",
            "cleanup_runtime_bridge_aux_island",
            "cleanup_runtime_bridge_aux_restore16",
            "cleanup_runtime_bridge_aux_restore_bridge_ids16",
        }
        and pipeline._formal_gpu_frame(world)
    )
    if cleanup_local16 and not restore_bridge_aux_values:
        program_name = (
            "cleanup_runtime_bridge_aux_island16"
            if program_name == "cleanup_runtime_bridge_aux_island"
            else "cleanup_runtime_bridge_aux16"
        )
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    program["restrict_to_active_tiles"].value = bool(affected_tile_dispatch)
    split_placeholder_roles = bool(
        active_tile_indirect and pipeline._placeholder_lazy_roles_enabled
    )
    program["use_split_placeholder_roles"].value = split_placeholder_roles
    resources.cell_state_pre.use(location=0)
    resources.cell_state_in.use(location=2)
    resources.cell_state_out.use(location=3)
    resources.island_in.use(location=4)
    resources.entity_in.use(location=5)
    resources.displaced_out.use(location=6)
    resources.active_tile_tex.use(location=7)
    resources.material_params.bind_to_storage_buffer(binding=0)
    tile_count_buffer = (
        resources.affected_tile_count
        if affected_tile_dispatch
        else resources.active_tile_count
    )
    tile_list_buffer = (
        resources.affected_tile_list
        if affected_tile_dispatch
        else resources.active_tile_list
    )
    dispatch_args = (
        resources.affected_tile_dispatch_args
        if affected_tile_dispatch
        else resources.active_tile_dispatch_args
    )
    tile_count_buffer.bind_to_storage_buffer(binding=1)
    tile_list_buffer.bind_to_storage_buffer(binding=2)
    if direct_bridge_inputs:
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=3)
    resources.affected_tile_flags.bind_to_storage_buffer(binding=4)
    if direct_bridge_aux_outputs:
        world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=5)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=7)
        if direct_bridge_island_input:
            resources.island_in.bind_to_image(3, read=False, write=True)
        else:
            resources.island_in.bind_to_image(0, read=False, write=True)
        resources.entity_in.bind_to_image(1, read=False, write=True)
    if direct_bridge_aux_inputs and not direct_bridge_aux_outputs:
        world.bridge.buffers["island_id"].bind_to_storage_buffer(binding=5)
        world.bridge.buffers["entity_id"].bind_to_storage_buffer(binding=6)
        world.bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=7)
    else:
        resources.island_out.bind_to_image(0, read=False, write=True)
        resources.entity_out.bind_to_image(1, read=False, write=True)
    if split_placeholder_roles:
        resources.displaced_in.bind_to_image(2)
    else:
        resources.displaced_in.bind_to_image(2, read=False, write=True)
    if active_tile_indirect:
        if cleanup_local16:
            retarget_program = pipeline.programs["retarget_active_tile_dispatch"]
            retarget_program["workgroups_per_tile"].value = 4
            tile_count_buffer.bind_to_storage_buffer(binding=0)
            dispatch_args.bind_to_storage_buffer(binding=1)
            retarget_program.run(1, 1, 1)
            pipeline._sync_compute_writes(ctx)
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid runtime cleanup requires indirect dispatch")
        program.run_indirect(dispatch_args)
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(ctx)
