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
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
    program.run_indirect(resources.active_tile_dispatch_args)
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
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
    resources.affected_tile_count.bind_to_storage_buffer(binding=4)
    resources.affected_tile_list.bind_to_storage_buffer(binding=5)
    if axis == "x":
        material_tex = resources.material_out
        phase_tex = resources.phase_out
        flags_tex = resources.flags_out
        timer_tex = resources.timer_out
        temp_tex = resources.temp_out
        integrity_tex = resources.integrity_out
        velocity_tex = resources.velocity_out
    else:
        material_tex = resources.material_in
        phase_tex = resources.phase_in
        flags_tex = resources.flags_in
        timer_tex = resources.timer_in
        temp_tex = resources.temp_in
        integrity_tex = resources.integrity_in
        velocity_tex = resources.velocity_in
    material_tex.bind_to_image(0, read=False, write=True)
    phase_tex.bind_to_image(1, read=False, write=True)
    flags_tex.bind_to_image(2, read=False, write=True)
    timer_tex.bind_to_image(3, read=False, write=True)
    temp_tex.bind_to_image(4, read=False, write=True)
    integrity_tex.bind_to_image(5, read=False, write=True)
    velocity_tex.bind_to_image(6, read=False, write=True)
    program.run_indirect(resources.affected_tile_dispatch_args)
    pipeline._sync_compute_writes(bridge.ctx)

    aux_program = pipeline.programs["prefetch_seam_boundary_bridge_aux_inputs"]
    if not hasattr(aux_program, "run_indirect"):
        raise RuntimeError("GPU liquid seam aux prefetch requires ModernGL ComputeShader.run_indirect")
    aux_program["cell_grid_size"].value = (world.width, world.height)
    aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    aux_program["tile_size"].value = int(world.active.tile_size)
    aux_program["seam_axis"].value = 0 if axis == "x" else 1
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=0)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=1)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=2)
    resources.affected_tile_count.bind_to_storage_buffer(binding=3)
    resources.affected_tile_list.bind_to_storage_buffer(binding=4)
    resources.entity_in.bind_to_image(0, read=False, write=True)
    resources.displaced_in.bind_to_image(1, read=False, write=True)
    aux_program.run_indirect(resources.affected_tile_dispatch_args)
    pipeline._sync_compute_writes(bridge.ctx)



def _build_placeholder_dirty_affected_tile_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    material_tex: Any,
    displaced_tex: Any,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
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

    pending_program = pipeline.programs["compact_placeholder_active_pending_affected_tiles"]
    if not hasattr(pending_program, "run_indirect"):
        raise RuntimeError("GPU liquid placeholder pending compaction requires ModernGL ComputeShader.run_indirect")
    material_table = world.bridge.shadow_typed_tables["material_table"]
    pending_program["cell_grid_size"].value = (world.width, world.height)
    pending_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    pending_program["tile_size"].value = int(world.active.tile_size)
    pending_program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
    pending_program["source_workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    pending_program["workgroups_per_tile"].value = int(pipeline._active_tile_workgroups_per_tile(world))
    material_tex.use(location=0)
    displaced_tex.use(location=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
    resources.affected_tile_flags.bind_to_storage_buffer(binding=5)
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
    program = pipeline.programs["tile_solve"]
    ctx = world.bridge.ctx
    assert ctx is not None
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("GPU liquid active tile solve requires ModernGL ComputeShader.run_indirect")
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    resources.material_in.use(location=0)
    resources.phase_in.use(location=1)
    resources.flags_in.use(location=2)
    resources.timer_in.use(location=3)
    resources.temp_in.use(location=4)
    resources.integrity_in.use(location=5)
    resources.velocity_in.use(location=6)
    resources.entity_in.use(location=8)
    resources.displaced_in.use(location=9)
    resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=2)
    resources.active_tile_list.bind_to_storage_buffer(binding=3)
    resources.material_out.bind_to_image(0, read=False, write=True)
    resources.phase_out.bind_to_image(1, read=False, write=True)
    resources.flags_out.bind_to_image(2, read=False, write=True)
    resources.timer_out.bind_to_image(3, read=False, write=True)
    resources.temp_out.bind_to_image(4, read=False, write=True)
    resources.integrity_out.bind_to_image(5, read=False, write=True)
    resources.velocity_out.bind_to_image(6, read=False, write=True)
    program.run_indirect(resources.active_tile_dispatch_args)
    pipeline._sync_compute_writes(ctx)



def _run_seam_pass(
    pipeline,
    program_name: str,
    world: "WorldEngine",
    read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
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
    read_resources[1].use(location=1)
    read_resources[2].use(location=2)
    read_resources[3].use(location=3)
    read_resources[4].use(location=4)
    read_resources[5].use(location=5)
    read_resources[6].use(location=6)
    active_tile_tex.use(location=7)
    pipeline.resources.entity_in.use(location=8)
    pipeline.resources.displaced_in.use(location=9)
    pipeline.resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources = pipeline.resources
    assert resources is not None
    resources.affected_tile_count.bind_to_storage_buffer(binding=2)
    resources.affected_tile_list.bind_to_storage_buffer(binding=3)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(1, read=False, write=True)
    write_resources[2].bind_to_image(2, read=False, write=True)
    write_resources[3].bind_to_image(3, read=False, write=True)
    write_resources[4].bind_to_image(4, read=False, write=True)
    write_resources[5].bind_to_image(5, read=False, write=True)
    write_resources[6].bind_to_image(6, read=False, write=True)
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
    read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
) -> None:
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_powder"].value = int(Phase.POWDER)
    read_resources[0].use(location=0)
    read_resources[1].use(location=1)
    read_resources[2].use(location=2)
    read_resources[3].use(location=3)
    read_resources[4].use(location=4)
    read_resources[5].use(location=5)
    read_resources[6].use(location=6)
    resources.active_tile_tex.use(location=7)
    resources.material_params.bind_to_storage_buffer(binding=0)
    world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=2)
    resources.active_tile_list.bind_to_storage_buffer(binding=3)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(1, read=False, write=True)
    write_resources[2].bind_to_image(2, read=False, write=True)
    write_resources[3].bind_to_image(3, read=False, write=True)
    write_resources[4].bind_to_image(4, read=False, write=True)
    write_resources[5].bind_to_image(5, read=False, write=True)
    write_resources[6].bind_to_image(6, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, program_name)
    pipeline._sync_compute_writes(ctx)



def _run_copy_core_state(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
) -> None:
    program = pipeline.programs["copy_core_state"]
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    read_resources[0].use(location=0)
    read_resources[1].use(location=1)
    read_resources[2].use(location=2)
    read_resources[3].use(location=3)
    read_resources[4].use(location=4)
    read_resources[5].use(location=5)
    read_resources[6].use(location=6)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(1, read=False, write=True)
    write_resources[2].bind_to_image(2, read=False, write=True)
    write_resources[3].bind_to_image(3, read=False, write=True)
    write_resources[4].bind_to_image(4, read=False, write=True)
    write_resources[5].bind_to_image(5, read=False, write=True)
    write_resources[6].bind_to_image(6, read=False, write=True)
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
    read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    displaced_in: Any,
    displaced_out: Any,
) -> None:
    program = pipeline.programs["copy_with_pending"]
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    read_resources[0].use(location=0)
    read_resources[1].use(location=1)
    read_resources[2].use(location=2)
    read_resources[3].use(location=3)
    read_resources[4].use(location=4)
    read_resources[5].use(location=5)
    read_resources[6].use(location=6)
    displaced_in.use(location=7)
    write_resources[0].bind_to_image(0, read=False, write=True)
    write_resources[1].bind_to_image(1, read=False, write=True)
    write_resources[2].bind_to_image(2, read=False, write=True)
    write_resources[3].bind_to_image(3, read=False, write=True)
    write_resources[4].bind_to_image(4, read=False, write=True)
    write_resources[5].bind_to_image(5, read=False, write=True)
    write_resources[6].bind_to_image(6, read=False, write=True)
    displaced_out.bind_to_image(7, read=False, write=True)
    resources.active_tile_count.bind_to_storage_buffer(binding=0)
    resources.active_tile_list.bind_to_storage_buffer(binding=1)
    if active_tile_indirect:
        pipeline._run_active_tile_indirect(program, resources, "copy with pending")
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
    read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    displaced_in: Any,
    displaced_out: Any,
) -> None:
    program = pipeline.programs["placeholder_displace"]
    ctx = world.bridge.ctx
    assert ctx is not None
    dirty_affected_dispatch = pipeline._formal_gpu_frame(world)
    if dirty_affected_dispatch:
        pipeline._build_placeholder_dirty_affected_tile_dispatch(
            world,
            resources,
            material_tex=read_resources[0],
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
    read_resources[1].use(location=1)
    read_resources[2].use(location=2)
    read_resources[3].use(location=3)
    read_resources[4].use(location=4)
    read_resources[5].use(location=5)
    read_resources[6].use(location=6)
    resources.active_tile_tex.use(location=7)
    displaced_in.use(location=8)
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
    write_resources[1].bind_to_image(1)
    write_resources[2].bind_to_image(2)
    write_resources[3].bind_to_image(3)
    write_resources[4].bind_to_image(4)
    write_resources[5].bind_to_image(5)
    write_resources[6].bind_to_image(6)
    displaced_out.bind_to_image(7)
    if dirty_affected_dispatch:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid placeholder displacement requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.affected_tile_dispatch_args)
    else:
        pipeline._run_active_tile_indirect(program, resources, "placeholder displacement")
    pipeline._sync_compute_writes(ctx)



def _run_cleanup_runtime(pipeline, world: "WorldEngine", resources: GPULiquidResources) -> None:
    program = pipeline.programs["cleanup_runtime"]
    ctx = world.bridge.ctx
    assert ctx is not None
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    resources.material_pre.use(location=0)
    resources.phase_pre.use(location=1)
    resources.material_in.use(location=2)
    resources.phase_in.use(location=3)
    resources.island_in.use(location=4)
    resources.entity_in.use(location=5)
    resources.displaced_out.use(location=6)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    resources.island_out.bind_to_image(0, read=False, write=True)
    resources.entity_out.bind_to_image(1, read=False, write=True)
    resources.displaced_in.bind_to_image(2, read=False, write=True)
    if active_tile_indirect:
        pipeline._run_active_tile_indirect(program, resources, "runtime cleanup")
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(ctx)
