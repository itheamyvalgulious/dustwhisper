from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_liquid import (
    MAX_MATERIALS,
    PASS_LOCAL_SIZE,
    TILE_SIZE,
    GPULiquidResources,
)


def release(pipeline) -> None:
    if pipeline.resources is None:
        return
    try:
        pipeline.resources.bridge_cell_copy_framebuffer.release()
    except Exception:
        pass
    for resource in (
        pipeline.resources.cell_state_pre,
        pipeline.resources.cell_state_in,
        pipeline.resources.cell_state_out,
        pipeline.resources.island_in,
        pipeline.resources.island_out,
        pipeline.resources.entity_in,
        pipeline.resources.entity_out,
        pipeline.resources.timer_in,
        pipeline.resources.timer_out,
        pipeline.resources.temp_in,
        pipeline.resources.temp_out,
        pipeline.resources.integrity_in,
        pipeline.resources.integrity_out,
        pipeline.resources.velocity_in,
        pipeline.resources.velocity_out,
        pipeline.resources.blocker_mask,
        pipeline.resources.liquid_flow_intent,
        pipeline.resources.active_tile_tex,
        pipeline.resources.active_tile_list,
        pipeline.resources.active_tile_count,
        pipeline.resources.active_tile_dispatch_args,
        pipeline.resources.affected_tile_list,
        pipeline.resources.affected_tile_count,
        pipeline.resources.affected_tile_dispatch_args,
        pipeline.resources.affected_tile_prefetch_dispatch_args,
        pipeline.resources.affected_tile_flags,
        pipeline.resources.tile_solve_snapshot,
        pipeline.resources.tile_snapshot_token,
        pipeline.resources.tile_snapshot_tile_tokens,
        pipeline.resources.provenance_in,
        pipeline.resources.provenance_out,
        pipeline.resources.placeholder_target_claims,
        pipeline.resources.displaced_in,
        pipeline.resources.displaced_out,
        pipeline.resources.material_params,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None



def _ensure_resources(pipeline, world: "WorldEngine") -> GPULiquidResources:
    ctx = world.bridge.ctx
    assert ctx is not None
    signature = (world.width, world.height, world.active.tile_width, world.active.tile_height)
    if pipeline.resources is not None and pipeline.resources.signature == signature:
        return pipeline.resources
    pipeline.release()
    cell_state_pre = ctx.texture((world.width, world.height), 1, dtype="u4")
    cell_state_in = ctx.texture((world.width, world.height), 1, dtype="u4")
    cell_state_out = ctx.texture((world.width, world.height), 1, dtype="u4")
    island_in = ctx.texture((world.width, world.height), 1, dtype="f4")
    island_out = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
    timer_in = ctx.texture((world.width, world.height), 1, dtype="u4")
    timer_out = ctx.texture((world.width, world.height), 1, dtype="u4")
    temp_in = ctx.texture((world.width, world.height), 1, dtype="f4")
    temp_out = ctx.texture((world.width, world.height), 1, dtype="f4")
    integrity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
    integrity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
    velocity_in = ctx.texture((world.width, world.height), 2, dtype="f4")
    velocity_out = ctx.texture((world.width, world.height), 2, dtype="f4")
    blocker_mask = ctx.texture((world.width, world.height), 1, dtype="u1")
    blocker_mask.write(np.zeros((world.height, world.width), dtype=np.uint8).tobytes())
    liquid_flow_intent = ctx.texture((world.width, world.height), 2, dtype="f4")
    active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
    tile_count = max(1, int(world.active.tile_width * world.active.tile_height))
    active_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
    active_tile_count = ctx.buffer(reserve=4, dynamic=True)
    active_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    affected_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
    affected_tile_count = ctx.buffer(reserve=4, dynamic=True)
    affected_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    affected_tile_prefetch_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    affected_tile_flags = ctx.buffer(reserve=max(4, tile_count * 4), dynamic=True)
    affected_tile_flags.write(np.zeros((tile_count,), dtype=np.uint32).tobytes())
    cell_count = max(1, int(world.width * world.height))
    snapshot_words = 2 if pipeline._compact_tile_solve_snapshot_enabled else 6
    tile_solve_snapshot = ctx.buffer(reserve=cell_count * snapshot_words * 4, dynamic=True)
    tile_snapshot_token = ctx.buffer(reserve=5 * 4, dynamic=True)
    tile_snapshot_token.write(np.zeros((5,), dtype=np.uint32).tobytes())
    tile_snapshot_tile_tokens = ctx.buffer(reserve=max(8, tile_count * 8), dynamic=True)
    tile_snapshot_tile_tokens.write(np.zeros((tile_count, 2), dtype=np.uint32).tobytes())
    provenance_in = ctx.buffer(reserve=cell_count * 4, dynamic=True)
    provenance_out = ctx.buffer(reserve=cell_count * 4, dynamic=True)
    placeholder_target_claims = ctx.buffer(reserve=cell_count * 4, dynamic=True)
    placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
    displaced_in = ctx.texture((world.width, world.height), 1, dtype="f4")
    displaced_out = ctx.texture((world.width, world.height), 1, dtype="f4")
    for texture in (
        cell_state_pre,
        cell_state_in,
        cell_state_out,
        island_in,
        island_out,
        entity_in,
        entity_out,
        timer_in,
        timer_out,
        temp_in,
        temp_out,
        integrity_in,
        integrity_out,
        velocity_in,
        velocity_out,
        blocker_mask,
        liquid_flow_intent,
        active_tile_tex,
        displaced_in,
        displaced_out,
    ):
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
    bridge_cell_copy_framebuffer = ctx.framebuffer(
        color_attachments=[
            cell_state_pre,
            cell_state_out,
            timer_out,
            temp_out,
            integrity_out,
        ]
    )
    pipeline.resources = GPULiquidResources(
        signature=signature,
        cell_state_pre=cell_state_pre,
        cell_state_in=cell_state_in,
        cell_state_out=cell_state_out,
        island_in=island_in,
        island_out=island_out,
        entity_in=entity_in,
        entity_out=entity_out,
        timer_in=timer_in,
        timer_out=timer_out,
        temp_in=temp_in,
        temp_out=temp_out,
        integrity_in=integrity_in,
        integrity_out=integrity_out,
        velocity_in=velocity_in,
        velocity_out=velocity_out,
        blocker_mask=blocker_mask,
        liquid_flow_intent=liquid_flow_intent,
        active_tile_tex=active_tile_tex,
        active_tile_list=active_tile_list,
        active_tile_count=active_tile_count,
        active_tile_dispatch_args=active_tile_dispatch_args,
        affected_tile_list=affected_tile_list,
        affected_tile_count=affected_tile_count,
        affected_tile_dispatch_args=affected_tile_dispatch_args,
        affected_tile_prefetch_dispatch_args=affected_tile_prefetch_dispatch_args,
        affected_tile_flags=affected_tile_flags,
        tile_solve_snapshot=tile_solve_snapshot,
        tile_snapshot_token=tile_snapshot_token,
        tile_snapshot_tile_tokens=tile_snapshot_tile_tokens,
        provenance_in=provenance_in,
        provenance_out=provenance_out,
        placeholder_target_claims=placeholder_target_claims,
        displaced_in=displaced_in,
        displaced_out=displaced_out,
        bridge_cell_copy_framebuffer=bridge_cell_copy_framebuffer,
        material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
    )
    return pipeline.resources



def _active_scheduler_gpu_authoritative(pipeline, world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return (
        pipeline._formal_gpu_frame(world)
        and "active_tile_ttl" in authoritative
        and "active_chunk_mask" in authoritative
    )



def _refresh_active_scheduler_from_ttl(pipeline, world: "WorldEngine") -> None:
    bridge = world.bridge
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for active refresh")
    bridge._ensure_active_scheduler_programs()
    bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")



def _upload_active_tile_mask(pipeline, resources: GPULiquidResources, tile_mask: np.ndarray) -> None:
    resources.active_tile_tex.write(np.asarray(tile_mask, dtype="f4").tobytes())



def _load_authoritative_active_tile_mask(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    expansion_radius: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge active scheduler resources")
    program = pipeline.programs["load_active_tiles"]
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["expansion_radius"].value = int(expansion_radius)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
    resources.active_tile_tex.bind_to_image(1, read=False, write=True)
    program.run(
        (world.active.tile_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
        (world.active.tile_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(bridge.ctx)



def _active_tile_workgroups_per_tile(pipeline, world: "WorldEngine") -> int:
    axis = max(1, (int(world.active.tile_size) + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
    return axis * axis



def _next_placeholder_claim_epoch(pipeline, resources: GPULiquidResources, world: "WorldEngine") -> int:
    pipeline._placeholder_claim_epoch += 1
    if pipeline._placeholder_claim_epoch >= 0x7FFFFFFF:
        cell_count = max(1, int(world.width * world.height))
        resources.placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
        pipeline._placeholder_claim_epoch = 1
    return pipeline._placeholder_claim_epoch



def _seam_workgroups_per_boundary(pipeline, axis: str, *, canonical: bool = False) -> int:
    if axis == "x":
        rows_per_group = int(pipeline._seam_x_multirow_frame_rows)
        if not canonical and rows_per_group == 4:
            return max(1, (TILE_SIZE + rows_per_group - 1) // rows_per_group)
        groups_x = max(1, (TILE_SIZE * 2 + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        groups_y = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
    elif axis == "y":
        if (
            pipeline._seam_y_shared_snapshot_enabled
            and pipeline._buoyancy_pass_fusion_enabled
        ):
            return 1
        groups_x = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        groups_y = 1
    else:
        raise ValueError(f"unknown liquid seam axis: {axis}")
    return groups_x * groups_y



def _reload_and_compact_active_cell_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
) -> None:
    pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
    pipeline._compact_active_tiles(
        world,
        resources,
        workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
    )



def _run_active_tile_indirect(pipeline, program: Any, resources: GPULiquidResources, pass_name: str) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError(f"GPU liquid {pass_name} requires ModernGL ComputeShader.run_indirect")
    program.run_indirect(resources.active_tile_dispatch_args)
