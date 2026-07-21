from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

# GPUMotionResources is imported at runtime (not just TYPE_CHECKING) because
# _ensure_resources constructs it. It is defined in the facade before the
# bucket import block, so the partial-init resolves this without a cycle.
from oracle_game.sim.gpu_motion import (
    GPUMotionResources,
    MAX_MATERIALS,
)


def release(pipeline) -> None:
    if pipeline.resources is None:
        return
    for resource in (
        pipeline.resources.cell_state_tex,
        pipeline.resources.cell_state_out_tex,
        pipeline.resources.velocity_tex,
        pipeline.resources.velocity_out_tex,
        pipeline.resources.temp_tex,
        pipeline.resources.temp_out_tex,
        pipeline.resources.timer_tex,
        pipeline.resources.timer_out_tex,
        pipeline.resources.integrity_tex,
        pipeline.resources.integrity_out_tex,
        pipeline.resources.flow_tex,
        pipeline.resources.ambient_tex,
        pipeline.resources.island_id_tex,
        pipeline.resources.island_id_out_tex,
        pipeline.resources.entity_id_tex,
        pipeline.resources.entity_id_out_tex,
        pipeline.resources.displaced_tex,
        pipeline.resources.displaced_out_tex,
        pipeline.resources.active_tile_tex,
        pipeline.resources.active_tile_list,
        pipeline.resources.active_tile_count,
        pipeline.resources.active_tile_dispatch_args,
        pipeline.resources.island_materialization_candidate_tile_list,
        pipeline.resources.island_materialization_candidate_tile_count,
        pipeline.resources.island_materialization_candidate_dispatch_args,
        pipeline.resources.powder_apply_tile_flags,
        pipeline.resources.powder_target_tex,
        pipeline.resources.powder_target_winner,
        pipeline.resources.powder_apply_incoming,
        pipeline.resources.powder_apply_outgoing,
        pipeline.resources.powder_apply_epoch,
        pipeline.resources.powder_direct_apply_unsafe,
        pipeline.resources.powder_source_cell_core_snapshot,
        pipeline.resources.powder_source_aux_snapshot,
        pipeline.resources.powder_reservations,
        pipeline.resources.powder_compact_reservations,
        pipeline.resources.powder_reservation_count,
        pipeline.resources.powder_provisional_moving_count,
        pipeline.resources.powder_reservation_dispatch_args,
        pipeline.resources.island_reservations,
        pipeline.resources.island_reservation_count,
        pipeline.resources.island_runtime_dispatch_args,
        pipeline.resources.island_apply_incoming,
        pipeline.resources.island_apply_outgoing,
        pipeline.resources.island_materialization_index,
        pipeline.resources.island_reservation_source_index,
        pipeline.resources.island_ids,
        pipeline.resources.island_bboxes,
        pipeline.resources.island_motion,
        pipeline.resources.island_shift_results,
        pipeline.resources.component_label_ping,
        pipeline.resources.component_label_pong,
        pipeline.resources.component_labels,
        pipeline.resources.component_island_ids,
        pipeline.resources.component_metadata,
        pipeline.resources.component_change_flag,
        pipeline.resources.material_params,
        pipeline.resources.material_contact_params,
        pipeline.resources.material_falling_params,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None


def _ensure_resources(pipeline, world: "WorldEngine") -> GPUMotionResources:
    ctx = world.bridge.ctx
    assert ctx is not None
    signature = (
        world.width,
        world.height,
        world.active.tile_width,
        world.active.tile_height,
        world.gas_width,
        world.gas_height,
        world.gas_cell_size,
    )
    if pipeline.resources is not None and pipeline.resources.signature == signature:
        return pipeline.resources
    pipeline.release()
    cell_state_tex = ctx.texture((world.width, world.height), 1, dtype="u4")
    cell_state_out_tex = ctx.texture((world.width, world.height), 1, dtype="u4")
    velocity_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
    velocity_out_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
    temp_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    temp_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    timer_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
    timer_out_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
    integrity_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    integrity_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    flow_tex = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
    ambient_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
    island_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    island_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    displaced_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    displaced_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
    tile_count = max(1, int(world.active.tile_width * world.active.tile_height))
    active_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
    active_tile_count = ctx.buffer(reserve=4, dynamic=True)
    active_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    island_materialization_candidate_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
    island_materialization_candidate_tile_count = ctx.buffer(reserve=4, dynamic=True)
    island_materialization_candidate_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    powder_apply_tile_flags = ctx.buffer(reserve=max(4, tile_count * 4), dynamic=True)
    powder_target_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
    cell_count = int(world.width * world.height)
    powder_target_winner = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    powder_apply_incoming = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    powder_apply_outgoing = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    powder_apply_epoch = ctx.buffer(reserve=max(8, cell_count * 2 * 4), dynamic=True)
    powder_apply_epoch.write(bytes(max(8, cell_count * 2 * 4)))
    powder_direct_apply_unsafe = ctx.buffer(reserve=4, dynamic=True)
    powder_source_cell_core_snapshot = ctx.buffer(
        reserve=max(4, cell_count * 5 * 4),
        dynamic=True,
    )
    powder_source_aux_snapshot = ctx.buffer(reserve=max(4, cell_count * 3 * 4), dynamic=True)
    powder_reservation_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    island_runtime_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
    island_apply_incoming = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    island_apply_outgoing = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    island_materialization_index = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
    component_label_ping = ctx.texture((world.width, world.height), 1, dtype="f4")
    component_label_pong = ctx.texture((world.width, world.height), 1, dtype="f4")
    for texture in (
        cell_state_tex,
        cell_state_out_tex,
        velocity_tex,
        velocity_out_tex,
        temp_tex,
        temp_out_tex,
        timer_tex,
        timer_out_tex,
        integrity_tex,
        integrity_out_tex,
        flow_tex,
        ambient_tex,
        island_id_tex,
        island_id_out_tex,
        entity_id_tex,
        entity_id_out_tex,
        displaced_tex,
        displaced_out_tex,
        active_tile_tex,
        powder_target_tex,
        component_label_ping,
        component_label_pong,
    ):
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
    pipeline.resources = GPUMotionResources(
        signature=signature,
        cell_state_tex=cell_state_tex,
        cell_state_out_tex=cell_state_out_tex,
        velocity_tex=velocity_tex,
        velocity_out_tex=velocity_out_tex,
        temp_tex=temp_tex,
        temp_out_tex=temp_out_tex,
        timer_tex=timer_tex,
        timer_out_tex=timer_out_tex,
        integrity_tex=integrity_tex,
        integrity_out_tex=integrity_out_tex,
        flow_tex=flow_tex,
        ambient_tex=ambient_tex,
        island_id_tex=island_id_tex,
        island_id_out_tex=island_id_out_tex,
        entity_id_tex=entity_id_tex,
        entity_id_out_tex=entity_id_out_tex,
        displaced_tex=displaced_tex,
        displaced_out_tex=displaced_out_tex,
        active_tile_tex=active_tile_tex,
        active_tile_list=active_tile_list,
        active_tile_count=active_tile_count,
        active_tile_dispatch_args=active_tile_dispatch_args,
        island_materialization_candidate_tile_list=island_materialization_candidate_tile_list,
        island_materialization_candidate_tile_count=island_materialization_candidate_tile_count,
        island_materialization_candidate_dispatch_args=island_materialization_candidate_dispatch_args,
        powder_apply_tile_flags=powder_apply_tile_flags,
        powder_target_tex=powder_target_tex,
        powder_target_winner=powder_target_winner,
        powder_apply_incoming=powder_apply_incoming,
        powder_apply_outgoing=powder_apply_outgoing,
        powder_apply_epoch=powder_apply_epoch,
        powder_direct_apply_unsafe=powder_direct_apply_unsafe,
        powder_source_cell_core_snapshot=powder_source_cell_core_snapshot,
        powder_source_aux_snapshot=powder_source_aux_snapshot,
        powder_reservations=ctx.buffer(reserve=4, dynamic=True),
        powder_compact_reservations=ctx.buffer(reserve=4, dynamic=True),
        powder_reservation_count=ctx.buffer(reserve=4, dynamic=True),
        powder_provisional_moving_count=ctx.buffer(reserve=4, dynamic=True),
        powder_reservation_dispatch_args=powder_reservation_dispatch_args,
        island_reservations=ctx.buffer(reserve=4, dynamic=True),
        island_reservation_count=ctx.buffer(reserve=4, dynamic=True),
        island_runtime_dispatch_args=island_runtime_dispatch_args,
        island_apply_incoming=island_apply_incoming,
        island_apply_outgoing=island_apply_outgoing,
        island_materialization_index=island_materialization_index,
        island_reservation_source_index=ctx.buffer(reserve=4, dynamic=True),
        island_ids=ctx.buffer(reserve=4, dynamic=True),
        island_bboxes=ctx.buffer(reserve=4, dynamic=True),
        island_motion=ctx.buffer(reserve=4, dynamic=True),
        island_shift_results=ctx.buffer(reserve=4, dynamic=True),
        component_label_ping=component_label_ping,
        component_label_pong=component_label_pong,
        component_labels=ctx.buffer(reserve=4, dynamic=True),
        component_island_ids=ctx.buffer(reserve=4, dynamic=True),
        component_metadata=ctx.buffer(reserve=world.width * world.height * 5 * 4, dynamic=True),
        component_change_flag=ctx.buffer(reserve=4, dynamic=True),
        material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        material_contact_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        material_falling_params=ctx.buffer(reserve=MAX_MATERIALS * 2 * 4 * 4, dynamic=True),
    )
    return pipeline.resources


def _write_dynamic_buffer(pipeline, ctx: Any, resources: GPUMotionResources, name: str, data: np.ndarray) -> None:
    buffer = getattr(resources, name)
    nbytes = max(4, int(data.nbytes))
    if buffer.size < nbytes:
        buffer.release()
        buffer = ctx.buffer(reserve=nbytes, dynamic=True)
        setattr(resources, name, buffer)
    else:
        buffer.orphan(nbytes)
    if data.nbytes > 0:
        buffer.write(np.ascontiguousarray(data).tobytes())


def _ensure_dynamic_buffer_capacity(pipeline, ctx: Any, resources: GPUMotionResources, name: str, nbytes: int) -> None:
    buffer = getattr(resources, name)
    required = max(4, int(nbytes))
    if buffer.size < required:
        buffer.release()
        setattr(resources, name, ctx.buffer(reserve=required, dynamic=True))
