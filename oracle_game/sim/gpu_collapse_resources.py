from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    GPUCollapseResources,
)



def _ensure_resources(pipeline, ctx: Any, width: int, height: int) -> GPUCollapseResources:
    signature = (width, height)
    if pipeline.resources is not None and pipeline.resources.signature == signature:
        return pipeline.resources
    pipeline.release()
    structural_tex = ctx.texture((width, height), 1, dtype="f4")
    support_ping = ctx.texture((width, height), 1, dtype="f4")
    support_pong = ctx.texture((width, height), 1, dtype="f4")
    material_tex = ctx.texture((width, height), 1, dtype="f4")
    material_out_tex = ctx.texture((width, height), 1, dtype="f4")
    phase_tex = ctx.texture((width, height), 1, dtype="f4")
    phase_out_tex = ctx.texture((width, height), 1, dtype="f4")
    pending_tex = ctx.texture((width, height), 1, dtype="f4")
    cell_flags_tex = ctx.texture((width, height), 1, dtype="f4")
    cell_flags_out_tex = ctx.texture((width, height), 1, dtype="f4")
    timer_tex = ctx.texture((width, height), 4, dtype="f4")
    timer_out_tex = ctx.texture((width, height), 4, dtype="f4")
    integrity_tex = ctx.texture((width, height), 1, dtype="f4")
    integrity_out_tex = ctx.texture((width, height), 1, dtype="f4")
    temp_tex = ctx.texture((width, height), 1, dtype="f4")
    temp_out_tex = ctx.texture((width, height), 1, dtype="f4")
    island_id_tex = ctx.texture((width, height), 1, dtype="f4")
    island_id_out_tex = ctx.texture((width, height), 1, dtype="f4")
    entity_id_tex = ctx.texture((width, height), 1, dtype="f4")
    entity_id_out_tex = ctx.texture((width, height), 1, dtype="f4")
    displaced_tex = ctx.texture((width, height), 1, dtype="f4")
    displaced_out_tex = ctx.texture((width, height), 1, dtype="f4")
    for texture in (
        structural_tex,
        support_ping,
        support_pong,
        material_tex,
        material_out_tex,
        phase_tex,
        phase_out_tex,
        pending_tex,
        cell_flags_tex,
        cell_flags_out_tex,
        timer_tex,
        timer_out_tex,
        integrity_tex,
        integrity_out_tex,
        temp_tex,
        temp_out_tex,
        island_id_tex,
        island_id_out_tex,
        entity_id_tex,
        entity_id_out_tex,
        displaced_tex,
        displaced_out_tex,
    ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
    cell_count = max(1, width * height)
    axis_tile_width = max(1, (int(width) + FORMAL_CONNECTED_TILE_LOCAL_SIZE - 1) // FORMAL_CONNECTED_TILE_LOCAL_SIZE)
    axis_tile_height = max(1, (int(height) + FORMAL_CONNECTED_TILE_LOCAL_SIZE - 1) // FORMAL_CONNECTED_TILE_LOCAL_SIZE)
    axis_mask_bytes = max(
        4,
        axis_tile_width
        * axis_tile_height
        * FORMAL_CONNECTED_TILE_LOCAL_SIZE
        * np.dtype(np.uint32).itemsize,
    )
    pipeline.resources = GPUCollapseResources(
        signature=signature,
        structural_tex=structural_tex,
        support_ping=support_ping,
        support_pong=support_pong,
        support_u8_ping=None,
        support_u8_pong=None,
        material_tex=material_tex,
        material_out_tex=material_out_tex,
        phase_tex=phase_tex,
        phase_out_tex=phase_out_tex,
        pending_tex=pending_tex,
        cell_flags_tex=cell_flags_tex,
        cell_flags_out_tex=cell_flags_out_tex,
        timer_tex=timer_tex,
        timer_out_tex=timer_out_tex,
        integrity_tex=integrity_tex,
        integrity_out_tex=integrity_out_tex,
        temp_tex=temp_tex,
        temp_out_tex=temp_out_tex,
        island_id_tex=island_id_tex,
        island_id_out_tex=island_id_out_tex,
        entity_id_tex=entity_id_tex,
        entity_id_out_tex=entity_id_out_tex,
        displaced_tex=displaced_tex,
        displaced_out_tex=displaced_out_tex,
        change_flag=ctx.buffer(reserve=4, dynamic=True),
        component_labels=ctx.buffer(reserve=cell_count * 4, dynamic=True),
        component_island_ids=ctx.buffer(reserve=cell_count * 4, dynamic=True),
        component_metadata=ctx.buffer(reserve=cell_count * 5 * 4, dynamic=True),
        component_flags=ctx.buffer(reserve=cell_count * 4, dynamic=True),
        component_invalid=ctx.buffer(reserve=cell_count * 4, dynamic=True),
        component_count=ctx.buffer(reserve=4, dynamic=True),
        component_dispatch_args=ctx.buffer(reserve=3 * 4, dynamic=True),
        region_flags=ctx.buffer(reserve=4, dynamic=True),
        support_tile_union_roots=None,
        support_tile_union_parent=None,
        support_tile_union_seeded=None,
        support_tile_union_edges=None,
        support_tile_union_edge_count=None,
        connected_tile_row_masks=ctx.buffer(reserve=axis_mask_bytes, dynamic=True),
        connected_tile_column_masks=ctx.buffer(reserve=axis_mask_bytes, dynamic=True),
        material_structural=ctx.buffer(reserve=4, dynamic=True),
        material_support_anchor=ctx.buffer(reserve=4, dynamic=True),
        material_collapse_behavior=ctx.buffer(reserve=4, dynamic=True),
        material_collapse_generation=ctx.buffer(reserve=4, dynamic=True),
        material_base_integrity=ctx.buffer(reserve=4, dynamic=True),
        material_spawn_temperature=ctx.buffer(reserve=4, dynamic=True),
    )
    return pipeline.resources


def _ensure_formal_connected_u8_support_textures(
    pipeline,
    ctx: Any,
    resources: GPUCollapseResources,
) -> tuple[Any, Any]:
    if resources.support_u8_ping is None:
        resources.support_u8_ping = ctx.texture(resources.signature, 1, dtype="u1")
        resources.support_u8_ping.filter = (ctx.NEAREST, ctx.NEAREST)
    if resources.support_u8_pong is None:
        resources.support_u8_pong = ctx.texture(resources.signature, 1, dtype="u1")
        resources.support_u8_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    return resources.support_u8_ping, resources.support_u8_pong


def _write_dynamic_buffer(pipeline, ctx: Any, resources: GPUCollapseResources, name: str, data: np.ndarray) -> None:
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


def _materialize_material_params(pipeline, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None:
        collapse_generation = material_table["collapse_generation_id"].astype(np.int32, copy=True)
        base_integrity = material_table["base_integrity"].astype(np.float32, copy=True)
        spawn_temperature = material_table["spawn_temperature"].astype(np.float32, copy=True)
        valid = material_table["name_hash"] != 0
        collapse_generation[~valid] = 0
        base_integrity[~valid] = 0.0
        spawn_temperature[~valid] = np.nan
    else:
        count = max(
            1,
            int(world.material_collapse_generation_id.shape[0]),
            int(world.material_base_integrity.shape[0]),
            int(world.material_spawn_temperature.shape[0]),
        )
        collapse_generation = np.zeros(count, dtype=np.int32)
        base_integrity = np.zeros(count, dtype=np.float32)
        spawn_temperature = np.full(count, np.nan, dtype=np.float32)
        collapse_generation[: world.material_collapse_generation_id.shape[0]] = world.material_collapse_generation_id
        base_integrity[: world.material_base_integrity.shape[0]] = world.material_base_integrity
        spawn_temperature[: world.material_spawn_temperature.shape[0]] = world.material_spawn_temperature
    spawn_temperature = np.nan_to_num(
        spawn_temperature,
        nan=np.float32(-3.4028234663852886e38),
        neginf=np.float32(-3.4028234663852886e38),
    ).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(collapse_generation, dtype=np.int32),
        np.ascontiguousarray(base_integrity, dtype=np.float32),
        np.ascontiguousarray(spawn_temperature, dtype=np.float32),
    )


def _classification_material_params(pipeline, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None:
        structural = material_table["is_structural"].astype(np.int32, copy=True)
        support_anchor = material_table["is_support_anchor"].astype(np.int32, copy=True)
        behavior = material_table["collapse_behavior_id"].astype(np.int32, copy=True)
        valid = material_table["name_hash"] != 0
        structural[~valid] = 0
        support_anchor[~valid] = 0
        behavior[~valid] = 0
    else:
        count = max(
            1,
            int(world.material_is_structural.shape[0]),
            int(world.material_is_support_anchor.shape[0]),
            int(world.material_collapse_behavior.shape[0]),
        )
        structural = np.zeros(count, dtype=np.int32)
        support_anchor = np.zeros(count, dtype=np.int32)
        behavior = np.zeros(count, dtype=np.int32)
        structural[: world.material_is_structural.shape[0]] = world.material_is_structural.astype(np.int32)
        support_anchor[: world.material_is_support_anchor.shape[0]] = world.material_is_support_anchor.astype(np.int32)
        behavior[: world.material_collapse_behavior.shape[0]] = world.material_collapse_behavior.astype(np.int32)
    return (
        np.ascontiguousarray(structural, dtype=np.int32),
        np.ascontiguousarray(support_anchor, dtype=np.int32),
        np.ascontiguousarray(behavior, dtype=np.int32),
    )
