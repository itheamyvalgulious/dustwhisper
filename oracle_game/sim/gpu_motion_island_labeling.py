from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE
)
from oracle_game.sim.gpu_motion_bridge import _pack_cell_state_texture


def label_falling_island_components(
    pipeline,
    world: "WorldEngine",
    island_id: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    labels, _metadata = pipeline.label_falling_island_component_metadata(world, island_id, bbox)
    return labels


def label_falling_island_component_metadata(
    pipeline,
    world: "WorldEngine",
    island_id: int,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    label_texture, metadata = pipeline.label_falling_island_component_metadata_texture(world, island_id, bbox)
    x0, y0, x1, y1 = bbox
    labels = np.rint(
        np.frombuffer(label_texture.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    return labels[max(0, y0):min(world.height, y1), max(0, x0):min(world.width, x1)].copy(), metadata


def label_falling_island_component_metadata_texture(
    pipeline,
    world: "WorldEngine",
    island_id: int,
    bbox: tuple[int, int, int, int],
) -> tuple[Any, np.ndarray]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["cell_core"]:
        resources.cell_state_tex.write(
            _pack_cell_state_texture(world.material_id, world.phase, world.cell_flags).tobytes()
        )
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y)

    init_program = pipeline.programs["island_component_init"]
    init_program["cell_grid_size"].value = (world.width, world.height)
    init_program["target_island_id"].value = int(island_id)
    init_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    resources.cell_state_tex.use(location=0)
    resources.island_id_tex.use(location=2)
    resources.component_label_ping.bind_to_image(3, read=False, write=True)
    init_program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    current = resources.component_label_ping
    scratch = resources.component_label_pong
    propagate = pipeline.programs["island_component_propagate"]
    propagate["cell_grid_size"].value = (world.width, world.height)
    if pipeline._formal_gpu_frame(world):
        x0, y0, x1, y1 = bbox
        clipped_width = max(0, min(world.width, int(x1)) - max(0, int(x0)))
        clipped_height = max(0, min(world.height, int(y1)) - max(0, int(y0)))
        pass_count = max(1, clipped_width + clipped_height)
        resources.component_change_flag.bind_to_storage_buffer(binding=0)
        for _ in range(pass_count):
            current.use(location=0)
            scratch.bind_to_image(1, read=False, write=True)
            propagate.run(group_x, group_y, 1)
            ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
            current, scratch = scratch, current
    else:
        while True:
            resources.component_change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
            current.use(location=0)
            scratch.bind_to_image(1, read=False, write=True)
            resources.component_change_flag.bind_to_storage_buffer(binding=0)
            propagate.run(group_x, group_y, 1)
            ctx.finish()
            changed = bool(np.frombuffer(resources.component_change_flag.read(size=4), dtype=np.uint32, count=1)[0])
            current, scratch = scratch, current
            if not changed:
                break

    metadata = pipeline._summarize_falling_island_label_texture(world, current)
    return current, metadata


def _summarize_falling_island_label_texture(pipeline, world: "WorldEngine", label_texture: Any) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    resources = pipeline._ensure_resources(world)
    cell_count = int(world.width * world.height)
    metadata = np.zeros((cell_count, 5), dtype=np.int32)
    metadata[:, 0] = int(world.width)
    metadata[:, 1] = int(world.height)
    pipeline._write_dynamic_buffer(ctx, resources, "component_metadata", metadata)
    program = pipeline.programs["summarize_falling_island_components"]
    program["cell_grid_size"].value = (world.width, world.height)
    label_texture.use(location=0)
    resources.component_metadata.bind_to_storage_buffer(binding=0)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    ctx.finish()
    summarized = np.frombuffer(
        resources.component_metadata.read(size=metadata.nbytes),
        dtype=np.int32,
        count=metadata.size,
    ).reshape((cell_count, 5))
    active_indices = np.flatnonzero(summarized[:, 4] > 0)
    if active_indices.size == 0:
        return np.zeros((0, 6), dtype=np.int32)
    labeled_metadata = np.zeros((int(active_indices.size), 6), dtype=np.int32)
    labeled_metadata[:, 0] = active_indices.astype(np.int32, copy=False) + 1
    labeled_metadata[:, 1:] = summarized[active_indices]
    return labeled_metadata


def relabel_falling_island_components(
    pipeline,
    world: "WorldEngine",
    labels: np.ndarray,
    component_labels: np.ndarray,
    component_island_ids: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or labels.size == 0 or component_labels.size == 0:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    full_labels = np.zeros((world.height, world.width), dtype=np.float32)
    x0, y0, x1, y1 = bbox
    clipped_x0 = max(0, int(x0))
    clipped_y0 = max(0, int(y0))
    clipped_x1 = min(world.width, int(x1))
    clipped_y1 = min(world.height, int(y1))
    if clipped_x0 >= clipped_x1 or clipped_y0 >= clipped_y1:
        return False
    label_height = clipped_y1 - clipped_y0
    label_width = clipped_x1 - clipped_x0
    full_labels[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = labels[:label_height, :label_width].astype(
        np.float32,
        copy=False,
    )
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
    resources.component_label_ping.write(full_labels.tobytes())
    pipeline._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
    pipeline._write_dynamic_buffer(
        ctx,
        resources,
        "component_island_ids",
        component_island_ids.astype(np.int32, copy=False),
    )
    program = pipeline.programs["relabel_falling_island_components"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["component_count"].value = int(component_labels.size)
    resources.island_id_tex.use(location=0)
    resources.component_label_ping.use(location=1)
    resources.island_id_out_tex.bind_to_image(2, read=False, write=True)
    resources.component_labels.bind_to_storage_buffer(binding=0)
    resources.component_island_ids.bind_to_storage_buffer(binding=1)
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if not pipeline.last_cpu_mirror_downloaded:
        pipeline._publish_bridge_island_id(world, resources, resources.island_id_out_tex)
        return True
    ctx.finish()
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    return True


def relabel_falling_island_component_texture(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    component_labels: np.ndarray,
    component_island_ids: np.ndarray,
) -> bool:
    ctx = world.bridge.ctx
    if ctx is None or component_labels.size == 0:
        return False
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
    pipeline._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
    pipeline._write_dynamic_buffer(
        ctx,
        resources,
        "component_island_ids",
        component_island_ids.astype(np.int32, copy=False),
    )
    program = pipeline.programs["relabel_falling_island_components"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["component_count"].value = int(component_labels.size)
    resources.island_id_tex.use(location=0)
    label_texture.use(location=1)
    resources.island_id_out_tex.bind_to_image(2, read=False, write=True)
    resources.component_labels.bind_to_storage_buffer(binding=0)
    resources.component_island_ids.bind_to_storage_buffer(binding=1)
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if not pipeline.last_cpu_mirror_downloaded:
        pipeline._publish_bridge_island_id(world, resources, resources.island_id_out_tex)
        return True
    ctx.finish()
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    return True
