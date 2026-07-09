from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE
from oracle_game.types import Phase

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    LOCAL_SIZE
)



def label_component_mask(
    pipeline,
    world: "WorldEngine",
    component_mask: np.ndarray,
    *,
    x0: int = 0,
    y0: int = 0,
) -> np.ndarray:
    label_texture, width, height = pipeline._label_component_mask_texture(world, component_mask, x0=x0, y0=y0)
    if width == 0 or height == 0:
        return np.zeros_like(component_mask, dtype=np.int32)
    return np.rint(np.frombuffer(label_texture.read(), dtype="f4").reshape((height, width))).astype(np.int32)


def materialize_component_mask(
    pipeline,
    world: "WorldEngine",
    component_mask: np.ndarray,
    x0: int,
    y0: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_texture, width, height = pipeline._label_component_mask_texture(world, component_mask, x0=x0, y0=y0)
    if width == 0 or height == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty.copy(), np.zeros((0, 5), dtype=np.int32)
    component_labels = pipeline.collect_component_labels(world, label_texture, width, height)
    if component_labels.size == 0:
        return component_labels, np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.int32)
    component_island_ids = np.asarray(
        [world.allocate_island_id() for _ in range(int(component_labels.size))],
        dtype=np.int32,
    )
    component_metadata = pipeline.summarize_labeled_component_texture(
        world,
        label_texture,
        component_labels,
        x0,
        y0,
        width,
        height,
    )
    pipeline.materialize_labeled_component_texture(
        world,
        label_texture,
        component_labels,
        component_island_ids,
        x0,
        y0,
        width,
        height,
    )
    return component_labels, component_island_ids, component_metadata


def materialize_component_texture(
    pipeline,
    world: "WorldEngine",
    component_texture: Any,
    width: int,
    height: int,
    x0: int,
    y0: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width == 0 or height == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty.copy(), np.zeros((0, 5), dtype=np.int32)
    label_texture, width, height = pipeline._label_component_texture(
        world,
        component_texture,
        width,
        height,
        x0=x0,
        y0=y0,
    )
    component_labels = pipeline.collect_component_labels(world, label_texture, width, height)
    if component_labels.size == 0:
        return component_labels, np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.int32)
    component_island_ids = np.asarray(
        [world.allocate_island_id() for _ in range(int(component_labels.size))],
        dtype=np.int32,
    )
    component_metadata = pipeline.summarize_labeled_component_texture(
        world,
        label_texture,
        component_labels,
        x0,
        y0,
        width,
        height,
    )
    pipeline.materialize_labeled_component_texture(
        world,
        label_texture,
        component_labels,
        component_island_ids,
        x0,
        y0,
        width,
        height,
    )
    return component_labels, component_island_ids, component_metadata


def materialize_component_texture_formal(
    pipeline,
    world: "WorldEngine",
    component_texture: Any,
    width: int,
    height: int,
    x0: int,
    y0: int,
    *,
    tile_mask_name: str | None = None,
) -> int:
    if not pipeline._formal_gpu_frame(world):
        raise RuntimeError("formal component texture materialization requires an active formal GPU frame")
    if width == 0 or height == 0:
        return 0
    with pipeline._profile_pass(world, "label_collect_components"):
        with pipeline._profile_pass(world, "label_collect_components.label"):
            label_texture, width, height = pipeline._label_component_texture(
                world,
                component_texture,
                width,
                height,
                x0=x0,
                y0=y0,
                tile_mask_name=tile_mask_name,
            )
        component_capacity = pipeline._prepare_formal_component_list_and_metadata(
            world,
            label_texture,
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name,
        )
    if component_capacity == 0:
        return 0
    island_id_base = pipeline._reserve_formal_component_island_ids(world, component_capacity)
    with pipeline._profile_pass(world, "materialize"):
        pipeline._materialize_compact_labeled_component_texture(
            world,
            label_texture,
            island_id_base,
            component_capacity,
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name,
        )
    with pipeline._profile_pass(world, "publish_runtime"):
        pipeline._publish_compact_component_island_runtime(
            world,
            island_id_base,
            component_capacity,
            x0,
            y0,
            width,
            height,
        )
    return component_capacity


def _reserve_formal_component_island_ids(pipeline, world: "WorldEngine", component_capacity: int) -> int:
    component_capacity = max(0, int(component_capacity))
    if component_capacity == 0:
        return 0
    next_id = max(1, int(getattr(world, "next_island_id", 1)))
    max_existing = max((int(island_id) for island_id in getattr(world, "islands", {})), default=0)
    island_id_base = max(next_id, max_existing + 1)
    world.next_island_id = max(next_id, island_id_base + component_capacity)
    return island_id_base


def _ensure_component_work_buffers(
    pipeline,
    ctx: Any,
    resources: GPUCollapseResources,
    component_capacity: int,
) -> None:
    component_capacity = max(1, int(component_capacity))
    label_bytes = component_capacity * np.dtype(np.int32).itemsize
    flag_bytes = component_capacity * np.dtype(np.uint32).itemsize
    metadata_bytes = component_capacity * 5 * np.dtype(np.int32).itemsize
    if resources.component_labels.size < label_bytes:
        resources.component_labels.release()
        resources.component_labels = ctx.buffer(reserve=label_bytes, dynamic=True)
    else:
        resources.component_labels.orphan(label_bytes)
    if resources.component_flags.size < flag_bytes:
        resources.component_flags.release()
        resources.component_flags = ctx.buffer(reserve=flag_bytes, dynamic=True)
    if resources.component_metadata.size < metadata_bytes:
        resources.component_metadata.release()
        resources.component_metadata = ctx.buffer(reserve=metadata_bytes, dynamic=True)
    else:
        resources.component_metadata.orphan(metadata_bytes)


def _collect_component_labels_gpu(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    width: int,
    height: int,
    *,
    empty_min: tuple[int, int] | None = None,
    tile_mask_name: str | None = None,
) -> int:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0:
        return 0
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    component_capacity = max(1, int(width) * int(height))
    pipeline._ensure_component_work_buffers(ctx, resources, component_capacity)
    empty_min_value = empty_min if empty_min is not None else (int(width), int(height))
    resources.component_count.write(np.zeros(1, dtype=np.uint32).tobytes())

    if pipeline._formal_gpu_frame(world) and tile_mask_name is not None:
        program = pipeline.programs["collect_component_labels_connected_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected component collect requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
        program["component_capacity"].value = int(component_capacity)
        program["empty_min"].value = (int(empty_min_value[0]), int(empty_min_value[1]))
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_labels.bind_to_storage_buffer(binding=1)
        resources.component_count.bind_to_storage_buffer(binding=2)
        resources.component_metadata.bind_to_storage_buffer(binding=3)
        world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
        program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        return component_capacity

    program = pipeline.programs["collect_component_labels"]
    program["region_size"].value = (int(width), int(height))
    program["empty_min"].value = (int(empty_min_value[0]), int(empty_min_value[1]))
    label_texture.use(location=0)
    resources.component_flags.bind_to_storage_buffer(binding=0)
    resources.component_labels.bind_to_storage_buffer(binding=1)
    resources.component_count.bind_to_storage_buffer(binding=2)
    resources.component_metadata.bind_to_storage_buffer(binding=3)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    return component_capacity


def _clear_component_label_flags_connected_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    label_texture: Any,
    width: int,
    height: int,
    component_capacity: int,
    tile_mask_name: str,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    program = pipeline.programs["clear_component_label_flags_connected_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected component flag clear requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
    program["component_capacity"].value = int(component_capacity)
    label_texture.use(location=0)
    resources.component_flags.bind_to_storage_buffer(binding=0)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _build_component_dispatch_args(
    pipeline,
    world: "WorldEngine",
    component_capacity: int,
    *,
    invocations_per_group: int = 256,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    resources = pipeline.resources
    if resources is None:
        raise RuntimeError("GPU collapse component dispatch requires allocated resources")
    program = pipeline.programs["build_component_dispatch_args"]
    program["component_capacity"].value = int(component_capacity)
    program["invocations_per_group"].value = int(invocations_per_group)
    resources.component_count.bind_to_storage_buffer(binding=0)
    resources.component_dispatch_args.bind_to_storage_buffer(binding=1)
    program.run(1, 1, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))


def _prepare_formal_component_list_and_metadata(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    tile_mask_name: str | None = None,
) -> int:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    with pipeline._profile_pass(world, "label_collect_components.collect_roots"):
        component_capacity = pipeline._collect_component_labels_gpu(
            world,
            label_texture,
            width,
            height,
            empty_min=(int(x0 + width), int(y0 + height)),
            tile_mask_name=tile_mask_name,
        )
    if component_capacity == 0:
        return 0
    # Keep the connected-tiles summary path explicit at the prepare stage so
    # formal connected materialization stays routed to
    # summarize_compact_components_connected_tiles instead of the legacy
    # full-grid summarize shader.
    with pipeline._profile_pass(world, "label_collect_components.summarize_metadata"):
        pipeline._summarize_formal_component_metadata(
            world,
            label_texture,
            x0,
            y0,
            width,
            height,
            component_capacity,
            tile_mask_name=tile_mask_name,
        )
    return component_capacity


def _summarize_formal_component_metadata(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    component_capacity: int,
    *,
    tile_mask_name: str | None = None,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    resources = pipeline._ensure_resources(ctx, width, height)

    summarize_program = pipeline.programs["summarize_compact_components"]
    summarize_program["region_size"].value = (int(width), int(height))
    summarize_program["region_origin"].value = (int(x0), int(y0))
    summarize_program["component_capacity"].value = int(component_capacity)
    label_texture.use(location=0)
    resources.component_flags.bind_to_storage_buffer(binding=0)
    resources.component_metadata.bind_to_storage_buffer(binding=1)
    if pipeline._formal_gpu_frame(world) and tile_mask_name is not None:
        summarize_program = pipeline.programs["summarize_compact_components_connected_tiles"]
        summarize_program["cell_grid_size"].value = (int(width), int(height))
        summarize_program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        summarize_program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
        summarize_program["region_origin"].value = (int(x0), int(y0))
        summarize_program["component_capacity"].value = int(component_capacity)
        label_texture.use(location=0)
        resources.component_flags.bind_to_storage_buffer(binding=0)
        resources.component_metadata.bind_to_storage_buffer(binding=1)
        world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
        summarize_program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    else:
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        summarize_program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)


def _materialize_compact_labeled_component_texture(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    island_id_base: int,
    component_capacity: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    tile_mask_name: str | None = None,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0 or component_capacity == 0:
        return
    with pipeline._profile_pass(world, "materialize.main"):
        pipeline._ensure_programs(ctx)
        resources = pipeline._ensure_resources(ctx, width, height)
        if not (pipeline._formal_gpu_frame(world) and tile_mask_name is not None):
            pipeline._upload_region_state(world, resources, x0, y0, width, height)
        collapse_generation, base_integrity, spawn_temperature = pipeline._materialize_material_params(world)
        pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
        pipeline._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
        pipeline._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

        connected_tiles = pipeline._formal_gpu_frame(world) and tile_mask_name is not None
        program = pipeline.programs[
            "materialize_compact_components_connected_tiles" if connected_tiles else "materialize_compact_components"
        ]
        if connected_tiles:
            program["cell_grid_size"].value = (int(width), int(height))
            program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
        else:
            program["region_size"].value = (int(width), int(height))
        program["label_capacity"].value = int(component_capacity)
        program["island_id_base"].value = int(island_id_base)
        program["material_count"].value = int(collapse_generation.size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        label_texture.use(location=0)
        resources.material_tex.use(location=1)
        resources.phase_tex.use(location=2)
        resources.cell_flags_tex.use(location=3)
        resources.timer_tex.use(location=4)
        resources.integrity_tex.use(location=5)
        resources.temp_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.timer_out_tex.bind_to_image(3, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(4, read=False, write=True)
        resources.temp_out_tex.bind_to_image(5, read=False, write=True)
        resources.material_collapse_generation.bind_to_storage_buffer(binding=0)
        resources.material_base_integrity.bind_to_storage_buffer(binding=1)
        resources.material_spawn_temperature.bind_to_storage_buffer(binding=2)
        resources.component_flags.bind_to_storage_buffer(binding=3)
        if connected_tiles:
            assert tile_mask_name is not None
            world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
            program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    with pipeline._profile_pass(world, "materialize.aux"):
        aux_program = pipeline.programs[
            "materialize_compact_components_aux_connected_tiles"
            if connected_tiles
            else "materialize_compact_components_aux"
        ]
        if connected_tiles:
            aux_program["cell_grid_size"].value = (int(width), int(height))
            aux_program["tile_grid_size"].value = (
                int(getattr(world.active, "tile_width", 1)),
                int(getattr(world.active, "tile_height", 1)),
            )
            aux_program["tile_size"].value = int(
                max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
            )
        else:
            aux_program["region_size"].value = (int(width), int(height))
        aux_program["label_capacity"].value = int(component_capacity)
        aux_program["island_id_base"].value = int(island_id_base)
        label_texture.use(location=0)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        resources.component_flags.bind_to_storage_buffer(binding=3)
        if connected_tiles:
            assert tile_mask_name is not None
            world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
            world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
            aux_program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        else:
            aux_program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "materialize.publish_bridge_outputs"):
        if connected_tiles:
            assert tile_mask_name is not None
            pipeline._publish_bridge_region_outputs_connected_tiles(world, resources, x0, y0, width, height, tile_mask_name)
        else:
            pipeline._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
    pipeline.last_cpu_mirror_downloaded = False


def _publish_compact_component_island_runtime(
    pipeline,
    world: "WorldEngine",
    island_id_base: int,
    component_capacity: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    if component_capacity == 0:
        return
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for island runtime")
    required_bytes = max(4, int(component_capacity) * ISLAND_RUNTIME_DTYPE.itemsize)
    bridge_buffer = bridge.buffers["island_runtime"]
    preserve_existing_runtime = "island_runtime" in bridge.gpu_authoritative_resources
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["island_runtime"] = bridge_buffer
        preserve_existing_runtime = False
    elif not preserve_existing_runtime:
        bridge_buffer.orphan(required_bytes)
    if not preserve_existing_runtime:
        bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())

    resources = pipeline._ensure_resources(ctx, width, height)
    pipeline._build_component_dispatch_args(world, component_capacity)
    program = pipeline.programs["publish_compact_component_island_runtime"]
    program["component_capacity"].value = int(component_capacity)
    program["island_id_base"].value = int(island_id_base)
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
    program["paging_buffer_origin"].value = (
        int(world.paging.buffer_origin_x),
        int(world.paging.buffer_origin_y),
    )
    resources.component_metadata.bind_to_storage_buffer(binding=0)
    bridge_buffer.bind_to_storage_buffer(binding=1)
    bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
    resources.component_count.bind_to_storage_buffer(binding=3)
    program.run_indirect(resources.component_dispatch_args)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative("island_runtime")


def _materialize_dense_labeled_component_texture(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    island_id_base: int,
    component_capacity: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0 or component_capacity == 0:
        return
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    pipeline._upload_region_state(world, resources, x0, y0, width, height)
    collapse_generation, base_integrity, spawn_temperature = pipeline._materialize_material_params(world)
    pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
    pipeline._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
    pipeline._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

    program = pipeline.programs["materialize_dense_components"]
    program["region_size"].value = (width, height)
    program["label_capacity"].value = int(component_capacity)
    program["island_id_base"].value = int(island_id_base)
    program["material_count"].value = int(collapse_generation.size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    label_texture.use(location=0)
    resources.material_tex.use(location=1)
    resources.phase_tex.use(location=2)
    resources.cell_flags_tex.use(location=3)
    resources.timer_tex.use(location=4)
    resources.integrity_tex.use(location=5)
    resources.temp_tex.use(location=6)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.material_out_tex.bind_to_image(0, read=False, write=True)
    resources.phase_out_tex.bind_to_image(1, read=False, write=True)
    resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
    resources.timer_out_tex.bind_to_image(3, read=False, write=True)
    resources.integrity_out_tex.bind_to_image(4, read=False, write=True)
    resources.temp_out_tex.bind_to_image(5, read=False, write=True)
    resources.material_collapse_generation.bind_to_storage_buffer(binding=0)
    resources.material_base_integrity.bind_to_storage_buffer(binding=1)
    resources.material_spawn_temperature.bind_to_storage_buffer(binding=2)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)

    aux_program = pipeline.programs["materialize_dense_components_aux"]
    aux_program["region_size"].value = (width, height)
    aux_program["label_capacity"].value = int(component_capacity)
    aux_program["island_id_base"].value = int(island_id_base)
    label_texture.use(location=0)
    resources.island_id_tex.use(location=7)
    resources.entity_id_tex.use(location=8)
    resources.displaced_tex.use(location=9)
    resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
    resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
    resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
    aux_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)
    pipeline._publish_bridge_region_outputs(world, resources, x0, y0, width, height)
    pipeline.last_cpu_mirror_downloaded = False


def _summarize_dense_component_metadata(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    component_capacity: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    init_program = pipeline.programs["init_dense_component_metadata"]
    init_program["component_capacity"].value = int(component_capacity)
    init_program["empty_min"].value = (int(x0 + width), int(y0 + height))
    resources.component_metadata.bind_to_storage_buffer(binding=0)
    init_program.run((int(component_capacity) + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)

    summarize_program = pipeline.programs["summarize_dense_components"]
    summarize_program["region_size"].value = (width, height)
    summarize_program["region_origin"].value = (int(x0), int(y0))
    summarize_program["component_capacity"].value = int(component_capacity)
    label_texture.use(location=0)
    resources.component_metadata.bind_to_storage_buffer(binding=0)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    summarize_program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _publish_dense_component_island_runtime(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    island_id_base: int,
    component_capacity: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    if component_capacity == 0:
        return
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._summarize_dense_component_metadata(world, label_texture, component_capacity, x0, y0, width, height)
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for island runtime")
    required_bytes = max(4, int(component_capacity) * ISLAND_RUNTIME_DTYPE.itemsize)
    bridge_buffer = bridge.buffers["island_runtime"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["island_runtime"] = bridge_buffer
    else:
        bridge_buffer.orphan(required_bytes)
    bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())

    resources = pipeline._ensure_resources(ctx, width, height)
    program = pipeline.programs["publish_dense_component_island_runtime"]
    program["component_capacity"].value = int(component_capacity)
    program["island_id_base"].value = int(island_id_base)
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
    program["paging_buffer_origin"].value = (
        int(world.paging.buffer_origin_x),
        int(world.paging.buffer_origin_y),
    )
    resources.component_metadata.bind_to_storage_buffer(binding=0)
    bridge_buffer.bind_to_storage_buffer(binding=1)
    bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
    program.run((int(component_capacity) + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative("island_runtime")


def _label_component_mask_texture(
    pipeline,
    world: "WorldEngine",
    component_mask: np.ndarray,
    *,
    x0: int = 0,
    y0: int = 0,
) -> tuple[Any, int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = component_mask.shape
    if width == 0 or height == 0:
        return None, width, height
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    resources.structural_tex.write(component_mask.astype("f4", copy=False).tobytes())
    return pipeline._label_component_texture(world, resources.structural_tex, width, height, x0=x0, y0=y0)


def collect_component_labels(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    width: int,
    height: int,
) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0:
        return np.zeros((0,), dtype=np.int32)
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    cell_count = max(1, int(width) * int(height))
    pipeline._ensure_component_work_buffers(ctx, resources, cell_count)
    resources.component_count.write(np.zeros(1, dtype=np.uint32).tobytes())

    program = pipeline.programs["collect_component_labels"]
    program["region_size"].value = (int(width), int(height))
    program["empty_min"].value = (int(width), int(height))
    label_texture.use(location=0)
    resources.component_flags.bind_to_storage_buffer(binding=0)
    resources.component_labels.bind_to_storage_buffer(binding=1)
    resources.component_count.bind_to_storage_buffer(binding=2)
    resources.component_metadata.bind_to_storage_buffer(binding=3)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    ctx.finish()

    component_count = int(np.frombuffer(resources.component_count.read(size=4), dtype=np.uint32, count=1)[0])
    if component_count <= 0:
        return np.zeros((0,), dtype=np.int32)
    return np.frombuffer(
        resources.component_labels.read(size=component_count * np.dtype(np.int32).itemsize),
        dtype=np.int32,
        count=component_count,
    ).copy()


def summarize_labeled_components(
    pipeline,
    world: "WorldEngine",
    labels: np.ndarray,
    component_labels: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = labels.shape
    component_count = int(component_labels.size)
    if width == 0 or height == 0 or component_count == 0:
        return np.zeros((0, 5), dtype=np.int32)
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    resources.support_ping.write(labels.astype("f4", copy=False).tobytes())
    return pipeline.summarize_labeled_component_texture(
        world,
        resources.support_ping,
        component_labels,
        x0,
        y0,
        width,
        height,
    )


def summarize_labeled_component_texture(
    pipeline,
    world: "WorldEngine",
    label_texture: Any,
    component_labels: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    component_count = int(component_labels.size)
    if width == 0 or height == 0 or component_count == 0:
        return np.zeros((0, 5), dtype=np.int32)
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    metadata = np.zeros((component_count, 5), dtype=np.int32)
    metadata[:, 0] = int(x0 + width)
    metadata[:, 1] = int(y0 + height)
    metadata[:, 2] = int(x0)
    metadata[:, 3] = int(y0)
    pipeline._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
    pipeline._write_dynamic_buffer(ctx, resources, "component_metadata", metadata)

    program = pipeline.programs["summarize_components"]
    program["region_size"].value = (width, height)
    program["region_origin"].value = (int(x0), int(y0))
    program["component_count"].value = component_count
    label_texture.use(location=0)
    resources.component_labels.bind_to_storage_buffer(binding=0)
    resources.component_metadata.bind_to_storage_buffer(binding=1)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    ctx.finish()
    return np.frombuffer(
        resources.component_metadata.read(size=metadata.nbytes),
        dtype=np.int32,
        count=metadata.size,
    ).reshape((component_count, 5)).copy()


def materialize_labeled_components(
    pipeline,
    world: "WorldEngine",
    labels: np.ndarray,
    component_labels: np.ndarray,
    component_island_ids: np.ndarray,
    x0: int,
    y0: int,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = labels.shape
    if width == 0 or height == 0 or component_labels.size == 0:
        return
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    resources.support_ping.write(labels.astype("f4", copy=False).tobytes())
    pipeline.materialize_labeled_component_texture(
        world,
        resources.support_ping,
        component_labels,
        component_island_ids,
        x0,
        y0,
        width,
        height,
    )
