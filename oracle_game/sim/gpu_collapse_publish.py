from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    LOCAL_SIZE
)



def _upload_region_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    ys = slice(y0, y0 + height)
    xs = slice(x0, x0 + width)
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    world._require_gpu_authoritative_resources(
        "collapse input",
        "cell_core",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )
    upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
    upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
    upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
    upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
    pipeline.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
    pipeline.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
    pipeline.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
    pipeline.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
    if upload_cell_state_from_cpu:
        resources.material_tex.write(world.material_id[ys, xs].astype("f4").tobytes())
        resources.phase_tex.write(world.phase[ys, xs].astype("f4").tobytes())
        resources.cell_flags_tex.write(world.cell_flags[ys, xs].astype("f4").tobytes())
        resources.timer_tex.write(world.timer_pack[ys, xs].astype("f4").tobytes())
        resources.integrity_tex.write(world.integrity[ys, xs].astype("f4").tobytes())
        resources.temp_tex.write(world.cell_temperature[ys, xs].astype("f4").tobytes())
    if upload_island_id_from_cpu:
        resources.island_id_tex.write(world.island_id[ys, xs].astype("f4").tobytes())
    if upload_entity_id_from_cpu:
        resources.entity_id_tex.write(world.entity_id[ys, xs].astype("f4").tobytes())
    if upload_displaced_from_cpu:
        resources.displaced_tex.write(world.placeholder_displaced_material[ys, xs].astype("f4").tobytes())
    pipeline._load_authoritative_bridge_region_inputs(world, resources, x0, y0, width, height)


def _load_authoritative_bridge_region_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative input state")
    if bridge.ctx is not world.bridge.ctx:
        raise RuntimeError("GPU collapse pipeline cannot consume authoritative bridge state from a separate GL context")

    world._require_gpu_authoritative_resources(
        "collapse input",
        "cell_core",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
        return

    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if copy_cell_core:
        program = pipeline.programs["load_bridge_region_cell"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["copy_cell_core"].value = bool(copy_cell_core)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        resources.material_tex.bind_to_image(0, read=False, write=True)
        resources.phase_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
        resources.timer_tex.bind_to_image(3, read=False, write=True)
        resources.integrity_tex.bind_to_image(4, read=False, write=True)
        resources.temp_tex.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(bridge.ctx)

    if copy_island_id or copy_entity_id or copy_displaced:
        program = pipeline.programs["load_bridge_region_cell_aux"]
        program["region_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["cell_grid_size"].value = (int(world.width), int(world.height))
        program["copy_island_id"].value = bool(copy_island_id)
        program["copy_entity_id"].value = bool(copy_entity_id)
        program["copy_displaced_material"].value = bool(copy_displaced)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        resources.island_id_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_tex.bind_to_image(2, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(bridge.ctx)


def _load_authoritative_bridge_connected_tile_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative input state")
    if bridge.ctx is not world.bridge.ctx:
        raise RuntimeError("GPU collapse pipeline cannot consume authoritative bridge state from a separate GL context")

    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
        return

    pipeline._ensure_programs(bridge.ctx)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected bridge input load requires tile_size <= 32")
    tile_grid_size = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )

    if copy_cell_core:
        program = pipeline.programs["load_bridge_connected_tile_cell"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected bridge cell input load requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = tile_grid_size
        program["tile_size"].value = int(tile_size)
        program["copy_cell_core"].value = bool(copy_cell_core)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        resources.material_tex.bind_to_image(0, read=False, write=True)
        resources.phase_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
        resources.timer_tex.bind_to_image(3, read=False, write=True)
        resources.integrity_tex.bind_to_image(4, read=False, write=True)
        resources.temp_tex.bind_to_image(5, read=False, write=True)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(bridge.ctx)

    if copy_island_id or copy_entity_id or copy_displaced:
        program = pipeline.programs["load_bridge_connected_tile_cell_aux"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("formal connected bridge aux input load requires ComputeShader.run_indirect")
        program["cell_grid_size"].value = (int(width), int(height))
        program["region_origin"].value = (int(x0), int(y0))
        program["world_grid_size"].value = (int(world.width), int(world.height))
        program["tile_grid_size"].value = tile_grid_size
        program["tile_size"].value = int(tile_size)
        program["copy_island_id"].value = bool(copy_island_id)
        program["copy_entity_id"].value = bool(copy_entity_id)
        program["copy_displaced_material"].value = bool(copy_displaced)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=0)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=3)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=4)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=5)
        resources.island_id_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_tex.bind_to_image(2, read=False, write=True)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
        pipeline._sync_compute_writes(bridge.ctx)


def _load_authoritative_bridge_pending_region(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if "collapse_delay_pending" not in bridge.gpu_authoritative_resources:
        return
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending state")
    program = pipeline.programs["load_bridge_region_pending"]
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
    resources.phase_tex.bind_to_image(1, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)


def _load_authoritative_bridge_connected_tile_pending(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if "collapse_delay_pending" not in bridge.gpu_authoritative_resources:
        return
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending state")
    pipeline._ensure_programs(bridge.ctx)
    tile_size = max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    if tile_size > FORMAL_CONNECTED_TILE_LOCAL_SIZE:
        raise RuntimeError("formal connected pending input load requires tile_size <= 32")
    program = pipeline.programs["load_bridge_connected_tile_pending"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected pending input load requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(tile_size)
    bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
    resources.phase_tex.bind_to_image(0, read=False, write=True)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(bridge.ctx)


def _publish_bridge_pending_region_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    pipeline._publish_bridge_pending_region_outputs_from_texture(
        world,
        resources,
        resources.support_pong,
        x0,
        y0,
        width,
        height,
    )


def _publish_bridge_pending_region_outputs_from_texture(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    pending_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    tile_mask_name: str | None = None,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative pending output")
    connected_tiles = tile_mask_name is not None
    program = pipeline.programs[
        "publish_bridge_region_pending_connected_tiles" if connected_tiles else "publish_bridge_region_pending"
    ]
    if connected_tiles and not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected pending publish requires ComputeShader.run_indirect")
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    if connected_tiles:
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
    pending_texture.use(location=0)
    bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
    if connected_tiles:
        assert tile_mask_name is not None
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    else:
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("collapse_delay_pending")


def _publish_bridge_region_mask(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    texture: Any,
    resource_name: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    mode: int = 0,
    tile_mask_name: str | None = None,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative runtime masks")
    connected_tiles = tile_mask_name is not None
    program = pipeline.programs["publish_bridge_region_mask_connected_tiles" if connected_tiles else "publish_bridge_region_mask"]
    if connected_tiles and not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected mask publish requires ComputeShader.run_indirect")
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["mode"].value = int(mode)
    if connected_tiles:
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
    texture.use(location=0)
    resources.structural_tex.use(location=1)
    bridge.buffers[resource_name].bind_to_storage_buffer(binding=0)
    if connected_tiles:
        assert tile_mask_name is not None
        bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
        bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
        bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
        program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    else:
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(resource_name)


def _publish_bridge_supported_unsupported_masks_connected_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    supported_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative runtime masks")
    program = pipeline.programs["publish_bridge_supported_unsupported_masks_connected_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected support mask publish requires ComputeShader.run_indirect")
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(
        max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
    )
    supported_texture.use(location=0)
    resources.structural_tex.use(location=1)
    bridge.buffers["collapse_supported_mask"].bind_to_storage_buffer(binding=0)
    bridge.buffers["collapse_unsupported_mask"].bind_to_storage_buffer(binding=1)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("collapse_supported_mask", "collapse_unsupported_mask")


def _publish_bridge_region_labels(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    label_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative component labels")
    program = pipeline.programs["publish_bridge_region_labels"]
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    label_texture.use(location=0)
    bridge.buffers["collapse_component_label"].bind_to_storage_buffer(binding=0)
    bridge.buffers["collapse_collapsed_cell_mask"].bind_to_storage_buffer(binding=1)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("collapse_component_label", "collapse_collapsed_cell_mask")


def _publish_bridge_region_labels_connected_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    label_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative component labels")
    program = pipeline.programs["publish_bridge_region_labels_connected_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected component label publish requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
    label_texture.use(location=0)
    bridge.buffers["collapse_component_label"].bind_to_storage_buffer(binding=0)
    bridge.buffers["collapse_collapsed_cell_mask"].bind_to_storage_buffer(binding=1)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("collapse_component_label", "collapse_collapsed_cell_mask")


def _publish_bridge_region_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative output state")
        return
    if bridge.ctx is not world.bridge.ctx:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU collapse pipeline cannot publish authoritative state from a separate GL context")
        return
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("collapse output", "cell_core")
        bridge.sync_world(world)

    program = pipeline.programs["publish_bridge_region_cell"]
    program["region_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    resources.material_out_tex.use(location=0)
    resources.phase_out_tex.use(location=1)
    resources.cell_flags_out_tex.use(location=2)
    resources.timer_out_tex.use(location=3)
    resources.integrity_out_tex.use(location=4)
    resources.temp_out_tex.use(location=5)
    resources.island_id_out_tex.use(location=6)
    resources.entity_id_out_tex.use(location=7)
    resources.displaced_out_tex.use(location=8)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )


def _publish_bridge_region_outputs_connected_tiles(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse pipeline requires bridge GPU resources for authoritative output state")
    if bridge.ctx is not world.bridge.ctx:
        raise RuntimeError("GPU collapse pipeline cannot publish authoritative state from a separate GL context")
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("collapse output", "cell_core")
        bridge.sync_world(world)

    program = pipeline.programs["publish_bridge_region_cell_connected_tiles"]
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected output publish requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(width), int(height))
    program["region_origin"].value = (int(x0), int(y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (
        int(getattr(world.active, "tile_width", 1)),
        int(getattr(world.active, "tile_height", 1)),
    )
    program["tile_size"].value = int(max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE))))
    resources.material_out_tex.use(location=0)
    resources.phase_out_tex.use(location=1)
    resources.cell_flags_out_tex.use(location=2)
    resources.timer_out_tex.use(location=3)
    resources.integrity_out_tex.use(location=4)
    resources.temp_out_tex.use(location=5)
    resources.island_id_out_tex.use(location=6)
    resources.entity_id_out_tex.use(location=7)
    resources.displaced_out_tex.use(location=8)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
    bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=4)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )


def _barrier_bits(pipeline) -> tuple[str, ...]:
    # collapse uses indirect dispatch in addition to the default
    # image/texture/storage sync.
    return (
        "SHADER_STORAGE_BARRIER_BIT",
        "SHADER_IMAGE_ACCESS_BARRIER_BIT",
        "TEXTURE_FETCH_BARRIER_BIT",
        "COMMAND_BARRIER_BIT",
    )


def _download_region_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    ys = slice(y0, y0 + height)
    xs = slice(x0, x0 + width)
    world.material_id[ys, xs] = np.rint(
        np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.int32)
    world.phase[ys, xs] = np.rint(
        np.frombuffer(resources.phase_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.uint8)
    world.cell_flags[ys, xs] = np.rint(
        np.frombuffer(resources.cell_flags_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.uint8)
    world.timer_pack[ys, xs] = np.rint(
        np.frombuffer(resources.timer_out_tex.read(), dtype="f4").reshape((height, width, 4))
    ).astype(np.uint8)
    world.integrity[ys, xs] = np.frombuffer(resources.integrity_out_tex.read(), dtype="f4").reshape((height, width))
    world.cell_temperature[ys, xs] = np.frombuffer(resources.temp_out_tex.read(), dtype="f4").reshape((height, width))
    world.island_id[ys, xs] = np.rint(
        np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.int32)
    world.entity_id[ys, xs] = np.rint(
        np.frombuffer(resources.entity_id_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.int32)
    world.placeholder_displaced_material[ys, xs] = np.rint(
        np.frombuffer(resources.displaced_out_tex.read(), dtype="f4").reshape((height, width))
    ).astype(np.int32)
