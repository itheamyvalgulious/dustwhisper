from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_collapse import GPUCollapseResources

from oracle_game.types import CollapseBehavior, Phase

from oracle_game.sim.gpu_collapse import (
    FORMAL_CONNECTED_TILE_COUNT_BUFFER,
    FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
    FORMAL_CONNECTED_TILE_LIST_BUFFER,
    FORMAL_CONNECTED_TILE_LOCAL_SIZE,
    LOCAL_SIZE
)



def solve_region(
    pipeline,
    world: "WorldEngine",
    structural_mask: np.ndarray,
    support_seed_mask: np.ndarray,
    *,
    x0: int = 0,
    y0: int = 0,
) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = structural_mask.shape
    if width == 0 or height == 0:
        return np.zeros_like(structural_mask, dtype=bool)
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    resources.structural_tex.write(structural_mask.astype("f4", copy=False).tobytes())
    resources.support_ping.write(support_seed_mask.astype("f4", copy=False).tobytes())
    resources.support_pong.write(support_seed_mask.astype("f4", copy=False).tobytes())
    current = pipeline.solve_region_textures(world, resources, width, height, x0=x0, y0=y0)
    supported = np.frombuffer(current.read(), dtype="f4").reshape((height, width)) > 0.5
    return structural_mask & ~supported


def solve_region_textures(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    width: int,
    height: int,
    *,
    x0: int = 0,
    y0: int = 0,
    publish_masks: bool = True,
) -> Any:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    if pipeline._formal_gpu_frame(world):
        tile_mask_name = pipeline._seed_formal_texture_region_tile_worklist(world, width, height)
        if tile_mask_name is not None:
            return pipeline._solve_formal_connected_tile_support_textures(
                world,
                resources,
                x0,
                y0,
                width,
                height,
                tile_mask_name,
                publish_masks=publish_masks,
            )
    current = resources.support_ping
    scratch = resources.support_pong
    jumps = pipeline._formal_jfa_jumps(width, height)
    for jump in jumps:
        current, scratch, _ = pipeline._run_pass(ctx, resources, current, scratch, width, height, jump, read_changed=False)
    if pipeline._formal_gpu_frame(world):
        current, scratch = pipeline._run_formal_support_refine_passes(
            ctx,
            resources,
            current,
            scratch,
            width,
            height,
            jumps,
        )
        if publish_masks:
            pipeline._publish_bridge_region_mask(world, resources, current, "collapse_supported_mask", x0, y0, width, height)
            pipeline._publish_bridge_region_mask(
                world,
                resources,
                current,
                "collapse_unsupported_mask",
                x0,
                y0,
                width,
                height,
                mode=1,
            )
    else:
        while True:
            current, scratch, changed = pipeline._run_pass(ctx, resources, current, scratch, width, height, 1)
            if not changed:
                break
    return current


def classify_world_structural_mask(pipeline, world: "WorldEngine") -> np.ndarray:
    structural, _, _ = pipeline.classify_region(world, 0, 0, world.width, world.height)
    return structural


def expand_region_to_component_bbox(
    pipeline,
    world: "WorldEngine",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int, int]:
    if pipeline._formal_gpu_frame(world):
        return pipeline._expand_formal_region_to_component_bbox(world, x0, y0, x1, y1)
    seed_x0 = max(0, int(x0) - 1)
    seed_y0 = max(0, int(y0) - 1)
    seed_x1 = min(world.width, int(x1) + 1)
    seed_y1 = min(world.height, int(y1) + 1)
    if seed_x0 >= seed_x1 or seed_y0 >= seed_y1:
        return (seed_x0, seed_y0, seed_x1, seed_y1)
    resources, width, height = pipeline.classify_region_textures(
        world,
        0,
        0,
        world.width,
        world.height,
        publish_masks=False,
    )
    pipeline.seed_structural_region_texture(world, resources, width, height, seed_x0, seed_y0, seed_x1, seed_y1)
    connected_texture = pipeline.solve_region_textures(world, resources, width, height, x0=0, y0=0, publish_masks=False)
    metadata = pipeline.summarize_labeled_component_texture(
        world,
        connected_texture,
        np.asarray([1], dtype=np.int32),
        0,
        0,
        width,
        height,
    )
    if metadata.size == 0:
        return (seed_x0, seed_y0, seed_x1, seed_y1)
    min_x, min_y, max_x, max_y, cell_count = (int(value) for value in metadata[0])
    if cell_count <= 0:
        return (seed_x0, seed_y0, seed_x1, seed_y1)
    return (
        min(seed_x0, min_x),
        min(seed_y0, min_y),
        max(seed_x1, max_x),
        max(seed_y1, max_y),
    )


def _expand_formal_region_to_component_bbox(
    pipeline,
    world: "WorldEngine",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int, int]:
    world_width = int(world.width)
    world_height = int(world.height)
    seed_x0 = max(0, min(world_width, int(x0)))
    seed_y0 = max(0, min(world_height, int(y0)))
    seed_x1 = max(0, min(world_width, int(x1)))
    seed_y1 = max(0, min(world_height, int(y1)))
    if seed_x0 >= seed_x1 or seed_y0 >= seed_y1:
        return (seed_x0, seed_y0, seed_x1, seed_y1)

    # Formal frames must not read component metadata back to the CPU to steer bbox growth.
    # The caller supplies an already halo-expanded dirty/event region; keep it tile-aligned
    # and let GPU eligibility masks restrict materialization to dirty-connected structure.
    tile_size = max(1, int(getattr(world.active, "tile_size", 32)))

    def align_down(value: int) -> int:
        return max(0, (int(value) // tile_size) * tile_size)

    def align_up(value: int, limit: int) -> int:
        return min(int(limit), ((int(value) + tile_size - 1) // tile_size) * tile_size)

    search_x0 = align_down(seed_x0)
    search_y0 = align_down(seed_y0)
    search_x1 = align_up(seed_x1, world_width)
    search_y1 = align_up(seed_y1, world_height)
    return (search_x0, search_y0, search_x1, search_y1)


def classify_region(
    pipeline,
    world: "WorldEngine",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    width = max(0, int(x1) - int(x0))
    height = max(0, int(y1) - int(y0))
    if width == 0 or height == 0:
        empty_bool = np.zeros((height, width), dtype=np.bool_)
        empty_int = np.zeros((height, width), dtype=np.int32)
        return empty_bool, empty_bool.copy(), empty_int
    resources, width, height = pipeline.classify_region_textures(world, x0, y0, x1, y1)
    ctx.finish()
    structural = np.frombuffer(resources.structural_tex.read(), dtype="f4").reshape((height, width)) > 0.5
    support_seed = np.frombuffer(resources.support_ping.read(), dtype="f4").reshape((height, width)) > 0.5
    behavior = np.rint(np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width))).astype(np.int32)
    return structural, support_seed, behavior


def classify_region_textures(
    pipeline,
    world: "WorldEngine",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    publish_masks: bool = True,
    treat_region_boundary_as_support: bool = False,
) -> tuple[GPUCollapseResources, int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    width = max(0, int(x1) - int(x0))
    height = max(0, int(y1) - int(y0))
    if width == 0 or height == 0:
        raise ValueError("classify_region_textures requires a non-empty region")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    pipeline._upload_region_state(world, resources, x0, y0, width, height)
    structural_params, support_params, behavior_params = pipeline._classification_material_params(world)
    pipeline._write_dynamic_buffer(ctx, resources, "material_structural", structural_params)
    pipeline._write_dynamic_buffer(ctx, resources, "material_support_anchor", support_params)
    pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_behavior", behavior_params)

    program = pipeline.programs["classify_cells"]
    program["region_size"].value = (width, height)
    program["region_origin"].value = (int(x0), int(y0))
    program["world_width"].value = int(world.width)
    program["world_height"].value = int(world.height)
    program["material_count"].value = int(behavior_params.size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["treat_region_boundary_as_support"].value = bool(treat_region_boundary_as_support)
    resources.material_tex.use(location=0)
    resources.phase_tex.use(location=1)
    resources.structural_tex.bind_to_image(2, read=False, write=True)
    resources.support_ping.bind_to_image(3, read=False, write=True)
    resources.material_out_tex.bind_to_image(4, read=False, write=True)
    resources.material_structural.bind_to_storage_buffer(binding=0)
    resources.material_support_anchor.bind_to_storage_buffer(binding=1)
    resources.material_collapse_behavior.bind_to_storage_buffer(binding=2)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if pipeline._formal_gpu_frame(world) and publish_masks:
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.structural_tex,
            "collapse_structural_mask",
            x0,
            y0,
            width,
            height,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.support_ping,
            "collapse_support_seed_mask",
            x0,
            y0,
            width,
            height,
        )
    return resources, width, height


def resolve_unsupported_outcomes(
    pipeline,
    world: "WorldEngine",
    unsupported: np.ndarray,
    behavior_region: np.ndarray,
    x0: int,
    y0: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = unsupported.shape
    if width == 0 or height == 0:
        empty = np.zeros_like(unsupported, dtype=np.bool_)
        return empty, empty.copy(), empty.copy()
    resources, width, height = pipeline.resolve_unsupported_outcome_textures(
        world,
        unsupported,
        behavior_region,
        x0,
        y0,
    )
    delayed_pending = np.frombuffer(resources.support_pong.read(), dtype="f4").reshape((height, width)) > 0.5
    immune_unsupported = np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((height, width)) > 0.5
    collapse_now = np.frombuffer(resources.phase_out_tex.read(), dtype="f4").reshape((height, width)) > 0.5
    if not pipeline._formal_gpu_frame(world):
        pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
        pending_region[:] = delayed_pending
    return delayed_pending, immune_unsupported, collapse_now


def resolve_unsupported_outcome_textures(
    pipeline,
    world: "WorldEngine",
    unsupported: np.ndarray,
    behavior_region: np.ndarray,
    x0: int,
    y0: int,
) -> tuple[GPUCollapseResources, int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    height, width = unsupported.shape
    if width == 0 or height == 0:
        raise ValueError("resolve_unsupported_outcome_textures requires a non-empty region")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(ctx, width, height)
    pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
    resources.structural_tex.write(unsupported.astype("f4", copy=False).tobytes())
    resources.support_ping.write(behavior_region.astype("f4", copy=False).tobytes())
    resources.phase_tex.write(pending_region.astype("f4", copy=False).tobytes())
    pipeline._load_authoritative_bridge_pending_region(world, resources, x0, y0, width, height)

    program = pipeline.programs["resolve_outcomes"]
    program["region_size"].value = (width, height)
    program["behavior_falling_island"].value = int(CollapseBehavior.FALLING_ISLAND)
    program["behavior_delayed"].value = int(CollapseBehavior.DELAYED)
    program["behavior_immune"].value = int(CollapseBehavior.IMMUNE)
    resources.structural_tex.use(location=0)
    resources.support_ping.use(location=1)
    resources.phase_tex.use(location=2)
    resources.support_pong.bind_to_image(3, read=False, write=True)
    resources.material_out_tex.bind_to_image(4, read=False, write=True)
    resources.phase_out_tex.bind_to_image(5, read=False, write=True)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if pipeline._formal_gpu_frame(world):
        pipeline._publish_bridge_pending_region_outputs(world, resources, x0, y0, width, height)
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.support_pong,
            "collapse_delayed_pending_mask",
            x0,
            y0,
            width,
            height,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.material_out_tex,
            "collapse_immune_unsupported_mask",
            x0,
            y0,
            width,
            height,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.phase_out_tex,
            "collapse_collapsed_cell_mask",
            x0,
            y0,
            width,
            height,
        )
    else:
        ctx.finish()
    return resources, width, height


def resolve_supported_outcome_textures(
    pipeline,
    world: "WorldEngine",
    resources: GPUCollapseResources,
    supported_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    eligibility_texture: Any | None = None,
    tile_mask_name: str | None = None,
) -> tuple[GPUCollapseResources, int, int]:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse pipeline requires a valid ModernGL context")
    if width == 0 or height == 0:
        raise ValueError("resolve_supported_outcome_textures requires a non-empty region")
    connected_tiles = pipeline._formal_gpu_frame(world) and tile_mask_name is not None
    if connected_tiles and "collapse_delay_pending" in world.bridge.gpu_authoritative_resources:
        assert tile_mask_name is not None
        pipeline._load_authoritative_bridge_connected_tile_pending(
            world,
            resources,
            x0,
            y0,
            width,
            height,
            tile_mask_name,
        )
    else:
        pending_region = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
        resources.phase_tex.write(pending_region.astype("f4", copy=False).tobytes())
        pipeline._load_authoritative_bridge_pending_region(world, resources, x0, y0, width, height)

    program = pipeline.programs[
        "resolve_outcomes_from_supported_connected_tiles" if connected_tiles else "resolve_outcomes_from_supported"
    ]
    if connected_tiles and not hasattr(program, "run_indirect"):
        raise RuntimeError("formal connected outcome resolve requires ComputeShader.run_indirect")
    program["region_size"].value = (width, height)
    program["behavior_falling_island"].value = int(CollapseBehavior.FALLING_ISLAND)
    program["behavior_delayed"].value = int(CollapseBehavior.DELAYED)
    program["behavior_immune"].value = int(CollapseBehavior.IMMUNE)
    program["use_eligibility"].value = eligibility_texture is not None
    if connected_tiles:
        program["tile_grid_size"].value = (
            int(getattr(world.active, "tile_width", 1)),
            int(getattr(world.active, "tile_height", 1)),
        )
        program["tile_size"].value = int(
            max(1, int(getattr(world.active, "tile_size", FORMAL_CONNECTED_TILE_LOCAL_SIZE)))
        )
    resources.structural_tex.use(location=0)
    supported_texture.use(location=1)
    resources.material_out_tex.use(location=2)
    resources.phase_tex.use(location=3)
    (eligibility_texture if eligibility_texture is not None else resources.structural_tex).use(location=7)
    resources.temp_out_tex.bind_to_image(4, read=False, write=True)
    resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
    resources.phase_out_tex.bind_to_image(6, read=False, write=True)
    if connected_tiles:
        assert tile_mask_name is not None
        world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=0)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=1)
        world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=2)
        program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    else:
        group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
    ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    if pipeline._formal_gpu_frame(world):
        pipeline._publish_bridge_pending_region_outputs_from_texture(
            world,
            resources,
            resources.temp_out_tex,
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name if connected_tiles else None,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.temp_out_tex,
            "collapse_delayed_pending_mask",
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name if connected_tiles else None,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.integrity_out_tex,
            "collapse_immune_unsupported_mask",
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name if connected_tiles else None,
        )
        pipeline._publish_bridge_region_mask(
            world,
            resources,
            resources.phase_out_tex,
            "collapse_collapsed_cell_mask",
            x0,
            y0,
            width,
            height,
            tile_mask_name=tile_mask_name if connected_tiles else None,
        )
    else:
        ctx.finish()
    return resources, width, height


def _run_pass(
    pipeline,
    ctx: Any,
    resources: GPUCollapseResources,
    current: Any,
    scratch: Any,
    width: int,
    height: int,
    jump: int,
    *,
    read_changed: bool = True,
) -> tuple[Any, Any, bool]:
    program = pipeline.programs["propagate"]
    resources.change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
    program["region_size"].value = (width, height)
    program["jump"].value = jump
    resources.structural_tex.use(location=0)
    current.use(location=1)
    scratch.bind_to_image(2, read=False, write=True)
    resources.change_flag.bind_to_storage_buffer(binding=0)
    group_x = (width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    if not read_changed:
        pipeline._sync_compute_writes(ctx)
        return scratch, current, True
    ctx.finish()
    changed = bool(np.frombuffer(resources.change_flag.read(), dtype=np.uint32, count=1)[0])
    return scratch, current, changed


def release(pipeline) -> None:
    if pipeline.resources is None:
        return
    for resource in (
        pipeline.resources.structural_tex,
        pipeline.resources.support_ping,
        pipeline.resources.support_pong,
        pipeline.resources.material_tex,
        pipeline.resources.material_out_tex,
        pipeline.resources.phase_tex,
        pipeline.resources.phase_out_tex,
        pipeline.resources.cell_flags_tex,
        pipeline.resources.cell_flags_out_tex,
        pipeline.resources.timer_tex,
        pipeline.resources.timer_out_tex,
        pipeline.resources.integrity_tex,
        pipeline.resources.integrity_out_tex,
        pipeline.resources.temp_tex,
        pipeline.resources.temp_out_tex,
        pipeline.resources.island_id_tex,
        pipeline.resources.island_id_out_tex,
        pipeline.resources.entity_id_tex,
        pipeline.resources.entity_id_out_tex,
        pipeline.resources.displaced_tex,
        pipeline.resources.displaced_out_tex,
        pipeline.resources.change_flag,
        pipeline.resources.component_labels,
        pipeline.resources.component_island_ids,
        pipeline.resources.component_metadata,
        pipeline.resources.component_flags,
        pipeline.resources.component_count,
        pipeline.resources.component_dispatch_args,
        pipeline.resources.region_flags,
        pipeline.resources.connected_tile_row_masks,
        pipeline.resources.connected_tile_column_masks,
        pipeline.resources.material_structural,
        pipeline.resources.material_support_anchor,
        pipeline.resources.material_collapse_behavior,
        pipeline.resources.material_collapse_generation,
        pipeline.resources.material_base_integrity,
        pipeline.resources.material_spawn_temperature,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None
