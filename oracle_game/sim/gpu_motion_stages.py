from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE
)


def step(pipeline, world: "WorldEngine", dt: float, *, solve_tile_mask: np.ndarray) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._reset_pass_profile()
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "powder_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "powder_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "powder_targets"):
        pipeline._run_powder_targets(world, resources, group_x, group_y, dt)
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        return pipeline._download_outputs(world, resources)
    return np.zeros((world.height, world.width, 2), dtype=np.int32)


def integrate_velocity(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    solve_tile_mask: np.ndarray,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "integrate_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "integrate_load_bridge_inputs"):
        pipeline._load_authoritative_integrate_inputs(world, resources, group_x, group_y)
    program = pipeline.programs["integrate_velocity"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["gas_cell_size"].value = world.gas_cell_size
    program["dt"].value = dt
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    resources.material_tex.use(location=1)
    resources.velocity_tex.use(location=2)
    resources.flow_tex.use(location=3)
    resources.active_tile_tex.use(location=4)
    resources.velocity_out_tex.bind_to_image(5, read=False, write=True)
    with pipeline._profile_pass(world, "integrate_velocity"):
        pipeline._run_active_tile_indirect(program, resources, "integrate velocity")
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "integrate_publish_bridge"):
        active_tile_indirect = pipeline._formal_gpu_frame(world)
        if not pipeline._publish_bridge_velocity_words(
            world,
            resources,
            active_tile_indirect=active_tile_indirect,
        ):
            pipeline._publish_bridge_outputs(
                world,
                resources,
                output_textures=False,
                velocity_out_active_only=True,
                active_tile_indirect=active_tile_indirect,
            )
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        world.velocity[:] = pipeline._download_velocity_output(world, resources)
