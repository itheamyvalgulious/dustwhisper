from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_liquid import GPULiquidResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_liquid import (
    MAX_MATERIALS,
    PASS_LOCAL_SIZE,
)


def step(
    pipeline,
    world: "WorldEngine",
    *,
    solve_tile_mask: np.ndarray,
    post_tile_mask: np.ndarray,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.reset_pass_profile()
    with pipeline._profile_pass(world, "liquid_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
    with pipeline._profile_pass(world, "liquid_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_inputs(world, resources)
    with pipeline._profile_pass(world, "liquid_compact_active_tiles"):
        pipeline._compact_active_tiles(world, resources)
    with pipeline._profile_pass(world, "liquid_tile_solve"):
        pipeline._run_tile_solve(world, resources)
    if pipeline._active_scheduler_gpu_authoritative(world):
        with pipeline._profile_pass(world, "liquid_load_active_mask"):
            pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
    else:
        with pipeline._profile_pass(world, "liquid_upload_active_mask"):
            pipeline._upload_active_tile_mask(resources, post_tile_mask)
    with pipeline._profile_pass(world, "liquid_compact_active_cell_tiles"):
        pipeline._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
        )
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_copy_tile_solve"):
            pipeline._run_copy_core_state(
                world,
                resources,
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
            )
        with pipeline._profile_pass(world, "liquid_build_seam_x_boundaries"):
            pipeline._build_seam_boundary_dispatch(world, resources, axis="x")
        with pipeline._profile_pass(world, "liquid_prefetch_seam_x_boundaries"):
            pipeline._prefetch_seam_boundary_bridge_inputs(world, resources, axis="x")
    with pipeline._profile_pass(world, "liquid_seam_x"):
        pipeline._run_seam_pass(
            "seam_x",
            world,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.active_tile_tex,
            boundary_dispatch=formal_gpu_frame,
        )
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_reload_seam_x_active_tiles"):
            pipeline._reload_and_compact_active_cell_tiles(world, resources)
        with pipeline._profile_pass(world, "liquid_copy_seam_x"):
            pipeline._run_copy_core_state(
                world,
                resources,
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
            )
        with pipeline._profile_pass(world, "liquid_build_seam_y_boundaries"):
            pipeline._build_seam_boundary_dispatch(world, resources, axis="y")
        with pipeline._profile_pass(world, "liquid_prefetch_seam_y_boundaries"):
            pipeline._prefetch_seam_boundary_bridge_inputs(world, resources, axis="y")
    with pipeline._profile_pass(world, "liquid_seam_y"):
        pipeline._run_seam_pass(
            "seam_y",
            world,
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            resources.active_tile_tex,
            boundary_dispatch=formal_gpu_frame,
        )
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_reload_seam_y_active_tiles"):
            pipeline._reload_and_compact_active_cell_tiles(world, resources)
    with pipeline._profile_pass(world, "liquid_buoyancy_sink"):
        pipeline._run_buoyancy_pass(
            "buoyancy_sink",
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
        )
    with pipeline._profile_pass(world, "liquid_buoyancy_float"):
        pipeline._run_buoyancy_pass(
            "buoyancy_float",
            world,
            resources,
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
        )
    with pipeline._profile_pass(world, "liquid_copy_for_placeholder"):
        pipeline._run_copy_for_placeholder(
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
        )
    with pipeline._profile_pass(world, "liquid_placeholder_displacement"):
        pipeline._run_placeholder_displacement(
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
        )
    with pipeline._profile_pass(world, "liquid_cleanup_runtime"):
        pipeline._run_cleanup_runtime(world, resources)
    if pipeline._active_scheduler_gpu_authoritative(world):
        with pipeline._profile_pass(world, "liquid_reload_flow_active_mask"):
            pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        with pipeline._profile_pass(world, "liquid_compact_flow_active_tiles"):
            pipeline._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
            )
    with pipeline._profile_pass(world, "liquid_flow_intent"):
        pipeline._run_liquid_intent_pass(world, resources)
    if pipeline._formal_gpu_frame(world):
        with pipeline._profile_pass(world, "liquid_refresh_active_scheduler"):
            pipeline._refresh_active_scheduler_from_ttl(world)
    with pipeline._profile_pass(world, "liquid_publish_bridge"):
        pipeline._publish_bridge_outputs(world, resources)
    pipeline.last_cpu_mirror_downloaded = not pipeline._formal_gpu_frame(world)
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_outputs(world, resources, use_in=True)



def prepare_motion_flow_intent(
    pipeline,
    world: "WorldEngine",
    *,
    solve_tile_mask: np.ndarray,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.reset_pass_profile()
    with pipeline._profile_pass(world, "liquid_pre_motion_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
    with pipeline._profile_pass(world, "liquid_pre_motion_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_flow_intent_inputs(world, resources)
    with pipeline._profile_pass(world, "liquid_pre_motion_compact_active_tiles"):
        pipeline._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
        )
    with pipeline._profile_pass(world, "liquid_pre_motion_flow_intent"):
        pipeline._run_liquid_intent_pass(world, resources)
    pipeline.last_cpu_mirror_downloaded = False



def _upload_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    solve_tile_mask: np.ndarray,
) -> None:
    world.bridge.sync_rule_tables(world)
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    world._require_gpu_authoritative_resources(
        "liquid input",
        "cell_core",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "active_tile_ttl",
    )
    upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
    upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
    upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
    upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
    upload_active_from_cpu = not pipeline._active_scheduler_gpu_authoritative(world)
    pipeline.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
    pipeline.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
    pipeline.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
    pipeline.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
    pipeline.last_cpu_active_upload_skipped = not upload_active_from_cpu
    if upload_cell_state_from_cpu:
        resources.material_pre.write(world.material_id.astype("f4").tobytes())
        resources.material_in.write(world.material_id.astype("f4").tobytes())
        resources.material_out.write(world.material_id.astype("f4").tobytes())
        resources.phase_pre.write(world.phase.astype("f4").tobytes())
        resources.phase_in.write(world.phase.astype("f4").tobytes())
        resources.phase_out.write(world.phase.astype("f4").tobytes())
        resources.flags_in.write(world.cell_flags.astype("f4").tobytes())
        resources.flags_out.write(world.cell_flags.astype("f4").tobytes())
        resources.timer_in.write(world.timer_pack.astype("f4").tobytes())
        resources.timer_out.write(world.timer_pack.astype("f4").tobytes())
        resources.temp_in.write(world.cell_temperature.astype("f4").tobytes())
        resources.temp_out.write(world.cell_temperature.astype("f4").tobytes())
        resources.integrity_in.write(world.integrity.astype("f4").tobytes())
        resources.integrity_out.write(world.integrity.astype("f4").tobytes())
        resources.velocity_in.write(world.velocity.astype("f4").tobytes())
        resources.velocity_out.write(world.velocity.astype("f4").tobytes())
    if upload_island_id_from_cpu:
        resources.island_in.write(world.island_id.astype("f4").tobytes())
        resources.island_out.write(world.island_id.astype("f4").tobytes())
    if upload_entity_id_from_cpu:
        resources.entity_in.write(world.entity_id.astype("f4").tobytes())
        resources.entity_out.write(world.entity_id.astype("f4").tobytes())
    if upload_active_from_cpu:
        pipeline._upload_active_tile_mask(resources, solve_tile_mask)
    else:
        pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
    if upload_displaced_from_cpu:
        resources.displaced_in.write(world.placeholder_displaced_material.astype("f4").tobytes())
        resources.displaced_out.write(world.placeholder_displaced_material.astype("f4").tobytes())
    material_table = world.bridge.shadow_typed_tables["material_table"]
    table_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
    if resources.material_params_signature != table_signature:
        params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        params[:count, 0] = material_table[:count]["density"]
        params[:count, 1] = material_table[:count]["base_integrity"]
        params[:count, 2] = material_table[:count]["liquid_solver_kind_id"]
        params[:count, 3] = material_table[:count]["render_group_id"].astype("f4")
        resources.material_params.write(params.tobytes())
        resources.material_params_signature = table_signature



def _load_authoritative_bridge_inputs(pipeline, world: "WorldEngine", resources: GPULiquidResources) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
        return
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative input state")
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    group_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    group_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    if active_tile_indirect:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.active_tile_compact"):
            pipeline._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
            )
    ran_copy = False
    if copy_cell_core:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_cell_in"):
            program = pipeline.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["copy_cell_core"].value = bool(copy_cell_core)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            resources.material_pre.bind_to_image(0, read=False, write=True)
            resources.material_in.bind_to_image(1, read=False, write=True)
            resources.phase_pre.bind_to_image(2, read=False, write=True)
            resources.phase_in.bind_to_image(3, read=False, write=True)
            resources.flags_in.bind_to_image(4, read=False, write=True)
            resources.timer_in.bind_to_image(5, read=False, write=True)
            resources.temp_in.bind_to_image(6, read=False, write=True)
            resources.integrity_in.bind_to_image(7, read=False, write=True)
            if active_tile_indirect:
                pipeline._run_active_tile_indirect(program, resources, "bridge cell input load")
            else:
                program.run(group_x, group_y, 1)
            ran_copy = True

        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_cell_out"):
            program = pipeline.programs["load_bridge_cell_out"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            resources.material_out.bind_to_image(0, read=False, write=True)
            resources.phase_out.bind_to_image(1, read=False, write=True)
            resources.flags_out.bind_to_image(2, read=False, write=True)
            resources.timer_out.bind_to_image(3, read=False, write=True)
            resources.temp_out.bind_to_image(4, read=False, write=True)
            resources.integrity_out.bind_to_image(5, read=False, write=True)
            resources.velocity_out.bind_to_image(6, read=False, write=True)
            if active_tile_indirect:
                pipeline._run_active_tile_indirect(program, resources, "bridge cell output load")
            else:
                program.run(group_x, group_y, 1)
            ran_copy = True

    if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_aux"):
            program = pipeline.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["copy_cell_core"].value = bool(copy_cell_core)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            resources.velocity_in.bind_to_image(0, read=False, write=True)
            resources.island_in.bind_to_image(1, read=False, write=True)
            resources.entity_in.bind_to_image(2, read=False, write=True)
            resources.displaced_in.bind_to_image(3, read=False, write=True)
            if active_tile_indirect:
                pipeline._run_active_tile_indirect(program, resources, "bridge aux input load")
            else:
                program.run(group_x, group_y, 1)
            ran_copy = True

    if ran_copy:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.sync"):
            pipeline._sync_compute_writes(bridge.ctx)



def _load_authoritative_bridge_flow_intent_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    if not (copy_cell_core or copy_entity_id or copy_displaced):
        return
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for flow-intent input state")
    pipeline._compact_active_tiles(
        world,
        resources,
        workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
    )
    program = pipeline.programs["load_bridge_flow_intent_inputs"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["copy_cell_core"].value = bool(copy_cell_core)
    program["copy_entity_id"].value = bool(copy_entity_id)
    program["copy_displaced_material"].value = bool(copy_displaced)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
    resources.active_tile_count.bind_to_storage_buffer(binding=4)
    resources.active_tile_list.bind_to_storage_buffer(binding=5)
    resources.material_in.bind_to_image(0, read=False, write=True)
    resources.phase_in.bind_to_image(1, read=False, write=True)
    resources.velocity_in.bind_to_image(2, read=False, write=True)
    resources.entity_in.bind_to_image(3, read=False, write=True)
    resources.displaced_in.bind_to_image(4, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, "bridge flow-intent input load")
    pipeline._sync_compute_writes(bridge.ctx)



def _publish_bridge_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    use_out: bool = False,
    velocity_use_out: bool = False,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative output state")
        return
    program = pipeline.programs["publish_bridge_cell"]
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
    material_tex = resources.material_out if use_out else resources.material_in
    phase_tex = resources.phase_out if use_out else resources.phase_in
    flags_tex = resources.flags_out if use_out else resources.flags_in
    timer_tex = resources.timer_out if use_out else resources.timer_in
    temp_tex = resources.temp_out if use_out else resources.temp_in
    integrity_tex = resources.integrity_out if use_out else resources.integrity_in
    velocity_tex = resources.velocity_out if velocity_use_out else resources.velocity_in
    material_tex.use(location=0)
    phase_tex.use(location=1)
    flags_tex.use(location=2)
    timer_tex.use(location=3)
    temp_tex.use(location=4)
    integrity_tex.use(location=5)
    velocity_tex.use(location=6)
    resources.island_out.use(location=7)
    resources.entity_out.use(location=8)
    resources.displaced_in.use(location=9)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
    resources.active_tile_count.bind_to_storage_buffer(binding=4)
    resources.active_tile_list.bind_to_storage_buffer(binding=5)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    if active_tile_indirect:
        pipeline._run_active_tile_indirect(program, resources, "bridge cell publish")
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )



def _run_liquid_intent_pass(pipeline, world: "WorldEngine", resources: GPULiquidResources) -> None:
    program = pipeline.programs["liquid_flow_intent"]
    ctx = world.bridge.ctx
    assert ctx is not None
    target_texture = resources.liquid_flow_intent
    if pipeline._formal_gpu_frame(world):
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for liquid flow intent")
        target_texture = bridge.textures["liquid_flow_intent"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["phase_liquid"].value = int(Phase.LIQUID)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    resources.material_in.use(location=0)
    resources.phase_in.use(location=1)
    resources.velocity_in.use(location=2)
    resources.active_tile_tex.use(location=3)
    resources.entity_in.use(location=4)
    resources.displaced_in.use(location=5)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    target_texture.bind_to_image(0, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, "flow intent")
    pipeline._sync_compute_writes(ctx)
    if pipeline._formal_gpu_frame(world):
        world.bridge.mark_gpu_authoritative("liquid_flow_intent")



def _download_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    use_in: bool = False,
    velocity_use_out: bool = False,
) -> None:
    material = resources.material_in if use_in else resources.material_out
    phase = resources.phase_in if use_in else resources.phase_out
    flags = resources.flags_in if use_in else resources.flags_out
    timer = resources.timer_in if use_in else resources.timer_out
    temp = resources.temp_in if use_in else resources.temp_out
    integrity = resources.integrity_in if use_in else resources.integrity_out
    velocity = resources.velocity_out if velocity_use_out else (resources.velocity_in if use_in else resources.velocity_out)
    displaced = resources.displaced_in if use_in else resources.displaced_out
    world.material_id[:] = np.rint(
        np.frombuffer(material.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.phase[:] = np.rint(
        np.frombuffer(phase.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.uint8)
    world.cell_flags[:] = np.rint(
        np.frombuffer(flags.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.uint8)
    world.timer_pack[:] = np.rint(
        np.frombuffer(timer.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.uint8)
    world.cell_temperature[:] = np.frombuffer(temp.read(), dtype="f4").reshape((world.height, world.width))
    world.integrity[:] = np.frombuffer(integrity.read(), dtype="f4").reshape((world.height, world.width))
    world.velocity[:] = np.frombuffer(velocity.read(), dtype="f4").reshape((world.height, world.width, 2))
    world.placeholder_displaced_material[:] = np.rint(
        np.frombuffer(displaced.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_out.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.entity_id[:] = np.rint(
        np.frombuffer(resources.entity_out.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)



def _barrier_bits(pipeline) -> tuple[str, ...]:
    # liquid touches the framebuffer (visible illumination publish) and
    # uses indirect/buffer-update paths alongside the default sync.
    return (
        "SHADER_IMAGE_ACCESS_BARRIER_BIT",
        "TEXTURE_FETCH_BARRIER_BIT",
        "FRAMEBUFFER_BARRIER_BIT",
        "SHADER_STORAGE_BARRIER_BIT",
        "COMMAND_BARRIER_BIT",
        "BUFFER_UPDATE_BARRIER_BIT",
    )
