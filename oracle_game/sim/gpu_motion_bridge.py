from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_motion import GPUMotionResources

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE, pack_island_runtime_upload

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE,
    MAX_MATERIALS,
    POWDER_RESERVATION_DTYPE,
    FALLING_ISLAND_RESERVATION_DTYPE
)


def _pack_cell_state_texture(
    material_id: np.ndarray,
    phase: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    material = np.clip(np.asarray(material_id), 0, 0xFFFF).astype(np.uint32)
    phase_u32 = np.clip(np.asarray(phase), 0, 0xFF).astype(np.uint32)
    flags_u32 = np.clip(np.asarray(flags), 0, 0xFF).astype(np.uint32)
    return np.ascontiguousarray(material | (phase_u32 << 16) | (flags_u32 << 24))


def _upload_inputs(pipeline, world: "WorldEngine", resources: GPUMotionResources, solve_tile_mask: np.ndarray) -> None:
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["cell_core"]:
        resources.cell_state_tex.write(
            _pack_cell_state_texture(world.material_id, world.phase, world.cell_flags).tobytes()
        )
        resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
    if upload_plan["flow_velocity"]:
        resources.flow_tex.write(world.flow_velocity.astype("f4").tobytes())
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    if upload_plan["entity_id"]:
        resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
    if upload_plan["placeholder_displaced_material"]:
        resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
    if upload_plan["active_tile_ttl"]:
        resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
    else:
        pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
    pipeline._compact_active_tiles(world, resources)
    pipeline._upload_material_rule_params(world, resources)


def _cpu_upload_plan(pipeline, world: "WorldEngine") -> dict[str, bool]:
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    heat_pipeline = getattr(getattr(world, "heat_solver", None), "gpu_pipeline", None)
    deferred_cell_core = bool(
        formal_gpu_frame
        and getattr(heat_pipeline, "_deferred_cell_core_frame_id", None)
        == int(getattr(world, "frame_id", 0))
        and isinstance(getattr(heat_pipeline, "_motion_handoff_candidate", None), dict)
    )
    required_resources = [
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "flow_velocity",
        "ambient_temperature",
        "active_tile_ttl",
    ]
    if not deferred_cell_core:
        required_resources.insert(0, "cell_core")
    world._require_gpu_authoritative_resources(
        "motion input",
        *required_resources,
    )
    return {
        "cell_core": not (formal_gpu_frame and ("cell_core" in authoritative or deferred_cell_core)),
        "island_id": not (formal_gpu_frame and "island_id" in authoritative),
        "entity_id": not (formal_gpu_frame and "entity_id" in authoritative),
        "placeholder_displaced_material": not (
            formal_gpu_frame and "placeholder_displaced_material" in authoritative
        ),
        "flow_velocity": not (formal_gpu_frame and "flow_velocity" in authoritative),
        "ambient_temperature": not (formal_gpu_frame and "ambient_temperature" in authoritative),
        "active_tile_ttl": not (formal_gpu_frame and "active_tile_ttl" in authoritative),
    }


def _record_cpu_upload_plan(pipeline, upload_plan: dict[str, bool]) -> None:
    pipeline.last_cpu_cell_state_upload_skipped = not upload_plan["cell_core"]
    pipeline.last_cpu_island_id_upload_skipped = not upload_plan["island_id"]
    pipeline.last_cpu_entity_id_upload_skipped = not upload_plan["entity_id"]
    pipeline.last_cpu_displaced_material_upload_skipped = not upload_plan["placeholder_displaced_material"]
    pipeline.last_cpu_flow_velocity_upload_skipped = not upload_plan["flow_velocity"]
    pipeline.last_cpu_ambient_upload_skipped = not upload_plan["ambient_temperature"]
    pipeline.last_cpu_active_upload_skipped = not upload_plan["active_tile_ttl"]


def _load_authoritative_active_tile_mask(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    expansion_radius: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU motion pipeline requires bridge active scheduler resources")
    program = pipeline.programs["load_active_tiles"]
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["expansion_radius"].value = int(expansion_radius)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
    resources.active_tile_tex.bind_to_image(1, read=False, write=True)
    program.run(
        (world.active.tile_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.active.tile_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(bridge.ctx)


def _upload_material_rule_params(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    world.bridge.sync_rule_tables(world)
    material_table = world.bridge.shadow_typed_tables["material_table"]
    table_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
    if resources.material_params_signature == table_signature:
        return
    params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
    contact = np.zeros((MAX_MATERIALS, 4), dtype="f4")
    falling = np.zeros((MAX_MATERIALS, 2, 4), dtype="f4")
    count = min(MAX_MATERIALS, material_table.shape[0])
    params[:count, 0] = material_table[:count]["max_dda_step"].astype("f4")
    params[:count, 1] = material_table[:count]["gravity_scale"].astype("f4")
    params[:count, 2] = material_table[:count]["wind_coupling"].astype("f4")
    params[:count, 3] = material_table[:count]["drag_scale"].astype("f4")
    contact[:count, 0] = material_table[:count]["friction"].astype("f4")
    contact[:count, 1] = material_table[:count]["elasticity"].astype("f4")
    contact[:count, 2] = material_table[:count]["powder_solver_kind_id"].astype("f4")
    falling[:count, 0, 0] = material_table[:count]["default_phase"].astype("f4")
    falling[:count, 0, 1] = material_table[:count]["render_group_id"].astype("f4")
    falling[:count, 0, 2] = material_table[:count]["falling_island_break_kind_id"].astype("f4")
    falling[:count, 0, 3] = material_table[:count]["powder_generation_id"].astype("f4")
    falling[:count, 1, 0] = material_table[:count]["base_integrity"].astype("f4")
    falling[:count, 1, 1] = material_table[:count]["spawn_temperature"].astype("f4")
    resources.material_params.write(params.tobytes())
    resources.material_contact_params.write(contact.tobytes())
    resources.material_falling_params.write(falling.reshape((MAX_MATERIALS * 2, 4)).tobytes())
    resources.material_params_signature = table_signature

# ``_formal_gpu_frame`` inherited from GPUPipelineBase.


def _bridge_authoritative_cell_blockers(pipeline, world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return (
        pipeline._formal_gpu_frame(world)
        and {"cell_core", "entity_id", "placeholder_displaced_material"}.issubset(authoritative)
    )


def _bridge_authoritative_powder_inputs(pipeline, world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return (
        pipeline._formal_gpu_frame(world)
        and pipeline._bridge_context_active(world)
        and {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(authoritative)
    )


def _bind_bridge_cell_blockers(pipeline, world: "WorldEngine", *, cell_binding: int = 8) -> None:
    bridge = world.bridge
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=cell_binding)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=cell_binding + 1)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=cell_binding + 2)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=cell_binding + 3)


def _bridge_authoritative_island_state(pipeline, world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return pipeline._formal_gpu_frame(world) and {"cell_core", "island_id"}.issubset(authoritative)


def _bind_bridge_island_state(pipeline, world: "WorldEngine", *, cell_binding: int = 7) -> None:
    bridge = world.bridge
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=cell_binding)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=cell_binding + 1)


def _bridge_context_active(pipeline, world: "WorldEngine") -> bool:
    return world.bridge.ctx is not None and world.bridge.ctx is pipeline._active_context(world)


def _active_context(pipeline, world: "WorldEngine") -> Any:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    return ctx


def _load_authoritative_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    group_x: int,
    group_y: int,
    *,
    use_existing_active_tile_dispatch: bool = False,
    load_gas_inputs: bool = True,
) -> bool:
    if not pipeline._formal_gpu_frame(world):
        return False
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative input state")
    if not pipeline._bridge_context_active(world):
        raise RuntimeError("GPU motion pipeline cannot consume authoritative bridge state from a separate GL context")

    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    copy_flow = bool(load_gas_inputs and "flow_velocity" in authoritative)
    copy_ambient = bool(load_gas_inputs and "ambient_temperature" in authoritative)
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_flow or copy_ambient):
        return False

    active_tile_indirect = bool(pipeline._active_scheduler_gpu_authoritative(world))
    if active_tile_indirect and not use_existing_active_tile_dispatch:
        pipeline._compact_active_tiles(world, resources)

    if copy_cell_core:
        program = pipeline.programs["load_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        program["copy_cell_core"].value = bool(copy_cell_core)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        resources.cell_state_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_tex.bind_to_image(3, read=False, write=True)
        resources.temp_tex.bind_to_image(4, read=False, write=True)
        resources.timer_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_tex.bind_to_image(6, read=False, write=True)
        if active_tile_indirect:
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            pipeline._run_active_tile_indirect(program, resources, "motion bridge cell load")
        else:
            program.run(group_x, group_y, 1)

    if copy_island_id or copy_entity_id or copy_displaced:
        program = pipeline.programs["load_bridge_cell_aux"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        program["copy_island_id"].value = bool(copy_island_id)
        program["copy_entity_id"].value = bool(copy_entity_id)
        program["copy_displaced_material"].value = bool(copy_displaced)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        resources.island_id_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_tex.bind_to_image(2, read=False, write=True)
        if active_tile_indirect:
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            pipeline._run_active_tile_indirect(program, resources, "motion bridge aux load")
        else:
            program.run(group_x, group_y, 1)

    if copy_flow or copy_ambient:
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program = pipeline.programs["load_bridge_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["copy_flow_velocity"].value = bool(copy_flow)
        program["copy_ambient"].value = bool(copy_ambient)
        bridge.textures["flow_velocity"].use(location=0)
        bridge.textures["ambient_temperature"].use(location=1)
        resources.flow_tex.bind_to_image(2, read=False, write=True)
        resources.ambient_tex.bind_to_image(3, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)

    pipeline._sync_compute_writes(bridge.ctx)
    return False


def _load_authoritative_integrate_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    group_x: int,
    group_y: int,
    *,
    use_existing_active_tile_dispatch: bool = False,
) -> bool:
    if not pipeline._formal_gpu_frame(world):
        return False
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative input state")
    if not pipeline._bridge_context_active(world):
        raise RuntimeError("GPU motion pipeline cannot consume authoritative bridge state from a separate GL context")

    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_flow = "flow_velocity" in authoritative
    active_tile_indirect = bool(pipeline._active_scheduler_gpu_authoritative(world))
    if not (copy_cell_core or copy_flow):
        return False

    if copy_cell_core and copy_flow and active_tile_indirect:
        # integrate_velocity can decode the authoritative bridge state in its
        # existing active-tile dispatch. It also refreshes resident textures so
        # subsequent falling-island stages see the same state as before.
        return True

    if active_tile_indirect and not use_existing_active_tile_dispatch:
        pipeline._compact_active_tiles(world, resources)

    if copy_cell_core:
        program = pipeline.programs["load_bridge_integrate_inputs"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        resources.cell_state_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_tex.bind_to_image(1, read=False, write=True)
        if active_tile_indirect:
            resources.active_tile_count.bind_to_storage_buffer(binding=1)
            resources.active_tile_list.bind_to_storage_buffer(binding=2)
            pipeline._run_active_tile_indirect(program, resources, "motion integrate bridge input load")
        else:
            program.run(group_x, group_y, 1)

    if copy_flow:
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program = pipeline.programs["load_bridge_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["copy_flow_velocity"].value = True
        program["copy_ambient"].value = False
        bridge.textures["flow_velocity"].use(location=0)
        bridge.textures["ambient_temperature"].use(location=1)
        resources.flow_tex.bind_to_image(2, read=False, write=True)
        resources.ambient_tex.bind_to_image(3, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)

    pipeline._sync_compute_writes(bridge.ctx)
    return False


def _load_authoritative_materialization_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    use_existing_active_tile_dispatch: bool,
) -> bool:
    if not (
        pipeline._bridge_authoritative_powder_inputs(world)
        and pipeline._active_scheduler_gpu_authoritative(world)
    ):
        return False
    bridge = world.bridge
    if not use_existing_active_tile_dispatch:
        pipeline._compact_active_tiles(world, resources)
    program = pipeline.programs["load_bridge_materialization_inputs"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
    resources.active_tile_count.bind_to_storage_buffer(binding=4)
    resources.active_tile_list.bind_to_storage_buffer(binding=5)
    resources.cell_state_tex.bind_to_image(0, read=False, write=True)
    resources.island_id_tex.bind_to_image(1, read=False, write=True)
    pipeline._run_active_tile_indirect(
        program,
        resources,
        "motion materialization bridge input load",
    )
    pipeline._sync_compute_writes(bridge.ctx)
    return True


def _publish_bridge_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    output_textures: bool,
    velocity_out_active_only: bool = False,
    active_tile_indirect: bool = False,
    active_tile_count_buffer: Any | None = None,
    active_tile_list_buffer: Any | None = None,
    active_tile_dispatch_args: Any | None = None,
    use_powder_apply_touch_sources: bool = False,
    use_packed_powder_aux: bool = False,
    sparse_powder_bridge_publish: bool = False,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative output state")
        return
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot publish authoritative state from a separate GL context")
        return

    cell_state_tex = resources.cell_state_out_tex if output_textures else resources.cell_state_tex
    velocity_tex = resources.velocity_out_tex
    temp_tex = resources.temp_out_tex if output_textures else resources.temp_tex
    timer_tex = resources.timer_out_tex if output_textures else resources.timer_tex
    integrity_tex = resources.integrity_out_tex if output_textures else resources.integrity_tex
    island_tex = resources.island_id_out_tex if output_textures else resources.island_id_tex
    entity_tex = resources.entity_id_out_tex if output_textures else resources.entity_id_tex
    displaced_tex = resources.displaced_out_tex if output_textures else resources.displaced_tex

    program = pipeline.programs["publish_bridge_cell"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["velocity_out_active_only"].value = bool(velocity_out_active_only)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    program["use_powder_apply_touch_sources"].value = bool(use_powder_apply_touch_sources)
    program["use_packed_powder_aux"].value = bool(use_packed_powder_aux)
    program["sparse_powder_bridge_publish"].value = bool(sparse_powder_bridge_publish)
    program["use_bridge_input_core"].value = bool(
        use_powder_apply_touch_sources and pipeline._bridge_authoritative_powder_inputs(world)
    )
    program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
    cell_state_tex.use(location=0)
    velocity_tex.use(location=3)
    temp_tex.use(location=4)
    timer_tex.use(location=5)
    integrity_tex.use(location=6)
    island_tex.use(location=7)
    entity_tex.use(location=8)
    displaced_tex.use(location=9)
    resources.velocity_tex.use(location=10)
    resources.active_tile_tex.use(location=11)
    if use_powder_apply_touch_sources:
        resources.cell_state_tex.use(location=12)
        resources.velocity_tex.use(location=15)
        resources.temp_tex.use(location=16)
        resources.timer_tex.use(location=17)
        resources.integrity_tex.use(location=18)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
    tile_count_buffer = active_tile_count_buffer if active_tile_count_buffer is not None else resources.active_tile_count
    tile_list_buffer = active_tile_list_buffer if active_tile_list_buffer is not None else resources.active_tile_list
    tile_count_buffer.bind_to_storage_buffer(binding=4)
    tile_list_buffer.bind_to_storage_buffer(binding=5)
    if use_powder_apply_touch_sources:
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=6)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=7)
        resources.powder_target_winner.bind_to_storage_buffer(binding=8)
    if active_tile_indirect:
        pipeline._run_active_tile_indirect(
            program,
            resources,
            "bridge cell publish",
            dispatch_args=active_tile_dispatch_args,
        )
    else:
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )


def _publish_bridge_velocity_words(
    pipeline,
    world: "WorldEngine",
    resources: GPUMotionResources,
    *,
    active_tile_indirect: bool,
) -> bool:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not (
        pipeline._formal_gpu_frame(world)
        and bool(active_tile_indirect)
        and bridge.enabled
        and bridge.ctx is not None
        and pipeline._bridge_context_active(world)
        and "cell_core" in bridge.gpu_authoritative_resources
    ):
        return False
    if "cell_core" not in bridge.buffers:
        return False

    program = pipeline.programs["publish_bridge_velocity_word"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    resources.velocity_out_tex.use(location=0)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    pipeline._run_active_tile_indirect(program, resources, "bridge velocity word publish")
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("cell_core")
    return True


def _publish_bridge_island_id(pipeline, world: "WorldEngine", resources: GPUMotionResources, island_tex: Any) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative island state")
        return
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot publish island state from a separate GL context")
        return
    program = pipeline.programs["publish_bridge_island_id"]
    program["cell_grid_size"].value = (world.width, world.height)
    island_tex.use(location=0)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=0)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("island_id")


def publish_bridge_falling_island_reservations(
    pipeline,
    world: "WorldEngine",
    reservation_count: int,
) -> bool:
    reservation_count = int(reservation_count)
    if reservation_count < 0:
        raise ValueError("reservation_count must be non-negative")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island reservations")
        return False
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot publish island reservations from a separate GL context")
        return False
    pipeline._ensure_programs(bridge.ctx)
    resources = pipeline._ensure_resources(world)
    required_bytes = max(4, reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize)
    bridge_buffer = bridge.buffers["island_reservation"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["island_reservation"] = bridge_buffer
    elif not pipeline._formal_gpu_frame(world):
        bridge_buffer.orphan(required_bytes)
    if pipeline._formal_gpu_frame(world):
        bridge.buffers["island_reservation_count"].write(np.array([0], dtype=np.int32).tobytes())
        if reservation_count > 0:
            with pipeline._profile_pass(world, "island_reservation_publish_bridge"):
                program = pipeline.programs["publish_falling_island_reservations"]
                program["reservation_capacity"].value = reservation_count
                resources.island_reservations.bind_to_storage_buffer(binding=0)
                resources.island_reservation_count.bind_to_storage_buffer(binding=1)
                bridge_buffer.bind_to_storage_buffer(binding=2)
                bridge.buffers["island_reservation_count"].bind_to_storage_buffer(binding=3)
                pipeline._run_island_reservation_indirect(
                    world,
                    resources,
                    program,
                    "falling island reservation publish",
                    reservation_capacity=reservation_count,
                    invocations_per_group=256,
                )
                bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
        bridge.mark_gpu_authoritative("island_reservation")
        return True
    if reservation_count > 0:
        bridge.ctx.copy_buffer(
            bridge_buffer,
            resources.island_reservations,
            size=reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
        )
    bridge.buffers["island_reservation_count"].write(np.array([reservation_count], dtype=np.int32).tobytes())
    return True


def publish_bridge_powder_reservations(
    pipeline,
    world: "WorldEngine",
    reservation_capacity: int,
) -> bool:
    reservation_capacity = int(reservation_capacity)
    if reservation_capacity < 0:
        raise ValueError("reservation_capacity must be non-negative")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for powder reservations")
        return False
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot publish powder reservations from a separate GL context")
        return False
    resources = pipeline._ensure_resources(world)
    required_bytes = max(4, reservation_capacity * POWDER_RESERVATION_DTYPE.itemsize)
    bridge_buffer = bridge.buffers["powder_reservation"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["powder_reservation"] = bridge_buffer
    if pipeline._formal_gpu_frame(world):
        with pipeline._profile_pass(world, "powder_publish_bridge"):
            resources.powder_reservations, bridge.buffers["powder_reservation"] = (
                bridge_buffer,
                resources.powder_reservations,
            )
            resources.powder_reservation_count, bridge.buffers["powder_reservation_count"] = (
                bridge.buffers["powder_reservation_count"],
                resources.powder_reservation_count,
            )
            bridge.clear_gpu_authoritative(
                "powder_reservation_compact",
                "powder_reservation_cpu_mirror",
            )
            bridge.mark_gpu_authoritative(
                "powder_reservation",
                "powder_reservation_standard",
            )
        return True
    else:
        # Keep capacity reserved by incremental collapse runtime admission so
        # later slots can append without discarding surviving islands.
        bridge_buffer.orphan(bridge_buffer.size)
    with pipeline._profile_pass(world, "powder_publish_bridge"):
        if reservation_capacity > 0:
            bridge.ctx.copy_buffer(
                bridge_buffer,
                resources.powder_reservations,
                size=reservation_capacity * POWDER_RESERVATION_DTYPE.itemsize,
            )
        bridge.ctx.copy_buffer(bridge.buffers["powder_reservation_count"], resources.powder_reservation_count, size=4)
    return True


def publish_bridge_compact_powder_reservations(
    pipeline,
    world: "WorldEngine",
    reservation_capacity: int,
) -> bool:
    reservation_capacity = int(reservation_capacity)
    if reservation_capacity < 0:
        raise ValueError("reservation_capacity must be non-negative")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU motion pipeline requires bridge GPU resources for compact powder reservations")
    if not pipeline._bridge_context_active(world):
        raise RuntimeError("GPU motion pipeline cannot publish compact powder reservations from a separate GL context")
    resources = pipeline._ensure_resources(world)
    required_bytes = max(4, reservation_capacity * 24)
    bridge_buffer = bridge.buffers["powder_reservation_compact"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["powder_reservation_compact"] = bridge_buffer
    with pipeline._profile_pass(world, "powder_publish_bridge"):
        resources.powder_compact_reservations, bridge.buffers["powder_reservation_compact"] = (
            bridge_buffer,
            resources.powder_compact_reservations,
        )
        resources.powder_reservation_count, bridge.buffers["powder_reservation_count"] = (
            bridge.buffers["powder_reservation_count"],
            resources.powder_reservation_count,
        )
        bridge.clear_gpu_authoritative(
            "powder_reservation_standard",
            "powder_reservation_cpu_mirror",
        )
        bridge.mark_gpu_authoritative(
            "powder_reservation",
            "powder_reservation_compact",
        )
    return True


def publish_bridge_falling_island_runtime_from_reservations(
    pipeline,
    world: "WorldEngine",
    reservation_count: int,
) -> bool:
    reservation_count = int(reservation_count)
    if reservation_count < 0:
        raise ValueError("reservation_count must be non-negative")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime")
        return False
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot publish island runtime from a separate GL context")
        return False
    formal_frame = pipeline._formal_gpu_frame(world)
    pipeline._ensure_programs(bridge.ctx)
    resources = pipeline._ensure_resources(world)
    required_bytes = max(4, reservation_count * ISLAND_RUNTIME_DTYPE.itemsize)
    bridge_buffer = bridge.buffers["island_runtime"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["island_runtime"] = bridge_buffer
    else:
        bridge_buffer.orphan(required_bytes)
    if not formal_frame:
        bridge_buffer.write(np.zeros((required_bytes,), dtype=np.uint8).tobytes())
    bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())
    if reservation_count > 0:
        with pipeline._profile_pass(world, "island_runtime_publish_bridge"):
            program = pipeline.programs["publish_falling_island_runtime"]
            program["reservation_count"].value = int(reservation_count)
            program["use_reservation_count_buffer"].value = bool(formal_frame)
            program["cell_grid_size"].value = (world.width, world.height)
            program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
            program["paging_buffer_origin"].value = (
                int(world.paging.buffer_origin_x),
                int(world.paging.buffer_origin_y),
            )
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            bridge_buffer.bind_to_storage_buffer(binding=1)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
            resources.island_reservation_count.bind_to_storage_buffer(binding=3)
            if formal_frame:
                pipeline._run_island_reservation_indirect(
                    world,
                    resources,
                    program,
                    "falling island runtime publish",
                    reservation_capacity=reservation_count,
                )
            else:
                group_x = (reservation_count + LOCAL_SIZE - 1) // LOCAL_SIZE
                program.run(group_x, 1, 1)
            bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
    pipeline.last_published_island_runtime_capacity = reservation_count
    if formal_frame:
        bridge.mark_gpu_authoritative("island_runtime")
    return True


def seed_bridge_falling_island_runtime_from_cpu(pipeline, world: "WorldEngine") -> int:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime seeding")
        return 0
    if not pipeline._bridge_context_active(world):
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU motion pipeline cannot seed island runtime from a separate GL context")
        return 0
    runtime = pack_island_runtime_upload(world)
    runtime_count = int(runtime.shape[0])
    required_bytes = max(4, runtime.nbytes)
    bridge_buffer = bridge.buffers["island_runtime"]
    if bridge_buffer.size < required_bytes:
        bridge_buffer.release()
        bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers["island_runtime"] = bridge_buffer
    else:
        bridge_buffer.orphan(required_bytes)
    if runtime.nbytes > 0:
        bridge_buffer.write(runtime.tobytes())
    bridge.buffers["island_runtime_count"].write(np.array([runtime_count], dtype=np.int32).tobytes())
    pipeline.last_published_island_runtime_capacity = runtime_count
    if pipeline._formal_gpu_frame(world):
        bridge.mark_gpu_authoritative("island_runtime")
    return runtime_count


def _download_outputs(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
    return np.rint(
        np.frombuffer(resources.powder_target_tex.read(), dtype="f4").reshape((world.height, world.width, 4))[
            :, :, :2
        ]
    ).astype(np.int32)


def _download_velocity_output(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
    velocity = world.velocity.copy()
    velocity_out = np.frombuffer(resources.velocity_out_tex.read(), dtype="f4").reshape(world.velocity.shape)
    active_tiles = np.frombuffer(resources.active_tile_tex.read(), dtype="f4").reshape(
        (world.active.tile_height, world.active.tile_width)
    )
    tile_size = int(world.active.tile_size)
    for tile_y, tile_x in np.argwhere(active_tiles > 0.5):
        x0 = int(tile_x) * tile_size
        y0 = int(tile_y) * tile_size
        x1 = min(world.width, x0 + tile_size)
        y1 = min(world.height, y0 + tile_size)
        velocity[y0:y1, x0:x1] = velocity_out[y0:y1, x0:x1]
    return velocity


def _upload_powder_apply_state(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    upload_plan = pipeline._cpu_upload_plan(world)
    pipeline._record_cpu_upload_plan(upload_plan)
    if upload_plan["cell_core"]:
        resources.cell_state_tex.write(
            _pack_cell_state_texture(world.material_id, world.phase, world.cell_flags).tobytes()
        )
        resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
        resources.temp_tex.write(world.cell_temperature.astype("f4").tobytes())
        resources.timer_tex.write(world.timer_pack.astype("f4").tobytes())
        resources.integrity_tex.write(world.integrity.astype("f4").tobytes())
    if upload_plan["island_id"]:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    if upload_plan["entity_id"]:
        resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
    if upload_plan["placeholder_displaced_material"]:
        resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
    if upload_plan["ambient_temperature"]:
        resources.ambient_tex.write(world.ambient_temperature.astype("f4").tobytes())


def _download_powder_apply_state(pipeline, world: "WorldEngine", resources: GPUMotionResources) -> None:
    packed_cell_state = np.frombuffer(
        resources.cell_state_out_tex.read(),
        dtype=np.uint32,
    ).reshape((world.height, world.width))
    world.material_id[:] = (packed_cell_state & 0xFFFF).astype(np.int32)
    world.phase[:] = ((packed_cell_state >> 16) & 0xFF).astype(np.uint8)
    world.cell_flags[:] = ((packed_cell_state >> 24) & 0xFF).astype(np.uint8)
    world.velocity[:] = np.frombuffer(resources.velocity_out_tex.read(), dtype="f4").reshape(world.velocity.shape)
    world.cell_temperature[:] = np.frombuffer(resources.temp_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    world.timer_pack[:] = np.rint(
        np.frombuffer(resources.timer_out_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.uint8)
    world.integrity[:] = np.frombuffer(resources.integrity_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.entity_id[:] = np.rint(
        np.frombuffer(resources.entity_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.placeholder_displaced_material[:] = np.rint(
        np.frombuffer(resources.displaced_out_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
