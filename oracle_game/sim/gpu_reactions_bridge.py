from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.types import CellFlag, ForceSource, Phase, ReactionType
from oracle_game.sim.gpu_collapse_dirty import COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER, COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER, COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER, COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER, _active_scheduler_gpu_authoritative, _ensure_material_flags_buffer, ensure_collapse_structure_dirty_tile_mask, ensure_collapse_structure_dirty_tile_queue, mark_collapse_structure_dirty_tiles_from_bridge_cell_core
from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    FORMAL_GPU_EMPTY_DEFERRED_BATCH,
    GPUDeferredActionBatch,
    GPUReactionBridgeInputLoads,
    GPUReactionResources,
    LOCAL_SIZE,
    MAX_EMITTED_LIGHTS,
)
from oracle_game.sim.gpu_timer_pack import unpack_cell_state, unpack_u8x4


def _load_authoritative_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    bridge_input_loads: GPUReactionBridgeInputLoads | None = None,
    reaction_group: str | None = None,
    profile_scope: str | None = None,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    if bridge_input_loads is None:
        bridge_input_loads = GPUReactionBridgeInputLoads()
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = bridge_input_loads.cell_core and "cell_core" in authoritative
    copy_gas = bridge_input_loads.gas and "gas_concentration" in authoritative
    copy_ambient = bridge_input_loads.ambient and "ambient_temperature" in authoritative
    copy_flow_velocity = bridge_input_loads.flow_velocity and "flow_velocity" in authoritative
    copy_cell_dose = bridge_input_loads.cell_dose and "cell_optical_dose" in authoritative
    copy_gas_dose = bridge_input_loads.gas_dose and "gas_optical_dose" in authoritative
    if not (
        copy_cell_core
        or copy_gas
        or copy_ambient
        or copy_flow_velocity
        or copy_cell_dose
        or copy_gas_dose
    ):
        return
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative input state")
    ran_copy = False
    ran_cell_copy = False
    if copy_cell_core:
        read_role_only_cell_core = pipeline._bridge_cell_core_read_role_only_load(reaction_group)
        with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_cell"):
            group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
            with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_cell_core"):
                program = pipeline.programs[
                    "load_bridge_cell_role" if read_role_only_cell_core else "load_bridge_cell"
                ]
                program["cell_grid_size"].value = (world.width, world.height)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                if read_role_only_cell_core:
                    cell_state_tex, _phase_tex, temp_tex, integrity_tex, _velocity_tex, _timer_tex = (
                        pipeline._current_cell_textures(resources)
                    )
                    cell_state_tex.bind_to_image(0, read=False, write=True)
                    temp_tex.bind_to_image(1, read=False, write=True)
                    integrity_tex.bind_to_image(2, read=False, write=True)
                else:
                    resources.cell_state_ping.bind_to_image(0, read=False, write=True)
                    resources.cell_state_pong.bind_to_image(1, read=False, write=True)
                    resources.temp_ping.bind_to_image(2, read=False, write=True)
                    resources.temp_pong.bind_to_image(3, read=False, write=True)
                    resources.integrity_ping.bind_to_image(4, read=False, write=True)
                    resources.integrity_pong.bind_to_image(5, read=False, write=True)
                if light_dose_guard_buffer is not None:
                    pipeline._run_light_dose_guarded_dispatch(
                        world,
                        resources,
                        program,
                        light_dose_guard_buffer,
                        group_x,
                        group_y,
                        1,
                    )
                else:
                    program.run(group_x, group_y, 1)
            with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_cell_aux"):
                program = pipeline.programs[
                    "load_bridge_cell_aux_role" if read_role_only_cell_core else "load_bridge_cell_aux"
                ]
                program["cell_grid_size"].value = (world.width, world.height)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                if read_role_only_cell_core:
                    _material_tex, _phase_tex, _temp_tex, _integrity_tex, velocity_tex, timer_tex = (
                        pipeline._current_cell_textures(resources)
                    )
                    velocity_tex.bind_to_image(0, read=False, write=True)
                    timer_tex.bind_to_image(1, read=False, write=True)
                else:
                    resources.velocity_ping.bind_to_image(0, read=False, write=True)
                    resources.velocity_pong.bind_to_image(1, read=False, write=True)
                    resources.timer_ping.bind_to_image(2, read=False, write=True)
                    resources.timer_pong.bind_to_image(3, read=False, write=True)
                if light_dose_guard_buffer is not None:
                    pipeline._run_light_dose_guarded_dispatch(
                        world,
                        resources,
                        program,
                        light_dose_guard_buffer,
                        group_x,
                        group_y,
                        1,
                    )
                else:
                    program.run(group_x, group_y, 1)
            ran_cell_copy = True
            ran_copy = True
    if copy_gas or copy_ambient or copy_flow_velocity:
        with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_gas"):
            program = pipeline.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["species_count"].value = int(world.gas_concentration.shape[0])
            program["copy_gas"].value = bool(copy_gas)
            program["copy_ambient"].value = bool(copy_ambient)
            program["copy_flow_velocity"].value = bool(copy_flow_velocity)
            bridge.textures["ambient_temperature"].use(location=0)
            bridge.textures["flow_velocity"].use(location=1)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=0)
            resources.gas_ping.bind_to_image(2, read=False, write=True)
            resources.gas_pong.bind_to_image(3, read=False, write=True)
            resources.ambient_ping.bind_to_image(4, read=False, write=True)
            resources.ambient_pong.bind_to_image(5, read=False, write=True)
            resources.flow_velocity_tex.bind_to_image(6, read=False, write=True)
            group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_z = int(world.gas_concentration.shape[0])
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    group_z,
                )
            else:
                program.run(group_x, group_y, group_z)
            ran_copy = True
    if copy_cell_dose or copy_gas_dose:
        with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_dose"):
            program = pipeline.programs["load_bridge_dose"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["light_count"].value = int(world.cell_optical_dose.shape[0])
            program["copy_cell_dose"].value = bool(copy_cell_dose)
            program["copy_gas_dose"].value = bool(copy_gas_dose)
            bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=0)
            bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=1)
            resources.cell_dose_tex.bind_to_image(0, read=False, write=True)
            resources.cell_dose_pong.bind_to_image(1, read=False, write=True)
            resources.gas_dose_tex.bind_to_image(2, read=False, write=True)
            resources.gas_dose_pong.bind_to_image(3, read=False, write=True)
            group_x = (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_z = int(world.cell_optical_dose.shape[0])
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    group_z,
                )
            else:
                program.run(group_x, group_y, group_z)
            ran_copy = True
    if ran_copy:
        if ran_cell_copy:
            with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_cell_sync"):
                pipeline._sync_compute_writes(bridge.ctx)
        else:
            pipeline._sync_compute_writes(bridge.ctx)



def _publish_bridge_cell_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    source_role: str | None = None,
    source_velocity_role: str | None = None,
    cell_reset_texture: Any | None = None,
    reaction_latched_texture: Any | None = None,
    cell_meta_texture: Any | None = None,
    packed_cell_meta_texture: Any | None = None,
    light_dose_guard_buffer: Any | None = None,
    mark_structure_dirty: bool = True,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative cell state")
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("reaction output", "cell_core")
        bridge.sync_world(world)
    cell_state_tex, _phase_tex, temp_tex, integrity_tex, velocity_tex, timer_tex = pipeline._cell_role_textures(
        resources,
        source_role or "pong",
    )
    if source_velocity_role is not None:
        velocity_tex = pipeline._cell_role_textures(resources, source_velocity_role)[4]
    fuse_structure_dirty_mark = False
    dirty_buffer = None
    dirty_count = None
    dirty_list = None
    dirty_dispatch_args = None
    material_flags_buffer = None
    material_count = 0
    if mark_structure_dirty and pipeline._formal_gpu_frame(world) and _active_scheduler_gpu_authoritative(world):
        dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_buffer is not None and dirty_queue is not None:
            dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
            material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
            fuse_structure_dirty_mark = True
    if mark_structure_dirty and not fuse_structure_dirty_mark:
        mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
            world,
            None,
            None,
            dispatch_guard_buffer=light_dose_guard_buffer,
            cell_state_texture=cell_state_tex,
        )
    program = pipeline.programs["publish_bridge_cell"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["use_cell_meta_texture"].value = cell_meta_texture is not None
    program["use_packed_cell_meta_texture"].value = packed_cell_meta_texture is not None
    program["mark_structure_dirty"].value = bool(fuse_structure_dirty_mark)
    program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
    program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
    program["tile_size"].value = int(world.active.tile_size)
    program["material_count"].value = int(material_count)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    cell_state_tex.use(location=0)
    temp_tex.use(location=2)
    integrity_tex.use(location=3)
    velocity_tex.use(location=4)
    timer_tex.use(location=5)
    (cell_reset_texture or resources.cell_reset_tex).use(location=6)
    (reaction_latched_texture or resources.reaction_latched_tex).use(location=7)
    (cell_meta_texture or resources.local_cell_meta_out).use(location=8)
    (packed_cell_meta_texture or resources.local_deferred_packed_out).use(location=9)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    if fuse_structure_dirty_mark:
        assert dirty_buffer is not None
        assert dirty_count is not None
        assert dirty_list is not None
        assert dirty_dispatch_args is not None
        assert material_flags_buffer is not None
        material_flags_buffer.bind_to_storage_buffer(binding=1)
        dirty_buffer.bind_to_storage_buffer(binding=2)
        dirty_count.bind_to_storage_buffer(binding=3)
        dirty_list.bind_to_storage_buffer(binding=4)
        dirty_dispatch_args.bind_to_storage_buffer(binding=5)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            program,
            light_dose_guard_buffer,
            group_x,
            group_y,
            1,
        )
    else:
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    if fuse_structure_dirty_mark:
        setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
        bridge.mark_gpu_authoritative(
            COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
        )
    bridge.mark_gpu_authoritative("cell_core", "material")



def _publish_bridge_gas_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    gas_texture: Any | None = None,
    ambient_texture: Any | None = None,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative gas state")
    program = pipeline.programs["publish_bridge_gas"]
    gas_texture = resources.gas_pong if gas_texture is None else gas_texture
    ambient_texture = resources.ambient_pong if ambient_texture is None else ambient_texture
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["species_count"].value = int(world.gas_concentration.shape[0])
    gas_texture.use(location=0)
    ambient_texture.use(location=1)
    bridge.textures["ambient_temperature"].bind_to_image(2, read=False, write=True)
    bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=0)
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_z = int(world.gas_concentration.shape[0])
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            program,
            light_dose_guard_buffer,
            group_x,
            group_y,
            group_z,
        )
    else:
        program.run(group_x, group_y, group_z)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("gas_concentration", "ambient_temperature")



def _publish_bridge_dose_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative optical dose state")
    light_count = int(world.cell_optical_dose.shape[0])
    cell_program = pipeline.programs["publish_bridge_cell_dose"]
    cell_program["cell_grid_size"].value = (world.width, world.height)
    cell_program["light_count"].value = light_count
    resources.cell_dose_pong.use(location=0)
    bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=0)
    cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            cell_program,
            light_dose_guard_buffer,
            cell_group_x,
            cell_group_y,
            light_count,
        )
    else:
        cell_program.run(cell_group_x, cell_group_y, light_count)
    gas_program = pipeline.programs["publish_bridge_gas_dose"]
    gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    gas_program["light_count"].value = light_count
    resources.gas_dose_pong.use(location=0)
    bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=0)
    gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            gas_program,
            light_dose_guard_buffer,
            gas_group_x,
            gas_group_y,
            light_count,
        )
    else:
        gas_program.run(gas_group_x, gas_group_y, light_count)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")



def _apply_flow_sources_to_bridge_velocity(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    light_dose_guard_buffer: Any | None = None,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative flow state")
    program = pipeline.programs["apply_bridge_flow_sources"]
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["gas_cell_size"].value = int(world.gas_cell_size)
    program["tile_size"].value = int(world.active.tile_size)
    program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    active_flow_source_layers = max(1, min(FLOW_SOURCE_LAYERS, int(flow_source_layers)))
    program["flow_source_layers"].value = active_flow_source_layers
    program["impulse_dt"].value = 1.0 / 60.0
    generation_validity = pipeline._flow_source_generation_validity_active(world)
    if pipeline._flow_source_generation_programs_enabled:
        program["flow_source_generation_validity_enabled"].value = generation_validity
        program["flow_source_generation"].value = int(resources.flow_source_generation)
    resources.flow_velocity_tex.use(location=0)
    resources.flow_source_tex.use(location=1)
    if pipeline._flow_source_generation_programs_enabled:
        resources.flow_source_generation_tex.use(location=2)
    bridge.textures["flow_velocity"].bind_to_image(2, read=False, write=True)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
    group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    if light_dose_guard_buffer is not None:
        pipeline._run_light_dose_guarded_dispatch(
            world,
            resources,
            program,
            light_dose_guard_buffer,
            group_x,
            group_y,
            1,
        )
    else:
        program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge._ensure_active_scheduler_programs()
    bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge.mark_gpu_authoritative("flow_velocity", "active_meta", "active_tile_ttl", "active_chunk_mask")
    pipeline._formal_active_mask_cache_key = None
    pipeline._copy_bridge_flow_velocity_to_reaction(world, resources)



def _publish_bridge_light_emitters(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative emitted light state")
    program = pipeline.programs["publish_bridge_light_emitters"]
    program["emitter_vec4_count"].value = int(MAX_EMITTED_LIGHTS * 2)
    program["counter_count"].value = 16
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=0)
    resources.light_emitter_count.bind_to_storage_buffer(binding=1)
    bridge.buffers["reaction_light_emitter"].bind_to_storage_buffer(binding=2)
    bridge.buffers["reaction_light_emitter_count"].bind_to_storage_buffer(binding=3)
    program.run((max(MAX_EMITTED_LIGHTS * 2, 16) + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative("reaction_light_emitter", "reaction_light_emitter_count")



def _download_cell_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    direct_core_outputs: bool = False,
    advance_velocity_role: bool = False,
    packed_local_cell_meta: bool = False,
    segment_cell_meta_already_accumulated: bool = False,
) -> None:
    if pipeline._formal_gpu_frame(world):
        rotate_formal_cell_roles = pipeline._formal_before_motion_cell_roles_active()
        if pipeline._formal_segment_batch_active():
            if not segment_cell_meta_already_accumulated:
                pipeline._accumulate_segment_cell_transient_state(
                    world,
                    resources,
                    direct_core_outputs=direct_core_outputs,
                    packed_local_cell_meta=packed_local_cell_meta,
                )
            if rotate_formal_cell_roles:
                pipeline._advance_formal_cell_read_role()
                if advance_velocity_role:
                    pipeline._advance_formal_velocity_read_role()
            else:
                pipeline._promote_cell_pong_to_ping(world, resources)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "cell")
        else:
            if rotate_formal_cell_roles:
                source_role = pipeline._formal_cell_write_role()
                source_velocity_role = (
                    pipeline._formal_velocity_write_role()
                    if advance_velocity_role
                    else pipeline._formal_velocity_read_role()
                )
                pipeline._publish_bridge_cell_state(
                    world,
                    resources,
                    source_role=source_role,
                    source_velocity_role=source_velocity_role,
                    cell_meta_texture=resources.local_cell_meta_out if direct_core_outputs else None,
                    packed_cell_meta_texture=resources.local_deferred_packed_out if packed_local_cell_meta else None,
                )
                pipeline._set_formal_cell_read_role(source_role)
                pipeline._set_formal_velocity_read_role(source_velocity_role)
            else:
                pipeline._publish_bridge_cell_state(
                    world,
                    resources,
                    cell_meta_texture=resources.local_cell_meta_out if direct_core_outputs else None,
                    packed_cell_meta_texture=resources.local_deferred_packed_out if packed_local_cell_meta else None,
                )
                pipeline._promote_cell_pong_to_ping(world, resources)
        pipeline.last_cpu_mirror_downloaded = False
        return
    pipeline.last_cpu_mirror_downloaded = True
    previous_material = world.material_id.copy()
    previous_phase = world.phase.copy()
    previous_island_id = world.island_id.copy()
    cell_state = np.frombuffer(resources.cell_state_pong.read(), dtype="u4").reshape((world.height, world.width))
    material, phase, packed_flags = unpack_cell_state(cell_state)
    world.material_id[:] = material
    world.phase[:] = phase
    world.cell_flags[:] = packed_flags
    world.cell_temperature[:] = np.frombuffer(resources.temp_pong.read(), dtype="f4").reshape((world.height, world.width))
    world.integrity[:] = np.frombuffer(resources.integrity_pong.read(), dtype="f4").reshape((world.height, world.width))
    world.timer_pack[:] = unpack_u8x4(
        np.frombuffer(resources.timer_pong.read(), dtype="u4").reshape((world.height, world.width))
    )
    if hasattr(resources, "velocity_pong"):
        world.velocity[:] = np.frombuffer(resources.velocity_pong.read(), dtype="f4").reshape((world.height, world.width, 2))
    cell_reset_mask = np.frombuffer(resources.cell_reset_tex.read(), dtype="f4").reshape((world.height, world.width)) > 0.5
    reaction_latched_mask = (
        np.frombuffer(resources.reaction_latched_tex.read(), dtype="f4").reshape((world.height, world.width)) > 0.5
    )
    world.cell_flags[cell_reset_mask] = 0
    world.cell_flags[reaction_latched_mask] |= np.uint8(int(CellFlag.REACTION_LATCHED))
    emptied_mask = cell_reset_mask & (world.material_id <= 0)
    if np.any(emptied_mask):
        world.velocity[emptied_mask] = 0.0
        ambient_cells = world.sample_ambient_to_cells()
        world.cell_temperature[emptied_mask] = ambient_cells[emptied_mask]
    non_placeholder_mask = (world.material_id <= 0) | ~np.vectorize(world._shadow_material_is_placeholder, otypes=[np.bool_])(
        world.material_id
    )
    world.entity_id[non_placeholder_mask] = 0
    world.placeholder_displaced_material[non_placeholder_mask] = 0
    invalid_island_mask = (world.island_id > 0) & (
        (world.phase != int(Phase.FALLING_ISLAND)) | (world.material_id <= 0)
    )
    changed_mask = (world.material_id != previous_material) | (world.phase != previous_phase)
    if np.any(changed_mask):
        for y, x in np.argwhere(changed_mask):
            previous_participates = world._cell_participates_in_collapse(
                int(previous_material[y, x]),
                int(previous_phase[y, x]),
            )
            current_participates = world._cell_participates_in_collapse(
                int(world.material_id[y, x]),
                int(world.phase[y, x]),
            )
            if previous_participates or current_participates:
                world._mark_collapse_dirty_rect(int(x), int(y), int(x) + 1, int(y) + 1)
    touched_island_ids = np.unique(previous_island_id[changed_mask | invalid_island_mask])
    world.island_id[invalid_island_mask] = 0
    world._refresh_island_records_for_ids(touched_island_ids.tolist())



def _download_gas_state(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if pipeline._formal_gpu_frame(world):
        if pipeline._formal_segment_batch_active():
            pipeline._promote_gas_pong_to_ping(world, resources)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "gas")
        else:
            pipeline._publish_bridge_gas_state(world, resources)
            pipeline._promote_gas_pong_to_ping(world, resources)
        pipeline.last_cpu_mirror_downloaded = False
        return
    pipeline.last_cpu_mirror_downloaded = True
    world.gas_concentration[:] = np.maximum(
        np.frombuffer(resources.gas_pong.read(), dtype="f4").reshape(world.gas_concentration.shape),
        0.0,
    )



def _download_dose_state(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if pipeline._formal_gpu_frame(world):
        if pipeline._formal_segment_batch_active():
            pipeline._promote_dose_pong_to_ping(world, resources)
            pipeline._mark_formal_bridge_publish_pending(world, resources, "dose")
        else:
            pipeline._publish_bridge_dose_state(world, resources)
            pipeline._promote_dose_pong_to_ping(world, resources)
        pipeline.last_cpu_mirror_downloaded = False
        return
    pipeline.last_cpu_mirror_downloaded = True
    world.cell_optical_dose[:] = np.maximum(
        np.frombuffer(resources.cell_dose_pong.read(), dtype="f4").reshape(world.cell_optical_dose.shape),
        0.0,
    )
    world.gas_optical_dose[:] = np.maximum(
        np.frombuffer(resources.gas_dose_pong.read(), dtype="f4").reshape(world.gas_optical_dose.shape),
        0.0,
    )



def _download_deferred_batch(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> GPUDeferredActionBatch:
    shape = (world.height, world.width, 4)
    if pipeline._formal_gpu_frame(world):
        unsupported = pipeline._unsupported_deferred_action_indices(world)
        if unsupported:
            raise RuntimeError(
                "GPU reaction pipeline encountered unsupported deferred action indices "
                f"{unsupported}; CPU fallback is disabled"
            )
        if pipeline._formal_segment_batch_active():
            pipeline._mark_formal_bridge_publish_pending(world, resources, "light_emitters")
        else:
            pipeline._publish_bridge_light_emitters(world, resources)
        return FORMAL_GPU_EMPTY_DEFERRED_BATCH
    reaction_counts = np.frombuffer(resources.light_emitter_count.read(), dtype=np.uint32, count=16).copy()
    emitted_light_count = int(reaction_counts[0])
    emitted_light_count = max(0, min(emitted_light_count, MAX_EMITTED_LIGHTS))
    gpu_local_action_counts = reaction_counts[1:9].copy()
    if emitted_light_count > 0:
        raw_emitters = np.frombuffer(resources.light_emitter_buffer.read(), dtype="f4").reshape(
            (MAX_EMITTED_LIGHTS, 2, 4)
        )
        emitted_lights = np.zeros((emitted_light_count, 8), dtype=np.float32)
        emitted_lights[:, 0:4] = raw_emitters[:emitted_light_count, 0, :]
        emitted_lights[:, 4:8] = raw_emitters[:emitted_light_count, 1, :]
    else:
        emitted_lights = np.zeros((0, 8), dtype=np.float32)
    return GPUDeferredActionBatch(
        action_lo=np.rint(np.frombuffer(resources.trigger_lo_tex.read(), dtype="f4").reshape(shape)).astype(np.int32),
        action_hi=np.rint(np.frombuffer(resources.trigger_hi_tex.read(), dtype="f4").reshape(shape)).astype(np.int32),
        scale_lo=np.frombuffer(resources.deferred_scale_lo_tex.read(), dtype="f4").reshape(shape).copy(),
        scale_hi=np.frombuffer(resources.deferred_scale_hi_tex.read(), dtype="f4").reshape(shape).copy(),
        emitted_lights=emitted_lights,
        emitted_material_mask=(
            np.frombuffer(resources.emitted_material_mask_tex.read(), dtype="f4").reshape((world.height, world.width)) > 0.5
        ),
        gpu_local_action_counts=gpu_local_action_counts,
    )



def _unsupported_deferred_action_indices(pipeline, world: "WorldEngine") -> list[int]:
    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    unsupported: list[int] = []
    for index, row in enumerate(action_table):
        reaction_type_id = int(row["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            continue
        if reaction_type_id in {
            int(ReactionType.HARM.value),
            int(ReactionType.MODIFY_TEMPERATURE.value),
            int(ReactionType.CONVERT_MATERIAL.value),
        }:
            continue
        if reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
            continue
        if reaction_type_id == int(ReactionType.EMIT_LIGHT.value) and int(row["light_type_id"]) >= 0:
            continue
        if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
            continue
        unsupported.append(int(index))
    return unsupported



def _append_flow_sources_from_gpu(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    may_have_flow_sources: bool = True,
    light_dose_guard_buffer: Any | None = None,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
) -> None:
    if not may_have_flow_sources:
        return
    if pipeline._formal_gpu_frame(world):
        pipeline._apply_flow_sources_to_bridge_velocity(
            world,
            resources,
            light_dose_guard_buffer=light_dose_guard_buffer,
            flow_source_layers=flow_source_layers,
        )
        pipeline.last_cpu_mirror_downloaded = False
        return
    flow = np.frombuffer(resources.flow_source_tex.read(), dtype="f4").reshape(
        (FLOW_SOURCE_LAYERS, world.gas_height, world.gas_width, 4)
    )
    source_layers, ys, xs = np.nonzero(flow[..., 3] > 0.0)
    if source_layers.size == 0:
        return
    emitted: list[ForceSource] = []
    for layer, gy, gx in zip(source_layers.tolist(), ys.tolist(), xs.tolist()):
        direction_x = float(flow[layer, gy, gx, 0])
        direction_y = float(flow[layer, gy, gx, 1])
        radius = float(flow[layer, gy, gx, 2])
        strength = float(flow[layer, gy, gx, 3])
        norm = float(np.hypot(direction_x, direction_y))
        if norm <= 1.0e-6 or radius <= 0.0 or strength <= 0.0:
            continue
        cell_x = int(gx) * int(world.gas_cell_size) + int(world.gas_cell_size) // 2
        cell_y = int(gy) * int(world.gas_cell_size) + int(world.gas_cell_size) // 2
        emitted.append(
            ForceSource(
                x=float(np.clip(cell_x, 0, world.width - 1)),
                y=float(np.clip(cell_y, 0, world.height - 1)),
                direction=(direction_x / norm, direction_y / norm),
                radius=radius,
                strength=strength,
                lifetime=1.0 / 60.0,
            )
        )
    if not emitted:
        return
    world.force_sources.extend(emitted)
    max_radius = int(np.ceil(max(source.radius for source in emitted)))
    min_x = max(0, int(min(source.x for source in emitted)) - max_radius)
    min_y = max(0, int(min(source.y for source in emitted)) - max_radius)
    max_x = min(world.width, int(max(source.x for source in emitted)) + max_radius + 1)
    max_y = min(world.height, int(max(source.y for source in emitted)) + max_radius + 1)
    world._mark_active_rect_runtime(min_x, min_y, max_x, max_y)
