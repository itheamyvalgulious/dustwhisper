from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_heat import (
    LOCAL_SIZE,
    MAX_GAS_SPECIES,
    MAX_MATERIALS,
    GPUHeatResources,
)


def release(pipeline) -> None:
    if pipeline.resources is None:
        return
    for resource in (
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
        pipeline.resources.island_id_tex,
        pipeline.resources.island_id_out_tex,
        pipeline.resources.entity_id_tex,
        pipeline.resources.entity_id_out_tex,
        pipeline.resources.displaced_tex,
        pipeline.resources.displaced_out_tex,
        pipeline.resources.velocity_tex,
        pipeline.resources.velocity_out_tex,
        pipeline.resources.temp_ping,
        pipeline.resources.temp_pong,
        pipeline.resources.phase_target_tex,
        pipeline.resources.boil_target_tex,
        pipeline.resources.gas_tex,
        pipeline.resources.gas_out_tex,
        pipeline.resources.condense_target_tex,
        pipeline.resources.ambient_ping,
        pipeline.resources.ambient_pong,
        pipeline.resources.active_tile_tex,
        pipeline.resources.material_params,
        pipeline.resources.material_response_params,
        pipeline.resources.material_phase_params,
        pipeline.resources.gas_params,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None


def _ensure_resources(pipeline, world: "WorldEngine") -> GPUHeatResources:
    ctx = world.bridge.ctx
    assert ctx is not None
    signature = (world.width, world.height, world.gas_width, world.gas_height, world.gas_concentration.shape[0])
    if pipeline.resources is not None and pipeline.resources.signature == signature:
        return pipeline.resources
    pipeline.release()
    gas_count = signature[4]
    material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    material_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    phase_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    phase_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    cell_flags_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    cell_flags_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    timer_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
    timer_out_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
    integrity_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    integrity_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    island_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    island_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    entity_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    displaced_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    displaced_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    velocity_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
    velocity_out_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
    temp_ping = ctx.texture((world.width, world.height), 1, dtype="f4")
    temp_pong = ctx.texture((world.width, world.height), 1, dtype="f4")
    phase_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    boil_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
    gas_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
    gas_out_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
    condense_target_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
    ambient_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
    ambient_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
    active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
    for texture in (
        material_tex,
        material_out_tex,
        phase_tex,
        phase_out_tex,
        cell_flags_tex,
        cell_flags_out_tex,
        timer_tex,
        timer_out_tex,
        integrity_tex,
        integrity_out_tex,
        island_id_tex,
        island_id_out_tex,
        entity_id_tex,
        entity_id_out_tex,
        displaced_tex,
        displaced_out_tex,
        velocity_tex,
        velocity_out_tex,
        temp_ping,
        temp_pong,
        phase_target_tex,
        boil_target_tex,
        gas_tex,
        gas_out_tex,
        condense_target_tex,
        ambient_ping,
        ambient_pong,
        active_tile_tex,
    ):
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
    material_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
    material_response_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
    material_phase_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
    gas_params = ctx.buffer(reserve=MAX_GAS_SPECIES * 4 * 4, dynamic=True)
    pipeline.resources = GPUHeatResources(
        signature=signature,
        material_tex=material_tex,
        material_out_tex=material_out_tex,
        phase_tex=phase_tex,
        phase_out_tex=phase_out_tex,
        cell_flags_tex=cell_flags_tex,
        cell_flags_out_tex=cell_flags_out_tex,
        timer_tex=timer_tex,
        timer_out_tex=timer_out_tex,
        integrity_tex=integrity_tex,
        integrity_out_tex=integrity_out_tex,
        island_id_tex=island_id_tex,
        island_id_out_tex=island_id_out_tex,
        entity_id_tex=entity_id_tex,
        entity_id_out_tex=entity_id_out_tex,
        displaced_tex=displaced_tex,
        displaced_out_tex=displaced_out_tex,
        velocity_tex=velocity_tex,
        velocity_out_tex=velocity_out_tex,
        temp_ping=temp_ping,
        temp_pong=temp_pong,
        phase_target_tex=phase_target_tex,
        boil_target_tex=boil_target_tex,
        gas_tex=gas_tex,
        gas_out_tex=gas_out_tex,
        condense_target_tex=condense_target_tex,
        ambient_ping=ambient_ping,
        ambient_pong=ambient_pong,
        active_tile_tex=active_tile_tex,
        material_params=material_params,
        material_response_params=material_response_params,
        material_phase_params=material_phase_params,
        gas_params=gas_params,
    )
    return pipeline.resources


def _upload_inputs(pipeline, world: "WorldEngine", resources: GPUHeatResources, solve_tile_mask: np.ndarray) -> None:
    world.bridge.sync_rule_tables(world)
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    world._require_gpu_authoritative_resources(
        "heat input",
        "cell_core",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "ambient_temperature",
        "gas_concentration",
        "active_tile_ttl",
    )
    upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
    upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
    upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
    upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
    upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
    upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
    upload_active_from_cpu = not (formal_gpu_frame and "active_tile_ttl" in authoritative)
    pipeline.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
    pipeline.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
    pipeline.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
    pipeline.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
    pipeline.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
    pipeline.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
    pipeline.last_cpu_active_upload_skipped = not upload_active_from_cpu
    if upload_cell_state_from_cpu:
        resources.material_tex.write(world.material_id.astype("f4").tobytes())
        resources.phase_tex.write(world.phase.astype("f4").tobytes())
        resources.cell_flags_tex.write(world.cell_flags.astype("f4").tobytes())
        resources.timer_tex.write(world.timer_pack.astype("f4").tobytes())
        resources.integrity_tex.write(world.integrity.astype("f4").tobytes())
        resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
        resources.velocity_out_tex.write(world.velocity.astype("f4").tobytes())
        resources.temp_ping.write(world.cell_temperature.astype("f4").tobytes())
        resources.temp_pong.write(world.cell_temperature.astype("f4").tobytes())
    if upload_island_id_from_cpu:
        resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
    if upload_entity_id_from_cpu:
        resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
    if upload_displaced_from_cpu:
        resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
    if upload_gas_from_cpu:
        resources.gas_tex.write(world.gas_concentration.astype("f4").tobytes())
        resources.gas_out_tex.write(world.gas_concentration.astype("f4").tobytes())
    if upload_ambient_from_cpu:
        resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
        resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
    if upload_active_from_cpu:
        resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
    else:
        pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=1)
    material_table = world.bridge.shadow_typed_tables["material_table"]
    material_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
    if resources.material_params_signature != material_signature:
        params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        response_params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        phase_params = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
        count = min(MAX_MATERIALS, material_table.shape[0])
        params[:count, 0] = material_table[:count]["conductivity"]
        params[:count, 1] = material_table[:count]["ambient_exchange_rate"]
        params[:count, 2] = material_table[:count]["melt_point"]
        params[:count, 3] = material_table[:count]["boil_point"]
        response_params[:count, 0] = material_table[:count]["heat_capacity"]
        response_params[:count, 1] = material_table[:count]["base_integrity"]
        response_params[:count, 2] = material_table[:count]["spawn_temperature"]
        response_params[:count, 3] = material_table[:count]["render_group_id"].astype("f4")
        phase_params[:count, 0] = material_table[:count]["default_phase"]
        phase_params[:count, 1] = material_table[:count]["melt_to_material_id"]
        phase_params[:count, 2] = material_table[:count]["freeze_to_material_id"]
        boil_species = material_table[:count]["boil_to_gas_species_id"].astype(np.int32)
        phase_params[:count, 3] = np.where(boil_species >= 0, boil_species + 1, 0)
        resources.material_params.write(params.tobytes())
        resources.material_response_params.write(response_params.tobytes())
        resources.material_phase_params.write(phase_params.tobytes())
        resources.material_params_signature = material_signature
    gas_table = world.bridge.shadow_typed_tables["gas_table"]
    gas_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
    if resources.gas_params_signature != gas_signature:
        gas_params = np.zeros((MAX_GAS_SPECIES, 4), dtype="f4")
        count = min(MAX_GAS_SPECIES, gas_table.shape[0])
        gas_params[:count, 0] = gas_table[:count]["condense_point"]
        gas_params[:count, 1] = gas_table[:count]["condense_to_material_id"].astype("f4")
        resources.gas_params.write(gas_params.tobytes())
        resources.gas_params_signature = gas_signature


def _load_authoritative_active_tile_mask(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    *,
    expansion_radius: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU heat pipeline requires bridge active scheduler resources")
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
