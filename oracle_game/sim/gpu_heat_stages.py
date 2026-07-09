from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _active_scheduler_gpu_authoritative,
    _ensure_material_flags_buffer,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
    mark_collapse_structure_dirty_tiles_from_bridge_cell_core,
)
from oracle_game.sim.gpu_heat import (
    FREEZE_COLD_NEIGHBOR_THRESHOLD,
    LOCAL_SIZE,
    MAX_GAS_SPECIES,
    GPUHeatResources,
    GPUHeatStageTargets,
)
from oracle_game.types import Phase


def step(
    pipeline,
    world: "WorldEngine",
    dt: float,
    *,
    solve_tile_mask: np.ndarray,
    ambient_iterations: int,
) -> GPUHeatStageTargets:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU heat pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    with pipeline._profile_pass(world, "upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
    gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "load_bridge_inputs"):
        pipeline._load_authoritative_bridge_inputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
    with pipeline._profile_pass(world, "ambient_diffuse"):
        pipeline._run_ambient_diffuse(world, resources, gas_group_x, gas_group_y, iterations=ambient_iterations)
    with pipeline._profile_pass(world, "cell_heat"):
        pipeline._run_cell_heat(world, dt, resources, group_x, group_y)
    with pipeline._profile_pass(world, "ambient_exchange"):
        pipeline._run_ambient_exchange(world, dt, resources, group_x, group_y)
    with pipeline._profile_pass(world, "ambient_feedback"):
        pipeline._run_ambient_feedback(world, dt, resources, gas_group_x, gas_group_y)
    with pipeline._profile_pass(world, "phase_targets"):
        pipeline._run_phase_targets(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "boil_targets"):
        pipeline._run_boil_targets(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "condense_targets"):
        pipeline._run_condense_targets(world, resources, gas_group_x, gas_group_y)
    with pipeline._profile_pass(world, "apply_cell_targets"):
        pipeline._run_apply_cell_targets(world, dt, resources, group_x, group_y)
    with pipeline._profile_pass(world, "apply_gas_targets"):
        pipeline._run_apply_gas_targets(world, dt, resources, gas_group_x, gas_group_y)
    with pipeline._profile_pass(world, "apply_condense_cells"):
        pipeline._run_apply_condense_cells(world, resources, group_x, group_y)
    with pipeline._profile_pass(world, "publish_bridge_outputs"):
        pipeline._publish_bridge_outputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
    pipeline.last_cpu_mirror_downloaded = not (
        getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        with pipeline._profile_pass(world, "download_outputs"):
            return pipeline._download_outputs(world, resources)
    return pipeline._empty_stage_targets(world)


# ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.


def _load_authoritative_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    copy_ambient = "ambient_temperature" in authoritative
    copy_gas = "gas_concentration" in authoritative
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient or copy_gas):
        return
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative input state")

    if copy_cell_core:
        program = pipeline.programs["load_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["copy_cell_core"].value = bool(copy_cell_core)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        resources.material_tex.bind_to_image(4, read=False, write=True)
        resources.phase_tex.bind_to_image(5, read=False, write=True)
        resources.cell_flags_tex.bind_to_image(6, read=False, write=True)
        resources.timer_tex.bind_to_image(7, read=False, write=True)
        resources.temp_ping.bind_to_image(0, read=False, write=True)
        resources.integrity_tex.bind_to_image(1, read=False, write=True)
        resources.velocity_tex.bind_to_image(2, read=False, write=True)
        program.run(group_x, group_y, 1)

    if copy_island_id or copy_entity_id or copy_displaced:
        program = pipeline.programs["load_bridge_cell_aux"]
        program["cell_grid_size"].value = (world.width, world.height)
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

    if copy_ambient or copy_gas:
        program = pipeline.programs["load_bridge_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["copy_ambient"].value = bool(copy_ambient)
        program["copy_gas"].value = bool(copy_gas)
        bridge.textures["ambient_temperature"].use(location=0)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
        resources.ambient_ping.bind_to_image(2, read=False, write=True)
        resources.gas_tex.bind_to_image(3, read=False, write=True)
        program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

    pipeline._sync_compute_writes(bridge.ctx)


def _run_cell_heat(pipeline, world: "WorldEngine", dt: float, resources: GPUHeatResources, group_x: int, group_y: int) -> None:
    program = pipeline.programs["cell_heat"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_ping.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.temp_pong.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_ambient_exchange(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["ambient_exchange"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_pong.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.temp_ping.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_ambient_diffuse(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
    *,
    iterations: int,
) -> None:
    if iterations <= 0:
        return
    ctx = world.bridge.ctx
    assert ctx is not None
    program = pipeline.programs["ambient_diffuse"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    for _ in range(iterations):
        resources.ambient_ping.use(location=3)
        resources.ambient_pong.bind_to_image(4, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        pipeline._sync_compute_writes(ctx)
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping


def _run_ambient_feedback(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    program = pipeline.programs["ambient_feedback"]
    ctx = world.bridge.ctx
    assert ctx is not None
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["gas_cell_size"].value = world.gas_cell_size
    program["tile_size"].value = world.active.tile_size
    program["dt"].value = dt
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.temp_pong.use(location=3)
    resources.ambient_ping.use(location=4)
    resources.ambient_pong.bind_to_image(5, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_phase_targets(pipeline, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
    program = pipeline.programs["phase_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
    pipeline._set_uniform_if_present(program, "freeze_cold_neighbor_threshold", FREEZE_COLD_NEIGHBOR_THRESHOLD)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.phase_tex.use(location=4)
    resources.temp_ping.use(location=5)
    resources.phase_target_tex.bind_to_image(6, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_boil_targets(pipeline, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
    program = pipeline.programs["boil_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.temp_ping.use(location=4)
    resources.boil_target_tex.bind_to_image(5, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_condense_targets(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    program = pipeline.programs["condense_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.gas_params.bind_to_storage_buffer(binding=3)
    resources.gas_tex.use(location=4)
    resources.ambient_pong.use(location=5)
    resources.condense_target_tex.bind_to_image(6, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_cell_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    with pipeline._profile_pass(world, "apply_cell_targets.main"):
        program = pipeline.programs["apply_cell_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        pipeline._set_uniform_if_present(program, "dt", dt)
        pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.phase_target_tex.use(location=3)
        resources.phase_tex.use(location=4)
        resources.cell_flags_tex.use(location=5)
        resources.timer_tex.use(location=6)
        resources.boil_target_tex.use(location=7)
        resources.temp_ping.use(location=8)
        resources.integrity_tex.use(location=9)
        resources.island_id_tex.use(location=10)
        resources.entity_id_tex.use(location=11)
        resources.displaced_tex.use(location=12)
        resources.ambient_pong.use(location=22)
        resources.velocity_tex.use(location=23)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.timer_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_pong.bind_to_image(4, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
        resources.island_id_out_tex.bind_to_image(6, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(7, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "apply_cell_targets.aux"):
        pipeline._run_apply_cell_aux_targets(world, dt, resources, group_x, group_y)


def _run_apply_cell_aux_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["apply_cell_aux_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "dt", dt)
    pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
    pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_phase_params.bind_to_storage_buffer(binding=3)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.phase_target_tex.use(location=3)
    resources.phase_tex.use(location=4)
    resources.boil_target_tex.use(location=5)
    resources.integrity_tex.use(location=6)
    resources.displaced_tex.use(location=7)
    resources.velocity_tex.use(location=8)
    resources.displaced_out_tex.bind_to_image(0, read=False, write=True)
    resources.velocity_out_tex.bind_to_image(1, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_gas_targets(
    pipeline,
    world: "WorldEngine",
    dt: float,
    resources: GPUHeatResources,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    program = pipeline.programs["apply_gas_targets"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    pipeline._set_uniform_if_present(program, "dt", dt)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.material_tex.use(location=1)
    resources.active_tile_tex.use(location=2)
    resources.material_response_params.bind_to_storage_buffer(binding=7)
    resources.gas_params.bind_to_storage_buffer(binding=3)
    resources.gas_tex.use(location=4)
    resources.boil_target_tex.use(location=5)
    resources.condense_target_tex.use(location=6)
    resources.material_out_tex.use(location=8)
    resources.gas_out_tex.bind_to_image(0, read=False, write=True)
    program.run(gas_group_x, gas_group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _run_apply_condense_cells(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    with pipeline._profile_pass(world, "apply_condense_cells.main"):
        program = pipeline.programs["apply_condense_cells"]
        ctx = world.bridge.ctx
        assert ctx is not None
        pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        pipeline._set_uniform_if_present(
            program,
            "gas_species_count",
            min(world.gas_concentration.shape[0], MAX_GAS_SPECIES),
        )
        pipeline._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        pipeline._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=8)
        resources.material_out_tex.use(location=4)
        resources.phase_out_tex.use(location=5)
        resources.cell_flags_out_tex.use(location=6)
        resources.timer_out_tex.use(location=9)
        resources.temp_pong.use(location=10)
        resources.integrity_out_tex.use(location=11)
        resources.island_id_out_tex.use(location=12)
        resources.entity_id_out_tex.use(location=13)
        resources.displaced_out_tex.use(location=22)
        resources.velocity_out_tex.use(location=23)
        resources.condense_target_tex.use(location=24)
        resources.material_tex.bind_to_image(0, read=False, write=True)
        resources.phase_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
        resources.timer_tex.bind_to_image(3, read=False, write=True)
        resources.temp_ping.bind_to_image(4, read=False, write=True)
        resources.integrity_tex.bind_to_image(5, read=False, write=True)
        resources.displaced_tex.bind_to_image(6, read=False, write=True)
        resources.velocity_tex.bind_to_image(7, read=False, write=True)
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)
        resources.island_id_tex, resources.island_id_out_tex = resources.island_id_out_tex, resources.island_id_tex
        resources.entity_id_tex, resources.entity_id_out_tex = resources.entity_id_out_tex, resources.entity_id_tex


def _run_apply_condense_cell_aux(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
) -> None:
    program = pipeline.programs["apply_condense_cell_aux"]
    ctx = world.bridge.ctx
    assert ctx is not None
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
    pipeline._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
    pipeline._set_uniform_if_present(program, "tile_size", world.active.tile_size)
    pipeline._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
    resources.active_tile_tex.use(location=2)
    resources.gas_params.bind_to_storage_buffer(binding=8)
    resources.material_out_tex.use(location=4)
    resources.phase_out_tex.use(location=5)
    resources.island_id_out_tex.use(location=12)
    resources.displaced_out_tex.use(location=22)
    resources.velocity_out_tex.use(location=23)
    resources.condense_target_tex.use(location=24)
    resources.displaced_tex.bind_to_image(0, read=False, write=True)
    resources.velocity_tex.bind_to_image(1, read=False, write=True)
    program.run(group_x, group_y, 1)
    pipeline._sync_compute_writes(ctx)


def _download_outputs(pipeline, world: "WorldEngine", resources: GPUHeatResources) -> GPUHeatStageTargets:
    world.material_id[:] = np.rint(
        np.frombuffer(resources.material_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.phase[:] = np.rint(
        np.frombuffer(resources.phase_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.uint8)
    world.cell_flags[:] = np.rint(
        np.frombuffer(resources.cell_flags_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.uint8)
    world.timer_pack[:] = np.rint(
        np.frombuffer(resources.timer_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
    ).astype(np.uint8)
    world.cell_temperature[:] = np.frombuffer(resources.temp_ping.read(), dtype="f4").reshape((world.height, world.width))
    world.integrity[:] = np.frombuffer(resources.integrity_tex.read(), dtype="f4").reshape((world.height, world.width))
    world.island_id[:] = np.rint(
        np.frombuffer(resources.island_id_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.entity_id[:] = np.rint(
        np.frombuffer(resources.entity_id_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.placeholder_displaced_material[:] = np.rint(
        np.frombuffer(resources.displaced_tex.read(), dtype="f4").reshape((world.height, world.width))
    ).astype(np.int32)
    world.velocity[:] = np.frombuffer(resources.velocity_tex.read(), dtype="f4").reshape((world.height, world.width, 2))
    world.ambient_temperature[:] = np.frombuffer(resources.ambient_pong.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
    world.gas_concentration[:] = np.frombuffer(resources.gas_out_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
    return GPUHeatStageTargets(
        phase_targets=np.rint(
            np.frombuffer(resources.phase_target_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32),
        boil_targets=np.rint(
            np.frombuffer(resources.boil_target_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32),
        condense_targets=(
            np.frombuffer(resources.condense_target_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
            > 0.5
        ),
    )


def _empty_stage_targets(pipeline, world: "WorldEngine") -> GPUHeatStageTargets:
    if pipeline._formal_gpu_frame(world):
        return GPUHeatStageTargets.empty_sentinel()
    return GPUHeatStageTargets(
        phase_targets=np.zeros((world.height, world.width), dtype=np.int32),
        boil_targets=np.zeros((world.height, world.width), dtype=np.int32),
        condense_targets=np.zeros(world.gas_concentration.shape, dtype=np.bool_),
    )


def _publish_bridge_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPUHeatResources,
    group_x: int,
    group_y: int,
    gas_group_x: int,
    gas_group_y: int,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative heat state")
    fuse_structure_dirty_mark = False
    dirty_buffer = None
    dirty_count = None
    dirty_list = None
    dirty_dispatch_args = None
    material_flags_buffer = None
    material_count = 0
    if pipeline._formal_gpu_frame(world) and _active_scheduler_gpu_authoritative(world):
        dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
        dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
        if dirty_buffer is not None and dirty_queue is not None:
            dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
            material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
            fuse_structure_dirty_mark = True
    with pipeline._profile_pass(world, "publish_bridge_outputs.collapse_dirty_mark"):
        if not fuse_structure_dirty_mark:
            mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
                world,
                resources.material_tex,
                resources.phase_tex,
            )
    with pipeline._profile_pass(world, "publish_bridge_outputs.cell"):
        cell_program = pipeline.programs["publish_bridge_cell"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        cell_program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
        cell_program["tile_size"].value = int(world.active.tile_size)
        cell_program["material_count"].value = int(material_count)
        cell_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        cell_program["mark_structure_dirty"].value = bool(fuse_structure_dirty_mark)
        cell_program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
        cell_program["material_tex"].value = 0
        cell_program["phase_tex"].value = 1
        cell_program["flags_tex"].value = 2
        cell_program["timer_tex"].value = 3
        cell_program["temp_tex"].value = 4
        cell_program["integrity_tex"].value = 5
        cell_program["island_tex"].value = 6
        cell_program["entity_tex"].value = 7
        cell_program["displaced_tex"].value = 8
        cell_program["velocity_tex"].value = 9
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.cell_flags_tex.use(location=2)
        resources.timer_tex.use(location=3)
        resources.temp_ping.use(location=4)
        resources.integrity_tex.use(location=5)
        resources.island_id_tex.use(location=6)
        resources.entity_id_tex.use(location=7)
        resources.displaced_tex.use(location=8)
        resources.velocity_tex.use(location=9)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        if fuse_structure_dirty_mark:
            assert dirty_buffer is not None
            assert dirty_count is not None
            assert dirty_list is not None
            assert dirty_dispatch_args is not None
            assert material_flags_buffer is not None
            material_flags_buffer.bind_to_storage_buffer(binding=4)
            dirty_buffer.bind_to_storage_buffer(binding=5)
            dirty_count.bind_to_storage_buffer(binding=6)
            dirty_list.bind_to_storage_buffer(binding=7)
            dirty_dispatch_args.bind_to_storage_buffer(binding=8)
    if not bool(getattr(world, "phase_c_defer_cell_publish", False)):
        cell_program.run(group_x, group_y, 1)

    with pipeline._profile_pass(world, "publish_bridge_outputs.gas"):
        gas_program = pipeline.programs["publish_bridge_gas"]
        gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        gas_program["species_count"].value = int(world.gas_concentration.shape[0])
        gas_program["ambient_tex"].value = 0
        gas_program["gas_tex"].value = 2
        resources.ambient_pong.use(location=0)
        resources.gas_out_tex.use(location=2)
        bridge.textures["ambient_temperature"].bind_to_image(1, read=False, write=True)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=4)
        gas_program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

    with pipeline._profile_pass(world, "publish_bridge_outputs.sync"):
        pipeline._sync_compute_writes(bridge.ctx)
        if fuse_structure_dirty_mark:
            setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
            bridge.mark_gpu_authoritative(
                COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
                COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
            )
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "gas_concentration",
        )


# ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
# from :class:`GPUPipelineBase`; the heat pass uses the default barrier
# bits (image-access | texture-fetch | shader-storage).
