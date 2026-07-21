from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_motion import (
    LOCAL_SIZE
)
from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _active_scheduler_gpu_authoritative as _collapse_active_scheduler_gpu_authoritative,
    _ensure_material_flags_buffer,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
)
from oracle_game.types import Phase


def can_consume_deferred_heat_core(pipeline, world: "WorldEngine") -> bool:
    if (
        not pipeline._formal_gpu_frame(world)
        or world.bridge.ctx is None
    ):
        return False
    if not pipeline._bridge_context_active(world) or not _collapse_active_scheduler_gpu_authoritative(world):
        return False
    return (
        ensure_collapse_structure_dirty_tile_mask(world) is not None
        and ensure_collapse_structure_dirty_tile_queue(world) is not None
    )


def can_consume_reaction_handoff(pipeline, world: "WorldEngine") -> bool:
    return bool(
        getattr(world, "reaction_motion_handoff_active", False)
        and pipeline.can_consume_deferred_heat_core(world)
    )


def _terminal_integrated_handoff(world: "WorldEngine") -> tuple[Any, Any] | None:
    reaction_pipeline = getattr(getattr(world, "reaction_solver", None), "gpu_pipeline", None)
    heat_pipeline = getattr(getattr(world, "heat_solver", None), "gpu_pipeline", None)
    reaction_handoff = getattr(reaction_pipeline, "_motion_handoff_candidate", None)
    if not (
        isinstance(reaction_handoff, dict)
        and reaction_handoff.get("terminal_integrated") is True
        and int(reaction_handoff.get("frame_id", -1)) == int(getattr(world, "frame_id", 0))
    ):
        return None
    return reaction_pipeline, heat_pipeline


def _finish_terminal_integrated_handoff(pipeline, reaction_pipeline: Any, heat_pipeline: Any) -> None:
    reaction_pipeline._motion_handoff_candidate = None
    heat_pipeline._motion_handoff_candidate = None
    heat_pipeline._deferred_cell_core_frame_id = None
    pipeline.last_cpu_mirror_downloaded = False


def _integrate_reaction_handoff(
    pipeline,
    world: "WorldEngine",
    resources: Any,
    candidate: dict[str, Any],
    dt: float,
) -> None:
    ctx = world.bridge.ctx
    assert ctx is not None
    dirty_mask = ensure_collapse_structure_dirty_tile_mask(world)
    dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
    if dirty_mask is None or dirty_queue is None:
        raise RuntimeError("reaction-to-motion handoff requires collapse dirty queue resources")
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    material_flags, material_count = _ensure_material_flags_buffer(world)
    program = pipeline.programs["integrate_reaction_handoff"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["gas_cell_size"].value = int(world.gas_cell_size)
    program["material_count"].value = int(material_count)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    program["dt"].value = float(dt)
    cell_state = candidate.get("cell_state")
    base_flags = candidate.get("base_flags")
    reaction_meta = candidate.get("meta")
    apply_reaction_meta = reaction_meta is not None
    program["use_cell_state_texture"].value = cell_state is not None
    program["use_base_flags_texture"].value = base_flags is not None
    program["apply_reaction_meta"].value = bool(apply_reaction_meta)
    program["clear_reaction_latched"].value = bool(
        pipeline._reaction_latch_handoff_clear_enabled
    )
    material = candidate.get("material")
    phase = candidate.get("phase")
    sampler_fallback = candidate["temp"]
    (material if material is not None else sampler_fallback).use(location=0)
    (phase if phase is not None else sampler_fallback).use(location=1)
    candidate["temp"].use(location=2)
    candidate["integrity"].use(location=3)
    candidate["velocity"].use(location=4)
    candidate["timer"].use(location=5)
    (reaction_meta if apply_reaction_meta else sampler_fallback).use(location=6)
    world.bridge.textures["flow_velocity"].use(location=8)
    resources.active_tile_tex.use(location=9)
    (base_flags if base_flags is not None else sampler_fallback).use(location=10)
    (cell_state if cell_state is not None else resources.cell_state_tex).use(location=11)
    world.bridge.textures["material"].bind_to_image(0, read=False, write=True)
    world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    resources.material_params.bind_to_storage_buffer(binding=1)
    material_flags.bind_to_storage_buffer(binding=2)
    dirty_mask.bind_to_storage_buffer(binding=3)
    dirty_count.bind_to_storage_buffer(binding=4)
    dirty_list.bind_to_storage_buffer(binding=5)
    dirty_dispatch_args.bind_to_storage_buffer(binding=6)
    program.run(
        (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(ctx)
    if pipeline._reaction_latch_handoff_clear_enabled:
        setattr(
            world,
            "_reaction_latches_handoff_cleared_frame_id",
            int(getattr(world, "frame_id", 0)),
        )
    setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
    world.bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    )


def step(pipeline, world: "WorldEngine", dt: float, *, solve_tile_mask: np.ndarray) -> np.ndarray:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
    pipeline._reset_pass_profile()
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    setattr(world, "_reaction_latches_handoff_cleared_frame_id", None)
    with pipeline._profile_pass(world, "powder_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask)
    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "powder_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=True,
            load_gas_inputs=False,
        )
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
    reaction_pipeline = getattr(getattr(world, "reaction_solver", None), "gpu_pipeline", None)
    heat_pipeline = getattr(getattr(world, "heat_solver", None), "gpu_pipeline", None)
    reaction_handoff = getattr(reaction_pipeline, "_motion_handoff_candidate", None)
    heat_handoff = getattr(heat_pipeline, "_motion_handoff_candidate", None)
    handoff = reaction_handoff if isinstance(reaction_handoff, dict) else heat_handoff
    terminal_handoff = _terminal_integrated_handoff(world)
    if terminal_handoff is not None:
        _finish_terminal_integrated_handoff(pipeline, *terminal_handoff)
        return
    use_reaction_handoff = bool(
        isinstance(handoff, dict)
        and int(handoff.get("frame_id", -1)) == int(getattr(world, "frame_id", 0))
        and pipeline.can_consume_reaction_handoff(world)
    )
    deferred_heat_core = bool(
        getattr(heat_pipeline, "_deferred_cell_core_frame_id", None)
        == int(getattr(world, "frame_id", 0))
    )
    if (
        deferred_heat_core
        and isinstance(handoff, dict)
        and handoff.get("cell_state") is None
        and handoff.get("base_flags") is None
    ):
        raise RuntimeError("deferred heat cell core requires packed cell state or a base flags texture")
    if deferred_heat_core and not use_reaction_handoff:
        raise RuntimeError("deferred heat cell core requires a same-frame motion handoff candidate")
    with pipeline._profile_pass(world, "integrate_load_bridge_inputs"):
        use_bridge_inputs = False
        if not use_reaction_handoff:
            use_bridge_inputs = pipeline._load_authoritative_integrate_inputs(
                world,
                resources,
                group_x,
                group_y,
                use_existing_active_tile_dispatch=True,
            )
    if use_reaction_handoff:
        with pipeline._profile_pass(world, "integrate_reaction_handoff"):
            pipeline._integrate_reaction_handoff(world, resources, handoff, dt)
        reaction_pipeline._motion_handoff_candidate = None
        heat_pipeline._motion_handoff_candidate = None
        heat_pipeline._deferred_cell_core_frame_id = None
        pipeline.last_cpu_mirror_downloaded = False
        return
    program = pipeline.programs["integrate_velocity"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = world.active.tile_size
    program["gas_cell_size"].value = world.gas_cell_size
    program["dt"].value = dt
    program["use_bridge_inputs"].value = bool(use_bridge_inputs)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    resources.cell_state_tex.use(location=1)
    resources.velocity_tex.use(location=2)
    if use_bridge_inputs:
        world.bridge.textures["flow_velocity"].use(location=3)
        world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=6)
    else:
        resources.flow_tex.use(location=3)
    resources.active_tile_tex.use(location=4)
    resources.velocity_out_tex.bind_to_image(5, read=False, write=True)
    resources.cell_state_tex.bind_to_image(6, read=False, write=True)
    resources.velocity_tex.bind_to_image(7, read=False, write=True)
    with pipeline._profile_pass(world, "integrate_velocity"):
        pipeline._run_active_tile_indirect(program, resources, "integrate velocity")
        pipeline._sync_compute_writes(ctx)
    with pipeline._profile_pass(world, "integrate_publish_bridge"):
        active_tile_indirect = pipeline._formal_gpu_frame(world)
        if use_bridge_inputs:
            world.bridge.mark_gpu_authoritative("cell_core")
        elif not pipeline._publish_bridge_velocity_words(
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
