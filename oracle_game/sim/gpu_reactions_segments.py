from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.sim.gpu_collapse_dirty import _active_scheduler_gpu_authoritative
from oracle_game.sim.gpu_reactions import (
    DIRECT_CORE_OUTPUT_REACTION_GROUPS,
    GPUReactionResources,
    LIGHT_DOSE_GUARD_BUFFER,
    LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING,
    LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING,
    LOCAL_SIZE,
    _SHADER_SUBS,
)


def _active_scheduler_gpu_authoritative(pipeline, world: "WorldEngine") -> bool:
    return (
        pipeline._formal_gpu_frame(world)
        and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
    )



def _formal_light_dose_guard_buffer(pipeline, world: "WorldEngine") -> Any | None:
    if not pipeline._formal_gpu_frame(world):
        return None
    bridge = world.bridge
    if LIGHT_DOSE_GUARD_BUFFER not in bridge.gpu_authoritative_resources:
        return None
    return bridge.buffers.get(LIGHT_DOSE_GUARD_BUFFER)



def _build_light_dose_guarded_dispatch_args(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    guard_buffer: Any,
    group_x: int,
    group_y: int,
    group_z: int = 1,
) -> Any:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU light-dose guarded dispatch requires a valid ModernGL context")
    program = pipeline.programs["build_light_dose_guarded_dispatch_args"]
    program["full_group_count"].value = (
        max(0, int(group_x)),
        max(1, int(group_y)),
        max(1, int(group_z)),
    )
    guard_buffer.bind_to_storage_buffer(binding=LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING)
    resources.light_dose_guarded_dispatch_args.bind_to_storage_buffer(
        binding=LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING,
    )
    program.run(1, 1, 1)
    pipeline._sync_storage_and_indirect_writes(ctx)
    return resources.light_dose_guarded_dispatch_args



def _run_light_dose_guarded_dispatch(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    program: Any,
    guard_buffer: Any,
    group_x: int,
    group_y: int,
    group_z: int = 1,
) -> None:
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("formal light-dose guarded reactions require ModernGL ComputeShader.run_indirect")
    dispatch_args = pipeline._build_light_dose_guarded_dispatch_args(
        world,
        resources,
        guard_buffer,
        group_x,
        group_y,
        group_z,
    )
    program.run_indirect(dispatch_args)



def _active_masks_for_cell_reaction_upload(
    pipeline,
    world: "WorldEngine",
    solve_cell_mask: object | None,
    *,
    reaction_group: str | None = None,
) -> tuple[object | None, object | None]:
    if (
        pipeline._active_scheduler_gpu_authoritative(world)
        and (
            solve_cell_mask is None
            or bool(getattr(solve_cell_mask, "full_gpu_authoritative", False))
            or pipeline._reaction_state_segment(reaction_group) == "before_motion"
        )
    ):
        return None, None
    return (
        solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
        np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
    )



def _reaction_state_segment(pipeline, reaction_group: str | None) -> str | None:
    if (
        reaction_group in {"material_material", "material_gas", "material_light", "gas_gas", "gas_light"}
        and pipeline._formal_segment_batch_base_key is not None
        and len(pipeline._formal_segment_batch_base_key) >= 3
        and pipeline._formal_segment_batch_base_key[2] in {"before_motion", "after_optics"}
    ):
        return str(pipeline._formal_segment_batch_base_key[2])
    if reaction_group in {
        "timed",
        "self",
        "material_material",
        "material_gas",
        "material_light",
        "gas_gas",
        "gas_light",
    }:
        return "before_motion"
    return None



def _bridge_cell_core_read_role_only_load(pipeline, reaction_group: str | None) -> bool:
    return reaction_group in DIRECT_CORE_OUTPUT_REACTION_GROUPS



def _formal_reaction_segment_base_key(
    pipeline,
    world: "WorldEngine",
    segment: str | None,
) -> tuple[object, ...] | None:
    if not pipeline._formal_gpu_frame(world) or segment not in {"before_motion", "after_optics"}:
        return None
    return (
        id(world),
        int(getattr(world, "frame_id", 0)),
        segment,
    )



def _formal_reaction_segment_cache_key(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    segment: str | None,
) -> tuple[object, ...] | None:
    base_key = pipeline._formal_reaction_segment_base_key(world, segment)
    if base_key is None:
        return None
    return (
        *base_key,
        id(resources),
        tuple(resources.signature),
    )



def _formal_reaction_state_cache_key(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    reaction_group: str | None,
) -> tuple[object, ...] | None:
    if not pipeline._formal_gpu_frame(world):
        return None
    segment = pipeline._reaction_state_segment(reaction_group)
    return pipeline._formal_reaction_segment_cache_key(world, resources, segment)



def _formal_reaction_active_mask_cache_key(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    reaction_group: str | None,
    *,
    expansion_radius: int,
    load_cell_mask: bool = True,
    load_gas_mask: bool = True,
) -> tuple[object, ...] | None:
    segment = pipeline._reaction_state_segment(reaction_group)
    segment_key = pipeline._formal_reaction_segment_cache_key(world, resources, segment)
    if segment_key is None or "active_tile_ttl" not in world.bridge.gpu_authoritative_resources:
        return None
    return (
        *segment_key,
        "active",
        int(expansion_radius),
        id(world.bridge.buffers.get("active_tile_ttl")),
        id(world.bridge.buffers.get("active_chunk_mask")),
        id(world.bridge.buffers.get("active_meta")),
        bool(load_cell_mask),
        bool(load_gas_mask),
    )



def _formal_reaction_state_cache_active(pipeline) -> bool:
    return pipeline._formal_state_cache_key is not None



def _formal_segment_batch_active(pipeline) -> bool:
    return (
        pipeline._formal_segment_batch_key is not None
        and pipeline._formal_state_cache_key is not None
        and pipeline._formal_segment_batch_key == pipeline._formal_state_cache_key
    )



def _formal_state_key_is_before_motion(pipeline) -> bool:
    key = pipeline._formal_state_cache_key
    return key is not None and len(key) >= 3 and key[2] == "before_motion"



def _formal_before_motion_cell_roles_active(pipeline) -> bool:
    return (
        pipeline._formal_state_key_is_before_motion()
        and pipeline._formal_cell_state_role_key == pipeline._formal_state_cache_key
    )



def _formal_cell_read_role(pipeline) -> str:
    if not pipeline._formal_state_key_is_before_motion():
        return "ping"
    if pipeline._formal_cell_state_role_key != pipeline._formal_state_cache_key:
        return "ping"
    return pipeline._formal_cell_state_read_role



def _formal_cell_write_role(pipeline) -> str:
    return "pong" if pipeline._formal_cell_read_role() == "ping" else "ping"



def _set_formal_cell_read_role(pipeline, role: str) -> None:
    if role not in {"ping", "pong"}:
        raise ValueError(f"unsupported formal cell state role {role!r}")
    if not pipeline._formal_state_key_is_before_motion():
        return
    pipeline._formal_cell_state_role_key = pipeline._formal_state_cache_key
    pipeline._formal_cell_state_read_role = role



def _advance_formal_cell_read_role(pipeline) -> None:
    pipeline._set_formal_cell_read_role(pipeline._formal_cell_write_role())



def _reset_formal_cell_read_role(pipeline) -> None:
    pipeline._formal_cell_state_role_key = None
    pipeline._formal_cell_state_read_role = "ping"



def _cell_role_textures(pipeline, resources: GPUReactionResources, role: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    if role == "ping":
        return (
            resources.material_ping,
            resources.phase_ping,
            resources.temp_ping,
            resources.integrity_ping,
            resources.velocity_ping,
            resources.timer_ping,
        )
    if role == "pong":
        return (
            resources.material_pong,
            resources.phase_pong,
            resources.temp_pong,
            resources.integrity_pong,
            resources.velocity_pong,
            resources.timer_pong,
        )
    raise ValueError(f"unsupported cell state role {role!r}")



def _current_cell_textures(pipeline, resources: GPUReactionResources) -> tuple[Any, Any, Any, Any, Any, Any]:
    return pipeline._cell_role_textures(resources, pipeline._formal_cell_read_role())



def _next_cell_textures(pipeline, resources: GPUReactionResources) -> tuple[Any, Any, Any, Any, Any, Any]:
    if not pipeline._formal_before_motion_cell_roles_active():
        return pipeline._cell_role_textures(resources, "pong")
    return pipeline._cell_role_textures(resources, pipeline._formal_cell_write_role())



def begin_formal_reaction_segment(pipeline, world: "WorldEngine", segment: str) -> bool:
    base_key = pipeline._formal_reaction_segment_base_key(world, segment)
    if base_key is None:
        return False
    pipeline._formal_segment_batch_base_key = base_key
    pipeline._formal_segment_batch_key = None
    pipeline._formal_light_counters_cleared_key = None
    pipeline._formal_pending_bridge_publish_key = None
    pipeline._formal_pending_bridge_publish.clear()
    pipeline._formal_pending_gas_delta_key = None
    pipeline._formal_active_mask_cache_key = None
    pipeline._formal_loaded_bridge_inputs_key = None
    pipeline._formal_loaded_bridge_inputs.clear()
    pipeline._reset_formal_cell_read_role()
    pipeline._phase_c_rxn_candidate = None
    return True



def end_formal_reaction_segment(pipeline, world: "WorldEngine", segment: str) -> None:
    base_key = pipeline._formal_reaction_segment_base_key(world, segment)
    if base_key is not None and pipeline._formal_segment_batch_base_key != base_key:
        return
    pipeline._formal_segment_batch_base_key = None
    pipeline._formal_segment_batch_key = None
    pipeline._formal_light_counters_cleared_key = None
    pipeline._formal_pending_bridge_publish_key = None
    pipeline._formal_pending_bridge_publish.clear()
    pipeline._formal_pending_gas_delta_key = None
    pipeline._formal_active_mask_cache_key = None
    pipeline._formal_loaded_bridge_inputs_key = None
    pipeline._formal_loaded_bridge_inputs.clear()
    pipeline._reset_formal_cell_read_role()



def _mark_formal_bridge_publish_pending(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *resource_names: str,
) -> None:
    if not pipeline._formal_segment_batch_active():
        return
    pending_key = pipeline._formal_segment_batch_key
    if pipeline._formal_pending_bridge_publish_key != pending_key:
        pipeline._formal_pending_bridge_publish_key = pending_key
        pipeline._formal_pending_bridge_publish.clear()
    pipeline._formal_pending_bridge_publish.update(str(name) for name in resource_names)



def flush_formal_reaction_segment(pipeline, world: "WorldEngine", segment: str) -> bool:
    if not pipeline._formal_gpu_frame(world) or pipeline.resources is None:
        return False
    segment_key = pipeline._formal_reaction_segment_cache_key(world, pipeline.resources, segment)
    gas_delta_flushed = pipeline._flush_formal_segment_gas_delta(world, pipeline.resources, segment_key)
    if (
        segment_key is None
        or pipeline._formal_segment_batch_key != segment_key
        or pipeline._formal_pending_bridge_publish_key != segment_key
    ):
        return gas_delta_flushed
    pending = set(pipeline._formal_pending_bridge_publish)
    if not pending:
        pipeline._formal_pending_bridge_publish_key = None
        return gas_delta_flushed
    if "cell" in pending:
        if bool(getattr(world, "phase_c_defer_cell_publish", False)):
            role = pipeline._formal_cell_read_role() if pipeline._formal_before_motion_cell_roles_active() else "pong"
            mt, pt, tp, it, vt, tm = pipeline._cell_role_textures(pipeline.resources, role)
            pipeline._phase_c_rxn_candidate = {
                "material": mt, "phase": pt, "temp": tp, "integrity": it,
                "velocity": vt, "timer": tm,
                "latched": pipeline.resources.segment_reaction_latched_tex,
            }
        with pipeline._profile_pass(world, "publish_bridge_cell"):
            if not bool(getattr(world, "phase_c_defer_cell_publish", False)):
                pipeline._publish_bridge_cell_state(
                    world,
                    pipeline.resources,
                    source_role=pipeline._formal_cell_read_role()
                    if pipeline._formal_before_motion_cell_roles_active()
                    else None,
                    cell_reset_texture=pipeline.resources.segment_cell_reset_tex,
                    reaction_latched_texture=pipeline.resources.segment_reaction_latched_tex,
                )
    if "gas" in pending:
        with pipeline._profile_pass(world, "publish_bridge_gas"):
            pipeline._publish_bridge_gas_state(world, pipeline.resources)
    if "dose" in pending:
        with pipeline._profile_pass(world, "publish_bridge_dose"):
            pipeline._publish_bridge_dose_state(world, pipeline.resources)
    if "light_emitters" in pending:
        with pipeline._profile_pass(world, "publish_bridge_light_emitters"):
            pipeline._publish_bridge_light_emitters(world, pipeline.resources)
    pipeline._formal_pending_bridge_publish.clear()
    pipeline._formal_pending_bridge_publish_key = None
    return True



def _clear_formal_segment_gas_delta(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    segment_key: tuple[object, ...],
) -> None:
    if pipeline._formal_pending_gas_delta_key == segment_key:
        return
    ctx = world.bridge.ctx
    assert ctx is not None
    gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
    clear_program = pipeline.programs["clear_cell_gas_delta"]
    clear_program["delta_count"].value = gas_delta_count
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
    with pipeline._profile_pass(world, "cell_gas_action_delta_segment_clear"):
        clear_program.run((gas_delta_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
    pipeline._formal_pending_gas_delta_key = segment_key



def _flush_formal_segment_gas_delta(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    segment_key: tuple[object, ...] | None,
) -> bool:
    if segment_key is None or pipeline._formal_pending_gas_delta_key != segment_key:
        return False
    ctx = world.bridge.ctx
    assert ctx is not None
    apply_program = pipeline.programs["apply_cell_gas_delta"]
    apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
    resources.gas_ping.use(location=0)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
    resources.gas_pong.bind_to_image(0, read=False, write=True)
    with pipeline._profile_pass(world, "cell_gas_action_delta_segment_apply"):
        apply_program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            int(world.gas_concentration.shape[0]),
        )
        pipeline._sync_compute_writes(ctx)
    pipeline._download_gas_state(world, resources)
    pipeline._formal_pending_gas_delta_key = None
    return True



def _clear_reaction_latches_on_bridge(pipeline, world: "WorldEngine") -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction latch clearing requires bridge GPU resources for authoritative state")
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("reaction latch clearing", "cell_core")
        bridge.sync_world(world)
    ctx = bridge.ctx
    if pipeline._clear_bridge_latches_program is None:
        pipeline._clear_bridge_latches_program = build_compute_shader(
            ctx, "reactions/_clear_bridge_latches_program.comp", _SHADER_SUBS
        )
    cell_count = int(world.width * world.height)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    pipeline._clear_bridge_latches_program["cell_count"].value = cell_count
    pipeline._clear_bridge_latches_program.run((cell_count + 255) // 256, 1, 1)
    pipeline._sync_compute_writes(ctx)
    bridge.mark_gpu_authoritative("cell_core", "material")



def _upload_active_masks(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    solve_cell_mask: object | None,
    solve_gas_mask: object | None,
    *,
    reaction_group: str | None = None,
    light_dose_guard_buffer: Any | None = None,
    load_cell_mask: bool = True,
    load_gas_mask: bool = True,
) -> None:
    active_authoritative = pipeline._active_scheduler_gpu_authoritative(world)
    pipeline.last_cpu_active_upload_skipped = bool(active_authoritative)
    if active_authoritative:
        cache_key = pipeline._formal_reaction_active_mask_cache_key(
            world,
            resources,
            reaction_group,
            expansion_radius=1,
            load_cell_mask=load_cell_mask,
            load_gas_mask=load_gas_mask,
        )
        existing_cache_key = pipeline._formal_active_mask_cache_key
        if cache_key is not None and existing_cache_key is not None and existing_cache_key[:-2] == cache_key[:-2]:
            existing_load_cell = bool(existing_cache_key[-2])
            existing_load_gas = bool(existing_cache_key[-1])
            if (not load_cell_mask or existing_load_cell) and (not load_gas_mask or existing_load_gas):
                return
            load_cell_mask = bool(load_cell_mask and not existing_load_cell)
            load_gas_mask = bool(load_gas_mask and not existing_load_gas)
            cache_key = (
                *cache_key[:-2],
                bool(load_cell_mask or existing_load_cell),
                bool(load_gas_mask or existing_load_gas),
            )
        elif cache_key is not None and existing_cache_key == cache_key:
            return
        pipeline._load_authoritative_active_masks(
            world,
            resources,
            expansion_radius=1,
            light_dose_guard_buffer=light_dose_guard_buffer,
            load_cell_mask=load_cell_mask,
            load_gas_mask=load_gas_mask,
        )
        pipeline._formal_active_mask_cache_key = cache_key
        return
    pipeline._formal_active_mask_cache_key = None
    if load_cell_mask:
        if solve_cell_mask is None:
            raise RuntimeError("CPU active mask upload requires a materialized cell mask")
        resources.active_cell_tex.write(np.asarray(solve_cell_mask, dtype="f4").tobytes())
    if load_gas_mask:
        if solve_gas_mask is None:
            raise RuntimeError("CPU active mask upload requires a materialized gas mask")
        resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())



def _load_authoritative_active_masks(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    expansion_radius: int,
    light_dose_guard_buffer: Any | None = None,
    load_cell_mask: bool = True,
    load_gas_mask: bool = True,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU reaction pipeline requires bridge active scheduler resources")
    active_mask_loads = []
    if load_cell_mask:
        active_mask_loads.append(("load_active_cell", resources.active_cell_tex, world.width, world.height))
    if load_gas_mask:
        active_mask_loads.append(("load_active_gas", resources.active_gas_tex, world.gas_width, world.gas_height))
    for name, texture, width, height in active_mask_loads:
        program = pipeline.programs[name]
        pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        pipeline._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        pipeline._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
        pipeline._set_uniform_if_present(program, "tile_size", int(world.active.tile_size))
        pipeline._set_uniform_if_present(program, "expansion_radius", int(expansion_radius))
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        texture.bind_to_image(1, read=False, write=True)
        with pipeline._profile_pass(world, name):
            group_x = (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE
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

