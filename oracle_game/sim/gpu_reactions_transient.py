from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.types import Phase
from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    GPUReactionBridgeInputLoads,
    GPUReactionResources,
    LOCAL_SIZE,
    MAX_ACTIONS,
    MAX_MATERIALS,
    MAX_SELF_RULES,
)


def release(pipeline) -> None:
    pipeline._formal_state_cache_key = None
    pipeline._formal_active_mask_cache_key = None
    pipeline._formal_loaded_bridge_inputs_key = None
    pipeline._formal_loaded_bridge_inputs.clear()
    pipeline._formal_segment_batch_base_key = None
    pipeline._formal_segment_batch_key = None
    pipeline._formal_light_counters_cleared_key = None
    pipeline._formal_pending_bridge_publish_key = None
    pipeline._formal_pending_bridge_publish.clear()
    pipeline._reset_formal_cell_read_role()
    if pipeline.resources is None:
        return
    for resource in (
        pipeline.resources.material_ping,
        pipeline.resources.material_pong,
        pipeline.resources.phase_ping,
        pipeline.resources.phase_pong,
        pipeline.resources.temp_ping,
        pipeline.resources.temp_pong,
        pipeline.resources.integrity_ping,
        pipeline.resources.integrity_pong,
        pipeline.resources.velocity_ping,
        pipeline.resources.velocity_pong,
        pipeline.resources.timer_ping,
        pipeline.resources.timer_pong,
        pipeline.resources.ambient_ping,
        pipeline.resources.ambient_pong,
        pipeline.resources.gas_ping,
        pipeline.resources.gas_pong,
        pipeline.resources.flow_velocity_tex,
        pipeline.resources.active_cell_tex,
        pipeline.resources.active_gas_tex,
        pipeline.resources.cell_dose_tex,
        pipeline.resources.cell_dose_pong,
        pipeline.resources.gas_dose_tex,
        pipeline.resources.gas_dose_pong,
        pipeline.resources.flow_source_tex,
        pipeline.resources.gas_delta_buffer,
        pipeline.resources.timed_candidate_count,
        pipeline.resources.timed_candidate_list,
        pipeline.resources.timed_candidate_dispatch_args,
        pipeline.resources.light_dose_guarded_dispatch_args,
        pipeline.resources.timed_candidate_marks,
        pipeline.resources.timed_material_target_list,
        pipeline.resources.timed_material_target_dispatch_args,
        pipeline.resources.timed_material_target_marks,
        pipeline.resources.trigger_lo_tex,
        pipeline.resources.trigger_hi_tex,
        pipeline.resources.deferred_scale_lo_tex,
        pipeline.resources.deferred_scale_hi_tex,
        pipeline.resources.cell_reset_tex,
        pipeline.resources.reaction_latched_tex,
        pipeline.resources.segment_cell_reset_tex,
        pipeline.resources.segment_reaction_latched_tex,
        pipeline.resources.emitted_material_mask_tex,
        pipeline.resources.local_material_out,
        pipeline.resources.local_phase_out,
        pipeline.resources.local_temp_out,
        pipeline.resources.local_integrity_out,
        pipeline.resources.local_timer_out,
        pipeline.resources.local_deferred_lo_out,
        pipeline.resources.local_deferred_hi_out,
        pipeline.resources.local_cell_meta_out,
        pipeline.resources.local_emit_cell_lo_out,
        pipeline.resources.local_emit_cell_hi_out,
        pipeline.resources.material_params,
        pipeline.resources.material_tags,
        pipeline.resources.gas_tags,
        pipeline.resources.material_slots_lo,
        pipeline.resources.material_slots_hi,
        pipeline.resources.action_meta,
        pipeline.resources.light_emitter_buffer,
        pipeline.resources.light_emitter_count,
        pipeline.resources.random_targets,
        pipeline.resources.action_i,
        pipeline.resources.action_f,
        pipeline.resources.mm_rule_i,
        pipeline.resources.mm_rule_f,
        pipeline.resources.mm_rule_tags,
        pipeline.resources.mg_rule_i,
        pipeline.resources.mg_rule_f,
        pipeline.resources.mg_rule_tags,
        pipeline.resources.rule_lhs_candidate_masks,
        pipeline.resources.ml_rule_i,
        pipeline.resources.ml_rule_f,
        pipeline.resources.ml_rule_tags,
        pipeline.resources.gg_rule_i,
        pipeline.resources.gg_rule_f,
        pipeline.resources.gg_rule_tags,
        pipeline.resources.gl_rule_i,
        pipeline.resources.gl_rule_f,
        pipeline.resources.gl_rule_tags,
        pipeline.resources.self_rule_i,
        pipeline.resources.self_rule_f,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None



def _upload_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    reaction_group: str | None = None,
    compiled_actions: tuple[np.ndarray, np.ndarray] | None = None,
    light_dose_guard_buffer: Any | None = None,
    publishes_gas: bool | None = None,
    flow_source_layers: int | None = None,
) -> None:
    world.bridge.sync_rule_tables(world)
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    bridge_loads = pipeline._bridge_input_load_requirements(
        world,
        reaction_group,
        compiled_actions,
        publishes_gas=publishes_gas,
    )
    profile_scope = pipeline._upload_state_profile_scope(reaction_group)
    cache_key = pipeline._formal_reaction_state_cache_key(world, resources, reaction_group)
    reuse_formal_state = cache_key is not None and pipeline._formal_state_cache_key == cache_key
    batch_key_started = False
    if cache_key is not None and pipeline._formal_segment_batch_base_key == cache_key[:3]:
        if pipeline._formal_segment_batch_key is None:
            pipeline._formal_segment_batch_key = cache_key
            batch_key_started = True
            pipeline._formal_loaded_bridge_inputs_key = None
            pipeline._formal_loaded_bridge_inputs.clear()
        elif pipeline._formal_segment_batch_key != cache_key:
            pipeline._formal_segment_batch_key = cache_key
            pipeline._formal_pending_bridge_publish_key = None
            pipeline._formal_pending_bridge_publish.clear()
            pipeline._formal_active_mask_cache_key = None
            pipeline._formal_loaded_bridge_inputs_key = None
            pipeline._formal_loaded_bridge_inputs.clear()
            batch_key_started = True
    if cache_key is None:
        pipeline._formal_state_cache_key = None
        pipeline._formal_active_mask_cache_key = None
        pipeline._formal_light_counters_cleared_key = None
        pipeline._formal_loaded_bridge_inputs_key = None
        pipeline._formal_loaded_bridge_inputs.clear()
        pipeline._reset_formal_cell_read_role()
        if pipeline._formal_segment_batch_base_key is None:
            pipeline._formal_segment_batch_key = None
            pipeline._formal_pending_bridge_publish_key = None
            pipeline._formal_pending_bridge_publish.clear()
    required_bridge_resources = bridge_loads.resource_names()
    if required_bridge_resources:
        world._require_gpu_authoritative_resources("reaction input", *required_bridge_resources)
    upload_cell_state_from_cpu = (
        bridge_loads.cell_core
        and not (formal_gpu_frame and "cell_core" in authoritative)
        and not reuse_formal_state
    )
    upload_gas_from_cpu = (
        bridge_loads.gas
        and not (formal_gpu_frame and "gas_concentration" in authoritative)
        and not reuse_formal_state
    )
    upload_ambient_from_cpu = (
        bridge_loads.ambient
        and not (formal_gpu_frame and "ambient_temperature" in authoritative)
        and not reuse_formal_state
    )
    upload_flow_velocity_from_cpu = (
        bridge_loads.flow_velocity
        and not (formal_gpu_frame and "flow_velocity" in authoritative)
        and not reuse_formal_state
    )
    upload_cell_dose_from_cpu = (
        bridge_loads.cell_dose
        and not (formal_gpu_frame and "cell_optical_dose" in authoritative)
        and not reuse_formal_state
    )
    upload_gas_dose_from_cpu = (
        bridge_loads.gas_dose
        and not (formal_gpu_frame and "gas_optical_dose" in authoritative)
        and not reuse_formal_state
    )
    pipeline.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
    pipeline.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
    pipeline.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
    pipeline.last_cpu_flow_velocity_upload_skipped = not upload_flow_velocity_from_cpu
    pipeline.last_cpu_cell_dose_upload_skipped = not upload_cell_dose_from_cpu
    pipeline.last_cpu_gas_dose_upload_skipped = not upload_gas_dose_from_cpu
    if upload_cell_state_from_cpu:
        resources.material_ping.write(world.material_id.astype("f4").tobytes())
        resources.phase_ping.write(world.phase.astype("f4").tobytes())
        resources.temp_ping.write(world.cell_temperature.astype("f4").tobytes())
        resources.integrity_ping.write(world.integrity.astype("f4").tobytes())
        resources.velocity_ping.write(world.velocity.astype("f4").tobytes())
        resources.velocity_pong.write(world.velocity.astype("f4").tobytes())
        resources.timer_ping.write(world.timer_pack.astype("f4").tobytes())
        resources.timer_pong.write(world.timer_pack.astype("f4").tobytes())
        resources.material_pong.write(world.material_id.astype("f4").tobytes())
        resources.phase_pong.write(world.phase.astype("f4").tobytes())
        resources.temp_pong.write(world.cell_temperature.astype("f4").tobytes())
        resources.integrity_pong.write(world.integrity.astype("f4").tobytes())
    if upload_ambient_from_cpu:
        resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
        resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
    if upload_gas_from_cpu:
        resources.gas_ping.write(world.gas_concentration.astype("f4").tobytes())
        resources.gas_pong.write(world.gas_concentration.astype("f4").tobytes())
    if upload_flow_velocity_from_cpu:
        resources.flow_velocity_tex.write(world.flow_velocity.astype("f4").tobytes())
    if upload_cell_dose_from_cpu:
        resources.cell_dose_tex.write(world.cell_optical_dose.astype("f4").tobytes())
        resources.cell_dose_pong.write(world.cell_optical_dose.astype("f4").tobytes())
    if upload_gas_dose_from_cpu:
        resources.gas_dose_tex.write(world.gas_optical_dose.astype("f4").tobytes())
        resources.gas_dose_pong.write(world.gas_optical_dose.astype("f4").tobytes())
    if formal_gpu_frame:
        clear_requirements = pipeline._transient_clear_requirements(reaction_group, compiled_actions)
        if flow_source_layers is not None:
            clear_requirements["flow_source_layers"] = max(
                1,
                min(FLOW_SOURCE_LAYERS, int(flow_source_layers)),
            )
        if clear_requirements["clear_light_counters"]:
            if cache_key is None:
                pipeline._formal_light_counters_cleared_key = None
            elif pipeline._formal_light_counters_cleared_key == cache_key:
                clear_requirements["clear_light_counters"] = False
            else:
                pipeline._formal_light_counters_cleared_key = cache_key
        if batch_key_started or (pipeline._formal_segment_batch_key == cache_key and not reuse_formal_state):
            with pipeline._profile_scoped_pass(world, profile_scope, "clear_segment_transient"):
                pipeline._clear_segment_transient_state(world, resources)
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient"):
            pipeline._clear_transient_state(
                world,
                resources,
                profile_scope=profile_scope,
                light_dose_guard_buffer=light_dose_guard_buffer,
                **clear_requirements,
            )
    else:
        resources.flow_source_tex.write(np.zeros((FLOW_SOURCE_LAYERS, world.gas_height, world.gas_width, 4), dtype="f4").tobytes())
        resources.trigger_lo_tex.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.trigger_hi_tex.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.deferred_scale_lo_tex.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.deferred_scale_hi_tex.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.cell_reset_tex.write(np.zeros((world.height, world.width), dtype="f4").tobytes())
        resources.reaction_latched_tex.write(np.zeros((world.height, world.width), dtype="f4").tobytes())
        resources.emitted_material_mask_tex.write(np.zeros((world.height, world.width), dtype="f4").tobytes())
        resources.local_emit_cell_lo_out.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.local_emit_cell_hi_out.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.local_timer_out.write(np.zeros((world.height, world.width, 4), dtype="f4").tobytes())
        resources.local_cell_meta_out.write(np.zeros((world.height, world.width, 2), dtype="f4").tobytes())
        resources.light_emitter_count.write(np.zeros((16,), dtype=np.uint32).tobytes())
    bridge_loads_to_run = bridge_loads
    if formal_gpu_frame and cache_key is not None and reuse_formal_state:
        bridge_loads_to_run = pipeline._missing_formal_bridge_input_loads(cache_key, bridge_loads)
    if not reuse_formal_state or bridge_loads_to_run.any():
        with pipeline._profile_scoped_pass(world, profile_scope, "load_bridge_inputs"):
            pipeline._load_authoritative_bridge_inputs(
                world,
                resources,
                bridge_input_loads=bridge_loads_to_run,
                reaction_group=reaction_group,
                profile_scope=profile_scope,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
        if cache_key is not None:
            pipeline._formal_state_cache_key = cache_key
            pipeline._record_formal_bridge_inputs_loaded(cache_key, bridge_loads)
            if not reuse_formal_state:
                pipeline._set_formal_cell_read_role("ping")
    material_table = world.bridge.shadow_typed_tables["material_table"]
    table_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
    if resources.material_params_signature != table_signature:
        params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        params[:count, 0] = material_table[:count]["base_integrity"]
        params[:count, 1] = material_table[:count]["default_phase"]
        params[:count, 2] = material_table[:count]["spawn_temperature"]
        resources.material_params.write(params.tobytes())
        resources.material_params_signature = table_signature
    pipeline._upload_random_targets(world, resources, material_table)



def _bridge_input_load_requirements(
    pipeline,
    world: "WorldEngine",
    reaction_group: str | None,
    compiled_actions: tuple[np.ndarray, np.ndarray] | None,
    *,
    publishes_gas: bool | None = None,
) -> GPUReactionBridgeInputLoads:
    if not pipeline._formal_gpu_frame(world):
        return GPUReactionBridgeInputLoads()
    if compiled_actions is None:
        return GPUReactionBridgeInputLoads()

    modifies_gas = pipeline._compiled_actions_include_modify_gas(compiled_actions)
    gas_published = modifies_gas if publishes_gas is None else bool(publishes_gas)
    reads_gas = reaction_group in {"material_gas", "gas_gas", "gas_light"} or modifies_gas or gas_published
    reads_ambient = reaction_group in {"gas_gas", "gas_light"} or gas_published
    reads_cell_dose = reaction_group == "material_light"
    reads_gas_dose = reaction_group == "gas_light"
    segment = pipeline._reaction_state_segment(reaction_group)
    if segment == "before_motion":
        return GPUReactionBridgeInputLoads(
            cell_core=True,
            gas=reads_gas,
            ambient=reads_ambient,
            flow_velocity=modifies_gas and pipeline._compiled_actions_include_flow_sources(compiled_actions),
            cell_dose=reads_cell_dose,
            gas_dose=reads_gas_dose,
        )
    if segment == "after_optics":
        return GPUReactionBridgeInputLoads(
            cell_core=True,
            gas=reads_gas,
            ambient=reads_ambient,
            flow_velocity=modifies_gas and pipeline._compiled_actions_include_flow_sources(compiled_actions),
            cell_dose=reads_cell_dose,
            gas_dose=reads_gas_dose,
        )
    return GPUReactionBridgeInputLoads()



def _missing_formal_bridge_input_loads(
    pipeline,
    cache_key: tuple[object, ...],
    bridge_loads: GPUReactionBridgeInputLoads,
) -> GPUReactionBridgeInputLoads:
    if pipeline._formal_loaded_bridge_inputs_key != cache_key:
        return bridge_loads
    loaded = pipeline._formal_loaded_bridge_inputs
    return GPUReactionBridgeInputLoads(
        cell_core=bridge_loads.cell_core and "cell_core" not in loaded,
        gas=bridge_loads.gas and "gas_concentration" not in loaded,
        ambient=bridge_loads.ambient and "ambient_temperature" not in loaded,
        flow_velocity=bridge_loads.flow_velocity and "flow_velocity" not in loaded,
        cell_dose=bridge_loads.cell_dose and "cell_optical_dose" not in loaded,
        gas_dose=bridge_loads.gas_dose and "gas_optical_dose" not in loaded,
    )



def _record_formal_bridge_inputs_loaded(
    pipeline,
    cache_key: tuple[object, ...],
    bridge_loads: GPUReactionBridgeInputLoads,
) -> None:
    if pipeline._formal_loaded_bridge_inputs_key != cache_key:
        pipeline._formal_loaded_bridge_inputs_key = cache_key
        pipeline._formal_loaded_bridge_inputs.clear()
    pipeline._formal_loaded_bridge_inputs.update(bridge_loads.resource_names())



def _transient_clear_requirements(
    pipeline,
    reaction_group: str | None,
    compiled_actions: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, bool]:
    emits_material = bool(
        compiled_actions is not None and pipeline._compiled_actions_include_emit_material(compiled_actions)
    )
    return {
        "clear_light_counters": True,
        "clear_flow_sources": bool(
            compiled_actions is not None and pipeline._compiled_actions_include_flow_sources(compiled_actions)
        ),
        "flow_source_layers": 16 if reaction_group == "timed" else FLOW_SOURCE_LAYERS,
        "clear_emit_material_mask": emits_material,
        "clear_emit_material_buffers": emits_material and reaction_group in {"gas_gas", "gas_light"},
    }



def _upload_random_targets(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    material_table: np.ndarray,
) -> None:
    chaos_convert_bit = int(world.tag_bits_by_name.get("chaos_convert", 0))
    random_targets_signature = (
        int(world.bridge.table_generations.get("materials", 0)),
        int(material_table.shape[0]),
        chaos_convert_bit,
    )
    if resources.random_targets_signature == random_targets_signature:
        return
    random_targets = [
        int(row["material_id"])
        for row in material_table
        if chaos_convert_bit != 0
        and bool(int(row["material_tag_mask"]) & chaos_convert_bit)
        and int(row["default_phase"]) == int(Phase.POWDER)
    ]
    packed_random_targets = np.zeros((MAX_MATERIALS,), dtype=np.int32)
    for index, material_id in enumerate(random_targets[:MAX_MATERIALS]):
        packed_random_targets[index] = int(material_id)
    pipeline.random_targets[:] = packed_random_targets
    pipeline.random_target_count = min(len(random_targets), MAX_MATERIALS)
    resources.random_targets.write(pipeline.random_targets.astype(np.int32, copy=False).tobytes())
    resources.random_targets_signature = random_targets_signature



def _clear_transient_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    clear_light_counters: bool = True,
    clear_flow_sources: bool = False,
    clear_emit_material_mask: bool = False,
    clear_emit_material_buffers: bool = False,
    flow_source_layers: int = FLOW_SOURCE_LAYERS,
    profile_scope: str | None = None,
    light_dose_guard_buffer: Any | None = None,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    ran_clear = False
    with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_full_cell_outputs_skipped"):
        pass
    if clear_light_counters:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_light_counters"):
            counter_program = pipeline.programs["clear_transient_light_counters"]
            resources.light_emitter_count.bind_to_storage_buffer(binding=0)
            counter_program.run(1, 1, 1)
            ran_clear = True
    else:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_light_counters_skipped"):
            pass

    if clear_emit_material_buffers:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_buffers"):
            emit_program = pipeline.programs["clear_transient_emit_material_buffers"]
            emit_program["cell_grid_size"].value = (world.width, world.height)
            resources.local_emit_cell_lo_out.bind_to_image(0, read=False, write=True)
            resources.local_emit_cell_hi_out.bind_to_image(1, read=False, write=True)
            resources.local_timer_out.bind_to_image(2, read=False, write=True)
            resources.local_cell_meta_out.bind_to_image(3, read=False, write=True)
            resources.emitted_material_mask_tex.bind_to_image(4, read=False, write=True)
            resources.cell_reset_tex.bind_to_image(5, read=False, write=True)
            resources.reaction_latched_tex.bind_to_image(6, read=False, write=True)
            group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    emit_program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    1,
                )
            else:
                emit_program.run(group_x, group_y, 1)
            ran_clear = True
    elif clear_emit_material_mask:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_mask"):
            mask_program = pipeline.programs["clear_transient_emit_material_mask"]
            mask_program["cell_grid_size"].value = (world.width, world.height)
            resources.emitted_material_mask_tex.bind_to_image(0, read=False, write=True)
            group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    mask_program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    1,
                )
            else:
                mask_program.run(group_x, group_y, 1)
            ran_clear = True
    else:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_skipped"):
            pass

    if clear_flow_sources:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_flow_sources"):
            flow_program = pipeline.programs["clear_transient_flow_sources"]
            flow_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            active_flow_source_layers = max(1, min(FLOW_SOURCE_LAYERS, int(flow_source_layers)))
            flow_program["flow_source_layers"].value = active_flow_source_layers
            resources.flow_source_tex.bind_to_image(0, read=False, write=True)
            group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            if light_dose_guard_buffer is not None:
                pipeline._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    flow_program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    active_flow_source_layers,
                )
            else:
                flow_program.run(group_x, group_y, active_flow_source_layers)
            ran_clear = True
    else:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_flow_sources_skipped"):
            pass
    if ran_clear:
        pipeline._sync_compute_writes(ctx)



def _clear_segment_transient_state(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    program = pipeline.programs["clear_segment_cell_transient_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.segment_cell_reset_tex.bind_to_image(0, read=False, write=True)
    resources.segment_reaction_latched_tex.bind_to_image(1, read=False, write=True)
    program.run(
        (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(ctx)



def _accumulate_segment_cell_transient_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    direct_core_outputs: bool = False,
) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    ctx = world.bridge.ctx
    if ctx is None:
        return
    program = pipeline.programs["accumulate_segment_cell_transient_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["use_local_cell_meta"].value = bool(direct_core_outputs)
    resources.cell_reset_tex.use(location=0)
    resources.reaction_latched_tex.use(location=1)
    resources.local_cell_meta_out.use(location=2)
    resources.segment_cell_reset_tex.bind_to_image(0, read=True, write=True)
    resources.segment_reaction_latched_tex.bind_to_image(1, read=True, write=True)
    program.run(
        (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(ctx)



def _upload_local_metadata(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    include_self_rules: bool = False,
) -> None:
    world.bridge.sync_rule_tables(world)
    material_table = world.bridge.shadow_typed_tables["material_table"]
    material_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
    if resources.material_slots_signature != material_signature:
        slots_lo = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
        slots_hi = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
        material_tags = np.zeros((MAX_MATERIALS, 4), dtype=np.uint32)
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        reaction_slots = material_table[:count]["reaction_slots"]
        slots_lo[:count] = reaction_slots[:, :4]
        slots_hi[:count] = reaction_slots[:, 4:8]
        material_tags[:count, 0] = material_table[:count]["material_tag_mask"]
        material_tags[:count, 1] = material_table[:count]["gas_tag_mask"]
        material_tags[:count, 2] = material_table[:count]["light_tag_mask"]
        resources.material_slots_lo.write(slots_lo.tobytes())
        resources.material_slots_hi.write(slots_hi.tobytes())
        resources.material_tags.write(material_tags.tobytes())
        resources.material_slots_signature = material_signature

    gas_table = world.bridge.shadow_typed_tables["gas_table"]
    gas_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
    if resources.gas_tags_signature != gas_signature:
        gas_tags = np.zeros((MAX_MATERIALS, 4), dtype=np.uint32)
        count = min(MAX_MATERIALS, int(gas_table.shape[0]))
        gas_tags[:count, 0] = gas_table[:count]["material_reaction_tag_mask"]
        gas_tags[:count, 1] = gas_table[:count]["light_reaction_tag_mask"]
        resources.gas_tags.write(gas_tags.tobytes())
        resources.gas_tags_signature = gas_signature

    action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
    action_signature = (world.bridge.table_generations.get("reactions", 0), int(action_table.shape[0]))
    if resources.action_meta_signature != action_signature:
        action_meta = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
        count = min(MAX_ACTIONS, int(action_table.shape[0]))
        action_meta[:count, 0] = action_table[:count]["duration"]
        resources.action_meta.write(action_meta.tobytes())
        resources.action_meta_signature = action_signature

    if not include_self_rules:
        return

    self_rule_table = world.bridge.shadow_typed_tables["self_rule_table"]
    self_rule_signature = (world.bridge.table_generations.get("reactions", 0), int(self_rule_table.shape[0]))
    if resources.self_rule_signature == self_rule_signature:
        return
    compiled_self_i = np.zeros((MAX_SELF_RULES, 4), dtype=np.int32)
    compiled_self_f = np.zeros((MAX_SELF_RULES, 4), dtype=np.float32)
    count = min(MAX_SELF_RULES, int(self_rule_table.shape[0]))
    if count > 0:
        rows = self_rule_table[:count]
        compiled_self_i[:count, 0] = rows["material_id"]
        compiled_self_i[:count, 1] = rows["trigger_slot_index"]
        compiled_self_i[:count, 2] = rows["phase_mask"]
        integrity_at_most = rows["integrity_at_most"]
        integrity_at_least = rows["integrity_at_least"]
        has_upper = ~np.isnan(integrity_at_most)
        has_lower = ~np.isnan(integrity_at_least)
        flags = np.zeros((count,), dtype=np.int32)
        flags[has_upper] |= 1
        flags[has_lower] |= 2
        compiled_self_i[:count, 3] = flags
        compiled_self_f[:count, 0] = rows["min_temperature"]
        compiled_self_f[:count, 1] = rows["max_temperature"]
        compiled_self_f[:count, 2] = np.where(has_upper, integrity_at_most, 0.0)
        compiled_self_f[:count, 3] = np.where(has_lower, integrity_at_least, 0.0)
    resources.self_rule_i.write(compiled_self_i.tobytes())
    resources.self_rule_f.write(compiled_self_f.tobytes())
    resources.self_rule_signature = self_rule_signature



def _promote_cell_pong_to_ping(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    program = pipeline.programs["promote_reaction_cell_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.material_pong.use(location=0)
    resources.phase_pong.use(location=1)
    resources.temp_pong.use(location=2)
    resources.integrity_pong.use(location=3)
    resources.velocity_pong.use(location=4)
    resources.timer_pong.use(location=5)
    resources.material_ping.bind_to_image(0, read=False, write=True)
    resources.phase_ping.bind_to_image(1, read=False, write=True)
    resources.temp_ping.bind_to_image(2, read=False, write=True)
    resources.integrity_ping.bind_to_image(3, read=False, write=True)
    resources.velocity_ping.bind_to_image(4, read=False, write=True)
    resources.timer_ping.bind_to_image(5, read=False, write=True)
    with pipeline._profile_pass(world, "promote_cell_pong"):
        program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        pipeline._sync_compute_writes(world.bridge.ctx)



def _copy_gas_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    gas_source: Any,
    ambient_source: Any,
    gas_destination: Any,
    ambient_destination: Any,
) -> None:
    if gas_source is gas_destination and ambient_source is ambient_destination:
        return
    program = pipeline.programs["promote_reaction_gas_state"]
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["gas_count"].value = int(world.gas_concentration.shape[0])
    gas_source.use(location=0)
    ambient_source.use(location=1)
    gas_destination.bind_to_image(0, read=False, write=True)
    ambient_destination.bind_to_image(1, read=False, write=True)
    program.run(
        (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        int(world.gas_concentration.shape[0]),
    )
    pipeline._sync_compute_writes(world.bridge.ctx)



def _promote_gas_pong_to_ping(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    with pipeline._profile_pass(world, "promote_gas_pong"):
        pipeline._copy_gas_state(
            world,
            resources,
            gas_source=resources.gas_pong,
            ambient_source=resources.ambient_pong,
            gas_destination=resources.gas_ping,
            ambient_destination=resources.ambient_ping,
        )



def _promote_gas_result(pipeline, world: "WorldEngine", resources: GPUReactionResources, gas_source: Any, ambient_source: Any) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    with pipeline._profile_pass(world, "promote_gas_pong"):
        if gas_source is resources.gas_ping and ambient_source is resources.ambient_ping:
            pipeline._copy_gas_state(
                world,
                resources,
                gas_source=gas_source,
                ambient_source=ambient_source,
                gas_destination=resources.gas_pong,
                ambient_destination=resources.ambient_pong,
            )
            return
        pipeline._copy_gas_state(
            world,
            resources,
            gas_source=gas_source,
            ambient_source=ambient_source,
            gas_destination=resources.gas_ping,
            ambient_destination=resources.ambient_ping,
        )



def _promote_dose_pong_to_ping(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    program = pipeline.programs["promote_reaction_dose_state"]
    light_count = int(world.cell_optical_dose.shape[0])
    program["cell_grid_size"].value = (world.width, world.height)
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    program["light_count"].value = light_count
    resources.cell_dose_pong.use(location=0)
    resources.gas_dose_pong.use(location=1)
    resources.cell_dose_tex.bind_to_image(0, read=False, write=True)
    resources.gas_dose_tex.bind_to_image(1, read=False, write=True)
    with pipeline._profile_pass(world, "promote_dose_pong"):
        program.run(
            (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
            light_count,
        )
        pipeline._sync_compute_writes(world.bridge.ctx)



def _copy_bridge_flow_velocity_to_reaction(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    program = pipeline.programs["copy_bridge_flow_velocity_to_reaction"]
    program["gas_grid_size"].value = (world.gas_width, world.gas_height)
    world.bridge.textures["flow_velocity"].use(location=0)
    resources.flow_velocity_tex.bind_to_image(0, read=False, write=True)
    program.run(
        (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(world.bridge.ctx)



def _sync_storage_and_indirect_writes(pipeline, ctx: Any | None) -> None:
    if ctx is None:
        return
    ctx.memory_barrier(
        ctx.SHADER_STORAGE_BARRIER_BIT
        | getattr(ctx, "COMMAND_BARRIER_BIT", 0)
        | ctx.TEXTURE_FETCH_BARRIER_BIT,
    )



def _sync_compute_writes(pipeline, ctx: Any | None) -> None:
    if ctx is None:
        return
    ctx.memory_barrier(
        ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT,
    )
