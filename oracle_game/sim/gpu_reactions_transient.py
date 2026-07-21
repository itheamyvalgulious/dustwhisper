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
from oracle_game.sim.gpu_timer_pack import pack_cell_state, pack_u8x4


def release(pipeline) -> None:
    pipeline._used_action_indices_cache.clear()
    pipeline._compiled_action_cache.clear()
    pipeline._material_light_packed_descriptor_cache_key = None
    pipeline._material_light_packed_descriptor_cache = None
    pipeline._material_pair_packed_descriptor_cache_key = None
    pipeline._material_pair_packed_descriptor_cache = None
    pipeline._material_pair_plan_cache.clear()
    pipeline._formal_state_cache_key = None
    pipeline._formal_active_mask_cache_key = None
    pipeline._formal_loaded_bridge_inputs_key = None
    pipeline._formal_loaded_bridge_inputs.clear()
    pipeline._formal_segment_batch_base_key = None
    pipeline._formal_segment_batch_key = None
    pipeline._reset_formal_segment_meta_lazy_zero()
    pipeline.last_terminal_segment_meta_lazy_zero_used = False
    pipeline.last_segment_meta_lazy_clear_skipped = False
    pipeline._formal_light_counters_cleared_key = None
    pipeline._formal_pending_bridge_publish_key = None
    pipeline._formal_pending_bridge_publish.clear()
    pipeline._motion_handoff_candidate = None
    pipeline._reset_formal_cell_read_role()
    pipeline._reset_formal_velocity_read_role()
    pipeline._clear_formal_external_cell_state()
    if pipeline.resources is None:
        return
    for resource in (
        pipeline.resources.cell_state_ping,
        pipeline.resources.cell_state_pong,
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
        pipeline.resources.expanded_active_tile_tex,
        pipeline.resources.active_gas_tex,
        pipeline.resources.cell_dose_tex,
        pipeline.resources.cell_dose_pong,
        pipeline.resources.gas_dose_tex,
        pipeline.resources.gas_dose_pong,
        pipeline.resources.flow_source_tex,
        pipeline.resources.flow_source_generation_tex,
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
        pipeline.resources.segment_cell_meta_tex,
        pipeline.resources.emitted_material_mask_tex,
        pipeline.resources.local_cell_state_out,
        pipeline.resources.handoff_material_tex,
        pipeline.resources.handoff_phase_tex,
        pipeline.resources.handoff_flags_tex,
        pipeline.resources.local_temp_out,
        pipeline.resources.local_integrity_out,
        pipeline.resources.local_timer_out,
        pipeline.resources.local_deferred_lo_out,
        pipeline.resources.local_deferred_hi_out,
        pipeline.resources.local_deferred_packed_out,
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
        pipeline.resources.material_pair_action_i,
        pipeline.resources.material_pair_action_f,
        pipeline.resources.mm_rule_i,
        pipeline.resources.mm_rule_f,
        pipeline.resources.mm_rule_tags,
        pipeline.resources.mg_rule_i,
        pipeline.resources.mg_rule_f,
        pipeline.resources.mg_rule_tags,
        pipeline.resources.material_pair_rule_i,
        pipeline.resources.material_pair_rule_f,
        pipeline.resources.material_pair_rule_tags,
        pipeline.resources.material_pair_lhs_candidate_masks,
        pipeline.resources.material_pair_terminal_material_tables,
        pipeline.resources.material_pair_terminal_action_tables,
        pipeline.resources.material_pair_terminal_rule_tables,
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
        pipeline.resources.self_rule_span_i,
    ):
        try:
            resource.release()
        except Exception:
            pass
    pipeline.resources = None


def _adopt_formal_heat_cell_state(
    pipeline,
    world: "WorldEngine",
    cache_key: tuple[object, ...] | None,
    bridge_loads: GPUReactionBridgeInputLoads,
) -> GPUReactionBridgeInputLoads:
    """Reuse heat's post-publish textures for formal before-motion reactions."""
    if cache_key is None or len(cache_key) < 5 or cache_key[2] != "before_motion":
        return bridge_loads
    heat_pipeline = getattr(getattr(world, "heat_solver", None), "gpu_pipeline", None)
    heat_resources = getattr(heat_pipeline, "resources", None)
    if heat_resources is None:
        return bridge_loads
    if getattr(heat_pipeline, "_last_formal_output_frame_id", None) != int(getattr(world, "frame_id", 0)):
        return bridge_loads
    heat_signature = tuple(getattr(heat_resources, "signature", ()))
    reaction_signature = tuple(cache_key[4]) if isinstance(cache_key[4], tuple) else ()
    if len(heat_signature) < 5 or len(reaction_signature) < 5 or heat_signature[:5] != reaction_signature[:5]:
        return bridge_loads
    pipeline._formal_external_cell_state_key = cache_key
    pipeline._formal_external_cell_state_textures = (
        heat_resources.cell_state_tex,
        heat_resources.cell_state_tex,
        heat_resources.temp_ping,
        heat_resources.integrity_tex,
        heat_resources.velocity_tex,
        heat_resources.timer_tex,
    )
    pipeline._formal_external_cell_flags_texture = None
    return GPUReactionBridgeInputLoads(
        cell_core=False,
        gas=bridge_loads.gas,
        ambient=bridge_loads.ambient,
        flow_velocity=bridge_loads.flow_velocity,
        cell_dose=bridge_loads.cell_dose,
        gas_dose=bridge_loads.gas_dose,
    )



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
    direct_bridge_cell_dose: bool = False,
    direct_bridge_gas_dose: bool = False,
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
    if formal_gpu_frame:
        bridge_loads = _adopt_formal_heat_cell_state(pipeline, world, cache_key, bridge_loads)
    bridge_copy_loads = GPUReactionBridgeInputLoads(
        cell_core=bridge_loads.cell_core,
        gas=bridge_loads.gas,
        ambient=bridge_loads.ambient,
        flow_velocity=bridge_loads.flow_velocity,
        cell_dose=bridge_loads.cell_dose and not (formal_gpu_frame and direct_bridge_cell_dose),
        gas_dose=bridge_loads.gas_dose and not (formal_gpu_frame and direct_bridge_gas_dose),
    )
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
        pipeline._reset_formal_velocity_read_role()
        pipeline._clear_formal_external_cell_state()
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
        bridge_copy_loads.cell_dose
        and not (formal_gpu_frame and "cell_optical_dose" in authoritative)
        and not reuse_formal_state
    )
    upload_gas_dose_from_cpu = (
        bridge_copy_loads.gas_dose
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
        packed_cell_state = pack_cell_state(world.material_id, world.phase, world.cell_flags).tobytes()
        resources.cell_state_ping.write(packed_cell_state)
        resources.temp_ping.write(world.cell_temperature.astype("f4").tobytes())
        resources.integrity_ping.write(world.integrity.astype("f4").tobytes())
        resources.velocity_ping.write(world.velocity.astype("f4").tobytes())
        resources.velocity_pong.write(world.velocity.astype("f4").tobytes())
        packed_timer = pack_u8x4(world.timer_pack).tobytes()
        resources.timer_ping.write(packed_timer)
        resources.timer_pong.write(packed_timer)
        resources.cell_state_pong.write(packed_cell_state)
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
        clear_segment_state = bool(
            batch_key_started
            or (pipeline._formal_segment_batch_key == cache_key and not reuse_formal_state)
        )
        lazy_segment_zero = bool(
            clear_segment_state
            and pipeline._terminal_segment_meta_lazy_zero_enabled
            and pipeline._timed_self_cell_flag_meta_enabled
            and pipeline._formal_segment_batch_key is not None
        )
        if lazy_segment_zero:
            pipeline._begin_formal_segment_meta_lazy_zero()
            pipeline.last_segment_meta_lazy_clear_skipped = True
        fuse_segment_light_counters = bool(
            clear_segment_state
            and not lazy_segment_zero
            and clear_requirements["clear_light_counters"]
            and getattr(
                pipeline,
                "_segment_meta_light_counter_clear_fusion_enabled",
                False,
            )
        )
        if clear_segment_state and not lazy_segment_zero:
            with pipeline._profile_scoped_pass(world, profile_scope, "clear_segment_transient"):
                pipeline._clear_segment_transient_state(
                    world,
                    resources,
                    clear_light_counters=fuse_segment_light_counters,
                )
        elif lazy_segment_zero:
            with pipeline._profile_scoped_pass(
                world,
                profile_scope,
                "clear_segment_transient_lazy_zero_skipped",
            ):
                pass
        if fuse_segment_light_counters:
            clear_requirements["clear_light_counters"] = False
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
        resources.local_timer_out.write(np.zeros((world.height, world.width), dtype="u4").tobytes())
        resources.local_cell_meta_out.write(np.zeros((world.height, world.width, 2), dtype="f4").tobytes())
        resources.light_emitter_count.write(np.zeros((16,), dtype=np.uint32).tobytes())
    bridge_loads_to_run = bridge_copy_loads
    if formal_gpu_frame and cache_key is not None and reuse_formal_state:
        bridge_loads_to_run = pipeline._missing_formal_bridge_input_loads(cache_key, bridge_copy_loads)
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
            pipeline._record_formal_bridge_inputs_loaded(cache_key, bridge_copy_loads)
            if pipeline._formal_external_cell_state_key == cache_key:
                pipeline._formal_loaded_bridge_inputs.add("cell_core")
            if not reuse_formal_state:
                pipeline._set_formal_cell_read_role("ping")
                pipeline._set_formal_velocity_read_role("ping")
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
    reads_gas = reaction_group in {"material_gas", "material_pair_fused", "gas_gas", "gas_light"} or modifies_gas or gas_published
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
        # Formal action batches never read this debug/download-only mask.
        "clear_emit_material_mask": False,
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

    generation_validity = bool(
        clear_flow_sources
        and pipeline._flow_source_generation_validity_active(world)
    )
    if generation_validity:
        with pipeline._profile_scoped_pass(world, profile_scope, "clear_transient_flow_sources"):
            pipeline._advance_flow_source_generation(world, resources)
    elif clear_flow_sources:
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


def _flow_source_generation_validity_active(pipeline, world: "WorldEngine") -> bool:
    return bool(
        pipeline._flow_source_generation_validity_enabled
        and pipeline._flow_source_generation_programs_enabled
        and pipeline._formal_gpu_frame(world)
    )


def _advance_flow_source_generation(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
) -> None:
    generation = int(resources.flow_source_generation)
    generation_limit = (
        int(np.iinfo(np.uint8).max)
        if pipeline._flow_source_generation_u8_programs_enabled
        else int(np.iinfo(np.uint32).max)
    )
    if generation >= generation_limit:
        program = pipeline.programs["clear_transient_flow_source_generations"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        resources.flow_source_generation_tex.bind_to_image(0, read=False, write=True)
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            FLOW_SOURCE_LAYERS,
        )
        pipeline._sync_compute_writes(world.bridge.ctx)
        generation = 0
    resources.flow_source_generation = generation + 1


def _bind_flow_source_generation_output(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    program: Any,
    *,
    binding: int,
) -> None:
    if not pipeline._flow_source_generation_programs_enabled:
        return
    enabled = pipeline._flow_source_generation_validity_active(world)
    pipeline._set_uniform_if_present(
        program,
        "flow_source_generation_validity_enabled",
        enabled,
    )
    pipeline._set_uniform_if_present(
        program,
        "flow_source_generation",
        int(resources.flow_source_generation),
    )
    resources.flow_source_generation_tex.bind_to_image(
        binding,
        read=False,
        write=True,
    )



def _clear_segment_transient_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    clear_light_counters: bool = False,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        return
    program = pipeline.programs[
        "clear_segment_cell_transient_state_light_counters"
        if clear_light_counters
        else "clear_segment_cell_transient_state"
    ]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.segment_cell_meta_tex.bind_to_image(0, read=False, write=True)
    if clear_light_counters:
        resources.light_emitter_count.bind_to_storage_buffer(binding=0)
    program.run(
        (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
        (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        1,
    )
    pipeline._sync_compute_writes(ctx)
    if pipeline._formal_segment_meta_lazy_key == pipeline._formal_segment_batch_key:
        pipeline._formal_segment_meta_physically_cleared = True
        pipeline._formal_segment_meta_logically_zero = False
        pipeline._formal_segment_all_prior_cell_meta_in_flags = False


def _begin_formal_segment_meta_lazy_zero(pipeline) -> None:
    segment_key = pipeline._formal_segment_batch_key
    if segment_key is None:
        raise RuntimeError("lazy segment metadata requires an active formal segment batch")
    pipeline._formal_segment_meta_lazy_key = segment_key
    pipeline._formal_segment_meta_logically_zero = True
    pipeline._formal_segment_meta_physically_cleared = False
    pipeline._formal_segment_all_prior_cell_meta_in_flags = True
    pipeline.last_terminal_segment_meta_lazy_zero_used = False


def _reset_formal_segment_meta_lazy_zero(pipeline) -> None:
    pipeline._formal_segment_meta_lazy_key = None
    pipeline._formal_segment_meta_logically_zero = False
    pipeline._formal_segment_meta_physically_cleared = False
    pipeline._formal_segment_all_prior_cell_meta_in_flags = False


def _record_formal_segment_cell_meta_in_flags(pipeline, carried_in_flags: bool) -> None:
    if (
        pipeline._formal_segment_meta_lazy_key != pipeline._formal_segment_batch_key
        or not pipeline._formal_segment_meta_logically_zero
        or pipeline._formal_segment_meta_physically_cleared
    ):
        return
    pipeline._formal_segment_all_prior_cell_meta_in_flags = bool(
        pipeline._formal_segment_all_prior_cell_meta_in_flags
        and carried_in_flags
    )


def _ensure_formal_segment_meta_physical_zero(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
) -> None:
    if (
        pipeline._formal_segment_meta_lazy_key != pipeline._formal_segment_batch_key
        or not pipeline._formal_segment_meta_logically_zero
        or pipeline._formal_segment_meta_physically_cleared
    ):
        return
    with pipeline._profile_pass(world, "clear_segment_transient_lazy_fallback"):
        pipeline._clear_segment_transient_state(
            world,
            resources,
            clear_light_counters=False,
        )
    pipeline.segment_meta_lazy_fallback_clear_count += 1


def _can_use_terminal_segment_meta_zero(pipeline) -> bool:
    return bool(
        pipeline._terminal_segment_meta_lazy_zero_enabled
        and pipeline._formal_segment_meta_lazy_key == pipeline._formal_segment_batch_key
        and pipeline._formal_segment_meta_logically_zero
        and not pipeline._formal_segment_meta_physically_cleared
        and pipeline._formal_segment_all_prior_cell_meta_in_flags
    )



def _accumulate_segment_cell_transient_state(
    pipeline,
    world: "WorldEngine",
    resources: GPUReactionResources,
    *,
    direct_core_outputs: bool = False,
    light_dose_guard_buffer: Any | None = None,
    packed_local_cell_meta: bool = False,
) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    ctx = world.bridge.ctx
    if ctx is None:
        return
    pipeline._ensure_formal_segment_meta_physical_zero(world, resources)
    program = pipeline.programs["accumulate_segment_cell_transient_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    program["use_local_cell_meta"].value = bool(direct_core_outputs)
    program["use_packed_local_cell_meta"].value = bool(packed_local_cell_meta)
    resources.cell_reset_tex.use(location=0)
    resources.reaction_latched_tex.use(location=1)
    resources.local_cell_meta_out.use(location=2)
    resources.local_deferred_packed_out.use(location=3)
    resources.segment_cell_meta_tex.bind_to_image(0, read=True, write=True)
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
    self_rule_signature = (
        world.bridge.table_generations.get("reactions", 0),
        int(self_rule_table.shape[0]),
        world.bridge.table_generations.get("materials", 0),
        int(material_table.shape[0]),
    )
    if resources.self_rule_signature == self_rule_signature:
        return
    compiled_self_i = np.zeros((MAX_SELF_RULES, 4), dtype=np.int32)
    compiled_self_f = np.zeros((MAX_SELF_RULES, 4), dtype=np.float32)
    compiled_self_span_i = np.zeros((MAX_SELF_RULES, 4), dtype=np.int32)
    direct_action_spans = True
    count = min(MAX_SELF_RULES, int(self_rule_table.shape[0]))
    if count > 0:
        rows = self_rule_table[:count]
        # A cell can only match rules for its own material. Stable grouping
        # preserves that material's rule order while removing all unrelated
        # rule iterations from the GPU hot path.
        order = np.argsort(rows["material_id"], kind="stable")
        rows = rows[order]
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
        compiled_self_span_i[:count] = compiled_self_i[:count]
        for rule_index in range(count):
            material_id = int(compiled_self_i[rule_index, 0])
            slot_index = int(compiled_self_i[rule_index, 1])
            if (
                material_id <= 0
                or material_id >= int(material_table.shape[0])
                or slot_index < 0
                or slot_index >= 8
            ):
                direct_action_spans = False
                break
            action_index = int(material_table[material_id]["reaction_slots"][slot_index])
            if action_index < 0 or action_index >= MAX_ACTIONS:
                direct_action_spans = False
                break
            compiled_self_span_i[rule_index, 0] = action_index
    if count <= 0:
        direct_action_spans = False
    material_tags = np.zeros((MAX_MATERIALS, 4), dtype=np.uint32)
    material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
    material_tags[:material_count, 0] = material_table[:material_count]["material_tag_mask"]
    material_tags[:material_count, 1] = material_table[:material_count]["gas_tag_mask"]
    material_tags[:material_count, 2] = material_table[:material_count]["light_tag_mask"]
    if count > 0:
        grouped_materials = compiled_self_i[:count, 0]
        for material_id in np.unique(grouped_materials):
            material_id = int(material_id)
            if material_id <= 0 or material_id >= MAX_MATERIALS:
                continue
            indices = np.flatnonzero(grouped_materials == material_id)
            start = int(indices[0])
            span_count = int(indices.size)
            material_tags[material_id, 3] = np.uint32(start | (span_count << 16))
    resources.material_tags.write(material_tags.tobytes())
    resources.self_rule_i.write(compiled_self_i.tobytes())
    resources.self_rule_f.write(compiled_self_f.tobytes())
    resources.self_rule_span_i.write(compiled_self_span_i.tobytes())
    resources.self_rule_span_direct_actions = bool(direct_action_spans)
    resources.self_rule_signature = self_rule_signature



def _promote_cell_pong_to_ping(pipeline, world: "WorldEngine", resources: GPUReactionResources) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    program = pipeline.programs["promote_reaction_cell_state"]
    program["cell_grid_size"].value = (world.width, world.height)
    resources.cell_state_pong.use(location=0)
    resources.temp_pong.use(location=1)
    resources.integrity_pong.use(location=2)
    resources.velocity_pong.use(location=3)
    resources.timer_pong.use(location=4)
    resources.cell_state_ping.bind_to_image(0, read=False, write=True)
    resources.temp_ping.bind_to_image(1, read=False, write=True)
    resources.integrity_ping.bind_to_image(2, read=False, write=True)
    resources.velocity_ping.bind_to_image(3, read=False, write=True)
    resources.timer_ping.bind_to_image(4, read=False, write=True)
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
        resources.gas_ping, resources.gas_pong = resources.gas_pong, resources.gas_ping



def _promote_gas_result(pipeline, world: "WorldEngine", resources: GPUReactionResources, gas_source: Any, ambient_source: Any) -> None:
    if not pipeline._formal_reaction_state_cache_active():
        return
    with pipeline._profile_pass(world, "promote_gas_pong"):
        if gas_source is resources.gas_ping and ambient_source is resources.ambient_ping:
            return
        if gas_source is not resources.gas_pong or ambient_source is not resources.ambient_pong:
            raise RuntimeError("formal reaction gas result must use a matching ping/pong texture pair")
        resources.gas_ping, resources.gas_pong = resources.gas_pong, resources.gas_ping
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping



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
