from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import time
from contextlib import contextmanager

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_reactions import (
    FLOW_SOURCE_LAYERS,
    GPUReactionResources,
    MAX_ACTIONS,
    MAX_EMITTED_LIGHTS,
    MAX_MATERIALS,
    MATERIAL_PAIR_RULE_I_ENTRY_COUNT,
    MAX_RULES,
    MAX_SELF_RULES,
    RULE_CANDIDATE_VECS,
)


def _record_profile_pass(
    pipeline,
    profile: dict[str, Any],
    name: str,
    elapsed_ms: float,
    *,
    gpu_timed: bool,
) -> None:
    entry = {
        "name": str(name),
        "cpu_ms": elapsed_ms,
        "gpu_ms": elapsed_ms if gpu_timed else None,
    }
    profile["passes"].append(entry)
    summary = profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
    summary["count"] += 1
    summary["cpu_ms"] += elapsed_ms
    if gpu_timed:
        summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

# ``_profile_pass`` is inherited from GPUPipelineBase (it records identical
# entries inline; ``_record_profile_pass`` is retained for
# ``_profile_scoped_pass`` below).



def _upload_state_profile_scope(pipeline, reaction_group: str | None) -> str | None:
    if reaction_group is None:
        return None
    return f"{reaction_group}_upload_state"



@contextmanager
def _profile_scoped_pass(pipeline, world: "WorldEngine", scope: str | None, name: str):
    profile = pipeline.last_pass_profile if bool(getattr(world, "profile_passes_enabled", False)) else None
    ctx = world.bridge.ctx if bool(getattr(world, "profile_passes_sync", False)) else None
    if profile is not None and ctx is not None:
        ctx.finish()
    start = time.perf_counter() if profile is not None else 0.0
    try:
        yield
    finally:
        if profile is not None:
            if ctx is not None:
                ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            pipeline._record_profile_pass(profile, name, elapsed_ms, gpu_timed=ctx is not None)
            if scope is not None:
                pipeline._record_profile_pass(profile, f"{scope}.{name}", elapsed_ms, gpu_timed=ctx is not None)



def _ensure_resources(pipeline, world: "WorldEngine") -> GPUReactionResources:
    ctx = world.bridge.ctx
    assert ctx is not None
    signature = (
        world.width,
        world.height,
        world.gas_width,
        world.gas_height,
        world.gas_concentration.shape[0],
        world.cell_optical_dose.shape[0],
    )
    if pipeline.resources is not None and pipeline.resources.signature == signature:
        return pipeline.resources
    segment_batch_base_key = pipeline._formal_segment_batch_base_key
    pipeline.release()
    pipeline._formal_segment_batch_base_key = segment_batch_base_key
    light_count = signature[5]
    gas_count = signature[4]
    cell_count = max(1, int(world.width * world.height))
    flow_generation_dtype = (
        "u1" if pipeline._flow_source_generation_u8_programs_enabled else "u4"
    )
    flow_generation_numpy_dtype = (
        np.uint8
        if pipeline._flow_source_generation_u8_programs_enabled
        else np.uint32
    )
    timed_candidate_zero = np.zeros((4,), dtype=np.uint32).tobytes()
    timed_dispatch_zero = np.zeros((3,), dtype=np.uint32).tobytes()
    timed_cell_marks_zero = np.zeros((cell_count,), dtype=np.uint32).tobytes()
    def tex(size, comps=1):
        texture = ctx.texture(size, comps, dtype="f4")
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
        return texture
    def uint_tex(size):
        texture = ctx.texture(size, 1, dtype="u4")
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
        return texture
    resources = GPUReactionResources(
        signature=signature,
        cell_state_ping=uint_tex((world.width, world.height)),
        cell_state_pong=uint_tex((world.width, world.height)),
        temp_ping=tex((world.width, world.height)),
        temp_pong=tex((world.width, world.height)),
        integrity_ping=tex((world.width, world.height)),
        integrity_pong=tex((world.width, world.height)),
        velocity_ping=tex((world.width, world.height), 2),
        velocity_pong=tex((world.width, world.height), 2),
        timer_ping=uint_tex((world.width, world.height)),
        timer_pong=uint_tex((world.width, world.height)),
        ambient_ping=tex((world.gas_width, world.gas_height)),
        ambient_pong=tex((world.gas_width, world.gas_height)),
        gas_ping=ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4"),
        gas_pong=ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4"),
        flow_velocity_tex=tex((world.gas_width, world.gas_height), 2),
        active_cell_tex=tex((world.width, world.height)),
        expanded_active_tile_tex=ctx.texture(
            (world.active.tile_width, world.active.tile_height),
            1,
            dtype="u1",
        ),
        active_gas_tex=tex((world.gas_width, world.gas_height)),
        cell_dose_tex=ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4"),
        cell_dose_pong=ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4"),
        gas_dose_tex=ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4"),
        gas_dose_pong=ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4"),
        flow_source_tex=ctx.texture_array((world.gas_width, world.gas_height, FLOW_SOURCE_LAYERS), 4, dtype="f4"),
        flow_source_generation_tex=ctx.texture_array(
            (world.gas_width, world.gas_height, FLOW_SOURCE_LAYERS),
            1,
            dtype=flow_generation_dtype,
        ),
        gas_delta_buffer=ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height * gas_count * np.dtype(np.int32).itemsize),
            dynamic=True,
        ),
        timed_candidate_count=ctx.buffer(timed_candidate_zero, dynamic=True),
        timed_candidate_list=ctx.buffer(reserve=cell_count * np.dtype(np.uint32).itemsize, dynamic=True),
        timed_candidate_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        light_dose_guarded_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        timed_candidate_marks=ctx.buffer(timed_cell_marks_zero, dynamic=True),
        timed_material_target_list=ctx.buffer(reserve=2 * cell_count * np.dtype(np.uint32).itemsize, dynamic=True),
        timed_material_target_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        timed_material_target_marks=ctx.buffer(timed_cell_marks_zero, dynamic=True),
        trigger_lo_tex=tex((world.width, world.height), 4),
        trigger_hi_tex=tex((world.width, world.height), 4),
        deferred_scale_lo_tex=tex((world.width, world.height), 4),
        deferred_scale_hi_tex=tex((world.width, world.height), 4),
        cell_reset_tex=tex((world.width, world.height)),
        reaction_latched_tex=tex((world.width, world.height)),
        segment_cell_meta_tex=tex((world.width, world.height), 2),
        emitted_material_mask_tex=tex((world.width, world.height)),
        local_cell_state_out=uint_tex((world.width, world.height)),
        handoff_material_tex=tex((world.width, world.height)),
        handoff_phase_tex=tex((world.width, world.height)),
        handoff_flags_tex=tex((world.width, world.height)),
        local_temp_out=tex((world.width, world.height)),
        local_integrity_out=tex((world.width, world.height)),
        local_timer_out=uint_tex((world.width, world.height)),
        local_deferred_lo_out=ctx.texture_array((world.width, world.height, 2), 4, dtype="f4"),
        local_deferred_hi_out=ctx.texture_array((world.width, world.height, 2), 4, dtype="f4"),
        local_deferred_packed_out=ctx.texture((world.width, world.height), 2, dtype="u4"),
        local_cell_meta_out=tex((world.width, world.height), 2),
        local_emit_cell_lo_out=tex((world.width, world.height), 4),
        local_emit_cell_hi_out=tex((world.width, world.height), 4),
        material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        material_tags=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        gas_tags=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        material_slots_lo=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        material_slots_hi=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        action_meta=ctx.buffer(reserve=MAX_ACTIONS * 4 * 4, dynamic=True),
        light_emitter_buffer=ctx.buffer(reserve=MAX_EMITTED_LIGHTS * 2 * 4 * 4, dynamic=True),
        light_emitter_count=ctx.buffer(reserve=16 * 4, dynamic=True),
        random_targets=ctx.buffer(reserve=MAX_MATERIALS * 4, dynamic=True),
        action_i=ctx.buffer(reserve=MAX_ACTIONS * 4 * 4, dynamic=True),
        action_f=ctx.buffer(reserve=MAX_ACTIONS * 4 * 4, dynamic=True),
        material_pair_action_i=ctx.buffer(reserve=MAX_ACTIONS * 4 * 4, dynamic=True),
        material_pair_action_f=ctx.buffer(reserve=MAX_ACTIONS * 4 * 4, dynamic=True),
        mm_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mm_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mm_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        material_pair_rule_i=ctx.buffer(
            reserve=MATERIAL_PAIR_RULE_I_ENTRY_COUNT * 4 * 4,
            dynamic=True,
        ),
        material_pair_rule_f=ctx.buffer(reserve=(MAX_RULES * 2 + 1) * 4 * 4, dynamic=True),
        material_pair_rule_tags=ctx.buffer(reserve=(MAX_RULES * 2 + 1) * 4 * 4, dynamic=True),
        material_pair_lhs_candidate_masks=ctx.buffer(
            reserve=MAX_MATERIALS * RULE_CANDIDATE_VECS * 2 * 4 * np.dtype(np.uint32).itemsize,
            dynamic=True,
        ),
        material_pair_terminal_material_tables=ctx.buffer(
            reserve=MAX_MATERIALS * (6 * 4 * np.dtype(np.uint32).itemsize + np.dtype(np.uint32).itemsize),
            dynamic=True,
        ),
        material_pair_terminal_action_tables=ctx.buffer(
            reserve=(MAX_ACTIONS * 3 * 4 + MAX_MATERIALS) * np.dtype(np.uint32).itemsize,
            dynamic=True,
        ),
        material_pair_terminal_rule_tables=ctx.buffer(
            reserve=(
                MATERIAL_PAIR_RULE_I_ENTRY_COUNT * 4
                + (MAX_RULES * 2 + 1) * 4 * 2
                + MAX_MATERIALS * RULE_CANDIDATE_VECS * 2 * 4
            )
            * np.dtype(np.uint32).itemsize,
            dynamic=True,
        ),
        rule_lhs_candidate_masks=ctx.buffer(
            reserve=MAX_MATERIALS * RULE_CANDIDATE_VECS * 4 * np.dtype(np.uint32).itemsize,
            dynamic=True,
        ),
        ml_rule_i=ctx.buffer(reserve=(MAX_RULES + 1) * 4 * 4, dynamic=True),
        ml_rule_f=ctx.buffer(reserve=(MAX_RULES + 1) * 4 * 4, dynamic=True),
        ml_rule_tags=ctx.buffer(reserve=(MAX_RULES + 1) * 4 * 4, dynamic=True),
        gg_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gg_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gg_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        self_rule_i=ctx.buffer(reserve=MAX_SELF_RULES * 4 * 4, dynamic=True),
        self_rule_f=ctx.buffer(reserve=MAX_SELF_RULES * 4 * 4, dynamic=True),
        self_rule_span_i=ctx.buffer(reserve=MAX_SELF_RULES * 4 * 4, dynamic=True),
    )
    resources.gas_ping.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.expanded_active_tile_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.cell_dose_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.cell_dose_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_dose_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_dose_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.flow_source_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.flow_source_generation_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.flow_source_generation_tex.write(
        np.zeros(
            (FLOW_SOURCE_LAYERS, world.gas_height, world.gas_width),
            dtype=flow_generation_numpy_dtype,
        ).tobytes()
    )
    resources.local_deferred_packed_out.filter = (ctx.NEAREST, ctx.NEAREST)
    pipeline.resources = resources
    return resources
