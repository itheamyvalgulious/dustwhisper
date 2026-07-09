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
    timed_candidate_zero = np.zeros((4,), dtype=np.uint32).tobytes()
    timed_dispatch_zero = np.zeros((3,), dtype=np.uint32).tobytes()
    timed_cell_marks_zero = np.zeros((cell_count,), dtype=np.uint32).tobytes()
    def tex(size, comps=1):
        texture = ctx.texture(size, comps, dtype="f4")
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
        return texture
    resources = GPUReactionResources(
        signature=signature,
        material_ping=tex((world.width, world.height)),
        material_pong=tex((world.width, world.height)),
        phase_ping=tex((world.width, world.height)),
        phase_pong=tex((world.width, world.height)),
        temp_ping=tex((world.width, world.height)),
        temp_pong=tex((world.width, world.height)),
        integrity_ping=tex((world.width, world.height)),
        integrity_pong=tex((world.width, world.height)),
        velocity_ping=tex((world.width, world.height), 2),
        velocity_pong=tex((world.width, world.height), 2),
        timer_ping=tex((world.width, world.height), 4),
        timer_pong=tex((world.width, world.height), 4),
        ambient_ping=tex((world.gas_width, world.gas_height)),
        ambient_pong=tex((world.gas_width, world.gas_height)),
        gas_ping=ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4"),
        gas_pong=ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4"),
        flow_velocity_tex=tex((world.gas_width, world.gas_height), 2),
        active_cell_tex=tex((world.width, world.height)),
        active_gas_tex=tex((world.gas_width, world.gas_height)),
        cell_dose_tex=ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4"),
        cell_dose_pong=ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4"),
        gas_dose_tex=ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4"),
        gas_dose_pong=ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4"),
        flow_source_tex=ctx.texture_array((world.gas_width, world.gas_height, FLOW_SOURCE_LAYERS), 4, dtype="f4"),
        gas_delta_buffer=ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height * gas_count * np.dtype(np.int32).itemsize),
            dynamic=True,
        ),
        timed_candidate_count=ctx.buffer(timed_candidate_zero, dynamic=True),
        timed_candidate_list=ctx.buffer(reserve=cell_count * np.dtype(np.uint32).itemsize, dynamic=True),
        timed_candidate_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        light_dose_guarded_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        timed_candidate_marks=ctx.buffer(timed_cell_marks_zero, dynamic=True),
        timed_material_target_list=ctx.buffer(reserve=cell_count * np.dtype(np.uint32).itemsize, dynamic=True),
        timed_material_target_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        timed_material_target_marks=ctx.buffer(timed_cell_marks_zero, dynamic=True),
        trigger_lo_tex=tex((world.width, world.height), 4),
        trigger_hi_tex=tex((world.width, world.height), 4),
        deferred_scale_lo_tex=tex((world.width, world.height), 4),
        deferred_scale_hi_tex=tex((world.width, world.height), 4),
        cell_reset_tex=tex((world.width, world.height)),
        reaction_latched_tex=tex((world.width, world.height)),
        segment_cell_reset_tex=tex((world.width, world.height)),
        segment_reaction_latched_tex=tex((world.width, world.height)),
        emitted_material_mask_tex=tex((world.width, world.height)),
        local_material_out=tex((world.width, world.height)),
        local_phase_out=tex((world.width, world.height)),
        local_temp_out=tex((world.width, world.height)),
        local_integrity_out=tex((world.width, world.height)),
        local_timer_out=tex((world.width, world.height), 4),
        local_deferred_lo_out=ctx.texture_array((world.width, world.height, 2), 4, dtype="f4"),
        local_deferred_hi_out=ctx.texture_array((world.width, world.height, 2), 4, dtype="f4"),
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
        mm_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mm_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mm_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        mg_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        rule_lhs_candidate_masks=ctx.buffer(
            reserve=MAX_MATERIALS * RULE_CANDIDATE_VECS * 4 * np.dtype(np.uint32).itemsize,
            dynamic=True,
        ),
        ml_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        ml_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        ml_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gg_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gg_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gg_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_i=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_f=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        gl_rule_tags=ctx.buffer(reserve=MAX_RULES * 4 * 4, dynamic=True),
        self_rule_i=ctx.buffer(reserve=MAX_SELF_RULES * 4 * 4, dynamic=True),
        self_rule_f=ctx.buffer(reserve=MAX_SELF_RULES * 4 * 4, dynamic=True),
    )
    resources.gas_ping.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.cell_dose_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.cell_dose_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_dose_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.gas_dose_pong.filter = (ctx.NEAREST, ctx.NEAREST)
    resources.flow_source_tex.filter = (ctx.NEAREST, ctx.NEAREST)
    pipeline.resources = resources
    return resources

