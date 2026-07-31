from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.gpu import CONSUME_POLICY_IDS, DIRECTION_IDS
from oracle_game.sim.gpu_motion_constants import GRAVITY_CELLS_PER_SECOND_SQ
from oracle_game.types import CellFlag, Phase

LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_ACTIONS = 128
MAX_RULES = 256
RULE_CANDIDATE_WORDS = (MAX_RULES + 31) // 32
RULE_CANDIDATE_VECS = (RULE_CANDIDATE_WORDS + 3) // 4
MAX_SELF_RULES = 256
MAX_MATERIAL_LIGHT_PACKED_RULES = 8
MAX_MATERIAL_PAIR_PACKED_RULES = 8
MATERIAL_LIGHT_PACKED_HEADER_OFFSET = MAX_RULES * 2 + 1
MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET = MATERIAL_LIGHT_PACKED_HEADER_OFFSET + MAX_MATERIALS
MATERIAL_PAIR_PACKED_HEADER_OFFSET = MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET = MATERIAL_PAIR_PACKED_HEADER_OFFSET + MAX_MATERIALS
MATERIAL_PAIR_RULE_I_ENTRY_COUNT = MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET + MAX_RULES
FLOW_SOURCE_LAYERS = 32
FLOW_SOURCE_GENERATION_BINDING = 7
# The self shader never reads gas_tags. Fused variants reuse its binding 7 for
# gas deltas while retaining the light-emitter SSBO at binding 14.
SELF_FUSED_GAS_DELTA_BINDING = 7
SELF_FUSED_FLOW_SOURCE_BINDING = 6
SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING = 1
MAX_EMITTED_LIGHTS = 256
GAS_DELTA_FIXED_SCALE = 1000000
LIGHT_DOSE_GUARD_BUFFER = "optics_light_dose_guard"
LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING = 12
LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING = 13

TYPE_NONE = 0
TYPE_HARM = 1
TYPE_MODIFY_TEMPERATURE = 2
TYPE_CONVERT_MATERIAL = 3
TYPE_DEFERRED = 4
TYPE_MODIFY_GAS = 5
TYPE_EMIT_LIGHT = 6
TYPE_EMIT_MATERIAL = 7
ACTION_FLAG_RANDOM_TARGET = 1
ACTION_FLAG_ALLOW_SUBUNIT_SCALE = 2
CONSUME_POLICY_NONE = int(CONSUME_POLICY_IDS["none"])
CONSUME_POLICY_LHS = int(CONSUME_POLICY_IDS["lhs"])
CONSUME_POLICY_RHS = int(CONSUME_POLICY_IDS["rhs"])
CONSUME_POLICY_BOTH = int(CONSUME_POLICY_IDS["both"])
DIRECT_CORE_OUTPUT_REACTION_GROUPS = frozenset(
    (
        "timed",
        "self",
        "material_material",
        "material_gas",
        "material_pair_fused",
        "material_light",
    )
)

# Superset of every {{NAME}} marker referenced by any reaction shader; the
# loader ignores unused keys, so one shared dict suffices for all passes.
_SHADER_SUBS = {
    "ACTION_FLAG_ALLOW_SUBUNIT_SCALE": ACTION_FLAG_ALLOW_SUBUNIT_SCALE,
    "CONSUME_POLICY_BOTH": CONSUME_POLICY_BOTH,
    "CONSUME_POLICY_LHS": CONSUME_POLICY_LHS,
    "CONSUME_POLICY_RHS": CONSUME_POLICY_RHS,
    "DIRECTION_ALL": DIRECTION_IDS["all"],
    "DIRECTION_DOWN": DIRECTION_IDS["down"],
    "DIRECTION_LEFT": DIRECTION_IDS["left"],
    "DIRECTION_RANDOM": DIRECTION_IDS["random"],
    "DIRECTION_RIGHT": DIRECTION_IDS["right"],
    "DIRECTION_SPEED": DIRECTION_IDS["speed"],
    "DIRECTION_UP": DIRECTION_IDS["up"],
    "ENABLE_LIGHT_EMITTER_OUTPUT": 1,
    "MATERIAL_PAIR_TERMINAL_HANDOFF": 0,
    "MATERIAL_PAIR_TERMINAL_DIRTY_FAST_EQUAL": 0,
    "MATERIAL_PAIR_TERMINAL_SEGMENT_META_ZERO": 0,
    "MATERIAL_PAIR_TERMINAL_SHARED_TRANSPOSE": 0,
    "PACK_CELL_META_IN_STATE": 0,
    "SELF_CACHE_CELL_STATE": 0,
    "SELF_RULE_DIRECT_ACTION_SPANS": 0,
    "SELF_SPARSE_INPLACE": 0,
    "DIRECT_GAS_DELTA_BINDING": 13,
    "REACTION_COUNTER_BINDING": 15,
    "BRIDGE_CELL_DOSE_BINDING": 14,
    "CLEAR_LIGHT_COUNTERS": 0,
    "FLOW_SOURCE_LAYERS": FLOW_SOURCE_LAYERS,
    "FLOW_SOURCE_GENERATION_VALIDITY": 0,
    "FLOW_SOURCE_GENERATION_BINDING": FLOW_SOURCE_GENERATION_BINDING,
    "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": "r32ui",
    "GAS_DELTA_FIXED_SCALE": GAS_DELTA_FIXED_SCALE,
    "GRAVITY_CELLS_PER_SECOND_SQ": GRAVITY_CELLS_PER_SECOND_SQ,
    "LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING": LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING,
    "LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING": LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING,
    "LOCAL_SIZE": LOCAL_SIZE,
    "LOCAL_SIZE_X": LOCAL_SIZE,
    "LOCAL_SIZE_Y": LOCAL_SIZE,
    "MAX_ACTIONS": MAX_ACTIONS,
    "MAX_EMITTED_LIGHTS": MAX_EMITTED_LIGHTS,
    "MAX_EMITTED_LIGHTS_TIMES_2": MAX_EMITTED_LIGHTS * 2,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
    "MAX_MATERIAL_LIGHT_PACKED_RULES": MAX_MATERIAL_LIGHT_PACKED_RULES,
    "MAX_MATERIAL_PAIR_PACKED_RULES": MAX_MATERIAL_PAIR_PACKED_RULES,
    "MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET": MATERIAL_LIGHT_PACKED_DESCRIPTOR_OFFSET,
    "MATERIAL_LIGHT_PACKED_HEADER_OFFSET": MATERIAL_LIGHT_PACKED_HEADER_OFFSET,
    "MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET": MATERIAL_PAIR_PACKED_DESCRIPTOR_OFFSET,
    "MATERIAL_PAIR_PACKED_HEADER_OFFSET": MATERIAL_PAIR_PACKED_HEADER_OFFSET,
    "MAX_MATERIALS_TIMES_RULE_CANDIDATE_VECS": MAX_MATERIALS * RULE_CANDIDATE_VECS,
    "MAX_RULES": MAX_RULES,
    "RULE_I_CAPACITY": MAX_RULES,
    "MAX_SELF_RULES": MAX_SELF_RULES,
    "PHASE_POWDER": int(Phase.POWDER),
    "REACTION_LATCHED_FLAG": int(CellFlag.REACTION_LATCHED),
    "REACTION_LATCHED_FLAG_SHIFTED_24": int(CellFlag.REACTION_LATCHED) << 24,
    "RULE_CANDIDATE_VECS": RULE_CANDIDATE_VECS,
    "RULE_CANDIDATE_WORDS": RULE_CANDIDATE_WORDS,
    "SELF_FUSED_FLOW_SOURCE_BINDING": SELF_FUSED_FLOW_SOURCE_BINDING,
    "SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING": SELF_FUSED_FLOW_SOURCE_GENERATION_BINDING,
    "SELF_FUSED_GAS_DELTA_BINDING": SELF_FUSED_GAS_DELTA_BINDING,
    "SELF_FUSED_GAS_OUTPUT": 0,
    "TIMED_EMIT_TARGET_PRODUCER": 0,
    "TIMED_SPARSE_INPLACE": 0,
    "TYPE_CONVERT_MATERIAL": TYPE_CONVERT_MATERIAL,
    "TYPE_DEFERRED": TYPE_DEFERRED,
    "TYPE_EMIT_LIGHT": TYPE_EMIT_LIGHT,
    "TYPE_EMIT_MATERIAL": TYPE_EMIT_MATERIAL,
    "TYPE_HARM": TYPE_HARM,
    "TYPE_MODIFY_GAS": TYPE_MODIFY_GAS,
    "TYPE_MODIFY_TEMPERATURE": TYPE_MODIFY_TEMPERATURE,
}


@dataclass(slots=True)
class GPUDeferredActionBatch:
    action_lo: np.ndarray
    action_hi: np.ndarray
    scale_lo: np.ndarray
    scale_hi: np.ndarray
    emitted_lights: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    emitted_material_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.bool_)
    )
    gpu_local_action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint32)
    )
    formal_gpu_empty: bool = False


FORMAL_GPU_EMPTY_DEFERRED_BATCH = GPUDeferredActionBatch(
    action_lo=np.zeros((0, 0, 4), dtype=np.int32),
    action_hi=np.zeros((0, 0, 4), dtype=np.int32),
    scale_lo=np.zeros((0, 0, 4), dtype=np.float32),
    scale_hi=np.zeros((0, 0, 4), dtype=np.float32),
    emitted_lights=np.zeros((0, 8), dtype=np.float32),
    emitted_material_mask=np.zeros((0, 0), dtype=np.bool_),
    gpu_local_action_counts=np.zeros((8,), dtype=np.uint32),
    formal_gpu_empty=True,
)


@dataclass(frozen=True, slots=True)
class GPUReactionBridgeInputLoads:
    cell_core: bool = True
    gas: bool = True
    ambient: bool = True
    flow_velocity: bool = True
    cell_dose: bool = True
    gas_dose: bool = True

    def any(self) -> bool:
        return any(
            (
                self.cell_core,
                self.gas,
                self.ambient,
                self.flow_velocity,
                self.cell_dose,
                self.gas_dose,
            )
        )

    def resource_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.cell_core:
            names.append("cell_core")
        if self.gas:
            names.append("gas_concentration")
        if self.ambient:
            names.append("ambient_temperature")
        if self.flow_velocity:
            names.append("flow_velocity")
        if self.cell_dose:
            names.append("cell_optical_dose")
        if self.gas_dose:
            names.append("gas_optical_dose")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class GPUReactionMaterialPairPlan:
    cache_key: tuple[object, ...]
    compiled_actions: tuple[np.ndarray, np.ndarray]
    packed_rule_i: np.ndarray
    packed_rule_f: np.ndarray
    packed_rule_tags: np.ndarray
    packed_lhs_candidate_masks: np.ndarray
    material_material_rule_count: int
    rule_count: int
    material_light_rule_count: int
    material_light_packed_descriptors: np.ndarray | None
    material_pair_packed_descriptors: np.ndarray | None
    modifies_gas: bool
    direct_modify_gas_layer_mask: int


@dataclass(slots=True)
class GPUReactionResources:
    signature: tuple[int, int, int, int, int, int]
    cell_state_ping: Any
    cell_state_pong: Any
    temp_ping: Any
    temp_pong: Any
    integrity_ping: Any
    integrity_pong: Any
    velocity_ping: Any
    velocity_pong: Any
    timer_ping: Any
    timer_pong: Any
    ambient_ping: Any
    ambient_pong: Any
    gas_ping: Any
    gas_pong: Any
    flow_velocity_tex: Any
    active_cell_tex: Any
    expanded_active_tile_tex: Any
    active_gas_tex: Any
    cell_dose_tex: Any
    cell_dose_pong: Any
    gas_dose_tex: Any
    gas_dose_pong: Any
    flow_source_tex: Any
    flow_source_generation_tex: Any
    gas_delta_buffer: Any
    timed_candidate_count: Any
    timed_candidate_list: Any
    timed_candidate_dispatch_args: Any
    light_dose_guarded_dispatch_args: Any
    timed_candidate_marks: Any
    timed_material_target_list: Any
    timed_material_target_dispatch_args: Any
    timed_material_target_marks: Any
    trigger_lo_tex: Any
    trigger_hi_tex: Any
    deferred_scale_lo_tex: Any
    deferred_scale_hi_tex: Any
    cell_reset_tex: Any
    reaction_latched_tex: Any
    segment_cell_meta_tex: Any
    emitted_material_mask_tex: Any
    local_cell_state_out: Any
    handoff_material_tex: Any
    handoff_phase_tex: Any
    handoff_flags_tex: Any
    local_temp_out: Any
    local_integrity_out: Any
    local_timer_out: Any
    local_deferred_lo_out: Any
    local_deferred_hi_out: Any
    local_deferred_packed_out: Any
    local_cell_meta_out: Any
    local_emit_cell_lo_out: Any
    local_emit_cell_hi_out: Any
    material_params: Any
    material_tags: Any
    gas_tags: Any
    material_slots_lo: Any
    material_slots_hi: Any
    action_meta: Any
    light_emitter_buffer: Any
    light_emitter_count: Any
    random_targets: Any
    action_i: Any
    action_f: Any
    material_pair_action_i: Any
    material_pair_action_f: Any
    mm_rule_i: Any
    mm_rule_f: Any
    mm_rule_tags: Any
    mg_rule_i: Any
    mg_rule_f: Any
    mg_rule_tags: Any
    material_pair_rule_i: Any
    material_pair_rule_f: Any
    material_pair_rule_tags: Any
    material_pair_lhs_candidate_masks: Any
    material_pair_terminal_material_tables: Any
    material_pair_terminal_action_tables: Any
    material_pair_terminal_rule_tables: Any
    rule_lhs_candidate_masks: Any
    ml_rule_i: Any
    ml_rule_f: Any
    ml_rule_tags: Any
    gg_rule_i: Any
    gg_rule_f: Any
    gg_rule_tags: Any
    gl_rule_i: Any
    gl_rule_f: Any
    gl_rule_tags: Any
    self_rule_i: Any
    self_rule_f: Any
    self_rule_span_i: Any
    self_rule_span_direct_actions: bool = False
    flow_source_generation: int = 0
    material_params_signature: tuple[int, int] | None = None
    material_slots_signature: tuple[int, int] | None = None
    gas_tags_signature: tuple[int, int] | None = None
    action_meta_signature: tuple[int, int] | None = None
    self_rule_signature: tuple[int, int] | None = None
    random_targets_signature: tuple[int, int, int] | None = None
    material_pair_plan_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_material_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_action_upload_key: tuple[object, ...] | None = None
    material_pair_terminal_rule_upload_key: tuple[object, ...] | None = None


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
    profile = (
        pipeline.last_pass_profile
        if bool(getattr(world, "profile_passes_enabled", False))
        else None
    )
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
                pipeline._record_profile_pass(
                    profile, f"{scope}.{name}", elapsed_ms, gpu_timed=ctx is not None
                )


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
    flow_generation_dtype = "u1" if pipeline._flow_source_generation_u8_programs_enabled else "u4"
    flow_generation_numpy_dtype = (
        np.uint8 if pipeline._flow_source_generation_u8_programs_enabled else np.uint32
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
        gas_dose_tex=ctx.texture_array(
            (world.gas_width, world.gas_height, light_count), 1, dtype="f4"
        ),
        gas_dose_pong=ctx.texture_array(
            (world.gas_width, world.gas_height, light_count), 1, dtype="f4"
        ),
        flow_source_tex=ctx.texture_array(
            (world.gas_width, world.gas_height, FLOW_SOURCE_LAYERS), 4, dtype="f4"
        ),
        flow_source_generation_tex=ctx.texture_array(
            (world.gas_width, world.gas_height, FLOW_SOURCE_LAYERS),
            1,
            dtype=flow_generation_dtype,
        ),
        gas_delta_buffer=ctx.buffer(
            reserve=max(
                4, world.gas_width * world.gas_height * gas_count * np.dtype(np.int32).itemsize
            ),
            dynamic=True,
        ),
        timed_candidate_count=ctx.buffer(timed_candidate_zero, dynamic=True),
        timed_candidate_list=ctx.buffer(
            reserve=cell_count * np.dtype(np.uint32).itemsize, dynamic=True
        ),
        timed_candidate_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        light_dose_guarded_dispatch_args=ctx.buffer(timed_dispatch_zero, dynamic=True),
        timed_candidate_marks=ctx.buffer(timed_cell_marks_zero, dynamic=True),
        timed_material_target_list=ctx.buffer(
            reserve=2 * cell_count * np.dtype(np.uint32).itemsize, dynamic=True
        ),
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
            reserve=MAX_MATERIALS
            * (6 * 4 * np.dtype(np.uint32).itemsize + np.dtype(np.uint32).itemsize),
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
