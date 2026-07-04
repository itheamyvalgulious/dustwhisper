from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from oracle_game.gpu import CONSUME_POLICY_IDS, DIRECTION_IDS, typed_material_id
from oracle_game.sim.gpu_collapse_dirty import mark_collapse_structure_dirty_tiles_from_bridge_cell_core
from oracle_game.types import ForceSource
from oracle_game.types import CellFlag, Phase, ReactionType


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_ACTIONS = 128
MAX_RULES = 256
RULE_CANDIDATE_WORDS = (MAX_RULES + 31) // 32
RULE_CANDIDATE_VECS = (RULE_CANDIDATE_WORDS + 3) // 4
MAX_SELF_RULES = 256
FLOW_SOURCE_LAYERS = 32
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
        "material_light",
    )
)


@dataclass(slots=True)
class GPUDeferredActionBatch:
    action_lo: np.ndarray
    action_hi: np.ndarray
    scale_lo: np.ndarray
    scale_hi: np.ndarray
    emitted_lights: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    emitted_material_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.bool_))
    gpu_local_action_counts: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint32))
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


@dataclass(slots=True)
class GPUReactionResources:
    signature: tuple[int, int, int, int, int, int]
    material_ping: Any
    material_pong: Any
    phase_ping: Any
    phase_pong: Any
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
    active_gas_tex: Any
    cell_dose_tex: Any
    cell_dose_pong: Any
    gas_dose_tex: Any
    gas_dose_pong: Any
    flow_source_tex: Any
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
    segment_cell_reset_tex: Any
    segment_reaction_latched_tex: Any
    emitted_material_mask_tex: Any
    local_material_out: Any
    local_phase_out: Any
    local_temp_out: Any
    local_integrity_out: Any
    local_timer_out: Any
    local_deferred_lo_out: Any
    local_deferred_hi_out: Any
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
    mm_rule_i: Any
    mm_rule_f: Any
    mm_rule_tags: Any
    mg_rule_i: Any
    mg_rule_f: Any
    mg_rule_tags: Any
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
    material_params_signature: tuple[int, int] | None = None
    material_slots_signature: tuple[int, int] | None = None
    gas_tags_signature: tuple[int, int] | None = None
    action_meta_signature: tuple[int, int] | None = None
    self_rule_signature: tuple[int, int] | None = None
    random_targets_signature: tuple[int, int, int] | None = None


class GPUReactionPipeline:
    def __init__(self) -> None:
        self.resources: GPUReactionResources | None = None
        self.programs: dict[str, Any] = {}
        self._clear_latches_program: Any | None = None
        self._clear_bridge_latches_program: Any | None = None
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_flow_velocity_upload_skipped = False
        self.last_cpu_cell_dose_upload_skipped = False
        self.last_cpu_gas_dose_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self.random_targets = np.zeros((MAX_MATERIALS,), dtype=np.int32)
        self.random_target_count = 0
        self._used_action_indices_cache: dict[tuple[object, ...], set[int] | None] = {}
        self._compiled_action_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray] | None] = {}
        self._formal_state_cache_key: tuple[object, ...] | None = None
        self._formal_active_mask_cache_key: tuple[object, ...] | None = None
        self._formal_loaded_bridge_inputs_key: tuple[object, ...] | None = None
        self._formal_loaded_bridge_inputs: set[str] = set()
        self._formal_segment_batch_base_key: tuple[object, ...] | None = None
        self._formal_segment_batch_key: tuple[object, ...] | None = None
        self._formal_light_counters_cleared_key: tuple[object, ...] | None = None
        self._formal_pending_bridge_publish_key: tuple[object, ...] | None = None
        self._formal_pending_bridge_publish: set[str] = set()
        self._formal_pending_gas_delta_key: tuple[object, ...] | None = None
        self._formal_cell_state_role_key: tuple[object, ...] | None = None
        self._formal_cell_state_read_role: str = "ping"

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    def _record_profile_pass(
        self,
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

    @contextmanager
    def _profile_pass(self, world: "WorldEngine", name: str):
        profile = self.last_pass_profile if bool(getattr(world, "profile_passes_enabled", False)) else None
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
                self._record_profile_pass(profile, name, elapsed_ms, gpu_timed=ctx is not None)

    def _upload_state_profile_scope(self, reaction_group: str | None) -> str | None:
        if reaction_group is None:
            return None
        return f"{reaction_group}_upload_state"

    @contextmanager
    def _profile_scoped_pass(self, world: "WorldEngine", scope: str | None, name: str):
        profile = self.last_pass_profile if bool(getattr(world, "profile_passes_enabled", False)) else None
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
                self._record_profile_pass(profile, name, elapsed_ms, gpu_timed=ctx is not None)
                if scope is not None:
                    self._record_profile_pass(profile, f"{scope}.{name}", elapsed_ms, gpu_timed=ctx is not None)

    def run_timed_actions(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: object | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        material_table = world.bridge.shadow_typed_tables["material_table"]
        with self._profile_pass(world, "timed_compile_actions"):
            used_indices = self._cached_used_action_indices_for_material_slots(world, material_table)
            compiled = self._compile_action_buffers_cached(world, action_table, used_indices)
        if compiled is None:
            return None
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "timed_upload_state"):
            self._upload_state(world, resources, reaction_group="timed", compiled_actions=compiled)
        upload_cell_mask, upload_gas_mask = self._active_masks_for_cell_reaction_upload(
            world,
            solve_cell_mask,
            reaction_group="timed",
        )
        with self._profile_pass(world, "timed_upload_active_masks"):
            self._upload_active_masks(
                world,
                resources,
                upload_cell_mask,
                upload_gas_mask,
                reaction_group="timed",
                load_gas_mask=False,
            )
        with self._profile_pass(world, "timed_upload_metadata"):
            self._upload_local_metadata(world, resources)
            resources.action_i.write(compiled[0].tobytes())
            resources.action_f.write(compiled[1].tobytes())
        formal_gpu_frame = self._formal_gpu_frame(world)
        self._run_local_cell_action_pass(
            world,
            resources,
            "timed_apply",
            apply_material_side_effects=self._compiled_actions_include_emit_material(compiled),
            apply_gas_side_effects=self._compiled_actions_include_modify_gas(compiled),
            modify_gas_layer_mask=self._compiled_modify_gas_layer_mask(compiled, world.gas_concentration.shape[0]),
            may_have_flow_sources=self._compiled_actions_include_flow_sources(compiled),
        )
        with self._profile_pass(world, "timed_publish_cell_state"):
            self._download_cell_state(world, resources, direct_core_outputs=formal_gpu_frame)
        with self._profile_pass(world, "timed_publish_deferred"):
            return self._download_deferred_batch(world, resources)

    def run_timed_triggers(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if not self.available(world):
            return None
        if self._formal_gpu_frame(world):
            raise RuntimeError("GPU reaction timed trigger readback is not allowed in formal GPU frames; CPU fallback is disabled")
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        self._upload_state(world, resources)
        self._upload_active_masks(
            world,
            resources,
            solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
            np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
        )
        self._upload_local_metadata(world, resources)
        program = self.programs["timed_trigger"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        resources.material_ping.use(location=0)
        resources.timer_ping.use(location=1)
        resources.active_cell_tex.use(location=2)
        resources.material_slots_lo.bind_to_storage_buffer(binding=0)
        resources.trigger_lo_tex.bind_to_image(3, read=False, write=True)
        resources.timer_pong.bind_to_image(4, read=False, write=True)
        program.run((world.width + LOCAL_SIZE - 1) // LOCAL_SIZE, (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE, 1)
        world.bridge.ctx.finish()
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_pong.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        return np.rint(
            np.frombuffer(resources.trigger_lo_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.int32)

    def run_self_triggers(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.available(world):
            return None
        if self._formal_gpu_frame(world):
            raise RuntimeError("GPU reaction self trigger readback is not allowed in formal GPU frames; CPU fallback is disabled")
        world.bridge.sync_rule_tables(world)
        self_rule_count = min(MAX_SELF_RULES, int(world.bridge.shadow_typed_tables["self_rule_table"].shape[0]))
        if self_rule_count <= 0:
            return None
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        self._upload_state(world, resources)
        self._upload_active_masks(
            world,
            resources,
            solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
            np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
        )
        self._upload_local_metadata(world, resources, include_self_rules=True)
        program = self.programs["self_trigger"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "self_rule_count", self_rule_count)
        resources.material_ping.use(location=0)
        resources.phase_ping.use(location=1)
        resources.temp_ping.use(location=2)
        resources.integrity_ping.use(location=3)
        resources.timer_ping.use(location=4)
        resources.active_cell_tex.use(location=5)
        resources.material_slots_lo.bind_to_storage_buffer(binding=0)
        resources.material_slots_hi.bind_to_storage_buffer(binding=1)
        resources.action_meta.bind_to_storage_buffer(binding=2)
        resources.self_rule_i.bind_to_storage_buffer(binding=3)
        resources.self_rule_f.bind_to_storage_buffer(binding=4)
        resources.timer_pong.bind_to_image(0, read=False, write=True)
        resources.trigger_lo_tex.bind_to_image(1, read=False, write=True)
        resources.trigger_hi_tex.bind_to_image(2, read=False, write=True)
        program.run((world.width + LOCAL_SIZE - 1) // LOCAL_SIZE, (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE, 1)
        world.bridge.ctx.finish()
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_pong.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        trigger_lo = np.rint(
            np.frombuffer(resources.trigger_lo_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.int32)
        trigger_hi = np.rint(
            np.frombuffer(resources.trigger_hi_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.int32)
        return (trigger_lo, trigger_hi)

    def run_self_actions(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: object | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["self_rule_table"]
        self_rule_count = min(MAX_SELF_RULES, int(rule_table.shape[0]))
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        material_table = world.bridge.shadow_typed_tables["material_table"]
        with self._profile_pass(world, "self_compile_actions"):
            used_indices = self._cached_used_action_indices_for_self_rules(world, rule_table, material_table)
            compiled = self._compile_action_buffers_cached(world, action_table, used_indices)
        if compiled is None:
            return None
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "self_upload_state"):
            self._upload_state(world, resources, reaction_group="self", compiled_actions=compiled)
        upload_cell_mask, upload_gas_mask = self._active_masks_for_cell_reaction_upload(
            world,
            solve_cell_mask,
            reaction_group="self",
        )
        with self._profile_pass(world, "self_upload_active_masks"):
            self._upload_active_masks(
                world,
                resources,
                upload_cell_mask,
                upload_gas_mask,
                reaction_group="self",
            )
        with self._profile_pass(world, "self_upload_metadata"):
            self._upload_local_metadata(world, resources, include_self_rules=True)
            resources.action_i.write(compiled[0].tobytes())
            resources.action_f.write(compiled[1].tobytes())
        self._run_local_cell_action_pass(
            world,
            resources,
            "self_apply",
            self_rule_count=self_rule_count,
            apply_material_side_effects=self._compiled_actions_include_emit_material(compiled),
            apply_gas_side_effects=self._compiled_actions_include_modify_gas(compiled),
            modify_gas_layer_mask=self._compiled_modify_gas_layer_mask(compiled, world.gas_concentration.shape[0]),
            may_have_flow_sources=self._compiled_actions_include_flow_sources(compiled),
        )
        with self._profile_pass(world, "self_publish_cell_state"):
            self._download_cell_state(
                world,
                resources,
                direct_core_outputs=self._formal_gpu_frame(world),
            )
        with self._profile_pass(world, "self_publish_deferred"):
            return self._download_deferred_batch(world, resources)

    def run_material_material(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: object | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_material_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        material_table = world.bridge.shadow_typed_tables["material_table"]
        used_indices = self._cached_used_action_indices_for_pair_rules(
            world,
            rule_table,
            material_table,
            rule_kind="material_material",
            lhs_tag_field="material_tag_mask",
        )
        compiled = self._compile_action_buffers_cached(world, action_table, used_indices)
        if compiled is None:
            return None
        rule_i, rule_f, rule_tags = self._compile_material_material_rules(rule_table)
        lhs_candidate_masks = self._compile_material_rule_candidate_masks(
            rule_table,
            material_table,
            selector_id_field="lhs_material_id",
            selector_tag_field="lhs_tag_mask",
            material_tag_field="material_tag_mask",
        )
        return self._run_cell_pass(
            world,
            "material_material",
            compiled,
            rule_i,
            rule_f,
            rule_tags,
            rule_count,
            solve_cell_mask,
            lhs_rule_candidate_masks=lhs_candidate_masks,
        )

    def run_material_gas(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: object | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_gas_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        material_table = world.bridge.shadow_typed_tables["material_table"]
        used_indices = self._cached_used_action_indices_for_pair_rules(
            world,
            rule_table,
            material_table,
            rule_kind="material_gas",
            lhs_tag_field="gas_tag_mask",
        )
        compiled = self._compile_action_buffers_cached(world, action_table, used_indices)
        if compiled is None:
            return None
        rule_i, rule_f, rule_tags = self._compile_material_gas_rules(rule_table)
        lhs_candidate_masks = self._compile_material_rule_candidate_masks(
            rule_table,
            material_table,
            selector_id_field="lhs_material_id",
            selector_tag_field="lhs_tag_mask",
            material_tag_field="gas_tag_mask",
        )
        return self._run_cell_pass(
            world,
            "material_gas",
            compiled,
            rule_i,
            rule_f,
            rule_tags,
            rule_count,
            solve_cell_mask,
            lhs_rule_candidate_masks=lhs_candidate_masks,
        )

    def run_material_light(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: object | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_light_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        material_table = world.bridge.shadow_typed_tables["material_table"]
        used_indices = self._cached_used_action_indices_for_pair_rules(
            world,
            rule_table,
            material_table,
            rule_kind="material_light",
            lhs_tag_field="light_tag_mask",
        )
        compiled = self._compile_action_buffers_cached(world, action_table, used_indices)
        if compiled is None:
            return None
        light_table = world.bridge.shadow_typed_tables["light_table"]
        rule_i, rule_f, rule_tags = self._compile_material_light_rules(rule_table, light_table)
        lhs_candidate_masks = self._compile_material_rule_candidate_masks(
            rule_table,
            material_table,
            selector_id_field="lhs_material_id",
            selector_tag_field="lhs_tag_mask",
            material_tag_field="light_tag_mask",
        )
        light_dose_guard = self._formal_light_dose_guard_buffer(world)
        if light_dose_guard is not None and self._formal_segment_batch_base_key is None:
            return self._run_formal_guarded_material_light(
                world,
                compiled,
                rule_i,
                rule_f,
                rule_tags,
                rule_count,
                solve_cell_mask,
                light_dose_guard,
                lhs_candidate_masks,
            )
        return self._run_cell_pass(
            world,
            "material_light",
            compiled,
            rule_i,
            rule_f,
            rule_tags,
            rule_count,
            solve_cell_mask,
            lhs_rule_candidate_masks=lhs_candidate_masks,
        )

    def _run_formal_guarded_material_light(
        self,
        world: "WorldEngine",
        compiled_actions: tuple[np.ndarray, np.ndarray],
        rule_i: np.ndarray,
        rule_f: np.ndarray,
        rule_tags: np.ndarray,
        rule_count: int,
        solve_cell_mask: object | None,
        light_dose_guard: Any,
        lhs_rule_candidate_masks: np.ndarray,
    ) -> GPUDeferredActionBatch:
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "material_light_upload_state"):
            self._upload_state(
                world,
                resources,
                reaction_group="material_light",
                compiled_actions=compiled_actions,
                light_dose_guard_buffer=light_dose_guard,
            )
        active_authoritative = self._active_scheduler_gpu_authoritative(world)
        with self._profile_pass(world, "material_light_upload_active_masks"):
            self._upload_active_masks(
                world,
                resources,
                None
                if active_authoritative
                else solve_cell_mask
                if solve_cell_mask is not None
                else np.ones((world.height, world.width), dtype=np.bool_),
                None if active_authoritative else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
                reaction_group="material_light",
                light_dose_guard_buffer=light_dose_guard,
            )
        with self._profile_pass(world, "material_light_upload_metadata"):
            self._upload_local_metadata(world, resources)
            resources.action_i.write(compiled_actions[0].tobytes())
            resources.action_f.write(compiled_actions[1].tobytes())
            resources.ml_rule_i.write(rule_i.tobytes())
            resources.ml_rule_f.write(rule_f.tobytes())
            resources.ml_rule_tags.write(rule_tags.tobytes())
            resources.rule_lhs_candidate_masks.write(lhs_rule_candidate_masks.tobytes())

        program = self.programs["material_light"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", rule_count)
        self._set_uniform_if_present(program, "rule_candidate_word_count", self._rule_candidate_word_count(rule_count))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        self._set_uniform_if_present(program, "direct_gas_delta_enabled", False)
        self._set_uniform_if_present(program, "direct_modify_gas_layer_mask", 0)
        material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        phase_in.use(location=1)
        temp_in.use(location=2)
        integrity_in.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        timer_in.use(location=6)
        resources.active_cell_tex.use(location=7)
        velocity_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.ml_rule_i.bind_to_storage_buffer(binding=3)
        resources.ml_rule_f.bind_to_storage_buffer(binding=4)
        resources.ml_rule_tags.bind_to_storage_buffer(binding=5)
        resources.material_tags.bind_to_storage_buffer(binding=6)
        resources.gas_tags.bind_to_storage_buffer(binding=7)
        resources.material_slots_lo.bind_to_storage_buffer(binding=8)
        resources.material_slots_hi.bind_to_storage_buffer(binding=9)
        resources.action_meta.bind_to_storage_buffer(binding=10)
        resources.random_targets.bind_to_storage_buffer(binding=11)
        resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
        self._bind_local_cell_action_output_images(resources, direct_core_outputs=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "material_light_shader"):
            dispatch_args = self._build_light_dose_guarded_dispatch_args(
                world,
                resources,
                light_dose_guard,
                group_x,
                group_y,
                1,
            )
            resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal light-dose guarded reactions require ModernGL ComputeShader.run_indirect")
            program.run_indirect(dispatch_args)
            self._sync_compute_writes(world.bridge.ctx)
        with self._profile_pass(world, "material_light_velocity_copy"):
            self._copy_current_velocity_to_next_role(
                world,
                resources,
                group_x,
                group_y,
                light_dose_guard_buffer=light_dose_guard,
            )

        has_rhs_consume = self._compiled_rules_include_rhs_consume(rule_tags)
        if self._compiled_actions_include_emit_material(compiled_actions):
            with self._profile_pass(world, "material_light_material_side_effects"):
                self._run_cell_material_side_effect_pass(
                    world,
                    resources,
                    direct_core_outputs=True,
                    light_dose_guard_buffer=light_dose_guard,
                )
        if self._compiled_actions_include_modify_gas(compiled_actions):
            may_have_flow_sources = self._compiled_actions_include_flow_sources(compiled_actions)
            with self._profile_pass(world, "material_light_gas_side_effects"):
                self._run_cell_gas_side_effect_pass(
                    world,
                    resources,
                    apply_action_side_effects=True,
                    may_have_flow_sources=may_have_flow_sources,
                    modify_gas_layer_mask=self._compiled_modify_gas_layer_mask(
                        compiled_actions,
                        world.gas_concentration.shape[0],
                    ),
                    direct_core_outputs=True,
                    light_dose_guard_buffer=light_dose_guard,
                )
        if has_rhs_consume:
            with self._profile_pass(world, "material_light_dose_consume"):
                self._run_material_light_dose_consume_pass(
                    world,
                    resources,
                    rule_count,
                    light_dose_guard_buffer=light_dose_guard,
                )
        with self._profile_pass(world, "material_light_publish_cell_state"):
            self._publish_bridge_cell_state(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard,
                mark_structure_dirty=self._compiled_actions_may_change_structure(compiled_actions),
            )
        with self._profile_pass(world, "material_light_publish_deferred"):
            return self._download_deferred_batch(world, resources)

    def run_gas_gas(
        self,
        world: "WorldEngine",
        *,
        solve_gas_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["gas_gas_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        used_indices = self._used_action_indices(rule_table)
        if used_indices is None:
            return None
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        compiled = self._compile_gas_action_buffers(action_table, used_indices)
        if compiled is None:
            return None
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        self._upload_state(world, resources, reaction_group="gas_gas", compiled_actions=compiled)
        self._upload_local_metadata(world, resources)
        self._upload_active_masks(
            world,
            resources,
            np.ones((world.height, world.width), dtype=np.bool_),
            solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
            reaction_group="gas_gas",
        )
        program = self.programs["gas_gas"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        resources.active_gas_tex.use(location=2)
        resources.material_ping.use(location=3)
        resources.temp_ping.use(location=4)
        resources.flow_velocity_tex.use(location=5)
        resources.gas_tags.bind_to_storage_buffer(binding=5)
        resources.material_params.bind_to_storage_buffer(binding=6)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        resources.action_i.write(compiled[0].tobytes())
        resources.action_f.write(compiled[1].tobytes())
        resources.action_i.bind_to_storage_buffer(binding=0)
        resources.action_f.bind_to_storage_buffer(binding=1)
        resources.flow_source_tex.bind_to_image(2, read=False, write=True)
        resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
        resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
        resources.local_timer_out.bind_to_image(5, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
        ping_is_primary = True
        for rule_index in range(rule_count):
            rule_compiled = self._compile_single_gas_gas_rule(rule_table[rule_index : rule_index + 1])
            resources.gg_rule_i.write(rule_compiled[0].tobytes())
            resources.gg_rule_f.write(rule_compiled[1].tobytes())
            resources.gg_rule_tags.write(rule_compiled[2].tobytes())
            resources.gg_rule_i.bind_to_storage_buffer(binding=2)
            resources.gg_rule_f.bind_to_storage_buffer(binding=3)
            resources.gg_rule_tags.bind_to_storage_buffer(binding=4)
            self._set_uniform_if_present(program, "rule_count", 1)
            if ping_is_primary:
                resources.gas_ping.use(location=0)
                resources.ambient_ping.use(location=1)
                resources.gas_pong.bind_to_image(0, read=False, write=True)
                resources.ambient_pong.bind_to_image(1, read=False, write=True)
            else:
                resources.gas_pong.use(location=0)
                resources.ambient_pong.use(location=1)
                resources.gas_ping.bind_to_image(0, read=False, write=True)
                resources.ambient_ping.bind_to_image(1, read=False, write=True)
            program.run(group_x, group_y, world.gas_concentration.shape[0])
            self._sync_compute_writes(world.bridge.ctx)
            ping_is_primary = not ping_is_primary
        final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
        final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
        if self._formal_gpu_frame(world):
            self.last_cpu_mirror_downloaded = False
            if self._formal_segment_batch_active():
                self._promote_gas_result(world, resources, final_gas, final_ambient)
                self._mark_formal_bridge_publish_pending(world, resources, "gas")
            else:
                self._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
                self._promote_gas_result(world, resources, final_gas, final_ambient)
        else:
            self.last_cpu_mirror_downloaded = True
            world.gas_concentration[:] = np.maximum(
                np.frombuffer(final_gas.read(), dtype="f4").reshape(world.gas_concentration.shape),
                0.0,
            )
            world.ambient_temperature[:] = np.frombuffer(final_ambient.read(), dtype="f4").reshape(world.ambient_temperature.shape)
        self._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=self._compiled_actions_include_flow_sources(compiled),
        )
        if self._compiled_actions_include_emit_material(compiled):
            self._scatter_local_emit_cell_outputs(world, resources)
            self._download_cell_state(world, resources)
        return self._download_deferred_batch(world, resources)

    def run_gas_light(
        self,
        world: "WorldEngine",
        *,
        solve_gas_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["gas_light_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        if self._has_unsupported_consume_policies(rule_table, {CONSUME_POLICY_NONE, CONSUME_POLICY_RHS, CONSUME_POLICY_BOTH}):
            return None
        used_indices = {int(value) for value in rule_table["result_action"].tolist() if int(value) >= 0}
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        action_compiled = self._compile_gas_light_action_buffers(
            action_table,
            used_indices,
        )
        if action_compiled is None:
            return None
        light_dose_guard = self._formal_light_dose_guard_buffer(world)
        if light_dose_guard is not None:
            return self._run_formal_guarded_gas_light(
                world,
                rule_table,
                action_compiled,
                rule_count,
                solve_gas_mask,
                light_dose_guard,
            )
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        self._upload_state(world, resources, reaction_group="gas_light", compiled_actions=action_compiled)
        self._upload_local_metadata(world, resources)
        self._upload_active_masks(
            world,
            resources,
            np.ones((world.height, world.width), dtype=np.bool_),
            solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
            reaction_group="gas_light",
        )
        light_table = world.bridge.shadow_typed_tables["light_table"]
        resources.action_i.write(action_compiled[0].tobytes())
        resources.action_f.write(action_compiled[1].tobytes())
        program = self.programs["gas_light"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        resources.gas_dose_tex.use(location=1)
        resources.active_gas_tex.use(location=2)
        resources.material_ping.use(location=4)
        resources.temp_ping.use(location=5)
        resources.flow_velocity_tex.use(location=6)
        resources.action_i.bind_to_storage_buffer(binding=2)
        resources.action_f.bind_to_storage_buffer(binding=3)
        resources.gas_tags.bind_to_storage_buffer(binding=5)
        resources.material_params.bind_to_storage_buffer(binding=6)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.flow_source_tex.bind_to_image(2, read=False, write=True)
        resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
        resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
        resources.local_timer_out.bind_to_image(5, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        ping_is_primary = True
        for rule_index in range(rule_count):
            rule_compiled = self._compile_single_gas_light_rule(rule_table[rule_index : rule_index + 1], light_table)
            resources.gl_rule_i.write(rule_compiled[0].tobytes())
            resources.gl_rule_f.write(rule_compiled[1].tobytes())
            resources.gl_rule_tags.write(rule_compiled[2].tobytes())
            resources.gl_rule_i.bind_to_storage_buffer(binding=0)
            resources.gl_rule_f.bind_to_storage_buffer(binding=1)
            resources.gl_rule_tags.bind_to_storage_buffer(binding=4)
            self._set_uniform_if_present(program, "rule_count", 1)
            if ping_is_primary:
                resources.gas_ping.use(location=0)
                resources.ambient_ping.use(location=3)
                resources.gas_pong.bind_to_image(0, read=False, write=True)
                resources.ambient_pong.bind_to_image(1, read=False, write=True)
            else:
                resources.gas_pong.use(location=0)
                resources.ambient_pong.use(location=3)
                resources.gas_ping.bind_to_image(0, read=False, write=True)
                resources.ambient_ping.bind_to_image(1, read=False, write=True)
            program.run(group_x, group_y, world.gas_concentration.shape[0])
            self._sync_compute_writes(world.bridge.ctx)
            ping_is_primary = not ping_is_primary
        final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
        final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
        if self._formal_gpu_frame(world):
            self.last_cpu_mirror_downloaded = False
            if self._formal_segment_batch_active():
                self._promote_gas_result(world, resources, final_gas, final_ambient)
                self._mark_formal_bridge_publish_pending(world, resources, "gas")
            else:
                self._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
                self._promote_gas_result(world, resources, final_gas, final_ambient)
        else:
            self.last_cpu_mirror_downloaded = True
            world.gas_concentration[:] = np.maximum(
                np.frombuffer(final_gas.read(), dtype="f4").reshape(world.gas_concentration.shape),
                0.0,
            )
            world.ambient_temperature[:] = np.frombuffer(final_ambient.read(), dtype="f4").reshape(world.ambient_temperature.shape)
        self._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=self._compiled_actions_include_flow_sources(action_compiled),
        )
        if self._compiled_actions_include_emit_material(action_compiled):
            self._scatter_local_emit_cell_outputs(world, resources)
            self._download_cell_state(world, resources)
        return self._download_deferred_batch(world, resources)

    def _run_formal_guarded_gas_light(
        self,
        world: "WorldEngine",
        rule_table: np.ndarray,
        action_compiled: tuple[np.ndarray, np.ndarray],
        rule_count: int,
        solve_gas_mask: np.ndarray | None,
        light_dose_guard: Any,
    ) -> GPUDeferredActionBatch:
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "gas_light_upload_state"):
            self._upload_state(
                world,
                resources,
                reaction_group="gas_light",
                compiled_actions=action_compiled,
                light_dose_guard_buffer=light_dose_guard,
            )
        active_authoritative = self._active_scheduler_gpu_authoritative(world)
        with self._profile_pass(world, "gas_light_upload_active_masks"):
            self._upload_active_masks(
                world,
                resources,
                None if active_authoritative else np.ones((world.height, world.width), dtype=np.bool_),
                None
                if active_authoritative
                else solve_gas_mask
                if solve_gas_mask is not None
                else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
                reaction_group="gas_light",
                light_dose_guard_buffer=light_dose_guard,
            )
        self._upload_local_metadata(world, resources)
        light_table = world.bridge.shadow_typed_tables["light_table"]
        resources.action_i.write(action_compiled[0].tobytes())
        resources.action_f.write(action_compiled[1].tobytes())
        program = self.programs["gas_light"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        resources.gas_dose_tex.use(location=1)
        resources.active_gas_tex.use(location=2)
        resources.material_ping.use(location=4)
        resources.temp_ping.use(location=5)
        resources.flow_velocity_tex.use(location=6)
        resources.action_i.bind_to_storage_buffer(binding=2)
        resources.action_f.bind_to_storage_buffer(binding=3)
        resources.gas_tags.bind_to_storage_buffer(binding=5)
        resources.material_params.bind_to_storage_buffer(binding=6)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.flow_source_tex.bind_to_image(2, read=False, write=True)
        resources.local_emit_cell_lo_out.bind_to_image(3, read=False, write=True)
        resources.local_emit_cell_hi_out.bind_to_image(4, read=False, write=True)
        resources.local_timer_out.bind_to_image(5, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(6, read=False, write=True)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_z = int(world.gas_concentration.shape[0])
        ping_is_primary = True
        for rule_index in range(rule_count):
            rule_compiled = self._compile_single_gas_light_rule(rule_table[rule_index : rule_index + 1], light_table)
            resources.gl_rule_i.write(rule_compiled[0].tobytes())
            resources.gl_rule_f.write(rule_compiled[1].tobytes())
            resources.gl_rule_tags.write(rule_compiled[2].tobytes())
            resources.gl_rule_i.bind_to_storage_buffer(binding=0)
            resources.gl_rule_f.bind_to_storage_buffer(binding=1)
            resources.gl_rule_tags.bind_to_storage_buffer(binding=4)
            self._set_uniform_if_present(program, "rule_count", 1)
            if ping_is_primary:
                resources.gas_ping.use(location=0)
                resources.ambient_ping.use(location=3)
                resources.gas_pong.bind_to_image(0, read=False, write=True)
                resources.ambient_pong.bind_to_image(1, read=False, write=True)
            else:
                resources.gas_pong.use(location=0)
                resources.ambient_pong.use(location=3)
                resources.gas_ping.bind_to_image(0, read=False, write=True)
                resources.ambient_ping.bind_to_image(1, read=False, write=True)
            with self._profile_pass(world, "gas_light_shader"):
                self._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    program,
                    light_dose_guard,
                    group_x,
                    group_y,
                    group_z,
                )
                self._sync_compute_writes(world.bridge.ctx)
            ping_is_primary = not ping_is_primary
        final_gas = resources.gas_ping if ping_is_primary else resources.gas_pong
        final_ambient = resources.ambient_ping if ping_is_primary else resources.ambient_pong
        with self._profile_pass(world, "gas_light_publish_gas_state"):
            self._publish_bridge_gas_state(
                world,
                resources,
                gas_texture=final_gas,
                ambient_texture=final_ambient,
                light_dose_guard_buffer=light_dose_guard,
            )
        self._append_flow_sources_from_gpu(
            world,
            resources,
            may_have_flow_sources=self._compiled_actions_include_flow_sources(action_compiled),
            light_dose_guard_buffer=light_dose_guard,
        )
        if self._compiled_actions_include_emit_material(action_compiled):
            self._scatter_local_emit_cell_outputs(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard,
            )
            self._publish_bridge_cell_state(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard,
            )
        return self._download_deferred_batch(world, resources)

    def clear_reaction_latches(self, world: "WorldEngine") -> bool:
        if not self.available(world):
            return False
        ctx = world.bridge.ctx
        if ctx is None:
            return False
        if self._formal_gpu_frame(world):
            self._clear_reaction_latches_on_bridge(world)
            self.last_cpu_mirror_downloaded = False
            return True
        if self._clear_latches_program is None:
            self._clear_latches_program = ctx.compute_shader(
                f"""
                #version 430
                layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
                uniform uint cell_count;
                layout(std430, binding=0) buffer CellFlags {{
                    uint flags[];
                }};
                void main() {{
                    uint index = gl_GlobalInvocationID.x;
                    if (index >= cell_count) {{
                        return;
                    }}
                    flags[index] = flags[index] & ~uint({int(CellFlag.REACTION_LATCHED)});
                }}
                """
            )
        flat_flags = np.asarray(world.cell_flags.reshape(-1), dtype=np.uint32)
        flag_buffer = ctx.buffer(flat_flags.tobytes())
        try:
            self._clear_latches_program["cell_count"].value = int(flat_flags.size)
            flag_buffer.bind_to_storage_buffer(binding=0)
            self._clear_latches_program.run((int(flat_flags.size) + 255) // 256, 1, 1)
            self._sync_compute_writes(ctx)
            world.cell_flags[:] = np.frombuffer(flag_buffer.read(), dtype=np.uint32).reshape(world.cell_flags.shape).astype(np.uint8)
        finally:
            flag_buffer.release()
        self.last_cpu_mirror_downloaded = True
        return True

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _active_scheduler_gpu_authoritative(self, world: "WorldEngine") -> bool:
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )

    def _formal_light_dose_guard_buffer(self, world: "WorldEngine") -> Any | None:
        if not self._formal_gpu_frame(world):
            return None
        bridge = world.bridge
        if LIGHT_DOSE_GUARD_BUFFER not in bridge.gpu_authoritative_resources:
            return None
        return bridge.buffers.get(LIGHT_DOSE_GUARD_BUFFER)

    def _build_light_dose_guarded_dispatch_args(
        self,
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
        program = self.programs["build_light_dose_guarded_dispatch_args"]
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
        self._sync_storage_and_indirect_writes(ctx)
        return resources.light_dose_guarded_dispatch_args

    def _run_light_dose_guarded_dispatch(
        self,
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
        dispatch_args = self._build_light_dose_guarded_dispatch_args(
            world,
            resources,
            guard_buffer,
            group_x,
            group_y,
            group_z,
        )
        program.run_indirect(dispatch_args)

    def _active_masks_for_cell_reaction_upload(
        self,
        world: "WorldEngine",
        solve_cell_mask: object | None,
        *,
        reaction_group: str | None = None,
    ) -> tuple[object | None, object | None]:
        if (
            self._active_scheduler_gpu_authoritative(world)
            and (
                solve_cell_mask is None
                or bool(getattr(solve_cell_mask, "full_gpu_authoritative", False))
                or self._reaction_state_segment(reaction_group) == "before_motion"
            )
        ):
            return None, None
        return (
            solve_cell_mask if solve_cell_mask is not None else np.ones((world.height, world.width), dtype=np.bool_),
            np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
        )

    def _reaction_state_segment(self, reaction_group: str | None) -> str | None:
        if (
            reaction_group in {"material_material", "material_gas", "material_light", "gas_gas", "gas_light"}
            and self._formal_segment_batch_base_key is not None
            and len(self._formal_segment_batch_base_key) >= 3
            and self._formal_segment_batch_base_key[2] in {"before_motion", "after_optics"}
        ):
            return str(self._formal_segment_batch_base_key[2])
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

    def _bridge_cell_core_read_role_only_load(self, reaction_group: str | None) -> bool:
        return reaction_group in DIRECT_CORE_OUTPUT_REACTION_GROUPS

    def _formal_reaction_segment_base_key(
        self,
        world: "WorldEngine",
        segment: str | None,
    ) -> tuple[object, ...] | None:
        if not self._formal_gpu_frame(world) or segment not in {"before_motion", "after_optics"}:
            return None
        return (
            id(world),
            int(getattr(world, "frame_id", 0)),
            segment,
        )

    def _formal_reaction_segment_cache_key(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        segment: str | None,
    ) -> tuple[object, ...] | None:
        base_key = self._formal_reaction_segment_base_key(world, segment)
        if base_key is None:
            return None
        return (
            *base_key,
            id(resources),
            tuple(resources.signature),
        )

    def _formal_reaction_state_cache_key(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        reaction_group: str | None,
    ) -> tuple[object, ...] | None:
        if not self._formal_gpu_frame(world):
            return None
        segment = self._reaction_state_segment(reaction_group)
        return self._formal_reaction_segment_cache_key(world, resources, segment)

    def _formal_reaction_active_mask_cache_key(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        reaction_group: str | None,
        *,
        expansion_radius: int,
        load_cell_mask: bool = True,
        load_gas_mask: bool = True,
    ) -> tuple[object, ...] | None:
        segment = self._reaction_state_segment(reaction_group)
        segment_key = self._formal_reaction_segment_cache_key(world, resources, segment)
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

    def _formal_reaction_state_cache_active(self) -> bool:
        return self._formal_state_cache_key is not None

    def _formal_segment_batch_active(self) -> bool:
        return (
            self._formal_segment_batch_key is not None
            and self._formal_state_cache_key is not None
            and self._formal_segment_batch_key == self._formal_state_cache_key
        )

    def _formal_state_key_is_before_motion(self) -> bool:
        key = self._formal_state_cache_key
        return key is not None and len(key) >= 3 and key[2] == "before_motion"

    def _formal_before_motion_cell_roles_active(self) -> bool:
        return (
            self._formal_state_key_is_before_motion()
            and self._formal_cell_state_role_key == self._formal_state_cache_key
        )

    def _formal_cell_read_role(self) -> str:
        if not self._formal_state_key_is_before_motion():
            return "ping"
        if self._formal_cell_state_role_key != self._formal_state_cache_key:
            return "ping"
        return self._formal_cell_state_read_role

    def _formal_cell_write_role(self) -> str:
        return "pong" if self._formal_cell_read_role() == "ping" else "ping"

    def _set_formal_cell_read_role(self, role: str) -> None:
        if role not in {"ping", "pong"}:
            raise ValueError(f"unsupported formal cell state role {role!r}")
        if not self._formal_state_key_is_before_motion():
            return
        self._formal_cell_state_role_key = self._formal_state_cache_key
        self._formal_cell_state_read_role = role

    def _advance_formal_cell_read_role(self) -> None:
        self._set_formal_cell_read_role(self._formal_cell_write_role())

    def _reset_formal_cell_read_role(self) -> None:
        self._formal_cell_state_role_key = None
        self._formal_cell_state_read_role = "ping"

    def _cell_role_textures(self, resources: GPUReactionResources, role: str) -> tuple[Any, Any, Any, Any, Any, Any]:
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

    def _current_cell_textures(self, resources: GPUReactionResources) -> tuple[Any, Any, Any, Any, Any, Any]:
        return self._cell_role_textures(resources, self._formal_cell_read_role())

    def _next_cell_textures(self, resources: GPUReactionResources) -> tuple[Any, Any, Any, Any, Any, Any]:
        if not self._formal_before_motion_cell_roles_active():
            return self._cell_role_textures(resources, "pong")
        return self._cell_role_textures(resources, self._formal_cell_write_role())

    def begin_formal_reaction_segment(self, world: "WorldEngine", segment: str) -> bool:
        base_key = self._formal_reaction_segment_base_key(world, segment)
        if base_key is None:
            return False
        self._formal_segment_batch_base_key = base_key
        self._formal_segment_batch_key = None
        self._formal_light_counters_cleared_key = None
        self._formal_pending_bridge_publish_key = None
        self._formal_pending_bridge_publish.clear()
        self._formal_pending_gas_delta_key = None
        self._formal_active_mask_cache_key = None
        self._formal_loaded_bridge_inputs_key = None
        self._formal_loaded_bridge_inputs.clear()
        self._reset_formal_cell_read_role()
        return True

    def end_formal_reaction_segment(self, world: "WorldEngine", segment: str) -> None:
        base_key = self._formal_reaction_segment_base_key(world, segment)
        if base_key is not None and self._formal_segment_batch_base_key != base_key:
            return
        self._formal_segment_batch_base_key = None
        self._formal_segment_batch_key = None
        self._formal_light_counters_cleared_key = None
        self._formal_pending_bridge_publish_key = None
        self._formal_pending_bridge_publish.clear()
        self._formal_pending_gas_delta_key = None
        self._formal_active_mask_cache_key = None
        self._formal_loaded_bridge_inputs_key = None
        self._formal_loaded_bridge_inputs.clear()
        self._reset_formal_cell_read_role()

    def _mark_formal_bridge_publish_pending(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *resource_names: str,
    ) -> None:
        if not self._formal_segment_batch_active():
            return
        pending_key = self._formal_segment_batch_key
        if self._formal_pending_bridge_publish_key != pending_key:
            self._formal_pending_bridge_publish_key = pending_key
            self._formal_pending_bridge_publish.clear()
        self._formal_pending_bridge_publish.update(str(name) for name in resource_names)

    def flush_formal_reaction_segment(self, world: "WorldEngine", segment: str) -> bool:
        if not self._formal_gpu_frame(world) or self.resources is None:
            return False
        segment_key = self._formal_reaction_segment_cache_key(world, self.resources, segment)
        gas_delta_flushed = self._flush_formal_segment_gas_delta(world, self.resources, segment_key)
        if (
            segment_key is None
            or self._formal_segment_batch_key != segment_key
            or self._formal_pending_bridge_publish_key != segment_key
        ):
            return gas_delta_flushed
        pending = set(self._formal_pending_bridge_publish)
        if not pending:
            self._formal_pending_bridge_publish_key = None
            return gas_delta_flushed
        if "cell" in pending:
            with self._profile_pass(world, "publish_bridge_cell"):
                self._publish_bridge_cell_state(
                    world,
                    self.resources,
                    source_role=self._formal_cell_read_role()
                    if self._formal_before_motion_cell_roles_active()
                    else None,
                    cell_reset_texture=self.resources.segment_cell_reset_tex,
                    reaction_latched_texture=self.resources.segment_reaction_latched_tex,
                )
        if "gas" in pending:
            with self._profile_pass(world, "publish_bridge_gas"):
                self._publish_bridge_gas_state(world, self.resources)
        if "dose" in pending:
            with self._profile_pass(world, "publish_bridge_dose"):
                self._publish_bridge_dose_state(world, self.resources)
        self._formal_pending_bridge_publish.clear()
        self._formal_pending_bridge_publish_key = None
        return True

    def _clear_formal_segment_gas_delta(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        segment_key: tuple[object, ...],
    ) -> None:
        if self._formal_pending_gas_delta_key == segment_key:
            return
        ctx = world.bridge.ctx
        assert ctx is not None
        gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
        clear_program = self.programs["clear_cell_gas_delta"]
        clear_program["delta_count"].value = gas_delta_count
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        with self._profile_pass(world, "cell_gas_action_delta_segment_clear"):
            clear_program.run((gas_delta_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        self._formal_pending_gas_delta_key = segment_key

    def _flush_formal_segment_gas_delta(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        segment_key: tuple[object, ...] | None,
    ) -> bool:
        if segment_key is None or self._formal_pending_gas_delta_key != segment_key:
            return False
        ctx = world.bridge.ctx
        assert ctx is not None
        apply_program = self.programs["apply_cell_gas_delta"]
        apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
        resources.gas_ping.use(location=0)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        resources.gas_pong.bind_to_image(0, read=False, write=True)
        with self._profile_pass(world, "cell_gas_action_delta_segment_apply"):
            apply_program.run(
                (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                int(world.gas_concentration.shape[0]),
            )
            self._sync_compute_writes(ctx)
        self._download_gas_state(world, resources)
        self._formal_pending_gas_delta_key = None
        return True

    def _clear_reaction_latches_on_bridge(self, world: "WorldEngine") -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction latch clearing requires bridge GPU resources for authoritative state")
        if "cell_core" not in bridge.gpu_authoritative_resources:
            world._require_gpu_authoritative_resources("reaction latch clearing", "cell_core")
            bridge.sync_world(world)
        ctx = bridge.ctx
        if self._clear_bridge_latches_program is None:
            self._clear_bridge_latches_program = ctx.compute_shader(
                f"""
                #version 430
                layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
                uniform uint cell_count;
                layout(std430, binding=0) buffer BridgeCellCore {{
                    uint cell_core[];
                }};
                void main() {{
                    uint index = gl_GlobalInvocationID.x;
                    if (index >= cell_count) {{
                        return;
                    }}
                    uint word_index = index * 5u;
                    cell_core[word_index] = cell_core[word_index] & ~uint({int(CellFlag.REACTION_LATCHED) << 24});
                }}
                """
            )
        cell_count = int(world.width * world.height)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        self._clear_bridge_latches_program["cell_count"].value = cell_count
        self._clear_bridge_latches_program.run((cell_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative("cell_core", "material")

    def release(self) -> None:
        self._formal_state_cache_key = None
        self._formal_active_mask_cache_key = None
        self._formal_loaded_bridge_inputs_key = None
        self._formal_loaded_bridge_inputs.clear()
        self._formal_segment_batch_base_key = None
        self._formal_segment_batch_key = None
        self._formal_light_counters_cleared_key = None
        self._formal_pending_bridge_publish_key = None
        self._formal_pending_bridge_publish.clear()
        self._reset_formal_cell_read_role()
        if self.resources is None:
            return
        for resource in (
            self.resources.material_ping,
            self.resources.material_pong,
            self.resources.phase_ping,
            self.resources.phase_pong,
            self.resources.temp_ping,
            self.resources.temp_pong,
            self.resources.integrity_ping,
            self.resources.integrity_pong,
            self.resources.velocity_ping,
            self.resources.velocity_pong,
            self.resources.timer_ping,
            self.resources.timer_pong,
            self.resources.ambient_ping,
            self.resources.ambient_pong,
            self.resources.gas_ping,
            self.resources.gas_pong,
            self.resources.flow_velocity_tex,
            self.resources.active_cell_tex,
            self.resources.active_gas_tex,
            self.resources.cell_dose_tex,
            self.resources.cell_dose_pong,
            self.resources.gas_dose_tex,
            self.resources.gas_dose_pong,
            self.resources.flow_source_tex,
            self.resources.gas_delta_buffer,
            self.resources.timed_candidate_count,
            self.resources.timed_candidate_list,
            self.resources.timed_candidate_dispatch_args,
            self.resources.light_dose_guarded_dispatch_args,
            self.resources.timed_candidate_marks,
            self.resources.timed_material_target_list,
            self.resources.timed_material_target_dispatch_args,
            self.resources.timed_material_target_marks,
            self.resources.trigger_lo_tex,
            self.resources.trigger_hi_tex,
            self.resources.deferred_scale_lo_tex,
            self.resources.deferred_scale_hi_tex,
            self.resources.cell_reset_tex,
            self.resources.reaction_latched_tex,
            self.resources.segment_cell_reset_tex,
            self.resources.segment_reaction_latched_tex,
            self.resources.emitted_material_mask_tex,
            self.resources.local_material_out,
            self.resources.local_phase_out,
            self.resources.local_temp_out,
            self.resources.local_integrity_out,
            self.resources.local_timer_out,
            self.resources.local_deferred_lo_out,
            self.resources.local_deferred_hi_out,
            self.resources.local_cell_meta_out,
            self.resources.local_emit_cell_lo_out,
            self.resources.local_emit_cell_hi_out,
            self.resources.material_params,
            self.resources.material_tags,
            self.resources.gas_tags,
            self.resources.material_slots_lo,
            self.resources.material_slots_hi,
            self.resources.action_meta,
            self.resources.light_emitter_buffer,
            self.resources.light_emitter_count,
            self.resources.random_targets,
            self.resources.action_i,
            self.resources.action_f,
            self.resources.mm_rule_i,
            self.resources.mm_rule_f,
            self.resources.mm_rule_tags,
            self.resources.mg_rule_i,
            self.resources.mg_rule_f,
            self.resources.mg_rule_tags,
            self.resources.rule_lhs_candidate_masks,
            self.resources.ml_rule_i,
            self.resources.ml_rule_f,
            self.resources.ml_rule_tags,
            self.resources.gg_rule_i,
            self.resources.gg_rule_f,
            self.resources.gg_rule_tags,
            self.resources.gl_rule_i,
            self.resources.gl_rule_f,
            self.resources.gl_rule_tags,
            self.resources.self_rule_i,
            self.resources.self_rule_f,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _run_cell_pass(
        self,
        world: "WorldEngine",
        program_name: str,
        compiled_actions: tuple[np.ndarray, np.ndarray],
        rule_i: np.ndarray,
        rule_f: np.ndarray,
        rule_tags: np.ndarray,
        rule_count: int,
        solve_cell_mask: object | None,
        lhs_rule_candidate_masks: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch:
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        has_rhs_consume = self._compiled_rules_include_rhs_consume(rule_tags)
        modifies_gas = self._compiled_actions_include_modify_gas(compiled_actions)
        gas_side_effects_required = modifies_gas or (program_name == "material_gas" and has_rhs_consume)
        with self._profile_pass(world, f"{program_name}_upload_state"):
            self._upload_state(
                world,
                resources,
                reaction_group=program_name,
                compiled_actions=compiled_actions,
                publishes_gas=gas_side_effects_required,
            )
        upload_cell_mask, upload_gas_mask = self._active_masks_for_cell_reaction_upload(
            world,
            solve_cell_mask,
            reaction_group=program_name,
        )
        with self._profile_pass(world, f"{program_name}_upload_active_masks"):
            self._upload_active_masks(
                world,
                resources,
                upload_cell_mask,
                upload_gas_mask,
                reaction_group=program_name,
            )
        rule_i_buffer = getattr(resources, f"{program_name[:2]}_rule_i", None)
        rule_f_buffer = getattr(resources, f"{program_name[:2]}_rule_f", None)
        rule_tags_buffer = None
        if program_name == "material_material":
            rule_i_buffer = resources.mm_rule_i
            rule_f_buffer = resources.mm_rule_f
            rule_tags_buffer = resources.mm_rule_tags
        elif program_name == "material_gas":
            rule_i_buffer = resources.mg_rule_i
            rule_f_buffer = resources.mg_rule_f
            rule_tags_buffer = resources.mg_rule_tags
        elif program_name == "material_light":
            rule_i_buffer = resources.ml_rule_i
            rule_f_buffer = resources.ml_rule_f
            rule_tags_buffer = resources.ml_rule_tags
        assert rule_i_buffer is not None and rule_f_buffer is not None and rule_tags_buffer is not None
        with self._profile_pass(world, f"{program_name}_upload_metadata"):
            self._upload_local_metadata(world, resources)
            resources.action_i.write(compiled_actions[0].tobytes())
            resources.action_f.write(compiled_actions[1].tobytes())
            rule_i_buffer.write(rule_i.tobytes())
            rule_f_buffer.write(rule_f.tobytes())
            rule_tags_buffer.write(rule_tags.tobytes())
            if lhs_rule_candidate_masks is not None:
                resources.rule_lhs_candidate_masks.write(lhs_rule_candidate_masks.tobytes())
        program = self.programs[program_name]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", rule_count)
        self._set_uniform_if_present(program, "rule_candidate_word_count", self._rule_candidate_word_count(rule_count))
        self._set_uniform_if_present(program, "has_rhs_consume", has_rhs_consume)
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        phase_in.use(location=1)
        temp_in.use(location=2)
        integrity_in.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        timer_in.use(location=6)
        resources.active_cell_tex.use(location=7)
        velocity_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        rule_i_buffer.bind_to_storage_buffer(binding=3)
        rule_f_buffer.bind_to_storage_buffer(binding=4)
        rule_tags_buffer.bind_to_storage_buffer(binding=5)
        resources.material_tags.bind_to_storage_buffer(binding=6)
        resources.gas_tags.bind_to_storage_buffer(binding=7)
        resources.material_slots_lo.bind_to_storage_buffer(binding=8)
        resources.material_slots_hi.bind_to_storage_buffer(binding=9)
        resources.action_meta.bind_to_storage_buffer(binding=10)
        resources.random_targets.bind_to_storage_buffer(binding=11)
        if program_name in {"material_material", "material_gas", "material_light"}:
            resources.rule_lhs_candidate_masks.bind_to_storage_buffer(binding=12)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        direct_core_outputs = self._formal_gpu_frame(world)
        may_have_flow_sources = (
            modifies_gas
            and self._compiled_actions_include_flow_sources(compiled_actions)
        )
        direct_action_gas_delta = (
            direct_core_outputs
            and modifies_gas
            and not has_rhs_consume
            and self._formal_segment_batch_active()
            and self._formal_segment_batch_key is not None
        )
        modify_gas_layer_mask = self._compiled_modify_gas_layer_mask(
            compiled_actions,
            world.gas_concentration.shape[0],
        )
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "direct_gas_delta_enabled", bool(direct_action_gas_delta))
        self._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
        if direct_action_gas_delta:
            assert self._formal_segment_batch_key is not None
            self._clear_formal_segment_gas_delta(world, resources, self._formal_segment_batch_key)
        self._bind_local_cell_action_output_images(resources, direct_core_outputs=direct_core_outputs)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, f"{program_name}_shader"):
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(world.bridge.ctx)
        if direct_core_outputs:
            with self._profile_pass(world, f"{program_name}_velocity_copy"):
                self._copy_current_velocity_to_next_role(world, resources, group_x, group_y)
        else:
            with self._profile_pass(world, f"{program_name}_scatter"):
                self._scatter_local_cell_action_outputs(
                    world,
                    resources,
                    group_x,
                    group_y,
                )
        if self._compiled_actions_include_emit_material(compiled_actions):
            with self._profile_pass(world, f"{program_name}_material_side_effects"):
                self._run_cell_material_side_effect_pass(
                    world,
                    resources,
                    direct_core_outputs=direct_core_outputs,
                )
        if gas_side_effects_required:
            with self._profile_pass(world, f"{program_name}_gas_side_effects"):
                self._run_cell_gas_side_effect_pass(
                    world,
                    resources,
                    apply_action_side_effects=modifies_gas,
                    material_gas_rule_count=rule_count if program_name == "material_gas" and has_rhs_consume else 0,
                    may_have_flow_sources=may_have_flow_sources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    direct_core_outputs=direct_core_outputs,
                    action_gas_delta_already_applied=direct_action_gas_delta,
                )
        if program_name == "material_light" and has_rhs_consume:
            with self._profile_pass(world, f"{program_name}_dose_consume"):
                self._run_material_light_dose_consume_pass(world, resources, rule_count)
        with self._profile_pass(world, f"{program_name}_publish_cell_state"):
            self._download_cell_state(world, resources, direct_core_outputs=direct_core_outputs)
        with self._profile_pass(world, f"{program_name}_publish_deferred"):
            return self._download_deferred_batch(world, resources)

    def _run_local_cell_action_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        program_name: str,
        *,
        self_rule_count: int = 0,
        apply_material_side_effects: bool = False,
        apply_gas_side_effects: bool = False,
        modify_gas_layer_mask: int | None = None,
        may_have_flow_sources: bool = True,
    ) -> None:
        program = self.programs[program_name]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", 0)
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        self._set_uniform_if_present(program, "self_rule_count", int(self_rule_count))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        direct_core_outputs = self._formal_gpu_frame(world)
        direct_action_gas_delta = (
            direct_core_outputs
            and program_name == "timed_apply"
            and apply_gas_side_effects
            and self._formal_segment_batch_active()
            and self._formal_segment_batch_key is not None
        )
        if modify_gas_layer_mask is None:
            modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
        self._set_uniform_if_present(program, "direct_gas_delta_enabled", bool(direct_action_gas_delta))
        self._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
        if direct_action_gas_delta:
            assert self._formal_segment_batch_key is not None
            self._clear_formal_segment_gas_delta(world, resources, self._formal_segment_batch_key)
        material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        phase_in.use(location=1)
        temp_in.use(location=2)
        integrity_in.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        timer_in.use(location=6)
        resources.active_cell_tex.use(location=7)
        velocity_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.mm_rule_i.bind_to_storage_buffer(binding=3)
        resources.mm_rule_f.bind_to_storage_buffer(binding=4)
        resources.mm_rule_tags.bind_to_storage_buffer(binding=5)
        resources.material_tags.bind_to_storage_buffer(binding=6)
        resources.gas_tags.bind_to_storage_buffer(binding=7)
        resources.material_slots_lo.bind_to_storage_buffer(binding=8)
        resources.material_slots_hi.bind_to_storage_buffer(binding=9)
        resources.action_meta.bind_to_storage_buffer(binding=10)
        resources.random_targets.bind_to_storage_buffer(binding=11)
        resources.self_rule_i.bind_to_storage_buffer(binding=12)
        resources.self_rule_f.bind_to_storage_buffer(binding=13)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        if direct_action_gas_delta:
            resources.gas_delta_buffer.bind_to_storage_buffer(binding=13)
        self._bind_local_cell_action_output_images(resources, direct_core_outputs=direct_core_outputs)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, f"{program_name}_shader"):
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(world.bridge.ctx)
        if direct_core_outputs:
            with self._profile_pass(world, f"{program_name}_velocity_copy"):
                self._copy_current_velocity_to_next_role(world, resources, group_x, group_y)
        else:
            with self._profile_pass(world, f"{program_name}_scatter"):
                self._scatter_local_cell_action_outputs(
                    world,
                    resources,
                    group_x,
                    group_y,
                )
        if apply_material_side_effects:
            with self._profile_pass(world, f"{program_name}_material_side_effects"):
                self._run_cell_material_side_effect_pass(
                    world,
                    resources,
                    direct_core_outputs=direct_core_outputs,
                )
        if apply_gas_side_effects:
            with self._profile_pass(world, f"{program_name}_gas_side_effects"):
                self._run_cell_gas_side_effect_pass(
                    world,
                    resources,
                    may_have_flow_sources=may_have_flow_sources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    direct_core_outputs=direct_core_outputs,
                    action_gas_delta_already_applied=direct_action_gas_delta,
                )

    def _run_timed_candidate_action_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        apply_material_side_effects: bool = False,
        apply_gas_side_effects: bool = False,
        modify_gas_layer_mask: int | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._prepare_timed_candidate_worklist(world, resources)
        if not self._formal_segment_batch_active():
            with self._profile_pass(world, "timed_candidate_clear_local_meta"):
                self._clear_timed_candidate_local_meta(world, resources)

        program = self.programs["timed_apply_candidates"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", 0)
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        material_in, phase_in, temp_in, integrity_in, velocity_in, timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        phase_in.use(location=1)
        temp_in.use(location=2)
        integrity_in.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        timer_in.use(location=6)
        resources.active_cell_tex.use(location=7)
        velocity_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.mm_rule_i.bind_to_storage_buffer(binding=3)
        resources.mm_rule_f.bind_to_storage_buffer(binding=4)
        resources.mm_rule_tags.bind_to_storage_buffer(binding=5)
        resources.material_tags.bind_to_storage_buffer(binding=6)
        resources.gas_tags.bind_to_storage_buffer(binding=7)
        resources.material_slots_lo.bind_to_storage_buffer(binding=8)
        resources.material_slots_hi.bind_to_storage_buffer(binding=9)
        resources.action_meta.bind_to_storage_buffer(binding=10)
        resources.random_targets.bind_to_storage_buffer(binding=11)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=12)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=13)
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        self._bind_local_cell_action_output_images(resources, direct_core_outputs=True)
        with self._profile_pass(world, "timed_apply_candidates_shader"):
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal timed reaction candidate apply requires ModernGL ComputeShader.run_indirect")
            program.run_indirect(resources.timed_candidate_dispatch_args)
            self._sync_compute_writes(ctx)
            self._sync_storage_and_indirect_writes(ctx)

        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "timed_apply_candidates_velocity_copy"):
            self._copy_current_velocity_to_next_role(world, resources, group_x, group_y)

        if apply_material_side_effects:
            with self._profile_pass(world, "timed_apply_material_side_effects"):
                self._run_cell_material_side_effect_pass(
                    world,
                    resources,
                    direct_core_outputs=True,
                    timed_candidate_outputs=True,
                )
        if apply_gas_side_effects:
            with self._profile_pass(world, "timed_apply_gas_side_effects"):
                self._run_cell_gas_side_effect_pass(
                    world,
                    resources,
                    modify_gas_layer_mask=modify_gas_layer_mask,
                    direct_core_outputs=True,
                    timed_candidate_outputs=True,
                )

    def _prepare_timed_candidate_worklist(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        setup_program = self.programs["clear_timed_candidate_worklist"]
        resources.timed_candidate_count.bind_to_storage_buffer(binding=0)
        resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=2)
        with self._profile_pass(world, "timed_candidates_clear"):
            setup_program.run(1, 1, 1)
            self._sync_storage_and_indirect_writes(ctx)

        compact_program = self.programs["compact_timed_candidates"]
        compact_program["cell_grid_size"].value = (world.width, world.height)
        material_in, _phase_in, _temp_in, _integrity_in, _velocity_in, timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        timer_in.use(location=1)
        resources.active_cell_tex.use(location=2)
        resources.material_slots_lo.bind_to_storage_buffer(binding=0)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=2)
        resources.timed_candidate_dispatch_args.bind_to_storage_buffer(binding=3)
        resources.timed_candidate_marks.bind_to_storage_buffer(binding=4)
        with self._profile_pass(world, "timed_candidates_compact"):
            compact_program.run(
                (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
            self._sync_storage_and_indirect_writes(ctx)

    def _clear_timed_candidate_local_meta(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        program = self.programs["clear_timed_candidate_local_meta"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.local_cell_meta_out.bind_to_image(0, read=False, write=True)
        program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _publish_timed_candidate_cell_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_gpu_frame(world):
            self._download_cell_state(world, resources)
            return
        rotate_formal_cell_roles = self._formal_before_motion_cell_roles_active()
        if self._formal_segment_batch_active():
            self._accumulate_timed_candidate_segment_cell_transient_state(world, resources)
            if rotate_formal_cell_roles:
                self._advance_formal_cell_read_role()
            else:
                self._promote_cell_pong_to_ping(world, resources)
            self._mark_formal_bridge_publish_pending(world, resources, "cell")
        else:
            if rotate_formal_cell_roles:
                source_role = self._formal_cell_write_role()
                self._publish_bridge_cell_state(
                    world,
                    resources,
                    source_role=source_role,
                    cell_meta_texture=resources.local_cell_meta_out,
                )
                self._set_formal_cell_read_role(source_role)
            else:
                self._publish_bridge_cell_state(
                    world,
                    resources,
                    cell_meta_texture=resources.local_cell_meta_out,
                )
                self._promote_cell_pong_to_ping(world, resources)
        self.last_cpu_mirror_downloaded = False

    def _accumulate_timed_candidate_segment_cell_transient_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
    ) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        ctx = world.bridge.ctx
        if ctx is None:
            return
        program = self.programs["accumulate_timed_candidate_segment_cell_transient_state"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.local_cell_meta_out.use(location=0)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=0)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=1)
        resources.segment_cell_reset_tex.bind_to_image(0, read=True, write=True)
        resources.segment_reaction_latched_tex.bind_to_image(1, read=True, write=True)
        with self._profile_pass(world, "accumulate_timed_candidate_segment_cell_transient_state"):
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal timed candidate segment accumulation requires indirect dispatch")
            program.run_indirect(resources.timed_candidate_dispatch_args)
            self._sync_compute_writes(ctx)

    def _bind_local_cell_action_output_images(
        self,
        resources: GPUReactionResources,
        *,
        direct_core_outputs: bool,
    ) -> None:
        if direct_core_outputs:
            material_out, phase_out, temp_out, integrity_out, _velocity_out, timer_out = self._next_cell_textures(resources)
        else:
            material_out = resources.local_material_out
            phase_out = resources.local_phase_out
            temp_out = resources.local_temp_out
            integrity_out = resources.local_integrity_out
            timer_out = resources.local_timer_out
        material_out.bind_to_image(0, read=False, write=True)
        phase_out.bind_to_image(1, read=False, write=True)
        temp_out.bind_to_image(2, read=False, write=True)
        integrity_out.bind_to_image(3, read=False, write=True)
        timer_out.bind_to_image(4, read=False, write=True)
        resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
        resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)

    def _scatter_local_cell_action_outputs(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        group_x: int,
        group_y: int,
        *,
        core_outputs_direct: bool = False,
    ) -> None:
        if core_outputs_direct:
            self._copy_current_velocity_to_next_role(world, resources, group_x, group_y)
            return

        program = self.programs["scatter_local_action_outputs"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        resources.local_material_out.use(location=0)
        resources.local_phase_out.use(location=1)
        resources.local_temp_out.use(location=2)
        resources.local_integrity_out.use(location=3)
        resources.local_timer_out.use(location=4)
        resources.local_deferred_lo_out.use(location=5)
        resources.local_deferred_hi_out.use(location=6)
        resources.local_cell_meta_out.use(location=7)
        resources.material_pong.bind_to_image(0, read=False, write=True)
        resources.phase_pong.bind_to_image(1, read=False, write=True)
        resources.temp_pong.bind_to_image(2, read=False, write=True)
        resources.integrity_pong.bind_to_image(3, read=False, write=True)
        resources.timer_pong.bind_to_image(4, read=False, write=True)
        resources.trigger_lo_tex.bind_to_image(5, read=False, write=True)
        resources.trigger_hi_tex.bind_to_image(6, read=False, write=True)
        resources.deferred_scale_lo_tex.bind_to_image(7, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(world.bridge.ctx)

        tail_program = self.programs["scatter_local_action_tail_outputs"]
        self._set_uniform_if_present(tail_program, "cell_grid_size", (world.width, world.height))
        resources.local_deferred_hi_out.use(location=5)
        resources.local_cell_meta_out.use(location=7)
        resources.deferred_scale_hi_tex.bind_to_image(0, read=False, write=True)
        resources.cell_reset_tex.bind_to_image(1, read=False, write=True)
        resources.reaction_latched_tex.bind_to_image(2, read=False, write=True)
        tail_program.run(group_x, group_y, 1)
        self._sync_compute_writes(world.bridge.ctx)

    def _copy_current_velocity_to_next_role(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        group_x: int,
        group_y: int,
        *,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        _material_in, _phase_in, _temp_in, _integrity_in, velocity_in, _timer_in = self._current_cell_textures(resources)
        _material_out, _phase_out, _temp_out, _integrity_out, velocity_out, _timer_out = self._next_cell_textures(resources)
        if velocity_in is velocity_out:
            return
        program = self.programs["copy_reaction_velocity_state"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        velocity_in.use(location=0)
        velocity_out.bind_to_image(0, read=False, write=True)
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(world.bridge.ctx)

    def _scatter_local_emit_cell_outputs(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        program = self.programs["scatter_local_emit_cell_outputs"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        resources.local_emit_cell_lo_out.use(location=0)
        resources.local_emit_cell_hi_out.use(location=1)
        resources.local_timer_out.use(location=2)
        resources.local_cell_meta_out.use(location=3)
        resources.material_pong.bind_to_image(0, read=False, write=True)
        resources.phase_pong.bind_to_image(1, read=False, write=True)
        resources.temp_pong.bind_to_image(2, read=False, write=True)
        resources.integrity_pong.bind_to_image(3, read=False, write=True)
        resources.velocity_pong.bind_to_image(4, read=False, write=True)
        resources.timer_pong.bind_to_image(5, read=False, write=True)
        resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(world.bridge.ctx)

    def _run_cell_gas_side_effect_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        apply_action_side_effects: bool = True,
        material_gas_rule_count: int = 0,
        may_have_flow_sources: bool = True,
        modify_gas_layer_mask: int | None = None,
        direct_core_outputs: bool = False,
        timed_candidate_outputs: bool = False,
        light_dose_guard_buffer: Any | None = None,
        action_gas_delta_already_applied: bool = False,
    ) -> None:
        if action_gas_delta_already_applied:
            if may_have_flow_sources:
                with self._profile_pass(world, "cell_gas_action_delta_flow_source_scatter"):
                    self._run_cell_gas_action_delta_pass(
                        world,
                        resources,
                        modify_gas_layer_mask=(
                            int(modify_gas_layer_mask)
                            if modify_gas_layer_mask is not None
                            else (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
                        ),
                        may_have_flow_sources=may_have_flow_sources,
                        direct_core_outputs=direct_core_outputs,
                        light_dose_guard_buffer=light_dose_guard_buffer,
                        gas_delta_already_applied=True,
                    )
                with self._profile_pass(world, "cell_gas_action_delta_flow_sources"):
                    self._append_flow_sources_from_gpu(
                        world,
                        resources,
                        may_have_flow_sources=may_have_flow_sources,
                        light_dose_guard_buffer=light_dose_guard_buffer,
                    )
            return
        if timed_candidate_outputs:
            if not apply_action_side_effects or material_gas_rule_count > 0:
                return
            if modify_gas_layer_mask is None:
                modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
            self._run_timed_candidate_gas_side_effect_pass(
                world,
                resources,
                modify_gas_layer_mask=int(modify_gas_layer_mask),
                may_have_flow_sources=may_have_flow_sources,
            )
            return
        if apply_action_side_effects and material_gas_rule_count <= 0:
            if modify_gas_layer_mask is None:
                modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
            self._run_cell_gas_action_delta_pass(
                world,
                resources,
                modify_gas_layer_mask=int(modify_gas_layer_mask),
                may_have_flow_sources=may_have_flow_sources,
                direct_core_outputs=direct_core_outputs,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
            return
        program = self.programs["cell_gas_side_effects"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "apply_action_side_effects", int(apply_action_side_effects))
        self._set_uniform_if_present(program, "material_gas_rule_count", int(material_gas_rule_count))
        self._set_uniform_if_present(program, "use_local_deferred_outputs", bool(direct_core_outputs))
        if material_gas_rule_count > 0 or modify_gas_layer_mask is None:
            modify_gas_layer_mask = (1 << min(31, int(world.gas_concentration.shape[0]))) - 1
        self._set_uniform_if_present(program, "modify_gas_layer_mask", int(modify_gas_layer_mask))
        material_in, phase_in, temp_in, _integrity_in, _velocity_in, _timer_in = self._current_cell_textures(resources)
        resources.gas_ping.use(location=0)
        resources.trigger_lo_tex.use(location=1)
        resources.trigger_hi_tex.use(location=2)
        resources.deferred_scale_lo_tex.use(location=3)
        resources.deferred_scale_hi_tex.use(location=4)
        material_in.use(location=5)
        phase_in.use(location=6)
        temp_in.use(location=7)
        resources.active_cell_tex.use(location=8)
        resources.local_deferred_lo_out.use(location=9)
        resources.local_deferred_hi_out.use(location=10)
        resources.action_i.bind_to_storage_buffer(binding=0)
        resources.action_f.bind_to_storage_buffer(binding=1)
        resources.mg_rule_i.bind_to_storage_buffer(binding=2)
        resources.mg_rule_f.bind_to_storage_buffer(binding=3)
        resources.mg_rule_tags.bind_to_storage_buffer(binding=4)
        resources.material_tags.bind_to_storage_buffer(binding=5)
        resources.gas_tags.bind_to_storage_buffer(binding=6)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.gas_pong.bind_to_image(0, read=False, write=True)
        resources.flow_source_tex.bind_to_image(1, read=False, write=True)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "cell_gas_side_effects_gather"):
            if light_dose_guard_buffer is not None:
                self._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    program,
                    light_dose_guard_buffer,
                    group_x,
                    group_y,
                    world.gas_concentration.shape[0],
                )
            else:
                program.run(group_x, group_y, world.gas_concentration.shape[0])
            self._sync_compute_writes(world.bridge.ctx)
        with self._profile_pass(world, "cell_gas_side_effects_publish"):
            if light_dose_guard_buffer is not None:
                self._publish_bridge_gas_state(
                    world,
                    resources,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                )
            else:
                self._download_gas_state(world, resources)
        with self._profile_pass(world, "cell_gas_side_effects_flow_sources"):
            self._append_flow_sources_from_gpu(
                world,
                resources,
                may_have_flow_sources=may_have_flow_sources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )

    def _run_cell_gas_action_delta_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        modify_gas_layer_mask: int,
        may_have_flow_sources: bool,
        direct_core_outputs: bool = False,
        light_dose_guard_buffer: Any | None = None,
        gas_delta_already_applied: bool = False,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        segment_key = self._formal_segment_batch_key
        batch_formal_delta = (
            self._formal_segment_batch_active()
            and direct_core_outputs
            and not may_have_flow_sources
            and segment_key is not None
        )
        gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
        if gas_delta_already_applied:
            pass
        elif batch_formal_delta:
            self._clear_formal_segment_gas_delta(world, resources, segment_key)
        else:
            clear_program = self.programs["clear_cell_gas_delta"]
            clear_program["delta_count"].value = gas_delta_count
            resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
            with self._profile_pass(world, "cell_gas_action_delta_clear"):
                clear_groups = (gas_delta_count + LOCAL_SIZE - 1) // LOCAL_SIZE
                if light_dose_guard_buffer is not None:
                    self._run_light_dose_guarded_dispatch(
                        world,
                        resources,
                        clear_program,
                        light_dose_guard_buffer,
                        clear_groups,
                        1,
                        1,
                    )
                else:
                    clear_program.run(clear_groups, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        scatter_program = self.programs["scatter_cell_gas_action_delta"]
        scatter_program["cell_grid_size"].value = (world.width, world.height)
        scatter_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        scatter_program["gas_cell_size"].value = int(world.gas_cell_size)
        scatter_program["gas_count"].value = int(world.gas_concentration.shape[0])
        scatter_program["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
        scatter_program["use_local_deferred_outputs"].value = bool(direct_core_outputs)
        scatter_program["gas_delta_already_applied"].value = bool(gas_delta_already_applied)
        resources.trigger_lo_tex.use(location=0)
        resources.trigger_hi_tex.use(location=1)
        resources.deferred_scale_lo_tex.use(location=2)
        resources.deferred_scale_hi_tex.use(location=3)
        resources.active_cell_tex.use(location=4)
        resources.local_deferred_lo_out.use(location=5)
        resources.local_deferred_hi_out.use(location=6)
        resources.action_i.bind_to_storage_buffer(binding=0)
        resources.action_f.bind_to_storage_buffer(binding=1)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.flow_source_tex.bind_to_image(0, read=False, write=True)
        with self._profile_pass(world, "cell_gas_action_delta_scatter"):
            scatter_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
            scatter_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
            if light_dose_guard_buffer is not None:
                self._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    scatter_program,
                    light_dose_guard_buffer,
                    scatter_group_x,
                    scatter_group_y,
                    1,
                )
            else:
                scatter_program.run(scatter_group_x, scatter_group_y, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        if batch_formal_delta or gas_delta_already_applied:
            return

        apply_program = self.programs["apply_cell_gas_delta"]
        apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
        resources.gas_ping.use(location=0)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        resources.gas_pong.bind_to_image(0, read=False, write=True)
        with self._profile_pass(world, "cell_gas_action_delta_apply"):
            apply_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            apply_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            apply_group_z = int(world.gas_concentration.shape[0])
            if light_dose_guard_buffer is not None:
                self._run_light_dose_guarded_dispatch(
                    world,
                    resources,
                    apply_program,
                    light_dose_guard_buffer,
                    apply_group_x,
                    apply_group_y,
                    apply_group_z,
                )
            else:
                apply_program.run(apply_group_x, apply_group_y, apply_group_z)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "cell_gas_action_delta_publish"):
            if light_dose_guard_buffer is not None:
                self._publish_bridge_gas_state(
                    world,
                    resources,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                )
            else:
                self._download_gas_state(world, resources)
        with self._profile_pass(world, "cell_gas_action_delta_flow_sources"):
            self._append_flow_sources_from_gpu(
                world,
                resources,
                may_have_flow_sources=may_have_flow_sources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )

    def _run_timed_candidate_gas_side_effect_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        modify_gas_layer_mask: int,
        may_have_flow_sources: bool,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        segment_key = self._formal_segment_batch_key
        batch_formal_delta = (
            self._formal_segment_batch_active()
            and not may_have_flow_sources
            and segment_key is not None
        )
        gas_delta_count = int(world.gas_width * world.gas_height * world.gas_concentration.shape[0])
        if batch_formal_delta:
            self._clear_formal_segment_gas_delta(world, resources, segment_key)
        else:
            clear_program = self.programs["clear_cell_gas_delta"]
            clear_program["delta_count"].value = gas_delta_count
            resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
            with self._profile_pass(world, "cell_gas_action_delta_clear"):
                clear_program.run((gas_delta_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        scatter_program = self.programs["scatter_cell_gas_action_delta_candidates"]
        scatter_program["cell_grid_size"].value = (world.width, world.height)
        scatter_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        scatter_program["gas_cell_size"].value = int(world.gas_cell_size)
        scatter_program["gas_count"].value = int(world.gas_concentration.shape[0])
        scatter_program["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
        resources.local_deferred_lo_out.use(location=0)
        resources.local_deferred_hi_out.use(location=1)
        resources.action_i.bind_to_storage_buffer(binding=0)
        resources.action_f.bind_to_storage_buffer(binding=1)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=3)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=4)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.flow_source_tex.bind_to_image(0, read=False, write=True)
        with self._profile_pass(world, "cell_gas_action_delta_scatter_candidates"):
            if not hasattr(scatter_program, "run_indirect"):
                raise RuntimeError("formal timed gas side-effect scatter requires indirect dispatch")
            scatter_program.run_indirect(resources.timed_candidate_dispatch_args)
            ctx.memory_barrier(
                ctx.SHADER_STORAGE_BARRIER_BIT
                | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
                | ctx.TEXTURE_FETCH_BARRIER_BIT
            )
        if batch_formal_delta:
            return

        apply_program = self.programs["apply_cell_gas_delta"]
        apply_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        apply_program["gas_count"].value = int(world.gas_concentration.shape[0])
        resources.gas_ping.use(location=0)
        resources.gas_delta_buffer.bind_to_storage_buffer(binding=0)
        resources.gas_pong.bind_to_image(0, read=False, write=True)
        with self._profile_pass(world, "cell_gas_action_delta_apply"):
            apply_program.run(
                (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                int(world.gas_concentration.shape[0]),
            )
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "cell_gas_action_delta_publish"):
            self._download_gas_state(world, resources)
        with self._profile_pass(world, "cell_gas_action_delta_flow_sources"):
            self._append_flow_sources_from_gpu(
                world,
                resources,
                may_have_flow_sources=may_have_flow_sources,
            )

    def _run_material_light_dose_consume_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        rule_count: int,
        *,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        cell_program = self.programs["material_light_cell_dose_consume"]
        self._set_uniform_if_present(cell_program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(cell_program, "light_count", world.cell_optical_dose.shape[0])
        self._set_uniform_if_present(cell_program, "rule_count", int(rule_count))
        resources.cell_dose_tex.use(location=0)
        resources.material_ping.use(location=1)
        resources.phase_ping.use(location=2)
        resources.temp_ping.use(location=3)
        resources.active_cell_tex.use(location=4)
        resources.ml_rule_i.bind_to_storage_buffer(binding=0)
        resources.ml_rule_f.bind_to_storage_buffer(binding=1)
        resources.ml_rule_tags.bind_to_storage_buffer(binding=2)
        resources.material_tags.bind_to_storage_buffer(binding=3)
        resources.cell_dose_pong.bind_to_image(5, read=False, write=True)
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_z = world.cell_optical_dose.shape[0]
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
                world,
                resources,
                cell_program,
                light_dose_guard_buffer,
                cell_group_x,
                cell_group_y,
                cell_group_z,
            )
        else:
            cell_program.run(cell_group_x, cell_group_y, cell_group_z)
        self._sync_compute_writes(world.bridge.ctx)

        gas_program = self.programs["material_light_gas_dose_consume"]
        self._set_uniform_if_present(gas_program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(gas_program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(gas_program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(gas_program, "light_count", world.gas_optical_dose.shape[0])
        self._set_uniform_if_present(gas_program, "rule_count", int(rule_count))
        resources.gas_dose_tex.use(location=0)
        resources.cell_dose_tex.use(location=1)
        resources.material_ping.use(location=2)
        resources.phase_ping.use(location=3)
        resources.temp_ping.use(location=4)
        resources.active_cell_tex.use(location=5)
        resources.ml_rule_i.bind_to_storage_buffer(binding=0)
        resources.ml_rule_f.bind_to_storage_buffer(binding=1)
        resources.ml_rule_tags.bind_to_storage_buffer(binding=2)
        resources.material_tags.bind_to_storage_buffer(binding=3)
        resources.gas_dose_pong.bind_to_image(6, read=False, write=True)
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_z = world.gas_optical_dose.shape[0]
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
                world,
                resources,
                gas_program,
                light_dose_guard_buffer,
                gas_group_x,
                gas_group_y,
                gas_group_z,
            )
        else:
            gas_program.run(gas_group_x, gas_group_y, gas_group_z)
        self._sync_compute_writes(world.bridge.ctx)
        if light_dose_guard_buffer is not None:
            self._publish_bridge_dose_state(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
        else:
            self._download_dose_state(world, resources)

    def _run_cell_material_side_effect_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        direct_core_outputs: bool = False,
        timed_candidate_outputs: bool = False,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        if timed_candidate_outputs:
            self._run_timed_candidate_material_side_effect_pass(world, resources)
            return
        program = self.programs["cell_material_side_effects"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "use_local_deferred_outputs", bool(direct_core_outputs))
        material_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = self._current_cell_textures(resources)
        material_in.use(location=0)
        velocity_in.use(location=1)
        resources.trigger_lo_tex.use(location=2)
        resources.trigger_hi_tex.use(location=3)
        resources.deferred_scale_lo_tex.use(location=4)
        resources.deferred_scale_hi_tex.use(location=5)
        temp_in.use(location=6)
        resources.local_deferred_lo_out.use(location=7)
        resources.local_deferred_hi_out.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        material_out, phase_out, temp_out, integrity_out, velocity_out, timer_out = self._next_cell_textures(resources)
        material_out.bind_to_image(0, read=False, write=True)
        phase_out.bind_to_image(1, read=False, write=True)
        temp_out.bind_to_image(2, read=False, write=True)
        integrity_out.bind_to_image(3, read=False, write=True)
        velocity_out.bind_to_image(4, read=False, write=True)
        timer_out.bind_to_image(5, read=False, write=True)
        resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(world.bridge.ctx)

    def _run_timed_candidate_material_side_effect_pass(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        material_in, _phase_in, temp_in, _integrity_in, velocity_in, _timer_in = self._current_cell_textures(resources)

        compact_program = self.programs["compact_timed_material_targets"]
        compact_program["cell_grid_size"].value = (world.width, world.height)
        velocity_in.use(location=0)
        resources.local_deferred_lo_out.use(location=1)
        resources.local_deferred_hi_out.use(location=2)
        resources.action_i.bind_to_storage_buffer(binding=0)
        resources.timed_candidate_list.bind_to_storage_buffer(binding=1)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=2)
        resources.timed_material_target_list.bind_to_storage_buffer(binding=3)
        resources.timed_material_target_dispatch_args.bind_to_storage_buffer(binding=4)
        resources.timed_material_target_marks.bind_to_storage_buffer(binding=5)
        with self._profile_pass(world, "timed_material_targets_compact"):
            if not hasattr(compact_program, "run_indirect"):
                raise RuntimeError("formal timed material side-effect target compaction requires indirect dispatch")
            compact_program.run_indirect(resources.timed_candidate_dispatch_args)
            self._sync_storage_and_indirect_writes(ctx)

        program = self.programs["cell_material_side_effects_candidates"]
        program["cell_grid_size"].value = (world.width, world.height)
        material_in.use(location=0)
        velocity_in.use(location=1)
        temp_in.use(location=2)
        resources.local_deferred_lo_out.use(location=3)
        resources.local_deferred_hi_out.use(location=4)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.timed_candidate_count.bind_to_storage_buffer(binding=3)
        resources.timed_candidate_marks.bind_to_storage_buffer(binding=4)
        resources.timed_material_target_list.bind_to_storage_buffer(binding=5)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        material_out, phase_out, temp_out, integrity_out, velocity_out, timer_out = self._next_cell_textures(resources)
        material_out.bind_to_image(0, read=False, write=True)
        phase_out.bind_to_image(1, read=False, write=True)
        temp_out.bind_to_image(2, read=False, write=True)
        integrity_out.bind_to_image(3, read=False, write=True)
        velocity_out.bind_to_image(4, read=False, write=True)
        timer_out.bind_to_image(5, read=False, write=True)
        resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
        with self._profile_pass(world, "timed_material_targets_apply"):
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("formal timed material side-effect apply requires indirect dispatch")
            program.run_indirect(resources.timed_material_target_dispatch_args)
            self._sync_compute_writes(ctx)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _ensure_resources(self, world: "WorldEngine") -> GPUReactionResources:
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
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        segment_batch_base_key = self._formal_segment_batch_base_key
        self.release()
        self._formal_segment_batch_base_key = segment_batch_base_key
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
        self.resources = resources
        return resources

    def _ensure_programs(self, ctx: Any | None) -> None:
        if not ctx or self.programs:
            return
        active_helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            uniform int expansion_radius;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            bool source_tile_active(ivec2 tile) {{
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                int index = tile.y * tile_grid_size.x + tile.x;
                return active_tile_ttl[index] > 0;
            }}
            bool expanded_tile_active(ivec2 tile) {{
                for (int source_y = tile.y - expansion_radius; source_y <= tile.y + expansion_radius; ++source_y) {{
                    for (int source_x = tile.x - expansion_radius; source_x <= tile.x + expansion_radius; ++source_x) {{
                        if (source_tile_active(ivec2(source_x, source_y))) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}
        """
        self.programs["load_active_cell"] = ctx.compute_shader(
            active_helper
            + """
            layout(r32f, binding=1) writeonly uniform image2D active_cell_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                ivec2 tile = ivec2(
                    min(gid.x / tile_size, tile_grid_size.x - 1),
                    min(gid.y / tile_size, tile_grid_size.y - 1)
                );
                imageStore(active_cell_img, gid, vec4(expanded_tile_active(tile) ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["load_active_gas"] = ctx.compute_shader(
            active_helper
            + """
            layout(r32f, binding=1) writeonly uniform image2D active_gas_img;
            bool gas_cell_active(ivec2 gas_cell) {
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                int tile_x0 = max(0, x0 / tile_size);
                int tile_y0 = max(0, y0 / tile_size);
                int tile_x1 = min(tile_grid_size.x, (x1 + tile_size - 1) / tile_size);
                int tile_y1 = min(tile_grid_size.y, (y1 + tile_size - 1) / tile_size);
                for (int tile_y = tile_y0 - expansion_radius; tile_y < tile_y1 + expansion_radius; ++tile_y) {
                    for (int tile_x = tile_x0 - expansion_radius; tile_x < tile_x1 + expansion_radius; ++tile_x) {
                        if (source_tile_active(ivec2(tile_x, tile_y))) {
                            return true;
                        }
                    }
                }
                return false;
            }
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {
                    return;
                }
                imageStore(active_gas_img, gid, vec4(gas_cell_active(gid) ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["load_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_ping_img;
            layout(r32f, binding=1) writeonly uniform image2D material_pong_img;
            layout(r32f, binding=2) writeonly uniform image2D phase_ping_img;
            layout(r32f, binding=3) writeonly uniform image2D phase_pong_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_ping_img;
            layout(r32f, binding=5) writeonly uniform image2D temp_pong_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_ping_img;
            layout(r32f, binding=7) writeonly uniform image2D integrity_pong_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                int word_index = cell_index * 5;
                uint word0 = bridge_cell_core[word_index];
                float material = float(word0 & 0xFFFFu);
                float phase = float((word0 >> 16u) & 0xFFu);
                float temperature = uintBitsToFloat(bridge_cell_core[word_index + 2]);
                float integrity = float(bridge_cell_core[word_index + 4] & 0xFFFFu);
                imageStore(material_ping_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_ping_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(temp_ping_img, gid, vec4(temperature, 0.0, 0.0, 0.0));
                imageStore(integrity_ping_img, gid, vec4(integrity, 0.0, 0.0, 0.0));
                imageStore(material_pong_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_pong_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(temp_pong_img, gid, vec4(temperature, 0.0, 0.0, 0.0));
                imageStore(integrity_pong_img, gid, vec4(integrity, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell_role"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                int word_index = cell_index * 5;
                uint word0 = bridge_cell_core[word_index];
                float material = float(word0 & 0xFFFFu);
                float phase = float((word0 >> 16u) & 0xFFu);
                float temperature = uintBitsToFloat(bridge_cell_core[word_index + 2]);
                float integrity = float(bridge_cell_core[word_index + 4] & 0xFFFFu);
                imageStore(material_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(temp_img, gid, vec4(temperature, 0.0, 0.0, 0.0));
                imageStore(integrity_img, gid, vec4(integrity, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(rg32f, binding=0) writeonly uniform image2D velocity_ping_img;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_pong_img;
            layout(rgba32f, binding=2) writeonly uniform image2D timer_ping_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_pong_img;
            vec4 unpack_timer(uint word) {{
                return vec4(
                    float(word & 0xFFu),
                    float((word >> 8u) & 0xFFu),
                    float((word >> 16u) & 0xFFu),
                    float((word >> 24u) & 0xFFu)
                );
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int word_index = (gid.y * cell_grid_size.x + gid.x) * 5;
                vec2 velocity = unpackHalf2x16(bridge_cell_core[word_index + 1]);
                vec4 timer = unpack_timer(bridge_cell_core[word_index + 3]);
                imageStore(velocity_ping_img, gid, vec4(velocity, 0.0, 0.0));
                imageStore(timer_ping_img, gid, timer);
                imageStore(velocity_pong_img, gid, vec4(velocity, 0.0, 0.0));
                imageStore(timer_pong_img, gid, timer);
            }}
            """
        )
        self.programs["load_bridge_cell_aux_role"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(rg32f, binding=0) writeonly uniform image2D velocity_img;
            layout(rgba32f, binding=1) writeonly uniform image2D timer_img;
            vec4 unpack_timer(uint word) {{
                return vec4(
                    float(word & 0xFFu),
                    float((word >> 8u) & 0xFFu),
                    float((word >> 16u) & 0xFFu),
                    float((word >> 24u) & 0xFFu)
                );
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int word_index = (gid.y * cell_grid_size.x + gid.x) * 5;
                vec2 velocity = unpackHalf2x16(bridge_cell_core[word_index + 1]);
                vec4 timer = unpack_timer(bridge_cell_core[word_index + 3]);
                imageStore(velocity_img, gid, vec4(velocity, 0.0, 0.0));
                imageStore(timer_img, gid, timer);
            }}
            """
        )
        self.programs["load_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int species_count;
            uniform bool copy_gas;
            uniform bool copy_ambient;
            uniform bool copy_flow_velocity;
            layout(binding=0) uniform sampler2D bridge_ambient_tex;
            layout(binding=1) uniform sampler2D bridge_flow_velocity_tex;
            layout(std430, binding=0) readonly buffer BridgeGasBuffer {{
                float bridge_gas[];
            }};
            layout(r32f, binding=2) writeonly uniform image2DArray gas_ping_img;
            layout(r32f, binding=3) writeonly uniform image2DArray gas_pong_img;
            layout(r32f, binding=4) writeonly uniform image2D ambient_ping_img;
            layout(r32f, binding=5) writeonly uniform image2D ambient_pong_img;
            layout(rg32f, binding=6) writeonly uniform image2D flow_velocity_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= species_count) {{
                    return;
                }}
                if (copy_gas) {{
                    int src_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                    float gas_value = bridge_gas[src_index];
                    imageStore(gas_ping_img, ivec3(gid, species), vec4(gas_value, 0.0, 0.0, 0.0));
                    imageStore(gas_pong_img, ivec3(gid, species), vec4(gas_value, 0.0, 0.0, 0.0));
                }}
                if (species == 0) {{
                    if (copy_ambient) {{
                        float ambient = texelFetch(bridge_ambient_tex, gid, 0).x;
                        imageStore(ambient_ping_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                        imageStore(ambient_pong_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                    }}
                    if (copy_flow_velocity) {{
                        vec2 flow = texelFetch(bridge_flow_velocity_tex, gid, 0).xy;
                        imageStore(flow_velocity_img, gid, vec4(flow, 0.0, 0.0));
                    }}
                }}
            }}
            """
        )
        self.programs["load_bridge_dose"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int light_count;
            uniform bool copy_cell_dose;
            uniform bool copy_gas_dose;
            layout(std430, binding=0) readonly buffer BridgeCellDoseBuffer {{
                float bridge_cell_dose[];
            }};
            layout(std430, binding=1) readonly buffer BridgeGasDoseBuffer {{
                float bridge_gas_dose[];
            }};
            layout(r32f, binding=0) writeonly uniform image2DArray cell_dose_ping_img;
            layout(r32f, binding=1) writeonly uniform image2DArray cell_dose_pong_img;
            layout(r32f, binding=2) writeonly uniform image2DArray gas_dose_ping_img;
            layout(r32f, binding=3) writeonly uniform image2DArray gas_dose_pong_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int light = int(gl_GlobalInvocationID.z);
                if (light >= light_count) {{
                    return;
                }}
                if (copy_cell_dose && gid.x < cell_grid_size.x && gid.y < cell_grid_size.y) {{
                    int cell_index = (light * cell_grid_size.y + gid.y) * cell_grid_size.x + gid.x;
                    float value = bridge_cell_dose[cell_index];
                    imageStore(cell_dose_ping_img, ivec3(gid, light), vec4(value, 0.0, 0.0, 0.0));
                    imageStore(cell_dose_pong_img, ivec3(gid, light), vec4(value, 0.0, 0.0, 0.0));
                }}
                if (copy_gas_dose && gid.x < gas_grid_size.x && gid.y < gas_grid_size.y) {{
                    int gas_index = (light * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                    float value = bridge_gas_dose[gas_index];
                    imageStore(gas_dose_ping_img, ivec3(gid, light), vec4(value, 0.0, 0.0, 0.0));
                    imageStore(gas_dose_pong_img, ivec3(gid, light), vec4(value, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["publish_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool use_cell_meta_texture;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D temp_tex;
            layout(binding=3) uniform sampler2D integrity_tex;
            layout(binding=4) uniform sampler2D velocity_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D cell_reset_tex;
            layout(binding=7) uniform sampler2D reaction_latched_tex;
            layout(binding=8) uniform sampler2D cell_meta_tex;
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;
            layout(std430, binding=0) buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}
            vec2 current_meta(ivec2 cell) {{
                if (use_cell_meta_texture) {{
                    return texelFetch(cell_meta_tex, cell, 0).xy;
                }}
                return vec2(
                    texelFetch(cell_reset_tex, cell, 0).x,
                    texelFetch(reaction_latched_tex, cell, 0).x
                );
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                int word_index = cell_index * 5;
                uint previous_word = bridge_cell_core[word_index];
                uint flags = (previous_word >> 24u) & 0xFFu;
                vec2 meta = current_meta(gid);
                if (meta.x > 0.5) {{
                    flags = 0u;
                }}
                if (meta.y > 0.5) {{
                    flags = flags | uint({int(CellFlag.REACTION_LATCHED)});
                }}
                uint material = uint(clamp(round(texelFetch(material_tex, gid, 0).x), 0.0, 65535.0));
                uint phase = uint(clamp(round(texelFetch(phase_tex, gid, 0).x), 0.0, 255.0));
                vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                float temperature = texelFetch(temp_tex, gid, 0).x;
                uint integrity = uint(clamp(round(texelFetch(integrity_tex, gid, 0).x), 0.0, 65535.0));
                bridge_cell_core[word_index] = material | (phase << 16u) | (flags << 24u);
                bridge_cell_core[word_index + 1] = packHalf2x16(velocity);
                bridge_cell_core[word_index + 2] = floatBitsToUint(temperature);
                bridge_cell_core[word_index + 3] = pack_timer(texelFetch(timer_tex, gid, 0));
                bridge_cell_core[word_index + 4] = integrity;
                imageStore(bridge_material_img, gid, vec4(float(material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int species_count;
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(binding=1) uniform sampler2D ambient_tex;
            layout(r32f, binding=2) writeonly uniform image2D bridge_ambient_img;
            layout(std430, binding=0) buffer BridgeGasBuffer {{
                float bridge_gas[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= species_count) {{
                    return;
                }}
                if (species == 0) {{
                    float ambient = texelFetch(ambient_tex, gid, 0).x;
                    imageStore(bridge_ambient_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                }}
                int dst_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                bridge_gas[dst_index] = max(texelFetch(gas_tex, ivec3(gid, species), 0).x, 0.0);
            }}
            """
        )
        self.programs["publish_bridge_cell_dose"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int light_count;
            layout(binding=0) uniform sampler2DArray cell_dose_tex;
            layout(std430, binding=0) buffer BridgeCellDoseBuffer {{
                float bridge_cell_dose[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int light = int(gl_GlobalInvocationID.z);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y || light >= light_count) {{
                    return;
                }}
                int dst_index = (light * cell_grid_size.y + gid.y) * cell_grid_size.x + gid.x;
                bridge_cell_dose[dst_index] = max(texelFetch(cell_dose_tex, ivec3(gid, light), 0).x, 0.0);
            }}
            """
        )
        self.programs["publish_bridge_gas_dose"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int light_count;
            layout(binding=0) uniform sampler2DArray gas_dose_tex;
            layout(std430, binding=0) buffer BridgeGasDoseBuffer {{
                float bridge_gas_dose[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int light = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || light >= light_count) {{
                    return;
                }}
                int dst_index = (light * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                bridge_gas_dose[dst_index] = max(texelFetch(gas_dose_tex, ivec3(gid, light), 0).x, 0.0);
            }}
            """
        )
        self.programs["apply_bridge_flow_sources"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            uniform int active_ttl_reset;
            uniform float impulse_dt;
            layout(binding=0) uniform sampler2D flow_velocity_tex;
            layout(binding=1) uniform sampler2DArray flow_source_tex;
            layout(rg32f, binding=2) writeonly uniform image2D bridge_flow_velocity_img;
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            int ceil_div(int value, int divisor) {{
                return (value + divisor - 1) / divisor;
            }}
            void mark_flow_source_active_tiles(ivec2 gas_cell) {{
                int x0 = max(0, gas_cell.x * gas_cell_size - gas_cell_size);
                int y0 = max(0, gas_cell.y * gas_cell_size - gas_cell_size);
                int x1 = (gas_cell.x + 2) * gas_cell_size;
                int y1 = (gas_cell.y + 2) * gas_cell_size;
                int tile_x0 = clamp(x0 / tile_size, 0, tile_grid_size.x);
                int tile_y0 = clamp(y0 / tile_size, 0, tile_grid_size.y);
                int tile_x1 = clamp(ceil_div(x1, tile_size), 0, tile_grid_size.x);
                int tile_y1 = clamp(ceil_div(y1, tile_size), 0, tile_grid_size.y);
                for (int tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {{
                    for (int tile_x = tile_x0; tile_x < tile_x1; ++tile_x) {{
                        active_tile_ttl[tile_y * tile_grid_size.x + tile_x] = active_ttl_reset;
                    }}
                }}
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                vec2 velocity = texelFetch(flow_velocity_tex, gid, 0).xy;
                for (int layer = 0; layer < {FLOW_SOURCE_LAYERS}; ++layer) {{
                    vec4 source = texelFetch(flow_source_tex, ivec3(gid, layer), 0);
                    vec2 direction = source.xy;
                    float radius = source.z;
                    float strength = source.w;
                    float norm = length(direction);
                    if (norm > 1.0e-6 && radius > 0.0 && strength > 0.0) {{
                        velocity += (direction / norm) * strength * impulse_dt;
                        mark_flow_source_active_tiles(gid);
                    }}
                }}
                imageStore(bridge_flow_velocity_img, gid, vec4(velocity, 0.0, 0.0));
            }}
            """
        )
        self.programs["promote_reaction_cell_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_src_tex;
            layout(binding=1) uniform sampler2D phase_src_tex;
            layout(binding=2) uniform sampler2D temp_src_tex;
            layout(binding=3) uniform sampler2D integrity_src_tex;
            layout(binding=4) uniform sampler2D velocity_src_tex;
            layout(binding=5) uniform sampler2D timer_src_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_dst_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_dst_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_dst_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_dst_img;
            layout(rg32f, binding=4) writeonly uniform image2D velocity_dst_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_dst_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(material_dst_img, gid, texelFetch(material_src_tex, gid, 0));
                imageStore(phase_dst_img, gid, texelFetch(phase_src_tex, gid, 0));
                imageStore(temp_dst_img, gid, texelFetch(temp_src_tex, gid, 0));
                imageStore(integrity_dst_img, gid, texelFetch(integrity_src_tex, gid, 0));
                imageStore(velocity_dst_img, gid, texelFetch(velocity_src_tex, gid, 0));
                imageStore(timer_dst_img, gid, texelFetch(timer_src_tex, gid, 0));
            }}
            """
        )
        self.programs["copy_reaction_velocity_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D velocity_src_tex;
            layout(rg32f, binding=0) writeonly uniform image2D velocity_dst_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(velocity_dst_img, gid, texelFetch(velocity_src_tex, gid, 0));
            }}
            """
        )
        self.programs["promote_reaction_gas_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int gas_count;
            layout(binding=0) uniform sampler2DArray gas_src_tex;
            layout(binding=1) uniform sampler2D ambient_src_tex;
            layout(r32f, binding=0) writeonly uniform image2DArray gas_dst_img;
            layout(r32f, binding=1) writeonly uniform image2D ambient_dst_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= gas_count) {{
                    return;
                }}
                imageStore(gas_dst_img, ivec3(gid, species), texelFetch(gas_src_tex, ivec3(gid, species), 0));
                if (species == 0) {{
                    imageStore(ambient_dst_img, gid, texelFetch(ambient_src_tex, gid, 0));
                }}
            }}
            """
        )
        self.programs["promote_reaction_dose_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int light_count;
            layout(binding=0) uniform sampler2DArray cell_dose_src_tex;
            layout(binding=1) uniform sampler2DArray gas_dose_src_tex;
            layout(r32f, binding=0) writeonly uniform image2DArray cell_dose_dst_img;
            layout(r32f, binding=1) writeonly uniform image2DArray gas_dose_dst_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int light = int(gl_GlobalInvocationID.z);
                if (light >= light_count) {{
                    return;
                }}
                if (gid.x < cell_grid_size.x && gid.y < cell_grid_size.y) {{
                    imageStore(cell_dose_dst_img, ivec3(gid, light), texelFetch(cell_dose_src_tex, ivec3(gid, light), 0));
                }}
                if (gid.x < gas_grid_size.x && gid.y < gas_grid_size.y) {{
                    imageStore(gas_dose_dst_img, ivec3(gid, light), texelFetch(gas_dose_src_tex, ivec3(gid, light), 0));
                }}
            }}
            """
        )
        self.programs["copy_bridge_flow_velocity_to_reaction"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            layout(binding=0) uniform sampler2D bridge_flow_velocity_tex;
            layout(rg32f, binding=0) writeonly uniform image2D flow_velocity_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                imageStore(flow_velocity_img, gid, texelFetch(bridge_flow_velocity_tex, gid, 0));
            }}
            """
        )
        self.programs["publish_bridge_light_emitters"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform uint emitter_vec4_count;
            uniform uint counter_count;
            layout(std430, binding=0) readonly buffer SourceEmitterBuffer {{
                vec4 src_emitters[];
            }};
            layout(std430, binding=1) readonly buffer SourceCounterBuffer {{
                uint src_counts[];
            }};
            layout(std430, binding=2) writeonly buffer BridgeEmitterBuffer {{
                vec4 bridge_emitters[];
            }};
            layout(std430, binding=3) writeonly buffer BridgeCounterBuffer {{
                uint bridge_counts[];
            }};
            void main() {{
                uint index = gl_GlobalInvocationID.x;
                uint valid_emitter_vec4_count = min(emitter_vec4_count, min(src_counts[0], uint({MAX_EMITTED_LIGHTS})) * 2u);
                if (index < valid_emitter_vec4_count) {{
                    bridge_emitters[index] = src_emitters[index];
                }}
                if (index < counter_count) {{
                    bridge_counts[index] = src_counts[index];
                }}
            }}
            """
        )
        self.programs["timed_trigger"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D timer_in_tex;
            layout(binding=2) uniform sampler2D active_cell_tex;
            layout(std430, binding=0) buffer MaterialSlotsLo {{
                ivec4 material_slots_lo[{MAX_MATERIALS}];
            }};
            layout(rgba32f, binding=3) writeonly uniform image2D trigger_lo_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                vec4 timers = texelFetch(timer_in_tex, gid, 0);
                if (texelFetch(active_cell_tex, gid, 0).x <= 0.5) {{
                    imageStore(trigger_lo_img, gid, vec4(0.0));
                    imageStore(timer_out_img, gid, timers);
                    return;
                }}
                vec4 updated = timers;
                vec4 triggered = vec4(0.0);
                ivec4 slots = material_slots_lo[clamp(material_id, 0, {MAX_MATERIALS - 1})];
                for (int slot_index = 0; slot_index < 4; ++slot_index) {{
                    int timer_value = int(round(timers[slot_index]));
                    if (timer_value > 0) {{
                        if (slots[slot_index] > 0) {{
                            triggered[slot_index] = float(slots[slot_index]);
                        }}
                        updated[slot_index] = float(max(0, timer_value - 1));
                    }}
                }}
                imageStore(trigger_lo_img, gid, triggered);
                imageStore(timer_out_img, gid, updated);
            }}
            """
        )
        self.programs["self_trigger"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int self_rule_count;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D temp_tex;
            layout(binding=3) uniform sampler2D integrity_tex;
            layout(binding=4) uniform sampler2D timer_in_tex;
            layout(binding=5) uniform sampler2D active_cell_tex;
            layout(std430, binding=0) buffer MaterialSlotsLo {{
                ivec4 material_slots_lo[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer MaterialSlotsHi {{
                ivec4 material_slots_hi[{MAX_MATERIALS}];
            }};
            layout(std430, binding=2) buffer ActionMeta {{
                ivec4 action_meta[{MAX_ACTIONS}];
            }};
            layout(std430, binding=3) buffer SelfRuleI {{
                ivec4 self_rule_i[{MAX_SELF_RULES}];
            }};
            layout(std430, binding=4) buffer SelfRuleF {{
                vec4 self_rule_f[{MAX_SELF_RULES}];
            }};
            layout(rgba32f, binding=0) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=1) writeonly uniform image2D trigger_lo_img;
            layout(rgba32f, binding=2) writeonly uniform image2D trigger_hi_img;
            int slot_action(int material_id, int slot_index) {{
                if (slot_index < 4) {{
                    return material_slots_lo[material_id][slot_index];
                }}
                return material_slots_hi[material_id][slot_index - 4];
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                vec4 timers = texelFetch(timer_in_tex, gid, 0);
                vec4 trigger_lo = vec4(0.0);
                vec4 trigger_hi = vec4(0.0);
                if (texelFetch(active_cell_tex, gid, 0).x <= 0.5) {{
                    imageStore(timer_out_img, gid, timers);
                    imageStore(trigger_lo_img, gid, trigger_lo);
                    imageStore(trigger_hi_img, gid, trigger_hi);
                    return;
                }}
                if (material_id <= 0) {{
                    imageStore(timer_out_img, gid, timers);
                    imageStore(trigger_lo_img, gid, trigger_lo);
                    imageStore(trigger_hi_img, gid, trigger_hi);
                    return;
                }}
                int phase_value = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                for (int rule_index = 0; rule_index < self_rule_count; ++rule_index) {{
                    ivec4 ri = self_rule_i[rule_index];
                    vec4 rf = self_rule_f[rule_index];
                    if (ri.x != material_id) {{
                        continue;
                    }}
                    if (ri.z != 0 && ((1 << phase_value) & ri.z) == 0) {{
                        continue;
                    }}
                    if (temp_value < rf.x || temp_value > rf.y) {{
                        continue;
                    }}
                    if ((ri.w & 1) != 0 && integrity_value > rf.z) {{
                        continue;
                    }}
                    if ((ri.w & 2) != 0 && integrity_value < rf.w) {{
                        continue;
                    }}
                    int slot_index = ri.y;
                    int action_index = slot_action(material_id, slot_index);
                    if (action_index <= 0) {{
                        continue;
                    }}
                    if (slot_index < 4) {{
                        if (timers[slot_index] <= 0.5) {{
                            int duration = action_meta[action_index].x;
                            if (duration > 0) {{
                                timers[slot_index] = float(duration);
                            }}
                            trigger_lo[slot_index] = float(action_index);
                        }}
                    }} else {{
                        trigger_hi[slot_index - 4] = float(action_index);
                    }}
                }}
                imageStore(timer_out_img, gid, timers);
                imageStore(trigger_lo_img, gid, trigger_lo);
                imageStore(trigger_hi_img, gid, trigger_hi);
            }}
            """
        )
        local_action_rule_storage = f"""
            layout(std430, binding=3) buffer RuleI {{
                ivec4 rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=4) buffer RuleF {{
                vec4 rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=5) buffer RuleTags {{
                uvec4 rule_tags[{MAX_RULES}];
            }};
        """
        helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int rule_count;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform ivec2 gas_grid_size;
            uniform int random_target_count;
            uniform bool direct_gas_delta_enabled;
            uniform uint direct_modify_gas_layer_mask;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D temp_tex;
            layout(binding=3) uniform sampler2D integrity_tex;
            layout(binding=4) uniform sampler2DArray gas_tex;
            layout(binding=5) uniform sampler2DArray dose_tex;
            layout(binding=6) uniform sampler2D timer_tex;
            layout(binding=7) uniform sampler2D active_cell_tex;
            layout(binding=8) uniform sampler2D velocity_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            {local_action_rule_storage}
            layout(std430, binding=6) buffer MaterialTags {{
                uvec4 material_tags[{MAX_MATERIALS}];
            }};
            layout(std430, binding=7) buffer GasTags {{
                uvec4 gas_tags[{MAX_MATERIALS}];
            }};
            layout(std430, binding=8) buffer MaterialSlotsLo {{
                ivec4 material_slots_lo[{MAX_MATERIALS}];
            }};
            layout(std430, binding=9) buffer MaterialSlotsHi {{
                ivec4 material_slots_hi[{MAX_MATERIALS}];
            }};
            layout(std430, binding=10) buffer ActionMeta {{
                ivec4 action_meta[{MAX_ACTIONS}];
            }};
            layout(std430, binding=11) buffer RandomTargets {{
                int random_targets[{MAX_MATERIALS}];
            }};
            layout(std430, binding=14) buffer LightEmitterBuffer {{
                vec4 light_emitters[{MAX_EMITTED_LIGHTS * 2}];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            // DIRECT_GAS_DELTA_STORAGE_BEGIN
            layout(std430, binding=13) buffer DirectGasDelta {{
                int direct_gas_delta[];
            }};
            // DIRECT_GAS_DELTA_STORAGE_END
            const uint CONSUME_POLICY_LHS_ID = {CONSUME_POLICY_LHS}u;
            const uint CONSUME_POLICY_RHS_ID = {CONSUME_POLICY_RHS}u;
            const uint CONSUME_POLICY_BOTH_ID = {CONSUME_POLICY_BOTH}u;
            bool mask_matches(uint value, uint required) {{
                return required == 0u || (value & required) == required;
            }}
            int clamp_material_id(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}
            int slot_action(int material_id, int slot_index) {{
                if (slot_index < 0) {{
                    return -1;
                }}
                material_id = clamp_material_id(material_id);
                if (slot_index < 4) {{
                    return material_slots_lo[material_id][slot_index];
                }}
                if (slot_index < 8) {{
                    return material_slots_hi[material_id][slot_index - 4];
                }}
                return -1;
            }}
            bool match_material_selector(int material_id, int required_id, uint required_mask) {{
                if (material_id <= 0) {{
                    return false;
                }}
                if (required_id >= 0 && material_id != required_id) {{
                    return false;
                }}
                material_id = clamp_material_id(material_id);
                if (!mask_matches(material_tags[material_id].x, required_mask)) {{
                    return false;
                }}
                return required_id >= 0 || required_mask != 0u;
            }}
            bool solve_active(ivec2 cell) {{
                return texelFetch(active_cell_tex, cell, 0).x > 0.5;
            }}
            void append_deferred(
                inout vec4 deferred_action_lo,
                inout vec4 deferred_action_hi,
                inout vec4 deferred_scale_lo,
                inout vec4 deferred_scale_hi,
                inout int deferred_count,
                int action_index,
                float scale
            ) {{
                if (deferred_count >= 8) {{
                    return;
                }}
                if (deferred_count < 4) {{
                    deferred_action_lo[deferred_count] = float(action_index);
                    deferred_scale_lo[deferred_count] = scale;
                }} else {{
                    int hi_index = deferred_count - 4;
                    deferred_action_hi[hi_index] = float(action_index);
                    deferred_scale_hi[hi_index] = scale;
                }}
                deferred_count += 1;
            }}
            // DIRECT_GAS_DELTA_FUNCTIONS_BEGIN
            bool direct_gas_layer_enabled(int layer) {{
                if (layer < 0 || layer >= gas_count) {{
                    return false;
                }}
                if (layer >= 31) {{
                    return true;
                }}
                return (direct_modify_gas_layer_mask & (1u << uint(layer))) != 0u;
            }}
            int direct_gas_delta_index(ivec2 gas_cell, int layer) {{
                return (layer * gas_grid_size.y + gas_cell.y) * gas_grid_size.x + gas_cell.x;
            }}
            bool apply_direct_gas_delta(ivec2 gid, ivec4 ai, vec4 af, float scale, inout int deferred_count) {{
                if (!direct_gas_delta_enabled || ai.x != {TYPE_MODIFY_GAS} || !direct_gas_layer_enabled(ai.y)) {{
                    return false;
                }}
                ivec2 gas_cell = ivec2(
                    min(gas_grid_size.x - 1, max(0, gid.x / gas_cell_size)),
                    min(gas_grid_size.y - 1, max(0, gid.y / gas_cell_size))
                );
                float delta = af.x * scale;
                int fixed_delta = int(round(clamp(
                    delta * float({GAS_DELTA_FIXED_SCALE}),
                    -2147483000.0,
                    2147483000.0
                )));
                if (fixed_delta != 0) {{
                    atomicAdd(direct_gas_delta[direct_gas_delta_index(gas_cell, ai.y)], fixed_delta);
                }}
                atomicAdd(reaction_counts[1], 1u);
                atomicAdd(reaction_counts[1 + {TYPE_MODIFY_GAS}], 1u);
                return true;
            }}
            // DIRECT_GAS_DELTA_FUNCTIONS_END
            vec2 light_direction(ivec2 gid, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return vec2(1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return vec2(-1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return vec2(0.0, -1.0);
                }}
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return vec2(0.0, 1.0);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                    float norm = length(velocity);
                    if (norm > 1.0e-5) {{
                        return velocity / norm;
                    }}
                }}
                return vec2(0.0, 0.0);
            }}
            void append_light_emitter(ivec2 gid, ivec4 ai, vec4 af, float scale) {{
                if (ai.y < 0) {{
                    return;
                }}
                uint emitter_index = atomicAdd(reaction_counts[0], 1u);
                if (emitter_index >= uint({MAX_EMITTED_LIGHTS})) {{
                    return;
                }}
                vec2 direction = light_direction(gid, ai.z);
                uint base_index = emitter_index * 2u;
                light_emitters[base_index] = vec4(float(gid.x), float(gid.y), direction.x, direction.y);
                light_emitters[base_index + 1u] = vec4(
                    max(0.1, af.x * scale),
                    af.y,
                    max(0.0, af.z),
                    float(ai.y)
                );
            }}
            void record_gpu_local_action(int action_type) {{
                if (
                    action_type == {TYPE_HARM}
                    || action_type == {TYPE_MODIFY_TEMPERATURE}
                    || action_type == {TYPE_CONVERT_MATERIAL}
                    || action_type == {TYPE_EMIT_LIGHT}
                ) {{
                    atomicAdd(reaction_counts[1], 1u);
                    atomicAdd(reaction_counts[1 + action_type], 1u);
                }}
            }}
            void apply_action(
                inout float material_value,
                inout float phase_value,
                inout float temp_value,
                inout float integrity_value,
                inout vec4 timer_value,
                inout float cell_reset_value,
                inout float reaction_latched_value,
                inout vec4 deferred_action_lo,
                inout vec4 deferred_action_hi,
                inout vec4 deferred_scale_lo,
                inout vec4 deferred_scale_hi,
                inout int deferred_count,
                ivec2 gid,
                int action_index,
                float scale
            ) {{
                if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                    return;
                }}
                ivec4 ai = action_i[action_index];
                vec4 af = action_f[action_index];
                record_gpu_local_action(ai.x);
                reaction_latched_value = 1.0;
                if (ai.x == {TYPE_HARM}) {{
                    float next_integrity = integrity_value - af.y * scale;
                    if (af.y < 0.0) {{
                        float base_integrity = material_params[int(material_value + 0.5)].x;
                        next_integrity = min(base_integrity, next_integrity);
                    }}
                    if (next_integrity <= 0.0) {{
                        cell_reset_value = 1.0;
                        reaction_latched_value = 0.0;
                        material_value = 0.0;
                        phase_value = 0.0;
                        integrity_value = 0.0;
                        timer_value = vec4(0.0);
                    }} else {{
                        integrity_value = next_integrity;
                    }}
                }} else if (ai.x == {TYPE_MODIFY_TEMPERATURE}) {{
                    temp_value += af.x * scale;
                }} else if (ai.x == {TYPE_CONVERT_MATERIAL}) {{
                    float harm_scale = scale;
                    if ((ai.z & {ACTION_FLAG_ALLOW_SUBUNIT_SCALE}) == 0) {{
                        harm_scale = max(1.0, scale);
                    }}
                    integrity_value -= af.z * harm_scale;
                    if (integrity_value <= af.w) {{
                        int target_material = ai.y;
                        if ((ai.z & 1) != 0 && random_target_count > 0) {{
                            uint selector_bits = (uint(gid.x) * 73856093u) ^ (uint(gid.y) * 19349663u);
                            int selector = int(selector_bits % uint(random_target_count));
                            target_material = random_targets[selector];
                            int current_material = int(material_value + 0.5);
                            if (target_material == current_material || target_material <= 0) {{
                                for (int offset = 1; offset < random_target_count; ++offset) {{
                                    int alternative = random_targets[(selector + offset) % random_target_count];
                                    if (alternative > 0 && alternative != current_material) {{
                                        target_material = alternative;
                                        break;
                                    }}
                                }}
                                if (target_material == current_material) {{
                                    target_material = 0;
                                }}
                            }}
                        }}
                        cell_reset_value = 1.0;
                        reaction_latched_value = 0.0;
                        timer_value = vec4(0.0);
                        if (target_material <= 0) {{
                            material_value = 0.0;
                            phase_value = 0.0;
                            integrity_value = 0.0;
                        }} else {{
                            material_value = float(target_material);
                            phase_value = material_params[target_material].y;
                            integrity_value = material_params[target_material].x;
                            float spawn_temp = material_params[target_material].z;
                            if (spawn_temp == spawn_temp) {{
                                temp_value = max(temp_value, spawn_temp);
                            }}
                        }}
                    }}
                }} else if (ai.x == {TYPE_MODIFY_GAS}) {{
                    if (!apply_direct_gas_delta(gid, ai, af, scale, deferred_count)) {{
                        append_deferred(
                            deferred_action_lo,
                            deferred_action_hi,
                            deferred_scale_lo,
                            deferred_scale_hi,
                            deferred_count,
                            action_index,
                            scale
                        );
                    }} else if (ai.w != 0) {{
                        append_deferred(
                            deferred_action_lo,
                            deferred_action_hi,
                            deferred_scale_lo,
                            deferred_scale_hi,
                            deferred_count,
                            action_index,
                            scale
                        );
                    }}
                }} else if (ai.x == {TYPE_EMIT_LIGHT}) {{
                    append_light_emitter(gid, ai, af, scale);
                }} else if (ai.x == {TYPE_EMIT_MATERIAL}) {{
                    append_deferred(
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        deferred_count,
                        action_index,
                        scale
                    );
                }} else if (ai.x == {TYPE_DEFERRED}) {{
                    append_deferred(
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        deferred_count,
                        action_index,
                        scale
                    );
                }}
            }}
            void apply_slot_trigger(
                inout float material_value,
                inout float phase_value,
                inout float temp_value,
                inout float integrity_value,
                inout vec4 timer_value,
                inout float cell_reset_value,
                inout float reaction_latched_value,
                inout vec4 deferred_action_lo,
                inout vec4 deferred_action_hi,
                inout vec4 deferred_scale_lo,
                inout vec4 deferred_scale_hi,
                inout int deferred_count,
                ivec2 gid,
                int slot_index,
                float scale
            ) {{
                int material_id = int(material_value + 0.5);
                if (material_id <= 0) {{
                    return;
                }}
                int action_index = slot_action(material_id, slot_index);
                if (action_index <= 0) {{
                    return;
                }}
                if (slot_index < 4 && timer_value[slot_index] <= 0.5) {{
                    int duration = action_meta[action_index].x;
                    if (duration > 0) {{
                        timer_value[slot_index] = float(duration);
                    }}
                }}
                apply_action(
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    cell_reset_value,
                    reaction_latched_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    deferred_count,
                    gid,
                    action_index,
                    scale
                );
            }}
            void consume_current_material(
                inout float material_value,
                inout float phase_value,
                inout float integrity_value,
                inout vec4 timer_value,
                inout float cell_reset_value,
                inout float reaction_latched_value,
                float scale
            ) {{
                int material_id = int(material_value + 0.5);
                if (material_id <= 0 || scale <= 0.0) {{
                    return;
                }}
                integrity_value = max(0.0, integrity_value - scale);
                if (integrity_value <= 1e-6) {{
                    cell_reset_value = 1.0;
                    reaction_latched_value = 0.0;
                    material_value = 0.0;
                    phase_value = 0.0;
                    integrity_value = 0.0;
                    timer_value = vec4(0.0);
                }}
            }}
        """
        direct_storage_begin = "            // DIRECT_GAS_DELTA_STORAGE_BEGIN\n"
        direct_storage_end = "            // DIRECT_GAS_DELTA_STORAGE_END\n"
        direct_functions_begin = "            // DIRECT_GAS_DELTA_FUNCTIONS_BEGIN\n"
        direct_functions_end = "            // DIRECT_GAS_DELTA_FUNCTIONS_END\n"

        def without_marked_block(source: str, begin: str, end: str, replacement: str = "") -> str:
            start = source.index(begin)
            stop = source.index(end, start) + len(end)
            return source[:start] + replacement + source[stop:]

        no_direct_helper = without_marked_block(helper, direct_storage_begin, direct_storage_end)
        no_direct_helper = without_marked_block(
            no_direct_helper,
            direct_functions_begin,
            direct_functions_end,
            """
            bool apply_direct_gas_delta(ivec2 gid, ivec4 ai, vec4 af, float scale, inout int deferred_count) {
                return false;
            }
            """,
        )
        self_apply_helper = no_direct_helper.replace(local_action_rule_storage, "")
        lhs_candidate_helper = f"""
            const int RULE_CANDIDATE_WORDS = {RULE_CANDIDATE_WORDS};
            const int RULE_CANDIDATE_VECS = {RULE_CANDIDATE_VECS};
            uniform int rule_candidate_word_count;
            layout(std430, binding=12) buffer RuleLhsCandidateMasks {{
                uvec4 rule_lhs_candidate_masks[{MAX_MATERIALS * RULE_CANDIDATE_VECS}];
            }};
            uint lhs_rule_candidate_word(int material_id, int word_index) {{
                if (word_index < 0 || word_index >= RULE_CANDIDATE_WORDS) {{
                    return 0u;
                }}
                material_id = clamp_material_id(material_id);
                uvec4 packed_words = rule_lhs_candidate_masks[material_id * RULE_CANDIDATE_VECS + (word_index / 4)];
                return packed_words[word_index & 3];
            }}
        """
        local_action_output_layout = """
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2DArray deferred_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2DArray deferred_hi_img;
            layout(rg32f, binding=7) writeonly uniform image2D cell_meta_out_img;
            void store_cell_outputs(
                ivec2 gid,
                float material_value,
                float phase_value,
                float temp_value,
                float integrity_value,
                vec4 timer_value,
                vec4 deferred_action_lo,
                vec4 deferred_action_hi,
                vec4 deferred_scale_lo,
                vec4 deferred_scale_hi,
                float cell_reset_value,
                float reaction_latched_value
            ) {
                imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, timer_value);
                imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                imageStore(cell_meta_out_img, gid, vec4(cell_reset_value, reaction_latched_value, 0.0, 0.0));
            }
            void apply_trigger_vector(
                inout float material_value,
                inout float phase_value,
                inout float temp_value,
                inout float integrity_value,
                inout vec4 timer_value,
                inout float cell_reset_value,
                inout float reaction_latched_value,
                inout vec4 deferred_action_lo,
                inout vec4 deferred_action_hi,
                inout vec4 deferred_scale_lo,
                inout vec4 deferred_scale_hi,
                inout int deferred_count,
                ivec2 gid,
                vec4 triggers
            ) {
                for (int slot_index = 0; slot_index < 4; ++slot_index) {
                    int action_index = int(round(triggers[slot_index]));
                    if (action_index <= 0) {
                        continue;
                    }
                    apply_action(
                        material_value,
                        phase_value,
                        temp_value,
                        integrity_value,
                        timer_value,
                        cell_reset_value,
                        reaction_latched_value,
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        deferred_count,
                        gid,
                        action_index,
                        1.0
                    );
                }
            }
        """
        self.programs["timed_apply"] = ctx.compute_shader(
            helper
            + local_action_output_layout
            + """
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                if (!solve_active(gid)) {
                    store_cell_outputs(
                        gid,
                        material_value,
                        phase_value,
                        temp_value,
                        integrity_value,
                        timer_value,
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        cell_reset_value,
                        reaction_latched_value
                    );
                    return;
                }
                int material_id = clamp_material_id(int(material_value + 0.5));
                ivec4 slots = material_slots_lo[material_id];
                vec4 triggers = vec4(0.0);
                bool has_trigger = false;
                for (int slot_index = 0; slot_index < 4; ++slot_index) {
                    int timer_count = int(round(timer_value[slot_index]));
                    if (timer_count <= 0) {
                        continue;
                    }
                    int action_index = slots[slot_index];
                    if (action_index > 0) {
                        has_trigger = true;
                        triggers[slot_index] = float(action_index);
                    }
                    timer_value[slot_index] = float(max(0, timer_count - 1));
                }
                if (has_trigger) {
                    apply_trigger_vector(
                        material_value,
                        phase_value,
                        temp_value,
                        integrity_value,
                        timer_value,
                        cell_reset_value,
                        reaction_latched_value,
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        deferred_count,
                        gid,
                        triggers
                    );
                }
                store_cell_outputs(
                    gid,
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    cell_reset_value,
                    reaction_latched_value
                );
            }
            """
        )
        self.programs["clear_timed_candidate_worklist"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            layout(std430, binding=0) buffer TimedCandidateCount {
                uint timed_candidate_count[];
            };
            layout(std430, binding=1) buffer TimedCandidateDispatchArgs {
                uint timed_candidate_dispatch_args[];
            };
            layout(std430, binding=2) buffer TimedMaterialTargetDispatchArgs {
                uint timed_material_target_dispatch_args[];
            };

            void main() {
                uint generation = timed_candidate_count[1] + 1u;
                if (generation == 0u) {
                    generation = 1u;
                }
                timed_candidate_count[0] = 0u;
                timed_candidate_count[1] = generation;
                timed_candidate_count[2] = 0u;
                timed_candidate_count[3] = 0u;
                timed_candidate_dispatch_args[0] = 0u;
                timed_candidate_dispatch_args[1] = 1u;
                timed_candidate_dispatch_args[2] = 1u;
                timed_material_target_dispatch_args[0] = 0u;
                timed_material_target_dispatch_args[1] = 1u;
                timed_material_target_dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["build_light_dose_guarded_dispatch_args"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform uvec3 full_group_count;
            layout(std430, binding={LIGHT_DOSE_GUARD_DISPATCH_GUARD_BINDING}) readonly buffer LightDoseGuard {{
                uint light_dose_guard[];
            }};
            layout(std430, binding={LIGHT_DOSE_GUARD_DISPATCH_ARGS_BINDING}) buffer DispatchArgs {{
                uint dispatch_args[];
            }};

            void main() {{
                if (light_dose_guard[0] == 0u) {{
                    dispatch_args[0] = 0u;
                    dispatch_args[1] = 1u;
                    dispatch_args[2] = 1u;
                    return;
                }}
                dispatch_args[0] = full_group_count.x;
                dispatch_args[1] = max(full_group_count.y, 1u);
                dispatch_args[2] = max(full_group_count.z, 1u);
            }}
            """
        )
        self.programs["compact_timed_candidates"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D timer_tex;
            layout(binding=2) uniform sampler2D active_cell_tex;
            layout(std430, binding=0) readonly buffer MaterialSlotsLo {{
                ivec4 material_slots_lo[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer TimedCandidateCount {{
                uint timed_candidate_count[];
            }};
            layout(std430, binding=2) buffer TimedCandidateList {{
                uint timed_candidate_list[];
            }};
            layout(std430, binding=3) buffer TimedCandidateDispatchArgs {{
                uint timed_candidate_dispatch_args[];
            }};
            layout(std430, binding=4) buffer TimedCandidateMarks {{
                uint timed_candidate_marks[];
            }};

            bool has_counting_timer(ivec2 gid) {{
                int material_id = clamp(int(texelFetch(material_tex, gid, 0).x + 0.5), 0, {MAX_MATERIALS - 1});
                ivec4 slots = material_slots_lo[material_id];
                vec4 timers = texelFetch(timer_tex, gid, 0);
                bool has_countdown = false;
                bool has_action = false;
                for (int slot = 0; slot < 4; ++slot) {{
                    int timer_count = int(round(timers[slot]));
                    if (timer_count <= 0) {{
                        continue;
                    }}
                    has_countdown = true;
                    has_action = has_action || slots[slot] > 0;
                }}
                return has_countdown || has_action;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                if (texelFetch(active_cell_tex, gid, 0).x <= 0.5 || !has_counting_timer(gid)) {{
                    return;
                }}
                uint cell_index = uint(gid.y * cell_grid_size.x + gid.x);
                uint slot = atomicAdd(timed_candidate_count[0], 1u);
                timed_candidate_list[slot] = cell_index;
                timed_candidate_marks[cell_index] = timed_candidate_count[1];
                atomicMax(timed_candidate_dispatch_args[0], (slot + 64u) / 64u);
            }}
            """
        )
        self.programs["timed_apply_candidates"] = ctx.compute_shader(
            no_direct_helper
            + local_action_output_layout
            + """
            layout(std430, binding=12) readonly buffer TimedCandidateList {
                uint timed_candidate_list[];
            };
            layout(std430, binding=13) readonly buffer TimedCandidateCount {
                uint timed_candidate_count[];
            };

            uint linear_thread_index() {
                return uint(gl_GlobalInvocationID.y) * (uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x))
                    + uint(gl_GlobalInvocationID.x);
            }

            void main() {
                uint candidate_slot = linear_thread_index();
                if (candidate_slot >= timed_candidate_count[0]) {
                    return;
                }
                uint cell_index = timed_candidate_list[candidate_slot];
                ivec2 gid = ivec2(
                    int(cell_index % uint(cell_grid_size.x)),
                    int(cell_index / uint(cell_grid_size.x))
                );
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                int material_id = clamp_material_id(int(material_value + 0.5));
                ivec4 slots = material_slots_lo[material_id];
                vec4 triggers = vec4(0.0);
                for (int slot_index = 0; slot_index < 4; ++slot_index) {
                    int timer_count = int(round(timer_value[slot_index]));
                    if (timer_count <= 0) {
                        continue;
                    }
                    int action_index = slots[slot_index];
                    if (action_index > 0) {
                        triggers[slot_index] = float(action_index);
                    }
                    timer_value[slot_index] = float(max(0, timer_count - 1));
                }
                apply_trigger_vector(
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    cell_reset_value,
                    reaction_latched_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    deferred_count,
                    gid,
                    triggers
                );
                store_cell_outputs(
                    gid,
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    cell_reset_value,
                    reaction_latched_value
                );
            }
            """
        )
        self.programs["self_apply"] = ctx.compute_shader(
            self_apply_helper
            + local_action_output_layout
            + f"""
            uniform int self_rule_count;
            layout(std430, binding=12) buffer SelfRuleI {{
                ivec4 self_rule_i[{MAX_SELF_RULES}];
            }};
            layout(std430, binding=13) buffer SelfRuleF {{
                vec4 self_rule_f[{MAX_SELF_RULES}];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                if (!solve_active(gid)) {{
                    store_cell_outputs(
                        gid,
                        material_value,
                        phase_value,
                        temp_value,
                        integrity_value,
                        timer_value,
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        cell_reset_value,
                        reaction_latched_value
                    );
                    return;
                }}
                int material_id = int(material_value + 0.5);
                if (material_id <= 0) {{
                    store_cell_outputs(
                        gid,
                        material_value,
                        phase_value,
                        temp_value,
                        integrity_value,
                        timer_value,
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        cell_reset_value,
                        reaction_latched_value
                    );
                    return;
                }}
                int phase_id = int(phase_value + 0.5);
                material_id = clamp_material_id(material_id);
                vec4 trigger_lo = vec4(0.0);
                vec4 trigger_hi = vec4(0.0);
                for (int rule_index = 0; rule_index < self_rule_count; ++rule_index) {{
                    ivec4 ri = self_rule_i[rule_index];
                    vec4 rf = self_rule_f[rule_index];
                    if (ri.x != material_id) {{
                        continue;
                    }}
                    if (ri.z != 0 && ((1 << phase_id) & ri.z) == 0) {{
                        continue;
                    }}
                    if (temp_value < rf.x || temp_value > rf.y) {{
                        continue;
                    }}
                    if ((ri.w & 1) != 0 && integrity_value > rf.z) {{
                        continue;
                    }}
                    if ((ri.w & 2) != 0 && integrity_value < rf.w) {{
                        continue;
                    }}
                    int slot_index = ri.y;
                    int action_index = slot_action(material_id, slot_index);
                    if (action_index <= 0) {{
                        continue;
                    }}
                    if (slot_index < 4) {{
                        if (timer_value[slot_index] <= 0.5) {{
                            int duration = action_meta[action_index].x;
                            if (duration > 0) {{
                                timer_value[slot_index] = float(duration);
                            }}
                            trigger_lo[slot_index] = float(action_index);
                        }}
                    }} else if (slot_index < 8) {{
                        trigger_hi[slot_index - 4] = float(action_index);
                    }}
                }}
                apply_trigger_vector(
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    cell_reset_value,
                    reaction_latched_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    deferred_count,
                    gid,
                    trigger_lo
                );
                apply_trigger_vector(
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    cell_reset_value,
                    reaction_latched_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    deferred_count,
                    gid,
                    trigger_hi
                );
                store_cell_outputs(
                    gid,
                    material_value,
                    phase_value,
                    temp_value,
                    integrity_value,
                    timer_value,
                    deferred_action_lo,
                    deferred_action_hi,
                    deferred_scale_lo,
                    deferred_scale_hi,
                    cell_reset_value,
                    reaction_latched_value
                );
            }}
            """
        )
        self.programs["scatter_local_action_outputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_local_tex;
            layout(binding=1) uniform sampler2D phase_local_tex;
            layout(binding=2) uniform sampler2D temp_local_tex;
            layout(binding=3) uniform sampler2D integrity_local_tex;
            layout(binding=4) uniform sampler2D timer_local_tex;
            layout(binding=5) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=6) uniform sampler2DArray deferred_hi_local_tex;
            layout(binding=7) uniform sampler2D cell_meta_local_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D deferred_action_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2D deferred_action_hi_img;
            layout(rgba32f, binding=7) writeonly uniform image2D deferred_scale_lo_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(material_out_img, gid, texelFetch(material_local_tex, gid, 0));
                imageStore(phase_out_img, gid, texelFetch(phase_local_tex, gid, 0));
                imageStore(temp_out_img, gid, texelFetch(temp_local_tex, gid, 0));
                imageStore(integrity_out_img, gid, texelFetch(integrity_local_tex, gid, 0));
                imageStore(timer_out_img, gid, texelFetch(timer_local_tex, gid, 0));
                imageStore(deferred_action_lo_img, gid, texelFetch(deferred_lo_local_tex, ivec3(gid, 0), 0));
                imageStore(deferred_action_hi_img, gid, texelFetch(deferred_hi_local_tex, ivec3(gid, 0), 0));
                imageStore(deferred_scale_lo_img, gid, texelFetch(deferred_lo_local_tex, ivec3(gid, 1), 0));
            }}
            """
        )
        self.programs["scatter_local_action_deferred_meta_outputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=1) uniform sampler2DArray deferred_hi_local_tex;
            layout(binding=2) uniform sampler2D cell_meta_local_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(rgba32f, binding=0) writeonly uniform image2D deferred_action_lo_img;
            layout(rgba32f, binding=1) writeonly uniform image2D deferred_action_hi_img;
            layout(rgba32f, binding=2) writeonly uniform image2D deferred_scale_lo_img;
            layout(rgba32f, binding=3) writeonly uniform image2D deferred_scale_hi_img;
            layout(r32f, binding=4) writeonly uniform image2D cell_reset_img;
            layout(r32f, binding=5) writeonly uniform image2D reaction_latched_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec4 meta = texelFetch(cell_meta_local_tex, gid, 0);
                imageStore(deferred_action_lo_img, gid, texelFetch(deferred_lo_local_tex, ivec3(gid, 0), 0));
                imageStore(deferred_action_hi_img, gid, texelFetch(deferred_hi_local_tex, ivec3(gid, 0), 0));
                imageStore(deferred_scale_lo_img, gid, texelFetch(deferred_lo_local_tex, ivec3(gid, 1), 0));
                imageStore(deferred_scale_hi_img, gid, texelFetch(deferred_hi_local_tex, ivec3(gid, 1), 0));
                imageStore(cell_reset_img, gid, vec4(meta.x, 0.0, 0.0, 0.0));
                imageStore(reaction_latched_img, gid, vec4(meta.y, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, texelFetch(velocity_tex, gid, 0));
            }}
            """
        )
        self.programs["scatter_local_action_tail_outputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=5) uniform sampler2DArray deferred_hi_local_tex;
            layout(binding=7) uniform sampler2D cell_meta_local_tex;
            layout(rgba32f, binding=0) writeonly uniform image2D deferred_scale_hi_img;
            layout(r32f, binding=1) writeonly uniform image2D cell_reset_img;
            layout(r32f, binding=2) writeonly uniform image2D reaction_latched_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec4 meta = texelFetch(cell_meta_local_tex, gid, 0);
                imageStore(deferred_scale_hi_img, gid, texelFetch(deferred_hi_local_tex, ivec3(gid, 1), 0));
                imageStore(cell_reset_img, gid, vec4(meta.x, 0.0, 0.0, 0.0));
                imageStore(reaction_latched_img, gid, vec4(meta.y, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["scatter_local_emit_cell_outputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D emit_cell_lo_tex;
            layout(binding=1) uniform sampler2D emit_cell_hi_tex;
            layout(binding=2) uniform sampler2D emit_timer_tex;
            layout(binding=3) uniform sampler2D emit_meta_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=4) writeonly uniform image2D velocity_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D emitted_material_mask_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec4 meta = texelFetch(emit_meta_tex, gid, 0);
                if (meta.x <= 0.5) {{
                    return;
                }}
                vec4 lo = texelFetch(emit_cell_lo_tex, gid, 0);
                vec4 hi = texelFetch(emit_cell_hi_tex, gid, 0);
                imageStore(material_out_img, gid, vec4(lo.x, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(lo.y, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(lo.z, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(lo.w, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, vec4(hi.xy, 0.0, 0.0));
                imageStore(timer_out_img, gid, texelFetch(emit_timer_tex, gid, 0));
                imageStore(emitted_material_mask_img, gid, vec4(meta.x, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["clear_transient_cell_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(rgba32f, binding=0) writeonly uniform image2D trigger_lo_img;
            layout(rgba32f, binding=1) writeonly uniform image2D trigger_hi_img;
            layout(rgba32f, binding=2) writeonly uniform image2D deferred_scale_lo_img;
            layout(rgba32f, binding=3) writeonly uniform image2D deferred_scale_hi_img;
            layout(r32f, binding=4) writeonly uniform image2D cell_reset_img;
            layout(r32f, binding=5) writeonly uniform image2D reaction_latched_img;
            layout(r32f, binding=6) writeonly uniform image2D emitted_material_mask_img;
            layout(rgba32f, binding=7) writeonly uniform image2D local_emit_cell_lo_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec4 zero4 = vec4(0.0);
                imageStore(trigger_lo_img, gid, zero4);
                imageStore(trigger_hi_img, gid, zero4);
                imageStore(deferred_scale_lo_img, gid, zero4);
                imageStore(deferred_scale_hi_img, gid, zero4);
                imageStore(cell_reset_img, gid, zero4);
                imageStore(reaction_latched_img, gid, zero4);
                imageStore(emitted_material_mask_img, gid, zero4);
                imageStore(local_emit_cell_lo_img, gid, zero4);
            }}
            """
        )
        self.programs["clear_transient_aux_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int flow_source_layers;
            layout(rgba32f, binding=0) writeonly uniform image2D local_emit_cell_hi_img;
            layout(rgba32f, binding=1) writeonly uniform image2D local_timer_img;
            layout(rg32f, binding=2) writeonly uniform image2D local_cell_meta_img;
            layout(rgba32f, binding=3) writeonly uniform image2DArray flow_source_img;
            layout(std430, binding=0) buffer LightEmitterCount {{
                uint light_emitter_count[];
            }};

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x < cell_grid_size.x && gid.y < cell_grid_size.y) {{
                    vec4 zero4 = vec4(0.0);
                    imageStore(local_emit_cell_hi_img, gid, zero4);
                    imageStore(local_timer_img, gid, zero4);
                    imageStore(local_cell_meta_img, gid, zero4);
                }}
                if (gid.x < gas_grid_size.x && gid.y < gas_grid_size.y) {{
                    for (int layer = 0; layer < flow_source_layers; ++layer) {{
                        imageStore(flow_source_img, ivec3(gid, layer), vec4(0.0));
                    }}
                }}
                uint linear = uint(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.x);
                if (linear < 16u) {{
                    light_emitter_count[linear] = 0u;
                }}
            }}
            """
        )
        self.programs["clear_transient_light_counters"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=16, local_size_y=1, local_size_z=1) in;
            layout(std430, binding=0) buffer LightEmitterCount {
                uint light_emitter_count[];
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index < 16u) {
                    light_emitter_count[index] = 0u;
                }
            }
            """
        )
        self.programs["clear_timed_candidate_local_meta"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(rg32f, binding=0) writeonly uniform image2D local_cell_meta_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(local_cell_meta_img, gid, vec4(0.0));
            }}
            """
        )
        self.programs["clear_transient_emit_material_mask"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(r32f, binding=0) writeonly uniform image2D emitted_material_mask_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(emitted_material_mask_img, gid, vec4(0.0));
            }}
            """
        )
        self.programs["clear_transient_emit_material_buffers"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(rgba32f, binding=0) writeonly uniform image2D local_emit_cell_lo_img;
            layout(rgba32f, binding=1) writeonly uniform image2D local_emit_cell_hi_img;
            layout(rgba32f, binding=2) writeonly uniform image2D local_timer_img;
            layout(rg32f, binding=3) writeonly uniform image2D local_cell_meta_img;
            layout(r32f, binding=4) writeonly uniform image2D emitted_material_mask_img;
            layout(r32f, binding=5) writeonly uniform image2D cell_reset_img;
            layout(r32f, binding=6) writeonly uniform image2D reaction_latched_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec4 zero4 = vec4(0.0);
                imageStore(local_emit_cell_lo_img, gid, zero4);
                imageStore(local_emit_cell_hi_img, gid, zero4);
                imageStore(local_timer_img, gid, zero4);
                imageStore(local_cell_meta_img, gid, zero4);
                imageStore(emitted_material_mask_img, gid, zero4);
                imageStore(cell_reset_img, gid, zero4);
                imageStore(reaction_latched_img, gid, zero4);
            }}
            """
        )
        self.programs["clear_transient_flow_sources"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int flow_source_layers;
            layout(rgba32f, binding=0) writeonly uniform image2DArray flow_source_img;

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= flow_source_layers) {{
                    return;
                }}
                imageStore(flow_source_img, gid, vec4(0.0));
            }}
            """
        )
        self.programs["clear_segment_cell_transient_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(r32f, binding=0) writeonly uniform image2D segment_cell_reset_img;
            layout(r32f, binding=1) writeonly uniform image2D segment_reaction_latched_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(segment_cell_reset_img, gid, vec4(0.0));
                imageStore(segment_reaction_latched_img, gid, vec4(0.0));
            }}
            """
        )
        self.programs["accumulate_segment_cell_transient_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool use_local_cell_meta;
            layout(binding=0) uniform sampler2D cell_reset_tex;
            layout(binding=1) uniform sampler2D reaction_latched_tex;
            layout(binding=2) uniform sampler2D local_cell_meta_tex;
            layout(r32f, binding=0) uniform image2D segment_cell_reset_img;
            layout(r32f, binding=1) uniform image2D segment_reaction_latched_img;

            vec2 current_meta(ivec2 cell) {{
                if (use_local_cell_meta) {{
                    return texelFetch(local_cell_meta_tex, cell, 0).xy;
                }}
                return vec2(
                    texelFetch(cell_reset_tex, cell, 0).x,
                    texelFetch(reaction_latched_tex, cell, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec2 meta = current_meta(gid);
                float current_reset = meta.x;
                float current_latched = meta.y;
                float segment_reset = max(imageLoad(segment_cell_reset_img, gid).x, current_reset);
                float segment_latched = imageLoad(segment_reaction_latched_img, gid).x;
                if (current_reset > 0.5) {{
                    segment_latched = 0.0;
                }}
                segment_latched = max(segment_latched, current_latched);
                imageStore(segment_cell_reset_img, gid, vec4(segment_reset, 0.0, 0.0, 0.0));
                imageStore(segment_reaction_latched_img, gid, vec4(segment_latched, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["accumulate_timed_candidate_segment_cell_transient_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D local_cell_meta_tex;
            layout(std430, binding=0) readonly buffer TimedCandidateList {{
                uint timed_candidate_list[];
            }};
            layout(std430, binding=1) readonly buffer TimedCandidateCount {{
                uint timed_candidate_count[];
            }};
            layout(r32f, binding=0) uniform image2D segment_cell_reset_img;
            layout(r32f, binding=1) uniform image2D segment_reaction_latched_img;

            uint linear_thread_index() {{
                return uint(gl_GlobalInvocationID.y) * (uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x))
                    + uint(gl_GlobalInvocationID.x);
            }}

            void main() {{
                uint candidate_slot = linear_thread_index();
                if (candidate_slot >= timed_candidate_count[0]) {{
                    return;
                }}
                uint cell_index = timed_candidate_list[candidate_slot];
                ivec2 gid = ivec2(
                    int(cell_index % uint(cell_grid_size.x)),
                    int(cell_index / uint(cell_grid_size.x))
                );
                vec2 meta = texelFetch(local_cell_meta_tex, gid, 0).xy;
                float current_reset = meta.x;
                float current_latched = meta.y;
                float segment_reset = max(imageLoad(segment_cell_reset_img, gid).x, current_reset);
                float segment_latched = imageLoad(segment_reaction_latched_img, gid).x;
                if (current_reset > 0.5) {{
                    segment_latched = 0.0;
                }}
                segment_latched = max(segment_latched, current_latched);
                imageStore(segment_cell_reset_img, gid, vec4(segment_reset, 0.0, 0.0, 0.0));
                imageStore(segment_reaction_latched_img, gid, vec4(segment_latched, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["cell_material_side_effects"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool use_local_deferred_outputs;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(binding=2) uniform sampler2D action_lo_tex;
            layout(binding=3) uniform sampler2D action_hi_tex;
            layout(binding=4) uniform sampler2D scale_lo_tex;
            layout(binding=5) uniform sampler2D scale_hi_tex;
            layout(binding=6) uniform sampler2D temp_tex;
            layout(binding=7) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=8) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=4) writeonly uniform image2D velocity_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D emitted_material_mask_img;

            uint deterministic_selector(int x, int y, uint count) {{
                if (count == 0u) {{
                    return 0u;
                }}
                uint mixed = (uint(x) * 73856093u) ^ (uint(y) * 19349663u);
                return mixed % count;
            }}

            ivec2 target_for_direction(ivec2 source, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return source + ivec2(0, 1);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return source + ivec2(0, -1);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return source + ivec2(-1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return source + ivec2(1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["random"]}) {{
                    uint selector = deterministic_selector(source.x, source.y, 9u);
                    int dx = int(selector % 3u) - 1;
                    int dy = int(selector / 3u) - 1;
                    return source + ivec2(dx, dy);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                    return source + ivec2(int(sign(velocity.x)), int(sign(velocity.y)));
                }}
                return source;
            }}

            bool action_emits_to_target(ivec2 source, ivec2 target, int action_index) {{
                if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                    return false;
                }}
                ivec4 ai = action_i[action_index];
                if (ai.x != {TYPE_EMIT_MATERIAL} || ai.y <= 0) {{
                    return false;
                }}
                ivec2 emitted_target = target_for_direction(source, ai.z);
                return emitted_target == target;
            }}

            int first_emitting_action(ivec2 source, ivec2 target, vec4 actions) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_emits_to_target(source, target, action_index)) {{
                        return action_index;
                    }}
                }}
                return -1;
            }}

            vec4 fetch_action_lo(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_lo_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_lo_tex, cell, 0);
            }}

            vec4 fetch_action_hi(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_hi_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_hi_tex, cell, 0);
            }}

            vec2 emitted_velocity(ivec2 source, ivec2 target, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                int phase_id = int(material_params[material_id].y + 0.5);
                if (phase_id != {int(Phase.POWDER)}) {{
                    return vec2(0.0);
                }}
                vec2 explicit_velocity = af.xy;
                if (length(explicit_velocity) > 1.0e-5) {{
                    return explicit_velocity;
                }}
                vec2 direction = vec2(target - source);
                float norm = max(1.0e-5, length(direction));
                return direction / norm * max(0.0, af.z);
            }}

            void emit_material(ivec2 target, ivec2 source, int action_index) {{
                ivec4 ai = action_i[action_index];
                vec4 af = action_f[action_index];
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                if (material_id <= 0) {{
                    return;
                }}
                float target_phase = material_params[material_id].y;
                float target_integrity = material_params[material_id].x;
                float current_temp = texelFetch(temp_tex, target, 0).x;
                float spawn_temp = material_params[material_id].z;
                imageStore(material_out_img, target, vec4(float(material_id), 0.0, 0.0, 0.0));
                imageStore(phase_out_img, target, vec4(target_phase, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, target, vec4(target_integrity, 0.0, 0.0, 0.0));
                if (spawn_temp == spawn_temp) {{
                    imageStore(temp_out_img, target, vec4(max(current_temp, spawn_temp), 0.0, 0.0, 0.0));
                }}
                imageStore(velocity_out_img, target, vec4(emitted_velocity(source, target, ai, af), 0.0, 0.0));
                imageStore(timer_out_img, target, vec4(0.0));
                imageStore(emitted_material_mask_img, target, vec4(1.0, 0.0, 0.0, 0.0));
                atomicAdd(reaction_counts[1], 1u);
                atomicAdd(reaction_counts[1 + {TYPE_EMIT_MATERIAL}], 1u);
            }}

            void main() {{
                ivec2 target = ivec2(gl_GlobalInvocationID.xy);
                if (target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                    return;
                }}
                if (texelFetch(material_tex, target, 0).x > 0.5) {{
                    return;
                }}
                for (int sy = max(0, target.y - 1); sy <= min(cell_grid_size.y - 1, target.y + 1); ++sy) {{
                    for (int sx = max(0, target.x - 1); sx <= min(cell_grid_size.x - 1, target.x + 1); ++sx) {{
                        ivec2 source = ivec2(sx, sy);
                        int action_index = first_emitting_action(source, target, fetch_action_lo(source));
                        if (action_index < 0) {{
                            action_index = first_emitting_action(source, target, fetch_action_hi(source));
                        }}
                        if (action_index >= 0) {{
                            emit_material(target, source, action_index);
                            return;
                        }}
                    }}
                }}
            }}
            """
        )
        self.programs["compact_timed_material_targets"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D velocity_tex;
            layout(binding=1) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=2) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) readonly buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=1) readonly buffer TimedCandidateList {{
                uint timed_candidate_list[];
            }};
            layout(std430, binding=2) buffer TimedCandidateCount {{
                uint timed_candidate_count[];
            }};
            layout(std430, binding=3) buffer TimedMaterialTargetList {{
                uint timed_material_target_list[];
            }};
            layout(std430, binding=4) buffer TimedMaterialTargetDispatchArgs {{
                uint timed_material_target_dispatch_args[];
            }};
            layout(std430, binding=5) buffer TimedMaterialTargetMarks {{
                uint timed_material_target_marks[];
            }};

            uint linear_thread_index() {{
                return uint(gl_GlobalInvocationID.y) * (uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x))
                    + uint(gl_GlobalInvocationID.x);
            }}

            uint deterministic_selector(int x, int y, uint count) {{
                if (count == 0u) {{
                    return 0u;
                }}
                uint mixed = (uint(x) * 73856093u) ^ (uint(y) * 19349663u);
                return mixed % count;
            }}

            ivec2 target_for_direction(ivec2 source, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return source + ivec2(0, 1);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return source + ivec2(0, -1);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return source + ivec2(-1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return source + ivec2(1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["random"]}) {{
                    uint selector = deterministic_selector(source.x, source.y, 9u);
                    int dx = int(selector % 3u) - 1;
                    int dy = int(selector / 3u) - 1;
                    return source + ivec2(dx, dy);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                    return source + ivec2(int(sign(velocity.x)), int(sign(velocity.y)));
                }}
                return source;
            }}

            void append_target(ivec2 target) {{
                if (target.x < 0 || target.y < 0 || target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                    return;
                }}
                uint target_index = uint(target.y * cell_grid_size.x + target.x);
                uint generation = timed_candidate_count[1];
                uint previous = atomicExchange(timed_material_target_marks[target_index], generation);
                if (previous == generation) {{
                    return;
                }}
                uint slot = atomicAdd(timed_candidate_count[2], 1u);
                timed_material_target_list[slot] = target_index;
                atomicMax(timed_material_target_dispatch_args[0], (slot + 64u) / 64u);
            }}

            void append_targets_for_actions(ivec2 source, vec4 actions) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                        continue;
                    }}
                    ivec4 ai = action_i[action_index];
                    if (ai.x != {TYPE_EMIT_MATERIAL} || ai.y <= 0) {{
                        continue;
                    }}
                    append_target(target_for_direction(source, ai.z));
                }}
            }}

            void main() {{
                uint candidate_slot = linear_thread_index();
                if (candidate_slot >= timed_candidate_count[0]) {{
                    return;
                }}
                uint cell_index = timed_candidate_list[candidate_slot];
                ivec2 source = ivec2(
                    int(cell_index % uint(cell_grid_size.x)),
                    int(cell_index / uint(cell_grid_size.x))
                );
                append_targets_for_actions(source, texelFetch(deferred_lo_local_tex, ivec3(source, 0), 0));
                append_targets_for_actions(source, texelFetch(deferred_hi_local_tex, ivec3(source, 0), 0));
            }}
            """
        )
        self.programs["cell_material_side_effects_candidates"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(binding=2) uniform sampler2D temp_tex;
            layout(binding=3) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=4) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=3) readonly buffer TimedCandidateCount {{
                uint timed_candidate_count[];
            }};
            layout(std430, binding=4) readonly buffer TimedCandidateMarks {{
                uint timed_candidate_marks[];
            }};
            layout(std430, binding=5) readonly buffer TimedMaterialTargetList {{
                uint timed_material_target_list[];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=4) writeonly uniform image2D velocity_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D emitted_material_mask_img;

            uint linear_thread_index() {{
                return uint(gl_GlobalInvocationID.y) * (uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x))
                    + uint(gl_GlobalInvocationID.x);
            }}

            uint deterministic_selector(int x, int y, uint count) {{
                if (count == 0u) {{
                    return 0u;
                }}
                uint mixed = (uint(x) * 73856093u) ^ (uint(y) * 19349663u);
                return mixed % count;
            }}

            bool source_is_timed_candidate(ivec2 source) {{
                if (source.x < 0 || source.y < 0 || source.x >= cell_grid_size.x || source.y >= cell_grid_size.y) {{
                    return false;
                }}
                uint source_index = uint(source.y * cell_grid_size.x + source.x);
                return timed_candidate_marks[source_index] == timed_candidate_count[1];
            }}

            ivec2 target_for_direction(ivec2 source, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return source + ivec2(0, 1);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return source + ivec2(0, -1);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return source + ivec2(-1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return source + ivec2(1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["random"]}) {{
                    uint selector = deterministic_selector(source.x, source.y, 9u);
                    int dx = int(selector % 3u) - 1;
                    int dy = int(selector / 3u) - 1;
                    return source + ivec2(dx, dy);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                    return source + ivec2(int(sign(velocity.x)), int(sign(velocity.y)));
                }}
                return source;
            }}

            bool action_emits_to_target(ivec2 source, ivec2 target, int action_index) {{
                if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                    return false;
                }}
                ivec4 ai = action_i[action_index];
                if (ai.x != {TYPE_EMIT_MATERIAL} || ai.y <= 0) {{
                    return false;
                }}
                return target_for_direction(source, ai.z) == target;
            }}

            int first_emitting_action(ivec2 source, ivec2 target, vec4 actions) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_emits_to_target(source, target, action_index)) {{
                        return action_index;
                    }}
                }}
                return -1;
            }}

            vec2 emitted_velocity(ivec2 source, ivec2 target, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                int phase_id = int(material_params[material_id].y + 0.5);
                if (phase_id != {int(Phase.POWDER)}) {{
                    return vec2(0.0);
                }}
                vec2 explicit_velocity = af.xy;
                if (length(explicit_velocity) > 1.0e-5) {{
                    return explicit_velocity;
                }}
                vec2 direction = vec2(target - source);
                float norm = max(1.0e-5, length(direction));
                return direction / norm * max(0.0, af.z);
            }}

            void emit_material(ivec2 target, ivec2 source, int action_index) {{
                ivec4 ai = action_i[action_index];
                vec4 af = action_f[action_index];
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                if (material_id <= 0) {{
                    return;
                }}
                float target_phase = material_params[material_id].y;
                float target_integrity = material_params[material_id].x;
                float current_temp = texelFetch(temp_tex, target, 0).x;
                float spawn_temp = material_params[material_id].z;
                imageStore(material_out_img, target, vec4(float(material_id), 0.0, 0.0, 0.0));
                imageStore(phase_out_img, target, vec4(target_phase, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, target, vec4(target_integrity, 0.0, 0.0, 0.0));
                if (spawn_temp == spawn_temp) {{
                    imageStore(temp_out_img, target, vec4(max(current_temp, spawn_temp), 0.0, 0.0, 0.0));
                }}
                imageStore(velocity_out_img, target, vec4(emitted_velocity(source, target, ai, af), 0.0, 0.0));
                imageStore(timer_out_img, target, vec4(0.0));
                imageStore(emitted_material_mask_img, target, vec4(1.0, 0.0, 0.0, 0.0));
                atomicAdd(reaction_counts[1], 1u);
                atomicAdd(reaction_counts[1 + {TYPE_EMIT_MATERIAL}], 1u);
            }}

            void main() {{
                uint target_slot = linear_thread_index();
                if (target_slot >= timed_candidate_count[2]) {{
                    return;
                }}
                uint target_index = timed_material_target_list[target_slot];
                ivec2 target = ivec2(
                    int(target_index % uint(cell_grid_size.x)),
                    int(target_index / uint(cell_grid_size.x))
                );
                if (texelFetch(material_tex, target, 0).x > 0.5) {{
                    return;
                }}
                for (int sy = max(0, target.y - 1); sy <= min(cell_grid_size.y - 1, target.y + 1); ++sy) {{
                    for (int sx = max(0, target.x - 1); sx <= min(cell_grid_size.x - 1, target.x + 1); ++sx) {{
                        ivec2 source = ivec2(sx, sy);
                        if (!source_is_timed_candidate(source)) {{
                            continue;
                        }}
                        int action_index = first_emitting_action(
                            source,
                            target,
                            texelFetch(deferred_lo_local_tex, ivec3(source, 0), 0)
                        );
                        if (action_index < 0) {{
                            action_index = first_emitting_action(
                                source,
                                target,
                                texelFetch(deferred_hi_local_tex, ivec3(source, 0), 0)
                            );
                        }}
                        if (action_index >= 0) {{
                            emit_material(target, source, action_index);
                            return;
                        }}
                    }}
                }}
            }}
            """
        )
        self.programs["clear_cell_gas_delta"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int delta_count;
            layout(std430, binding=0) buffer GasDelta {{
                int gas_delta[];
            }};

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= delta_count) {{
                    return;
                }}
                gas_delta[index] = 0;
            }}
            """
        )
        self.programs["scatter_cell_gas_action_delta"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform uint modify_gas_layer_mask;
            uniform bool use_local_deferred_outputs;
            uniform bool gas_delta_already_applied;
            layout(binding=0) uniform sampler2D action_lo_tex;
            layout(binding=1) uniform sampler2D action_hi_tex;
            layout(binding=2) uniform sampler2D scale_lo_tex;
            layout(binding=3) uniform sampler2D scale_hi_tex;
            layout(binding=4) uniform sampler2D active_cell_tex;
            layout(binding=5) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=6) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=1) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer GasDelta {{
                int gas_delta[];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(rgba32f, binding=0) writeonly uniform image2DArray flow_source_img;

            bool action_layer_enabled(int layer) {{
                if (layer < 0 || layer >= gas_count) {{
                    return false;
                }}
                if (layer >= 31) {{
                    return true;
                }}
                return (modify_gas_layer_mask & (1u << uint(layer))) != 0u;
            }}

            int gas_delta_index(ivec2 gas_cell, int layer) {{
                return (layer * gas_grid_size.y + gas_cell.y) * gas_grid_size.x + gas_cell.x;
            }}

            void write_flow_sources(ivec2 gas_cell, ivec4 ai, vec4 af, float scale, int layer_base) {{
                if (ai.w == 0 || layer_base < 0 || layer_base >= {FLOW_SOURCE_LAYERS}) {{
                    return;
                }}
                float strength = af.y * max(scale, 0.0);
                float radius = af.z;
                if (strength <= 0.0 || radius <= 0.0) {{
                    return;
                }}
                int direction_id = ai.z;
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["left"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(-1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["up"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(0.0, -1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["down"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(0.0, 1.0, radius, strength));
                }} else if (
                    direction_id == {DIRECTION_IDS["all"]}
                    && abs(af.w) > 1.0e-5
                    && layer_base + 3 < {FLOW_SOURCE_LAYERS}
                ) {{
                    float flow_sign = af.w > 0.0 ? 1.0 : -1.0;
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(-flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 1), vec4(flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 2), vec4(0.0, -flow_sign, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 3), vec4(0.0, flow_sign, radius, strength));
                }}
            }}

            void scatter_action_vector(ivec2 gas_cell, vec4 actions, vec4 scales, int slot_offset) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                        continue;
                    }}
                    ivec4 ai = action_i[action_index];
                    if (ai.x != {TYPE_MODIFY_GAS} || !action_layer_enabled(ai.y)) {{
                        continue;
                    }}
                    vec4 af = action_f[action_index];
                    if (!gas_delta_already_applied) {{
                        float delta = af.x * scales[slot];
                        int fixed_delta = int(round(clamp(
                            delta * float({GAS_DELTA_FIXED_SCALE}),
                            -2147483000.0,
                            2147483000.0
                        )));
                        if (fixed_delta != 0) {{
                            atomicAdd(gas_delta[gas_delta_index(gas_cell, ai.y)], fixed_delta);
                        }}
                        atomicAdd(reaction_counts[1], 1u);
                        atomicAdd(reaction_counts[1 + {TYPE_MODIFY_GAS}], 1u);
                    }}
                    write_flow_sources(gas_cell, ai, af, scales[slot], (slot_offset + slot) * 4);
                }}
            }}

            vec4 fetch_action_lo(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_lo_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_lo_tex, cell, 0);
            }}

            vec4 fetch_action_hi(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_hi_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_hi_tex, cell, 0);
            }}

            vec4 fetch_scale_lo(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_lo_local_tex, ivec3(cell, 1), 0);
                }}
                return texelFetch(scale_lo_tex, cell, 0);
            }}

            vec4 fetch_scale_hi(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_hi_local_tex, ivec3(cell, 1), 0);
                }}
                return texelFetch(scale_hi_tex, cell, 0);
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                if (texelFetch(active_cell_tex, cell, 0).x <= 0.5) {{
                    return;
                }}
                ivec2 gas_cell = ivec2(
                    min(gas_grid_size.x - 1, max(0, cell.x / gas_cell_size)),
                    min(gas_grid_size.y - 1, max(0, cell.y / gas_cell_size))
                );
                scatter_action_vector(
                    gas_cell,
                    fetch_action_lo(cell),
                    fetch_scale_lo(cell),
                    0
                );
                scatter_action_vector(
                    gas_cell,
                    fetch_action_hi(cell),
                    fetch_scale_hi(cell),
                    4
                );
            }}
            """
        )
        self.programs["scatter_cell_gas_action_delta_candidates"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform uint modify_gas_layer_mask;
            layout(binding=0) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=1) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=1) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer GasDelta {{
                int gas_delta[];
            }};
            layout(std430, binding=3) readonly buffer TimedCandidateList {{
                uint timed_candidate_list[];
            }};
            layout(std430, binding=4) readonly buffer TimedCandidateCount {{
                uint timed_candidate_count[];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(rgba32f, binding=0) writeonly uniform image2DArray flow_source_img;

            uint linear_thread_index() {{
                return uint(gl_GlobalInvocationID.y) * (uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x))
                    + uint(gl_GlobalInvocationID.x);
            }}

            bool action_layer_enabled(int layer) {{
                if (layer < 0 || layer >= gas_count) {{
                    return false;
                }}
                if (layer >= 31) {{
                    return true;
                }}
                return (modify_gas_layer_mask & (1u << uint(layer))) != 0u;
            }}

            int gas_delta_index(ivec2 gas_cell, int layer) {{
                return (layer * gas_grid_size.y + gas_cell.y) * gas_grid_size.x + gas_cell.x;
            }}

            void write_flow_sources(ivec2 gas_cell, ivec4 ai, vec4 af, float scale, int layer_base) {{
                if (ai.w == 0 || layer_base < 0 || layer_base >= {FLOW_SOURCE_LAYERS}) {{
                    return;
                }}
                float strength = af.y * max(scale, 0.0);
                float radius = af.z;
                if (strength <= 0.0 || radius <= 0.0) {{
                    return;
                }}
                int direction_id = ai.z;
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["left"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(-1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["up"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(0.0, -1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["down"]}) {{
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(0.0, 1.0, radius, strength));
                }} else if (
                    direction_id == {DIRECTION_IDS["all"]}
                    && abs(af.w) > 1.0e-5
                    && layer_base + 3 < {FLOW_SOURCE_LAYERS}
                ) {{
                    float flow_sign = af.w > 0.0 ? 1.0 : -1.0;
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base), vec4(-flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 1), vec4(flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 2), vec4(0.0, -flow_sign, radius, strength));
                    imageStore(flow_source_img, ivec3(gas_cell, layer_base + 3), vec4(0.0, flow_sign, radius, strength));
                }}
            }}

            void scatter_action_vector(ivec2 gas_cell, vec4 actions, vec4 scales, int slot_offset) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                        continue;
                    }}
                    ivec4 ai = action_i[action_index];
                    if (ai.x != {TYPE_MODIFY_GAS} || !action_layer_enabled(ai.y)) {{
                        continue;
                    }}
                    vec4 af = action_f[action_index];
                    float delta = af.x * scales[slot];
                    int fixed_delta = int(round(clamp(
                        delta * float({GAS_DELTA_FIXED_SCALE}),
                        -2147483000.0,
                        2147483000.0
                    )));
                    if (fixed_delta != 0) {{
                        atomicAdd(gas_delta[gas_delta_index(gas_cell, ai.y)], fixed_delta);
                    }}
                    write_flow_sources(gas_cell, ai, af, scales[slot], (slot_offset + slot) * 4);
                    atomicAdd(reaction_counts[1], 1u);
                    atomicAdd(reaction_counts[1 + {TYPE_MODIFY_GAS}], 1u);
                }}
            }}

            void main() {{
                uint candidate_slot = linear_thread_index();
                if (candidate_slot >= timed_candidate_count[0]) {{
                    return;
                }}
                uint cell_index = timed_candidate_list[candidate_slot];
                ivec2 cell = ivec2(
                    int(cell_index % uint(cell_grid_size.x)),
                    int(cell_index / uint(cell_grid_size.x))
                );
                ivec2 gas_cell = ivec2(
                    min(gas_grid_size.x - 1, max(0, cell.x / gas_cell_size)),
                    min(gas_grid_size.y - 1, max(0, cell.y / gas_cell_size))
                );
                scatter_action_vector(
                    gas_cell,
                    texelFetch(deferred_lo_local_tex, ivec3(cell, 0), 0),
                    texelFetch(deferred_lo_local_tex, ivec3(cell, 1), 0),
                    0
                );
                scatter_action_vector(
                    gas_cell,
                    texelFetch(deferred_hi_local_tex, ivec3(cell, 0), 0),
                    texelFetch(deferred_hi_local_tex, ivec3(cell, 1), 0),
                    4
                );
            }}
            """
        )
        self.programs["apply_cell_gas_delta"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int gas_count;
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(std430, binding=0) readonly buffer GasDelta {{
                int gas_delta[];
            }};
            layout(r32f, binding=0) writeonly uniform image2DArray gas_out_img;

            int gas_delta_index(ivec3 gas_cell) {{
                return (gas_cell.z * gas_grid_size.y + gas_cell.y) * gas_grid_size.x + gas_cell.x;
            }}

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= gas_count) {{
                    return;
                }}
                float gas_value = texelFetch(gas_tex, gid, 0).x;
                float delta = float(gas_delta[gas_delta_index(gid)]) / float({GAS_DELTA_FIXED_SCALE});
                imageStore(gas_out_img, gid, vec4(max(0.0, gas_value + delta), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["cell_gas_side_effects"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform int apply_action_side_effects;
            uniform int material_gas_rule_count;
            uniform uint modify_gas_layer_mask;
            uniform bool use_local_deferred_outputs;
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(binding=1) uniform sampler2D action_lo_tex;
            layout(binding=2) uniform sampler2D action_hi_tex;
            layout(binding=3) uniform sampler2D scale_lo_tex;
            layout(binding=4) uniform sampler2D scale_hi_tex;
            layout(binding=5) uniform sampler2D material_tex;
            layout(binding=6) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D temp_tex;
            layout(binding=8) uniform sampler2D active_cell_tex;
            layout(binding=9) uniform sampler2DArray deferred_lo_local_tex;
            layout(binding=10) uniform sampler2DArray deferred_hi_local_tex;
            layout(std430, binding=0) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=1) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer MaterialGasRuleI {{
                ivec4 material_gas_rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=3) buffer MaterialGasRuleF {{
                vec4 material_gas_rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=4) buffer MaterialGasRuleTags {{
                uvec4 material_gas_rule_tags[{MAX_RULES}];
            }};
            layout(std430, binding=5) buffer MaterialTags {{
                uvec4 material_tags[{MAX_MATERIALS}];
            }};
            layout(std430, binding=6) buffer GasTags {{
                uvec4 gas_tags[{MAX_MATERIALS}];
            }};
            int clamp_material_id(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(r32f, binding=0) writeonly uniform image2DArray gas_out_img;
            layout(rgba32f, binding=1) writeonly uniform image2DArray flow_source_img;

            void write_flow_sources(ivec2 cell, ivec4 ai, vec4 af, float scale, int layer_base) {{
                if (ai.w == 0) {{
                    return;
                }}
                float strength = af.y * max(scale, 0.0);
                float radius = af.z;
                if (strength <= 0.0 || radius <= 0.0) {{
                    return;
                }}
                int direction_id = ai.z;
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    imageStore(flow_source_img, ivec3(cell, layer_base), vec4(1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["left"]}) {{
                    imageStore(flow_source_img, ivec3(cell, layer_base), vec4(-1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["up"]}) {{
                    imageStore(flow_source_img, ivec3(cell, layer_base), vec4(0.0, -1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["down"]}) {{
                    imageStore(flow_source_img, ivec3(cell, layer_base), vec4(0.0, 1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["all"]} && abs(af.w) > 1.0e-5) {{
                    float flow_sign = af.w > 0.0 ? 1.0 : -1.0;
                    imageStore(flow_source_img, ivec3(cell, layer_base), vec4(-flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, layer_base + 1), vec4(flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, layer_base + 2), vec4(0.0, -flow_sign, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, layer_base + 3), vec4(0.0, flow_sign, radius, strength));
                }}
            }}

            bool mask_matches(uint value, uint required) {{
                return required == 0u || (value & required) == required;
            }}

            int best_matching_material_reaction_gas_layer(ivec2 gas_cell, int required_id, uint required_mask) {{
                if (required_id >= 0) {{
                    if (!mask_matches(gas_tags[required_id].x, required_mask)) {{
                        return -1;
                    }}
                    return required_id;
                }}
                if (required_mask == 0u) {{
                    return -1;
                }}
                float best = -1.0;
                int best_species = -1;
                for (int species_id = 0; species_id < gas_count; ++species_id) {{
                    if (!mask_matches(gas_tags[species_id].x, required_mask)) {{
                        continue;
                    }}
                    float candidate = texelFetch(gas_tex, ivec3(gas_cell, species_id), 0).x;
                    if (candidate > best) {{
                        best = candidate;
                        best_species = species_id;
                    }}
                }}
                return best_species;
            }}

            bool material_cell_matches_rule(ivec2 cell, ivec4 ri, vec4 rf, uvec4 rt) {{
                if (texelFetch(active_cell_tex, cell, 0).x <= 0.5) {{
                    return false;
                }}
                int material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                if (material_id <= 0) {{
                    return false;
                }}
                if (ri.x >= 0 && material_id != ri.x) {{
                    return false;
                }}
                int clamped_material = clamp(material_id, 0, {MAX_MATERIALS - 1});
                if (!mask_matches(material_tags[clamped_material].y, rt.x)) {{
                    return false;
                }}
                int phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                int phase_mask = int(rt.z);
                if (phase_mask != 0 && ((1 << phase_id) & phase_mask) == 0) {{
                    return false;
                }}
                float temp_value = texelFetch(temp_tex, cell, 0).x;
                return temp_value >= rf.x && temp_value <= rf.y;
            }}

            void apply_material_gas_rhs_consume(ivec2 gas_cell, int layer, ivec2 cell, inout float gas_value) {{
                for (int rule_index = 0; rule_index < material_gas_rule_count; ++rule_index) {{
                    ivec4 ri = material_gas_rule_i[rule_index];
                    vec4 rf = material_gas_rule_f[rule_index];
                    uvec4 rt = material_gas_rule_tags[rule_index];
                    int consume_policy = int(rt.w);
                    if (consume_policy != {CONSUME_POLICY_RHS} && consume_policy != {CONSUME_POLICY_BOTH}) {{
                        continue;
                    }}
                    if (!material_cell_matches_rule(cell, ri, rf, rt)) {{
                        continue;
                    }}
                    int species_id = best_matching_material_reaction_gas_layer(gas_cell, ri.y, rt.y);
                    if (species_id != layer) {{
                        continue;
                    }}
                    float concentration = texelFetch(gas_tex, ivec3(gas_cell, species_id), 0).x;
                    if (concentration < rf.z) {{
                        continue;
                    }}
                    gas_value = max(0.0, gas_value - concentration * max(rf.w, 0.0));
                }}
            }}

            void apply_action_vector(ivec2 gas_cell, int layer, vec4 actions, vec4 scales, int slot_offset, inout float gas_value) {{
                for (int slot = 0; slot < 4; ++slot) {{
                    int action_index = int(round(actions[slot]));
                    if (action_index <= 0 || action_index >= {MAX_ACTIONS}) {{
                        continue;
                    }}
                    ivec4 ai = action_i[action_index];
                    if (ai.x != {TYPE_MODIFY_GAS} || ai.y != layer) {{
                        continue;
                    }}
                    vec4 af = action_f[action_index];
                    float scale = scales[slot];
                    gas_value = max(0.0, gas_value + af.x * scale);
                    write_flow_sources(gas_cell, ai, af, scale, (slot_offset + slot) * 4);
                    atomicAdd(reaction_counts[1], 1u);
                    atomicAdd(reaction_counts[1 + {TYPE_MODIFY_GAS}], 1u);
                }}
            }}

            vec4 fetch_action_lo(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_lo_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_lo_tex, cell, 0);
            }}

            vec4 fetch_action_hi(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_hi_local_tex, ivec3(cell, 0), 0);
                }}
                return texelFetch(action_hi_tex, cell, 0);
            }}

            vec4 fetch_scale_lo(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_lo_local_tex, ivec3(cell, 1), 0);
                }}
                return texelFetch(scale_lo_tex, cell, 0);
            }}

            vec4 fetch_scale_hi(ivec2 cell) {{
                if (use_local_deferred_outputs) {{
                    return texelFetch(deferred_hi_local_tex, ivec3(cell, 1), 0);
                }}
                return texelFetch(scale_hi_tex, cell, 0);
            }}

            bool action_layer_enabled(int layer) {{
                if (layer < 0) {{
                    return false;
                }}
                if (layer >= 31) {{
                    return true;
                }}
                return (modify_gas_layer_mask & (1u << uint(layer))) != 0u;
            }}

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= gas_count) {{
                    return;
                }}
                float gas_value = texelFetch(gas_tex, gid, 0).x;
                bool apply_actions_for_layer = apply_action_side_effects != 0 && action_layer_enabled(gid.z);
                if (!apply_actions_for_layer && material_gas_rule_count <= 0) {{
                    imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
                    return;
                }}
                int x0 = gid.x * gas_cell_size;
                int y0 = gid.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                for (int y = y0; y < y1; ++y) {{
                    for (int x = x0; x < x1; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (apply_actions_for_layer) {{
                            apply_action_vector(
                                gid.xy,
                                gid.z,
                                fetch_action_lo(cell),
                                fetch_scale_lo(cell),
                                0,
                                gas_value
                            );
                            apply_action_vector(
                                gid.xy,
                                gid.z,
                                fetch_action_hi(cell),
                                fetch_scale_hi(cell),
                                4,
                                gas_value
                            );
                        }}
                        if (material_gas_rule_count > 0) {{
                            apply_material_gas_rhs_consume(gid.xy, gid.z, cell, gas_value);
                        }}
                    }}
                }}
                imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["material_light_cell_dose_consume"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int light_count;
            uniform int rule_count;
            layout(binding=0) uniform sampler2DArray cell_dose_tex;
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D temp_tex;
            layout(binding=4) uniform sampler2D active_cell_tex;
            layout(std430, binding=0) buffer MaterialLightRuleI {{
                ivec4 material_light_rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=1) buffer MaterialLightRuleF {{
                vec4 material_light_rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=2) buffer MaterialLightRuleTags {{
                uvec4 material_light_rule_tags[{MAX_RULES}];
            }};
            layout(std430, binding=3) buffer MaterialTags {{
                uvec4 material_tags[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=5) writeonly uniform image2DArray cell_dose_out_img;

            bool mask_matches(uint value, uint required) {{
                return required == 0u || (value & required) == required;
            }}

            bool material_cell_matches_rule(ivec2 cell, ivec4 ri, vec4 rf, uvec4 rt) {{
                if (texelFetch(active_cell_tex, cell, 0).x <= 0.5) {{
                    return false;
                }}
                int material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                if (material_id <= 0) {{
                    return false;
                }}
                if (ri.x >= 0 && material_id != ri.x) {{
                    return false;
                }}
                int clamped_material = clamp(material_id, 0, {MAX_MATERIALS - 1});
                if (!mask_matches(material_tags[clamped_material].z, rt.x)) {{
                    return false;
                }}
                int phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                int phase_mask = int(rt.z);
                if (phase_mask != 0 && ((1 << phase_id) & phase_mask) == 0) {{
                    return false;
                }}
                float temp_value = texelFetch(temp_tex, cell, 0).x;
                return temp_value >= rf.x && temp_value <= rf.y;
            }}

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y || gid.z >= light_count) {{
                    return;
                }}
                ivec2 cell = gid.xy;
                int layer = gid.z;
                float dose = texelFetch(cell_dose_tex, gid, 0).x;
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {{
                    ivec4 ri = material_light_rule_i[rule_index];
                    vec4 rf = material_light_rule_f[rule_index];
                    uvec4 rt = material_light_rule_tags[rule_index];
                    int consume_policy = int(rt.w);
                    if (consume_policy != {CONSUME_POLICY_RHS} && consume_policy != {CONSUME_POLICY_BOTH}) {{
                        continue;
                    }}
                    if (ri.y != layer || dose < rf.z) {{
                        continue;
                    }}
                    if (!material_cell_matches_rule(cell, ri, rf, rt)) {{
                        continue;
                    }}
                    dose = max(0.0, dose - dose * max(rf.w, 0.0));
                }}
                imageStore(cell_dose_out_img, gid, vec4(dose, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["material_light_gas_dose_consume"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int light_count;
            uniform int rule_count;
            layout(binding=0) uniform sampler2DArray gas_dose_tex;
            layout(binding=1) uniform sampler2DArray cell_dose_tex;
            layout(binding=2) uniform sampler2D material_tex;
            layout(binding=3) uniform sampler2D phase_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D active_cell_tex;
            layout(std430, binding=0) buffer MaterialLightRuleI {{
                ivec4 material_light_rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=1) buffer MaterialLightRuleF {{
                vec4 material_light_rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=2) buffer MaterialLightRuleTags {{
                uvec4 material_light_rule_tags[{MAX_RULES}];
            }};
            layout(std430, binding=3) buffer MaterialTags {{
                uvec4 material_tags[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=6) writeonly uniform image2DArray gas_dose_out_img;

            bool mask_matches(uint value, uint required) {{
                return required == 0u || (value & required) == required;
            }}

            bool material_cell_matches_rule(ivec2 cell, ivec4 ri, vec4 rf, uvec4 rt) {{
                if (texelFetch(active_cell_tex, cell, 0).x <= 0.5) {{
                    return false;
                }}
                int material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                if (material_id <= 0) {{
                    return false;
                }}
                if (ri.x >= 0 && material_id != ri.x) {{
                    return false;
                }}
                int clamped_material = clamp(material_id, 0, {MAX_MATERIALS - 1});
                if (!mask_matches(material_tags[clamped_material].z, rt.x)) {{
                    return false;
                }}
                int phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                int phase_mask = int(rt.z);
                if (phase_mask != 0 && ((1 << phase_id) & phase_mask) == 0) {{
                    return false;
                }}
                float temp_value = texelFetch(temp_tex, cell, 0).x;
                return temp_value >= rf.x && temp_value <= rf.y;
            }}

            float consumed_cell_light_dose(ivec2 cell, int layer) {{
                float dose = texelFetch(cell_dose_tex, ivec3(cell, layer), 0).x;
                float consumed = 0.0;
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {{
                    ivec4 ri = material_light_rule_i[rule_index];
                    vec4 rf = material_light_rule_f[rule_index];
                    uvec4 rt = material_light_rule_tags[rule_index];
                    int consume_policy = int(rt.w);
                    if (consume_policy != {CONSUME_POLICY_RHS} && consume_policy != {CONSUME_POLICY_BOTH}) {{
                        continue;
                    }}
                    if (ri.y != layer || dose < rf.z) {{
                        continue;
                    }}
                    if (!material_cell_matches_rule(cell, ri, rf, rt)) {{
                        continue;
                    }}
                    float scale = dose * max(rf.w, 0.0);
                    consumed += scale;
                    dose = max(0.0, dose - scale);
                }}
                return consumed;
            }}

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= light_count) {{
                    return;
                }}
                int x0 = gid.x * gas_cell_size;
                int y0 = gid.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                float consumed = 0.0;
                for (int y = y0; y < y1; ++y) {{
                    for (int x = x0; x < x1; ++x) {{
                        consumed += consumed_cell_light_dose(ivec2(x, y), gid.z) * 0.08;
                    }}
                }}
                float dose = texelFetch(gas_dose_tex, gid, 0).x;
                imageStore(gas_dose_out_img, gid, vec4(max(0.0, dose - consumed), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["material_material"] = ctx.compute_shader(
            helper
            + lhs_candidate_helper
            + """
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2DArray deferred_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2DArray deferred_hi_img;
            layout(rg32f, binding=7) writeonly uniform image2D cell_meta_out_img;
            uniform bool has_rhs_consume;
            bool cell_in_bounds(ivec2 cell) {
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }
            bool source_matches_material_rule(ivec2 source, ivec4 ri, vec4 rf, uvec4 rt) {
                if (!cell_in_bounds(source) || !solve_active(source)) {
                    return false;
                }
                int source_material = int(texelFetch(material_tex, source, 0).x + 0.5);
                if (source_material <= 0) {
                    return false;
                }
                if (ri.x >= 0 && source_material != ri.x) {
                    return false;
                }
                if (!mask_matches(material_tags[clamp_material_id(source_material)].x, rt.x)) {
                    return false;
                }
                int source_phase = int(texelFetch(phase_tex, source, 0).x + 0.5);
                int phase_mask = int(rt.z);
                if (phase_mask != 0 && ((1 << source_phase) & phase_mask) == 0) {
                    return false;
                }
                float source_temp = texelFetch(temp_tex, source, 0).x;
                return source_temp >= rf.x && source_temp <= rf.y;
            }
            bool source_selects_rhs_target(ivec2 source, ivec2 target, int required_id, uint required_mask) {
                if (!cell_in_bounds(target)) {
                    return false;
                }
                int target_material = int(texelFetch(material_tex, target, 0).x + 0.5);
                if (!match_material_selector(target_material, required_id, required_mask)) {
                    return false;
                }
                ivec2 left = source + ivec2(-1, 0);
                if (cell_in_bounds(left) && match_material_selector(int(texelFetch(material_tex, left, 0).x + 0.5), required_id, required_mask)) {
                    return left == target;
                }
                ivec2 right = source + ivec2(1, 0);
                if (cell_in_bounds(right) && match_material_selector(int(texelFetch(material_tex, right, 0).x + 0.5), required_id, required_mask)) {
                    return right == target;
                }
                ivec2 up = source + ivec2(0, -1);
                if (cell_in_bounds(up) && match_material_selector(int(texelFetch(material_tex, up, 0).x + 0.5), required_id, required_mask)) {
                    return up == target;
                }
                ivec2 down = source + ivec2(0, 1);
                if (cell_in_bounds(down) && match_material_selector(int(texelFetch(material_tex, down, 0).x + 0.5), required_id, required_mask)) {
                    return down == target;
                }
                return false;
            }
            void consume_rhs_material_sources(
                ivec2 target,
                inout float material_value,
                inout float phase_value,
                inout float integrity_value,
                inout vec4 timer_value,
                inout float cell_reset_value,
                inout float reaction_latched_value
            ) {
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    if (rt.w != CONSUME_POLICY_RHS_ID && rt.w != CONSUME_POLICY_BOTH_ID) {
                        continue;
                    }
                    ivec2 source = target + ivec2(1, 0);
                    if (source_matches_material_rule(source, ri, rf, rt) && source_selects_rhs_target(source, target, ri.y, rt.y)) {
                        consume_current_material(material_value, phase_value, integrity_value, timer_value, cell_reset_value, reaction_latched_value, rf.w);
                    }
                    source = target + ivec2(-1, 0);
                    if (source_matches_material_rule(source, ri, rf, rt) && source_selects_rhs_target(source, target, ri.y, rt.y)) {
                        consume_current_material(material_value, phase_value, integrity_value, timer_value, cell_reset_value, reaction_latched_value, rf.w);
                    }
                    source = target + ivec2(0, 1);
                    if (source_matches_material_rule(source, ri, rf, rt) && source_selects_rhs_target(source, target, ri.y, rt.y)) {
                        consume_current_material(material_value, phase_value, integrity_value, timer_value, cell_reset_value, reaction_latched_value, rf.w);
                    }
                    source = target + ivec2(0, -1);
                    if (source_matches_material_rule(source, ri, rf, rt) && source_selects_rhs_target(source, target, ri.y, rt.y)) {
                        consume_current_material(material_value, phase_value, integrity_value, timer_value, cell_reset_value, reaction_latched_value, rf.w);
                    }
                }
            }
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                int material_id = int(material_value + 0.5);
                if (solve_active(gid) && material_id > 0) {
                    for (int word_index = 0; word_index < rule_candidate_word_count; ++word_index) {
                        uint candidate_bits = lhs_rule_candidate_word(material_id, word_index);
                        while (candidate_bits != 0u) {
                            int bit_index = findLSB(candidate_bits);
                            int rule_index = word_index * 32 + bit_index;
                            candidate_bits &= candidate_bits - 1u;
                            if (rule_index >= rule_count) {
                                continue;
                            }
                            ivec4 ri = rule_i[rule_index];
                            vec4 rf = rule_f[rule_index];
                            uvec4 rt = rule_tags[rule_index];
                            if (ri.x >= 0 && material_id != ri.x) {
                                continue;
                            }
                            if (!mask_matches(material_tags[clamp_material_id(material_id)].x, rt.x)) {
                                continue;
                            }
                            int phase_mask = int(rt.z);
                            float rule_scale = rf.w;
                            if (phase_mask != 0 && ((1 << int(phase_value + 0.5)) & phase_mask) == 0) {
                                continue;
                            }
                            if (temp_value < rf.x || temp_value > rf.y) {
                                continue;
                            }
                            bool matched = false;
                            ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                            ivec2 right = ivec2(min(gid.x + 1, cell_grid_size.x - 1), gid.y);
                            ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                            ivec2 up = ivec2(gid.x, min(gid.y + 1, cell_grid_size.y - 1));
                            matched = matched || match_material_selector(int(texelFetch(material_tex, left, 0).x + 0.5), ri.y, rt.y);
                            matched = matched || match_material_selector(int(texelFetch(material_tex, right, 0).x + 0.5), ri.y, rt.y);
                            matched = matched || match_material_selector(int(texelFetch(material_tex, down, 0).x + 0.5), ri.y, rt.y);
                            matched = matched || match_material_selector(int(texelFetch(material_tex, up, 0).x + 0.5), ri.y, rt.y);
                            if (matched) {
                                if (ri.w >= 0) {
                                    apply_slot_trigger(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.w,
                                        rule_scale
                                    );
                                } else if (ri.z >= 0) {
                                    apply_action(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.z,
                                        rule_scale
                                    );
                                }
                                if (rt.w == CONSUME_POLICY_LHS_ID || rt.w == CONSUME_POLICY_BOTH_ID) {
                                    consume_current_material(
                                        material_value,
                                        phase_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        rule_scale
                                    );
                                }
                            }
                        }
                    }
                }
                if (has_rhs_consume) {
                    consume_rhs_material_sources(
                        gid,
                        material_value,
                        phase_value,
                        integrity_value,
                        timer_value,
                        cell_reset_value,
                        reaction_latched_value
                    );
                }
                imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, timer_value);
                imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                imageStore(cell_meta_out_img, gid, vec4(cell_reset_value, reaction_latched_value, 0.0, 0.0));
            }
            """
        )
        self.programs["material_gas"] = ctx.compute_shader(
            helper
            + lhs_candidate_helper
            + """
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2DArray deferred_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2DArray deferred_hi_img;
            layout(rg32f, binding=7) writeonly uniform image2D cell_meta_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                if (!solve_active(gid)) {
                    imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                    imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                    imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                    imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                    imageStore(timer_out_img, gid, timer_value);
                    imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                    imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                    imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                    imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                    imageStore(cell_meta_out_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }
                ivec2 gas_cell = ivec2(
                    min(gid.x / gas_cell_size, textureSize(gas_tex, 0).x - 1),
                    min(gid.y / gas_cell_size, textureSize(gas_tex, 0).y - 1)
                );
                int material_id = int(material_value + 0.5);
                if (material_id > 0) {
                    for (int word_index = 0; word_index < rule_candidate_word_count; ++word_index) {
                        uint candidate_bits = lhs_rule_candidate_word(material_id, word_index);
                        while (candidate_bits != 0u) {
                            int bit_index = findLSB(candidate_bits);
                            int rule_index = word_index * 32 + bit_index;
                            candidate_bits &= candidate_bits - 1u;
                            if (rule_index >= rule_count) {
                                continue;
                            }
                            ivec4 ri = rule_i[rule_index];
                            vec4 rf = rule_f[rule_index];
                            uvec4 rt = rule_tags[rule_index];
                            if (ri.x >= 0 && material_id != ri.x) {
                                continue;
                            }
                            if (!mask_matches(material_tags[clamp_material_id(material_id)].y, rt.x)) {
                                continue;
                            }
                            int phase_mask = int(rt.z);
                            float rule_scale = rf.w;
                            if (phase_mask != 0 && ((1 << int(phase_value + 0.5)) & phase_mask) == 0) {
                                continue;
                            }
                            if (temp_value < rf.x || temp_value > rf.y) {
                                continue;
                            }
                            float concentration = 0.0;
                            if (ri.y >= 0) {
                                if (mask_matches(gas_tags[ri.y].x, rt.y)) {
                                    concentration = texelFetch(gas_tex, ivec3(gas_cell, ri.y), 0).x;
                                }
                            } else if (rt.y != 0u) {
                                for (int species_id = 0; species_id < gas_count; ++species_id) {
                                    if (!mask_matches(gas_tags[species_id].x, rt.y)) {
                                        continue;
                                    }
                                    concentration = max(concentration, texelFetch(gas_tex, ivec3(gas_cell, species_id), 0).x);
                                }
                            } else {
                                continue;
                            }
                            if (concentration >= rf.z) {
                                if (ri.w >= 0) {
                                    apply_slot_trigger(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.w,
                                        concentration * rule_scale
                                    );
                                } else if (ri.z >= 0) {
                                    apply_action(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.z,
                                        concentration * rule_scale
                                    );
                                }
                                if (rt.w == CONSUME_POLICY_LHS_ID || rt.w == CONSUME_POLICY_BOTH_ID) {
                                    consume_current_material(
                                        material_value,
                                        phase_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        concentration * rule_scale
                                    );
                                }
                            }
                        }
                    }
                }
                imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, timer_value);
                imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                imageStore(cell_meta_out_img, gid, vec4(cell_reset_value, reaction_latched_value, 0.0, 0.0));
            }
            """
        )
        self.programs["material_light"] = ctx.compute_shader(
            helper
            + lhs_candidate_helper
            + """
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2DArray deferred_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2DArray deferred_hi_img;
            layout(rg32f, binding=7) writeonly uniform image2D cell_meta_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                float temp_value = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                vec4 deferred_action_lo = vec4(0.0);
                vec4 deferred_action_hi = vec4(0.0);
                vec4 deferred_scale_lo = vec4(0.0);
                vec4 deferred_scale_hi = vec4(0.0);
                float cell_reset_value = 0.0;
                float reaction_latched_value = 0.0;
                int deferred_count = 0;
                if (!solve_active(gid)) {
                    imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                    imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                    imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                    imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                    imageStore(timer_out_img, gid, timer_value);
                    imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                    imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                    imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                    imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                    imageStore(cell_meta_out_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }
                int material_id = int(material_value + 0.5);
                if (material_id > 0) {
                    for (int word_index = 0; word_index < rule_candidate_word_count; ++word_index) {
                        uint candidate_bits = lhs_rule_candidate_word(material_id, word_index);
                        while (candidate_bits != 0u) {
                            int bit_index = findLSB(candidate_bits);
                            int rule_index = word_index * 32 + bit_index;
                            candidate_bits &= candidate_bits - 1u;
                            if (rule_index >= rule_count) {
                                continue;
                            }
                            ivec4 ri = rule_i[rule_index];
                            vec4 rf = rule_f[rule_index];
                            uvec4 rt = rule_tags[rule_index];
                            if (ri.x >= 0 && material_id != ri.x) {
                                continue;
                            }
                            if (!mask_matches(material_tags[clamp_material_id(material_id)].z, rt.x)) {
                                continue;
                            }
                            float rule_scale = rf.w;
                            int phase_mask = int(rt.z);
                            if (phase_mask != 0 && ((1 << int(phase_value + 0.5)) & phase_mask) == 0) {
                                continue;
                            }
                            if (temp_value < rf.x || temp_value > rf.y) {
                                continue;
                            }
                            if (ri.y < 0) {
                                continue;
                            }
                            float dose = texelFetch(dose_tex, ivec3(gid, ri.y), 0).x;
                            if (dose >= rf.z) {
                                if (ri.w >= 0) {
                                    apply_slot_trigger(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.w,
                                        dose * rule_scale
                                    );
                                } else if (ri.z >= 0) {
                                    apply_action(
                                        material_value,
                                        phase_value,
                                        temp_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        deferred_action_lo,
                                        deferred_action_hi,
                                        deferred_scale_lo,
                                        deferred_scale_hi,
                                        deferred_count,
                                        gid,
                                        ri.z,
                                        dose * rule_scale
                                    );
                                }
                                if (rt.w == CONSUME_POLICY_LHS_ID || rt.w == CONSUME_POLICY_BOTH_ID) {
                                    consume_current_material(
                                        material_value,
                                        phase_value,
                                        integrity_value,
                                        timer_value,
                                        cell_reset_value,
                                        reaction_latched_value,
                                        dose * rule_scale
                                    );
                                }
                            }
                        }
                    }
                }
                imageStore(material_out_img, gid, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, timer_value);
                imageStore(deferred_lo_img, ivec3(gid, 0), deferred_action_lo);
                imageStore(deferred_lo_img, ivec3(gid, 1), deferred_scale_lo);
                imageStore(deferred_hi_img, ivec3(gid, 0), deferred_action_hi);
                imageStore(deferred_hi_img, ivec3(gid, 1), deferred_scale_hi);
                imageStore(cell_meta_out_img, gid, vec4(cell_reset_value, reaction_latched_value, 0.0, 0.0));
            }
            """
        )
        self.programs["gas_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform int rule_count;
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(binding=1) uniform sampler2D ambient_tex;
            layout(binding=2) uniform sampler2D active_gas_tex;
            layout(binding=3) uniform sampler2D material_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D flow_velocity_tex;
            layout(std430, binding=0) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=1) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=2) buffer RuleI {{
                ivec4 rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=3) buffer RuleF {{
                vec4 rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=4) buffer RuleTags {{
                uvec4 rule_tags[{MAX_RULES}];
            }};
            layout(std430, binding=5) buffer GasTags {{
                uvec4 gas_tags[{MAX_MATERIALS}];
            }};
            layout(std430, binding=6) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=14) buffer LightEmitterBuffer {{
                vec4 light_emitters[{MAX_EMITTED_LIGHTS * 2}];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(r32f, binding=0) writeonly uniform image2DArray gas_out_img;
            layout(r32f, binding=1) writeonly uniform image2D ambient_out_img;
            layout(rgba32f, binding=2) writeonly uniform image2DArray flow_source_img;
            layout(rgba32f, binding=3) writeonly uniform image2D emit_cell_lo_img;
            layout(rgba32f, binding=4) writeonly uniform image2D emit_cell_hi_img;
            layout(rgba32f, binding=5) writeonly uniform image2D emit_timer_img;
            layout(rg32f, binding=6) writeonly uniform image2D emit_meta_img;

            void write_flow_sources(ivec2 cell, ivec4 ai, vec4 af, float scale) {{
                if (ai.w == 0) {{
                    return;
                }}
                float strength = af.y * max(scale, 0.0);
                float radius = af.z;
                if (strength <= 0.0 || radius <= 0.0) {{
                    return;
                }}
                int direction_id = ai.z;
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["left"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(-1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["up"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(0.0, -1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["down"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(0.0, 1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["all"]} && abs(af.w) > 1.0e-5) {{
                    float flow_sign = af.w > 0.0 ? 1.0 : -1.0;
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(-flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 1), vec4(flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 2), vec4(0.0, -flow_sign, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 3), vec4(0.0, flow_sign, radius, strength));
                }}
            }}

            void record_gas_action(int action_type) {{
                if (
                    action_type == {TYPE_MODIFY_GAS}
                    || action_type == {TYPE_MODIFY_TEMPERATURE}
                    || action_type == {TYPE_EMIT_LIGHT}
                    || action_type == {TYPE_EMIT_MATERIAL}
                ) {{
                    atomicAdd(reaction_counts[1], 1u);
                    atomicAdd(reaction_counts[1 + action_type], 1u);
                }}
            }}

            ivec2 gas_origin_cell(ivec2 gas_cell) {{
                return ivec2(
                    min(cell_grid_size.x - 1, gas_cell.x * gas_cell_size + gas_cell_size / 2),
                    min(cell_grid_size.y - 1, gas_cell.y * gas_cell_size + gas_cell_size / 2)
                );
            }}

            vec2 gas_flow_direction(ivec2 gas_cell) {{
                vec2 flow = texelFetch(flow_velocity_tex, gas_cell, 0).xy;
                float norm = length(flow);
                if (norm > 1.0e-5) {{
                    return flow / norm;
                }}
                return vec2(0.0);
            }}

            vec2 light_direction(ivec2 gas_cell, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return vec2(1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return vec2(-1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return vec2(0.0, -1.0);
                }}
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return vec2(0.0, 1.0);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    return gas_flow_direction(gas_cell);
                }}
                return vec2(0.0);
            }}

            ivec2 target_for_direction(ivec2 source, ivec2 gas_cell, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return source + ivec2(0, 1);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return source + ivec2(0, -1);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return source + ivec2(-1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return source + ivec2(1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["random"]}) {{
                    uint selector = (uint(source.x) * 73856093u) ^ (uint(source.y) * 19349663u);
                    int dx = int(selector % 3u) - 1;
                    int dy = int((selector % 9u) / 3u) - 1;
                    return source + ivec2(dx, dy);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 flow = texelFetch(flow_velocity_tex, gas_cell, 0).xy;
                    return source + ivec2(int(sign(flow.x)), int(sign(flow.y)));
                }}
                return source;
            }}

            vec2 emitted_velocity(ivec2 source, ivec2 target, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                int phase_id = int(material_params[material_id].y + 0.5);
                if (phase_id != {int(Phase.POWDER)}) {{
                    return vec2(0.0);
                }}
                vec2 explicit_velocity = af.xy;
                if (length(explicit_velocity) > 1.0e-5) {{
                    return explicit_velocity;
                }}
                vec2 direction = vec2(target - source);
                float norm = max(1.0e-5, length(direction));
                return direction / norm * max(0.0, af.z);
            }}

            void append_light_emitter(ivec2 gas_cell, ivec4 ai, vec4 af, float scale) {{
                if (ai.y < 0) {{
                    return;
                }}
                uint emitter_index = atomicAdd(reaction_counts[0], 1u);
                if (emitter_index >= uint({MAX_EMITTED_LIGHTS})) {{
                    return;
                }}
                ivec2 origin = gas_origin_cell(gas_cell);
                vec2 direction = light_direction(gas_cell, ai.z);
                uint base_index = emitter_index * 2u;
                light_emitters[base_index] = vec4(float(origin.x), float(origin.y), direction.x, direction.y);
                light_emitters[base_index + 1u] = vec4(max(0.1, af.x * scale), af.y, max(0.0, af.z), float(ai.y));
            }}

            void emit_material(ivec2 gas_cell, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                if (material_id <= 0) {{
                    return;
                }}
                ivec2 source = gas_origin_cell(gas_cell);
                ivec2 target = target_for_direction(source, gas_cell, ai.z);
                if (target.x < 0 || target.y < 0 || target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                    return;
                }}
                if (texelFetch(material_tex, target, 0).x > 0.5) {{
                    return;
                }}
                float target_phase = material_params[material_id].y;
                float target_integrity = material_params[material_id].x;
                float current_temp = texelFetch(temp_tex, target, 0).x;
                float spawn_temp = material_params[material_id].z;
                float target_temp = current_temp;
                if (spawn_temp == spawn_temp) {{
                    target_temp = max(current_temp, spawn_temp);
                }}
                imageStore(emit_cell_lo_img, target, vec4(float(material_id), target_phase, target_temp, target_integrity));
                imageStore(emit_cell_hi_img, target, vec4(emitted_velocity(source, target, ai, af), 0.0, 0.0));
                imageStore(emit_timer_img, target, vec4(0.0));
                imageStore(emit_meta_img, target, vec4(1.0, 0.0, 0.0, 0.0));
            }}

            int best_matching_gas_layer(ivec2 cell, int required_id, uint required_mask) {{
                if (required_id >= 0) {{
                    if (required_mask != 0u && (gas_tags[required_id].x & required_mask) != required_mask) {{
                        return -1;
                    }}
                    return required_id;
                }}
                if (required_mask == 0u) {{
                    return -1;
                }}
                float best = -1.0;
                int best_species = -1;
                for (int species_id = 0; species_id < gas_count; ++species_id) {{
                    if ((gas_tags[species_id].x & required_mask) != required_mask) {{
                        continue;
                    }}
                    float candidate = texelFetch(gas_tex, ivec3(cell, species_id), 0).x;
                    if (candidate > best) {{
                        best = candidate;
                        best_species = species_id;
                    }}
                }}
                return best_species;
            }}

            float matching_gas_value(ivec2 cell, int required_id, uint required_mask) {{
                int species_id = best_matching_gas_layer(cell, required_id, required_mask);
                if (species_id < 0) {{
                    return 0.0;
                }}
                return texelFetch(gas_tex, ivec3(cell, species_id), 0).x;
            }}

            bool selector_matches_current_layer(ivec2 cell, int layer, int required_id, uint required_mask) {{
                return layer == best_matching_gas_layer(cell, required_id, required_mask);
            }}

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= gas_count) {{
                    return;
                }}
                float gas_value = texelFetch(gas_tex, gid, 0).x;
                float ambient_value = texelFetch(ambient_tex, gid.xy, 0).x;
                if (texelFetch(active_gas_tex, gid.xy, 0).x <= 0.5) {{
                    imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
                    if (gid.z == 0) {{
                        imageStore(ambient_out_img, gid.xy, vec4(ambient_value, 0.0, 0.0, 0.0));
                    }}
                    return;
                }}
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {{
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    if ((ri.x < 0 && rt.x == 0u) || (ri.y < 0 && rt.y == 0u)) {{
                        continue;
                    }}
                    if (ambient_value < rf.x || ambient_value > rf.y) {{
                        continue;
                    }}
                    float lhs_value = matching_gas_value(gid.xy, ri.x, rt.x);
                    float rhs_value = matching_gas_value(gid.xy, ri.y, rt.y);
                    if (lhs_value < rf.z || rhs_value < rf.z) {{
                        continue;
                    }}
                    float scale = min(lhs_value, rhs_value) * max(rf.w, 0.0);
                    ivec4 ai = ivec4(0);
                    vec4 af = vec4(0.0);
                    if (ri.z >= 0) {{
                        ai = action_i[ri.z];
                        af = action_f[ri.z];
                    }}
                    if (ri.z >= 0 && ai.x == {TYPE_MODIFY_GAS} && gid.z == ai.y) {{
                        record_gas_action(ai.x);
                        gas_value = max(0.0, gas_value + af.x * scale);
                        write_flow_sources(gid.xy, ai, af, scale);
                    }} else if (ri.z >= 0 && ai.x == {TYPE_MODIFY_TEMPERATURE} && gid.z == 0) {{
                        record_gas_action(ai.x);
                        ambient_value += af.x * scale;
                    }} else if (ri.z >= 0 && ai.x == {TYPE_EMIT_LIGHT} && gid.z == 0) {{
                        record_gas_action(ai.x);
                        append_light_emitter(gid.xy, ai, af, scale);
                    }} else if (ri.z >= 0 && ai.x == {TYPE_EMIT_MATERIAL} && gid.z == 0) {{
                        record_gas_action(ai.x);
                        emit_material(gid.xy, ai, af);
                    }}
                    int consume_policy = int(rt.z);
                    if ((consume_policy == {CONSUME_POLICY_LHS} || consume_policy == {CONSUME_POLICY_BOTH})
                        && selector_matches_current_layer(gid.xy, gid.z, ri.x, rt.x)) {{
                        gas_value = max(0.0, gas_value - scale);
                    }}
                    if ((consume_policy == {CONSUME_POLICY_RHS} || consume_policy == {CONSUME_POLICY_BOTH})
                        && selector_matches_current_layer(gid.xy, gid.z, ri.y, rt.y)) {{
                        gas_value = max(0.0, gas_value - scale);
                    }}
                }}
                imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
                if (gid.z == 0) {{
                    imageStore(ambient_out_img, gid.xy, vec4(ambient_value, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["gas_light"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform int rule_count;
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(binding=1) uniform sampler2DArray gas_dose_tex;
            layout(binding=2) uniform sampler2D active_gas_tex;
            layout(binding=3) uniform sampler2D ambient_tex;
            layout(binding=4) uniform sampler2D material_tex;
            layout(binding=5) uniform sampler2D temp_tex;
            layout(binding=6) uniform sampler2D flow_velocity_tex;
            layout(std430, binding=0) buffer RuleI {{
                ivec4 rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=1) buffer RuleF {{
                vec4 rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=2) buffer ActionI {{
                ivec4 action_i[{MAX_ACTIONS}];
            }};
            layout(std430, binding=3) buffer ActionF {{
                vec4 action_f[{MAX_ACTIONS}];
            }};
            layout(std430, binding=4) buffer RuleTags {{
                uvec4 rule_tags[{MAX_RULES}];
            }};
            layout(std430, binding=5) buffer GasTags {{
                uvec4 gas_tags[{MAX_MATERIALS}];
            }};
            layout(std430, binding=6) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=14) buffer LightEmitterBuffer {{
                vec4 light_emitters[{MAX_EMITTED_LIGHTS * 2}];
            }};
            layout(std430, binding=15) buffer ReactionCounters {{
                uint reaction_counts[16];
            }};
            layout(r32f, binding=0) writeonly uniform image2DArray gas_out_img;
            layout(r32f, binding=1) writeonly uniform image2D ambient_out_img;
            layout(rgba32f, binding=2) writeonly uniform image2DArray flow_source_img;
            layout(rgba32f, binding=3) writeonly uniform image2D emit_cell_lo_img;
            layout(rgba32f, binding=4) writeonly uniform image2D emit_cell_hi_img;
            layout(rgba32f, binding=5) writeonly uniform image2D emit_timer_img;
            layout(rg32f, binding=6) writeonly uniform image2D emit_meta_img;
            void write_flow_sources(ivec2 cell, ivec4 ai, vec4 af, float scale) {{
                if (ai.w == 0) {{
                    return;
                }}
                float strength = af.y * max(scale, 0.0);
                float radius = af.z;
                if (strength <= 0.0 || radius <= 0.0) {{
                    return;
                }}
                int direction_id = ai.z;
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["left"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(-1.0, 0.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["up"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(0.0, -1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["down"]}) {{
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(0.0, 1.0, radius, strength));
                }} else if (direction_id == {DIRECTION_IDS["all"]} && abs(af.w) > 1.0e-5) {{
                    float flow_sign = af.w > 0.0 ? 1.0 : -1.0;
                    imageStore(flow_source_img, ivec3(cell, 0), vec4(-flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 1), vec4(flow_sign, 0.0, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 2), vec4(0.0, -flow_sign, radius, strength));
                    imageStore(flow_source_img, ivec3(cell, 3), vec4(0.0, flow_sign, radius, strength));
                }}
            }}
            void record_gas_action(int action_type) {{
                if (
                    action_type == {TYPE_MODIFY_GAS}
                    || action_type == {TYPE_MODIFY_TEMPERATURE}
                    || action_type == {TYPE_EMIT_LIGHT}
                    || action_type == {TYPE_EMIT_MATERIAL}
                ) {{
                    atomicAdd(reaction_counts[1], 1u);
                    atomicAdd(reaction_counts[1 + action_type], 1u);
                }}
            }}
            ivec2 gas_origin_cell(ivec2 gas_cell) {{
                return ivec2(
                    min(cell_grid_size.x - 1, gas_cell.x * gas_cell_size + gas_cell_size / 2),
                    min(cell_grid_size.y - 1, gas_cell.y * gas_cell_size + gas_cell_size / 2)
                );
            }}
            vec2 gas_flow_direction(ivec2 gas_cell) {{
                vec2 flow = texelFetch(flow_velocity_tex, gas_cell, 0).xy;
                float norm = length(flow);
                if (norm > 1.0e-5) {{
                    return flow / norm;
                }}
                return vec2(0.0);
            }}
            vec2 light_direction(ivec2 gas_cell, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return vec2(1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return vec2(-1.0, 0.0);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return vec2(0.0, -1.0);
                }}
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return vec2(0.0, 1.0);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    return gas_flow_direction(gas_cell);
                }}
                return vec2(0.0);
            }}
            ivec2 target_for_direction(ivec2 source, ivec2 gas_cell, int direction_id) {{
                if (direction_id == {DIRECTION_IDS["down"]}) {{
                    return source + ivec2(0, 1);
                }}
                if (direction_id == {DIRECTION_IDS["up"]}) {{
                    return source + ivec2(0, -1);
                }}
                if (direction_id == {DIRECTION_IDS["left"]}) {{
                    return source + ivec2(-1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["right"]}) {{
                    return source + ivec2(1, 0);
                }}
                if (direction_id == {DIRECTION_IDS["random"]}) {{
                    uint selector = (uint(source.x) * 73856093u) ^ (uint(source.y) * 19349663u);
                    int dx = int(selector % 3u) - 1;
                    int dy = int((selector % 9u) / 3u) - 1;
                    return source + ivec2(dx, dy);
                }}
                if (direction_id == {DIRECTION_IDS["speed"]}) {{
                    vec2 flow = texelFetch(flow_velocity_tex, gas_cell, 0).xy;
                    return source + ivec2(int(sign(flow.x)), int(sign(flow.y)));
                }}
                return source;
            }}
            vec2 emitted_velocity(ivec2 source, ivec2 target, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                int phase_id = int(material_params[material_id].y + 0.5);
                if (phase_id != {int(Phase.POWDER)}) {{
                    return vec2(0.0);
                }}
                vec2 explicit_velocity = af.xy;
                if (length(explicit_velocity) > 1.0e-5) {{
                    return explicit_velocity;
                }}
                vec2 direction = vec2(target - source);
                float norm = max(1.0e-5, length(direction));
                return direction / norm * max(0.0, af.z);
            }}
            void append_light_emitter(ivec2 gas_cell, ivec4 ai, vec4 af, float scale) {{
                if (ai.y < 0) {{
                    return;
                }}
                uint emitter_index = atomicAdd(reaction_counts[0], 1u);
                if (emitter_index >= uint({MAX_EMITTED_LIGHTS})) {{
                    return;
                }}
                ivec2 origin = gas_origin_cell(gas_cell);
                vec2 direction = light_direction(gas_cell, ai.z);
                uint base_index = emitter_index * 2u;
                light_emitters[base_index] = vec4(float(origin.x), float(origin.y), direction.x, direction.y);
                light_emitters[base_index + 1u] = vec4(max(0.1, af.x * scale), af.y, max(0.0, af.z), float(ai.y));
            }}
            void emit_material(ivec2 gas_cell, ivec4 ai, vec4 af) {{
                int material_id = clamp(ai.y, 0, {MAX_MATERIALS - 1});
                if (material_id <= 0) {{
                    return;
                }}
                ivec2 source = gas_origin_cell(gas_cell);
                ivec2 target = target_for_direction(source, gas_cell, ai.z);
                if (target.x < 0 || target.y < 0 || target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                    return;
                }}
                if (texelFetch(material_tex, target, 0).x > 0.5) {{
                    return;
                }}
                float target_phase = material_params[material_id].y;
                float target_integrity = material_params[material_id].x;
                float current_temp = texelFetch(temp_tex, target, 0).x;
                float spawn_temp = material_params[material_id].z;
                float target_temp = current_temp;
                if (spawn_temp == spawn_temp) {{
                    target_temp = max(current_temp, spawn_temp);
                }}
                imageStore(emit_cell_lo_img, target, vec4(float(material_id), target_phase, target_temp, target_integrity));
                imageStore(emit_cell_hi_img, target, vec4(emitted_velocity(source, target, ai, af), 0.0, 0.0));
                imageStore(emit_timer_img, target, vec4(0.0));
                imageStore(emit_meta_img, target, vec4(1.0, 0.0, 0.0, 0.0));
            }}
            bool mask_matches(uint value, uint required) {{
                if (required == 0u) {{
                    return true;
                }}
                return (value & required) == required;
            }}
            int best_matching_gas_layer(ivec2 cell, int required_id, uint required_mask) {{
                if (required_id >= 0) {{
                    if (!mask_matches(gas_tags[required_id].y, required_mask)) {{
                        return -1;
                    }}
                    return required_id;
                }}
                if (required_mask == 0u) {{
                    return -1;
                }}
                float best = -1.0;
                int best_species = -1;
                for (int species_id = 0; species_id < gas_count; ++species_id) {{
                    if (!mask_matches(gas_tags[species_id].y, required_mask)) {{
                        continue;
                    }}
                    float candidate = texelFetch(gas_tex, ivec3(cell, species_id), 0).x;
                    if (candidate > best) {{
                        best = candidate;
                        best_species = species_id;
                    }}
                }}
                return best_species;
            }}
            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= gas_count) {{
                    return;
                }}
                float gas_value = texelFetch(gas_tex, gid, 0).x;
                float ambient_value = texelFetch(ambient_tex, gid.xy, 0).x;
                if (texelFetch(active_gas_tex, gid.xy, 0).x <= 0.5) {{
                    imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
                    if (gid.z == 0) {{
                        imageStore(ambient_out_img, gid.xy, vec4(ambient_value, 0.0, 0.0, 0.0));
                    }}
                    return;
                }}
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {{
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    if (ambient_value < rf.x || ambient_value > rf.y) {{
                        continue;
                    }}
                    int source_species = best_matching_gas_layer(gid.xy, ri.x, rt.y);
                    float source_value = 0.0;
                    if (source_species >= 0) {{
                        source_value = texelFetch(gas_tex, ivec3(gid.xy, source_species), 0).x;
                    }}
                    float dose = texelFetch(gas_dose_tex, ivec3(gid.xy, ri.y), 0).x;
                    if (source_value >= rf.z && dose >= rf.z) {{
                        float scale = min(source_value, dose) * rf.w;
                        ivec4 ai = ivec4(0);
                        vec4 af = vec4(0.0);
                        if (ri.z >= 0) {{
                            ai = action_i[ri.z];
                            af = action_f[ri.z];
                        }}
                        if (ri.z >= 0 && ai.x == {TYPE_MODIFY_GAS} && gid.z == ai.y) {{
                            record_gas_action(ai.x);
                            gas_value = max(0.0, gas_value + af.x * scale);
                            write_flow_sources(gid.xy, ai, af, scale);
                        }} else if (ri.z >= 0 && ai.x == {TYPE_MODIFY_TEMPERATURE} && gid.z == 0) {{
                            record_gas_action(ai.x);
                            ambient_value += af.x * scale;
                        }} else if (ri.z >= 0 && ai.x == {TYPE_EMIT_LIGHT} && gid.z == 0) {{
                            record_gas_action(ai.x);
                            append_light_emitter(gid.xy, ai, af, scale);
                        }} else if (ri.z >= 0 && ai.x == {TYPE_EMIT_MATERIAL} && gid.z == 0) {{
                            record_gas_action(ai.x);
                            emit_material(gid.xy, ai, af);
                        }}
                        int consume_policy = int(rt.z);
                        if ((consume_policy == {CONSUME_POLICY_RHS} || consume_policy == {CONSUME_POLICY_BOTH})
                            && gid.z == source_species) {{
                            gas_value = max(0.0, gas_value - scale);
                        }}
                    }}
                }}
                imageStore(gas_out_img, gid, vec4(gas_value, 0.0, 0.0, 0.0));
                if (gid.z == 0) {{
                    imageStore(ambient_out_img, gid.xy, vec4(ambient_value, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )

    def _upload_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        reaction_group: str | None = None,
        compiled_actions: tuple[np.ndarray, np.ndarray] | None = None,
        light_dose_guard_buffer: Any | None = None,
        publishes_gas: bool | None = None,
    ) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        bridge_loads = self._bridge_input_load_requirements(
            world,
            reaction_group,
            compiled_actions,
            publishes_gas=publishes_gas,
        )
        profile_scope = self._upload_state_profile_scope(reaction_group)
        cache_key = self._formal_reaction_state_cache_key(world, resources, reaction_group)
        reuse_formal_state = cache_key is not None and self._formal_state_cache_key == cache_key
        batch_key_started = False
        if cache_key is not None and self._formal_segment_batch_base_key == cache_key[:3]:
            if self._formal_segment_batch_key is None:
                self._formal_segment_batch_key = cache_key
                batch_key_started = True
                self._formal_loaded_bridge_inputs_key = None
                self._formal_loaded_bridge_inputs.clear()
            elif self._formal_segment_batch_key != cache_key:
                self._formal_segment_batch_key = cache_key
                self._formal_pending_bridge_publish_key = None
                self._formal_pending_bridge_publish.clear()
                self._formal_active_mask_cache_key = None
                self._formal_loaded_bridge_inputs_key = None
                self._formal_loaded_bridge_inputs.clear()
                batch_key_started = True
        if cache_key is None:
            self._formal_state_cache_key = None
            self._formal_active_mask_cache_key = None
            self._formal_light_counters_cleared_key = None
            self._formal_loaded_bridge_inputs_key = None
            self._formal_loaded_bridge_inputs.clear()
            self._reset_formal_cell_read_role()
            if self._formal_segment_batch_base_key is None:
                self._formal_segment_batch_key = None
                self._formal_pending_bridge_publish_key = None
                self._formal_pending_bridge_publish.clear()
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
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
        self.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
        self.last_cpu_flow_velocity_upload_skipped = not upload_flow_velocity_from_cpu
        self.last_cpu_cell_dose_upload_skipped = not upload_cell_dose_from_cpu
        self.last_cpu_gas_dose_upload_skipped = not upload_gas_dose_from_cpu
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
            clear_requirements = self._transient_clear_requirements(reaction_group, compiled_actions)
            if clear_requirements["clear_light_counters"]:
                if cache_key is None:
                    self._formal_light_counters_cleared_key = None
                elif self._formal_light_counters_cleared_key == cache_key:
                    clear_requirements["clear_light_counters"] = False
                else:
                    self._formal_light_counters_cleared_key = cache_key
            if batch_key_started or (self._formal_segment_batch_key == cache_key and not reuse_formal_state):
                with self._profile_scoped_pass(world, profile_scope, "clear_segment_transient"):
                    self._clear_segment_transient_state(world, resources)
            with self._profile_scoped_pass(world, profile_scope, "clear_transient"):
                self._clear_transient_state(
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
            bridge_loads_to_run = self._missing_formal_bridge_input_loads(cache_key, bridge_loads)
        if not reuse_formal_state or bridge_loads_to_run.any():
            with self._profile_scoped_pass(world, profile_scope, "load_bridge_inputs"):
                self._load_authoritative_bridge_inputs(
                    world,
                    resources,
                    bridge_input_loads=bridge_loads_to_run,
                    reaction_group=reaction_group,
                    profile_scope=profile_scope,
                    light_dose_guard_buffer=light_dose_guard_buffer,
                )
            if cache_key is not None:
                self._formal_state_cache_key = cache_key
                self._record_formal_bridge_inputs_loaded(cache_key, bridge_loads)
                if not reuse_formal_state:
                    self._set_formal_cell_read_role("ping")
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
        self._upload_random_targets(world, resources, material_table)

    def _bridge_input_load_requirements(
        self,
        world: "WorldEngine",
        reaction_group: str | None,
        compiled_actions: tuple[np.ndarray, np.ndarray] | None,
        *,
        publishes_gas: bool | None = None,
    ) -> GPUReactionBridgeInputLoads:
        if not self._formal_gpu_frame(world):
            return GPUReactionBridgeInputLoads()
        if compiled_actions is None:
            return GPUReactionBridgeInputLoads()

        modifies_gas = self._compiled_actions_include_modify_gas(compiled_actions)
        gas_published = modifies_gas if publishes_gas is None else bool(publishes_gas)
        reads_gas = reaction_group in {"material_gas", "gas_gas", "gas_light"} or modifies_gas or gas_published
        reads_ambient = reaction_group in {"gas_gas", "gas_light"} or gas_published
        reads_cell_dose = reaction_group == "material_light"
        reads_gas_dose = reaction_group == "gas_light"
        segment = self._reaction_state_segment(reaction_group)
        if segment == "before_motion":
            return GPUReactionBridgeInputLoads(
                cell_core=True,
                gas=reads_gas,
                ambient=reads_ambient,
                flow_velocity=modifies_gas and self._compiled_actions_include_flow_sources(compiled_actions),
                cell_dose=reads_cell_dose,
                gas_dose=reads_gas_dose,
            )
        if segment == "after_optics":
            return GPUReactionBridgeInputLoads(
                cell_core=True,
                gas=reads_gas,
                ambient=reads_ambient,
                flow_velocity=modifies_gas and self._compiled_actions_include_flow_sources(compiled_actions),
                cell_dose=reads_cell_dose,
                gas_dose=reads_gas_dose,
            )
        return GPUReactionBridgeInputLoads()

    def _missing_formal_bridge_input_loads(
        self,
        cache_key: tuple[object, ...],
        bridge_loads: GPUReactionBridgeInputLoads,
    ) -> GPUReactionBridgeInputLoads:
        if self._formal_loaded_bridge_inputs_key != cache_key:
            return bridge_loads
        loaded = self._formal_loaded_bridge_inputs
        return GPUReactionBridgeInputLoads(
            cell_core=bridge_loads.cell_core and "cell_core" not in loaded,
            gas=bridge_loads.gas and "gas_concentration" not in loaded,
            ambient=bridge_loads.ambient and "ambient_temperature" not in loaded,
            flow_velocity=bridge_loads.flow_velocity and "flow_velocity" not in loaded,
            cell_dose=bridge_loads.cell_dose and "cell_optical_dose" not in loaded,
            gas_dose=bridge_loads.gas_dose and "gas_optical_dose" not in loaded,
        )

    def _record_formal_bridge_inputs_loaded(
        self,
        cache_key: tuple[object, ...],
        bridge_loads: GPUReactionBridgeInputLoads,
    ) -> None:
        if self._formal_loaded_bridge_inputs_key != cache_key:
            self._formal_loaded_bridge_inputs_key = cache_key
            self._formal_loaded_bridge_inputs.clear()
        self._formal_loaded_bridge_inputs.update(bridge_loads.resource_names())

    def _transient_clear_requirements(
        self,
        reaction_group: str | None,
        compiled_actions: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, bool]:
        emits_material = bool(
            compiled_actions is not None and self._compiled_actions_include_emit_material(compiled_actions)
        )
        return {
            "clear_light_counters": True,
            "clear_flow_sources": bool(
                compiled_actions is not None and self._compiled_actions_include_flow_sources(compiled_actions)
            ),
            "clear_emit_material_mask": emits_material,
            "clear_emit_material_buffers": emits_material and reaction_group in {"gas_gas", "gas_light"},
        }

    def _upload_random_targets(
        self,
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
        self.random_targets[:] = packed_random_targets
        self.random_target_count = min(len(random_targets), MAX_MATERIALS)
        resources.random_targets.write(self.random_targets.astype(np.int32, copy=False).tobytes())
        resources.random_targets_signature = random_targets_signature

    def _clear_transient_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        clear_light_counters: bool = True,
        clear_flow_sources: bool = False,
        clear_emit_material_mask: bool = False,
        clear_emit_material_buffers: bool = False,
        profile_scope: str | None = None,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        ran_clear = False
        with self._profile_scoped_pass(world, profile_scope, "clear_transient_full_cell_outputs_skipped"):
            pass
        if clear_light_counters:
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_light_counters"):
                counter_program = self.programs["clear_transient_light_counters"]
                resources.light_emitter_count.bind_to_storage_buffer(binding=0)
                counter_program.run(1, 1, 1)
                ran_clear = True
        else:
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_light_counters_skipped"):
                pass

        if clear_emit_material_buffers:
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_buffers"):
                emit_program = self.programs["clear_transient_emit_material_buffers"]
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
                    self._run_light_dose_guarded_dispatch(
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
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_mask"):
                mask_program = self.programs["clear_transient_emit_material_mask"]
                mask_program["cell_grid_size"].value = (world.width, world.height)
                resources.emitted_material_mask_tex.bind_to_image(0, read=False, write=True)
                group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
                group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
                if light_dose_guard_buffer is not None:
                    self._run_light_dose_guarded_dispatch(
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
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_emit_material_skipped"):
                pass

        if clear_flow_sources:
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_flow_sources"):
                flow_program = self.programs["clear_transient_flow_sources"]
                flow_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
                flow_program["flow_source_layers"].value = FLOW_SOURCE_LAYERS
                resources.flow_source_tex.bind_to_image(0, read=False, write=True)
                group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
                group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
                if light_dose_guard_buffer is not None:
                    self._run_light_dose_guarded_dispatch(
                        world,
                        resources,
                        flow_program,
                        light_dose_guard_buffer,
                        group_x,
                        group_y,
                        FLOW_SOURCE_LAYERS,
                    )
                else:
                    flow_program.run(group_x, group_y, FLOW_SOURCE_LAYERS)
                ran_clear = True
        else:
            with self._profile_scoped_pass(world, profile_scope, "clear_transient_flow_sources_skipped"):
                pass
        if ran_clear:
            self._sync_compute_writes(ctx)

    def _clear_segment_transient_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        program = self.programs["clear_segment_cell_transient_state"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.segment_cell_reset_tex.bind_to_image(0, read=False, write=True)
        resources.segment_reaction_latched_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _accumulate_segment_cell_transient_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        direct_core_outputs: bool = False,
    ) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        ctx = world.bridge.ctx
        if ctx is None:
            return
        program = self.programs["accumulate_segment_cell_transient_state"]
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
        self._sync_compute_writes(ctx)

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        bridge_input_loads: GPUReactionBridgeInputLoads | None = None,
        reaction_group: str | None = None,
        profile_scope: str | None = None,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        if not self._formal_gpu_frame(world):
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
            read_role_only_cell_core = self._bridge_cell_core_read_role_only_load(reaction_group)
            with self._profile_scoped_pass(world, profile_scope, "load_bridge_cell"):
                group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
                group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
                with self._profile_scoped_pass(world, profile_scope, "load_bridge_cell_core"):
                    program = self.programs[
                        "load_bridge_cell_role" if read_role_only_cell_core else "load_bridge_cell"
                    ]
                    program["cell_grid_size"].value = (world.width, world.height)
                    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                    if read_role_only_cell_core:
                        material_tex, phase_tex, temp_tex, integrity_tex, _velocity_tex, _timer_tex = (
                            self._current_cell_textures(resources)
                        )
                        material_tex.bind_to_image(0, read=False, write=True)
                        phase_tex.bind_to_image(1, read=False, write=True)
                        temp_tex.bind_to_image(2, read=False, write=True)
                        integrity_tex.bind_to_image(3, read=False, write=True)
                    else:
                        resources.material_ping.bind_to_image(0, read=False, write=True)
                        resources.material_pong.bind_to_image(1, read=False, write=True)
                        resources.phase_ping.bind_to_image(2, read=False, write=True)
                        resources.phase_pong.bind_to_image(3, read=False, write=True)
                        resources.temp_ping.bind_to_image(4, read=False, write=True)
                        resources.temp_pong.bind_to_image(5, read=False, write=True)
                        resources.integrity_ping.bind_to_image(6, read=False, write=True)
                        resources.integrity_pong.bind_to_image(7, read=False, write=True)
                    if light_dose_guard_buffer is not None:
                        self._run_light_dose_guarded_dispatch(
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
                with self._profile_scoped_pass(world, profile_scope, "load_bridge_cell_aux"):
                    program = self.programs[
                        "load_bridge_cell_aux_role" if read_role_only_cell_core else "load_bridge_cell_aux"
                    ]
                    program["cell_grid_size"].value = (world.width, world.height)
                    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                    if read_role_only_cell_core:
                        _material_tex, _phase_tex, _temp_tex, _integrity_tex, velocity_tex, timer_tex = (
                            self._current_cell_textures(resources)
                        )
                        velocity_tex.bind_to_image(0, read=False, write=True)
                        timer_tex.bind_to_image(1, read=False, write=True)
                    else:
                        resources.velocity_ping.bind_to_image(0, read=False, write=True)
                        resources.velocity_pong.bind_to_image(1, read=False, write=True)
                        resources.timer_ping.bind_to_image(2, read=False, write=True)
                        resources.timer_pong.bind_to_image(3, read=False, write=True)
                    if light_dose_guard_buffer is not None:
                        self._run_light_dose_guarded_dispatch(
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
            with self._profile_scoped_pass(world, profile_scope, "load_bridge_gas"):
                program = self.programs["load_bridge_gas"]
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
                    self._run_light_dose_guarded_dispatch(
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
            with self._profile_scoped_pass(world, profile_scope, "load_bridge_dose"):
                program = self.programs["load_bridge_dose"]
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
                    self._run_light_dose_guarded_dispatch(
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
                with self._profile_scoped_pass(world, profile_scope, "load_bridge_cell_sync"):
                    self._sync_compute_writes(bridge.ctx)
            else:
                self._sync_compute_writes(bridge.ctx)

    def _upload_active_masks(
        self,
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
        active_authoritative = self._active_scheduler_gpu_authoritative(world)
        self.last_cpu_active_upload_skipped = bool(active_authoritative)
        if active_authoritative:
            cache_key = self._formal_reaction_active_mask_cache_key(
                world,
                resources,
                reaction_group,
                expansion_radius=1,
                load_cell_mask=load_cell_mask,
                load_gas_mask=load_gas_mask,
            )
            existing_cache_key = self._formal_active_mask_cache_key
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
            self._load_authoritative_active_masks(
                world,
                resources,
                expansion_radius=1,
                light_dose_guard_buffer=light_dose_guard_buffer,
                load_cell_mask=load_cell_mask,
                load_gas_mask=load_gas_mask,
            )
            self._formal_active_mask_cache_key = cache_key
            return
        self._formal_active_mask_cache_key = None
        if load_cell_mask:
            if solve_cell_mask is None:
                raise RuntimeError("CPU active mask upload requires a materialized cell mask")
            resources.active_cell_tex.write(np.asarray(solve_cell_mask, dtype="f4").tobytes())
        if load_gas_mask:
            if solve_gas_mask is None:
                raise RuntimeError("CPU active mask upload requires a materialized gas mask")
            resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())

    def _load_authoritative_active_masks(
        self,
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
            program = self.programs[name]
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
            self._set_uniform_if_present(program, "tile_size", int(world.active.tile_size))
            self._set_uniform_if_present(program, "expansion_radius", int(expansion_radius))
            bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
            texture.bind_to_image(1, read=False, write=True)
            with self._profile_pass(world, name):
                group_x = (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE
                group_y = (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE
                if light_dose_guard_buffer is not None:
                    self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(bridge.ctx)

    def _upload_local_metadata(
        self,
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

    def _promote_cell_pong_to_ping(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        program = self.programs["promote_reaction_cell_state"]
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
        with self._profile_pass(world, "promote_cell_pong"):
            program.run(
                (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
            self._sync_compute_writes(world.bridge.ctx)

    def _copy_gas_state(
        self,
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
        program = self.programs["promote_reaction_gas_state"]
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
        self._sync_compute_writes(world.bridge.ctx)

    def _promote_gas_pong_to_ping(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        with self._profile_pass(world, "promote_gas_pong"):
            self._copy_gas_state(
                world,
                resources,
                gas_source=resources.gas_pong,
                ambient_source=resources.ambient_pong,
                gas_destination=resources.gas_ping,
                ambient_destination=resources.ambient_ping,
            )

    def _promote_gas_result(self, world: "WorldEngine", resources: GPUReactionResources, gas_source: Any, ambient_source: Any) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        with self._profile_pass(world, "promote_gas_pong"):
            if gas_source is resources.gas_ping and ambient_source is resources.ambient_ping:
                self._copy_gas_state(
                    world,
                    resources,
                    gas_source=gas_source,
                    ambient_source=ambient_source,
                    gas_destination=resources.gas_pong,
                    ambient_destination=resources.ambient_pong,
                )
                return
            self._copy_gas_state(
                world,
                resources,
                gas_source=gas_source,
                ambient_source=ambient_source,
                gas_destination=resources.gas_ping,
                ambient_destination=resources.ambient_ping,
            )

    def _promote_dose_pong_to_ping(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        program = self.programs["promote_reaction_dose_state"]
        light_count = int(world.cell_optical_dose.shape[0])
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["light_count"].value = light_count
        resources.cell_dose_pong.use(location=0)
        resources.gas_dose_pong.use(location=1)
        resources.cell_dose_tex.bind_to_image(0, read=False, write=True)
        resources.gas_dose_tex.bind_to_image(1, read=False, write=True)
        with self._profile_pass(world, "promote_dose_pong"):
            program.run(
                (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                light_count,
            )
            self._sync_compute_writes(world.bridge.ctx)

    def _copy_bridge_flow_velocity_to_reaction(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_reaction_state_cache_active():
            return
        program = self.programs["copy_bridge_flow_velocity_to_reaction"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        world.bridge.textures["flow_velocity"].use(location=0)
        resources.flow_velocity_tex.bind_to_image(0, read=False, write=True)
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(world.bridge.ctx)

    def _publish_bridge_cell_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        source_role: str | None = None,
        cell_reset_texture: Any | None = None,
        reaction_latched_texture: Any | None = None,
        cell_meta_texture: Any | None = None,
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
        material_tex, phase_tex, temp_tex, integrity_tex, velocity_tex, timer_tex = self._cell_role_textures(
            resources,
            source_role or "pong",
        )
        if mark_structure_dirty:
            mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
                world,
                material_tex,
                phase_tex,
                dispatch_guard_buffer=light_dose_guard_buffer,
            )
        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["use_cell_meta_texture"].value = cell_meta_texture is not None
        material_tex.use(location=0)
        phase_tex.use(location=1)
        temp_tex.use(location=2)
        integrity_tex.use(location=3)
        velocity_tex.use(location=4)
        timer_tex.use(location=5)
        (cell_reset_texture or resources.cell_reset_tex).use(location=6)
        (reaction_latched_texture or resources.reaction_latched_tex).use(location=7)
        (cell_meta_texture or resources.local_cell_meta_out).use(location=8)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("cell_core", "material")

    def _publish_bridge_gas_state(
        self,
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
        program = self.programs["publish_bridge_gas"]
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
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("gas_concentration", "ambient_temperature")

    def _publish_bridge_dose_state(
        self,
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
        cell_program = self.programs["publish_bridge_cell_dose"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        cell_program["light_count"].value = light_count
        resources.cell_dose_pong.use(location=0)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=0)
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        gas_program = self.programs["publish_bridge_gas_dose"]
        gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        gas_program["light_count"].value = light_count
        resources.gas_dose_pong.use(location=0)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=0)
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")

    def _apply_flow_sources_to_bridge_velocity(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative flow state")
        program = self.programs["apply_bridge_flow_sources"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["tile_size"].value = int(world.active.tile_size)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["impulse_dt"].value = 1.0 / 60.0
        resources.flow_velocity_tex.use(location=0)
        resources.flow_source_tex.use(location=1)
        bridge.textures["flow_velocity"].bind_to_image(2, read=False, write=True)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if light_dose_guard_buffer is not None:
            self._run_light_dose_guarded_dispatch(
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
        self._sync_compute_writes(bridge.ctx)
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        bridge.mark_gpu_authoritative("flow_velocity", "active_meta", "active_tile_ttl", "active_chunk_mask")
        self._formal_active_mask_cache_key = None
        self._copy_bridge_flow_velocity_to_reaction(world, resources)

    def _publish_bridge_light_emitters(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative emitted light state")
        program = self.programs["publish_bridge_light_emitters"]
        program["emitter_vec4_count"].value = int(MAX_EMITTED_LIGHTS * 2)
        program["counter_count"].value = 16
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=0)
        resources.light_emitter_count.bind_to_storage_buffer(binding=1)
        bridge.buffers["reaction_light_emitter"].bind_to_storage_buffer(binding=2)
        bridge.buffers["reaction_light_emitter_count"].bind_to_storage_buffer(binding=3)
        program.run((max(MAX_EMITTED_LIGHTS * 2, 16) + 255) // 256, 1, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("reaction_light_emitter", "reaction_light_emitter_count")

    def _download_cell_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        direct_core_outputs: bool = False,
    ) -> None:
        if self._formal_gpu_frame(world):
            rotate_formal_cell_roles = self._formal_before_motion_cell_roles_active()
            if self._formal_segment_batch_active():
                self._accumulate_segment_cell_transient_state(
                    world,
                    resources,
                    direct_core_outputs=direct_core_outputs,
                )
                if rotate_formal_cell_roles:
                    self._advance_formal_cell_read_role()
                else:
                    self._promote_cell_pong_to_ping(world, resources)
                self._mark_formal_bridge_publish_pending(world, resources, "cell")
            else:
                if rotate_formal_cell_roles:
                    source_role = self._formal_cell_write_role()
                    self._publish_bridge_cell_state(
                        world,
                        resources,
                        source_role=source_role,
                        cell_meta_texture=resources.local_cell_meta_out if direct_core_outputs else None,
                    )
                    self._set_formal_cell_read_role(source_role)
                else:
                    self._publish_bridge_cell_state(
                        world,
                        resources,
                        cell_meta_texture=resources.local_cell_meta_out if direct_core_outputs else None,
                    )
                    self._promote_cell_pong_to_ping(world, resources)
            self.last_cpu_mirror_downloaded = False
            return
        self.last_cpu_mirror_downloaded = True
        previous_material = world.material_id.copy()
        previous_phase = world.phase.copy()
        previous_island_id = world.island_id.copy()
        world.material_id[:] = np.rint(
            np.frombuffer(resources.material_pong.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(resources.phase_pong.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(resources.temp_pong.read(), dtype="f4").reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(resources.integrity_pong.read(), dtype="f4").reshape((world.height, world.width))
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_pong.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
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

    def _download_gas_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if self._formal_gpu_frame(world):
            if self._formal_segment_batch_active():
                self._promote_gas_pong_to_ping(world, resources)
                self._mark_formal_bridge_publish_pending(world, resources, "gas")
            else:
                self._publish_bridge_gas_state(world, resources)
                self._promote_gas_pong_to_ping(world, resources)
            self.last_cpu_mirror_downloaded = False
            return
        self.last_cpu_mirror_downloaded = True
        world.gas_concentration[:] = np.maximum(
            np.frombuffer(resources.gas_pong.read(), dtype="f4").reshape(world.gas_concentration.shape),
            0.0,
        )

    def _download_dose_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if self._formal_gpu_frame(world):
            if self._formal_segment_batch_active():
                self._promote_dose_pong_to_ping(world, resources)
                self._mark_formal_bridge_publish_pending(world, resources, "dose")
            else:
                self._publish_bridge_dose_state(world, resources)
                self._promote_dose_pong_to_ping(world, resources)
            self.last_cpu_mirror_downloaded = False
            return
        self.last_cpu_mirror_downloaded = True
        world.cell_optical_dose[:] = np.maximum(
            np.frombuffer(resources.cell_dose_pong.read(), dtype="f4").reshape(world.cell_optical_dose.shape),
            0.0,
        )
        world.gas_optical_dose[:] = np.maximum(
            np.frombuffer(resources.gas_dose_pong.read(), dtype="f4").reshape(world.gas_optical_dose.shape),
            0.0,
        )

    def _download_deferred_batch(self, world: "WorldEngine", resources: GPUReactionResources) -> GPUDeferredActionBatch:
        shape = (world.height, world.width, 4)
        if self._formal_gpu_frame(world):
            unsupported = self._unsupported_deferred_action_indices(world)
            if unsupported:
                raise RuntimeError(
                    "GPU reaction pipeline encountered unsupported deferred action indices "
                    f"{unsupported}; CPU fallback is disabled"
                )
            self._publish_bridge_light_emitters(world, resources)
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

    def _unsupported_deferred_action_indices(self, world: "WorldEngine") -> list[int]:
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
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        may_have_flow_sources: bool = True,
        light_dose_guard_buffer: Any | None = None,
    ) -> None:
        if not may_have_flow_sources:
            return
        if self._formal_gpu_frame(world):
            self._apply_flow_sources_to_bridge_velocity(
                world,
                resources,
                light_dose_guard_buffer=light_dose_guard_buffer,
            )
            self.last_cpu_mirror_downloaded = False
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

    def _compile_action_buffers(
        self,
        action_table: np.ndarray,
        used_indices: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
        action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
        action_count = int(action_table.shape[0])
        if action_count > MAX_ACTIONS:
            return None
        for index in range(action_count):
            row = action_table[index]
            action_i[index, 3] = max(0, int(row["duration"]))
            if used_indices is not None and index not in used_indices:
                action_i[index, 0] = TYPE_NONE
                continue
            reaction_type_id = int(row["reaction_type_id"])
            flags = 0
            if reaction_type_id == int(ReactionType.HARM.value):
                action_i[index, 0] = TYPE_HARM
                action_f[index, 1] = float(row["value"])
            elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
                action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
                action_f[index, 0] = float(row["delta"])
            elif reaction_type_id == int(ReactionType.CONVERT_MATERIAL.value):
                action_i[index, 0] = TYPE_CONVERT_MATERIAL
                if int(row["flags"]) & ACTION_FLAG_RANDOM_TARGET:
                    flags |= 1
                if int(row["flags"]) & ACTION_FLAG_ALLOW_SUBUNIT_SCALE:
                    flags |= ACTION_FLAG_ALLOW_SUBUNIT_SCALE
                action_i[index, 1] = int(row["target_material_id"])
                action_i[index, 2] = flags
                action_f[index, 2] = float(row["harm_per_frame"])
                action_f[index, 3] = float(row["integrity_threshold"])
            elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
                action_i[index, 0] = TYPE_MODIFY_GAS
                action_i[index, 1] = int(row["gas_species_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
                action_f[index, 0] = float(row["speed"]) * 0.1
                action_f[index, 1] = float(row["strength"])
                action_f[index, 2] = float(row["range_cells"])
                action_f[index, 3] = float(row["speed"])
            elif (
                reaction_type_id == int(ReactionType.EMIT_LIGHT.value)
                and int(row["light_type_id"]) >= 0
            ):
                action_i[index, 0] = TYPE_EMIT_LIGHT
                action_i[index, 1] = int(row["light_type_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["strength"])
                action_f[index, 1] = float(row["range_cells"])
                action_f[index, 2] = float(row["beam_width"])
            elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
                action_i[index, 0] = TYPE_EMIT_MATERIAL
                action_i[index, 1] = int(row["emit_material_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["velocity"][0])
                action_f[index, 1] = float(row["velocity"][1])
                action_f[index, 2] = float(row["speed"])
            elif reaction_type_id == int(ReactionType.NONE.value):
                action_i[index, 0] = TYPE_NONE
            else:
                action_i[index, 0] = TYPE_DEFERRED
        return action_i, action_f

    def _compile_action_buffers_cached(
        self,
        world: "WorldEngine",
        action_table: np.ndarray,
        used_indices: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        used_key = None if used_indices is None else tuple(sorted(int(index) for index in used_indices))
        key = (
            int(world.bridge.table_generations.get("reactions", 0)),
            int(action_table.shape[0]),
            used_key,
        )
        if key not in self._compiled_action_cache:
            if len(self._compiled_action_cache) > 64:
                self._compiled_action_cache.clear()
            self._compiled_action_cache[key] = self._compile_action_buffers(action_table, used_indices)
        return self._compiled_action_cache[key]

    def _compiled_actions_include_modify_gas(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.any(compiled_actions[0][:, 0] == TYPE_MODIFY_GAS))

    def _compiled_actions_include_flow_sources(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        action_i = np.asarray(compiled_actions[0], dtype=np.int32)
        return bool(np.any((action_i[:, 0] == TYPE_MODIFY_GAS) & (action_i[:, 3] != 0)))

    @staticmethod
    def _compiled_modify_gas_layer_mask(compiled_actions: tuple[np.ndarray, np.ndarray], gas_count: int) -> int:
        action_i = np.asarray(compiled_actions[0], dtype=np.int32)
        mask = 0
        for raw_layer in action_i[action_i[:, 0] == TYPE_MODIFY_GAS, 1].tolist():
            layer = int(raw_layer)
            if layer < 0:
                continue
            if layer >= int(gas_count):
                continue
            if layer >= 31:
                return (1 << min(31, int(gas_count))) - 1
            mask |= 1 << layer
        return mask

    def _compiled_actions_include_emit_material(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.any(compiled_actions[0][:, 0] == TYPE_EMIT_MATERIAL))

    def _compiled_actions_may_change_structure(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        action_types = np.asarray(compiled_actions[0][:, 0], dtype=np.int32)
        return bool(
            np.any(
                (action_types == TYPE_HARM)
                | (action_types == TYPE_CONVERT_MATERIAL)
                | (action_types == TYPE_EMIT_MATERIAL)
            )
        )

    def _compiled_rules_include_rhs_consume(self, rule_tags: np.ndarray) -> bool:
        consume_policies = np.asarray(rule_tags[:, 3], dtype=np.uint32)
        return bool(np.any((consume_policies == CONSUME_POLICY_RHS) | (consume_policies == CONSUME_POLICY_BOTH)))

    def _compile_gas_action_buffers(
        self,
        action_table: np.ndarray,
        used_indices: set[int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
        action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
        action_count = int(action_table.shape[0])
        if action_count > MAX_ACTIONS:
            return None
        for index in range(action_count):
            row = action_table[index]
            if index not in used_indices:
                action_i[index, 0] = TYPE_NONE
                continue
            reaction_type_id = int(row["reaction_type_id"])
            if reaction_type_id == int(ReactionType.NONE.value):
                action_i[index, 0] = TYPE_NONE
            elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
                action_i[index, 0] = TYPE_MODIFY_GAS
                action_i[index, 1] = int(row["gas_species_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
                action_f[index, 0] = float(row["speed"]) * 0.1
                action_f[index, 1] = float(row["strength"])
                action_f[index, 2] = float(row["range_cells"])
                action_f[index, 3] = float(row["speed"])
            elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
                action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
                action_f[index, 0] = float(row["delta"])
            elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value) and int(row["light_type_id"]) >= 0:
                action_i[index, 0] = TYPE_EMIT_LIGHT
                action_i[index, 1] = int(row["light_type_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["strength"])
                action_f[index, 1] = float(row["range_cells"])
                action_f[index, 2] = float(row["beam_width"])
            elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
                action_i[index, 0] = TYPE_EMIT_MATERIAL
                action_i[index, 1] = int(row["emit_material_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["velocity"][0])
                action_f[index, 1] = float(row["velocity"][1])
                action_f[index, 2] = float(row["speed"])
            else:
                return None
        return action_i, action_f

    def _compile_gas_light_action_buffers(
        self,
        action_table: np.ndarray,
        used_indices: set[int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
        action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
        action_count = int(action_table.shape[0])
        if action_count > MAX_ACTIONS:
            return None
        for index in range(action_count):
            row = action_table[index]
            if index not in used_indices:
                action_i[index, 0] = TYPE_NONE
                continue
            reaction_type_id = int(row["reaction_type_id"])
            if reaction_type_id == int(ReactionType.NONE.value):
                action_i[index, 0] = TYPE_NONE
            elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
                action_i[index, 0] = TYPE_MODIFY_GAS
                action_i[index, 1] = int(row["gas_species_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
                action_f[index, 0] = float(row["speed"]) * 0.1
                action_f[index, 1] = float(row["strength"])
                action_f[index, 2] = float(row["range_cells"])
                action_f[index, 3] = float(row["speed"])
            elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
                action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
                action_f[index, 0] = float(row["delta"])
            elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value) and int(row["light_type_id"]) >= 0:
                action_i[index, 0] = TYPE_EMIT_LIGHT
                action_i[index, 1] = int(row["light_type_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["strength"])
                action_f[index, 1] = float(row["range_cells"])
                action_f[index, 2] = float(row["beam_width"])
            elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
                action_i[index, 0] = TYPE_EMIT_MATERIAL
                action_i[index, 1] = int(row["emit_material_id"])
                action_i[index, 2] = int(row["direction_id"])
                action_f[index, 0] = float(row["velocity"][0])
                action_f[index, 1] = float(row["velocity"][1])
                action_f[index, 2] = float(row["speed"])
            else:
                return None
        return action_i, action_f

    @staticmethod
    def _modify_gas_action_requires_cpu_flow_side_effect(row: np.void) -> bool:
        strength = float(row["strength"])
        radius = int(row["range_cells"])
        if strength <= 0.0 or radius <= 0:
            return False
        velocity = np.asarray(row["velocity"], dtype=np.float32)
        if float(np.hypot(float(velocity[0]), float(velocity[1]))) > 1.0e-6:
            return True
        direction_id = int(row["direction_id"])
        if direction_id != 0:
            return True
        return abs(float(row["speed"])) > 1.0e-6

    @staticmethod
    def _rule_candidate_word_count(rule_count: int) -> int:
        return min(RULE_CANDIDATE_WORDS, max(0, (int(rule_count) + 31) // 32))

    @staticmethod
    def _empty_rule_candidate_masks() -> np.ndarray:
        return np.zeros((MAX_MATERIALS, RULE_CANDIDATE_VECS, 4), dtype=np.uint32)

    @staticmethod
    def _set_rule_candidate(mask_table: np.ndarray, material_id: int, rule_index: int) -> None:
        if material_id <= 0 or material_id >= MAX_MATERIALS or rule_index < 0 or rule_index >= MAX_RULES:
            return
        word_index = rule_index // 32
        mask_table[material_id, word_index // 4, word_index % 4] |= np.uint32(1 << (rule_index % 32))

    def _compile_material_rule_candidate_masks(
        self,
        rule_table: np.ndarray,
        material_table: np.ndarray,
        *,
        selector_id_field: str,
        selector_tag_field: str,
        material_tag_field: str,
    ) -> np.ndarray:
        masks = self._empty_rule_candidate_masks()
        count = min(MAX_RULES, int(rule_table.shape[0]))
        material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
        rule_field_names = rule_table.dtype.names or ()
        material_field_names = material_table.dtype.names or ()
        for rule_index, rule in enumerate(rule_table[:count]):
            selector_id = int(rule[selector_id_field]) if selector_id_field in rule_field_names else -1
            if selector_id > 0:
                self._set_rule_candidate(masks, selector_id, rule_index)
                continue
            selector_tag_mask = int(rule[selector_tag_field]) if selector_tag_field in rule_field_names else 0
            if selector_tag_mask != 0 and material_tag_field in material_field_names:
                tag_values = np.asarray(material_table[material_tag_field], dtype=np.uint32)
                required = np.uint32(selector_tag_mask)
                for material_id in range(1, material_count):
                    if (tag_values[material_id] & required) == required:
                        self._set_rule_candidate(masks, material_id, rule_index)
                continue
            for material_id in range(1, MAX_MATERIALS):
                self._set_rule_candidate(masks, material_id, rule_index)
        return masks

    def _compile_material_material_rules(self, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
        rule_i[:, 3] = -1
        rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
        rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
        count = min(MAX_RULES, int(rule_table.shape[0]))
        lhs_ids = rule_table[:count]["lhs_material_id"]
        rhs_ids = rule_table[:count]["rhs_material_id"]
        rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
        rule_i[:count, 1] = np.where(rhs_ids > 0, rhs_ids, -1)
        rule_i[:count, 2] = rule_table[:count]["result_action"]
        rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
        rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
        rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
        rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
        rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
        rule_f[:count, 0] = rule_table[:count]["min_temperature"]
        rule_f[:count, 1] = rule_table[:count]["max_temperature"]
        rule_f[:count, 2] = rule_table[:count]["threshold"]
        rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
        return rule_i, rule_f, rule_tags

    def _compile_material_gas_rules(self, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
        rule_i[:, 3] = -1
        rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
        rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
        count = min(MAX_RULES, int(rule_table.shape[0]))
        lhs_ids = rule_table[:count]["lhs_material_id"]
        rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
        rule_i[:count, 1] = rule_table[:count]["rhs_gas_id"]
        rule_i[:count, 2] = rule_table[:count]["result_action"]
        rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
        rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
        rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
        rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
        rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
        rule_f[:count, 0] = rule_table[:count]["min_temperature"]
        rule_f[:count, 1] = rule_table[:count]["max_temperature"]
        rule_f[:count, 2] = rule_table[:count]["threshold"]
        rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
        return rule_i, rule_f, rule_tags

    def _compile_material_light_rules(
        self,
        rule_table: np.ndarray,
        light_table: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
        rule_i[:, 3] = -1
        rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
        rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
        count = min(MAX_RULES, int(rule_table.shape[0]))
        lhs_ids = rule_table[:count]["lhs_material_id"]
        rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
        rhs_light_ids = rule_table[:count]["rhs_light_id"].astype(np.int32)
        dose_channels = np.full((count,), -1, dtype=np.int32)
        valid = (rhs_light_ids >= 0) & (rhs_light_ids < int(light_table.shape[0]))
        dose_channels[valid] = light_table[rhs_light_ids[valid]]["dose_channel_id"].astype(np.int32)
        rule_i[:count, 1] = dose_channels
        rule_i[:count, 2] = rule_table[:count]["result_action"]
        rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
        rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
        rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
        rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
        rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
        rule_f[:count, 0] = rule_table[:count]["min_temperature"]
        rule_f[:count, 1] = rule_table[:count]["max_temperature"]
        rule_f[:count, 2] = rule_table[:count]["threshold"]
        rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
        return rule_i, rule_f, rule_tags

    def _compile_gas_gas_rules(self, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
        rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
        rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
        count = min(MAX_RULES, int(rule_table.shape[0]))
        rule_i[:count, 0] = rule_table[:count]["lhs_gas_id"]
        rule_i[:count, 1] = rule_table[:count]["rhs_gas_id"]
        rule_i[:count, 2] = rule_table[:count]["result_action"]
        rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
        rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
        rule_tags[:count, 2] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
        rule_f[:count, 0] = rule_table[:count]["min_temperature"]
        rule_f[:count, 1] = rule_table[:count]["max_temperature"]
        rule_f[:count, 2] = rule_table[:count]["threshold"]
        rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
        return rule_i, rule_f, rule_tags

    def _compile_single_gas_gas_rule(self, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._compile_gas_gas_rules(rule_table[:1])

    def _compile_gas_light_rules(
        self,
        rule_table: np.ndarray,
        light_table: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
        rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
        rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
        count = min(MAX_RULES, int(rule_table.shape[0]))
        rhs_gas_ids = rule_table[:count]["rhs_gas_id"].astype(np.int32)
        rule_i[:count, 0] = np.where(rhs_gas_ids >= 0, rhs_gas_ids, -1)
        rhs_light_ids = rule_table[:count]["rhs_light_id"].astype(np.int32)
        dose_channels = np.full((count,), -1, dtype=np.int32)
        valid = (rhs_light_ids >= 0) & (rhs_light_ids < int(light_table.shape[0]))
        dose_channels[valid] = light_table[rhs_light_ids[valid]]["dose_channel_id"].astype(np.int32)
        rule_i[:count, 1] = dose_channels
        rule_i[:count, 2] = rule_table[:count]["result_action"]
        rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
        rule_tags[:count, 2] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
        rule_f[:count, 0] = rule_table[:count]["min_temperature"]
        rule_f[:count, 1] = rule_table[:count]["max_temperature"]
        rule_f[:count, 2] = rule_table[:count]["threshold"]
        rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
        return rule_i, rule_f, rule_tags

    def _compile_single_gas_light_rule(
        self,
        rule_table: np.ndarray,
        light_table: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._compile_gas_light_rules(rule_table[:1], light_table)

    def _used_action_indices(self, rule_table: np.ndarray) -> set[int] | None:
        used_indices: set[int] = set()
        for raw_value in rule_table["result_action"].tolist():
            action_index = int(raw_value)
            if action_index < 0:
                continue
            if action_index >= MAX_ACTIONS:
                return None
            used_indices.add(action_index)
        return used_indices

    def _used_action_indices_for_material_slots(self, material_table: np.ndarray) -> set[int] | None:
        if "reaction_slots" not in material_table.dtype.names:
            return None
        used_indices: set[int] = set()
        for raw_action in np.asarray(material_table["reaction_slots"], dtype=np.int32).reshape(-1):
            action_index = int(raw_action)
            if action_index < 0:
                continue
            if action_index >= MAX_ACTIONS:
                return None
            used_indices.add(action_index)
        return used_indices

    def _cached_used_action_indices_for_material_slots(
        self,
        world: "WorldEngine",
        material_table: np.ndarray,
    ) -> set[int] | None:
        key = (
            "material_slots",
            int(world.bridge.table_generations.get("materials", 0)),
            int(material_table.shape[0]),
        )
        if key not in self._used_action_indices_cache:
            if len(self._used_action_indices_cache) > 64:
                self._used_action_indices_cache.clear()
            self._used_action_indices_cache[key] = self._used_action_indices_for_material_slots(material_table)
        return self._used_action_indices_cache[key]

    def _used_action_indices_for_self_rules(
        self,
        rule_table: np.ndarray,
        material_table: np.ndarray,
    ) -> set[int] | None:
        if "trigger_slot_index" not in rule_table.dtype.names or "reaction_slots" not in material_table.dtype.names:
            return None
        used_indices: set[int] = set()
        material_count = int(material_table.shape[0])
        for rule in rule_table:
            slot_index = int(rule["trigger_slot_index"])
            if slot_index < 0:
                continue
            if slot_index >= 8:
                return None
            material_id = int(rule["material_id"]) if "material_id" in rule_table.dtype.names else -1
            if material_id > 0:
                if material_id >= material_count:
                    return None
                raw_actions = np.asarray(material_table["reaction_slots"][material_id : material_id + 1, slot_index])
            else:
                raw_actions = np.asarray(material_table["reaction_slots"][:, slot_index])
            for raw_action in np.asarray(raw_actions, dtype=np.int32).reshape(-1):
                action_index = int(raw_action)
                if action_index < 0:
                    continue
                if action_index >= MAX_ACTIONS:
                    return None
                used_indices.add(action_index)
        return used_indices

    def _cached_used_action_indices_for_self_rules(
        self,
        world: "WorldEngine",
        rule_table: np.ndarray,
        material_table: np.ndarray,
    ) -> set[int] | None:
        key = (
            "self_rules",
            int(world.bridge.table_generations.get("reactions", 0)),
            int(world.bridge.table_generations.get("materials", 0)),
            int(rule_table.shape[0]),
            int(material_table.shape[0]),
        )
        if key not in self._used_action_indices_cache:
            if len(self._used_action_indices_cache) > 64:
                self._used_action_indices_cache.clear()
            self._used_action_indices_cache[key] = self._used_action_indices_for_self_rules(rule_table, material_table)
        return self._used_action_indices_cache[key]

    def _used_action_indices_for_pair_rules(
        self,
        rule_table: np.ndarray,
        material_table: np.ndarray,
        *,
        lhs_tag_field: str,
    ) -> set[int] | None:
        used_indices = self._used_action_indices(rule_table)
        if used_indices is None:
            return None
        if "trigger_slot_index" not in rule_table.dtype.names or "reaction_slots" not in material_table.dtype.names:
            return used_indices
        material_count = int(material_table.shape[0])
        for rule in rule_table:
            slot_index = int(rule["trigger_slot_index"])
            if slot_index < 0:
                continue
            if slot_index >= 8:
                return None
            lhs_material_id = int(rule["lhs_material_id"]) if "lhs_material_id" in rule_table.dtype.names else -1
            lhs_tag_mask = int(rule["lhs_tag_mask"]) if "lhs_tag_mask" in rule_table.dtype.names else 0
            if lhs_material_id > 0:
                if lhs_material_id >= material_count:
                    return None
                candidates = material_table[lhs_material_id : lhs_material_id + 1]
            elif lhs_tag_mask != 0:
                if lhs_tag_field not in material_table.dtype.names:
                    return None
                masks = np.asarray(material_table[lhs_tag_field], dtype=np.uint32)
                candidates = material_table[(masks & np.uint32(lhs_tag_mask)) == np.uint32(lhs_tag_mask)]
            else:
                candidates = material_table
            for raw_action in np.asarray(candidates["reaction_slots"][:, slot_index], dtype=np.int32).reshape(-1):
                action_index = int(raw_action)
                if action_index < 0:
                    continue
                if action_index >= MAX_ACTIONS:
                    return None
                used_indices.add(action_index)
        return used_indices

    def _cached_used_action_indices_for_pair_rules(
        self,
        world: "WorldEngine",
        rule_table: np.ndarray,
        material_table: np.ndarray,
        *,
        rule_kind: str,
        lhs_tag_field: str,
    ) -> set[int] | None:
        key = (
            "pair_rules",
            str(rule_kind),
            str(lhs_tag_field),
            int(world.bridge.table_generations.get("reactions", 0)),
            int(world.bridge.table_generations.get("materials", 0)),
            int(rule_table.shape[0]),
            int(material_table.shape[0]),
        )
        if key not in self._used_action_indices_cache:
            if len(self._used_action_indices_cache) > 64:
                self._used_action_indices_cache.clear()
            self._used_action_indices_cache[key] = self._used_action_indices_for_pair_rules(
                rule_table,
                material_table,
                lhs_tag_field=lhs_tag_field,
            )
        return self._used_action_indices_cache[key]

    @staticmethod
    def _has_unsupported_consume_policies(rule_table: np.ndarray, supported_ids: set[int]) -> bool:
        if "consume_policy_id" not in rule_table.dtype.names:
            return False
        for raw_value in rule_table["consume_policy_id"].tolist():
            if int(raw_value) not in supported_ids:
                return True
        return False

    def _set_uniform_if_present(self, program: Any, name: str, value: Any) -> None:
        try:
            program[name].value = value
        except KeyError:
            return

    def _sync_storage_and_indirect_writes(self, ctx: Any | None) -> None:
        if ctx is None:
            return
        ctx.memory_barrier(
            ctx.SHADER_STORAGE_BARRIER_BIT
            | getattr(ctx, "COMMAND_BARRIER_BIT", 0)
            | ctx.TEXTURE_FETCH_BARRIER_BIT,
        )

    def _sync_compute_writes(self, ctx: Any | None) -> None:
        if ctx is None:
            return
        ctx.memory_barrier(
            ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT,
        )
