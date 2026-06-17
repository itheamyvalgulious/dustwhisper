from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oracle_game.gpu import CONSUME_POLICY_IDS, DIRECTION_IDS, typed_material_id
from oracle_game.types import ForceSource
from oracle_game.types import CellFlag, Phase, ReactionType


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_ACTIONS = 128
MAX_RULES = 256
MAX_SELF_RULES = 256
FLOW_SOURCE_LAYERS = 32
MAX_EMITTED_LIGHTS = 256

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
    trigger_lo_tex: Any
    trigger_hi_tex: Any
    deferred_scale_lo_tex: Any
    deferred_scale_hi_tex: Any
    cell_reset_tex: Any
    reaction_latched_tex: Any
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
        self.random_targets = np.zeros((MAX_MATERIALS,), dtype=np.int32)
        self.random_target_count = 0

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def run_timed_actions(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        compiled = self._compile_action_buffers(world.bridge.shadow_typed_tables["reaction_action_table"])
        if compiled is None:
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
        self._upload_local_metadata(world, resources)
        resources.action_i.write(compiled[0].tobytes())
        resources.action_f.write(compiled[1].tobytes())
        self._run_local_cell_action_pass(
            world,
            resources,
            "timed_apply",
            apply_material_side_effects=self._compiled_actions_include_emit_material(compiled),
            apply_gas_side_effects=self._compiled_actions_include_modify_gas(compiled),
        )
        self._download_cell_state(world, resources)
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
        solve_cell_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        self_rule_count = min(MAX_SELF_RULES, int(world.bridge.shadow_typed_tables["self_rule_table"].shape[0]))
        compiled = self._compile_action_buffers(world.bridge.shadow_typed_tables["reaction_action_table"])
        if compiled is None:
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
        resources.action_i.write(compiled[0].tobytes())
        resources.action_f.write(compiled[1].tobytes())
        self._run_local_cell_action_pass(
            world,
            resources,
            "self_apply",
            self_rule_count=self_rule_count,
            apply_material_side_effects=self._compiled_actions_include_emit_material(compiled),
            apply_gas_side_effects=self._compiled_actions_include_modify_gas(compiled),
        )
        self._download_cell_state(world, resources)
        return self._download_deferred_batch(world, resources)

    def run_material_material(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_material_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        compiled = self._compile_action_buffers(world.bridge.shadow_typed_tables["reaction_action_table"])
        if compiled is None:
            return None
        rule_i, rule_f, rule_tags = self._compile_material_material_rules(rule_table)
        return self._run_cell_pass(world, "material_material", compiled, rule_i, rule_f, rule_tags, rule_count, solve_cell_mask)

    def run_material_gas(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_gas_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        compiled = self._compile_action_buffers(world.bridge.shadow_typed_tables["reaction_action_table"])
        if compiled is None:
            return None
        rule_i, rule_f, rule_tags = self._compile_material_gas_rules(rule_table)
        return self._run_cell_pass(world, "material_gas", compiled, rule_i, rule_f, rule_tags, rule_count, solve_cell_mask)

    def run_material_light(
        self,
        world: "WorldEngine",
        *,
        solve_cell_mask: np.ndarray | None = None,
    ) -> GPUDeferredActionBatch | None:
        if not self.available(world):
            return None
        world.bridge.sync_rule_tables(world)
        rule_table = world.bridge.shadow_typed_tables["material_light_rule_table"]
        rule_count = int(rule_table.shape[0])
        if rule_count <= 0 or rule_count > MAX_RULES:
            return None
        compiled = self._compile_action_buffers(world.bridge.shadow_typed_tables["reaction_action_table"])
        if compiled is None:
            return None
        light_table = world.bridge.shadow_typed_tables["light_table"]
        rule_i, rule_f, rule_tags = self._compile_material_light_rules(rule_table, light_table)
        return self._run_cell_pass(world, "material_light", compiled, rule_i, rule_f, rule_tags, rule_count, solve_cell_mask)

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
        self._upload_state(world, resources)
        self._upload_local_metadata(world, resources)
        self._upload_active_masks(
            world,
            resources,
            np.ones((world.height, world.width), dtype=np.bool_),
            solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
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
            self._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
            self.last_cpu_mirror_downloaded = False
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
        self._ensure_programs(world.bridge.ctx)
        resources = self._ensure_resources(world)
        self._upload_state(world, resources)
        self._upload_local_metadata(world, resources)
        self._upload_active_masks(
            world,
            resources,
            np.ones((world.height, world.width), dtype=np.bool_),
            solve_gas_mask if solve_gas_mask is not None else np.ones((world.gas_height, world.gas_width), dtype=np.bool_),
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
            self._publish_bridge_gas_state(world, resources, gas_texture=final_gas, ambient_texture=final_ambient)
            self.last_cpu_mirror_downloaded = False
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
            self.resources.trigger_lo_tex,
            self.resources.trigger_hi_tex,
            self.resources.deferred_scale_lo_tex,
            self.resources.deferred_scale_hi_tex,
            self.resources.cell_reset_tex,
            self.resources.reaction_latched_tex,
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
        solve_cell_mask: np.ndarray | None,
    ) -> GPUDeferredActionBatch:
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
        resources.action_i.write(compiled_actions[0].tobytes())
        resources.action_f.write(compiled_actions[1].tobytes())
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
        rule_i_buffer.write(rule_i.tobytes())
        rule_f_buffer.write(rule_f.tobytes())
        rule_tags_buffer.write(rule_tags.tobytes())
        program = self.programs[program_name]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", rule_count)
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        resources.material_ping.use(location=0)
        resources.phase_ping.use(location=1)
        resources.temp_ping.use(location=2)
        resources.integrity_ping.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        resources.timer_ping.use(location=6)
        resources.active_cell_tex.use(location=7)
        resources.velocity_ping.use(location=8)
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
        resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.local_material_out.bind_to_image(0, read=False, write=True)
        resources.local_phase_out.bind_to_image(1, read=False, write=True)
        resources.local_temp_out.bind_to_image(2, read=False, write=True)
        resources.local_integrity_out.bind_to_image(3, read=False, write=True)
        resources.local_timer_out.bind_to_image(4, read=False, write=True)
        resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
        resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(world.bridge.ctx)
        self._scatter_local_cell_action_outputs(world, resources, group_x, group_y)
        has_rhs_consume = self._compiled_rules_include_rhs_consume(rule_tags)
        if self._compiled_actions_include_emit_material(compiled_actions):
            self._run_cell_material_side_effect_pass(world, resources)
        if self._compiled_actions_include_modify_gas(compiled_actions) or (program_name == "material_gas" and has_rhs_consume):
            may_have_flow_sources = (
                self._compiled_actions_include_modify_gas(compiled_actions)
                and self._compiled_actions_include_flow_sources(compiled_actions)
            )
            self._run_cell_gas_side_effect_pass(
                world,
                resources,
                apply_action_side_effects=self._compiled_actions_include_modify_gas(compiled_actions),
                material_gas_rule_count=rule_count if program_name == "material_gas" and has_rhs_consume else 0,
                may_have_flow_sources=may_have_flow_sources,
            )
        if program_name == "material_light" and has_rhs_consume:
            self._run_material_light_dose_consume_pass(world, resources, rule_count)
        self._download_cell_state(world, resources)
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
    ) -> None:
        program = self.programs[program_name]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "rule_count", 0)
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "random_target_count", int(self.random_target_count))
        self._set_uniform_if_present(program, "self_rule_count", int(self_rule_count))
        resources.material_ping.use(location=0)
        resources.phase_ping.use(location=1)
        resources.temp_ping.use(location=2)
        resources.integrity_ping.use(location=3)
        resources.gas_ping.use(location=4)
        resources.cell_dose_tex.use(location=5)
        resources.timer_ping.use(location=6)
        resources.active_cell_tex.use(location=7)
        resources.velocity_ping.use(location=8)
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
        resources.local_material_out.bind_to_image(0, read=False, write=True)
        resources.local_phase_out.bind_to_image(1, read=False, write=True)
        resources.local_temp_out.bind_to_image(2, read=False, write=True)
        resources.local_integrity_out.bind_to_image(3, read=False, write=True)
        resources.local_timer_out.bind_to_image(4, read=False, write=True)
        resources.local_deferred_lo_out.bind_to_image(5, read=False, write=True)
        resources.local_deferred_hi_out.bind_to_image(6, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(7, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(world.bridge.ctx)
        self._scatter_local_cell_action_outputs(world, resources, group_x, group_y)
        if apply_material_side_effects:
            self._run_cell_material_side_effect_pass(world, resources)
        if apply_gas_side_effects:
            self._run_cell_gas_side_effect_pass(world, resources)

    def _scatter_local_cell_action_outputs(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        group_x: int,
        group_y: int,
    ) -> None:
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

    def _scatter_local_emit_cell_outputs(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
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
    ) -> None:
        program = self.programs["cell_gas_side_effects"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "gas_count", world.gas_concentration.shape[0])
        self._set_uniform_if_present(program, "apply_action_side_effects", int(apply_action_side_effects))
        self._set_uniform_if_present(program, "material_gas_rule_count", int(material_gas_rule_count))
        resources.gas_ping.use(location=0)
        resources.trigger_lo_tex.use(location=1)
        resources.trigger_hi_tex.use(location=2)
        resources.deferred_scale_lo_tex.use(location=3)
        resources.deferred_scale_hi_tex.use(location=4)
        resources.material_ping.use(location=5)
        resources.phase_ping.use(location=6)
        resources.temp_ping.use(location=7)
        resources.active_cell_tex.use(location=8)
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
        program.run(group_x, group_y, world.gas_concentration.shape[0])
        self._sync_compute_writes(world.bridge.ctx)
        self._download_gas_state(world, resources)
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
        cell_program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            world.cell_optical_dose.shape[0],
        )
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
        gas_program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            world.gas_optical_dose.shape[0],
        )
        self._sync_compute_writes(world.bridge.ctx)
        self._download_dose_state(world, resources)

    def _run_cell_material_side_effect_pass(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        program = self.programs["cell_material_side_effects"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        resources.material_ping.use(location=0)
        resources.velocity_ping.use(location=1)
        resources.trigger_lo_tex.use(location=2)
        resources.trigger_hi_tex.use(location=3)
        resources.deferred_scale_lo_tex.use(location=4)
        resources.deferred_scale_hi_tex.use(location=5)
        resources.temp_ping.use(location=6)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.action_i.bind_to_storage_buffer(binding=1)
        resources.action_f.bind_to_storage_buffer(binding=2)
        resources.light_emitter_count.bind_to_storage_buffer(binding=15)
        resources.material_pong.bind_to_image(0, read=False, write=True)
        resources.phase_pong.bind_to_image(1, read=False, write=True)
        resources.temp_pong.bind_to_image(2, read=False, write=True)
        resources.integrity_pong.bind_to_image(3, read=False, write=True)
        resources.velocity_pong.bind_to_image(4, read=False, write=True)
        resources.timer_pong.bind_to_image(5, read=False, write=True)
        resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(world.bridge.ctx)

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
        self.release()
        light_count = signature[5]
        gas_count = signature[4]
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
            trigger_lo_tex=tex((world.width, world.height), 4),
            trigger_hi_tex=tex((world.width, world.height), 4),
            deferred_scale_lo_tex=tex((world.width, world.height), 4),
            deferred_scale_hi_tex=tex((world.width, world.height), 4),
            cell_reset_tex=tex((world.width, world.height)),
            reaction_latched_tex=tex((world.width, world.height)),
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
            uniform bool copy_cell_core;
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
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y || !copy_cell_core) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                int word_index = cell_index * 5;
                uint word0 = bridge_cell_core[word_index];
                float material = float(word0 & 0xFFFFu);
                float phase = float((word0 >> 16u) & 0xFFu);
                vec2 velocity = unpackHalf2x16(bridge_cell_core[word_index + 1]);
                float temperature = uintBitsToFloat(bridge_cell_core[word_index + 2]);
                vec4 timer = unpack_timer(bridge_cell_core[word_index + 3]);
                float integrity = float(bridge_cell_core[word_index + 4] & 0xFFFFu);
                imageStore(material_ping_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(material_pong_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_ping_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(phase_pong_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(temp_ping_img, gid, vec4(temperature, 0.0, 0.0, 0.0));
                imageStore(temp_pong_img, gid, vec4(temperature, 0.0, 0.0, 0.0));
                imageStore(integrity_ping_img, gid, vec4(integrity, 0.0, 0.0, 0.0));
                imageStore(integrity_pong_img, gid, vec4(integrity, 0.0, 0.0, 0.0));
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
                imageStore(velocity_pong_img, gid, vec4(velocity, 0.0, 0.0));
                imageStore(timer_ping_img, gid, timer);
                imageStore(timer_pong_img, gid, timer);
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
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D temp_tex;
            layout(binding=3) uniform sampler2D integrity_tex;
            layout(binding=4) uniform sampler2D velocity_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D cell_reset_tex;
            layout(binding=7) uniform sampler2D reaction_latched_tex;
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;
            layout(std430, binding=0) buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
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
                if (texelFetch(cell_reset_tex, gid, 0).x > 0.5) {{
                    flags = 0u;
                }}
                if (texelFetch(reaction_latched_tex, gid, 0).x > 0.5) {{
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
            uniform float impulse_dt;
            layout(binding=0) uniform sampler2D flow_velocity_tex;
            layout(binding=1) uniform sampler2DArray flow_source_tex;
            layout(rg32f, binding=2) writeonly uniform image2D bridge_flow_velocity_img;
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
                    }}
                }}
                imageStore(bridge_flow_velocity_img, gid, vec4(velocity, 0.0, 0.0));
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
                if (index < emitter_vec4_count) {{
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
        helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int rule_count;
            uniform int gas_cell_size;
            uniform int gas_count;
            uniform int random_target_count;
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
            layout(std430, binding=3) buffer RuleI {{
                ivec4 rule_i[{MAX_RULES}];
            }};
            layout(std430, binding=4) buffer RuleF {{
                vec4 rule_f[{MAX_RULES}];
            }};
            layout(std430, binding=5) buffer RuleTags {{
                uvec4 rule_tags[{MAX_RULES}];
            }};
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
                    append_deferred(
                        deferred_action_lo,
                        deferred_action_hi,
                        deferred_scale_lo,
                        deferred_scale_hi,
                        deferred_count,
                        action_index,
                        scale
                    );
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
            helper
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
        self.programs["cell_material_side_effects"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(binding=2) uniform sampler2D action_lo_tex;
            layout(binding=3) uniform sampler2D action_hi_tex;
            layout(binding=4) uniform sampler2D scale_lo_tex;
            layout(binding=5) uniform sampler2D scale_hi_tex;
            layout(binding=6) uniform sampler2D temp_tex;
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
                        int action_index = first_emitting_action(source, target, texelFetch(action_lo_tex, source, 0));
                        if (action_index < 0) {{
                            action_index = first_emitting_action(source, target, texelFetch(action_hi_tex, source, 0));
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
            layout(binding=0) uniform sampler2DArray gas_tex;
            layout(binding=1) uniform sampler2D action_lo_tex;
            layout(binding=2) uniform sampler2D action_hi_tex;
            layout(binding=3) uniform sampler2D scale_lo_tex;
            layout(binding=4) uniform sampler2D scale_hi_tex;
            layout(binding=5) uniform sampler2D material_tex;
            layout(binding=6) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D temp_tex;
            layout(binding=8) uniform sampler2D active_cell_tex;
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

            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || gid.z >= gas_count) {{
                    return;
                }}
                float gas_value = texelFetch(gas_tex, gid, 0).x;
                int x0 = gid.x * gas_cell_size;
                int y0 = gid.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                for (int y = y0; y < y1; ++y) {{
                    for (int x = x0; x < x1; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (apply_action_side_effects != 0) {{
                            apply_action_vector(
                                gid.xy,
                                gid.z,
                                texelFetch(action_lo_tex, cell, 0),
                                texelFetch(scale_lo_tex, cell, 0),
                                0,
                                gas_value
                            );
                            apply_action_vector(
                                gid.xy,
                                gid.z,
                                texelFetch(action_hi_tex, cell, 0),
                                texelFetch(scale_hi_tex, cell, 0),
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
            + """
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=3) writeonly uniform image2D integrity_out_img;
            layout(rgba32f, binding=4) writeonly uniform image2D timer_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2DArray deferred_lo_img;
            layout(rgba32f, binding=6) writeonly uniform image2DArray deferred_hi_img;
            layout(rg32f, binding=7) writeonly uniform image2D cell_meta_out_img;
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
                if (solve_active(gid)) {
                    for (int rule_index = 0; rule_index < rule_count; ++rule_index) {
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    int material_id = int(material_value + 0.5);
                    if (material_id <= 0) {
                        continue;
                    }
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
                consume_rhs_material_sources(
                    gid,
                    material_value,
                    phase_value,
                    integrity_value,
                    timer_value,
                    cell_reset_value,
                    reaction_latched_value
                );
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
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    int material_id = int(material_value + 0.5);
                    if (material_id <= 0) {
                        continue;
                    }
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
                for (int rule_index = 0; rule_index < rule_count; ++rule_index) {
                    ivec4 ri = rule_i[rule_index];
                    vec4 rf = rule_f[rule_index];
                    uvec4 rt = rule_tags[rule_index];
                    int material_id = int(material_value + 0.5);
                    if (material_id <= 0) {
                        continue;
                    }
                    if (ri.x >= 0 && material_id != ri.x) {
                        continue;
                    }
                    if (!mask_matches(material_tags[clamp_material_id(material_id)].z, rt.x)) {
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

    def _upload_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "reaction input",
            "cell_core",
            "gas_concentration",
            "ambient_temperature",
            "flow_velocity",
            "cell_optical_dose",
            "gas_optical_dose",
        )
        upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
        upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
        upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
        upload_flow_velocity_from_cpu = not (formal_gpu_frame and "flow_velocity" in authoritative)
        upload_cell_dose_from_cpu = not (formal_gpu_frame and "cell_optical_dose" in authoritative)
        upload_gas_dose_from_cpu = not (formal_gpu_frame and "gas_optical_dose" in authoritative)
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
            self._clear_transient_state(world, resources)
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
            resources.light_emitter_buffer.write(np.zeros((MAX_EMITTED_LIGHTS, 2, 4), dtype="f4").tobytes())
            resources.light_emitter_count.write(np.zeros((16,), dtype=np.uint32).tobytes())
        self._load_authoritative_bridge_inputs(world, resources)
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
        chaos_convert_bit = int(world.tag_bits_by_name.get("chaos_convert", 0))
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

    def _clear_transient_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        cell_program = self.programs["clear_transient_cell_state"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        resources.trigger_lo_tex.bind_to_image(0, read=False, write=True)
        resources.trigger_hi_tex.bind_to_image(1, read=False, write=True)
        resources.deferred_scale_lo_tex.bind_to_image(2, read=False, write=True)
        resources.deferred_scale_hi_tex.bind_to_image(3, read=False, write=True)
        resources.cell_reset_tex.bind_to_image(4, read=False, write=True)
        resources.reaction_latched_tex.bind_to_image(5, read=False, write=True)
        resources.emitted_material_mask_tex.bind_to_image(6, read=False, write=True)
        resources.local_emit_cell_lo_out.bind_to_image(7, read=False, write=True)
        cell_program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )

        aux_program = self.programs["clear_transient_aux_state"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        aux_program["flow_source_layers"].value = FLOW_SOURCE_LAYERS
        resources.local_emit_cell_hi_out.bind_to_image(0, read=False, write=True)
        resources.local_timer_out.bind_to_image(1, read=False, write=True)
        resources.local_cell_meta_out.bind_to_image(2, read=False, write=True)
        resources.flow_source_tex.bind_to_image(3, read=False, write=True)
        resources.light_emitter_count.bind_to_storage_buffer(binding=0)
        aux_program.run(
            (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        resources.light_emitter_buffer.write(np.zeros((MAX_EMITTED_LIGHTS, 2, 4), dtype="f4").tobytes())
        self._sync_compute_writes(ctx)

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_gas = "gas_concentration" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        copy_flow_velocity = "flow_velocity" in authoritative
        copy_cell_dose = "cell_optical_dose" in authoritative
        copy_gas_dose = "gas_optical_dose" in authoritative
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
        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = True
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_ping.bind_to_image(0, read=False, write=True)
            resources.material_pong.bind_to_image(1, read=False, write=True)
            resources.phase_ping.bind_to_image(2, read=False, write=True)
            resources.phase_pong.bind_to_image(3, read=False, write=True)
            resources.temp_ping.bind_to_image(4, read=False, write=True)
            resources.temp_pong.bind_to_image(5, read=False, write=True)
            resources.integrity_ping.bind_to_image(6, read=False, write=True)
            resources.integrity_pong.bind_to_image(7, read=False, write=True)
            program.run(
                (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
            program = self.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.velocity_ping.bind_to_image(0, read=False, write=True)
            resources.velocity_pong.bind_to_image(1, read=False, write=True)
            resources.timer_ping.bind_to_image(2, read=False, write=True)
            resources.timer_pong.bind_to_image(3, read=False, write=True)
            program.run(
                (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
        if copy_gas or copy_ambient or copy_flow_velocity:
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
            program.run(
                (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
                int(world.gas_concentration.shape[0]),
            )
        if copy_cell_dose or copy_gas_dose:
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
            program.run(
                (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                int(world.cell_optical_dose.shape[0]),
            )
        self._sync_compute_writes(bridge.ctx)

    def _upload_active_masks(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
    ) -> None:
        active_authoritative = (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        self.last_cpu_active_upload_skipped = bool(active_authoritative)
        if active_authoritative:
            self._load_authoritative_active_masks(world, resources, expansion_radius=1)
            return
        resources.active_cell_tex.write(np.asarray(solve_cell_mask, dtype="f4").tobytes())
        resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())

    def _load_authoritative_active_masks(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction pipeline requires bridge active scheduler resources")
        for name, texture, width, height in (
            ("load_active_cell", resources.active_cell_tex, world.width, world.height),
            ("load_active_gas", resources.active_gas_tex, world.gas_width, world.gas_height),
        ):
            program = self.programs[name]
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
            self._set_uniform_if_present(program, "tile_size", int(world.active.tile_size))
            self._set_uniform_if_present(program, "expansion_radius", int(expansion_radius))
            bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
            texture.bind_to_image(1, read=False, write=True)
            program.run(
                (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
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

    def _publish_bridge_cell_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative cell state")
        if "cell_core" not in bridge.gpu_authoritative_resources:
            world._require_gpu_authoritative_resources("reaction output", "cell_core")
            bridge.sync_world(world)
        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.material_pong.use(location=0)
        resources.phase_pong.use(location=1)
        resources.temp_pong.use(location=2)
        resources.integrity_pong.use(location=3)
        resources.velocity_pong.use(location=4)
        resources.timer_pong.use(location=5)
        resources.cell_reset_tex.use(location=6)
        resources.reaction_latched_tex.use(location=7)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("cell_core", "material")

    def _publish_bridge_gas_state(
        self,
        world: "WorldEngine",
        resources: GPUReactionResources,
        *,
        gas_texture: Any | None = None,
        ambient_texture: Any | None = None,
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
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            int(world.gas_concentration.shape[0]),
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("gas_concentration", "ambient_temperature")

    def _publish_bridge_dose_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
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
        cell_program.run(
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            light_count,
        )
        gas_program = self.programs["publish_bridge_gas_dose"]
        gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        gas_program["light_count"].value = light_count
        resources.gas_dose_pong.use(location=0)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=0)
        gas_program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            light_count,
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")

    def _apply_flow_sources_to_bridge_velocity(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU reaction pipeline requires bridge GPU resources for authoritative flow state")
        program = self.programs["apply_bridge_flow_sources"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["impulse_dt"].value = 1.0 / 60.0
        resources.flow_velocity_tex.use(location=0)
        resources.flow_source_tex.use(location=1)
        bridge.textures["flow_velocity"].bind_to_image(2, read=False, write=True)
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("flow_velocity")
        world._mark_active_rect_runtime(0, 0, world.width, world.height)

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

    def _download_cell_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if self._formal_gpu_frame(world):
            self._publish_bridge_cell_state(world, resources)
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
            self._publish_bridge_gas_state(world, resources)
            self.last_cpu_mirror_downloaded = False
            return
        self.last_cpu_mirror_downloaded = True
        world.gas_concentration[:] = np.maximum(
            np.frombuffer(resources.gas_pong.read(), dtype="f4").reshape(world.gas_concentration.shape),
            0.0,
        )

    def _download_dose_state(self, world: "WorldEngine", resources: GPUReactionResources) -> None:
        if self._formal_gpu_frame(world):
            self._publish_bridge_dose_state(world, resources)
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
    ) -> None:
        if not may_have_flow_sources:
            return
        if self._formal_gpu_frame(world):
            self._apply_flow_sources_to_bridge_velocity(world, resources)
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

    def _compiled_actions_include_modify_gas(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.any(compiled_actions[0][:, 0] == TYPE_MODIFY_GAS))

    def _compiled_actions_include_flow_sources(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        action_i = np.asarray(compiled_actions[0], dtype=np.int32)
        return bool(np.any((action_i[:, 0] == TYPE_MODIFY_GAS) & (action_i[:, 3] != 0)))

    def _compiled_actions_include_emit_material(self, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.any(compiled_actions[0][:, 0] == TYPE_EMIT_MATERIAL))

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

    def _sync_compute_writes(self, ctx: Any | None) -> None:
        if ctx is None:
            return
        ctx.memory_barrier(
            ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT,
        )
