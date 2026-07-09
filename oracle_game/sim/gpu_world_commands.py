from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from oracle_game.gpu import RENDER_GROUP_IDS, moderngl, typed_gas_id, typed_material_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.gpu_placeholders import MAX_MATERIALS, PASS_LOCAL_SIZE
from oracle_game.sim.shader_loader import build_compute_shader, shader_source
from oracle_game.types import Phase, WorldCommand


COMMAND_KIND_IDS = {
    "inject_material": 1,
    "write_material_region": 2,
    "inject_temperature": 3,
    "inject_velocity": 4,
    "inject_gas": 5,
}
CARRIER_IDS = {"cell": 1, "flow": 2, "both": 3}
MODE_IDS = {"set": 1, "add": 2}

# Superset of every {{NAME}} marker referenced by any world_commands
# shader; the loader ignores unused keys, so one shared dict suffices
# for all passes.
_SHADER_SUBS = {
    "KIND_INJECT_MATERIAL": COMMAND_KIND_IDS["inject_material"],
    "KIND_WRITE_MATERIAL_REGION": COMMAND_KIND_IDS["write_material_region"],
    "KIND_INJECT_TEMPERATURE": COMMAND_KIND_IDS["inject_temperature"],
    "KIND_INJECT_VELOCITY": COMMAND_KIND_IDS["inject_velocity"],
    "KIND_INJECT_GAS": COMMAND_KIND_IDS["inject_gas"],
    "CARRIER_CELL": CARRIER_IDS["cell"],
    "CARRIER_FLOW": CARRIER_IDS["flow"],
    "CARRIER_BOTH": CARRIER_IDS["both"],
    "MODE_SET": MODE_IDS["set"],
    "MODE_ADD": MODE_IDS["add"],
    "PASS_LOCAL_SIZE": PASS_LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
    "PHASE_LIQUID": int(Phase.LIQUID),
}


@dataclass(slots=True)
class GPUWorldCommandResources:
    signature: tuple[int, int, int, int, int]
    material: Any
    phase: Any
    flags: Any
    timer: Any
    temp: Any
    integrity: Any
    velocity: Any
    island: Any
    entity: Any
    displaced: Any
    ambient: Any
    flow_velocity: Any
    gas_concentration: Any
    command_i0: Any
    command_i1: Any
    command_i2: Any
    command_f: Any
    material_params: Any
    material_params_signature: tuple[int, int] | None = None


class GPUWorldCommandPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUWorldCommandResources | None = None
        self.programs: dict[str, Any] = {}
        self._thread_contexts: dict[int, Any] = {}
        self._thread_resources: dict[int, GPUWorldCommandResources] = {}
        self._thread_programs: dict[int, dict[str, Any]] = {}
        self._ephemeral_context_keys: set[int] = set()
        self._active_thread_id: int | None = None
        self.last_backend = "idle"
        self.last_cpu_mirror_downloaded = False

    def available(self, world: "WorldEngine") -> bool:
        # Overrides :meth:`GPUPipelineBase.available`: this pipeline can spin
        # up an ephemeral standalone context, so it is also available whenever
        # moderngl itself is importable (the base only checks the live bridge).
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(
            (moderngl is not None)
            or (world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)
        )

    def apply(self, world: "WorldEngine", commands: list[WorldCommand]) -> None:
        if not commands:
            self.last_backend = "idle"
            return
        ctx = self._context_for_current_thread(world)
        active_key = self._active_thread_id
        try:
            if ctx is None or getattr(ctx, "version_code", 0) < 430:
                raise RuntimeError("GPU world command pipeline requires a valid ModernGL context")
            self._activate_thread_resources()
            self._ensure_programs(ctx)
            resources = self._ensure_resources(world)
            command_i0, command_i1, command_i2, command_f = self._pack_commands(world, commands)
            self._upload_inputs(world, resources, command_i0, command_i1, command_i2, command_f)
            self._run_cell_commands(world, resources, command_count=len(commands))
            self._run_gas_commands(world, resources, command_count=len(commands))
            self._publish_bridge_outputs(world, resources)
            self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
            if self.last_cpu_mirror_downloaded:
                ctx.finish()
                self._download_outputs(world, resources)
            self.last_backend = "gpu"
        finally:
            if active_key is not None and active_key in self._ephemeral_context_keys:
                self._release_context_key(active_key)

    def release(self) -> None:
        for resources in list(self._thread_resources.values()):
            for resource in (
                resources.material,
                resources.phase,
                resources.flags,
                resources.timer,
                resources.temp,
                resources.integrity,
                resources.velocity,
                resources.island,
                resources.entity,
                resources.displaced,
                resources.ambient,
                resources.flow_velocity,
                resources.gas_concentration,
                resources.command_i0,
                resources.command_i1,
                resources.command_i2,
                resources.command_f,
                resources.material_params,
            ):
                try:
                    resource.release()
                except Exception:
                    pass
        for programs in list(self._thread_programs.values()):
            for program in programs.values():
                try:
                    program.release()
                except Exception:
                    pass
        for key, ctx in list(self._thread_contexts.items()):
            if key not in self._ephemeral_context_keys:
                continue
            try:
                ctx.release()
            except Exception:
                pass
        self._thread_contexts.clear()
        self._thread_resources.clear()
        self._thread_programs.clear()
        self._ephemeral_context_keys.clear()
        self.resources = None
        self.programs = {}
        self._active_thread_id = None

    def _context_for_current_thread(self, world: "WorldEngine") -> Any | None:
        if world.bridge.ctx is not None and world.bridge.owner_thread_id == threading.get_ident():
            thread_id = threading.get_ident()
            self._active_thread_id = thread_id
            self._thread_contexts[thread_id] = world.bridge.ctx
            return world.bridge.ctx
        if getattr(world, "simulation_backend", "") == "gpu":
            return None
        if moderngl is None:
            thread_id = threading.get_ident()
            self._active_thread_id = thread_id
            ctx = world.bridge.ctx
            if ctx is not None:
                self._thread_contexts[thread_id] = ctx
            return ctx
        errors: list[Exception] = []
        for kwargs in ({"require": 430, "backend": "egl"}, {"require": 430}):
            try:
                ctx = moderngl.create_standalone_context(**kwargs)
                context_key = id(ctx)
                self._active_thread_id = context_key
                self._thread_contexts[context_key] = ctx
                self._ephemeral_context_keys.add(context_key)
                return ctx
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        return None

    def _activate_thread_resources(self) -> None:
        thread_id = self._active_thread_id
        if thread_id is None:
            thread_id = threading.get_ident()
            self._active_thread_id = thread_id
        self.resources = self._thread_resources.get(thread_id)
        self.programs = self._thread_programs.setdefault(thread_id, {})

    def _release_context_key(self, key: int) -> None:
        resources = self._thread_resources.pop(key, None)
        if resources is not None:
            for resource in (
                resources.material,
                resources.phase,
                resources.flags,
                resources.timer,
                resources.temp,
                resources.integrity,
                resources.velocity,
                resources.island,
                resources.entity,
                resources.displaced,
                resources.ambient,
                resources.flow_velocity,
                resources.gas_concentration,
                resources.command_i0,
                resources.command_i1,
                resources.command_i2,
                resources.command_f,
                resources.material_params,
            ):
                try:
                    resource.release()
                except Exception:
                    pass
        programs = self._thread_programs.pop(key, None)
        if programs is not None:
            for program in programs.values():
                try:
                    program.release()
                except Exception:
                    pass
        ctx = self._thread_contexts.pop(key, None)
        if ctx is not None:
            try:
                ctx.release()
            except Exception:
                pass
        self._ephemeral_context_keys.discard(key)
        if self._active_thread_id == key:
            self.resources = None
            self.programs = {}
            self._active_thread_id = None

    def _active_context(self) -> Any:
        thread_id = self._active_thread_id
        if thread_id is None:
            thread_id = threading.get_ident()
            self._active_thread_id = thread_id
        ctx = self._thread_contexts.get(thread_id)
        if ctx is None:
            raise RuntimeError("GPU world command pipeline context is not active")
        return ctx

    def _ensure_resources(self, world: "WorldEngine") -> GPUWorldCommandResources:
        ctx = self._active_context()
        signature = (
            world.width,
            world.height,
            world.gas_width,
            world.gas_height,
            int(world.gas_concentration.shape[0]),
        )
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        if self.resources is not None:
            for resource in (
                self.resources.material,
                self.resources.phase,
                self.resources.flags,
                self.resources.timer,
                self.resources.temp,
                self.resources.integrity,
                self.resources.velocity,
                self.resources.island,
                self.resources.entity,
                self.resources.displaced,
                self.resources.ambient,
                self.resources.flow_velocity,
                self.resources.gas_concentration,
                self.resources.command_i0,
                self.resources.command_i1,
                self.resources.command_i2,
                self.resources.command_f,
                self.resources.material_params,
            ):
                try:
                    resource.release()
                except Exception:
                    pass
        cell_count = world.width * world.height
        gas_count = world.gas_width * world.gas_height
        species_count = int(world.gas_concentration.shape[0])
        self.resources = GPUWorldCommandResources(
            signature=signature,
            material=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            phase=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            flags=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            timer=ctx.buffer(reserve=max(4, cell_count * 4 * 4), dynamic=True),
            temp=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            integrity=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            velocity=ctx.buffer(reserve=max(4, cell_count * 2 * 4), dynamic=True),
            island=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            entity=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            displaced=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            ambient=ctx.buffer(reserve=max(4, gas_count * 4), dynamic=True),
            flow_velocity=ctx.buffer(reserve=max(4, gas_count * 2 * 4), dynamic=True),
            gas_concentration=ctx.buffer(reserve=max(4, species_count * gas_count * 4), dynamic=True),
            command_i0=ctx.buffer(reserve=4 * 4, dynamic=True),
            command_i1=ctx.buffer(reserve=4 * 4, dynamic=True),
            command_i2=ctx.buffer(reserve=4 * 4, dynamic=True),
            command_f=ctx.buffer(reserve=4 * 4, dynamic=True),
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        )
        assert self._active_thread_id is not None
        self._thread_resources[self._active_thread_id] = self.resources
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["cell_commands"] = build_compute_shader(
            ctx, "world_commands/cell_commands.comp", _SHADER_SUBS,
            includes=["world_commands/_common.comp"],
        )
        self.programs["gas_commands"] = build_compute_shader(
            ctx, "world_commands/gas_commands.comp", _SHADER_SUBS,
            includes=["world_commands/_common.comp"],
        )
        self.programs["load_bridge_cell"] = build_compute_shader(
            ctx, "world_commands/load_bridge_cell.comp", _SHADER_SUBS,
        )
        self.programs["load_bridge_gas"] = build_compute_shader(
            ctx, "world_commands/load_bridge_gas.comp", _SHADER_SUBS,
        )
        self.programs["publish_bridge_cell"] = build_compute_shader(
            ctx, "world_commands/publish_bridge_cell.comp", _SHADER_SUBS,
        )
        self.programs["publish_bridge_gas"] = build_compute_shader(
            ctx, "world_commands/publish_bridge_gas.comp", _SHADER_SUBS,
        )

    def _pack_commands(self, world: "WorldEngine", commands: list[WorldCommand]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        command_i0 = np.zeros((len(commands), 4), dtype=np.int32)
        command_i1 = np.zeros((len(commands), 4), dtype=np.int32)
        command_i2 = np.zeros((len(commands), 4), dtype=np.int32)
        command_f = np.zeros((len(commands), 4), dtype=np.float32)
        for index, command in enumerate(commands):
            kind_id = COMMAND_KIND_IDS[command.kind]
            x, y = world._queued_command_xy(command)
            command_i0[index, 0] = kind_id
            if command.kind == "inject_gas":
                gy, gx = world.cell_xy_to_gas(x, y)
                command_i0[index, 1] = int(gx)
                command_i0[index, 2] = int(gy)
            else:
                command_i0[index, 1] = int(x)
                command_i0[index, 2] = int(y)
            command_i0[index, 3] = int(command.payload.get("radius", 0))

            if command.kind == "inject_material":
                material_id = self._typed_material_alias_id(world, material_table, str(command.payload["material"]))
                if material_id <= 0:
                    raise KeyError(command.payload["material"])
                command_i1[index, 2] = material_id
            elif command.kind == "write_material_region":
                material_id = self._typed_material_alias_id(world, material_table, str(command.payload["material"]))
                if material_id <= 0:
                    raise KeyError(command.payload["material"])
                command_i0[index, 1] = int(x)
                command_i0[index, 2] = int(y)
                command_i1[index, 0] = int(command.payload["width"])
                command_i1[index, 1] = int(command.payload["height"])
                command_i1[index, 2] = material_id
            elif command.kind == "inject_temperature":
                command_f[index, 0] = float(command.payload["delta"])
            elif command.kind == "inject_velocity":
                carrier = str(command.payload.get("carrier", "cell"))
                mode = str(command.payload.get("mode", "add"))
                if carrier not in CARRIER_IDS:
                    raise ValueError(f"unsupported velocity carrier: {carrier}")
                if mode not in MODE_IDS:
                    raise ValueError(f"unsupported velocity mode: {mode}")
                if carrier in {"flow", "both"}:
                    gas_center_x = min(world.gas_width - 1, max(0, int(x) // world.gas_cell_size))
                    gas_center_y = min(world.gas_height - 1, max(0, int(y) // world.gas_cell_size))
                    gas_radius = max(0, (int(command.payload.get("radius", 0)) + world.gas_cell_size - 1) // world.gas_cell_size)
                    command_i1[index, 0] = int(gas_center_x)
                    command_i1[index, 1] = int(gas_center_y)
                    command_i1[index, 2] = int(gas_radius)
                command_i2[index, 0] = CARRIER_IDS[carrier]
                command_i2[index, 1] = MODE_IDS[mode]
                velocity = command.payload["velocity"]
                command_f[index, 0] = float(velocity[0])
                command_f[index, 1] = float(velocity[1])
            elif command.kind == "inject_gas":
                species_id = int(typed_gas_id(gas_table, str(command.payload["species"])))
                if species_id < 0:
                    raise KeyError(command.payload["species"])
                command_i1[index, 3] = species_id
                command_f[index, 0] = float(command.payload["amount"])
        return command_i0, command_i1, command_i2, command_f

    def _typed_material_alias_id(self, world: "WorldEngine", material_table: np.ndarray, name: str) -> int:
        candidates = [str(name)]
        canonical = world._canonical_material_input_name(name)
        if canonical is not None and canonical not in candidates:
            candidates.append(canonical)
        for candidate in candidates:
            material_id = int(typed_material_id(material_table, candidate))
            if material_id <= 0 or material_id >= int(material_table.shape[0]):
                continue
            if int(material_table[material_id]["name_hash"]) == 0:
                continue
            return material_id
        return 0

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPUWorldCommandResources,
        command_i0: np.ndarray,
        command_i1: np.ndarray,
        command_i2: np.ndarray,
        command_f: np.ndarray,
    ) -> None:
        cell_count = world.width * world.height
        gas_count = world.gas_width * world.gas_height
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "world command input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "flow_velocity",
            "gas_concentration",
        )
        if not formal_gpu_frame:
            resources.material.write(np.ascontiguousarray(world.material_id.astype(np.int32)).tobytes())
            resources.phase.write(np.ascontiguousarray(world.phase.astype(np.int32)).tobytes())
            resources.flags.write(np.ascontiguousarray(world.cell_flags.astype(np.int32)).tobytes())
            resources.timer.write(np.ascontiguousarray(world.timer_pack.astype(np.int32).reshape(cell_count, 4)).tobytes())
            resources.temp.write(np.ascontiguousarray(world.cell_temperature.astype(np.float32)).tobytes())
            resources.integrity.write(np.ascontiguousarray(world.integrity.astype(np.float32)).tobytes())
            resources.velocity.write(np.ascontiguousarray(world.velocity.astype(np.float32).reshape(cell_count, 2)).tobytes())
            resources.island.write(np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes())
            resources.entity.write(np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes())
            resources.displaced.write(np.ascontiguousarray(world.placeholder_displaced_material.astype(np.int32)).tobytes())
            resources.ambient.write(np.ascontiguousarray(world.ambient_temperature.astype(np.float32)).tobytes())
            resources.flow_velocity.write(np.ascontiguousarray(world.flow_velocity.astype(np.float32).reshape(gas_count, 2)).tobytes())
            resources.gas_concentration.write(np.ascontiguousarray(world.gas_concentration.astype(np.float32)).tobytes())
        self._write_dynamic_buffer(world, resources, "command_i0", command_i0)
        self._write_dynamic_buffer(world, resources, "command_i1", command_i1)
        self._write_dynamic_buffer(world, resources, "command_i2", command_i2)
        self._write_dynamic_buffer(world, resources, "command_f", command_f)
        self._write_material_params(world, resources)
        self._load_authoritative_bridge_inputs(world, resources)

    def _write_dynamic_buffer(
        self,
        world: "WorldEngine",
        resources: GPUWorldCommandResources,
        name: str,
        data: np.ndarray,
    ) -> None:
        ctx = self._active_context()
        buffer = getattr(resources, name)
        nbytes = max(4, int(data.nbytes))
        if buffer.size < nbytes:
            buffer.release()
            buffer = ctx.buffer(reserve=nbytes, dynamic=True)
            setattr(resources, name, buffer)
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    def _write_material_params(self, world: "WorldEngine", resources: GPUWorldCommandResources) -> None:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        params = np.zeros((MAX_MATERIALS, 4), dtype=np.float32)
        params[:, 3] = np.nan
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        valid_indices = np.nonzero(material_table[:count]["name_hash"] != 0)[0]
        params[valid_indices, 0] = material_table[:count]["default_phase"][valid_indices].astype(np.float32)
        params[valid_indices, 1] = material_table[:count]["base_integrity"][valid_indices].astype(np.float32)
        params[valid_indices, 2] = material_table[:count]["render_group_id"][valid_indices].astype(np.float32)
        params[valid_indices, 3] = material_table[:count]["spawn_temperature"][valid_indices].astype(np.float32)
        resources.material_params.write(params.tobytes())
        resources.material_params_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))

    def _run_cell_commands(self, world: "WorldEngine", resources: GPUWorldCommandResources, *, command_count: int) -> None:
        ctx = self._active_context()
        program = self.programs["cell_commands"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["command_count"].value = int(command_count)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["render_group_placeholder"].value = int(RENDER_GROUP_IDS["placeholder"])
        resources.material.bind_to_storage_buffer(binding=0)
        resources.phase.bind_to_storage_buffer(binding=1)
        resources.flags.bind_to_storage_buffer(binding=2)
        resources.timer.bind_to_storage_buffer(binding=3)
        resources.temp.bind_to_storage_buffer(binding=4)
        resources.integrity.bind_to_storage_buffer(binding=5)
        resources.velocity.bind_to_storage_buffer(binding=6)
        resources.island.bind_to_storage_buffer(binding=7)
        resources.entity.bind_to_storage_buffer(binding=8)
        resources.displaced.bind_to_storage_buffer(binding=9)
        resources.ambient.bind_to_storage_buffer(binding=10)
        resources.command_i0.bind_to_storage_buffer(binding=11)
        resources.command_i1.bind_to_storage_buffer(binding=12)
        resources.command_i2.bind_to_storage_buffer(binding=13)
        resources.command_f.bind_to_storage_buffer(binding=14)
        resources.material_params.bind_to_storage_buffer(binding=15)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _run_gas_commands(self, world: "WorldEngine", resources: GPUWorldCommandResources, *, command_count: int) -> None:
        ctx = self._active_context()
        program = self.programs["gas_commands"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_species_count"].value = int(world.gas_concentration.shape[0])
        program["command_count"].value = int(command_count)
        resources.flow_velocity.bind_to_storage_buffer(binding=0)
        resources.gas_concentration.bind_to_storage_buffer(binding=1)
        resources.command_i0.bind_to_storage_buffer(binding=2)
        resources.command_i1.bind_to_storage_buffer(binding=3)
        resources.command_i2.bind_to_storage_buffer(binding=4)
        resources.command_f.bind_to_storage_buffer(binding=5)
        program.run(
            (world.gas_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.gas_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`
    # (formerly inlined here verbatim).

    def _bridge_context_active(self, world: "WorldEngine") -> bool:
        return self._active_context() is world.bridge.ctx

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPUWorldCommandResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU world command pipeline requires bridge GPU resources for authoritative input state")
        if not self._bridge_context_active(world):
            raise RuntimeError("GPU world command pipeline cannot consume authoritative bridge state from a separate GL context")

        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.material.bind_to_storage_buffer(binding=4)
            resources.phase.bind_to_storage_buffer(binding=5)
            resources.flags.bind_to_storage_buffer(binding=6)
            resources.timer.bind_to_storage_buffer(binding=7)
            resources.temp.bind_to_storage_buffer(binding=8)
            resources.integrity.bind_to_storage_buffer(binding=9)
            resources.velocity.bind_to_storage_buffer(binding=10)
            resources.island.bind_to_storage_buffer(binding=11)
            resources.entity.bind_to_storage_buffer(binding=12)
            resources.displaced.bind_to_storage_buffer(binding=13)
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )

        copy_flow_velocity = "flow_velocity" in authoritative
        copy_gas_concentration = "gas_concentration" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        if copy_flow_velocity or copy_gas_concentration or copy_ambient:
            program = self.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["species_count"].value = int(world.gas_concentration.shape[0])
            program["copy_flow_velocity"].value = bool(copy_flow_velocity)
            program["copy_gas_concentration"].value = bool(copy_gas_concentration)
            program["copy_ambient_temperature"].value = bool(copy_ambient)
            bridge.textures["flow_velocity"].use(location=0)
            bridge.textures["ambient_temperature"].use(location=4)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
            resources.flow_velocity.bind_to_storage_buffer(binding=2)
            resources.gas_concentration.bind_to_storage_buffer(binding=3)
            resources.ambient.bind_to_storage_buffer(binding=5)
            program.run(
                (world.gas_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.gas_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                int(world.gas_concentration.shape[0]),
            )

        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_flow_velocity or copy_gas_concentration or copy_ambient:
            self._sync_compute_writes(self._active_context())

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPUWorldCommandResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU world command pipeline requires bridge GPU resources for authoritative output state")
            return
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU world command pipeline cannot publish authoritative state from a separate GL context")
            return

        cell_program = self.programs["publish_bridge_cell"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        resources.material.bind_to_storage_buffer(binding=0)
        resources.phase.bind_to_storage_buffer(binding=1)
        resources.flags.bind_to_storage_buffer(binding=2)
        resources.timer.bind_to_storage_buffer(binding=3)
        resources.temp.bind_to_storage_buffer(binding=4)
        resources.integrity.bind_to_storage_buffer(binding=5)
        resources.velocity.bind_to_storage_buffer(binding=6)
        resources.island.bind_to_storage_buffer(binding=7)
        resources.entity.bind_to_storage_buffer(binding=8)
        resources.displaced.bind_to_storage_buffer(binding=9)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=10)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=11)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=12)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=13)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        cell_program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )

        gas_program = self.programs["publish_bridge_gas"]
        gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        gas_program["species_count"].value = int(world.gas_concentration.shape[0])
        resources.flow_velocity.bind_to_storage_buffer(binding=0)
        resources.gas_concentration.bind_to_storage_buffer(binding=1)
        bridge.textures["flow_velocity"].bind_to_image(2, read=False, write=True)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=3)
        gas_program.run(
            (world.gas_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.gas_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            int(world.gas_concentration.shape[0]),
        )

        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "flow_velocity",
            "gas_concentration",
        )

    # ``_sync_compute_writes`` / ``_barrier_bits`` are inherited from
    # :class:`GPUPipelineBase`; the local barrier mask matched the base
    # default (SHADER_IMAGE_ACCESS | TEXTURE_FETCH | SHADER_STORAGE).

    def _download_outputs(self, world: "WorldEngine", resources: GPUWorldCommandResources) -> None:
        world.material_id[:] = np.frombuffer(resources.material.read(), dtype=np.int32).reshape((world.height, world.width))
        world.phase[:] = np.frombuffer(resources.phase.read(), dtype=np.int32).reshape((world.height, world.width)).astype(np.uint8)
        world.cell_flags[:] = np.frombuffer(resources.flags.read(), dtype=np.int32).reshape((world.height, world.width)).astype(np.uint8)
        timer = np.frombuffer(resources.timer.read(), dtype=np.int32).reshape((world.height, world.width, 4))
        world.timer_pack[:] = np.clip(timer, 0, 255).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(resources.temp.read(), dtype=np.float32).reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(resources.integrity.read(), dtype=np.float32).reshape((world.height, world.width))
        world.velocity[:] = np.frombuffer(resources.velocity.read(), dtype=np.float32).reshape((world.height, world.width, 2))
        world.island_id[:] = np.frombuffer(resources.island.read(), dtype=np.int32).reshape((world.height, world.width))
        world.entity_id[:] = np.frombuffer(resources.entity.read(), dtype=np.int32).reshape((world.height, world.width))
        world.placeholder_displaced_material[:] = np.frombuffer(resources.displaced.read(), dtype=np.int32).reshape(
            (world.height, world.width)
        )
        world.flow_velocity[:] = np.frombuffer(resources.flow_velocity.read(), dtype=np.float32).reshape(
            (world.gas_height, world.gas_width, 2)
        )
        world.gas_concentration[:] = np.frombuffer(resources.gas_concentration.read(), dtype=np.float32).reshape(
            world.gas_concentration.shape
        )
