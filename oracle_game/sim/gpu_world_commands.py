from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from oracle_game.gpu import RENDER_GROUP_IDS, moderngl, typed_gas_id, typed_material_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.gpu_placeholders import MAX_MATERIALS, PASS_LOCAL_SIZE
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
        common = f"""
            #version 430
            #define KIND_INJECT_MATERIAL {COMMAND_KIND_IDS["inject_material"]}
            #define KIND_WRITE_MATERIAL_REGION {COMMAND_KIND_IDS["write_material_region"]}
            #define KIND_INJECT_TEMPERATURE {COMMAND_KIND_IDS["inject_temperature"]}
            #define KIND_INJECT_VELOCITY {COMMAND_KIND_IDS["inject_velocity"]}
            #define KIND_INJECT_GAS {COMMAND_KIND_IDS["inject_gas"]}
            #define CARRIER_CELL {CARRIER_IDS["cell"]}
            #define CARRIER_FLOW {CARRIER_IDS["flow"]}
            #define CARRIER_BOTH {CARRIER_IDS["both"]}
            #define MODE_SET {MODE_IDS["set"]}
            #define MODE_ADD {MODE_IDS["add"]}
        """
        self.programs["cell_commands"] = ctx.compute_shader(
            common
            + f"""
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int command_count;
            uniform int phase_falling_island;
            uniform int render_group_placeholder;

            layout(std430, binding=0) buffer MaterialBuffer {{ int material_id[]; }};
            layout(std430, binding=1) buffer PhaseBuffer {{ int phase[]; }};
            layout(std430, binding=2) buffer FlagsBuffer {{ int flags[]; }};
            layout(std430, binding=3) buffer TimerBuffer {{ ivec4 timer_pack[]; }};
            layout(std430, binding=4) buffer TemperatureBuffer {{ float temperature[]; }};
            layout(std430, binding=5) buffer IntegrityBuffer {{ float integrity[]; }};
            layout(std430, binding=6) buffer VelocityBuffer {{ vec2 velocity[]; }};
            layout(std430, binding=7) buffer IslandBuffer {{ int island_id[]; }};
            layout(std430, binding=8) buffer EntityBuffer {{ int entity_id[]; }};
            layout(std430, binding=9) buffer DisplacedBuffer {{ int displaced_material[]; }};
            layout(std430, binding=10) readonly buffer AmbientBuffer {{ float ambient_temperature[]; }};
            layout(std430, binding=11) readonly buffer CommandI0 {{ ivec4 command_i0[]; }};
            layout(std430, binding=12) readonly buffer CommandI1 {{ ivec4 command_i1[]; }};
            layout(std430, binding=13) readonly buffer CommandI2 {{ ivec4 command_i2[]; }};
            layout(std430, binding=14) readonly buffer CommandF {{ vec4 command_f[]; }};
            layout(std430, binding=15) readonly buffer MaterialParams {{ vec4 material_params[{MAX_MATERIALS}]; }};

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            int material_slot(int mid) {{
                return clamp(mid, 0, {MAX_MATERIALS - 1});
            }}

            bool is_placeholder_material(int mid) {{
                return mid > 0 && int(material_params[material_slot(mid)].z + 0.5) == render_group_placeholder;
            }}

            int default_phase_for(int mid) {{
                return int(material_params[material_slot(mid)].x + 0.5);
            }}

            float base_integrity_for(int mid) {{
                return material_params[material_slot(mid)].y;
            }}

            float spawn_temperature_for(int mid) {{
                return material_params[material_slot(mid)].w;
            }}

            bool in_circle(ivec2 cell, ivec2 center, int radius) {{
                ivec2 delta = cell - center;
                return dot(delta, delta) <= radius * radius;
            }}

            void write_material(int index, int target_material) {{
                int previous_material = material_id[index];
                int previous_phase = phase[index];
                int previous_displaced = displaced_material[index];
                int target_phase = default_phase_for(target_material);
                bool target_placeholder = is_placeholder_material(target_material);
                bool previous_placeholder = is_placeholder_material(previous_material);

                material_id[index] = target_material;
                phase[index] = target_phase;
                flags[index] = 0;
                timer_pack[index] = ivec4(0);
                integrity[index] = base_integrity_for(target_material);
                float spawn_temperature = spawn_temperature_for(target_material);
                if (!isnan(spawn_temperature)) {{
                    temperature[index] = max(temperature[index], spawn_temperature);
                }}
                if (target_placeholder) {{
                    if (previous_placeholder) {{
                        displaced_material[index] = previous_displaced;
                    }} else if (previous_material != 0 && previous_phase == {int(Phase.LIQUID)}) {{
                        displaced_material[index] = previous_material;
                    }} else {{
                        displaced_material[index] = 0;
                    }}
                }} else {{
                    entity_id[index] = 0;
                    displaced_material[index] = 0;
                }}
                if (target_phase != phase_falling_island) {{
                    island_id[index] = 0;
                }}
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int index = cell_index(gid);
                for (int i = 0; i < command_count; ++i) {{
                    ivec4 c0 = command_i0[i];
                    ivec4 c1 = command_i1[i];
                    ivec4 c2 = command_i2[i];
                    vec4 cf = command_f[i];
                    if (c0.x == KIND_INJECT_MATERIAL) {{
                        if (c1.z > 0 && in_circle(gid, c0.yz, max(0, c0.w))) {{
                            write_material(index, c1.z);
                        }}
                    }} else if (c0.x == KIND_WRITE_MATERIAL_REGION) {{
                        if (
                            c1.z > 0
                            && gid.x >= c0.y
                            && gid.y >= c0.z
                            && gid.x < c0.y + max(0, c1.x)
                            && gid.y < c0.z + max(0, c1.y)
                        ) {{
                            write_material(index, c1.z);
                        }}
                    }} else if (c0.x == KIND_INJECT_TEMPERATURE) {{
                        if (in_circle(gid, c0.yz, max(0, c0.w))) {{
                            temperature[index] += cf.x;
                        }}
                    }} else if (c0.x == KIND_INJECT_VELOCITY) {{
                        if ((c2.x == CARRIER_CELL || c2.x == CARRIER_BOTH) && in_circle(gid, c0.yz, max(0, c0.w))) {{
                            if (c2.y == MODE_SET) {{
                                velocity[index] = cf.xy;
                            }} else {{
                                velocity[index] += cf.xy;
                            }}
                        }}
                    }}
                }}
            }}
            """
        )
        self.programs["gas_commands"] = ctx.compute_shader(
            common
            + f"""
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int gas_species_count;
            uniform int command_count;

            layout(std430, binding=0) buffer FlowVelocityBuffer {{ vec2 flow_velocity[]; }};
            layout(std430, binding=1) buffer GasConcentrationBuffer {{ float gas_concentration[]; }};
            layout(std430, binding=2) readonly buffer CommandI0 {{ ivec4 command_i0[]; }};
            layout(std430, binding=3) readonly buffer CommandI1 {{ ivec4 command_i1[]; }};
            layout(std430, binding=4) readonly buffer CommandI2 {{ ivec4 command_i2[]; }};
            layout(std430, binding=5) readonly buffer CommandF {{ vec4 command_f[]; }};

            int gas_index(ivec2 cell) {{
                return cell.y * gas_grid_size.x + cell.x;
            }}

            bool in_circle(ivec2 cell, ivec2 center, int radius) {{
                ivec2 delta = cell - center;
                return dot(delta, delta) <= radius * radius;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                int index = gas_index(gid);
                for (int i = 0; i < command_count; ++i) {{
                    ivec4 c0 = command_i0[i];
                    ivec4 c1 = command_i1[i];
                    ivec4 c2 = command_i2[i];
                    vec4 cf = command_f[i];
                    if (c0.x == KIND_INJECT_GAS) {{
                        int species_id = c1.w;
                        if (species_id >= 0 && species_id < gas_species_count && gid.x == c0.y && gid.y == c0.z) {{
                            int gas_offset = species_id * gas_grid_size.x * gas_grid_size.y + index;
                            gas_concentration[gas_offset] = max(0.0, gas_concentration[gas_offset] + cf.x);
                        }}
                    }} else if (c0.x == KIND_INJECT_VELOCITY) {{
                        if ((c2.x == CARRIER_FLOW || c2.x == CARRIER_BOTH) && in_circle(gid, c1.xy, max(0, c1.z))) {{
                            if (c2.y == MODE_SET) {{
                                flow_velocity[index] = cf.xy;
                            }} else {{
                                flow_velocity[index] += cf.xy;
                            }}
                        }}
                    }}
                }}
            }}
            """
        )
        self.programs["load_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool copy_cell_core;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{ uint bridge_cell_core[]; }};
            layout(std430, binding=1) readonly buffer BridgeIslandBuffer {{ int bridge_island_id[]; }};
            layout(std430, binding=2) readonly buffer BridgeEntityBuffer {{ int bridge_entity_id[]; }};
            layout(std430, binding=3) readonly buffer BridgeDisplacedBuffer {{ int bridge_displaced[]; }};
            layout(std430, binding=4) buffer MaterialBuffer {{ int material_id[]; }};
            layout(std430, binding=5) buffer PhaseBuffer {{ int phase[]; }};
            layout(std430, binding=6) buffer FlagsBuffer {{ int flags[]; }};
            layout(std430, binding=7) buffer TimerBuffer {{ ivec4 timer_pack[]; }};
            layout(std430, binding=8) buffer TemperatureBuffer {{ float temperature[]; }};
            layout(std430, binding=9) buffer IntegrityBuffer {{ float integrity[]; }};
            layout(std430, binding=10) buffer VelocityBuffer {{ vec2 velocity[]; }};
            layout(std430, binding=11) buffer IslandBuffer {{ int island_id[]; }};
            layout(std430, binding=12) buffer EntityBuffer {{ int entity_id[]; }};
            layout(std430, binding=13) buffer DisplacedBuffer {{ int displaced_material[]; }};

            ivec4 unpack_timer(uint word) {{
                return ivec4(
                    int(word & 0xFFu),
                    int((word >> 8u) & 0xFFu),
                    int((word >> 16u) & 0xFFu),
                    int((word >> 24u) & 0xFFu)
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                if (copy_cell_core) {{
                    int word_index = cell_index * 5;
                    uint word0 = bridge_cell_core[word_index];
                    material_id[cell_index] = int(word0 & 0xFFFFu);
                    phase[cell_index] = int((word0 >> 16u) & 0xFFu);
                    flags[cell_index] = int((word0 >> 24u) & 0xFFu);
                    velocity[cell_index] = unpackHalf2x16(bridge_cell_core[word_index + 1]);
                    temperature[cell_index] = uintBitsToFloat(bridge_cell_core[word_index + 2]);
                    timer_pack[cell_index] = unpack_timer(bridge_cell_core[word_index + 3]);
                    integrity[cell_index] = float(bridge_cell_core[word_index + 4] & 0xFFFFu);
                }}
                if (copy_island_id) {{
                    island_id[cell_index] = bridge_island_id[cell_index];
                }}
                if (copy_entity_id) {{
                    entity_id[cell_index] = bridge_entity_id[cell_index];
                }}
                if (copy_displaced_material) {{
                    displaced_material[cell_index] = bridge_displaced[cell_index];
                }}
            }}
            """
        )
        self.programs["load_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int species_count;
            uniform bool copy_flow_velocity;
            uniform bool copy_gas_concentration;
            uniform bool copy_ambient_temperature;

            layout(binding=0) uniform sampler2D bridge_flow_velocity_tex;
            layout(std430, binding=1) readonly buffer BridgeGasBuffer {{ float bridge_gas[]; }};
            layout(std430, binding=2) buffer FlowVelocityBuffer {{ vec2 flow_velocity[]; }};
            layout(std430, binding=3) buffer GasConcentrationBuffer {{ float gas_concentration[]; }};
            layout(binding=4) uniform sampler2D bridge_ambient_tex;
            layout(std430, binding=5) buffer AmbientBuffer {{ float ambient_temperature[]; }};

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= species_count) {{
                    return;
                }}
                int cell_index = gid.y * gas_grid_size.x + gid.x;
                if (species == 0 && copy_flow_velocity) {{
                    flow_velocity[cell_index] = texelFetch(bridge_flow_velocity_tex, gid, 0).xy;
                }}
                if (species == 0 && copy_ambient_temperature) {{
                    ambient_temperature[cell_index] = texelFetch(bridge_ambient_tex, gid, 0).x;
                }}
                if (copy_gas_concentration) {{
                    int gas_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                    gas_concentration[gas_index] = max(bridge_gas[gas_index], 0.0);
                }}
            }}
            """
        )
        self.programs["publish_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

            layout(std430, binding=0) readonly buffer MaterialBuffer {{ int material_id[]; }};
            layout(std430, binding=1) readonly buffer PhaseBuffer {{ int phase[]; }};
            layout(std430, binding=2) readonly buffer FlagsBuffer {{ int flags[]; }};
            layout(std430, binding=3) readonly buffer TimerBuffer {{ ivec4 timer_pack[]; }};
            layout(std430, binding=4) readonly buffer TemperatureBuffer {{ float temperature[]; }};
            layout(std430, binding=5) readonly buffer IntegrityBuffer {{ float integrity[]; }};
            layout(std430, binding=6) readonly buffer VelocityBuffer {{ vec2 velocity[]; }};
            layout(std430, binding=7) readonly buffer IslandBuffer {{ int island_id[]; }};
            layout(std430, binding=8) readonly buffer EntityBuffer {{ int entity_id[]; }};
            layout(std430, binding=9) readonly buffer DisplacedBuffer {{ int displaced_material[]; }};
            layout(std430, binding=10) writeonly buffer BridgeCellCoreBuffer {{ uint bridge_cell_core[]; }};
            layout(std430, binding=11) writeonly buffer BridgeIslandBuffer {{ int bridge_island_id[]; }};
            layout(std430, binding=12) writeonly buffer BridgeEntityBuffer {{ int bridge_entity_id[]; }};
            layout(std430, binding=13) writeonly buffer BridgeDisplacedBuffer {{ int bridge_displaced[]; }};
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

            uint pack_timer(ivec4 timer) {{
                uvec4 value = uvec4(clamp(timer, ivec4(0), ivec4(255)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                uint material = uint(clamp(material_id[cell_index], 0, 65535));
                uint phase_value = uint(clamp(phase[cell_index], 0, 255));
                uint flag_value = uint(clamp(flags[cell_index], 0, 255));
                uint integrity_value = uint(clamp(round(integrity[cell_index]), 0.0, 65535.0));
                int word_index = cell_index * 5;
                bridge_cell_core[word_index] = material | (phase_value << 16u) | (flag_value << 24u);
                bridge_cell_core[word_index + 1] = packHalf2x16(velocity[cell_index]);
                bridge_cell_core[word_index + 2] = floatBitsToUint(temperature[cell_index]);
                bridge_cell_core[word_index + 3] = pack_timer(timer_pack[cell_index]);
                bridge_cell_core[word_index + 4] = integrity_value;
                bridge_island_id[cell_index] = island_id[cell_index];
                bridge_entity_id[cell_index] = entity_id[cell_index];
                bridge_displaced[cell_index] = displaced_material[cell_index];
                imageStore(bridge_material_img, gid, vec4(float(material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int species_count;

            layout(std430, binding=0) readonly buffer FlowVelocityBuffer {{ vec2 flow_velocity[]; }};
            layout(std430, binding=1) readonly buffer GasConcentrationBuffer {{ float gas_concentration[]; }};
            layout(rg32f, binding=2) writeonly uniform image2D bridge_flow_velocity_img;
            layout(std430, binding=3) writeonly buffer BridgeGasBuffer {{ float bridge_gas[]; }};

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= species_count) {{
                    return;
                }}
                int cell_index = gid.y * gas_grid_size.x + gid.x;
                if (species == 0) {{
                    imageStore(bridge_flow_velocity_img, gid, vec4(flow_velocity[cell_index], 0.0, 0.0));
                }}
                int gas_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                bridge_gas[gas_index] = max(gas_concentration[gas_index], 0.0);
            }}
            """
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
