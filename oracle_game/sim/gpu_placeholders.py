from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import PLACEHOLDER_DTYPE, RENDER_GROUP_IDS, typed_material_id
from oracle_game.types import EntityPlaceholder, Phase


PASS_LOCAL_SIZE = 8
MAX_MATERIALS = 256


@dataclass(slots=True)
class GPUPlaceholderResources:
    signature: tuple[int, int, int, int]
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
    placeholders: Any
    material_params: Any
    placeholder_capacity: int = 0
    material_params_signature: tuple[int, int] | None = None


class GPUPlaceholderPipeline:
    def __init__(self) -> None:
        self.resources: GPUPlaceholderResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_backend = "idle"
        self.last_cpu_mirror_downloaded = False

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def apply(self, world: "WorldEngine", placeholders: list[EntityPlaceholder]) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU placeholder pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        world.bridge.sync_rule_tables(world)
        placeholder_upload = self._pack_placeholder_upload(world, placeholders)
        self._upload_inputs(world, resources, placeholder_upload)
        self._load_authoritative_bridge_inputs(world, resources)
        program = self.programs["apply_placeholders"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["placeholder_count"].value = int(len(placeholder_upload))
        program["phase_liquid"].value = int(Phase.LIQUID)
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
        resources.placeholders.bind_to_storage_buffer(binding=11)
        resources.material_params.bind_to_storage_buffer(binding=12)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources)
        self.last_backend = "gpu"

    def release(self) -> None:
        if self.resources is None:
            return
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
            self.resources.placeholders,
            self.resources.material_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUPlaceholderResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height)
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        cell_count = world.width * world.height
        self.resources = GPUPlaceholderResources(
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
            ambient=ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True),
            placeholders=ctx.buffer(reserve=max(4, PLACEHOLDER_DTYPE.itemsize), dynamic=True),
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["apply_placeholders"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int placeholder_count;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform int render_group_placeholder;

            struct Placeholder {{
                int entity_id;
                int buffer_x;
                int buffer_y;
                int world_x;
                int world_y;
                int width;
                int height;
                int material_id;
            }};

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
            layout(std430, binding=11) readonly buffer PlaceholderBuffer {{ Placeholder placeholders[]; }};
            layout(std430, binding=12) readonly buffer MaterialParams {{ vec4 material_params[{MAX_MATERIALS}]; }};

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            int material_slot(int mid) {{
                return clamp(mid, 0, {MAX_MATERIALS - 1});
            }}

            bool is_placeholder_material(int mid) {{
                if (mid <= 0) {{
                    return false;
                }}
                return int(material_params[material_slot(mid)].z + 0.5) == render_group_placeholder;
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

            float ambient_for_cell(ivec2 cell) {{
                int gx = clamp(cell.x / max(1, gas_cell_size), 0, gas_grid_size.x - 1);
                int gy = clamp(cell.y / max(1, gas_cell_size), 0, gas_grid_size.y - 1);
                return ambient_temperature[gy * gas_grid_size.x + gx];
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}

                int desired_entity = 0;
                int desired_material = 0;
                for (int i = 0; i < placeholder_count; ++i) {{
                    Placeholder item = placeholders[i];
                    int item_width = max(0, item.width);
                    int item_height = max(0, item.height);
                    if (
                        item.entity_id > 0
                        && item.material_id > 0
                        && is_placeholder_material(item.material_id)
                        && gid.x >= item.buffer_x
                        && gid.y >= item.buffer_y
                        && gid.x < item.buffer_x + item_width
                        && gid.y < item.buffer_y + item_height
                    ) {{
                        desired_entity = item.entity_id;
                        desired_material = item.material_id;
                    }}
                }}

                int index = cell_index(gid);
                int current_material = material_id[index];
                int current_phase = phase[index];
                int current_entity = entity_id[index];
                int current_displaced = displaced_material[index];
                bool current_placeholder = is_placeholder_material(current_material);
                bool keep_current = (
                    desired_entity > 0
                    && current_entity == desired_entity
                    && current_material > 0
                    && current_placeholder
                );

                int next_material = current_material;
                int next_phase = current_phase;
                int next_flags = flags[index];
                ivec4 next_timer = timer_pack[index];
                float next_temperature = temperature[index];
                float next_integrity = integrity[index];
                vec2 next_velocity = velocity[index];
                int next_island = island_id[index];
                int next_entity = current_entity;
                int next_displaced = current_displaced;

                if (current_entity > 0 && !keep_current) {{
                    next_entity = 0;
                    if (current_placeholder) {{
                        if (current_displaced > 0) {{
                            next_material = current_displaced;
                            next_phase = phase_liquid;
                            next_flags = 0;
                            next_timer = ivec4(0);
                            next_integrity = base_integrity_for(current_displaced);
                            next_displaced = 0;
                        }} else {{
                            next_material = 0;
                            next_phase = 0;
                            next_flags = 0;
                            next_timer = ivec4(0);
                            next_temperature = ambient_for_cell(gid);
                            next_integrity = 0.0;
                            next_velocity = vec2(0.0);
                            next_island = 0;
                            next_displaced = 0;
                        }}
                    }}
                }}

                if (keep_current) {{
                    next_entity = desired_entity;
                }} else if (desired_entity > 0 && (next_material == 0 || next_phase == phase_liquid)) {{
                    int displaced = (next_material > 0 && next_phase == phase_liquid) ? next_material : 0;
                    int placeholder_phase = default_phase_for(desired_material);
                    float spawn_temperature = spawn_temperature_for(desired_material);
                    next_material = desired_material;
                    next_phase = placeholder_phase;
                    next_flags = 0;
                    next_timer = ivec4(0);
                    next_integrity = base_integrity_for(desired_material);
                    if (!isnan(spawn_temperature)) {{
                        next_temperature = max(next_temperature, spawn_temperature);
                    }}
                    if (placeholder_phase != phase_falling_island) {{
                        next_island = 0;
                    }}
                    next_entity = desired_entity;
                    next_displaced = displaced;
                }}

                material_id[index] = next_material;
                phase[index] = next_phase;
                flags[index] = next_flags;
                timer_pack[index] = next_timer;
                temperature[index] = next_temperature;
                integrity[index] = next_integrity;
                velocity[index] = next_velocity;
                island_id[index] = next_island;
                entity_id[index] = next_entity;
                displaced_material[index] = next_displaced;
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
        self.programs["load_bridge_ambient"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            layout(binding=0) uniform sampler2D bridge_ambient_tex;
            layout(std430, binding=1) buffer AmbientBuffer {{ float ambient_temperature[]; }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                int index = gid.y * gas_grid_size.x + gid.x;
                ambient_temperature[index] = texelFetch(bridge_ambient_tex, gid, 0).x;
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

    def _pack_placeholder_upload(
        self,
        world: "WorldEngine",
        placeholders: list[EntityPlaceholder],
    ) -> np.ndarray:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        packed = np.zeros((len(placeholders),), dtype=PLACEHOLDER_DTYPE)
        for index, placeholder in enumerate(placeholders):
            if placeholder.world_x is not None and placeholder.world_y is not None:
                world_x = int(placeholder.world_x)
                world_y = int(placeholder.world_y)
            else:
                world_x, world_y = world.paging.buffer_to_world(int(placeholder.x), int(placeholder.y))
            packed[index]["entity_id"] = int(placeholder.entity_id)
            packed[index]["buffer_x"] = int(placeholder.x)
            packed[index]["buffer_y"] = int(placeholder.y)
            packed[index]["world_x"] = int(world_x)
            packed[index]["world_y"] = int(world_y)
            packed[index]["width"] = int(placeholder.width)
            packed[index]["height"] = int(placeholder.height)
            packed[index]["material_id"] = int(typed_material_id(material_table, placeholder.material))
        return packed

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPUPlaceholderResources,
        placeholder_upload: np.ndarray,
    ) -> None:
        cell_count = world.width * world.height
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "placeholder input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
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
        self._write_placeholder_buffer(world, resources, placeholder_upload)
        self._write_material_params(world, resources)

    def _write_placeholder_buffer(
        self,
        world: "WorldEngine",
        resources: GPUPlaceholderResources,
        placeholder_upload: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        nbytes = max(4, int(placeholder_upload.nbytes))
        if resources.placeholders.size < nbytes:
            resources.placeholders.release()
            resources.placeholders = ctx.buffer(reserve=nbytes, dynamic=True)
            resources.placeholder_capacity = nbytes
        else:
            resources.placeholders.orphan(nbytes)
        if placeholder_upload.nbytes > 0:
            resources.placeholders.write(np.ascontiguousarray(placeholder_upload).tobytes())

    def _write_material_params(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        params = np.zeros((MAX_MATERIALS, 4), dtype=np.float32)
        params[:, 3] = np.nan
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        valid_indices = np.nonzero(material_table[:count]["name_hash"] != 0)[0]
        params[valid_indices, 0] = material_table[:count]["default_phase"][valid_indices].astype(np.float32)
        params[valid_indices, 1] = material_table[:count]["base_integrity"][valid_indices].astype(np.float32)
        params[valid_indices, 2] = material_table[:count]["render_group_id"][valid_indices].astype(np.float32)
        params[valid_indices, 3] = material_table[:count]["spawn_temperature"][valid_indices].astype(np.float32)
        resources.material_params.write(params.tobytes())
        resources.material_params_signature = signature

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU placeholder pipeline requires bridge GPU resources for authoritative input state")
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
        if copy_ambient:
            program = self.programs["load_bridge_ambient"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            bridge.textures["ambient_temperature"].use(location=0)
            resources.ambient.bind_to_storage_buffer(binding=1)
            program.run(
                (world.gas_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.gas_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU placeholder pipeline requires bridge GPU resources for authoritative output state")
            return
        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
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
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(ctx, "SHADER_IMAGE_ACCESS_BARRIER_BIT", 0)
            | getattr(ctx, "TEXTURE_FETCH_BARRIER_BIT", 0)
        )

    def _download_outputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
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
