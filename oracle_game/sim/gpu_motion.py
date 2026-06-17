from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE, pack_island_runtime_upload
from oracle_game.types import Phase


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_ISLAND_DDA_STEP = 4

POWDER_RESOLVE_BLOCKED = 0
POWDER_RESOLVE_DDA = 1
POWDER_RESOLVE_FALLBACK = 2
POWDER_RESOLVE_STALE = 3
POWDER_SOLVER_SUSPENDED = 2
ISLAND_RESOLVE_BLOCKED = 0
ISLAND_RESOLVE_DIRECT = 1
ISLAND_RESOLVE_RERESOLVED = 2
ISLAND_RESOLVE_STALE = 3
FALLING_ISLAND_BREAK_STABLE = 2


POWDER_RESERVATION_DTYPE = np.dtype(
    [
        ("source_xy", "<i4", (2,)),
        ("desired_target_xy", "<i4", (2,)),
        ("reserved_target_xy", "<i4", (2,)),
        ("resolved_target_xy", "<i4", (2,)),
        ("velocity_xy", "<f4", (2,)),
        ("material_id", "<i4"),
        ("resolve_state", "<i4"),
    ]
)


def powder_reservation_dtype() -> np.dtype:
    return POWDER_RESERVATION_DTYPE


FALLING_ISLAND_RESERVATION_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("buffer_bbox", "<i4", (4,)),
        ("velocity_xy", "<f4", (2,)),
        ("subcell_offset", "<f4", (2,)),
        ("target_shift", "<i4", (2,)),
        ("reserved_shift", "<i4", (2,)),
        ("resolved_shift", "<i4", (2,)),
        ("resolve_state", "<i4"),
    ]
)


def falling_island_reservation_dtype() -> np.dtype:
    return FALLING_ISLAND_RESERVATION_DTYPE


@dataclass(slots=True)
class GPUMotionResources:
    signature: tuple[int, ...]
    material_tex: Any
    material_out_tex: Any
    phase_tex: Any
    phase_out_tex: Any
    cell_flags_tex: Any
    cell_flags_out_tex: Any
    velocity_tex: Any
    velocity_out_tex: Any
    temp_tex: Any
    temp_out_tex: Any
    timer_tex: Any
    timer_out_tex: Any
    integrity_tex: Any
    integrity_out_tex: Any
    flow_tex: Any
    ambient_tex: Any
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    active_tile_tex: Any
    powder_target_tex: Any
    powder_reservations: Any
    powder_reservation_count: Any
    island_reservations: Any
    island_reservation_count: Any
    island_ids: Any
    island_bboxes: Any
    island_motion: Any
    island_shift_results: Any
    component_label_ping: Any
    component_label_pong: Any
    component_labels: Any
    component_island_ids: Any
    component_metadata: Any
    component_change_flag: Any
    material_params: Any
    material_contact_params: Any
    material_falling_params: Any
    material_params_signature: tuple[int, int] | None = None


class GPUMotionPipeline:
    def __init__(self) -> None:
        self.resources: GPUMotionResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_cpu_flow_velocity_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_published_island_runtime_capacity = 0

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def step(self, world: "WorldEngine", *, solve_tile_mask: np.ndarray) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        self._run_powder_targets(world, resources, group_x, group_y)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            return self._download_outputs(world, resources)
        return np.zeros((world.height, world.width, 2), dtype=np.int32)

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.material_tex,
            self.resources.material_out_tex,
            self.resources.phase_tex,
            self.resources.phase_out_tex,
            self.resources.cell_flags_tex,
            self.resources.cell_flags_out_tex,
            self.resources.velocity_tex,
            self.resources.velocity_out_tex,
            self.resources.temp_tex,
            self.resources.temp_out_tex,
            self.resources.timer_tex,
            self.resources.timer_out_tex,
            self.resources.integrity_tex,
            self.resources.integrity_out_tex,
            self.resources.flow_tex,
            self.resources.ambient_tex,
            self.resources.island_id_tex,
            self.resources.island_id_out_tex,
            self.resources.entity_id_tex,
            self.resources.entity_id_out_tex,
            self.resources.displaced_tex,
            self.resources.displaced_out_tex,
            self.resources.active_tile_tex,
            self.resources.powder_target_tex,
            self.resources.powder_reservations,
            self.resources.powder_reservation_count,
            self.resources.island_reservations,
            self.resources.island_reservation_count,
            self.resources.island_ids,
            self.resources.island_bboxes,
            self.resources.island_motion,
            self.resources.island_shift_results,
            self.resources.component_label_ping,
            self.resources.component_label_pong,
            self.resources.component_labels,
            self.resources.component_island_ids,
            self.resources.component_metadata,
            self.resources.component_change_flag,
            self.resources.material_params,
            self.resources.material_contact_params,
            self.resources.material_falling_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUMotionResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (
            world.width,
            world.height,
            world.active.tile_width,
            world.active.tile_height,
            world.gas_width,
            world.gas_height,
            world.gas_cell_size,
        )
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        velocity_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        velocity_out_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        temp_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        temp_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        timer_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        timer_out_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        integrity_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        flow_tex = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        ambient_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        island_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        powder_target_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        component_label_ping = ctx.texture((world.width, world.height), 1, dtype="f4")
        component_label_pong = ctx.texture((world.width, world.height), 1, dtype="f4")
        for texture in (
            material_tex,
            material_out_tex,
            phase_tex,
            phase_out_tex,
            cell_flags_tex,
            cell_flags_out_tex,
            velocity_tex,
            velocity_out_tex,
            temp_tex,
            temp_out_tex,
            timer_tex,
            timer_out_tex,
            integrity_tex,
            integrity_out_tex,
            flow_tex,
            ambient_tex,
            island_id_tex,
            island_id_out_tex,
            entity_id_tex,
            entity_id_out_tex,
            displaced_tex,
            displaced_out_tex,
            active_tile_tex,
            powder_target_tex,
            component_label_ping,
            component_label_pong,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        self.resources = GPUMotionResources(
            signature=signature,
            material_tex=material_tex,
            material_out_tex=material_out_tex,
            phase_tex=phase_tex,
            phase_out_tex=phase_out_tex,
            cell_flags_tex=cell_flags_tex,
            cell_flags_out_tex=cell_flags_out_tex,
            velocity_tex=velocity_tex,
            velocity_out_tex=velocity_out_tex,
            temp_tex=temp_tex,
            temp_out_tex=temp_out_tex,
            timer_tex=timer_tex,
            timer_out_tex=timer_out_tex,
            integrity_tex=integrity_tex,
            integrity_out_tex=integrity_out_tex,
            flow_tex=flow_tex,
            ambient_tex=ambient_tex,
            island_id_tex=island_id_tex,
            island_id_out_tex=island_id_out_tex,
            entity_id_tex=entity_id_tex,
            entity_id_out_tex=entity_id_out_tex,
            displaced_tex=displaced_tex,
            displaced_out_tex=displaced_out_tex,
            active_tile_tex=active_tile_tex,
            powder_target_tex=powder_target_tex,
            powder_reservations=ctx.buffer(reserve=4, dynamic=True),
            powder_reservation_count=ctx.buffer(reserve=4, dynamic=True),
            island_reservations=ctx.buffer(reserve=4, dynamic=True),
            island_reservation_count=ctx.buffer(reserve=4, dynamic=True),
            island_ids=ctx.buffer(reserve=4, dynamic=True),
            island_bboxes=ctx.buffer(reserve=4, dynamic=True),
            island_motion=ctx.buffer(reserve=4, dynamic=True),
            island_shift_results=ctx.buffer(reserve=4, dynamic=True),
            component_label_ping=component_label_ping,
            component_label_pong=component_label_pong,
            component_labels=ctx.buffer(reserve=4, dynamic=True),
            component_island_ids=ctx.buffer(reserve=4, dynamic=True),
            component_metadata=ctx.buffer(reserve=world.width * world.height * 5 * 4, dynamic=True),
            component_change_flag=ctx.buffer(reserve=4, dynamic=True),
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
            material_contact_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
            material_falling_params=ctx.buffer(reserve=MAX_MATERIALS * 2 * 4 * 4, dynamic=True),
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform int expansion_radius;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(r32f, binding=1) writeonly uniform image2D active_tile_img;
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
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= tile_grid_size.x || gid.y >= tile_grid_size.y) {{
                    return;
                }}
                imageStore(active_tile_img, gid, vec4(expanded_tile_active(gid) ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["integrate_velocity"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int gas_cell_size;
            uniform float dt;
            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 params[{MAX_MATERIALS}];
            }};
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D velocity_tex;
            layout(binding=3) uniform sampler2D flow_tex;
            layout(binding=4) uniform sampler2D active_tile_tex;
            layout(rg32f, binding=5) writeonly uniform image2D velocity_out_img;

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (material_id > 0 && solve_cell_active(gid)) {{
                    material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                    vec4 material_params = params[material_id];
                    velocity.y += material_params.y * dt * 24.0;
                    ivec2 gas_cell = ivec2(
                        min(gid.x / gas_cell_size, gas_grid_size.x - 1),
                        min(gid.y / gas_cell_size, gas_grid_size.y - 1)
                    );
                    vec2 flow = texelFetch(flow_tex, gas_cell, 0).xy;
                    velocity += flow * material_params.z * dt * 4.0;
                    velocity *= max(0.0, 1.0 - material_params.w * dt);
                }}
                imageStore(velocity_out_img, gid, vec4(velocity, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool copy_cell_core;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=2) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=3) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_img;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_img;

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
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                if (copy_cell_core) {{
                    int word_index = cell_index * 5;
                    uint word0 = bridge_cell_core[word_index];
                    imageStore(material_img, gid, vec4(float(word0 & 0xFFFFu), 0.0, 0.0, 0.0));
                    imageStore(phase_img, gid, vec4(float((word0 >> 16u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(flags_img, gid, vec4(float((word0 >> 24u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(velocity_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
                    imageStore(temp_img, gid, vec4(uintBitsToFloat(bridge_cell_core[word_index + 2]), 0.0, 0.0, 0.0));
                    imageStore(timer_img, gid, unpack_timer(bridge_cell_core[word_index + 3]));
                    imageStore(integrity_img, gid, vec4(float(bridge_cell_core[word_index + 4] & 0xFFFFu), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["load_bridge_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;
            layout(std430, binding=1) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=2) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=3) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                if (copy_island_id) {{
                    imageStore(island_img, gid, vec4(float(bridge_island_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_entity_id) {{
                    imageStore(entity_img, gid, vec4(float(bridge_entity_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_displaced_material) {{
                    imageStore(displaced_img, gid, vec4(float(bridge_displaced[cell_index]), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["load_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform bool copy_flow_velocity;
            uniform bool copy_ambient;
            layout(binding=0) uniform sampler2D bridge_flow_tex;
            layout(binding=1) uniform sampler2D bridge_ambient_tex;
            layout(rg32f, binding=2) writeonly uniform image2D flow_img;
            layout(r32f, binding=3) writeonly uniform image2D ambient_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                if (copy_flow_velocity) {{
                    imageStore(flow_img, gid, vec4(texelFetch(bridge_flow_tex, gid, 0).xy, 0.0, 0.0));
                }}
                if (copy_ambient) {{
                    imageStore(ambient_img, gid, vec4(texelFetch(bridge_ambient_tex, gid, 0).x, 0.0, 0.0, 0.0));
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
            layout(binding=2) uniform sampler2D flags_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D island_tex;
            layout(binding=8) uniform sampler2D entity_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;
            layout(std430, binding=0) writeonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=2) writeonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=3) writeonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
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
                uint material = uint(clamp(round(texelFetch(material_tex, gid, 0).x), 0.0, 65535.0));
                uint phase = uint(clamp(round(texelFetch(phase_tex, gid, 0).x), 0.0, 255.0));
                uint flags = uint(clamp(round(texelFetch(flags_tex, gid, 0).x), 0.0, 255.0));
                vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                float temperature = texelFetch(temp_tex, gid, 0).x;
                uint integrity = uint(clamp(round(texelFetch(integrity_tex, gid, 0).x), 0.0, 65535.0));
                int island = int(round(texelFetch(island_tex, gid, 0).x));
                int entity = int(round(texelFetch(entity_tex, gid, 0).x));
                int displaced = int(round(texelFetch(displaced_tex, gid, 0).x));
                int word_index = cell_index * 5;
                bridge_cell_core[word_index] = material | (phase << 16u) | (flags << 24u);
                bridge_cell_core[word_index + 1] = packHalf2x16(velocity);
                bridge_cell_core[word_index + 2] = floatBitsToUint(temperature);
                bridge_cell_core[word_index + 3] = pack_timer(texelFetch(timer_tex, gid, 0));
                bridge_cell_core[word_index + 4] = integrity;
                bridge_island_id[cell_index] = island;
                bridge_entity_id[cell_index] = entity;
                bridge_displaced[cell_index] = displaced;
                imageStore(bridge_material_img, gid, vec4(float(material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge_island_id"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D island_tex;
            layout(std430, binding=0) writeonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                bridge_island_id[cell_index] = int(round(texelFetch(island_tex, gid, 0).x));
            }}
            """
        )
        self.programs["copy_scalar_texture"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 grid_size;
            layout(binding=0) uniform sampler2D source_tex;
            layout(r32f, binding=1) writeonly uniform image2D dest_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {{
                    return;
                }}
                imageStore(dest_img, gid, vec4(texelFetch(source_tex, gid, 0).x, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["powder_targets"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;
            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 params[{MAX_MATERIALS}];
            }};
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D active_tile_tex;
            layout(rg32f, binding=5) writeonly uniform image2D powder_target_img;

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            ivec2 dda_target(ivec2 cell, ivec2 desired) {{
                if (desired.x == 0 && desired.y == 0) {{
                    return cell;
                }}
                ivec2 current = ivec2(0, 0);
                ivec2 furthest = cell;
                int dx = abs(desired.x);
                int dy = -abs(desired.y);
                int sx = current.x < desired.x ? 1 : -1;
                int sy = current.y < desired.y ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current.x == desired.x && current.y == desired.y) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current.x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current.y += sy;
                    }}
                    ivec2 sample_cell = cell + current;
                    if (sample_cell.x < 0 || sample_cell.y < 0 || sample_cell.x >= cell_grid_size.x || sample_cell.y >= cell_grid_size.y) {{
                        break;
                    }}
                    if (texelFetch(material_tex, sample_cell, 0).x > 0.5) {{
                        break;
                    }}
                    furthest = sample_cell;
                }}
                return furthest;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 target = gid;
                if (solve_cell_active(gid)) {{
                    int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                    int phase_id = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                    if (material_id > 0 && phase_id == phase_powder) {{
                        int max_step = int(params[material_id].x + 0.5);
                        vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                        int desired_dx = int(clamp(round(velocity.x), float(-max_step), float(max_step)));
                        int desired_dy = int(clamp(round(velocity.y), float(-max_step), float(max_step)));
                        target = dda_target(gid, ivec2(desired_dx, desired_dy));
                    }}
                }}
                imageStore(powder_target_img, gid, vec4(float(target.x), float(target.y), 0.0, 0.0));
            }}
            """
        )
        self.programs["island_component_init"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int target_island_id;
            uniform int phase_falling_island;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D island_id_tex;
            layout(r32f, binding=3) writeonly uniform image2D component_label_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                int phase_id = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                int island_id = int(texelFetch(island_id_tex, gid, 0).x + 0.5);
                float label = 0.0;
                if (material_id > 0 && phase_id == phase_falling_island && island_id == target_island_id) {{
                    label = float(gid.y * cell_grid_size.x + gid.x + 1);
                }}
                imageStore(component_label_img, gid, vec4(label, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["island_component_propagate"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D label_in_tex;
            layout(r32f, binding=1) writeonly uniform image2D label_out_img;
            layout(std430, binding=0) buffer ComponentChangeFlag {{
                uint component_change_flag;
            }};

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float current = texelFetch(label_in_tex, gid, 0).x;
                if (current <= 0.5) {{
                    imageStore(label_out_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }}
                float next_label = current;
                ivec2 offsets[4] = ivec2[4](
                    ivec2(-1, 0),
                    ivec2(1, 0),
                    ivec2(0, -1),
                    ivec2(0, 1)
                );
                for (int index = 0; index < 4; ++index) {{
                    ivec2 sample_cell = gid + offsets[index];
                    if (
                        sample_cell.x < 0
                        || sample_cell.y < 0
                        || sample_cell.x >= cell_grid_size.x
                        || sample_cell.y >= cell_grid_size.y
                    ) {{
                        continue;
                    }}
                    float candidate = texelFetch(label_in_tex, sample_cell, 0).x;
                    if (candidate > 0.5 && candidate < next_label) {{
                        next_label = candidate;
                    }}
                }}
                imageStore(label_out_img, gid, vec4(next_label, 0.0, 0.0, 0.0));
                if (abs(next_label - current) > 0.5) {{
                    atomicOr(component_change_flag, 1u);
                }}
            }}
            """
        )
        self.programs["relabel_falling_island_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int component_count;

            layout(binding=0) uniform sampler2D island_id_tex;
            layout(binding=1) uniform sampler2D component_label_tex;
            layout(r32f, binding=2) writeonly uniform image2D island_id_out_img;

            layout(std430, binding=0) buffer ComponentLabels {{
                int component_labels[];
            }};
            layout(std430, binding=1) buffer ComponentIslandIds {{
                int component_island_ids[];
            }};

            int island_id_for_label(int label) {{
                for (int index = 0; index < component_count; ++index) {{
                    if (component_labels[index] == label) {{
                        return component_island_ids[index];
                    }}
                }}
                return 0;
            }}

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                int current_island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                int next_island_id = label > 0 ? island_id_for_label(label) : current_island_id;
                if (label > 0 && next_island_id <= 0) {{
                    next_island_id = current_island_id;
                }}
                imageStore(island_id_out_img, cell, vec4(float(next_island_id), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["summarize_falling_island_components"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

            layout(binding=0) uniform sampler2D component_label_tex;
            layout(std430, binding=0) buffer ComponentMetadata {{
                int component_metadata[];
            }};

            void main() {{
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                int label = int(texelFetch(component_label_tex, cell, 0).x + 0.5);
                if (label <= 0) {{
                    return;
                }}
                int index = label - 1;
                if (index < 0 || index >= cell_grid_size.x * cell_grid_size.y) {{
                    return;
                }}
                int base = index * 5;
                atomicMin(component_metadata[base + 0], cell.x);
                atomicMin(component_metadata[base + 1], cell.y);
                atomicMax(component_metadata[base + 2], cell.x + 1);
                atomicMax(component_metadata[base + 3], cell.y + 1);
                atomicAdd(component_metadata[base + 4], 1);
            }}
            """
        )
        self.programs["island_shifts"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int island_count;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D island_id_tex;
            layout(std430, binding=0) buffer IslandIdBuffer {{
                int island_ids[];
            }};
            layout(std430, binding=1) buffer IslandBBoxBuffer {{
                ivec4 island_bboxes[];
            }};
            layout(std430, binding=2) buffer IslandMotionBuffer {{
                vec4 island_motion[];
            }};
            layout(std430, binding=3) buffer IslandShiftBuffer {{
                ivec4 island_shifts[];
            }};

            bool can_shift_delta(ivec4 island_bbox, int island_id, int dx, int dy) {{
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        if (int(texelFetch(island_id_tex, ivec2(x, y), 0).x + 0.5) != island_id) {{
                            continue;
                        }}
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= cell_grid_size.x || ny >= cell_grid_size.y) {{
                            return false;
                        }}
                        int material_id = int(texelFetch(material_tex, ivec2(nx, ny), 0).x + 0.5);
                        if (material_id == 0) {{
                            continue;
                        }}
                        if (int(texelFetch(island_id_tex, ivec2(nx, ny), 0).x + 0.5) == island_id) {{
                            continue;
                        }}
                        return false;
                    }}
                }}
                return true;
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                if (gid >= island_count) {{
                    return;
                }}
                int island_id = island_ids[gid];
                ivec4 island_bbox = island_bboxes[gid];
                if (island_id <= 0 || island_bbox.z <= island_bbox.x || island_bbox.w <= island_bbox.y) {{
                    island_motion[gid] = vec4(0.0);
                    island_shifts[gid] = ivec4(0);
                    return;
                }}
                vec4 motion = island_motion[gid];
                float total_x = motion.z + motion.x;
                float total_y = motion.w + motion.y;
                int target_dx = int(clamp(round(total_x), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                int target_dy = int(clamp(round(total_y), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                island_motion[gid] = vec4(motion.x, motion.y, total_x - float(target_dx), total_y - float(target_dy));
                int current_x = 0;
                int current_y = 0;
                int furthest_x = 0;
                int furthest_y = 0;
                if (target_dx == 0 && target_dy == 0) {{
                    island_shifts[gid] = ivec4(0, 0, target_dx, target_dy);
                    return;
                }}
                int dx = abs(target_dx);
                int dy = -abs(target_dy);
                int sx = current_x < target_dx ? 1 : -1;
                int sy = current_y < target_dy ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current_x == target_dx && current_y == target_dy) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current_x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current_y += sy;
                    }}
                    if (!can_shift_delta(island_bbox, island_id, current_x, current_y)) {{
                        break;
                    }}
                    furthest_x = current_x;
                    furthest_y = current_y;
                }}
                island_shifts[gid] = ivec4(furthest_x, furthest_y, target_dx, target_dy);
            }}
            """
        )
        self.programs["pack_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int island_count;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandIds {{
                int island_ids[];
            }};
            layout(std430, binding=1) buffer IslandBboxes {{
                ivec4 island_bboxes[];
            }};
            layout(std430, binding=2) buffer IslandMotion {{
                vec4 island_motion[];
            }};
            layout(std430, binding=3) buffer IslandShifts {{
                ivec4 island_shifts[];
            }};
            layout(std430, binding=4) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};
            layout(std430, binding=5) buffer IslandReservationCount {{
                int reservation_count;
            }};

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                if (gid == 0) {{
                    reservation_count = island_count;
                }}
                if (gid >= island_count) {{
                    return;
                }}
                ivec4 bbox = island_bboxes[gid];
                vec4 motion = island_motion[gid];
                ivec4 shifts = island_shifts[gid];
                reservations[gid].island_id = island_ids[gid];
                reservations[gid].bbox_x = bbox.x;
                reservations[gid].bbox_y = bbox.y;
                reservations[gid].bbox_z = bbox.z;
                reservations[gid].bbox_w = bbox.w;
                reservations[gid].velocity_x = motion.x;
                reservations[gid].velocity_y = motion.y;
                reservations[gid].subcell_x = motion.z;
                reservations[gid].subcell_y = motion.w;
                reservations[gid].target_dx = shifts.z;
                reservations[gid].target_dy = shifts.w;
                reservations[gid].reserved_dx = shifts.x;
                reservations[gid].reserved_dy = shifts.y;
                reservations[gid].resolved_dx = shifts.x;
                reservations[gid].resolved_dy = shifts.y;
                if (shifts.x != 0 || shifts.y != 0 || (shifts.z == 0 && shifts.w == 0)) {{
                    reservations[gid].resolve_state = {ISLAND_RESOLVE_DIRECT};
                }} else {{
                    reservations[gid].resolve_state = {ISLAND_RESOLVE_BLOCKED};
                }}
            }}
            """
        )
        self.programs["publish_falling_island_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int reservation_count;
            uniform ivec2 cell_grid_size;
            uniform ivec2 paging_origin;
            uniform ivec2 paging_buffer_origin;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) readonly buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};
            layout(std430, binding=1) writeonly buffer IslandRuntimeWords {{
                int runtime_words[];
            }};
            layout(std430, binding=2) buffer IslandRuntimeCount {{
                int runtime_count;
            }};

            int positive_mod(int value, int divisor) {{
                int result = value % divisor;
                return result < 0 ? result + divisor : result;
            }}

            bool is_settled(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            void store_runtime_word(int record_index, int word_offset, int value) {{
                runtime_words[record_index * {ISLAND_RUNTIME_DTYPE.itemsize // 4} + word_offset] = value;
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                if (gid >= reservation_count) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[gid];
                if (reservation.island_id <= 0 || reservation.resolve_state == {ISLAND_RESOLVE_STALE} || is_settled(reservation)) {{
                    return;
                }}

                int out_index = atomicAdd(runtime_count, 1);
                int x0 = reservation.bbox_x + reservation.resolved_dx;
                int y0 = reservation.bbox_y + reservation.resolved_dy;
                int x1 = reservation.bbox_z + reservation.resolved_dx;
                int y1 = reservation.bbox_w + reservation.resolved_dy;
                int width = max(0, x1 - x0);
                int height = max(0, y1 - y0);
                int world_x0 = paging_origin.x + positive_mod(x0 - paging_buffer_origin.x, cell_grid_size.x);
                int world_y0 = paging_origin.y + positive_mod(y0 - paging_buffer_origin.y, cell_grid_size.y);
                bool reached_target = (
                    reservation.resolved_dx == reservation.target_dx
                    && reservation.resolved_dy == reservation.target_dy
                );
                float velocity_x = reached_target ? reservation.velocity_x * 0.98 : reservation.velocity_x;
                float velocity_y = reservation.velocity_y;
                float subcell_x = reached_target ? reservation.subcell_x : 0.0;
                float subcell_y = reached_target ? reservation.subcell_y : 0.0;

                store_runtime_word(out_index, 0, reservation.island_id);
                store_runtime_word(out_index, 1, x0);
                store_runtime_word(out_index, 2, y0);
                store_runtime_word(out_index, 3, x1);
                store_runtime_word(out_index, 4, y1);
                store_runtime_word(out_index, 5, world_x0);
                store_runtime_word(out_index, 6, world_y0);
                store_runtime_word(out_index, 7, world_x0 + width);
                store_runtime_word(out_index, 8, world_y0 + height);
                store_runtime_word(out_index, 9, floatBitsToInt(velocity_x));
                store_runtime_word(out_index, 10, floatBitsToInt(velocity_y));
                store_runtime_word(out_index, 11, floatBitsToInt(subcell_x));
                store_runtime_word(out_index, 12, floatBitsToInt(subcell_y));
            }}
            """
        )
        self.programs["publish_powder_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int reservation_capacity;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) readonly buffer SourceReservations {{
                PowderReservation source_reservations[];
            }};
            layout(std430, binding=1) readonly buffer SourceReservationCount {{
                int source_reservation_count;
            }};
            layout(std430, binding=2) writeonly buffer DestReservations {{
                PowderReservation dest_reservations[];
            }};
            layout(std430, binding=3) writeonly buffer DestReservationCount {{
                int dest_reservation_count;
            }};

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                int count = clamp(source_reservation_count, 0, reservation_capacity);
                if (gid == 0) {{
                    dest_reservation_count = count;
                }}
                if (gid >= count) {{
                    return;
                }}
                dest_reservations[gid] = source_reservations[gid];
            }}
            """
        )
        self.programs["unpack_bridge_island_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int runtime_count;

            layout(std430, binding=0) readonly buffer BridgeIslandRuntimeWords {{
                int runtime_words[];
            }};
            layout(std430, binding=1) buffer IslandIds {{
                int island_ids[];
            }};
            layout(std430, binding=2) buffer IslandBboxes {{
                ivec4 island_bboxes[];
            }};
            layout(std430, binding=3) buffer IslandMotion {{
                vec4 island_motion[];
            }};

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                if (gid >= runtime_count) {{
                    return;
                }}
                int base = gid * {ISLAND_RUNTIME_DTYPE.itemsize // 4};
                int island_id = runtime_words[base];
                if (island_id <= 0) {{
                    island_ids[gid] = 0;
                    island_bboxes[gid] = ivec4(0);
                    island_motion[gid] = vec4(0.0);
                    return;
                }}
                island_ids[gid] = island_id;
                island_bboxes[gid] = ivec4(
                    runtime_words[base + 1],
                    runtime_words[base + 2],
                    runtime_words[base + 3],
                    runtime_words[base + 4]
                );
                island_motion[gid] = vec4(
                    intBitsToFloat(runtime_words[base + 9]),
                    intBitsToFloat(runtime_words[base + 10]) + 0.9,
                    intBitsToFloat(runtime_words[base + 11]),
                    intBitsToFloat(runtime_words[base + 12])
                );
            }}
            """
        )
        self.programs["resolve_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform int process_rank;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};
            layout(std430, binding=1) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D island_id_tex;
            layout(r32f, binding=2) coherent uniform image2D shadow_material_img;

            ivec4 order_key(FallingIslandReservation reservation) {{
                if (reservation.target_dy > 0) {{
                    return ivec4(0, -reservation.bbox_w, reservation.bbox_x, reservation.island_id);
                }}
                if (reservation.target_dy < 0) {{
                    return ivec4(1, reservation.bbox_y, reservation.bbox_x, reservation.island_id);
                }}
                if (reservation.target_dx > 0) {{
                    return ivec4(2, -reservation.bbox_z, reservation.bbox_y, reservation.island_id);
                }}
                if (reservation.target_dx < 0) {{
                    return ivec4(3, reservation.bbox_x, reservation.bbox_y, reservation.island_id);
                }}
                return ivec4(4, reservation.bbox_y, reservation.bbox_x, reservation.island_id);
            }}

            bool key_less(ivec4 lhs, ivec4 rhs) {{
                if (lhs.x != rhs.x) {{
                    return lhs.x < rhs.x;
                }}
                if (lhs.y != rhs.y) {{
                    return lhs.y < rhs.y;
                }}
                if (lhs.z != rhs.z) {{
                    return lhs.z < rhs.z;
                }}
                return lhs.w < rhs.w;
            }}

            int index_for_rank(int rank) {{
                for (int index = 0; index < reservation_count; ++index) {{
                    ivec4 candidate_key = order_key(reservations[index]);
                    int lower_count = 0;
                    for (int other = 0; other < reservation_count; ++other) {{
                        if (other == index) {{
                            continue;
                        }}
                        ivec4 other_key = order_key(reservations[other]);
                        if (key_less(other_key, candidate_key)) {{
                            lower_count += 1;
                        }}
                    }}
                    if (lower_count == rank) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (
                    cell.x < reservation.bbox_x
                    || cell.x >= reservation.bbox_z
                    || cell.y < reservation.bbox_y
                    || cell.y >= reservation.bbox_w
                ) {{
                    return false;
                }}
                int island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                return island_id == reservation.island_id && material_id > 0;
            }}

            bool can_shift_delta(FallingIslandReservation reservation, int dx, int dy) {{
                for (int y = reservation.bbox_y; y < reservation.bbox_w; ++y) {{
                    for (int x = reservation.bbox_x; x < reservation.bbox_z; ++x) {{
                        ivec2 source = ivec2(x, y);
                        if (!source_matches(reservation, source)) {{
                            continue;
                        }}
                        ivec2 target = source + ivec2(dx, dy);
                        if (target.x < 0 || target.y < 0 || target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                            return false;
                        }}
                        if (imageLoad(shadow_material_img, target).x <= 0.5) {{
                            continue;
                        }}
                        if (source_matches(reservation, target)) {{
                            continue;
                        }}
                        return false;
                    }}
                }}
                return true;
            }}

            ivec2 resolve_shift(FallingIslandReservation reservation) {{
                ivec2 target_shift = ivec2(reservation.target_dx, reservation.target_dy);
                if (target_shift.x == 0 && target_shift.y == 0) {{
                    return ivec2(0, 0);
                }}
                ivec2 current = ivec2(0, 0);
                ivec2 furthest = ivec2(0, 0);
                int dx = abs(target_shift.x);
                int dy = -abs(target_shift.y);
                int sx = current.x < target_shift.x ? 1 : -1;
                int sy = current.y < target_shift.y ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current.x == target_shift.x && current.y == target_shift.y) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current.x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current.y += sy;
                    }}
                    if (!can_shift_delta(reservation, current.x, current.y)) {{
                        break;
                    }}
                    furthest = current;
                }}
                return furthest;
            }}

            vec2 collision_response(vec2 velocity, ivec2 attempted_delta, ivec2 actual_delta, float friction, float elasticity) {{
                float tangential_scale = max(0.0, 1.0 - clamp(friction, 0.0, 1.0));
                float bounce = max(0.0, elasticity);
                float vx = velocity.x;
                float vy = velocity.y;
                if (attempted_delta.x != actual_delta.x && attempted_delta.x != 0) {{
                    float normal_vx = abs(vx) > 1.0e-6 ? vx : float(attempted_delta.x);
                    vx = -normal_vx * bounce;
                    vy *= tangential_scale;
                }}
                if (attempted_delta.y != actual_delta.y && attempted_delta.y != 0) {{
                    float normal_vy = abs(vy) > 1.0e-6 ? vy : float(attempted_delta.y);
                    vy = -normal_vy * bounce;
                    vx *= tangential_scale;
                }}
                if (abs(vx) < 1.0e-6) {{
                    vx = 0.0;
                }}
                if (abs(vy) < 1.0e-6) {{
                    vy = 0.0;
                }}
                return vec2(vx, vy);
            }}

            vec2 island_collision_response(FallingIslandReservation reservation, ivec2 actual_shift) {{
                float friction_sum = 0.0;
                float elasticity_sum = 0.0;
                int material_count = 0;
                for (int y = reservation.bbox_y; y < reservation.bbox_w; ++y) {{
                    for (int x = reservation.bbox_x; x < reservation.bbox_z; ++x) {{
                        ivec2 source = ivec2(x, y);
                        if (!source_matches(reservation, source)) {{
                            continue;
                        }}
                        int material_id = clamp(int(texelFetch(material_tex, source, 0).x + 0.5), 0, {MAX_MATERIALS - 1});
                        friction_sum += contact_params[material_id].x;
                        elasticity_sum += contact_params[material_id].y;
                        material_count += 1;
                    }}
                }}
                if (material_count <= 0) {{
                    return vec2(reservation.velocity_x, reservation.velocity_y);
                }}
                float inv_count = 1.0 / float(material_count);
                return collision_response(
                    vec2(reservation.velocity_x, reservation.velocity_y),
                    ivec2(reservation.target_dx, reservation.target_dy),
                    actual_shift,
                    friction_sum * inv_count,
                    elasticity_sum * inv_count
                );
            }}

            void apply_shadow_shift(FallingIslandReservation reservation, ivec2 shift) {{
                if (shift.x == 0 && shift.y == 0) {{
                    return;
                }}
                for (int y = reservation.bbox_y; y < reservation.bbox_w; ++y) {{
                    for (int x = reservation.bbox_x; x < reservation.bbox_z; ++x) {{
                        ivec2 source = ivec2(x, y);
                        if (source_matches(reservation, source)) {{
                            imageStore(shadow_material_img, source, vec4(0.0, 0.0, 0.0, 0.0));
                        }}
                    }}
                }}
                memoryBarrierImage();
                for (int y = reservation.bbox_y; y < reservation.bbox_w; ++y) {{
                    for (int x = reservation.bbox_x; x < reservation.bbox_z; ++x) {{
                        ivec2 source = ivec2(x, y);
                        if (!source_matches(reservation, source)) {{
                            continue;
                        }}
                        ivec2 target = source + shift;
                        float material_id = texelFetch(material_tex, source, 0).x;
                        imageStore(shadow_material_img, target, vec4(material_id, 0.0, 0.0, 0.0));
                    }}
                }}
            }}

            void main() {{
                if (reservation_count <= 0 || process_rank < 0 || process_rank >= reservation_count) {{
                    return;
                }}
                int index = index_for_rank(process_rank);
                if (index < 0) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[index];
                ivec2 actual_shift = resolve_shift(reservation);
                reservations[index].resolved_dx = actual_shift.x;
                reservations[index].resolved_dy = actual_shift.y;
                if (actual_shift.x == 0 && actual_shift.y == 0 && reservation.target_dx == 0 && reservation.target_dy == 0) {{
                    reservations[index].resolve_state = {ISLAND_RESOLVE_DIRECT};
                }} else if (actual_shift.x == 0 && actual_shift.y == 0) {{
                    reservations[index].resolve_state = {ISLAND_RESOLVE_BLOCKED};
                }} else if (actual_shift.x == reservation.reserved_dx && actual_shift.y == reservation.reserved_dy) {{
                    reservations[index].resolve_state = {ISLAND_RESOLVE_DIRECT};
                }} else {{
                    reservations[index].resolve_state = {ISLAND_RESOLVE_RERESOLVED};
                }}
                if (
                    (actual_shift.x != 0 || actual_shift.y != 0)
                    && (actual_shift.x != reservation.target_dx || actual_shift.y != reservation.target_dy)
                ) {{
                    vec2 response_velocity = island_collision_response(reservation, actual_shift);
                    reservations[index].velocity_x = response_velocity.x;
                    reservations[index].velocity_y = response_velocity.y;
                }}
                apply_shadow_shift(reservation, actual_shift);
            }}
            """
        )
        self.programs["resolve_powder_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=1) buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=3) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D velocity_tex;
            layout(binding=3) uniform sampler2D active_tile_tex;
            layout(binding=4) uniform sampler2D powder_target_tex;
            layout(r32f, binding=5) coherent uniform image2D resolve_material_img;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            bool path_is_clear(ivec2 source, ivec2 target) {{
                ivec2 desired = target - source;
                if (desired.x == 0 && desired.y == 0) {{
                    return true;
                }}
                ivec2 current = ivec2(0, 0);
                int dx = abs(desired.x);
                int dy = -abs(desired.y);
                int sx = current.x < desired.x ? 1 : -1;
                int sy = current.y < desired.y ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current.x == desired.x && current.y == desired.y) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current.x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current.y += sy;
                    }}
                    ivec2 sample_cell = source + current;
                    if (!in_bounds(sample_cell)) {{
                        return false;
                    }}
                    if (imageLoad(resolve_material_img, sample_cell).x > 0.5) {{
                        return false;
                    }}
                }}
                return true;
            }}

            ivec2 fallback_candidate(ivec2 source, int material_id, int index) {{
                float gravity = material_params[clamp(material_id, 0, {MAX_MATERIALS - 1})].y;
                if (gravity >= 0.0) {{
                    if (index == 0) return source + ivec2(0, 1);
                    if (index == 1) return source + ivec2(-1, 1);
                    return source + ivec2(1, 1);
                }}
                if (index == 0) return source + ivec2(0, -1);
                if (index == 1) return source + ivec2(-1, -1);
                return source + ivec2(1, -1);
            }}

            void append_reservation(
                ivec2 source,
                ivec2 desired,
                ivec2 reserved,
                ivec2 resolved,
                vec2 velocity,
                int material_id,
                int resolve_state
            ) {{
                int index = powder_reservation_count;
                powder_reservation_count += 1;
                reservations[index].source_xy = source;
                reservations[index].desired_target_xy = desired;
                reservations[index].reserved_target_xy = reserved;
                reservations[index].resolved_target_xy = resolved;
                reservations[index].velocity_xy = velocity;
                reservations[index].material_id = material_id;
                reservations[index].resolve_state = resolve_state;
            }}

            void main() {{
                powder_reservation_count = 0;
                for (int y = 0; y < cell_grid_size.y; ++y) {{
                    for (int x = 0; x < cell_grid_size.x; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        imageStore(resolve_material_img, cell, vec4(texelFetch(material_tex, cell, 0).x, 0.0, 0.0, 0.0));
                    }}
                }}
                memoryBarrierImage();

                for (int y = cell_grid_size.y - 2; y >= 0; --y) {{
                    for (int x = 0; x < cell_grid_size.x; ++x) {{
                        ivec2 source = ivec2(x, y);
                        int material_id = int(texelFetch(material_tex, source, 0).x + 0.5);
                        if (
                            !solve_cell_active(source)
                            || material_id <= 0
                            || int(texelFetch(phase_tex, source, 0).x + 0.5) != phase_powder
                        ) {{
                            continue;
                        }}
                        material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                        vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                        int max_step = int(material_params[material_id].x + 0.5);
                        ivec2 desired = source + ivec2(
                            int(clamp(round(velocity.x), float(-max_step), float(max_step))),
                            int(clamp(round(velocity.y), float(-max_step), float(max_step)))
                        );
                        ivec2 reserved = ivec2(round(texelFetch(powder_target_tex, source, 0).xy));
                        ivec2 resolved = source;
                        int resolve_state = {POWDER_RESOLVE_BLOCKED};
                        if (int(round(imageLoad(resolve_material_img, source).x)) != material_id) {{
                            append_reservation(source, desired, reserved, source, velocity, material_id, {POWDER_RESOLVE_STALE});
                            continue;
                        }}
                        if ((reserved.x != source.x || reserved.y != source.y) && in_bounds(reserved) && path_is_clear(source, reserved)) {{
                            imageStore(resolve_material_img, source, vec4(0.0, 0.0, 0.0, 0.0));
                            imageStore(resolve_material_img, reserved, vec4(float(material_id), 0.0, 0.0, 0.0));
                            append_reservation(source, desired, reserved, reserved, velocity, material_id, {POWDER_RESOLVE_DDA});
                            continue;
                        }}
                        if (int(contact_params[material_id].z + 0.5) != {POWDER_SOLVER_SUSPENDED}) {{
                            for (int candidate_index = 0; candidate_index < 3; ++candidate_index) {{
                                ivec2 fallback = fallback_candidate(source, material_id, candidate_index);
                                if (in_bounds(fallback) && imageLoad(resolve_material_img, fallback).x <= 0.5) {{
                                    imageStore(resolve_material_img, source, vec4(0.0, 0.0, 0.0, 0.0));
                                    imageStore(resolve_material_img, fallback, vec4(float(material_id), 0.0, 0.0, 0.0));
                                    resolved = fallback;
                                    resolve_state = {POWDER_RESOLVE_FALLBACK};
                                    break;
                                }}
                            }}
                        }}
                        append_reservation(source, desired, reserved, resolved, velocity, material_id, resolve_state);
                    }}
                }}
            }}
            """
        )
        self.programs["apply_powder_fast_path"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;
            uniform int max_powder_step;

            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=2) buffer PowderReservationCount {{
                int powder_reservation_count;
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D cell_flags_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(binding=10) uniform sampler2D active_tile_tex;

            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_out_img;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            int material_at(ivec2 cell) {{
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            ivec2 desired_target(ivec2 source, int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                int max_step = int(material_params[material_id].x + 0.5);
                max_step = clamp(max_step, 0, max_powder_step);
                return source + ivec2(
                    int(clamp(round(velocity.x), float(-max_step), float(max_step))),
                    int(clamp(round(velocity.y), float(-max_step), float(max_step)))
                );
            }}

            bool path_is_clear(ivec2 source, ivec2 target, ivec2 destination) {{
                ivec2 desired = target - source;
                if (desired.x == 0 && desired.y == 0) {{
                    return true;
                }}
                ivec2 current = ivec2(0, 0);
                int dx = abs(desired.x);
                int dy = -abs(desired.y);
                int sx = current.x < desired.x ? 1 : -1;
                int sy = current.y < desired.y ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current.x == desired.x && current.y == desired.y) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current.x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current.y += sy;
                    }}
                    ivec2 sample_cell = source + current;
                    if (!in_bounds(sample_cell)) {{
                        return false;
                    }}
                    if (sample_cell == destination) {{
                        continue;
                    }}
                    if (material_at(sample_cell) > 0) {{
                        return false;
                    }}
                }}
                return true;
            }}

            ivec2 dda_target(ivec2 source, int material_id, ivec2 desired) {{
                if (desired == source) {{
                    return source;
                }}
                ivec2 delta = desired - source;
                ivec2 current = ivec2(0, 0);
                ivec2 furthest = source;
                int dx = abs(delta.x);
                int dy = -abs(delta.y);
                int sx = current.x < delta.x ? 1 : -1;
                int sy = current.y < delta.y ? 1 : -1;
                int err = dx + dy;
                while (true) {{
                    if (current.x == delta.x && current.y == delta.y) {{
                        break;
                    }}
                    int e2 = 2 * err;
                    if (e2 >= dy) {{
                        err += dy;
                        current.x += sx;
                    }}
                    if (e2 <= dx) {{
                        err += dx;
                        current.y += sy;
                    }}
                    ivec2 sample_cell = source + current;
                    if (!in_bounds(sample_cell)) {{
                        break;
                    }}
                    if (material_at(sample_cell) > 0) {{
                        break;
                    }}
                    furthest = sample_cell;
                }}
                return furthest;
            }}

            ivec2 fallback_candidate(ivec2 source, int material_id, int index) {{
                float gravity = material_params[clamp(material_id, 0, {MAX_MATERIALS - 1})].y;
                if (gravity >= 0.0) {{
                    if (index == 0) return source + ivec2(0, 1);
                    if (index == 1) return source + ivec2(-1, 1);
                    return source + ivec2(1, 1);
                }}
                if (index == 0) return source + ivec2(0, -1);
                if (index == 1) return source + ivec2(-1, -1);
                return source + ivec2(1, -1);
            }}

            ivec2 resolved_target(ivec2 source, int material_id) {{
                ivec2 desired = desired_target(source, material_id);
                ivec2 reserved = dda_target(source, material_id, desired);
                if (reserved != source && in_bounds(reserved) && path_is_clear(source, reserved, reserved)) {{
                    return reserved;
                }}
                if (int(contact_params[clamp(material_id, 0, {MAX_MATERIALS - 1})].z + 0.5) != {POWDER_SOLVER_SUSPENDED}) {{
                    for (int candidate_index = 0; candidate_index < 3; ++candidate_index) {{
                        ivec2 fallback = fallback_candidate(source, material_id, candidate_index);
                        if (in_bounds(fallback) && material_at(fallback) <= 0) {{
                            return fallback;
                        }}
                    }}
                }}
                return source;
            }}

            bool source_claims_destination(ivec2 source, ivec2 destination, out ivec2 resolved) {{
                resolved = source;
                if (!in_bounds(source) || !solve_cell_active(source)) {{
                    return false;
                }}
                int source_material = material_at(source);
                if (source_material <= 0 || phase_at(source) != phase_powder) {{
                    return false;
                }}
                resolved = resolved_target(source, source_material);
                return resolved == destination && resolved != source;
            }}

            bool better_source(ivec2 candidate, ivec2 current_best) {{
                if (current_best.x < 0) {{
                    return true;
                }}
                if (candidate.y != current_best.y) {{
                    return candidate.y > current_best.y;
                }}
                return candidate.x < current_best.x;
            }}

            ivec2 winning_source_for(ivec2 destination) {{
                ivec2 winner = ivec2(-1, -1);
                for (int dy = -max_powder_step; dy <= max_powder_step; ++dy) {{
                    for (int dx = -max_powder_step; dx <= max_powder_step; ++dx) {{
                        ivec2 candidate = destination + ivec2(dx, dy);
                        ivec2 resolved;
                        if (source_claims_destination(candidate, destination, resolved) && better_source(candidate, winner)) {{
                            winner = candidate;
                        }}
                    }}
                }}
                return winner;
            }}

            bool source_moves_out(ivec2 source) {{
                if (!solve_cell_active(source)) {{
                    return false;
                }}
                int source_material = material_at(source);
                if (source_material <= 0 || phase_at(source) != phase_powder) {{
                    return false;
                }}
                ivec2 target = resolved_target(source, source_material);
                if (target == source) {{
                    return false;
                }}
                ivec2 winner = winning_source_for(target);
                return winner == source;
            }}

            vec2 collision_response(vec2 velocity, ivec2 attempted_delta, ivec2 actual_delta, int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                float friction = clamp(contact_params[material_id].x, 0.0, 1.0);
                float elasticity = max(0.0, contact_params[material_id].y);
                float tangential_scale = max(0.0, 1.0 - friction);
                float vx = velocity.x;
                float vy = velocity.y;
                if (attempted_delta.x != actual_delta.x && attempted_delta.x != 0) {{
                    float normal_vx = abs(vx) > 1.0e-6 ? vx : float(attempted_delta.x);
                    vx = -normal_vx * elasticity;
                    vy *= tangential_scale;
                }}
                if (attempted_delta.y != actual_delta.y && attempted_delta.y != 0) {{
                    float normal_vy = abs(vy) > 1.0e-6 ? vy : float(attempted_delta.y);
                    vy = -normal_vy * elasticity;
                    vx *= tangential_scale;
                }}
                if (abs(vx) < 1.0e-6) vx = 0.0;
                if (abs(vy) < 1.0e-6) vy = 0.0;
                return vec2(vx, vy);
            }}

            void store_payload(
                ivec2 dst,
                float material_value,
                float phase_value,
                float flags_value,
                vec2 velocity_value,
                float temp_value,
                vec4 timer_value,
                float integrity_value
            ) {{
                imageStore(material_out_img, dst, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, dst, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, dst, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, dst, vec4(velocity_value, 0.0, 0.0));
                imageStore(temp_out_img, dst, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, dst, timer_value);
                imageStore(integrity_out_img, dst, vec4(integrity_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                store_payload(
                    dst,
                    texelFetch(material_tex, dst, 0).x,
                    texelFetch(phase_tex, dst, 0).x,
                    texelFetch(cell_flags_tex, dst, 0).x,
                    texelFetch(velocity_tex, dst, 0).xy,
                    texelFetch(temp_tex, dst, 0).x,
                    texelFetch(timer_tex, dst, 0),
                    texelFetch(integrity_tex, dst, 0).x
                );
            }}

            void store_clear(ivec2 dst) {{
                store_payload(dst, 0.0, 0.0, 0.0, vec2(0.0), 0.0, vec4(0.0), 0.0);
            }}

            void store_incoming(ivec2 dst, ivec2 src) {{
                int material_id = material_at(src);
                ivec2 desired_delta = desired_target(src, material_id) - src;
                ivec2 actual_delta = dst - src;
                vec2 velocity_value = texelFetch(velocity_tex, src, 0).xy;
                if (desired_delta != actual_delta) {{
                    velocity_value = collision_response(velocity_value, desired_delta, actual_delta, material_id);
                }}
                store_payload(
                    dst,
                    texelFetch(material_tex, src, 0).x,
                    texelFetch(phase_tex, src, 0).x,
                    texelFetch(cell_flags_tex, src, 0).x,
                    velocity_value,
                    texelFetch(temp_tex, src, 0).x,
                    texelFetch(timer_tex, src, 0),
                    texelFetch(integrity_tex, src, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                if (gid.x == 0 && gid.y == 0) {{
                    powder_reservation_count = 0;
                }}
                ivec2 incoming = winning_source_for(gid);
                if (incoming.x >= 0) {{
                    store_incoming(gid, incoming);
                    return;
                }}
                if (source_moves_out(gid)) {{
                    store_clear(gid);
                    return;
                }}
                store_original(gid);
            }}
            """
        )
        self.programs["apply_powder_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=1) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=2) readonly buffer PowderReservationCount {{
                int reservation_counter;
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D cell_flags_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_out_img;

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
            }}

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_counter, 0, cell_grid_size.x * cell_grid_size.y);
                }}
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool is_moving(PowderReservation reservation) {{
                return (
                    (reservation.resolve_state == {POWDER_RESOLVE_DDA} || reservation.resolve_state == {POWDER_RESOLVE_FALLBACK})
                    && !same_cell(reservation.source_xy, reservation.resolved_target_xy)
                );
            }}

            vec2 collision_response(vec2 velocity, ivec2 attempted_delta, ivec2 actual_delta, int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                float friction = clamp(contact_params[material_id].x, 0.0, 1.0);
                float elasticity = max(0.0, contact_params[material_id].y);
                float tangential_scale = max(0.0, 1.0 - friction);
                float vx = velocity.x;
                float vy = velocity.y;
                if (attempted_delta.x != actual_delta.x && attempted_delta.x != 0) {{
                    float normal_vx = abs(vx) > 1.0e-6 ? vx : float(attempted_delta.x);
                    vx = -normal_vx * elasticity;
                    vy *= tangential_scale;
                }}
                if (attempted_delta.y != actual_delta.y && attempted_delta.y != 0) {{
                    float normal_vy = abs(vy) > 1.0e-6 ? vy : float(attempted_delta.y);
                    vy = -normal_vy * elasticity;
                    vx *= tangential_scale;
                }}
                if (abs(vx) < 1.0e-6) {{
                    vx = 0.0;
                }}
                if (abs(vy) < 1.0e-6) {{
                    vy = 0.0;
                }}
                return vec2(vx, vy);
            }}

            int incoming_reservation_index(ivec2 cell) {{
                int count = active_reservation_count();
                for (int index = 0; index < count; ++index) {{
                    PowderReservation reservation = reservations[index];
                    if (is_moving(reservation) && same_cell(reservation.resolved_target_xy, cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                int count = active_reservation_count();
                for (int index = 0; index < count; ++index) {{
                    PowderReservation reservation = reservations[index];
                    if (reservation.resolve_state != {POWDER_RESOLVE_STALE} && same_cell(reservation.source_xy, cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(
                ivec2 dst,
                float material_value,
                float phase_value,
                float flags_value,
                vec2 velocity_value,
                float temp_value,
                vec4 timer_value,
                float integrity_value,
                float island_value,
                float entity_value,
                float displaced_value
            ) {{
                imageStore(material_out_img, dst, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, dst, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, dst, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, dst, vec4(velocity_value, 0.0, 0.0));
                imageStore(temp_out_img, dst, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, dst, timer_value);
                imageStore(integrity_out_img, dst, vec4(integrity_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst, vec2 velocity_override, bool override_velocity) {{
                vec2 velocity_value = override_velocity ? velocity_override : texelFetch(velocity_tex, dst, 0).xy;
                store_payload(
                    dst,
                    texelFetch(material_tex, dst, 0).x,
                    texelFetch(phase_tex, dst, 0).x,
                    texelFetch(cell_flags_tex, dst, 0).x,
                    velocity_value,
                    texelFetch(temp_tex, dst, 0).x,
                    texelFetch(timer_tex, dst, 0),
                    texelFetch(integrity_tex, dst, 0).x,
                    texelFetch(island_id_tex, dst, 0).x,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_clear(ivec2 dst) {{
                store_payload(dst, 0.0, 0.0, 0.0, vec2(0.0), 0.0, vec4(0.0), 0.0, 0.0, 0.0, 0.0);
            }}

            void store_incoming(ivec2 dst, PowderReservation reservation) {{
                ivec2 src = reservation.source_xy;
                int material_id = int(texelFetch(material_tex, src, 0).x + 0.5);
                vec2 velocity_value = texelFetch(velocity_tex, src, 0).xy;
                ivec2 desired_delta = reservation.desired_target_xy - reservation.source_xy;
                ivec2 actual_delta = reservation.resolved_target_xy - reservation.source_xy;
                if (material_id != 0 && (desired_delta.x != actual_delta.x || desired_delta.y != actual_delta.y)) {{
                    velocity_value = collision_response(velocity_value, desired_delta, actual_delta, material_id);
                }}
                store_payload(
                    dst,
                    texelFetch(material_tex, src, 0).x,
                    texelFetch(phase_tex, src, 0).x,
                    texelFetch(cell_flags_tex, src, 0).x,
                    velocity_value,
                    texelFetch(temp_tex, src, 0).x,
                    texelFetch(timer_tex, src, 0),
                    texelFetch(integrity_tex, src, 0).x,
                    texelFetch(island_id_tex, src, 0).x,
                    texelFetch(entity_id_tex, src, 0).x,
                    texelFetch(displaced_tex, src, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int incoming_index = incoming_reservation_index(gid);
                int outgoing_index = outgoing_reservation_index(gid);
                if (incoming_index >= 0) {{
                    store_incoming(gid, reservations[incoming_index]);
                    return;
                }}
                if (outgoing_index >= 0) {{
                    PowderReservation reservation = reservations[outgoing_index];
                    if (is_moving(reservation)) {{
                        store_clear(gid);
                        return;
                    }}
                    int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                    ivec2 desired_delta = reservation.desired_target_xy - reservation.source_xy;
                    vec2 velocity_value = texelFetch(velocity_tex, gid, 0).xy;
                    if (desired_delta.x != 0 || desired_delta.y != 0) {{
                        velocity_value = collision_response(velocity_value, desired_delta, ivec2(0, 0), material_id);
                    }} else {{
                        velocity_value *= 0.2;
                    }}
                    store_original(gid, velocity_value, true);
                    return;
                }}
                store_original(gid, vec2(0.0), false);
            }}
            """
        )
        self.programs["apply_powder_reservation_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=2) readonly buffer PowderReservationCount {{
                int reservation_counter;
            }};

            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
            }}

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_counter, 0, cell_grid_size.x * cell_grid_size.y);
                }}
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool is_moving(PowderReservation reservation) {{
                return (
                    (reservation.resolve_state == {POWDER_RESOLVE_DDA} || reservation.resolve_state == {POWDER_RESOLVE_FALLBACK})
                    && !same_cell(reservation.source_xy, reservation.resolved_target_xy)
                );
            }}

            int incoming_reservation_index(ivec2 cell) {{
                int count = active_reservation_count();
                for (int index = 0; index < count; ++index) {{
                    PowderReservation reservation = reservations[index];
                    if (is_moving(reservation) && same_cell(reservation.resolved_target_xy, cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                int count = active_reservation_count();
                for (int index = 0; index < count; ++index) {{
                    PowderReservation reservation = reservations[index];
                    if (reservation.resolve_state != {POWDER_RESOLVE_STALE} && same_cell(reservation.source_xy, cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(ivec2 dst, float island_value, float entity_value, float displaced_value) {{
                imageStore(island_id_out_img, dst, vec4(island_value, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, dst, vec4(entity_value, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, dst, vec4(displaced_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                store_payload(
                    dst,
                    texelFetch(island_id_tex, dst, 0).x,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_clear(ivec2 dst) {{
                store_payload(dst, 0.0, 0.0, 0.0);
            }}

            void store_incoming(ivec2 dst, PowderReservation reservation) {{
                ivec2 src = reservation.source_xy;
                store_payload(
                    dst,
                    texelFetch(island_id_tex, src, 0).x,
                    texelFetch(entity_id_tex, src, 0).x,
                    texelFetch(displaced_tex, src, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int incoming_index = incoming_reservation_index(gid);
                int outgoing_index = outgoing_reservation_index(gid);
                if (incoming_index >= 0) {{
                    store_incoming(gid, reservations[incoming_index]);
                    return;
                }}
                if (outgoing_index >= 0) {{
                    PowderReservation reservation = reservations[outgoing_index];
                    if (is_moving(reservation)) {{
                        store_clear(gid);
                        return;
                    }}
                }}
                store_original(gid);
            }}
            """
        )
        self.programs["apply_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int reservation_count;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D cell_flags_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
            layout(binding=20) uniform sampler2D ambient_tex;

            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_out_img;

            bool is_moving(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool in_source_bbox(FallingIslandReservation reservation, ivec2 cell) {{
                return cell.x >= reservation.bbox_x
                    && cell.x < reservation.bbox_z
                    && cell.y >= reservation.bbox_y
                    && cell.y < reservation.bbox_w;
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (!is_moving(reservation) || !in_source_bbox(reservation, cell)) {{
                    return false;
                }}
                int source_island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int source_material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id && source_material_id > 0;
            }}

            int incoming_reservation_index(ivec2 cell, out ivec2 source_cell) {{
                for (int index = 0; index < reservation_count; ++index) {{
                    FallingIslandReservation reservation = reservations[index];
                    ivec2 source = cell - ivec2(reservation.resolved_dx, reservation.resolved_dy);
                    if (source.x < 0 || source.y < 0 || source.x >= cell_grid_size.x || source.y >= cell_grid_size.y) {{
                        continue;
                    }}
                    if (source_matches(reservation, source)) {{
                        source_cell = source;
                        return index;
                    }}
                }}
                source_cell = cell;
                return -1;
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                for (int index = 0; index < reservation_count; ++index) {{
                    if (source_matches(reservations[index], cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(
                ivec2 dst,
                float material_value,
                float phase_value,
                float flags_value,
                vec2 velocity_value,
                float temp_value,
                vec4 timer_value,
                float integrity_value,
                float island_value,
                float entity_value,
                float displaced_value
            ) {{
                imageStore(material_out_img, dst, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, dst, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, dst, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, dst, vec4(velocity_value, 0.0, 0.0));
                imageStore(temp_out_img, dst, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, dst, timer_value);
                imageStore(integrity_out_img, dst, vec4(integrity_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                store_payload(
                    dst,
                    texelFetch(material_tex, dst, 0).x,
                    texelFetch(phase_tex, dst, 0).x,
                    texelFetch(cell_flags_tex, dst, 0).x,
                    texelFetch(velocity_tex, dst, 0).xy,
                    texelFetch(temp_tex, dst, 0).x,
                    texelFetch(timer_tex, dst, 0),
                    texelFetch(integrity_tex, dst, 0).x,
                    texelFetch(island_id_tex, dst, 0).x,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_clear(ivec2 dst) {{
                ivec2 gas_cell = ivec2(
                    min(dst.x / gas_cell_size, gas_grid_size.x - 1),
                    min(dst.y / gas_cell_size, gas_grid_size.y - 1)
                );
                float ambient = texelFetch(ambient_tex, gas_cell, 0).x;
                store_payload(dst, 0.0, 0.0, 0.0, vec2(0.0), ambient, vec4(0.0), 0.0, 0.0, 0.0, 0.0);
            }}

            void store_incoming(ivec2 dst, FallingIslandReservation reservation, ivec2 src) {{
                store_payload(
                    dst,
                    texelFetch(material_tex, src, 0).x,
                    texelFetch(phase_tex, src, 0).x,
                    texelFetch(cell_flags_tex, src, 0).x,
                    texelFetch(velocity_tex, src, 0).xy,
                    texelFetch(temp_tex, src, 0).x,
                    texelFetch(timer_tex, src, 0),
                    texelFetch(integrity_tex, src, 0).x,
                    float(reservation.island_id),
                    texelFetch(entity_id_tex, src, 0).x,
                    texelFetch(displaced_tex, src, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 source_cell;
                int incoming_index = incoming_reservation_index(gid, source_cell);
                if (incoming_index >= 0) {{
                    store_incoming(gid, reservations[incoming_index], source_cell);
                    return;
                }}
                if (outgoing_reservation_index(gid) >= 0) {{
                    store_clear(gid);
                    return;
                }}
                store_original(gid);
            }}
            """
        )
        self.programs["apply_falling_island_reservation_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            bool is_moving(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool in_source_bbox(FallingIslandReservation reservation, ivec2 cell) {{
                return cell.x >= reservation.bbox_x
                    && cell.x < reservation.bbox_z
                    && cell.y >= reservation.bbox_y
                    && cell.y < reservation.bbox_w;
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (!is_moving(reservation) || !in_source_bbox(reservation, cell)) {{
                    return false;
                }}
                int source_island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int source_material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id && source_material_id > 0;
            }}

            int incoming_reservation_index(ivec2 cell, out ivec2 source_cell) {{
                for (int index = 0; index < reservation_count; ++index) {{
                    FallingIslandReservation reservation = reservations[index];
                    ivec2 source = cell - ivec2(reservation.resolved_dx, reservation.resolved_dy);
                    if (source.x < 0 || source.y < 0 || source.x >= cell_grid_size.x || source.y >= cell_grid_size.y) {{
                        continue;
                    }}
                    if (source_matches(reservation, source)) {{
                        source_cell = source;
                        return index;
                    }}
                }}
                source_cell = cell;
                return -1;
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                for (int index = 0; index < reservation_count; ++index) {{
                    if (source_matches(reservations[index], cell)) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(ivec2 dst, float island_value, float entity_value, float displaced_value) {{
                imageStore(island_id_out_img, dst, vec4(island_value, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, dst, vec4(entity_value, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, dst, vec4(displaced_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                store_payload(
                    dst,
                    texelFetch(island_id_tex, dst, 0).x,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_clear(ivec2 dst) {{
                store_payload(dst, 0.0, 0.0, 0.0);
            }}

            void store_incoming(ivec2 dst, FallingIslandReservation reservation, ivec2 src) {{
                store_payload(
                    dst,
                    float(reservation.island_id),
                    texelFetch(entity_id_tex, src, 0).x,
                    texelFetch(displaced_tex, src, 0).x
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 source_cell;
                int incoming_index = incoming_reservation_index(gid, source_cell);
                if (incoming_index >= 0) {{
                    store_incoming(gid, reservations[incoming_index], source_cell);
                    return;
                }}
                if (outgoing_reservation_index(gid) >= 0) {{
                    store_clear(gid);
                    return;
                }}
                store_original(gid);
            }}
            """
        )
        self.programs["apply_falling_island_materialization"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform int phase_falling_island;
            uniform int phase_powder;
            uniform int phase_static_solid;
            uniform int mode;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};
            layout(std430, binding=1) buffer MaterialFallingParams {{
                vec4 falling_params[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D cell_flags_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D timer_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=6) writeonly uniform image2D integrity_out_img;

            int clamp_material(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}

            vec4 rule_params(int material_id) {{
                return falling_params[clamp_material(material_id) * 2];
            }}

            vec4 lifecycle_params(int material_id) {{
                return falling_params[clamp_material(material_id) * 2 + 1];
            }}

            bool is_placeholder(int material_id) {{
                return int(rule_params(material_id).y + 0.5) == 7;
            }}

            int powder_generation_id(int material_id) {{
                return int(rule_params(material_id).w + 0.5);
            }}

            int default_phase_id(int material_id) {{
                return int(rule_params(material_id).x + 0.5);
            }}

            int falling_break_kind(int material_id) {{
                return int(rule_params(material_id).z + 0.5);
            }}

            int same_island_neighbor_count(ivec2 cell, int island_id) {{
                int count = 0;
                ivec2 offsets[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
                for (int i = 0; i < 4; ++i) {{
                    ivec2 neighbor = cell + offsets[i];
                    if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= cell_grid_size.x || neighbor.y >= cell_grid_size.y) {{
                        continue;
                    }}
                    int neighbor_island = int(texelFetch(island_id_tex, neighbor, 0).x + 0.5);
                    int neighbor_phase = int(texelFetch(phase_tex, neighbor, 0).x + 0.5);
                    int neighbor_material = int(texelFetch(material_tex, neighbor, 0).x + 0.5);
                    if (neighbor_island == island_id && neighbor_phase == phase_falling_island && neighbor_material > 0) {{
                        count += 1;
                    }}
                }}
                return count;
            }}

            bool falling_source_matches(ivec2 cell, int island_id, int material_id) {{
                if (material_id <= 0 || island_id <= 0) {{
                    return false;
                }}
                return int(texelFetch(phase_tex, cell, 0).x + 0.5) == phase_falling_island;
            }}

            bool shed_fragment(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                if (falling_break_kind(material_id) == {FALLING_ISLAND_BREAK_STABLE}) {{
                    return false;
                }}
                return same_island_neighbor_count(cell, island_id) < 2 && powder_generation_id(material_id) > 0;
            }}

            bool settle_fragment(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                return same_island_neighbor_count(cell, island_id) < 2 && powder_generation_id(material_id) > 0;
            }}

            bool is_settle_reservation(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            int settle_reservation_index(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return -1;
                }}
                for (int index = 0; index < reservation_count; ++index) {{
                    FallingIslandReservation reservation = reservations[index];
                    if (!is_settle_reservation(reservation) || reservation.island_id != island_id) {{
                        continue;
                    }}
                    if (
                        cell.x >= reservation.bbox_x
                        && cell.x < reservation.bbox_z
                        && cell.y >= reservation.bbox_y
                        && cell.y < reservation.bbox_w
                    ) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(
                ivec2 dst,
                float material_value,
                float phase_value,
                float flags_value,
                vec2 velocity_value,
                float temp_value,
                vec4 timer_value,
                float integrity_value,
                float island_value,
                float entity_value,
                float displaced_value
            ) {{
                imageStore(material_out_img, dst, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, dst, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, dst, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, dst, vec4(velocity_value, 0.0, 0.0));
                imageStore(temp_out_img, dst, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, dst, timer_value);
                imageStore(integrity_out_img, dst, vec4(integrity_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                int material_id = int(texelFetch(material_tex, dst, 0).x + 0.5);
                int phase_id = int(texelFetch(phase_tex, dst, 0).x + 0.5);
                float island_value = texelFetch(island_id_tex, dst, 0).x;
                if (island_value > 0.5 && (material_id <= 0 || phase_id != phase_falling_island)) {{
                    island_value = 0.0;
                }}
                store_payload(
                    dst,
                    texelFetch(material_tex, dst, 0).x,
                    texelFetch(phase_tex, dst, 0).x,
                    texelFetch(cell_flags_tex, dst, 0).x,
                    texelFetch(velocity_tex, dst, 0).xy,
                    texelFetch(temp_tex, dst, 0).x,
                    texelFetch(timer_tex, dst, 0),
                    texelFetch(integrity_tex, dst, 0).x,
                    island_value,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_generated_powder(ivec2 dst, int source_material_id) {{
                int generated_id = powder_generation_id(source_material_id);
                vec4 generated_lifecycle = lifecycle_params(generated_id);
                float old_temp = texelFetch(temp_tex, dst, 0).x;
                float spawn_temp = generated_lifecycle.y;
                float entity_value = texelFetch(entity_id_tex, dst, 0).x;
                float displaced_value = 0.0;
                if (!is_placeholder(generated_id)) {{
                    entity_value = 0.0;
                }}
                store_payload(
                    dst,
                    float(generated_id),
                    float(phase_powder),
                    0.0,
                    texelFetch(velocity_tex, dst, 0).xy,
                    max(old_temp, spawn_temp),
                    vec4(0.0),
                    generated_lifecycle.x,
                    0.0,
                    entity_value,
                    displaced_value
                );
            }}

            void store_settled_static(ivec2 dst, int material_id) {{
                float entity_value = texelFetch(entity_id_tex, dst, 0).x;
                float displaced_value = texelFetch(displaced_tex, dst, 0).x;
                if (material_id <= 0 || !is_placeholder(material_id)) {{
                    entity_value = 0.0;
                    displaced_value = 0.0;
                }}
                int default_phase = default_phase_id(material_id);
                if (default_phase == 0) {{
                    default_phase = phase_static_solid;
                }}
                store_payload(
                    dst,
                    texelFetch(material_tex, dst, 0).x,
                    float(default_phase),
                    texelFetch(cell_flags_tex, dst, 0).x,
                    texelFetch(velocity_tex, dst, 0).xy,
                    texelFetch(temp_tex, dst, 0).x,
                    texelFetch(timer_tex, dst, 0),
                    texelFetch(integrity_tex, dst, 0).x,
                    0.0,
                    entity_value,
                    displaced_value
                );
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int island_id = int(texelFetch(island_id_tex, gid, 0).x + 0.5);
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (mode == 0) {{
                    if (shed_fragment(gid, island_id, material_id)) {{
                        store_generated_powder(gid, material_id);
                        return;
                    }}
                    store_original(gid);
                    return;
                }}
                if (settle_reservation_index(gid, island_id, material_id) >= 0) {{
                    if (settle_fragment(gid, island_id, material_id)) {{
                        store_generated_powder(gid, material_id);
                    }} else {{
                        store_settled_static(gid, material_id);
                    }}
                    return;
                }}
                store_original(gid);
            }}
            """
        )
        self.programs["apply_falling_island_materialization_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform int phase_falling_island;
            uniform int mode;

            struct FallingIslandReservation {{
                int island_id;
                int bbox_x;
                int bbox_y;
                int bbox_z;
                int bbox_w;
                float velocity_x;
                float velocity_y;
                float subcell_x;
                float subcell_y;
                int target_dx;
                int target_dy;
                int reserved_dx;
                int reserved_dy;
                int resolved_dx;
                int resolved_dy;
                int resolve_state;
            }};

            layout(std430, binding=0) buffer IslandReservations {{
                FallingIslandReservation reservations[];
            }};
            layout(std430, binding=1) buffer MaterialFallingParams {{
                vec4 falling_params[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            int clamp_material(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}

            vec4 rule_params(int material_id) {{
                return falling_params[clamp_material(material_id) * 2];
            }}

            bool is_placeholder(int material_id) {{
                return int(rule_params(material_id).y + 0.5) == 7;
            }}

            int powder_generation_id(int material_id) {{
                return int(rule_params(material_id).w + 0.5);
            }}

            int falling_break_kind(int material_id) {{
                return int(rule_params(material_id).z + 0.5);
            }}

            int same_island_neighbor_count(ivec2 cell, int island_id) {{
                int count = 0;
                ivec2 offsets[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
                for (int i = 0; i < 4; ++i) {{
                    ivec2 neighbor = cell + offsets[i];
                    if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= cell_grid_size.x || neighbor.y >= cell_grid_size.y) {{
                        continue;
                    }}
                    int neighbor_island = int(texelFetch(island_id_tex, neighbor, 0).x + 0.5);
                    int neighbor_phase = int(texelFetch(phase_tex, neighbor, 0).x + 0.5);
                    int neighbor_material = int(texelFetch(material_tex, neighbor, 0).x + 0.5);
                    if (neighbor_island == island_id && neighbor_phase == phase_falling_island && neighbor_material > 0) {{
                        count += 1;
                    }}
                }}
                return count;
            }}

            bool falling_source_matches(ivec2 cell, int island_id, int material_id) {{
                if (material_id <= 0 || island_id <= 0) {{
                    return false;
                }}
                return int(texelFetch(phase_tex, cell, 0).x + 0.5) == phase_falling_island;
            }}

            bool shed_fragment(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                if (falling_break_kind(material_id) == {FALLING_ISLAND_BREAK_STABLE}) {{
                    return false;
                }}
                return same_island_neighbor_count(cell, island_id) < 2 && powder_generation_id(material_id) > 0;
            }}

            bool settle_fragment(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                return same_island_neighbor_count(cell, island_id) < 2 && powder_generation_id(material_id) > 0;
            }}

            bool is_settle_reservation(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            int settle_reservation_index(ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return -1;
                }}
                for (int index = 0; index < reservation_count; ++index) {{
                    FallingIslandReservation reservation = reservations[index];
                    if (!is_settle_reservation(reservation) || reservation.island_id != island_id) {{
                        continue;
                    }}
                    if (
                        cell.x >= reservation.bbox_x
                        && cell.x < reservation.bbox_z
                        && cell.y >= reservation.bbox_y
                        && cell.y < reservation.bbox_w
                    ) {{
                        return index;
                    }}
                }}
                return -1;
            }}

            void store_payload(ivec2 dst, float island_value, float entity_value, float displaced_value) {{
                imageStore(island_id_out_img, dst, vec4(island_value, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, dst, vec4(entity_value, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, dst, vec4(displaced_value, 0.0, 0.0, 0.0));
            }}

            void store_original(ivec2 dst) {{
                int material_id = int(texelFetch(material_tex, dst, 0).x + 0.5);
                int phase_id = int(texelFetch(phase_tex, dst, 0).x + 0.5);
                float island_value = texelFetch(island_id_tex, dst, 0).x;
                if (island_value > 0.5 && (material_id <= 0 || phase_id != phase_falling_island)) {{
                    island_value = 0.0;
                }}
                store_payload(
                    dst,
                    island_value,
                    texelFetch(entity_id_tex, dst, 0).x,
                    texelFetch(displaced_tex, dst, 0).x
                );
            }}

            void store_generated_powder(ivec2 dst, int source_material_id) {{
                int generated_id = powder_generation_id(source_material_id);
                float entity_value = texelFetch(entity_id_tex, dst, 0).x;
                float displaced_value = 0.0;
                if (!is_placeholder(generated_id)) {{
                    entity_value = 0.0;
                }}
                store_payload(dst, 0.0, entity_value, displaced_value);
            }}

            void store_settled_static(ivec2 dst, int material_id) {{
                float entity_value = texelFetch(entity_id_tex, dst, 0).x;
                float displaced_value = texelFetch(displaced_tex, dst, 0).x;
                if (material_id <= 0 || !is_placeholder(material_id)) {{
                    entity_value = 0.0;
                    displaced_value = 0.0;
                }}
                store_payload(dst, 0.0, entity_value, displaced_value);
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int island_id = int(texelFetch(island_id_tex, gid, 0).x + 0.5);
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (mode == 0) {{
                    if (shed_fragment(gid, island_id, material_id)) {{
                        store_generated_powder(gid, material_id);
                        return;
                    }}
                    store_original(gid);
                    return;
                }}
                if (settle_reservation_index(gid, island_id, material_id) >= 0) {{
                    if (settle_fragment(gid, island_id, material_id)) {{
                        store_generated_powder(gid, material_id);
                    }} else {{
                        store_settled_static(gid, material_id);
                    }}
                    return;
                }}
                store_original(gid);
            }}
            """
        )

    def _upload_inputs(self, world: "WorldEngine", resources: GPUMotionResources, solve_tile_mask: np.ndarray) -> None:
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
            resources.phase_tex.write(world.phase.astype("f4").tobytes())
            resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
        if upload_plan["flow_velocity"]:
            resources.flow_tex.write(world.flow_velocity.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        if upload_plan["active_tile_ttl"]:
            resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=1)
        resources.powder_target_tex.write(np.zeros((world.height, world.width, 2), dtype="f4").tobytes())
        self._upload_material_rule_params(world, resources)

    def _cpu_upload_plan(self, world: "WorldEngine") -> dict[str, bool]:
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "motion input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "flow_velocity",
            "ambient_temperature",
            "active_tile_ttl",
        )
        return {
            "cell_core": not (formal_gpu_frame and "cell_core" in authoritative),
            "island_id": not (formal_gpu_frame and "island_id" in authoritative),
            "entity_id": not (formal_gpu_frame and "entity_id" in authoritative),
            "placeholder_displaced_material": not (
                formal_gpu_frame and "placeholder_displaced_material" in authoritative
            ),
            "flow_velocity": not (formal_gpu_frame and "flow_velocity" in authoritative),
            "ambient_temperature": not (formal_gpu_frame and "ambient_temperature" in authoritative),
            "active_tile_ttl": not (formal_gpu_frame and "active_tile_ttl" in authoritative),
        }

    def _record_cpu_upload_plan(self, upload_plan: dict[str, bool]) -> None:
        self.last_cpu_cell_state_upload_skipped = not upload_plan["cell_core"]
        self.last_cpu_island_id_upload_skipped = not upload_plan["island_id"]
        self.last_cpu_entity_id_upload_skipped = not upload_plan["entity_id"]
        self.last_cpu_displaced_material_upload_skipped = not upload_plan["placeholder_displaced_material"]
        self.last_cpu_flow_velocity_upload_skipped = not upload_plan["flow_velocity"]
        self.last_cpu_ambient_upload_skipped = not upload_plan["ambient_temperature"]
        self.last_cpu_active_upload_skipped = not upload_plan["active_tile_ttl"]

    def _load_authoritative_active_tile_mask(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU motion pipeline requires bridge active scheduler resources")
        program = self.programs["load_active_tiles"]
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["expansion_radius"].value = int(expansion_radius)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        resources.active_tile_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.active.tile_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.active.tile_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)

    def _upload_material_rule_params(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        world.bridge.sync_rule_tables(world)
        material_table = world.bridge.shadow_typed_tables["material_table"]
        table_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        if resources.material_params_signature == table_signature:
            return
        params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        contact = np.zeros((MAX_MATERIALS, 4), dtype="f4")
        falling = np.zeros((MAX_MATERIALS, 2, 4), dtype="f4")
        count = min(MAX_MATERIALS, material_table.shape[0])
        params[:count, 0] = material_table[:count]["max_dda_step"].astype("f4")
        params[:count, 1] = material_table[:count]["gravity_scale"].astype("f4")
        params[:count, 2] = material_table[:count]["wind_coupling"].astype("f4")
        params[:count, 3] = material_table[:count]["drag_scale"].astype("f4")
        contact[:count, 0] = material_table[:count]["friction"].astype("f4")
        contact[:count, 1] = material_table[:count]["elasticity"].astype("f4")
        contact[:count, 2] = material_table[:count]["powder_solver_kind_id"].astype("f4")
        falling[:count, 0, 0] = material_table[:count]["default_phase"].astype("f4")
        falling[:count, 0, 1] = material_table[:count]["render_group_id"].astype("f4")
        falling[:count, 0, 2] = material_table[:count]["falling_island_break_kind_id"].astype("f4")
        falling[:count, 0, 3] = material_table[:count]["powder_generation_id"].astype("f4")
        falling[:count, 1, 0] = material_table[:count]["base_integrity"].astype("f4")
        falling[:count, 1, 1] = material_table[:count]["spawn_temperature"].astype("f4")
        resources.material_params.write(params.tobytes())
        resources.material_contact_params.write(contact.tobytes())
        resources.material_falling_params.write(falling.reshape((MAX_MATERIALS * 2, 4)).tobytes())
        resources.material_params_signature = table_signature

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _bridge_context_active(self, world: "WorldEngine") -> bool:
        return world.bridge.ctx is not None and world.bridge.ctx is self._active_context(world)

    def _active_context(self, world: "WorldEngine") -> Any:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        return ctx

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        group_x: int,
        group_y: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative input state")
        if not self._bridge_context_active(world):
            raise RuntimeError("GPU motion pipeline cannot consume authoritative bridge state from a separate GL context")

        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        copy_flow = "flow_velocity" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_flow or copy_ambient):
            return

        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.velocity_tex.bind_to_image(3, read=False, write=True)
            resources.temp_tex.bind_to_image(4, read=False, write=True)
            resources.timer_tex.bind_to_image(5, read=False, write=True)
            resources.integrity_tex.bind_to_image(6, read=False, write=True)
            program.run(group_x, group_y, 1)

        if copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.island_id_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_tex.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)

        if copy_flow or copy_ambient:
            gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            program = self.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["copy_flow_velocity"].value = bool(copy_flow)
            program["copy_ambient"].value = bool(copy_ambient)
            bridge.textures["flow_velocity"].use(location=0)
            bridge.textures["ambient_temperature"].use(location=1)
            resources.flow_tex.bind_to_image(2, read=False, write=True)
            resources.ambient_tex.bind_to_image(3, read=False, write=True)
            program.run(gas_group_x, gas_group_y, 1)

        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        output_textures: bool,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative output state")
            return
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot publish authoritative state from a separate GL context")
            return

        material_tex = resources.material_out_tex if output_textures else resources.material_tex
        phase_tex = resources.phase_out_tex if output_textures else resources.phase_tex
        flags_tex = resources.cell_flags_out_tex if output_textures else resources.cell_flags_tex
        velocity_tex = resources.velocity_out_tex
        temp_tex = resources.temp_out_tex if output_textures else resources.temp_tex
        timer_tex = resources.timer_out_tex if output_textures else resources.timer_tex
        integrity_tex = resources.integrity_out_tex if output_textures else resources.integrity_tex
        island_tex = resources.island_id_out_tex if output_textures else resources.island_id_tex
        entity_tex = resources.entity_id_out_tex if output_textures else resources.entity_id_tex
        displaced_tex = resources.displaced_out_tex if output_textures else resources.displaced_tex

        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        material_tex.use(location=0)
        phase_tex.use(location=1)
        flags_tex.use(location=2)
        velocity_tex.use(location=3)
        temp_tex.use(location=4)
        timer_tex.use(location=5)
        integrity_tex.use(location=6)
        island_tex.use(location=7)
        entity_tex.use(location=8)
        displaced_tex.use(location=9)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    def _publish_bridge_island_id(self, world: "WorldEngine", resources: GPUMotionResources, island_tex: Any) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for authoritative island state")
            return
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot publish island state from a separate GL context")
            return
        program = self.programs["publish_bridge_island_id"]
        program["cell_grid_size"].value = (world.width, world.height)
        island_tex.use(location=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=0)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("island_id")

    def publish_bridge_falling_island_reservations(
        self,
        world: "WorldEngine",
        reservation_count: int,
    ) -> bool:
        reservation_count = int(reservation_count)
        if reservation_count < 0:
            raise ValueError("reservation_count must be non-negative")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island reservations")
            return False
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot publish island reservations from a separate GL context")
            return False
        resources = self._ensure_resources(world)
        required_bytes = max(4, reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["island_reservation"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_reservation"] = bridge_buffer
        else:
            bridge_buffer.orphan(required_bytes)
        if reservation_count > 0:
            bridge.ctx.copy_buffer(
                bridge_buffer,
                resources.island_reservations,
                size=reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
            )
        bridge.buffers["island_reservation_count"].write(np.array([reservation_count], dtype=np.int32).tobytes())
        if self._formal_gpu_frame(world):
            bridge.mark_gpu_authoritative("island_reservation")
        return True

    def publish_bridge_powder_reservations(
        self,
        world: "WorldEngine",
        reservation_capacity: int,
    ) -> bool:
        reservation_capacity = int(reservation_capacity)
        if reservation_capacity < 0:
            raise ValueError("reservation_capacity must be non-negative")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for powder reservations")
            return False
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot publish powder reservations from a separate GL context")
            return False
        resources = self._ensure_resources(world)
        required_bytes = max(4, reservation_capacity * POWDER_RESERVATION_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["powder_reservation"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["powder_reservation"] = bridge_buffer
        if self._formal_gpu_frame(world):
            program = self.programs["publish_powder_reservations"]
            program["reservation_capacity"].value = reservation_capacity
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            bridge_buffer.bind_to_storage_buffer(binding=2)
            bridge.buffers["powder_reservation_count"].bind_to_storage_buffer(binding=3)
            program.run((reservation_capacity + 255) // 256, 1, 1)
            bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
            bridge.mark_gpu_authoritative("powder_reservation")
            return True
        else:
            bridge_buffer.orphan(required_bytes)
        if reservation_capacity > 0:
            bridge.ctx.copy_buffer(
                bridge_buffer,
                resources.powder_reservations,
                size=reservation_capacity * POWDER_RESERVATION_DTYPE.itemsize,
            )
        bridge.ctx.copy_buffer(bridge.buffers["powder_reservation_count"], resources.powder_reservation_count, size=4)
        return True

    def publish_bridge_falling_island_runtime_from_reservations(
        self,
        world: "WorldEngine",
        reservation_count: int,
    ) -> bool:
        reservation_count = int(reservation_count)
        if reservation_count < 0:
            raise ValueError("reservation_count must be non-negative")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime")
            return False
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot publish island runtime from a separate GL context")
            return False
        self._ensure_programs(bridge.ctx)
        resources = self._ensure_resources(world)
        required_bytes = max(4, reservation_count * ISLAND_RUNTIME_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["island_runtime"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_runtime"] = bridge_buffer
        else:
            bridge_buffer.orphan(required_bytes)
        bridge_buffer.write(np.zeros((required_bytes,), dtype=np.uint8).tobytes())
        bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())
        if reservation_count > 0:
            program = self.programs["publish_falling_island_runtime"]
            program["reservation_count"].value = int(reservation_count)
            program["cell_grid_size"].value = (world.width, world.height)
            program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
            program["paging_buffer_origin"].value = (
                int(world.paging.buffer_origin_x),
                int(world.paging.buffer_origin_y),
            )
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            bridge_buffer.bind_to_storage_buffer(binding=1)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
            group_x = (reservation_count + LOCAL_SIZE - 1) // LOCAL_SIZE
            program.run(group_x, 1, 1)
            bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
        self.last_published_island_runtime_capacity = reservation_count
        if self._formal_gpu_frame(world):
            bridge.mark_gpu_authoritative("island_runtime")
        return True

    def seed_bridge_falling_island_runtime_from_cpu(self, world: "WorldEngine") -> int:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime seeding")
            return 0
        if not self._bridge_context_active(world):
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU motion pipeline cannot seed island runtime from a separate GL context")
            return 0
        runtime = pack_island_runtime_upload(world)
        runtime_count = int(runtime.shape[0])
        required_bytes = max(4, runtime.nbytes)
        bridge_buffer = bridge.buffers["island_runtime"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_runtime"] = bridge_buffer
        else:
            bridge_buffer.orphan(required_bytes)
        if runtime.nbytes > 0:
            bridge_buffer.write(runtime.tobytes())
        bridge.buffers["island_runtime_count"].write(np.array([runtime_count], dtype=np.int32).tobytes())
        self.last_published_island_runtime_capacity = runtime_count
        if self._formal_gpu_frame(world):
            bridge.mark_gpu_authoritative("island_runtime")
        return runtime_count

    def _copy_scalar_texture(self, ctx: Any, source_tex: Any, dest_tex: Any, width: int, height: int) -> None:
        program = self.programs["copy_scalar_texture"]
        program["grid_size"].value = (int(width), int(height))
        source_tex.use(location=0)
        dest_tex.bind_to_image(1, read=False, write=True)
        group_x = (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(ctx, "SHADER_IMAGE_ACCESS_BARRIER_BIT", 0)
            | getattr(ctx, "TEXTURE_FETCH_BARRIER_BIT", 0)
        )

    def integrate_velocity(
        self,
        world: "WorldEngine",
        dt: float,
        *,
        solve_tile_mask: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        program = self.programs["integrate_velocity"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["gas_cell_size"].value = world.gas_cell_size
        program["dt"].value = dt
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.velocity_tex.use(location=2)
        resources.flow_tex.use(location=3)
        resources.active_tile_tex.use(location=4)
        resources.velocity_out_tex.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        self._publish_bridge_outputs(world, resources, output_textures=False)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            world.velocity[:] = self._download_velocity_output(world, resources)

    def _run_powder_targets(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["powder_targets"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_powder"].value = int(Phase.POWDER)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.phase_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.active_tile_tex.use(location=4)
        resources.powder_target_tex.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)

    def _download_outputs(self, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
        return np.rint(
            np.frombuffer(resources.powder_target_tex.read(), dtype="f4").reshape((world.height, world.width, 2))
        ).astype(np.int32)

    def _download_velocity_output(self, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
        return np.frombuffer(resources.velocity_out_tex.read(), dtype="f4").reshape(world.velocity.shape)

    def plan_powder_reservations(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
        solve_cell_mask: np.ndarray,
    ) -> np.ndarray:
        powder_targets = self.step(world, solve_tile_mask=solve_tile_mask)
        reservations = self._build_powder_reservations(world, solve_cell_mask, powder_targets)
        self.upload_powder_reservations(world, reservations)
        return reservations

    def upload_powder_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        resources = self._ensure_resources(world)
        self._write_dynamic_buffer(ctx, resources, "powder_reservations", reservations)
        resources.powder_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())

    def resolve_and_apply_powders(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        if self._formal_gpu_frame(world):
            self._dispatch_apply_powder_fast_path(world, resources, group_x, group_y)
            return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
        self._run_powder_targets(world, resources, group_x, group_y)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "powder_reservations", world.width * world.height * POWDER_RESERVATION_DTYPE.itemsize)
        resources.powder_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
        program = self.programs["resolve_powder_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_powder"].value = int(Phase.POWDER)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.material_params.bind_to_storage_buffer(binding=2)
        resources.material_contact_params.bind_to_storage_buffer(binding=3)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.velocity_tex.use(location=2)
        resources.active_tile_tex.use(location=3)
        resources.powder_target_tex.use(location=4)
        resources.material_out_tex.bind_to_image(5, read=True, write=True)
        program.run(1, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world):
            self.publish_bridge_powder_reservations(world, world.width * world.height)
            self._dispatch_apply_powder_reservations(world, resources, None)
            return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
        reservation_count = int(np.frombuffer(resources.powder_reservation_count.read(size=4), dtype=np.int32, count=1)[0])
        reservation_count = max(0, min(reservation_count, world.width * world.height))
        self._dispatch_apply_powder_reservations(world, resources, reservation_count)
        return self._read_powder_reservations(resources, reservation_count)

    def _dispatch_apply_powder_fast_path(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        group_x: int,
        group_y: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["apply_powder_fast_path"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_powder"].value = int(Phase.POWDER)
        program["max_powder_step"].value = 3
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.cell_flags_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.temp_tex.use(location=4)
        resources.timer_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.active_tile_tex.use(location=10)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.timer_out_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_outputs(world, resources, output_textures=True)
        self.publish_bridge_powder_reservations(world, 0)
        self.last_cpu_mirror_downloaded = False

    def apply_powder_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> bool:
        ctx = world.bridge.ctx
        if ctx is None:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.upload_powder_reservations(world, reservations)
        self._dispatch_apply_powder_reservations(world, resources, int(len(reservations)))
        return True

    def apply_falling_island_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or len(reservations) == 0:
            return False
        moving = np.any(reservations["resolved_shift"] != 0, axis=1)
        if not bool(np.any(moving)):
            self.upload_falling_island_reservations(world, reservations)
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.upload_falling_island_reservations(world, reservations)
        self._dispatch_apply_falling_island_reservations(world, resources, int(len(reservations)))
        return True

    def apply_uploaded_falling_island_reservations(self, world: "WorldEngine", reservation_count: int) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or int(reservation_count) <= 0:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        resources.island_reservation_count.write(np.array([int(reservation_count)], dtype=np.int32).tobytes())
        self._dispatch_apply_falling_island_reservations(world, resources, int(reservation_count))
        return True

    def shed_falling_island_fragments(self, world: "WorldEngine") -> bool:
        ctx = world.bridge.ctx
        if ctx is None:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.upload_falling_island_reservations(world, np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE))
        self._dispatch_apply_falling_island_materialization(world, resources, reservation_count=0, mode=0)
        return True

    def apply_falling_island_settlements(self, world: "WorldEngine", reservations: np.ndarray) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or len(reservations) == 0:
            return False
        settling = (
            (reservations["resolve_state"] != ISLAND_RESOLVE_STALE)
            & np.any(reservations["target_shift"] != 0, axis=1)
            & ~np.any(reservations["resolved_shift"] != 0, axis=1)
        )
        if not bool(np.any(settling)):
            self.upload_falling_island_reservations(world, reservations)
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.upload_falling_island_reservations(world, reservations)
        self._dispatch_apply_falling_island_materialization(world, resources, int(len(reservations)), mode=1)
        return True

    def apply_uploaded_falling_island_settlements(self, world: "WorldEngine", reservation_count: int) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or int(reservation_count) <= 0:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        resources.island_reservation_count.write(np.array([int(reservation_count)], dtype=np.int32).tobytes())
        self._dispatch_apply_falling_island_materialization(world, resources, int(reservation_count), mode=1)
        return True

    def _dispatch_apply_powder_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        reservation_count: int | None,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._upload_powder_apply_state(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        program = self.programs["apply_powder_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
        program["use_reservation_count_buffer"].value = reservation_count is None
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.cell_flags_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.temp_tex.use(location=4)
        resources.timer_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.timer_out_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        aux_program = self.programs["apply_powder_reservation_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
        aux_program["use_reservation_count_buffer"].value = reservation_count is None
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        aux_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_outputs(world, resources, output_textures=True)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_powder_apply_state(world, resources)

    def _dispatch_apply_falling_island_materialization(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        reservation_count: int,
        *,
        mode: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._upload_powder_apply_state(world, resources)
        self._upload_material_rule_params(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        program = self.programs["apply_falling_island_materialization"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["reservation_count"].value = int(reservation_count)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["phase_powder"].value = int(Phase.POWDER)
        program["phase_static_solid"].value = int(Phase.STATIC_SOLID)
        program["mode"].value = int(mode)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_falling_params.bind_to_storage_buffer(binding=1)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.cell_flags_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.temp_tex.use(location=4)
        resources.timer_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.timer_out_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        aux_program = self.programs["apply_falling_island_materialization_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["reservation_count"].value = int(reservation_count)
        aux_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        aux_program["mode"].value = int(mode)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_falling_params.bind_to_storage_buffer(binding=1)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        aux_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_outputs(world, resources, output_textures=True)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_powder_apply_state(world, resources)

    def _dispatch_apply_falling_island_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._upload_powder_apply_state(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        program = self.programs["apply_falling_island_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["reservation_count"].value = int(reservation_count)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.cell_flags_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.temp_tex.use(location=4)
        resources.timer_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.ambient_tex.use(location=20)
        resources.material_out_tex.bind_to_image(0, read=False, write=True)
        resources.phase_out_tex.bind_to_image(1, read=False, write=True)
        resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(3, read=False, write=True)
        resources.temp_out_tex.bind_to_image(4, read=False, write=True)
        resources.timer_out_tex.bind_to_image(5, read=False, write=True)
        resources.integrity_out_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        aux_program = self.programs["apply_falling_island_reservation_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["reservation_count"].value = int(reservation_count)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        aux_program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        self._publish_bridge_outputs(world, resources, output_textures=True)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_powder_apply_state(world, resources)

    def _read_powder_reservations(self, resources: GPUMotionResources, reservation_count: int) -> np.ndarray:
        if reservation_count <= 0:
            return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
        return np.frombuffer(
            resources.powder_reservations.read(size=reservation_count * POWDER_RESERVATION_DTYPE.itemsize),
            dtype=POWDER_RESERVATION_DTYPE,
            count=reservation_count,
        ).copy()

    def _upload_powder_apply_state(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
            resources.phase_tex.write(world.phase.astype("f4").tobytes())
            resources.cell_flags_tex.write(world.cell_flags.astype("f4").tobytes())
            resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
            resources.temp_tex.write(world.cell_temperature.astype("f4").tobytes())
            resources.timer_tex.write(world.timer_pack.astype("f4").tobytes())
            resources.integrity_tex.write(world.integrity.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        if upload_plan["entity_id"]:
            resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
        if upload_plan["placeholder_displaced_material"]:
            resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
        if upload_plan["ambient_temperature"]:
            resources.ambient_tex.write(world.ambient_temperature.astype("f4").tobytes())

    def _download_powder_apply_state(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        world.material_id[:] = np.rint(
            np.frombuffer(resources.material_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(resources.phase_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_flags[:] = np.rint(
            np.frombuffer(resources.cell_flags_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.velocity[:] = np.frombuffer(resources.velocity_out_tex.read(), dtype="f4").reshape(world.velocity.shape)
        world.cell_temperature[:] = np.frombuffer(resources.temp_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_out_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        world.integrity[:] = np.frombuffer(resources.integrity_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.entity_id[:] = np.rint(
            np.frombuffer(resources.entity_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.placeholder_displaced_material[:] = np.rint(
            np.frombuffer(resources.displaced_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)

    def _build_powder_reservations(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
        powder_targets: np.ndarray,
    ) -> np.ndarray:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        reservations: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[float, float], int]] = []
        for y in range(world.height - 2, -1, -1):
            active_xs = np.flatnonzero(solve_cell_mask[y])
            if active_xs.size == 0:
                continue
            for x in active_xs.tolist():
                material_id = int(world.material_id[y, x])
                if material_id <= 0 or int(world.phase[y, x]) != int(Phase.POWDER):
                    continue
                max_step = 0
                if material_id < material_table.shape[0]:
                    max_step = int(material_table[material_id]["max_dda_step"])
                velocity = world.velocity[y, x]
                desired_dx = int(np.clip(np.rint(float(velocity[0])), -max_step, max_step))
                desired_dy = int(np.clip(np.rint(float(velocity[1])), -max_step, max_step))
                reservations.append(
                    (
                        (int(x), int(y)),
                        (int(x) + desired_dx, int(y) + desired_dy),
                        (int(powder_targets[y, x, 0]), int(powder_targets[y, x, 1])),
                        (int(x), int(y)),
                        (float(velocity[0]), float(velocity[1])),
                        material_id,
                        POWDER_RESOLVE_BLOCKED,
                    )
                )
        packed = np.zeros((len(reservations),), dtype=POWDER_RESERVATION_DTYPE)
        for index, (
            source_xy,
            desired_target_xy,
            reserved_target_xy,
            resolved_target_xy,
            velocity_xy,
            material_id,
            resolve_state,
        ) in enumerate(reservations):
            packed[index]["source_xy"] = np.asarray(source_xy, dtype=np.int32)
            packed[index]["desired_target_xy"] = np.asarray(desired_target_xy, dtype=np.int32)
            packed[index]["reserved_target_xy"] = np.asarray(reserved_target_xy, dtype=np.int32)
            packed[index]["resolved_target_xy"] = np.asarray(resolved_target_xy, dtype=np.int32)
            packed[index]["velocity_xy"] = np.asarray(velocity_xy, dtype=np.float32)
            packed[index]["material_id"] = int(material_id)
            packed[index]["resolve_state"] = int(resolve_state)
        return packed

    def plan_uploaded_falling_island_reservations(
        self,
        world: "WorldEngine",
        *,
        island_ids: list[int] | None = None,
        motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> int:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        runtime = pack_island_runtime_upload(world)
        if island_ids is not None:
            wanted = set(int(island_id) for island_id in island_ids)
            runtime = runtime[np.isin(runtime["island_id"], np.fromiter(wanted, dtype=np.int32, count=len(wanted)))]
        if motion_overrides:
            for island_id, (velocity_xy, subcell_offset) in motion_overrides.items():
                matches = np.nonzero(runtime["island_id"] == int(island_id))[0]
                if matches.size == 0:
                    continue
                index = int(matches[0])
                runtime[index]["velocity_xy"] = np.asarray(velocity_xy, dtype=np.float32)
                runtime[index]["subcell_offset"] = np.asarray(subcell_offset, dtype=np.float32)
        if runtime.size == 0:
            reservations = np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
            self.upload_falling_island_reservations(world, reservations)
            return 0
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
        packed_ids = np.ascontiguousarray(runtime["island_id"].astype(np.int32))
        packed_bboxes = np.ascontiguousarray(runtime["buffer_bbox"].astype(np.int32))
        packed_motion = np.zeros((runtime.shape[0], 4), dtype=np.float32)
        packed_motion[:, :2] = runtime["velocity_xy"]
        packed_motion[:, 2:] = runtime["subcell_offset"]
        packed_shifts = np.zeros((runtime.shape[0], 4), dtype=np.int32)
        empty_reservations = np.zeros((runtime.shape[0],), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
        self._write_dynamic_buffer(ctx, resources, "island_ids", packed_ids)
        self._write_dynamic_buffer(ctx, resources, "island_bboxes", packed_bboxes)
        self._write_dynamic_buffer(ctx, resources, "island_motion", packed_motion)
        self._write_dynamic_buffer(ctx, resources, "island_shift_results", packed_shifts)
        self._write_dynamic_buffer(ctx, resources, "island_reservations", empty_reservations)
        resources.island_reservation_count.write(np.array([int(runtime.shape[0])], dtype=np.int32).tobytes())
        program = self.programs["island_shifts"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["island_count"].value = int(runtime.shape[0])
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        group_x = (runtime.shape[0] + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        pack_program = self.programs["pack_falling_island_reservations"]
        pack_program["island_count"].value = int(runtime.shape[0])
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        resources.island_reservations.bind_to_storage_buffer(binding=4)
        resources.island_reservation_count.bind_to_storage_buffer(binding=5)
        pack_program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        return int(runtime.shape[0])

    def plan_uploaded_falling_island_reservations_from_bridge_runtime(
        self,
        world: "WorldEngine",
        runtime_capacity: int,
    ) -> int:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        runtime_capacity = int(runtime_capacity)
        if runtime_capacity <= 0:
            self.upload_falling_island_reservations(
                world,
                np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE),
            )
            return 0
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU motion pipeline requires bridge GPU resources for island runtime planning")
        if self._formal_gpu_frame(world) and "island_runtime" not in bridge.gpu_authoritative_resources:
            raise RuntimeError("GPU motion pipeline requires GPU-authoritative island_runtime for bridge runtime planning")
        if not self._bridge_context_active(world):
            raise RuntimeError("GPU motion pipeline cannot consume island runtime from a separate GL context")

        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        packed_ids = np.zeros((runtime_capacity,), dtype=np.int32)
        packed_bboxes = np.zeros((runtime_capacity, 4), dtype=np.int32)
        packed_motion = np.zeros((runtime_capacity, 4), dtype=np.float32)
        packed_shifts = np.zeros((runtime_capacity, 4), dtype=np.int32)
        empty_reservations = np.zeros((runtime_capacity,), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
        self._write_dynamic_buffer(ctx, resources, "island_ids", packed_ids)
        self._write_dynamic_buffer(ctx, resources, "island_bboxes", packed_bboxes)
        self._write_dynamic_buffer(ctx, resources, "island_motion", packed_motion)
        self._write_dynamic_buffer(ctx, resources, "island_shift_results", packed_shifts)
        self._write_dynamic_buffer(ctx, resources, "island_reservations", empty_reservations)
        resources.island_reservation_count.write(np.array([runtime_capacity], dtype=np.int32).tobytes())

        group_x = (runtime_capacity + LOCAL_SIZE - 1) // LOCAL_SIZE
        unpack_program = self.programs["unpack_bridge_island_runtime"]
        unpack_program["runtime_count"].value = runtime_capacity
        bridge.buffers["island_runtime"].bind_to_storage_buffer(binding=0)
        resources.island_ids.bind_to_storage_buffer(binding=1)
        resources.island_bboxes.bind_to_storage_buffer(binding=2)
        resources.island_motion.bind_to_storage_buffer(binding=3)
        unpack_program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
        program = self.programs["island_shifts"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["island_count"].value = runtime_capacity
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        pack_program = self.programs["pack_falling_island_reservations"]
        pack_program["island_count"].value = runtime_capacity
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        resources.island_reservations.bind_to_storage_buffer(binding=4)
        resources.island_reservation_count.bind_to_storage_buffer(binding=5)
        pack_program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        return runtime_capacity

    def plan_falling_island_reservations(
        self,
        world: "WorldEngine",
        *,
        island_ids: list[int] | None = None,
        motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        reservation_count = self.plan_uploaded_falling_island_reservations(
            world,
            island_ids=island_ids,
            motion_overrides=motion_overrides,
        )
        resources = self._ensure_resources(world)
        return self._read_falling_island_reservations(resources, reservation_count)

    def upload_falling_island_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            return
        resources = self._ensure_resources(world)
        self._write_dynamic_buffer(ctx, resources, "island_reservations", reservations)
        resources.island_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())

    def resolve_falling_island_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        if len(reservations) == 0:
            self.upload_falling_island_reservations(world, reservations)
            return reservations
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._write_dynamic_buffer(ctx, resources, "island_reservations", reservations)
        resources.island_reservation_count.write(np.array([len(reservations)], dtype=np.int32).tobytes())
        self._dispatch_resolve_falling_island_reservations(world, resources, int(len(reservations)))
        self.publish_bridge_falling_island_reservations(world, int(len(reservations)))
        self.publish_bridge_falling_island_runtime_from_reservations(world, int(len(reservations)))
        resolved = self._read_falling_island_reservations(resources, int(len(reservations)))
        resources.island_reservation_count.write(np.array([len(resolved)], dtype=np.int32).tobytes())
        return resolved

    def resolve_uploaded_falling_island_reservations(self, world: "WorldEngine", reservation_count: int) -> bool:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        reservation_count = int(reservation_count)
        if reservation_count <= 0:
            resources = self._ensure_resources(world)
            resources.island_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        resources.island_reservation_count.write(np.array([reservation_count], dtype=np.int32).tobytes())
        self._dispatch_resolve_falling_island_reservations(world, resources, reservation_count)
        self.publish_bridge_falling_island_reservations(world, reservation_count)
        self.publish_bridge_falling_island_runtime_from_reservations(world, reservation_count)
        return True

    def _dispatch_resolve_falling_island_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._upload_material_rule_params(world, resources)
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
        self._copy_scalar_texture(ctx, resources.material_tex, resources.material_out_tex, world.width, world.height)
        program = self.programs["resolve_falling_island_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["reservation_count"].value = int(reservation_count)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.material_out_tex.bind_to_image(2, read=True, write=True)
        for rank in range(int(reservation_count)):
            program["process_rank"].value = int(rank)
            program.run(1, 1, 1)
            ctx.memory_barrier(
                ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
                | ctx.SHADER_STORAGE_BARRIER_BIT
                | ctx.TEXTURE_FETCH_BARRIER_BIT
            )
            if not self._formal_gpu_frame(world):
                ctx.finish()

    def _read_falling_island_reservations(self, resources: GPUMotionResources, reservation_count: int) -> np.ndarray:
        reservation_count = int(reservation_count)
        if reservation_count <= 0:
            return np.zeros((0,), dtype=FALLING_ISLAND_RESERVATION_DTYPE)
        return np.frombuffer(
            resources.island_reservations.read(size=reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize),
            dtype=FALLING_ISLAND_RESERVATION_DTYPE,
            count=reservation_count,
        ).copy()

    def label_falling_island_components(
        self,
        world: "WorldEngine",
        island_id: int,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        labels, _metadata = self.label_falling_island_component_metadata(world, island_id, bbox)
        return labels

    def label_falling_island_component_metadata(
        self,
        world: "WorldEngine",
        island_id: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        label_texture, metadata = self.label_falling_island_component_metadata_texture(world, island_id, bbox)
        x0, y0, x1, y1 = bbox
        labels = np.rint(
            np.frombuffer(label_texture.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        return labels[max(0, y0):min(world.height, y1), max(0, x0):min(world.width, x1)].copy(), metadata

    def label_falling_island_component_metadata_texture(
        self,
        world: "WorldEngine",
        island_id: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[Any, np.ndarray]:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
            resources.phase_tex.write(world.phase.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)

        init_program = self.programs["island_component_init"]
        init_program["cell_grid_size"].value = (world.width, world.height)
        init_program["target_island_id"].value = int(island_id)
        init_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=2)
        resources.component_label_ping.bind_to_image(3, read=False, write=True)
        init_program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

        current = resources.component_label_ping
        scratch = resources.component_label_pong
        propagate = self.programs["island_component_propagate"]
        propagate["cell_grid_size"].value = (world.width, world.height)
        if self._formal_gpu_frame(world):
            x0, y0, x1, y1 = bbox
            clipped_width = max(0, min(world.width, int(x1)) - max(0, int(x0)))
            clipped_height = max(0, min(world.height, int(y1)) - max(0, int(y0)))
            pass_count = max(1, clipped_width + clipped_height)
            resources.component_change_flag.bind_to_storage_buffer(binding=0)
            for _ in range(pass_count):
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                propagate.run(group_x, group_y, 1)
                ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
                current, scratch = scratch, current
        else:
            while True:
                resources.component_change_flag.write(np.zeros(1, dtype=np.uint32).tobytes())
                current.use(location=0)
                scratch.bind_to_image(1, read=False, write=True)
                resources.component_change_flag.bind_to_storage_buffer(binding=0)
                propagate.run(group_x, group_y, 1)
                ctx.finish()
                changed = bool(np.frombuffer(resources.component_change_flag.read(size=4), dtype=np.uint32, count=1)[0])
                current, scratch = scratch, current
                if not changed:
                    break

        metadata = self._summarize_falling_island_label_texture(world, current)
        return current, metadata

    def _summarize_falling_island_label_texture(self, world: "WorldEngine", label_texture: Any) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        resources = self._ensure_resources(world)
        cell_count = int(world.width * world.height)
        metadata = np.zeros((cell_count, 5), dtype=np.int32)
        metadata[:, 0] = int(world.width)
        metadata[:, 1] = int(world.height)
        self._write_dynamic_buffer(ctx, resources, "component_metadata", metadata)
        program = self.programs["summarize_falling_island_components"]
        program["cell_grid_size"].value = (world.width, world.height)
        label_texture.use(location=0)
        resources.component_metadata.bind_to_storage_buffer(binding=0)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        ctx.finish()
        summarized = np.frombuffer(
            resources.component_metadata.read(size=metadata.nbytes),
            dtype=np.int32,
            count=metadata.size,
        ).reshape((cell_count, 5))
        active_indices = np.flatnonzero(summarized[:, 4] > 0)
        if active_indices.size == 0:
            return np.zeros((0, 6), dtype=np.int32)
        labeled_metadata = np.zeros((int(active_indices.size), 6), dtype=np.int32)
        labeled_metadata[:, 0] = active_indices.astype(np.int32, copy=False) + 1
        labeled_metadata[:, 1:] = summarized[active_indices]
        return labeled_metadata

    def relabel_falling_island_components(
        self,
        world: "WorldEngine",
        labels: np.ndarray,
        component_labels: np.ndarray,
        component_island_ids: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or labels.size == 0 or component_labels.size == 0:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        full_labels = np.zeros((world.height, world.width), dtype=np.float32)
        x0, y0, x1, y1 = bbox
        clipped_x0 = max(0, int(x0))
        clipped_y0 = max(0, int(y0))
        clipped_x1 = min(world.width, int(x1))
        clipped_y1 = min(world.height, int(y1))
        if clipped_x0 >= clipped_x1 or clipped_y0 >= clipped_y1:
            return False
        label_height = clipped_y1 - clipped_y0
        label_width = clipped_x1 - clipped_x0
        full_labels[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = labels[:label_height, :label_width].astype(
            np.float32,
            copy=False,
        )
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        resources.component_label_ping.write(full_labels.tobytes())
        self._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
        self._write_dynamic_buffer(
            ctx,
            resources,
            "component_island_ids",
            component_island_ids.astype(np.int32, copy=False),
        )
        program = self.programs["relabel_falling_island_components"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["component_count"].value = int(component_labels.size)
        resources.island_id_tex.use(location=0)
        resources.component_label_ping.use(location=1)
        resources.island_id_out_tex.bind_to_image(2, read=False, write=True)
        resources.component_labels.bind_to_storage_buffer(binding=0)
        resources.component_island_ids.bind_to_storage_buffer(binding=1)
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if not self.last_cpu_mirror_downloaded:
            self._publish_bridge_island_id(world, resources, resources.island_id_out_tex)
            return True
        ctx.finish()
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        return True

    def relabel_falling_island_component_texture(
        self,
        world: "WorldEngine",
        label_texture: Any,
        component_labels: np.ndarray,
        component_island_ids: np.ndarray,
    ) -> bool:
        ctx = world.bridge.ctx
        if ctx is None or component_labels.size == 0:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        self._write_dynamic_buffer(ctx, resources, "component_labels", component_labels.astype(np.int32, copy=False))
        self._write_dynamic_buffer(
            ctx,
            resources,
            "component_island_ids",
            component_island_ids.astype(np.int32, copy=False),
        )
        program = self.programs["relabel_falling_island_components"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["component_count"].value = int(component_labels.size)
        resources.island_id_tex.use(location=0)
        label_texture.use(location=1)
        resources.island_id_out_tex.bind_to_image(2, read=False, write=True)
        resources.component_labels.bind_to_storage_buffer(binding=0)
        resources.component_island_ids.bind_to_storage_buffer(binding=1)
        program.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if not self.last_cpu_mirror_downloaded:
            self._publish_bridge_island_id(world, resources, resources.island_id_out_tex)
            return True
        ctx.finish()
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_id_out_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        return True

    def resolve_falling_island_shifts(
        self,
        world: "WorldEngine",
        *,
        island_ids: list[int] | None = None,
        motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> dict[int, tuple[int, int]]:
        reservations = self.plan_falling_island_reservations(
            world,
            island_ids=island_ids,
            motion_overrides=motion_overrides,
        )
        return {
            int(record["island_id"]): (int(record["reserved_shift"][0]), int(record["reserved_shift"][1]))
            for record in reservations
        }

    def _write_dynamic_buffer(self, ctx: Any, resources: GPUMotionResources, name: str, data: np.ndarray) -> None:
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

    def _ensure_dynamic_buffer_capacity(self, ctx: Any, resources: GPUMotionResources, name: str, nbytes: int) -> None:
        buffer = getattr(resources, name)
        required = max(4, int(nbytes))
        if buffer.size < required:
            buffer.release()
            setattr(resources, name, ctx.buffer(reserve=required, dynamic=True))
        else:
            buffer.orphan(required)
