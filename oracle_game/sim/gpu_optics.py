from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_light_id


LOCAL_SIZE = 8
MAX_LIGHTS = 8
MAX_MATERIALS = 256
MAX_EMITTERS = 256
MAX_RAY_STACK = 64
ACTIVE_CELL_TEXTURE_UNIT = 6
ACTIVE_GAS_TEXTURE_UNIT = 7


@dataclass(slots=True)
class GPUOpticsResources:
    signature: tuple[int, int, int, int, int]
    material_tex: Any
    active_cell_tex: Any
    active_gas_tex: Any
    cell_dose: Any
    gas_dose: Any
    illum_layers: Any
    visible_tex: Any
    emitter_buffer: Any
    emitter_count_buffer: Any
    light_buffer: Any
    optics_buffer: Any
    light_buffer_signature: tuple[int, int] | None = None
    optics_buffer_signature: tuple[int, int, int, int, int] | None = None


class GPUOpticsPipeline:
    def __init__(self) -> None:
        self.resources: GPUOpticsResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_active_upload_skipped = False

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def step(
        self,
        world: "WorldEngine",
        emitters: list[dict[str, object]],
        *,
        solve_cell_mask: np.ndarray | None = None,
        solve_gas_mask: np.ndarray | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        if solve_cell_mask is None:
            solve_cell_mask = np.ones((world.height, world.width), dtype=np.bool_)
        if solve_gas_mask is None:
            solve_gas_mask = np.ones((world.gas_height, world.gas_width), dtype=np.bool_)
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._upload_inputs(world, resources, emitters, solve_cell_mask=solve_cell_mask, solve_gas_mask=solve_gas_mask)
        self._run_emitter_buffer_rays(world, resources, resources.emitter_buffer, resources.emitter_count_buffer)
        if self._formal_gpu_frame(world) and "reaction_light_emitter_count" in world.bridge.gpu_authoritative_resources:
            self._run_emitter_buffer_rays(
                world,
                resources,
                world.bridge.buffers["reaction_light_emitter"],
                world.bridge.buffers["reaction_light_emitter_count"],
            )
        self._compose_visible_illumination(world, resources)
        self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources)

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.material_tex,
            self.resources.active_cell_tex,
            self.resources.active_gas_tex,
            self.resources.cell_dose,
            self.resources.gas_dose,
            self.resources.illum_layers,
            self.resources.visible_tex,
            self.resources.emitter_buffer,
            self.resources.emitter_count_buffer,
            self.resources.light_buffer,
            self.resources.optics_buffer,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUOpticsResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height, world.cell_optical_dose.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        light_count = signature[4]
        material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        active_cell_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        active_gas_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        cell_dose = ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4")
        gas_dose = ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4")
        illum_layers = ctx.texture_array((world.width, world.height, light_count), 4, dtype="f4")
        visible_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        for texture in (material_tex, active_cell_tex, active_gas_tex, cell_dose, gas_dose, illum_layers, visible_tex):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        emitter_buffer = ctx.buffer(reserve=MAX_EMITTERS * 8 * 4, dynamic=True)
        emitter_count_buffer = ctx.buffer(reserve=16 * 4, dynamic=True)
        light_buffer = ctx.buffer(reserve=MAX_LIGHTS * 2 * 4 * 4, dynamic=True)
        optics_buffer = ctx.buffer(reserve=MAX_MATERIALS * MAX_LIGHTS * 4 * 4, dynamic=True)
        self.resources = GPUOpticsResources(
            signature=signature,
            material_tex=material_tex,
            active_cell_tex=active_cell_tex,
            active_gas_tex=active_gas_tex,
            cell_dose=cell_dose,
            gas_dose=gas_dose,
            illum_layers=illum_layers,
            visible_tex=visible_tex,
            emitter_buffer=emitter_buffer,
            emitter_count_buffer=emitter_count_buffer,
            light_buffer=light_buffer,
            optics_buffer=optics_buffer,
        )
        return self.resources

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        helper = f"""
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding={ACTIVE_CELL_TEXTURE_UNIT}) uniform sampler2D active_cell_tex;
            layout(binding={ACTIVE_GAS_TEXTURE_UNIT}) uniform sampler2D active_gas_tex;
            layout(std430, binding=3) buffer OpticsBuffer {{
                vec4 optics_params[{MAX_MATERIALS * MAX_LIGHTS}];
            }};
            int material_id_at(ivec2 cell) {{
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}
            vec4 optics_at(int material_id, int light_id) {{
                return optics_params[material_id * {MAX_LIGHTS} + light_id];
            }}
            bool solve_cell_active(ivec2 cell) {{
                return texelFetch(active_cell_tex, cell, 0).x > 0.5;
            }}
            bool solve_gas_active(ivec2 gas_cell) {{
                return texelFetch(active_gas_tex, gas_cell, 0).x > 0.5;
            }}
        """
        active_helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            uniform int expansion_radius;
            uniform bool force_all_active;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=1) readonly buffer EmitterBuffer {{
                vec4 emitters[];
            }};
            layout(std430, binding=2) readonly buffer EmitterCounterBuffer {{
                uint emitter_counts[];
            }};
            layout(std430, binding=3) readonly buffer LightBuffer {{
                vec4 light_params[{MAX_LIGHTS * 2}];
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
            bool emitter_reaches_cell(ivec2 cell) {{
                uint count = min(emitter_counts[0], uint({MAX_EMITTERS}));
                for (uint emitter_index = 0u; emitter_index < count; ++emitter_index) {{
                    vec4 origin_direction = emitters[emitter_index * 2u];
                    vec4 params = emitters[emitter_index * 2u + 1u];
                    int light_id = int(params.w + 0.5);
                    int max_bounce = 0;
                    if (light_id >= 0 && light_id < {MAX_LIGHTS}) {{
                        max_bounce = max(0, int(light_params[{MAX_LIGHTS} + light_id].z + 0.5));
                    }}
                    float reach = max(0.0, params.y) * float(max_bounce + 1) + float(tile_size * expansion_radius);
                    vec2 delta = abs(vec2(cell) - origin_direction.xy);
                    if (delta.x <= reach && delta.y <= reach) {{
                        return true;
                    }}
                }}
                return false;
            }}
            bool emitter_reaches_gas_cell(ivec2 gas_cell) {{
                ivec2 cell0 = gas_cell * gas_cell_size;
                ivec2 cell1 = min(cell_grid_size - ivec2(1), cell0 + ivec2(max(gas_cell_size - 1, 0)));
                return emitter_reaches_cell(cell0) || emitter_reaches_cell(cell1);
            }}
        """
        self.programs["load_active_cell"] = ctx.compute_shader(
            active_helper
            + """
            layout(r32f, binding=4) writeonly uniform image2D active_cell_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                ivec2 tile = ivec2(
                    min(gid.x / tile_size, tile_grid_size.x - 1),
                    min(gid.y / tile_size, tile_grid_size.y - 1)
                );
                bool is_active = force_all_active || expanded_tile_active(tile) || emitter_reaches_cell(gid);
                imageStore(active_cell_img, gid, vec4(is_active ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["load_active_gas"] = ctx.compute_shader(
            active_helper
            + """
            layout(r32f, binding=4) writeonly uniform image2D active_gas_img;
            bool gas_cell_has_active_tile(ivec2 gas_cell) {
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
                bool is_active = force_all_active || gas_cell_has_active_tile(gid) || emitter_reaches_gas_cell(gid);
                imageStore(active_gas_img, gid, vec4(is_active ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["trace_emitters"] = ctx.compute_shader(
            helper
            + """
            uniform uint max_emitters;
            uniform int dose_channel_count;
            layout(std430, binding=1) readonly buffer EmitterBuffer {
                vec4 emitters[];
            };
            layout(std430, binding=2) readonly buffer EmitterCounterBuffer {
                uint emitter_counts[];
            };
            layout(r32f, binding=4) coherent uniform image2DArray cell_dose_img;
            layout(r32f, binding=5) coherent uniform image2DArray gas_dose_img;
            layout(std430, binding=4) buffer LightBuffer {
                vec4 light_params[__LIGHT_PARAM_COUNT__];
            };
            void deposit_light(ivec2 cell, int dose_channel, float energy) {
                if (solve_cell_active(cell)) {
                    ivec3 dose_coord = ivec3(cell, dose_channel);
                    float cell_dose = imageLoad(cell_dose_img, dose_coord).x;
                    imageStore(cell_dose_img, dose_coord, vec4(cell_dose + energy, 0.0, 0.0, 0.0));
                }
                ivec2 gas_cell = ivec2(cell.x / gas_cell_size, cell.y / gas_cell_size);
                if (
                    gas_cell.x >= 0
                    && gas_cell.y >= 0
                    && gas_cell.x < gas_grid_size.x
                    && gas_cell.y < gas_grid_size.y
                    && solve_gas_active(gas_cell)
                ) {
                    ivec3 gas_coord = ivec3(gas_cell, dose_channel);
                    float gas_dose = imageLoad(gas_dose_img, gas_coord).x;
                    imageStore(gas_dose_img, gas_coord, vec4(gas_dose + energy * 0.08, 0.0, 0.0, 0.0));
                }
            }
            vec2 ray_direction(vec2 direction, float spread, int ray_index, out int ray_count) {
                if (abs(direction.x) < 1.0e-5 && abs(direction.y) < 1.0e-5) {
                    ray_count = 8;
                    if (ray_index == 0) {
                        return vec2(1.0, 0.0);
                    }
                    if (ray_index == 1) {
                        return vec2(-1.0, 0.0);
                    }
                    if (ray_index == 2) {
                        return vec2(0.0, 1.0);
                    }
                    if (ray_index == 3) {
                        return vec2(0.0, -1.0);
                    }
                    if (ray_index == 4) {
                        return vec2(0.707, 0.707);
                    }
                    if (ray_index == 5) {
                        return vec2(-0.707, 0.707);
                    }
                    if (ray_index == 6) {
                        return vec2(0.707, -0.707);
                    }
                    return vec2(-0.707, -0.707);
                }
                ray_count = 3;
                float angle = atan(direction.y, direction.x);
                float offset = ray_index == 0 ? -spread : (ray_index == 2 ? spread : 0.0);
                return vec2(cos(angle + offset), sin(angle + offset));
            }
            void trace_single_ray(
                vec2 origin,
                vec2 ray_dir,
                float strength,
                int range_cells,
                int light_id,
                int dose_channel,
                int max_bounce
            ) {
                if (dose_channel < 0 || dose_channel >= dose_channel_count) {
                    return;
                }
                vec4 stack_pos_dir[%d];
                vec2 stack_energy_bounce[%d];
                int stack_count = 1;
                stack_pos_dir[0] = vec4(origin + vec2(0.5), ray_dir);
                stack_energy_bounce[0] = vec2(strength, 0.0);
                while (stack_count > 0) {
                    stack_count -= 1;
                    vec4 ray = stack_pos_dir[stack_count];
                    vec2 pos = ray.xy;
                    vec2 dir = ray.zw;
                    float energy = stack_energy_bounce[stack_count].x;
                    int bounce = int(stack_energy_bounce[stack_count].y + 0.5);
                    if (energy <= 0.02 || bounce > max_bounce) {
                        continue;
                    }
                    for (int step_index = 0; step_index < range_cells; ++step_index) {
                        if (energy <= 0.02) {
                            break;
                        }
                        pos += dir;
                        ivec2 cell = ivec2(pos);
                        if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {
                            break;
                        }
                        deposit_light(cell, dose_channel, energy);
                        int material_id = material_id_at(cell);
                        if (material_id == 0) {
                            energy *= 0.97;
                        } else {
                            vec4 opt = optics_at(material_id, light_id);
                            float absorbed = energy * opt.x;
                            float scattered = energy * opt.y;
                            float refracted = energy * opt.z;
                            float remaining = max(0.0, energy - absorbed - scattered * 0.5 - refracted * 0.25);
                            if (bounce < max_bounce) {
                                float angle = atan(dir.y, dir.x);
                                if (scattered >= 0.05 && stack_count < %d) {
                                    vec2 scatter_dir = vec2(cos(angle + 0.6), sin(angle + 0.6));
                                    stack_pos_dir[stack_count] = vec4(pos, scatter_dir);
                                    stack_energy_bounce[stack_count] = vec2(scattered * 0.75, float(bounce + 1));
                                    stack_count += 1;
                                }
                                if (refracted >= 0.05 && stack_count < %d) {
                                    vec2 refract_dir = vec2(cos(angle - 0.2), sin(angle - 0.2));
                                    stack_pos_dir[stack_count] = vec4(pos, refract_dir);
                                    stack_energy_bounce[stack_count] = vec2(refracted * 0.8, float(bounce + 1));
                                    stack_count += 1;
                                }
                            }
                            energy = remaining;
                        }
                    }
                }
            }
            void main() {
                uint emitter_count = min(emitter_counts[0], max_emitters);
                for (uint emitter_index = 0u; emitter_index < emitter_count; ++emitter_index) {
                    uint base_index = emitter_index * 2u;
                    vec4 origin_dir = emitters[base_index];
                    vec4 emitter_meta = emitters[base_index + 1u];
                    int light_id = int(emitter_meta.w + 0.5);
                    if (light_id < 0 || light_id >= __MAX_LIGHTS__) {
                        continue;
                    }
                    vec4 light_color = light_params[light_id];
                    vec4 light_meta = light_params[__MAX_LIGHTS__ + light_id];
                    int dose_channel = int(light_color.w + 0.5);
                    int max_bounce = max(0, int(light_meta.z + 0.5));
                    int default_range = max(0, int(light_meta.w + 0.5));
                    int range_cells = int(emitter_meta.y + 0.5);
                    if (range_cells <= 0) {
                        range_cells = default_range;
                    }
                    if (range_cells <= 0 || dose_channel < 0) {
                        continue;
                    }
                    int ray_count = 0;
                    for (int ray_index = 0; ray_index < 8; ++ray_index) {
                        vec2 ray_dir = ray_direction(origin_dir.zw, max(0.0, emitter_meta.z), ray_index, ray_count);
                        if (ray_index >= ray_count) {
                            break;
                        }
                        trace_single_ray(
                            origin_dir.xy,
                            ray_dir,
                            max(0.1, emitter_meta.x),
                            range_cells,
                            light_id,
                            dose_channel,
                            max_bounce
                        );
                    }
                }
            }
            """
            .replace("__LIGHT_PARAM_COUNT__", str(MAX_LIGHTS * 2))
            .replace("__MAX_LIGHTS__", str(MAX_LIGHTS))
            % (MAX_RAY_STACK, MAX_RAY_STACK, MAX_RAY_STACK, MAX_RAY_STACK)
        )
        self.programs["compose_visible"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=__LOCAL_SIZE__, local_size_y=__LOCAL_SIZE__, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int gas_cell_size;
            uniform int light_count;
            uniform int dose_channel_count;
            layout(binding=1) uniform sampler2DArray cell_dose_tex;
            layout(binding=2) uniform sampler2DArray gas_dose_tex;
            layout(rgba32f, binding=6) writeonly uniform image2D visible_img;
            layout(std430, binding=4) buffer LightBuffer {
                vec4 light_params[__LIGHT_PARAM_COUNT__];
            };
            vec3 accent_for_channel(int visual_channel) {
                int channel = ((visual_channel % 3) + 3) % 3;
                if (channel == 0) {
                    return vec3(1.0, 0.2, 0.1);
                }
                if (channel == 1) {
                    return vec3(0.25, 1.0, 0.2);
                }
                return vec3(0.1, 0.3, 1.0);
            }
            vec3 tint_for_style(int style_id) {
                if (style_id == 2) {
                    return vec3(0.85, 1.2, 0.95);
                }
                if (style_id == 3) {
                    return vec3(1.2, 0.55, 0.35);
                }
                if (style_id == 4) {
                    return vec3(0.55, 1.05, 1.25);
                }
                return vec3(1.0, 1.0, 1.0);
            }
            vec3 scales_for_style(int style_id) {
                if (style_id == 2) {
                    return vec3(0.10, 0.03, 0.07);
                }
                if (style_id == 3) {
                    return vec3(0.09, 0.025, 0.05);
                }
                if (style_id == 4) {
                    return vec3(0.09, 0.025, 0.06);
                }
                return vec3(0.11, 0.02, 0.04);
            }
            void main() {
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
                if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {
                    return;
                }
                ivec2 gas_cell = ivec2(cell.x / gas_cell_size, cell.y / gas_cell_size);
                vec3 frame = vec3(0.0);
                int count = min(light_count, __MAX_LIGHTS__);
                for (int light_id = 0; light_id < count; ++light_id) {
                    vec4 light_color = light_params[light_id];
                    vec4 light_meta = light_params[__MAX_LIGHTS__ + light_id];
                    int dose_channel = int(light_color.w + 0.5);
                    if (dose_channel < 0 || dose_channel >= dose_channel_count) {
                        continue;
                    }
                    float cell_dose = texelFetch(cell_dose_tex, ivec3(cell, dose_channel), 0).x;
                    float gas_haze = 0.0;
                    if (
                        gas_cell.x >= 0
                        && gas_cell.y >= 0
                        && gas_cell.x < gas_grid_size.x
                        && gas_cell.y < gas_grid_size.y
                    ) {
                        gas_haze = texelFetch(gas_dose_tex, ivec3(gas_cell, dose_channel), 0).x;
                    }
                    if (cell_dose <= 0.0 && gas_haze <= 0.0) {
                        continue;
                    }
                    int visual_channel = int(light_meta.x + 0.5);
                    int style_id = int(light_meta.y + 0.5);
                    vec3 tint = tint_for_style(style_id);
                    vec3 scales = scales_for_style(style_id);
                    vec3 accent = accent_for_channel(visual_channel);
                    frame += (
                        cell_dose * (light_color.rgb * tint * scales.x)
                        + cell_dose * (accent * scales.y)
                        + gas_haze * (tint * scales.z)
                    );
                }
                imageStore(visible_img, cell, vec4(frame, 1.0));
            }
            """
            .replace("__LOCAL_SIZE__", str(LOCAL_SIZE))
            .replace("__LIGHT_PARAM_COUNT__", str(MAX_LIGHTS * 2))
            .replace("__MAX_LIGHTS__", str(MAX_LIGHTS))
        )
        self.programs["publish_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int dose_channel_count;
            layout(binding=0) uniform sampler2D visible_tex;
            layout(binding=1) uniform sampler2DArray cell_dose_tex;
            layout(rgba32f, binding=2) writeonly uniform image2D bridge_light_img;
            layout(rgba32f, binding=3) writeonly uniform image2D bridge_visible_img;
            layout(std430, binding=4) writeonly buffer BridgeCellDoseBuffer {{
                float bridge_cell_dose[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int channel = int(gl_GlobalInvocationID.z);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y || channel >= dose_channel_count) {{
                    return;
                }}
                if (channel == 0) {{
                    vec3 visible = texelFetch(visible_tex, gid, 0).xyz;
                    imageStore(bridge_light_img, gid, vec4(visible, 1.0));
                    imageStore(bridge_visible_img, gid, vec4(visible, 1.0));
                }}
                int dst_index = (channel * cell_grid_size.y + gid.y) * cell_grid_size.x + gid.x;
                bridge_cell_dose[dst_index] = texelFetch(cell_dose_tex, ivec3(gid, channel), 0).x;
            }}
            """
        )
        self.programs["publish_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int dose_channel_count;
            layout(binding=0) uniform sampler2DArray gas_dose_tex;
            layout(std430, binding=1) writeonly buffer BridgeGasDoseBuffer {{
                float bridge_gas_dose[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int channel = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || channel >= dose_channel_count) {{
                    return;
                }}
                int dst_index = (channel * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                bridge_gas_dose[dst_index] = texelFetch(gas_dose_tex, ivec3(gid, channel), 0).x;
            }}
            """
        )
        self.programs["clear_bridge_outputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform int dose_channel_count;
            layout(rgba32f, binding=0) writeonly uniform image2D bridge_light_img;
            layout(rgba32f, binding=1) writeonly uniform image2D bridge_visible_img;
            layout(std430, binding=0) writeonly buffer BridgeCellDoseBuffer {{
                float bridge_cell_dose[];
            }};
            layout(std430, binding=1) writeonly buffer BridgeGasDoseBuffer {{
                float bridge_gas_dose[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int channel = int(gl_GlobalInvocationID.z);
                if (gid.x < cell_grid_size.x && gid.y < cell_grid_size.y && channel < dose_channel_count) {{
                    int cell_index = (channel * cell_grid_size.y + gid.y) * cell_grid_size.x + gid.x;
                    bridge_cell_dose[cell_index] = 0.0;
                    if (channel == 0) {{
                        imageStore(bridge_light_img, gid, vec4(0.0, 0.0, 0.0, 1.0));
                        imageStore(bridge_visible_img, gid, vec4(0.0, 0.0, 0.0, 1.0));
                    }}
                }}
                if (gid.x < gas_grid_size.x && gid.y < gas_grid_size.y && channel < dose_channel_count) {{
                    int gas_index = (channel * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                    bridge_gas_dose[gas_index] = 0.0;
                }}
            }}
            """
        )

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        emitters: list[dict[str, object]],
        *,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
    ) -> int:
        world.bridge.sync_rule_tables(world)
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources("optics input", "material", "active_tile_ttl")
        active_authoritative = formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        self.last_cpu_active_upload_skipped = bool(active_authoritative)
        if not self._bridge_material_authoritative(world):
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        light_count = world.cell_optical_dose.shape[0]
        zero_cell = np.zeros((light_count, world.height, world.width), dtype="f4")
        zero_gas = np.zeros((light_count, world.gas_height, world.gas_width), dtype="f4")
        zero_illum = np.zeros((light_count, world.height, world.width, 4), dtype="f4")
        zero_visible = np.zeros((world.height, world.width, 4), dtype="f4")
        resources.cell_dose.write(zero_cell.tobytes())
        resources.gas_dose.write(zero_gas.tobytes())
        resources.illum_layers.write(zero_illum.tobytes())
        resources.visible_tex.write(zero_visible.tobytes())

        light_table = world.bridge.shadow_typed_tables["light_table"]
        emitter_data = np.zeros((MAX_EMITTERS * 2, 4), dtype="f4")
        emitter_count = 0
        for emitter in emitters:
            if emitter_count >= MAX_EMITTERS:
                break
            light_id = typed_light_id(light_table, str(emitter["light_type"]))
            if light_id < 0:
                continue
            direction = emitter["direction"]
            emitter_data[emitter_count * 2] = (
                float(emitter["origin"][0]),
                float(emitter["origin"][1]),
                float(direction[0]),
                float(direction[1]),
            )
            emitter_data[emitter_count * 2 + 1] = (
                float(emitter["strength"]),
                float(emitter["range_cells"]),
                float(emitter["spread"]),
                float(light_id),
            )
            emitter_count += 1
        resources.emitter_buffer.write(emitter_data.tobytes())
        emitter_counts = np.zeros((16,), dtype=np.uint32)
        emitter_counts[0] = np.uint32(emitter_count)
        resources.emitter_count_buffer.write(emitter_counts.tobytes())

        light_signature = (world.bridge.table_generations.get("lights", 0), int(light_table.shape[0]))
        if resources.light_buffer_signature != light_signature:
            light_colors = np.zeros((MAX_LIGHTS * 2, 4), dtype="f4")
            count = min(MAX_LIGHTS, int(light_table.shape[0]))
            light_colors[:count, :3] = light_table[:count]["color"]
            light_colors[:count, 3] = light_table[:count]["dose_channel_id"].astype(np.float32)
            light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 0] = light_table[:count]["visual_channel"].astype(np.float32)
            light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 1] = light_table[:count]["render_style_id"].astype(np.float32)
            light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 2] = light_table[:count]["max_bounce"].astype(np.float32)
            light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 3] = light_table[:count]["default_range"].astype(np.float32)
            resources.light_buffer.write(light_colors.tobytes())
            resources.light_buffer_signature = light_signature

        if active_authoritative:
            self._load_authoritative_active_masks(
                world,
                resources,
                force_all_active="reaction_light_emitter_count" in world.bridge.gpu_authoritative_resources,
            )
        else:
            resources.active_cell_tex.write(np.asarray(solve_cell_mask, dtype="f4").tobytes())
            resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())

        material_table = world.bridge.shadow_typed_tables["material_table"]
        optics_table = world.bridge.shadow_typed_tables["optics_table"]
        optics_signature = (
            world.bridge.table_generations.get("materials", 0),
            world.bridge.table_generations.get("lights", 0),
            world.bridge.table_generations.get("optics", 0),
            int(material_table.shape[0]),
            int(optics_table.shape[0]),
        )
        if resources.optics_buffer_signature != optics_signature:
            optics = np.zeros((MAX_MATERIALS * MAX_LIGHTS, 4), dtype="f4")
            for row in optics_table:
                material_id = int(row["material_id"])
                light_id = int(row["light_type_id"])
                if material_id < 0 or material_id >= MAX_MATERIALS or light_id < 0 or light_id >= MAX_LIGHTS:
                    continue
                optics[material_id * MAX_LIGHTS + light_id] = (
                    float(row["absorption"]),
                    float(row["scattering"]),
                    float(row["refraction"]),
                    0.0,
                )
            resources.optics_buffer.write(optics.tobytes())
            resources.optics_buffer_signature = optics_signature
        return emitter_count

    def _load_authoritative_active_masks(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        *,
        force_all_active: bool,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge active scheduler resources")
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
            self._set_uniform_if_present(program, "expansion_radius", 1)
            self._set_uniform_if_present(program, "force_all_active", bool(force_all_active))
            bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
            resources.emitter_buffer.bind_to_storage_buffer(binding=1)
            resources.emitter_count_buffer.bind_to_storage_buffer(binding=2)
            resources.light_buffer.bind_to_storage_buffer(binding=3)
            texture.bind_to_image(4, read=False, write=True)
            program.run(
                (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(bridge.ctx)

    def _set_uniform_if_present(self, program: Any, name: str, value: Any) -> None:
        try:
            program[name].value = value
        except KeyError:
            return

    def _bridge_material_authoritative(self, world: "WorldEngine") -> bool:
        return (
            self._formal_gpu_frame(world)
            and "material" in world.bridge.gpu_authoritative_resources
            and "material" in world.bridge.textures
        )

    def _bind_material_input(self, world: "WorldEngine", resources: GPUOpticsResources, *, location: int = 0) -> None:
        if self._bridge_material_authoritative(world):
            world.bridge.textures["material"].use(location=location)
            return
        resources.material_tex.use(location=location)

    def _run_emitter_buffer_rays(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        emitter_buffer: Any,
        emitter_count_buffer: Any,
    ) -> None:
        program = self.programs["trace_emitters"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["max_emitters"].value = MAX_EMITTERS
        program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
        self._bind_material_input(world, resources, location=0)
        resources.active_cell_tex.use(location=ACTIVE_CELL_TEXTURE_UNIT)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        emitter_buffer.bind_to_storage_buffer(binding=1)
        emitter_count_buffer.bind_to_storage_buffer(binding=2)
        resources.optics_buffer.bind_to_storage_buffer(binding=3)
        resources.light_buffer.bind_to_storage_buffer(binding=4)
        resources.cell_dose.bind_to_image(4, read=True, write=True)
        resources.gas_dose.bind_to_image(5, read=True, write=True)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

    def _compose_visible_illumination(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        program = self.programs["compose_visible"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["light_count"].value = min(MAX_LIGHTS, int(world.bridge.shadow_typed_tables["light_table"].shape[0]))
        program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
        resources.cell_dose.use(location=1)
        resources.gas_dose.use(location=2)
        resources.visible_tex.bind_to_image(6, read=False, write=True)
        resources.light_buffer.bind_to_storage_buffer(binding=4)
        groups_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, 1)
        self._sync_compute_writes(ctx)

    def _download_outputs(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        light_count = world.cell_optical_dose.shape[0]
        cell = np.frombuffer(resources.cell_dose.read(), dtype="f4").reshape((light_count, world.height, world.width))
        gas = np.frombuffer(resources.gas_dose.read(), dtype="f4").reshape((light_count, world.gas_height, world.gas_width))
        visible = np.frombuffer(resources.visible_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        world.cell_optical_dose[:] = cell
        world.gas_optical_dose[:] = gas
        world.visible_illumination[:] = visible[..., :3]

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge GPU resources for authoritative optics state")
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        cell_program = self.programs["publish_bridge_cell"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        cell_program["dose_channel_count"].value = dose_channel_count
        cell_program["visible_tex"].value = 0
        cell_program["cell_dose_tex"].value = 1
        resources.visible_tex.use(location=0)
        resources.cell_dose.use(location=1)
        bridge.textures["light"].bind_to_image(2, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(3, read=False, write=True)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=4)
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_program.run(cell_group_x, cell_group_y, dose_channel_count)

        gas_program = self.programs["publish_bridge_gas"]
        gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        gas_program["dose_channel_count"].value = dose_channel_count
        gas_program["gas_dose_tex"].value = 0
        resources.gas_dose.use(location=0)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=1)
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_program.run(gas_group_x, gas_group_y, dose_channel_count)

        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "light",
            "visible_illumination",
            "cell_optical_dose",
            "gas_optical_dose",
        )

    def clear_outputs(self, world: "WorldEngine") -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge GPU resources for clearing optics state")
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        program = self.programs["clear_bridge_outputs"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["dose_channel_count"].value = dose_channel_count
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(1, read=False, write=True)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=0)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=1)
        groups_x = (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, dose_channel_count)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            "light",
            "visible_illumination",
            "cell_optical_dose",
            "gas_optical_dose",
        )
        self.last_cpu_mirror_downloaded = False

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
            | getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0),
        )
