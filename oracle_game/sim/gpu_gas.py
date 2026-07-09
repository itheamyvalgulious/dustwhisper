from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_gas_id
from oracle_game.sim.gpu_base import GPUPipelineBase


LOCAL_SIZE = 8
REDUCE_LOCAL_SIZE = 256
MAX_SPECIES = 8
ACTIVE_GAS_TEXTURE_UNIT = 7


@dataclass(slots=True)
class GPUGasResources:
    signature: tuple[int, int, int]
    velocity_ping: Any
    velocity_pong: Any
    divergence: Any
    thermo_pressure: Any
    density_tex: Any
    pressure_ping: Any
    pressure_pong: Any
    ambient_ping: Any
    ambient_pong: Any
    gas_ping: Any
    gas_pong: Any
    active_gas_tex: Any
    species_params: Any
    species_force_params: Any
    force_sources: Any
    density_reduce_ping: Any
    density_reduce_pong: Any
    species_params_signature: tuple[int, int] | None = None


class GPUGasPipeline(GPUPipelineBase):
    def __init__(self, pressure_iterations: int = 12) -> None:
        self.pressure_iterations = pressure_iterations
        self.resources: GPUGasResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_velocity_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` is inherited from :class:`GPUPipelineBase`.

    def step(self, world: "WorldEngine", dt: float, *, solve_gas_mask: np.ndarray) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU gas pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.reset_pass_profile()
        with self._profile_pass(world, "upload_inputs"):
            self._upload_inputs(world, resources, solve_gas_mask)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)

        with self._profile_pass(world, "advect_velocity"):
            self._run_advect_velocity(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "force_sources"):
            self._run_force_sources(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "thermo_fields"):
            self._run_thermo_fields(world, resources, group_x, group_y)
        with self._profile_pass(world, "thermo_forces"):
            self._run_thermo_forces(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "divergence"):
            self._run_divergence(world, resources, group_x, group_y)
        with self._profile_pass(world, "pressure_jacobi"):
            self._run_pressure_jacobi(world, resources, group_x, group_y)
        with self._profile_pass(world, "projection"):
            self._run_projection(world, resources, group_x, group_y)
        with self._profile_pass(world, "species"):
            self._run_species(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "ambient"):
            self._run_ambient(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "publish_bridge_outputs"):
            self._publish_bridge_outputs(world, resources, group_x, group_y)
        self.last_cpu_mirror_downloaded = not (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            with self._profile_pass(world, "download_outputs"):
                self._download_outputs(world, resources)

    # ``reset_pass_profile`` / ``_profile_pass`` are inherited from
    # :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.velocity_ping,
            self.resources.velocity_pong,
            self.resources.divergence,
            self.resources.thermo_pressure,
            self.resources.density_tex,
            self.resources.pressure_ping,
            self.resources.pressure_pong,
            self.resources.ambient_ping,
            self.resources.ambient_pong,
            self.resources.gas_ping,
            self.resources.gas_pong,
            self.resources.active_gas_tex,
            self.resources.species_params,
            self.resources.species_force_params,
            self.resources.force_sources,
            self.resources.density_reduce_ping,
            self.resources.density_reduce_pong,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUGasResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.gas_width, world.gas_height, world.gas_concentration.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        species_count = signature[2]
        velocity_ping = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        velocity_pong = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        divergence = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        thermo_pressure = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        density_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        pressure_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        pressure_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        gas_ping = ctx.texture_array((world.gas_width, world.gas_height, species_count), 1, dtype="f4")
        gas_pong = ctx.texture_array((world.gas_width, world.gas_height, species_count), 1, dtype="f4")
        active_gas_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        for texture in (
            velocity_ping,
            velocity_pong,
            divergence,
            thermo_pressure,
            density_tex,
            pressure_ping,
            pressure_pong,
            ambient_ping,
            ambient_pong,
            gas_ping,
            gas_pong,
            active_gas_tex,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        species_params = ctx.buffer(reserve=MAX_SPECIES * 4 * 4, dynamic=True)
        species_force_params = ctx.buffer(reserve=MAX_SPECIES * 4 * 4, dynamic=True)
        force_sources = ctx.buffer(reserve=4, dynamic=True)
        density_reduce_ping = ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True)
        density_reduce_pong = ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True)
        self.resources = GPUGasResources(
            signature=signature,
            velocity_ping=velocity_ping,
            velocity_pong=velocity_pong,
            divergence=divergence,
            thermo_pressure=thermo_pressure,
            density_tex=density_tex,
            pressure_ping=pressure_ping,
            pressure_pong=pressure_pong,
            ambient_ping=ambient_ping,
            ambient_pong=ambient_pong,
            gas_ping=gas_ping,
            gas_pong=gas_pong,
            active_gas_tex=active_gas_tex,
            species_params=species_params,
            species_force_params=species_force_params,
            force_sources=force_sources,
            density_reduce_ping=density_reduce_ping,
            density_reduce_pong=density_reduce_pong,
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 grid_size;
            layout(binding={ACTIVE_GAS_TEXTURE_UNIT}) uniform sampler2D active_gas_tex;
            vec2 clamp_pos(vec2 pos) {{
                return clamp(pos, vec2(0.5), vec2(grid_size) - vec2(0.5001));
            }}
            bool solve_active(ivec2 cell) {{
                return texelFetch(active_gas_tex, cell, 0).x > 0.5;
            }}
            vec2 bilerp_vec2(sampler2D tex, vec2 pos) {{
                pos = clamp_pos(pos);
                vec2 base = pos - vec2(0.5);
                ivec2 p0 = ivec2(floor(base));
                ivec2 p1 = min(p0 + ivec2(1), grid_size - ivec2(1));
                vec2 fracv = fract(base);
                vec2 a = texelFetch(tex, p0, 0).xy;
                vec2 b = texelFetch(tex, ivec2(p1.x, p0.y), 0).xy;
                vec2 c = texelFetch(tex, ivec2(p0.x, p1.y), 0).xy;
                vec2 d = texelFetch(tex, p1, 0).xy;
                return mix(mix(a, b, fracv.x), mix(c, d, fracv.x), fracv.y);
            }}
            float bilerp_float(sampler2D tex, vec2 pos) {{
                pos = clamp_pos(pos);
                vec2 base = pos - vec2(0.5);
                ivec2 p0 = ivec2(floor(base));
                ivec2 p1 = min(p0 + ivec2(1), grid_size - ivec2(1));
                vec2 fracv = fract(base);
                float a = texelFetch(tex, p0, 0).x;
                float b = texelFetch(tex, ivec2(p1.x, p0.y), 0).x;
                float c = texelFetch(tex, ivec2(p0.x, p1.y), 0).x;
                float d = texelFetch(tex, p1, 0).x;
                return mix(mix(a, b, fracv.x), mix(c, d, fracv.x), fracv.y);
            }}
            float bilerp_layer(sampler2DArray tex, vec2 pos, int layer) {{
                pos = clamp_pos(pos);
                vec2 base = pos - vec2(0.5);
                ivec2 p0 = ivec2(floor(base));
                ivec2 p1 = min(p0 + ivec2(1), grid_size - ivec2(1));
                vec2 fracv = fract(base);
                float a = texelFetch(tex, ivec3(p0, layer), 0).x;
                float b = texelFetch(tex, ivec3(p1.x, p0.y, layer), 0).x;
                float c = texelFetch(tex, ivec3(p0.x, p1.y, layer), 0).x;
                float d = texelFetch(tex, ivec3(p1, layer), 0).x;
                return mix(mix(a, b, fracv.x), mix(c, d, fracv.x), fracv.y);
            }}
        """
        self.programs["load_bridge"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 grid_size;
            uniform int species_count;
            uniform bool copy_velocity;
            uniform bool copy_ambient;
            uniform bool copy_gas;

            layout(binding=0) uniform sampler2D bridge_velocity_tex;
            layout(binding=1) uniform sampler2D bridge_ambient_tex;
            layout(std430, binding=2) readonly buffer BridgeGasBuffer {{
                float bridge_gas[];
            }};
            layout(rg32f, binding=3) writeonly uniform image2D velocity_ping_img;
            layout(r32f, binding=4) writeonly uniform image2D ambient_ping_img;
            layout(r32f, binding=5) writeonly uniform image2DArray gas_ping_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y || species >= species_count) {{
                    return;
                }}
                if (species == 0) {{
                    if (copy_velocity) {{
                        imageStore(velocity_ping_img, gid, vec4(texelFetch(bridge_velocity_tex, gid, 0).xy, 0.0, 0.0));
                    }}
                    if (copy_ambient) {{
                        imageStore(ambient_ping_img, gid, vec4(texelFetch(bridge_ambient_tex, gid, 0).x, 0.0, 0.0, 0.0));
                    }}
                }}
                if (copy_gas) {{
                    int gas_index = (species * grid_size.y + gid.y) * grid_size.x + gid.x;
                    imageStore(gas_ping_img, ivec3(gid, species), vec4(max(bridge_gas[gas_index], 0.0), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["load_active_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            uniform int expansion_radius;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(r32f, binding=1) writeonly uniform image2D active_gas_img;
            bool source_tile_active(ivec2 tile) {{
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                int index = tile.y * tile_grid_size.x + tile.x;
                return active_tile_ttl[index] > 0;
            }}
            bool gas_cell_active(ivec2 gas_cell) {{
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                int tile_x0 = max(0, x0 / tile_size);
                int tile_y0 = max(0, y0 / tile_size);
                int tile_x1 = min(tile_grid_size.x, (x1 + tile_size - 1) / tile_size);
                int tile_y1 = min(tile_grid_size.y, (y1 + tile_size - 1) / tile_size);
                for (int tile_y = tile_y0 - expansion_radius; tile_y < tile_y1 + expansion_radius; ++tile_y) {{
                    for (int tile_x = tile_x0 - expansion_radius; tile_x < tile_x1 + expansion_radius; ++tile_x) {{
                        if (source_tile_active(ivec2(tile_x, tile_y))) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                imageStore(active_gas_img, gid, vec4(gas_cell_active(gid) ? 1.0 : 0.0, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["advect_velocity"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            uniform float damping;
            layout(binding=0) uniform sampler2D velocity_in_tex;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                vec2 vel = texelFetch(velocity_in_tex, gid, 0).xy;
                if (!solve_active(gid)) {
                    imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
                    return;
                }
                vec2 prev = vec2(gid) + vec2(0.5) - vel * dt;
                vec2 advected = bilerp_vec2(velocity_in_tex, prev) * damping;
                imageStore(velocity_out_img, gid, vec4(advected, 0.0, 0.0));
            }
            """
        )
        self.programs["divergence"] = ctx.compute_shader(
            helper
            + """
            layout(binding=0) uniform sampler2D velocity_tex;
            layout(r32f, binding=1) writeonly uniform image2D divergence_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                if (!solve_active(gid)) {
                    imageStore(divergence_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, grid_size.y - 1));
                float div = 0.5 * (
                    texelFetch(velocity_tex, right, 0).x - texelFetch(velocity_tex, left, 0).x +
                    texelFetch(velocity_tex, up, 0).y - texelFetch(velocity_tex, down, 0).y
                );
                imageStore(divergence_img, gid, vec4(div, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["force_sources"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            uniform float gas_cell_size;
            uniform int force_count;
            layout(std430, binding=0) buffer ForceSourceBuffer {
                vec4 force_data[];
            };
            layout(binding=0) uniform sampler2D velocity_in_tex;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                vec2 vel = texelFetch(velocity_in_tex, gid, 0).xy;
                if (solve_active(gid)) {
                    for (int force_index = 0; force_index < force_count; ++force_index) {
                        vec4 source = force_data[force_index * 2];
                        vec4 params = force_data[force_index * 2 + 1];
                        vec2 grid_pos = source.xy / gas_cell_size;
                        float radius = max(params.x / gas_cell_size, 1.0);
                        vec2 delta = vec2(gid) - grid_pos;
                        float influence = exp(-dot(delta, delta) / (radius * radius));
                        vel += source.zw * params.y * dt * influence;
                    }
                }
                imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
            }
            """
        )
        self.programs["thermo_fields"] = ctx.compute_shader(
            helper
            + f"""
            uniform int species_count;
            layout(std430, binding=0) buffer SpeciesForceParamBuffer {{
                vec4 force_params[{MAX_SPECIES}];
            }};
            layout(binding=1) uniform sampler2D ambient_tex;
            layout(binding=2) uniform sampler2DArray gas_tex;
            layout(r32f, binding=3) writeonly uniform image2D thermo_pressure_img;
            layout(r32f, binding=4) writeonly uniform image2D density_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {{
                    return;
                }}
                float pressure_coeff = 0.0;
                float density = 0.0;
                for (int species = 0; species < species_count; ++species) {{
                    float concentration = texelFetch(gas_tex, ivec3(gid, species), 0).x;
                    pressure_coeff += concentration * force_params[species].x;
                    density += concentration * force_params[species].y;
                }}
                float temperature = max(texelFetch(ambient_tex, gid, 0).x, 0.1);
                if (!solve_active(gid)) {{
                    imageStore(thermo_pressure_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    imageStore(density_img, gid, vec4(density, 0.0, 0.0, 0.0));
                    return;
                }}
                imageStore(thermo_pressure_img, gid, vec4(temperature * pressure_coeff, 0.0, 0.0, 0.0));
                imageStore(density_img, gid, vec4(density, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["density_extract"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 grid_size;
            layout(binding=0) uniform sampler2D density_tex;
            layout(std430, binding=0) buffer DensityValues {{
                float values[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {{
                    return;
                }}
                int index = gid.y * grid_size.x + gid.x;
                values[index] = texelFetch(density_tex, gid, 0).x;
            }}
            """
        )
        self.programs["density_reduce"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={REDUCE_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int input_count;
            layout(std430, binding=0) buffer InputValues {{
                float input_values[];
            }};
            layout(std430, binding=1) buffer OutputValues {{
                float output_values[];
            }};
            void main() {{
                int out_index = int(gl_GlobalInvocationID.x);
                int in_index = out_index * 2;
                if (in_index >= input_count) {{
                    return;
                }}
                float total = input_values[in_index];
                if (in_index + 1 < input_count) {{
                    total += input_values[in_index + 1];
                }}
                output_values[out_index] = total;
            }}
            """
        )
        self.programs["jacobi"] = ctx.compute_shader(
            helper
            + """
            layout(binding=0) uniform sampler2D pressure_tex;
            layout(binding=1) uniform sampler2D divergence_tex;
            layout(r32f, binding=2) writeonly uniform image2D pressure_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                float center = texelFetch(pressure_tex, gid, 0).x;
                if (!solve_active(gid)) {
                    imageStore(pressure_out_img, gid, vec4(center, 0.0, 0.0, 0.0));
                    return;
                }
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, grid_size.y - 1));
                float p = (
                    texelFetch(pressure_tex, left, 0).x +
                    texelFetch(pressure_tex, right, 0).x +
                    texelFetch(pressure_tex, down, 0).x +
                    texelFetch(pressure_tex, up, 0).x -
                    texelFetch(divergence_tex, gid, 0).x
                ) * 0.25;
                imageStore(pressure_out_img, gid, vec4(p, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["thermo_forces"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            uniform int density_cell_count;
            layout(std430, binding=4) buffer DensitySumBuffer {
                float density_sum[];
            };
            layout(binding=0) uniform sampler2D velocity_in_tex;
            layout(binding=1) uniform sampler2D thermo_pressure_tex;
            layout(binding=2) uniform sampler2D density_tex;
            layout(rg32f, binding=3) writeonly uniform image2D velocity_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                vec2 vel = texelFetch(velocity_in_tex, gid, 0).xy;
                if (!solve_active(gid)) {
                    imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
                    return;
                }
                float mean_density = density_cell_count > 0 ? density_sum[0] / float(density_cell_count) : 0.0;
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, grid_size.y - 1));
                float grad_x = 0.5 * (
                    texelFetch(thermo_pressure_tex, right, 0).x -
                    texelFetch(thermo_pressure_tex, left, 0).x
                );
                float grad_y = 0.5 * (
                    texelFetch(thermo_pressure_tex, up, 0).x -
                    texelFetch(thermo_pressure_tex, down, 0).x
                );
                float density = texelFetch(density_tex, gid, 0).x;
                float inv_density = 1.0 / max(density, 0.25);
                vel.x -= grad_x * inv_density * dt * 0.2;
                vel.y -= grad_y * inv_density * dt * 0.2;
                vel.y += (density - mean_density) * dt * 0.08;
                imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
            }
            """
        )
        self.programs["projection"] = ctx.compute_shader(
            helper
            + """
            layout(binding=0) uniform sampler2D pressure_tex;
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(rg32f, binding=2) writeonly uniform image2D velocity_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {
                    return;
                }
                vec2 vel = texelFetch(velocity_tex, gid, 0).xy;
                if (!solve_active(gid)) {
                    imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
                    return;
                }
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, grid_size.y - 1));
                float grad_x = texelFetch(pressure_tex, right, 0).x - texelFetch(pressure_tex, left, 0).x;
                float grad_y = texelFetch(pressure_tex, up, 0).x - texelFetch(pressure_tex, down, 0).x;
                vel -= 0.5 * vec2(grad_x, grad_y);
                imageStore(velocity_out_img, gid, vec4(vel, 0.0, 0.0));
            }
            """
        )
        self.programs["species"] = ctx.compute_shader(
            helper
            + f"""
            uniform float dt;
            uniform int species_count;
            uniform int air_index;
            layout(std430, binding=0) buffer SpeciesParamBuffer {{
                vec4 params[{MAX_SPECIES}];
            }};
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(binding=2) uniform sampler2DArray gas_in_tex;
            layout(r32f, binding=3) writeonly uniform image2DArray gas_out_img;
            void main() {{
                ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y || gid.z >= species_count) {{
                    return;
                }}
                float current = texelFetch(gas_in_tex, gid, 0).x;
                if (!solve_active(gid.xy)) {{
                    imageStore(gas_out_img, gid, vec4(current, 0.0, 0.0, 0.0));
                    return;
                }}
                vec2 pos = vec2(gid.xy) + vec2(0.5);
                vec2 vel = bilerp_vec2(velocity_tex, pos);
                vel.y -= params[gid.z].w;
                vec2 prev = pos - vel * dt;
                float advected = bilerp_layer(gas_in_tex, prev, gid.z);
                vec2 left_pos = vec2(max(gid.x - 1, 0), gid.y) + vec2(0.5);
                vec2 right_pos = vec2(min(gid.x + 1, grid_size.x - 1), gid.y) + vec2(0.5);
                vec2 down_pos = vec2(gid.x, max(gid.y - 1, 0)) + vec2(0.5);
                vec2 up_pos = vec2(gid.x, min(gid.y + 1, grid_size.y - 1)) + vec2(0.5);
                vec2 left_vel = bilerp_vec2(velocity_tex, left_pos);
                vec2 right_vel = bilerp_vec2(velocity_tex, right_pos);
                vec2 down_vel = bilerp_vec2(velocity_tex, down_pos);
                vec2 up_vel = bilerp_vec2(velocity_tex, up_pos);
                left_vel.y -= params[gid.z].w;
                right_vel.y -= params[gid.z].w;
                down_vel.y -= params[gid.z].w;
                up_vel.y -= params[gid.z].w;
                float advected_left = bilerp_layer(gas_in_tex, left_pos - left_vel * dt, gid.z);
                float advected_right = bilerp_layer(gas_in_tex, right_pos - right_vel * dt, gid.z);
                float advected_down = bilerp_layer(gas_in_tex, down_pos - down_vel * dt, gid.z);
                float advected_up = bilerp_layer(gas_in_tex, up_pos - up_vel * dt, gid.z);
                float lap = advected_left + advected_right + advected_down + advected_up - 4.0 * advected;
                float value = advected + params[gid.z].x * dt * lap;
                value *= max(0.0, 1.0 - params[gid.z].y * dt);
                value = max(value, 0.0);
                if (gid.z == air_index) {{
                    value = max(value, 0.3);
                }}
                imageStore(gas_out_img, gid, vec4(value, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["ambient"] = ctx.compute_shader(
            helper
            + f"""
            uniform float dt;
            uniform int species_count;
            layout(std430, binding=0) buffer SpeciesParamBuffer {{
                vec4 params[{MAX_SPECIES}];
            }};
            layout(binding=1) uniform sampler2D velocity_tex;
            layout(binding=2) uniform sampler2D ambient_in_tex;
            layout(binding=3) uniform sampler2DArray gas_tex;
            layout(r32f, binding=4) writeonly uniform image2D ambient_out_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y) {{
                    return;
                }}
                float current = texelFetch(ambient_in_tex, gid, 0).x;
                if (!solve_active(gid)) {{
                    imageStore(ambient_out_img, gid, vec4(current, 0.0, 0.0, 0.0));
                    return;
                }}
                vec2 pos = vec2(gid) + vec2(0.5);
                vec2 vel = bilerp_vec2(velocity_tex, pos);
                vec2 prev = pos - vel * dt;
                float ambient = bilerp_float(ambient_in_tex, prev);
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, grid_size.y - 1));
                float lap = texelFetch(ambient_in_tex, left, 0).x + texelFetch(ambient_in_tex, right, 0).x +
                            texelFetch(ambient_in_tex, down, 0).x + texelFetch(ambient_in_tex, up, 0).x -
                            4.0 * ambient;
                ambient += 0.08 * lap;
                for (int species = 0; species < species_count; ++species) {{
                    ambient += texelFetch(gas_tex, ivec3(gid, species), 0).x * params[species].z * 0.01;
                }}
                imageStore(ambient_out_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 grid_size;
            uniform int species_count;
            layout(binding=0) uniform sampler2D velocity_tex;
            layout(binding=1) uniform sampler2D ambient_tex;
            layout(binding=2) uniform sampler2D pressure_tex;
            layout(binding=3) uniform sampler2D thermo_pressure_tex;
            layout(binding=4) uniform sampler2DArray gas_tex;
            layout(rg32f, binding=5) writeonly uniform image2D bridge_velocity_img;
            layout(r32f, binding=6) writeonly uniform image2D bridge_ambient_img;
            layout(r32f, binding=7) writeonly uniform image2D bridge_pressure_img;
            layout(std430, binding=8) writeonly buffer BridgeGasBuffer {{
                float bridge_gas[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= grid_size.x || gid.y >= grid_size.y || species >= species_count) {{
                    return;
                }}
                if (species == 0) {{
                    vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                    float ambient = texelFetch(ambient_tex, gid, 0).x;
                    float pressure = texelFetch(pressure_tex, gid, 0).x + texelFetch(thermo_pressure_tex, gid, 0).x;
                    imageStore(bridge_velocity_img, gid, vec4(velocity, 0.0, 0.0));
                    imageStore(bridge_ambient_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                    imageStore(bridge_pressure_img, gid, vec4(pressure, 0.0, 0.0, 0.0));
                }}
                int dst_index = (species * grid_size.y + gid.y) * grid_size.x + gid.x;
                bridge_gas[dst_index] = max(texelFetch(gas_tex, ivec3(gid, species), 0).x, 0.0);
            }}
            """
        )

    def _upload_inputs(self, world: "WorldEngine", resources: GPUGasResources, solve_gas_mask: np.ndarray) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "gas input",
            "flow_velocity",
            "ambient_temperature",
            "gas_concentration",
            "active_tile_ttl",
        )
        upload_velocity_from_cpu = not (formal_gpu_frame and "flow_velocity" in authoritative)
        upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
        upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
        upload_active_from_cpu = not (formal_gpu_frame and "active_tile_ttl" in authoritative)
        self.last_cpu_velocity_upload_skipped = not upload_velocity_from_cpu
        self.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
        self.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_velocity_from_cpu:
            resources.velocity_ping.write(world.flow_velocity.astype("f4").tobytes())
            resources.velocity_pong.write(world.flow_velocity.astype("f4").tobytes())
        if upload_ambient_from_cpu:
            resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
            resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
        resources.thermo_pressure.write(np.zeros_like(world.ambient_temperature, dtype="f4").tobytes())
        resources.density_tex.write(np.zeros_like(world.ambient_temperature, dtype="f4").tobytes())
        resources.pressure_ping.write(np.zeros_like(world.ambient_temperature, dtype="f4").tobytes())
        resources.pressure_pong.write(np.zeros_like(world.ambient_temperature, dtype="f4").tobytes())
        if upload_gas_from_cpu:
            resources.gas_ping.write(world.gas_concentration.astype("f4").tobytes())
            resources.gas_pong.write(world.gas_concentration.astype("f4").tobytes())
        if upload_active_from_cpu:
            resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_gas_mask(world, resources, expansion_radius=1)
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        table_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
        if resources.species_params_signature == table_signature:
            return
        params = np.zeros((MAX_SPECIES, 4), dtype="f4")
        force_params = np.zeros((MAX_SPECIES, 4), dtype="f4")
        count = min(MAX_SPECIES, gas_table.shape[0])
        params[:count, 0] = gas_table[:count]["diffusion_rate"]
        params[:count, 1] = gas_table[:count]["decay_rate"]
        params[:count, 2] = gas_table[:count]["temperature_coupling"]
        params[:count, 3] = gas_table[:count]["buoyancy"]
        force_params[:count, 0] = gas_table[:count]["pressure_factor"]
        force_params[:count, 1] = gas_table[:count]["density_factor"]
        resources.species_params.write(params.tobytes())
        resources.species_force_params.write(force_params.tobytes())
        resources.species_params_signature = table_signature

    def _load_authoritative_active_gas_mask(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge active scheduler resources")
        program = self.programs["load_active_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["tile_size"].value = int(world.active.tile_size)
        program["expansion_radius"].value = int(expansion_radius)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        resources.active_gas_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)

    def _force_source_upload(self, world: "WorldEngine") -> np.ndarray:
        force_count = len(world.force_sources)
        force_data = np.zeros((max(1, force_count) * 2, 4), dtype=np.float32)
        for index, force in enumerate(world.force_sources):
            buffer_x, buffer_y = world._force_source_buffer_position(force)
            force_data[index * 2] = (
                float(buffer_x),
                float(buffer_y),
                float(force.direction[0]),
                float(force.direction[1]),
            )
            force_data[index * 2 + 1] = (
                float(force.radius),
                float(force.strength),
                float(force.lifetime),
                0.0,
            )
        return force_data

    def _write_dynamic_buffer(self, ctx: Any, resources: GPUGasResources, name: str, data: np.ndarray) -> None:
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

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_velocity = "flow_velocity" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        copy_gas = "gas_concentration" in authoritative
        if not (copy_velocity or copy_ambient or copy_gas):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge GPU resources for authoritative input state")
        program = self.programs["load_bridge"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["copy_velocity"].value = bool(copy_velocity)
        program["copy_ambient"].value = bool(copy_ambient)
        program["copy_gas"].value = bool(copy_gas)
        bridge.textures["flow_velocity"].use(location=0)
        bridge.textures["ambient_temperature"].use(location=1)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=2)
        resources.velocity_ping.bind_to_image(3, read=False, write=True)
        resources.ambient_ping.bind_to_image(4, read=False, write=True)
        resources.gas_ping.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, int(world.gas_concentration.shape[0]))
        self._sync_compute_writes(bridge.ctx)

    def _run_advect_velocity(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["advect_velocity"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["damping"].value = 0.995
        resources.velocity_ping.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_pong.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_force_sources(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        if not world.force_sources:
            return
        program = self.programs["force_sources"]
        ctx = world.bridge.ctx
        assert ctx is not None
        force_data = self._force_source_upload(world)
        self._write_dynamic_buffer(ctx, resources, "force_sources", force_data)
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["gas_cell_size"].value = float(world.gas_cell_size)
        program["force_count"].value = len(world.force_sources)
        resources.force_sources.bind_to_storage_buffer(binding=0)
        resources.velocity_pong.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_ping.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping
        for force in list(world.force_sources):
            force.lifetime -= dt
        world.force_sources[:] = [force for force in world.force_sources if force.lifetime > 0.0]

    def _run_divergence(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["divergence"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        resources.velocity_ping.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.divergence.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_thermo_fields(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["thermo_fields"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = world.gas_concentration.shape[0]
        resources.species_force_params.bind_to_storage_buffer(binding=0)
        resources.ambient_ping.use(location=1)
        resources.gas_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.thermo_pressure.bind_to_image(3, read=False, write=True)
        resources.density_tex.bind_to_image(4, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_density_reduction(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        cell_count = int(world.gas_width * world.gas_height)
        if cell_count <= 0:
            return
        zero = np.zeros((cell_count,), dtype=np.float32)
        self._write_dynamic_buffer(ctx, resources, "density_reduce_ping", zero)
        self._write_dynamic_buffer(ctx, resources, "density_reduce_pong", zero)

        extract = self.programs["density_extract"]
        extract["grid_size"].value = (world.gas_width, world.gas_height)
        resources.density_tex.use(location=0)
        resources.density_reduce_ping.bind_to_storage_buffer(binding=0)
        extract.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

        src = resources.density_reduce_ping
        dst = resources.density_reduce_pong
        input_count = cell_count
        reduce_program = self.programs["density_reduce"]
        while input_count > 1:
            output_count = (input_count + 1) // 2
            reduce_program["input_count"].value = input_count
            src.bind_to_storage_buffer(binding=0)
            dst.bind_to_storage_buffer(binding=1)
            reduce_program.run((output_count + REDUCE_LOCAL_SIZE - 1) // REDUCE_LOCAL_SIZE, 1, 1)
            self._sync_compute_writes(ctx)
            src, dst = dst, src
            input_count = output_count
        resources.density_reduce_ping = src
        resources.density_reduce_pong = dst

    def _run_thermo_forces(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        self._run_density_reduction(world, resources, group_x, group_y)
        program = self.programs["thermo_forces"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["density_cell_count"].value = int(world.gas_width * world.gas_height)
        resources.velocity_pong.use(location=0)
        resources.thermo_pressure.use(location=1)
        resources.density_tex.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.density_reduce_ping.bind_to_storage_buffer(binding=4)
        resources.velocity_ping.bind_to_image(3, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_pressure_jacobi(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["jacobi"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        for _ in range(self.pressure_iterations):
            resources.pressure_ping.use(location=0)
            resources.divergence.use(location=1)
            resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
            resources.pressure_pong.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
            resources.pressure_ping, resources.pressure_pong = resources.pressure_pong, resources.pressure_ping

    def _run_projection(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["projection"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        resources.pressure_ping.use(location=0)
        resources.velocity_ping.use(location=1)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_pong.bind_to_image(2, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping

    def _run_species(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["species"]
        ctx = world.bridge.ctx
        assert ctx is not None
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = world.gas_concentration.shape[0]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["species_count"].value = species_count
        program["air_index"].value = typed_gas_id(gas_table, "air")
        resources.species_params.bind_to_storage_buffer(binding=0)
        resources.velocity_ping.use(location=1)
        resources.gas_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.gas_pong.bind_to_image(3, read=False, write=True)
        program.run(group_x, group_y, species_count)
        self._sync_compute_writes(ctx)
        resources.gas_ping, resources.gas_pong = resources.gas_pong, resources.gas_ping

    def _run_ambient(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["ambient"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["species_count"].value = world.gas_concentration.shape[0]
        resources.species_params.bind_to_storage_buffer(binding=0)
        resources.velocity_ping.use(location=1)
        resources.ambient_ping.use(location=2)
        resources.gas_ping.use(location=3)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.ambient_pong.bind_to_image(4, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping

    def _download_outputs(self, world: "WorldEngine", resources: GPUGasResources) -> None:
        velocity = np.frombuffer(resources.velocity_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width, 2))
        ambient = np.frombuffer(resources.ambient_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        pressure = np.frombuffer(resources.pressure_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        thermo_pressure = np.frombuffer(resources.thermo_pressure.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        gas = np.frombuffer(resources.gas_ping.read(), dtype="f4").reshape((world.gas_concentration.shape[0], world.gas_height, world.gas_width))
        world.flow_velocity[:] = velocity
        world.ambient_temperature[:] = ambient
        world.pressure_ping[:] = pressure + thermo_pressure
        world.gas_concentration[:] = np.maximum(gas, 0.0)

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge GPU resources for authoritative gas state")
        program = self.programs["publish_bridge"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["velocity_tex"].value = 0
        program["ambient_tex"].value = 1
        program["pressure_tex"].value = 2
        program["thermo_pressure_tex"].value = 3
        program["gas_tex"].value = 4
        resources.velocity_ping.use(location=0)
        resources.ambient_ping.use(location=1)
        resources.pressure_ping.use(location=2)
        resources.thermo_pressure.use(location=3)
        resources.gas_ping.use(location=4)
        bridge.textures["flow_velocity"].bind_to_image(5, read=False, write=True)
        bridge.textures["ambient_temperature"].bind_to_image(6, read=False, write=True)
        bridge.textures["pressure_ping"].bind_to_image(7, read=False, write=True)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=8)
        program.run(group_x, group_y, int(world.gas_concentration.shape[0]))
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "flow_velocity",
            "ambient_temperature",
            "pressure_ping",
            "gas_concentration",
        )

    # ``_sync_compute_writes`` is inherited from :class:`GPUPipelineBase`
    # (uses the default ``_barrier_bits`` set: image access + texture fetch +
    # shader storage).
