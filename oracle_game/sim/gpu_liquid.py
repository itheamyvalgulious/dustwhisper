from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_material_id
from oracle_game.types import Phase

TILE_SIZE = 32
TILE_LOCAL_SIZE = TILE_SIZE
PASS_LOCAL_SIZE = 8
MAX_MATERIALS = 256


@dataclass(slots=True)
class GPULiquidResources:
    signature: tuple[int, int, int, int]
    material_pre: Any
    material_in: Any
    material_out: Any
    phase_pre: Any
    phase_in: Any
    phase_out: Any
    island_in: Any
    island_out: Any
    entity_in: Any
    entity_out: Any
    flags_in: Any
    flags_out: Any
    timer_in: Any
    timer_out: Any
    temp_in: Any
    temp_out: Any
    integrity_in: Any
    integrity_out: Any
    velocity_in: Any
    velocity_out: Any
    active_tile_tex: Any
    displaced_in: Any
    displaced_out: Any
    material_params: Any
    material_params_signature: tuple[int, int] | None = None


class GPULiquidPipeline:
    def __init__(self) -> None:
        self.resources: GPULiquidResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_cpu_active_upload_skipped = False

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def step(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
        post_tile_mask: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
        self._load_authoritative_bridge_inputs(world, resources)
        group_x = world.active.tile_width
        group_y = world.active.tile_height
        self._run_tile_solve(world, resources, group_x, group_y)
        if self._formal_gpu_frame(world):
            self._publish_bridge_outputs(world, resources, use_out=True)
            self.last_cpu_mirror_downloaded = False
            return
        if self._active_scheduler_gpu_authoritative(world):
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=1)
        else:
            self._upload_active_tile_mask(resources, post_tile_mask)
        self._run_seam_pass(
            "seam_x",
            world,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.active_tile_tex,
        )
        self._run_seam_pass(
            "seam_y",
            world,
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            resources.active_tile_tex,
        )
        self._run_buoyancy_pass(
            "buoyancy_sink",
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
        )
        self._run_buoyancy_pass(
            "buoyancy_float",
            world,
            resources,
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
        )
        self._run_copy_for_placeholder(
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
        )
        self._run_placeholder_displacement(
            world,
            resources,
            (
                resources.material_out,
                resources.phase_out,
                resources.flags_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.material_in,
                resources.phase_in,
                resources.flags_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
        )
        self._run_cleanup_runtime(world, resources)
        self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources, use_in=True)

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.material_pre,
            self.resources.material_in,
            self.resources.material_out,
            self.resources.phase_pre,
            self.resources.phase_in,
            self.resources.phase_out,
            self.resources.island_in,
            self.resources.island_out,
            self.resources.entity_in,
            self.resources.entity_out,
            self.resources.flags_in,
            self.resources.flags_out,
            self.resources.timer_in,
            self.resources.timer_out,
            self.resources.temp_in,
            self.resources.temp_out,
            self.resources.integrity_in,
            self.resources.integrity_out,
            self.resources.velocity_in,
            self.resources.velocity_out,
            self.resources.active_tile_tex,
            self.resources.displaced_in,
            self.resources.displaced_out,
            self.resources.material_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPULiquidResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.active.tile_width, world.active.tile_height)
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        material_pre = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_pre = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        flags_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        flags_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        timer_in = ctx.texture((world.width, world.height), 4, dtype="f4")
        timer_out = ctx.texture((world.width, world.height), 4, dtype="f4")
        temp_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        temp_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        velocity_in = ctx.texture((world.width, world.height), 2, dtype="f4")
        velocity_out = ctx.texture((world.width, world.height), 2, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        displaced_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        for texture in (
            material_pre,
            material_in,
            material_out,
            phase_pre,
            phase_in,
            phase_out,
            island_in,
            island_out,
            entity_in,
            entity_out,
            flags_in,
            flags_out,
            timer_in,
            timer_out,
            temp_in,
            temp_out,
            integrity_in,
            integrity_out,
            velocity_in,
            velocity_out,
            active_tile_tex,
            displaced_in,
            displaced_out,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        self.resources = GPULiquidResources(
            signature=signature,
            material_pre=material_pre,
            material_in=material_in,
            material_out=material_out,
            phase_pre=phase_pre,
            phase_in=phase_in,
            phase_out=phase_out,
            island_in=island_in,
            island_out=island_out,
            entity_in=entity_in,
            entity_out=entity_out,
            flags_in=flags_in,
            flags_out=flags_out,
            timer_in=timer_in,
            timer_out=timer_out,
            temp_in=temp_in,
            temp_out=temp_out,
            integrity_in=integrity_in,
            integrity_out=integrity_out,
            velocity_in=velocity_in,
            velocity_out=velocity_out,
            active_tile_tex=active_tile_tex,
            displaced_in=displaced_in,
            displaced_out=displaced_out,
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
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
        self.programs["tile_solve"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={TILE_LOCAL_SIZE}, local_size_y={TILE_LOCAL_SIZE}, local_size_z=1) in;
            const int TILE_SIZE = {TILE_SIZE};
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int phase_liquid;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            shared float s_material[TILE_SIZE][TILE_SIZE];
            shared float s_phase[TILE_SIZE][TILE_SIZE];
            shared float s_flags[TILE_SIZE][TILE_SIZE];
            shared int s_source_x[TILE_SIZE][TILE_SIZE];
            shared int s_source_y[TILE_SIZE][TILE_SIZE];

            void clear_cell(int y, int x) {{
                s_material[y][x] = 0.0;
                s_phase[y][x] = 0.0;
                s_flags[y][x] = 0.0;
                s_source_x[y][x] = -1;
                s_source_y[y][x] = -1;
            }}

            void write_cell(
                int y,
                int x,
                float material,
                float phase,
                float flagsv,
                int source_x,
                int source_y
            ) {{
                s_material[y][x] = material;
                s_phase[y][x] = phase;
                s_flags[y][x] = flagsv;
                s_source_x[y][x] = source_x;
                s_source_y[y][x] = source_y;
            }}

            int liquid_kind_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].z + 0.5);
            }}

            void main() {{
                ivec2 local = ivec2(gl_LocalInvocationID.xy);
                ivec2 tile = ivec2(gl_WorkGroupID.xy);
                if (tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                ivec2 cell = tile * TILE_SIZE + local;
                bool in_bounds = cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
                float material = in_bounds ? texelFetch(material_in_tex, cell, 0).x : 0.0;
                float phase = in_bounds ? texelFetch(phase_in_tex, cell, 0).x : 0.0;
                float flagsv = in_bounds ? texelFetch(flags_in_tex, cell, 0).x : 0.0;
                vec4 timerv = in_bounds ? texelFetch(timer_in_tex, cell, 0) : vec4(0.0);
                float tempv = in_bounds ? texelFetch(temp_in_tex, cell, 0).x : 0.0;
                float integrityv = in_bounds ? texelFetch(integrity_in_tex, cell, 0).x : 0.0;
                vec2 velocityv = in_bounds ? texelFetch(velocity_in_tex, cell, 0).xy : vec2(0.0);
                s_material[local.y][local.x] = material;
                s_phase[local.y][local.x] = phase;
                s_flags[local.y][local.x] = flagsv;
                s_source_x[local.y][local.x] = in_bounds ? local.x : -1;
                s_source_y[local.y][local.x] = in_bounds ? local.y : -1;
                barrier();

                bool tile_active = texelFetch(active_tile_tex, tile, 0).x > 0.5;
                if (tile_active) {{
                    for (int row = TILE_SIZE - 2; row >= 0; --row) {{
                        barrier();
                        if (local.y == row && local.x == 0) {{
                            int world_row = tile.y * TILE_SIZE + row;
                            if (world_row + 1 >= cell_grid_size.y) {{
                                continue;
                            }}
                            float row_material_snapshot[TILE_SIZE];
                            float row_phase_snapshot[TILE_SIZE];
                            int row_in_bounds[TILE_SIZE];
                            int row_kind[TILE_SIZE];
                            int below_empty[TILE_SIZE];
                            int claimed_lateral_dest[TILE_SIZE];
                            for (int x = 0; x < TILE_SIZE; ++x) {{
                                ivec2 row_cell = tile * TILE_SIZE + ivec2(x, row);
                                bool row_valid = row_cell.x < cell_grid_size.x && row_cell.y < cell_grid_size.y;
                                row_in_bounds[x] = row_valid ? 1 : 0;
                                row_material_snapshot[x] = row_valid ? s_material[row][x] : 0.0;
                                row_phase_snapshot[x] = row_valid ? s_phase[row][x] : 0.0;
                                row_kind[x] = 0;
                                below_empty[x] = 0;
                                claimed_lateral_dest[x] = 0;
                                if (
                                    row_valid
                                    && row_material_snapshot[x] > 0.5
                                    && int(row_phase_snapshot[x] + 0.5) == phase_liquid
                                ) {{
                                    row_kind[x] = liquid_kind_for(row_material_snapshot[x]);
                                }}
                                if (
                                    row_valid
                                    && row + 1 < TILE_SIZE
                                    && row_cell.y + 1 < cell_grid_size.y
                                    && s_material[row + 1][x] < 0.5
                                ) {{
                                    below_empty[x] = 1;
                                }}
                            }}
                            for (int x = 0; x < TILE_SIZE; ++x) {{
                                if (row_in_bounds[x] == 0) {{
                                    continue;
                                }}
                                if (row_material_snapshot[x] < 0.5 || int(row_phase_snapshot[x] + 0.5) != phase_liquid) {{
                                    continue;
                                }}
                                int dst_x = x;
                                int dst_y = row;
                                if (below_empty[x] != 0) {{
                                    dst_y = row + 1;
                                }} else if (row_kind[x] == 1) {{
                                    if (
                                        x > 0
                                        && row_in_bounds[x - 1] != 0
                                        && row_material_snapshot[x - 1] < 0.5
                                        && claimed_lateral_dest[x - 1] == 0
                                    ) {{
                                        dst_x = x - 1;
                                        claimed_lateral_dest[x - 1] = 1;
                                    }} else if (
                                        x + 1 < TILE_SIZE
                                        && row_in_bounds[x + 1] != 0
                                        && row_material_snapshot[x + 1] < 0.5
                                        && claimed_lateral_dest[x + 1] == 0
                                    ) {{
                                        dst_x = x + 1;
                                        claimed_lateral_dest[x + 1] = 1;
                                    }}
                                }}
                                if (dst_x != x || dst_y != row) {{
                                    write_cell(
                                        dst_y,
                                        dst_x,
                                        s_material[row][x],
                                        s_phase[row][x],
                                        s_flags[row][x],
                                        s_source_x[row][x],
                                        s_source_y[row][x]
                                    );
                                    clear_cell(row, x);
                                }}
                            }}
                        }}
                    }}
                }}
                barrier();
                if (in_bounds) {{
                    int source_x = s_source_x[local.y][local.x];
                    int source_y = s_source_y[local.y][local.x];
                    bool has_source = source_x >= 0 && source_y >= 0;
                    ivec2 source_cell = tile * TILE_SIZE + ivec2(max(source_x, 0), max(source_y, 0));
                    vec4 out_timer = has_source ? texelFetch(timer_in_tex, source_cell, 0) : vec4(0.0);
                    float out_temp = has_source ? texelFetch(temp_in_tex, source_cell, 0).x : 0.0;
                    float out_integrity = has_source ? texelFetch(integrity_in_tex, source_cell, 0).x : 0.0;
                    vec2 out_velocity = has_source ? texelFetch(velocity_in_tex, source_cell, 0).xy : vec2(0.0);
                    imageStore(material_out_img, cell, vec4(s_material[local.y][local.x], 0.0, 0.0, 0.0));
                    imageStore(phase_out_img, cell, vec4(s_phase[local.y][local.x], 0.0, 0.0, 0.0));
                    imageStore(flags_out_img, cell, vec4(s_flags[local.y][local.x], 0.0, 0.0, 0.0));
                    imageStore(timer_out_img, cell, out_timer);
                    imageStore(temp_out_img, cell, vec4(out_temp, 0.0, 0.0, 0.0));
                    imageStore(integrity_out_img, cell, vec4(out_integrity, 0.0, 0.0, 0.0));
                    imageStore(velocity_out_img, cell, vec4(out_velocity, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["seam_x"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            const int TILE_SIZE = {TILE_SIZE};
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int phase_liquid;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            bool boundary_active(ivec2 cell) {{
                int tile_y = min(tile_grid_size.y - 1, max(0, cell.y / TILE_SIZE));
                if (cell.x % TILE_SIZE == TILE_SIZE - 1 && cell.x + 1 < cell_grid_size.x) {{
                    int left_tile = cell.x / TILE_SIZE;
                    int right_tile = min(tile_grid_size.x - 1, left_tile + 1);
                    return texelFetch(active_tile_tex, ivec2(left_tile, tile_y), 0).x > 0.5
                        || texelFetch(active_tile_tex, ivec2(right_tile, tile_y), 0).x > 0.5;
                }}
                if (cell.x % TILE_SIZE == 0 && cell.x > 0) {{
                    int right_tile = cell.x / TILE_SIZE;
                    int left_tile = max(0, right_tile - 1);
                    return texelFetch(active_tile_tex, ivec2(left_tile, tile_y), 0).x > 0.5
                        || texelFetch(active_tile_tex, ivec2(right_tile, tile_y), 0).x > 0.5;
                }}
                return false;
            }}

            void store_state(
                ivec2 cell,
                float material,
                float phase,
                float flagsv,
                vec4 timerv,
                float tempv,
                float integrityv,
                vec2 velocityv
            ) {{
                imageStore(material_out_img, cell, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, cell, vec4(flagsv, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, timerv);
                imageStore(temp_out_img, cell, vec4(tempv, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, cell, vec4(integrityv, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, cell, vec4(velocityv, 0.0, 0.0));
            }}

            int liquid_kind_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].z + 0.5);
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (boundary_active(gid)) {{
                    if (gid.x % TILE_SIZE == TILE_SIZE - 1 && gid.x + 1 < cell_grid_size.x) {{
                        ivec2 right = ivec2(gid.x + 1, gid.y);
                        float right_material = texelFetch(material_in_tex, right, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_liquid && liquid_kind_for(material) == 1 && right_material < 0.5) {{
                            store_state(gid, 0.0, 0.0, 0.0, vec4(0.0), 0.0, 0.0, vec2(0.0));
                            return;
                        }}
                    }} else if (gid.x % TILE_SIZE == 0 && gid.x > 0) {{
                        ivec2 left = ivec2(gid.x - 1, gid.y);
                        float left_material = texelFetch(material_in_tex, left, 0).x;
                        float left_phase = texelFetch(phase_in_tex, left, 0).x;
                        if (material < 0.5 && left_material > 0.5 && int(left_phase + 0.5) == phase_liquid && liquid_kind_for(left_material) == 1) {{
                            store_state(
                                gid,
                                left_material,
                                left_phase,
                                texelFetch(flags_in_tex, left, 0).x,
                                texelFetch(timer_in_tex, left, 0),
                                texelFetch(temp_in_tex, left, 0).x,
                                texelFetch(integrity_in_tex, left, 0).x,
                                texelFetch(velocity_in_tex, left, 0).xy
                            );
                            return;
                        }}
                    }}
                }}
                store_state(gid, material, phase, flagsv, timerv, tempv, integrityv, velocityv);
            }}
            """
        )
        self.programs["seam_y"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            const int TILE_SIZE = {TILE_SIZE};
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int phase_liquid;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            bool boundary_active(ivec2 cell) {{
                int tile_x = min(tile_grid_size.x - 1, max(0, cell.x / TILE_SIZE));
                if (cell.y % TILE_SIZE == TILE_SIZE - 1 && cell.y + 1 < cell_grid_size.y) {{
                    int top_tile = cell.y / TILE_SIZE;
                    int bottom_tile = min(tile_grid_size.y - 1, top_tile + 1);
                    return texelFetch(active_tile_tex, ivec2(tile_x, top_tile), 0).x > 0.5
                        || texelFetch(active_tile_tex, ivec2(tile_x, bottom_tile), 0).x > 0.5;
                }}
                if (cell.y % TILE_SIZE == 0 && cell.y > 0) {{
                    int bottom_tile = cell.y / TILE_SIZE;
                    int top_tile = max(0, bottom_tile - 1);
                    return texelFetch(active_tile_tex, ivec2(tile_x, top_tile), 0).x > 0.5
                        || texelFetch(active_tile_tex, ivec2(tile_x, bottom_tile), 0).x > 0.5;
                }}
                return false;
            }}

            void store_state(
                ivec2 cell,
                float material,
                float phase,
                float flagsv,
                vec4 timerv,
                float tempv,
                float integrityv,
                vec2 velocityv
            ) {{
                imageStore(material_out_img, cell, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, cell, vec4(flagsv, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, timerv);
                imageStore(temp_out_img, cell, vec4(tempv, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, cell, vec4(integrityv, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, cell, vec4(velocityv, 0.0, 0.0));
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (boundary_active(gid)) {{
                    if (gid.y % TILE_SIZE == TILE_SIZE - 1 && gid.y + 1 < cell_grid_size.y) {{
                        ivec2 below = ivec2(gid.x, gid.y + 1);
                        float below_material = texelFetch(material_in_tex, below, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_liquid && below_material < 0.5) {{
                            store_state(gid, 0.0, 0.0, 0.0, vec4(0.0), 0.0, 0.0, vec2(0.0));
                            return;
                        }}
                    }} else if (gid.y % TILE_SIZE == 0 && gid.y > 0) {{
                        ivec2 above = ivec2(gid.x, gid.y - 1);
                        float above_material = texelFetch(material_in_tex, above, 0).x;
                        float above_phase = texelFetch(phase_in_tex, above, 0).x;
                        if (material < 0.5 && above_material > 0.5 && int(above_phase + 0.5) == phase_liquid) {{
                            store_state(
                                gid,
                                above_material,
                                above_phase,
                                texelFetch(flags_in_tex, above, 0).x,
                                texelFetch(timer_in_tex, above, 0),
                                texelFetch(temp_in_tex, above, 0).x,
                                texelFetch(integrity_in_tex, above, 0).x,
                                texelFetch(velocity_in_tex, above, 0).xy
                            );
                            return;
                        }}
                    }}
                }}
                store_state(gid, material, phase, flagsv, timerv, tempv, integrityv, velocityv);
            }}
            """
        )
        self.programs["buoyancy_sink"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_liquid;
            uniform int phase_powder;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            bool tile_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(tile_grid_size.x - 1, max(0, cell.x / tile_size)),
                    min(tile_grid_size.y - 1, max(0, cell.y / tile_size))
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            void store_state(
                ivec2 cell,
                float material,
                float phase,
                float flagsv,
                vec4 timerv,
                float tempv,
                float integrityv,
                vec2 velocityv
            ) {{
                imageStore(material_out_img, cell, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, cell, vec4(flagsv, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, timerv);
                imageStore(temp_out_img, cell, vec4(tempv, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, cell, vec4(integrityv, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, cell, vec4(velocityv, 0.0, 0.0));
            }}

            float density_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return material_params[material_id].x;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (tile_active(gid)) {{
                    if (gid.y + 1 < cell_grid_size.y) {{
                        ivec2 below = ivec2(gid.x, gid.y + 1);
                        float below_material = texelFetch(material_in_tex, below, 0).x;
                        float below_phase = texelFetch(phase_in_tex, below, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_powder && below_material > 0.5 && int(below_phase + 0.5) == phase_liquid) {{
                            if (density_for(material) > density_for(below_material)) {{
                                store_state(
                                    gid,
                                    below_material,
                                    below_phase,
                                    texelFetch(flags_in_tex, below, 0).x,
                                    texelFetch(timer_in_tex, below, 0),
                                    texelFetch(temp_in_tex, below, 0).x,
                                    texelFetch(integrity_in_tex, below, 0).x,
                                    texelFetch(velocity_in_tex, below, 0).xy
                                );
                                return;
                            }}
                        }}
                    }}
                    if (gid.y > 0) {{
                        ivec2 above = ivec2(gid.x, gid.y - 1);
                        float above_material = texelFetch(material_in_tex, above, 0).x;
                        float above_phase = texelFetch(phase_in_tex, above, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_liquid && above_material > 0.5 && int(above_phase + 0.5) == phase_powder) {{
                            if (density_for(above_material) > density_for(material)) {{
                                store_state(
                                    gid,
                                    above_material,
                                    above_phase,
                                    texelFetch(flags_in_tex, above, 0).x,
                                    texelFetch(timer_in_tex, above, 0),
                                    texelFetch(temp_in_tex, above, 0).x,
                                    texelFetch(integrity_in_tex, above, 0).x,
                                    texelFetch(velocity_in_tex, above, 0).xy
                                );
                                return;
                            }}
                        }}
                    }}
                }}
                store_state(gid, material, phase, flagsv, timerv, tempv, integrityv, velocityv);
            }}
            """
        )
        self.programs["buoyancy_float"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_liquid;
            uniform int phase_powder;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            bool tile_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(tile_grid_size.x - 1, max(0, cell.x / tile_size)),
                    min(tile_grid_size.y - 1, max(0, cell.y / tile_size))
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            void store_state(
                ivec2 cell,
                float material,
                float phase,
                float flagsv,
                vec4 timerv,
                float tempv,
                float integrityv,
                vec2 velocityv
            ) {{
                imageStore(material_out_img, cell, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, cell, vec4(flagsv, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, timerv);
                imageStore(temp_out_img, cell, vec4(tempv, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, cell, vec4(integrityv, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, cell, vec4(velocityv, 0.0, 0.0));
            }}

            float density_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return material_params[material_id].x;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (tile_active(gid)) {{
                    if (gid.y + 1 < cell_grid_size.y) {{
                        ivec2 below = ivec2(gid.x, gid.y + 1);
                        float below_material = texelFetch(material_in_tex, below, 0).x;
                        float below_phase = texelFetch(phase_in_tex, below, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_liquid && below_material > 0.5 && int(below_phase + 0.5) == phase_powder) {{
                            if (density_for(below_material) < density_for(material)) {{
                                store_state(
                                    gid,
                                    below_material,
                                    below_phase,
                                    texelFetch(flags_in_tex, below, 0).x,
                                    texelFetch(timer_in_tex, below, 0),
                                    texelFetch(temp_in_tex, below, 0).x,
                                    texelFetch(integrity_in_tex, below, 0).x,
                                    texelFetch(velocity_in_tex, below, 0).xy
                                );
                                return;
                            }}
                        }}
                    }}
                    if (gid.y > 0) {{
                        ivec2 above = ivec2(gid.x, gid.y - 1);
                        float above_material = texelFetch(material_in_tex, above, 0).x;
                        float above_phase = texelFetch(phase_in_tex, above, 0).x;
                        if (material > 0.5 && int(phase + 0.5) == phase_powder && above_material > 0.5 && int(above_phase + 0.5) == phase_liquid) {{
                            if (density_for(material) < density_for(above_material)) {{
                                store_state(
                                    gid,
                                    above_material,
                                    above_phase,
                                    texelFetch(flags_in_tex, above, 0).x,
                                    texelFetch(timer_in_tex, above, 0),
                                    texelFetch(temp_in_tex, above, 0).x,
                                    texelFetch(integrity_in_tex, above, 0).x,
                                    texelFetch(velocity_in_tex, above, 0).xy
                                );
                                return;
                            }}
                        }}
                    }}
                }}
                store_state(gid, material, phase, flagsv, timerv, tempv, integrityv, velocityv);
            }}
            """
        )
        self.programs["copy_with_pending"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D displaced_in_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=7) writeonly uniform image2D displaced_out_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                imageStore(material_out_img, gid, vec4(texelFetch(material_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(texelFetch(phase_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, gid, vec4(texelFetch(flags_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, texelFetch(timer_in_tex, gid, 0));
                imageStore(temp_out_img, gid, vec4(texelFetch(temp_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(texelFetch(integrity_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, vec4(texelFetch(velocity_in_tex, gid, 0).xy, 0.0, 0.0));
                imageStore(displaced_out_img, gid, vec4(texelFetch(displaced_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["placeholder_displace"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_liquid;
            uniform int placeholder_material_id;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(binding=8) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) uniform image2D material_out_img;
            layout(r32f, binding=1) uniform image2D phase_out_img;
            layout(r32f, binding=2) uniform image2D flags_out_img;
            layout(rgba32f, binding=3) uniform image2D timer_out_img;
            layout(r32f, binding=4) uniform image2D temp_out_img;
            layout(r32f, binding=5) uniform image2D integrity_out_img;
            layout(rg32f, binding=6) uniform image2D velocity_out_img;
            layout(r32f, binding=7) uniform image2D displaced_out_img;

            bool tile_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(tile_grid_size.x - 1, max(0, cell.x / tile_size)),
                    min(tile_grid_size.y - 1, max(0, cell.y / tile_size))
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            bool is_placeholder(ivec2 cell) {{
                return int(texelFetch(material_in_tex, cell, 0).x + 0.5) == placeholder_material_id;
            }}

            float base_integrity_for(int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                return material_params[material_id].y;
            }}

            bool try_emit(ivec2 target, float liquid_material, float source_temp, vec2 source_velocity, vec2 push_velocity) {{
                if (target.x < 0 || target.y < 0 || target.x >= cell_grid_size.x || target.y >= cell_grid_size.y) {{
                    return false;
                }}
                if (texelFetch(material_in_tex, target, 0).x > 0.5) {{
                    return false;
                }}
                int liquid_id = int(liquid_material + 0.5);
                imageStore(material_out_img, target, vec4(liquid_material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, target, vec4(float(phase_liquid), 0.0, 0.0, 0.0));
                imageStore(flags_out_img, target, vec4(0.0, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, target, vec4(0.0));
                imageStore(temp_out_img, target, vec4(source_temp, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, target, vec4(base_integrity_for(liquid_id), 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, target, vec4(source_velocity + push_velocity, 0.0, 0.0));
                return true;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                if (!tile_active(gid) || !is_placeholder(gid)) {{
                    return;
                }}
                float liquid_material = texelFetch(displaced_in_tex, gid, 0).x;
                if (liquid_material < 0.5) {{
                    return;
                }}
                int left = gid.x;
                while (left > 0 && is_placeholder(ivec2(left - 1, gid.y))) {{
                    left -= 1;
                }}
                int right = gid.x + 1;
                while (right < cell_grid_size.x && is_placeholder(ivec2(right, gid.y))) {{
                    right += 1;
                }}
                int seg_len = right - left;
                int idx = gid.x - left;
                bool prefer_left = idx * 2 < seg_len;
                bool top_exposed = gid.y == 0 || !is_placeholder(ivec2(gid.x, gid.y - 1));
                int left_target_x = left - 1 - idx;
                int right_target_x = right + (seg_len - 1 - idx);
                float source_temp = texelFetch(temp_in_tex, gid, 0).x;
                vec2 source_velocity = texelFetch(velocity_in_tex, gid, 0).xy;
                bool emitted = false;
                if (prefer_left) {{
                    emitted = try_emit(ivec2(left_target_x, gid.y), liquid_material, source_temp, source_velocity, vec2(-1.2, -0.15));
                    if (!emitted) {{
                        emitted = try_emit(ivec2(right_target_x, gid.y), liquid_material, source_temp, source_velocity, vec2(1.2, -0.15));
                    }}
                    if (!emitted && top_exposed) {{
                        emitted = try_emit(ivec2(left_target_x, gid.y - 1), liquid_material, source_temp, source_velocity, vec2(-0.8, -0.65));
                    }}
                    if (!emitted && top_exposed) {{
                        emitted = try_emit(ivec2(right_target_x, gid.y - 1), liquid_material, source_temp, source_velocity, vec2(0.8, -0.65));
                    }}
                }} else {{
                    emitted = try_emit(ivec2(right_target_x, gid.y), liquid_material, source_temp, source_velocity, vec2(1.2, -0.15));
                    if (!emitted) {{
                        emitted = try_emit(ivec2(left_target_x, gid.y), liquid_material, source_temp, source_velocity, vec2(-1.2, -0.15));
                    }}
                    if (!emitted && top_exposed) {{
                        emitted = try_emit(ivec2(right_target_x, gid.y - 1), liquid_material, source_temp, source_velocity, vec2(0.8, -0.65));
                    }}
                    if (!emitted && top_exposed) {{
                        emitted = try_emit(ivec2(left_target_x, gid.y - 1), liquid_material, source_temp, source_velocity, vec2(-0.8, -0.65));
                    }}
                }}
                if (emitted) {{
                    imageStore(displaced_out_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["cleanup_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int phase_falling_island;
            layout(binding=0) uniform sampler2D material_pre_tex;
            layout(binding=1) uniform sampler2D phase_pre_tex;
            layout(binding=2) uniform sampler2D material_final_tex;
            layout(binding=3) uniform sampler2D phase_final_tex;
            layout(binding=4) uniform sampler2D island_in_tex;
            layout(binding=5) uniform sampler2D entity_in_tex;
            layout(binding=6) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            bool is_placeholder_material(int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int pre_material = int(texelFetch(material_pre_tex, gid, 0).x + 0.5);
                int pre_phase = int(texelFetch(phase_pre_tex, gid, 0).x + 0.5);
                int final_material = int(texelFetch(material_final_tex, gid, 0).x + 0.5);
                int final_phase = int(texelFetch(phase_final_tex, gid, 0).x + 0.5);
                bool changed = pre_material != final_material || pre_phase != final_phase;
                float island_value = texelFetch(island_in_tex, gid, 0).x;
                float entity_value = texelFetch(entity_in_tex, gid, 0).x;
                float displaced_value = texelFetch(displaced_in_tex, gid, 0).x;
                if (changed && !is_placeholder_material(final_material)) {{
                    entity_value = 0.0;
                    displaced_value = 0.0;
                }}
                if (
                    changed
                    && island_value > 0.5
                    && (final_phase != phase_falling_island || final_material <= 0)
                ) {{
                    island_value = 0.0;
                }}
                imageStore(island_out_img, gid, vec4(island_value, 0.0, 0.0, 0.0));
                imageStore(entity_out_img, gid, vec4(entity_value, 0.0, 0.0, 0.0));
                imageStore(displaced_out_img, gid, vec4(displaced_value, 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform bool copy_cell_core;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_pre_img;
            layout(r32f, binding=1) writeonly uniform image2D material_in_img;
            layout(r32f, binding=2) writeonly uniform image2D phase_pre_img;
            layout(r32f, binding=3) writeonly uniform image2D phase_in_img;
            layout(r32f, binding=4) writeonly uniform image2D flags_in_img;
            layout(rgba32f, binding=5) writeonly uniform image2D timer_in_img;
            layout(r32f, binding=6) writeonly uniform image2D temp_in_img;
            layout(r32f, binding=7) writeonly uniform image2D integrity_in_img;

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
                    float material = float(word0 & 0xFFFFu);
                    float phase = float((word0 >> 16u) & 0xFFu);
                    imageStore(material_pre_img, gid, vec4(material, 0.0, 0.0, 0.0));
                    imageStore(material_in_img, gid, vec4(material, 0.0, 0.0, 0.0));
                    imageStore(phase_pre_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                    imageStore(phase_in_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                    imageStore(flags_in_img, gid, vec4(float((word0 >> 24u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(temp_in_img, gid, vec4(uintBitsToFloat(bridge_cell_core[word_index + 2]), 0.0, 0.0, 0.0));
                    imageStore(timer_in_img, gid, unpack_timer(bridge_cell_core[word_index + 3]));
                    imageStore(integrity_in_img, gid, vec4(float(bridge_cell_core[word_index + 4] & 0xFFFFu), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["load_bridge_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
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
            layout(rg32f, binding=0) writeonly uniform image2D velocity_in_img;
            layout(r32f, binding=1) writeonly uniform image2D island_in_img;
            layout(r32f, binding=2) writeonly uniform image2D entity_in_img;
            layout(r32f, binding=3) writeonly uniform image2D displaced_in_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                if (copy_cell_core) {{
                    int word_index = cell_index * 5;
                    imageStore(velocity_in_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
                }}
                if (copy_island_id) {{
                    imageStore(island_in_img, gid, vec4(float(bridge_island_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_entity_id) {{
                    imageStore(entity_in_img, gid, vec4(float(bridge_entity_id[cell_index]), 0.0, 0.0, 0.0));
                }}
                if (copy_displaced_material) {{
                    imageStore(displaced_in_img, gid, vec4(float(bridge_displaced[cell_index]), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["publish_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D flags_tex;
            layout(binding=3) uniform sampler2D timer_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D integrity_tex;
            layout(binding=6) uniform sampler2D velocity_tex;
            layout(binding=7) uniform sampler2D island_tex;
            layout(binding=8) uniform sampler2D entity_tex;
            layout(binding=9) uniform sampler2D displaced_tex;
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
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

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

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        solve_tile_mask: np.ndarray,
    ) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "liquid input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "active_tile_ttl",
        )
        upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
        upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
        upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
        upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
        upload_active_from_cpu = not self._active_scheduler_gpu_authoritative(world)
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
        self.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
        self.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_cell_state_from_cpu:
            resources.material_pre.write(world.material_id.astype("f4").tobytes())
            resources.material_in.write(world.material_id.astype("f4").tobytes())
            resources.material_out.write(world.material_id.astype("f4").tobytes())
            resources.phase_pre.write(world.phase.astype("f4").tobytes())
            resources.phase_in.write(world.phase.astype("f4").tobytes())
            resources.phase_out.write(world.phase.astype("f4").tobytes())
            resources.flags_in.write(world.cell_flags.astype("f4").tobytes())
            resources.flags_out.write(world.cell_flags.astype("f4").tobytes())
            resources.timer_in.write(world.timer_pack.astype("f4").tobytes())
            resources.timer_out.write(world.timer_pack.astype("f4").tobytes())
            resources.temp_in.write(world.cell_temperature.astype("f4").tobytes())
            resources.temp_out.write(world.cell_temperature.astype("f4").tobytes())
            resources.integrity_in.write(world.integrity.astype("f4").tobytes())
            resources.integrity_out.write(world.integrity.astype("f4").tobytes())
            resources.velocity_in.write(world.velocity.astype("f4").tobytes())
            resources.velocity_out.write(world.velocity.astype("f4").tobytes())
        if upload_island_id_from_cpu:
            resources.island_in.write(world.island_id.astype("f4").tobytes())
            resources.island_out.write(world.island_id.astype("f4").tobytes())
        if upload_entity_id_from_cpu:
            resources.entity_in.write(world.entity_id.astype("f4").tobytes())
            resources.entity_out.write(world.entity_id.astype("f4").tobytes())
        if upload_active_from_cpu:
            self._upload_active_tile_mask(resources, solve_tile_mask)
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        if upload_displaced_from_cpu:
            resources.displaced_in.write(world.placeholder_displaced_material.astype("f4").tobytes())
            resources.displaced_out.write(world.placeholder_displaced_material.astype("f4").tobytes())
        material_table = world.bridge.shadow_typed_tables["material_table"]
        table_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        if resources.material_params_signature != table_signature:
            params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
            count = min(MAX_MATERIALS, int(material_table.shape[0]))
            params[:count, 0] = material_table[:count]["density"]
            params[:count, 1] = material_table[:count]["base_integrity"]
            params[:count, 2] = material_table[:count]["liquid_solver_kind_id"]
            params[:count, 3] = material_table[:count]["render_group_id"].astype("f4")
            resources.material_params.write(params.tobytes())
            resources.material_params_signature = table_signature

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _active_scheduler_gpu_authoritative(self, world: "WorldEngine") -> bool:
        return self._formal_gpu_frame(world) and "active_tile_ttl" in world.bridge.gpu_authoritative_resources

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative input state")
        group_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        group_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_pre.bind_to_image(0, read=False, write=True)
            resources.material_in.bind_to_image(1, read=False, write=True)
            resources.phase_pre.bind_to_image(2, read=False, write=True)
            resources.phase_in.bind_to_image(3, read=False, write=True)
            resources.flags_in.bind_to_image(4, read=False, write=True)
            resources.timer_in.bind_to_image(5, read=False, write=True)
            resources.temp_in.bind_to_image(6, read=False, write=True)
            resources.integrity_in.bind_to_image(7, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(bridge.ctx)

        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.velocity_in.bind_to_image(0, read=False, write=True)
            resources.island_in.bind_to_image(1, read=False, write=True)
            resources.entity_in.bind_to_image(2, read=False, write=True)
            resources.displaced_in.bind_to_image(3, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPULiquidResources, *, use_out: bool = False) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative output state")
            return
        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        material_tex = resources.material_out if use_out else resources.material_in
        phase_tex = resources.phase_out if use_out else resources.phase_in
        flags_tex = resources.flags_out if use_out else resources.flags_in
        timer_tex = resources.timer_out if use_out else resources.timer_in
        temp_tex = resources.temp_out if use_out else resources.temp_in
        integrity_tex = resources.integrity_out if use_out else resources.integrity_in
        velocity_tex = resources.velocity_out if use_out else resources.velocity_in
        material_tex.use(location=0)
        phase_tex.use(location=1)
        flags_tex.use(location=2)
        timer_tex.use(location=3)
        temp_tex.use(location=4)
        integrity_tex.use(location=5)
        velocity_tex.use(location=6)
        resources.island_out.use(location=7)
        resources.entity_out.use(location=8)
        resources.displaced_in.use(location=9)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
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

    def _upload_active_tile_mask(self, resources: GPULiquidResources, tile_mask: np.ndarray) -> None:
        resources.active_tile_tex.write(np.asarray(tile_mask, dtype="f4").tobytes())

    def _load_authoritative_active_tile_mask(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge active scheduler resources")
        program = self.programs["load_active_tiles"]
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["expansion_radius"].value = int(expansion_radius)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        resources.active_tile_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.active.tile_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.active.tile_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)

    def _run_tile_solve(self, world: "WorldEngine", resources: GPULiquidResources, group_x: int, group_y: int) -> None:
        program = self.programs["tile_solve"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["phase_liquid"].value = int(Phase.LIQUID)
        resources.material_in.use(location=0)
        resources.phase_in.use(location=1)
        resources.flags_in.use(location=2)
        resources.timer_in.use(location=3)
        resources.temp_in.use(location=4)
        resources.integrity_in.use(location=5)
        resources.velocity_in.use(location=6)
        resources.active_tile_tex.use(location=7)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_out.bind_to_image(0, read=False, write=True)
        resources.phase_out.bind_to_image(1, read=False, write=True)
        resources.flags_out.bind_to_image(2, read=False, write=True)
        resources.timer_out.bind_to_image(3, read=False, write=True)
        resources.temp_out.bind_to_image(4, read=False, write=True)
        resources.integrity_out.bind_to_image(5, read=False, write=True)
        resources.velocity_out.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_seam_pass(
        self,
        program_name: str,
        world: "WorldEngine",
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        active_tile_tex: Any,
    ) -> None:
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["phase_liquid"].value = int(Phase.LIQUID)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        active_tile_tex.use(location=7)
        self.resources.material_params.bind_to_storage_buffer(binding=0)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _run_buoyancy_pass(
        self,
        program_name: str,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> None:
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_powder"].value = int(Phase.POWDER)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        resources.active_tile_tex.use(location=7)
        resources.material_params.bind_to_storage_buffer(binding=0)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _run_copy_for_placeholder(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        displaced_in: Any,
        displaced_out: Any,
    ) -> None:
        program = self.programs["copy_with_pending"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        displaced_in.use(location=7)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        displaced_out.bind_to_image(7, read=False, write=True)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _run_placeholder_displacement(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        displaced_in: Any,
        displaced_out: Any,
    ) -> None:
        program = self.programs["placeholder_displace"]
        ctx = world.bridge.ctx
        assert ctx is not None
        material_table = world.bridge.shadow_typed_tables["material_table"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        resources.active_tile_tex.use(location=7)
        displaced_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        write_resources[0].bind_to_image(0)
        write_resources[1].bind_to_image(1)
        write_resources[2].bind_to_image(2)
        write_resources[3].bind_to_image(3)
        write_resources[4].bind_to_image(4)
        write_resources[5].bind_to_image(5)
        write_resources[6].bind_to_image(6)
        displaced_out.bind_to_image(7)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _run_cleanup_runtime(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["cleanup_runtime"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_pre.use(location=0)
        resources.phase_pre.use(location=1)
        resources.material_in.use(location=2)
        resources.phase_in.use(location=3)
        resources.island_in.use(location=4)
        resources.entity_in.use(location=5)
        resources.displaced_out.use(location=6)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.island_out.bind_to_image(0, read=False, write=True)
        resources.entity_out.bind_to_image(1, read=False, write=True)
        resources.displaced_in.bind_to_image(2, read=False, write=True)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(ctx)

    def _download_outputs(self, world: "WorldEngine", resources: GPULiquidResources, *, use_in: bool = False) -> None:
        material = resources.material_in if use_in else resources.material_out
        phase = resources.phase_in if use_in else resources.phase_out
        flags = resources.flags_in if use_in else resources.flags_out
        timer = resources.timer_in if use_in else resources.timer_out
        temp = resources.temp_in if use_in else resources.temp_out
        integrity = resources.integrity_in if use_in else resources.integrity_out
        velocity = resources.velocity_in if use_in else resources.velocity_out
        displaced = resources.displaced_in if use_in else resources.displaced_out
        world.material_id[:] = np.rint(
            np.frombuffer(material.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(phase.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_flags[:] = np.rint(
            np.frombuffer(flags.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.timer_pack[:] = np.rint(
            np.frombuffer(timer.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(temp.read(), dtype="f4").reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(integrity.read(), dtype="f4").reshape((world.height, world.width))
        world.velocity[:] = np.frombuffer(velocity.read(), dtype="f4").reshape((world.height, world.width, 2))
        world.placeholder_displaced_material[:] = np.rint(
            np.frombuffer(displaced.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_out.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.entity_id[:] = np.rint(
            np.frombuffer(resources.entity_out.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
            | getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0),
        )
