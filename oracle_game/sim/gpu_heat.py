from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _active_scheduler_gpu_authoritative,
    _ensure_material_flags_buffer,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
    mark_collapse_structure_dirty_tiles_from_bridge_cell_core,
)
from oracle_game.types import Phase


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_GAS_SPECIES = 256
FREEZE_COLD_NEIGHBOR_THRESHOLD = 4


@dataclass(slots=True)
class GPUHeatResources:
    signature: tuple[int, int, int, int, int]
    material_tex: Any
    material_out_tex: Any
    phase_tex: Any
    phase_out_tex: Any
    cell_flags_tex: Any
    cell_flags_out_tex: Any
    timer_tex: Any
    timer_out_tex: Any
    integrity_tex: Any
    integrity_out_tex: Any
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    velocity_tex: Any
    velocity_out_tex: Any
    temp_ping: Any
    temp_pong: Any
    phase_target_tex: Any
    boil_target_tex: Any
    gas_tex: Any
    gas_out_tex: Any
    condense_target_tex: Any
    ambient_ping: Any
    ambient_pong: Any
    active_tile_tex: Any
    material_params: Any
    material_response_params: Any
    material_phase_params: Any
    gas_params: Any
    material_params_signature: tuple[int, int] | None = None
    gas_params_signature: tuple[int, int] | None = None


@dataclass(slots=True)
class GPUHeatStageTargets:
    phase_targets: np.ndarray
    boil_targets: np.ndarray
    condense_targets: np.ndarray

    @property
    def empty(self) -> bool:
        return (
            self.phase_targets.size == 0
            and self.boil_targets.size == 0
            and self.condense_targets.size == 0
        )

    @classmethod
    def empty_sentinel(cls) -> "GPUHeatStageTargets":
        return cls(
            phase_targets=np.zeros((0, 0), dtype=np.int32),
            boil_targets=np.zeros((0, 0), dtype=np.int32),
            condense_targets=np.zeros((0, 0, 0), dtype=np.bool_),
        )


class GPUHeatPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUHeatResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def step(
        self,
        world: "WorldEngine",
        dt: float,
        *,
        solve_tile_mask: np.ndarray,
        ambient_iterations: int,
    ) -> GPUHeatStageTargets:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU heat pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
        with self._profile_pass(world, "ambient_diffuse"):
            self._run_ambient_diffuse(world, resources, gas_group_x, gas_group_y, iterations=ambient_iterations)
        with self._profile_pass(world, "cell_heat"):
            self._run_cell_heat(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "ambient_exchange"):
            self._run_ambient_exchange(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "ambient_feedback"):
            self._run_ambient_feedback(world, dt, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "phase_targets"):
            self._run_phase_targets(world, resources, group_x, group_y)
        with self._profile_pass(world, "boil_targets"):
            self._run_boil_targets(world, resources, group_x, group_y)
        with self._profile_pass(world, "condense_targets"):
            self._run_condense_targets(world, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "apply_cell_targets"):
            self._run_apply_cell_targets(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "apply_gas_targets"):
            self._run_apply_gas_targets(world, dt, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "apply_condense_cells"):
            self._run_apply_condense_cells(world, resources, group_x, group_y)
        with self._profile_pass(world, "publish_bridge_outputs"):
            self._publish_bridge_outputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
        self.last_cpu_mirror_downloaded = not (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            with self._profile_pass(world, "download_outputs"):
                return self._download_outputs(world, resources)
        return self._empty_stage_targets(world)

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
            self.resources.timer_tex,
            self.resources.timer_out_tex,
            self.resources.integrity_tex,
            self.resources.integrity_out_tex,
            self.resources.island_id_tex,
            self.resources.island_id_out_tex,
            self.resources.entity_id_tex,
            self.resources.entity_id_out_tex,
            self.resources.displaced_tex,
            self.resources.displaced_out_tex,
            self.resources.velocity_tex,
            self.resources.velocity_out_tex,
            self.resources.temp_ping,
            self.resources.temp_pong,
            self.resources.phase_target_tex,
            self.resources.boil_target_tex,
            self.resources.gas_tex,
            self.resources.gas_out_tex,
            self.resources.condense_target_tex,
            self.resources.ambient_ping,
            self.resources.ambient_pong,
            self.resources.active_tile_tex,
            self.resources.material_params,
            self.resources.material_response_params,
            self.resources.material_phase_params,
            self.resources.gas_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUHeatResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height, world.gas_concentration.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        gas_count = signature[4]
        material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        timer_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        timer_out_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        integrity_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        velocity_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        velocity_out_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        temp_ping = ctx.texture((world.width, world.height), 1, dtype="f4")
        temp_pong = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        boil_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        gas_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        gas_out_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        condense_target_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        ambient_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        for texture in (
            material_tex,
            material_out_tex,
            phase_tex,
            phase_out_tex,
            cell_flags_tex,
            cell_flags_out_tex,
            timer_tex,
            timer_out_tex,
            integrity_tex,
            integrity_out_tex,
            island_id_tex,
            island_id_out_tex,
            entity_id_tex,
            entity_id_out_tex,
            displaced_tex,
            displaced_out_tex,
            velocity_tex,
            velocity_out_tex,
            temp_ping,
            temp_pong,
            phase_target_tex,
            boil_target_tex,
            gas_tex,
            gas_out_tex,
            condense_target_tex,
            ambient_ping,
            ambient_pong,
            active_tile_tex,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        material_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        material_response_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        material_phase_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        gas_params = ctx.buffer(reserve=MAX_GAS_SPECIES * 4 * 4, dynamic=True)
        self.resources = GPUHeatResources(
            signature=signature,
            material_tex=material_tex,
            material_out_tex=material_out_tex,
            phase_tex=phase_tex,
            phase_out_tex=phase_out_tex,
            cell_flags_tex=cell_flags_tex,
            cell_flags_out_tex=cell_flags_out_tex,
            timer_tex=timer_tex,
            timer_out_tex=timer_out_tex,
            integrity_tex=integrity_tex,
            integrity_out_tex=integrity_out_tex,
            island_id_tex=island_id_tex,
            island_id_out_tex=island_id_out_tex,
            entity_id_tex=entity_id_tex,
            entity_id_out_tex=entity_id_out_tex,
            displaced_tex=displaced_tex,
            displaced_out_tex=displaced_out_tex,
            velocity_tex=velocity_tex,
            velocity_out_tex=velocity_out_tex,
            temp_ping=temp_ping,
            temp_pong=temp_pong,
            phase_target_tex=phase_target_tex,
            boil_target_tex=boil_target_tex,
            gas_tex=gas_tex,
            gas_out_tex=gas_out_tex,
            condense_target_tex=condense_target_tex,
            ambient_ping=ambient_ping,
            ambient_pong=ambient_pong,
            active_tile_tex=active_tile_tex,
            material_params=material_params,
            material_response_params=material_response_params,
            material_phase_params=material_phase_params,
            gas_params=gas_params,
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=7) buffer MaterialResponseBuffer {{
                vec4 response_params[{MAX_MATERIALS}];
            }};
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D active_tile_tex;
            int material_id_at(ivec2 cell) {{
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}
            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}
            bool solve_gas_cell_active(ivec2 gas_cell) {{
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                int tile_x0 = max(0, x0 / tile_size);
                int tile_y0 = max(0, y0 / tile_size);
                int tile_x1 = min(tile_grid_size.x, (x1 + tile_size - 1) / tile_size);
                int tile_y1 = min(tile_grid_size.y, (y1 + tile_size - 1) / tile_size);
                for (int tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {{
                    for (int tile_x = tile_x0; tile_x < tile_x1; ++tile_x) {{
                        if (texelFetch(active_tile_tex, ivec2(tile_x, tile_y), 0).x > 0.5) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}
        """
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
            bool expanded_tile_active(ivec2 tile) {{
                int radius = max(0, expansion_radius);
                int source_y0 = max(0, tile.y - radius);
                int source_y1 = min(tile_grid_size.y - 1, tile.y + radius);
                int source_x0 = max(0, tile.x - radius);
                int source_x1 = min(tile_grid_size.x - 1, tile.x + radius);
                for (int source_y = source_y0; source_y <= source_y1; ++source_y) {{
                    int row_index = source_y * tile_grid_size.x;
                    for (int source_x = source_x0; source_x <= source_x1; ++source_x) {{
                        if (active_tile_ttl[row_index + source_x] > 0) {{
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
        self.programs["cell_heat"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            layout(binding=3) uniform sampler2D temp_in_tex;
            layout(binding=4) uniform sampler2D ambient_tex;
            layout(r32f, binding=5) writeonly uniform image2D temp_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float center = texelFetch(temp_in_tex, gid, 0).x;
                if (!solve_cell_active(gid)) {
                    imageStore(temp_out_img, gid, vec4(center, 0.0, 0.0, 0.0));
                    return;
                }
                int material_id = material_id_at(gid);
                float conductivity = params[material_id].x;
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, cell_grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, cell_grid_size.y - 1));
                float avg_neighbors = (
                    texelFetch(temp_in_tex, left, 0).x +
                    texelFetch(temp_in_tex, right, 0).x +
                    texelFetch(temp_in_tex, down, 0).x +
                    texelFetch(temp_in_tex, up, 0).x
                ) * 0.25;
                float heat_capacity = max(response_params[material_id].x, 1.0e-4);
                float updated = center + (conductivity / heat_capacity) * (avg_neighbors - center) * dt * 0.35;
                imageStore(temp_out_img, gid, vec4(updated, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["ambient_exchange"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            layout(binding=3) uniform sampler2D temp_in_tex;
            layout(binding=4) uniform sampler2D ambient_tex;
            layout(r32f, binding=5) writeonly uniform image2D temp_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {
                    return;
                }
                float center = texelFetch(temp_in_tex, gid, 0).x;
                if (!solve_cell_active(gid)) {
                    imageStore(temp_out_img, gid, vec4(center, 0.0, 0.0, 0.0));
                    return;
                }
                int material_id = material_id_at(gid);
                float exchange = params[material_id].y;
                float heat_capacity = max(response_params[material_id].x, 1.0e-4);
                ivec2 gas_cell = ivec2(
                    min(gid.x / gas_cell_size, gas_grid_size.x - 1),
                    min(gid.y / gas_cell_size, gas_grid_size.y - 1)
                );
                float ambient = texelFetch(ambient_tex, gas_cell, 0).x;
                float updated = center + (ambient - center) * (exchange / heat_capacity) * dt;
                imageStore(temp_out_img, gid, vec4(updated, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["ambient_feedback"] = ctx.compute_shader(
            helper
            + """
            uniform float dt;
            layout(binding=3) uniform sampler2D temp_tex;
            layout(binding=4) uniform sampler2D ambient_in_tex;
            layout(r32f, binding=5) writeonly uniform image2D ambient_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {
                    return;
                }
                float ambient = texelFetch(ambient_in_tex, gid, 0).x;
                if (!solve_gas_cell_active(gid)) {
                    imageStore(ambient_out_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                    return;
                }
                float accum = 0.0;
                int count = 0;
                for (int local_y = 0; local_y < gas_cell_size; ++local_y) {
                    for (int local_x = 0; local_x < gas_cell_size; ++local_x) {
                        ivec2 cell = ivec2(gid.x * gas_cell_size + local_x, gid.y * gas_cell_size + local_y);
                        if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {
                            continue;
                        }
                        int material_id = material_id_at(cell);
                        float exchange = params[material_id].y;
                        float heat_capacity = max(response_params[material_id].x, 1.0e-4);
                        float cell_temp = texelFetch(temp_tex, cell, 0).x;
                        float delta = (ambient - cell_temp) * (exchange / heat_capacity) * dt;
                        accum += -delta * 0.02;
                        count += 1;
                    }
                }
                float updated = ambient;
                if (count > 0) {
                    updated += accum / float(count);
                }
                imageStore(ambient_out_img, gid, vec4(updated, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["ambient_diffuse"] = ctx.compute_shader(
            helper
            + """
            layout(binding=3) uniform sampler2D ambient_in_tex;
            layout(r32f, binding=4) writeonly uniform image2D ambient_out_img;
            void main() {
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {
                    return;
                }
                float ambient = texelFetch(ambient_in_tex, gid, 0).x;
                if (!solve_gas_cell_active(gid)) {
                    imageStore(ambient_out_img, gid, vec4(ambient, 0.0, 0.0, 0.0));
                    return;
                }
                ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                ivec2 right = ivec2(min(gid.x + 1, gas_grid_size.x - 1), gid.y);
                ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                ivec2 up = ivec2(gid.x, min(gid.y + 1, gas_grid_size.y - 1));
                float lap = (
                    texelFetch(ambient_in_tex, left, 0).x +
                    texelFetch(ambient_in_tex, right, 0).x +
                    texelFetch(ambient_in_tex, down, 0).x +
                    texelFetch(ambient_in_tex, up, 0).x -
                    4.0 * ambient
                );
                imageStore(ambient_out_img, gid, vec4(ambient + 0.08 * lap, 0.0, 0.0, 0.0));
            }
            """
        )
        self.programs["phase_targets"] = ctx.compute_shader(
            helper
            + f"""
            uniform int phase_liquid;
            uniform int freeze_cold_neighbor_threshold;
            layout(std430, binding=3) buffer MaterialPhaseBuffer {{
                ivec4 phase_params[{MAX_MATERIALS}];
            }};
            layout(binding=4) uniform sampler2D phase_tex;
            layout(binding=5) uniform sampler2D temp_tex;
            layout(r32f, binding=6) writeonly uniform image2D phase_target_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                if (!solve_cell_active(gid)) {{
                    imageStore(phase_target_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }}
                int material_id = material_id_at(gid);
                if (material_id <= 0) {{
                    imageStore(phase_target_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }}
                vec4 material_param = params[material_id];
                ivec4 transition = phase_params[material_id];
                float melt_point = material_param.z;
                float temperature = texelFetch(temp_tex, gid, 0).x;
                int target_material = 0;
                if (!isnan(melt_point)) {{
                    if (transition.y > 0 && temperature > melt_point) {{
                        target_material = transition.y;
                    }} else if (transition.z > 0 && transition.x == phase_liquid && temperature < melt_point) {{
                        ivec2 left = ivec2(max(gid.x - 1, 0), gid.y);
                        ivec2 right = ivec2(min(gid.x + 1, cell_grid_size.x - 1), gid.y);
                        ivec2 down = ivec2(gid.x, max(gid.y - 1, 0));
                        ivec2 up = ivec2(gid.x, min(gid.y + 1, cell_grid_size.y - 1));
                        int cold_count = 1;
                        cold_count += texelFetch(temp_tex, left, 0).x < melt_point ? 1 : 0;
                        cold_count += texelFetch(temp_tex, right, 0).x < melt_point ? 1 : 0;
                        cold_count += texelFetch(temp_tex, down, 0).x < melt_point ? 1 : 0;
                        cold_count += texelFetch(temp_tex, up, 0).x < melt_point ? 1 : 0;
                        if (cold_count >= freeze_cold_neighbor_threshold) {{
                            target_material = transition.z;
                        }}
                    }}
                }}
                imageStore(phase_target_img, gid, vec4(float(target_material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["apply_cell_targets"] = ctx.compute_shader(
            helper
            + f"""
            uniform float dt;
            uniform int phase_falling_island;
            uniform int phase_liquid;
            layout(std430, binding=3) buffer MaterialPhaseBuffer {{
                ivec4 phase_params[{MAX_MATERIALS}];
            }};
            layout(binding=3) uniform sampler2D phase_target_tex;
            layout(binding=4) uniform sampler2D phase_tex;
            layout(binding=5) uniform sampler2D cell_flags_tex;
            layout(binding=6) uniform sampler2D timer_tex;
            layout(binding=7) uniform sampler2D boil_target_tex;
            layout(binding=8) uniform sampler2D temp_tex;
            layout(binding=9) uniform sampler2D integrity_tex;
            layout(binding=10) uniform sampler2D island_id_tex;
            layout(binding=11) uniform sampler2D entity_id_tex;
            layout(binding=12) uniform sampler2D displaced_tex;
            layout(binding=22) uniform sampler2D ambient_tex;
            layout(binding=23) uniform sampler2D velocity_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(r32f, binding=6) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=7) writeonly uniform image2D entity_id_out_img;

            bool is_placeholder_material(int material_id) {{
                if (material_id <= 0 || material_id >= {MAX_MATERIALS}) {{
                    return false;
                }}
                return int(response_params[material_id].w + 0.5) == 7;
            }}

            void store_payload(
                ivec2 cell,
                float material_value,
                float phase_value,
                float flags_value,
                vec4 timer_value,
                float temp_value,
                float integrity_value,
                float island_value,
                float entity_value,
                float displaced_value,
                vec2 velocity_value
            ) {{
                imageStore(material_out_img, cell, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, cell, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_out_img, cell, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, cell, timer_value);
                imageStore(temp_out_img, cell, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, cell, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(island_id_out_img, cell, vec4(island_value, 0.0, 0.0, 0.0));
                imageStore(entity_id_out_img, cell, vec4(entity_value, 0.0, 0.0, 0.0));
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                bool is_active = solve_cell_active(gid);
                float material_value = texelFetch(material_tex, gid, 0).x;
                float phase_value = texelFetch(phase_tex, gid, 0).x;
                int previous_material = int(material_value + 0.5);
                int previous_phase = int(phase_value + 0.5);
                float flags_value = texelFetch(cell_flags_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_tex, gid, 0);
                float temperature = texelFetch(temp_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_tex, gid, 0).x;
                float island_value = texelFetch(island_id_tex, gid, 0).x;
                float entity_value = texelFetch(entity_id_tex, gid, 0).x;
                float displaced_value = texelFetch(displaced_tex, gid, 0).x;
                vec2 velocity_value = texelFetch(velocity_tex, gid, 0).xy;

                int target_material = is_active ? int(texelFetch(phase_target_tex, gid, 0).x + 0.5) : 0;
                if (target_material > 0) {{
                    target_material = clamp(target_material, 0, {MAX_MATERIALS - 1});
                    int resolved_phase = phase_params[target_material].x;
                    float spawn_temperature = response_params[target_material].z;
                    if (!isnan(spawn_temperature)) {{
                        temperature = max(temperature, spawn_temperature);
                    }}
                    if (is_placeholder_material(target_material)) {{
                        if (is_placeholder_material(previous_material)) {{
                            displaced_value = texelFetch(displaced_tex, gid, 0).x;
                        }} else if (previous_material != 0 && previous_phase == phase_liquid) {{
                            displaced_value = float(previous_material);
                        }} else {{
                            displaced_value = 0.0;
                        }}
                    }} else if (previous_material != 0 && previous_phase == phase_liquid) {{
                        entity_value = 0.0;
                        displaced_value = 0.0;
                    }} else {{
                        entity_value = 0.0;
                        displaced_value = 0.0;
                    }}
                    material_value = float(target_material);
                    phase_value = float(resolved_phase);
                    flags_value = 0.0;
                    timer_value = vec4(0.0);
                    integrity_value = response_params[target_material].y;
                    island_value = resolved_phase == phase_falling_island ? island_value : 0.0;
                }}

                int boil_target = is_active ? int(texelFetch(boil_target_tex, gid, 0).x + 0.5) : 0;
                if (boil_target > 0) {{
                    integrity_value -= 0.5 * dt;
                    if (integrity_value <= 0.0) {{
                        ivec2 gas_cell = ivec2(
                            min(gid.x / gas_cell_size, gas_grid_size.x - 1),
                            min(gid.y / gas_cell_size, gas_grid_size.y - 1)
                        );
                        material_value = 0.0;
                        phase_value = 0.0;
                        flags_value = 0.0;
                        timer_value = vec4(0.0);
                        temperature = texelFetch(ambient_tex, gas_cell, 0).x;
                        integrity_value = 0.0;
                        island_value = 0.0;
                        entity_value = 0.0;
                        displaced_value = 0.0;
                        velocity_value = vec2(0.0);
                    }}
                }}
                store_payload(
                    gid,
                    material_value,
                    phase_value,
                    flags_value,
                    timer_value,
                    temperature,
                    integrity_value,
                    island_value,
                    entity_value,
                    displaced_value,
                    velocity_value
                );
            }}
            """
        )
        self.programs["apply_cell_aux_targets"] = ctx.compute_shader(
            helper
            + f"""
            uniform float dt;
            uniform int phase_falling_island;
            uniform int phase_liquid;
            layout(std430, binding=3) buffer MaterialPhaseBuffer {{
                ivec4 phase_params[{MAX_MATERIALS}];
            }};
            layout(binding=3) uniform sampler2D phase_target_tex;
            layout(binding=4) uniform sampler2D phase_tex;
            layout(binding=5) uniform sampler2D boil_target_tex;
            layout(binding=6) uniform sampler2D integrity_tex;
            layout(binding=7) uniform sampler2D displaced_tex;
            layout(binding=8) uniform sampler2D velocity_tex;
            layout(r32f, binding=0) writeonly uniform image2D displaced_out_img;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_out_img;

            bool is_placeholder_material(int material_id) {{
                if (material_id <= 0 || material_id >= {MAX_MATERIALS}) {{
                    return false;
                }}
                return int(response_params[material_id].w + 0.5) == 7;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                bool is_active = solve_cell_active(gid);
                int previous_material = int(texelFetch(material_tex, gid, 0).x + 0.5);
                int previous_phase = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                float displaced_value = texelFetch(displaced_tex, gid, 0).x;
                vec2 velocity_value = texelFetch(velocity_tex, gid, 0).xy;

                int target_material = is_active ? int(texelFetch(phase_target_tex, gid, 0).x + 0.5) : 0;
                if (target_material > 0) {{
                    target_material = clamp(target_material, 0, {MAX_MATERIALS - 1});
                    if (is_placeholder_material(target_material)) {{
                        if (is_placeholder_material(previous_material)) {{
                            displaced_value = texelFetch(displaced_tex, gid, 0).x;
                        }} else if (previous_material != 0 && previous_phase == phase_liquid) {{
                            displaced_value = float(previous_material);
                        }} else {{
                            displaced_value = 0.0;
                        }}
                    }} else {{
                        displaced_value = 0.0;
                    }}
                }}

                int boil_target = is_active ? int(texelFetch(boil_target_tex, gid, 0).x + 0.5) : 0;
                if (boil_target > 0) {{
                    float integrity_value = texelFetch(integrity_tex, gid, 0).x - 0.5 * dt;
                    if (integrity_value <= 0.0) {{
                        displaced_value = 0.0;
                        velocity_value = vec2(0.0);
                    }}
                }}
                imageStore(displaced_out_img, gid, vec4(displaced_value, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, vec4(velocity_value, 0.0, 0.0));
            }}
            """
        )
        self.programs["boil_targets"] = ctx.compute_shader(
            helper
            + f"""
            layout(std430, binding=3) buffer MaterialPhaseBuffer {{
                ivec4 phase_params[{MAX_MATERIALS}];
            }};
            layout(binding=4) uniform sampler2D temp_tex;
            layout(r32f, binding=5) writeonly uniform image2D boil_target_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                if (!solve_cell_active(gid)) {{
                    imageStore(boil_target_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }}
                int material_id = material_id_at(gid);
                if (material_id <= 0) {{
                    imageStore(boil_target_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    return;
                }}
                float boil_point = params[material_id].w;
                int gas_species_plus_one = phase_params[material_id].w;
                float temperature = texelFetch(temp_tex, gid, 0).x;
                int target_species = 0;
                if (!isnan(boil_point) && gas_species_plus_one > 0 && temperature > boil_point) {{
                    target_species = gas_species_plus_one;
                }}
                imageStore(boil_target_img, gid, vec4(float(target_species), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["condense_targets"] = ctx.compute_shader(
            helper
            + f"""
            uniform int gas_species_count;
            layout(std430, binding=3) buffer GasCondenseBuffer {{
                vec4 gas_params[{MAX_GAS_SPECIES}];
            }};
            layout(binding=4) uniform sampler2DArray gas_tex;
            layout(binding=5) uniform sampler2D ambient_tex;
            layout(r32f, binding=6) writeonly uniform image2DArray condense_target_img;
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                if (!solve_gas_cell_active(gid)) {{
                    for (int species = 0; species < gas_species_count; ++species) {{
                        imageStore(condense_target_img, ivec3(gid, species), vec4(0.0, 0.0, 0.0, 0.0));
                    }}
                    return;
                }}
                float ambient = texelFetch(ambient_tex, gid, 0).x;
                for (int species = 0; species < gas_species_count; ++species) {{
                    vec4 species_param = gas_params[species];
                    float condense_point = species_param.x;
                    int target_material_id = int(species_param.y + 0.5);
                    float concentration = texelFetch(gas_tex, ivec3(gid, species), 0).x;
                    float target = 0.0;
                    if (!isnan(condense_point) && target_material_id > 0 && ambient < condense_point && concentration > 0.7) {{
                        target = 1.0;
                    }}
                    imageStore(condense_target_img, ivec3(gid, species), vec4(target, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        self.programs["apply_gas_targets"] = ctx.compute_shader(
            helper
            + f"""
            uniform int gas_species_count;
            uniform float dt;
            layout(std430, binding=3) buffer GasCondenseBuffer {{
                vec4 gas_params[{MAX_GAS_SPECIES}];
            }};
            layout(binding=4) uniform sampler2DArray gas_tex;
            layout(binding=5) uniform sampler2D boil_target_tex;
            layout(binding=6) uniform sampler2DArray condense_target_tex;
            layout(binding=8) uniform sampler2D material_after_tex;
            layout(r32f, binding=0) writeonly uniform image2DArray gas_out_img;

            int empty_count_after_cell_targets(ivec2 gas_cell) {{
                int empty_count = 0;
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                for (int local_y = 0; local_y < gas_cell_size; ++local_y) {{
                    for (int local_x = 0; local_x < gas_cell_size; ++local_x) {{
                        ivec2 cell = ivec2(x0 + local_x, y0 + local_y);
                        if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                            continue;
                        }}
                        if (int(texelFetch(material_after_tex, cell, 0).x + 0.5) == 0) {{
                            empty_count += 1;
                        }}
                    }}
                }}
                return empty_count;
            }}

            int boil_count_for_species(ivec2 gas_cell, int species) {{
                int boil_count = 0;
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                for (int local_y = 0; local_y < gas_cell_size; ++local_y) {{
                    for (int local_x = 0; local_x < gas_cell_size; ++local_x) {{
                        ivec2 cell = ivec2(x0 + local_x, y0 + local_y);
                        if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                            continue;
                        }}
                        if (int(texelFetch(boil_target_tex, cell, 0).x + 0.5) == species + 1) {{
                            boil_count += 1;
                        }}
                    }}
                }}
                return boil_count;
            }}

            int condense_rank_for_species(ivec2 gas_cell, int species) {{
                int rank = 0;
                for (int other = 0; other <= species; ++other) {{
                    int target_material = int(gas_params[other].y + 0.5);
                    bool condenses = texelFetch(condense_target_tex, ivec3(gas_cell, other), 0).x > 0.5;
                    if (target_material > 0 && condenses) {{
                        rank += 1;
                    }}
                }}
                return rank;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y) {{
                    return;
                }}
                bool is_active = solve_gas_cell_active(gid);
                int empty_count = is_active ? empty_count_after_cell_targets(gid) : 0;
                for (int species = 0; species < gas_species_count; ++species) {{
                    float value = texelFetch(gas_tex, ivec3(gid, species), 0).x;
                    if (is_active) {{
                        value += float(boil_count_for_species(gid, species)) * 0.6 * dt;
                        int target_material = int(gas_params[species].y + 0.5);
                        bool condenses = texelFetch(condense_target_tex, ivec3(gid, species), 0).x > 0.5;
                        int condense_rank = condense_rank_for_species(gid, species);
                        if (target_material > 0 && condenses && condense_rank > 0 && empty_count >= condense_rank) {{
                            value = max(0.0, value - 0.6);
                        }}
                    }}
                    imageStore(gas_out_img, ivec3(gid, species), vec4(value, 0.0, 0.0, 0.0));
                }}
            }}
            """
        )
        condense_cell_helper = f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 gas_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int gas_cell_size;
            uniform int tile_size;
            uniform int gas_species_count;
            uniform int phase_falling_island;
            uniform int phase_liquid;
            layout(std430, binding=3) buffer MaterialPhaseBuffer {{
                ivec4 phase_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=7) buffer MaterialResponseBuffer {{
                vec4 response_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=8) buffer GasCondenseBuffer {{
                vec4 gas_params[{MAX_GAS_SPECIES}];
            }};
            layout(binding=2) uniform sampler2D active_tile_tex;
            bool solve_gas_cell_active(ivec2 gas_cell) {{
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                int x1 = min(cell_grid_size.x, x0 + gas_cell_size);
                int y1 = min(cell_grid_size.y, y0 + gas_cell_size);
                int tile_x0 = max(0, x0 / tile_size);
                int tile_y0 = max(0, y0 / tile_size);
                int tile_x1 = min(tile_grid_size.x, (x1 + tile_size - 1) / tile_size);
                int tile_y1 = min(tile_grid_size.y, (y1 + tile_size - 1) / tile_size);
                for (int tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {{
                    for (int tile_x = tile_x0; tile_x < tile_x1; ++tile_x) {{
                        if (texelFetch(active_tile_tex, ivec2(tile_x, tile_y), 0).x > 0.5) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}
        """
        self.programs["apply_condense_cells"] = ctx.compute_shader(
            condense_cell_helper
            + f"""
            layout(binding=4) uniform sampler2D material_after_tex;
            layout(binding=5) uniform sampler2D phase_after_tex;
            layout(binding=6) uniform sampler2D cell_flags_after_tex;
            layout(binding=9) uniform sampler2D timer_after_tex;
            layout(binding=10) uniform sampler2D temp_after_tex;
            layout(binding=11) uniform sampler2D integrity_after_tex;
            layout(binding=12) uniform sampler2D island_id_after_tex;
            layout(binding=13) uniform sampler2D entity_id_after_tex;
            layout(binding=22) uniform sampler2D displaced_after_tex;
            layout(binding=23) uniform sampler2D velocity_after_tex;
            layout(binding=24) uniform sampler2DArray condense_target_tex;
            layout(r32f, binding=0) writeonly uniform image2D material_final_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_final_img;
            layout(r32f, binding=2) writeonly uniform image2D cell_flags_final_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_final_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_final_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_final_img;
            layout(r32f, binding=6) writeonly uniform image2D displaced_final_img;
            layout(rg32f, binding=7) writeonly uniform image2D velocity_final_img;

            bool is_placeholder_material(int material_id) {{
                if (material_id <= 0 || material_id >= {MAX_MATERIALS}) {{
                    return false;
                }}
                return int(response_params[material_id].w + 0.5) == 7;
            }}

            void store_payload(
                ivec2 cell,
                float material_value,
                float phase_value,
                float flags_value,
                vec4 timer_value,
                float temp_value,
                float integrity_value,
                float island_value,
                float entity_value,
                float displaced_value,
                vec2 velocity_value
            ) {{
                imageStore(material_final_img, cell, vec4(material_value, 0.0, 0.0, 0.0));
                imageStore(phase_final_img, cell, vec4(phase_value, 0.0, 0.0, 0.0));
                imageStore(cell_flags_final_img, cell, vec4(flags_value, 0.0, 0.0, 0.0));
                imageStore(timer_final_img, cell, timer_value);
                imageStore(temp_final_img, cell, vec4(temp_value, 0.0, 0.0, 0.0));
                imageStore(integrity_final_img, cell, vec4(integrity_value, 0.0, 0.0, 0.0));
                imageStore(displaced_final_img, cell, vec4(displaced_value, 0.0, 0.0, 0.0));
                imageStore(velocity_final_img, cell, vec4(velocity_value, 0.0, 0.0));
            }}

            void copy_after_targets(ivec2 cell) {{
                store_payload(
                    cell,
                    texelFetch(material_after_tex, cell, 0).x,
                    texelFetch(phase_after_tex, cell, 0).x,
                    texelFetch(cell_flags_after_tex, cell, 0).x,
                    texelFetch(timer_after_tex, cell, 0),
                    texelFetch(temp_after_tex, cell, 0).x,
                    texelFetch(integrity_after_tex, cell, 0).x,
                    texelFetch(island_id_after_tex, cell, 0).x,
                    texelFetch(entity_id_after_tex, cell, 0).x,
                    texelFetch(displaced_after_tex, cell, 0).x,
                    texelFetch(velocity_after_tex, cell, 0).xy
                );
            }}

            int empty_rank_in_gas_cell(ivec2 cell, ivec2 gas_cell) {{
                int rank = 0;
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                for (int local_y = 0; local_y < gas_cell_size; ++local_y) {{
                    for (int local_x = 0; local_x < gas_cell_size; ++local_x) {{
                        ivec2 probe = ivec2(x0 + local_x, y0 + local_y);
                        if (probe.x >= cell_grid_size.x || probe.y >= cell_grid_size.y) {{
                            continue;
                        }}
                        if (int(texelFetch(material_after_tex, probe, 0).x + 0.5) == 0) {{
                            rank += 1;
                        }}
                        if (all(equal(probe, cell))) {{
                            return rank;
                        }}
                    }}
                }}
                return 0;
            }}

            int target_material_for_empty_rank(ivec2 gas_cell, int empty_rank) {{
                int condense_rank = 0;
                for (int species = 0; species < gas_species_count; ++species) {{
                    int target_material = int(gas_params[species].y + 0.5);
                    bool condenses = texelFetch(condense_target_tex, ivec3(gas_cell, species), 0).x > 0.5;
                    if (target_material > 0 && condenses) {{
                        condense_rank += 1;
                        if (condense_rank == empty_rank) {{
                            return clamp(target_material, 0, {MAX_MATERIALS - 1});
                        }}
                    }}
                }}
                return 0;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 gas_cell = ivec2(
                    min(gid.x / gas_cell_size, gas_grid_size.x - 1),
                    min(gid.y / gas_cell_size, gas_grid_size.y - 1)
                );
                float material_value = texelFetch(material_after_tex, gid, 0).x;
                float phase_value = texelFetch(phase_after_tex, gid, 0).x;
                float flags_value = texelFetch(cell_flags_after_tex, gid, 0).x;
                vec4 timer_value = texelFetch(timer_after_tex, gid, 0);
                float temperature = texelFetch(temp_after_tex, gid, 0).x;
                float integrity_value = texelFetch(integrity_after_tex, gid, 0).x;
                float island_value = texelFetch(island_id_after_tex, gid, 0).x;
                float entity_value = texelFetch(entity_id_after_tex, gid, 0).x;
                float displaced_value = texelFetch(displaced_after_tex, gid, 0).x;
                vec2 velocity_value = texelFetch(velocity_after_tex, gid, 0).xy;
                int current_material = int(material_value + 0.5);
                if (!solve_gas_cell_active(gas_cell) || current_material != 0) {{
                    store_payload(
                        gid,
                        material_value,
                        phase_value,
                        flags_value,
                        timer_value,
                        temperature,
                        integrity_value,
                        island_value,
                        entity_value,
                        displaced_value,
                        velocity_value
                    );
                    return;
                }}
                int empty_rank = empty_rank_in_gas_cell(gid, gas_cell);
                int target_material = target_material_for_empty_rank(gas_cell, empty_rank);
                if (target_material <= 0) {{
                    store_payload(
                        gid,
                        material_value,
                        phase_value,
                        flags_value,
                        timer_value,
                        temperature,
                        integrity_value,
                        island_value,
                        entity_value,
                        displaced_value,
                        velocity_value
                    );
                    return;
                }}
                int resolved_phase = phase_params[target_material].x;
                float spawn_temperature = response_params[target_material].z;
                if (!isnan(spawn_temperature)) {{
                    temperature = max(temperature, spawn_temperature);
                }}
                entity_value = 0.0;
                displaced_value = 0.0;
                island_value = resolved_phase == phase_falling_island ? island_value : 0.0;
                store_payload(
                    gid,
                    float(target_material),
                    float(resolved_phase),
                    0.0,
                    vec4(0.0),
                    temperature,
                    response_params[target_material].y,
                    island_value,
                    entity_value,
                    displaced_value,
                    velocity_value
                );
            }}
            """
        )
        self.programs["apply_condense_cell_aux"] = ctx.compute_shader(
            condense_cell_helper
            + f"""
            layout(binding=4) uniform sampler2D material_after_tex;
            layout(binding=5) uniform sampler2D phase_after_tex;
            layout(binding=12) uniform sampler2D island_id_after_tex;
            layout(binding=22) uniform sampler2D displaced_after_tex;
            layout(binding=23) uniform sampler2D velocity_after_tex;
            layout(binding=24) uniform sampler2DArray condense_target_tex;
            layout(r32f, binding=0) writeonly uniform image2D displaced_final_img;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_final_img;

            int empty_rank_in_gas_cell(ivec2 cell, ivec2 gas_cell) {{
                int rank = 0;
                int x0 = gas_cell.x * gas_cell_size;
                int y0 = gas_cell.y * gas_cell_size;
                for (int local_y = 0; local_y < gas_cell_size; ++local_y) {{
                    for (int local_x = 0; local_x < gas_cell_size; ++local_x) {{
                        ivec2 probe = ivec2(x0 + local_x, y0 + local_y);
                        if (probe.x >= cell_grid_size.x || probe.y >= cell_grid_size.y) {{
                            continue;
                        }}
                        if (int(texelFetch(material_after_tex, probe, 0).x + 0.5) == 0) {{
                            rank += 1;
                        }}
                        if (all(equal(probe, cell))) {{
                            return rank;
                        }}
                    }}
                }}
                return 0;
            }}

            int target_material_for_empty_rank(ivec2 gas_cell, int empty_rank) {{
                int condense_rank = 0;
                for (int species = 0; species < gas_species_count; ++species) {{
                    int target_material = int(gas_params[species].y + 0.5);
                    bool condenses = texelFetch(condense_target_tex, ivec3(gas_cell, species), 0).x > 0.5;
                    if (target_material > 0 && condenses) {{
                        condense_rank += 1;
                        if (condense_rank == empty_rank) {{
                            return clamp(target_material, 0, {MAX_MATERIALS - 1});
                        }}
                    }}
                }}
                return 0;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 gas_cell = ivec2(
                    min(gid.x / gas_cell_size, gas_grid_size.x - 1),
                    min(gid.y / gas_cell_size, gas_grid_size.y - 1)
                );
                float displaced_value = texelFetch(displaced_after_tex, gid, 0).x;
                vec2 velocity_value = texelFetch(velocity_after_tex, gid, 0).xy;
                int current_material = int(texelFetch(material_after_tex, gid, 0).x + 0.5);
                if (solve_gas_cell_active(gas_cell) && current_material == 0) {{
                    int empty_rank = empty_rank_in_gas_cell(gid, gas_cell);
                    int target_material = target_material_for_empty_rank(gas_cell, empty_rank);
                    if (target_material > 0) {{
                        displaced_value = 0.0;
                    }}
                }}
                imageStore(displaced_final_img, gid, vec4(displaced_value, 0.0, 0.0, 0.0));
                imageStore(velocity_final_img, gid, vec4(velocity_value, 0.0, 0.0));
            }}
            """
        )

        self.programs["publish_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int material_count;
            uniform int phase_falling_island;
            uniform bool mark_structure_dirty;
            uniform bool write_cell_core;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D flags_tex;
            layout(binding=3) uniform sampler2D timer_tex;
            layout(binding=4) uniform sampler2D temp_tex;
            layout(binding=5) uniform sampler2D integrity_tex;
            layout(binding=6) uniform sampler2D island_tex;
            layout(binding=7) uniform sampler2D entity_tex;
            layout(binding=8) uniform sampler2D displaced_tex;
            layout(binding=9) uniform sampler2D velocity_tex;
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;
            layout(std430, binding=0) buffer BridgeCellCoreBuffer {{
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
            layout(std430, binding=4) readonly buffer MaterialParticipationBuffer {{
                uint material_participates[];
            }};
            layout(std430, binding=5) buffer DirtyTileMaskBuffer {{
                uint dirty_tile_mask[];
            }};
            layout(std430, binding=6) buffer DirtyTileCountBuffer {{
                uint dirty_tile_count[];
            }};
            layout(std430, binding=7) buffer DirtyTileListBuffer {{
                ivec2 dirty_tile_list[];
            }};
            layout(std430, binding=8) buffer DirtyTileDispatchArgsBuffer {{
                uint dirty_tile_dispatch_args[];
            }};
            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}
            uint structure_flags(int material_id, int phase_value) {{
                if (material_id <= 0 || material_id >= material_count || phase_value == phase_falling_island) {{
                    return 0u;
                }}
                return material_participates[material_id];
            }}
            void append_dirty_tile(ivec2 tile) {{
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                uint previous = atomicOr(dirty_tile_mask[tile_index], 1u);
                if ((previous & 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(dirty_tile_count[0], 1u);
                dirty_tile_list[int(slot)] = tile;
                atomicMax(dirty_tile_dispatch_args[0], slot + 1u);
            }}
            void mark_dirty_if_structure_changed(ivec2 gid, uint previous_word, uint material, uint phase) {{
                if (!mark_structure_dirty) {{
                    return;
                }}
                int old_material = int(previous_word & 0xFFFFu);
                int old_phase = int((previous_word >> 16u) & 0xFFu);
                uint old_flags = structure_flags(old_material, old_phase);
                uint new_flags = structure_flags(int(material), int(phase));
                if (old_flags == new_flags) {{
                    return;
                }}
                ivec2 tile = gid / tile_size;
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = gid - tile_origin;
                append_dirty_tile(tile);
                if (local_cell.x == 0) {{
                    append_dirty_tile(tile + ivec2(-1, 0));
                }}
                if (local_cell.x + 1 >= tile_size) {{
                    append_dirty_tile(tile + ivec2(1, 0));
                }}
                if (local_cell.y == 0) {{
                    append_dirty_tile(tile + ivec2(0, -1));
                }}
                if (local_cell.y + 1 >= tile_size) {{
                    append_dirty_tile(tile + ivec2(0, 1));
                }}
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
                uint previous_word = bridge_cell_core[word_index];
                mark_dirty_if_structure_changed(gid, previous_word, material, phase);
                if (write_cell_core) {{
                    bridge_cell_core[word_index] = material | (phase << 16u) | (flags << 24u);
                    bridge_cell_core[word_index + 1] = packHalf2x16(velocity);
                    bridge_cell_core[word_index + 2] = floatBitsToUint(temperature);
                    bridge_cell_core[word_index + 3] = pack_timer(texelFetch(timer_tex, gid, 0));
                    bridge_cell_core[word_index + 4] = integrity;
                    imageStore(bridge_material_img, gid, vec4(float(material), 0.0, 0.0, 0.0));
                }}
                bridge_island_id[cell_index] = island;
                bridge_entity_id[cell_index] = entity;
                bridge_displaced[cell_index] = displaced;
            }}
            """
        )
        self.programs["publish_bridge_gas"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 gas_grid_size;
            uniform int species_count;
            layout(binding=0) uniform sampler2D ambient_tex;
            layout(binding=2) uniform sampler2DArray gas_tex;
            layout(r32f, binding=1) writeonly uniform image2D bridge_ambient_img;
            layout(std430, binding=4) writeonly buffer BridgeGasBuffer {{
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
                int gas_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                bridge_gas[gas_index] = max(texelFetch(gas_tex, ivec3(gid, species), 0).x, 0.0);
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
            layout(r32f, binding=4) writeonly uniform image2D material_img;
            layout(r32f, binding=5) writeonly uniform image2D phase_img;
            layout(r32f, binding=6) writeonly uniform image2D flags_img;
            layout(rgba32f, binding=7) writeonly uniform image2D timer_img;
            layout(r32f, binding=0) writeonly uniform image2D temp_img;
            layout(r32f, binding=1) writeonly uniform image2D integrity_img;
            layout(rg32f, binding=2) writeonly uniform image2D velocity_img;

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
            uniform int species_count;
            uniform bool copy_ambient;
            uniform bool copy_gas;

            layout(binding=0) uniform sampler2D bridge_ambient_tex;
            layout(std430, binding=1) readonly buffer BridgeGasBuffer {{
                float bridge_gas[];
            }};
            layout(r32f, binding=2) writeonly uniform image2D ambient_img;
            layout(r32f, binding=3) writeonly uniform image2DArray gas_img;

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                int species = int(gl_GlobalInvocationID.z);
                if (gid.x >= gas_grid_size.x || gid.y >= gas_grid_size.y || species >= species_count) {{
                    return;
                }}
                if (species == 0 && copy_ambient) {{
                    imageStore(ambient_img, gid, vec4(texelFetch(bridge_ambient_tex, gid, 0).x, 0.0, 0.0, 0.0));
                }}
                if (copy_gas) {{
                    int gas_index = (species * gas_grid_size.y + gid.y) * gas_grid_size.x + gid.x;
                    imageStore(gas_img, ivec3(gid, species), vec4(max(bridge_gas[gas_index], 0.0), 0.0, 0.0, 0.0));
                }}
            }}
            """
        )

    def _upload_inputs(self, world: "WorldEngine", resources: GPUHeatResources, solve_tile_mask: np.ndarray) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "heat input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "gas_concentration",
            "active_tile_ttl",
        )
        upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
        upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
        upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
        upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
        upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
        upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
        upload_active_from_cpu = not (formal_gpu_frame and "active_tile_ttl" in authoritative)
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
        self.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
        self.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
        self.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
        self.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_cell_state_from_cpu:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
            resources.phase_tex.write(world.phase.astype("f4").tobytes())
            resources.cell_flags_tex.write(world.cell_flags.astype("f4").tobytes())
            resources.timer_tex.write(world.timer_pack.astype("f4").tobytes())
            resources.integrity_tex.write(world.integrity.astype("f4").tobytes())
            resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
            resources.velocity_out_tex.write(world.velocity.astype("f4").tobytes())
            resources.temp_ping.write(world.cell_temperature.astype("f4").tobytes())
            resources.temp_pong.write(world.cell_temperature.astype("f4").tobytes())
        if upload_island_id_from_cpu:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        if upload_entity_id_from_cpu:
            resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
        if upload_displaced_from_cpu:
            resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
        if upload_gas_from_cpu:
            resources.gas_tex.write(world.gas_concentration.astype("f4").tobytes())
            resources.gas_out_tex.write(world.gas_concentration.astype("f4").tobytes())
        if upload_ambient_from_cpu:
            resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
            resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
        if upload_active_from_cpu:
            resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=1)
        material_table = world.bridge.shadow_typed_tables["material_table"]
        material_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        if resources.material_params_signature != material_signature:
            params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
            response_params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
            phase_params = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
            count = min(MAX_MATERIALS, material_table.shape[0])
            params[:count, 0] = material_table[:count]["conductivity"]
            params[:count, 1] = material_table[:count]["ambient_exchange_rate"]
            params[:count, 2] = material_table[:count]["melt_point"]
            params[:count, 3] = material_table[:count]["boil_point"]
            response_params[:count, 0] = material_table[:count]["heat_capacity"]
            response_params[:count, 1] = material_table[:count]["base_integrity"]
            response_params[:count, 2] = material_table[:count]["spawn_temperature"]
            response_params[:count, 3] = material_table[:count]["render_group_id"].astype("f4")
            phase_params[:count, 0] = material_table[:count]["default_phase"]
            phase_params[:count, 1] = material_table[:count]["melt_to_material_id"]
            phase_params[:count, 2] = material_table[:count]["freeze_to_material_id"]
            boil_species = material_table[:count]["boil_to_gas_species_id"].astype(np.int32)
            phase_params[:count, 3] = np.where(boil_species >= 0, boil_species + 1, 0)
            resources.material_params.write(params.tobytes())
            resources.material_response_params.write(response_params.tobytes())
            resources.material_phase_params.write(phase_params.tobytes())
            resources.material_params_signature = material_signature
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        gas_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
        if resources.gas_params_signature != gas_signature:
            gas_params = np.zeros((MAX_GAS_SPECIES, 4), dtype="f4")
            count = min(MAX_GAS_SPECIES, gas_table.shape[0])
            gas_params[:count, 0] = gas_table[:count]["condense_point"]
            gas_params[:count, 1] = gas_table[:count]["condense_to_material_id"].astype("f4")
            resources.gas_params.write(gas_params.tobytes())
            resources.gas_params_signature = gas_signature

    def _load_authoritative_active_tile_mask(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge active scheduler resources")
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

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        copy_gas = "gas_concentration" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient or copy_gas):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative input state")

        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(4, read=False, write=True)
            resources.phase_tex.bind_to_image(5, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(6, read=False, write=True)
            resources.timer_tex.bind_to_image(7, read=False, write=True)
            resources.temp_ping.bind_to_image(0, read=False, write=True)
            resources.integrity_tex.bind_to_image(1, read=False, write=True)
            resources.velocity_tex.bind_to_image(2, read=False, write=True)
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

        if copy_ambient or copy_gas:
            program = self.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["species_count"].value = int(world.gas_concentration.shape[0])
            program["copy_ambient"].value = bool(copy_ambient)
            program["copy_gas"].value = bool(copy_gas)
            bridge.textures["ambient_temperature"].use(location=0)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
            resources.ambient_ping.bind_to_image(2, read=False, write=True)
            resources.gas_tex.bind_to_image(3, read=False, write=True)
            program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

        self._sync_compute_writes(bridge.ctx)

    def _run_cell_heat(self, world: "WorldEngine", dt: float, resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["cell_heat"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_ping.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.temp_pong.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_ambient_exchange(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["ambient_exchange"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_pong.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.temp_ping.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_ambient_diffuse(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
        *,
        iterations: int,
    ) -> None:
        if iterations <= 0:
            return
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["ambient_diffuse"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        for _ in range(iterations):
            resources.ambient_ping.use(location=3)
            resources.ambient_pong.bind_to_image(4, read=False, write=True)
            program.run(gas_group_x, gas_group_y, 1)
            self._sync_compute_writes(ctx)
            resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping

    def _run_ambient_feedback(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["ambient_feedback"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["tile_size"].value = world.active.tile_size
        program["dt"].value = dt
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_pong.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.ambient_pong.bind_to_image(5, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_phase_targets(self, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["phase_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        self._set_uniform_if_present(program, "freeze_cold_neighbor_threshold", FREEZE_COLD_NEIGHBOR_THRESHOLD)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.phase_tex.use(location=4)
        resources.temp_ping.use(location=5)
        resources.phase_target_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_boil_targets(self, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["boil_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.temp_ping.use(location=4)
        resources.boil_target_tex.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_condense_targets(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["condense_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=3)
        resources.gas_tex.use(location=4)
        resources.ambient_pong.use(location=5)
        resources.condense_target_tex.bind_to_image(6, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_cell_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        with self._profile_pass(world, "apply_cell_targets.main"):
            program = self.programs["apply_cell_targets"]
            ctx = world.bridge.ctx
            assert ctx is not None
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
            self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
            self._set_uniform_if_present(program, "dt", dt)
            self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
            self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
            resources.material_params.bind_to_storage_buffer(binding=0)
            resources.material_tex.use(location=1)
            resources.active_tile_tex.use(location=2)
            resources.material_phase_params.bind_to_storage_buffer(binding=3)
            resources.material_response_params.bind_to_storage_buffer(binding=7)
            resources.phase_target_tex.use(location=3)
            resources.phase_tex.use(location=4)
            resources.cell_flags_tex.use(location=5)
            resources.timer_tex.use(location=6)
            resources.boil_target_tex.use(location=7)
            resources.temp_ping.use(location=8)
            resources.integrity_tex.use(location=9)
            resources.island_id_tex.use(location=10)
            resources.entity_id_tex.use(location=11)
            resources.displaced_tex.use(location=12)
            resources.ambient_pong.use(location=22)
            resources.velocity_tex.use(location=23)
            resources.material_out_tex.bind_to_image(0, read=False, write=True)
            resources.phase_out_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
            resources.timer_out_tex.bind_to_image(3, read=False, write=True)
            resources.temp_pong.bind_to_image(4, read=False, write=True)
            resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
            resources.island_id_out_tex.bind_to_image(6, read=False, write=True)
            resources.entity_id_out_tex.bind_to_image(7, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "apply_cell_targets.aux"):
            self._run_apply_cell_aux_targets(world, dt, resources, group_x, group_y)

    def _run_apply_cell_aux_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["apply_cell_aux_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.phase_target_tex.use(location=3)
        resources.phase_tex.use(location=4)
        resources.boil_target_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.displaced_tex.use(location=7)
        resources.velocity_tex.use(location=8)
        resources.displaced_out_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_gas_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["apply_gas_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=3)
        resources.gas_tex.use(location=4)
        resources.boil_target_tex.use(location=5)
        resources.condense_target_tex.use(location=6)
        resources.material_out_tex.use(location=8)
        resources.gas_out_tex.bind_to_image(0, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_condense_cells(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        with self._profile_pass(world, "apply_condense_cells.main"):
            program = self.programs["apply_condense_cells"]
            ctx = world.bridge.ctx
            assert ctx is not None
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
            self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
            self._set_uniform_if_present(
                program,
                "gas_species_count",
                min(world.gas_concentration.shape[0], MAX_GAS_SPECIES),
            )
            self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
            self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
            resources.active_tile_tex.use(location=2)
            resources.material_phase_params.bind_to_storage_buffer(binding=3)
            resources.material_response_params.bind_to_storage_buffer(binding=7)
            resources.gas_params.bind_to_storage_buffer(binding=8)
            resources.material_out_tex.use(location=4)
            resources.phase_out_tex.use(location=5)
            resources.cell_flags_out_tex.use(location=6)
            resources.timer_out_tex.use(location=9)
            resources.temp_pong.use(location=10)
            resources.integrity_out_tex.use(location=11)
            resources.island_id_out_tex.use(location=12)
            resources.entity_id_out_tex.use(location=13)
            resources.displaced_out_tex.use(location=22)
            resources.velocity_out_tex.use(location=23)
            resources.condense_target_tex.use(location=24)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.timer_tex.bind_to_image(3, read=False, write=True)
            resources.temp_ping.bind_to_image(4, read=False, write=True)
            resources.integrity_tex.bind_to_image(5, read=False, write=True)
            resources.displaced_tex.bind_to_image(6, read=False, write=True)
            resources.velocity_tex.bind_to_image(7, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
            resources.island_id_tex, resources.island_id_out_tex = resources.island_id_out_tex, resources.island_id_tex
            resources.entity_id_tex, resources.entity_id_out_tex = resources.entity_id_out_tex, resources.entity_id_tex

    def _run_apply_condense_cell_aux(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["apply_condense_cell_aux"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        resources.active_tile_tex.use(location=2)
        resources.gas_params.bind_to_storage_buffer(binding=8)
        resources.material_out_tex.use(location=4)
        resources.phase_out_tex.use(location=5)
        resources.island_id_out_tex.use(location=12)
        resources.displaced_out_tex.use(location=22)
        resources.velocity_out_tex.use(location=23)
        resources.condense_target_tex.use(location=24)
        resources.displaced_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_tex.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _download_outputs(self, world: "WorldEngine", resources: GPUHeatResources) -> GPUHeatStageTargets:
        world.material_id[:] = np.rint(
            np.frombuffer(resources.material_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(resources.phase_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_flags[:] = np.rint(
            np.frombuffer(resources.cell_flags_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(resources.temp_ping.read(), dtype="f4").reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(resources.integrity_tex.read(), dtype="f4").reshape((world.height, world.width))
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_id_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.entity_id[:] = np.rint(
            np.frombuffer(resources.entity_id_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.placeholder_displaced_material[:] = np.rint(
            np.frombuffer(resources.displaced_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.velocity[:] = np.frombuffer(resources.velocity_tex.read(), dtype="f4").reshape((world.height, world.width, 2))
        world.ambient_temperature[:] = np.frombuffer(resources.ambient_pong.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        world.gas_concentration[:] = np.frombuffer(resources.gas_out_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
        return GPUHeatStageTargets(
            phase_targets=np.rint(
                np.frombuffer(resources.phase_target_tex.read(), dtype="f4").reshape((world.height, world.width))
            ).astype(np.int32),
            boil_targets=np.rint(
                np.frombuffer(resources.boil_target_tex.read(), dtype="f4").reshape((world.height, world.width))
            ).astype(np.int32),
            condense_targets=(
                np.frombuffer(resources.condense_target_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
                > 0.5
            ),
        )

    def _empty_stage_targets(self, world: "WorldEngine") -> GPUHeatStageTargets:
        if self._formal_gpu_frame(world):
            return GPUHeatStageTargets.empty_sentinel()
        return GPUHeatStageTargets(
            phase_targets=np.zeros((world.height, world.width), dtype=np.int32),
            boil_targets=np.zeros((world.height, world.width), dtype=np.int32),
            condense_targets=np.zeros(world.gas_concentration.shape, dtype=np.bool_),
        )

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative heat state")
        fuse_structure_dirty_mark = False
        dirty_buffer = None
        dirty_count = None
        dirty_list = None
        dirty_dispatch_args = None
        material_flags_buffer = None
        material_count = 0
        if self._formal_gpu_frame(world) and _active_scheduler_gpu_authoritative(world):
            dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
            dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
            if dirty_buffer is not None and dirty_queue is not None:
                dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
                material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
                fuse_structure_dirty_mark = True
        with self._profile_pass(world, "publish_bridge_outputs.collapse_dirty_mark"):
            if not fuse_structure_dirty_mark:
                mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
                    world,
                    resources.material_tex,
                    resources.phase_tex,
                )
        with self._profile_pass(world, "publish_bridge_outputs.cell"):
            cell_program = self.programs["publish_bridge_cell"]
            cell_program["cell_grid_size"].value = (world.width, world.height)
            cell_program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
            cell_program["tile_size"].value = int(world.active.tile_size)
            cell_program["material_count"].value = int(material_count)
            cell_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            cell_program["mark_structure_dirty"].value = bool(fuse_structure_dirty_mark)
            cell_program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
            cell_program["material_tex"].value = 0
            cell_program["phase_tex"].value = 1
            cell_program["flags_tex"].value = 2
            cell_program["timer_tex"].value = 3
            cell_program["temp_tex"].value = 4
            cell_program["integrity_tex"].value = 5
            cell_program["island_tex"].value = 6
            cell_program["entity_tex"].value = 7
            cell_program["displaced_tex"].value = 8
            cell_program["velocity_tex"].value = 9
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.cell_flags_tex.use(location=2)
            resources.timer_tex.use(location=3)
            resources.temp_ping.use(location=4)
            resources.integrity_tex.use(location=5)
            resources.island_id_tex.use(location=6)
            resources.entity_id_tex.use(location=7)
            resources.displaced_tex.use(location=8)
            resources.velocity_tex.use(location=9)
            bridge.textures["material"].bind_to_image(0, read=False, write=True)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            if fuse_structure_dirty_mark:
                assert dirty_buffer is not None
                assert dirty_count is not None
                assert dirty_list is not None
                assert dirty_dispatch_args is not None
                assert material_flags_buffer is not None
                material_flags_buffer.bind_to_storage_buffer(binding=4)
                dirty_buffer.bind_to_storage_buffer(binding=5)
                dirty_count.bind_to_storage_buffer(binding=6)
                dirty_list.bind_to_storage_buffer(binding=7)
                dirty_dispatch_args.bind_to_storage_buffer(binding=8)
        if not bool(getattr(world, "phase_c_defer_cell_publish", False)):
            cell_program.run(group_x, group_y, 1)

        with self._profile_pass(world, "publish_bridge_outputs.gas"):
            gas_program = self.programs["publish_bridge_gas"]
            gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            gas_program["species_count"].value = int(world.gas_concentration.shape[0])
            gas_program["ambient_tex"].value = 0
            gas_program["gas_tex"].value = 2
            resources.ambient_pong.use(location=0)
            resources.gas_out_tex.use(location=2)
            bridge.textures["ambient_temperature"].bind_to_image(1, read=False, write=True)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=4)
            gas_program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

        with self._profile_pass(world, "publish_bridge_outputs.sync"):
            self._sync_compute_writes(bridge.ctx)
            if fuse_structure_dirty_mark:
                setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
                bridge.mark_gpu_authoritative(
                    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
                )
            bridge.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
                "ambient_temperature",
                "gas_concentration",
            )

    # ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
    # from :class:`GPUPipelineBase`; the heat pass uses the default barrier
    # bits (image-access | texture-fetch | shader-storage).
