from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
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
    liquid_flow_intent: Any
    active_tile_tex: Any
    active_tile_list: Any
    active_tile_count: Any
    active_tile_dispatch_args: Any
    affected_tile_list: Any
    affected_tile_count: Any
    affected_tile_dispatch_args: Any
    affected_tile_prefetch_dispatch_args: Any
    affected_tile_flags: Any
    placeholder_target_claims: Any
    displaced_in: Any
    displaced_out: Any
    bridge_cell_copy_framebuffer: Any
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
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self._placeholder_claim_epoch = 1

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

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
            if profile is None:
                return
            if ctx is not None:
                ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            entry = {
                "name": str(name),
                "cpu_ms": elapsed_ms,
                "gpu_ms": elapsed_ms if ctx is not None else None,
            }
            profile["passes"].append(entry)
            summary = profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
            summary["count"] += 1
            summary["cpu_ms"] += elapsed_ms
            if ctx is not None:
                summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

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
        self.reset_pass_profile()
        with self._profile_pass(world, "liquid_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
        with self._profile_pass(world, "liquid_load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources)
        with self._profile_pass(world, "liquid_compact_active_tiles"):
            self._compact_active_tiles(world, resources)
        with self._profile_pass(world, "liquid_tile_solve"):
            self._run_tile_solve(world, resources)
        if self._active_scheduler_gpu_authoritative(world):
            with self._profile_pass(world, "liquid_load_active_mask"):
                self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        else:
            with self._profile_pass(world, "liquid_upload_active_mask"):
                self._upload_active_tile_mask(resources, post_tile_mask)
        with self._profile_pass(world, "liquid_compact_active_cell_tiles"):
            self._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
            )
        formal_gpu_frame = self._formal_gpu_frame(world)
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_copy_tile_solve"):
                self._run_copy_core_state(
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
            with self._profile_pass(world, "liquid_build_seam_x_boundaries"):
                self._build_seam_boundary_dispatch(world, resources, axis="x")
            with self._profile_pass(world, "liquid_prefetch_seam_x_boundaries"):
                self._prefetch_seam_boundary_bridge_inputs(world, resources, axis="x")
        with self._profile_pass(world, "liquid_seam_x"):
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
                boundary_dispatch=formal_gpu_frame,
            )
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_reload_seam_x_active_tiles"):
                self._reload_and_compact_active_cell_tiles(world, resources)
            with self._profile_pass(world, "liquid_copy_seam_x"):
                self._run_copy_core_state(
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
            with self._profile_pass(world, "liquid_build_seam_y_boundaries"):
                self._build_seam_boundary_dispatch(world, resources, axis="y")
            with self._profile_pass(world, "liquid_prefetch_seam_y_boundaries"):
                self._prefetch_seam_boundary_bridge_inputs(world, resources, axis="y")
        with self._profile_pass(world, "liquid_seam_y"):
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
                boundary_dispatch=formal_gpu_frame,
            )
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_reload_seam_y_active_tiles"):
                self._reload_and_compact_active_cell_tiles(world, resources)
        with self._profile_pass(world, "liquid_buoyancy_sink"):
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
        with self._profile_pass(world, "liquid_buoyancy_float"):
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
        with self._profile_pass(world, "liquid_copy_for_placeholder"):
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
        with self._profile_pass(world, "liquid_placeholder_displacement"):
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
        with self._profile_pass(world, "liquid_cleanup_runtime"):
            self._run_cleanup_runtime(world, resources)
        if self._active_scheduler_gpu_authoritative(world):
            with self._profile_pass(world, "liquid_reload_flow_active_mask"):
                self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
            with self._profile_pass(world, "liquid_compact_flow_active_tiles"):
                self._compact_active_tiles(
                    world,
                    resources,
                    workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
                )
        with self._profile_pass(world, "liquid_flow_intent"):
            self._run_liquid_intent_pass(world, resources)
        if self._formal_gpu_frame(world):
            with self._profile_pass(world, "liquid_refresh_active_scheduler"):
                self._refresh_active_scheduler_from_ttl(world)
        with self._profile_pass(world, "liquid_publish_bridge"):
            self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources, use_in=True)

    def prepare_motion_flow_intent(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.reset_pass_profile()
        with self._profile_pass(world, "liquid_pre_motion_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
        with self._profile_pass(world, "liquid_pre_motion_load_bridge_inputs"):
            self._load_authoritative_bridge_flow_intent_inputs(world, resources)
        with self._profile_pass(world, "liquid_pre_motion_compact_active_tiles"):
            self._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
            )
        with self._profile_pass(world, "liquid_pre_motion_flow_intent"):
            self._run_liquid_intent_pass(world, resources)
        self.last_cpu_mirror_downloaded = False

    def release(self) -> None:
        if self.resources is None:
            return
        try:
            self.resources.bridge_cell_copy_framebuffer.release()
        except Exception:
            pass
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
            self.resources.liquid_flow_intent,
            self.resources.active_tile_tex,
            self.resources.active_tile_list,
            self.resources.active_tile_count,
            self.resources.active_tile_dispatch_args,
            self.resources.affected_tile_list,
            self.resources.affected_tile_count,
            self.resources.affected_tile_dispatch_args,
            self.resources.affected_tile_prefetch_dispatch_args,
            self.resources.affected_tile_flags,
            self.resources.placeholder_target_claims,
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
        liquid_flow_intent = ctx.texture((world.width, world.height), 2, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        tile_count = max(1, int(world.active.tile_width * world.active.tile_height))
        active_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        active_tile_count = ctx.buffer(reserve=4, dynamic=True)
        active_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        affected_tile_count = ctx.buffer(reserve=4, dynamic=True)
        affected_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_prefetch_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_flags = ctx.buffer(reserve=max(4, tile_count * 4), dynamic=True)
        affected_tile_flags.write(np.zeros((tile_count,), dtype=np.uint32).tobytes())
        cell_count = max(1, int(world.width * world.height))
        placeholder_target_claims = ctx.buffer(reserve=cell_count * 4, dynamic=True)
        placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
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
            liquid_flow_intent,
            active_tile_tex,
            displaced_in,
            displaced_out,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        bridge_cell_copy_framebuffer = ctx.framebuffer(
            color_attachments=[
                material_pre,
                material_out,
                phase_pre,
                phase_out,
                flags_out,
                timer_out,
                temp_out,
                integrity_out,
            ]
        )
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
            liquid_flow_intent=liquid_flow_intent,
            active_tile_tex=active_tile_tex,
            active_tile_list=active_tile_list,
            active_tile_count=active_tile_count,
            active_tile_dispatch_args=active_tile_dispatch_args,
            affected_tile_list=affected_tile_list,
            affected_tile_count=affected_tile_count,
            affected_tile_dispatch_args=affected_tile_dispatch_args,
            affected_tile_prefetch_dispatch_args=affected_tile_prefetch_dispatch_args,
            affected_tile_flags=affected_tile_flags,
            placeholder_target_claims=placeholder_target_claims,
            displaced_in=displaced_in,
            displaced_out=displaced_out,
            bridge_cell_copy_framebuffer=bridge_cell_copy_framebuffer,
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
        self.programs["clear_active_tile_dispatch"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            layout(std430, binding=0) buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=1) buffer ActiveTileDispatchArgsBuffer {
                uint active_tile_dispatch_args[];
            };
            void main() {
                active_tile_count[0] = 0u;
                active_tile_dispatch_args[0] = 0u;
                active_tile_dispatch_args[1] = 1u;
                active_tile_dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["compact_active_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform uint workgroups_per_tile;
            layout(binding=0) uniform sampler2D active_tile_tex;
            layout(std430, binding=0) buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=1) buffer ActiveTileListBuffer {
                ivec2 active_tile_list[];
            };
            layout(std430, binding=2) buffer ActiveTileDispatchArgsBuffer {
                uint active_tile_dispatch_args[];
            };
            void main() {
                int tile_count = tile_grid_size.x * tile_grid_size.y;
                int index = int(gl_GlobalInvocationID.x);
                if (index >= tile_count) {
                    return;
                }
                ivec2 tile = ivec2(index % tile_grid_size.x, index / tile_grid_size.x);
                if (texelFetch(active_tile_tex, tile, 0).x <= 0.5) {
                    return;
                }
                uint slot = atomicAdd(active_tile_count[0], 1u);
                active_tile_list[slot] = tile;
                atomicMax(active_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
            }
            """
        )
        self.programs["compact_active_tiles_from_chunks"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform int chunk_tiles;
            uniform uint workgroups_per_tile;
            layout(std430, binding=0) buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=1) buffer ActiveTileListBuffer {
                ivec2 active_tile_list[];
            };
            layout(std430, binding=2) buffer ActiveTileDispatchArgsBuffer {
                uint active_tile_dispatch_args[];
            };
            layout(std430, binding=3) readonly buffer ActiveChunkCountBuffer {
                uint active_chunk_count[];
            };
            layout(std430, binding=4) readonly buffer ActiveChunkListBuffer {
                ivec2 active_chunk_list[];
            };
            layout(std430, binding=5) readonly buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };
            void main() {
                uint chunk_index = gl_WorkGroupID.x;
                if (chunk_index >= active_chunk_count[0]) {
                    return;
                }
                ivec2 chunk = active_chunk_list[int(chunk_index)];
                int chunk_side = max(chunk_tiles, 1);
                int tile_slots = chunk_side * chunk_side;
                for (int tile_slot = int(gl_LocalInvocationIndex); tile_slot < tile_slots; tile_slot += 256) {
                    ivec2 local_tile = ivec2(tile_slot % chunk_side, tile_slot / chunk_side);
                    ivec2 tile = chunk * chunk_side + local_tile;
                    if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                        continue;
                    }
                    int tile_index = tile.y * tile_grid_size.x + tile.x;
                    if (active_tile_ttl[tile_index] <= 0) {
                        continue;
                    }
                    uint slot = atomicAdd(active_tile_count[0], 1u);
                    active_tile_list[slot] = tile;
                    atomicMax(active_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
                }
            }
            """
        )
        self.programs["compact_placeholder_dirty_affected_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int dirty_rect_count;
            uniform uint workgroups_per_tile;

            struct PlaceholderDirtyRect {
                int buffer_x0;
                int buffer_y0;
                int buffer_x1;
                int buffer_y1;
                int world_x0;
                int world_y0;
                int width;
                int height;
            };

            layout(std430, binding=0) buffer AffectedTileCountBuffer {
                uint affected_tile_count[];
            };
            layout(std430, binding=1) buffer AffectedTileListBuffer {
                ivec2 affected_tile_list[];
            };
            layout(std430, binding=2) buffer AffectedTileDispatchArgsBuffer {
                uint affected_tile_dispatch_args[];
            };
            layout(std430, binding=3) buffer AffectedTileFlagsBuffer {
                uint affected_tile_flags[];
            };
            layout(std430, binding=4) readonly buffer PlaceholderDirtyRectBuffer {
                PlaceholderDirtyRect dirty_rects[];
            };

            void emit_tile(ivec2 tile) {
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return;
                }
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(affected_tile_flags[tile_index], 0u, 1u) != 0u) {
                    return;
                }
                uint slot = atomicAdd(affected_tile_count[0], 1u);
                affected_tile_list[int(slot)] = tile;
                atomicMax(affected_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
            }

            void main() {
                uint rect_index = gl_GlobalInvocationID.x;
                if (rect_index >= uint(max(dirty_rect_count, 0))) {
                    return;
                }
                PlaceholderDirtyRect rect = dirty_rects[int(rect_index)];
                if (rect.buffer_x1 <= rect.buffer_x0 || rect.buffer_y1 <= rect.buffer_y0) {
                    return;
                }
                int min_x = max(0, min(rect.buffer_x0, rect.buffer_x1 - 1));
                int min_y = max(0, min(rect.buffer_y0, rect.buffer_y1 - 1));
                int max_x = max(0, max(rect.buffer_x0, rect.buffer_x1 - 1));
                int max_y = max(0, max(rect.buffer_y0, rect.buffer_y1 - 1));
                int tile_min_x = clamp(min_x / max(tile_size, 1) - 1, 0, tile_grid_size.x - 1);
                int tile_min_y = clamp(min_y / max(tile_size, 1) - 1, 0, tile_grid_size.y - 1);
                int tile_max_x = clamp(max_x / max(tile_size, 1) + 1, 0, tile_grid_size.x - 1);
                int tile_max_y = clamp(max_y / max(tile_size, 1) + 1, 0, tile_grid_size.y - 1);
                for (int tile_y = tile_min_y; tile_y <= tile_max_y; ++tile_y) {
                    for (int tile_x = tile_min_x; tile_x <= tile_max_x; ++tile_x) {
                        emit_tile(ivec2(tile_x, tile_y));
                    }
                }
            }
            """
        )
        self.programs["compact_placeholder_active_pending_affected_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=64, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int placeholder_material_id;
            uniform uint source_workgroups_per_tile;
            uniform uint workgroups_per_tile;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D displaced_tex;

            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=2) buffer AffectedTileCountBuffer {{
                uint affected_tile_count[];
            }};
            layout(std430, binding=3) buffer AffectedTileListBuffer {{
                ivec2 affected_tile_list[];
            }};
            layout(std430, binding=4) buffer AffectedTileDispatchArgsBuffer {{
                uint affected_tile_dispatch_args[];
            }};
            layout(std430, binding=5) buffer AffectedTileFlagsBuffer {{
                uint affected_tile_flags[];
            }};

            void emit_tile(ivec2 tile) {{
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                if (atomicCompSwap(affected_tile_flags[tile_index], 0u, 1u) != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(affected_tile_count[0], 1u);
                affected_tile_list[int(slot)] = tile;
                atomicMax(affected_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
            }}

            void emit_pending_tile_region(ivec2 tile) {{
                for (int y = tile.y - 1; y <= tile.y + 1; ++y) {{
                    for (int x = tile.x - 1; x <= tile.x + 1; ++x) {{
                        emit_tile(ivec2(x, y));
                    }}
                }}
            }}

            void main() {{
                uint groups_per_tile = max(source_workgroups_per_tile, 1u);
                uint group_index = gl_WorkGroupID.x;
                uint active_index = group_index / groups_per_tile;
                if (active_index >= active_tile_count[0]) {{
                    return;
                }}
                uint subgroup = group_index % groups_per_tile;
                ivec2 tile = active_tile_list[int(active_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int safe_tile_size = max(tile_size, 1);
                int slot_count = safe_tile_size * safe_tile_size;
                uint stride = groups_per_tile * 64u;
                for (
                    uint slot = subgroup * 64u + uint(gl_LocalInvocationIndex);
                    slot < uint(slot_count);
                    slot += stride
                ) {{
                    int local_slot = int(slot);
                    ivec2 local_cell = ivec2(local_slot % safe_tile_size, local_slot / safe_tile_size);
                    ivec2 cell = tile * safe_tile_size + local_cell;
                    if (cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                        continue;
                    }}
                    int material = int(texelFetch(material_tex, cell, 0).x + 0.5);
                    float displaced = texelFetch(displaced_tex, cell, 0).x;
                    if (material == placeholder_material_id && displaced > 0.5) {{
                        emit_pending_tile_region(tile);
                        return;
                    }}
                }}
            }}
            """
        )
        self.programs["compact_seam_x_boundaries_from_active_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform uint source_workgroups_per_tile;
            uniform uint workgroups_per_boundary;
            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {
                ivec2 active_tile_list[];
            };
            layout(std430, binding=2) buffer AffectedTileCountBuffer {
                uint affected_tile_count[];
            };
            layout(std430, binding=3) buffer AffectedTileListBuffer {
                ivec2 affected_tile_list[];
            };
            layout(std430, binding=4) buffer AffectedTileDispatchArgsBuffer {
                uint affected_tile_dispatch_args[];
            };
            layout(std430, binding=5) readonly buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };

            bool tile_active(ivec2 tile) {
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return false;
                }
                return active_tile_ttl[tile.y * tile_grid_size.x + tile.x] > 0;
            }

            void emit_boundary(ivec2 boundary_tile) {
                if (
                    boundary_tile.x < 0
                    || boundary_tile.y < 0
                    || boundary_tile.x + 1 >= tile_grid_size.x
                    || boundary_tile.y >= tile_grid_size.y
                ) {
                    return;
                }
                uint slot = atomicAdd(affected_tile_count[0], 1u);
                affected_tile_list[int(slot)] = boundary_tile;
                atomicMax(affected_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_boundary, 1u));
            }

            void main() {
                uint groups_per_tile = max(source_workgroups_per_tile, 1u);
                uint group_index = gl_WorkGroupID.x;
                if ((group_index % groups_per_tile) != 0u) {
                    return;
                }
                uint active_index = group_index / groups_per_tile;
                if (active_index >= active_tile_count[0]) {
                    return;
                }
                ivec2 tile = active_tile_list[int(active_index)];
                if (!tile_active(tile)) {
                    return;
                }
                emit_boundary(tile);
                if (tile.x > 0 && !tile_active(tile + ivec2(-1, 0))) {
                    emit_boundary(tile + ivec2(-1, 0));
                }
            }
            """
        )
        self.programs["compact_seam_y_boundaries_from_active_tiles"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform uint source_workgroups_per_tile;
            uniform uint workgroups_per_boundary;
            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {
                ivec2 active_tile_list[];
            };
            layout(std430, binding=2) buffer AffectedTileCountBuffer {
                uint affected_tile_count[];
            };
            layout(std430, binding=3) buffer AffectedTileListBuffer {
                ivec2 affected_tile_list[];
            };
            layout(std430, binding=4) buffer AffectedTileDispatchArgsBuffer {
                uint affected_tile_dispatch_args[];
            };
            layout(std430, binding=5) readonly buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };

            bool tile_active(ivec2 tile) {
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {
                    return false;
                }
                return active_tile_ttl[tile.y * tile_grid_size.x + tile.x] > 0;
            }

            void emit_boundary(ivec2 boundary_tile) {
                if (
                    boundary_tile.x < 0
                    || boundary_tile.y < 0
                    || boundary_tile.x >= tile_grid_size.x
                    || boundary_tile.y + 1 >= tile_grid_size.y
                ) {
                    return;
                }
                uint slot = atomicAdd(affected_tile_count[0], 1u);
                affected_tile_list[int(slot)] = boundary_tile;
                atomicMax(affected_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_boundary, 1u));
            }

            void main() {
                uint groups_per_tile = max(source_workgroups_per_tile, 1u);
                uint group_index = gl_WorkGroupID.x;
                if ((group_index % groups_per_tile) != 0u) {
                    return;
                }
                uint active_index = group_index / groups_per_tile;
                if (active_index >= active_tile_count[0]) {
                    return;
                }
                ivec2 tile = active_tile_list[int(active_index)];
                if (!tile_active(tile)) {
                    return;
                }
                emit_boundary(tile);
                if (tile.y > 0 && !tile_active(tile + ivec2(0, -1))) {
                    emit_boundary(tile + ivec2(0, -1));
                }
            }
            """
        )
        self.programs["prefetch_seam_boundary_bridge_inputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int seam_axis;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=3) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer AffectedTileCountBuffer {{
                uint affected_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer AffectedTileListBuffer {{
                ivec2 affected_tile_list[];
            }};

            layout(r32f, binding=0) writeonly uniform image2D material_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_img;

            vec4 unpack_timer(uint word) {{
                return vec4(
                    float(word & 0xFFu),
                    float((word >> 8u) & 0xFFu),
                    float((word >> 16u) & 0xFFu),
                    float((word >> 24u) & 0xFFu)
                );
            }}

            bool boundary_dispatch_cell(out ivec2 cell) {{
                int groups_x;
                int groups_y;
                if (seam_axis == 0) {{
                    groups_x = max(1, (tile_size * 2 + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                    groups_y = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                }} else {{
                    groups_x = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                    groups_y = 1;
                }}
                int workgroups_per_boundary = max(1, groups_x * groups_y);
                uint group_index = gl_WorkGroupID.x;
                uint boundary_index = group_index / uint(workgroups_per_boundary);
                if (boundary_index >= affected_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_boundary));
                ivec2 subtile_xy = ivec2(subtile % groups_x, subtile / groups_x);
                ivec2 boundary_tile = affected_tile_list[int(boundary_index)];
                if (boundary_tile.x < 0 || boundary_tile.y < 0 || boundary_tile.x >= tile_grid_size.x || boundary_tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 origin;
                ivec2 region_size;
                if (seam_axis == 0) {{
                    if (boundary_tile.x + 1 >= tile_grid_size.x) {{
                        return false;
                    }}
                    origin = boundary_tile * tile_size;
                    region_size = ivec2(tile_size * 2, tile_size);
                }} else {{
                    if (boundary_tile.y + 1 >= tile_grid_size.y) {{
                        return false;
                    }}
                    origin = ivec2(boundary_tile.x * tile_size, (boundary_tile.y + 1) * tile_size - 1);
                    region_size = ivec2(tile_size, 2);
                }}
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                if (seam_axis != 0 && local_cell.y >= 2) {{
                    return false;
                }}
                cell = origin + local_cell;
                ivec2 region_end = min(origin + region_size, cell_grid_size);
                return cell.x >= 0 && cell.y >= 0 && cell.x < region_end.x && cell.y < region_end.y;
            }}

            bool tile_is_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    clamp(cell.x / max(tile_size, 1), 0, tile_grid_size.x - 1),
                    clamp(cell.y / max(tile_size, 1), 0, tile_grid_size.y - 1)
                );
                return active_tile_ttl[tile.y * tile_grid_size.x + tile.x] > 0;
            }}

            void main() {{
                ivec2 gid;
                if (!boundary_dispatch_cell(gid) || tile_is_active(gid)) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                int word_index = cell_index * 5;
                uint word0 = bridge_cell_core[word_index];
                float material = float(word0 & 0xFFFFu);
                float phase = float((word0 >> 16u) & 0xFFu);
                imageStore(material_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_img, gid, vec4(float((word0 >> 24u) & 0xFFu), 0.0, 0.0, 0.0));
                imageStore(timer_img, gid, unpack_timer(bridge_cell_core[word_index + 3]));
                imageStore(temp_img, gid, vec4(uintBitsToFloat(bridge_cell_core[word_index + 2]), 0.0, 0.0, 0.0));
                imageStore(integrity_img, gid, vec4(float(bridge_cell_core[word_index + 4] & 0xFFFFu), 0.0, 0.0, 0.0));
                imageStore(velocity_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
            }}
            """
        )
        self.programs["prefetch_seam_boundary_bridge_aux_inputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int seam_axis;

            layout(std430, binding=0) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=1) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced_material[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=3) readonly buffer AffectedTileCountBuffer {{
                uint affected_tile_count[];
            }};
            layout(std430, binding=4) readonly buffer AffectedTileListBuffer {{
                ivec2 affected_tile_list[];
            }};

            layout(r32f, binding=0) writeonly uniform image2D entity_img;
            layout(r32f, binding=1) writeonly uniform image2D displaced_img;

            bool boundary_dispatch_cell(out ivec2 cell) {{
                int groups_x;
                int groups_y;
                if (seam_axis == 0) {{
                    groups_x = max(1, (tile_size * 2 + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                    groups_y = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                }} else {{
                    groups_x = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                    groups_y = 1;
                }}
                int workgroups_per_boundary = max(1, groups_x * groups_y);
                uint group_index = gl_WorkGroupID.x;
                uint boundary_index = group_index / uint(workgroups_per_boundary);
                if (boundary_index >= affected_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_boundary));
                ivec2 subtile_xy = ivec2(subtile % groups_x, subtile / groups_x);
                ivec2 boundary_tile = affected_tile_list[int(boundary_index)];
                if (boundary_tile.x < 0 || boundary_tile.y < 0 || boundary_tile.x >= tile_grid_size.x || boundary_tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 origin;
                ivec2 region_size;
                if (seam_axis == 0) {{
                    if (boundary_tile.x + 1 >= tile_grid_size.x) {{
                        return false;
                    }}
                    origin = boundary_tile * tile_size;
                    region_size = ivec2(tile_size * 2, tile_size);
                }} else {{
                    if (boundary_tile.y + 1 >= tile_grid_size.y) {{
                        return false;
                    }}
                    origin = ivec2(boundary_tile.x * tile_size, (boundary_tile.y + 1) * tile_size - 1);
                    region_size = ivec2(tile_size, 2);
                }}
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                if (seam_axis != 0 && local_cell.y >= 2) {{
                    return false;
                }}
                cell = origin + local_cell;
                ivec2 region_end = min(origin + region_size, cell_grid_size);
                return cell.x >= 0 && cell.y >= 0 && cell.x < region_end.x && cell.y < region_end.y;
            }}

            bool tile_is_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    clamp(cell.x / max(tile_size, 1), 0, tile_grid_size.x - 1),
                    clamp(cell.y / max(tile_size, 1), 0, tile_grid_size.y - 1)
                );
                return active_tile_ttl[tile.y * tile_grid_size.x + tile.x] > 0;
            }}

            void main() {{
                ivec2 gid;
                if (!boundary_dispatch_cell(gid) || tile_is_active(gid)) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                imageStore(entity_img, gid, vec4(float(bridge_entity_id[cell_index]), 0.0, 0.0, 0.0));
                imageStore(displaced_img, gid, vec4(float(bridge_displaced_material[cell_index]), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["tile_solve"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={TILE_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            const int TILE_SIZE = {TILE_SIZE};
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int active_ttl_reset;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=8) uniform sampler2D entity_in_tex;
            layout(binding=9) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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
            shared int s_blocked[TILE_SIZE][TILE_SIZE];
            shared int s_changed[TILE_SIZE][TILE_SIZE];
            shared int s_source_x[TILE_SIZE][TILE_SIZE];
            shared int s_source_y[TILE_SIZE][TILE_SIZE];
            shared int s_down_target_x[TILE_SIZE];
            shared int s_down_source_x[TILE_SIZE];
            shared int s_down_segment_claimed[TILE_SIZE];
            shared int s_row_liquid_flag[TILE_SIZE];
            shared int s_below_empty_flag[TILE_SIZE];
            shared int s_row_any_liquid;
            shared int s_row_any_below_empty;
            shared int s_liquid_left_blocker[TILE_SIZE];
            shared int s_liquid_right_blocker[TILE_SIZE];
            shared int s_empty_left_blocker[TILE_SIZE];
            shared int s_empty_right_blocker[TILE_SIZE];
            shared int s_liquid_left_scan_tmp[TILE_SIZE];
            shared int s_liquid_right_scan_tmp[TILE_SIZE];
            shared int s_empty_left_scan_tmp[TILE_SIZE];
            shared int s_empty_right_scan_tmp[TILE_SIZE];
            shared int s_liquid_run_start[TILE_SIZE];
            shared int s_liquid_run_end[TILE_SIZE];
            shared int s_empty_segment_start[TILE_SIZE];
            shared int s_empty_segment_end[TILE_SIZE];
            shared int s_liquid_run_first_empty_x[TILE_SIZE];
            shared int s_lateral_target_x[TILE_SIZE];
            shared int s_lateral_source_x[TILE_SIZE];

            void clear_cell(int y, int x) {{
                s_material[y][x] = 0.0;
                s_phase[y][x] = 0.0;
                s_flags[y][x] = 0.0;
                s_blocked[y][x] = 0;
                s_changed[y][x] = 1;
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
                s_changed[y][x] = 1;
                s_source_x[y][x] = source_x;
                s_source_y[y][x] = source_y;
            }}

            int liquid_kind_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].z + 0.5);
            }}

            bool is_placeholder_material(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            bool is_reachable_empty(int y, int x) {{
                int phase = int(s_phase[y][x] + 0.5);
                return s_material[y][x] < 0.5
                    && phase != phase_liquid
                    && phase != phase_falling_island
                    && s_blocked[y][x] == 0;
            }}

            bool is_vertical_empty(int y, int x) {{
                int phase = int(s_phase[y][x] + 0.5);
                return s_material[y][x] < 0.5
                    && phase != phase_liquid
                    && phase != phase_falling_island
                    && s_blocked[y][x] == 0;
            }}

            bool local_cell_in_world(ivec2 tile, int y, int x) {{
                ivec2 cell = tile * TILE_SIZE + ivec2(x, y);
                return x >= 0
                    && y >= 0
                    && x < TILE_SIZE
                    && y < TILE_SIZE
                    && cell.x >= 0
                    && cell.y >= 0
                    && cell.x < cell_grid_size.x
                    && cell.y < cell_grid_size.y;
            }}

            bool is_liquid_cell(int y, int x) {{
                return s_material[y][x] > 0.5
                    && int(s_phase[y][x] + 0.5) == phase_liquid;
            }}

            bool is_columnar_liquid_cell(int y, int x) {{
                return is_liquid_cell(y, x) && liquid_kind_for(s_material[y][x]) == 1;
            }}

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / TILE_SIZE, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / TILE_SIZE, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            void main() {{
                ivec2 local = ivec2(gl_LocalInvocationID.x, 0);
                uint active_index = gl_WorkGroupID.x;
                if (active_index >= active_tile_count[0]) {{
                    return;
                }}
                ivec2 tile = active_tile_list[int(active_index)];
                if (tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                for (int y = 0; y < TILE_SIZE; ++y) {{
                    ivec2 cell = tile * TILE_SIZE + ivec2(local.x, y);
                    bool in_bounds = cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
                    float material = in_bounds ? texelFetch(material_in_tex, cell, 0).x : 0.0;
                    float phase = in_bounds ? texelFetch(phase_in_tex, cell, 0).x : 0.0;
                    float flagsv = in_bounds ? texelFetch(flags_in_tex, cell, 0).x : 0.0;
                    float entityv = in_bounds ? texelFetch(entity_in_tex, cell, 0).x : 0.0;
                    float displacedv = in_bounds ? texelFetch(displaced_in_tex, cell, 0).x : 0.0;
                    s_material[y][local.x] = material;
                    s_phase[y][local.x] = phase;
                    s_flags[y][local.x] = flagsv;
                    s_blocked[y][local.x] = (
                        int(phase + 0.5) == phase_falling_island
                        || entityv > 0.5
                        || displacedv > 0.5
                        || is_placeholder_material(material)
                    ) ? 1 : 0;
                    s_source_x[y][local.x] = in_bounds ? local.x : -1;
                    s_source_y[y][local.x] = in_bounds ? y : -1;
                    s_changed[y][local.x] = 0;
                }}
                barrier();

                for (int row = TILE_SIZE - 2; row >= 0; --row) {{
                    if (local.x == 0) {{
                        s_row_any_liquid = 0;
                        s_row_any_below_empty = 0;
                    }}
                    barrier();

                    bool row_liquid = local_cell_in_world(tile, row, local.x)
                        && row + 1 < TILE_SIZE
                        && tile.y * TILE_SIZE + row + 1 < cell_grid_size.y
                        && is_columnar_liquid_cell(row, local.x);
                    bool below_empty = row + 1 < TILE_SIZE
                        && local_cell_in_world(tile, row + 1, local.x)
                        && is_vertical_empty(row + 1, local.x);
                    s_row_liquid_flag[local.x] = row_liquid ? 1 : 0;
                    s_below_empty_flag[local.x] = below_empty ? 1 : 0;
                    if (row_liquid) {{
                        atomicOr(s_row_any_liquid, 1);
                    }}
                    if (below_empty) {{
                        atomicOr(s_row_any_below_empty, 1);
                    }}
                    memoryBarrierShared();
                    barrier();

                    if (s_row_any_liquid == 0) {{
                        continue;
                    }}

                    s_down_target_x[local.x] = -1;
                    s_down_source_x[local.x] = TILE_SIZE;
                    s_down_segment_claimed[local.x] = 0;
                    int source_x = -1;
                    int target_x = -1;

                    if (s_row_any_below_empty != 0) {{
                        s_liquid_left_blocker[local.x] = row_liquid ? -1 : local.x;
                        s_liquid_right_blocker[local.x] = row_liquid ? TILE_SIZE : local.x;
                        s_empty_left_blocker[local.x] = below_empty ? -1 : local.x;
                        s_empty_right_blocker[local.x] = below_empty ? TILE_SIZE : local.x;
                        s_liquid_run_start[local.x] = -1;
                        s_liquid_run_end[local.x] = -1;
                        s_empty_segment_start[local.x] = -1;
                        s_empty_segment_end[local.x] = -1;
                        s_liquid_run_first_empty_x[local.x] = TILE_SIZE;
                        barrier();

                        for (int stride = 1; stride < TILE_SIZE; stride *= 2) {{
                            int liquid_left = s_liquid_left_blocker[local.x];
                            int liquid_right = s_liquid_right_blocker[local.x];
                            int empty_left = s_empty_left_blocker[local.x];
                            int empty_right = s_empty_right_blocker[local.x];
                            if (local.x >= stride) {{
                                liquid_left = max(liquid_left, s_liquid_left_blocker[local.x - stride]);
                                empty_left = max(empty_left, s_empty_left_blocker[local.x - stride]);
                            }}
                            if (local.x + stride < TILE_SIZE) {{
                                liquid_right = min(liquid_right, s_liquid_right_blocker[local.x + stride]);
                                empty_right = min(empty_right, s_empty_right_blocker[local.x + stride]);
                            }}
                            s_liquid_left_scan_tmp[local.x] = liquid_left;
                            s_liquid_right_scan_tmp[local.x] = liquid_right;
                            s_empty_left_scan_tmp[local.x] = empty_left;
                            s_empty_right_scan_tmp[local.x] = empty_right;
                            barrier();
                            s_liquid_left_blocker[local.x] = s_liquid_left_scan_tmp[local.x];
                            s_liquid_right_blocker[local.x] = s_liquid_right_scan_tmp[local.x];
                            s_empty_left_blocker[local.x] = s_empty_left_scan_tmp[local.x];
                            s_empty_right_blocker[local.x] = s_empty_right_scan_tmp[local.x];
                            barrier();
                        }}

                        if (s_row_liquid_flag[local.x] != 0) {{
                            int liquid_start = s_liquid_left_blocker[local.x] + 1;
                            int liquid_end = s_liquid_right_blocker[local.x];
                            s_liquid_run_start[local.x] = liquid_start;
                            s_liquid_run_end[local.x] = liquid_end;
                            if (s_below_empty_flag[local.x] != 0) {{
                                atomicMin(s_liquid_run_first_empty_x[liquid_start], local.x);
                            }}
                        }}
                        if (s_below_empty_flag[local.x] != 0) {{
                            s_empty_segment_start[local.x] = s_empty_left_blocker[local.x] + 1;
                            s_empty_segment_end[local.x] = s_empty_right_blocker[local.x];
                        }}
                        memoryBarrierShared();
                        barrier();

                        if (s_row_liquid_flag[local.x] != 0) {{
                            int liquid_start = s_liquid_run_start[local.x];
                            int liquid_end = s_liquid_run_end[local.x];
                            int first_empty_x = s_liquid_run_first_empty_x[liquid_start];
                            if (first_empty_x < TILE_SIZE) {{
                                s_down_segment_claimed[local.x] = 1;
                                int empty_start = s_empty_segment_start[first_empty_x];
                                int empty_end = s_empty_segment_end[first_empty_x];
                                int move_count = min(liquid_end - liquid_start, empty_end - empty_start);
                                int source_offset = local.x - liquid_start;
                                if (source_offset < move_count) {{
                                    int target_base = clamp(liquid_start, empty_start, empty_end - move_count);
                                    int target_x = target_base + source_offset;
                                    s_down_target_x[local.x] = target_x;
                                    atomicMin(s_down_source_x[target_x], local.x);
                                }}
                            }}
                        }}
                        memoryBarrierShared();
                        barrier();

                        if (s_down_source_x[local.x] == TILE_SIZE) {{
                            s_down_source_x[local.x] = -1;
                        }}
                        barrier();

                        source_x = s_down_source_x[local.x];
                        if (source_x >= 0) {{
                            write_cell(
                                row + 1,
                                local.x,
                                s_material[row][source_x],
                                s_phase[row][source_x],
                                s_flags[row][source_x],
                                s_source_x[row][source_x],
                                s_source_y[row][source_x]
                            );
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(source_x, row));
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row + 1));
                        }}
                        barrier();
                        target_x = s_down_target_x[local.x];
                        if (target_x >= 0 && s_down_source_x[target_x] == local.x) {{
                            clear_cell(row, local.x);
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row));
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(target_x, row + 1));
                        }}
                        barrier();

                        if (
                            s_down_segment_claimed[local.x] == 0
                            && local_cell_in_world(tile, row, local.x)
                            && local_cell_in_world(tile, row + 1, local.x)
                            && is_liquid_cell(row, local.x)
                            && is_vertical_empty(row + 1, local.x)
                            && s_down_source_x[local.x] < 0
                        ) {{
                            write_cell(
                                row + 1,
                                local.x,
                                s_material[row][local.x],
                                s_phase[row][local.x],
                                s_flags[row][local.x],
                                s_source_x[row][local.x],
                                s_source_y[row][local.x]
                            );
                            clear_cell(row, local.x);
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row));
                            refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row + 1));
                        }}
                        barrier();
                    }}
                    barrier();

                    s_lateral_target_x[local.x] = -1;
                    s_lateral_source_x[local.x] = -1;
                    barrier();

                    if (
                        s_down_segment_claimed[local.x] == 0
                        && local_cell_in_world(tile, row, local.x)
                        && local_cell_in_world(tile, row + 1, local.x)
                        && is_columnar_liquid_cell(row, local.x)
                    ) {{
                        if (
                            local.x > 0
                            && local_cell_in_world(tile, row, local.x - 1)
                            && is_reachable_empty(row, local.x - 1)
                        ) {{
                            s_lateral_target_x[local.x] = local.x - 1;
                        }} else if (
                            local.x + 1 < TILE_SIZE
                            && local_cell_in_world(tile, row, local.x + 1)
                            && is_reachable_empty(row, local.x + 1)
                        ) {{
                            s_lateral_target_x[local.x] = local.x + 1;
                        }}
                    }}
                    barrier();

                    int owner_x = -1;
                    if (local.x > 0 && s_lateral_target_x[local.x - 1] == local.x) {{
                        owner_x = local.x - 1;
                    }} else if (local.x + 1 < TILE_SIZE && s_lateral_target_x[local.x + 1] == local.x) {{
                        owner_x = local.x + 1;
                    }}
                    s_lateral_source_x[local.x] = owner_x;
                    barrier();

                    source_x = s_lateral_source_x[local.x];
                    if (source_x >= 0) {{
                        write_cell(
                            row,
                            local.x,
                            s_material[row][source_x],
                            s_phase[row][source_x],
                            s_flags[row][source_x],
                            s_source_x[row][source_x],
                            s_source_y[row][source_x]
                        );
                        refresh_changed_cell(tile * TILE_SIZE + ivec2(source_x, row));
                        refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row));
                    }}
                    barrier();

                    target_x = s_lateral_target_x[local.x];
                    if (target_x >= 0 && s_lateral_source_x[target_x] == local.x) {{
                        clear_cell(row, local.x);
                        refresh_changed_cell(tile * TILE_SIZE + ivec2(local.x, row));
                        refresh_changed_cell(tile * TILE_SIZE + ivec2(target_x, row));
                    }}
                    barrier();
                }}
                barrier();
                for (int y = 0; y < TILE_SIZE; ++y) {{
                    ivec2 cell = tile * TILE_SIZE + ivec2(local.x, y);
                    bool in_bounds = cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
                    if (in_bounds) {{
                        if (s_changed[y][local.x] == 0) {{
                            continue;
                        }}
                        int source_x = s_source_x[y][local.x];
                        int source_y = s_source_y[y][local.x];
                        bool has_source = source_x >= 0 && source_y >= 0;
                        ivec2 source_cell = tile * TILE_SIZE + ivec2(max(source_x, 0), max(source_y, 0));
                        vec4 out_timer = has_source ? texelFetch(timer_in_tex, source_cell, 0) : vec4(0.0);
                        float out_temp = has_source ? texelFetch(temp_in_tex, source_cell, 0).x : 0.0;
                        float out_integrity = has_source ? texelFetch(integrity_in_tex, source_cell, 0).x : 0.0;
                        vec2 out_velocity = has_source ? texelFetch(velocity_in_tex, source_cell, 0).xy : vec2(0.0);
                        imageStore(material_out_img, cell, vec4(s_material[y][local.x], 0.0, 0.0, 0.0));
                        imageStore(phase_out_img, cell, vec4(s_phase[y][local.x], 0.0, 0.0, 0.0));
                        imageStore(flags_out_img, cell, vec4(s_flags[y][local.x], 0.0, 0.0, 0.0));
                        imageStore(timer_out_img, cell, out_timer);
                        imageStore(temp_out_img, cell, vec4(out_temp, 0.0, 0.0, 0.0));
                        imageStore(integrity_out_img, cell, vec4(out_integrity, 0.0, 0.0, 0.0));
                        imageStore(velocity_out_img, cell, vec4(out_velocity, 0.0, 0.0));
                    }}
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
            uniform int active_ttl_reset;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform bool use_boundary_dispatch;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(binding=8) uniform sampler2D entity_in_tex;
            layout(binding=9) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer AffectedTileCountBuffer {{
                uint affected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer AffectedTileListBuffer {{
                ivec2 affected_tile_list[];
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

            bool vertical_boundary_active(int boundary_x, int y) {{
                if (boundary_x <= 0 || boundary_x >= cell_grid_size.x) {{
                    return false;
                }}
                int tile_y = min(tile_grid_size.y - 1, max(0, y / TILE_SIZE));
                int right_tile = min(tile_grid_size.x - 1, boundary_x / TILE_SIZE);
                int left_tile = max(0, right_tile - 1);
                return texelFetch(active_tile_tex, ivec2(left_tile, tile_y), 0).x > 0.5
                    || texelFetch(active_tile_tex, ivec2(right_tile, tile_y), 0).x > 0.5;
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

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool is_tile_liquid(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                float material = texelFetch(material_in_tex, cell, 0).x;
                float phase = texelFetch(phase_in_tex, cell, 0).x;
                return material > 0.5 && int(phase + 0.5) == phase_liquid && liquid_kind_for(material) == 1;
            }}

            bool is_placeholder_material(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            bool is_reachable_empty(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                float material = texelFetch(material_in_tex, cell, 0).x;
                int phase = int(texelFetch(phase_in_tex, cell, 0).x + 0.5);
                float entity = texelFetch(entity_in_tex, cell, 0).x;
                float displaced = texelFetch(displaced_in_tex, cell, 0).x;
                return material < 0.5
                    && phase != phase_liquid
                    && phase != phase_falling_island
                    && entity < 0.5
                    && displaced < 0.5
                    && !is_placeholder_material(material);
            }}

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / TILE_SIZE, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / TILE_SIZE, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            bool left_to_right_move(
                int boundary_x,
                int y,
                out int source_base,
                out int target_base,
                out int move_count
            ) {{
                source_base = 0;
                target_base = 0;
                move_count = 0;
                if (boundary_x <= 0 || boundary_x >= cell_grid_size.x) {{
                    return false;
                }}
                int source_end = boundary_x - 1;
                if (!is_tile_liquid(ivec2(source_end, y)) || !is_reachable_empty(ivec2(boundary_x, y))) {{
                    return false;
                }}
                int source_tile_start = (source_end / TILE_SIZE) * TILE_SIZE;
                int source_start = source_end;
                while (source_start > source_tile_start && is_tile_liquid(ivec2(source_start - 1, y))) {{
                    source_start -= 1;
                }}
                int target_start = boundary_x;
                int target_end = target_start;
                int target_tile_end = min(cell_grid_size.x, target_start + TILE_SIZE);
                while (target_end < target_tile_end && is_reachable_empty(ivec2(target_end, y))) {{
                    target_end += 1;
                }}
                move_count = min(source_end - source_start + 1, target_end - target_start);
                if (move_count <= 0) {{
                    return false;
                }}
                source_base = source_end - move_count + 1;
                target_base = target_start;
                return true;
            }}

            bool right_to_left_move(
                int boundary_x,
                int y,
                out int source_base,
                out int target_base,
                out int move_count
            ) {{
                source_base = 0;
                target_base = 0;
                move_count = 0;
                if (boundary_x <= 0 || boundary_x >= cell_grid_size.x) {{
                    return false;
                }}
                int source_start = boundary_x;
                if (!is_tile_liquid(ivec2(source_start, y)) || !is_reachable_empty(ivec2(boundary_x - 1, y))) {{
                    return false;
                }}
                int source_end = source_start + 1;
                int source_tile_end = min(cell_grid_size.x, source_start + TILE_SIZE);
                while (source_end < source_tile_end && is_tile_liquid(ivec2(source_end, y))) {{
                    source_end += 1;
                }}
                int target_end = boundary_x;
                int target_start = target_end - 1;
                int target_tile_start = max(0, target_end - TILE_SIZE);
                while (target_start > target_tile_start && is_reachable_empty(ivec2(target_start - 1, y))) {{
                    target_start -= 1;
                }}
                move_count = min(source_end - source_start, target_end - target_start);
                if (move_count <= 0) {{
                    return false;
                }}
                source_base = source_start;
                target_base = target_end - move_count;
                return true;
            }}

            bool try_apply_horizontal_boundary(ivec2 gid, int boundary_x) {{
                int source_base;
                int target_base;
                int move_count;
                if (left_to_right_move(boundary_x, gid.y, source_base, target_base, move_count)) {{
                    if (is_tile_liquid(gid) && gid.x >= source_base && gid.x < source_base + move_count) {{
                        store_state(gid, 0.0, 0.0, 0.0, vec4(0.0), 0.0, 0.0, vec2(0.0));
                        refresh_changed_cell(gid);
                        return true;
                    }}
                    int target_offset = gid.x - target_base;
                    if (is_reachable_empty(gid) && target_offset >= 0 && target_offset < move_count) {{
                        ivec2 source = ivec2(source_base + target_offset, gid.y);
                        store_state(
                            gid,
                            texelFetch(material_in_tex, source, 0).x,
                            texelFetch(phase_in_tex, source, 0).x,
                            texelFetch(flags_in_tex, source, 0).x,
                            texelFetch(timer_in_tex, source, 0),
                            texelFetch(temp_in_tex, source, 0).x,
                            texelFetch(integrity_in_tex, source, 0).x,
                            texelFetch(velocity_in_tex, source, 0).xy
                        );
                        refresh_changed_cell(source);
                        refresh_changed_cell(gid);
                        return true;
                    }}
                }}
                if (right_to_left_move(boundary_x, gid.y, source_base, target_base, move_count)) {{
                    if (is_tile_liquid(gid) && gid.x >= source_base && gid.x < source_base + move_count) {{
                        store_state(gid, 0.0, 0.0, 0.0, vec4(0.0), 0.0, 0.0, vec2(0.0));
                        refresh_changed_cell(gid);
                        return true;
                    }}
                    int target_offset = gid.x - target_base;
                    if (is_reachable_empty(gid) && target_offset >= 0 && target_offset < move_count) {{
                        ivec2 source = ivec2(source_base + target_offset, gid.y);
                        store_state(
                            gid,
                            texelFetch(material_in_tex, source, 0).x,
                            texelFetch(phase_in_tex, source, 0).x,
                            texelFetch(flags_in_tex, source, 0).x,
                            texelFetch(timer_in_tex, source, 0),
                            texelFetch(temp_in_tex, source, 0).x,
                            texelFetch(integrity_in_tex, source, 0).x,
                            texelFetch(velocity_in_tex, source, 0).xy
                        );
                        refresh_changed_cell(source);
                        refresh_changed_cell(gid);
                        return true;
                    }}
                }}
                return false;
            }}

            bool boundary_dispatch_cell(out ivec2 cell, out int boundary_x) {{
                int groups_x = max(1, (TILE_SIZE * 2 + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int groups_y = max(1, (TILE_SIZE + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_boundary = groups_x * groups_y;
                uint group_index = gl_WorkGroupID.x;
                uint boundary_index = group_index / uint(workgroups_per_boundary);
                if (boundary_index >= affected_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_boundary));
                ivec2 subtile_xy = ivec2(subtile % groups_x, subtile / groups_x);
                ivec2 boundary_tile = affected_tile_list[int(boundary_index)];
                if (
                    boundary_tile.x < 0
                    || boundary_tile.y < 0
                    || boundary_tile.x + 1 >= tile_grid_size.x
                    || boundary_tile.y >= tile_grid_size.y
                ) {{
                    return false;
                }}
                ivec2 origin = ivec2(boundary_tile.x * TILE_SIZE, boundary_tile.y * TILE_SIZE);
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = origin + local_cell;
                ivec2 region_end = min(origin + ivec2(TILE_SIZE * 2, TILE_SIZE), cell_grid_size);
                boundary_x = (boundary_tile.x + 1) * TILE_SIZE;
                return cell.x < region_end.x && cell.y < region_end.y;
            }}

            void main() {{
                ivec2 gid;
                int dispatch_boundary_x = -1;
                if (use_boundary_dispatch) {{
                    if (!boundary_dispatch_cell(gid, dispatch_boundary_x)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (use_boundary_dispatch) {{
                    if (try_apply_horizontal_boundary(gid, dispatch_boundary_x)) {{
                        return;
                    }}
                    return;
                }}
                int source_boundary = ((gid.x / TILE_SIZE) + 1) * TILE_SIZE;
                if (source_boundary < cell_grid_size.x && vertical_boundary_active(source_boundary, gid.y)) {{
                    if (try_apply_horizontal_boundary(gid, source_boundary)) {{
                        return;
                    }}
                }}
                int target_boundary = (gid.x / TILE_SIZE) * TILE_SIZE;
                if (target_boundary > 0 && vertical_boundary_active(target_boundary, gid.y)) {{
                    if (try_apply_horizontal_boundary(gid, target_boundary)) {{
                        return;
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
            uniform int active_ttl_reset;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform bool use_boundary_dispatch;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D active_tile_tex;
            layout(binding=8) uniform sampler2D entity_in_tex;
            layout(binding=9) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer AffectedTileCountBuffer {{
                uint affected_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer AffectedTileListBuffer {{
                ivec2 affected_tile_list[];
            }};
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

            int liquid_kind_for(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].z + 0.5);
            }}

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool is_tile_liquid(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                float material = texelFetch(material_in_tex, cell, 0).x;
                float phase = texelFetch(phase_in_tex, cell, 0).x;
                return material > 0.5 && int(phase + 0.5) == phase_liquid && liquid_kind_for(material) == 1;
            }}

            bool is_placeholder_material(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            bool is_reachable_empty(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                float material = texelFetch(material_in_tex, cell, 0).x;
                int phase = int(texelFetch(phase_in_tex, cell, 0).x + 0.5);
                float entity = texelFetch(entity_in_tex, cell, 0).x;
                float displaced = texelFetch(displaced_in_tex, cell, 0).x;
                return material < 0.5
                    && phase != phase_liquid
                    && phase != phase_falling_island
                    && entity < 0.5
                    && displaced < 0.5
                    && !is_placeholder_material(material);
            }}

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / TILE_SIZE, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / TILE_SIZE, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            bool down_move_from_liquid(
                int row_start,
                int row_end,
                int top_y,
                int probe_x,
                out int source_base,
                out int target_base,
                out int move_count
            ) {{
                source_base = 0;
                target_base = 0;
                move_count = 0;
                if (!is_tile_liquid(ivec2(probe_x, top_y))) {{
                    return false;
                }}
                int liquid_start = probe_x;
                while (liquid_start > row_start && is_tile_liquid(ivec2(liquid_start - 1, top_y))) {{
                    liquid_start -= 1;
                }}
                int liquid_end = probe_x + 1;
                while (liquid_end < row_end && is_tile_liquid(ivec2(liquid_end, top_y))) {{
                    liquid_end += 1;
                }}
                int empty_start = -1;
                int empty_end = -1;
                for (int x = liquid_start; x < liquid_end; ++x) {{
                    if (is_reachable_empty(ivec2(x, top_y + 1))) {{
                        empty_start = x;
                        while (empty_start > row_start && is_reachable_empty(ivec2(empty_start - 1, top_y + 1))) {{
                            empty_start -= 1;
                        }}
                        empty_end = x + 1;
                        while (empty_end < row_end && is_reachable_empty(ivec2(empty_end, top_y + 1))) {{
                            empty_end += 1;
                        }}
                        break;
                    }}
                }}
                if (empty_start < 0) {{
                    return false;
                }}
                move_count = min(liquid_end - liquid_start, empty_end - empty_start);
                if (move_count <= 0) {{
                    return false;
                }}
                source_base = liquid_start;
                target_base = clamp(liquid_start, empty_start, empty_end - move_count);
                return true;
            }}

            bool down_move_from_empty(
                int row_start,
                int row_end,
                int bottom_y,
                int probe_x,
                out int source_base,
                out int target_base,
                out int move_count
            ) {{
                source_base = 0;
                target_base = 0;
                move_count = 0;
                if (!is_reachable_empty(ivec2(probe_x, bottom_y))) {{
                    return false;
                }}
                int empty_start = probe_x;
                while (empty_start > row_start && is_reachable_empty(ivec2(empty_start - 1, bottom_y))) {{
                    empty_start -= 1;
                }}
                int empty_end = probe_x + 1;
                while (empty_end < row_end && is_reachable_empty(ivec2(empty_end, bottom_y))) {{
                    empty_end += 1;
                }}
                int liquid_probe = -1;
                for (int x = empty_start; x < empty_end; ++x) {{
                    if (is_tile_liquid(ivec2(x, bottom_y - 1))) {{
                        liquid_probe = x;
                        break;
                    }}
                }}
                if (liquid_probe < 0) {{
                    return false;
                }}
                int liquid_start = liquid_probe;
                while (liquid_start > row_start && is_tile_liquid(ivec2(liquid_start - 1, bottom_y - 1))) {{
                    liquid_start -= 1;
                }}
                int liquid_end = liquid_probe + 1;
                while (liquid_end < row_end && is_tile_liquid(ivec2(liquid_end, bottom_y - 1))) {{
                    liquid_end += 1;
                }}
                empty_start = -1;
                empty_end = -1;
                for (int x = liquid_start; x < liquid_end; ++x) {{
                    if (is_reachable_empty(ivec2(x, bottom_y))) {{
                        empty_start = x;
                        while (empty_start > row_start && is_reachable_empty(ivec2(empty_start - 1, bottom_y))) {{
                            empty_start -= 1;
                        }}
                        empty_end = x + 1;
                        while (empty_end < row_end && is_reachable_empty(ivec2(empty_end, bottom_y))) {{
                            empty_end += 1;
                        }}
                        break;
                    }}
                }}
                if (empty_start < 0) {{
                    return false;
                }}
                move_count = min(liquid_end - liquid_start, empty_end - empty_start);
                if (move_count <= 0) {{
                    return false;
                }}
                source_base = liquid_start;
                target_base = clamp(liquid_start, empty_start, empty_end - move_count);
                return true;
            }}

            bool try_apply_vertical_boundary(ivec2 gid, int boundary_y) {{
                if (boundary_y <= 0 || boundary_y >= cell_grid_size.y) {{
                    return false;
                }}
                int row_start = (gid.x / TILE_SIZE) * TILE_SIZE;
                int row_end = min(cell_grid_size.x, row_start + TILE_SIZE);
                int source_base;
                int target_base;
                int move_count;
                if (gid.y == boundary_y - 1) {{
                    if (down_move_from_liquid(row_start, row_end, gid.y, gid.x, source_base, target_base, move_count)) {{
                        if (is_tile_liquid(gid) && gid.x >= source_base && gid.x < source_base + move_count) {{
                            store_state(gid, 0.0, 0.0, 0.0, vec4(0.0), 0.0, 0.0, vec2(0.0));
                            refresh_changed_cell(gid);
                            return true;
                        }}
                    }}
                    return false;
                }}
                if (gid.y == boundary_y) {{
                    if (down_move_from_empty(row_start, row_end, gid.y, gid.x, source_base, target_base, move_count)) {{
                        int target_offset = gid.x - target_base;
                        if (target_offset >= 0 && target_offset < move_count && is_reachable_empty(gid)) {{
                            ivec2 source = ivec2(source_base + target_offset, gid.y - 1);
                            store_state(
                                gid,
                                texelFetch(material_in_tex, source, 0).x,
                                texelFetch(phase_in_tex, source, 0).x,
                                texelFetch(flags_in_tex, source, 0).x,
                                texelFetch(timer_in_tex, source, 0),
                                texelFetch(temp_in_tex, source, 0).x,
                                texelFetch(integrity_in_tex, source, 0).x,
                                texelFetch(velocity_in_tex, source, 0).xy
                            );
                            refresh_changed_cell(source);
                            refresh_changed_cell(gid);
                            return true;
                        }}
                    }}
                }}
                return false;
            }}

            bool boundary_dispatch_cell(out ivec2 cell, out int boundary_y) {{
                int groups_x = max(1, (TILE_SIZE + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int groups_y = 1;
                int workgroups_per_boundary = groups_x * groups_y;
                uint group_index = gl_WorkGroupID.x;
                uint boundary_index = group_index / uint(workgroups_per_boundary);
                if (boundary_index >= affected_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_boundary));
                ivec2 subtile_xy = ivec2(subtile % groups_x, subtile / groups_x);
                ivec2 boundary_tile = affected_tile_list[int(boundary_index)];
                if (
                    boundary_tile.x < 0
                    || boundary_tile.y < 0
                    || boundary_tile.x >= tile_grid_size.x
                    || boundary_tile.y + 1 >= tile_grid_size.y
                ) {{
                    return false;
                }}
                boundary_y = (boundary_tile.y + 1) * TILE_SIZE;
                ivec2 origin = ivec2(boundary_tile.x * TILE_SIZE, boundary_y - 1);
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                if (local_cell.y >= 2) {{
                    return false;
                }}
                cell = origin + local_cell;
                ivec2 region_end = min(origin + ivec2(TILE_SIZE, 2), cell_grid_size);
                return cell.x < region_end.x && cell.y < region_end.y;
            }}

            void main() {{
                ivec2 gid;
                int dispatch_boundary_y = -1;
                if (use_boundary_dispatch) {{
                    if (!boundary_dispatch_cell(gid, dispatch_boundary_y)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                if (use_boundary_dispatch) {{
                    if (try_apply_vertical_boundary(gid, dispatch_boundary_y)) {{
                        return;
                    }}
                    return;
                }}
                if (boundary_active(gid)) {{
                    if (gid.y % TILE_SIZE == TILE_SIZE - 1 && gid.y + 1 < cell_grid_size.y) {{
                        if (try_apply_vertical_boundary(gid, gid.y + 1)) {{
                            return;
                        }}
                    }} else if (gid.y % TILE_SIZE == 0 && gid.y > 0) {{
                        if (try_apply_vertical_boundary(gid, gid.y)) {{
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
            uniform int active_ttl_reset;
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
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                {{
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
                                refresh_changed_cell(gid);
                                refresh_changed_cell(below);
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
                                refresh_changed_cell(gid);
                                refresh_changed_cell(above);
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
            uniform int active_ttl_reset;
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
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                float material = texelFetch(material_in_tex, gid, 0).x;
                float phase = texelFetch(phase_in_tex, gid, 0).x;
                float flagsv = texelFetch(flags_in_tex, gid, 0).x;
                vec4 timerv = texelFetch(timer_in_tex, gid, 0);
                float tempv = texelFetch(temp_in_tex, gid, 0).x;
                float integrityv = texelFetch(integrity_in_tex, gid, 0).x;
                vec2 velocityv = texelFetch(velocity_in_tex, gid, 0).xy;
                {{
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
                                refresh_changed_cell(gid);
                                refresh_changed_cell(below);
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
                                refresh_changed_cell(gid);
                                refresh_changed_cell(above);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(binding=7) uniform sampler2D displaced_in_tex;
            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;
            layout(r32f, binding=7) writeonly uniform image2D displaced_out_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
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
        self.programs["copy_core_state"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;
            layout(binding=0) uniform sampler2D material_in_tex;
            layout(binding=1) uniform sampler2D phase_in_tex;
            layout(binding=2) uniform sampler2D flags_in_tex;
            layout(binding=3) uniform sampler2D timer_in_tex;
            layout(binding=4) uniform sampler2D temp_in_tex;
            layout(binding=5) uniform sampler2D integrity_in_tex;
            layout(binding=6) uniform sampler2D velocity_in_tex;
            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
                }}
                imageStore(material_out_img, gid, vec4(texelFetch(material_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(texelFetch(phase_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, gid, vec4(texelFetch(flags_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, texelFetch(timer_in_tex, gid, 0));
                imageStore(temp_out_img, gid, vec4(texelFetch(temp_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(integrity_out_img, gid, vec4(texelFetch(integrity_in_tex, gid, 0).x, 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, vec4(texelFetch(velocity_in_tex, gid, 0).xy, 0.0, 0.0));
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
            uniform int active_ttl_reset;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform int placeholder_material_id;
            uniform uint placeholder_claim_epoch;
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
            layout(std430, binding=1) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=3) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=4) buffer AffectedTileFlagsBuffer {{
                uint affected_tile_flags[];
            }};
            layout(std430, binding=5) buffer PlaceholderTargetClaimsBuffer {{
                uint placeholder_target_claims[];
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

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            float base_integrity_for(int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                return material_params[material_id].y;
            }}

            void refresh_changed_cell(ivec2 changed_cell) {{
                if (
                    changed_cell.x < 0
                    || changed_cell.y < 0
                    || changed_cell.x >= cell_grid_size.x
                    || changed_cell.y >= cell_grid_size.y
                ) {{
                    return;
                }}
                ivec2 changed_tile = ivec2(
                    clamp(changed_cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(changed_cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[changed_tile.y * tile_grid_size.x + changed_tile.x] = active_ttl_reset;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                affected_tile_flags[tile.y * tile_grid_size.x + tile.x] = 0u;
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            bool target_empty(ivec2 target) {{
                if (!in_bounds(target)) {{
                    return false;
                }}
                if (texelFetch(material_in_tex, target, 0).x > 0.5) {{
                    return false;
                }}
                int target_phase = int(texelFetch(phase_in_tex, target, 0).x + 0.5);
                if (target_phase == phase_liquid || target_phase == phase_falling_island) {{
                    return false;
                }}
                if (texelFetch(displaced_in_tex, target, 0).x > 0.5) {{
                    return false;
                }}
                return true;
            }}

            bool side_lane_reachable(int side, int target_x, int target_y, int left, int right) {{
                if (target_y < 0 || target_y >= cell_grid_size.y) {{
                    return false;
                }}
                if (side < 0) {{
                    for (int x = target_x; x < left; ++x) {{
                        if (!target_empty(ivec2(x, target_y))) {{
                            return false;
                        }}
                    }}
                    return true;
                }}
                for (int x = right; x <= target_x; ++x) {{
                    if (!target_empty(ivec2(x, target_y))) {{
                        return false;
                    }}
                }}
                return true;
            }}

            bool segment_top_exposed(int left, int right, int source_y) {{
                if (source_y == 0) {{
                    return true;
                }}
                for (int x = left; x < right; ++x) {{
                    if (!is_placeholder(ivec2(x, source_y - 1))) {{
                        return true;
                    }}
                }}
                return false;
            }}

            int side_lane_capacity(int side, bool top_lane, int seg_len, int left, int right, int source_y) {{
                int target_y = top_lane ? source_y - 1 : source_y;
                if (target_y < 0 || target_y >= cell_grid_size.y) {{
                    return 0;
                }}
                int capacity = 0;
                for (int slot = 0; slot < seg_len; ++slot) {{
                    int target_x = side < 0 ? left - 1 - slot : right + slot;
                    if (side_lane_reachable(side, target_x, target_y, left, right)) {{
                        capacity += 1;
                    }}
                }}
                return capacity;
            }}

            int side_capacity(int side, bool top_exposed, int seg_len, int left, int right, int source_y) {{
                int capacity = side_lane_capacity(side, false, seg_len, left, right, source_y);
                if (top_exposed) {{
                    capacity += side_lane_capacity(side, true, seg_len, left, right, source_y);
                }}
                return capacity;
            }}

            int placeholder_left_quota(int displaced_count, int left_capacity, int right_capacity) {{
                int total_capacity = left_capacity + right_capacity;
                if (displaced_count <= 0 || total_capacity <= 0) {{
                    return 0;
                }}
                int numerator = displaced_count * left_capacity;
                int quota = numerator / total_capacity;
                int remainder = numerator - quota * total_capacity;
                if (remainder * 2 >= total_capacity) {{
                    quota += 1;
                }}
                return clamp(quota, 0, displaced_count);
            }}

            bool try_claim_target(ivec2 target) {{
                int target_index = target.y * cell_grid_size.x + target.x;
                for (int attempt = 0; attempt < 2; ++attempt) {{
                    uint observed = placeholder_target_claims[target_index];
                    if (observed == placeholder_claim_epoch) {{
                        return false;
                    }}
                    uint previous = atomicCompSwap(
                        placeholder_target_claims[target_index],
                        observed,
                        placeholder_claim_epoch
                    );
                    if (previous == observed) {{
                        return true;
                    }}
                }}
                return false;
            }}

            bool try_emit(
                ivec2 target,
                int side,
                int left,
                int right,
                float liquid_material,
                float source_temp,
                vec2 source_velocity,
                vec2 push_velocity
            ) {{
                if (!in_bounds(target)) {{
                    return false;
                }}
                if (!side_lane_reachable(side, target.x, target.y, left, right)) {{
                    return false;
                }}
                if (!try_claim_target(target)) {{
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
                refresh_changed_cell(target);
                return true;
            }}

            bool try_emit_side_lane(
                int side,
                bool top_lane,
                int start_slot,
                int seg_len,
                int left,
                int right,
                int source_y,
                float liquid_material,
                float source_temp,
                vec2 source_velocity
            ) {{
                int target_y = top_lane ? source_y - 1 : source_y;
                if (target_y < 0 || target_y >= cell_grid_size.y) {{
                    return false;
                }}
                for (int offset = 0; offset < seg_len; ++offset) {{
                    int slot = (start_slot + offset) % seg_len;
                    int target_x = side < 0 ? left - 1 - slot : right + slot;
                    vec2 push_velocity = top_lane
                        ? vec2(float(side) * 0.8, -0.65)
                        : vec2(float(side) * 1.2, -0.15);
                    if (try_emit(
                        ivec2(target_x, target_y),
                        side,
                        left,
                        right,
                        liquid_material,
                        source_temp,
                        source_velocity,
                        push_velocity
                    )) {{
                        return true;
                    }}
                }}
                return false;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                if (!is_placeholder(gid)) {{
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
                int displaced_count = 0;
                int displaced_rank = 0;
                for (int x = left; x < right; ++x) {{
                    if (texelFetch(displaced_in_tex, ivec2(x, gid.y), 0).x > 0.5) {{
                        if (x < gid.x) {{
                            displaced_rank += 1;
                        }}
                        displaced_count += 1;
                    }}
                }}
                if (displaced_count <= 0) {{
                    return;
                }}
                bool top_exposed = segment_top_exposed(left, right, gid.y);
                int left_capacity = side_capacity(-1, top_exposed, seg_len, left, right, gid.y);
                int right_capacity = side_capacity(1, top_exposed, seg_len, left, right, gid.y);
                int left_quota = placeholder_left_quota(displaced_count, left_capacity, right_capacity);
                bool prefer_left = displaced_rank < left_quota;
                int side = prefer_left ? -1 : 1;
                int side_rank = prefer_left ? displaced_rank : displaced_count - 1 - displaced_rank;
                side_rank = clamp(side_rank, 0, max(seg_len - 1, 0));
                float source_temp = texelFetch(temp_in_tex, gid, 0).x;
                vec2 source_velocity = texelFetch(velocity_in_tex, gid, 0).xy;
                bool emitted = false;
                emitted = try_emit_side_lane(side, false, side_rank, seg_len, left, right, gid.y, liquid_material, source_temp, source_velocity);
                if (!emitted && top_exposed) {{
                    emitted = try_emit_side_lane(side, true, side_rank, seg_len, left, right, gid.y, liquid_material, source_temp, source_velocity);
                }}
                if (emitted) {{
                    imageStore(displaced_out_img, gid, vec4(0.0, 0.0, 0.0, 0.0));
                    refresh_changed_cell(gid);
                }} else {{
                    refresh_changed_cell(gid);
                }}
            }}
            """
        )
        self.programs["cleanup_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_falling_island;
            uniform bool use_active_tile_dispatch;
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
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            bool is_placeholder_material(int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
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
        self.programs["liquid_flow_intent"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D velocity_tex;
            layout(binding=3) uniform sampler2D active_tile_tex;
            layout(binding=4) uniform sampler2D entity_tex;
            layout(binding=5) uniform sampler2D displaced_tex;
            layout(std430, binding=0) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(rg32f, binding=0) writeonly uniform image2D liquid_flow_intent_img;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool tile_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(tile_grid_size.x - 1, max(0, cell.x / tile_size)),
                    min(tile_grid_size.y - 1, max(0, cell.y / tile_size))
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            bool is_placeholder_material(float material) {{
                int material_id = clamp(int(material + 0.5), 0, {MAX_MATERIALS - 1});
                return int(material_params[material_id].w + 0.5) == 7;
            }}

            bool reachable_empty(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                float material = texelFetch(material_tex, cell, 0).x;
                int phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                float entity = texelFetch(entity_tex, cell, 0).x;
                float displaced = texelFetch(displaced_tex, cell, 0).x;
                return material < 0.5
                    && phase_id != phase_liquid
                    && phase_id != phase_falling_island
                    && entity < 0.5
                    && displaced < 0.5
                    && !is_placeholder_material(material);
            }}

            int flow_sign(ivec2 cell, vec2 velocity) {{
                if (velocity.x > 0.25) {{
                    return 1;
                }}
                if (velocity.x < -0.25) {{
                    return -1;
                }}
                uint hashv = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u;
                return (hashv & 1u) == 0u ? -1 : 1;
            }}

            vec2 liquid_intent(ivec2 cell, vec2 velocity) {{
                int direction = flow_sign(cell, velocity);
                ivec2 below = cell + ivec2(0, 1);
                if (reachable_empty(below)) {{
                    return vec2(float(direction) * 0.35, 2.0);
                }}
                ivec2 side = cell + ivec2(direction, 0);
                if (reachable_empty(side)) {{
                    return vec2(float(direction) * 2.0, 0.0);
                }}
                ivec2 diag = cell + ivec2(direction, 1);
                if (reachable_empty(diag)) {{
                    return vec2(float(direction) * 1.5, 1.0);
                }}
                ivec2 other_side = cell + ivec2(-direction, 0);
                if (reachable_empty(other_side)) {{
                    return vec2(float(-direction) * 2.0, 0.0);
                }}
                ivec2 other_diag = cell + ivec2(-direction, 1);
                if (reachable_empty(other_diag)) {{
                    return vec2(float(-direction) * 1.5, 1.0);
                }}
                return velocity * 0.35;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                vec2 intent = vec2(0.0);
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                int phase_id = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                if (material_id > 0 && phase_id == phase_liquid) {{
                    intent = liquid_intent(gid, velocity) - velocity;
                }}
                imageStore(liquid_flow_intent_img, gid, vec4(intent, 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool copy_cell_core;
            uniform bool use_active_tile_dispatch;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
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
        self.programs["load_bridge_flow_intent_inputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool copy_cell_core;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=2) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced[];
            }};
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_in_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_in_img;
            layout(rg32f, binding=2) writeonly uniform image2D velocity_in_img;
            layout(r32f, binding=3) writeonly uniform image2D entity_in_img;
            layout(r32f, binding=4) writeonly uniform image2D displaced_in_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                int cell_index = gid.y * cell_grid_size.x + gid.x;
                if (copy_cell_core) {{
                    int word_index = cell_index * 5;
                    uint word0 = bridge_cell_core[word_index];
                    imageStore(material_in_img, gid, vec4(float(word0 & 0xFFFFu), 0.0, 0.0, 0.0));
                    imageStore(phase_in_img, gid, vec4(float((word0 >> 16u) & 0xFFu), 0.0, 0.0, 0.0));
                    imageStore(velocity_in_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
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
        self.programs["load_bridge_cell_out"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_out_img;
            layout(r32f, binding=1) writeonly uniform image2D phase_out_img;
            layout(r32f, binding=2) writeonly uniform image2D flags_out_img;
            layout(rgba32f, binding=3) writeonly uniform image2D timer_out_img;
            layout(r32f, binding=4) writeonly uniform image2D temp_out_img;
            layout(r32f, binding=5) writeonly uniform image2D integrity_out_img;
            layout(rg32f, binding=6) writeonly uniform image2D velocity_out_img;

            vec4 unpack_timer(uint word) {{
                return vec4(
                    float(word & 0xFFu),
                    float((word >> 8u) & 0xFFu),
                    float((word >> 16u) & 0xFFu),
                    float((word >> 24u) & 0xFFu)
                );
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
                }}
                int word_index = (gid.y * cell_grid_size.x + gid.x) * 5;
                uint word0 = bridge_cell_core[word_index];
                float material = float(word0 & 0xFFFFu);
                float phase = float((word0 >> 16u) & 0xFFu);
                imageStore(material_out_img, gid, vec4(material, 0.0, 0.0, 0.0));
                imageStore(phase_out_img, gid, vec4(phase, 0.0, 0.0, 0.0));
                imageStore(flags_out_img, gid, vec4(float((word0 >> 24u) & 0xFFu), 0.0, 0.0, 0.0));
                imageStore(temp_out_img, gid, vec4(uintBitsToFloat(bridge_cell_core[word_index + 2]), 0.0, 0.0, 0.0));
                imageStore(timer_out_img, gid, unpack_timer(bridge_cell_core[word_index + 3]));
                imageStore(integrity_out_img, gid, vec4(float(bridge_cell_core[word_index + 4] & 0xFFFFu), 0.0, 0.0, 0.0));
                imageStore(velocity_out_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={PASS_LOCAL_SIZE}, local_size_y={PASS_LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool copy_cell_core;
            uniform bool copy_island_id;
            uniform bool copy_entity_id;
            uniform bool copy_displaced_material;
            uniform bool use_active_tile_dispatch;
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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(rg32f, binding=0) writeonly uniform image2D velocity_in_img;
            layout(r32f, binding=1) writeonly uniform image2D island_in_img;
            layout(r32f, binding=2) writeonly uniform image2D entity_in_img;
            layout(r32f, binding=3) writeonly uniform image2D displaced_in_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;
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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {PASS_LOCAL_SIZE - 1}) / {PASS_LOCAL_SIZE});
                int workgroups_per_tile = groups_per_tile_axis * groups_per_tile_axis;
                uint group_index = gl_WorkGroupID.x;
                uint active_tile_index = group_index / uint(workgroups_per_tile);
                if (active_tile_index >= active_tile_count[0]) {{
                    return false;
                }}
                int subtile = int(group_index % uint(workgroups_per_tile));
                ivec2 subtile_xy = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return false;
                }}
                ivec2 tile_origin = tile * tile_size;
                ivec2 local_cell = subtile_xy * {PASS_LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (use_active_tile_dispatch) {{
                    if (!active_dispatch_cell(gid)) {{
                        return;
                    }}
                }} else {{
                    gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                        return;
                    }}
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
        authoritative = world.bridge.gpu_authoritative_resources
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in authoritative
            and "active_chunk_mask" in authoritative
        )

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
        active_tile_indirect = self._formal_gpu_frame(world)
        group_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        group_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        if active_tile_indirect:
            with self._profile_pass(world, "liquid_load_bridge_inputs.active_tile_compact"):
                self._compact_active_tiles(
                    world,
                    resources,
                    workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
                )
        ran_copy = False
        if copy_cell_core:
            with self._profile_pass(world, "liquid_load_bridge_inputs.load_cell_in"):
                program = self.programs["load_bridge_cell"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["copy_cell_core"].value = bool(copy_cell_core)
                program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                resources.material_pre.bind_to_image(0, read=False, write=True)
                resources.material_in.bind_to_image(1, read=False, write=True)
                resources.phase_pre.bind_to_image(2, read=False, write=True)
                resources.phase_in.bind_to_image(3, read=False, write=True)
                resources.flags_in.bind_to_image(4, read=False, write=True)
                resources.timer_in.bind_to_image(5, read=False, write=True)
                resources.temp_in.bind_to_image(6, read=False, write=True)
                resources.integrity_in.bind_to_image(7, read=False, write=True)
                if active_tile_indirect:
                    self._run_active_tile_indirect(program, resources, "bridge cell input load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

            with self._profile_pass(world, "liquid_load_bridge_inputs.load_cell_out"):
                program = self.programs["load_bridge_cell_out"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                resources.material_out.bind_to_image(0, read=False, write=True)
                resources.phase_out.bind_to_image(1, read=False, write=True)
                resources.flags_out.bind_to_image(2, read=False, write=True)
                resources.timer_out.bind_to_image(3, read=False, write=True)
                resources.temp_out.bind_to_image(4, read=False, write=True)
                resources.integrity_out.bind_to_image(5, read=False, write=True)
                resources.velocity_out.bind_to_image(6, read=False, write=True)
                if active_tile_indirect:
                    self._run_active_tile_indirect(program, resources, "bridge cell output load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
            with self._profile_pass(world, "liquid_load_bridge_inputs.load_aux"):
                program = self.programs["load_bridge_cell_aux"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["copy_cell_core"].value = bool(copy_cell_core)
                program["copy_island_id"].value = bool(copy_island_id)
                program["copy_entity_id"].value = bool(copy_entity_id)
                program["copy_displaced_material"].value = bool(copy_displaced)
                program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
                bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
                bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                resources.velocity_in.bind_to_image(0, read=False, write=True)
                resources.island_in.bind_to_image(1, read=False, write=True)
                resources.entity_in.bind_to_image(2, read=False, write=True)
                resources.displaced_in.bind_to_image(3, read=False, write=True)
                if active_tile_indirect:
                    self._run_active_tile_indirect(program, resources, "bridge aux input load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

        if ran_copy:
            with self._profile_pass(world, "liquid_load_bridge_inputs.sync"):
                self._sync_compute_writes(bridge.ctx)

    def _load_authoritative_bridge_flow_intent_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if not (copy_cell_core or copy_entity_id or copy_displaced):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for flow-intent input state")
        self._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
        )
        program = self.programs["load_bridge_flow_intent_inputs"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["copy_cell_core"].value = bool(copy_cell_core)
        program["copy_entity_id"].value = bool(copy_entity_id)
        program["copy_displaced_material"].value = bool(copy_displaced)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
        resources.active_tile_count.bind_to_storage_buffer(binding=4)
        resources.active_tile_list.bind_to_storage_buffer(binding=5)
        resources.material_in.bind_to_image(0, read=False, write=True)
        resources.phase_in.bind_to_image(1, read=False, write=True)
        resources.velocity_in.bind_to_image(2, read=False, write=True)
        resources.entity_in.bind_to_image(3, read=False, write=True)
        resources.displaced_in.bind_to_image(4, read=False, write=True)
        self._run_active_tile_indirect(program, resources, "bridge flow-intent input load")
        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        use_out: bool = False,
        velocity_use_out: bool = False,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative output state")
            return
        program = self.programs["publish_bridge_cell"]
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        material_tex = resources.material_out if use_out else resources.material_in
        phase_tex = resources.phase_out if use_out else resources.phase_in
        flags_tex = resources.flags_out if use_out else resources.flags_in
        timer_tex = resources.timer_out if use_out else resources.timer_in
        temp_tex = resources.temp_out if use_out else resources.temp_in
        integrity_tex = resources.integrity_out if use_out else resources.integrity_in
        velocity_tex = resources.velocity_out if velocity_use_out else resources.velocity_in
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
        resources.active_tile_count.bind_to_storage_buffer(binding=4)
        resources.active_tile_list.bind_to_storage_buffer(binding=5)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "bridge cell publish")
        else:
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

    def _refresh_active_scheduler_from_ttl(self, world: "WorldEngine") -> None:
        bridge = world.bridge
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for active refresh")
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")

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

    def _active_tile_workgroups_per_tile(self, world: "WorldEngine") -> int:
        axis = max(1, (int(world.active.tile_size) + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        return axis * axis

    def _next_placeholder_claim_epoch(self, resources: GPULiquidResources, world: "WorldEngine") -> int:
        self._placeholder_claim_epoch += 1
        if self._placeholder_claim_epoch >= 0x7FFFFFFF:
            cell_count = max(1, int(world.width * world.height))
            resources.placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
            self._placeholder_claim_epoch = 1
        return self._placeholder_claim_epoch

    def _seam_workgroups_per_boundary(self, axis: str) -> int:
        if axis == "x":
            groups_x = max(1, (TILE_SIZE * 2 + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
            groups_y = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        elif axis == "y":
            groups_x = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
            groups_y = 1
        else:
            raise ValueError(f"unknown liquid seam axis: {axis}")
        return groups_x * groups_y

    def _reload_and_compact_active_cell_tiles(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
    ) -> None:
        self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        self._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
        )

    def _build_seam_boundary_dispatch(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        axis: str,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        program = self.programs[f"compact_seam_{axis}_boundaries_from_active_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU liquid seam {axis} boundary compaction requires ModernGL ComputeShader.run_indirect")
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["source_workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        program["workgroups_per_boundary"].value = int(self._seam_workgroups_per_boundary(axis))
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
        program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _prefetch_seam_boundary_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        axis: str,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        if axis not in ("x", "y"):
            raise ValueError(f"unknown liquid seam axis: {axis}")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid seam prefetch requires bridge GPU resources")
        world._require_gpu_authoritative_resources(
            "liquid seam boundary prefetch",
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
            "active_tile_ttl",
        )
        program = self.programs["prefetch_seam_boundary_bridge_inputs"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid seam prefetch requires ModernGL ComputeShader.run_indirect")
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["seam_axis"].value = 0 if axis == "x" else 1
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.affected_tile_count.bind_to_storage_buffer(binding=4)
        resources.affected_tile_list.bind_to_storage_buffer(binding=5)
        if axis == "x":
            material_tex = resources.material_out
            phase_tex = resources.phase_out
            flags_tex = resources.flags_out
            timer_tex = resources.timer_out
            temp_tex = resources.temp_out
            integrity_tex = resources.integrity_out
            velocity_tex = resources.velocity_out
        else:
            material_tex = resources.material_in
            phase_tex = resources.phase_in
            flags_tex = resources.flags_in
            timer_tex = resources.timer_in
            temp_tex = resources.temp_in
            integrity_tex = resources.integrity_in
            velocity_tex = resources.velocity_in
        material_tex.bind_to_image(0, read=False, write=True)
        phase_tex.bind_to_image(1, read=False, write=True)
        flags_tex.bind_to_image(2, read=False, write=True)
        timer_tex.bind_to_image(3, read=False, write=True)
        temp_tex.bind_to_image(4, read=False, write=True)
        integrity_tex.bind_to_image(5, read=False, write=True)
        velocity_tex.bind_to_image(6, read=False, write=True)
        program.run_indirect(resources.affected_tile_dispatch_args)
        self._sync_compute_writes(bridge.ctx)

        aux_program = self.programs["prefetch_seam_boundary_bridge_aux_inputs"]
        if not hasattr(aux_program, "run_indirect"):
            raise RuntimeError("GPU liquid seam aux prefetch requires ModernGL ComputeShader.run_indirect")
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        aux_program["tile_size"].value = int(world.active.tile_size)
        aux_program["seam_axis"].value = 0 if axis == "x" else 1
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=0)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=1)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=2)
        resources.affected_tile_count.bind_to_storage_buffer(binding=3)
        resources.affected_tile_list.bind_to_storage_buffer(binding=4)
        resources.entity_in.bind_to_image(0, read=False, write=True)
        resources.displaced_in.bind_to_image(1, read=False, write=True)
        aux_program.run_indirect(resources.affected_tile_dispatch_args)
        self._sync_compute_writes(bridge.ctx)

    def _build_placeholder_dirty_affected_tile_dispatch(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        material_tex: Any,
        displaced_tex: Any,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        dirty_rect_count = int(len(world.bridge_frame_placeholder_dirty_rects))
        if dirty_rect_count > 0:
            world.bridge.ensure_world_resources(world)
            if "placeholder_dirty_rect" not in world.bridge.buffers:
                raise RuntimeError("GPU liquid placeholder displacement requires placeholder dirty rect buffer")
            program = self.programs["compact_placeholder_dirty_affected_tiles"]
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["dirty_rect_count"].value = dirty_rect_count
            program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
            resources.affected_tile_count.bind_to_storage_buffer(binding=0)
            resources.affected_tile_list.bind_to_storage_buffer(binding=1)
            resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            resources.affected_tile_flags.bind_to_storage_buffer(binding=3)
            world.bridge.buffers["placeholder_dirty_rect"].bind_to_storage_buffer(binding=4)
            program.run((dirty_rect_count + 63) // 64, 1, 1)
            self._sync_compute_writes(ctx)

        pending_program = self.programs["compact_placeholder_active_pending_affected_tiles"]
        if not hasattr(pending_program, "run_indirect"):
            raise RuntimeError("GPU liquid placeholder pending compaction requires ModernGL ComputeShader.run_indirect")
        material_table = world.bridge.shadow_typed_tables["material_table"]
        pending_program["cell_grid_size"].value = (world.width, world.height)
        pending_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        pending_program["tile_size"].value = int(world.active.tile_size)
        pending_program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
        pending_program["source_workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        pending_program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        material_tex.use(location=0)
        displaced_tex.use(location=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
        resources.affected_tile_flags.bind_to_storage_buffer(binding=5)
        pending_program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _compact_active_tiles(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        workgroups_per_tile: int = 1,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        compact_workgroups_per_tile = int(max(1, workgroups_per_tile))
        if self._active_scheduler_gpu_authoritative(world):
            bridge = world.bridge
            bridge._ensure_active_scheduler_programs()
            bridge._refresh_active_chunks_and_meta(world, read_meta=False)
            compact_program = self.programs["compact_active_tiles_from_chunks"]
            if not hasattr(compact_program, "run_indirect"):
                raise RuntimeError("GPU liquid active chunk compaction requires ModernGL ComputeShader.run_indirect")
            compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
            compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
            resources.active_tile_count.bind_to_storage_buffer(binding=0)
            resources.active_tile_list.bind_to_storage_buffer(binding=1)
            resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
            bridge.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
            bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
            compact_program.run_indirect(bridge.buffers["active_chunk_dispatch_args"])
        else:
            tile_count = int(world.active.tile_width * world.active.tile_height)
            compact_program = self.programs["compact_active_tiles"]
            compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
            resources.active_tile_tex.use(location=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=0)
            resources.active_tile_list.bind_to_storage_buffer(binding=1)
            resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            compact_program.run((tile_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)

    def _run_active_tile_indirect(self, program: Any, resources: GPULiquidResources, pass_name: str) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU liquid {pass_name} requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.active_tile_dispatch_args)

    def _run_tile_solve(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["tile_solve"]
        ctx = world.bridge.ctx
        assert ctx is not None
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid active tile solve requires ModernGL ComputeShader.run_indirect")
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_in.use(location=0)
        resources.phase_in.use(location=1)
        resources.flags_in.use(location=2)
        resources.timer_in.use(location=3)
        resources.temp_in.use(location=4)
        resources.integrity_in.use(location=5)
        resources.velocity_in.use(location=6)
        resources.entity_in.use(location=8)
        resources.displaced_in.use(location=9)
        resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
        resources.material_out.bind_to_image(0, read=False, write=True)
        resources.phase_out.bind_to_image(1, read=False, write=True)
        resources.flags_out.bind_to_image(2, read=False, write=True)
        resources.timer_out.bind_to_image(3, read=False, write=True)
        resources.temp_out.bind_to_image(4, read=False, write=True)
        resources.integrity_out.bind_to_image(5, read=False, write=True)
        resources.velocity_out.bind_to_image(6, read=False, write=True)
        program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _run_seam_pass(
        self,
        program_name: str,
        world: "WorldEngine",
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        active_tile_tex: Any,
        *,
        boundary_dispatch: bool = False,
    ) -> None:
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["use_boundary_dispatch"].value = bool(boundary_dispatch)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        active_tile_tex.use(location=7)
        self.resources.entity_in.use(location=8)
        self.resources.displaced_in.use(location=9)
        self.resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources = self.resources
        assert resources is not None
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        if boundary_dispatch:
            if not hasattr(program, "run_indirect"):
                raise RuntimeError(f"GPU liquid {program_name} requires ModernGL ComputeShader.run_indirect")
            program.run_indirect(resources.affected_tile_dispatch_args)
        else:
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
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
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
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        self._run_active_tile_indirect(program, resources, program_name)
        self._sync_compute_writes(ctx)

    def _run_copy_core_state(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> None:
        program = self.programs["copy_core_state"]
        ctx = world.bridge.ctx
        assert ctx is not None
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "copy core state")
        else:
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
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
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
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "copy with pending")
        else:
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
        dirty_affected_dispatch = self._formal_gpu_frame(world)
        if dirty_affected_dispatch:
            self._build_placeholder_dirty_affected_tile_dispatch(
                world,
                resources,
                material_tex=read_resources[0],
                displaced_tex=displaced_in,
            )
        material_table = world.bridge.shadow_typed_tables["material_table"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
        program["placeholder_claim_epoch"].value = int(self._next_placeholder_claim_epoch(resources, world))
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
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        if dirty_affected_dispatch:
            resources.affected_tile_count.bind_to_storage_buffer(binding=2)
            resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        else:
            resources.active_tile_count.bind_to_storage_buffer(binding=2)
            resources.active_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_flags.bind_to_storage_buffer(binding=4)
        resources.placeholder_target_claims.bind_to_storage_buffer(binding=5)
        write_resources[0].bind_to_image(0)
        write_resources[1].bind_to_image(1)
        write_resources[2].bind_to_image(2)
        write_resources[3].bind_to_image(3)
        write_resources[4].bind_to_image(4)
        write_resources[5].bind_to_image(5)
        write_resources[6].bind_to_image(6)
        displaced_out.bind_to_image(7)
        if dirty_affected_dispatch:
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("GPU liquid placeholder displacement requires ModernGL ComputeShader.run_indirect")
            program.run_indirect(resources.affected_tile_dispatch_args)
        else:
            self._run_active_tile_indirect(program, resources, "placeholder displacement")
        self._sync_compute_writes(ctx)

    def _run_cleanup_runtime(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["cleanup_runtime"]
        ctx = world.bridge.ctx
        assert ctx is not None
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        resources.material_pre.use(location=0)
        resources.phase_pre.use(location=1)
        resources.material_in.use(location=2)
        resources.phase_in.use(location=3)
        resources.island_in.use(location=4)
        resources.entity_in.use(location=5)
        resources.displaced_out.use(location=6)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        resources.island_out.bind_to_image(0, read=False, write=True)
        resources.entity_out.bind_to_image(1, read=False, write=True)
        resources.displaced_in.bind_to_image(2, read=False, write=True)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "runtime cleanup")
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(ctx)

    def _run_liquid_intent_pass(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["liquid_flow_intent"]
        ctx = world.bridge.ctx
        assert ctx is not None
        target_texture = resources.liquid_flow_intent
        if self._formal_gpu_frame(world):
            bridge = world.bridge
            bridge.ensure_world_resources(world)
            if not bridge.enabled or bridge.ctx is None:
                raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for liquid flow intent")
            target_texture = bridge.textures["liquid_flow_intent"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_in.use(location=0)
        resources.phase_in.use(location=1)
        resources.velocity_in.use(location=2)
        resources.active_tile_tex.use(location=3)
        resources.entity_in.use(location=4)
        resources.displaced_in.use(location=5)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        target_texture.bind_to_image(0, read=False, write=True)
        self._run_active_tile_indirect(program, resources, "flow intent")
        self._sync_compute_writes(ctx)
        if self._formal_gpu_frame(world):
            world.bridge.mark_gpu_authoritative("liquid_flow_intent")

    def _download_outputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        use_in: bool = False,
        velocity_use_out: bool = False,
    ) -> None:
        material = resources.material_in if use_in else resources.material_out
        phase = resources.phase_in if use_in else resources.phase_out
        flags = resources.flags_in if use_in else resources.flags_out
        timer = resources.timer_in if use_in else resources.timer_out
        temp = resources.temp_in if use_in else resources.temp_out
        integrity = resources.integrity_in if use_in else resources.integrity_out
        velocity = resources.velocity_out if velocity_use_out else (resources.velocity_in if use_in else resources.velocity_out)
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
            | getattr(ctx, "FRAMEBUFFER_BARRIER_BIT", 0)
            | getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(ctx, "BUFFER_UPDATE_BARRIER_BIT", 0),
        )
