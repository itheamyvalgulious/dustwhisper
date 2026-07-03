from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE, pack_island_runtime_upload
from oracle_game.types import Phase


LOCAL_SIZE = 8
POWDER_RESERVATION_LOCAL_SIZE = 64
ISLAND_RESERVATION_LINEAR_LOCAL_SIZE = 256
ACTIVE_TILE_WORKGROUP_AXIS = 4
ACTIVE_TILE_WORKGROUPS_PER_TILE = ACTIVE_TILE_WORKGROUP_AXIS * ACTIVE_TILE_WORKGROUP_AXIS
MAX_MATERIALS = 256
MAX_ISLAND_DDA_STEP = 4
INDEX_EMPTY = 2147483647
FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING = 1
FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING = 2
FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION = 4
FALLING_ISLAND_INDEX_CLEAR_SOURCE = 8
FALLING_ISLAND_INDEX_CLEAR_APPLY = (
    FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING | FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING
)

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
    active_tile_list: Any
    active_tile_count: Any
    active_tile_dispatch_args: Any
    island_materialization_candidate_tile_list: Any
    island_materialization_candidate_tile_count: Any
    island_materialization_candidate_dispatch_args: Any
    powder_apply_tile_flags: Any
    powder_target_tex: Any
    powder_target_winner: Any
    powder_apply_incoming: Any
    powder_apply_outgoing: Any
    powder_reservations: Any
    powder_reservation_count: Any
    powder_reservation_dispatch_args: Any
    island_reservations: Any
    island_reservation_count: Any
    island_runtime_dispatch_args: Any
    island_apply_incoming: Any
    island_apply_outgoing: Any
    island_materialization_index: Any
    island_reservation_source_index: Any
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
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    def _profile_enabled(self, world: "WorldEngine") -> bool:
        return bool(getattr(world, "profile_passes_enabled", False))

    def _reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    def reset_pass_profile(self) -> None:
        self._reset_pass_profile()

    @contextmanager
    def _profile_pass(self, world: "WorldEngine", name: str):
        if not self._profile_enabled(world):
            yield
            return
        ctx = world.bridge.ctx if bool(getattr(world, "profile_passes_sync", False)) else None
        if ctx is not None:
            ctx.finish()
        start = time.perf_counter()
        try:
            yield
        finally:
            if ctx is not None:
                ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            entry = {"name": str(name), "cpu_ms": elapsed_ms, "gpu_ms": elapsed_ms if ctx is not None else None}
            self.last_pass_profile["passes"].append(entry)
            summary = self.last_pass_profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
            summary["count"] += 1
            summary["cpu_ms"] += elapsed_ms
            if ctx is not None:
                summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

    def available(self, world: "WorldEngine") -> bool:
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        return bool(world.bridge.enabled and world.bridge.ctx is not None and world.bridge.ctx.version_code >= 430)

    def step(self, world: "WorldEngine", dt: float, *, solve_tile_mask: np.ndarray) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._reset_pass_profile()
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "powder_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "powder_load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        with self._profile_pass(world, "powder_targets"):
            self._run_powder_targets(world, resources, group_x, group_y, dt)
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
            self.resources.active_tile_list,
            self.resources.active_tile_count,
            self.resources.active_tile_dispatch_args,
            self.resources.island_materialization_candidate_tile_list,
            self.resources.island_materialization_candidate_tile_count,
            self.resources.island_materialization_candidate_dispatch_args,
            self.resources.powder_apply_tile_flags,
            self.resources.powder_target_tex,
            self.resources.powder_target_winner,
            self.resources.powder_apply_incoming,
            self.resources.powder_apply_outgoing,
            self.resources.powder_reservations,
            self.resources.powder_reservation_count,
            self.resources.powder_reservation_dispatch_args,
            self.resources.island_reservations,
            self.resources.island_reservation_count,
            self.resources.island_runtime_dispatch_args,
            self.resources.island_apply_incoming,
            self.resources.island_apply_outgoing,
            self.resources.island_materialization_index,
            self.resources.island_reservation_source_index,
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
        tile_count = max(1, int(world.active.tile_width * world.active.tile_height))
        active_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        active_tile_count = ctx.buffer(reserve=4, dynamic=True)
        active_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        island_materialization_candidate_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        island_materialization_candidate_tile_count = ctx.buffer(reserve=4, dynamic=True)
        island_materialization_candidate_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        powder_apply_tile_flags = ctx.buffer(reserve=max(4, tile_count * 4), dynamic=True)
        powder_target_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        cell_count = int(world.width * world.height)
        powder_target_winner = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
        powder_apply_incoming = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
        powder_apply_outgoing = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
        powder_reservation_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        island_runtime_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        island_apply_incoming = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
        island_apply_outgoing = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
        island_materialization_index = ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True)
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
            active_tile_list=active_tile_list,
            active_tile_count=active_tile_count,
            active_tile_dispatch_args=active_tile_dispatch_args,
            island_materialization_candidate_tile_list=island_materialization_candidate_tile_list,
            island_materialization_candidate_tile_count=island_materialization_candidate_tile_count,
            island_materialization_candidate_dispatch_args=island_materialization_candidate_dispatch_args,
            powder_apply_tile_flags=powder_apply_tile_flags,
            powder_target_tex=powder_target_tex,
            powder_target_winner=powder_target_winner,
            powder_apply_incoming=powder_apply_incoming,
            powder_apply_outgoing=powder_apply_outgoing,
            powder_reservations=ctx.buffer(reserve=4, dynamic=True),
            powder_reservation_count=ctx.buffer(reserve=4, dynamic=True),
            powder_reservation_dispatch_args=powder_reservation_dispatch_args,
            island_reservations=ctx.buffer(reserve=4, dynamic=True),
            island_reservation_count=ctx.buffer(reserve=4, dynamic=True),
            island_runtime_dispatch_args=island_runtime_dispatch_args,
            island_apply_incoming=island_apply_incoming,
            island_apply_outgoing=island_apply_outgoing,
            island_materialization_index=island_materialization_index,
            island_reservation_source_index=ctx.buffer(reserve=4, dynamic=True),
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
        self.programs["build_falling_island_materialization_candidate_dispatch"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_falling_island;
            uniform uint workgroups_per_tile;

            layout(binding=0) uniform sampler2D phase_tex;
            layout(binding=1) uniform sampler2D island_id_tex;

            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=2) buffer CandidateTileCountBuffer {{
                uint candidate_tile_count[];
            }};
            layout(std430, binding=3) buffer CandidateTileListBuffer {{
                ivec2 candidate_tile_list[];
            }};
            layout(std430, binding=4) buffer CandidateDispatchArgsBuffer {{
                uint candidate_dispatch_args[];
            }};

            shared uint tile_has_candidate;

            bool candidate_cell(ivec2 cell) {{
                int phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                int island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                return phase_id == phase_falling_island || island_id > 0;
            }}

            void main() {{
                uint active_tile_index = gl_WorkGroupID.x;
                if (active_tile_index >= active_tile_count[0]) {{
                    return;
                }}
                if (gl_LocalInvocationIndex == 0u) {{
                    tile_has_candidate = 0u;
                }}
                barrier();

                ivec2 tile = active_tile_list[int(active_tile_index)];
                if (tile.x >= 0 && tile.y >= 0 && tile.x < tile_grid_size.x && tile.y < tile_grid_size.y) {{
                    ivec2 tile_origin = tile * tile_size;
                    ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                    ivec2 tile_extent = max(tile_end - tile_origin, ivec2(0));
                    int cell_count = tile_extent.x * tile_extent.y;
                    for (int offset = int(gl_LocalInvocationIndex); offset < cell_count; offset += 256) {{
                        ivec2 local_cell = ivec2(offset % tile_extent.x, offset / tile_extent.x);
                        if (candidate_cell(tile_origin + local_cell)) {{
                            atomicExchange(tile_has_candidate, 1u);
                            break;
                        }}
                    }}
                }}
                barrier();

                if (gl_LocalInvocationIndex == 0u && tile_has_candidate != 0u) {{
                    uint slot = atomicAdd(candidate_tile_count[0], 1u);
                    candidate_tile_list[int(slot)] = tile;
                    atomicMax(candidate_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
                }}
            }}
            """
        )
        self.programs["build_powder_reservation_dispatch"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform int invocations_per_group;
            uniform int max_reservation_count;
            layout(std430, binding=6) readonly buffer PowderReservationCount {
                int powder_reservation_count;
            };
            layout(std430, binding=7) buffer PowderReservationDispatchArgs {
                uint dispatch_args[];
            };
            void main() {
                int group_size = max(1, invocations_per_group);
                int count = clamp(powder_reservation_count, 0, max_reservation_count);
                dispatch_args[0] = uint(max(1, (count + group_size - 1) / group_size));
                dispatch_args[1] = 1u;
                dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["build_island_runtime_dispatch"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform int invocations_per_group;
            uniform int runtime_capacity;

            layout(std430, binding=6) readonly buffer IslandRuntimeCount {
                int runtime_count;
            };
            layout(std430, binding=7) buffer IslandRuntimeDispatchArgs {
                uint dispatch_args[];
            };

            void main() {
                int group_size = max(1, invocations_per_group);
                int count = clamp(runtime_count, 0, max(runtime_capacity, 0));
                dispatch_args[0] = uint(max(1, (count + group_size - 1) / group_size));
                dispatch_args[1] = 1u;
                dispatch_args[2] = 1u;
            }
            """
        )
        self.programs["clear_powder_affected_tile_dispatch"] = ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;
            layout(std430, binding=0) buffer PowderAffectedTileFlags {
                uint affected_tile_flags[];
            };
            layout(std430, binding=1) buffer ActiveTileCountBuffer {
                uint active_tile_count[];
            };
            layout(std430, binding=2) buffer ActiveTileDispatchArgsBuffer {
                uint active_tile_dispatch_args[];
            };
            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index == 0u) {
                    active_tile_count[0] = 0u;
                    active_tile_dispatch_args[0] = 0u;
                    active_tile_dispatch_args[1] = 1u;
                    active_tile_dispatch_args[2] = 1u;
                }
                if (index < uint(tile_count)) {
                    affected_tile_flags[index] = 0u;
                }
            }
            """
        )
        self.programs["build_powder_apply_dispatch"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform uint workgroups_per_tile;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) readonly buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer PowderAffectedTileFlags {{
                uint affected_tile_flags[];
            }};
            layout(std430, binding=3) buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=4) buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=5) buffer ActiveTileDispatchArgsBuffer {{
                uint active_tile_dispatch_args[];
            }};

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
            }}

            bool is_moving(PowderReservation reservation) {{
                return (
                    (reservation.resolve_state == {POWDER_RESOLVE_DDA} || reservation.resolve_state == {POWDER_RESOLVE_FALLBACK})
                    && !same_cell(reservation.source_xy, reservation.resolved_target_xy)
                );
            }}

            void append_tile_for_cell(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                uint previous = atomicExchange(affected_tile_flags[tile_index], 1u);
                if (previous != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(active_tile_count[0], 1u);
                active_tile_list[int(slot)] = tile;
                atomicMax(active_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                if (reservation.resolve_state == {POWDER_RESOLVE_STALE}) {{
                    return;
                }}
                append_tile_for_cell(reservation.source_xy);
                if (is_moving(reservation)) {{
                    append_tile_for_cell(reservation.resolved_target_xy);
                }}
            }}
            """
        )
        self.programs["build_falling_island_apply_dispatch"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform uint workgroups_per_tile;
            uniform int operation;

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
            layout(std430, binding=1) readonly buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=2) buffer FallingIslandAffectedTileFlags {{
                uint affected_tile_flags[];
            }};
            layout(std430, binding=3) buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=4) buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=5) buffer ActiveTileDispatchArgsBuffer {{
                uint active_tile_dispatch_args[];
            }};

            int active_reservation_count() {{
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool valid_reservation(FallingIslandReservation reservation) {{
                return reservation.island_id > 0 && reservation.resolve_state != {ISLAND_RESOLVE_STALE};
            }}

            bool moving_reservation(FallingIslandReservation reservation) {{
                return valid_reservation(reservation)
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool settling_reservation(FallingIslandReservation reservation) {{
                return valid_reservation(reservation)
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            void append_tile(ivec2 tile) {{
                if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                    return;
                }}
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                uint previous = atomicExchange(affected_tile_flags[tile_index], 1u);
                if (previous != 0u) {{
                    return;
                }}
                uint slot = atomicAdd(active_tile_count[0], 1u);
                active_tile_list[int(slot)] = tile;
                atomicMax(active_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
            }}

            void append_bbox(ivec4 bbox) {{
                int x0 = clamp(min(bbox.x, bbox.z), 0, cell_grid_size.x);
                int y0 = clamp(min(bbox.y, bbox.w), 0, cell_grid_size.y);
                int x1 = clamp(max(bbox.x, bbox.z), 0, cell_grid_size.x);
                int y1 = clamp(max(bbox.y, bbox.w), 0, cell_grid_size.y);
                if (x1 <= x0 || y1 <= y0) {{
                    return;
                }}
                int tile_x0 = clamp(x0 / tile_size, 0, tile_grid_size.x - 1);
                int tile_y0 = clamp(y0 / tile_size, 0, tile_grid_size.y - 1);
                int tile_x1 = clamp((x1 + tile_size - 1) / tile_size, 0, tile_grid_size.x);
                int tile_y1 = clamp((y1 + tile_size - 1) / tile_size, 0, tile_grid_size.y);
                for (int tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {{
                    for (int tile_x = tile_x0; tile_x < tile_x1; ++tile_x) {{
                        append_tile(ivec2(tile_x, tile_y));
                    }}
                }}
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= active_reservation_count()) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[index];
                ivec4 source_bbox = ivec4(
                    reservation.bbox_x,
                    reservation.bbox_y,
                    reservation.bbox_z,
                    reservation.bbox_w
                );
                if (operation == 0) {{
                    if (!moving_reservation(reservation)) {{
                        return;
                    }}
                    append_bbox(source_bbox);
                    append_bbox(source_bbox + ivec4(
                        reservation.resolved_dx,
                        reservation.resolved_dy,
                        reservation.resolved_dx,
                        reservation.resolved_dy
                    ));
                    return;
                }}
                if (operation == 2) {{
                    if (!valid_reservation(reservation)) {{
                        return;
                    }}
                    ivec4 target_bbox = source_bbox + ivec4(
                        reservation.target_dx,
                        reservation.target_dy,
                        reservation.target_dx,
                        reservation.target_dy
                    );
                    append_bbox(ivec4(
                        min(source_bbox.x, target_bbox.x),
                        min(source_bbox.y, target_bbox.y),
                        max(source_bbox.z, target_bbox.z),
                        max(source_bbox.w, target_bbox.w)
                    ));
                    return;
                }}
                if (settling_reservation(reservation)) {{
                    append_bbox(source_bbox);
                }}
            }}
            """
        )
        self.programs["clear_powder_target_winners_for_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) readonly buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer PowderTargetWinners {{
                int target_winners[];
            }};

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            void clear_cell(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return;
                }}
                target_winners[cell.y * cell_grid_size.x + cell.x] = 2147483647;
            }}

            void clear_fallback_candidates(ivec2 source) {{
                clear_cell(source + ivec2(0, 1));
                clear_cell(source + ivec2(-1, 1));
                clear_cell(source + ivec2(1, 1));
                clear_cell(source + ivec2(0, -1));
                clear_cell(source + ivec2(-1, -1));
                clear_cell(source + ivec2(1, -1));
            }}

            void clear_path_cells(ivec2 source, ivec2 target) {{
                ivec2 desired = target - source;
                if (desired.x == 0 && desired.y == 0) {{
                    return;
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
                    clear_cell(source + current);
                }}
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                clear_cell(reservation.source_xy);
                clear_cell(reservation.desired_target_xy);
                clear_cell(reservation.reserved_target_xy);
                clear_path_cells(reservation.source_xy, reservation.reserved_target_xy);
                clear_fallback_candidates(reservation.source_xy);
            }}
            """
        )
        self.programs["clear_powder_apply_index_for_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

            struct PowderReservation {{
                ivec2 source_xy;
                ivec2 desired_target_xy;
                ivec2 reserved_target_xy;
                ivec2 resolved_target_xy;
                vec2 velocity_xy;
                int material_id;
                int resolve_state;
            }};

            layout(std430, binding=0) readonly buffer PowderReservations {{
                PowderReservation reservations[];
            }};
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer PowderTargetWinners {{
                int target_winners[];
            }};
            layout(std430, binding=3) buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=4) buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            void clear_cell(ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                target_winners[cell_index] = 2147483647;
                apply_incoming[cell_index] = -1;
                apply_outgoing[cell_index] = -1;
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                clear_cell(reservation.source_xy);
                clear_cell(reservation.resolved_target_xy);
            }}
            """
        )
        self.programs["clear_powder_apply_index_for_active_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;

            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=2) buffer PowderTargetWinners {{
                int target_winners[];
            }};
            layout(std430, binding=3) buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=4) buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 cell;
                if (!active_dispatch_cell(cell)) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                target_winners[cell_index] = 2147483647;
                apply_incoming[cell_index] = -1;
                apply_outgoing[cell_index] = -1;
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
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D velocity_tex;
            layout(binding=3) uniform sampler2D flow_tex;
            layout(binding=4) uniform sampler2D active_tile_tex;
            layout(rg32f, binding=5) writeonly uniform image2D velocity_out_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (material_id > 0) {{
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;
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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
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
        self.programs["load_bridge_integrate_inputs"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;

            layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D material_img;
            layout(rg32f, binding=1) writeonly uniform image2D velocity_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
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
                imageStore(material_img, gid, vec4(float(word0 & 0xFFFFu), 0.0, 0.0, 0.0));
                imageStore(velocity_img, gid, vec4(unpackHalf2x16(bridge_cell_core[word_index + 1]), 0.0, 0.0));
            }}
            """
        )
        self.programs["load_bridge_cell_aux"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool use_active_tile_dispatch;
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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(r32f, binding=0) writeonly uniform image2D island_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform bool velocity_out_active_only;
            uniform bool use_active_tile_dispatch;
            uniform bool use_powder_apply_touch_sources;
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
            layout(binding=10) uniform sampler2D velocity_fallback_tex;
            layout(binding=11) uniform sampler2D active_tile_tex;
            layout(binding=12) uniform sampler2D material_input_tex;
            layout(binding=13) uniform sampler2D phase_input_tex;
            layout(binding=14) uniform sampler2D flags_input_tex;
            layout(binding=15) uniform sampler2D velocity_input_tex;
            layout(binding=16) uniform sampler2D temp_input_tex;
            layout(binding=17) uniform sampler2D timer_input_tex;
            layout(binding=18) uniform sampler2D integrity_input_tex;
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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=6) readonly buffer PowderApplyIncoming {{
                int powder_apply_incoming[];
            }};
            layout(std430, binding=7) readonly buffer PowderApplyOutgoing {{
                int powder_apply_outgoing[];
            }};

            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
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
                bool use_output_core = true;
                if (use_powder_apply_touch_sources) {{
                    use_output_core = powder_apply_incoming[cell_index] >= 0 || powder_apply_outgoing[cell_index] >= 0;
                }}
                float material_value;
                float phase_value;
                float flags_value;
                vec2 velocity;
                float temperature;
                vec4 timer;
                float integrity_value;
                if (use_output_core) {{
                    material_value = texelFetch(material_tex, gid, 0).x;
                    phase_value = texelFetch(phase_tex, gid, 0).x;
                    flags_value = texelFetch(flags_tex, gid, 0).x;
                    velocity = texelFetch(velocity_tex, gid, 0).xy;
                    temperature = texelFetch(temp_tex, gid, 0).x;
                    timer = texelFetch(timer_tex, gid, 0);
                    integrity_value = texelFetch(integrity_tex, gid, 0).x;
                }} else {{
                    material_value = texelFetch(material_input_tex, gid, 0).x;
                    phase_value = texelFetch(phase_input_tex, gid, 0).x;
                    flags_value = texelFetch(flags_input_tex, gid, 0).x;
                    velocity = texelFetch(velocity_input_tex, gid, 0).xy;
                    temperature = texelFetch(temp_input_tex, gid, 0).x;
                    timer = texelFetch(timer_input_tex, gid, 0);
                    integrity_value = texelFetch(integrity_input_tex, gid, 0).x;
                }}
                uint material = uint(clamp(round(material_value), 0.0, 65535.0));
                uint phase = uint(clamp(round(phase_value), 0.0, 255.0));
                uint flags = uint(clamp(round(flags_value), 0.0, 255.0));
                if (velocity_out_active_only && !solve_cell_active(gid)) {{
                    velocity = texelFetch(velocity_fallback_tex, gid, 0).xy;
                }}
                uint integrity = uint(clamp(round(integrity_value), 0.0, 65535.0));
                int island = int(round(texelFetch(island_tex, gid, 0).x));
                int entity = int(round(texelFetch(entity_tex, gid, 0).x));
                int displaced = int(round(texelFetch(displaced_tex, gid, 0).x));
                int word_index = cell_index * 5;
                bridge_cell_core[word_index] = material | (phase << 16u) | (flags << 24u);
                bridge_cell_core[word_index + 1] = packHalf2x16(velocity);
                bridge_cell_core[word_index + 2] = floatBitsToUint(temperature);
                bridge_cell_core[word_index + 3] = pack_timer(timer);
                bridge_cell_core[word_index + 4] = integrity;
                bridge_island_id[cell_index] = island;
                bridge_entity_id[cell_index] = entity;
                bridge_displaced[cell_index] = displaced;
                imageStore(bridge_material_img, gid, vec4(float(material), 0.0, 0.0, 0.0));
            }}
            """
        )
        self.programs["publish_bridge_velocity_word"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            layout(binding=0) uniform sampler2D velocity_tex;
            layout(std430, binding=0) writeonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
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
                bridge_cell_core[cell_index * 5 + 1] = packHalf2x16(texelFetch(velocity_tex, gid, 0).xy);
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
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform bool use_bridge_authoritative_blockers;
            uniform bool use_liquid_flow_intent;
            uniform float dt;
            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=2) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=8) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=9) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=10) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=11) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced_material[];
            }};
            layout(binding=1) uniform sampler2D material_tex;
            layout(binding=2) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D velocity_tex;
            layout(binding=4) uniform sampler2D active_tile_tex;
            layout(binding=6) uniform sampler2D entity_id_tex;
            layout(binding=7) uniform sampler2D displaced_tex;
            layout(binding=8) uniform sampler2D liquid_flow_intent_tex;
            layout(rgba32f, binding=5) writeonly uniform image2D powder_target_img;

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            int material_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int(bridge_cell_core[cell_index(cell) * 5] & 0xFFFFu);
                }}
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int((bridge_cell_core[cell_index(cell) * 5] >> 16u) & 0xFFu);
                }}
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            bool entity_blocks(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return bridge_entity_id[cell_index(cell)] != 0;
                }}
                return texelFetch(entity_id_tex, cell, 0).x > 0.5;
            }}

            bool displaced_blocks(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return bridge_displaced_material[cell_index(cell)] != 0;
                }}
                return texelFetch(displaced_tex, cell, 0).x > 0.5;
            }}

            bool blocks_dda_target(ivec2 cell) {{
                int phase_id = phase_at(cell);
                return material_at(cell) > 0
                    || phase_id == phase_falling_island
                    || entity_blocks(cell)
                    || displaced_blocks(cell);
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
                    if (blocks_dda_target(sample_cell)) {{
                        break;
                    }}
                    furthest = sample_cell;
                }}
                return furthest;
            }}

            void main() {{
                ivec2 gid;
                if (!active_dispatch_cell(gid)) {{
                    return;
                }}
                ivec2 target = gid;
                ivec2 raw_target = gid;
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                int phase_id = int(texelFetch(phase_tex, gid, 0).x + 0.5);
                if (material_id > 0 && (phase_id == phase_powder || phase_id == phase_liquid)) {{
                    int max_step = int(params[material_id].x + 0.5);
                    vec2 velocity = texelFetch(velocity_tex, gid, 0).xy;
                    if (phase_id == phase_liquid && use_liquid_flow_intent) {{
                        velocity += texelFetch(liquid_flow_intent_tex, gid, 0).xy;
                    }}
                    vec2 frame_delta = velocity * dt;
                    int desired_dx = int(clamp(round(frame_delta.x), float(-max_step), float(max_step)));
                    int desired_dy = int(clamp(round(frame_delta.y), float(-max_step), float(max_step)));
                    raw_target = gid + ivec2(desired_dx, desired_dy);
                    target = dda_target(gid, ivec2(desired_dx, desired_dy));
                }}
                imageStore(
                    powder_target_img,
                    gid,
                    vec4(float(target.x), float(target.y), float(raw_target.x), float(raw_target.y))
                );
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
            uniform bool use_island_count_buffer;
            uniform bool use_bridge_authoritative_state;
            uniform float dt;
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
            layout(std430, binding=4) readonly buffer IslandCountBuffer {{
                int island_count_buffer;
            }};
            layout(std430, binding=5) readonly buffer MaterialParamBuffer {{
                vec4 material_params[];
            }};
            layout(std430, binding=7) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=8) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};

            int active_island_count() {{
                if (use_island_count_buffer) {{
                    return clamp(island_count_buffer, 0, max(island_count, 0));
                }}
                return max(island_count, 0);
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            int material_at(ivec2 cell) {{
                if (use_bridge_authoritative_state) {{
                    return int(bridge_cell_core[cell_index(cell) * 5] & 0xFFFFu);
                }}
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int island_at(ivec2 cell) {{
                if (use_bridge_authoritative_state) {{
                    return bridge_island_id[cell_index(cell)];
                }}
                return int(texelFetch(island_id_tex, cell, 0).x + 0.5);
            }}

            bool bbox_in_bounds(ivec4 island_bbox) {{
                return island_bbox.x >= 0
                    && island_bbox.y >= 0
                    && island_bbox.z <= cell_grid_size.x
                    && island_bbox.w <= cell_grid_size.y
                    && island_bbox.z > island_bbox.x
                    && island_bbox.w > island_bbox.y;
            }}

            bool has_source_cell(ivec4 island_bbox, int island_id) {{
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (island_at(cell) == island_id && material_at(cell) > 0) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}

            int clamp_material(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}

            int gravity_fallback_dy(ivec4 island_bbox, int island_id, float velocity_y) {{
                float gravity_sum = 0.0;
                int material_count = 0;
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (island_at(cell) != island_id) {{
                            continue;
                        }}
                        int material_id = material_at(cell);
                        if (material_id <= 0) {{
                            continue;
                        }}
                        gravity_sum += material_params[clamp_material(material_id)].y;
                        material_count += 1;
                    }}
                }}
                if (material_count <= 0) {{
                    return 0;
                }}
                float mean_gravity = gravity_sum / float(material_count);
                if (abs(mean_gravity) > 1.0e-6) {{
                    return mean_gravity > 0.0 ? 1 : -1;
                }}
                if (abs(velocity_y) <= 1.0e-6) {{
                    return 0;
                }}
                return velocity_y > 0.0 ? 1 : -1;
            }}

            bool can_shift_delta(ivec4 island_bbox, int island_id, int dx, int dy) {{
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        if (island_at(ivec2(x, y)) != island_id) {{
                            continue;
                        }}
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= cell_grid_size.x || ny >= cell_grid_size.y) {{
                            return false;
                        }}
                        int material_id = material_at(ivec2(nx, ny));
                        if (material_id == 0) {{
                            continue;
                        }}
                        if (island_at(ivec2(nx, ny)) == island_id) {{
                            continue;
                        }}
                        return false;
                    }}
                }}
                return true;
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                if (gid >= active_island_count()) {{
                    return;
                }}
                int island_id = island_ids[gid];
                ivec4 island_bbox = island_bboxes[gid];
                if (island_id <= 0 || !bbox_in_bounds(island_bbox) || !has_source_cell(island_bbox, island_id)) {{
                    island_motion[gid] = vec4(0.0);
                    island_shifts[gid] = ivec4(0);
                    island_ids[gid] = 0;
                    return;
                }}
                vec4 motion = island_motion[gid];
                float total_x = motion.z + motion.x * dt;
                float total_y = motion.w + motion.y * dt;
                int target_dx = int(clamp(round(total_x), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                int target_dy = int(clamp(round(total_y), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                float residual_x = total_x - float(target_dx);
                float residual_y = total_y - float(target_dy);
                if (target_dx == 0 && target_dy == 0) {{
                    int fallback_dy = gravity_fallback_dy(island_bbox, island_id, motion.y);
                    if (fallback_dy != 0) {{
                        target_dy = fallback_dy;
                        residual_y = 0.0;
                    }}
                }}
                island_motion[gid] = vec4(motion.x, motion.y, residual_x, residual_y);
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
        self.programs["plan_bridge_runtime_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int runtime_capacity;
            uniform bool use_bridge_authoritative_state;
            uniform float dt;
            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D island_id_tex;
            layout(std430, binding=0) readonly buffer BridgeIslandRuntimeWords {{
                int runtime_words[];
            }};
            layout(std430, binding=1) buffer IslandReservationWords {{
                ivec4 reservation_words[];
            }};
            layout(std430, binding=2) buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=3) readonly buffer BridgeIslandRuntimeCount {{
                int runtime_count;
            }};
            layout(std430, binding=4) readonly buffer MaterialParamBuffer {{
                vec4 material_params[];
            }};
            layout(std430, binding=7) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=8) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};

            int active_runtime_count() {{
                return clamp(runtime_count, 0, max(runtime_capacity, 0));
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            int material_at(ivec2 cell) {{
                if (use_bridge_authoritative_state) {{
                    return int(bridge_cell_core[cell_index(cell) * 5] & 0xFFFFu);
                }}
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int island_at(ivec2 cell) {{
                if (use_bridge_authoritative_state) {{
                    return bridge_island_id[cell_index(cell)];
                }}
                return int(texelFetch(island_id_tex, cell, 0).x + 0.5);
            }}

            bool bbox_in_bounds(ivec4 island_bbox) {{
                return island_bbox.x >= 0
                    && island_bbox.y >= 0
                    && island_bbox.z <= cell_grid_size.x
                    && island_bbox.w <= cell_grid_size.y
                    && island_bbox.z > island_bbox.x
                    && island_bbox.w > island_bbox.y;
            }}

            bool has_source_cell(ivec4 island_bbox, int island_id) {{
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (island_at(cell) == island_id && material_at(cell) > 0) {{
                            return true;
                        }}
                    }}
                }}
                return false;
            }}

            int clamp_material(int material_id) {{
                return clamp(material_id, 0, {MAX_MATERIALS - 1});
            }}

            int gravity_fallback_dy(ivec4 island_bbox, int island_id, float velocity_y) {{
                float gravity_sum = 0.0;
                int material_count = 0;
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        ivec2 cell = ivec2(x, y);
                        if (island_at(cell) != island_id) {{
                            continue;
                        }}
                        int material_id = material_at(cell);
                        if (material_id <= 0) {{
                            continue;
                        }}
                        gravity_sum += material_params[clamp_material(material_id)].y;
                        material_count += 1;
                    }}
                }}
                if (material_count <= 0) {{
                    return 0;
                }}
                float mean_gravity = gravity_sum / float(material_count);
                if (abs(mean_gravity) > 1.0e-6) {{
                    return mean_gravity > 0.0 ? 1 : -1;
                }}
                if (abs(velocity_y) <= 1.0e-6) {{
                    return 0;
                }}
                return velocity_y > 0.0 ? 1 : -1;
            }}

            bool can_shift_delta(ivec4 island_bbox, int island_id, int dx, int dy) {{
                for (int y = island_bbox.y; y < island_bbox.w; ++y) {{
                    for (int x = island_bbox.x; x < island_bbox.z; ++x) {{
                        if (island_at(ivec2(x, y)) != island_id) {{
                            continue;
                        }}
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= cell_grid_size.x || ny >= cell_grid_size.y) {{
                            return false;
                        }}
                        int material_id = material_at(ivec2(nx, ny));
                        if (material_id == 0) {{
                            continue;
                        }}
                        if (island_at(ivec2(nx, ny)) == island_id) {{
                            continue;
                        }}
                        return false;
                    }}
                }}
                return true;
            }}

            ivec4 runtime_bbox(int gid) {{
                int base = gid * {ISLAND_RUNTIME_DTYPE.itemsize // 4};
                return ivec4(
                    runtime_words[base + 1],
                    runtime_words[base + 2],
                    runtime_words[base + 3],
                    runtime_words[base + 4]
                );
            }}

            vec4 runtime_motion(int gid) {{
                int base = gid * {ISLAND_RUNTIME_DTYPE.itemsize // 4};
                return vec4(
                    intBitsToFloat(runtime_words[base + 9]),
                    intBitsToFloat(runtime_words[base + 10]) + 0.9,
                    intBitsToFloat(runtime_words[base + 11]),
                    intBitsToFloat(runtime_words[base + 12])
                );
            }}

            void write_reservation(int gid, int island_id, ivec4 bbox, vec4 motion, ivec4 shifts) {{
                int word_base = gid * 4;
                int resolve_state = {ISLAND_RESOLVE_BLOCKED};
                if (shifts.x != 0 || shifts.y != 0 || (shifts.z == 0 && shifts.w == 0)) {{
                    resolve_state = {ISLAND_RESOLVE_DIRECT};
                }}
                reservation_words[word_base] = ivec4(island_id, bbox.x, bbox.y, bbox.z);
                reservation_words[word_base + 1] = ivec4(
                    bbox.w,
                    floatBitsToInt(motion.x),
                    floatBitsToInt(motion.y),
                    floatBitsToInt(motion.z)
                );
                reservation_words[word_base + 2] = ivec4(
                    floatBitsToInt(motion.w),
                    shifts.z,
                    shifts.w,
                    shifts.x
                );
                reservation_words[word_base + 3] = ivec4(shifts.y, shifts.x, shifts.y, resolve_state);
            }}

            void write_empty_reservation(int gid) {{
                write_reservation(gid, 0, ivec4(0), vec4(0.0), ivec4(0));
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                int count = active_runtime_count();
                if (gid == 0) {{
                    reservation_count = count;
                }}
                if (gid >= count) {{
                    return;
                }}

                int base = gid * {ISLAND_RUNTIME_DTYPE.itemsize // 4};
                int island_id = runtime_words[base];
                ivec4 island_bbox = runtime_bbox(gid);
                if (island_id <= 0) {{
                    write_empty_reservation(gid);
                    return;
                }}
                if (!bbox_in_bounds(island_bbox) || !has_source_cell(island_bbox, island_id)) {{
                    write_reservation(gid, 0, island_bbox, vec4(0.0), ivec4(0));
                    return;
                }}

                vec4 motion = runtime_motion(gid);
                float total_x = motion.z + motion.x * dt;
                float total_y = motion.w + motion.y * dt;
                int target_dx = int(clamp(round(total_x), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                int target_dy = int(clamp(round(total_y), float(-{MAX_ISLAND_DDA_STEP}), float({MAX_ISLAND_DDA_STEP})));
                float residual_x = total_x - float(target_dx);
                float residual_y = total_y - float(target_dy);
                if (target_dx == 0 && target_dy == 0) {{
                    int fallback_dy = gravity_fallback_dy(island_bbox, island_id, motion.y);
                    if (fallback_dy != 0) {{
                        target_dy = fallback_dy;
                        residual_y = 0.0;
                    }}
                }}

                vec4 updated_motion = vec4(motion.x, motion.y, residual_x, residual_y);
                int current_x = 0;
                int current_y = 0;
                int furthest_x = 0;
                int furthest_y = 0;
                if (target_dx == 0 && target_dy == 0) {{
                    write_reservation(gid, island_id, island_bbox, updated_motion, ivec4(0, 0, target_dx, target_dy));
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
                write_reservation(gid, island_id, island_bbox, updated_motion, ivec4(furthest_x, furthest_y, target_dx, target_dy));
            }}
            """
        )
        self.programs["pack_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={ISLAND_RESERVATION_LINEAR_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int island_count;
            uniform bool use_island_count_buffer;

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
            layout(std430, binding=4) buffer IslandReservationWords {{
                ivec4 reservation_words[];
            }};
            layout(std430, binding=5) buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=6) readonly buffer IslandCountBuffer {{
                int island_count_buffer;
            }};

            int active_island_count() {{
                if (use_island_count_buffer) {{
                    return clamp(island_count_buffer, 0, max(island_count, 0));
                }}
                return max(island_count, 0);
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                int count = active_island_count();
                if (gid == 0) {{
                    reservation_count = count;
                }}
                if (gid >= count) {{
                    return;
                }}
                ivec4 bbox = island_bboxes[gid];
                vec4 motion = island_motion[gid];
                ivec4 shifts = island_shifts[gid];
                int word_base = gid * 4;
                int resolve_state = {ISLAND_RESOLVE_BLOCKED};
                if (shifts.x != 0 || shifts.y != 0 || (shifts.z == 0 && shifts.w == 0)) {{
                    resolve_state = {ISLAND_RESOLVE_DIRECT};
                }}
                reservation_words[word_base] = ivec4(island_ids[gid], bbox.x, bbox.y, bbox.z);
                reservation_words[word_base + 1] = ivec4(
                    bbox.w,
                    floatBitsToInt(motion.x),
                    floatBitsToInt(motion.y),
                    floatBitsToInt(motion.z)
                );
                reservation_words[word_base + 2] = ivec4(
                    floatBitsToInt(motion.w),
                    shifts.z,
                    shifts.w,
                    shifts.x
                );
                reservation_words[word_base + 3] = ivec4(shifts.y, shifts.x, shifts.y, resolve_state);
            }}
            """
        )
        self.programs["publish_falling_island_runtime"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;
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
            layout(std430, binding=3) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
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

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

            void store_runtime_word(int record_index, int word_offset, int value) {{
                runtime_words[record_index * {ISLAND_RUNTIME_DTYPE.itemsize // 4} + word_offset] = value;
            }}

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (gid >= count) {{
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
        self.programs["publish_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int reservation_capacity;

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

            layout(std430, binding=0) readonly buffer SourceReservations {{
                FallingIslandReservation source_reservations[];
            }};
            layout(std430, binding=1) readonly buffer SourceReservationCount {{
                int source_reservation_count;
            }};
            layout(std430, binding=2) writeonly buffer DestReservations {{
                FallingIslandReservation dest_reservations[];
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
            uniform int runtime_capacity;

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
            layout(std430, binding=4) readonly buffer BridgeIslandRuntimeCount {{
                int runtime_count;
            }};

            void main() {{
                int gid = int(gl_GlobalInvocationID.x);
                int count = clamp(runtime_count, 0, max(runtime_capacity, 0));
                if (gid >= count) {{
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
        self.programs["fill_falling_island_reservation_source_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

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
            layout(std430, binding=1) readonly buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=2) buffer IslandReservationSourceIndex {{
                int reservation_source_index[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D island_id_tex;

            int active_reservation_count() {{
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
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

            void main() {{
                int index = int(gl_WorkGroupID.x);
                if (index >= active_reservation_count()) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[index];
                if (reservation.island_id <= 0 || reservation.resolve_state == {ISLAND_RESOLVE_STALE}) {{
                    return;
                }}
                int x0 = clamp(min(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y0 = clamp(min(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int x1 = clamp(max(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y1 = clamp(max(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int bbox_width = x1 - x0;
                int bbox_height = y1 - y0;
                if (bbox_width <= 0 || bbox_height <= 0) {{
                    return;
                }}
                int bbox_area = bbox_width * bbox_height;
                int local_stride = int(gl_WorkGroupSize.x);
                for (int offset = int(gl_LocalInvocationID.x); offset < bbox_area; offset += local_stride) {{
                    ivec2 source = ivec2(x0 + (offset % bbox_width), y0 + (offset / bbox_width));
                    if (!source_matches(reservation, source)) {{
                        continue;
                    }}
                    atomicMin(reservation_source_index[source.y * cell_grid_size.x + source.x], index);
                }}
            }}
            """
        )
        self.programs["resolve_falling_island_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;

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
            layout(std430, binding=2) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
            }};
            layout(std430, binding=3) readonly buffer IslandReservationSourceIndex {{
                int reservation_source_index[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D island_id_tex;

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

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

            int reservation_source_index_for_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return -1;
                }}
                int index = reservation_source_index[cell_index(cell)];
                int count = active_reservation_count();
                if (index < 0 || index >= count) {{
                    return -1;
                }}
                if (!source_matches(reservations[index], cell)) {{
                    return -1;
                }}
                return index;
            }}

            bool reservation_clears_cell(FallingIslandReservation blocker, ivec2 cell) {{
                ivec2 blocker_shift = ivec2(blocker.reserved_dx, blocker.reserved_dy);
                if (blocker_shift.x == 0 && blocker_shift.y == 0) {{
                    return false;
                }}
                if (!source_matches(blocker, cell)) {{
                    return false;
                }}
                ivec2 shifted_cell = cell + blocker_shift;
                return shifted_cell.x != cell.x || shifted_cell.y != cell.y;
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
                        int target_material_id = int(texelFetch(material_tex, target, 0).x + 0.5);
                        if (target_material_id == 0) {{
                            continue;
                        }}
                        if (source_matches(reservation, target)) {{
                            continue;
                        }}
                        int blocker_index = reservation_source_index_for_cell(target);
                        if (blocker_index >= 0 && reservation_clears_cell(reservations[blocker_index], target)) {{
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

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (count <= 0 || index < 0 || index >= count) {{
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
            }}
            """
        )
        self.programs["generate_powder_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform float dt;

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
            layout(std430, binding=4) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=5) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=2) uniform sampler2D velocity_tex;
            layout(binding=3) uniform sampler2D active_tile_tex;
            layout(binding=4) uniform sampler2D powder_target_tex;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            int material_at(ivec2 cell) {{
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            void append_reservation(
                ivec2 source,
                ivec2 desired,
                ivec2 reserved,
                vec2 velocity,
                int material_id
            ) {{
                int index = atomicAdd(powder_reservation_count, 1);
                reservations[index].source_xy = source;
                reservations[index].desired_target_xy = desired;
                reservations[index].reserved_target_xy = reserved;
                reservations[index].resolved_target_xy = source;
                reservations[index].velocity_xy = velocity;
                reservations[index].material_id = material_id;
                reservations[index].resolve_state = {POWDER_RESOLVE_BLOCKED};
            }}

            void main() {{
                ivec2 source;
                if (dt < 0.0) {{
                    return;
                }}
                if (!active_dispatch_cell(source)) {{
                    return;
                }}
                int material_id = material_at(source);
                int phase_id = phase_at(source);
                if (material_id <= 0 || (phase_id != phase_powder && phase_id != phase_liquid)) {{
                    return;
                }}
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                vec4 target_payload = texelFetch(powder_target_tex, source, 0);
                ivec2 reserved = ivec2(round(target_payload.xy));
                ivec2 desired = ivec2(round(target_payload.zw));
                append_reservation(source, desired, reserved, velocity, material_id);
            }}
            """
        )
        self.programs["clear_powder_target_winners"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int cell_count;
            layout(std430, binding=0) buffer PowderTargetWinners {{
                int target_winners[];
            }};

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= cell_count) {{
                    return;
                }}
                target_winners[index] = 2147483647;
            }}
            """
        )
        self.programs["index_powder_target_winners"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform bool use_bridge_authoritative_blockers;

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
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer PowderTargetWinners {{
                int target_winners[];
            }};
            layout(std430, binding=8) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=9) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=10) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=11) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced_material[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D active_tile_tex;
            layout(binding=5) uniform sampler2D entity_id_tex;
            layout(binding=6) uniform sampler2D displaced_tex;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            int material_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int(bridge_cell_core[cell_index(cell) * 5] & 0xFFFFu);
                }}
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int((bridge_cell_core[cell_index(cell) * 5] >> 16u) & 0xFFu);
                }}
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            bool blocks_dda_target(ivec2 cell) {{
                return material_at(cell) > 0
                    || phase_at(cell) == phase_falling_island
                    || (
                        use_bridge_authoritative_blockers
                        ? bridge_entity_id[cell_index(cell)] != 0
                        : texelFetch(entity_id_tex, cell, 0).x > 0.5
                    )
                    || (
                        use_bridge_authoritative_blockers
                        ? bridge_displaced_material[cell_index(cell)] != 0
                        : texelFetch(displaced_tex, cell, 0).x > 0.5
                    );
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
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
                    if (sample_cell == target) {{
                        continue;
                    }}
                    if (blocks_dda_target(sample_cell)) {{
                        return false;
                    }}
                }}
                return true;
            }}

            bool reservation_has_live_source(PowderReservation reservation) {{
                ivec2 source = reservation.source_xy;
                if (!in_bounds(source) || !solve_cell_active(source)) {{
                    return false;
                }}
                int material_id = material_at(source);
                int phase_id = phase_at(source);
                return material_id > 0
                    && (phase_id == phase_powder || phase_id == phase_liquid)
                    && material_id == reservation.material_id;
            }}

            int winner_rank(PowderReservation reservation) {{
                ivec2 source = reservation.source_xy;
                int rank_y = (cell_grid_size.y - 1) - clamp(source.y, 0, cell_grid_size.y - 1);
                int rank_x = clamp(source.x, 0, cell_grid_size.x - 1);
                return rank_y * cell_grid_size.x + rank_x;
            }}

            void index_path_winners(PowderReservation reservation) {{
                ivec2 source = reservation.source_xy;
                ivec2 target = reservation.reserved_target_xy;
                ivec2 desired = target - source;
                if (desired.x == 0 && desired.y == 0) {{
                    return;
                }}
                int rank = winner_rank(reservation);
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
                    ivec2 path_cell = source + current;
                    if (!in_bounds(path_cell)) {{
                        return;
                    }}
                    atomicMin(target_winners[cell_index(path_cell)], rank);
                }}
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                ivec2 source = reservation.source_xy;
                ivec2 target = reservation.reserved_target_xy;
                if (
                    !reservation_has_live_source(reservation)
                    || same_cell(target, source)
                    || !in_bounds(target)
                    || !path_is_clear(source, target)
                ) {{
                    return;
                }}
                index_path_winners(reservation);
            }}
            """
        )
        self.programs["resolve_powder_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int phase_powder;
            uniform int phase_liquid;
            uniform int phase_falling_island;
            uniform bool use_bridge_authoritative_blockers;

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
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer MaterialParams {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=3) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=4) readonly buffer PowderTargetWinners {{
                int target_winners[];
            }};
            layout(std430, binding=8) readonly buffer BridgeCellCoreBuffer {{
                uint bridge_cell_core[];
            }};
            layout(std430, binding=9) readonly buffer BridgeIslandBuffer {{
                int bridge_island_id[];
            }};
            layout(std430, binding=10) readonly buffer BridgeEntityBuffer {{
                int bridge_entity_id[];
            }};
            layout(std430, binding=11) readonly buffer BridgeDisplacedBuffer {{
                int bridge_displaced_material[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=3) uniform sampler2D active_tile_tex;
            layout(binding=5) uniform sampler2D entity_id_tex;
            layout(binding=6) uniform sampler2D displaced_tex;

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            int cell_index(ivec2 cell) {{
                return cell.y * cell_grid_size.x + cell.x;
            }}

            bool solve_cell_active(ivec2 cell) {{
                ivec2 tile = ivec2(
                    min(cell.x / tile_size, tile_grid_size.x - 1),
                    min(cell.y / tile_size, tile_grid_size.y - 1)
                );
                return texelFetch(active_tile_tex, tile, 0).x > 0.5;
            }}

            int material_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int(bridge_cell_core[cell_index(cell) * 5] & 0xFFFFu);
                }}
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                if (use_bridge_authoritative_blockers) {{
                    return int((bridge_cell_core[cell_index(cell) * 5] >> 16u) & 0xFFu);
                }}
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            bool blocks_dda_target(ivec2 cell) {{
                return material_at(cell) > 0
                    || phase_at(cell) == phase_falling_island
                    || (
                        use_bridge_authoritative_blockers
                        ? bridge_entity_id[cell_index(cell)] != 0
                        : texelFetch(entity_id_tex, cell, 0).x > 0.5
                    )
                    || (
                        use_bridge_authoritative_blockers
                        ? bridge_displaced_material[cell_index(cell)] != 0
                        : texelFetch(displaced_tex, cell, 0).x > 0.5
                    );
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
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
                    if (sample_cell == target) {{
                        continue;
                    }}
                    if (blocks_dda_target(sample_cell)) {{
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

            bool better_source(ivec2 candidate, ivec2 current_best) {{
                if (current_best.x < 0) {{
                    return true;
                }}
                if (candidate.y != current_best.y) {{
                    return candidate.y > current_best.y;
                }}
                return candidate.x < current_best.x;
            }}

            bool reservation_has_live_source(PowderReservation reservation) {{
                ivec2 source = reservation.source_xy;
                if (!in_bounds(source) || !solve_cell_active(source)) {{
                    return false;
                }}
                int material_id = material_at(source);
                int phase_id = phase_at(source);
                return material_id > 0
                    && (phase_id == phase_powder || phase_id == phase_liquid)
                    && material_id == reservation.material_id;
            }}

            int source_rank(ivec2 source) {{
                int rank_y = (cell_grid_size.y - 1) - clamp(source.y, 0, cell_grid_size.y - 1);
                int rank_x = clamp(source.x, 0, cell_grid_size.x - 1);
                return rank_y * cell_grid_size.x + rank_x;
            }}

            int target_winner_rank(ivec2 target) {{
                if (!in_bounds(target)) {{
                    return -1;
                }}
                int rank = target_winners[target.y * cell_grid_size.x + target.x];
                if (rank == 2147483647) {{
                    return -1;
                }}
                return rank;
            }}

            ivec2 resolved_path_prefix(PowderReservation reservation) {{
                ivec2 source = reservation.source_xy;
                ivec2 target = reservation.reserved_target_xy;
                ivec2 desired = target - source;
                if (desired.x == 0 && desired.y == 0) {{
                    return source;
                }}
                int rank = source_rank(source);
                ivec2 current = ivec2(0, 0);
                ivec2 furthest = source;
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
                    ivec2 path_cell = source + current;
                    if (!in_bounds(path_cell)) {{
                        break;
                    }}
                    if (!same_cell(path_cell, target) && blocks_dda_target(path_cell)) {{
                        break;
                    }}
                    if (target_winner_rank(path_cell) != rank) {{
                        break;
                    }}
                    furthest = path_cell;
                }}
                return furthest;
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                ivec2 source = reservation.source_xy;
                int material_id = clamp(reservation.material_id, 0, {MAX_MATERIALS - 1});
                ivec2 resolved = source;
                int resolve_state = {POWDER_RESOLVE_BLOCKED};
                ivec2 reserved = reservation.reserved_target_xy;
                bool live_source = reservation_has_live_source(reservation);
                if (
                    live_source
                    && !same_cell(reserved, source)
                    && in_bounds(reserved)
                    && path_is_clear(source, reserved)
                ) {{
                    ivec2 path_resolved = resolved_path_prefix(reservation);
                    if (!same_cell(path_resolved, source)) {{
                        resolved = path_resolved;
                        resolve_state = {POWDER_RESOLVE_DDA};
                    }}
                }}
                if (
                    live_source
                    && phase_at(source) == phase_powder
                    && int(contact_params[material_id].z + 0.5) != {POWDER_SOLVER_SUSPENDED}
                ) {{
                    for (int candidate_index = 0; candidate_index < 3; ++candidate_index) {{
                        ivec2 fallback = fallback_candidate(source, material_id, candidate_index);
                        if (
                            resolve_state != {POWDER_RESOLVE_DDA}
                            &&
                            in_bounds(fallback)
                            && !blocks_dda_target(fallback)
                            && target_winner_rank(fallback) < 0
                        ) {{
                            resolved = fallback;
                            resolve_state = {POWDER_RESOLVE_FALLBACK};
                            break;
                        }}
                    }}
                }}
                reservations[index].resolved_target_xy = resolved;
                reservations[index].resolve_state = resolve_state;
            }}
            """
        )
        self.programs["clear_powder_apply_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int cell_count;
            layout(std430, binding=0) buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=1) buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= cell_count) {{
                    return;
                }}
                apply_incoming[index] = -1;
                apply_outgoing[index] = -1;
            }}
            """
        )
        self.programs["clear_falling_island_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform int cell_count;
            uniform int clear_flags;

            layout(std430, binding=0) buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=1) buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=2) buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};
            layout(std430, binding=3) buffer IslandReservationSourceIndex {{
                int reservation_source_index[];
            }};

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                if (index >= cell_count) {{
                    return;
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING}) != 0) {{
                    apply_incoming[index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING}) != 0) {{
                    apply_outgoing[index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION}) != 0) {{
                    materialization_index[index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_SOURCE}) != 0) {{
                    reservation_source_index[index] = {INDEX_EMPTY};
                }}
            }}
            """
        )
        self.programs["clear_falling_island_index_for_active_tiles"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int clear_flags;

            layout(std430, binding=0) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=1) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};
            layout(std430, binding=2) buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=3) buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=4) buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};
            layout(std430, binding=5) buffer IslandReservationSourceIndex {{
                int reservation_source_index[];
            }};

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void main() {{
                ivec2 cell;
                if (!active_dispatch_cell(cell)) {{
                    return;
                }}
                int cell_index = cell.y * cell_grid_size.x + cell.x;
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING}) != 0) {{
                    apply_incoming[cell_index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING}) != 0) {{
                    apply_outgoing[cell_index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION}) != 0) {{
                    materialization_index[cell_index] = {INDEX_EMPTY};
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_SOURCE}) != 0) {{
                    reservation_source_index[cell_index] = {INDEX_EMPTY};
                }}
            }}
            """
        )
        self.programs["clear_falling_island_index_for_reservations"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int clear_flags;

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
            layout(std430, binding=1) readonly buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=2) buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=3) buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=4) buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};
            layout(std430, binding=5) buffer IslandReservationSourceIndex {{
                int reservation_source_index[];
            }};

            int active_reservation_count() {{
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool valid_reservation(FallingIslandReservation reservation) {{
                return reservation.island_id > 0 && reservation.resolve_state != {ISLAND_RESOLVE_STALE};
            }}

            bool moving_reservation(FallingIslandReservation reservation) {{
                return valid_reservation(reservation)
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool settling_reservation(FallingIslandReservation reservation) {{
                return valid_reservation(reservation)
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            void clear_cell(ivec2 cell, int cell_clear_flags) {{
                int index = cell.y * cell_grid_size.x + cell.x;
                if ((cell_clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING}) != 0) {{
                    apply_incoming[index] = {INDEX_EMPTY};
                }}
                if ((cell_clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING}) != 0) {{
                    apply_outgoing[index] = {INDEX_EMPTY};
                }}
                if ((cell_clear_flags & {FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION}) != 0) {{
                    materialization_index[index] = {INDEX_EMPTY};
                }}
                if ((cell_clear_flags & {FALLING_ISLAND_INDEX_CLEAR_SOURCE}) != 0) {{
                    reservation_source_index[index] = {INDEX_EMPTY};
                }}
            }}

            void clear_bbox(ivec4 bbox, int cell_clear_flags) {{
                int x0 = clamp(min(bbox.x, bbox.z), 0, cell_grid_size.x);
                int y0 = clamp(min(bbox.y, bbox.w), 0, cell_grid_size.y);
                int x1 = clamp(max(bbox.x, bbox.z), 0, cell_grid_size.x);
                int y1 = clamp(max(bbox.y, bbox.w), 0, cell_grid_size.y);
                int bbox_width = x1 - x0;
                int bbox_height = y1 - y0;
                if (bbox_width <= 0 || bbox_height <= 0) {{
                    return;
                }}
                int bbox_area = bbox_width * bbox_height;
                int local_stride = int(gl_WorkGroupSize.x);
                for (int offset = int(gl_LocalInvocationID.x); offset < bbox_area; offset += local_stride) {{
                    ivec2 cell = ivec2(x0 + (offset % bbox_width), y0 + (offset / bbox_width));
                    clear_cell(cell, cell_clear_flags);
                }}
            }}

            void main() {{
                int reservation_index = int(gl_WorkGroupID.x);
                if (reservation_index >= active_reservation_count()) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[reservation_index];
                if (!valid_reservation(reservation)) {{
                    return;
                }}
                ivec4 source_bbox = ivec4(
                    reservation.bbox_x,
                    reservation.bbox_y,
                    reservation.bbox_z,
                    reservation.bbox_w
                );
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_SOURCE}) != 0) {{
                    clear_bbox(source_bbox, {FALLING_ISLAND_INDEX_CLEAR_SOURCE});
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING}) != 0 && moving_reservation(reservation)) {{
                    clear_bbox(source_bbox, {FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING});
                }}
                if ((clear_flags & {FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING}) != 0 && moving_reservation(reservation)) {{
                    clear_bbox(
                        source_bbox + ivec4(
                            reservation.resolved_dx,
                            reservation.resolved_dy,
                            reservation.resolved_dx,
                            reservation.resolved_dy
                        ),
                        {FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING}
                    );
                }}
                if (
                    (clear_flags & {FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION}) != 0
                    && settling_reservation(reservation)
                ) {{
                    clear_bbox(source_bbox, {FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION});
                }}
            }}
            """
        )
        self.programs["fill_falling_island_apply_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int phase_falling_island;

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
            layout(std430, binding=1) readonly buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=2) buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=3) buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D island_id_tex;

            int active_reservation_count() {{
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool is_moving(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                int source_island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int source_material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                int source_phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id
                    && source_material_id > 0
                    && source_phase_id == phase_falling_island;
            }}

            void main() {{
                int index = int(gl_WorkGroupID.x);
                if (index >= active_reservation_count()) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[index];
                if (!is_moving(reservation) || reservation.island_id <= 0) {{
                    return;
                }}
                int x0 = clamp(min(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y0 = clamp(min(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int x1 = clamp(max(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y1 = clamp(max(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int bbox_width = x1 - x0;
                int bbox_height = y1 - y0;
                if (bbox_width <= 0 || bbox_height <= 0) {{
                    return;
                }}
                int bbox_area = bbox_width * bbox_height;
                int local_stride = int(gl_WorkGroupSize.x);
                for (int offset = int(gl_LocalInvocationID.x); offset < bbox_area; offset += local_stride) {{
                    ivec2 source = ivec2(x0 + (offset % bbox_width), y0 + (offset / bbox_width));
                    if (!source_matches(reservation, source)) {{
                        continue;
                    }}
                    int source_index = source.y * cell_grid_size.x + source.x;
                    atomicMin(apply_outgoing[source_index], index);
                    ivec2 target = source + ivec2(reservation.resolved_dx, reservation.resolved_dy);
                    if (!in_bounds(target)) {{
                        continue;
                    }}
                    int target_index = target.y * cell_grid_size.x + target.x;
                    atomicMin(apply_incoming[target_index], index);
                }}
            }}
            """
        )
        self.programs["fill_falling_island_materialization_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;
            uniform int phase_falling_island;

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
            layout(std430, binding=1) readonly buffer IslandReservationCount {{
                int reservation_count;
            }};
            layout(std430, binding=2) buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D island_id_tex;

            int active_reservation_count() {{
                return clamp(reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool is_settle_reservation(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.target_dx != 0 || reservation.target_dy != 0)
                    && reservation.resolved_dx == 0
                    && reservation.resolved_dy == 0;
            }}

            bool source_matches(FallingIslandReservation reservation, ivec2 cell) {{
                if (!in_bounds(cell)) {{
                    return false;
                }}
                int source_island_id = int(texelFetch(island_id_tex, cell, 0).x + 0.5);
                int source_material_id = int(texelFetch(material_tex, cell, 0).x + 0.5);
                int source_phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id
                    && source_material_id > 0
                    && source_phase_id == phase_falling_island;
            }}

            void main() {{
                int index = int(gl_WorkGroupID.x);
                if (index >= active_reservation_count()) {{
                    return;
                }}
                FallingIslandReservation reservation = reservations[index];
                if (!is_settle_reservation(reservation) || reservation.island_id <= 0) {{
                    return;
                }}
                int x0 = clamp(min(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y0 = clamp(min(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int x1 = clamp(max(reservation.bbox_x, reservation.bbox_z), 0, cell_grid_size.x);
                int y1 = clamp(max(reservation.bbox_y, reservation.bbox_w), 0, cell_grid_size.y);
                int bbox_width = x1 - x0;
                int bbox_height = y1 - y0;
                if (bbox_width <= 0 || bbox_height <= 0) {{
                    return;
                }}
                int bbox_area = bbox_width * bbox_height;
                int local_stride = int(gl_WorkGroupSize.x);
                for (int offset = int(gl_LocalInvocationID.x); offset < bbox_area; offset += local_stride) {{
                    ivec2 source = ivec2(x0 + (offset % bbox_width), y0 + (offset / bbox_width));
                    if (!source_matches(reservation, source)) {{
                        continue;
                    }}
                    int source_index = source.y * cell_grid_size.x + source.x;
                    atomicMin(materialization_index[source_index], index);
                }}
            }}
            """
        )
        self.programs["index_powder_apply_winners"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

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
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) buffer PowderTargetWinners {{
                int target_winners[];
            }};

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
            }}

            bool is_moving(PowderReservation reservation) {{
                return (
                    (reservation.resolve_state == {POWDER_RESOLVE_DDA} || reservation.resolve_state == {POWDER_RESOLVE_FALLBACK})
                    && !same_cell(reservation.source_xy, reservation.resolved_target_xy)
                );
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            int source_rank(ivec2 source) {{
                int rank_y = (cell_grid_size.y - 1) - clamp(source.y, 0, cell_grid_size.y - 1);
                int rank_x = clamp(source.x, 0, cell_grid_size.x - 1);
                return rank_y * cell_grid_size.x + rank_x;
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                if (!is_moving(reservation) || !in_bounds(reservation.source_xy) || !in_bounds(reservation.resolved_target_xy)) {{
                    return;
                }}
                int target_index = reservation.resolved_target_xy.y * cell_grid_size.x + reservation.resolved_target_xy.x;
                atomicMin(target_winners[target_index], source_rank(reservation.source_xy));
            }}
            """
        )
        self.programs["fill_powder_apply_index"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={POWDER_RESERVATION_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

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
            layout(std430, binding=1) readonly buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=2) readonly buffer PowderTargetWinners {{
                int target_winners[];
            }};
            layout(std430, binding=3) buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=4) buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};

            bool in_bounds(ivec2 cell) {{
                return cell.x >= 0 && cell.y >= 0 && cell.x < cell_grid_size.x && cell.y < cell_grid_size.y;
            }}

            bool same_cell(ivec2 lhs, ivec2 rhs) {{
                return lhs.x == rhs.x && lhs.y == rhs.y;
            }}

            bool is_moving(PowderReservation reservation) {{
                return (
                    (reservation.resolve_state == {POWDER_RESOLVE_DDA} || reservation.resolve_state == {POWDER_RESOLVE_FALLBACK})
                    && !same_cell(reservation.source_xy, reservation.resolved_target_xy)
                );
            }}

            int active_reservation_count() {{
                return clamp(powder_reservation_count, 0, cell_grid_size.x * cell_grid_size.y);
            }}

            int source_rank(ivec2 source) {{
                int rank_y = (cell_grid_size.y - 1) - clamp(source.y, 0, cell_grid_size.y - 1);
                int rank_x = clamp(source.x, 0, cell_grid_size.x - 1);
                return rank_y * cell_grid_size.x + rank_x;
            }}

            void main() {{
                int index = int(gl_GlobalInvocationID.x);
                int count = active_reservation_count();
                if (index >= count) {{
                    return;
                }}
                PowderReservation reservation = reservations[index];
                if (reservation.resolve_state == {POWDER_RESOLVE_STALE} || !in_bounds(reservation.source_xy)) {{
                    return;
                }}
                if (!is_moving(reservation) || !in_bounds(reservation.resolved_target_xy)) {{
                    return;
                }}
                int source_index = reservation.source_xy.y * cell_grid_size.x + reservation.source_xy.x;
                int target_index = reservation.resolved_target_xy.y * cell_grid_size.x + reservation.resolved_target_xy.x;
                if (target_winners[target_index] == source_rank(reservation.source_xy)) {{
                    apply_outgoing[source_index] = index;
                    apply_incoming[target_index] = index;
                }} else {{
                    reservations[index].resolved_target_xy = reservation.source_xy;
                    reservations[index].resolve_state = {POWDER_RESOLVE_BLOCKED};
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
            uniform int phase_falling_island;
            uniform int max_powder_step;
            uniform float dt;

            layout(std430, binding=0) buffer MaterialParamBuffer {{
                vec4 material_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=1) buffer MaterialContactParams {{
                vec4 contact_params[{MAX_MATERIALS}];
            }};
            layout(std430, binding=2) buffer PowderReservationCount {{
                int powder_reservation_count;
            }};
            layout(std430, binding=11) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
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
            uniform int active_ttl_reset;

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

            void refresh_changed_cell(ivec2 cell) {{
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                int tile_index = tile.y * tile_grid_size.x + tile.x;
                active_tile_ttl[tile_index] = active_ttl_reset;
            }}

            int material_at(ivec2 cell) {{
                return int(texelFetch(material_tex, cell, 0).x + 0.5);
            }}

            int phase_at(ivec2 cell) {{
                return int(texelFetch(phase_tex, cell, 0).x + 0.5);
            }}

            bool blocks_dda_target(ivec2 cell) {{
                return material_at(cell) > 0
                    || phase_at(cell) == phase_falling_island
                    || texelFetch(entity_id_tex, cell, 0).x > 0.5
                    || texelFetch(displaced_tex, cell, 0).x > 0.5;
            }}

            ivec2 desired_target(ivec2 source, int material_id) {{
                material_id = clamp(material_id, 0, {MAX_MATERIALS - 1});
                vec2 velocity = texelFetch(velocity_tex, source, 0).xy;
                int max_step = int(material_params[material_id].x + 0.5);
                max_step = clamp(max_step, 0, max_powder_step);
                vec2 frame_delta = velocity * dt;
                return source + ivec2(
                    int(clamp(round(frame_delta.x), float(-max_step), float(max_step))),
                    int(clamp(round(frame_delta.y), float(-max_step), float(max_step)))
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
                    if (blocks_dda_target(sample_cell)) {{
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
                    if (blocks_dda_target(sample_cell)) {{
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
                        if (in_bounds(fallback) && !blocks_dda_target(fallback)) {{
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
                store_payload(dst, 0.0, 0.0, 0.0, vec2(0.0), 0.0, vec4(0.0), 0.0, 0.0, 0.0, 0.0);
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
                ivec2 incoming = winning_source_for(gid);
                if (incoming.x >= 0) {{
                    refresh_changed_cell(gid);
                    store_incoming(gid, incoming);
                    return;
                }}
                if (source_moves_out(gid)) {{
                    refresh_changed_cell(gid);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int active_ttl_reset;
            uniform int phase_falling_island;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;
            uniform bool skip_untouched_original_stores;

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
            layout(std430, binding=3) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=5) readonly buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void refresh_changed_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[tile.y * tile_grid_size.x + tile.x] = active_ttl_reset;
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
                return apply_incoming[cell.y * cell_grid_size.x + cell.x];
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                return apply_outgoing[cell.y * cell_grid_size.x + cell.x];
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
                int incoming_index = incoming_reservation_index(gid);
                int outgoing_index = outgoing_reservation_index(gid);
                if (incoming_index >= 0) {{
                    refresh_changed_cell(gid);
                    store_incoming(gid, reservations[incoming_index]);
                    return;
                }}
                if (outgoing_index >= 0) {{
                    PowderReservation reservation = reservations[outgoing_index];
                    if (is_moving(reservation)) {{
                        refresh_changed_cell(gid);
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
                if (skip_untouched_original_stores) {{
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;

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
            layout(std430, binding=4) readonly buffer PowderApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=5) readonly buffer PowderApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
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
                return apply_incoming[cell.y * cell_grid_size.x + cell.x];
            }}

            int outgoing_reservation_index(ivec2 cell) {{
                return apply_outgoing[cell.y * cell_grid_size.x + cell.x];
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int gas_cell_size;
            uniform int active_ttl_reset;
            uniform int phase_falling_island;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;

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
            layout(std430, binding=2) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
            }};
            layout(std430, binding=3) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=5) readonly buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

            bool is_moving(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void refresh_changed_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[tile.y * tile_grid_size.x + tile.x] = active_ttl_reset;
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
                int source_phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id
                    && source_material_id > 0
                    && source_phase_id == phase_falling_island;
            }}

            int indexed_apply_incoming(ivec2 cell) {{
                int count = active_reservation_count();
                int value = apply_incoming[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            int indexed_apply_outgoing(ivec2 cell) {{
                int count = active_reservation_count();
                int value = apply_outgoing[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            bool source_cell_for_incoming(int reservation_index, ivec2 cell, out ivec2 source_cell) {{
                FallingIslandReservation reservation = reservations[reservation_index];
                source_cell = cell - ivec2(reservation.resolved_dx, reservation.resolved_dy);
                if (source_cell.x < 0 || source_cell.y < 0 || source_cell.x >= cell_grid_size.x || source_cell.y >= cell_grid_size.y) {{
                    return false;
                }}
                return source_matches(reservation, source_cell);
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
                ivec2 source_cell;
                int incoming_index = indexed_apply_incoming(gid);
                if (incoming_index >= 0 && source_cell_for_incoming(incoming_index, gid, source_cell)) {{
                    refresh_changed_cell(gid);
                    store_incoming(gid, reservations[incoming_index], source_cell);
                    return;
                }}
                int outgoing_index = indexed_apply_outgoing(gid);
                if (outgoing_index >= 0 && source_matches(reservations[outgoing_index], gid)) {{
                    refresh_changed_cell(gid);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int active_ttl_reset;
            uniform int phase_falling_island;
            uniform int reservation_count;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;

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
            layout(std430, binding=2) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
            }};
            layout(std430, binding=3) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer IslandApplyIncoming {{
                int apply_incoming[];
            }};
            layout(std430, binding=5) readonly buffer IslandApplyOutgoing {{
                int apply_outgoing[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

            bool is_moving(FallingIslandReservation reservation) {{
                return reservation.resolve_state != {ISLAND_RESOLVE_STALE}
                    && (reservation.resolved_dx != 0 || reservation.resolved_dy != 0);
            }}

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void refresh_changed_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[tile.y * tile_grid_size.x + tile.x] = active_ttl_reset;
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
                int source_phase_id = int(texelFetch(phase_tex, cell, 0).x + 0.5);
                return source_island_id == reservation.island_id
                    && source_material_id > 0
                    && source_phase_id == phase_falling_island;
            }}

            int indexed_apply_incoming(ivec2 cell) {{
                int count = active_reservation_count();
                int value = apply_incoming[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            int indexed_apply_outgoing(ivec2 cell) {{
                int count = active_reservation_count();
                int value = apply_outgoing[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            bool source_cell_for_incoming(int reservation_index, ivec2 cell, out ivec2 source_cell) {{
                FallingIslandReservation reservation = reservations[reservation_index];
                source_cell = cell - ivec2(reservation.resolved_dx, reservation.resolved_dy);
                if (source_cell.x < 0 || source_cell.y < 0 || source_cell.x >= cell_grid_size.x || source_cell.y >= cell_grid_size.y) {{
                    return false;
                }}
                return source_matches(reservation, source_cell);
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
                ivec2 source_cell;
                int incoming_index = indexed_apply_incoming(gid);
                if (incoming_index >= 0 && source_cell_for_incoming(incoming_index, gid, source_cell)) {{
                    refresh_changed_cell(gid);
                    store_incoming(gid, reservations[incoming_index], source_cell);
                    return;
                }}
                int outgoing_index = indexed_apply_outgoing(gid);
                if (outgoing_index >= 0 && source_matches(reservations[outgoing_index], gid)) {{
                    refresh_changed_cell(gid);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int reservation_count;
            uniform int phase_falling_island;
            uniform int phase_powder;
            uniform int phase_static_solid;
            uniform int mode;
            uniform int active_ttl_reset;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;

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
            layout(std430, binding=2) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
            }};
            layout(std430, binding=3) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
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

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void refresh_changed_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[tile.y * tile_grid_size.x + tile.x] = active_ttl_reset;
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

            int indexed_materialization_reservation(ivec2 cell) {{
                int count = active_reservation_count();
                int value = materialization_index[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            bool indexed_settle_reservation_matches(
                int reservation_index,
                ivec2 cell,
                int island_id,
                int material_id
            ) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                FallingIslandReservation reservation = reservations[reservation_index];
                if (!is_settle_reservation(reservation) || reservation.island_id != island_id) {{
                    return false;
                }}
                return cell.x >= reservation.bbox_x
                    && cell.x < reservation.bbox_z
                    && cell.y >= reservation.bbox_y
                    && cell.y < reservation.bbox_w;
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
                int island_id = int(texelFetch(island_id_tex, gid, 0).x + 0.5);
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (mode == 0) {{
                    if (shed_fragment(gid, island_id, material_id)) {{
                        refresh_changed_cell(gid);
                        store_generated_powder(gid, material_id);
                        return;
                    }}
                    store_original(gid);
                    return;
                }}
                int settle_index = indexed_materialization_reservation(gid);
                if (
                    settle_index >= 0
                    && indexed_settle_reservation_matches(settle_index, gid, island_id, material_id)
                ) {{
                    refresh_changed_cell(gid);
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
            uniform ivec2 tile_grid_size;
            uniform int tile_size;
            uniform int reservation_count;
            uniform int phase_falling_island;
            uniform int mode;
            uniform int active_ttl_reset;
            uniform bool use_reservation_count_buffer;
            uniform bool use_active_tile_dispatch;

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
            layout(std430, binding=2) readonly buffer IslandReservationCount {{
                int reservation_count_buffer;
            }};
            layout(std430, binding=3) buffer ActiveTileTTLBuffer {{
                int active_tile_ttl[];
            }};
            layout(std430, binding=4) readonly buffer IslandMaterializationIndex {{
                int materialization_index[];
            }};
            layout(std430, binding=6) readonly buffer ActiveTileCountBuffer {{
                uint active_tile_count[];
            }};
            layout(std430, binding=7) readonly buffer ActiveTileListBuffer {{
                ivec2 active_tile_list[];
            }};

            layout(binding=0) uniform sampler2D material_tex;
            layout(binding=1) uniform sampler2D phase_tex;
            layout(binding=7) uniform sampler2D island_id_tex;
            layout(binding=8) uniform sampler2D entity_id_tex;
            layout(binding=9) uniform sampler2D displaced_tex;

            layout(r32f, binding=0) writeonly uniform image2D island_id_out_img;
            layout(r32f, binding=1) writeonly uniform image2D entity_id_out_img;
            layout(r32f, binding=2) writeonly uniform image2D displaced_out_img;

            int active_reservation_count() {{
                if (use_reservation_count_buffer) {{
                    return clamp(reservation_count_buffer, 0, max(reservation_count, 0));
                }}
                return max(reservation_count, 0);
            }}

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

            bool active_dispatch_cell(out ivec2 cell) {{
                int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
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
                ivec2 local_cell = subtile_xy * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
                cell = tile_origin + local_cell;
                ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
                return cell.x < tile_end.x && cell.y < tile_end.y;
            }}

            void refresh_changed_cell(ivec2 cell) {{
                if (cell.x < 0 || cell.y < 0 || cell.x >= cell_grid_size.x || cell.y >= cell_grid_size.y) {{
                    return;
                }}
                ivec2 tile = ivec2(
                    clamp(cell.x / tile_size, 0, tile_grid_size.x - 1),
                    clamp(cell.y / tile_size, 0, tile_grid_size.y - 1)
                );
                active_tile_ttl[tile.y * tile_grid_size.x + tile.x] = active_ttl_reset;
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

            int indexed_materialization_reservation(ivec2 cell) {{
                int count = active_reservation_count();
                int value = materialization_index[cell.y * cell_grid_size.x + cell.x];
                if (value < 0 || value >= count || value == {INDEX_EMPTY}) {{
                    return -1;
                }}
                return value;
            }}

            bool indexed_settle_reservation_matches(int reservation_index, ivec2 cell, int island_id, int material_id) {{
                if (!falling_source_matches(cell, island_id, material_id)) {{
                    return false;
                }}
                FallingIslandReservation reservation = reservations[reservation_index];
                if (!is_settle_reservation(reservation) || reservation.island_id != island_id) {{
                    return false;
                }}
                return cell.x >= reservation.bbox_x
                    && cell.x < reservation.bbox_z
                    && cell.y >= reservation.bbox_y
                    && cell.y < reservation.bbox_w;
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
                    refresh_changed_cell(dst);
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
                int island_id = int(texelFetch(island_id_tex, gid, 0).x + 0.5);
                int material_id = int(texelFetch(material_tex, gid, 0).x + 0.5);
                if (mode == 0) {{
                    if (shed_fragment(gid, island_id, material_id)) {{
                        refresh_changed_cell(gid);
                        store_generated_powder(gid, material_id);
                        return;
                    }}
                    store_original(gid);
                    return;
                }}
                int settle_index = indexed_materialization_reservation(gid);
                if (settle_index >= 0 && indexed_settle_reservation_matches(settle_index, gid, island_id, material_id)) {{
                    refresh_changed_cell(gid);
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
        if upload_plan["entity_id"]:
            resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
        if upload_plan["placeholder_displaced_material"]:
            resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
        if upload_plan["active_tile_ttl"]:
            resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        self._compact_active_tiles(world, resources)
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

    def _active_tile_workgroups_per_tile(self, world: "WorldEngine") -> int:
        if int(world.active.tile_size) == LOCAL_SIZE * ACTIVE_TILE_WORKGROUP_AXIS:
            return ACTIVE_TILE_WORKGROUPS_PER_TILE
        axis = max(1, (int(world.active.tile_size) + LOCAL_SIZE - 1) // LOCAL_SIZE)
        return axis * axis

    def _active_scheduler_gpu_authoritative(self, world: "WorldEngine") -> bool:
        authoritative = world.bridge.gpu_authoritative_resources
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in authoritative
            and "active_chunk_mask" in authoritative
        )

    def _compact_active_tiles(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        workgroups_per_tile = int(self._active_tile_workgroups_per_tile(world))
        if self._active_scheduler_gpu_authoritative(world):
            bridge = world.bridge
            bridge._ensure_active_scheduler_programs()
            bridge._refresh_active_chunks_and_meta(world, read_meta=False)
            compact_program = self.programs["compact_active_tiles_from_chunks"]
            if not hasattr(compact_program, "run_indirect"):
                raise RuntimeError("GPU motion active chunk compaction requires ModernGL ComputeShader.run_indirect")
            compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
            compact_program["workgroups_per_tile"].value = workgroups_per_tile
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
            compact_program["workgroups_per_tile"].value = workgroups_per_tile
            resources.active_tile_tex.use(location=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=0)
            resources.active_tile_list.bind_to_storage_buffer(binding=1)
            resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            compact_program.run((tile_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)

    def _build_active_tile_count_dispatch_args(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["build_powder_reservation_dispatch"]
        program["invocations_per_group"].value = 1
        program["max_reservation_count"].value = int(world.active.tile_width * world.active.tile_height)
        resources.active_tile_count.bind_to_storage_buffer(binding=6)
        resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

    def _build_falling_island_materialization_candidate_dispatch(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=0)
        resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        self._build_active_tile_count_dispatch_args(world, resources)
        program = self.programs["build_falling_island_materialization_candidate_dispatch"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        resources.phase_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.island_materialization_candidate_tile_count.bind_to_storage_buffer(binding=2)
        resources.island_materialization_candidate_tile_list.bind_to_storage_buffer(binding=3)
        resources.island_materialization_candidate_dispatch_args.bind_to_storage_buffer(binding=4)
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU motion falling island materialization candidate dispatch requires indirect dispatch")
        program.run_indirect(resources.island_runtime_dispatch_args)
        self._sync_compute_writes(ctx)

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

    def _bridge_authoritative_cell_blockers(self, world: "WorldEngine") -> bool:
        authoritative = world.bridge.gpu_authoritative_resources
        return (
            self._formal_gpu_frame(world)
            and {"cell_core", "entity_id", "placeholder_displaced_material"}.issubset(authoritative)
        )

    def _bind_bridge_cell_blockers(self, world: "WorldEngine", *, cell_binding: int = 8) -> None:
        bridge = world.bridge
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=cell_binding)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=cell_binding + 1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=cell_binding + 2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=cell_binding + 3)

    def _bridge_authoritative_island_state(self, world: "WorldEngine") -> bool:
        authoritative = world.bridge.gpu_authoritative_resources
        return self._formal_gpu_frame(world) and {"cell_core", "island_id"}.issubset(authoritative)

    def _bind_bridge_island_state(self, world: "WorldEngine", *, cell_binding: int = 7) -> None:
        bridge = world.bridge
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=cell_binding)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=cell_binding + 1)

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
        *,
        use_existing_active_tile_dispatch: bool = False,
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

        active_tile_indirect = bool(self._active_scheduler_gpu_authoritative(world))
        if active_tile_indirect and not use_existing_active_tile_dispatch:
            self._compact_active_tiles(world, resources)

        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.velocity_tex.bind_to_image(3, read=False, write=True)
            resources.temp_tex.bind_to_image(4, read=False, write=True)
            resources.timer_tex.bind_to_image(5, read=False, write=True)
            resources.integrity_tex.bind_to_image(6, read=False, write=True)
            if active_tile_indirect:
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                self._run_active_tile_indirect(program, resources, "motion bridge cell load")
            else:
                program.run(group_x, group_y, 1)

        if copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.island_id_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_tex.bind_to_image(2, read=False, write=True)
            if active_tile_indirect:
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                self._run_active_tile_indirect(program, resources, "motion bridge aux load")
            else:
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

    def _load_authoritative_integrate_inputs(
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
        copy_flow = "flow_velocity" in authoritative
        if not (copy_cell_core or copy_flow):
            return

        active_tile_indirect = bool(self._active_scheduler_gpu_authoritative(world))
        if active_tile_indirect:
            self._compact_active_tiles(world, resources)

        if copy_cell_core:
            program = self.programs["load_bridge_integrate_inputs"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.velocity_tex.bind_to_image(1, read=False, write=True)
            if active_tile_indirect:
                resources.active_tile_count.bind_to_storage_buffer(binding=1)
                resources.active_tile_list.bind_to_storage_buffer(binding=2)
                self._run_active_tile_indirect(program, resources, "motion integrate bridge input load")
            else:
                program.run(group_x, group_y, 1)

        if copy_flow:
            gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            program = self.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["copy_flow_velocity"].value = True
            program["copy_ambient"].value = False
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
        velocity_out_active_only: bool = False,
        active_tile_indirect: bool = False,
        active_tile_count_buffer: Any | None = None,
        active_tile_list_buffer: Any | None = None,
        active_tile_dispatch_args: Any | None = None,
        use_powder_apply_touch_sources: bool = False,
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
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["velocity_out_active_only"].value = bool(velocity_out_active_only)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        program["use_powder_apply_touch_sources"].value = bool(use_powder_apply_touch_sources)
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
        resources.velocity_tex.use(location=10)
        resources.active_tile_tex.use(location=11)
        if use_powder_apply_touch_sources:
            resources.material_tex.use(location=12)
            resources.phase_tex.use(location=13)
            resources.cell_flags_tex.use(location=14)
            resources.velocity_tex.use(location=15)
            resources.temp_tex.use(location=16)
            resources.timer_tex.use(location=17)
            resources.integrity_tex.use(location=18)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
        tile_count_buffer = active_tile_count_buffer if active_tile_count_buffer is not None else resources.active_tile_count
        tile_list_buffer = active_tile_list_buffer if active_tile_list_buffer is not None else resources.active_tile_list
        tile_count_buffer.bind_to_storage_buffer(binding=4)
        tile_list_buffer.bind_to_storage_buffer(binding=5)
        if use_powder_apply_touch_sources:
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=6)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=7)
        if active_tile_indirect:
            self._run_active_tile_indirect(
                program,
                resources,
                "bridge cell publish",
                dispatch_args=active_tile_dispatch_args,
            )
        else:
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

    def _publish_bridge_velocity_words(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        active_tile_indirect: bool,
    ) -> bool:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not (
            self._formal_gpu_frame(world)
            and bool(active_tile_indirect)
            and bridge.enabled
            and bridge.ctx is not None
            and self._bridge_context_active(world)
            and "cell_core" in bridge.gpu_authoritative_resources
        ):
            return False
        if "cell_core" not in bridge.buffers:
            return False

        program = self.programs["publish_bridge_velocity_word"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        resources.velocity_out_tex.use(location=0)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        self._run_active_tile_indirect(program, resources, "bridge velocity word publish")
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative("cell_core")
        return True

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
        self._ensure_programs(bridge.ctx)
        resources = self._ensure_resources(world)
        required_bytes = max(4, reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize)
        bridge_buffer = bridge.buffers["island_reservation"]
        if bridge_buffer.size < required_bytes:
            bridge_buffer.release()
            bridge_buffer = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers["island_reservation"] = bridge_buffer
        elif not self._formal_gpu_frame(world):
            bridge_buffer.orphan(required_bytes)
        if self._formal_gpu_frame(world):
            bridge.buffers["island_reservation_count"].write(np.array([0], dtype=np.int32).tobytes())
            if reservation_count > 0:
                with self._profile_pass(world, "island_reservation_publish_bridge"):
                    program = self.programs["publish_falling_island_reservations"]
                    program["reservation_capacity"].value = reservation_count
                    resources.island_reservations.bind_to_storage_buffer(binding=0)
                    resources.island_reservation_count.bind_to_storage_buffer(binding=1)
                    bridge_buffer.bind_to_storage_buffer(binding=2)
                    bridge.buffers["island_reservation_count"].bind_to_storage_buffer(binding=3)
                    self._run_island_reservation_indirect(
                        world,
                        resources,
                        program,
                        "falling island reservation publish",
                        reservation_capacity=reservation_count,
                        invocations_per_group=256,
                    )
                    bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
            bridge.mark_gpu_authoritative("island_reservation")
            return True
        if reservation_count > 0:
            bridge.ctx.copy_buffer(
                bridge_buffer,
                resources.island_reservations,
                size=reservation_count * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
            )
        bridge.buffers["island_reservation_count"].write(np.array([reservation_count], dtype=np.int32).tobytes())
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
            with self._profile_pass(world, "powder_publish_bridge"):
                bridge.buffers["powder_reservation_count"].write(np.array([0], dtype=np.int32).tobytes())
                program = self.programs["publish_powder_reservations"]
                program["reservation_capacity"].value = reservation_capacity
                resources.powder_reservations.bind_to_storage_buffer(binding=0)
                resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
                bridge_buffer.bind_to_storage_buffer(binding=2)
                bridge.buffers["powder_reservation_count"].bind_to_storage_buffer(binding=3)
                self._run_powder_reservation_indirect(
                    world,
                    resources,
                    program,
                    "powder reservation publish",
                    invocations_per_group=256,
                )
                bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
                bridge.mark_gpu_authoritative("powder_reservation")
            return True
        else:
            bridge_buffer.orphan(required_bytes)
        with self._profile_pass(world, "powder_publish_bridge"):
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
        formal_frame = self._formal_gpu_frame(world)
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
        if not formal_frame:
            bridge_buffer.write(np.zeros((required_bytes,), dtype=np.uint8).tobytes())
        bridge.buffers["island_runtime_count"].write(np.array([0], dtype=np.int32).tobytes())
        if reservation_count > 0:
            with self._profile_pass(world, "island_runtime_publish_bridge"):
                program = self.programs["publish_falling_island_runtime"]
                program["reservation_count"].value = int(reservation_count)
                program["use_reservation_count_buffer"].value = bool(formal_frame)
                program["cell_grid_size"].value = (world.width, world.height)
                program["paging_origin"].value = (int(world.paging.origin_x), int(world.paging.origin_y))
                program["paging_buffer_origin"].value = (
                    int(world.paging.buffer_origin_x),
                    int(world.paging.buffer_origin_y),
                )
                resources.island_reservations.bind_to_storage_buffer(binding=0)
                bridge_buffer.bind_to_storage_buffer(binding=1)
                bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=2)
                resources.island_reservation_count.bind_to_storage_buffer(binding=3)
                if formal_frame:
                    self._run_island_reservation_indirect(
                        world,
                        resources,
                        program,
                        "falling island runtime publish",
                        reservation_capacity=reservation_count,
                    )
                else:
                    group_x = (reservation_count + LOCAL_SIZE - 1) // LOCAL_SIZE
                    program.run(group_x, 1, 1)
                bridge.ctx.memory_barrier(bridge.ctx.SHADER_STORAGE_BARRIER_BIT)
        self.last_published_island_runtime_capacity = reservation_count
        if formal_frame:
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

    def _swap_powder_apply_textures(self, resources: GPUMotionResources) -> None:
        resources.material_tex, resources.material_out_tex = resources.material_out_tex, resources.material_tex
        resources.phase_tex, resources.phase_out_tex = resources.phase_out_tex, resources.phase_tex
        resources.cell_flags_tex, resources.cell_flags_out_tex = resources.cell_flags_out_tex, resources.cell_flags_tex
        resources.velocity_tex, resources.velocity_out_tex = resources.velocity_out_tex, resources.velocity_tex
        resources.temp_tex, resources.temp_out_tex = resources.temp_out_tex, resources.temp_tex
        resources.timer_tex, resources.timer_out_tex = resources.timer_out_tex, resources.timer_tex
        resources.integrity_tex, resources.integrity_out_tex = resources.integrity_out_tex, resources.integrity_tex
        resources.island_id_tex, resources.island_id_out_tex = resources.island_id_out_tex, resources.island_id_tex
        resources.entity_id_tex, resources.entity_id_out_tex = resources.entity_id_out_tex, resources.entity_id_tex
        resources.displaced_tex, resources.displaced_out_tex = resources.displaced_out_tex, resources.displaced_tex

    def _sync_compute_writes(self, ctx: Any) -> None:
        ctx.memory_barrier(
            getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(ctx, "SHADER_IMAGE_ACCESS_BARRIER_BIT", 0)
            | getattr(ctx, "TEXTURE_FETCH_BARRIER_BIT", 0)
            | getattr(ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )

    def _run_active_tile_indirect(
        self,
        program: Any,
        resources: GPUMotionResources,
        pass_name: str,
        *,
        dispatch_args: Any | None = None,
    ) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.active_tile_dispatch_args if dispatch_args is None else dispatch_args)

    def _refresh_authoritative_active_scheduler_after_apply(self, world: "WorldEngine", pass_name: str) -> None:
        if not (self._formal_gpu_frame(world) and "active_tile_ttl" in world.bridge.gpu_authoritative_resources):
            return
        with self._profile_pass(world, pass_name):
            world.bridge._ensure_active_scheduler_programs()
            world.bridge._refresh_active_chunks_and_meta(world, read_meta=False)
            world.bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _build_powder_reservation_dispatch_args(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        invocations_per_group: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["build_powder_reservation_dispatch"]
        program["invocations_per_group"].value = int(invocations_per_group)
        program["max_reservation_count"].value = int(world.width * world.height)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=6)
        resources.powder_reservation_dispatch_args.bind_to_storage_buffer(binding=7)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

    def _run_powder_reservation_indirect(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        program: Any,
        pass_name: str,
        *,
        invocations_per_group: int = POWDER_RESERVATION_LOCAL_SIZE,
    ) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
        self._build_powder_reservation_dispatch_args(
            world,
            resources,
            invocations_per_group=int(invocations_per_group),
        )
        program.run_indirect(resources.powder_reservation_dispatch_args)

    def _build_island_reservation_dispatch_args(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        reservation_capacity: int,
        invocations_per_group: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["build_powder_reservation_dispatch"]
        program["invocations_per_group"].value = int(invocations_per_group)
        program["max_reservation_count"].value = int(reservation_capacity)
        resources.island_reservation_count.bind_to_storage_buffer(binding=6)
        resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

    def _run_island_reservation_indirect(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        program: Any,
        pass_name: str,
        *,
        reservation_capacity: int,
        invocations_per_group: int = LOCAL_SIZE,
    ) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
        self._build_island_reservation_dispatch_args(
            world,
            resources,
            reservation_capacity=int(reservation_capacity),
            invocations_per_group=int(invocations_per_group),
        )
        program.run_indirect(resources.island_runtime_dispatch_args)

    def _build_island_runtime_dispatch_args(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        runtime_capacity: int,
        invocations_per_group: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["build_island_runtime_dispatch"]
        program["invocations_per_group"].value = int(invocations_per_group)
        program["runtime_capacity"].value = int(runtime_capacity)
        world.bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=6)
        resources.island_runtime_dispatch_args.bind_to_storage_buffer(binding=7)
        program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

    def _run_island_runtime_indirect(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        program: Any,
        pass_name: str,
        *,
        runtime_capacity: int,
        invocations_per_group: int = LOCAL_SIZE,
        before_run: Any | None = None,
    ) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU motion {pass_name} requires ModernGL ComputeShader.run_indirect")
        self._build_island_runtime_dispatch_args(
            world,
            resources,
            runtime_capacity=int(runtime_capacity),
            invocations_per_group=int(invocations_per_group),
        )
        if before_run is not None:
            before_run()
        program.run_indirect(resources.island_runtime_dispatch_args)

    def _clear_powder_target_winners_for_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["clear_powder_target_winners_for_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        self._run_powder_reservation_indirect(
            world,
            resources,
            program,
            "powder target winner reservation clear",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _clear_powder_apply_index_for_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["clear_powder_apply_index_for_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
        self._run_powder_reservation_indirect(
            world,
            resources,
            program,
            "powder apply index reservation clear",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _clear_powder_apply_index_for_active_tiles(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["clear_powder_apply_index_for_active_tiles"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.powder_target_winner.bind_to_storage_buffer(binding=2)
        resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
        resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
        self._run_active_tile_indirect(program, resources, "powder apply affected tile index clear")
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _build_powder_apply_dispatch(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        tile_count = int(world.active.tile_width * world.active.tile_height)
        clear_program = self.programs["clear_powder_affected_tile_dispatch"]
        clear_program["tile_count"].value = tile_count
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        clear_program.run((tile_count + 255) // 256, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))

        build_program = self.programs["build_powder_apply_dispatch"]
        build_program["cell_grid_size"].value = (world.width, world.height)
        build_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        build_program["tile_size"].value = int(world.active.tile_size)
        build_program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=2)
        resources.active_tile_count.bind_to_storage_buffer(binding=3)
        resources.active_tile_list.bind_to_storage_buffer(binding=4)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=5)
        self._run_powder_reservation_indirect(
            world,
            resources,
            build_program,
            "powder apply affected tile dispatch build",
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))

    def _build_falling_island_apply_dispatch(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        reservation_count: int,
        operation: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        reservation_count = max(0, int(reservation_count))
        tile_count = int(world.active.tile_width * world.active.tile_height)
        clear_program = self.programs["clear_powder_affected_tile_dispatch"]
        clear_program["tile_count"].value = tile_count
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
        clear_program.run((tile_count + 255) // 256, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))
        if reservation_count <= 0:
            return

        build_program = self.programs["build_falling_island_apply_dispatch"]
        build_program["cell_grid_size"].value = (world.width, world.height)
        build_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        build_program["tile_size"].value = int(world.active.tile_size)
        build_program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        build_program["operation"].value = int(operation)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=1)
        resources.powder_apply_tile_flags.bind_to_storage_buffer(binding=2)
        resources.active_tile_count.bind_to_storage_buffer(binding=3)
        resources.active_tile_list.bind_to_storage_buffer(binding=4)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=5)
        self._run_island_reservation_indirect(
            world,
            resources,
            build_program,
            "falling island apply affected tile dispatch build",
            reservation_capacity=reservation_count,
            invocations_per_group=POWDER_RESERVATION_LOCAL_SIZE,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | getattr(ctx, "COMMAND_BARRIER_BIT", 0))

    def _ensure_falling_island_index_capacity(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        cell_bytes = int(world.width * world.height * np.dtype(np.int32).itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_apply_incoming", cell_bytes)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_apply_outgoing", cell_bytes)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_materialization_index", cell_bytes)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_reservation_source_index", cell_bytes)

    def _clear_falling_island_index(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        pass_name: str,
        clear_flags: int,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._ensure_falling_island_index_capacity(world, resources)
        with self._profile_pass(world, pass_name):
            if self._formal_gpu_frame(world):
                reservation_count = max(0, int(reservation_count))
                if reservation_count <= 0:
                    return
                program = self.programs["clear_falling_island_index_for_reservations"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["clear_flags"].value = int(clear_flags)
                resources.island_reservations.bind_to_storage_buffer(binding=0)
                resources.island_reservation_count.bind_to_storage_buffer(binding=1)
                resources.island_apply_incoming.bind_to_storage_buffer(binding=2)
                resources.island_apply_outgoing.bind_to_storage_buffer(binding=3)
                resources.island_materialization_index.bind_to_storage_buffer(binding=4)
                resources.island_reservation_source_index.bind_to_storage_buffer(binding=5)
                self._run_island_reservation_indirect(
                    world,
                    resources,
                    program,
                    "falling island index reservation-domain clear",
                    reservation_capacity=reservation_count,
                    invocations_per_group=1,
                )
            else:
                cell_count = int(world.width * world.height)
                program = self.programs["clear_falling_island_index"]
                program["cell_count"].value = cell_count
                program["clear_flags"].value = int(clear_flags)
                resources.island_apply_incoming.bind_to_storage_buffer(binding=0)
                resources.island_apply_outgoing.bind_to_storage_buffer(binding=1)
                resources.island_materialization_index.bind_to_storage_buffer(binding=2)
                resources.island_reservation_source_index.bind_to_storage_buffer(binding=3)
                program.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

    def _dispatch_index_falling_island_reservation_sources(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._clear_falling_island_index(
            world,
            resources,
            pass_name="island_reservation_source_index_clear",
            clear_flags=FALLING_ISLAND_INDEX_CLEAR_SOURCE,
            reservation_count=int(reservation_count),
        )
        with self._profile_pass(world, "island_reservation_source_index_build"):
            program = self.programs["fill_falling_island_reservation_source_index"]
            program["cell_grid_size"].value = (world.width, world.height)
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            resources.island_reservation_count.bind_to_storage_buffer(binding=1)
            resources.island_reservation_source_index.bind_to_storage_buffer(binding=2)
            resources.material_tex.use(location=0)
            resources.island_id_tex.use(location=1)
            self._run_island_reservation_indirect(
                world,
                resources,
                program,
                "falling island reservation source index build",
                reservation_capacity=int(reservation_count),
                invocations_per_group=1,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    def _dispatch_index_falling_island_apply(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._clear_falling_island_index(
            world,
            resources,
            pass_name="island_apply_index_clear",
            clear_flags=FALLING_ISLAND_INDEX_CLEAR_APPLY,
            reservation_count=int(reservation_count),
        )
        with self._profile_pass(world, "island_apply_index_build"):
            program = self.programs["fill_falling_island_apply_index"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            resources.island_reservation_count.bind_to_storage_buffer(binding=1)
            resources.island_apply_incoming.bind_to_storage_buffer(binding=2)
            resources.island_apply_outgoing.bind_to_storage_buffer(binding=3)
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.island_id_tex.use(location=7)
            self._run_island_reservation_indirect(
                world,
                resources,
                program,
                "falling island apply index build",
                reservation_capacity=int(reservation_count),
                invocations_per_group=1,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

    def _dispatch_index_falling_island_materialization(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        *,
        reservation_count: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._clear_falling_island_index(
            world,
            resources,
            pass_name="island_materialization_index_clear",
            clear_flags=FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION,
            reservation_count=int(reservation_count),
        )
        with self._profile_pass(world, "island_materialization_index_build"):
            program = self.programs["fill_falling_island_materialization_index"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            resources.island_reservation_count.bind_to_storage_buffer(binding=1)
            resources.island_materialization_index.bind_to_storage_buffer(binding=2)
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.island_id_tex.use(location=7)
            self._run_island_reservation_indirect(
                world,
                resources,
                program,
                "falling island materialization index build",
                reservation_capacity=int(reservation_count),
                invocations_per_group=1,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

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
        with self._profile_pass(world, "integrate_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "integrate_load_bridge_inputs"):
            self._load_authoritative_integrate_inputs(world, resources, group_x, group_y)
        program = self.programs["integrate_velocity"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["gas_cell_size"].value = world.gas_cell_size
        program["dt"].value = dt
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        resources.material_tex.use(location=1)
        resources.velocity_tex.use(location=2)
        resources.flow_tex.use(location=3)
        resources.active_tile_tex.use(location=4)
        resources.velocity_out_tex.bind_to_image(5, read=False, write=True)
        with self._profile_pass(world, "integrate_velocity"):
            self._run_active_tile_indirect(program, resources, "integrate velocity")
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "integrate_publish_bridge"):
            active_tile_indirect = self._formal_gpu_frame(world)
            if not self._publish_bridge_velocity_words(
                world,
                resources,
                active_tile_indirect=active_tile_indirect,
            ):
                self._publish_bridge_outputs(
                    world,
                    resources,
                    output_textures=False,
                    velocity_out_active_only=True,
                    active_tile_indirect=active_tile_indirect,
                )
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
        dt: float,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        del group_x, group_y
        program = self.programs["powder_targets"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_powder"].value = int(Phase.POWDER)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["dt"].value = float(dt)
        use_bridge_blockers = self._bridge_authoritative_cell_blockers(world)
        program["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
        use_liquid_flow_intent = (
            self._formal_gpu_frame(world)
            and "liquid_flow_intent" in world.bridge.gpu_authoritative_resources
            and self._bridge_context_active(world)
        )
        program["use_liquid_flow_intent"].value = bool(use_liquid_flow_intent)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        if use_bridge_blockers:
            self._bind_bridge_cell_blockers(world, cell_binding=8)
        resources.material_tex.use(location=1)
        resources.phase_tex.use(location=2)
        resources.velocity_tex.use(location=3)
        resources.active_tile_tex.use(location=4)
        resources.entity_id_tex.use(location=6)
        resources.displaced_tex.use(location=7)
        if use_liquid_flow_intent:
            world.bridge.textures["liquid_flow_intent"].use(location=8)
        else:
            resources.velocity_tex.use(location=8)
        resources.powder_target_tex.bind_to_image(5, read=False, write=True)
        self._run_active_tile_indirect(program, resources, "powder target generation")
        self._sync_compute_writes(ctx)

    def _run_generate_powder_reservations(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        dt: float,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        generate = self.programs["generate_powder_reservations"]
        generate["cell_grid_size"].value = (world.width, world.height)
        generate["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        generate["tile_size"].value = world.active.tile_size
        generate["phase_powder"].value = int(Phase.POWDER)
        generate["phase_liquid"].value = int(Phase.LIQUID)
        generate["dt"].value = float(dt)
        resources.powder_reservations.bind_to_storage_buffer(binding=0)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
        resources.material_params.bind_to_storage_buffer(binding=2)
        resources.material_contact_params.bind_to_storage_buffer(binding=3)
        resources.active_tile_count.bind_to_storage_buffer(binding=4)
        resources.active_tile_list.bind_to_storage_buffer(binding=5)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.velocity_tex.use(location=2)
        resources.active_tile_tex.use(location=3)
        resources.powder_target_tex.use(location=4)
        self._run_active_tile_indirect(generate, resources, "powder reservation generation")
        self._sync_compute_writes(ctx)

    def _download_outputs(self, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
        return np.rint(
            np.frombuffer(resources.powder_target_tex.read(), dtype="f4").reshape((world.height, world.width, 4))[
                :, :, :2
            ]
        ).astype(np.int32)

    def _download_velocity_output(self, world: "WorldEngine", resources: GPUMotionResources) -> np.ndarray:
        velocity = world.velocity.copy()
        velocity_out = np.frombuffer(resources.velocity_out_tex.read(), dtype="f4").reshape(world.velocity.shape)
        active_tiles = np.frombuffer(resources.active_tile_tex.read(), dtype="f4").reshape(
            (world.active.tile_height, world.active.tile_width)
        )
        tile_size = int(world.active.tile_size)
        for tile_y, tile_x in np.argwhere(active_tiles > 0.5):
            x0 = int(tile_x) * tile_size
            y0 = int(tile_y) * tile_size
            x1 = min(world.width, x0 + tile_size)
            y1 = min(world.height, y0 + tile_size)
            velocity[y0:y1, x0:x1] = velocity_out[y0:y1, x0:x1]
        return velocity

    def plan_powder_reservations(
        self,
        world: "WorldEngine",
        dt: float,
        *,
        solve_tile_mask: np.ndarray,
        solve_cell_mask: np.ndarray,
    ) -> np.ndarray:
        powder_targets = self.step(world, dt, solve_tile_mask=solve_tile_mask)
        reservations = self._build_powder_reservations(world, solve_cell_mask, powder_targets, dt)
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
        dt: float,
        *,
        solve_tile_mask: np.ndarray,
    ) -> np.ndarray:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU motion pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "powder_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "powder_load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)
        with self._profile_pass(world, "powder_targets"):
            self._run_powder_targets(world, resources, group_x, group_y, dt)
        with self._profile_pass(world, "powder_buffer_prepare"):
            cell_count = world.width * world.height
            self._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_reservations",
                cell_count * POWDER_RESERVATION_DTYPE.itemsize,
            )
            self._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_target_winner",
                cell_count * np.dtype(np.int32).itemsize,
            )
            self._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_apply_incoming",
                cell_count * np.dtype(np.int32).itemsize,
            )
            self._ensure_dynamic_buffer_capacity(
                ctx,
                resources,
                "powder_apply_outgoing",
                cell_count * np.dtype(np.int32).itemsize,
            )
            resources.powder_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
        with self._profile_pass(world, "powder_generate"):
            self._run_generate_powder_reservations(world, resources, dt)
        with self._profile_pass(world, "powder_index_targets"):
            self._clear_powder_target_winners_for_reservations(world, resources)

            index_winners = self.programs["index_powder_target_winners"]
            index_winners["cell_grid_size"].value = (world.width, world.height)
            index_winners["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            index_winners["tile_size"].value = world.active.tile_size
            index_winners["phase_powder"].value = int(Phase.POWDER)
            index_winners["phase_liquid"].value = int(Phase.LIQUID)
            index_winners["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            use_bridge_blockers = self._bridge_authoritative_cell_blockers(world)
            index_winners["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            resources.powder_target_winner.bind_to_storage_buffer(binding=2)
            if use_bridge_blockers:
                self._bind_bridge_cell_blockers(world, cell_binding=8)
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.active_tile_tex.use(location=3)
            resources.entity_id_tex.use(location=5)
            resources.displaced_tex.use(location=6)
            self._run_powder_reservation_indirect(
                world,
                resources,
                index_winners,
                "powder target winner indexing",
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        with self._profile_pass(world, "powder_resolve"):
            resolve = self.programs["resolve_powder_reservations"]
            resolve["cell_grid_size"].value = (world.width, world.height)
            resolve["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            resolve["tile_size"].value = world.active.tile_size
            resolve["phase_powder"].value = int(Phase.POWDER)
            resolve["phase_liquid"].value = int(Phase.LIQUID)
            resolve["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            use_bridge_blockers = self._bridge_authoritative_cell_blockers(world)
            resolve["use_bridge_authoritative_blockers"].value = bool(use_bridge_blockers)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            resources.material_params.bind_to_storage_buffer(binding=2)
            resources.material_contact_params.bind_to_storage_buffer(binding=3)
            resources.powder_target_winner.bind_to_storage_buffer(binding=4)
            if use_bridge_blockers:
                self._bind_bridge_cell_blockers(world, cell_binding=8)
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.active_tile_tex.use(location=3)
            resources.entity_id_tex.use(location=5)
            resources.displaced_tex.use(location=6)
            self._run_powder_reservation_indirect(
                world,
                resources,
                resolve,
                "powder reservation resolve",
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if self._formal_gpu_frame(world):
            self.publish_bridge_powder_reservations(world, world.width * world.height)
            self._dispatch_index_powder_apply(world, resources)
            self._dispatch_apply_powder_reservations(world, resources, None)
            return np.zeros((0,), dtype=POWDER_RESERVATION_DTYPE)
        reservation_count = int(np.frombuffer(resources.powder_reservation_count.read(size=4), dtype=np.int32, count=1)[0])
        reservation_count = max(0, min(reservation_count, world.width * world.height))
        self._dispatch_index_powder_apply(world, resources)
        self._dispatch_apply_powder_reservations(world, resources, reservation_count)
        return self._read_powder_reservations(resources, reservation_count)

    def _dispatch_apply_powder_fast_path(
        self,
        world: "WorldEngine",
        resources: GPUMotionResources,
        group_x: int,
        group_y: int,
        dt: float,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["apply_powder_fast_path"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["phase_powder"].value = int(Phase.POWDER)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["max_powder_step"].value = 3
        program["dt"].value = float(dt)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_contact_params.bind_to_storage_buffer(binding=1)
        resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=11)
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
        world.bridge._ensure_active_scheduler_programs()
        world.bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        resources.powder_reservation_count.write(np.array([0], dtype=np.int32).tobytes())
        self.publish_bridge_powder_reservations(world, 0)
        self.last_cpu_mirror_downloaded = False

    def apply_powder_reservations(self, world: "WorldEngine", reservations: np.ndarray) -> bool:
        ctx = world.bridge.ctx
        if ctx is None:
            return False
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.upload_powder_reservations(world, reservations)
        self._dispatch_index_powder_apply(world, resources)
        self._dispatch_apply_powder_reservations(world, resources, int(len(reservations)))
        return True

    def _dispatch_index_powder_apply(self, world: "WorldEngine", resources: GPUMotionResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        cell_count = int(world.width * world.height)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "powder_target_winner", cell_count * np.dtype(np.int32).itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_incoming", cell_count * np.dtype(np.int32).itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "powder_apply_outgoing", cell_count * np.dtype(np.int32).itemsize)
        with self._profile_pass(world, "powder_index_apply"):
            if self._formal_gpu_frame(world):
                self._build_powder_apply_dispatch(world, resources)
                self._clear_powder_apply_index_for_active_tiles(world, resources)
                self._clear_powder_apply_index_for_reservations(world, resources)
            else:
                clear_apply = self.programs["clear_powder_apply_index"]
                clear_apply["cell_count"].value = cell_count
                resources.powder_apply_incoming.bind_to_storage_buffer(binding=0)
                resources.powder_apply_outgoing.bind_to_storage_buffer(binding=1)
                clear_apply.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

                clear_winners = self.programs["clear_powder_target_winners"]
                clear_winners["cell_count"].value = cell_count
                resources.powder_target_winner.bind_to_storage_buffer(binding=0)
                clear_winners.run((cell_count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

            index_winners = self.programs["index_powder_apply_winners"]
            index_winners["cell_grid_size"].value = (world.width, world.height)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            resources.powder_target_winner.bind_to_storage_buffer(binding=2)
            self._run_powder_reservation_indirect(
                world,
                resources,
                index_winners,
                "powder apply winner indexing",
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

            fill_index = self.programs["fill_powder_apply_index"]
            fill_index["cell_grid_size"].value = (world.width, world.height)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=1)
            resources.powder_target_winner.bind_to_storage_buffer(binding=2)
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=3)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=4)
            self._run_powder_reservation_indirect(
                world,
                resources,
                fill_index,
                "powder apply index fill",
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)

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
        if not self._formal_gpu_frame(world):
            resources.island_reservation_count.write(np.array([int(reservation_count)], dtype=np.int32).tobytes())
        self._dispatch_apply_falling_island_reservations(world, resources, int(reservation_count))
        if self._formal_gpu_frame(world):
            self._dispatch_apply_falling_island_materialization(
                world,
                resources,
                reservation_count=int(reservation_count),
                mode=0,
                use_existing_active_tile_dispatch=True,
            )
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
        if not self._formal_gpu_frame(world):
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
        formal_frame = self._formal_gpu_frame(world)
        self._upload_powder_apply_state(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if formal_frame:
            self._build_powder_apply_dispatch(world, resources)
        self._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=formal_frame,
        )
        with self._profile_pass(world, "powder_apply_main"):
            program = self.programs["apply_powder_reservations"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
            program_members = getattr(program, "_members", {})
            if "reservation_count" in program_members:
                program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
            if "use_reservation_count_buffer" in program_members:
                program["use_reservation_count_buffer"].value = reservation_count is None
            if "use_active_tile_dispatch" in program_members:
                program["use_active_tile_dispatch"].value = bool(formal_frame)
            if "skip_untouched_original_stores" in program_members:
                program["skip_untouched_original_stores"].value = bool(formal_frame)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.material_contact_params.bind_to_storage_buffer(binding=1)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
            world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
            resources.active_tile_count.bind_to_storage_buffer(binding=6)
            resources.active_tile_list.bind_to_storage_buffer(binding=7)
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
            if formal_frame:
                self._run_active_tile_indirect(program, resources, "powder reservation apply")
            else:
                program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)

        with self._profile_pass(world, "powder_apply_aux"):
            aux_program = self.programs["apply_powder_reservation_aux"]
            aux_program["cell_grid_size"].value = (world.width, world.height)
            aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            aux_program["tile_size"].value = int(world.active.tile_size)
            aux_members = getattr(aux_program, "_members", {})
            if "reservation_count" in aux_members:
                aux_program["reservation_count"].value = 0 if reservation_count is None else int(reservation_count)
            if "use_reservation_count_buffer" in aux_members:
                aux_program["use_reservation_count_buffer"].value = reservation_count is None
            if "use_active_tile_dispatch" in aux_members:
                aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
            resources.powder_reservations.bind_to_storage_buffer(binding=0)
            resources.powder_reservation_count.bind_to_storage_buffer(binding=2)
            resources.powder_apply_incoming.bind_to_storage_buffer(binding=4)
            resources.powder_apply_outgoing.bind_to_storage_buffer(binding=5)
            resources.active_tile_count.bind_to_storage_buffer(binding=6)
            resources.active_tile_list.bind_to_storage_buffer(binding=7)
            resources.island_id_tex.use(location=7)
            resources.entity_id_tex.use(location=8)
            resources.displaced_tex.use(location=9)
            resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
            if formal_frame:
                self._run_active_tile_indirect(aux_program, resources, "powder reservation aux apply")
            else:
                aux_program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        self._publish_bridge_outputs(
            world,
            resources,
            output_textures=True,
            active_tile_indirect=formal_frame,
            use_powder_apply_touch_sources=formal_frame,
        )
        self._refresh_authoritative_active_scheduler_after_apply(world, "active_refresh_after_powder")
        self.last_cpu_mirror_downloaded = not formal_frame
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
        inputs_already_loaded: bool = False,
        use_existing_active_tile_dispatch: bool = False,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        formal_frame = self._formal_gpu_frame(world)
        if formal_frame and int(mode) == 0:
            self._sync_compute_writes(ctx)
        self._upload_powder_apply_state(world, resources)
        self._upload_material_rule_params(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        formal_mode_zero = formal_frame and int(mode) == 0
        if formal_frame:
            if formal_mode_zero and inputs_already_loaded:
                if not self._active_scheduler_gpu_authoritative(world):
                    self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
                self._compact_active_tiles(world, resources)
            elif formal_mode_zero and not self._active_scheduler_gpu_authoritative(world):
                self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
                self._compact_active_tiles(world, resources)
            elif not formal_mode_zero:
                self._build_falling_island_apply_dispatch(
                    world,
                    resources,
                    reservation_count=int(reservation_count),
                    operation=1,
                )
        if not inputs_already_loaded:
            reuse_active_dispatch = bool(
                formal_frame and (not formal_mode_zero or use_existing_active_tile_dispatch)
            )
            self._load_authoritative_bridge_inputs(
                world,
                resources,
                group_x,
                group_y,
                use_existing_active_tile_dispatch=reuse_active_dispatch,
            )
        materialization_tile_count_buffer = resources.active_tile_count
        materialization_tile_list_buffer = resources.active_tile_list
        materialization_dispatch_args = resources.active_tile_dispatch_args
        if formal_mode_zero:
            self._build_falling_island_materialization_candidate_dispatch(world, resources)
            materialization_tile_count_buffer = resources.island_materialization_candidate_tile_count
            materialization_tile_list_buffer = resources.island_materialization_candidate_tile_list
            materialization_dispatch_args = resources.island_materialization_candidate_dispatch_args
        if int(mode) != 0:
            self._dispatch_index_falling_island_materialization(
                world,
                resources,
                reservation_count=int(reservation_count),
            )
        program = self.programs["apply_falling_island_materialization"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["reservation_count"].value = int(reservation_count)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["phase_powder"].value = int(Phase.POWDER)
        program["phase_static_solid"].value = int(Phase.STATIC_SOLID)
        program["mode"].value = int(mode)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["use_reservation_count_buffer"].value = bool(formal_frame)
        program["use_active_tile_dispatch"].value = bool(formal_frame)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_falling_params.bind_to_storage_buffer(binding=1)
        resources.island_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.island_materialization_index.bind_to_storage_buffer(binding=4)
        materialization_tile_count_buffer.bind_to_storage_buffer(binding=6)
        materialization_tile_list_buffer.bind_to_storage_buffer(binding=7)
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
        with self._profile_pass(world, "island_materialization_main"):
            if formal_frame:
                self._run_active_tile_indirect(
                    program,
                    resources,
                    "falling island materialization",
                    dispatch_args=materialization_dispatch_args,
                )
            else:
                program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)

        aux_program = self.programs["apply_falling_island_materialization_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        aux_program["tile_size"].value = int(world.active.tile_size)
        aux_program["reservation_count"].value = int(reservation_count)
        aux_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        aux_program["mode"].value = int(mode)
        aux_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        aux_program["use_reservation_count_buffer"].value = bool(formal_frame)
        aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.material_falling_params.bind_to_storage_buffer(binding=1)
        resources.island_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.island_materialization_index.bind_to_storage_buffer(binding=4)
        materialization_tile_count_buffer.bind_to_storage_buffer(binding=6)
        materialization_tile_list_buffer.bind_to_storage_buffer(binding=7)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        with self._profile_pass(world, "island_materialization_aux"):
            if formal_frame:
                self._run_active_tile_indirect(
                    aux_program,
                    resources,
                    "falling island materialization aux",
                    dispatch_args=materialization_dispatch_args,
                )
            else:
                aux_program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "island_materialization_bridge_publish"):
            self._publish_bridge_outputs(
                world,
                resources,
                output_textures=True,
                active_tile_indirect=formal_frame,
                active_tile_count_buffer=materialization_tile_count_buffer,
                active_tile_list_buffer=materialization_tile_list_buffer,
                active_tile_dispatch_args=materialization_dispatch_args,
            )
        self._refresh_authoritative_active_scheduler_after_apply(
            world,
            "active_refresh_after_falling_island_materialization",
        )
        self.last_cpu_mirror_downloaded = not formal_frame
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
        formal_frame = self._formal_gpu_frame(world)
        self._upload_powder_apply_state(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if formal_frame:
            self._build_falling_island_apply_dispatch(
                world,
                resources,
                reservation_count=int(reservation_count),
                operation=0,
            )
        self._load_authoritative_bridge_inputs(
            world,
            resources,
            group_x,
            group_y,
            use_existing_active_tile_dispatch=formal_frame,
        )
        self._dispatch_index_falling_island_apply(
            world,
            resources,
            reservation_count=int(reservation_count),
        )
        program = self.programs["apply_falling_island_reservations"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["gas_cell_size"].value = world.gas_cell_size
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["reservation_count"].value = int(reservation_count)
        program["use_reservation_count_buffer"].value = bool(formal_frame)
        program["use_active_tile_dispatch"].value = bool(formal_frame)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.island_apply_incoming.bind_to_storage_buffer(binding=4)
        resources.island_apply_outgoing.bind_to_storage_buffer(binding=5)
        resources.active_tile_count.bind_to_storage_buffer(binding=6)
        resources.active_tile_list.bind_to_storage_buffer(binding=7)
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
        with self._profile_pass(world, "island_apply_main"):
            if formal_frame:
                self._run_active_tile_indirect(program, resources, "falling island reservation apply")
            else:
                program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)

        aux_program = self.programs["apply_falling_island_reservation_aux"]
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        aux_program["tile_size"].value = int(world.active.tile_size)
        aux_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        aux_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        aux_program["reservation_count"].value = int(reservation_count)
        aux_program["use_reservation_count_buffer"].value = bool(formal_frame)
        aux_program["use_active_tile_dispatch"].value = bool(formal_frame)
        resources.island_reservations.bind_to_storage_buffer(binding=0)
        resources.island_reservation_count.bind_to_storage_buffer(binding=2)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.island_apply_incoming.bind_to_storage_buffer(binding=4)
        resources.island_apply_outgoing.bind_to_storage_buffer(binding=5)
        resources.active_tile_count.bind_to_storage_buffer(binding=6)
        resources.active_tile_list.bind_to_storage_buffer(binding=7)
        resources.material_tex.use(location=0)
        resources.phase_tex.use(location=1)
        resources.island_id_tex.use(location=7)
        resources.entity_id_tex.use(location=8)
        resources.displaced_tex.use(location=9)
        resources.island_id_out_tex.bind_to_image(0, read=False, write=True)
        resources.entity_id_out_tex.bind_to_image(1, read=False, write=True)
        resources.displaced_out_tex.bind_to_image(2, read=False, write=True)
        with self._profile_pass(world, "island_apply_aux"):
            if formal_frame:
                self._run_active_tile_indirect(aux_program, resources, "falling island reservation aux apply")
            else:
                aux_program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "island_apply_bridge_publish"):
            self._publish_bridge_outputs(world, resources, output_textures=True, active_tile_indirect=formal_frame)
        self._refresh_authoritative_active_scheduler_after_apply(
            world,
            "active_refresh_after_falling_island_reservation",
        )
        self.last_cpu_mirror_downloaded = not formal_frame
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
        dt: float,
    ) -> np.ndarray:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        reservations: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[float, float], int]] = []
        for y in range(world.height - 2, -1, -1):
            active_xs = np.flatnonzero(solve_cell_mask[y])
            if active_xs.size == 0:
                continue
            for x in active_xs.tolist():
                material_id = int(world.material_id[y, x])
                phase_id = int(world.phase[y, x])
                if material_id <= 0 or phase_id not in (int(Phase.POWDER), int(Phase.LIQUID)):
                    continue
                max_step = 0
                if material_id < material_table.shape[0]:
                    max_step = int(material_table[material_id]["max_dda_step"])
                velocity = world.velocity[y, x]
                frame_delta_x = float(velocity[0]) * float(dt)
                frame_delta_y = float(velocity[1]) * float(dt)
                desired_dx = int(np.clip(np.rint(frame_delta_x), -max_step, max_step))
                desired_dy = int(np.clip(np.rint(frame_delta_y), -max_step, max_step))
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
        dt: float,
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
        self._upload_material_rule_params(world, resources)
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
        program["use_island_count_buffer"].value = False
        use_bridge_state = self._bridge_authoritative_island_state(world)
        program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
        program["dt"].value = float(dt)
        resources.material_tex.use(location=0)
        resources.island_id_tex.use(location=1)
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        resources.material_params.bind_to_storage_buffer(binding=5)
        if use_bridge_state:
            self._bind_bridge_island_state(world, cell_binding=7)
        group_x = (runtime.shape[0] + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        pack_program = self.programs["pack_falling_island_reservations"]
        pack_program["island_count"].value = int(runtime.shape[0])
        pack_program["use_island_count_buffer"].value = False
        resources.island_ids.bind_to_storage_buffer(binding=0)
        resources.island_bboxes.bind_to_storage_buffer(binding=1)
        resources.island_motion.bind_to_storage_buffer(binding=2)
        resources.island_shift_results.bind_to_storage_buffer(binding=3)
        resources.island_reservations.bind_to_storage_buffer(binding=4)
        resources.island_reservation_count.bind_to_storage_buffer(binding=5)
        pack_group_x = (runtime.shape[0] + ISLAND_RESERVATION_LINEAR_LOCAL_SIZE - 1) // ISLAND_RESERVATION_LINEAR_LOCAL_SIZE
        pack_program.run(pack_group_x, 1, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        return int(runtime.shape[0])

    def plan_uploaded_falling_island_reservations_from_bridge_runtime(
        self,
        world: "WorldEngine",
        dt: float,
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
        formal_frame = self._formal_gpu_frame(world)
        if formal_frame:
            with self._profile_pass(world, "island_shift_planning"):
                self._ensure_bridge_runtime_reservation_capacity(ctx, resources, runtime_capacity)
                upload_plan = self._cpu_upload_plan(world)
                self._record_cpu_upload_plan(upload_plan)
                if upload_plan["cell_core"]:
                    resources.material_tex.write(world.material_id.astype("f4").tobytes())
                if upload_plan["island_id"]:
                    resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
                cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
                cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
                self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
                self._upload_material_rule_params(world, resources)
                program = self.programs["plan_bridge_runtime_falling_island_reservations"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["runtime_capacity"].value = runtime_capacity
                use_bridge_state = self._bridge_authoritative_island_state(world)
                program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
                program["dt"].value = float(dt)
                resources.material_tex.use(location=0)
                resources.island_id_tex.use(location=1)
                bridge.buffers["island_runtime"].bind_to_storage_buffer(binding=0)
                resources.island_reservations.bind_to_storage_buffer(binding=1)
                resources.island_reservation_count.bind_to_storage_buffer(binding=2)
                bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=3)
                resources.material_params.bind_to_storage_buffer(binding=4)
                before_plan_run = None
                if use_bridge_state:
                    def rebind_bridge_island_state() -> None:
                        self._bind_bridge_island_state(world, cell_binding=7)

                    before_plan_run = rebind_bridge_island_state
                self._run_island_runtime_indirect(
                    world,
                    resources,
                    program,
                    "bridge runtime falling island reservation planning",
                    runtime_capacity=runtime_capacity,
                    before_run=before_plan_run,
                )
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
            # The returned value is only a buffer capacity upper bound. The actual
            # reservation count remains GPU-authoritative in island_reservation_count.
            return runtime_capacity

        with self._profile_pass(world, "island_runtime_unpack"):
            self._ensure_bridge_runtime_planning_capacity(ctx, resources, runtime_capacity)
            resources.island_reservation_count.write(np.array([0], dtype=np.int32).tobytes())

            unpack_program = self.programs["unpack_bridge_island_runtime"]
            unpack_program["runtime_capacity"].value = runtime_capacity
            bridge.buffers["island_runtime"].bind_to_storage_buffer(binding=0)
            resources.island_ids.bind_to_storage_buffer(binding=1)
            resources.island_bboxes.bind_to_storage_buffer(binding=2)
            resources.island_motion.bind_to_storage_buffer(binding=3)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=4)
            self._run_island_runtime_indirect(
                world,
                resources,
                unpack_program,
                "bridge island runtime unpack",
                runtime_capacity=runtime_capacity,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        with self._profile_pass(world, "island_shift_planning"):
            upload_plan = self._cpu_upload_plan(world)
            self._record_cpu_upload_plan(upload_plan)
            if upload_plan["cell_core"]:
                resources.material_tex.write(world.material_id.astype("f4").tobytes())
            if upload_plan["island_id"]:
                resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
            cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
            cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
            self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
            self._upload_material_rule_params(world, resources)
            program = self.programs["island_shifts"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["island_count"].value = runtime_capacity
            program["use_island_count_buffer"].value = True
            use_bridge_state = self._bridge_authoritative_island_state(world)
            program["use_bridge_authoritative_state"].value = bool(use_bridge_state)
            program["dt"].value = float(dt)
            resources.material_tex.use(location=0)
            resources.island_id_tex.use(location=1)
            resources.island_ids.bind_to_storage_buffer(binding=0)
            resources.island_bboxes.bind_to_storage_buffer(binding=1)
            resources.island_motion.bind_to_storage_buffer(binding=2)
            resources.island_shift_results.bind_to_storage_buffer(binding=3)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=4)
            resources.material_params.bind_to_storage_buffer(binding=5)
            before_shift_run = None
            if use_bridge_state:
                def rebind_bridge_island_state() -> None:
                    self._bind_bridge_island_state(world, cell_binding=7)

                before_shift_run = rebind_bridge_island_state
            self._run_island_runtime_indirect(
                world,
                resources,
                program,
                "bridge island shift planning",
                runtime_capacity=runtime_capacity,
                before_run=before_shift_run,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)

        with self._profile_pass(world, "island_reservation_packing"):
            pack_program = self.programs["pack_falling_island_reservations"]
            pack_program["island_count"].value = runtime_capacity
            pack_program["use_island_count_buffer"].value = True
            resources.island_ids.bind_to_storage_buffer(binding=0)
            resources.island_bboxes.bind_to_storage_buffer(binding=1)
            resources.island_motion.bind_to_storage_buffer(binding=2)
            resources.island_shift_results.bind_to_storage_buffer(binding=3)
            resources.island_reservations.bind_to_storage_buffer(binding=4)
            resources.island_reservation_count.bind_to_storage_buffer(binding=5)
            bridge.buffers["island_runtime_count"].bind_to_storage_buffer(binding=6)
            self._run_island_runtime_indirect(
                world,
                resources,
                pack_program,
                "bridge island reservation packing",
                runtime_capacity=runtime_capacity,
                invocations_per_group=ISLAND_RESERVATION_LINEAR_LOCAL_SIZE,
            )
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        # The returned value is only a buffer capacity upper bound. The actual
        # reservation count remains GPU-authoritative in island_reservation_count.
        return runtime_capacity

    def _ensure_bridge_runtime_reservation_capacity(
        self,
        ctx: Any,
        resources: GPUMotionResources,
        runtime_capacity: int,
    ) -> None:
        runtime_capacity = max(0, int(runtime_capacity))
        self._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "island_reservations",
            runtime_capacity * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
        )

    def _ensure_bridge_runtime_planning_capacity(
        self,
        ctx: Any,
        resources: GPUMotionResources,
        runtime_capacity: int,
    ) -> None:
        runtime_capacity = max(0, int(runtime_capacity))
        int_itemsize = np.dtype(np.int32).itemsize
        float_itemsize = np.dtype(np.float32).itemsize
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_ids", runtime_capacity * int_itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_bboxes", runtime_capacity * 4 * int_itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_motion", runtime_capacity * 4 * float_itemsize)
        self._ensure_dynamic_buffer_capacity(ctx, resources, "island_shift_results", runtime_capacity * 4 * int_itemsize)
        self._ensure_dynamic_buffer_capacity(
            ctx,
            resources,
            "island_reservations",
            runtime_capacity * FALLING_ISLAND_RESERVATION_DTYPE.itemsize,
        )

    def plan_falling_island_reservations(
        self,
        world: "WorldEngine",
        dt: float,
        *,
        island_ids: list[int] | None = None,
        motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        reservation_count = self.plan_uploaded_falling_island_reservations(
            world,
            dt,
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

    def resolve_uploaded_falling_island_reservations(
        self,
        world: "WorldEngine",
        reservation_count: int,
    ) -> bool:
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
        if not self._formal_gpu_frame(world):
            resources.island_reservation_count.write(np.array([reservation_count], dtype=np.int32).tobytes())
        self._dispatch_resolve_falling_island_reservations(world, resources, reservation_count)
        if not self._formal_gpu_frame(world):
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
        formal_frame = self._formal_gpu_frame(world)
        self._upload_material_rule_params(world, resources)
        upload_plan = self._cpu_upload_plan(world)
        self._record_cpu_upload_plan(upload_plan)
        if upload_plan["cell_core"]:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
        if upload_plan["island_id"]:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if formal_frame:
            self._build_falling_island_apply_dispatch(
                world,
                resources,
                reservation_count=int(reservation_count),
                operation=2,
            )
        self._load_authoritative_bridge_inputs(
            world,
            resources,
            cell_group_x,
            cell_group_y,
            use_existing_active_tile_dispatch=formal_frame,
        )
        self._dispatch_index_falling_island_reservation_sources(
            world,
            resources,
            reservation_count=int(reservation_count),
        )
        with self._profile_pass(world, "island_reservation_resolve"):
            program = self.programs["resolve_falling_island_reservations"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["reservation_count"].value = int(reservation_count)
            program["use_reservation_count_buffer"].value = bool(formal_frame)
            resources.island_reservations.bind_to_storage_buffer(binding=0)
            resources.material_contact_params.bind_to_storage_buffer(binding=1)
            resources.island_reservation_count.bind_to_storage_buffer(binding=2)
            resources.island_reservation_source_index.bind_to_storage_buffer(binding=3)
            resources.material_tex.use(location=0)
            resources.island_id_tex.use(location=1)
            if formal_frame:
                self._run_island_reservation_indirect(
                    world,
                    resources,
                    program,
                    "falling island reservation resolve",
                    reservation_capacity=int(reservation_count),
                )
            else:
                group_x = (int(reservation_count) + LOCAL_SIZE - 1) // LOCAL_SIZE
                program.run(group_x, 1, 1)
            ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
        if not formal_frame:
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
        dt: float,
        *,
        island_ids: list[int] | None = None,
        motion_overrides: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> dict[int, tuple[int, int]]:
        reservations = self.plan_falling_island_reservations(
            world,
            dt,
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
