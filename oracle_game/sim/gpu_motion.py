from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE, pack_island_runtime_upload
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader
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


# Substitution markers shared by every shader in ``shaders/motion/``.  Passing
# the same superset dict to every ``build_compute_shader`` call is cheap (the
# loader only touches markers actually present in each file) and keeps the
# call sites uniform.  Derived entries mirror the inline Python expressions
# the original f-strings used (e.g. ``{LOCAL_SIZE - 1}`` -> ``{{LOCAL_SIZE_MINUS_1}}``).
_SHADER_SUBS: dict[str, object] = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "LOCAL_SIZE_MINUS_1": LOCAL_SIZE - 1,
    "POWDER_RESERVATION_LOCAL_SIZE": POWDER_RESERVATION_LOCAL_SIZE,
    "ISLAND_RESERVATION_LINEAR_LOCAL_SIZE": ISLAND_RESERVATION_LINEAR_LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
    "MAX_ISLAND_DDA_STEP": MAX_ISLAND_DDA_STEP,
    "INDEX_EMPTY": INDEX_EMPTY,
    "FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING": FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING,
    "FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING": FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING,
    "FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION": FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION,
    "FALLING_ISLAND_INDEX_CLEAR_SOURCE": FALLING_ISLAND_INDEX_CLEAR_SOURCE,
    "FALLING_ISLAND_BREAK_STABLE": FALLING_ISLAND_BREAK_STABLE,
    "POWDER_RESOLVE_BLOCKED": POWDER_RESOLVE_BLOCKED,
    "POWDER_RESOLVE_DDA": POWDER_RESOLVE_DDA,
    "POWDER_RESOLVE_FALLBACK": POWDER_RESOLVE_FALLBACK,
    "POWDER_RESOLVE_STALE": POWDER_RESOLVE_STALE,
    "POWDER_SOLVER_SUSPENDED": POWDER_SOLVER_SUSPENDED,
    "ISLAND_RESOLVE_BLOCKED": ISLAND_RESOLVE_BLOCKED,
    "ISLAND_RESOLVE_DIRECT": ISLAND_RESOLVE_DIRECT,
    "ISLAND_RESOLVE_RERESOLVED": ISLAND_RESOLVE_RERESOLVED,
    "ISLAND_RESOLVE_STALE": ISLAND_RESOLVE_STALE,
    "ISLAND_RUNTIME_WORDS": ISLAND_RUNTIME_DTYPE.itemsize // 4,
}


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


class GPUMotionPipeline(GPUPipelineBase):
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

    def _reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``available`` inherited from GPUPipelineBase.

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
        self.programs["load_active_tiles"] = build_compute_shader(ctx, "motion/load_active_tiles.comp", _SHADER_SUBS)
        self.programs["clear_active_tile_dispatch"] = build_compute_shader(ctx, "motion/clear_active_tile_dispatch.comp", _SHADER_SUBS)
        self.programs["compact_active_tiles"] = build_compute_shader(ctx, "motion/compact_active_tiles.comp", _SHADER_SUBS)
        self.programs["compact_active_tiles_from_chunks"] = build_compute_shader(ctx, "motion/compact_active_tiles_from_chunks.comp", _SHADER_SUBS)
        self.programs["build_falling_island_materialization_candidate_dispatch"] = build_compute_shader(ctx, "motion/build_falling_island_materialization_candidate_dispatch.comp", _SHADER_SUBS)
        self.programs["build_powder_reservation_dispatch"] = build_compute_shader(ctx, "motion/build_powder_reservation_dispatch.comp", _SHADER_SUBS)
        self.programs["build_island_runtime_dispatch"] = build_compute_shader(ctx, "motion/build_island_runtime_dispatch.comp", _SHADER_SUBS)
        self.programs["clear_powder_affected_tile_dispatch"] = build_compute_shader(ctx, "motion/clear_powder_affected_tile_dispatch.comp", _SHADER_SUBS)
        self.programs["build_powder_apply_dispatch"] = build_compute_shader(ctx, "motion/build_powder_apply_dispatch.comp", _SHADER_SUBS)
        self.programs["build_falling_island_apply_dispatch"] = build_compute_shader(ctx, "motion/build_falling_island_apply_dispatch.comp", _SHADER_SUBS)
        self.programs["clear_powder_target_winners_for_reservations"] = build_compute_shader(ctx, "motion/clear_powder_target_winners_for_reservations.comp", _SHADER_SUBS)
        self.programs["clear_powder_apply_index_for_reservations"] = build_compute_shader(ctx, "motion/clear_powder_apply_index_for_reservations.comp", _SHADER_SUBS)
        self.programs["clear_powder_apply_index_for_active_tiles"] = build_compute_shader(ctx, "motion/clear_powder_apply_index_for_active_tiles.comp", _SHADER_SUBS)
        self.programs["integrate_velocity"] = build_compute_shader(ctx, "motion/integrate_velocity.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "motion/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_integrate_inputs"] = build_compute_shader(ctx, "motion/load_bridge_integrate_inputs.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux"] = build_compute_shader(ctx, "motion/load_bridge_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_gas"] = build_compute_shader(ctx, "motion/load_bridge_gas.comp", _SHADER_SUBS)
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "motion/publish_bridge_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_velocity_word"] = build_compute_shader(ctx, "motion/publish_bridge_velocity_word.comp", _SHADER_SUBS)
        self.programs["publish_bridge_island_id"] = build_compute_shader(ctx, "motion/publish_bridge_island_id.comp", _SHADER_SUBS)
        self.programs["copy_scalar_texture"] = build_compute_shader(ctx, "motion/copy_scalar_texture.comp", _SHADER_SUBS)
        self.programs["powder_targets"] = build_compute_shader(ctx, "motion/powder_targets.comp", _SHADER_SUBS)
        self.programs["island_component_init"] = build_compute_shader(ctx, "motion/island_component_init.comp", _SHADER_SUBS)
        self.programs["island_component_propagate"] = build_compute_shader(ctx, "motion/island_component_propagate.comp", _SHADER_SUBS)
        self.programs["relabel_falling_island_components"] = build_compute_shader(ctx, "motion/relabel_falling_island_components.comp", _SHADER_SUBS)
        self.programs["summarize_falling_island_components"] = build_compute_shader(ctx, "motion/summarize_falling_island_components.comp", _SHADER_SUBS)
        self.programs["island_shifts"] = build_compute_shader(ctx, "motion/island_shifts.comp", _SHADER_SUBS)
        self.programs["plan_bridge_runtime_falling_island_reservations"] = build_compute_shader(ctx, "motion/plan_bridge_runtime_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["pack_falling_island_reservations"] = build_compute_shader(ctx, "motion/pack_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["publish_falling_island_runtime"] = build_compute_shader(ctx, "motion/publish_falling_island_runtime.comp", _SHADER_SUBS)
        self.programs["publish_powder_reservations"] = build_compute_shader(ctx, "motion/publish_powder_reservations.comp", _SHADER_SUBS)
        self.programs["publish_falling_island_reservations"] = build_compute_shader(ctx, "motion/publish_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["unpack_bridge_island_runtime"] = build_compute_shader(ctx, "motion/unpack_bridge_island_runtime.comp", _SHADER_SUBS)
        self.programs["fill_falling_island_reservation_source_index"] = build_compute_shader(ctx, "motion/fill_falling_island_reservation_source_index.comp", _SHADER_SUBS)
        self.programs["resolve_falling_island_reservations"] = build_compute_shader(ctx, "motion/resolve_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["generate_powder_reservations"] = build_compute_shader(ctx, "motion/generate_powder_reservations.comp", _SHADER_SUBS)
        self.programs["clear_powder_target_winners"] = build_compute_shader(ctx, "motion/clear_powder_target_winners.comp", _SHADER_SUBS)
        self.programs["index_powder_target_winners"] = build_compute_shader(ctx, "motion/index_powder_target_winners.comp", _SHADER_SUBS)
        self.programs["resolve_powder_reservations"] = build_compute_shader(ctx, "motion/resolve_powder_reservations.comp", _SHADER_SUBS)
        self.programs["clear_powder_apply_index"] = build_compute_shader(ctx, "motion/clear_powder_apply_index.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index"] = build_compute_shader(ctx, "motion/clear_falling_island_index.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index_for_active_tiles"] = build_compute_shader(ctx, "motion/clear_falling_island_index_for_active_tiles.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index_for_reservations"] = build_compute_shader(ctx, "motion/clear_falling_island_index_for_reservations.comp", _SHADER_SUBS)
        self.programs["fill_falling_island_apply_index"] = build_compute_shader(ctx, "motion/fill_falling_island_apply_index.comp", _SHADER_SUBS)
        self.programs["fill_falling_island_materialization_index"] = build_compute_shader(ctx, "motion/fill_falling_island_materialization_index.comp", _SHADER_SUBS)
        self.programs["index_powder_apply_winners"] = build_compute_shader(ctx, "motion/index_powder_apply_winners.comp", _SHADER_SUBS)
        self.programs["fill_powder_apply_index"] = build_compute_shader(ctx, "motion/fill_powder_apply_index.comp", _SHADER_SUBS)
        self.programs["apply_powder_fast_path"] = build_compute_shader(ctx, "motion/apply_powder_fast_path.comp", _SHADER_SUBS)
        self.programs["apply_powder_reservations"] = build_compute_shader(ctx, "motion/apply_powder_reservations.comp", _SHADER_SUBS)
        self.programs["apply_powder_reservation_aux"] = build_compute_shader(ctx, "motion/apply_powder_reservation_aux.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_reservations"] = build_compute_shader(ctx, "motion/apply_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_reservation_aux"] = build_compute_shader(ctx, "motion/apply_falling_island_reservation_aux.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_materialization"] = build_compute_shader(ctx, "motion/apply_falling_island_materialization.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_materialization_aux"] = build_compute_shader(ctx, "motion/apply_falling_island_materialization_aux.comp", _SHADER_SUBS)

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

    # ``_formal_gpu_frame`` inherited from GPUPipelineBase.

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
        program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
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

    def _barrier_bits(self) -> tuple[str, ...]:
        # motion uses indirect dispatch + buffer updates in addition to the
        # default image/texture/storage sync.
        return (
            "SHADER_STORAGE_BARRIER_BIT",
            "SHADER_IMAGE_ACCESS_BARRIER_BIT",
            "TEXTURE_FETCH_BARRIER_BIT",
            "COMMAND_BARRIER_BIT",
            "BUFFER_UPDATE_BARRIER_BIT",
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
            self._dispatch_apply_powder_reservations(
                world,
                resources,
                None,
                inputs_already_loaded=True,
            )
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
        *,
        inputs_already_loaded: bool = False,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        formal_frame = self._formal_gpu_frame(world)
        self._upload_powder_apply_state(world, resources)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        if formal_frame:
            self._build_powder_apply_dispatch(world, resources)
        if not (formal_frame and inputs_already_loaded):
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
                use_bridge_state = self._bridge_authoritative_island_state(world)
                if upload_plan["cell_core"]:
                    resources.material_tex.write(world.material_id.astype("f4").tobytes())
                if upload_plan["island_id"]:
                    resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
                if not use_bridge_state:
                    cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
                    cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
                    self._load_authoritative_bridge_inputs(world, resources, cell_group_x, cell_group_y)
                self._upload_material_rule_params(world, resources)
                program = self.programs["plan_bridge_runtime_falling_island_reservations"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["runtime_capacity"].value = runtime_capacity
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
