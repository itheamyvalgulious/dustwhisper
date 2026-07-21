from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import ISLAND_RUNTIME_DTYPE, pack_island_runtime_upload
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.types import CellFlag


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
    "REACTION_LATCHED_FLAG": int(CellFlag.REACTION_LATCHED),
    "DIRECT_BRIDGE_OUTPUTS": 0,
    "ISLAND_APPLY_CHANGED_ONLY": 0,
    "POWDER_APPLY_INDEX_EPOCH": 0,
    "POWDER_CLEAR_LOCAL_SIZE": 8,
    "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 0,
    "POWDER_COMPACT_RESERVATION": 0,
    "POWDER_COMPACT_LAZY_EXPAND": 0,
    "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 0,
    "POWDER_PROVISIONAL_MOVING_WORKLIST": 0,
    "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 0,
    "POWDER_SOURCE_TILE_PRODUCER": 0,
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
    cell_state_tex: Any
    cell_state_out_tex: Any
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
    powder_apply_epoch: Any
    powder_direct_apply_unsafe: Any
    powder_source_cell_core_snapshot: Any
    powder_source_aux_snapshot: Any
    powder_reservations: Any
    powder_compact_reservations: Any
    powder_reservation_count: Any
    powder_provisional_moving_count: Any
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




from oracle_game.sim.gpu_motion_resources import (
    release,
    _ensure_resources,
    _write_dynamic_buffer,
    _ensure_dynamic_buffer_capacity
)
from oracle_game.sim.gpu_motion_bridge import (
    _upload_inputs,
    _cpu_upload_plan,
    _record_cpu_upload_plan,
    _load_authoritative_active_tile_mask,
    _upload_material_rule_params,
    _bridge_authoritative_cell_blockers,
    _bridge_authoritative_powder_inputs,
    _bind_bridge_cell_blockers,
    _bridge_authoritative_island_state,
    _bind_bridge_island_state,
    _bridge_context_active,
    _active_context,
    _load_authoritative_bridge_inputs,
    _load_authoritative_integrate_inputs,
    _load_authoritative_materialization_inputs,
    _publish_bridge_outputs,
    _publish_bridge_velocity_words,
    _publish_bridge_island_id,
    publish_bridge_falling_island_reservations,
    publish_bridge_powder_reservations,
    publish_bridge_compact_powder_reservations,
    publish_bridge_falling_island_runtime_from_reservations,
    seed_bridge_falling_island_runtime_from_cpu,
    _download_outputs,
    _download_velocity_output,
    _upload_powder_apply_state,
    _download_powder_apply_state
)
from oracle_game.sim.gpu_motion_dispatch import (
    _active_tile_workgroups_per_tile,
    _active_scheduler_gpu_authoritative,
    _compact_active_tiles,
    _build_active_tile_count_dispatch_args,
    _build_falling_island_materialization_candidate_dispatch,
    _copy_scalar_texture,
    _swap_powder_apply_textures,
    _barrier_bits,
    _run_active_tile_indirect,
    _refresh_authoritative_active_scheduler_after_apply,
    _build_powder_reservation_dispatch_args,
    _run_powder_reservation_indirect,
    _build_island_reservation_dispatch_args,
    _run_island_reservation_indirect,
    _build_island_runtime_dispatch_args,
    _run_island_runtime_indirect,
    _build_powder_apply_dispatch,
    _build_falling_island_apply_dispatch,
    _ensure_falling_island_index_capacity,
    _clear_falling_island_index,
    _ensure_bridge_runtime_reservation_capacity,
    _ensure_bridge_runtime_planning_capacity
)
from oracle_game.sim.gpu_motion_powder import (
    _clear_powder_target_winners_for_reservations,
    _clear_powder_apply_index_for_reservations,
    _clear_powder_apply_index_for_active_tiles,
    _run_powder_targets,
    _run_generate_powder_reservations,
    plan_powder_reservations,
    upload_powder_reservations,
    resolve_and_apply_powders,
    _dispatch_apply_powder_fast_path,
    apply_powder_reservations,
    _dispatch_index_powder_apply,
    _dispatch_apply_powder_reservations,
    _powder_direct_apply_is_safe,
    _read_powder_reservations,
    materialize_compact_powder_reservations,
    _build_powder_reservations
)
from oracle_game.sim.gpu_motion_island import (
    _dispatch_index_falling_island_reservation_sources,
    _dispatch_index_falling_island_apply,
    _dispatch_index_falling_island_materialization,
    apply_falling_island_reservations,
    apply_uploaded_falling_island_reservations,
    shed_falling_island_fragments,
    apply_falling_island_settlements,
    apply_uploaded_falling_island_settlements,
    _dispatch_apply_falling_island_materialization,
    _dispatch_apply_falling_island_reservations,
    plan_uploaded_falling_island_reservations,
    plan_uploaded_falling_island_reservations_from_bridge_runtime,
    plan_falling_island_reservations,
    upload_falling_island_reservations,
    resolve_falling_island_reservations,
    resolve_uploaded_falling_island_reservations,
    _dispatch_resolve_falling_island_reservations,
    _read_falling_island_reservations,
    resolve_falling_island_shifts
)
from oracle_game.sim.gpu_motion_island_labeling import (
    label_falling_island_components,
    label_falling_island_component_metadata,
    label_falling_island_component_metadata_texture,
    _summarize_falling_island_label_texture,
    relabel_falling_island_components,
    relabel_falling_island_component_texture
)
from oracle_game.sim.gpu_motion_stages import (
    step,
    integrate_velocity,
    can_consume_deferred_heat_core,
    can_consume_reaction_handoff,
    _integrate_reaction_handoff,
)


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
        self._falling_island_materialization_bridge_fusion_enabled = True
        self._falling_island_apply_bridge_fusion_enabled = True
        self._powder_aux_index_scratch_fusion_enabled = True
        self._powder_sparse_bridge_publish_enabled = True
        self._powder_direct_bridge_apply_enabled = False
        # Generation stamps avoid clearing active-tile index arrays each
        # formal powder apply while retaining the legacy non-formal path.
        self._powder_apply_index_epoch_enabled = True
        self._powder_target_clear_local64_enabled = True
        self._powder_apply_epoch = 0
        self._powder_generated_path_reuse_enabled = False
        # Generation and resolve observe the same blocker state.  This
        # candidate moves fallback blocker reads into the spatial generation
        # dispatch while leaving winner arbitration in resolve.
        self._powder_precomputed_fallback_blockers_enabled = True
        # Candidate: compact generated reservations whose DDA stayed at source
        # and whose three fallback cells were preclassified blocked have the
        # canonical blocked result without rereading source state in resolve.
        self._powder_trivial_blocked_classification_enabled = True
        # Resolve publishes at most two apply tiles per reservation.  Deduping
        # those tile IDs in shared memory removes repeated global atomics while
        # preserving the canonical reservation and bridge outputs.
        self._powder_apply_tile_workgroup_dedup_enabled = True
        # Formal generated reservations have disjoint source/target cells, so
        # one reservation invocation can publish directly to the bridge and
        # avoid the full-cell index/apply/publish chain. Uploaded reservations
        # remain on the canonical path via the strict runtime gate.
        self._powder_source_indexed_direct_apply_enabled = True
        # Formal generated reservations can use a private packed ABI while the
        # public/CPU reservation buffer remains the canonical 48-byte format.
        self._powder_compact_reservation_enabled = True
        # Keep the compact stream authoritative during ordinary GPU frames;
        # materialize the public ABI only when a debugger/readback observes it.
        self._powder_compact_reservation_lazy_expand_enabled = True
        self.compact_powder_reservation_materialization_count = 0
        # Candidate: resolve appends only provisionally moving reservation
        # indices so source-indexed direct apply can skip blocked records.
        self._powder_provisional_moving_worklist_enabled = True
        # Generation finalizes terminal blocked records and emits a sparse
        # canonical-index stream for the remaining resolve work.
        self._powder_nontrivial_resolve_worklist_enabled = True
        self._falling_island_materialization_minimal_hydration_enabled = True
        # Island reservation resolve only consumes packed cell state and island
        # ownership. Keep the narrower bridge hydration opt-in until its
        # frame-level performance is validated against the canonical loader.
        self._falling_island_resolve_minimal_hydration_enabled = True
        # Direct materialization already targets authoritative bridge storage.
        # This candidate avoids rewriting unchanged cells in affected tiles.
        self._falling_island_materialization_changed_only_enabled = True
        # Experimental: direct bridge apply can skip cells with no indexed
        # incoming/outgoing reservation because their bridge payload is unchanged.
        self._falling_island_apply_changed_only_enabled = True
        self._reaction_latch_handoff_clear_enabled = True
        # The reaction terminal shader can complete velocity integration. The
        # normal path remains canonical until redundant motion input setup is
        # validated independently at frame level.


    def _reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``available`` inherited from GPUPipelineBase.

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
        self.programs["integrate_reaction_handoff"] = build_compute_shader(ctx, "motion/integrate_reaction_handoff.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "motion/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_integrate_inputs"] = build_compute_shader(ctx, "motion/load_bridge_integrate_inputs.comp", _SHADER_SUBS)
        self.programs["load_bridge_materialization_inputs"] = build_compute_shader(ctx, "motion/load_bridge_materialization_inputs.comp", _SHADER_SUBS)
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
        self.programs["generate_powder_reservations_compact"] = build_compute_shader(
            ctx,
            "motion/generate_powder_reservations.comp",
            {**_SHADER_SUBS, "POWDER_COMPACT_RESERVATION": 1},
        )
        self.programs[
            "generate_powder_reservations_compact_nontrivial_worklist"
        ] = build_compute_shader(
            ctx,
            "motion/generate_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
                "POWDER_SOURCE_TILE_PRODUCER": 1,
            },
        )
        self.programs["clear_powder_target_winners"] = build_compute_shader(ctx, "motion/clear_powder_target_winners.comp", _SHADER_SUBS)
        self.programs["clear_powder_target_winners_local64"] = build_compute_shader(
            ctx,
            "motion/clear_powder_target_winners.comp",
            {**_SHADER_SUBS, "POWDER_CLEAR_LOCAL_SIZE": 64},
        )
        self.programs["index_powder_target_winners"] = build_compute_shader(ctx, "motion/index_powder_target_winners.comp", _SHADER_SUBS)
        self.programs["resolve_powder_reservations"] = build_compute_shader(ctx, "motion/resolve_powder_reservations.comp", _SHADER_SUBS)
        self.programs["resolve_powder_reservations_tile_dedup"] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {**_SHADER_SUBS, "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1},
        )
        self.programs["resolve_powder_reservations_compact"] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {**_SHADER_SUBS, "POWDER_COMPACT_RESERVATION": 1},
        )
        self.programs["resolve_powder_reservations_compact_trivial_blocked"] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
            },
        )
        self.programs["resolve_powder_reservations_compact_tile_dedup"] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1,
            },
        )
        self.programs[
            "resolve_powder_reservations_compact_tile_dedup_trivial_blocked"
        ] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1,
                "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
            },
        )
        self.programs[
            "resolve_powder_reservations_compact_tile_dedup_trivial_blocked_moving_worklist"
        ] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1,
                "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
                "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
            },
        )
        self.programs[
            "resolve_powder_reservations_compact_trivial_blocked_moving_worklist_"
            "nontrivial_worklist"
        ] = build_compute_shader(
            ctx,
            "motion/resolve_powder_reservations.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
                "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
                "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
                "POWDER_SOURCE_TILE_PRODUCER": 1,
            },
        )
        self.programs["apply_powder_reservations_source_indexed_direct"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations_source_indexed_direct.comp",
            _SHADER_SUBS,
        )
        self.programs["apply_powder_reservations_source_indexed_direct_compact"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations_source_indexed_direct.comp",
            {**_SHADER_SUBS, "POWDER_COMPACT_RESERVATION": 1},
        )
        self.programs["apply_powder_reservations_source_indexed_direct_compact_lazy"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations_source_indexed_direct.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_COMPACT_LAZY_EXPAND": 1,
            },
        )
        self.programs[
            "apply_powder_reservations_source_indexed_direct_compact_lazy_moving_worklist"
        ] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations_source_indexed_direct.comp",
            {
                **_SHADER_SUBS,
                "POWDER_COMPACT_RESERVATION": 1,
                "POWDER_COMPACT_LAZY_EXPAND": 1,
                "POWDER_PROVISIONAL_MOVING_WORKLIST": 1,
            },
        )
        self.programs["expand_compact_powder_reservations"] = build_compute_shader(
            ctx,
            "motion/expand_compact_powder_reservations.comp",
            _SHADER_SUBS,
        )
        self.programs["clear_powder_apply_index"] = build_compute_shader(ctx, "motion/clear_powder_apply_index.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index"] = build_compute_shader(ctx, "motion/clear_falling_island_index.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index_for_active_tiles"] = build_compute_shader(ctx, "motion/clear_falling_island_index_for_active_tiles.comp", _SHADER_SUBS)
        self.programs["clear_falling_island_index_for_reservations"] = build_compute_shader(ctx, "motion/clear_falling_island_index_for_reservations.comp", _SHADER_SUBS)
        self.programs["fill_falling_island_apply_index"] = build_compute_shader(ctx, "motion/fill_falling_island_apply_index.comp", _SHADER_SUBS)
        self.programs["fill_falling_island_materialization_index"] = build_compute_shader(ctx, "motion/fill_falling_island_materialization_index.comp", _SHADER_SUBS)
        self.programs["index_powder_apply_winners"] = build_compute_shader(ctx, "motion/index_powder_apply_winners.comp", _SHADER_SUBS)
        self.programs["fill_powder_apply_index_legacy"] = build_compute_shader(
            ctx, "motion/fill_powder_apply_index.comp", _SHADER_SUBS
        )
        self.programs["fill_powder_apply_index"] = build_compute_shader(
            ctx, "motion/fill_powder_apply_index.comp", {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1}
        )
        self.programs["apply_powder_fast_path"] = build_compute_shader(ctx, "motion/apply_powder_fast_path.comp", _SHADER_SUBS)
        self.programs["apply_powder_reservations_legacy"] = build_compute_shader(
            ctx, "motion/apply_powder_reservations.comp", _SHADER_SUBS
        )
        self.programs["apply_powder_reservations_bridge_legacy"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1},
        )
        self.programs["apply_powder_reservations"] = build_compute_shader(
            ctx, "motion/apply_powder_reservations.comp", {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1}
        )
        self.programs["apply_powder_reservations_bridge"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1, "POWDER_APPLY_INDEX_EPOCH": 1},
        )
        self.programs["detect_powder_direct_apply_unsafe"] = build_compute_shader(
            ctx,
            "motion/detect_powder_direct_apply_unsafe.comp",
            _SHADER_SUBS,
        )
        self.programs["apply_powder_reservation_aux_legacy"] = build_compute_shader(
            ctx, "motion/apply_powder_reservation_aux.comp", _SHADER_SUBS
        )
        self.programs["apply_powder_reservation_aux"] = build_compute_shader(
            ctx, "motion/apply_powder_reservation_aux.comp", {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1}
        )
        self.programs["apply_falling_island_reservations"] = build_compute_shader(ctx, "motion/apply_falling_island_reservations.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_reservations_bridge"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_reservations.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1},
        )
        self.programs["apply_falling_island_reservations_bridge_changed_only"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_reservations.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_OUTPUTS": 1,
                "ISLAND_APPLY_CHANGED_ONLY": 1,
            },
        )
        self.programs["apply_falling_island_reservation_aux"] = build_compute_shader(ctx, "motion/apply_falling_island_reservation_aux.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_materialization"] = build_compute_shader(ctx, "motion/apply_falling_island_materialization.comp", _SHADER_SUBS)
        self.programs["apply_falling_island_materialization_bridge"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_materialization.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1},
        )
        self.programs["apply_falling_island_materialization_bridge_changed_only"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_materialization.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_OUTPUTS": 2,
            },
        )
        self.programs["apply_falling_island_materialization_aux"] = build_compute_shader(ctx, "motion/apply_falling_island_materialization_aux.comp", _SHADER_SUBS)

    release = release
    _ensure_resources = _ensure_resources
    _write_dynamic_buffer = _write_dynamic_buffer
    _ensure_dynamic_buffer_capacity = _ensure_dynamic_buffer_capacity

    _upload_inputs = _upload_inputs
    _cpu_upload_plan = _cpu_upload_plan
    _record_cpu_upload_plan = _record_cpu_upload_plan
    _load_authoritative_active_tile_mask = _load_authoritative_active_tile_mask
    _upload_material_rule_params = _upload_material_rule_params
    _bridge_authoritative_cell_blockers = _bridge_authoritative_cell_blockers
    _bridge_authoritative_powder_inputs = _bridge_authoritative_powder_inputs
    _bind_bridge_cell_blockers = _bind_bridge_cell_blockers
    _bridge_authoritative_island_state = _bridge_authoritative_island_state
    _bind_bridge_island_state = _bind_bridge_island_state
    _bridge_context_active = _bridge_context_active
    _active_context = _active_context
    _load_authoritative_bridge_inputs = _load_authoritative_bridge_inputs
    _load_authoritative_integrate_inputs = _load_authoritative_integrate_inputs
    _load_authoritative_materialization_inputs = _load_authoritative_materialization_inputs
    _publish_bridge_outputs = _publish_bridge_outputs
    _publish_bridge_velocity_words = _publish_bridge_velocity_words
    _publish_bridge_island_id = _publish_bridge_island_id
    publish_bridge_falling_island_reservations = publish_bridge_falling_island_reservations
    publish_bridge_powder_reservations = publish_bridge_powder_reservations
    publish_bridge_compact_powder_reservations = publish_bridge_compact_powder_reservations
    publish_bridge_falling_island_runtime_from_reservations = publish_bridge_falling_island_runtime_from_reservations
    seed_bridge_falling_island_runtime_from_cpu = seed_bridge_falling_island_runtime_from_cpu
    _download_outputs = _download_outputs
    _download_velocity_output = _download_velocity_output
    _upload_powder_apply_state = _upload_powder_apply_state
    _download_powder_apply_state = _download_powder_apply_state

    _active_tile_workgroups_per_tile = _active_tile_workgroups_per_tile
    _active_scheduler_gpu_authoritative = _active_scheduler_gpu_authoritative
    _compact_active_tiles = _compact_active_tiles
    _build_active_tile_count_dispatch_args = _build_active_tile_count_dispatch_args
    _build_falling_island_materialization_candidate_dispatch = _build_falling_island_materialization_candidate_dispatch
    _copy_scalar_texture = _copy_scalar_texture
    _swap_powder_apply_textures = _swap_powder_apply_textures
    _barrier_bits = _barrier_bits
    _run_active_tile_indirect = _run_active_tile_indirect
    _refresh_authoritative_active_scheduler_after_apply = _refresh_authoritative_active_scheduler_after_apply
    _build_powder_reservation_dispatch_args = _build_powder_reservation_dispatch_args
    _run_powder_reservation_indirect = _run_powder_reservation_indirect
    _build_island_reservation_dispatch_args = _build_island_reservation_dispatch_args
    _run_island_reservation_indirect = _run_island_reservation_indirect
    _build_island_runtime_dispatch_args = _build_island_runtime_dispatch_args
    _run_island_runtime_indirect = _run_island_runtime_indirect
    _build_powder_apply_dispatch = _build_powder_apply_dispatch
    _build_falling_island_apply_dispatch = _build_falling_island_apply_dispatch
    _ensure_falling_island_index_capacity = _ensure_falling_island_index_capacity
    _clear_falling_island_index = _clear_falling_island_index
    _ensure_bridge_runtime_reservation_capacity = _ensure_bridge_runtime_reservation_capacity
    _ensure_bridge_runtime_planning_capacity = _ensure_bridge_runtime_planning_capacity

    _clear_powder_target_winners_for_reservations = _clear_powder_target_winners_for_reservations
    _clear_powder_apply_index_for_reservations = _clear_powder_apply_index_for_reservations
    _clear_powder_apply_index_for_active_tiles = _clear_powder_apply_index_for_active_tiles
    _run_powder_targets = _run_powder_targets
    _run_generate_powder_reservations = _run_generate_powder_reservations
    plan_powder_reservations = plan_powder_reservations
    upload_powder_reservations = upload_powder_reservations
    resolve_and_apply_powders = resolve_and_apply_powders
    _dispatch_apply_powder_fast_path = _dispatch_apply_powder_fast_path
    apply_powder_reservations = apply_powder_reservations
    _dispatch_index_powder_apply = _dispatch_index_powder_apply
    _dispatch_apply_powder_reservations = _dispatch_apply_powder_reservations
    _powder_direct_apply_is_safe = _powder_direct_apply_is_safe
    _read_powder_reservations = _read_powder_reservations
    materialize_compact_powder_reservations = materialize_compact_powder_reservations
    _build_powder_reservations = _build_powder_reservations

    _dispatch_index_falling_island_reservation_sources = _dispatch_index_falling_island_reservation_sources
    _dispatch_index_falling_island_apply = _dispatch_index_falling_island_apply
    _dispatch_index_falling_island_materialization = _dispatch_index_falling_island_materialization
    apply_falling_island_reservations = apply_falling_island_reservations
    apply_uploaded_falling_island_reservations = apply_uploaded_falling_island_reservations
    shed_falling_island_fragments = shed_falling_island_fragments
    apply_falling_island_settlements = apply_falling_island_settlements
    apply_uploaded_falling_island_settlements = apply_uploaded_falling_island_settlements
    _dispatch_apply_falling_island_materialization = _dispatch_apply_falling_island_materialization
    _dispatch_apply_falling_island_reservations = _dispatch_apply_falling_island_reservations
    plan_uploaded_falling_island_reservations = plan_uploaded_falling_island_reservations
    plan_uploaded_falling_island_reservations_from_bridge_runtime = plan_uploaded_falling_island_reservations_from_bridge_runtime
    plan_falling_island_reservations = plan_falling_island_reservations
    upload_falling_island_reservations = upload_falling_island_reservations
    resolve_falling_island_reservations = resolve_falling_island_reservations
    resolve_uploaded_falling_island_reservations = resolve_uploaded_falling_island_reservations
    _dispatch_resolve_falling_island_reservations = _dispatch_resolve_falling_island_reservations
    _read_falling_island_reservations = _read_falling_island_reservations
    resolve_falling_island_shifts = resolve_falling_island_shifts

    label_falling_island_components = label_falling_island_components
    label_falling_island_component_metadata = label_falling_island_component_metadata
    label_falling_island_component_metadata_texture = label_falling_island_component_metadata_texture
    _summarize_falling_island_label_texture = _summarize_falling_island_label_texture
    relabel_falling_island_components = relabel_falling_island_components
    relabel_falling_island_component_texture = relabel_falling_island_component_texture

    step = step
    integrate_velocity = integrate_velocity
    can_consume_deferred_heat_core = can_consume_deferred_heat_core
    can_consume_reaction_handoff = can_consume_reaction_handoff
    _integrate_reaction_handoff = _integrate_reaction_handoff
