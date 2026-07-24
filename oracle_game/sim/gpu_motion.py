from __future__ import annotations

from typing import Any

import numpy as np  # noqa: F401  # monkeypatch target in tests (gpu_motion.np)

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.gpu import (  # noqa: F401  # monkeypatch target in tests
    ISLAND_RUNTIME_DTYPE,
    pack_island_runtime_upload,
)
from oracle_game.sim.gpu_base import GPUPipelineBase

# Facade re-exports: the constants and reservation dtypes live in the leaf
# module ``gpu_motion_constants`` so the gpu_motion_* satellites can import them
# without cycling back through this hub.
from oracle_game.sim.gpu_motion_constants import (
    ACTIVE_TILE_WORKGROUP_AXIS,  # noqa: F401  # facade re-export
    ACTIVE_TILE_WORKGROUPS_PER_TILE,  # noqa: F401  # facade re-export
    FALLING_ISLAND_BREAK_STABLE,
    FALLING_ISLAND_INDEX_CLEAR_APPLY,  # noqa: F401  # facade re-export
    FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING,
    FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING,
    FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION,
    FALLING_ISLAND_INDEX_CLEAR_SOURCE,
    FALLING_ISLAND_RESERVATION_DTYPE,  # noqa: F401  # facade re-export
    INDEX_EMPTY,
    ISLAND_RESERVATION_LINEAR_LOCAL_SIZE,
    ISLAND_RESOLVE_BLOCKED,
    ISLAND_RESOLVE_DIRECT,
    ISLAND_RESOLVE_RERESOLVED,
    ISLAND_RESOLVE_STALE,
    LOCAL_SIZE,
    MAX_ISLAND_DDA_STEP,
    MAX_MATERIALS,
    POWDER_RESERVATION_DTYPE,  # noqa: F401  # facade re-export
    POWDER_RESERVATION_LOCAL_SIZE,
    POWDER_RESOLVE_BLOCKED,
    POWDER_RESOLVE_DDA,
    POWDER_RESOLVE_FALLBACK,
    POWDER_RESOLVE_STALE,
    POWDER_SOLVER_SUSPENDED,
    falling_island_reservation_dtype,  # noqa: F401  # facade re-export
    powder_reservation_dtype,  # noqa: F401  # facade re-export
)
from oracle_game.sim.gpu_motion_resources import GPUMotionResources
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.types import CellFlag

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


from oracle_game.sim.gpu_motion_bridge import (
    _active_context,
    _bind_bridge_cell_blockers,
    _bind_bridge_island_state,
    _bridge_authoritative_cell_blockers,
    _bridge_authoritative_island_state,
    _bridge_authoritative_powder_inputs,
    _bridge_context_active,
    _cpu_upload_plan,
    _download_outputs,
    _download_powder_apply_state,
    _download_velocity_output,
    _load_authoritative_active_tile_mask,
    _load_authoritative_bridge_inputs,
    _load_authoritative_integrate_inputs,
    _load_authoritative_materialization_inputs,
    _publish_bridge_island_id,
    _publish_bridge_outputs,
    _publish_bridge_velocity_words,
    _record_cpu_upload_plan,
    _upload_inputs,
    _upload_material_rule_params,
    _upload_powder_apply_state,
    publish_bridge_compact_powder_reservations,
    publish_bridge_falling_island_reservations,
    publish_bridge_falling_island_runtime_from_reservations,
    publish_bridge_powder_reservations,
    seed_bridge_falling_island_runtime_from_cpu,
)
from oracle_game.sim.gpu_motion_dispatch import (
    _active_scheduler_gpu_authoritative,
    _active_tile_workgroups_per_tile,
    _barrier_bits,
    _build_active_tile_count_dispatch_args,
    _build_falling_island_apply_dispatch,
    _build_falling_island_materialization_candidate_dispatch,
    _build_island_reservation_dispatch_args,
    _build_island_runtime_dispatch_args,
    _build_powder_apply_dispatch,
    _build_powder_reservation_dispatch_args,
    _clear_falling_island_index,
    _compact_active_tiles,
    _copy_scalar_texture,
    _ensure_bridge_runtime_planning_capacity,
    _ensure_bridge_runtime_reservation_capacity,
    _ensure_falling_island_index_capacity,
    _refresh_authoritative_active_scheduler_after_apply,
    _run_active_tile_indirect,
    _run_island_reservation_indirect,
    _run_island_runtime_indirect,
    _run_powder_reservation_indirect,
    _swap_powder_apply_textures,
)
from oracle_game.sim.gpu_motion_island import (
    _dispatch_apply_falling_island_materialization,
    _dispatch_apply_falling_island_reservations,
    _dispatch_index_falling_island_apply,
    _dispatch_index_falling_island_materialization,
    _dispatch_index_falling_island_reservation_sources,
    _dispatch_resolve_falling_island_reservations,
    _read_falling_island_reservations,
    apply_falling_island_reservations,
    apply_falling_island_settlements,
    apply_uploaded_falling_island_reservations,
    apply_uploaded_falling_island_settlements,
    plan_falling_island_reservations,
    plan_uploaded_falling_island_reservations,
    plan_uploaded_falling_island_reservations_from_bridge_runtime,
    resolve_falling_island_reservations,
    resolve_falling_island_shifts,
    resolve_uploaded_falling_island_reservations,
    shed_falling_island_fragments,
    upload_falling_island_reservations,
)
from oracle_game.sim.gpu_motion_island_labeling import (
    _summarize_falling_island_label_texture,
    label_falling_island_component_metadata,
    label_falling_island_component_metadata_texture,
    label_falling_island_components,
    relabel_falling_island_component_texture,
    relabel_falling_island_components,
)
from oracle_game.sim.gpu_motion_powder import (
    _build_powder_reservations,
    _clear_powder_apply_index_for_active_tiles,
    _clear_powder_apply_index_for_reservations,
    _clear_powder_target_winners_for_reservations,
    _dispatch_apply_powder_fast_path,
    _dispatch_apply_powder_reservations,
    _dispatch_index_powder_apply,
    _powder_direct_apply_is_safe,
    _read_powder_reservations,
    _run_generate_powder_reservations,
    _run_powder_targets,
    apply_powder_reservations,
    materialize_compact_powder_reservations,
    plan_powder_reservations,
    resolve_and_apply_powders,
    upload_powder_reservations,
)
from oracle_game.sim.gpu_motion_resources import (
    _ensure_dynamic_buffer_capacity,
    _ensure_resources,
    _write_dynamic_buffer,
    release,
)
from oracle_game.sim.gpu_motion_stages import (
    _integrate_reaction_handoff,
    can_consume_deferred_heat_core,
    can_consume_reaction_handoff,
    integrate_velocity,
    step,
)


class GPUMotionPipeline(GPUPipelineBase):
    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
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
        self._falling_island_materialization_bridge_fusion_enabled = (
            self.engine_config.falling_island_materialization_bridge_fusion_enabled
        )
        self._falling_island_apply_bridge_fusion_enabled = (
            self.engine_config.falling_island_apply_bridge_fusion_enabled
        )
        self._powder_aux_index_scratch_fusion_enabled = (
            self.engine_config.powder_aux_index_scratch_fusion_enabled
        )
        self._powder_sparse_bridge_publish_enabled = (
            self.engine_config.powder_sparse_bridge_publish_enabled
        )
        # Generation stamps avoid clearing active-tile index arrays each
        # formal powder apply while retaining the legacy non-formal path.
        self._powder_apply_index_epoch_enabled = self.engine_config.powder_apply_index_epoch_enabled
        self._powder_target_clear_local64_enabled = (
            self.engine_config.powder_target_clear_local64_enabled
        )
        self._powder_apply_epoch = 0
        # Generation and resolve observe the same blocker state.  This
        # candidate moves fallback blocker reads into the spatial generation
        # dispatch while leaving winner arbitration in resolve.
        self._powder_precomputed_fallback_blockers_enabled = (
            self.engine_config.powder_precomputed_fallback_blockers_enabled
        )
        # Candidate: compact generated reservations whose DDA stayed at source
        # and whose three fallback cells were preclassified blocked have the
        # canonical blocked result without rereading source state in resolve.
        self._powder_trivial_blocked_classification_enabled = (
            self.engine_config.powder_trivial_blocked_classification_enabled
        )
        # Resolve publishes at most two apply tiles per reservation.  Deduping
        # those tile IDs in shared memory removes repeated global atomics while
        # preserving the canonical reservation and bridge outputs.
        self._powder_apply_tile_workgroup_dedup_enabled = (
            self.engine_config.powder_apply_tile_workgroup_dedup_enabled
        )
        # Formal generated reservations have disjoint source/target cells, so
        # one reservation invocation can publish directly to the bridge and
        # avoid the full-cell index/apply/publish chain. Uploaded reservations
        # remain on the canonical path via the strict runtime gate.
        self._powder_source_indexed_direct_apply_enabled = (
            self.engine_config.powder_source_indexed_direct_apply_enabled
        )
        # Formal generated reservations can use a private packed ABI while the
        # public/CPU reservation buffer remains the canonical 48-byte format.
        self._powder_compact_reservation_enabled = (
            self.engine_config.powder_compact_reservation_enabled
        )
        # Keep the compact stream authoritative during ordinary GPU frames;
        # materialize the public ABI only when a debugger/readback observes it.
        self._powder_compact_reservation_lazy_expand_enabled = (
            self.engine_config.powder_compact_reservation_lazy_expand_enabled
        )
        self.compact_powder_reservation_materialization_count = 0
        # Candidate: resolve appends only provisionally moving reservation
        # indices so source-indexed direct apply can skip blocked records.
        self._powder_provisional_moving_worklist_enabled = (
            self.engine_config.powder_provisional_moving_worklist_enabled
        )
        # Generation finalizes terminal blocked records and emits a sparse
        # canonical-index stream for the remaining resolve work.
        self._powder_nontrivial_resolve_worklist_enabled = (
            self.engine_config.powder_nontrivial_resolve_worklist_enabled
        )
        self._falling_island_materialization_minimal_hydration_enabled = (
            self.engine_config.falling_island_materialization_minimal_hydration_enabled
        )
        # Island reservation resolve only consumes packed cell state and island
        # ownership. Keep the narrower bridge hydration opt-in until its
        # frame-level performance is validated against the canonical loader.
        self._falling_island_resolve_minimal_hydration_enabled = (
            self.engine_config.falling_island_resolve_minimal_hydration_enabled
        )
        # Direct materialization already targets authoritative bridge storage.
        # This candidate avoids rewriting unchanged cells in affected tiles.
        self._falling_island_materialization_changed_only_enabled = (
            self.engine_config.falling_island_materialization_changed_only_enabled
        )
        # Experimental: direct bridge apply can skip cells with no indexed
        # incoming/outgoing reservation because their bridge payload is unchanged.
        self._falling_island_apply_changed_only_enabled = (
            self.engine_config.falling_island_apply_changed_only_enabled
        )
        self._reaction_latch_handoff_clear_enabled = (
            self.engine_config.reaction_latch_handoff_clear_enabled
        )
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
        self.programs["load_active_tiles"] = build_compute_shader(
            ctx, "motion/load_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["clear_active_tile_dispatch"] = build_compute_shader(
            ctx, "_shared/clear_active_tile_dispatch.comp", _SHADER_SUBS
        )
        self.programs["compact_active_tiles"] = build_compute_shader(
            ctx, "_shared/compact_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["compact_active_tiles_from_chunks"] = build_compute_shader(
            ctx, "_shared/compact_active_tiles_from_chunks.comp", _SHADER_SUBS
        )
        self.programs["build_falling_island_materialization_candidate_dispatch"] = (
            build_compute_shader(
                ctx,
                "motion/build_falling_island_materialization_candidate_dispatch.comp",
                _SHADER_SUBS,
            )
        )
        self.programs["build_powder_reservation_dispatch"] = build_compute_shader(
            ctx, "motion/build_powder_reservation_dispatch.comp", _SHADER_SUBS
        )
        self.programs["build_island_runtime_dispatch"] = build_compute_shader(
            ctx, "motion/build_island_runtime_dispatch.comp", _SHADER_SUBS
        )
        self.programs["clear_powder_affected_tile_dispatch"] = build_compute_shader(
            ctx, "motion/clear_powder_affected_tile_dispatch.comp", _SHADER_SUBS
        )
        self.programs["build_powder_apply_dispatch"] = build_compute_shader(
            ctx, "motion/build_powder_apply_dispatch.comp", _SHADER_SUBS
        )
        self.programs["build_falling_island_apply_dispatch"] = build_compute_shader(
            ctx, "motion/build_falling_island_apply_dispatch.comp", _SHADER_SUBS
        )
        self.programs["clear_powder_target_winners_for_reservations"] = build_compute_shader(
            ctx, "motion/clear_powder_target_winners_for_reservations.comp", _SHADER_SUBS
        )
        self.programs["clear_powder_apply_index_for_reservations"] = build_compute_shader(
            ctx, "motion/clear_powder_apply_index_for_reservations.comp", _SHADER_SUBS
        )
        self.programs["clear_powder_apply_index_for_active_tiles"] = build_compute_shader(
            ctx, "motion/clear_powder_apply_index_for_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["integrate_velocity"] = build_compute_shader(
            ctx, "motion/integrate_velocity.comp", _SHADER_SUBS
        )
        self.programs["integrate_reaction_handoff"] = build_compute_shader(
            ctx, "motion/integrate_reaction_handoff.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell"] = build_compute_shader(
            ctx, "motion/load_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_integrate_inputs"] = build_compute_shader(
            ctx, "motion/load_bridge_integrate_inputs.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_materialization_inputs"] = build_compute_shader(
            ctx, "motion/load_bridge_materialization_inputs.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_aux"] = build_compute_shader(
            ctx, "motion/load_bridge_cell_aux.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_gas"] = build_compute_shader(
            ctx, "motion/load_bridge_gas.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_cell"] = build_compute_shader(
            ctx, "motion/publish_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_velocity_word"] = build_compute_shader(
            ctx, "motion/publish_bridge_velocity_word.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_island_id"] = build_compute_shader(
            ctx, "motion/publish_bridge_island_id.comp", _SHADER_SUBS
        )
        self.programs["copy_scalar_texture"] = build_compute_shader(
            ctx, "motion/copy_scalar_texture.comp", _SHADER_SUBS
        )
        self.programs["powder_targets"] = build_compute_shader(
            ctx, "motion/powder_targets.comp", _SHADER_SUBS
        )
        self.programs["island_component_init"] = build_compute_shader(
            ctx, "motion/island_component_init.comp", _SHADER_SUBS
        )
        self.programs["island_component_propagate"] = build_compute_shader(
            ctx, "motion/island_component_propagate.comp", _SHADER_SUBS
        )
        self.programs["relabel_falling_island_components"] = build_compute_shader(
            ctx, "motion/relabel_falling_island_components.comp", _SHADER_SUBS
        )
        self.programs["summarize_falling_island_components"] = build_compute_shader(
            ctx, "motion/summarize_falling_island_components.comp", _SHADER_SUBS
        )
        self.programs["island_shifts"] = build_compute_shader(
            ctx, "motion/island_shifts.comp", _SHADER_SUBS
        )
        self.programs["plan_bridge_runtime_falling_island_reservations"] = build_compute_shader(
            ctx, "motion/plan_bridge_runtime_falling_island_reservations.comp", _SHADER_SUBS
        )
        self.programs["pack_falling_island_reservations"] = build_compute_shader(
            ctx, "motion/pack_falling_island_reservations.comp", _SHADER_SUBS
        )
        self.programs["publish_falling_island_runtime"] = build_compute_shader(
            ctx, "motion/publish_falling_island_runtime.comp", _SHADER_SUBS
        )
        self.programs["publish_powder_reservations"] = build_compute_shader(
            ctx, "motion/publish_powder_reservations.comp", _SHADER_SUBS
        )
        self.programs["publish_falling_island_reservations"] = build_compute_shader(
            ctx, "motion/publish_falling_island_reservations.comp", _SHADER_SUBS
        )
        self.programs["unpack_bridge_island_runtime"] = build_compute_shader(
            ctx, "motion/unpack_bridge_island_runtime.comp", _SHADER_SUBS
        )
        self.programs["fill_falling_island_reservation_source_index"] = build_compute_shader(
            ctx, "motion/fill_falling_island_reservation_source_index.comp", _SHADER_SUBS
        )
        self.programs["resolve_falling_island_reservations"] = build_compute_shader(
            ctx, "motion/resolve_falling_island_reservations.comp", _SHADER_SUBS
        )
        self.programs["generate_powder_reservations"] = build_compute_shader(
            ctx, "motion/generate_powder_reservations.comp", _SHADER_SUBS
        )
        self.programs["generate_powder_reservations_compact"] = build_compute_shader(
            ctx,
            "motion/generate_powder_reservations.comp",
            {**_SHADER_SUBS, "POWDER_COMPACT_RESERVATION": 1},
        )
        self.programs["generate_powder_reservations_compact_nontrivial_worklist"] = (
            build_compute_shader(
                ctx,
                "motion/generate_powder_reservations.comp",
                {
                    **_SHADER_SUBS,
                    "POWDER_COMPACT_RESERVATION": 1,
                    "POWDER_NONTRIVIAL_RESOLVE_WORKLIST": 1,
                    "POWDER_SOURCE_TILE_PRODUCER": 1,
                },
            )
        )
        self.programs["clear_powder_target_winners"] = build_compute_shader(
            ctx, "motion/clear_powder_target_winners.comp", _SHADER_SUBS
        )
        self.programs["clear_powder_target_winners_local64"] = build_compute_shader(
            ctx,
            "motion/clear_powder_target_winners.comp",
            {**_SHADER_SUBS, "POWDER_CLEAR_LOCAL_SIZE": 64},
        )
        self.programs["index_powder_target_winners"] = build_compute_shader(
            ctx, "motion/index_powder_target_winners.comp", _SHADER_SUBS
        )
        self.programs["resolve_powder_reservations"] = build_compute_shader(
            ctx, "motion/resolve_powder_reservations.comp", _SHADER_SUBS
        )
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
        self.programs["resolve_powder_reservations_compact_tile_dedup_trivial_blocked"] = (
            build_compute_shader(
                ctx,
                "motion/resolve_powder_reservations.comp",
                {
                    **_SHADER_SUBS,
                    "POWDER_COMPACT_RESERVATION": 1,
                    "POWDER_APPLY_TILE_WORKGROUP_DEDUP": 1,
                    "POWDER_TRIVIAL_BLOCKED_CLASSIFICATION": 1,
                },
            )
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
        self.programs["apply_powder_reservations_source_indexed_direct_compact"] = (
            build_compute_shader(
                ctx,
                "motion/apply_powder_reservations_source_indexed_direct.comp",
                {**_SHADER_SUBS, "POWDER_COMPACT_RESERVATION": 1},
            )
        )
        self.programs["apply_powder_reservations_source_indexed_direct_compact_lazy"] = (
            build_compute_shader(
                ctx,
                "motion/apply_powder_reservations_source_indexed_direct.comp",
                {
                    **_SHADER_SUBS,
                    "POWDER_COMPACT_RESERVATION": 1,
                    "POWDER_COMPACT_LAZY_EXPAND": 1,
                },
            )
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
        self.programs["clear_powder_apply_index"] = build_compute_shader(
            ctx, "motion/clear_powder_apply_index.comp", _SHADER_SUBS
        )
        self.programs["clear_falling_island_index"] = build_compute_shader(
            ctx, "motion/clear_falling_island_index.comp", _SHADER_SUBS
        )
        self.programs["clear_falling_island_index_for_active_tiles"] = build_compute_shader(
            ctx, "motion/clear_falling_island_index_for_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["clear_falling_island_index_for_reservations"] = build_compute_shader(
            ctx, "motion/clear_falling_island_index_for_reservations.comp", _SHADER_SUBS
        )
        self.programs["fill_falling_island_apply_index"] = build_compute_shader(
            ctx, "motion/fill_falling_island_apply_index.comp", _SHADER_SUBS
        )
        self.programs["fill_falling_island_materialization_index"] = build_compute_shader(
            ctx, "motion/fill_falling_island_materialization_index.comp", _SHADER_SUBS
        )
        self.programs["index_powder_apply_winners"] = build_compute_shader(
            ctx, "motion/index_powder_apply_winners.comp", _SHADER_SUBS
        )
        self.programs["fill_powder_apply_index_legacy"] = build_compute_shader(
            ctx, "motion/fill_powder_apply_index.comp", _SHADER_SUBS
        )
        self.programs["fill_powder_apply_index"] = build_compute_shader(
            ctx,
            "motion/fill_powder_apply_index.comp",
            {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1},
        )
        self.programs["apply_powder_fast_path"] = build_compute_shader(
            ctx, "motion/apply_powder_fast_path.comp", _SHADER_SUBS
        )
        self.programs["apply_powder_reservations_legacy"] = build_compute_shader(
            ctx, "motion/apply_powder_reservations.comp", _SHADER_SUBS
        )
        self.programs["apply_powder_reservations"] = build_compute_shader(
            ctx,
            "motion/apply_powder_reservations.comp",
            {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1},
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
            ctx,
            "motion/apply_powder_reservation_aux.comp",
            {**_SHADER_SUBS, "POWDER_APPLY_INDEX_EPOCH": 1},
        )
        self.programs["apply_falling_island_reservations"] = build_compute_shader(
            ctx, "motion/apply_falling_island_reservations.comp", _SHADER_SUBS
        )
        self.programs["apply_falling_island_reservations_bridge"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_reservations.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1},
        )
        self.programs["apply_falling_island_reservations_bridge_changed_only"] = (
            build_compute_shader(
                ctx,
                "motion/apply_falling_island_reservations.comp",
                {
                    **_SHADER_SUBS,
                    "DIRECT_BRIDGE_OUTPUTS": 1,
                    "ISLAND_APPLY_CHANGED_ONLY": 1,
                },
            )
        )
        self.programs["apply_falling_island_reservation_aux"] = build_compute_shader(
            ctx, "motion/apply_falling_island_reservation_aux.comp", _SHADER_SUBS
        )
        self.programs["apply_falling_island_materialization"] = build_compute_shader(
            ctx, "motion/apply_falling_island_materialization.comp", _SHADER_SUBS
        )
        self.programs["apply_falling_island_materialization_bridge"] = build_compute_shader(
            ctx,
            "motion/apply_falling_island_materialization.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_OUTPUTS": 1},
        )
        self.programs["apply_falling_island_materialization_bridge_changed_only"] = (
            build_compute_shader(
                ctx,
                "motion/apply_falling_island_materialization.comp",
                {
                    **_SHADER_SUBS,
                    "DIRECT_BRIDGE_OUTPUTS": 2,
                },
            )
        )
        self.programs["apply_falling_island_materialization_aux"] = build_compute_shader(
            ctx, "motion/apply_falling_island_materialization_aux.comp", _SHADER_SUBS
        )

    # ------------------------------------------------------------------
    # Satellite method delegates (W3: retired the `_x = _x` class grafts).
    #
    # Each body resolves the bare function name through this module's global
    # namespace -- method bodies never see class scope -- i.e. the satellite
    # function imported at the top of this file, bound at import time exactly
    # like the historical grafts.  Monkeypatch semantics are unchanged:
    # patching the attribute on the class or on an instance shadows/replaces
    # the delegate, while patching the satellite module's attribute does NOT
    # affect calls through the pipeline.
    # ------------------------------------------------------------------

    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def _ensure_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_resources(self, *args, **kwargs)

    def _write_dynamic_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _write_dynamic_buffer(self, *args, **kwargs)

    def _ensure_dynamic_buffer_capacity(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_dynamic_buffer_capacity(self, *args, **kwargs)

    def _upload_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_inputs(self, *args, **kwargs)

    def _cpu_upload_plan(self, *args: Any, **kwargs: Any) -> Any:
        return _cpu_upload_plan(self, *args, **kwargs)

    def _record_cpu_upload_plan(self, *args: Any, **kwargs: Any) -> Any:
        return _record_cpu_upload_plan(self, *args, **kwargs)

    def _load_authoritative_active_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_active_tile_mask(self, *args, **kwargs)

    def _upload_material_rule_params(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_material_rule_params(self, *args, **kwargs)

    def _bridge_authoritative_cell_blockers(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_authoritative_cell_blockers(self, *args, **kwargs)

    def _bridge_authoritative_powder_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_authoritative_powder_inputs(self, *args, **kwargs)

    def _bind_bridge_cell_blockers(self, *args: Any, **kwargs: Any) -> Any:
        return _bind_bridge_cell_blockers(self, *args, **kwargs)

    def _bridge_authoritative_island_state(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_authoritative_island_state(self, *args, **kwargs)

    def _bind_bridge_island_state(self, *args: Any, **kwargs: Any) -> Any:
        return _bind_bridge_island_state(self, *args, **kwargs)

    def _bridge_context_active(self, *args: Any, **kwargs: Any) -> Any:
        return _bridge_context_active(self, *args, **kwargs)

    def _active_context(self, *args: Any, **kwargs: Any) -> Any:
        return _active_context(self, *args, **kwargs)

    def _load_authoritative_bridge_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_bridge_inputs(self, *args, **kwargs)

    def _load_authoritative_integrate_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_integrate_inputs(self, *args, **kwargs)

    def _load_authoritative_materialization_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_materialization_inputs(self, *args, **kwargs)

    def _publish_bridge_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_outputs(self, *args, **kwargs)

    def _publish_bridge_velocity_words(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_velocity_words(self, *args, **kwargs)

    def _publish_bridge_island_id(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_island_id(self, *args, **kwargs)

    def publish_bridge_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return publish_bridge_falling_island_reservations(self, *args, **kwargs)

    def publish_bridge_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return publish_bridge_powder_reservations(self, *args, **kwargs)

    def publish_bridge_compact_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return publish_bridge_compact_powder_reservations(self, *args, **kwargs)

    def publish_bridge_falling_island_runtime_from_reservations(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return publish_bridge_falling_island_runtime_from_reservations(self, *args, **kwargs)

    def seed_bridge_falling_island_runtime_from_cpu(self, *args: Any, **kwargs: Any) -> Any:
        return seed_bridge_falling_island_runtime_from_cpu(self, *args, **kwargs)

    def _download_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _download_outputs(self, *args, **kwargs)

    def _download_velocity_output(self, *args: Any, **kwargs: Any) -> Any:
        return _download_velocity_output(self, *args, **kwargs)

    def _upload_powder_apply_state(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_powder_apply_state(self, *args, **kwargs)

    def _download_powder_apply_state(self, *args: Any, **kwargs: Any) -> Any:
        return _download_powder_apply_state(self, *args, **kwargs)

    def _active_tile_workgroups_per_tile(self, *args: Any, **kwargs: Any) -> Any:
        return _active_tile_workgroups_per_tile(self, *args, **kwargs)

    def _active_scheduler_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return _active_scheduler_gpu_authoritative(self, *args, **kwargs)

    def _compact_active_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _compact_active_tiles(self, *args, **kwargs)

    def _build_active_tile_count_dispatch_args(self, *args: Any, **kwargs: Any) -> Any:
        return _build_active_tile_count_dispatch_args(self, *args, **kwargs)

    def _build_falling_island_materialization_candidate_dispatch(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return _build_falling_island_materialization_candidate_dispatch(self, *args, **kwargs)

    def _copy_scalar_texture(self, *args: Any, **kwargs: Any) -> Any:
        return _copy_scalar_texture(self, *args, **kwargs)

    def _swap_powder_apply_textures(self, *args: Any, **kwargs: Any) -> Any:
        return _swap_powder_apply_textures(self, *args, **kwargs)

    def _barrier_bits(self, *args: Any, **kwargs: Any) -> Any:
        return _barrier_bits(self, *args, **kwargs)

    def _run_active_tile_indirect(self, *args: Any, **kwargs: Any) -> Any:
        return _run_active_tile_indirect(self, *args, **kwargs)

    def _refresh_authoritative_active_scheduler_after_apply(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_authoritative_active_scheduler_after_apply(self, *args, **kwargs)

    def _build_powder_reservation_dispatch_args(self, *args: Any, **kwargs: Any) -> Any:
        return _build_powder_reservation_dispatch_args(self, *args, **kwargs)

    def _run_powder_reservation_indirect(self, *args: Any, **kwargs: Any) -> Any:
        return _run_powder_reservation_indirect(self, *args, **kwargs)

    def _build_island_reservation_dispatch_args(self, *args: Any, **kwargs: Any) -> Any:
        return _build_island_reservation_dispatch_args(self, *args, **kwargs)

    def _run_island_reservation_indirect(self, *args: Any, **kwargs: Any) -> Any:
        return _run_island_reservation_indirect(self, *args, **kwargs)

    def _build_island_runtime_dispatch_args(self, *args: Any, **kwargs: Any) -> Any:
        return _build_island_runtime_dispatch_args(self, *args, **kwargs)

    def _run_island_runtime_indirect(self, *args: Any, **kwargs: Any) -> Any:
        return _run_island_runtime_indirect(self, *args, **kwargs)

    def _build_powder_apply_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return _build_powder_apply_dispatch(self, *args, **kwargs)

    def _build_falling_island_apply_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return _build_falling_island_apply_dispatch(self, *args, **kwargs)

    def _ensure_falling_island_index_capacity(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_falling_island_index_capacity(self, *args, **kwargs)

    def _clear_falling_island_index(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_falling_island_index(self, *args, **kwargs)

    def _ensure_bridge_runtime_reservation_capacity(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_bridge_runtime_reservation_capacity(self, *args, **kwargs)

    def _ensure_bridge_runtime_planning_capacity(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_bridge_runtime_planning_capacity(self, *args, **kwargs)

    def _clear_powder_target_winners_for_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_powder_target_winners_for_reservations(self, *args, **kwargs)

    def _clear_powder_apply_index_for_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_powder_apply_index_for_reservations(self, *args, **kwargs)

    def _clear_powder_apply_index_for_active_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_powder_apply_index_for_active_tiles(self, *args, **kwargs)

    def _run_powder_targets(self, *args: Any, **kwargs: Any) -> Any:
        return _run_powder_targets(self, *args, **kwargs)

    def _run_generate_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _run_generate_powder_reservations(self, *args, **kwargs)

    def plan_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return plan_powder_reservations(self, *args, **kwargs)

    def upload_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return upload_powder_reservations(self, *args, **kwargs)

    def resolve_and_apply_powders(self, *args: Any, **kwargs: Any) -> Any:
        return resolve_and_apply_powders(self, *args, **kwargs)

    def _dispatch_apply_powder_fast_path(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_apply_powder_fast_path(self, *args, **kwargs)

    def apply_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return apply_powder_reservations(self, *args, **kwargs)

    def _dispatch_index_powder_apply(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_index_powder_apply(self, *args, **kwargs)

    def _dispatch_apply_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_apply_powder_reservations(self, *args, **kwargs)

    def _powder_direct_apply_is_safe(self, *args: Any, **kwargs: Any) -> Any:
        return _powder_direct_apply_is_safe(self, *args, **kwargs)

    def _read_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _read_powder_reservations(self, *args, **kwargs)

    def materialize_compact_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return materialize_compact_powder_reservations(self, *args, **kwargs)

    def _build_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _build_powder_reservations(self, *args, **kwargs)

    def _dispatch_index_falling_island_reservation_sources(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_index_falling_island_reservation_sources(self, *args, **kwargs)

    def _dispatch_index_falling_island_apply(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_index_falling_island_apply(self, *args, **kwargs)

    def _dispatch_index_falling_island_materialization(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_index_falling_island_materialization(self, *args, **kwargs)

    def apply_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return apply_falling_island_reservations(self, *args, **kwargs)

    def apply_uploaded_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return apply_uploaded_falling_island_reservations(self, *args, **kwargs)

    def shed_falling_island_fragments(self, *args: Any, **kwargs: Any) -> Any:
        return shed_falling_island_fragments(self, *args, **kwargs)

    def apply_falling_island_settlements(self, *args: Any, **kwargs: Any) -> Any:
        return apply_falling_island_settlements(self, *args, **kwargs)

    def apply_uploaded_falling_island_settlements(self, *args: Any, **kwargs: Any) -> Any:
        return apply_uploaded_falling_island_settlements(self, *args, **kwargs)

    def _dispatch_apply_falling_island_materialization(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_apply_falling_island_materialization(self, *args, **kwargs)

    def _dispatch_apply_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_apply_falling_island_reservations(self, *args, **kwargs)

    def plan_uploaded_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return plan_uploaded_falling_island_reservations(self, *args, **kwargs)

    def plan_uploaded_falling_island_reservations_from_bridge_runtime(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return plan_uploaded_falling_island_reservations_from_bridge_runtime(self, *args, **kwargs)

    def plan_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return plan_falling_island_reservations(self, *args, **kwargs)

    def upload_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return upload_falling_island_reservations(self, *args, **kwargs)

    def resolve_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return resolve_falling_island_reservations(self, *args, **kwargs)

    def resolve_uploaded_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return resolve_uploaded_falling_island_reservations(self, *args, **kwargs)

    def _dispatch_resolve_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch_resolve_falling_island_reservations(self, *args, **kwargs)

    def _read_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _read_falling_island_reservations(self, *args, **kwargs)

    def resolve_falling_island_shifts(self, *args: Any, **kwargs: Any) -> Any:
        return resolve_falling_island_shifts(self, *args, **kwargs)

    def label_falling_island_components(self, *args: Any, **kwargs: Any) -> Any:
        return label_falling_island_components(self, *args, **kwargs)

    def label_falling_island_component_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return label_falling_island_component_metadata(self, *args, **kwargs)

    def label_falling_island_component_metadata_texture(self, *args: Any, **kwargs: Any) -> Any:
        return label_falling_island_component_metadata_texture(self, *args, **kwargs)

    def _summarize_falling_island_label_texture(self, *args: Any, **kwargs: Any) -> Any:
        return _summarize_falling_island_label_texture(self, *args, **kwargs)

    def relabel_falling_island_components(self, *args: Any, **kwargs: Any) -> Any:
        return relabel_falling_island_components(self, *args, **kwargs)

    def relabel_falling_island_component_texture(self, *args: Any, **kwargs: Any) -> Any:
        return relabel_falling_island_component_texture(self, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def integrate_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return integrate_velocity(self, *args, **kwargs)

    def can_consume_deferred_heat_core(self, *args: Any, **kwargs: Any) -> Any:
        return can_consume_deferred_heat_core(self, *args, **kwargs)

    def can_consume_reaction_handoff(self, *args: Any, **kwargs: Any) -> Any:
        return can_consume_reaction_handoff(self, *args, **kwargs)

    def _integrate_reaction_handoff(self, *args: Any, **kwargs: Any) -> Any:
        return _integrate_reaction_handoff(self, *args, **kwargs)
