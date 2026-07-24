from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.sim.gpu_base import GPUPipelineBase

# Facade re-exports: the constants and resources dataclass live in the leaf
# module ``gpu_liquid_resources`` so the gpu_liquid_* satellites can import them
# without cycling back through this hub.
from oracle_game.sim.gpu_liquid_resources import (
    MAX_MATERIALS,
    PASS_LOCAL_SIZE,
    TILE_LOCAL_SIZE,
    TILE_SIZE,
    GPULiquidResources,
)
from oracle_game.sim.shader_loader import build_compute_shader

_SHADER_SUBS = {
    "PASS_LOCAL_SIZE": PASS_LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "TILE_LOCAL_SIZE": TILE_LOCAL_SIZE,
    "TILE_SIZE": TILE_SIZE,
    "PASS_LOCAL_SIZE_MINUS_1": PASS_LOCAL_SIZE - 1,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
    "TILE_WARP_EXTENSIONS": "",
    "TILE_WARP_FAST_PATH": 0,
    "TILE_WARP_DIRECT_VERTICAL_MAPPING": 0,
    "TILE_WARP_PROVENANCE_ROW_STREAM": 0,
    "TILE_WARP_LANE_CHANGE_VOTE": 0,
    "DIRECT_BRIDGE_INPUTS": 0,
    "DIRECT_BRIDGE_CELL_INPUTS": 0,
    "TILE_SNAPSHOT_OUTPUT": 0,
    "TILE_COMPACT_SNAPSHOT": 0,
    "TILE_SNAPSHOT_PRE_STATE": 0,
    "TILE_PACKED_PRE_STATE_BLOCKER": 0,
    "TILE_SNAPSHOT_STATE_ELISION": 0,
    "BUOYANCY_SHARED_SINK_CACHE": 0,
    "SEAM_SNAPSHOT_INPUT": 0,
    "SEAM_COMPACT_SNAPSHOT": 0,
    "SEAM_ROW_LEADER": 0,
    "SEAM_ROWS_PER_GROUP": 1,
    "DIRECT_BRIDGE_AUX_OUTPUTS": 0,
    "DIRECT_BRIDGE_AUX_INPUTS": 0,
    "DIRECT_BRIDGE_ISLAND_INPUTS": 0,
    "DIRECT_BRIDGE_ENTITY_INPUTS": 0,
    "LIQUID_PROVENANCE": 0,
    "PROVENANCE_TERMINAL": 0,
    "FUSE_CLEANUP": 0,
    "FUSE_ACTIVE_DECAY": 0,
    "RESTORE_BRIDGE_AUX_OUTPUTS": 0,
    "PROVENANCE_ACTIVE_MASK_CACHE": 0,
    "PROVENANCE_SHARED_META_CACHE": 0,
    "PROVENANCE_LAZY_AUX": 0,
    "TILE_LIQUID_KIND_CACHE": 0,
    "TILE_BLOCKER_MASK_INPUT": 0,
}


from oracle_game.sim.gpu_liquid_bridge import (
    _barrier_bits,
    _download_outputs,
    _load_authoritative_bridge_flow_intent_inputs,
    _load_authoritative_bridge_inputs,
    _publish_bridge_outputs,
    _run_liquid_intent_pass,
    _upload_inputs,
    prepare_motion_flow_intent,
    step,
)
from oracle_game.sim.gpu_liquid_resources import (
    _active_scheduler_gpu_authoritative,
    _active_tile_workgroups_per_tile,
    _ensure_resources,
    _load_authoritative_active_tile_mask,
    _next_placeholder_claim_epoch,
    _refresh_active_scheduler_from_ttl,
    _reload_and_compact_active_cell_tiles,
    _run_active_tile_indirect,
    _seam_workgroups_per_boundary,
    _upload_active_tile_mask,
    release,
)
from oracle_game.sim.gpu_liquid_solve import (
    _build_placeholder_dirty_affected_tile_dispatch,
    _build_seam_boundary_dispatch,
    _compact_active_tiles,
    _prefetch_seam_boundary_bridge_inputs,
    _run_buoyancy_pass,
    _run_cleanup_runtime,
    _run_copy_core_state,
    _run_copy_for_placeholder,
    _run_placeholder_displacement,
    _run_provenance_init,
    _run_seam_pass,
    _run_tile_solve,
)


class GPULiquidPipeline(GPUPipelineBase):
    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
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
        self.tile_solve_warp_fast_path = False
        self._tile_warp_direct_vertical_mapping_enabled = (
            self.engine_config.tile_warp_direct_vertical_mapping_enabled
        )
        self._tile_warp_provenance_row_stream_enabled = (
            self.engine_config.tile_warp_provenance_row_stream_enabled
        )
        # A per-lane register flag plus one warp vote avoids shared atomic
        # contention while preserving the tile-level active TTL update.
        self._tile_warp_lane_change_vote_enabled = (
            self.engine_config.tile_warp_lane_change_vote_enabled
        )
        self._tile_solve_bridge_hydration_fusion_enabled = (
            self.engine_config.tile_solve_bridge_hydration_fusion_enabled
        )
        self._placeholder_lazy_roles_enabled = self.engine_config.placeholder_lazy_roles_enabled
        self._tile_solve_snapshot_output_fusion_enabled = (
            self.engine_config.tile_solve_snapshot_output_fusion_enabled
        )
        self._compact_tile_solve_snapshot_enabled = (
            self.engine_config.compact_tile_solve_snapshot_enabled
        )
        # Candidate: cell_state_in already receives the tile result. Keep only
        # packed source/pre-state metadata in the compact snapshot and let seam
        # X sample that canonical state texture directly.
        self._tile_snapshot_state_elision_enabled = (
            self.engine_config.tile_snapshot_state_elision_enabled
        )
        self._tile_snapshot_state_elision_frame_enabled = False
        self.last_tile_snapshot_state_elision_used = False
        # Candidate: the warp snapshot-pre specialization only needs one
        # immutable 16-bit destination state plus one mutable blocker bit per
        # cell. Pack those separately instead of reserving a uint per cell.
        self._tile_packed_pre_state_blocker_enabled = (
            self.engine_config.tile_packed_pre_state_blocker_enabled
        )
        self._buoyancy_pass_fusion_enabled = self.engine_config.buoyancy_pass_fusion_enabled
        self._seam_x_row_leader_enabled = self.engine_config.seam_x_row_leader_enabled
        # Pack four independent row-leader warps into one workgroup. Frames
        # outside the strict gate retain the canonical one-warp layout.
        self._seam_x_multirow_rows = 4
        self._seam_x_multirow_frame_rows = 0
        # Do not launch seam hydration workgroups when every tile is already
        # resident and the prefetch shader would immediately return.
        self._seam_prefetch_zero_full_active_enabled = (
            self.engine_config.seam_prefetch_zero_full_active_enabled
        )
        self._seam_y_shared_snapshot_enabled = self.engine_config.seam_y_shared_snapshot_enabled
        self._bridge_aux_cleanup_fusion_enabled = (
            self.engine_config.bridge_aux_cleanup_fusion_enabled
        )
        # The cell-independent cleanup copy is faster as four 16x16 groups per
        # 32x32 tile than as sixteen 8x8 groups on the target GPU.
        self._cleanup_local16_enabled = self.engine_config.cleanup_local16_enabled
        self.last_cleanup_flow_fusion_used = False
        self._flow_intent_shared_halo_enabled = self.engine_config.flow_intent_shared_halo_enabled
        # Experimental: terminal flow already resolves provenance while
        # filling its shared halo. Preserve that resolution per halo entry so
        # the output lane can reuse it for velocity and core materialization.
        self._flow_intent_provenance_shared_meta_cache_enabled = (
            self.engine_config.flow_intent_provenance_shared_meta_cache_enabled
        )
        # Terminal flow reads authoritative bridge aux state. Defer those
        # reads until a probed cell is otherwise reachable-empty instead of
        # hydrating two aux values for every shared-halo entry.
        self._flow_intent_provenance_lazy_aux_enabled = (
            self.engine_config.flow_intent_provenance_lazy_aux_enabled
        )
        self._flow_active_decay_fusion_frame_id: int | None = None
        self.last_flow_active_decay_fusion_used = False
        # Carry liquid payload provenance through the tile/seam passes and
        # materialize the authoritative bridge once at flow termination.  The
        # formal gate preserves the canonical path for partial/CPU frames.
        self._provenance_terminal_enabled = self.engine_config.provenance_terminal_enabled
        self._provenance_terminal_frame_enabled = False
        # Experimental: when every tile is already active, tile solve writes
        # every provenance entry and the identity init dispatch can be zeroed.
        # Partial-active frames retain the full init for newly activated tiles.
        self._provenance_init_fusion_enabled = self.engine_config.provenance_init_fusion_enabled
        self._provenance_init_fusion_frame_enabled = False
        self.last_provenance_init_fusion_used = False
        self.last_provenance_terminal_used = False
        self.last_provenance_cleanup_terminal_fusion_used = False
        # Candidate v2: cleanup ordinary active cells while buoyancy already
        # owns their final state, then restore only placeholder-affected tiles
        # after displacement. This keeps cleanup out of the flow halo.
        self._buoyancy_cleanup_split_fusion_enabled = (
            self.engine_config.buoyancy_cleanup_split_fusion_enabled
        )
        self._buoyancy_cleanup_split_fusion_frame_enabled = False
        self.last_buoyancy_cleanup_split_fusion_used = False
        # Candidate: retain immutable destination material/phase in the compact
        # tile snapshot so fused buoyancy cleanup can avoid bridge AoS reads.
        self._buoyancy_snapshot_pre_state_enabled = (
            self.engine_config.buoyancy_snapshot_pre_state_enabled
        )
        self._buoyancy_snapshot_pre_state_frame_enabled = False
        self.last_buoyancy_snapshot_pre_state_used = False
        # Candidate: full-active formal frames reuse the deterministic vertical
        # sink transform within each 8x8 workgroup. Partial frames execute the
        # canonical texture path inside the same specialization.
        self._buoyancy_shared_sink_cache_enabled = (
            self.engine_config.buoyancy_shared_sink_cache_enabled
        )
        self._buoyancy_shared_sink_cache_frame_enabled = False
        self.last_buoyancy_shared_sink_cache_used = False
        # Candidate: hydrate the tile blocker predicate plus displaced IDs,
        # then read island/entity identities directly during cleanup. Frames
        # outside the compact snapshot-pre schedule retain canonical hydration.
        self._buoyancy_blocker_displaced_hydration_enabled = (
            self.engine_config.buoyancy_blocker_displaced_hydration_enabled
        )
        self._buoyancy_blocker_displaced_hydration_frame_enabled = False
        self.last_buoyancy_blocker_displaced_hydration_used = False
        self._blocker_displaced_hydration_frame_enabled = False

    # ``available`` inherited from GPUPipelineBase.
    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``_formal_gpu_frame`` inherited from GPUPipelineBase.
    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = build_compute_shader(
            ctx, "liquid/load_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["clear_active_tile_dispatch"] = build_compute_shader(
            ctx, "_shared/clear_active_tile_dispatch.comp", _SHADER_SUBS
        )
        self.programs["retarget_active_tile_dispatch"] = build_compute_shader(
            ctx, "liquid/retarget_active_tile_dispatch.comp", _SHADER_SUBS
        )
        self.programs["retarget_seam_prefetch_dispatch"] = build_compute_shader(
            ctx,
            "liquid/retarget_seam_prefetch_dispatch.comp",
            _SHADER_SUBS,
        )
        self.programs["compact_active_tiles"] = build_compute_shader(
            ctx, "_shared/compact_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["compact_active_tiles_from_chunks"] = build_compute_shader(
            ctx, "_shared/compact_active_tiles_from_chunks.comp", _SHADER_SUBS
        )
        self.programs["compact_placeholder_dirty_affected_tiles"] = build_compute_shader(
            ctx, "liquid/compact_placeholder_dirty_affected_tiles.comp", _SHADER_SUBS
        )
        self.programs["compact_placeholder_active_pending_affected_tiles"] = build_compute_shader(
            ctx, "liquid/compact_placeholder_active_pending_affected_tiles.comp", _SHADER_SUBS
        )
        self.programs["compact_placeholder_active_pending_affected_tiles_bridge_aux"] = (
            build_compute_shader(
                ctx,
                "liquid/compact_placeholder_active_pending_affected_tiles.comp",
                {**_SHADER_SUBS, "DIRECT_BRIDGE_AUX_INPUTS": 1},
            )
        )
        self.programs["clear_affected_tile_flags"] = build_compute_shader(
            ctx, "liquid/clear_affected_tile_flags.comp", _SHADER_SUBS
        )
        self.programs["compact_seam_x_boundaries_from_active_tiles"] = build_compute_shader(
            ctx, "liquid/compact_seam_x_boundaries_from_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["compact_seam_y_boundaries_from_active_tiles"] = build_compute_shader(
            ctx, "liquid/compact_seam_y_boundaries_from_active_tiles.comp", _SHADER_SUBS
        )
        self.programs["prefetch_seam_boundary_bridge_inputs"] = build_compute_shader(
            ctx, "liquid/prefetch_seam_boundary_bridge_inputs.comp", _SHADER_SUBS
        )
        self.programs["prefetch_seam_boundary_bridge_aux_inputs"] = build_compute_shader(
            ctx, "liquid/prefetch_seam_boundary_bridge_aux_inputs.comp", _SHADER_SUBS
        )
        required_warp_extensions = {
            "GL_NV_gpu_shader5",
            "GL_NV_shader_thread_group",
            "GL_NV_shader_thread_shuffle",
        }
        available_extensions = set(getattr(ctx, "extensions", ()))
        warp_size = 0
        if required_warp_extensions.issubset(available_extensions):
            warp_size_program = build_compute_shader(ctx, "_shared/query_nv_warp_size.comp")
            warp_size_buffer = ctx.buffer(reserve=np.dtype(np.uint32).itemsize)
            try:
                warp_size_buffer.bind_to_storage_buffer(binding=0)
                warp_size_program.run(1, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
                warp_size = int(np.frombuffer(warp_size_buffer.read(), dtype=np.uint32, count=1)[0])
            finally:
                warp_size_buffer.release()
                warp_size_program.release()
        self.tile_solve_warp_fast_path = bool(
            TILE_LOCAL_SIZE == 32 and warp_size == TILE_LOCAL_SIZE
        )
        tile_solve_subs = dict(_SHADER_SUBS)
        tile_solve_subs["TILE_WARP_FAST_PATH"] = int(self.tile_solve_warp_fast_path)
        tile_solve_subs["TILE_WARP_DIRECT_VERTICAL_MAPPING"] = int(
            self._tile_warp_direct_vertical_mapping_enabled
        )
        tile_solve_subs["TILE_NO_LIQUID_FAST_PATH"] = 0
        tile_solve_subs["TILE_WARP_LANE_CHANGE_VOTE"] = int(
            self._tile_warp_lane_change_vote_enabled
        )
        tile_solve_subs["TILE_PACKED_PRE_STATE_BLOCKER"] = int(
            self._tile_packed_pre_state_blocker_enabled
        )
        if self.tile_solve_warp_fast_path:
            tile_solve_subs["TILE_WARP_EXTENSIONS"] = "\n".join(
                (
                    "#extension GL_NV_gpu_shader5 : require",
                    "#extension GL_NV_shader_thread_group : require",
                    "#extension GL_NV_shader_thread_shuffle : require",
                )
            )
        self.programs["tile_solve"] = build_compute_shader(
            ctx, "liquid/tile_solve.comp", tile_solve_subs
        )
        self.programs["tile_solve_bridge"] = build_compute_shader(
            ctx,
            "liquid/tile_solve.comp",
            {**tile_solve_subs, "DIRECT_BRIDGE_INPUTS": 1},
        )
        self.programs["init_liquid_provenance"] = build_compute_shader(
            ctx,
            "liquid/init_provenance.comp",
            _SHADER_SUBS,
        )
        self.programs["retarget_provenance_init_dispatch"] = build_compute_shader(
            ctx,
            "liquid/retarget_provenance_init_dispatch.comp",
            _SHADER_SUBS,
        )
        if self._buoyancy_snapshot_pre_state_enabled:
            self.programs["advance_tile_snapshot_token"] = build_compute_shader(
                ctx,
                "liquid/advance_tile_snapshot_token.comp",
            )
            self.programs["validate_tile_snapshot_coverage"] = build_compute_shader(
                ctx,
                "liquid/validate_tile_snapshot_coverage.comp",
            )
        self.programs["tile_solve_bridge_snapshot"] = build_compute_shader(
            ctx,
            "liquid/tile_solve.comp",
            {
                **tile_solve_subs,
                "DIRECT_BRIDGE_INPUTS": 1,
                "TILE_SNAPSHOT_OUTPUT": 1,
                "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
            },
        )
        self.programs["tile_solve_bridge_snapshot_provenance"] = build_compute_shader(
            ctx,
            "liquid/tile_solve.comp",
            {
                **tile_solve_subs,
                "DIRECT_BRIDGE_INPUTS": 1,
                "TILE_SNAPSHOT_OUTPUT": 1,
                "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "LIQUID_PROVENANCE": 1,
            },
        )
        self.programs["tile_solve_bridge_snapshot_provenance_blocker"] = build_compute_shader(
            ctx,
            "liquid/tile_solve.comp",
            {
                **tile_solve_subs,
                "DIRECT_BRIDGE_INPUTS": 1,
                "TILE_SNAPSHOT_OUTPUT": 1,
                "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "LIQUID_PROVENANCE": 1,
                "TILE_BLOCKER_MASK_INPUT": 1,
            },
        )
        if self.tile_solve_warp_fast_path:
            self.programs["tile_solve_bridge_snapshot_row_stream"] = build_compute_shader(
                ctx,
                "liquid/tile_solve.comp",
                {
                    **tile_solve_subs,
                    "DIRECT_BRIDGE_INPUTS": 1,
                    "TILE_SNAPSHOT_OUTPUT": 1,
                    "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                    "TILE_WARP_PROVENANCE_ROW_STREAM": 1,
                },
            )
            self.programs["tile_solve_bridge_snapshot_row_stream_provenance"] = (
                build_compute_shader(
                    ctx,
                    "liquid/tile_solve.comp",
                    {
                        **tile_solve_subs,
                        "DIRECT_BRIDGE_INPUTS": 1,
                        "TILE_SNAPSHOT_OUTPUT": 1,
                        "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                        "TILE_WARP_PROVENANCE_ROW_STREAM": 1,
                        "LIQUID_PROVENANCE": 1,
                    },
                )
            )
            self.programs["tile_solve_bridge_snapshot_row_stream_provenance_blocker"] = (
                build_compute_shader(
                    ctx,
                    "liquid/tile_solve.comp",
                    {
                        **tile_solve_subs,
                        "DIRECT_BRIDGE_INPUTS": 1,
                        "TILE_SNAPSHOT_OUTPUT": 1,
                        "TILE_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                        "TILE_WARP_PROVENANCE_ROW_STREAM": 1,
                        "LIQUID_PROVENANCE": 1,
                        "TILE_BLOCKER_MASK_INPUT": 1,
                    },
                )
            )
        if self._buoyancy_snapshot_pre_state_enabled:
            for name, row_stream, direct_aux, blocker_mask in (
                ("tile_solve_bridge_snapshot_provenance_snapshot_pre", 0, 0, 0),
                ("tile_solve_bridge_snapshot_provenance_blocker_snapshot_pre", 0, 0, 1),
                ("tile_solve_bridge_snapshot_row_stream_provenance_snapshot_pre", 1, 0, 0),
                ("tile_solve_bridge_snapshot_row_stream_provenance_blocker_snapshot_pre", 1, 0, 1),
            ):
                if row_stream and not self.tile_solve_warp_fast_path:
                    continue
                self.programs[name] = build_compute_shader(
                    ctx,
                    "liquid/tile_solve.comp",
                    {
                        **tile_solve_subs,
                        "DIRECT_BRIDGE_INPUTS": 1,
                        "DIRECT_BRIDGE_AUX_INPUTS": direct_aux,
                        "TILE_SNAPSHOT_OUTPUT": 1,
                        "TILE_COMPACT_SNAPSHOT": 1,
                        "TILE_WARP_PROVENANCE_ROW_STREAM": row_stream,
                        "LIQUID_PROVENANCE": 1,
                        "TILE_SNAPSHOT_PRE_STATE": 1,
                        "TILE_BLOCKER_MASK_INPUT": blocker_mask,
                    },
                )
                if self._tile_snapshot_state_elision_enabled:
                    self.programs[f"{name}_state_elided"] = build_compute_shader(
                        ctx,
                        "liquid/tile_solve.comp",
                        {
                            **tile_solve_subs,
                            "DIRECT_BRIDGE_INPUTS": 1,
                            "DIRECT_BRIDGE_AUX_INPUTS": direct_aux,
                            "TILE_SNAPSHOT_OUTPUT": 1,
                            "TILE_COMPACT_SNAPSHOT": 1,
                            "TILE_WARP_PROVENANCE_ROW_STREAM": row_stream,
                            "LIQUID_PROVENANCE": 1,
                            "TILE_SNAPSHOT_PRE_STATE": 1,
                            "TILE_SNAPSHOT_STATE_ELISION": 1,
                            "TILE_BLOCKER_MASK_INPUT": blocker_mask,
                        },
                    )
        self.programs["seam_x"] = build_compute_shader(ctx, "liquid/seam_x.comp", _SHADER_SUBS)
        self.programs["seam_x_snapshot"] = build_compute_shader(
            ctx,
            "liquid/seam_x.comp",
            {
                **_SHADER_SUBS,
                "SEAM_SNAPSHOT_INPUT": 1,
                "SEAM_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
            },
        )
        self.programs["seam_x_snapshot_row_leader"] = build_compute_shader(
            ctx,
            "liquid/seam_x.comp",
            {
                **_SHADER_SUBS,
                "SEAM_SNAPSHOT_INPUT": 1,
                "SEAM_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "SEAM_ROW_LEADER": 1,
            },
        )
        self.programs["seam_x_snapshot_row_leader_provenance"] = build_compute_shader(
            ctx,
            "liquid/seam_x.comp",
            {
                **_SHADER_SUBS,
                "SEAM_SNAPSHOT_INPUT": 1,
                "SEAM_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "SEAM_ROW_LEADER": 1,
                "LIQUID_PROVENANCE": 1,
            },
        )
        self.programs["seam_x_snapshot_row_leader4"] = build_compute_shader(
            ctx,
            "liquid/seam_x.comp",
            {
                **_SHADER_SUBS,
                "SEAM_SNAPSHOT_INPUT": 1,
                "SEAM_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "SEAM_ROW_LEADER": 1,
                "SEAM_ROWS_PER_GROUP": 4,
            },
        )
        self.programs["seam_x_snapshot_row_leader4_provenance"] = build_compute_shader(
            ctx,
            "liquid/seam_x.comp",
            {
                **_SHADER_SUBS,
                "SEAM_SNAPSHOT_INPUT": 1,
                "SEAM_COMPACT_SNAPSHOT": int(self._compact_tile_solve_snapshot_enabled),
                "SEAM_ROW_LEADER": 1,
                "SEAM_ROWS_PER_GROUP": 4,
                "LIQUID_PROVENANCE": 1,
            },
        )
        if self._buoyancy_snapshot_pre_state_enabled:
            for name, row_leader, rows_per_group in (
                ("seam_x_snapshot_row_leader_provenance_snapshot_pre", 1, 1),
                ("seam_x_snapshot_row_leader4_provenance_snapshot_pre", 1, 4),
            ):
                self.programs[name] = build_compute_shader(
                    ctx,
                    "liquid/seam_x.comp",
                    {
                        **_SHADER_SUBS,
                        "SEAM_SNAPSHOT_INPUT": 1,
                        "SEAM_COMPACT_SNAPSHOT": 1,
                        "SEAM_ROW_LEADER": row_leader,
                        "SEAM_ROWS_PER_GROUP": rows_per_group,
                        "LIQUID_PROVENANCE": 1,
                        "TILE_SNAPSHOT_PRE_STATE": 1,
                    },
                )
                if self._tile_snapshot_state_elision_enabled:
                    self.programs[f"{name}_state_elided"] = build_compute_shader(
                        ctx,
                        "liquid/seam_x.comp",
                        {
                            **_SHADER_SUBS,
                            "SEAM_SNAPSHOT_INPUT": 1,
                            "SEAM_COMPACT_SNAPSHOT": 1,
                            "SEAM_ROW_LEADER": row_leader,
                            "SEAM_ROWS_PER_GROUP": rows_per_group,
                            "LIQUID_PROVENANCE": 1,
                            "TILE_SNAPSHOT_PRE_STATE": 1,
                            "TILE_SNAPSHOT_STATE_ELISION": 1,
                        },
                    )
        self.programs["seam_y"] = build_compute_shader(ctx, "liquid/seam_y.comp", _SHADER_SUBS)
        self.programs["seam_y_shared_snapshot"] = build_compute_shader(
            ctx,
            "liquid/seam_y_shared_snapshot.comp",
            _SHADER_SUBS,
        )
        self.programs["seam_y_shared_snapshot_provenance_aux"] = build_compute_shader(
            ctx,
            "liquid/seam_y_shared_snapshot.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_AUX_INPUTS": 1,
                "LIQUID_PROVENANCE": 1,
            },
        )
        self.programs["buoyancy_float"] = build_compute_shader(
            ctx, "liquid/buoyancy_float.comp", _SHADER_SUBS
        )
        self.programs["buoyancy_fused"] = build_compute_shader(
            ctx, "liquid/buoyancy_fused.comp", _SHADER_SUBS
        )
        self.programs["buoyancy_fused_provenance"] = build_compute_shader(
            ctx, "liquid/buoyancy_fused.comp", {**_SHADER_SUBS, "LIQUID_PROVENANCE": 1}
        )
        self.programs["buoyancy_fused_provenance_cleanup"] = build_compute_shader(
            ctx,
            "liquid/buoyancy_fused.comp",
            {**_SHADER_SUBS, "LIQUID_PROVENANCE": 1, "FUSE_CLEANUP": 1},
        )
        if self._buoyancy_snapshot_pre_state_enabled:
            self.programs["buoyancy_fused_provenance_cleanup_snapshot_pre"] = build_compute_shader(
                ctx,
                "liquid/buoyancy_fused.comp",
                {
                    **_SHADER_SUBS,
                    "LIQUID_PROVENANCE": 1,
                    "FUSE_CLEANUP": 1,
                    "TILE_SNAPSHOT_PRE_STATE": 1,
                },
            )
            if self._tile_snapshot_state_elision_enabled:
                self.programs["buoyancy_fused_provenance_cleanup_snapshot_pre_state_elided"] = (
                    build_compute_shader(
                        ctx,
                        "liquid/buoyancy_fused.comp",
                        {
                            **_SHADER_SUBS,
                            "LIQUID_PROVENANCE": 1,
                            "FUSE_CLEANUP": 1,
                            "TILE_SNAPSHOT_PRE_STATE": 1,
                            "TILE_SNAPSHOT_STATE_ELISION": 1,
                        },
                    )
                )
            if self._buoyancy_shared_sink_cache_enabled:
                self.programs["buoyancy_fused_provenance_cleanup_snapshot_pre_shared_sink"] = (
                    build_compute_shader(
                        ctx,
                        "liquid/buoyancy_fused.comp",
                        {
                            **_SHADER_SUBS,
                            "LIQUID_PROVENANCE": 1,
                            "FUSE_CLEANUP": 1,
                            "TILE_SNAPSHOT_PRE_STATE": 1,
                            "BUOYANCY_SHARED_SINK_CACHE": 1,
                        },
                    )
                )
                if self._tile_snapshot_state_elision_enabled:
                    self.programs[
                        "buoyancy_fused_provenance_cleanup_snapshot_pre_state_elided_shared_sink"
                    ] = build_compute_shader(
                        ctx,
                        "liquid/buoyancy_fused.comp",
                        {
                            **_SHADER_SUBS,
                            "LIQUID_PROVENANCE": 1,
                            "FUSE_CLEANUP": 1,
                            "TILE_SNAPSHOT_PRE_STATE": 1,
                            "TILE_SNAPSHOT_STATE_ELISION": 1,
                            "BUOYANCY_SHARED_SINK_CACHE": 1,
                        },
                    )
        self.programs["copy_with_pending"] = build_compute_shader(
            ctx, "liquid/copy_with_pending.comp", _SHADER_SUBS
        )
        self.programs["copy_with_pending_provenance"] = build_compute_shader(
            ctx, "liquid/copy_with_pending.comp", {**_SHADER_SUBS, "LIQUID_PROVENANCE": 1}
        )
        self.programs["copy_core_state"] = build_compute_shader(
            ctx, "liquid/copy_core_state.comp", _SHADER_SUBS
        )
        self.programs["placeholder_displace"] = build_compute_shader(
            ctx, "liquid/placeholder_displace.comp", _SHADER_SUBS
        )
        self.programs["placeholder_displace_provenance"] = build_compute_shader(
            ctx, "liquid/placeholder_displace.comp", {**_SHADER_SUBS, "LIQUID_PROVENANCE": 1}
        )
        self.programs["cleanup_runtime"] = build_compute_shader(
            ctx, "liquid/cleanup_runtime.comp", _SHADER_SUBS
        )
        self.programs["cleanup_runtime_bridge"] = build_compute_shader(
            ctx,
            "liquid/cleanup_runtime.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_INPUTS": 1},
        )
        self.programs["cleanup_runtime_bridge_aux"] = build_compute_shader(
            ctx,
            "liquid/cleanup_runtime.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_INPUTS": 1, "DIRECT_BRIDGE_AUX_OUTPUTS": 1},
        )
        self.programs["cleanup_runtime_bridge_aux16"] = build_compute_shader(
            ctx,
            "liquid/cleanup_runtime.comp",
            {
                **_SHADER_SUBS,
                "PASS_LOCAL_SIZE": 16,
                "PASS_LOCAL_SIZE_MINUS_1": 15,
                "DIRECT_BRIDGE_INPUTS": 1,
                "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
            },
        )
        self.programs["cleanup_runtime_bridge_aux_restore16"] = build_compute_shader(
            ctx,
            "liquid/cleanup_runtime.comp",
            {
                **_SHADER_SUBS,
                "PASS_LOCAL_SIZE": 16,
                "PASS_LOCAL_SIZE_MINUS_1": 15,
                "DIRECT_BRIDGE_INPUTS": 1,
                "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
                "RESTORE_BRIDGE_AUX_OUTPUTS": 1,
            },
        )
        self.programs["cleanup_runtime_bridge_aux_restore_bridge_ids16"] = build_compute_shader(
            ctx,
            "liquid/cleanup_runtime.comp",
            {
                **_SHADER_SUBS,
                "PASS_LOCAL_SIZE": 16,
                "PASS_LOCAL_SIZE_MINUS_1": 15,
                "DIRECT_BRIDGE_INPUTS": 1,
                "DIRECT_BRIDGE_AUX_OUTPUTS": 1,
                "DIRECT_BRIDGE_ISLAND_INPUTS": 1,
                "DIRECT_BRIDGE_ENTITY_INPUTS": 1,
                "RESTORE_BRIDGE_AUX_OUTPUTS": 1,
            },
        )
        self.programs["liquid_flow_intent"] = build_compute_shader(
            ctx, "liquid/liquid_flow_intent.comp", _SHADER_SUBS
        )
        self.programs["liquid_flow_intent_shared_halo"] = build_compute_shader(
            ctx,
            "liquid/liquid_flow_intent_shared_halo.comp",
            _SHADER_SUBS,
        )
        self.programs["liquid_flow_intent_shared_halo_provenance_bridge_aux"] = (
            build_compute_shader(
                ctx,
                "liquid/liquid_flow_intent_shared_halo.comp",
                {
                    **_SHADER_SUBS,
                    "DIRECT_BRIDGE_AUX_INPUTS": 1,
                    "LIQUID_PROVENANCE": 1,
                    "PROVENANCE_TERMINAL": 1,
                },
            )
        )
        self.programs["liquid_flow_intent_shared_halo_provenance_shared_meta"] = (
            build_compute_shader(
                ctx,
                "liquid/liquid_flow_intent_shared_halo.comp",
                {
                    **_SHADER_SUBS,
                    "DIRECT_BRIDGE_AUX_INPUTS": 1,
                    "LIQUID_PROVENANCE": 1,
                    "PROVENANCE_TERMINAL": 1,
                    "PROVENANCE_ACTIVE_MASK_CACHE": 1,
                    "PROVENANCE_SHARED_META_CACHE": 1,
                },
            )
        )
        self.programs["liquid_flow_intent_shared_halo_provenance_shared_meta_lazy_aux"] = (
            build_compute_shader(
                ctx,
                "liquid/liquid_flow_intent_shared_halo.comp",
                {
                    **_SHADER_SUBS,
                    "DIRECT_BRIDGE_AUX_INPUTS": 1,
                    "LIQUID_PROVENANCE": 1,
                    "PROVENANCE_TERMINAL": 1,
                    "PROVENANCE_ACTIVE_MASK_CACHE": 1,
                    "PROVENANCE_SHARED_META_CACHE": 1,
                    "PROVENANCE_LAZY_AUX": 1,
                },
            )
        )
        self.programs["load_bridge_cell"] = build_compute_shader(
            ctx, "liquid/load_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_blocker_displaced"] = build_compute_shader(
            ctx,
            "liquid/load_bridge_blocker_displaced.comp",
            _SHADER_SUBS,
        )
        self.programs["load_bridge_flow_intent_inputs"] = build_compute_shader(
            ctx, "liquid/load_bridge_flow_intent_inputs.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_out"] = build_compute_shader(
            ctx, "liquid/load_bridge_cell_out.comp", _SHADER_SUBS
        )
        self.programs["load_bridge_cell_aux"] = build_compute_shader(
            ctx, "liquid/load_bridge_cell_aux.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_cell"] = build_compute_shader(
            ctx, "liquid/publish_bridge_cell.comp", _SHADER_SUBS
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

    # --- resources bucket ---
    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def _ensure_resources(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_resources(self, *args, **kwargs)

    def _active_scheduler_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return _active_scheduler_gpu_authoritative(self, *args, **kwargs)

    def _refresh_active_scheduler_from_ttl(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_active_scheduler_from_ttl(self, *args, **kwargs)

    def _upload_active_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_active_tile_mask(self, *args, **kwargs)

    def _load_authoritative_active_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_active_tile_mask(self, *args, **kwargs)

    def _active_tile_workgroups_per_tile(self, *args: Any, **kwargs: Any) -> Any:
        return _active_tile_workgroups_per_tile(self, *args, **kwargs)

    def _next_placeholder_claim_epoch(self, *args: Any, **kwargs: Any) -> Any:
        return _next_placeholder_claim_epoch(self, *args, **kwargs)

    def _seam_workgroups_per_boundary(self, *args: Any, **kwargs: Any) -> Any:
        return _seam_workgroups_per_boundary(self, *args, **kwargs)

    def _reload_and_compact_active_cell_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _reload_and_compact_active_cell_tiles(self, *args, **kwargs)

    def _run_active_tile_indirect(self, *args: Any, **kwargs: Any) -> Any:
        return _run_active_tile_indirect(self, *args, **kwargs)

    # --- solve bucket ---
    def _build_seam_boundary_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return _build_seam_boundary_dispatch(self, *args, **kwargs)

    def _prefetch_seam_boundary_bridge_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _prefetch_seam_boundary_bridge_inputs(self, *args, **kwargs)

    def _build_placeholder_dirty_affected_tile_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return _build_placeholder_dirty_affected_tile_dispatch(self, *args, **kwargs)

    def _compact_active_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _compact_active_tiles(self, *args, **kwargs)

    def _run_tile_solve(self, *args: Any, **kwargs: Any) -> Any:
        return _run_tile_solve(self, *args, **kwargs)

    def _run_seam_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_seam_pass(self, *args, **kwargs)

    def _run_provenance_init(self, *args: Any, **kwargs: Any) -> Any:
        return _run_provenance_init(self, *args, **kwargs)

    def _run_buoyancy_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_buoyancy_pass(self, *args, **kwargs)

    def _run_copy_core_state(self, *args: Any, **kwargs: Any) -> Any:
        return _run_copy_core_state(self, *args, **kwargs)

    def _run_copy_for_placeholder(self, *args: Any, **kwargs: Any) -> Any:
        return _run_copy_for_placeholder(self, *args, **kwargs)

    def _run_placeholder_displacement(self, *args: Any, **kwargs: Any) -> Any:
        return _run_placeholder_displacement(self, *args, **kwargs)

    def _run_cleanup_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _run_cleanup_runtime(self, *args, **kwargs)

    # --- bridge bucket ---
    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def prepare_motion_flow_intent(self, *args: Any, **kwargs: Any) -> Any:
        return prepare_motion_flow_intent(self, *args, **kwargs)

    def _upload_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _upload_inputs(self, *args, **kwargs)

    def _load_authoritative_bridge_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_bridge_inputs(self, *args, **kwargs)

    def _load_authoritative_bridge_flow_intent_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return _load_authoritative_bridge_flow_intent_inputs(self, *args, **kwargs)

    def _publish_bridge_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _publish_bridge_outputs(self, *args, **kwargs)

    def _run_liquid_intent_pass(self, *args: Any, **kwargs: Any) -> Any:
        return _run_liquid_intent_pass(self, *args, **kwargs)

    def _download_outputs(self, *args: Any, **kwargs: Any) -> Any:
        return _download_outputs(self, *args, **kwargs)

    def _barrier_bits(self, *args: Any, **kwargs: Any) -> Any:
        return _barrier_bits(self, *args, **kwargs)
