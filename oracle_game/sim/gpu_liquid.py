from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_material_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.types import Phase

TILE_SIZE = 32
TILE_LOCAL_SIZE = TILE_SIZE
PASS_LOCAL_SIZE = 8
MAX_MATERIALS = 256

_SHADER_SUBS = {
    "PASS_LOCAL_SIZE": PASS_LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "TILE_LOCAL_SIZE": TILE_LOCAL_SIZE,
    "TILE_SIZE": TILE_SIZE,
    "PASS_LOCAL_SIZE_MINUS_1": PASS_LOCAL_SIZE - 1,
    "MAX_MATERIALS_MINUS_1": MAX_MATERIALS - 1,
}


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


from oracle_game.sim.gpu_liquid_resources import (
    release,
    _ensure_resources,
    _active_scheduler_gpu_authoritative,
    _refresh_active_scheduler_from_ttl,
    _upload_active_tile_mask,
    _load_authoritative_active_tile_mask,
    _active_tile_workgroups_per_tile,
    _next_placeholder_claim_epoch,
    _seam_workgroups_per_boundary,
    _reload_and_compact_active_cell_tiles,
    _run_active_tile_indirect
)

from oracle_game.sim.gpu_liquid_solve import (
    _build_seam_boundary_dispatch,
    _prefetch_seam_boundary_bridge_inputs,
    _build_placeholder_dirty_affected_tile_dispatch,
    _compact_active_tiles,
    _run_tile_solve,
    _run_seam_pass,
    _run_buoyancy_pass,
    _run_copy_core_state,
    _run_copy_for_placeholder,
    _run_placeholder_displacement,
    _run_cleanup_runtime
)

from oracle_game.sim.gpu_liquid_bridge import (
    step,
    prepare_motion_flow_intent,
    _upload_inputs,
    _load_authoritative_bridge_inputs,
    _load_authoritative_bridge_flow_intent_inputs,
    _publish_bridge_outputs,
    _run_liquid_intent_pass,
    _download_outputs,
    _barrier_bits
)


class GPULiquidPipeline(GPUPipelineBase):
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
    # ``available`` inherited from GPUPipelineBase.
    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``_formal_gpu_frame`` inherited from GPUPipelineBase.
    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = build_compute_shader(ctx, "liquid/load_active_tiles.comp", _SHADER_SUBS)
        self.programs["clear_active_tile_dispatch"] = build_compute_shader(ctx, "liquid/clear_active_tile_dispatch.comp", _SHADER_SUBS)
        self.programs["compact_active_tiles"] = build_compute_shader(ctx, "liquid/compact_active_tiles.comp", _SHADER_SUBS)
        self.programs["compact_active_tiles_from_chunks"] = build_compute_shader(ctx, "liquid/compact_active_tiles_from_chunks.comp", _SHADER_SUBS)
        self.programs["compact_placeholder_dirty_affected_tiles"] = build_compute_shader(ctx, "liquid/compact_placeholder_dirty_affected_tiles.comp", _SHADER_SUBS)
        self.programs["compact_placeholder_active_pending_affected_tiles"] = build_compute_shader(ctx, "liquid/compact_placeholder_active_pending_affected_tiles.comp", _SHADER_SUBS)
        self.programs["compact_seam_x_boundaries_from_active_tiles"] = build_compute_shader(ctx, "liquid/compact_seam_x_boundaries_from_active_tiles.comp", _SHADER_SUBS)
        self.programs["compact_seam_y_boundaries_from_active_tiles"] = build_compute_shader(ctx, "liquid/compact_seam_y_boundaries_from_active_tiles.comp", _SHADER_SUBS)
        self.programs["prefetch_seam_boundary_bridge_inputs"] = build_compute_shader(ctx, "liquid/prefetch_seam_boundary_bridge_inputs.comp", _SHADER_SUBS)
        self.programs["prefetch_seam_boundary_bridge_aux_inputs"] = build_compute_shader(ctx, "liquid/prefetch_seam_boundary_bridge_aux_inputs.comp", _SHADER_SUBS)
        self.programs["tile_solve"] = build_compute_shader(ctx, "liquid/tile_solve.comp", _SHADER_SUBS)
        self.programs["seam_x"] = build_compute_shader(ctx, "liquid/seam_x.comp", _SHADER_SUBS)
        self.programs["seam_y"] = build_compute_shader(ctx, "liquid/seam_y.comp", _SHADER_SUBS)
        self.programs["buoyancy_sink"] = build_compute_shader(ctx, "liquid/buoyancy_sink.comp", _SHADER_SUBS)
        self.programs["buoyancy_float"] = build_compute_shader(ctx, "liquid/buoyancy_float.comp", _SHADER_SUBS)
        self.programs["copy_with_pending"] = build_compute_shader(ctx, "liquid/copy_with_pending.comp", _SHADER_SUBS)
        self.programs["copy_core_state"] = build_compute_shader(ctx, "liquid/copy_core_state.comp", _SHADER_SUBS)
        self.programs["placeholder_displace"] = build_compute_shader(ctx, "liquid/placeholder_displace.comp", _SHADER_SUBS)
        self.programs["cleanup_runtime"] = build_compute_shader(ctx, "liquid/cleanup_runtime.comp", _SHADER_SUBS)
        self.programs["liquid_flow_intent"] = build_compute_shader(ctx, "liquid/liquid_flow_intent.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "liquid/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_flow_intent_inputs"] = build_compute_shader(ctx, "liquid/load_bridge_flow_intent_inputs.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_out"] = build_compute_shader(ctx, "liquid/load_bridge_cell_out.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux"] = build_compute_shader(ctx, "liquid/load_bridge_cell_aux.comp", _SHADER_SUBS)
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "liquid/publish_bridge_cell.comp", _SHADER_SUBS)

    # --- resources bucket ---
    release = release
    _ensure_resources = _ensure_resources
    _active_scheduler_gpu_authoritative = _active_scheduler_gpu_authoritative
    _refresh_active_scheduler_from_ttl = _refresh_active_scheduler_from_ttl
    _upload_active_tile_mask = _upload_active_tile_mask
    _load_authoritative_active_tile_mask = _load_authoritative_active_tile_mask
    _active_tile_workgroups_per_tile = _active_tile_workgroups_per_tile
    _next_placeholder_claim_epoch = _next_placeholder_claim_epoch
    _seam_workgroups_per_boundary = _seam_workgroups_per_boundary
    _reload_and_compact_active_cell_tiles = _reload_and_compact_active_cell_tiles
    _run_active_tile_indirect = _run_active_tile_indirect

    # --- solve bucket ---
    _build_seam_boundary_dispatch = _build_seam_boundary_dispatch
    _prefetch_seam_boundary_bridge_inputs = _prefetch_seam_boundary_bridge_inputs
    _build_placeholder_dirty_affected_tile_dispatch = _build_placeholder_dirty_affected_tile_dispatch
    _compact_active_tiles = _compact_active_tiles
    _run_tile_solve = _run_tile_solve
    _run_seam_pass = _run_seam_pass
    _run_buoyancy_pass = _run_buoyancy_pass
    _run_copy_core_state = _run_copy_core_state
    _run_copy_for_placeholder = _run_copy_for_placeholder
    _run_placeholder_displacement = _run_placeholder_displacement
    _run_cleanup_runtime = _run_cleanup_runtime

    # --- bridge bucket ---
    step = step
    prepare_motion_flow_intent = prepare_motion_flow_intent
    _upload_inputs = _upload_inputs
    _load_authoritative_bridge_inputs = _load_authoritative_bridge_inputs
    _load_authoritative_bridge_flow_intent_inputs = _load_authoritative_bridge_flow_intent_inputs
    _publish_bridge_outputs = _publish_bridge_outputs
    _run_liquid_intent_pass = _run_liquid_intent_pass
    _download_outputs = _download_outputs
    _barrier_bits = _barrier_bits
