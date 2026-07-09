from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.gpu import ISLAND_RUNTIME_DTYPE
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.sim.gpu_collapse_dirty import (
    clear_collapse_structure_dirty_tile_queue_on_gpu,
    ensure_collapse_structure_dirty_tile_queue,
    get_collapse_structure_dirty_tile_bounds,
)
from oracle_game.types import CollapseBehavior, Phase


LOCAL_SIZE = 8
FORMAL_CONNECTED_TILE_LOCAL_SIZE = 32

# Substitution table for every ``{{NAME}}`` marker used by the external
# ``shaders/collapse/*.comp`` files.  Compound f-string interpolations (e.g.
# ``LOCAL_SIZE - 1``) were baked in as literal values during extraction because
# the loader only substitutes bare-identifier markers; the underlying constants
# are frozen module values, so this is behaviour-identical to the old f-strings.
_SHADER_SUBS: dict[str, Any] = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "FORMAL_CONNECTED_TILE_LOCAL_SIZE": FORMAL_CONNECTED_TILE_LOCAL_SIZE,
}
FORMAL_DEFERRED_REGION_REQUEST_CAPACITY = 256
FORMAL_DEFERRED_REGION_REQUEST_COUNT_BUFFER = "collapse_deferred_region_request_count"
FORMAL_DEFERRED_REGION_REQUEST_BUFFER = "collapse_deferred_region_requests"
FORMAL_CONNECTED_FRONTIER_BUFFER = "collapse_connected_frontier_mask"
FORMAL_CONNECTED_FRONTIER_SCRATCH_BUFFER = "collapse_connected_frontier_scratch_mask"
FORMAL_CONNECTED_PROCESSED_BUFFER = "collapse_connected_processed_mask"
FORMAL_CONNECTED_TILE_SEED_BUFFER = "collapse_connected_tile_seed_mask"
FORMAL_CONNECTED_TILE_FRONTIER_BUFFER = "collapse_connected_tile_frontier_mask"
FORMAL_CONNECTED_TILE_SCRATCH_BUFFER = "collapse_connected_tile_scratch_mask"
FORMAL_CONNECTED_TILE_LIST_BUFFER = "collapse_connected_tile_list"
FORMAL_CONNECTED_TILE_COUNT_BUFFER = "collapse_connected_tile_count"
FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_dispatch_args"
FORMAL_CONNECTED_TILE_FRONTIER_LIST_BUFFER = "collapse_connected_tile_frontier_list"
FORMAL_CONNECTED_TILE_FRONTIER_COUNT_BUFFER = "collapse_connected_tile_frontier_count"
FORMAL_CONNECTED_TILE_FRONTIER_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_frontier_dispatch_args"
FORMAL_CONNECTED_TILE_SCRATCH_LIST_BUFFER = "collapse_connected_tile_scratch_list"
FORMAL_CONNECTED_TILE_SCRATCH_COUNT_BUFFER = "collapse_connected_tile_scratch_count"
FORMAL_CONNECTED_TILE_SCRATCH_DISPATCH_ARGS_BUFFER = "collapse_connected_tile_scratch_dispatch_args"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_LIST_BUFFER = "collapse_connected_cell_frontier_tile_list"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_LIST_BUFFER = "collapse_connected_cell_frontier_tile_scratch_list"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_FLAGS_BUFFER = "collapse_connected_cell_frontier_tile_flags"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_FLAGS_BUFFER = "collapse_connected_cell_frontier_tile_scratch_flags"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_COUNT_BUFFER = "collapse_connected_cell_frontier_tile_count"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_COUNT_BUFFER = "collapse_connected_cell_frontier_tile_scratch_count"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_DISPATCH_ARGS_BUFFER = "collapse_connected_cell_frontier_tile_dispatch_args"
FORMAL_CONNECTED_CELL_FRONTIER_TILE_SCRATCH_DISPATCH_ARGS_BUFFER = (
    "collapse_connected_cell_frontier_tile_scratch_dispatch_args"
)
FORMAL_CONNECTED_TILE_REFINE_PASS_COUNT = 2
FORMAL_CONNECTED_DIRTY_JUMP_ROUNDS = 4


@dataclass(slots=True)
class GPUCollapseResources:
    signature: tuple[int, int]
    structural_tex: Any
    support_ping: Any
    support_pong: Any
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
    temp_tex: Any
    temp_out_tex: Any
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    change_flag: Any
    component_labels: Any
    component_island_ids: Any
    component_metadata: Any
    component_flags: Any
    component_count: Any
    component_dispatch_args: Any
    region_flags: Any
    connected_tile_row_masks: Any
    connected_tile_column_masks: Any
    material_structural: Any
    material_support_anchor: Any
    material_collapse_behavior: Any
    material_collapse_generation: Any
    material_base_integrity: Any
    material_spawn_temperature: Any


from oracle_game.sim.gpu_collapse_resources import (
    _ensure_resources,
    _write_dynamic_buffer,
    _materialize_material_params,
    _classification_material_params
)
from oracle_game.sim.gpu_collapse_publish import (
    _upload_region_state,
    _load_authoritative_bridge_region_inputs,
    _load_authoritative_bridge_connected_tile_inputs,
    _load_authoritative_bridge_pending_region,
    _load_authoritative_bridge_connected_tile_pending,
    _publish_bridge_pending_region_outputs,
    _publish_bridge_pending_region_outputs_from_texture,
    _publish_bridge_region_mask,
    _publish_bridge_supported_unsupported_masks_connected_tiles,
    _publish_bridge_region_labels,
    _publish_bridge_region_labels_connected_tiles,
    _publish_bridge_region_outputs,
    _publish_bridge_region_outputs_connected_tiles,
    _barrier_bits,
    _download_region_state
)
from oracle_game.sim.gpu_collapse_formal import (
    prewarm_formal_connected_resources,
    _formal_jfa_jumps,
    _formal_jfa_profile_jump_bands,
    _formal_support_unit_pass_count,
    _formal_label_unit_pass_count,
    _formal_support_refine_round_count,
    _formal_label_refine_round_count,
    _run_formal_support_refine_passes,
    seed_structural_region_texture,
    connected_structural_region_texture,
    copy_mask_texture,
    _copy_mask_texture_connected_tiles,
    _ensure_formal_connected_axis_mask_buffers,
    _build_formal_connected_axis_masks,
    detect_connected_internal_boundary_flags,
    _ensure_formal_deferred_region_request_buffers,
    _ensure_formal_connected_frontier_buffers,
    _ensure_formal_connected_frontier_buffers_impl,
    _seed_formal_texture_region_tile_worklist,
    _clear_formal_connected_cell_buffer_names,
    _clear_formal_connected_tile_mask_buffers,
    _clear_formal_connected_tile_worklist,
    _clear_formal_connected_tile_worklists,
    _clear_formal_connected_cell_buffer_connected_tiles,
    reset_formal_connected_frontier,
    clear_formal_connected_frontier_buffer,
    clear_formal_deferred_region_requests,
    execute_formal_connected_expansion,
    execute_formal_connected_dirty_tile_queue
)
from oracle_game.sim.gpu_collapse_formal_solve import (
    solve_formal_connected_region_textures,
    _solve_formal_connected_dirty_tile_textures,
    _solve_formal_connected_tile_textures,
    _prepare_formal_connected_tile_resources,
    _prepare_formal_connected_tile_resources_without_input_upload,
    _prepare_formal_connected_tile_resources_impl,
    _clamp_formal_connected_region,
    _formal_connected_dirty_tile_queue_resource_region,
    _formal_connected_dirty_tile_resource_region_from_tile_bounds,
    _formal_connected_resource_region_from_bbox,
    _local_formal_connected_rect,
    _classify_formal_connected_tile_textures,
    _solve_formal_connected_frontier_texture,
    _solve_formal_connected_dirty_cell_frontier_texture,
    _formal_connected_expansion_pass_count,
    _formal_connected_tile_jump_schedule,
    _formal_connected_tile_refine_pass_count,
    _formal_connected_tile_support_frontier_pass_count,
    _formal_connected_component_label_frontier_pass_count,
    _formal_connected_dirty_tile_jump_schedule,
    _formal_connected_dirty_jump_schedule,
    _formal_connected_cell_jump_schedule,
    _next_formal_connected_cell_frontier_generation,
    _filter_formal_connected_eligibility,
    connected_structural_frontier_texture,
    drain_formal_deferred_region_requests,
    enqueue_connected_internal_boundary_deferred_regions,
    exclude_internal_boundary_connected_texture_to_frontier,
    exclude_internal_boundary_connected_texture
)
from oracle_game.sim.gpu_collapse_frontier import (
    _solve_formal_connected_tile_frontier,
    _solve_formal_connected_dirty_tile_frontier,
    _compact_formal_connected_tile_mask,
    _seed_formal_connected_tile_frontier,
    _seed_formal_connected_tile_frontier_from_dirty_queue,
    _expand_formal_connected_tile_frontier,
    _clear_formal_connected_cell_frontier_tiles,
    _accumulate_formal_connected_cell_frontier_tiles,
    _seed_formal_connected_cell_frontier,
    _seed_formal_connected_cell_frontier_from_dirty_queue,
    _expand_formal_connected_cell_frontier,
    _copy_formal_connected_buffer_to_texture,
    _solve_formal_connected_tile_support_textures,
    _seed_formal_connected_tile_support_frontier,
    _expand_formal_connected_tile_support_frontier,
    _run_formal_connected_tile_support_pass,
    _run_formal_connected_tile_support_refine_passes
)
from oracle_game.sim.gpu_collapse_labeling import (
    label_component_mask,
    materialize_component_mask,
    materialize_component_texture,
    materialize_component_texture_formal,
    _reserve_formal_component_island_ids,
    _ensure_component_work_buffers,
    _collect_component_labels_gpu,
    _clear_component_label_flags_connected_tiles,
    _build_component_dispatch_args,
    _prepare_formal_component_list_and_metadata,
    _summarize_formal_component_metadata,
    _materialize_compact_labeled_component_texture,
    _publish_compact_component_island_runtime,
    _materialize_dense_labeled_component_texture,
    _summarize_dense_component_metadata,
    _publish_dense_component_island_runtime,
    _label_component_mask_texture,
    collect_component_labels,
    summarize_labeled_components,
    summarize_labeled_component_texture,
    materialize_labeled_components
)
from oracle_game.sim.gpu_collapse_labeling_formal import (
    _label_component_texture,
    _label_component_texture_connected_tiles_from_texture_init,
    _label_component_texture_connected_tiles,
    _seed_formal_component_label_frontier,
    _expand_formal_component_label_frontier,
    _run_formal_connected_component_label_pass,
    _run_formal_connected_component_label_refine_passes,
    _copy_formal_component_label_buffer_to_texture,
    _run_formal_label_refine_passes,
    materialize_labeled_component_texture
)
from oracle_game.sim.gpu_collapse_stages import (
    solve_region,
    solve_region_textures,
    classify_world_structural_mask,
    expand_region_to_component_bbox,
    _expand_formal_region_to_component_bbox,
    classify_region,
    classify_region_textures,
    resolve_unsupported_outcomes,
    resolve_unsupported_outcome_textures,
    resolve_supported_outcome_textures,
    _run_pass,
    release
)


class GPUCollapsePipeline(GPUPipelineBase):
    FORMAL_EXPAND_LEFT = 1
    FORMAL_EXPAND_TOP = 2
    FORMAL_EXPAND_RIGHT = 4
    FORMAL_EXPAND_BOTTOM = 8

    def __init__(self) -> None:
        self.resources: GPUCollapseResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self._last_formal_connected_tile_mask_name: str | None = None
        self._formal_connected_cell_frontier_generation = 0

    # ``reset_pass_profile`` inherited from GPUPipelineBase.
    # ``_profile_pass`` inherited from GPUPipelineBase.
    # ``available`` inherited from GPUPipelineBase.

    def _ensure_programs(self, ctx: Any | None) -> None:
        if not ctx or self.programs:
            return
        self.programs["classify_cells"] = build_compute_shader(ctx, "collapse/classify_cells.comp", _SHADER_SUBS)
        self.programs["propagate"] = build_compute_shader(ctx, "collapse/propagate.comp", _SHADER_SUBS)
        self.programs["component_label_init"] = build_compute_shader(ctx, "collapse/component_label_init.comp", _SHADER_SUBS)
        self.programs["seed_structural_region"] = build_compute_shader(ctx, "collapse/seed_structural_region.comp", _SHADER_SUBS)
        self.programs["copy_mask_texture"] = build_compute_shader(ctx, "collapse/copy_mask_texture.comp", _SHADER_SUBS)
        self.programs["copy_mask_texture_connected_tiles"] = build_compute_shader(ctx, "collapse/copy_mask_texture_connected_tiles.comp", _SHADER_SUBS)
        self.programs["filter_formal_connected_eligibility"] = build_compute_shader(ctx, "collapse/filter_formal_connected_eligibility.comp", _SHADER_SUBS)
        self.programs["filter_formal_connected_eligibility_connected_tiles"] = build_compute_shader(ctx, "collapse/filter_formal_connected_eligibility_connected_tiles.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_frontier_rect"] = build_compute_shader(ctx, "collapse/seed_formal_connected_frontier_rect.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_tile_frontier"] = build_compute_shader(ctx, "collapse/seed_formal_connected_tile_frontier.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_tile_frontier_from_dirty_queue"] = build_compute_shader(ctx, "collapse/seed_formal_connected_tile_frontier_from_dirty_queue.comp", _SHADER_SUBS)
        self.programs["expand_formal_connected_tiles"] = build_compute_shader(ctx, "collapse/expand_formal_connected_tiles.comp", _SHADER_SUBS)
        self.programs["classify_formal_connected_tiles"] = build_compute_shader(ctx, "collapse/classify_formal_connected_tiles.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_tile_worklist"] = build_compute_shader(ctx, "collapse/clear_formal_connected_tile_worklist.comp", _SHADER_SUBS)
        self.programs["compact_formal_connected_tile_mask"] = build_compute_shader(ctx, "collapse/compact_formal_connected_tile_mask.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_cell_buffer"] = build_compute_shader(ctx, "collapse/clear_formal_connected_cell_buffer.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_tile_mask_buffer"] = build_compute_shader(ctx, "collapse/clear_formal_connected_tile_mask_buffer.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_tile_masks_by_list"] = build_compute_shader(ctx, "collapse/clear_formal_connected_tile_masks_by_list.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_cell_buffer_connected_tiles"] = build_compute_shader(ctx, "collapse/clear_formal_connected_cell_buffer_connected_tiles.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_cell_frontier_tiles"] = build_compute_shader(ctx, "collapse/clear_formal_connected_cell_frontier_tiles.comp", _SHADER_SUBS)
        self.programs["clear_formal_connected_cell_frontier_tile_flags_by_list"] = build_compute_shader(ctx, "collapse/clear_formal_connected_cell_frontier_tile_flags_by_list.comp", _SHADER_SUBS)
        self.programs["reset_formal_connected_cell_frontier_tiles"] = build_compute_shader(ctx, "collapse/reset_formal_connected_cell_frontier_tiles.comp", _SHADER_SUBS)
        self.programs["accumulate_formal_connected_cell_frontier_tiles"] = build_compute_shader(ctx, "collapse/accumulate_formal_connected_cell_frontier_tiles.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_tile_support_frontier"] = build_compute_shader(ctx, "collapse/seed_formal_connected_tile_support_frontier.comp", _SHADER_SUBS)
        self.programs["expand_formal_connected_tile_support_frontier"] = build_compute_shader(ctx, "collapse/expand_formal_connected_tile_support_frontier.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_cell_frontier"] = build_compute_shader(ctx, "collapse/seed_formal_connected_cell_frontier.comp", _SHADER_SUBS)
        self.programs["seed_formal_connected_cell_frontier_from_dirty_queue"] = build_compute_shader(ctx, "collapse/seed_formal_connected_cell_frontier_from_dirty_queue.comp", _SHADER_SUBS)
        self.programs["expand_formal_connected_cells_by_tile"] = build_compute_shader(ctx, "collapse/expand_formal_connected_cells_by_tile.comp", _SHADER_SUBS)
        self.programs["copy_formal_connected_buffer_to_texture"] = build_compute_shader(ctx, "collapse/copy_formal_connected_buffer_to_texture.comp", _SHADER_SUBS)
        self.programs["seed_structural_frontier_region"] = build_compute_shader(ctx, "collapse/seed_structural_frontier_region.comp", _SHADER_SUBS)
        self.programs["publish_internal_boundary_frontier"] = build_compute_shader(ctx, "collapse/publish_internal_boundary_frontier.comp", _SHADER_SUBS)
        self.programs["detect_connected_internal_boundary_flags"] = build_compute_shader(ctx, "collapse/detect_connected_internal_boundary_flags.comp", _SHADER_SUBS)
        self.programs["enqueue_connected_internal_boundary_deferred_regions"] = build_compute_shader(ctx, "collapse/enqueue_connected_internal_boundary_deferred_regions.comp", _SHADER_SUBS)
        self.programs["seed_internal_boundary_region"] = build_compute_shader(ctx, "collapse/seed_internal_boundary_region.comp", _SHADER_SUBS)
        self.programs["exclude_boundary_connected_mask"] = build_compute_shader(ctx, "collapse/exclude_boundary_connected_mask.comp", _SHADER_SUBS)
        self.programs["build_formal_connected_axis_masks"] = build_compute_shader(ctx, "collapse/build_formal_connected_axis_masks.comp", _SHADER_SUBS)
        self.programs["propagate_formal_connected_tiles"] = build_compute_shader(ctx, "collapse/propagate_formal_connected_tiles.comp", _SHADER_SUBS)
        self.programs["propagate_formal_connected_component_labels"] = build_compute_shader(ctx, "collapse/propagate_formal_connected_component_labels.comp", _SHADER_SUBS)
        self.programs["component_label_propagate"] = build_compute_shader(ctx, "collapse/component_label_propagate.comp", _SHADER_SUBS)
        self.programs["seed_formal_component_label_frontier"] = build_compute_shader(ctx, "collapse/seed_formal_component_label_frontier.comp", _SHADER_SUBS)
        self.programs["expand_formal_component_label_frontier"] = build_compute_shader(ctx, "collapse/expand_formal_component_label_frontier.comp", _SHADER_SUBS)
        self.programs["copy_formal_component_label_buffer_to_texture"] = build_compute_shader(ctx, "collapse/copy_formal_component_label_buffer_to_texture.comp", _SHADER_SUBS)
        self.programs["collect_component_labels"] = build_compute_shader(ctx, "collapse/collect_component_labels.comp", _SHADER_SUBS)
        self.programs["clear_component_label_flags_connected_tiles"] = build_compute_shader(ctx, "collapse/clear_component_label_flags_connected_tiles.comp", _SHADER_SUBS)
        self.programs["collect_component_labels_connected_tiles"] = build_compute_shader(ctx, "collapse/collect_component_labels_connected_tiles.comp", _SHADER_SUBS)
        self.programs["build_component_dispatch_args"] = build_compute_shader(ctx, "collapse/build_component_dispatch_args.comp", _SHADER_SUBS)
        self.programs["index_compact_component_labels"] = build_compute_shader(ctx, "collapse/index_compact_component_labels.comp", _SHADER_SUBS)
        self.programs["summarize_compact_components"] = build_compute_shader(ctx, "collapse/summarize_compact_components.comp", _SHADER_SUBS)
        self.programs["summarize_compact_components_connected_tiles"] = build_compute_shader(ctx, "collapse/summarize_compact_components_connected_tiles.comp", _SHADER_SUBS)
        self.programs["summarize_components"] = build_compute_shader(ctx, "collapse/summarize_components.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes"] = build_compute_shader(ctx, "collapse/resolve_outcomes.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes_from_supported"] = build_compute_shader(ctx, "collapse/resolve_outcomes_from_supported.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes_from_supported_connected_tiles"] = build_compute_shader(ctx, "collapse/resolve_outcomes_from_supported_connected_tiles.comp", _SHADER_SUBS)
        self.programs["materialize_components"] = build_compute_shader(ctx, "collapse/materialize_components.comp", _SHADER_SUBS)
        self.programs["materialize_components_aux"] = build_compute_shader(ctx, "collapse/materialize_components_aux.comp", _SHADER_SUBS)
        self.programs["materialize_compact_components"] = build_compute_shader(ctx, "collapse/materialize_compact_components.comp", _SHADER_SUBS)
        self.programs["materialize_compact_components_aux"] = build_compute_shader(ctx, "collapse/materialize_compact_components_aux.comp", _SHADER_SUBS)
        self.programs["materialize_compact_components_connected_tiles"] = build_compute_shader(ctx, "collapse/materialize_compact_components_connected_tiles.comp", _SHADER_SUBS)
        self.programs["materialize_compact_components_aux_connected_tiles"] = build_compute_shader(ctx, "collapse/materialize_compact_components_aux_connected_tiles.comp", _SHADER_SUBS)
        self.programs["materialize_dense_components"] = build_compute_shader(ctx, "collapse/materialize_dense_components.comp", _SHADER_SUBS)
        self.programs["materialize_dense_components_aux"] = build_compute_shader(ctx, "collapse/materialize_dense_components_aux.comp", _SHADER_SUBS)
        self.programs["init_dense_component_metadata"] = build_compute_shader(ctx, "collapse/init_dense_component_metadata.comp", _SHADER_SUBS)
        self.programs["summarize_dense_components"] = build_compute_shader(ctx, "collapse/summarize_dense_components.comp", _SHADER_SUBS)
        self.programs["publish_dense_component_island_runtime"] = build_compute_shader(ctx, "collapse/publish_dense_component_island_runtime.comp", _SHADER_SUBS)
        self.programs["publish_compact_component_island_runtime"] = build_compute_shader(ctx, "collapse/publish_compact_component_island_runtime.comp", _SHADER_SUBS)
        self.programs["load_bridge_region_cell"] = build_compute_shader(ctx, "collapse/load_bridge_region_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_connected_tile_cell"] = build_compute_shader(ctx, "collapse/load_bridge_connected_tile_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_region_cell_aux"] = build_compute_shader(ctx, "collapse/load_bridge_region_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_connected_tile_cell_aux"] = build_compute_shader(ctx, "collapse/load_bridge_connected_tile_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_connected_tile_pending"] = build_compute_shader(ctx, "collapse/load_bridge_connected_tile_pending.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_cell"] = build_compute_shader(ctx, "collapse/publish_bridge_region_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_cell_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_cell_connected_tiles.comp", _SHADER_SUBS)
        self.programs["load_bridge_region_pending"] = build_compute_shader(ctx, "collapse/load_bridge_region_pending.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_pending"] = build_compute_shader(ctx, "collapse/publish_bridge_region_pending.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_pending_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_pending_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_mask"] = build_compute_shader(ctx, "collapse/publish_bridge_region_mask.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_mask_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_mask_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_supported_unsupported_masks_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_supported_unsupported_masks_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_labels"] = build_compute_shader(ctx, "collapse/publish_bridge_region_labels.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_labels_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_labels_connected_tiles.comp", _SHADER_SUBS)

    _ensure_resources = _ensure_resources
    _write_dynamic_buffer = _write_dynamic_buffer
    _materialize_material_params = _materialize_material_params
    _classification_material_params = _classification_material_params

    _upload_region_state = _upload_region_state
    _load_authoritative_bridge_region_inputs = _load_authoritative_bridge_region_inputs
    _load_authoritative_bridge_connected_tile_inputs = _load_authoritative_bridge_connected_tile_inputs
    _load_authoritative_bridge_pending_region = _load_authoritative_bridge_pending_region
    _load_authoritative_bridge_connected_tile_pending = _load_authoritative_bridge_connected_tile_pending
    _publish_bridge_pending_region_outputs = _publish_bridge_pending_region_outputs
    _publish_bridge_pending_region_outputs_from_texture = _publish_bridge_pending_region_outputs_from_texture
    _publish_bridge_region_mask = _publish_bridge_region_mask
    _publish_bridge_supported_unsupported_masks_connected_tiles = _publish_bridge_supported_unsupported_masks_connected_tiles
    _publish_bridge_region_labels = _publish_bridge_region_labels
    _publish_bridge_region_labels_connected_tiles = _publish_bridge_region_labels_connected_tiles
    _publish_bridge_region_outputs = _publish_bridge_region_outputs
    _publish_bridge_region_outputs_connected_tiles = _publish_bridge_region_outputs_connected_tiles
    _barrier_bits = _barrier_bits
    _download_region_state = _download_region_state

    prewarm_formal_connected_resources = prewarm_formal_connected_resources
    _formal_jfa_jumps = staticmethod(_formal_jfa_jumps)
    _formal_jfa_profile_jump_bands = staticmethod(_formal_jfa_profile_jump_bands)
    _formal_support_unit_pass_count = staticmethod(_formal_support_unit_pass_count)
    _formal_label_unit_pass_count = staticmethod(_formal_label_unit_pass_count)
    _formal_support_refine_round_count = staticmethod(_formal_support_refine_round_count)
    _formal_label_refine_round_count = staticmethod(_formal_label_refine_round_count)
    _run_formal_support_refine_passes = _run_formal_support_refine_passes
    seed_structural_region_texture = seed_structural_region_texture
    connected_structural_region_texture = connected_structural_region_texture
    copy_mask_texture = copy_mask_texture
    _copy_mask_texture_connected_tiles = _copy_mask_texture_connected_tiles
    _ensure_formal_connected_axis_mask_buffers = _ensure_formal_connected_axis_mask_buffers
    _build_formal_connected_axis_masks = _build_formal_connected_axis_masks
    detect_connected_internal_boundary_flags = detect_connected_internal_boundary_flags
    _ensure_formal_deferred_region_request_buffers = _ensure_formal_deferred_region_request_buffers
    _ensure_formal_connected_frontier_buffers = _ensure_formal_connected_frontier_buffers
    _ensure_formal_connected_frontier_buffers_impl = _ensure_formal_connected_frontier_buffers_impl
    _seed_formal_texture_region_tile_worklist = _seed_formal_texture_region_tile_worklist
    _clear_formal_connected_cell_buffer_names = _clear_formal_connected_cell_buffer_names
    _clear_formal_connected_tile_mask_buffers = _clear_formal_connected_tile_mask_buffers
    _clear_formal_connected_tile_worklist = _clear_formal_connected_tile_worklist
    _clear_formal_connected_tile_worklists = _clear_formal_connected_tile_worklists
    _clear_formal_connected_cell_buffer_connected_tiles = _clear_formal_connected_cell_buffer_connected_tiles
    reset_formal_connected_frontier = reset_formal_connected_frontier
    clear_formal_connected_frontier_buffer = clear_formal_connected_frontier_buffer
    clear_formal_deferred_region_requests = clear_formal_deferred_region_requests
    execute_formal_connected_expansion = execute_formal_connected_expansion
    execute_formal_connected_dirty_tile_queue = execute_formal_connected_dirty_tile_queue

    solve_formal_connected_region_textures = solve_formal_connected_region_textures
    _solve_formal_connected_dirty_tile_textures = _solve_formal_connected_dirty_tile_textures
    _solve_formal_connected_tile_textures = _solve_formal_connected_tile_textures
    _prepare_formal_connected_tile_resources = _prepare_formal_connected_tile_resources
    _prepare_formal_connected_tile_resources_without_input_upload = _prepare_formal_connected_tile_resources_without_input_upload
    _prepare_formal_connected_tile_resources_impl = _prepare_formal_connected_tile_resources_impl
    _clamp_formal_connected_region = _clamp_formal_connected_region
    _formal_connected_dirty_tile_queue_resource_region = _formal_connected_dirty_tile_queue_resource_region
    _formal_connected_dirty_tile_resource_region_from_tile_bounds = _formal_connected_dirty_tile_resource_region_from_tile_bounds
    _formal_connected_resource_region_from_bbox = _formal_connected_resource_region_from_bbox
    _local_formal_connected_rect = staticmethod(_local_formal_connected_rect)
    _classify_formal_connected_tile_textures = _classify_formal_connected_tile_textures
    _solve_formal_connected_frontier_texture = _solve_formal_connected_frontier_texture
    _solve_formal_connected_dirty_cell_frontier_texture = _solve_formal_connected_dirty_cell_frontier_texture
    _formal_connected_expansion_pass_count = _formal_connected_expansion_pass_count
    _formal_connected_tile_jump_schedule = _formal_connected_tile_jump_schedule
    _formal_connected_tile_refine_pass_count = _formal_connected_tile_refine_pass_count
    _formal_connected_tile_support_frontier_pass_count = _formal_connected_tile_support_frontier_pass_count
    _formal_connected_component_label_frontier_pass_count = _formal_connected_component_label_frontier_pass_count
    _formal_connected_dirty_tile_jump_schedule = _formal_connected_dirty_tile_jump_schedule
    _formal_connected_dirty_jump_schedule = staticmethod(_formal_connected_dirty_jump_schedule)
    _formal_connected_cell_jump_schedule = staticmethod(_formal_connected_cell_jump_schedule)
    _next_formal_connected_cell_frontier_generation = _next_formal_connected_cell_frontier_generation
    _filter_formal_connected_eligibility = _filter_formal_connected_eligibility
    connected_structural_frontier_texture = connected_structural_frontier_texture
    drain_formal_deferred_region_requests = drain_formal_deferred_region_requests
    enqueue_connected_internal_boundary_deferred_regions = enqueue_connected_internal_boundary_deferred_regions
    exclude_internal_boundary_connected_texture_to_frontier = exclude_internal_boundary_connected_texture_to_frontier
    exclude_internal_boundary_connected_texture = exclude_internal_boundary_connected_texture

    _solve_formal_connected_tile_frontier = _solve_formal_connected_tile_frontier
    _solve_formal_connected_dirty_tile_frontier = _solve_formal_connected_dirty_tile_frontier
    _compact_formal_connected_tile_mask = _compact_formal_connected_tile_mask
    _seed_formal_connected_tile_frontier = _seed_formal_connected_tile_frontier
    _seed_formal_connected_tile_frontier_from_dirty_queue = _seed_formal_connected_tile_frontier_from_dirty_queue
    _expand_formal_connected_tile_frontier = _expand_formal_connected_tile_frontier
    _clear_formal_connected_cell_frontier_tiles = _clear_formal_connected_cell_frontier_tiles
    _accumulate_formal_connected_cell_frontier_tiles = _accumulate_formal_connected_cell_frontier_tiles
    _seed_formal_connected_cell_frontier = _seed_formal_connected_cell_frontier
    _seed_formal_connected_cell_frontier_from_dirty_queue = _seed_formal_connected_cell_frontier_from_dirty_queue
    _expand_formal_connected_cell_frontier = _expand_formal_connected_cell_frontier
    _copy_formal_connected_buffer_to_texture = _copy_formal_connected_buffer_to_texture
    _solve_formal_connected_tile_support_textures = _solve_formal_connected_tile_support_textures
    _seed_formal_connected_tile_support_frontier = _seed_formal_connected_tile_support_frontier
    _expand_formal_connected_tile_support_frontier = _expand_formal_connected_tile_support_frontier
    _run_formal_connected_tile_support_pass = _run_formal_connected_tile_support_pass
    _run_formal_connected_tile_support_refine_passes = _run_formal_connected_tile_support_refine_passes

    label_component_mask = label_component_mask
    materialize_component_mask = materialize_component_mask
    materialize_component_texture = materialize_component_texture
    materialize_component_texture_formal = materialize_component_texture_formal
    _reserve_formal_component_island_ids = _reserve_formal_component_island_ids
    _ensure_component_work_buffers = _ensure_component_work_buffers
    _collect_component_labels_gpu = _collect_component_labels_gpu
    _clear_component_label_flags_connected_tiles = _clear_component_label_flags_connected_tiles
    _build_component_dispatch_args = _build_component_dispatch_args
    _prepare_formal_component_list_and_metadata = _prepare_formal_component_list_and_metadata
    _summarize_formal_component_metadata = _summarize_formal_component_metadata
    _materialize_compact_labeled_component_texture = _materialize_compact_labeled_component_texture
    _publish_compact_component_island_runtime = _publish_compact_component_island_runtime
    _materialize_dense_labeled_component_texture = _materialize_dense_labeled_component_texture
    _summarize_dense_component_metadata = _summarize_dense_component_metadata
    _publish_dense_component_island_runtime = _publish_dense_component_island_runtime
    _label_component_mask_texture = _label_component_mask_texture
    collect_component_labels = collect_component_labels
    summarize_labeled_components = summarize_labeled_components
    summarize_labeled_component_texture = summarize_labeled_component_texture
    materialize_labeled_components = materialize_labeled_components

    _label_component_texture = _label_component_texture
    _label_component_texture_connected_tiles_from_texture_init = _label_component_texture_connected_tiles_from_texture_init
    _label_component_texture_connected_tiles = _label_component_texture_connected_tiles
    _seed_formal_component_label_frontier = _seed_formal_component_label_frontier
    _expand_formal_component_label_frontier = _expand_formal_component_label_frontier
    _run_formal_connected_component_label_pass = _run_formal_connected_component_label_pass
    _run_formal_connected_component_label_refine_passes = _run_formal_connected_component_label_refine_passes
    _copy_formal_component_label_buffer_to_texture = _copy_formal_component_label_buffer_to_texture
    _run_formal_label_refine_passes = _run_formal_label_refine_passes
    materialize_labeled_component_texture = materialize_labeled_component_texture

    solve_region = solve_region
    solve_region_textures = solve_region_textures
    classify_world_structural_mask = classify_world_structural_mask
    expand_region_to_component_bbox = expand_region_to_component_bbox
    _expand_formal_region_to_component_bbox = _expand_formal_region_to_component_bbox
    classify_region = classify_region
    classify_region_textures = classify_region_textures
    resolve_unsupported_outcomes = resolve_unsupported_outcomes
    resolve_unsupported_outcome_textures = resolve_unsupported_outcome_textures
    resolve_supported_outcome_textures = resolve_supported_outcome_textures
    _run_pass = _run_pass
    release = release

