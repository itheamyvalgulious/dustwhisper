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
    "PUBLISH_RUNTIME_MASKS": 0,
    "PUBLISH_CLASSIFICATION_MASKS": 0,
    "DIRECT_BRIDGE_INPUTS": 0,
    "DIRECT_BEHAVIOR_INPUTS": 0,
    "SNAPSHOT_PENDING": 0,
    "PUBLISH_COMPONENT_LABELS": 0,
    "PUBLISH_IMMUNE_DIRECT": 0,
    "PUBLISH_DELAYED_DIRECT": 0,
    "PACKED_INCREMENTAL_SNAPSHOT": 0,
    "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 0,
    "SUPPORT_JFA_U8": 0,
    "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 0,
    "SUPPORT_TEXTURE_U8": 0,
    "SUMMARIZE_COMPONENT_METADATA": 0,
    "WRITE_FILTERED_COMPONENT_LABELS": 0,
    "INITIALIZE_LABEL_TILE_UNION": 0,
    "INVALID_COMPONENT_GENERATION_VALIDITY": 0,
    "COMPONENT_FLAG_GENERATION_VALIDITY": 0,
    "SUPPORT_JFA_NV32_ROW_HYDRATE": 0,
    "SUPPORT_JFA_EXTENSIONS": "",
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
    support_u8_ping: Any | None
    support_u8_pong: Any | None
    material_tex: Any
    material_out_tex: Any
    phase_tex: Any
    phase_out_tex: Any
    pending_tex: Any
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
    component_invalid: Any
    component_count: Any
    component_dispatch_args: Any
    region_flags: Any
    support_tile_union_roots: Any | None
    support_tile_union_parent: Any | None
    support_tile_union_seeded: Any | None
    support_tile_union_edges: Any | None
    support_tile_union_edge_count: Any | None
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
    _ensure_formal_connected_u8_support_textures,
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
    _invalidate_persistent_dense_tile_worklist,
    _persistent_dense_tile_worklist_signature,
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
    _begin_formal_connected_tile_support,
    _run_formal_connected_tile_support_slice,
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
    _begin_formal_connected_component_labeling,
    _run_formal_connected_component_label_slice,
    _publish_formal_connected_component_labels,
    _ensure_formal_connected_component_label_union_buffers,
    _begin_formal_connected_component_label_union,
    _run_formal_connected_component_label_union_slice,
    _materialize_formal_connected_component_label_union,
    _seed_formal_component_labels_and_axis_masks,
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
from oracle_game.sim.gpu_collapse_incremental import (
    advance_formal_connected_dirty_tile_queue,
    advance_formal_runtime_admission,
    has_active_formal_dirty_epoch,
    _validate_and_collect_formal_dirty_epoch_labels,
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
        self._persistent_dense_tile_worklist_enabled = True
        self._persistent_dense_tile_worklist_signature: tuple[int, ...] | None = None
        self.persistent_dense_tile_worklist_hits = 0
        self.persistent_dense_tile_worklist_rebuilds = 0
        self.persistent_dense_tile_worklist_invalidations = 0
        self._support_outcome_publish_fusion_enabled = True
        self._classification_mask_publish_fusion_enabled = True
        self._classification_bridge_hydration_fusion_enabled = True
        self._label_seed_materialize_axis_fusion_enabled = True
        self._support_tile_union_enabled = False
        self._support_tile_union_atomic_union_enabled = False
        self._support_jfa_image_barrier_elision_enabled = False
        # Experimental: write each support tile row-major so a warp stores a
        # contiguous image row instead of issuing a vertical-stride store.
        # Keep the canonical traversal available until frame-level A/B proves
        # a stable win on the target GPU.
        self._support_jfa_row_major_output_enabled = True
        self._incremental_support_jfa_u8_enabled = True
        self._support_jfa_u8_propagated_source_mask_elision_enabled = True
        # A 32-lane NV warp hydrates one structural tile row per lane and
        # ballots support texels into row masks. Other devices keep the
        # canonical scalar row scan.
        self._support_jfa_nv32_row_hydrate_enabled = True
        self._support_jfa_nv32_row_hydrate_supported = False
        self._incremental_classification_support_axis_u8_fusion_enabled = True
        self._outcome_label_tile_union_enabled = True
        self._incremental_collapse_pipeline_enabled = True
        self._incremental_jfa_four_frame_balance_enabled = True
        # Balance the coarse support and terminal label work across the four
        # epoch phases so the worst frame does not carry both peaks.
        self._incremental_phase_peak_v3_balance_enabled = True
        self._incremental_support_outcome_publish_fusion_enabled = False
        self._incremental_direct_immune_publish_enabled = True
        self._incremental_direct_delayed_publish_enabled = True
        self._incremental_packed_cell_snapshot_enabled = False
        self._incremental_materialize_metadata_fusion_enabled = True
        self._incremental_materialize_filter_fusion_enabled = True
        self._incremental_label_union_materialize_validation_fusion_enabled = True
        # Incremental validation normally clears one uint per possible label.
        # A generation token makes prior invalid marks semantically stale.
        self._incremental_component_invalid_generation_enabled = True
        self._component_invalid_generation = 0
        self._incremental_component_flag_generation_enabled = True
        self._active_component_flag_generation = 0
        # Initialize label-union tile roots in the outcome resolve workgroup
        # while preserving the canonical outcome texture for other consumers.
        self._incremental_outcome_label_local_fusion_enabled = True
        self._runtime_admission_stride_dispatch_enabled = True
        self._formal_dirty_epoch: Any | None = None
        self._pending_formal_runtime_admission: Any | None = None
        self.incremental_collapse_epoch_sequence = 0
        self.incremental_collapse_epochs_started = 0
        self.incremental_collapse_epochs_completed = 0
        self.incremental_collapse_epochs_aborted = 0
        self.last_incremental_collapse_phase: int | None = None
        self.last_incremental_collapse_epoch_id: int | None = None
        self.last_incremental_collapse_epoch_started_frame_id: int | None = None
        self.last_incremental_runtime_admission_slot: int | None = None
        self.incremental_collapse_runtime_admissions_started = 0
        self.incremental_collapse_runtime_admissions_completed = 0
        self.incremental_collapse_runtime_admissions_aborted = 0

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
        self.programs["classify_formal_connected_tiles_publish"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles.comp",
            {**_SHADER_SUBS, "PUBLISH_CLASSIFICATION_MASKS": 1},
        )
        self.programs["classify_formal_connected_tiles_bridge"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles.comp",
            {**_SHADER_SUBS, "DIRECT_BRIDGE_INPUTS": 1},
        )
        self.programs["classify_formal_connected_tiles_bridge_publish"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_INPUTS": 1,
                "PUBLISH_CLASSIFICATION_MASKS": 1,
            },
        )
        self.programs["classify_formal_connected_tiles_bridge_publish_incremental"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_INPUTS": 1,
                "PUBLISH_CLASSIFICATION_MASKS": 1,
                "SNAPSHOT_PENDING": 1,
            },
        )
        self.programs["classify_formal_connected_tiles_bridge_publish_incremental_packed"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_INPUTS": 1,
                "PUBLISH_CLASSIFICATION_MASKS": 1,
                "SNAPSHOT_PENDING": 1,
                "PACKED_INCREMENTAL_SNAPSHOT": 1,
            },
        )
        self.programs["classify_formal_connected_tiles_bridge_publish_incremental_axis_u8"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles_support_axis_u8.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_INPUTS": 1,
                "PUBLISH_CLASSIFICATION_MASKS": 1,
                "SNAPSHOT_PENDING": 1,
            },
        )
        self.programs["classify_formal_connected_tiles_bridge_publish_incremental_packed_axis_u8"] = build_compute_shader(
            ctx,
            "collapse/classify_formal_connected_tiles_support_axis_u8.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BRIDGE_INPUTS": 1,
                "PUBLISH_CLASSIFICATION_MASKS": 1,
                "SNAPSHOT_PENDING": 1,
                "PACKED_INCREMENTAL_SNAPSHOT": 1,
            },
        )
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
        self.programs["build_formal_connected_axis_masks_u8"] = build_compute_shader(
            ctx,
            "collapse/build_formal_connected_axis_masks.comp",
            {**_SHADER_SUBS, "SUPPORT_JFA_U8": 1},
        )
        self.programs["propagate_formal_connected_tiles"] = build_compute_shader(
            ctx,
            "collapse/propagate_formal_connected_tile_rows.comp",
            _SHADER_SUBS,
        )
        self.programs["propagate_formal_connected_tiles_row_major"] = build_compute_shader(
            ctx,
            "collapse/propagate_formal_connected_tile_rows.comp",
            {**_SHADER_SUBS, "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1},
        )
        self.programs["propagate_formal_connected_tiles_u8"] = build_compute_shader(
            ctx,
            "collapse/propagate_formal_connected_tile_rows.comp",
            {**_SHADER_SUBS, "SUPPORT_JFA_U8": 1},
        )
        self.programs["propagate_formal_connected_tiles_u8_row_major"] = build_compute_shader(
            ctx,
            "collapse/propagate_formal_connected_tile_rows.comp",
            {
                **_SHADER_SUBS,
                "SUPPORT_JFA_U8": 1,
                "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1,
            },
        )
        self.programs["propagate_formal_connected_tiles_u8_source_mask_elision"] = (
            build_compute_shader(
                ctx,
                "collapse/propagate_formal_connected_tile_rows.comp",
                {
                    **_SHADER_SUBS,
                    "SUPPORT_JFA_U8": 1,
                    "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 1,
                },
            )
        )
        self.programs["propagate_formal_connected_tiles_u8_row_major_source_mask_elision"] = (
            build_compute_shader(
                ctx,
                "collapse/propagate_formal_connected_tile_rows.comp",
                {
                    **_SHADER_SUBS,
                    "SUPPORT_JFA_U8": 1,
                    "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1,
                    "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 1,
                },
            )
        )
        required_support_ballot_extensions = {
            "GL_NV_gpu_shader5",
            "GL_NV_shader_thread_group",
        }
        available_extensions = set(getattr(ctx, "extensions", ()))
        support_warp_size = 0
        if (
            self._support_jfa_nv32_row_hydrate_enabled
            and required_support_ballot_extensions.issubset(available_extensions)
        ):
            warp_size_program = build_compute_shader(
                ctx,
                "heat/query_nv_warp_size.comp",
            )
            warp_size_buffer = ctx.buffer(reserve=np.dtype(np.uint32).itemsize)
            try:
                warp_size_buffer.bind_to_storage_buffer(binding=0)
                warp_size_program.run(1, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
                support_warp_size = int(
                    np.frombuffer(
                        warp_size_buffer.read(),
                        dtype=np.uint32,
                        count=1,
                    )[0]
                )
            finally:
                warp_size_buffer.release()
                warp_size_program.release()
        self._support_jfa_nv32_row_hydrate_supported = support_warp_size == 32
        if self._support_jfa_nv32_row_hydrate_supported:
            self.programs[
                "propagate_formal_connected_tiles_u8_row_major_source_mask_elision_nv32"
            ] = build_compute_shader(
                ctx,
                "collapse/propagate_formal_connected_tile_rows.comp",
                {
                    **_SHADER_SUBS,
                    "SUPPORT_JFA_U8": 1,
                    "SUPPORT_JFA_ROW_MAJOR_OUTPUT": 1,
                    "SUPPORT_JFA_PROPAGATED_SOURCE_MASK_ELISION": 1,
                    "SUPPORT_JFA_NV32_ROW_HYDRATE": 1,
                    "SUPPORT_JFA_EXTENSIONS": "\n".join(
                        (
                            "#extension GL_NV_gpu_shader5 : require",
                            "#extension GL_NV_shader_thread_group : require",
                        )
                    ),
                },
            )
        self.programs["support_tile_union_local"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_local.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_edges"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_edges.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_hook"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_hook.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_shortcut"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_shortcut.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_atomic_hook"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_atomic_hook.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_atomic_shortcut"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_atomic_shortcut.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_seed"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_seed.comp",
            _SHADER_SUBS,
        )
        self.programs["support_tile_union_materialize"] = build_compute_shader(
            ctx,
            "collapse/support_tile_union_materialize.comp",
            _SHADER_SUBS,
        )
        self.programs["label_tile_union_local"] = build_compute_shader(
            ctx,
            "collapse/label_tile_union_local.comp",
            _SHADER_SUBS,
        )
        self.programs["label_tile_union_hook"] = build_compute_shader(
            ctx,
            "collapse/label_tile_union_hook.comp",
            _SHADER_SUBS,
        )
        self.programs["label_tile_union_build_dispatch"] = build_compute_shader(
            ctx,
            "collapse/label_tile_union_build_dispatch.comp",
            _SHADER_SUBS,
        )
        self.programs["label_tile_union_materialize"] = build_compute_shader(
            ctx,
            "collapse/label_tile_union_materialize.comp",
            _SHADER_SUBS,
        )
        self.programs["propagate_formal_connected_component_labels"] = build_compute_shader(ctx, "collapse/propagate_formal_connected_component_labels.comp", _SHADER_SUBS)
        self.programs["component_label_propagate"] = build_compute_shader(ctx, "collapse/component_label_propagate.comp", _SHADER_SUBS)
        self.programs["seed_formal_component_label_frontier"] = build_compute_shader(ctx, "collapse/seed_formal_component_label_frontier.comp", _SHADER_SUBS)
        self.programs["seed_formal_component_labels_and_axis_masks"] = build_compute_shader(
            ctx,
            "collapse/seed_formal_component_labels_and_axis_masks.comp",
            _SHADER_SUBS,
        )
        self.programs["expand_formal_component_label_frontier"] = build_compute_shader(ctx, "collapse/expand_formal_component_label_frontier.comp", _SHADER_SUBS)
        self.programs["copy_formal_component_label_buffer_to_texture"] = build_compute_shader(ctx, "collapse/copy_formal_component_label_buffer_to_texture.comp", _SHADER_SUBS)
        self.programs["collect_component_labels"] = build_compute_shader(ctx, "collapse/collect_component_labels.comp", _SHADER_SUBS)
        self.programs["clear_component_label_flags_connected_tiles"] = build_compute_shader(ctx, "collapse/clear_component_label_flags_connected_tiles.comp", _SHADER_SUBS)
        self.programs["collect_component_labels_connected_tiles"] = build_compute_shader(ctx, "collapse/collect_component_labels_connected_tiles.comp", _SHADER_SUBS)
        self.programs["collect_component_labels_connected_tiles_generation"] = build_compute_shader(
            ctx,
            "collapse/collect_component_labels_connected_tiles.comp",
            {**_SHADER_SUBS, "INVALID_COMPONENT_GENERATION_VALIDITY": 1},
        )
        self.programs["clear_incremental_component_invalid"] = build_compute_shader(
            ctx,
            "collapse/clear_incremental_component_invalid.comp",
            _SHADER_SUBS,
        )
        self.programs["validate_incremental_component_labels"] = build_compute_shader(
            ctx,
            "collapse/validate_incremental_component_labels.comp",
            _SHADER_SUBS,
        )
        self.programs["validate_incremental_component_labels_packed"] = build_compute_shader(
            ctx,
            "collapse/validate_incremental_component_labels.comp",
            {**_SHADER_SUBS, "PACKED_INCREMENTAL_SNAPSHOT": 1},
        )
        self.programs["validate_incremental_component_labels_union_materialize"] = build_compute_shader(
            ctx,
            "collapse/validate_incremental_component_labels_union_materialize.comp",
            _SHADER_SUBS,
        )
        self.programs["validate_incremental_component_labels_union_materialize_generation"] = build_compute_shader(
            ctx,
            "collapse/validate_incremental_component_labels_union_materialize.comp",
            {**_SHADER_SUBS, "INVALID_COMPONENT_GENERATION_VALIDITY": 1},
        )
        self.programs["validate_incremental_component_labels_union_materialize_packed"] = build_compute_shader(
            ctx,
            "collapse/validate_incremental_component_labels_union_materialize.comp",
            {**_SHADER_SUBS, "PACKED_INCREMENTAL_SNAPSHOT": 1},
        )
        self.programs["filter_incremental_component_labels"] = build_compute_shader(
            ctx,
            "collapse/filter_incremental_component_labels.comp",
            _SHADER_SUBS,
        )
        self.programs["materialize_incremental_components_bridge"] = build_compute_shader(
            ctx,
            "collapse/materialize_incremental_components_bridge.comp",
            _SHADER_SUBS,
        )
        self.programs["materialize_incremental_components_bridge_metadata"] = build_compute_shader(
            ctx,
            "collapse/materialize_incremental_components_bridge.comp",
            {**_SHADER_SUBS, "SUMMARIZE_COMPONENT_METADATA": 1},
        )
        self.programs["materialize_incremental_components_bridge_filter"] = build_compute_shader(
            ctx,
            "collapse/materialize_incremental_components_bridge.comp",
            {**_SHADER_SUBS, "WRITE_FILTERED_COMPONENT_LABELS": 1},
        )
        self.programs["materialize_incremental_components_bridge_metadata_filter"] = build_compute_shader(
            ctx,
            "collapse/materialize_incremental_components_bridge.comp",
            {
                **_SHADER_SUBS,
                "SUMMARIZE_COMPONENT_METADATA": 1,
                "WRITE_FILTERED_COMPONENT_LABELS": 1,
            },
        )
        self.programs["materialize_incremental_components_bridge_metadata_filter_generation"] = build_compute_shader(
            ctx,
            "collapse/materialize_incremental_components_bridge.comp",
            {
                **_SHADER_SUBS,
                "SUMMARIZE_COMPONENT_METADATA": 1,
                "WRITE_FILTERED_COMPONENT_LABELS": 1,
                "COMPONENT_FLAG_GENERATION_VALIDITY": 1,
            },
        )
        self.programs["publish_incremental_outcome_masks"] = build_compute_shader(
            ctx,
            "collapse/publish_incremental_outcome_masks.comp",
            _SHADER_SUBS,
        )
        self.programs["build_component_dispatch_args"] = build_compute_shader(ctx, "collapse/build_component_dispatch_args.comp", _SHADER_SUBS)
        self.programs["index_compact_component_labels"] = build_compute_shader(ctx, "collapse/index_compact_component_labels.comp", _SHADER_SUBS)
        self.programs["summarize_compact_components"] = build_compute_shader(ctx, "collapse/summarize_compact_components.comp", _SHADER_SUBS)
        self.programs["summarize_compact_components_connected_tiles"] = build_compute_shader(ctx, "collapse/summarize_compact_components_connected_tiles.comp", _SHADER_SUBS)
        self.programs["summarize_components"] = build_compute_shader(ctx, "collapse/summarize_components.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes"] = build_compute_shader(ctx, "collapse/resolve_outcomes.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes_from_supported"] = build_compute_shader(ctx, "collapse/resolve_outcomes_from_supported.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes_from_supported_connected_tiles"] = build_compute_shader(ctx, "collapse/resolve_outcomes_from_supported_connected_tiles.comp", _SHADER_SUBS)
        self.programs["resolve_outcomes_from_supported_connected_tiles_publish"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {**_SHADER_SUBS, "PUBLISH_RUNTIME_MASKS": 1},
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {**_SHADER_SUBS, "DIRECT_BEHAVIOR_INPUTS": 1},
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_publish"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_RUNTIME_MASKS": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_immune"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {**_SHADER_SUBS, "PUBLISH_IMMUNE_DIRECT": 1},
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_publish_immune"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "PUBLISH_RUNTIME_MASKS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_immune"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_publish_immune"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_RUNTIME_MASKS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_outcomes"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_publish_outcomes"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "PUBLISH_RUNTIME_MASKS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_outcomes"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_outcomes_packed"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
                "PACKED_INCREMENTAL_SNAPSHOT": 1,
            },
        )
        self.programs["resolve_outcomes_from_supported_connected_tiles_bridge_publish_outcomes"] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_RUNTIME_MASKS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
            },
        )
        support_u8_outcome_variants = (
            ("", {}),
            ("_publish", {"PUBLISH_RUNTIME_MASKS": 1}),
            ("_bridge", {"DIRECT_BEHAVIOR_INPUTS": 1}),
            (
                "_bridge_publish",
                {"DIRECT_BEHAVIOR_INPUTS": 1, "PUBLISH_RUNTIME_MASKS": 1},
            ),
            ("_immune", {"PUBLISH_IMMUNE_DIRECT": 1}),
            (
                "_publish_immune",
                {"PUBLISH_RUNTIME_MASKS": 1, "PUBLISH_IMMUNE_DIRECT": 1},
            ),
            (
                "_bridge_immune",
                {"DIRECT_BEHAVIOR_INPUTS": 1, "PUBLISH_IMMUNE_DIRECT": 1},
            ),
            (
                "_bridge_publish_immune",
                {
                    "DIRECT_BEHAVIOR_INPUTS": 1,
                    "PUBLISH_RUNTIME_MASKS": 1,
                    "PUBLISH_IMMUNE_DIRECT": 1,
                },
            ),
            (
                "_outcomes",
                {"PUBLISH_IMMUNE_DIRECT": 1, "PUBLISH_DELAYED_DIRECT": 1},
            ),
            (
                "_publish_outcomes",
                {
                    "PUBLISH_RUNTIME_MASKS": 1,
                    "PUBLISH_IMMUNE_DIRECT": 1,
                    "PUBLISH_DELAYED_DIRECT": 1,
                },
            ),
            (
                "_bridge_outcomes",
                {
                    "DIRECT_BEHAVIOR_INPUTS": 1,
                    "PUBLISH_IMMUNE_DIRECT": 1,
                    "PUBLISH_DELAYED_DIRECT": 1,
                },
            ),
            (
                "_bridge_outcomes_packed",
                {
                    "DIRECT_BEHAVIOR_INPUTS": 1,
                    "PUBLISH_IMMUNE_DIRECT": 1,
                    "PUBLISH_DELAYED_DIRECT": 1,
                    "PACKED_INCREMENTAL_SNAPSHOT": 1,
                },
            ),
            (
                "_bridge_publish_outcomes",
                {
                    "DIRECT_BEHAVIOR_INPUTS": 1,
                    "PUBLISH_RUNTIME_MASKS": 1,
                    "PUBLISH_IMMUNE_DIRECT": 1,
                    "PUBLISH_DELAYED_DIRECT": 1,
                },
            ),
        )
        for program_suffix, substitutions in support_u8_outcome_variants:
            self.programs[
                f"resolve_outcomes_from_supported_connected_tiles{program_suffix}_u8"
            ] = build_compute_shader(
                ctx,
                "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
                {**_SHADER_SUBS, **substitutions, "SUPPORT_TEXTURE_U8": 1},
            )
        self.programs[
            "resolve_outcomes_from_supported_connected_tiles_bridge_outcomes_u8_label_local"
        ] = build_compute_shader(
            ctx,
            "collapse/resolve_outcomes_from_supported_connected_tiles.comp",
            {
                **_SHADER_SUBS,
                "DIRECT_BEHAVIOR_INPUTS": 1,
                "PUBLISH_IMMUNE_DIRECT": 1,
                "PUBLISH_DELAYED_DIRECT": 1,
                "SUPPORT_TEXTURE_U8": 1,
                "INITIALIZE_LABEL_TILE_UNION": 1,
            },
        )
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
        self.programs["publish_bridge_region_cell_connected_tiles_incremental"] = build_compute_shader(
            ctx,
            "collapse/publish_bridge_region_cell_connected_tiles.comp",
            {**_SHADER_SUBS, "PUBLISH_COMPONENT_LABELS": 1},
        )
        self.programs["load_bridge_region_pending"] = build_compute_shader(ctx, "collapse/load_bridge_region_pending.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_pending"] = build_compute_shader(ctx, "collapse/publish_bridge_region_pending.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_pending_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_pending_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_mask"] = build_compute_shader(ctx, "collapse/publish_bridge_region_mask.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_mask_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_mask_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_supported_unsupported_masks_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_supported_unsupported_masks_connected_tiles.comp", _SHADER_SUBS)
        self.programs["publish_bridge_supported_unsupported_masks_connected_tiles_u8"] = build_compute_shader(
            ctx,
            "collapse/publish_bridge_supported_unsupported_masks_connected_tiles.comp",
            {**_SHADER_SUBS, "SUPPORT_TEXTURE_U8": 1},
        )
        self.programs["publish_bridge_region_labels"] = build_compute_shader(ctx, "collapse/publish_bridge_region_labels.comp", _SHADER_SUBS)
        self.programs["publish_bridge_region_labels_connected_tiles"] = build_compute_shader(ctx, "collapse/publish_bridge_region_labels_connected_tiles.comp", _SHADER_SUBS)

    _ensure_resources = _ensure_resources
    _ensure_formal_connected_u8_support_textures = _ensure_formal_connected_u8_support_textures
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
    _invalidate_persistent_dense_tile_worklist = _invalidate_persistent_dense_tile_worklist
    _persistent_dense_tile_worklist_signature_for = _persistent_dense_tile_worklist_signature
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
    advance_formal_connected_dirty_tile_queue = advance_formal_connected_dirty_tile_queue
    advance_formal_runtime_admission = advance_formal_runtime_admission
    has_active_formal_dirty_epoch = has_active_formal_dirty_epoch
    _validate_and_collect_formal_dirty_epoch_labels = _validate_and_collect_formal_dirty_epoch_labels

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
    _begin_formal_connected_tile_support = _begin_formal_connected_tile_support
    _run_formal_connected_tile_support_slice = _run_formal_connected_tile_support_slice
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
    _begin_formal_connected_component_labeling = _begin_formal_connected_component_labeling
    _run_formal_connected_component_label_slice = _run_formal_connected_component_label_slice
    _publish_formal_connected_component_labels = _publish_formal_connected_component_labels
    _ensure_formal_connected_component_label_union_buffers = _ensure_formal_connected_component_label_union_buffers
    _begin_formal_connected_component_label_union = _begin_formal_connected_component_label_union
    _run_formal_connected_component_label_union_slice = _run_formal_connected_component_label_union_slice
    _materialize_formal_connected_component_label_union = _materialize_formal_connected_component_label_union
    _seed_formal_component_labels_and_axis_masks = _seed_formal_component_labels_and_axis_masks
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
