from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.sim.gpu_collapse_dirty import clear_collapse_structure_dirty_tile_queue_on_gpu
from oracle_game.types import Phase

if TYPE_CHECKING:
    from oracle_game.sim.gpu_collapse import GPUCollapseResources
    from oracle_game.world import WorldEngine


@dataclass(slots=True)
class FormalDirtyCollapseEpoch:
    epoch_id: int
    phase: int
    started_frame_id: int
    world_signature: tuple[int, ...]
    resources: GPUCollapseResources
    x0: int
    y0: int
    width: int
    height: int
    tile_mask_name: str
    support_schedule: tuple[int, ...]
    support_current: Any
    support_scratch: Any
    support_round: int = 0
    outcome_texture: Any | None = None
    label_schedule: tuple[int, ...] = ()
    label_current: Any | None = None
    label_scratch: Any | None = None
    label_round: int = 0
    label_texture: Any | None = None
    label_tile_union_enabled: bool = False
    label_union_edge_capacity: int = 0
    label_union_round_count: int = 0
    label_union_round: int = 0
    label_union_materialized: bool = False
    label_union_materialize_validation_deferred: bool = False
    label_union_local_ready: bool = False
    packed_cell_snapshot: bool = False


@dataclass(slots=True)
class FormalRuntimeAdmission:
    epoch_id: int
    world_signature: tuple[int, ...]
    resources: GPUCollapseResources
    island_id_base: int
    component_capacity: int
    x0: int
    y0: int
    width: int
    height: int
    next_slot: int = 1
    slot_count: int = 4


def _formal_dirty_epoch_world_signature(world: WorldEngine) -> tuple[int, ...]:
    ctx = world.bridge.ctx
    paging = world.paging
    return (
        int(world.width),
        int(world.height),
        int(world.active.tile_width),
        int(world.active.tile_height),
        int(world.active.tile_size),
        int(getattr(paging, "buffer_origin_x", 0)),
        int(getattr(paging, "buffer_origin_y", 0)),
        int(world.bridge.table_generations.get("materials", 0)),
        id(ctx),
    )


def has_active_formal_dirty_epoch(pipeline) -> bool:
    return pipeline._formal_dirty_epoch is not None


def advance_formal_runtime_admission(pipeline, world: WorldEngine) -> int | None:
    admission = pipeline._pending_formal_runtime_admission
    if admission is None:
        return None
    if (
        pipeline.resources is not admission.resources
        or admission.world_signature != _formal_dirty_epoch_world_signature(world)
    ):
        pipeline._pending_formal_runtime_admission = None
        pipeline.incremental_collapse_runtime_admissions_aborted += 1
        world.collapse_dirty_regions.append((0, 0, int(world.width), int(world.height)))
        return None
    slot = int(admission.next_slot)
    with pipeline._profile_pass(world, f"incremental.runtime_admission.slot{slot}"):
        pipeline._publish_compact_component_island_runtime(
            world,
            admission.island_id_base,
            admission.component_capacity,
            admission.x0,
            admission.y0,
            admission.width,
            admission.height,
            admission_slot=slot,
            admission_stride=admission.slot_count,
        )
    admission.next_slot += 1
    pipeline.last_incremental_runtime_admission_slot = slot
    if admission.next_slot >= admission.slot_count:
        pipeline._pending_formal_runtime_admission = None
        pipeline.incremental_collapse_runtime_admissions_completed += 1
    return admission.component_capacity


def _require_epoch_resources(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch) -> bool:
    if pipeline.resources is not epoch.resources:
        world.collapse_dirty_regions.append((0, 0, int(world.width), int(world.height)))
        pipeline._formal_dirty_epoch = None
        pipeline.incremental_collapse_epochs_aborted += 1
        return False
    if epoch.world_signature != _formal_dirty_epoch_world_signature(world):
        # Preserve correctness over a stale commit. The CPU region is retained
        # for the normal rebuild path after a paging/resize/context transition.
        world.collapse_dirty_regions.append((0, 0, int(world.width), int(world.height)))
        pipeline._formal_dirty_epoch = None
        pipeline.incremental_collapse_epochs_aborted += 1
        return False
    return True


def _begin_formal_dirty_epoch(pipeline, world: WorldEngine) -> FormalDirtyCollapseEpoch:
    resource_region = (0, 0, int(world.width), int(world.height))
    resources, x0, y0, width, height = pipeline._prepare_formal_connected_tile_resources_without_input_upload(
        world,
        resource_region,
    )
    with pipeline._profile_pass(world, "incremental.phase0.tile_region_worklist"):
        tile_mask_name = pipeline._seed_formal_texture_region_tile_worklist(world, width, height)
        if tile_mask_name is None:
            raise RuntimeError("formal incremental collapse requires a non-empty connected tile worklist")
        pipeline._last_formal_connected_tile_mask_name = tile_mask_name

    # The connected worklist is epoch-owned. Clear the producer queue now so
    # changes from the following simulation frames accumulate for the next job.
    with pipeline._profile_pass(world, "incremental.phase0.claim_dirty_queue"):
        if not clear_collapse_structure_dirty_tile_queue_on_gpu(world):
            raise RuntimeError("formal incremental collapse failed to claim the dirty tile queue")

    with pipeline._profile_pass(world, "incremental.phase0.connected_bridge_input_load"):
        pending_gpu_authoritative = "collapse_delay_pending" in world.bridge.gpu_authoritative_resources
        if not pending_gpu_authoritative:
            pending = world.collapse_delay_pending[y0 : y0 + height, x0 : x0 + width]
            resources.pending_tex.write(np.asarray(pending, dtype=np.float32).tobytes())
    packed_cell_snapshot = bool(
        pending_gpu_authoritative
        and pipeline._incremental_packed_cell_snapshot_enabled
        and pipeline._formal_gpu_frame(world)
        and pipeline._classification_bridge_hydration_fusion_enabled
        and "cell_core" in world.bridge.gpu_authoritative_resources
    )
    fuse_classification_support_axis_u8 = bool(
        pipeline._incremental_classification_support_axis_u8_fusion_enabled
        and pipeline._incremental_support_jfa_u8_enabled
        and not pipeline._support_tile_union_enabled
        and pending_gpu_authoritative
        and pipeline._formal_gpu_frame(world)
        and pipeline._classification_bridge_hydration_fusion_enabled
        and "cell_core" in world.bridge.gpu_authoritative_resources
    )
    support_u8_current = None
    if fuse_classification_support_axis_u8:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("incremental collapse fused support classification requires a GL context")
        support_u8_current, _ = pipeline._ensure_formal_connected_u8_support_textures(
            ctx,
            resources,
        )
    with pipeline._profile_pass(world, "incremental.phase0.classify_filter"):
        classify_kwargs: dict[str, Any] = {}
        if fuse_classification_support_axis_u8:
            classify_kwargs = {
                "build_support_axis_masks_u8": True,
                "support_seed_u8_texture": support_u8_current,
            }
        pipeline._classify_formal_connected_tile_textures(
            world,
            resources,
            tile_mask_name,
            x0,
            y0,
            width,
            height,
            publish_runtime_masks=True,
            snapshot_pending=pending_gpu_authoritative,
            packed_incremental_snapshot=packed_cell_snapshot,
            **classify_kwargs,
        )
    with pipeline._profile_pass(world, "incremental.phase0.support_begin"):
        support_begin_kwargs: dict[str, bool] = {}
        if fuse_classification_support_axis_u8:
            support_begin_kwargs["axis_masks_prebuilt"] = True
        support_current, support_scratch, support_schedule = pipeline._begin_formal_connected_tile_support(
            world,
            resources,
            width,
            height,
            tile_mask_name,
            use_u8=bool(pipeline._incremental_support_jfa_u8_enabled),
            **support_begin_kwargs,
        )

    pipeline.incremental_collapse_epoch_sequence += 1
    return FormalDirtyCollapseEpoch(
        epoch_id=int(pipeline.incremental_collapse_epoch_sequence),
        phase=0,
        started_frame_id=int(world.frame_id),
        world_signature=_formal_dirty_epoch_world_signature(world),
        resources=resources,
        x0=x0,
        y0=y0,
        width=width,
        height=height,
        tile_mask_name=tile_mask_name,
        support_schedule=support_schedule,
        support_current=support_current,
        support_scratch=support_scratch,
        label_tile_union_enabled=bool(pipeline._outcome_label_tile_union_enabled),
        packed_cell_snapshot=packed_cell_snapshot,
    )


def _run_support_slice(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch, stop: int) -> None:
    start = int(epoch.support_round)
    stop = min(len(epoch.support_schedule), max(start, int(stop)))
    epoch.support_current, epoch.support_scratch = pipeline._run_formal_connected_tile_support_slice(
        world,
        epoch.resources,
        epoch.support_current,
        epoch.support_scratch,
        epoch.width,
        epoch.height,
        epoch.tile_mask_name,
        epoch.support_schedule,
        start,
        stop,
    )
    epoch.support_round = stop


def _run_label_slice(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch, stop: int) -> None:
    if epoch.label_current is None or epoch.label_scratch is None or epoch.outcome_texture is None:
        raise RuntimeError("formal incremental collapse label slice started before label initialization")
    start = int(epoch.label_round)
    stop = min(len(epoch.label_schedule), max(start, int(stop)))
    epoch.label_current, epoch.label_scratch = pipeline._run_formal_connected_component_label_slice(
        world,
        epoch.resources,
        epoch.outcome_texture,
        epoch.label_current,
        epoch.label_scratch,
        epoch.width,
        epoch.height,
        epoch.tile_mask_name,
        epoch.label_schedule,
        start,
        stop,
    )
    epoch.label_round = stop
    epoch.label_texture = epoch.label_current


def _advance_phase0(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch) -> None:
    with pipeline._profile_pass(world, "incremental.phase0.support_slice"):
        # Keep the cached-path prefix stable. Peak balancing moves only the
        # trailing refine/outcome boundary, so phase 0 does not become the new peak.
        if (
            pipeline._incremental_phase_peak_v3_balance_enabled
            and pipeline._incremental_jfa_four_frame_balance_enabled
        ):
            support_stop = 2
        else:
            support_stop = (
                3
                if pipeline._incremental_jfa_four_frame_balance_enabled
                else (3 if pipeline._persistent_dense_tile_worklist_enabled else 2)
            )
        _run_support_slice(pipeline, world, epoch, support_stop)
    epoch.phase = 1


def _finish_support_and_resolve_outcome(
    pipeline,
    world: WorldEngine,
    epoch: FormalDirtyCollapseEpoch,
    *,
    profile_prefix: str,
) -> None:
    with pipeline._profile_pass(world, f"{profile_prefix}.support_slice"):
        _run_support_slice(pipeline, world, epoch, len(epoch.support_schedule))
    fuse_support_publish = bool(
        pipeline._incremental_support_outcome_publish_fusion_enabled
    )
    publish_immune_direct = bool(
        pipeline._incremental_direct_immune_publish_enabled
    )
    publish_delayed_direct = bool(
        pipeline._incremental_direct_delayed_publish_enabled
        and pipeline._incremental_jfa_four_frame_balance_enabled
    )
    initialize_label_tile_union = bool(
        epoch.label_tile_union_enabled
        and pipeline._incremental_outcome_label_local_fusion_enabled
        and pipeline._formal_gpu_frame(world)
        and pipeline._classification_bridge_hydration_fusion_enabled
        and "cell_core" in world.bridge.gpu_authoritative_resources
        and (
            epoch.support_current is epoch.resources.support_u8_ping
            or epoch.support_current is epoch.resources.support_u8_pong
        )
        and not fuse_support_publish
        and publish_immune_direct
        and publish_delayed_direct
        and not epoch.packed_cell_snapshot
    )
    if initialize_label_tile_union:
        pipeline._ensure_formal_connected_component_label_union_buffers(
            world,
            epoch.resources,
            epoch.width,
            epoch.height,
        )
    if not fuse_support_publish:
        with pipeline._profile_pass(world, f"{profile_prefix}.publish_support_masks"):
            pipeline._publish_bridge_supported_unsupported_masks_connected_tiles(
                world,
                epoch.resources,
                epoch.support_current,
                epoch.x0,
                epoch.y0,
                epoch.width,
                epoch.height,
                tile_mask_name=epoch.tile_mask_name,
            )
    with pipeline._profile_pass(world, f"{profile_prefix}.resolve_outcome"):
        outcome_resources, _, _ = pipeline.resolve_supported_outcome_textures(
            world,
            epoch.resources,
            epoch.support_current,
            epoch.x0,
            epoch.y0,
            epoch.width,
            epoch.height,
            eligibility_texture=epoch.resources.structural_tex,
            tile_mask_name=epoch.tile_mask_name,
            publish_runtime_masks=fuse_support_publish,
            publish_outputs=False,
            pending_already_loaded=True,
            publish_immune_direct=publish_immune_direct,
            publish_delayed_direct=publish_delayed_direct,
            packed_material_snapshot=epoch.packed_cell_snapshot,
            initialize_label_tile_union=initialize_label_tile_union,
        )
        epoch.outcome_texture = outcome_resources.phase_out_tex
        epoch.label_union_local_ready = initialize_label_tile_union
    if not publish_immune_direct:
        with pipeline._profile_pass(world, f"{profile_prefix}.publish_immune_mask"):
            _publish_epoch_masks(pipeline, world, epoch, publish_delayed=False, publish_immune=True)


def _advance_phase1(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch) -> None:
    if pipeline._incremental_jfa_four_frame_balance_enabled:
        refine_pass_count = min(
            len(epoch.support_schedule),
            max(0, int(pipeline._formal_connected_tile_refine_pass_count(world))),
        )
        coarse_stop = len(epoch.support_schedule) - refine_pass_count
        with pipeline._profile_pass(world, "incremental.phase1.support_slice"):
            _run_support_slice(pipeline, world, epoch, coarse_stop)
    else:
        _finish_support_and_resolve_outcome(
            pipeline,
            world,
            epoch,
            profile_prefix="incremental.phase1",
        )
    epoch.phase = 2


def _advance_phase2(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch) -> None:
    if pipeline._incremental_jfa_four_frame_balance_enabled:
        _finish_support_and_resolve_outcome(
            pipeline,
            world,
            epoch,
            profile_prefix="incremental.phase2",
        )
    if epoch.outcome_texture is None:
        raise RuntimeError("formal incremental collapse label initialization requires resolved outcomes")
    if epoch.label_tile_union_enabled:
        with pipeline._profile_pass(world, "incremental.phase2.label_union_begin"):
            epoch.label_union_round_count, epoch.label_union_edge_capacity = (
                pipeline._begin_formal_connected_component_label_union(
                    world,
                    epoch.resources,
                    epoch.outcome_texture,
                    epoch.width,
                    epoch.height,
                    epoch.tile_mask_name,
                    local_components_ready=epoch.label_union_local_ready,
                )
            )
        with pipeline._profile_pass(world, "incremental.phase2.label_union_slice"):
            epoch.label_union_round = pipeline._run_formal_connected_component_label_union_slice(
                world,
                epoch.resources,
                epoch.label_union_edge_capacity,
                epoch.label_union_round,
                epoch.label_union_round_count,
            )
        if pipeline._incremental_phase_peak_v3_balance_enabled:
            if epoch.label_union_round < epoch.label_union_round_count:
                raise RuntimeError("formal incremental collapse cannot materialize an incomplete label union")
            if pipeline._incremental_label_union_materialize_validation_fusion_enabled:
                # The completed union buffers are collapse-private and remain
                # frozen until phase 3. No phase-2 consumer observes the label
                # texture, so validation can materialize it while consuming it.
                epoch.label_union_materialize_validation_deferred = True
            else:
                with pipeline._profile_pass(world, "incremental.phase2.label_union_materialize"):
                    epoch.label_texture, epoch.label_scratch = (
                        pipeline._materialize_formal_connected_component_label_union(
                            world,
                            epoch.resources,
                            epoch.width,
                            epoch.height,
                            epoch.tile_mask_name,
                        )
                    )
                epoch.label_union_materialized = True
    else:
        with pipeline._profile_pass(world, "incremental.phase2.label_begin"):
            epoch.label_current, epoch.label_scratch, epoch.label_schedule = (
                pipeline._begin_formal_connected_component_labeling(
                    world,
                    epoch.resources,
                    epoch.outcome_texture,
                    epoch.width,
                    epoch.height,
                    epoch.tile_mask_name,
                )
            )
        with pipeline._profile_pass(world, "incremental.phase2.label_slice"):
            # Validation and materialization leave phase 3 with substantial fixed
            # work. Five trailing rounds balance the two label phases at 1080p.
            _run_label_slice(pipeline, world, epoch, max(0, len(epoch.label_schedule) - 5))
    if not (
        pipeline._incremental_direct_delayed_publish_enabled
        and pipeline._incremental_jfa_four_frame_balance_enabled
    ):
        with pipeline._profile_pass(world, "incremental.phase2.publish_outcome_masks"):
            _publish_epoch_masks(pipeline, world, epoch, publish_delayed=True, publish_immune=False)
    epoch.phase = 3


def _publish_epoch_masks(
    pipeline,
    world: WorldEngine,
    epoch: FormalDirtyCollapseEpoch,
    *,
    publish_delayed: bool,
    publish_immune: bool,
) -> None:
    from oracle_game.sim.gpu_collapse import (
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
    )

    resources = epoch.resources
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal incremental outcome publish requires a valid GL context")
    program = pipeline.programs["publish_incremental_outcome_masks"]
    program["cell_grid_size"].value = (int(epoch.width), int(epoch.height))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["region_origin"].value = (int(epoch.x0), int(epoch.y0))
    program["tile_grid_size"].value = (
        int(world.active.tile_width),
        int(world.active.tile_height),
    )
    program["tile_size"].value = int(world.active.tile_size)
    program["publish_delayed"].value = bool(publish_delayed)
    program["publish_immune"].value = bool(publish_immune)
    resources.temp_out_tex.use(location=0)
    resources.integrity_out_tex.use(location=1)
    world.bridge.buffers["collapse_delay_pending"].bind_to_storage_buffer(binding=0)
    world.bridge.buffers["collapse_delayed_pending_mask"].bind_to_storage_buffer(binding=1)
    world.bridge.buffers["collapse_immune_unsupported_mask"].bind_to_storage_buffer(binding=2)
    world.bridge.buffers[epoch.tile_mask_name].bind_to_storage_buffer(binding=3)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=4)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=5)
    program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)
    authoritative: list[str] = []
    if publish_delayed:
        authoritative.extend(("collapse_delay_pending", "collapse_delayed_pending_mask"))
    if publish_immune:
        authoritative.append("collapse_immune_unsupported_mask")
    world.bridge.mark_gpu_authoritative(*authoritative)


def _validate_and_collect_formal_dirty_epoch_labels(
    pipeline,
    world: WorldEngine,
    resources: GPUCollapseResources,
    label_texture: Any,
    scratch_texture: Any,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tile_mask_name: str,
    *,
    packed_cell_snapshot: bool = False,
    materialize_label_union: bool = False,
) -> tuple[Any, int]:
    from oracle_game.sim.gpu_collapse import (
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
    )

    if scratch_texture is None:
        raise RuntimeError("formal incremental collapse validation requires a label scratch texture")
    if label_texture is None:
        raise RuntimeError("formal incremental collapse validation requires a label texture")
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal incremental collapse validation requires a valid GL context")
    component_capacity = max(1, int(width) * int(height))
    generation_validity = bool(
        pipeline._incremental_component_invalid_generation_enabled
        and pipeline._formal_gpu_frame(world)
        and materialize_label_union
        and not packed_cell_snapshot
    )
    invalid_generation: int | None = None
    clear_invalid = not generation_validity
    if generation_validity:
        previous_generation = int(pipeline._component_invalid_generation)
        clear_invalid = previous_generation <= 0 or previous_generation >= 0xFFFFFFFF
        invalid_generation = 1 if clear_invalid else previous_generation + 1
        pipeline._component_invalid_generation = invalid_generation
    component_flag_generation = (
        invalid_generation
        if invalid_generation is not None
        and pipeline._incremental_component_flag_generation_enabled
        and pipeline._incremental_materialize_metadata_fusion_enabled
        and pipeline._incremental_materialize_filter_fusion_enabled
        else None
    )
    pipeline._active_component_flag_generation = int(component_flag_generation or 0)
    if clear_invalid:
        with pipeline._profile_pass(world, "incremental.validation.clear_invalid"):
            clear_program = pipeline.programs["clear_incremental_component_invalid"]
            clear_program["component_capacity"].value = component_capacity
            resources.component_invalid.bind_to_storage_buffer(binding=0)
            clear_program.run((component_capacity + 255) // 256, 1, 1)
            pipeline._sync_compute_writes(ctx)

    if materialize_label_union:
        validate_program_name = (
            "validate_incremental_component_labels_union_materialize_generation"
            if invalid_generation is not None
            else "validate_incremental_component_labels_union_materialize"
        )
        if packed_cell_snapshot:
            validate_program_name += "_packed"
    else:
        validate_program_name = (
            "validate_incremental_component_labels_packed"
            if packed_cell_snapshot
            else "validate_incremental_component_labels"
        )
    validate_program = pipeline.programs[validate_program_name]
    validate_program["cell_grid_size"].value = (int(width), int(height))
    validate_program["world_grid_size"].value = (int(world.width), int(world.height))
    validate_program["region_origin"].value = (int(x0), int(y0))
    validate_program["tile_grid_size"].value = (
        int(world.active.tile_width),
        int(world.active.tile_height),
    )
    validate_program["tile_size"].value = int(world.active.tile_size)
    validate_program["component_capacity"].value = component_capacity
    if invalid_generation is not None:
        validate_program["invalid_generation"].value = invalid_generation
    if materialize_label_union:
        roots = resources.support_tile_union_roots
        parents = resources.support_tile_union_parent
        if roots is None or parents is None:
            raise RuntimeError("formal incremental collapse fused validation requires label union buffers")
        label_texture.bind_to_image(0, read=False, write=True)
        roots.bind_to_storage_buffer(binding=6)
        parents.bind_to_storage_buffer(binding=7)
    else:
        label_texture.use(location=0)
    if packed_cell_snapshot:
        resources.component_labels.bind_to_storage_buffer(binding=5)
    else:
        resources.material_tex.use(location=1)
        resources.phase_tex.use(location=2)
    world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    resources.component_invalid.bind_to_storage_buffer(binding=1)
    world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=2)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=3)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=4)
    with pipeline._profile_pass(world, "incremental.validation.compare_live"):
        validate_program.run_indirect(
            world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER]
        )
        pipeline._sync_compute_writes(ctx)

    component_capacity = pipeline._prepare_formal_component_list_and_metadata(
        world,
        label_texture,
        x0,
        y0,
        width,
        height,
        tile_mask_name=tile_mask_name,
        reject_invalid_components=True,
        defer_metadata_summary=bool(
            pipeline._incremental_materialize_metadata_fusion_enabled
        ),
        invalid_generation=invalid_generation,
        component_flag_generation=component_flag_generation,
    )

    if pipeline._incremental_materialize_filter_fusion_enabled:
        return label_texture, component_capacity

    filter_program = pipeline.programs["filter_incremental_component_labels"]
    filter_program["cell_grid_size"].value = (int(width), int(height))
    filter_program["tile_grid_size"].value = (
        int(world.active.tile_width),
        int(world.active.tile_height),
    )
    filter_program["tile_size"].value = int(world.active.tile_size)
    filter_program["component_capacity"].value = int(component_capacity)
    label_texture.use(location=0)
    scratch_texture.bind_to_image(1, read=False, write=True)
    resources.component_flags.bind_to_storage_buffer(binding=0)
    world.bridge.buffers[tile_mask_name].bind_to_storage_buffer(binding=1)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=2)
    world.bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=3)
    filter_program.run_indirect(world.bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)
    return scratch_texture, component_capacity


def _materialize_formal_dirty_epoch_direct(
    pipeline,
    world: WorldEngine,
    epoch: FormalDirtyCollapseEpoch,
    island_id_base: int,
    component_capacity: int,
) -> None:
    from oracle_game.sim.gpu_collapse import (
        FORMAL_CONNECTED_TILE_COUNT_BUFFER,
        FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER,
        FORMAL_CONNECTED_TILE_LIST_BUFFER,
    )

    if epoch.label_texture is None:
        raise RuntimeError("formal incremental direct materialization requires a completed label texture")
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("formal incremental direct materialization requires a valid GL context")
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("formal incremental direct materialization requires bridge GPU resources")
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("incremental collapse materialization", "cell_core")
        bridge.sync_world(world)

    resources = epoch.resources
    collapse_generation, base_integrity, spawn_temperature = pipeline._materialize_material_params(world)
    pipeline._write_dynamic_buffer(ctx, resources, "material_collapse_generation", collapse_generation)
    pipeline._write_dynamic_buffer(ctx, resources, "material_base_integrity", base_integrity)
    pipeline._write_dynamic_buffer(ctx, resources, "material_spawn_temperature", spawn_temperature)

    summarize_metadata = bool(
        pipeline._incremental_materialize_metadata_fusion_enabled
    )
    filter_labels = bool(pipeline._incremental_materialize_filter_fusion_enabled)
    if filter_labels and epoch.label_scratch is None:
        raise RuntimeError("formal incremental fused label filtering requires a scratch texture")
    program_name = "materialize_incremental_components_bridge"
    if summarize_metadata:
        program_name += "_metadata"
    if filter_labels:
        program_name += "_filter"
    component_flag_generation = int(pipeline._active_component_flag_generation)
    if component_flag_generation > 0:
        if program_name != "materialize_incremental_components_bridge_metadata_filter":
            raise RuntimeError(
                "component flag generation requires fused metadata and label filtering"
            )
        program_name += "_generation"
    program = pipeline.programs[program_name]
    program["cell_grid_size"].value = (int(epoch.width), int(epoch.height))
    program["region_origin"].value = (int(epoch.x0), int(epoch.y0))
    program["world_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (
        int(world.active.tile_width),
        int(world.active.tile_height),
    )
    program["tile_size"].value = int(world.active.tile_size)
    program["label_capacity"].value = int(component_capacity)
    program["island_id_base"].value = int(island_id_base)
    program["material_count"].value = int(collapse_generation.size)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    if component_flag_generation > 0:
        program["component_flag_generation"].value = component_flag_generation
    epoch.label_texture.use(location=0)
    resources.material_collapse_generation.bind_to_storage_buffer(binding=0)
    resources.material_base_integrity.bind_to_storage_buffer(binding=1)
    resources.material_spawn_temperature.bind_to_storage_buffer(binding=2)
    resources.component_flags.bind_to_storage_buffer(binding=3)
    bridge.buffers[epoch.tile_mask_name].bind_to_storage_buffer(binding=4)
    bridge.buffers[FORMAL_CONNECTED_TILE_COUNT_BUFFER].bind_to_storage_buffer(binding=5)
    bridge.buffers[FORMAL_CONNECTED_TILE_LIST_BUFFER].bind_to_storage_buffer(binding=6)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=7)
    bridge.buffers["island_id"].bind_to_storage_buffer(binding=8)
    bridge.buffers["entity_id"].bind_to_storage_buffer(binding=9)
    bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=10)
    bridge.buffers["collapse_component_label"].bind_to_storage_buffer(binding=11)
    bridge.buffers["collapse_collapsed_cell_mask"].bind_to_storage_buffer(binding=12)
    if summarize_metadata:
        resources.component_metadata.bind_to_storage_buffer(binding=13)
    if component_flag_generation > 0:
        resources.component_invalid.bind_to_storage_buffer(binding=14)
    bridge.textures["material"].bind_to_image(0, read=False, write=True)
    if filter_labels:
        epoch.label_scratch.bind_to_image(1, read=False, write=True)
    program.run_indirect(bridge.buffers[FORMAL_CONNECTED_TILE_DISPATCH_ARGS_BUFFER])
    pipeline._sync_compute_writes(ctx)
    if filter_labels:
        epoch.label_texture, epoch.label_scratch = epoch.label_scratch, epoch.label_texture
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "collapse_component_label",
        "collapse_collapsed_cell_mask",
    )
    pipeline.last_cpu_mirror_downloaded = False


def _commit_formal_dirty_epoch(pipeline, world: WorldEngine, epoch: FormalDirtyCollapseEpoch) -> int:
    fuse_label_union_materialize_validation = bool(
        epoch.label_tile_union_enabled
        and not epoch.label_union_materialized
        and epoch.label_union_materialize_validation_deferred
        and pipeline._incremental_label_union_materialize_validation_fusion_enabled
    )
    if epoch.label_tile_union_enabled:
        if not epoch.label_union_materialized and not fuse_label_union_materialize_validation:
            with pipeline._profile_pass(world, "incremental.phase3.label_union_materialize"):
                epoch.label_texture, epoch.label_scratch = (
                    pipeline._materialize_formal_connected_component_label_union(
                        world,
                        epoch.resources,
                        epoch.width,
                        epoch.height,
                        epoch.tile_mask_name,
                    )
                )
            epoch.label_union_materialized = True
        elif fuse_label_union_materialize_validation:
            if epoch.label_union_round < epoch.label_union_round_count:
                raise RuntimeError("formal incremental collapse cannot validate an incomplete label union")
            epoch.label_texture = epoch.resources.support_ping
            epoch.label_scratch = epoch.resources.support_pong
    else:
        if epoch.label_texture is None:
            raise RuntimeError("formal incremental collapse commit requires a completed label texture")
        with pipeline._profile_pass(world, "incremental.phase3.label_slice"):
            _run_label_slice(pipeline, world, epoch, len(epoch.label_schedule))
    if epoch.label_texture is None:
        raise RuntimeError("formal incremental collapse commit requires a completed label texture")
    with pipeline._profile_pass(world, "incremental.phase3.validate_and_collect"):
        validation_args = (
            world,
            epoch.resources,
            epoch.label_texture,
            epoch.label_scratch,
            epoch.x0,
            epoch.y0,
            epoch.width,
            epoch.height,
            epoch.tile_mask_name,
        )
        validation_kwargs: dict[str, bool] = {}
        if epoch.packed_cell_snapshot:
            validation_kwargs["packed_cell_snapshot"] = True
        if fuse_label_union_materialize_validation:
            validation_kwargs["materialize_label_union"] = True
        if validation_kwargs:
            epoch.label_texture, component_capacity = (
                pipeline._validate_and_collect_formal_dirty_epoch_labels(
                    *validation_args,
                    **validation_kwargs,
                )
            )
        else:
            epoch.label_texture, component_capacity = (
                pipeline._validate_and_collect_formal_dirty_epoch_labels(*validation_args)
            )
    if fuse_label_union_materialize_validation:
        epoch.label_union_materialized = True
    if component_capacity == 0:
        pipeline._publish_formal_connected_component_labels(
            world,
            epoch.resources,
            epoch.label_texture,
            epoch.x0,
            epoch.y0,
            epoch.width,
            epoch.height,
            epoch.tile_mask_name,
        )
        return 0
    island_id_base = pipeline._reserve_formal_component_island_ids(world, component_capacity)
    with pipeline._profile_pass(world, "incremental.phase3.materialize_direct"):
        _materialize_formal_dirty_epoch_direct(
            pipeline,
            world,
            epoch,
            island_id_base,
            component_capacity,
        )
    with pipeline._profile_pass(world, "incremental.phase3.publish_runtime"):
        pipeline._publish_compact_component_island_runtime(
            world,
            island_id_base,
            component_capacity,
            epoch.x0,
            epoch.y0,
            epoch.width,
            epoch.height,
            admission_slot=0,
            admission_stride=4,
        )
    pipeline._pending_formal_runtime_admission = FormalRuntimeAdmission(
        epoch_id=epoch.epoch_id,
        world_signature=epoch.world_signature,
        resources=epoch.resources,
        island_id_base=island_id_base,
        component_capacity=component_capacity,
        x0=epoch.x0,
        y0=epoch.y0,
        width=epoch.width,
        height=epoch.height,
    )
    pipeline.incremental_collapse_runtime_admissions_started += 1
    return component_capacity


def advance_formal_connected_dirty_tile_queue(pipeline, world: WorldEngine) -> int | None:
    epoch = pipeline._formal_dirty_epoch
    if epoch is None:
        epoch = _begin_formal_dirty_epoch(pipeline, world)
        pipeline._formal_dirty_epoch = epoch
        pipeline.incremental_collapse_epochs_started += 1
    if not _require_epoch_resources(pipeline, world, epoch):
        return None
    phase = int(epoch.phase)
    pipeline.last_incremental_collapse_phase = phase
    pipeline.last_incremental_collapse_epoch_id = int(epoch.epoch_id)
    pipeline.last_incremental_collapse_epoch_started_frame_id = int(epoch.started_frame_id)
    with pipeline._profile_pass(world, f"incremental.phase{phase}"):
        if phase == 0:
            _advance_phase0(pipeline, world, epoch)
            return None
        if phase == 1:
            _advance_phase1(pipeline, world, epoch)
            return None
        if phase == 2:
            _advance_phase2(pipeline, world, epoch)
            return None
        if phase != 3:
            raise RuntimeError(f"invalid formal incremental collapse phase {phase}")
        component_capacity = _commit_formal_dirty_epoch(pipeline, world, epoch)
    pipeline._formal_dirty_epoch = None
    pipeline.incremental_collapse_epochs_completed += 1
    return component_capacity
