from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine
    from oracle_game.sim.gpu_liquid import GPULiquidResources

from oracle_game.types import Phase

from oracle_game.sim.gpu_liquid import (
    MAX_MATERIALS,
    PASS_LOCAL_SIZE,
    TILE_SIZE,
)


def _pack_timer_texture(timer_pack: np.ndarray) -> np.ndarray:
    timer = np.asarray(timer_pack, dtype=np.uint8)
    return np.ascontiguousarray(
        timer[..., 0].astype(np.uint32)
        | (timer[..., 1].astype(np.uint32) << 8)
        | (timer[..., 2].astype(np.uint32) << 16)
        | (timer[..., 3].astype(np.uint32) << 24)
    )


def _unpack_timer_texture(packed: np.ndarray) -> np.ndarray:
    timer = np.asarray(packed, dtype=np.uint32)
    return np.stack(
        (
            timer & 0xFF,
            (timer >> 8) & 0xFF,
            (timer >> 16) & 0xFF,
            (timer >> 24) & 0xFF,
        ),
        axis=-1,
    ).astype(np.uint8)


def _pack_cell_state_texture(
    material_id: np.ndarray,
    phase: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    material = np.clip(np.asarray(material_id), 0, 0xFFFF).astype(np.uint32)
    phase_u32 = np.clip(np.asarray(phase), 0, 0xFF).astype(np.uint32)
    flags_u32 = np.clip(np.asarray(flags), 0, 0xFF).astype(np.uint32)
    return np.ascontiguousarray(material | (phase_u32 << 16) | (flags_u32 << 24))


def step(
    pipeline,
    world: "WorldEngine",
    *,
    solve_tile_mask: np.ndarray,
    post_tile_mask: np.ndarray,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.reset_pass_profile()
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
    pipeline.last_provenance_terminal_used = False
    pipeline.last_provenance_init_fusion_used = False
    pipeline._provenance_init_fusion_frame_enabled = False
    pipeline._seam_x_multirow_frame_rows = 0
    pipeline._provenance_terminal_frame_enabled = bool(
        pipeline._provenance_terminal_enabled
        and formal_gpu_frame
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and pipeline._flow_intent_shared_halo_enabled
        and pipeline._tile_solve_snapshot_output_fusion_enabled
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and pipeline._compact_tile_solve_snapshot_enabled
        and pipeline._seam_x_row_leader_enabled
        and pipeline._seam_y_shared_snapshot_enabled
        and pipeline._buoyancy_pass_fusion_enabled
        and pipeline._active_scheduler_gpu_authoritative(world)
        and {
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    pipeline._provenance_init_fusion_frame_enabled = bool(
        pipeline._provenance_init_fusion_enabled
        and pipeline._provenance_terminal_frame_enabled
    )
    pipeline.last_buoyancy_cleanup_split_fusion_used = False
    pipeline._buoyancy_cleanup_split_fusion_frame_enabled = bool(
        pipeline._buoyancy_cleanup_split_fusion_enabled
        and pipeline._provenance_terminal_frame_enabled
        and pipeline._buoyancy_pass_fusion_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and pipeline._placeholder_lazy_roles_enabled
        and {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    pipeline.last_buoyancy_snapshot_pre_state_used = False
    pipeline._buoyancy_snapshot_pre_state_frame_enabled = bool(
        pipeline._buoyancy_snapshot_pre_state_enabled
        and pipeline._buoyancy_cleanup_split_fusion_frame_enabled
        and pipeline._provenance_terminal_frame_enabled
        and pipeline._tile_solve_snapshot_output_fusion_enabled
        and pipeline._compact_tile_solve_snapshot_enabled
        and not pipeline._tile_solve_liquid_kind_cache_enabled
        and MAX_MATERIALS == 256
        and TILE_SIZE * TILE_SIZE < 0xFFFF
    )
    pipeline.last_tile_snapshot_state_elision_used = False
    pipeline._tile_snapshot_state_elision_frame_enabled = bool(
        pipeline._tile_snapshot_state_elision_enabled
        and pipeline._buoyancy_snapshot_pre_state_frame_enabled
        and pipeline._compact_tile_solve_snapshot_enabled
    )
    pipeline.last_buoyancy_shared_sink_cache_used = False
    pipeline._buoyancy_shared_sink_cache_frame_enabled = bool(
        pipeline._buoyancy_shared_sink_cache_enabled
        and formal_gpu_frame
        and pipeline._buoyancy_snapshot_pre_state_frame_enabled
        and pipeline._buoyancy_pass_fusion_enabled
        and pipeline._provenance_terminal_frame_enabled
        and PASS_LOCAL_SIZE == 8
    )
    pipeline.last_provenance_cleanup_terminal_fusion_used = False
    pipeline._provenance_cleanup_terminal_fusion_frame_enabled = bool(
        pipeline._provenance_cleanup_terminal_fusion_enabled
        and not pipeline._buoyancy_cleanup_split_fusion_frame_enabled
        and pipeline._provenance_terminal_frame_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and pipeline._placeholder_lazy_roles_enabled
        and {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    pipeline.last_buoyancy_blocker_displaced_hydration_used = False
    pipeline._buoyancy_blocker_displaced_hydration_frame_enabled = bool(
        pipeline._buoyancy_blocker_displaced_hydration_enabled
        and pipeline._buoyancy_cleanup_split_fusion_frame_enabled
        and pipeline._buoyancy_snapshot_pre_state_frame_enabled
        and pipeline._provenance_terminal_frame_enabled
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and pipeline._tile_solve_snapshot_output_fusion_enabled
        and pipeline._compact_tile_solve_snapshot_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and pipeline._placeholder_lazy_roles_enabled
        and not pipeline._bridge_aux_residency_enabled
        and {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    pipeline._blocker_displaced_hydration_frame_enabled = bool(
        pipeline._provenance_cleanup_terminal_fusion_frame_enabled
        or pipeline._buoyancy_blocker_displaced_hydration_frame_enabled
    )
    pipeline.last_cleanup_flow_fusion_used = False
    pipeline._cleanup_flow_fusion_frame_enabled = bool(
        pipeline._cleanup_flow_fusion_enabled
        and formal_gpu_frame
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and not pipeline._provenance_terminal_frame_enabled
        and pipeline._flow_intent_shared_halo_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    pipeline.last_flow_active_decay_fusion_used = False
    pipeline._flow_active_decay_fusion_frame_id = None
    pipeline._flow_active_decay_fusion_frame_enabled = bool(
        pipeline._flow_active_decay_fusion_enabled
        and formal_gpu_frame
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
        and pipeline._flow_intent_shared_halo_enabled
        and not pipeline._provenance_terminal_frame_enabled
        and not pipeline._cleanup_flow_fusion_frame_enabled
        and not pipeline._bridge_aux_residency_enabled
        and pipeline._active_scheduler_gpu_authoritative(world)
    )
    with pipeline._profile_pass(world, "liquid_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
    with pipeline._profile_pass(world, "liquid_load_bridge_inputs"):
        active_tiles_ready_for_solve = pipeline._load_authoritative_bridge_inputs(
            world,
            resources,
            next_workgroups_per_tile=1,
        )
    if not active_tiles_ready_for_solve:
        with pipeline._profile_pass(world, "liquid_compact_active_tiles"):
            pipeline._compact_active_tiles(world, resources)
    if pipeline._provenance_terminal_frame_enabled:
        with pipeline._profile_pass(world, "liquid_init_provenance"):
            pipeline._run_provenance_init(
                world,
                resources,
                skip_when_all_tiles_active=pipeline._provenance_init_fusion_frame_enabled,
            )
        pipeline.last_provenance_init_fusion_used = bool(
            pipeline._provenance_init_fusion_frame_enabled
        )
    with pipeline._profile_pass(world, "liquid_tile_solve"):
        pipeline._run_tile_solve(world, resources)
    if pipeline._active_scheduler_gpu_authoritative(world):
        with pipeline._profile_pass(world, "liquid_load_active_mask"):
            pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
    else:
        with pipeline._profile_pass(world, "liquid_upload_active_mask"):
            pipeline._upload_active_tile_mask(resources, post_tile_mask)
    with pipeline._profile_pass(world, "liquid_compact_active_cell_tiles"):
        pipeline._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
        )
    direct_bridge_aux_inputs = bool(
        formal_gpu_frame
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    seam_y_shared_snapshot = bool(
        formal_gpu_frame
        and pipeline._seam_y_shared_snapshot_enabled
        and pipeline._buoyancy_pass_fusion_enabled
    )
    tile_snapshot_fusion = bool(
        formal_gpu_frame
        and pipeline._tile_solve_snapshot_output_fusion_enabled
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and pipeline._active_scheduler_gpu_authoritative(world)
        and {
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    requested_seam_rows = int(pipeline._seam_x_multirow_rows)
    pipeline._seam_x_multirow_frame_rows = (
        requested_seam_rows
        if requested_seam_rows == 4
        and tile_snapshot_fusion
        and pipeline._seam_x_row_leader_enabled
        and pipeline.tile_solve_warp_fast_path
        and int(world.active.tile_size) == TILE_SIZE
        else 0
    )
    if formal_gpu_frame:
        if not tile_snapshot_fusion:
            with pipeline._profile_pass(world, "liquid_copy_tile_solve"):
                pipeline._run_copy_core_state(
                    world,
                    resources,
                    (
                        resources.cell_state_out,
                        resources.timer_out,
                        resources.temp_out,
                        resources.integrity_out,
                        resources.velocity_out,
                    ),
                    (
                        resources.cell_state_in,
                        resources.timer_in,
                        resources.temp_in,
                        resources.integrity_in,
                        resources.velocity_in,
                    ),
                )
        with pipeline._profile_pass(world, "liquid_build_seam_x_boundaries"):
            pipeline._build_seam_boundary_dispatch(world, resources, axis="x")
        with pipeline._profile_pass(world, "liquid_prefetch_seam_x_boundaries"):
            pipeline._prefetch_seam_boundary_bridge_inputs(world, resources, axis="x")
    with pipeline._profile_pass(world, "liquid_seam_x"):
        seam_x_program = "seam_x"
        if tile_snapshot_fusion:
            if pipeline._seam_x_multirow_frame_rows == 4:
                seam_x_program = (
                    "seam_x_snapshot_row_leader4_provenance"
                    if pipeline._provenance_terminal_frame_enabled
                    else "seam_x_snapshot_row_leader4"
                )
            elif pipeline._seam_x_row_leader_enabled:
                seam_x_program = (
                    "seam_x_snapshot_row_leader_provenance"
                    if pipeline._provenance_terminal_frame_enabled
                    else "seam_x_snapshot_row_leader"
                )
            else:
                seam_x_program = "seam_x_snapshot"
            if pipeline._buoyancy_snapshot_pre_state_frame_enabled:
                if not pipeline._provenance_terminal_frame_enabled:
                    raise RuntimeError("liquid snapshot pre-state requires provenance seam")
                seam_x_program += "_snapshot_pre"
                if pipeline._tile_snapshot_state_elision_frame_enabled:
                    seam_x_program += "_state_elided"
        elif direct_bridge_aux_inputs:
            seam_x_program = "seam_x_bridge_aux"
        pipeline._run_seam_pass(
            seam_x_program,
            world,
            (
                resources.cell_state_in if tile_snapshot_fusion else resources.cell_state_out,
                resources.timer_in if tile_snapshot_fusion else resources.timer_out,
                resources.temp_in if tile_snapshot_fusion else resources.temp_out,
                resources.integrity_in if tile_snapshot_fusion else resources.integrity_out,
                resources.velocity_in if tile_snapshot_fusion else resources.velocity_out,
            ),
            (
                resources.cell_state_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.active_tile_tex,
            boundary_dispatch=formal_gpu_frame,
        )
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_reload_seam_x_active_tiles"):
            pipeline._reload_and_compact_active_cell_tiles(world, resources)
        if not seam_y_shared_snapshot:
            with pipeline._profile_pass(world, "liquid_copy_seam_x"):
                pipeline._run_copy_core_state(
                    world,
                    resources,
                    (
                        resources.cell_state_in,
                        resources.timer_in,
                        resources.temp_in,
                        resources.integrity_in,
                        resources.velocity_in,
                    ),
                    (
                        resources.cell_state_out,
                        resources.timer_out,
                        resources.temp_out,
                        resources.integrity_out,
                        resources.velocity_out,
                    ),
                )
        with pipeline._profile_pass(world, "liquid_build_seam_y_boundaries"):
            pipeline._build_seam_boundary_dispatch(world, resources, axis="y")
        with pipeline._profile_pass(world, "liquid_prefetch_seam_y_boundaries"):
            pipeline._prefetch_seam_boundary_bridge_inputs(world, resources, axis="y")
    with pipeline._profile_pass(world, "liquid_seam_y"):
        seam_y_read_resources = (
            resources.cell_state_in,
            resources.timer_in,
            resources.temp_in,
            resources.integrity_in,
            resources.velocity_in,
        )
        seam_y_write_resources = (
            seam_y_read_resources
            if seam_y_shared_snapshot
            else (
                resources.cell_state_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            )
        )
        pipeline._run_seam_pass(
            (
                "seam_y_shared_snapshot_provenance_aux"
                if pipeline._provenance_terminal_frame_enabled
                else "seam_y_shared_snapshot_aux"
                if seam_y_shared_snapshot and direct_bridge_aux_inputs
                else "seam_y_shared_snapshot"
                if seam_y_shared_snapshot
                else "seam_y_bridge_aux"
                if direct_bridge_aux_inputs
                else "seam_y"
            ),
            world,
            seam_y_read_resources,
            seam_y_write_resources,
            resources.active_tile_tex,
            boundary_dispatch=formal_gpu_frame,
        )
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_reload_seam_y_active_tiles"):
            pipeline._reload_and_compact_active_cell_tiles(world, resources)
    if pipeline._buoyancy_pass_fusion_enabled:
        buoyancy_read_resources = (
            (
                resources.cell_state_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            )
            if seam_y_shared_snapshot
            else (
                resources.cell_state_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            )
        )
        buoyancy_write_resources = (
            (
                resources.cell_state_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            )
            if seam_y_shared_snapshot
            else (
                resources.cell_state_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            )
        )
        with pipeline._profile_pass(world, "liquid_buoyancy_fused"):
            pipeline._run_buoyancy_pass(
                (
                    "buoyancy_fused_provenance_cleanup_snapshot_pre"
                    if pipeline._buoyancy_snapshot_pre_state_frame_enabled
                    else
                    "buoyancy_fused_provenance_cleanup"
                    if pipeline._buoyancy_cleanup_split_fusion_frame_enabled
                    else "buoyancy_fused_provenance"
                    if pipeline._provenance_terminal_frame_enabled
                    else "buoyancy_fused"
                )
                + (
                    "_state_elided"
                    if pipeline._tile_snapshot_state_elision_frame_enabled
                    else ""
                )
                + (
                    "_shared_sink"
                    if pipeline._buoyancy_shared_sink_cache_frame_enabled
                    else ""
                ),
                world,
                resources,
                buoyancy_read_resources,
                buoyancy_write_resources,
                sink_fallback_resources=(
                    buoyancy_read_resources if seam_y_shared_snapshot else None
                ),
            )
        pipeline.last_buoyancy_snapshot_pre_state_used = bool(
            pipeline._buoyancy_snapshot_pre_state_frame_enabled
        )
        pipeline.last_tile_snapshot_state_elision_used = bool(
            pipeline._tile_snapshot_state_elision_frame_enabled
        )
        pipeline.last_buoyancy_shared_sink_cache_used = bool(
            pipeline._buoyancy_shared_sink_cache_frame_enabled
        )
        if pipeline._provenance_terminal_frame_enabled:
            resources.provenance_in, resources.provenance_out = (
                resources.provenance_out,
                resources.provenance_in,
            )
        # The fused pass produces the same final role as float without writing
        # an intermediate texture set. Keep all downstream role assumptions.
        if not seam_y_shared_snapshot:
            resources.cell_state_in, resources.cell_state_out = resources.cell_state_out, resources.cell_state_in
            resources.timer_in, resources.timer_out = resources.timer_out, resources.timer_in
            resources.temp_in, resources.temp_out = resources.temp_out, resources.temp_in
            resources.integrity_in, resources.integrity_out = resources.integrity_out, resources.integrity_in
            resources.velocity_in, resources.velocity_out = resources.velocity_out, resources.velocity_in
    else:
        with pipeline._profile_pass(world, "liquid_buoyancy_sink"):
            pipeline._run_buoyancy_pass(
                "buoyancy_sink",
                world,
                resources,
                (
                    resources.cell_state_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.cell_state_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
            )
        with pipeline._profile_pass(world, "liquid_buoyancy_float"):
            pipeline._run_buoyancy_pass(
                "buoyancy_float",
                world,
                resources,
                (
                    resources.cell_state_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                (
                    resources.cell_state_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
            )
    lazy_placeholder_roles = bool(
        formal_gpu_frame and pipeline._placeholder_lazy_roles_enabled
    )
    if lazy_placeholder_roles:
        with pipeline._profile_pass(world, "liquid_build_placeholder_affected_tiles"):
            pipeline._build_placeholder_dirty_affected_tile_dispatch(
                world,
                resources,
                cell_state_tex=resources.cell_state_out,
                displaced_tex=resources.displaced_in,
            )
    with pipeline._profile_pass(world, "liquid_copy_for_placeholder"):
        pipeline._run_copy_for_placeholder(
            world,
            resources,
            (
                resources.cell_state_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.cell_state_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
            affected_tile_dispatch=lazy_placeholder_roles,
        )
    with pipeline._profile_pass(world, "liquid_placeholder_displacement"):
        pipeline._run_placeholder_displacement(
            world,
            resources,
            (
                resources.cell_state_out,
                resources.timer_out,
                resources.temp_out,
                resources.integrity_out,
                resources.velocity_out,
            ),
            (
                resources.cell_state_in,
                resources.timer_in,
                resources.temp_in,
                resources.integrity_in,
                resources.velocity_in,
            ),
            resources.displaced_in,
            resources.displaced_out,
            affected_dispatch_prepared=lazy_placeholder_roles,
        )
    if not pipeline._cleanup_flow_fusion_frame_enabled:
        if pipeline._buoyancy_cleanup_split_fusion_frame_enabled:
            with pipeline._profile_pass(world, "liquid_cleanup_placeholder_affected"):
                pipeline._run_cleanup_runtime(
                    world,
                    resources,
                    affected_tile_dispatch=True,
                    restore_bridge_aux_values=True,
                )
            pipeline.last_buoyancy_cleanup_split_fusion_used = True
        elif not pipeline._provenance_cleanup_terminal_fusion_frame_enabled:
            with pipeline._profile_pass(world, "liquid_cleanup_runtime"):
                pipeline._run_cleanup_runtime(world, resources)
    if pipeline._active_scheduler_gpu_authoritative(world):
        with pipeline._profile_pass(world, "liquid_reload_flow_active_mask"):
            pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        with pipeline._profile_pass(world, "liquid_compact_flow_active_tiles"):
            pipeline._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
            )
    with pipeline._profile_pass(world, "liquid_flow_intent"):
        pipeline._run_liquid_intent_pass(
            world,
            resources,
            publish_bridge_outputs=formal_gpu_frame,
        )
    if formal_gpu_frame:
        with pipeline._profile_pass(world, "liquid_refresh_active_scheduler"):
            pipeline._refresh_active_scheduler_from_ttl(world)
    else:
        with pipeline._profile_pass(world, "liquid_publish_bridge"):
            pipeline._publish_bridge_outputs(world, resources)
    pipeline.last_cpu_mirror_downloaded = not formal_gpu_frame
    if pipeline.last_cpu_mirror_downloaded:
        ctx.finish()
        pipeline._download_outputs(world, resources, use_in=True)



def prepare_motion_flow_intent(
    pipeline,
    world: "WorldEngine",
    *,
    solve_tile_mask: np.ndarray,
) -> None:
    ctx = world.bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
    pipeline._ensure_programs(ctx)
    resources = pipeline._ensure_resources(world)
    pipeline.reset_pass_profile()
    with pipeline._profile_pass(world, "liquid_pre_motion_upload_inputs"):
        pipeline._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
    with pipeline._profile_pass(world, "liquid_pre_motion_load_bridge_inputs"):
        pipeline._load_authoritative_bridge_flow_intent_inputs(world, resources)
    with pipeline._profile_pass(world, "liquid_pre_motion_compact_active_tiles"):
        pipeline._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
        )
    with pipeline._profile_pass(world, "liquid_pre_motion_flow_intent"):
        pipeline._run_liquid_intent_pass(world, resources)
    pipeline.last_cpu_mirror_downloaded = False



def _upload_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    solve_tile_mask: np.ndarray,
) -> None:
    world.bridge.sync_rule_tables(world)
    authoritative = world.bridge.gpu_authoritative_resources
    formal_gpu_frame = pipeline._formal_gpu_frame(world)
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
    upload_active_from_cpu = not pipeline._active_scheduler_gpu_authoritative(world)
    pipeline.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
    pipeline.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
    pipeline.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
    pipeline.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
    pipeline.last_cpu_active_upload_skipped = not upload_active_from_cpu
    if upload_cell_state_from_cpu:
        packed_cell_state = _pack_cell_state_texture(
            world.material_id,
            world.phase,
            world.cell_flags,
        )
        resources.cell_state_pre.write(packed_cell_state.tobytes())
        resources.cell_state_in.write(packed_cell_state.tobytes())
        resources.cell_state_out.write(packed_cell_state.tobytes())
        packed_timer = _pack_timer_texture(world.timer_pack)
        resources.timer_in.write(packed_timer.tobytes())
        resources.timer_out.write(packed_timer.tobytes())
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
        pipeline._upload_active_tile_mask(resources, solve_tile_mask)
    else:
        pipeline._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
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



def _load_authoritative_bridge_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    next_workgroups_per_tile: int | None = None,
) -> bool:
    if not pipeline._formal_gpu_frame(world):
        return False
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    copy_cell_core = "cell_core" in authoritative
    direct_bridge_cell_core = bool(copy_cell_core and pipeline._tile_solve_bridge_hydration_fusion_enabled)
    hydrate_cell_core = bool(copy_cell_core and not direct_bridge_cell_core)
    copy_island_id = "island_id" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    blocker_mask_inputs = bool(
        pipeline._blocker_displaced_hydration_frame_enabled
        and copy_entity_id
        and copy_displaced
    )
    direct_bridge_aux_inputs = bool(
        copy_cell_core
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and pipeline._bridge_aux_residency_enabled
        and {"island_id", "entity_id", "placeholder_displaced_material"}.issubset(
            authoritative
        )
    )
    skip_island_hydration = bool(
        copy_cell_core
        and pipeline._tile_solve_bridge_hydration_fusion_enabled
        and pipeline._cleanup_bridge_island_residency_enabled
        and pipeline._bridge_aux_cleanup_fusion_enabled
        and not direct_bridge_aux_inputs
        and "island_id" in authoritative
    )
    copy_island_id = bool(
        copy_island_id and not (skip_island_hydration or blocker_mask_inputs)
    )
    hydrate_island_id = bool(copy_island_id and not direct_bridge_aux_inputs)
    copy_entity_id = bool(
        copy_entity_id and not (direct_bridge_aux_inputs or blocker_mask_inputs)
    )
    copy_displaced = bool(
        copy_displaced and not (direct_bridge_aux_inputs or blocker_mask_inputs)
    )
    if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
        return False
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative input state")
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    group_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    group_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
    if active_tile_indirect:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.active_tile_compact"):
            pipeline._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
            )
    ran_copy = False
    if blocker_mask_inputs:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_blocker_displaced"):
            program = pipeline.programs["load_bridge_blocker_displaced"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=0)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=1)
            resources.active_tile_count.bind_to_storage_buffer(binding=2)
            resources.active_tile_list.bind_to_storage_buffer(binding=3)
            resources.blocker_mask.bind_to_image(0, read=False, write=True)
            resources.displaced_in.bind_to_image(1, read=False, write=True)
            pipeline._run_active_tile_indirect(
                program,
                resources,
                "bridge blocker/displaced input load",
            )
            pipeline.last_buoyancy_blocker_displaced_hydration_used = bool(
                pipeline._buoyancy_blocker_displaced_hydration_frame_enabled
            )
            ran_copy = True
    if hydrate_cell_core:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_cell_in"):
            program = pipeline.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["copy_cell_core"].value = True
            program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=4)
            resources.active_tile_list.bind_to_storage_buffer(binding=5)
            resources.cell_state_pre.bind_to_image(0, read=False, write=True)
            resources.cell_state_in.bind_to_image(1, read=False, write=True)
            resources.timer_in.bind_to_image(5, read=False, write=True)
            resources.temp_in.bind_to_image(6, read=False, write=True)
            resources.integrity_in.bind_to_image(7, read=False, write=True)
            if active_tile_indirect:
                pipeline._run_active_tile_indirect(program, resources, "bridge cell input load")
            else:
                program.run(group_x, group_y, 1)
            ran_copy = True

    if hydrate_cell_core or hydrate_island_id or copy_entity_id or copy_displaced:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.load_aux"):
            program = pipeline.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["copy_cell_core"].value = bool(hydrate_cell_core)
            program["copy_island_id"].value = bool(hydrate_island_id)
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
                pipeline._run_active_tile_indirect(program, resources, "bridge aux input load")
            else:
                program.run(group_x, group_y, 1)
            ran_copy = True

    active_tiles_ready_for_next = bool(
        active_tile_indirect
        and (ran_copy or direct_bridge_aux_inputs)
        and next_workgroups_per_tile is not None
    )
    needs_sync = bool(active_tiles_ready_for_next)
    if ran_copy:
        needs_sync = True
    if needs_sync:
        with pipeline._profile_pass(world, "liquid_load_bridge_inputs.sync"):
            if active_tiles_ready_for_next:
                # Bridge loaders consume 16 workgroups per 32x32 tile, while
                # tile solve consumes one. Retarget the already compacted list
                # before the existing barrier instead of rescanning chunks.
                retarget_program = pipeline.programs["retarget_active_tile_dispatch"]
                retarget_program["workgroups_per_tile"].value = max(
                    1,
                    int(next_workgroups_per_tile),
                )
                resources.active_tile_count.bind_to_storage_buffer(binding=0)
                resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
                retarget_program.run(1, 1, 1)
            pipeline._sync_compute_writes(bridge.ctx)
    return active_tiles_ready_for_next



def _load_authoritative_bridge_flow_intent_inputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
) -> None:
    if not pipeline._formal_gpu_frame(world):
        return
    bridge = world.bridge
    authoritative = bridge.gpu_authoritative_resources
    direct_flow_inputs = bool(
        pipeline._flow_intent_bridge_residency_enabled
        and {
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(authoritative)
    )
    if direct_flow_inputs:
        # The pre-motion flow-intent shader reads the authoritative bridge
        # buffers directly. The caller still compacts active tiles before the
        # dispatch, so no texture hydration is needed here.
        return
    copy_cell_core = "cell_core" in authoritative
    copy_entity_id = "entity_id" in authoritative
    copy_displaced = "placeholder_displaced_material" in authoritative
    direct_bridge_aux_inputs = bool(
        pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(authoritative)
    )
    copy_entity_id = bool(copy_entity_id and not direct_bridge_aux_inputs)
    copy_displaced = bool(copy_displaced and not direct_bridge_aux_inputs)
    if not (copy_cell_core or copy_entity_id or copy_displaced):
        return
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for flow-intent input state")
    pipeline._compact_active_tiles(
        world,
        resources,
        workgroups_per_tile=pipeline._active_tile_workgroups_per_tile(world),
    )
    program = pipeline.programs["load_bridge_flow_intent_inputs"]
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
    resources.cell_state_in.bind_to_image(0, read=False, write=True)
    resources.velocity_in.bind_to_image(2, read=False, write=True)
    resources.entity_in.bind_to_image(3, read=False, write=True)
    resources.displaced_in.bind_to_image(4, read=False, write=True)
    pipeline._run_active_tile_indirect(program, resources, "bridge flow-intent input load")
    pipeline._sync_compute_writes(bridge.ctx)



def _publish_bridge_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    use_out: bool = False,
    velocity_use_out: bool = False,
) -> None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        if pipeline._formal_gpu_frame(world):
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative output state")
        return
    program = pipeline.programs["publish_bridge_cell"]
    active_tile_indirect = pipeline._formal_gpu_frame(world)
    program["cell_grid_size"].value = (world.width, world.height)
    program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    program["tile_size"].value = int(world.active.tile_size)
    program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
    program["write_cell_core"].value = not bool(
        getattr(world, "phase_c_defer_cell_publish", False)
    )
    cell_state_tex = resources.cell_state_out if use_out else resources.cell_state_in
    timer_tex = resources.timer_out if use_out else resources.timer_in
    temp_tex = resources.temp_out if use_out else resources.temp_in
    integrity_tex = resources.integrity_out if use_out else resources.integrity_in
    velocity_tex = resources.velocity_out if velocity_use_out else resources.velocity_in
    cell_state_tex.use(location=0)
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
        pipeline._run_active_tile_indirect(program, resources, "bridge cell publish")
    else:
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    pipeline._sync_compute_writes(bridge.ctx)
    bridge.mark_gpu_authoritative(
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
    )



def _run_liquid_intent_pass(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    publish_bridge_outputs: bool = False,
) -> None:
    provenance_terminal = bool(
        publish_bridge_outputs
        and pipeline._provenance_terminal_frame_enabled
        and not bool(getattr(world, "phase_c_defer_cell_publish", False))
    )
    provenance_cleanup_terminal = bool(
        provenance_terminal
        and pipeline._provenance_cleanup_terminal_fusion_frame_enabled
    )
    cleanup_flow_fusion = bool(
        publish_bridge_outputs and pipeline._cleanup_flow_fusion_frame_enabled
    )
    active_decay_fusion = bool(
        publish_bridge_outputs and pipeline._flow_active_decay_fusion_frame_enabled
    )
    direct_flow_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and not publish_bridge_outputs
        and pipeline._flow_intent_bridge_residency_enabled
        and {
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
        }.issubset(world.bridge.gpu_authoritative_resources)
    )
    direct_bridge_aux_inputs = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._bridge_aux_residency_enabled
        and {"entity_id", "placeholder_displaced_material"}.issubset(
            world.bridge.gpu_authoritative_resources
        )
    )
    if provenance_terminal:
        # The terminal gather must see authoritative entity/displaced halo
        # values even when the broad aux-residency candidate is disabled.
        program_name = (
            "liquid_flow_intent_shared_halo_provenance_cleanup_bridge_aux"
            if provenance_cleanup_terminal
            else "liquid_flow_intent_shared_halo_provenance_shared_meta_lazy_aux"
            if (
                pipeline._flow_intent_provenance_shared_meta_cache_enabled
                and pipeline._flow_intent_provenance_lazy_aux_enabled
            )
            else "liquid_flow_intent_shared_halo_provenance_shared_meta"
            if pipeline._flow_intent_provenance_shared_meta_cache_enabled
            else "liquid_flow_intent_shared_halo_provenance_mask_cache"
            if pipeline._flow_intent_active_mask_cache_enabled
            else "liquid_flow_intent_shared_halo_provenance_bridge_aux"
        )
    elif cleanup_flow_fusion:
        program_name = "liquid_flow_intent_shared_halo_cleanup"
    elif active_decay_fusion:
        program_name = "liquid_flow_intent_shared_halo_active_decay"
    elif pipeline._flow_intent_shared_halo_enabled:
        if direct_flow_inputs:
            program_name = "liquid_flow_intent_shared_halo_resident"
        elif direct_bridge_aux_inputs:
            program_name = "liquid_flow_intent_shared_halo_bridge_aux"
        else:
            program_name = "liquid_flow_intent_shared_halo"
    else:
        program_name = (
            "liquid_flow_intent_resident"
            if direct_flow_inputs
            else "liquid_flow_intent_bridge_aux"
            if direct_bridge_aux_inputs
            else "liquid_flow_intent"
        )
    program = pipeline.programs[program_name]
    ctx = world.bridge.ctx
    assert ctx is not None
    target_texture = resources.liquid_flow_intent
    if pipeline._formal_gpu_frame(world):
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
    program["publish_bridge_outputs"].value = bool(publish_bridge_outputs)
    program["publish_bridge_aux_outputs"].value = bool(
        publish_bridge_outputs
        and not (
            pipeline._formal_gpu_frame(world)
            and pipeline._bridge_aux_cleanup_fusion_enabled
        )
    )
    program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
    program["resident_hydrate_inputs"].value = bool(direct_flow_inputs)
    split_placeholder_roles = bool(
        publish_bridge_outputs and pipeline._placeholder_lazy_roles_enabled
    )
    program["use_split_placeholder_roles"].value = split_placeholder_roles
    seam_y_shared_snapshot_roles = bool(
        pipeline._formal_gpu_frame(world)
        and pipeline._seam_y_shared_snapshot_enabled
        and pipeline._buoyancy_pass_fusion_enabled
    )
    pipeline._set_uniform_if_present(
        program,
        "buoyancy_fused_roles",
        bool(
            pipeline._buoyancy_pass_fusion_enabled
            and not seam_y_shared_snapshot_roles
        ),
    )
    if provenance_terminal:
        program["provenance_full_grid"].value = True
    resources.cell_state_in.use(location=0)
    resources.cell_state_out.use(location=1)
    resources.velocity_in.use(location=2)
    resources.active_tile_tex.use(location=3)
    resources.entity_in.use(location=4)
    resources.displaced_in.use(location=5)
    if cleanup_flow_fusion or provenance_cleanup_terminal:
        resources.displaced_out.use(location=6)
    if cleanup_flow_fusion:
        resources.entity_in.bind_to_image(4, read=False, write=True)
        resources.island_in.bind_to_image(6, read=False, write=True)
    resources.timer_in.use(location=7)
    resources.temp_in.use(location=8)
    resources.integrity_in.use(location=9)
    resources.island_out.use(location=10)
    resources.entity_out.use(location=11)
    resources.velocity_out.use(location=12)
    resources.timer_out.use(location=13)
    resources.temp_out.use(location=14)
    resources.integrity_out.use(location=15)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.active_tile_count.bind_to_storage_buffer(binding=1)
    resources.active_tile_list.bind_to_storage_buffer(binding=2)
    resources.affected_tile_flags.bind_to_storage_buffer(binding=7)
    if active_decay_fusion:
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=11)
    target_texture.bind_to_image(0, read=False, write=True)
    if direct_flow_inputs:
        resources.cell_state_in.bind_to_image(2, read=False, write=True)
        resources.velocity_in.bind_to_image(3, read=False, write=True)
        resources.entity_in.bind_to_image(4, read=False, write=True)
        resources.displaced_in.bind_to_image(5, read=False, write=True)
    if publish_bridge_outputs:
        bridge = world.bridge
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=3)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=4)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=5)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
        bridge.textures["material"].bind_to_image(1, read=False, write=True)
    elif direct_flow_inputs:
        bridge = world.bridge
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=3)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=5)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
    elif direct_bridge_aux_inputs:
        bridge = world.bridge
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=4)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=5)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=6)
    if provenance_terminal:
        bridge.ensure_cell_core_spare(world)
        if bridge.cell_core_spare is None:
            raise RuntimeError("liquid provenance terminal requires a cell-core spare")
        resources.provenance_out.bind_to_storage_buffer(binding=8)
        resources.provenance_in.bind_to_storage_buffer(binding=9)
        bridge.cell_core_spare.bind_to_storage_buffer(binding=10)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
    else:
        pipeline._run_active_tile_indirect(program, resources, "flow intent")
    pipeline._sync_compute_writes(ctx)
    if pipeline._formal_gpu_frame(world):
        world.bridge.mark_gpu_authoritative("liquid_flow_intent")
        if publish_bridge_outputs:
            world.bridge.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
            )
        if provenance_terminal:
            live = bridge.buffers.get("cell_core")
            spare = bridge.cell_core_spare
            if live is None or spare is None or live is spare:
                raise RuntimeError("liquid provenance terminal did not produce a distinct core spare")
            bridge.buffers["cell_core"], bridge.cell_core_spare = spare, live
            pipeline.last_provenance_terminal_used = True
        if provenance_cleanup_terminal:
            pipeline.last_provenance_cleanup_terminal_fusion_used = True
        if cleanup_flow_fusion:
            pipeline.last_cleanup_flow_fusion_used = True
        if active_decay_fusion:
            world.bridge.mark_gpu_authoritative("active_tile_ttl")
            pipeline.last_flow_active_decay_fusion_used = True
            pipeline._flow_active_decay_fusion_frame_id = int(
                getattr(world, "frame_id", 0)
            )



def _download_outputs(
    pipeline,
    world: "WorldEngine",
    resources: GPULiquidResources,
    *,
    use_in: bool = False,
    velocity_use_out: bool = False,
) -> None:
    cell_state = resources.cell_state_in if use_in else resources.cell_state_out
    timer = resources.timer_in if use_in else resources.timer_out
    temp = resources.temp_in if use_in else resources.temp_out
    integrity = resources.integrity_in if use_in else resources.integrity_out
    velocity = resources.velocity_out if velocity_use_out else (resources.velocity_in if use_in else resources.velocity_out)
    displaced = resources.displaced_in if use_in else resources.displaced_out
    packed_cell_state = np.frombuffer(cell_state.read(), dtype=np.uint32).reshape(
        (world.height, world.width)
    )
    world.material_id[:] = (packed_cell_state & 0xFFFF).astype(np.int32)
    world.phase[:] = ((packed_cell_state >> 16) & 0xFF).astype(np.uint8)
    world.cell_flags[:] = ((packed_cell_state >> 24) & 0xFF).astype(np.uint8)
    world.timer_pack[:] = _unpack_timer_texture(
        np.frombuffer(timer.read(), dtype=np.uint32).reshape((world.height, world.width))
    )
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



def _barrier_bits(pipeline) -> tuple[str, ...]:
    # liquid touches the framebuffer (visible illumination publish) and
    # uses indirect/buffer-update paths alongside the default sync.
    return (
        "SHADER_IMAGE_ACCESS_BARRIER_BIT",
        "TEXTURE_FETCH_BARRIER_BIT",
        "FRAMEBUFFER_BARRIER_BIT",
        "SHADER_STORAGE_BARRIER_BIT",
        "COMMAND_BARRIER_BIT",
        "BUFFER_UPDATE_BARRIER_BIT",
    )
