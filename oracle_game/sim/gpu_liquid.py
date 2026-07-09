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

    def step(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
        post_tile_mask: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.reset_pass_profile()
        with self._profile_pass(world, "liquid_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
        with self._profile_pass(world, "liquid_load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources)
        with self._profile_pass(world, "liquid_compact_active_tiles"):
            self._compact_active_tiles(world, resources)
        with self._profile_pass(world, "liquid_tile_solve"):
            self._run_tile_solve(world, resources)
        if self._active_scheduler_gpu_authoritative(world):
            with self._profile_pass(world, "liquid_load_active_mask"):
                self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        else:
            with self._profile_pass(world, "liquid_upload_active_mask"):
                self._upload_active_tile_mask(resources, post_tile_mask)
        with self._profile_pass(world, "liquid_compact_active_cell_tiles"):
            self._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
            )
        formal_gpu_frame = self._formal_gpu_frame(world)
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_copy_tile_solve"):
                self._run_copy_core_state(
                    world,
                    resources,
                    (
                        resources.material_out,
                        resources.phase_out,
                        resources.flags_out,
                        resources.timer_out,
                        resources.temp_out,
                        resources.integrity_out,
                        resources.velocity_out,
                    ),
                    (
                        resources.material_in,
                        resources.phase_in,
                        resources.flags_in,
                        resources.timer_in,
                        resources.temp_in,
                        resources.integrity_in,
                        resources.velocity_in,
                    ),
                )
            with self._profile_pass(world, "liquid_build_seam_x_boundaries"):
                self._build_seam_boundary_dispatch(world, resources, axis="x")
            with self._profile_pass(world, "liquid_prefetch_seam_x_boundaries"):
                self._prefetch_seam_boundary_bridge_inputs(world, resources, axis="x")
        with self._profile_pass(world, "liquid_seam_x"):
            self._run_seam_pass(
                "seam_x",
                world,
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                resources.active_tile_tex,
                boundary_dispatch=formal_gpu_frame,
            )
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_reload_seam_x_active_tiles"):
                self._reload_and_compact_active_cell_tiles(world, resources)
            with self._profile_pass(world, "liquid_copy_seam_x"):
                self._run_copy_core_state(
                    world,
                    resources,
                    (
                        resources.material_in,
                        resources.phase_in,
                        resources.flags_in,
                        resources.timer_in,
                        resources.temp_in,
                        resources.integrity_in,
                        resources.velocity_in,
                    ),
                    (
                        resources.material_out,
                        resources.phase_out,
                        resources.flags_out,
                        resources.timer_out,
                        resources.temp_out,
                        resources.integrity_out,
                        resources.velocity_out,
                    ),
                )
            with self._profile_pass(world, "liquid_build_seam_y_boundaries"):
                self._build_seam_boundary_dispatch(world, resources, axis="y")
            with self._profile_pass(world, "liquid_prefetch_seam_y_boundaries"):
                self._prefetch_seam_boundary_bridge_inputs(world, resources, axis="y")
        with self._profile_pass(world, "liquid_seam_y"):
            self._run_seam_pass(
                "seam_y",
                world,
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                resources.active_tile_tex,
                boundary_dispatch=formal_gpu_frame,
            )
        if formal_gpu_frame:
            with self._profile_pass(world, "liquid_reload_seam_y_active_tiles"):
                self._reload_and_compact_active_cell_tiles(world, resources)
        with self._profile_pass(world, "liquid_buoyancy_sink"):
            self._run_buoyancy_pass(
                "buoyancy_sink",
                world,
                resources,
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
            )
        with self._profile_pass(world, "liquid_buoyancy_float"):
            self._run_buoyancy_pass(
                "buoyancy_float",
                world,
                resources,
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
            )
        with self._profile_pass(world, "liquid_copy_for_placeholder"):
            self._run_copy_for_placeholder(
                world,
                resources,
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                resources.displaced_in,
                resources.displaced_out,
            )
        with self._profile_pass(world, "liquid_placeholder_displacement"):
            self._run_placeholder_displacement(
                world,
                resources,
                (
                    resources.material_out,
                    resources.phase_out,
                    resources.flags_out,
                    resources.timer_out,
                    resources.temp_out,
                    resources.integrity_out,
                    resources.velocity_out,
                ),
                (
                    resources.material_in,
                    resources.phase_in,
                    resources.flags_in,
                    resources.timer_in,
                    resources.temp_in,
                    resources.integrity_in,
                    resources.velocity_in,
                ),
                resources.displaced_in,
                resources.displaced_out,
            )
        with self._profile_pass(world, "liquid_cleanup_runtime"):
            self._run_cleanup_runtime(world, resources)
        if self._active_scheduler_gpu_authoritative(world):
            with self._profile_pass(world, "liquid_reload_flow_active_mask"):
                self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
            with self._profile_pass(world, "liquid_compact_flow_active_tiles"):
                self._compact_active_tiles(
                    world,
                    resources,
                    workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
                )
        with self._profile_pass(world, "liquid_flow_intent"):
            self._run_liquid_intent_pass(world, resources)
        if self._formal_gpu_frame(world):
            with self._profile_pass(world, "liquid_refresh_active_scheduler"):
                self._refresh_active_scheduler_from_ttl(world)
        with self._profile_pass(world, "liquid_publish_bridge"):
            self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources, use_in=True)

    def prepare_motion_flow_intent(
        self,
        world: "WorldEngine",
        *,
        solve_tile_mask: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU liquid pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.reset_pass_profile()
        with self._profile_pass(world, "liquid_pre_motion_upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask=solve_tile_mask)
        with self._profile_pass(world, "liquid_pre_motion_load_bridge_inputs"):
            self._load_authoritative_bridge_flow_intent_inputs(world, resources)
        with self._profile_pass(world, "liquid_pre_motion_compact_active_tiles"):
            self._compact_active_tiles(
                world,
                resources,
                workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
            )
        with self._profile_pass(world, "liquid_pre_motion_flow_intent"):
            self._run_liquid_intent_pass(world, resources)
        self.last_cpu_mirror_downloaded = False

    def release(self) -> None:
        if self.resources is None:
            return
        try:
            self.resources.bridge_cell_copy_framebuffer.release()
        except Exception:
            pass
        for resource in (
            self.resources.material_pre,
            self.resources.material_in,
            self.resources.material_out,
            self.resources.phase_pre,
            self.resources.phase_in,
            self.resources.phase_out,
            self.resources.island_in,
            self.resources.island_out,
            self.resources.entity_in,
            self.resources.entity_out,
            self.resources.flags_in,
            self.resources.flags_out,
            self.resources.timer_in,
            self.resources.timer_out,
            self.resources.temp_in,
            self.resources.temp_out,
            self.resources.integrity_in,
            self.resources.integrity_out,
            self.resources.velocity_in,
            self.resources.velocity_out,
            self.resources.liquid_flow_intent,
            self.resources.active_tile_tex,
            self.resources.active_tile_list,
            self.resources.active_tile_count,
            self.resources.active_tile_dispatch_args,
            self.resources.affected_tile_list,
            self.resources.affected_tile_count,
            self.resources.affected_tile_dispatch_args,
            self.resources.affected_tile_prefetch_dispatch_args,
            self.resources.affected_tile_flags,
            self.resources.placeholder_target_claims,
            self.resources.displaced_in,
            self.resources.displaced_out,
            self.resources.material_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPULiquidResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.active.tile_width, world.active.tile_height)
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        material_pre = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_pre = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        flags_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        flags_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        timer_in = ctx.texture((world.width, world.height), 4, dtype="f4")
        timer_out = ctx.texture((world.width, world.height), 4, dtype="f4")
        temp_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        temp_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        velocity_in = ctx.texture((world.width, world.height), 2, dtype="f4")
        velocity_out = ctx.texture((world.width, world.height), 2, dtype="f4")
        liquid_flow_intent = ctx.texture((world.width, world.height), 2, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        tile_count = max(1, int(world.active.tile_width * world.active.tile_height))
        active_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        active_tile_count = ctx.buffer(reserve=4, dynamic=True)
        active_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_list = ctx.buffer(reserve=max(8, tile_count * 2 * 4), dynamic=True)
        affected_tile_count = ctx.buffer(reserve=4, dynamic=True)
        affected_tile_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_prefetch_dispatch_args = ctx.buffer(reserve=3 * 4, dynamic=True)
        affected_tile_flags = ctx.buffer(reserve=max(4, tile_count * 4), dynamic=True)
        affected_tile_flags.write(np.zeros((tile_count,), dtype=np.uint32).tobytes())
        cell_count = max(1, int(world.width * world.height))
        placeholder_target_claims = ctx.buffer(reserve=cell_count * 4, dynamic=True)
        placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
        displaced_in = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_out = ctx.texture((world.width, world.height), 1, dtype="f4")
        for texture in (
            material_pre,
            material_in,
            material_out,
            phase_pre,
            phase_in,
            phase_out,
            island_in,
            island_out,
            entity_in,
            entity_out,
            flags_in,
            flags_out,
            timer_in,
            timer_out,
            temp_in,
            temp_out,
            integrity_in,
            integrity_out,
            velocity_in,
            velocity_out,
            liquid_flow_intent,
            active_tile_tex,
            displaced_in,
            displaced_out,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        bridge_cell_copy_framebuffer = ctx.framebuffer(
            color_attachments=[
                material_pre,
                material_out,
                phase_pre,
                phase_out,
                flags_out,
                timer_out,
                temp_out,
                integrity_out,
            ]
        )
        self.resources = GPULiquidResources(
            signature=signature,
            material_pre=material_pre,
            material_in=material_in,
            material_out=material_out,
            phase_pre=phase_pre,
            phase_in=phase_in,
            phase_out=phase_out,
            island_in=island_in,
            island_out=island_out,
            entity_in=entity_in,
            entity_out=entity_out,
            flags_in=flags_in,
            flags_out=flags_out,
            timer_in=timer_in,
            timer_out=timer_out,
            temp_in=temp_in,
            temp_out=temp_out,
            integrity_in=integrity_in,
            integrity_out=integrity_out,
            velocity_in=velocity_in,
            velocity_out=velocity_out,
            liquid_flow_intent=liquid_flow_intent,
            active_tile_tex=active_tile_tex,
            active_tile_list=active_tile_list,
            active_tile_count=active_tile_count,
            active_tile_dispatch_args=active_tile_dispatch_args,
            affected_tile_list=affected_tile_list,
            affected_tile_count=affected_tile_count,
            affected_tile_dispatch_args=affected_tile_dispatch_args,
            affected_tile_prefetch_dispatch_args=affected_tile_prefetch_dispatch_args,
            affected_tile_flags=affected_tile_flags,
            placeholder_target_claims=placeholder_target_claims,
            displaced_in=displaced_in,
            displaced_out=displaced_out,
            bridge_cell_copy_framebuffer=bridge_cell_copy_framebuffer,
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        )
        return self.resources

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

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        solve_tile_mask: np.ndarray,
    ) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
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
        upload_active_from_cpu = not self._active_scheduler_gpu_authoritative(world)
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
        self.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
        self.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_cell_state_from_cpu:
            resources.material_pre.write(world.material_id.astype("f4").tobytes())
            resources.material_in.write(world.material_id.astype("f4").tobytes())
            resources.material_out.write(world.material_id.astype("f4").tobytes())
            resources.phase_pre.write(world.phase.astype("f4").tobytes())
            resources.phase_in.write(world.phase.astype("f4").tobytes())
            resources.phase_out.write(world.phase.astype("f4").tobytes())
            resources.flags_in.write(world.cell_flags.astype("f4").tobytes())
            resources.flags_out.write(world.cell_flags.astype("f4").tobytes())
            resources.timer_in.write(world.timer_pack.astype("f4").tobytes())
            resources.timer_out.write(world.timer_pack.astype("f4").tobytes())
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
            self._upload_active_tile_mask(resources, solve_tile_mask)
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
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

    # ``_formal_gpu_frame`` inherited from GPUPipelineBase.

    def _active_scheduler_gpu_authoritative(self, world: "WorldEngine") -> bool:
        authoritative = world.bridge.gpu_authoritative_resources
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in authoritative
            and "active_chunk_mask" in authoritative
        )

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative input state")
        active_tile_indirect = self._formal_gpu_frame(world)
        group_x = (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        group_y = (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE
        if active_tile_indirect:
            with self._profile_pass(world, "liquid_load_bridge_inputs.active_tile_compact"):
                self._compact_active_tiles(
                    world,
                    resources,
                    workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
                )
        ran_copy = False
        if copy_cell_core:
            with self._profile_pass(world, "liquid_load_bridge_inputs.load_cell_in"):
                program = self.programs["load_bridge_cell"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["copy_cell_core"].value = bool(copy_cell_core)
                program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                resources.material_pre.bind_to_image(0, read=False, write=True)
                resources.material_in.bind_to_image(1, read=False, write=True)
                resources.phase_pre.bind_to_image(2, read=False, write=True)
                resources.phase_in.bind_to_image(3, read=False, write=True)
                resources.flags_in.bind_to_image(4, read=False, write=True)
                resources.timer_in.bind_to_image(5, read=False, write=True)
                resources.temp_in.bind_to_image(6, read=False, write=True)
                resources.integrity_in.bind_to_image(7, read=False, write=True)
                if active_tile_indirect:
                    self._run_active_tile_indirect(program, resources, "bridge cell input load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

            with self._profile_pass(world, "liquid_load_bridge_inputs.load_cell_out"):
                program = self.programs["load_bridge_cell_out"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
                bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
                resources.active_tile_count.bind_to_storage_buffer(binding=4)
                resources.active_tile_list.bind_to_storage_buffer(binding=5)
                resources.material_out.bind_to_image(0, read=False, write=True)
                resources.phase_out.bind_to_image(1, read=False, write=True)
                resources.flags_out.bind_to_image(2, read=False, write=True)
                resources.timer_out.bind_to_image(3, read=False, write=True)
                resources.temp_out.bind_to_image(4, read=False, write=True)
                resources.integrity_out.bind_to_image(5, read=False, write=True)
                resources.velocity_out.bind_to_image(6, read=False, write=True)
                if active_tile_indirect:
                    self._run_active_tile_indirect(program, resources, "bridge cell output load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
            with self._profile_pass(world, "liquid_load_bridge_inputs.load_aux"):
                program = self.programs["load_bridge_cell_aux"]
                program["cell_grid_size"].value = (world.width, world.height)
                program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
                program["tile_size"].value = int(world.active.tile_size)
                program["copy_cell_core"].value = bool(copy_cell_core)
                program["copy_island_id"].value = bool(copy_island_id)
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
                    self._run_active_tile_indirect(program, resources, "bridge aux input load")
                else:
                    program.run(group_x, group_y, 1)
                ran_copy = True

        if ran_copy:
            with self._profile_pass(world, "liquid_load_bridge_inputs.sync"):
                self._sync_compute_writes(bridge.ctx)

    def _load_authoritative_bridge_flow_intent_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        if not (copy_cell_core or copy_entity_id or copy_displaced):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for flow-intent input state")
        self._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
        )
        program = self.programs["load_bridge_flow_intent_inputs"]
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
        resources.material_in.bind_to_image(0, read=False, write=True)
        resources.phase_in.bind_to_image(1, read=False, write=True)
        resources.velocity_in.bind_to_image(2, read=False, write=True)
        resources.entity_in.bind_to_image(3, read=False, write=True)
        resources.displaced_in.bind_to_image(4, read=False, write=True)
        self._run_active_tile_indirect(program, resources, "bridge flow-intent input load")
        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        use_out: bool = False,
        velocity_use_out: bool = False,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for authoritative output state")
            return
        program = self.programs["publish_bridge_cell"]
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
        material_tex = resources.material_out if use_out else resources.material_in
        phase_tex = resources.phase_out if use_out else resources.phase_in
        flags_tex = resources.flags_out if use_out else resources.flags_in
        timer_tex = resources.timer_out if use_out else resources.timer_in
        temp_tex = resources.temp_out if use_out else resources.temp_in
        integrity_tex = resources.integrity_out if use_out else resources.integrity_in
        velocity_tex = resources.velocity_out if velocity_use_out else resources.velocity_in
        material_tex.use(location=0)
        phase_tex.use(location=1)
        flags_tex.use(location=2)
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
            self._run_active_tile_indirect(program, resources, "bridge cell publish")
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    def _refresh_active_scheduler_from_ttl(self, world: "WorldEngine") -> None:
        bridge = world.bridge
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge GPU resources for active refresh")
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
        bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _upload_active_tile_mask(self, resources: GPULiquidResources, tile_mask: np.ndarray) -> None:
        resources.active_tile_tex.write(np.asarray(tile_mask, dtype="f4").tobytes())

    def _load_authoritative_active_tile_mask(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid pipeline requires bridge active scheduler resources")
        program = self.programs["load_active_tiles"]
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["expansion_radius"].value = int(expansion_radius)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        resources.active_tile_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.active.tile_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.active.tile_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)

    def _active_tile_workgroups_per_tile(self, world: "WorldEngine") -> int:
        axis = max(1, (int(world.active.tile_size) + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        return axis * axis

    def _next_placeholder_claim_epoch(self, resources: GPULiquidResources, world: "WorldEngine") -> int:
        self._placeholder_claim_epoch += 1
        if self._placeholder_claim_epoch >= 0x7FFFFFFF:
            cell_count = max(1, int(world.width * world.height))
            resources.placeholder_target_claims.write(np.zeros((cell_count,), dtype=np.uint32).tobytes())
            self._placeholder_claim_epoch = 1
        return self._placeholder_claim_epoch

    def _seam_workgroups_per_boundary(self, axis: str) -> int:
        if axis == "x":
            groups_x = max(1, (TILE_SIZE * 2 + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
            groups_y = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
        elif axis == "y":
            groups_x = max(1, (TILE_SIZE + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE)
            groups_y = 1
        else:
            raise ValueError(f"unknown liquid seam axis: {axis}")
        return groups_x * groups_y

    def _reload_and_compact_active_cell_tiles(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
    ) -> None:
        self._load_authoritative_active_tile_mask(world, resources, expansion_radius=0)
        self._compact_active_tiles(
            world,
            resources,
            workgroups_per_tile=self._active_tile_workgroups_per_tile(world),
        )

    def _build_seam_boundary_dispatch(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        axis: str,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        program = self.programs[f"compact_seam_{axis}_boundaries_from_active_tiles"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU liquid seam {axis} boundary compaction requires ModernGL ComputeShader.run_indirect")
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["source_workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        program["workgroups_per_boundary"].value = int(self._seam_workgroups_per_boundary(axis))
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
        program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _prefetch_seam_boundary_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        axis: str,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        if axis not in ("x", "y"):
            raise ValueError(f"unknown liquid seam axis: {axis}")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU liquid seam prefetch requires bridge GPU resources")
        world._require_gpu_authoritative_resources(
            "liquid seam boundary prefetch",
            "cell_core",
            "entity_id",
            "placeholder_displaced_material",
            "active_tile_ttl",
        )
        program = self.programs["prefetch_seam_boundary_bridge_inputs"]
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid seam prefetch requires ModernGL ComputeShader.run_indirect")
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["seam_axis"].value = 0 if axis == "x" else 1
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=1)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=2)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=3)
        resources.affected_tile_count.bind_to_storage_buffer(binding=4)
        resources.affected_tile_list.bind_to_storage_buffer(binding=5)
        if axis == "x":
            material_tex = resources.material_out
            phase_tex = resources.phase_out
            flags_tex = resources.flags_out
            timer_tex = resources.timer_out
            temp_tex = resources.temp_out
            integrity_tex = resources.integrity_out
            velocity_tex = resources.velocity_out
        else:
            material_tex = resources.material_in
            phase_tex = resources.phase_in
            flags_tex = resources.flags_in
            timer_tex = resources.timer_in
            temp_tex = resources.temp_in
            integrity_tex = resources.integrity_in
            velocity_tex = resources.velocity_in
        material_tex.bind_to_image(0, read=False, write=True)
        phase_tex.bind_to_image(1, read=False, write=True)
        flags_tex.bind_to_image(2, read=False, write=True)
        timer_tex.bind_to_image(3, read=False, write=True)
        temp_tex.bind_to_image(4, read=False, write=True)
        integrity_tex.bind_to_image(5, read=False, write=True)
        velocity_tex.bind_to_image(6, read=False, write=True)
        program.run_indirect(resources.affected_tile_dispatch_args)
        self._sync_compute_writes(bridge.ctx)

        aux_program = self.programs["prefetch_seam_boundary_bridge_aux_inputs"]
        if not hasattr(aux_program, "run_indirect"):
            raise RuntimeError("GPU liquid seam aux prefetch requires ModernGL ComputeShader.run_indirect")
        aux_program["cell_grid_size"].value = (world.width, world.height)
        aux_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        aux_program["tile_size"].value = int(world.active.tile_size)
        aux_program["seam_axis"].value = 0 if axis == "x" else 1
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=0)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=1)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=2)
        resources.affected_tile_count.bind_to_storage_buffer(binding=3)
        resources.affected_tile_list.bind_to_storage_buffer(binding=4)
        resources.entity_in.bind_to_image(0, read=False, write=True)
        resources.displaced_in.bind_to_image(1, read=False, write=True)
        aux_program.run_indirect(resources.affected_tile_dispatch_args)
        self._sync_compute_writes(bridge.ctx)

    def _build_placeholder_dirty_affected_tile_dispatch(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        material_tex: Any,
        displaced_tex: Any,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.affected_tile_count.bind_to_storage_buffer(binding=0)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        dirty_rect_count = int(len(world.bridge_frame_placeholder_dirty_rects))
        if dirty_rect_count > 0:
            world.bridge.ensure_world_resources(world)
            if "placeholder_dirty_rect" not in world.bridge.buffers:
                raise RuntimeError("GPU liquid placeholder displacement requires placeholder dirty rect buffer")
            program = self.programs["compact_placeholder_dirty_affected_tiles"]
            program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            program["tile_size"].value = int(world.active.tile_size)
            program["dirty_rect_count"].value = dirty_rect_count
            program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
            resources.affected_tile_count.bind_to_storage_buffer(binding=0)
            resources.affected_tile_list.bind_to_storage_buffer(binding=1)
            resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            resources.affected_tile_flags.bind_to_storage_buffer(binding=3)
            world.bridge.buffers["placeholder_dirty_rect"].bind_to_storage_buffer(binding=4)
            program.run((dirty_rect_count + 63) // 64, 1, 1)
            self._sync_compute_writes(ctx)

        pending_program = self.programs["compact_placeholder_active_pending_affected_tiles"]
        if not hasattr(pending_program, "run_indirect"):
            raise RuntimeError("GPU liquid placeholder pending compaction requires ModernGL ComputeShader.run_indirect")
        material_table = world.bridge.shadow_typed_tables["material_table"]
        pending_program["cell_grid_size"].value = (world.width, world.height)
        pending_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        pending_program["tile_size"].value = int(world.active.tile_size)
        pending_program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
        pending_program["source_workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        pending_program["workgroups_per_tile"].value = int(self._active_tile_workgroups_per_tile(world))
        material_tex.use(location=0)
        displaced_tex.use(location=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_dispatch_args.bind_to_storage_buffer(binding=4)
        resources.affected_tile_flags.bind_to_storage_buffer(binding=5)
        pending_program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _compact_active_tiles(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        workgroups_per_tile: int = 1,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        clear_program = self.programs["clear_active_tile_dispatch"]
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=1)
        clear_program.run(1, 1, 1)
        self._sync_compute_writes(ctx)

        compact_workgroups_per_tile = int(max(1, workgroups_per_tile))
        if self._active_scheduler_gpu_authoritative(world):
            bridge = world.bridge
            bridge._ensure_active_scheduler_programs()
            bridge._refresh_active_chunks_and_meta(world, read_meta=False)
            compact_program = self.programs["compact_active_tiles_from_chunks"]
            if not hasattr(compact_program, "run_indirect"):
                raise RuntimeError("GPU liquid active chunk compaction requires ModernGL ComputeShader.run_indirect")
            compact_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
            compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
            compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
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
            compact_program["workgroups_per_tile"].value = compact_workgroups_per_tile
            resources.active_tile_tex.use(location=0)
            resources.active_tile_count.bind_to_storage_buffer(binding=0)
            resources.active_tile_list.bind_to_storage_buffer(binding=1)
            resources.active_tile_dispatch_args.bind_to_storage_buffer(binding=2)
            compact_program.run((tile_count + 255) // 256, 1, 1)
        self._sync_compute_writes(ctx)

    def _run_active_tile_indirect(self, program: Any, resources: GPULiquidResources, pass_name: str) -> None:
        if not hasattr(program, "run_indirect"):
            raise RuntimeError(f"GPU liquid {pass_name} requires ModernGL ComputeShader.run_indirect")
        program.run_indirect(resources.active_tile_dispatch_args)

    def _run_tile_solve(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["tile_solve"]
        ctx = world.bridge.ctx
        assert ctx is not None
        if not hasattr(program, "run_indirect"):
            raise RuntimeError("GPU liquid active tile solve requires ModernGL ComputeShader.run_indirect")
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        resources.material_in.use(location=0)
        resources.phase_in.use(location=1)
        resources.flags_in.use(location=2)
        resources.timer_in.use(location=3)
        resources.temp_in.use(location=4)
        resources.integrity_in.use(location=5)
        resources.velocity_in.use(location=6)
        resources.entity_in.use(location=8)
        resources.displaced_in.use(location=9)
        resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
        resources.material_out.bind_to_image(0, read=False, write=True)
        resources.phase_out.bind_to_image(1, read=False, write=True)
        resources.flags_out.bind_to_image(2, read=False, write=True)
        resources.timer_out.bind_to_image(3, read=False, write=True)
        resources.temp_out.bind_to_image(4, read=False, write=True)
        resources.integrity_out.bind_to_image(5, read=False, write=True)
        resources.velocity_out.bind_to_image(6, read=False, write=True)
        program.run_indirect(resources.active_tile_dispatch_args)
        self._sync_compute_writes(ctx)

    def _run_seam_pass(
        self,
        program_name: str,
        world: "WorldEngine",
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        active_tile_tex: Any,
        *,
        boundary_dispatch: bool = False,
    ) -> None:
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["use_boundary_dispatch"].value = bool(boundary_dispatch)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        active_tile_tex.use(location=7)
        self.resources.entity_in.use(location=8)
        self.resources.displaced_in.use(location=9)
        self.resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources = self.resources
        assert resources is not None
        resources.affected_tile_count.bind_to_storage_buffer(binding=2)
        resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        if boundary_dispatch:
            if not hasattr(program, "run_indirect"):
                raise RuntimeError(f"GPU liquid {program_name} requires ModernGL ComputeShader.run_indirect")
            program.run_indirect(resources.affected_tile_dispatch_args)
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(ctx)

    def _run_buoyancy_pass(
        self,
        program_name: str,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> None:
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_powder"].value = int(Phase.POWDER)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        resources.active_tile_tex.use(location=7)
        resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        resources.active_tile_count.bind_to_storage_buffer(binding=2)
        resources.active_tile_list.bind_to_storage_buffer(binding=3)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        self._run_active_tile_indirect(program, resources, program_name)
        self._sync_compute_writes(ctx)

    def _run_copy_core_state(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> None:
        program = self.programs["copy_core_state"]
        ctx = world.bridge.ctx
        assert ctx is not None
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "copy core state")
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(ctx)

    def _run_copy_for_placeholder(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        displaced_in: Any,
        displaced_out: Any,
    ) -> None:
        program = self.programs["copy_with_pending"]
        ctx = world.bridge.ctx
        assert ctx is not None
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        displaced_in.use(location=7)
        write_resources[0].bind_to_image(0, read=False, write=True)
        write_resources[1].bind_to_image(1, read=False, write=True)
        write_resources[2].bind_to_image(2, read=False, write=True)
        write_resources[3].bind_to_image(3, read=False, write=True)
        write_resources[4].bind_to_image(4, read=False, write=True)
        write_resources[5].bind_to_image(5, read=False, write=True)
        write_resources[6].bind_to_image(6, read=False, write=True)
        displaced_out.bind_to_image(7, read=False, write=True)
        resources.active_tile_count.bind_to_storage_buffer(binding=0)
        resources.active_tile_list.bind_to_storage_buffer(binding=1)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "copy with pending")
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(ctx)

    def _run_placeholder_displacement(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        read_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        write_resources: tuple[Any, Any, Any, Any, Any, Any, Any],
        displaced_in: Any,
        displaced_out: Any,
    ) -> None:
        program = self.programs["placeholder_displace"]
        ctx = world.bridge.ctx
        assert ctx is not None
        dirty_affected_dispatch = self._formal_gpu_frame(world)
        if dirty_affected_dispatch:
            self._build_placeholder_dirty_affected_tile_dispatch(
                world,
                resources,
                material_tex=read_resources[0],
                displaced_tex=displaced_in,
            )
        material_table = world.bridge.shadow_typed_tables["material_table"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = world.active.tile_size
        program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["placeholder_material_id"].value = typed_material_id(material_table, "placeholder_solid")
        program["placeholder_claim_epoch"].value = int(self._next_placeholder_claim_epoch(resources, world))
        read_resources[0].use(location=0)
        read_resources[1].use(location=1)
        read_resources[2].use(location=2)
        read_resources[3].use(location=3)
        read_resources[4].use(location=4)
        read_resources[5].use(location=5)
        read_resources[6].use(location=6)
        resources.active_tile_tex.use(location=7)
        displaced_in.use(location=8)
        resources.material_params.bind_to_storage_buffer(binding=0)
        world.bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=1)
        if dirty_affected_dispatch:
            resources.affected_tile_count.bind_to_storage_buffer(binding=2)
            resources.affected_tile_list.bind_to_storage_buffer(binding=3)
        else:
            resources.active_tile_count.bind_to_storage_buffer(binding=2)
            resources.active_tile_list.bind_to_storage_buffer(binding=3)
        resources.affected_tile_flags.bind_to_storage_buffer(binding=4)
        resources.placeholder_target_claims.bind_to_storage_buffer(binding=5)
        write_resources[0].bind_to_image(0)
        write_resources[1].bind_to_image(1)
        write_resources[2].bind_to_image(2)
        write_resources[3].bind_to_image(3)
        write_resources[4].bind_to_image(4)
        write_resources[5].bind_to_image(5)
        write_resources[6].bind_to_image(6)
        displaced_out.bind_to_image(7)
        if dirty_affected_dispatch:
            if not hasattr(program, "run_indirect"):
                raise RuntimeError("GPU liquid placeholder displacement requires ModernGL ComputeShader.run_indirect")
            program.run_indirect(resources.affected_tile_dispatch_args)
        else:
            self._run_active_tile_indirect(program, resources, "placeholder displacement")
        self._sync_compute_writes(ctx)

    def _run_cleanup_runtime(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["cleanup_runtime"]
        ctx = world.bridge.ctx
        assert ctx is not None
        active_tile_indirect = self._formal_gpu_frame(world)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["tile_size"].value = int(world.active.tile_size)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["use_active_tile_dispatch"].value = bool(active_tile_indirect)
        resources.material_pre.use(location=0)
        resources.phase_pre.use(location=1)
        resources.material_in.use(location=2)
        resources.phase_in.use(location=3)
        resources.island_in.use(location=4)
        resources.entity_in.use(location=5)
        resources.displaced_out.use(location=6)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        resources.island_out.bind_to_image(0, read=False, write=True)
        resources.entity_out.bind_to_image(1, read=False, write=True)
        resources.displaced_in.bind_to_image(2, read=False, write=True)
        if active_tile_indirect:
            self._run_active_tile_indirect(program, resources, "runtime cleanup")
        else:
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(ctx)

    def _run_liquid_intent_pass(self, world: "WorldEngine", resources: GPULiquidResources) -> None:
        program = self.programs["liquid_flow_intent"]
        ctx = world.bridge.ctx
        assert ctx is not None
        target_texture = resources.liquid_flow_intent
        if self._formal_gpu_frame(world):
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
        resources.material_in.use(location=0)
        resources.phase_in.use(location=1)
        resources.velocity_in.use(location=2)
        resources.active_tile_tex.use(location=3)
        resources.entity_in.use(location=4)
        resources.displaced_in.use(location=5)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.active_tile_count.bind_to_storage_buffer(binding=1)
        resources.active_tile_list.bind_to_storage_buffer(binding=2)
        target_texture.bind_to_image(0, read=False, write=True)
        self._run_active_tile_indirect(program, resources, "flow intent")
        self._sync_compute_writes(ctx)
        if self._formal_gpu_frame(world):
            world.bridge.mark_gpu_authoritative("liquid_flow_intent")

    def _download_outputs(
        self,
        world: "WorldEngine",
        resources: GPULiquidResources,
        *,
        use_in: bool = False,
        velocity_use_out: bool = False,
    ) -> None:
        material = resources.material_in if use_in else resources.material_out
        phase = resources.phase_in if use_in else resources.phase_out
        flags = resources.flags_in if use_in else resources.flags_out
        timer = resources.timer_in if use_in else resources.timer_out
        temp = resources.temp_in if use_in else resources.temp_out
        integrity = resources.integrity_in if use_in else resources.integrity_out
        velocity = resources.velocity_out if velocity_use_out else (resources.velocity_in if use_in else resources.velocity_out)
        displaced = resources.displaced_in if use_in else resources.displaced_out
        world.material_id[:] = np.rint(
            np.frombuffer(material.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(phase.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_flags[:] = np.rint(
            np.frombuffer(flags.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.timer_pack[:] = np.rint(
            np.frombuffer(timer.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
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

    def _barrier_bits(self) -> tuple[str, ...]:
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
