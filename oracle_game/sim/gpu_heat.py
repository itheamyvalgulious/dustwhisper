from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    _active_scheduler_gpu_authoritative,
    _ensure_material_flags_buffer,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
    mark_collapse_structure_dirty_tiles_from_bridge_cell_core,
)
from oracle_game.sim.shader_loader import build_compute_shader, shader_source
from oracle_game.types import Phase


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_GAS_SPECIES = 256
FREEZE_COLD_NEIGHBOR_THRESHOLD = 4

# Superset of every {{NAME}} marker referenced by any heat shader; the loader
# ignores unused keys, so one shared dict suffices for all passes.
_SHADER_SUBS = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_GAS_SPECIES": MAX_GAS_SPECIES,
    "MAX_MATERIALS_MINUS_ONE": MAX_MATERIALS - 1,
}


@dataclass(slots=True)
class GPUHeatResources:
    signature: tuple[int, int, int, int, int]
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
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    velocity_tex: Any
    velocity_out_tex: Any
    temp_ping: Any
    temp_pong: Any
    phase_target_tex: Any
    boil_target_tex: Any
    gas_tex: Any
    gas_out_tex: Any
    condense_target_tex: Any
    ambient_ping: Any
    ambient_pong: Any
    active_tile_tex: Any
    material_params: Any
    material_response_params: Any
    material_phase_params: Any
    gas_params: Any
    material_params_signature: tuple[int, int] | None = None
    gas_params_signature: tuple[int, int] | None = None


@dataclass(slots=True)
class GPUHeatStageTargets:
    phase_targets: np.ndarray
    boil_targets: np.ndarray
    condense_targets: np.ndarray

    @property
    def empty(self) -> bool:
        return (
            self.phase_targets.size == 0
            and self.boil_targets.size == 0
            and self.condense_targets.size == 0
        )

    @classmethod
    def empty_sentinel(cls) -> "GPUHeatStageTargets":
        return cls(
            phase_targets=np.zeros((0, 0), dtype=np.int32),
            boil_targets=np.zeros((0, 0), dtype=np.int32),
            condense_targets=np.zeros((0, 0, 0), dtype=np.bool_),
        )


class GPUHeatPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUHeatResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def step(
        self,
        world: "WorldEngine",
        dt: float,
        *,
        solve_tile_mask: np.ndarray,
        ambient_iterations: int,
    ) -> GPUHeatStageTargets:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU heat pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        with self._profile_pass(world, "upload_inputs"):
            self._upload_inputs(world, resources, solve_tile_mask)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
        with self._profile_pass(world, "ambient_diffuse"):
            self._run_ambient_diffuse(world, resources, gas_group_x, gas_group_y, iterations=ambient_iterations)
        with self._profile_pass(world, "cell_heat"):
            self._run_cell_heat(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "ambient_exchange"):
            self._run_ambient_exchange(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "ambient_feedback"):
            self._run_ambient_feedback(world, dt, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "phase_targets"):
            self._run_phase_targets(world, resources, group_x, group_y)
        with self._profile_pass(world, "boil_targets"):
            self._run_boil_targets(world, resources, group_x, group_y)
        with self._profile_pass(world, "condense_targets"):
            self._run_condense_targets(world, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "apply_cell_targets"):
            self._run_apply_cell_targets(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "apply_gas_targets"):
            self._run_apply_gas_targets(world, dt, resources, gas_group_x, gas_group_y)
        with self._profile_pass(world, "apply_condense_cells"):
            self._run_apply_condense_cells(world, resources, group_x, group_y)
        with self._profile_pass(world, "publish_bridge_outputs"):
            self._publish_bridge_outputs(world, resources, group_x, group_y, gas_group_x, gas_group_y)
        self.last_cpu_mirror_downloaded = not (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            with self._profile_pass(world, "download_outputs"):
                return self._download_outputs(world, resources)
        return self._empty_stage_targets(world)

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
            self.resources.timer_tex,
            self.resources.timer_out_tex,
            self.resources.integrity_tex,
            self.resources.integrity_out_tex,
            self.resources.island_id_tex,
            self.resources.island_id_out_tex,
            self.resources.entity_id_tex,
            self.resources.entity_id_out_tex,
            self.resources.displaced_tex,
            self.resources.displaced_out_tex,
            self.resources.velocity_tex,
            self.resources.velocity_out_tex,
            self.resources.temp_ping,
            self.resources.temp_pong,
            self.resources.phase_target_tex,
            self.resources.boil_target_tex,
            self.resources.gas_tex,
            self.resources.gas_out_tex,
            self.resources.condense_target_tex,
            self.resources.ambient_ping,
            self.resources.ambient_pong,
            self.resources.active_tile_tex,
            self.resources.material_params,
            self.resources.material_response_params,
            self.resources.material_phase_params,
            self.resources.gas_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUHeatResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height, world.gas_concentration.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        gas_count = signature[4]
        material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        material_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        cell_flags_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        timer_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        timer_out_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        integrity_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        integrity_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        island_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        entity_id_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        displaced_out_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        velocity_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        velocity_out_tex = ctx.texture((world.width, world.height), 2, dtype="f4")
        temp_ping = ctx.texture((world.width, world.height), 1, dtype="f4")
        temp_pong = ctx.texture((world.width, world.height), 1, dtype="f4")
        phase_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        boil_target_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        gas_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        gas_out_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        condense_target_tex = ctx.texture_array((world.gas_width, world.gas_height, gas_count), 1, dtype="f4")
        ambient_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        active_tile_tex = ctx.texture((world.active.tile_width, world.active.tile_height), 1, dtype="f4")
        for texture in (
            material_tex,
            material_out_tex,
            phase_tex,
            phase_out_tex,
            cell_flags_tex,
            cell_flags_out_tex,
            timer_tex,
            timer_out_tex,
            integrity_tex,
            integrity_out_tex,
            island_id_tex,
            island_id_out_tex,
            entity_id_tex,
            entity_id_out_tex,
            displaced_tex,
            displaced_out_tex,
            velocity_tex,
            velocity_out_tex,
            temp_ping,
            temp_pong,
            phase_target_tex,
            boil_target_tex,
            gas_tex,
            gas_out_tex,
            condense_target_tex,
            ambient_ping,
            ambient_pong,
            active_tile_tex,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        material_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        material_response_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        material_phase_params = ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True)
        gas_params = ctx.buffer(reserve=MAX_GAS_SPECIES * 4 * 4, dynamic=True)
        self.resources = GPUHeatResources(
            signature=signature,
            material_tex=material_tex,
            material_out_tex=material_out_tex,
            phase_tex=phase_tex,
            phase_out_tex=phase_out_tex,
            cell_flags_tex=cell_flags_tex,
            cell_flags_out_tex=cell_flags_out_tex,
            timer_tex=timer_tex,
            timer_out_tex=timer_out_tex,
            integrity_tex=integrity_tex,
            integrity_out_tex=integrity_out_tex,
            island_id_tex=island_id_tex,
            island_id_out_tex=island_id_out_tex,
            entity_id_tex=entity_id_tex,
            entity_id_out_tex=entity_id_out_tex,
            displaced_tex=displaced_tex,
            displaced_out_tex=displaced_out_tex,
            velocity_tex=velocity_tex,
            velocity_out_tex=velocity_out_tex,
            temp_ping=temp_ping,
            temp_pong=temp_pong,
            phase_target_tex=phase_target_tex,
            boil_target_tex=boil_target_tex,
            gas_tex=gas_tex,
            gas_out_tex=gas_out_tex,
            condense_target_tex=condense_target_tex,
            ambient_ping=ambient_ping,
            ambient_pong=ambient_pong,
            active_tile_tex=active_tile_tex,
            material_params=material_params,
            material_response_params=material_response_params,
            material_phase_params=material_phase_params,
            gas_params=gas_params,
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = build_compute_shader(ctx, "heat/load_active_tiles.comp", _SHADER_SUBS)
        self.programs["cell_heat"] = build_compute_shader(ctx, "heat/cell_heat.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_exchange"] = build_compute_shader(ctx, "heat/ambient_exchange.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_feedback"] = build_compute_shader(ctx, "heat/ambient_feedback.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_diffuse"] = build_compute_shader(ctx, "heat/ambient_diffuse.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["phase_targets"] = build_compute_shader(ctx, "heat/phase_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_cell_targets"] = build_compute_shader(ctx, "heat/apply_cell_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_cell_aux_targets"] = build_compute_shader(ctx, "heat/apply_cell_aux_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["boil_targets"] = build_compute_shader(ctx, "heat/boil_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["condense_targets"] = build_compute_shader(ctx, "heat/condense_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_gas_targets"] = build_compute_shader(ctx, "heat/apply_gas_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_condense_cells"] = build_compute_shader(ctx, "heat/apply_condense_cells.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["apply_condense_cell_aux"] = build_compute_shader(ctx, "heat/apply_condense_cell_aux.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "heat/publish_bridge_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_gas"] = build_compute_shader(ctx, "heat/publish_bridge_gas.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "heat/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux"] = build_compute_shader(ctx, "heat/load_bridge_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_gas"] = build_compute_shader(ctx, "heat/load_bridge_gas.comp", _SHADER_SUBS)

    def _upload_inputs(self, world: "WorldEngine", resources: GPUHeatResources, solve_tile_mask: np.ndarray) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "heat input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "gas_concentration",
            "active_tile_ttl",
        )
        upload_cell_state_from_cpu = not (formal_gpu_frame and "cell_core" in authoritative)
        upload_island_id_from_cpu = not (formal_gpu_frame and "island_id" in authoritative)
        upload_entity_id_from_cpu = not (formal_gpu_frame and "entity_id" in authoritative)
        upload_displaced_from_cpu = not (formal_gpu_frame and "placeholder_displaced_material" in authoritative)
        upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
        upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
        upload_active_from_cpu = not (formal_gpu_frame and "active_tile_ttl" in authoritative)
        self.last_cpu_cell_state_upload_skipped = not upload_cell_state_from_cpu
        self.last_cpu_island_id_upload_skipped = not upload_island_id_from_cpu
        self.last_cpu_entity_id_upload_skipped = not upload_entity_id_from_cpu
        self.last_cpu_displaced_material_upload_skipped = not upload_displaced_from_cpu
        self.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
        self.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_cell_state_from_cpu:
            resources.material_tex.write(world.material_id.astype("f4").tobytes())
            resources.phase_tex.write(world.phase.astype("f4").tobytes())
            resources.cell_flags_tex.write(world.cell_flags.astype("f4").tobytes())
            resources.timer_tex.write(world.timer_pack.astype("f4").tobytes())
            resources.integrity_tex.write(world.integrity.astype("f4").tobytes())
            resources.velocity_tex.write(world.velocity.astype("f4").tobytes())
            resources.velocity_out_tex.write(world.velocity.astype("f4").tobytes())
            resources.temp_ping.write(world.cell_temperature.astype("f4").tobytes())
            resources.temp_pong.write(world.cell_temperature.astype("f4").tobytes())
        if upload_island_id_from_cpu:
            resources.island_id_tex.write(world.island_id.astype("f4").tobytes())
        if upload_entity_id_from_cpu:
            resources.entity_id_tex.write(world.entity_id.astype("f4").tobytes())
        if upload_displaced_from_cpu:
            resources.displaced_tex.write(world.placeholder_displaced_material.astype("f4").tobytes())
        if upload_gas_from_cpu:
            resources.gas_tex.write(world.gas_concentration.astype("f4").tobytes())
            resources.gas_out_tex.write(world.gas_concentration.astype("f4").tobytes())
        if upload_ambient_from_cpu:
            resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
            resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
        if upload_active_from_cpu:
            resources.active_tile_tex.write(np.asarray(solve_tile_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_tile_mask(world, resources, expansion_radius=1)
        material_table = world.bridge.shadow_typed_tables["material_table"]
        material_signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        if resources.material_params_signature != material_signature:
            params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
            response_params = np.zeros((MAX_MATERIALS, 4), dtype="f4")
            phase_params = np.zeros((MAX_MATERIALS, 4), dtype=np.int32)
            count = min(MAX_MATERIALS, material_table.shape[0])
            params[:count, 0] = material_table[:count]["conductivity"]
            params[:count, 1] = material_table[:count]["ambient_exchange_rate"]
            params[:count, 2] = material_table[:count]["melt_point"]
            params[:count, 3] = material_table[:count]["boil_point"]
            response_params[:count, 0] = material_table[:count]["heat_capacity"]
            response_params[:count, 1] = material_table[:count]["base_integrity"]
            response_params[:count, 2] = material_table[:count]["spawn_temperature"]
            response_params[:count, 3] = material_table[:count]["render_group_id"].astype("f4")
            phase_params[:count, 0] = material_table[:count]["default_phase"]
            phase_params[:count, 1] = material_table[:count]["melt_to_material_id"]
            phase_params[:count, 2] = material_table[:count]["freeze_to_material_id"]
            boil_species = material_table[:count]["boil_to_gas_species_id"].astype(np.int32)
            phase_params[:count, 3] = np.where(boil_species >= 0, boil_species + 1, 0)
            resources.material_params.write(params.tobytes())
            resources.material_response_params.write(response_params.tobytes())
            resources.material_phase_params.write(phase_params.tobytes())
            resources.material_params_signature = material_signature
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        gas_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
        if resources.gas_params_signature != gas_signature:
            gas_params = np.zeros((MAX_GAS_SPECIES, 4), dtype="f4")
            count = min(MAX_GAS_SPECIES, gas_table.shape[0])
            gas_params[:count, 0] = gas_table[:count]["condense_point"]
            gas_params[:count, 1] = gas_table[:count]["condense_to_material_id"].astype("f4")
            resources.gas_params.write(gas_params.tobytes())
            resources.gas_params_signature = gas_signature

    def _load_authoritative_active_tile_mask(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge active scheduler resources")
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

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        copy_gas = "gas_concentration" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient or copy_gas):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative input state")

        if copy_cell_core:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            resources.material_tex.bind_to_image(4, read=False, write=True)
            resources.phase_tex.bind_to_image(5, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(6, read=False, write=True)
            resources.timer_tex.bind_to_image(7, read=False, write=True)
            resources.temp_ping.bind_to_image(0, read=False, write=True)
            resources.integrity_tex.bind_to_image(1, read=False, write=True)
            resources.velocity_tex.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)

        if copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell_aux"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.island_id_tex.bind_to_image(0, read=False, write=True)
            resources.entity_id_tex.bind_to_image(1, read=False, write=True)
            resources.displaced_tex.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)

        if copy_ambient or copy_gas:
            program = self.programs["load_bridge_gas"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["species_count"].value = int(world.gas_concentration.shape[0])
            program["copy_ambient"].value = bool(copy_ambient)
            program["copy_gas"].value = bool(copy_gas)
            bridge.textures["ambient_temperature"].use(location=0)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
            resources.ambient_ping.bind_to_image(2, read=False, write=True)
            resources.gas_tex.bind_to_image(3, read=False, write=True)
            program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

        self._sync_compute_writes(bridge.ctx)

    def _run_cell_heat(self, world: "WorldEngine", dt: float, resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["cell_heat"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_ping.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.temp_pong.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_ambient_exchange(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["ambient_exchange"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_pong.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.temp_ping.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_ambient_diffuse(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
        *,
        iterations: int,
    ) -> None:
        if iterations <= 0:
            return
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["ambient_diffuse"]
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        for _ in range(iterations):
            resources.ambient_ping.use(location=3)
            resources.ambient_pong.bind_to_image(4, read=False, write=True)
            program.run(gas_group_x, gas_group_y, 1)
            self._sync_compute_writes(ctx)
            resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping

    def _run_ambient_feedback(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["ambient_feedback"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["tile_size"].value = world.active.tile_size
        program["dt"].value = dt
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.temp_pong.use(location=3)
        resources.ambient_ping.use(location=4)
        resources.ambient_pong.bind_to_image(5, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_phase_targets(self, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["phase_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        self._set_uniform_if_present(program, "freeze_cold_neighbor_threshold", FREEZE_COLD_NEIGHBOR_THRESHOLD)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.phase_tex.use(location=4)
        resources.temp_ping.use(location=5)
        resources.phase_target_tex.bind_to_image(6, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_boil_targets(self, world: "WorldEngine", resources: GPUHeatResources, group_x: int, group_y: int) -> None:
        program = self.programs["boil_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.temp_ping.use(location=4)
        resources.boil_target_tex.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_condense_targets(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["condense_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=3)
        resources.gas_tex.use(location=4)
        resources.ambient_pong.use(location=5)
        resources.condense_target_tex.bind_to_image(6, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_cell_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        with self._profile_pass(world, "apply_cell_targets.main"):
            program = self.programs["apply_cell_targets"]
            ctx = world.bridge.ctx
            assert ctx is not None
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
            self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
            self._set_uniform_if_present(program, "dt", dt)
            self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
            self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
            resources.material_params.bind_to_storage_buffer(binding=0)
            resources.material_tex.use(location=1)
            resources.active_tile_tex.use(location=2)
            resources.material_phase_params.bind_to_storage_buffer(binding=3)
            resources.material_response_params.bind_to_storage_buffer(binding=7)
            resources.phase_target_tex.use(location=3)
            resources.phase_tex.use(location=4)
            resources.cell_flags_tex.use(location=5)
            resources.timer_tex.use(location=6)
            resources.boil_target_tex.use(location=7)
            resources.temp_ping.use(location=8)
            resources.integrity_tex.use(location=9)
            resources.island_id_tex.use(location=10)
            resources.entity_id_tex.use(location=11)
            resources.displaced_tex.use(location=12)
            resources.ambient_pong.use(location=22)
            resources.velocity_tex.use(location=23)
            resources.material_out_tex.bind_to_image(0, read=False, write=True)
            resources.phase_out_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_out_tex.bind_to_image(2, read=False, write=True)
            resources.timer_out_tex.bind_to_image(3, read=False, write=True)
            resources.temp_pong.bind_to_image(4, read=False, write=True)
            resources.integrity_out_tex.bind_to_image(5, read=False, write=True)
            resources.island_id_out_tex.bind_to_image(6, read=False, write=True)
            resources.entity_id_out_tex.bind_to_image(7, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
        with self._profile_pass(world, "apply_cell_targets.aux"):
            self._run_apply_cell_aux_targets(world, dt, resources, group_x, group_y)

    def _run_apply_cell_aux_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["apply_cell_aux_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "dt", dt)
        self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
        self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_phase_params.bind_to_storage_buffer(binding=3)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.phase_target_tex.use(location=3)
        resources.phase_tex.use(location=4)
        resources.boil_target_tex.use(location=5)
        resources.integrity_tex.use(location=6)
        resources.displaced_tex.use(location=7)
        resources.velocity_tex.use(location=8)
        resources.displaced_out_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_out_tex.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_gas_targets(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUHeatResources,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        program = self.programs["apply_gas_targets"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        self._set_uniform_if_present(program, "dt", dt)
        resources.material_params.bind_to_storage_buffer(binding=0)
        resources.material_tex.use(location=1)
        resources.active_tile_tex.use(location=2)
        resources.material_response_params.bind_to_storage_buffer(binding=7)
        resources.gas_params.bind_to_storage_buffer(binding=3)
        resources.gas_tex.use(location=4)
        resources.boil_target_tex.use(location=5)
        resources.condense_target_tex.use(location=6)
        resources.material_out_tex.use(location=8)
        resources.gas_out_tex.bind_to_image(0, read=False, write=True)
        program.run(gas_group_x, gas_group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_apply_condense_cells(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        with self._profile_pass(world, "apply_condense_cells.main"):
            program = self.programs["apply_condense_cells"]
            ctx = world.bridge.ctx
            assert ctx is not None
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
            self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
            self._set_uniform_if_present(
                program,
                "gas_species_count",
                min(world.gas_concentration.shape[0], MAX_GAS_SPECIES),
            )
            self._set_uniform_if_present(program, "phase_falling_island", int(Phase.FALLING_ISLAND))
            self._set_uniform_if_present(program, "phase_liquid", int(Phase.LIQUID))
            resources.active_tile_tex.use(location=2)
            resources.material_phase_params.bind_to_storage_buffer(binding=3)
            resources.material_response_params.bind_to_storage_buffer(binding=7)
            resources.gas_params.bind_to_storage_buffer(binding=8)
            resources.material_out_tex.use(location=4)
            resources.phase_out_tex.use(location=5)
            resources.cell_flags_out_tex.use(location=6)
            resources.timer_out_tex.use(location=9)
            resources.temp_pong.use(location=10)
            resources.integrity_out_tex.use(location=11)
            resources.island_id_out_tex.use(location=12)
            resources.entity_id_out_tex.use(location=13)
            resources.displaced_out_tex.use(location=22)
            resources.velocity_out_tex.use(location=23)
            resources.condense_target_tex.use(location=24)
            resources.material_tex.bind_to_image(0, read=False, write=True)
            resources.phase_tex.bind_to_image(1, read=False, write=True)
            resources.cell_flags_tex.bind_to_image(2, read=False, write=True)
            resources.timer_tex.bind_to_image(3, read=False, write=True)
            resources.temp_ping.bind_to_image(4, read=False, write=True)
            resources.integrity_tex.bind_to_image(5, read=False, write=True)
            resources.displaced_tex.bind_to_image(6, read=False, write=True)
            resources.velocity_tex.bind_to_image(7, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
            resources.island_id_tex, resources.island_id_out_tex = resources.island_id_out_tex, resources.island_id_tex
            resources.entity_id_tex, resources.entity_id_out_tex = resources.entity_id_out_tex, resources.entity_id_tex

    def _run_apply_condense_cell_aux(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
    ) -> None:
        program = self.programs["apply_condense_cell_aux"]
        ctx = world.bridge.ctx
        assert ctx is not None
        self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
        self._set_uniform_if_present(program, "gas_cell_size", world.gas_cell_size)
        self._set_uniform_if_present(program, "tile_size", world.active.tile_size)
        self._set_uniform_if_present(program, "gas_species_count", min(world.gas_concentration.shape[0], MAX_GAS_SPECIES))
        resources.active_tile_tex.use(location=2)
        resources.gas_params.bind_to_storage_buffer(binding=8)
        resources.material_out_tex.use(location=4)
        resources.phase_out_tex.use(location=5)
        resources.island_id_out_tex.use(location=12)
        resources.displaced_out_tex.use(location=22)
        resources.velocity_out_tex.use(location=23)
        resources.condense_target_tex.use(location=24)
        resources.displaced_tex.bind_to_image(0, read=False, write=True)
        resources.velocity_tex.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _download_outputs(self, world: "WorldEngine", resources: GPUHeatResources) -> GPUHeatStageTargets:
        world.material_id[:] = np.rint(
            np.frombuffer(resources.material_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.phase[:] = np.rint(
            np.frombuffer(resources.phase_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.cell_flags[:] = np.rint(
            np.frombuffer(resources.cell_flags_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.uint8)
        world.timer_pack[:] = np.rint(
            np.frombuffer(resources.timer_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        ).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(resources.temp_ping.read(), dtype="f4").reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(resources.integrity_tex.read(), dtype="f4").reshape((world.height, world.width))
        world.island_id[:] = np.rint(
            np.frombuffer(resources.island_id_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.entity_id[:] = np.rint(
            np.frombuffer(resources.entity_id_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.placeholder_displaced_material[:] = np.rint(
            np.frombuffer(resources.displaced_tex.read(), dtype="f4").reshape((world.height, world.width))
        ).astype(np.int32)
        world.velocity[:] = np.frombuffer(resources.velocity_tex.read(), dtype="f4").reshape((world.height, world.width, 2))
        world.ambient_temperature[:] = np.frombuffer(resources.ambient_pong.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        world.gas_concentration[:] = np.frombuffer(resources.gas_out_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
        return GPUHeatStageTargets(
            phase_targets=np.rint(
                np.frombuffer(resources.phase_target_tex.read(), dtype="f4").reshape((world.height, world.width))
            ).astype(np.int32),
            boil_targets=np.rint(
                np.frombuffer(resources.boil_target_tex.read(), dtype="f4").reshape((world.height, world.width))
            ).astype(np.int32),
            condense_targets=(
                np.frombuffer(resources.condense_target_tex.read(), dtype="f4").reshape(world.gas_concentration.shape)
                > 0.5
            ),
        )

    def _empty_stage_targets(self, world: "WorldEngine") -> GPUHeatStageTargets:
        if self._formal_gpu_frame(world):
            return GPUHeatStageTargets.empty_sentinel()
        return GPUHeatStageTargets(
            phase_targets=np.zeros((world.height, world.width), dtype=np.int32),
            boil_targets=np.zeros((world.height, world.width), dtype=np.int32),
            condense_targets=np.zeros(world.gas_concentration.shape, dtype=np.bool_),
        )

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPUHeatResources,
        group_x: int,
        group_y: int,
        gas_group_x: int,
        gas_group_y: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU heat pipeline requires bridge GPU resources for authoritative heat state")
        fuse_structure_dirty_mark = False
        dirty_buffer = None
        dirty_count = None
        dirty_list = None
        dirty_dispatch_args = None
        material_flags_buffer = None
        material_count = 0
        if self._formal_gpu_frame(world) and _active_scheduler_gpu_authoritative(world):
            dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
            dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
            if dirty_buffer is not None and dirty_queue is not None:
                dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
                material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
                fuse_structure_dirty_mark = True
        with self._profile_pass(world, "publish_bridge_outputs.collapse_dirty_mark"):
            if not fuse_structure_dirty_mark:
                mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
                    world,
                    resources.material_tex,
                    resources.phase_tex,
                )
        with self._profile_pass(world, "publish_bridge_outputs.cell"):
            cell_program = self.programs["publish_bridge_cell"]
            cell_program["cell_grid_size"].value = (world.width, world.height)
            cell_program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
            cell_program["tile_size"].value = int(world.active.tile_size)
            cell_program["material_count"].value = int(material_count)
            cell_program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
            cell_program["mark_structure_dirty"].value = bool(fuse_structure_dirty_mark)
            cell_program["write_cell_core"].value = not bool(getattr(world, "phase_c_defer_cell_publish", False))
            cell_program["material_tex"].value = 0
            cell_program["phase_tex"].value = 1
            cell_program["flags_tex"].value = 2
            cell_program["timer_tex"].value = 3
            cell_program["temp_tex"].value = 4
            cell_program["integrity_tex"].value = 5
            cell_program["island_tex"].value = 6
            cell_program["entity_tex"].value = 7
            cell_program["displaced_tex"].value = 8
            cell_program["velocity_tex"].value = 9
            resources.material_tex.use(location=0)
            resources.phase_tex.use(location=1)
            resources.cell_flags_tex.use(location=2)
            resources.timer_tex.use(location=3)
            resources.temp_ping.use(location=4)
            resources.integrity_tex.use(location=5)
            resources.island_id_tex.use(location=6)
            resources.entity_id_tex.use(location=7)
            resources.displaced_tex.use(location=8)
            resources.velocity_tex.use(location=9)
            bridge.textures["material"].bind_to_image(0, read=False, write=True)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            if fuse_structure_dirty_mark:
                assert dirty_buffer is not None
                assert dirty_count is not None
                assert dirty_list is not None
                assert dirty_dispatch_args is not None
                assert material_flags_buffer is not None
                material_flags_buffer.bind_to_storage_buffer(binding=4)
                dirty_buffer.bind_to_storage_buffer(binding=5)
                dirty_count.bind_to_storage_buffer(binding=6)
                dirty_list.bind_to_storage_buffer(binding=7)
                dirty_dispatch_args.bind_to_storage_buffer(binding=8)
        if not bool(getattr(world, "phase_c_defer_cell_publish", False)):
            cell_program.run(group_x, group_y, 1)

        with self._profile_pass(world, "publish_bridge_outputs.gas"):
            gas_program = self.programs["publish_bridge_gas"]
            gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            gas_program["species_count"].value = int(world.gas_concentration.shape[0])
            gas_program["ambient_tex"].value = 0
            gas_program["gas_tex"].value = 2
            resources.ambient_pong.use(location=0)
            resources.gas_out_tex.use(location=2)
            bridge.textures["ambient_temperature"].bind_to_image(1, read=False, write=True)
            bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=4)
            gas_program.run(gas_group_x, gas_group_y, int(world.gas_concentration.shape[0]))

        with self._profile_pass(world, "publish_bridge_outputs.sync"):
            self._sync_compute_writes(bridge.ctx)
            if fuse_structure_dirty_mark:
                setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
                bridge.mark_gpu_authoritative(
                    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
                    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
                )
            bridge.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
                "ambient_temperature",
                "gas_concentration",
            )

    # ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
    # from :class:`GPUPipelineBase`; the heat pass uses the default barrier
    # bits (image-access | texture-fetch | shader-storage).
