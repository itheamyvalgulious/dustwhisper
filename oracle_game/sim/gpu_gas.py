from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_gas_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader


LOCAL_SIZE = 8
REDUCE_LOCAL_SIZE = 256
MAX_SPECIES = 8
ACTIVE_GAS_TEXTURE_UNIT = 7
SPECIES_COOPERATIVE_LOCAL_X = 8
SPECIES_COOPERATIVE_LOCAL_Y = 4
PRESSURE_CONE_CORE_X = 48
PRESSURE_CONE_CORE_Y = 16
PRESSURE_CONE_LOCAL_X = 16
PRESSURE_CONE_LOCAL_Y = 16
PRESSURE_CONE_JACOBI_ITERATIONS = 11
PRESSURE_CONE_HALO = PRESSURE_CONE_JACOBI_ITERATIONS + 1
PRESSURE_PAIR_LOCAL_SIZE = 16
PRESSURE_PAIR_HALO = 2
PRESSURE_PAIR_CORE_SIZE = PRESSURE_PAIR_LOCAL_SIZE - 2 * PRESSURE_PAIR_HALO

# Superset of every {{NAME}} marker referenced by any gas shader; the loader
# ignores unused keys, so one shared dict suffices for all passes.
_SHADER_SUBS = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "REDUCE_LOCAL_SIZE": REDUCE_LOCAL_SIZE,
    "MAX_SPECIES": MAX_SPECIES,
    "ACTIVE_GAS_TEXTURE_UNIT": ACTIVE_GAS_TEXTURE_UNIT,
    "SPECIES_COOPERATIVE_LOCAL_X": SPECIES_COOPERATIVE_LOCAL_X,
    "SPECIES_COOPERATIVE_LOCAL_Y": SPECIES_COOPERATIVE_LOCAL_Y,
    "PRESSURE_CONE_CORE_X": PRESSURE_CONE_CORE_X,
    "PRESSURE_CONE_CORE_Y": PRESSURE_CONE_CORE_Y,
    "PRESSURE_CONE_LOCAL_X": PRESSURE_CONE_LOCAL_X,
    "PRESSURE_CONE_LOCAL_Y": PRESSURE_CONE_LOCAL_Y,
    "PRESSURE_CONE_JACOBI_ITERATIONS": PRESSURE_CONE_JACOBI_ITERATIONS,
    "PRESSURE_CONE_HALO": PRESSURE_CONE_HALO,
    "PRESSURE_PAIR_LOCAL_SIZE": PRESSURE_PAIR_LOCAL_SIZE,
    "PRESSURE_PAIR_HALO": PRESSURE_PAIR_HALO,
    "PRESSURE_PAIR_CORE_SIZE": PRESSURE_PAIR_CORE_SIZE,
}


@dataclass(slots=True)
class GPUGasResources:
    signature: tuple[int, int, int]
    velocity_ping: Any
    velocity_pong: Any
    divergence: Any
    thermo_pressure: Any
    density_tex: Any
    pressure_ping: Any
    pressure_pong: Any
    pressure_cone_shadow: Any
    ambient_ping: Any
    ambient_pong: Any
    gas_ping: Any
    gas_pong: Any
    active_gas_tex: Any
    species_params: Any
    species_force_params: Any
    force_sources: Any
    density_reduce_ping: Any
    density_reduce_pong: Any
    species_params_signature: tuple[int, int] | None = None


class GPUGasPipeline(GPUPipelineBase):
    def __init__(self, pressure_iterations: int = 12) -> None:
        self.pressure_iterations = pressure_iterations
        self.resources: GPUGasResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_velocity_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_divergence_pressure_seed_used = False
        self.last_species_terminal_cooperative_used = False
        self.last_density_tree_reduction_used = False
        self._divergence_pressure_seed_enabled = True
        self._species_terminal_cooperative_enabled = True
        self._pressure_projection_dependency_cone_enabled = False
        self._pressure_jacobi_pair_enabled = True
        self._density_tree_reduction_enabled = True
        # Candidate: force-source application and thermo-field construction
        # write disjoint outputs and can share one dispatch. Keep disabled
        # until formal raw-byte/A-B retention; the canonical two-pass path is
        # unchanged by default.
        self._force_thermo_fusion_enabled = False
        self.last_force_thermo_fusion_used = False
        self.last_pressure_jacobi_pair_used = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` is inherited from :class:`GPUPipelineBase`.

    def step(self, world: "WorldEngine", dt: float, *, solve_gas_mask: np.ndarray) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU gas pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        self.reset_pass_profile()
        self.last_divergence_pressure_seed_used = False
        self.last_species_terminal_cooperative_used = False
        self.last_density_tree_reduction_used = False
        self.last_force_thermo_fusion_used = False
        self.last_pressure_jacobi_pair_used = False
        with self._profile_pass(world, "upload_inputs"):
            self._upload_inputs(world, resources, solve_gas_mask)
        group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "load_bridge_inputs"):
            self._load_authoritative_bridge_inputs(world, resources, group_x, group_y)

        with self._profile_pass(world, "advect_velocity"):
            self._run_advect_velocity(world, dt, resources, group_x, group_y)
        force_thermo_fusion = bool(
            self._force_thermo_fusion_enabled
            and self._density_tree_reduction_enabled
            and bool(world.force_sources)
        )
        self.last_force_thermo_fusion_used = force_thermo_fusion
        if force_thermo_fusion:
            with self._profile_pass(world, "force_sources_thermo_fields"):
                self._run_force_sources_thermo_fields(world, dt, resources, group_x, group_y)
        else:
            with self._profile_pass(world, "force_sources"):
                self._run_force_sources(world, dt, resources, group_x, group_y)
            with self._profile_pass(world, "thermo_fields"):
                self._run_thermo_fields(world, resources, group_x, group_y)
        with self._profile_pass(world, "thermo_forces"):
            self._run_thermo_forces(world, dt, resources, group_x, group_y)
        with self._profile_pass(world, "divergence"):
            self._run_divergence(world, resources, group_x, group_y)
        if self._can_run_pressure_projection_dependency_cone():
            with self._profile_pass(world, "pressure_projection_dependency_cone"):
                self._run_pressure_projection_dependency_cone(world, resources)
        else:
            with self._profile_pass(world, "pressure_jacobi"):
                self._run_pressure_jacobi(world, resources, group_x, group_y)
            with self._profile_pass(world, "projection"):
                self._run_projection(world, resources, group_x, group_y)
        if self._can_run_species_terminal_cooperative(world):
            with self._profile_pass(world, "species_ambient_publish_cooperative"):
                self._run_species_terminal_cooperative(world, dt, resources)
        else:
            with self._profile_pass(world, "species"):
                self._run_species(world, dt, resources, group_x, group_y)
            with self._profile_pass(world, "ambient"):
                self._run_ambient(world, dt, resources, group_x, group_y)
            with self._profile_pass(world, "publish_bridge_outputs"):
                self._publish_bridge_outputs(world, resources, group_x, group_y)
        self.last_cpu_mirror_downloaded = not (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            with self._profile_pass(world, "download_outputs"):
                self._download_outputs(world, resources)

    # ``reset_pass_profile`` / ``_profile_pass`` are inherited from
    # :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.velocity_ping,
            self.resources.velocity_pong,
            self.resources.divergence,
            self.resources.thermo_pressure,
            self.resources.density_tex,
            self.resources.pressure_ping,
            self.resources.pressure_pong,
            self.resources.pressure_cone_shadow,
            self.resources.ambient_ping,
            self.resources.ambient_pong,
            self.resources.gas_ping,
            self.resources.gas_pong,
            self.resources.active_gas_tex,
            self.resources.species_params,
            self.resources.species_force_params,
            self.resources.force_sources,
            self.resources.density_reduce_ping,
            self.resources.density_reduce_pong,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUGasResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.gas_width, world.gas_height, world.gas_concentration.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        species_count = signature[2]
        velocity_ping = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        velocity_pong = ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        divergence = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        thermo_pressure = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        density_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        pressure_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        pressure_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        pressure_cone_shadow = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_ping = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        ambient_pong = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        gas_ping = ctx.texture_array((world.gas_width, world.gas_height, species_count), 1, dtype="f4")
        gas_pong = ctx.texture_array((world.gas_width, world.gas_height, species_count), 1, dtype="f4")
        active_gas_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        for texture in (
            velocity_ping,
            velocity_pong,
            divergence,
            thermo_pressure,
            density_tex,
            pressure_ping,
            pressure_pong,
            pressure_cone_shadow,
            ambient_ping,
            ambient_pong,
            gas_ping,
            gas_pong,
            active_gas_tex,
        ):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        species_params = ctx.buffer(reserve=MAX_SPECIES * 4 * 4, dynamic=True)
        species_force_params = ctx.buffer(reserve=MAX_SPECIES * 4 * 4, dynamic=True)
        force_sources = ctx.buffer(reserve=4, dynamic=True)
        density_reduce_ping = ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True)
        density_reduce_pong = ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True)
        self.resources = GPUGasResources(
            signature=signature,
            velocity_ping=velocity_ping,
            velocity_pong=velocity_pong,
            divergence=divergence,
            thermo_pressure=thermo_pressure,
            density_tex=density_tex,
            pressure_ping=pressure_ping,
            pressure_pong=pressure_pong,
            pressure_cone_shadow=pressure_cone_shadow,
            ambient_ping=ambient_ping,
            ambient_pong=ambient_pong,
            gas_ping=gas_ping,
            gas_pong=gas_pong,
            active_gas_tex=active_gas_tex,
            species_params=species_params,
            species_force_params=species_force_params,
            force_sources=force_sources,
            density_reduce_ping=density_reduce_ping,
            density_reduce_pong=density_reduce_pong,
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_bridge"] = build_compute_shader(ctx, "gas/load_bridge.comp", _SHADER_SUBS)
        self.programs["load_active_gas"] = build_compute_shader(ctx, "gas/load_active_gas.comp", _SHADER_SUBS)
        self.programs["advect_velocity"] = build_compute_shader(ctx, "gas/advect_velocity.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["divergence"] = build_compute_shader(ctx, "gas/divergence.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["divergence_pressure_seed"] = build_compute_shader(
            ctx,
            "gas/divergence_pressure_seed.comp",
            _SHADER_SUBS,
            includes=["gas/_common.comp"],
        )
        self.programs["force_sources"] = build_compute_shader(ctx, "gas/force_sources.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["force_sources_thermo_fields_density_buffer"] = build_compute_shader(
            ctx,
            "gas/force_sources_thermo_fields_density_buffer.comp",
            _SHADER_SUBS,
            includes=["gas/_common.comp"],
        )
        self.programs["thermo_fields"] = build_compute_shader(ctx, "gas/thermo_fields.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["thermo_fields_density_buffer"] = build_compute_shader(
            ctx,
            "gas/thermo_fields_density_buffer.comp",
            _SHADER_SUBS,
            includes=["gas/_common.comp"],
        )
        self.programs["density_extract"] = build_compute_shader(ctx, "gas/density_extract.comp", _SHADER_SUBS)
        self.programs["density_reduce"] = build_compute_shader(ctx, "gas/density_reduce.comp", _SHADER_SUBS)
        self.programs["density_reduce_tree"] = build_compute_shader(
            ctx,
            "gas/density_reduce_tree.comp",
            _SHADER_SUBS,
        )
        self.programs["jacobi"] = build_compute_shader(ctx, "gas/jacobi.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["jacobi_pair"] = build_compute_shader(
            ctx,
            "gas/jacobi_pair.comp",
            _SHADER_SUBS,
        )
        self.programs["thermo_forces"] = build_compute_shader(ctx, "gas/thermo_forces.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["projection"] = build_compute_shader(ctx, "gas/projection.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["pressure_projection_dependency_cone"] = build_compute_shader(
            ctx,
            "gas/pressure_projection_dependency_cone.comp",
            _SHADER_SUBS,
        )
        self.programs["species"] = build_compute_shader(ctx, "gas/species.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["species_terminal_cooperative"] = build_compute_shader(
            ctx,
            "gas/species_terminal_cooperative.comp",
            _SHADER_SUBS,
        )
        self.programs["ambient"] = build_compute_shader(ctx, "gas/ambient.comp", _SHADER_SUBS, includes=["gas/_common.comp"])
        self.programs["publish_bridge"] = build_compute_shader(ctx, "gas/publish_bridge.comp", _SHADER_SUBS)

    def _upload_inputs(self, world: "WorldEngine", resources: GPUGasResources, solve_gas_mask: np.ndarray) -> None:
        world.bridge.sync_rule_tables(world)
        authoritative = world.bridge.gpu_authoritative_resources
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "gas input",
            "flow_velocity",
            "ambient_temperature",
            "gas_concentration",
            "active_tile_ttl",
        )
        upload_velocity_from_cpu = not (formal_gpu_frame and "flow_velocity" in authoritative)
        upload_ambient_from_cpu = not (formal_gpu_frame and "ambient_temperature" in authoritative)
        upload_gas_from_cpu = not (formal_gpu_frame and "gas_concentration" in authoritative)
        upload_active_from_cpu = not (formal_gpu_frame and "active_tile_ttl" in authoritative)
        self.last_cpu_velocity_upload_skipped = not upload_velocity_from_cpu
        self.last_cpu_ambient_upload_skipped = not upload_ambient_from_cpu
        self.last_cpu_gas_upload_skipped = not upload_gas_from_cpu
        self.last_cpu_active_upload_skipped = not upload_active_from_cpu
        if upload_velocity_from_cpu:
            resources.velocity_ping.write(world.flow_velocity.astype("f4").tobytes())
            resources.velocity_pong.write(world.flow_velocity.astype("f4").tobytes())
        if upload_ambient_from_cpu:
            resources.ambient_ping.write(world.ambient_temperature.astype("f4").tobytes())
            resources.ambient_pong.write(world.ambient_temperature.astype("f4").tobytes())
        pressure_seeded_by_divergence = self._use_divergence_pressure_seed()
        pressure_zero: bytes | None = None
        if not pressure_seeded_by_divergence:
            pressure_zero = np.zeros_like(world.ambient_temperature, dtype="f4").tobytes()
            resources.pressure_ping.write(pressure_zero)
        if not formal_gpu_frame:
            if pressure_zero is None:
                pressure_zero = np.zeros_like(world.ambient_temperature, dtype="f4").tobytes()
            resources.thermo_pressure.write(pressure_zero)
            resources.density_tex.write(pressure_zero)
            if not pressure_seeded_by_divergence:
                resources.pressure_pong.write(pressure_zero)
        if upload_gas_from_cpu:
            resources.gas_ping.write(world.gas_concentration.astype("f4").tobytes())
            resources.gas_pong.write(world.gas_concentration.astype("f4").tobytes())
        if upload_active_from_cpu:
            resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())
        else:
            self._load_authoritative_active_gas_mask(world, resources, expansion_radius=1)
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        table_signature = (world.bridge.table_generations.get("gases", 0), int(gas_table.shape[0]))
        if resources.species_params_signature == table_signature:
            return
        params = np.zeros((MAX_SPECIES, 4), dtype="f4")
        force_params = np.zeros((MAX_SPECIES, 4), dtype="f4")
        count = min(MAX_SPECIES, gas_table.shape[0])
        params[:count, 0] = gas_table[:count]["diffusion_rate"]
        params[:count, 1] = gas_table[:count]["decay_rate"]
        params[:count, 2] = gas_table[:count]["temperature_coupling"]
        params[:count, 3] = gas_table[:count]["buoyancy"]
        force_params[:count, 0] = gas_table[:count]["pressure_factor"]
        force_params[:count, 1] = gas_table[:count]["density_factor"]
        resources.species_params.write(params.tobytes())
        resources.species_force_params.write(force_params.tobytes())
        resources.species_params_signature = table_signature

    def _load_authoritative_active_gas_mask(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        *,
        expansion_radius: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge active scheduler resources")
        program = self.programs["load_active_gas"]
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["cell_grid_size"].value = (world.width, world.height)
        program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["tile_size"].value = int(world.active.tile_size)
        program["expansion_radius"].value = int(expansion_radius)
        bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        resources.active_gas_tex.bind_to_image(1, read=False, write=True)
        program.run(
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)

    def _force_source_upload(self, world: "WorldEngine") -> np.ndarray:
        force_count = len(world.force_sources)
        force_data = np.zeros((max(1, force_count) * 2, 4), dtype=np.float32)
        for index, force in enumerate(world.force_sources):
            buffer_x, buffer_y = world._force_source_buffer_position(force)
            force_data[index * 2] = (
                float(buffer_x),
                float(buffer_y),
                float(force.direction[0]),
                float(force.direction[1]),
            )
            force_data[index * 2 + 1] = (
                float(force.radius),
                float(force.strength),
                float(force.lifetime),
                0.0,
            )
        return force_data

    def _write_dynamic_buffer(self, ctx: Any, resources: GPUGasResources, name: str, data: np.ndarray) -> None:
        buffer = getattr(resources, name)
        nbytes = max(4, int(data.nbytes))
        if buffer.size < nbytes:
            buffer.release()
            buffer = ctx.buffer(reserve=nbytes, dynamic=True)
            setattr(resources, name, buffer)
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _load_authoritative_bridge_inputs(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_velocity = "flow_velocity" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        copy_gas = "gas_concentration" in authoritative
        if not (copy_velocity or copy_ambient or copy_gas):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge GPU resources for authoritative input state")
        program = self.programs["load_bridge"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["copy_velocity"].value = bool(copy_velocity)
        program["copy_ambient"].value = bool(copy_ambient)
        program["copy_gas"].value = bool(copy_gas)
        bridge.textures["flow_velocity"].use(location=0)
        bridge.textures["ambient_temperature"].use(location=1)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=2)
        resources.velocity_ping.bind_to_image(3, read=False, write=True)
        resources.ambient_ping.bind_to_image(4, read=False, write=True)
        resources.gas_ping.bind_to_image(5, read=False, write=True)
        program.run(group_x, group_y, int(world.gas_concentration.shape[0]))
        self._sync_compute_writes(bridge.ctx)

    def _run_advect_velocity(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["advect_velocity"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["damping"].value = 0.995
        resources.velocity_ping.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_pong.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_force_sources(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        if not world.force_sources:
            return
        program = self.programs["force_sources"]
        ctx = world.bridge.ctx
        assert ctx is not None
        force_data = self._force_source_upload(world)
        self._write_dynamic_buffer(ctx, resources, "force_sources", force_data)
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["gas_cell_size"].value = float(world.gas_cell_size)
        program["force_count"].value = len(world.force_sources)
        resources.force_sources.bind_to_storage_buffer(binding=0)
        resources.velocity_pong.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_ping.bind_to_image(1, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping
        for force in list(world.force_sources):
            force.lifetime -= dt
        world.force_sources[:] = [force for force in world.force_sources if force.lifetime > 0.0]

    def _run_force_sources_thermo_fields(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        """Fuse force application with density/thermo field generation.

        The two canonical passes have no intermediate data dependency: the
        force pass writes velocity while thermo fields read ambient/gas and
        write pressure/density. This candidate keeps the exact ping-pong
        roles and density reduction input, but pays one dispatch/barrier.
        """
        if not world.force_sources:
            raise RuntimeError("force/thermo fusion requires at least one force source")
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["force_sources_thermo_fields_density_buffer"]
        force_data = self._force_source_upload(world)
        self._write_dynamic_buffer(ctx, resources, "force_sources", force_data)
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = float(dt)
        program["gas_cell_size"].value = float(world.gas_cell_size)
        program["force_count"].value = int(len(world.force_sources))
        program["species_count"].value = int(world.gas_concentration.shape[0])
        resources.force_sources.bind_to_storage_buffer(binding=0)
        resources.species_force_params.bind_to_storage_buffer(binding=1)
        resources.velocity_pong.use(location=0)
        resources.ambient_ping.use(location=1)
        resources.gas_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_ping.bind_to_image(0, read=False, write=True)
        resources.thermo_pressure.bind_to_image(1, read=False, write=True)
        resources.density_tex.bind_to_image(2, read=False, write=True)
        resources.density_reduce_ping.bind_to_storage_buffer(binding=5)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping
        for force in list(world.force_sources):
            force.lifetime -= dt
        world.force_sources[:] = [force for force in world.force_sources if force.lifetime > 0.0]

    def _run_divergence(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        pressure_seed = self._use_divergence_pressure_seed()
        program = self.programs["divergence_pressure_seed" if pressure_seed else "divergence"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        resources.velocity_ping.use(location=0)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.divergence.bind_to_image(1, read=False, write=True)
        if pressure_seed:
            resources.pressure_pong.bind_to_image(2, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        if pressure_seed:
            resources.pressure_ping, resources.pressure_pong = resources.pressure_pong, resources.pressure_ping
            self.last_divergence_pressure_seed_used = True

    def _run_thermo_fields(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        tree_reduction = bool(self._density_tree_reduction_enabled)
        program = self.programs["thermo_fields_density_buffer" if tree_reduction else "thermo_fields"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = world.gas_concentration.shape[0]
        resources.species_force_params.bind_to_storage_buffer(binding=0)
        resources.ambient_ping.use(location=1)
        resources.gas_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.thermo_pressure.bind_to_image(3, read=False, write=True)
        resources.density_tex.bind_to_image(4, read=False, write=True)
        if tree_reduction:
            resources.density_reduce_ping.bind_to_storage_buffer(binding=5)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_density_reduction(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        cell_count = int(world.gas_width * world.gas_height)
        if cell_count <= 0:
            return

        tree_reduction = bool(self._density_tree_reduction_enabled)
        if not tree_reduction:
            extract = self.programs["density_extract"]
            extract["grid_size"].value = (world.gas_width, world.gas_height)
            resources.density_tex.use(location=0)
            resources.density_reduce_ping.bind_to_storage_buffer(binding=0)
            extract.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)

        src = resources.density_reduce_ping
        dst = resources.density_reduce_pong
        input_count = cell_count
        reduce_program = self.programs["density_reduce_tree" if tree_reduction else "density_reduce"]
        while input_count > 1:
            values_per_group = REDUCE_LOCAL_SIZE * 2 if tree_reduction else 2
            output_count = (input_count + values_per_group - 1) // values_per_group
            reduce_program["input_count"].value = input_count
            src.bind_to_storage_buffer(binding=0)
            dst.bind_to_storage_buffer(binding=1)
            dispatch_groups = output_count if tree_reduction else (output_count + REDUCE_LOCAL_SIZE - 1) // REDUCE_LOCAL_SIZE
            reduce_program.run(dispatch_groups, 1, 1)
            self._sync_compute_writes(ctx)
            src, dst = dst, src
            input_count = output_count
        resources.density_reduce_ping = src
        resources.density_reduce_pong = dst
        self.last_density_tree_reduction_used = tree_reduction

    def _run_thermo_forces(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        self._run_density_reduction(world, resources, group_x, group_y)
        program = self.programs["thermo_forces"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["density_cell_count"].value = int(world.gas_width * world.gas_height)
        resources.velocity_pong.use(location=0)
        resources.thermo_pressure.use(location=1)
        resources.density_tex.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.density_reduce_ping.bind_to_storage_buffer(binding=4)
        resources.velocity_ping.bind_to_image(3, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)

    def _run_pressure_jacobi(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["jacobi"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        remaining_iterations = self.pressure_iterations - (1 if self._use_divergence_pressure_seed() else 0)
        remaining_iterations = max(0, remaining_iterations)
        pair_count = 0
        if self._can_run_pressure_jacobi_pair(world):
            pair_count = remaining_iterations // 2
        if pair_count > 0:
            pair_program = self.programs["jacobi_pair"]
            pair_program["grid_size"].value = (world.gas_width, world.gas_height)
            pair_group_x = (
                world.gas_width + PRESSURE_PAIR_CORE_SIZE - 1
            ) // PRESSURE_PAIR_CORE_SIZE
            pair_group_y = (
                world.gas_height + PRESSURE_PAIR_CORE_SIZE - 1
            ) // PRESSURE_PAIR_CORE_SIZE
            for _ in range(pair_count):
                resources.pressure_ping.use(location=0)
                resources.divergence.use(location=1)
                resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
                resources.pressure_pong.bind_to_image(2, read=False, write=True)
                resources.pressure_cone_shadow.bind_to_image(
                    3,
                    read=False,
                    write=True,
                )
                pair_program.run(pair_group_x, pair_group_y, 1)
                self._sync_compute_writes(ctx)
                (
                    resources.pressure_ping,
                    resources.pressure_pong,
                    resources.pressure_cone_shadow,
                ) = (
                    resources.pressure_cone_shadow,
                    resources.pressure_pong,
                    resources.pressure_ping,
                )
            self.last_pressure_jacobi_pair_used = True
        for _ in range(remaining_iterations - pair_count * 2):
            resources.pressure_ping.use(location=0)
            resources.divergence.use(location=1)
            resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
            resources.pressure_pong.bind_to_image(2, read=False, write=True)
            program.run(group_x, group_y, 1)
            self._sync_compute_writes(ctx)
            resources.pressure_ping, resources.pressure_pong = resources.pressure_pong, resources.pressure_ping

    def _can_run_pressure_jacobi_pair(self, world: "WorldEngine") -> bool:
        return bool(
            self._pressure_jacobi_pair_enabled
            and self._formal_gpu_frame(world)
            and self._use_divergence_pressure_seed()
        )

    def _use_divergence_pressure_seed(self) -> bool:
        return bool(self._divergence_pressure_seed_enabled and self.pressure_iterations > 0)

    def _can_run_pressure_projection_dependency_cone(self) -> bool:
        return bool(
            self._pressure_projection_dependency_cone_enabled
            and self._use_divergence_pressure_seed()
            and self.pressure_iterations == PRESSURE_CONE_JACOBI_ITERATIONS + 1
        )

    def _run_pressure_projection_dependency_cone(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
    ) -> None:
        program = self.programs["pressure_projection_dependency_cone"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        resources.pressure_ping.use(location=0)
        resources.divergence.use(location=1)
        resources.velocity_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.pressure_pong.bind_to_image(0, read=False, write=True)
        resources.velocity_pong.bind_to_image(1, read=False, write=True)
        resources.pressure_cone_shadow.bind_to_image(2, read=False, write=True)
        program.run(
            (world.gas_width + PRESSURE_CONE_CORE_X - 1) // PRESSURE_CONE_CORE_X,
            (world.gas_height + PRESSURE_CONE_CORE_Y - 1) // PRESSURE_CONE_CORE_Y,
            1,
        )
        self._sync_compute_writes(ctx)
        resources.pressure_ping, resources.pressure_pong, resources.pressure_cone_shadow = (
            resources.pressure_pong,
            resources.pressure_cone_shadow,
            resources.pressure_ping,
        )
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping

    def _can_run_species_terminal_cooperative(self, world: "WorldEngine") -> bool:
        species_count = int(world.gas_concentration.shape[0])
        return bool(self._species_terminal_cooperative_enabled and 0 < species_count <= MAX_SPECIES)

    def _run_projection(self, world: "WorldEngine", resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["projection"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        resources.pressure_ping.use(location=0)
        resources.velocity_ping.use(location=1)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.velocity_pong.bind_to_image(2, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.velocity_ping, resources.velocity_pong = resources.velocity_pong, resources.velocity_ping

    def _run_species(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["species"]
        ctx = world.bridge.ctx
        assert ctx is not None
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = world.gas_concentration.shape[0]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["species_count"].value = species_count
        program["air_index"].value = typed_gas_id(gas_table, "air")
        resources.species_params.bind_to_storage_buffer(binding=0)
        resources.velocity_ping.use(location=1)
        resources.gas_ping.use(location=2)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.gas_pong.bind_to_image(3, read=False, write=True)
        program.run(group_x, group_y, species_count)
        self._sync_compute_writes(ctx)
        resources.gas_ping, resources.gas_pong = resources.gas_pong, resources.gas_ping

    def _run_ambient(self, world: "WorldEngine", dt: float, resources: GPUGasResources, group_x: int, group_y: int) -> None:
        program = self.programs["ambient"]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["species_count"].value = world.gas_concentration.shape[0]
        resources.species_params.bind_to_storage_buffer(binding=0)
        resources.velocity_ping.use(location=1)
        resources.ambient_ping.use(location=2)
        resources.gas_ping.use(location=3)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.ambient_pong.bind_to_image(4, read=False, write=True)
        program.run(group_x, group_y, 1)
        self._sync_compute_writes(ctx)
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping

    def _run_species_terminal_cooperative(
        self,
        world: "WorldEngine",
        dt: float,
        resources: GPUGasResources,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas cooperative terminal requires bridge GPU resources")
        gas_table = bridge.shadow_typed_tables["gas_table"]
        species_count = int(world.gas_concentration.shape[0])
        program = self.programs["species_terminal_cooperative"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["dt"].value = dt
        program["species_count"].value = species_count
        program["air_index"].value = typed_gas_id(gas_table, "air")
        resources.species_params.bind_to_storage_buffer(binding=0)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=1)
        resources.velocity_ping.use(location=0)
        resources.gas_ping.use(location=1)
        resources.ambient_ping.use(location=2)
        resources.pressure_ping.use(location=3)
        resources.thermo_pressure.use(location=4)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        resources.gas_pong.bind_to_image(0, read=False, write=True)
        resources.ambient_pong.bind_to_image(1, read=False, write=True)
        bridge.textures["flow_velocity"].bind_to_image(2, read=False, write=True)
        bridge.textures["ambient_temperature"].bind_to_image(3, read=False, write=True)
        bridge.textures["pressure_ping"].bind_to_image(4, read=False, write=True)
        program.run(
            (world.gas_width + SPECIES_COOPERATIVE_LOCAL_X - 1) // SPECIES_COOPERATIVE_LOCAL_X,
            (world.gas_height + SPECIES_COOPERATIVE_LOCAL_Y - 1) // SPECIES_COOPERATIVE_LOCAL_Y,
            1,
        )
        self._sync_compute_writes(bridge.ctx)
        resources.gas_ping, resources.gas_pong = resources.gas_pong, resources.gas_ping
        resources.ambient_ping, resources.ambient_pong = resources.ambient_pong, resources.ambient_ping
        bridge.mark_gpu_authoritative(
            "flow_velocity",
            "ambient_temperature",
            "pressure_ping",
            "gas_concentration",
        )
        self.last_species_terminal_cooperative_used = True

    def _download_outputs(self, world: "WorldEngine", resources: GPUGasResources) -> None:
        velocity = np.frombuffer(resources.velocity_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width, 2))
        ambient = np.frombuffer(resources.ambient_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        pressure = np.frombuffer(resources.pressure_ping.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        thermo_pressure = np.frombuffer(resources.thermo_pressure.read(), dtype="f4").reshape((world.gas_height, world.gas_width))
        gas = np.frombuffer(resources.gas_ping.read(), dtype="f4").reshape((world.gas_concentration.shape[0], world.gas_height, world.gas_width))
        world.flow_velocity[:] = velocity
        world.ambient_temperature[:] = ambient
        world.pressure_ping[:] = pressure + thermo_pressure
        world.gas_concentration[:] = np.maximum(gas, 0.0)

    def _publish_bridge_outputs(
        self,
        world: "WorldEngine",
        resources: GPUGasResources,
        group_x: int,
        group_y: int,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU gas pipeline requires bridge GPU resources for authoritative gas state")
        program = self.programs["publish_bridge"]
        program["grid_size"].value = (world.gas_width, world.gas_height)
        program["species_count"].value = int(world.gas_concentration.shape[0])
        program["velocity_tex"].value = 0
        program["ambient_tex"].value = 1
        program["pressure_tex"].value = 2
        program["thermo_pressure_tex"].value = 3
        program["gas_tex"].value = 4
        resources.velocity_ping.use(location=0)
        resources.ambient_ping.use(location=1)
        resources.pressure_ping.use(location=2)
        resources.thermo_pressure.use(location=3)
        resources.gas_ping.use(location=4)
        bridge.textures["flow_velocity"].bind_to_image(5, read=False, write=True)
        bridge.textures["ambient_temperature"].bind_to_image(6, read=False, write=True)
        bridge.textures["pressure_ping"].bind_to_image(7, read=False, write=True)
        bridge.buffers["gas_concentration"].bind_to_storage_buffer(binding=8)
        program.run(group_x, group_y, int(world.gas_concentration.shape[0]))
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "flow_velocity",
            "ambient_temperature",
            "pressure_ping",
            "gas_concentration",
        )

    # ``_sync_compute_writes`` is inherited from :class:`GPUPipelineBase`
    # (uses the default ``_barrier_bits`` set: image access + texture fetch +
    # shader storage).
