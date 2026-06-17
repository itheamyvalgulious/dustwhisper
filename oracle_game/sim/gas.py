from __future__ import annotations

import numpy as np

from oracle_game.gpu import typed_gas_id
from oracle_game.sim.gpu_gas import GPUGasPipeline
from oracle_game.sim.utils import (
    advect_scalar,
    advect_vector,
    centered_gradient_x,
    centered_gradient_y,
    cross_neighbor_sum,
    expand_bool_mask,
    laplace,
    tile_mask_to_gas_mask,
)


GAS_ACTIVITY_EPSILON = 1e-4


class GasSolver:
    def __init__(self, pressure_iterations: int = 12) -> None:
        self.pressure_iterations = pressure_iterations
        self.gpu_pipeline = GPUGasPipeline(pressure_iterations=pressure_iterations)
        self.last_backend = "idle"
        self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_solve_gas_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_pressure_iterations = 0
        self.last_force_source_count_before = 0
        self.last_force_source_count_after = 0
        self.last_velocity_changed = False
        self.last_ambient_changed = False
        self.last_gas_changed = False
        self.last_pressure_range = np.zeros((2,), dtype=np.float32)
        self.last_ambient_range = np.zeros((2,), dtype=np.float32)
        self.last_flow_speed_range = np.zeros((2,), dtype=np.float32)
        self.last_species_total_concentration = np.zeros((0,), dtype=np.float32)
        self.last_species_active_concentration = np.zeros((0,), dtype=np.float32)

    def step(self, world: "WorldEngine", dt: float) -> None:
        self.reset_runtime_state(world)
        world.bridge.sync_rule_tables(world)
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "gas")
        formal_gpu_frame = (
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        active_scheduler_gpu_authoritative = (
            formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        if formal_gpu_frame and not active_scheduler_gpu_authoritative:
            world._require_gpu_stage("active scheduler gas solve masks")
        if active_scheduler_gpu_authoritative:
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            solve_tile_mask = self._solve_tile_mask(world)
        if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
            return
        solve_gas_mask = self._solve_gas_mask(world, solve_tile_mask)
        self.last_solve_tile_mask = solve_tile_mask.copy()
        self.last_solve_gas_mask = solve_gas_mask.copy()
        self.last_pressure_iterations = int(self.pressure_iterations)
        self.last_force_source_count_before = int(len(world.force_sources))
        if formal_gpu_frame:
            previous_flow_velocity = None
            previous_ambient_temperature = None
            previous_gas_concentration = None
        else:
            previous_flow_velocity = world.flow_velocity.copy()
            previous_ambient_temperature = world.ambient_temperature.copy()
            previous_gas_concentration = world.gas_concentration.copy()
        if gpu_available:
            self.gpu_pipeline.step(world, dt, solve_gas_mask=solve_gas_mask)
            self.last_backend = "gpu"
        else:
            world._require_cpu_oracle_backend("gas")
            self.last_backend = "cpu"
            self._step_cpu_active(world, dt, solve_gas_mask)
        self.last_force_source_count_after = int(len(world.force_sources))
        if gpu_available and not self.gpu_pipeline.last_cpu_mirror_downloaded:
            velocity_changed = ambient_changed = gas_changed = True
            self.last_velocity_changed = True
            self.last_ambient_changed = True
            self.last_gas_changed = True
        else:
            assert previous_flow_velocity is not None
            assert previous_ambient_temperature is not None
            assert previous_gas_concentration is not None
            velocity_changed, ambient_changed, gas_changed = self._finalize_runtime_state(
                world,
                solve_tile_mask,
                solve_gas_mask,
                previous_flow_velocity,
                previous_ambient_temperature,
                previous_gas_concentration,
            )
        self._refresh_active_regions(
            world,
            solve_tile_mask,
            velocity_changed=False if active_scheduler_gpu_authoritative else velocity_changed,
            ambient_changed=False if active_scheduler_gpu_authoritative else ambient_changed,
            gas_changed=False if active_scheduler_gpu_authoritative else gas_changed,
        )

    def _step_cpu_active(self, world: "WorldEngine", dt: float, solve_gas_mask: np.ndarray) -> None:
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = min(int(world.gas_concentration.shape[0]), int(gas_table.shape[0]))
        air_id = typed_gas_id(gas_table, "air")
        velocity = world.flow_velocity.copy()
        advected_velocity = advect_vector(world.flow_velocity, world.flow_velocity, dt)
        velocity[solve_gas_mask] = advected_velocity[solve_gas_mask] * 0.995
        for force in list(world.force_sources):
            self._apply_force_to_flow(world, velocity, force, dt, solve_gas_mask=solve_gas_mask)
            force.lifetime -= dt
        world.force_sources[:] = [force for force in world.force_sources if force.lifetime > 0.0]

        thermo_pressure, thermo_density = self._pressure_and_density_fields(world)
        mean_density = float(np.mean(thermo_density))
        thermo_pressure = np.where(solve_gas_mask, thermo_pressure, 0.0).astype(np.float32, copy=False)
        thermo_density = np.where(solve_gas_mask, thermo_density, 0.0).astype(np.float32, copy=False)
        self._apply_pressure_density_forces(
            velocity,
            thermo_pressure,
            thermo_density,
            dt,
            solve_gas_mask=solve_gas_mask,
            mean_density=mean_density,
        )
        divergence = self._divergence(velocity, solve_gas_mask=solve_gas_mask)
        pressure = np.zeros_like(world.pressure_ping)
        for _ in range(self.pressure_iterations):
            next_pressure = pressure.copy()
            jacobi = (cross_neighbor_sum(pressure) - divergence) * 0.25
            next_pressure[solve_gas_mask] = jacobi[solve_gas_mask]
            pressure = next_pressure
        pressure_grad_x = centered_gradient_x(pressure)
        pressure_grad_y = centered_gradient_y(pressure)
        velocity[..., 0][solve_gas_mask] -= pressure_grad_x[solve_gas_mask]
        velocity[..., 1][solve_gas_mask] -= pressure_grad_y[solve_gas_mask]
        updated_pressure = thermo_pressure + pressure
        world.pressure_ping[solve_gas_mask] = updated_pressure[solve_gas_mask]
        world.flow_velocity[solve_gas_mask] = velocity[solve_gas_mask]

        for species_id in range(species_count):
            species = gas_table[species_id]
            field = world.gas_concentration[species_id]
            species_velocity = velocity.copy()
            species_velocity[..., 1] -= float(species["buoyancy"])
            advected = advect_scalar(field, species_velocity, dt)
            diffused = advected + float(species["diffusion_rate"]) * dt * laplace(advected)
            diffused *= max(0.0, 1.0 - float(species["decay_rate"]) * dt)
            diffused = np.maximum(diffused, 0.0)
            if species_id == air_id:
                diffused = np.maximum(diffused, 0.3)
            world.gas_concentration[species_id, solve_gas_mask] = diffused[solve_gas_mask]

        ambient = advect_scalar(world.ambient_temperature, velocity, dt)
        ambient += 0.08 * (cross_neighbor_sum(world.ambient_temperature) - 4.0 * ambient)
        for species_id in range(species_count):
            ambient += world.gas_concentration[species_id] * float(gas_table[species_id]["temperature_coupling"]) * 0.01
        world.ambient_temperature[solve_gas_mask] = ambient[solve_gas_mask]

    def _solve_tile_mask(self, world: "WorldEngine") -> np.ndarray:
        active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
        if not world.force_sources:
            return expand_bool_mask(active_tiles, radius=1)
        seeded_tiles = active_tiles.copy()
        tile_size = world.active.tile_size
        for force in world.force_sources:
            x0 = max(0, int(np.floor(force.x - force.radius)))
            y0 = max(0, int(np.floor(force.y - force.radius)))
            x1 = min(world.width, int(np.ceil(force.x + force.radius + 1.0)))
            y1 = min(world.height, int(np.ceil(force.y + force.radius + 1.0)))
            if x0 >= x1 or y0 >= y1:
                continue
            tile_x0 = max(0, x0 // tile_size)
            tile_y0 = max(0, y0 // tile_size)
            tile_x1 = min(world.active.tile_width, (x1 + tile_size - 1) // tile_size)
            tile_y1 = min(world.active.tile_height, (y1 + tile_size - 1) // tile_size)
            seeded_tiles[tile_y0:tile_y1, tile_x0:tile_x1] = True
        return expand_bool_mask(seeded_tiles, radius=1)

    def _solve_gas_mask(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> np.ndarray:
        return tile_mask_to_gas_mask(
            solve_tile_mask,
            tile_size=world.active.tile_size,
            gas_cell_size=world.gas_cell_size,
            width=world.width,
            height=world.height,
            gas_width=world.gas_width,
            gas_height=world.gas_height,
        )

    def _finalize_runtime_state(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
        previous_flow_velocity: np.ndarray,
        previous_ambient_temperature: np.ndarray,
        previous_gas_concentration: np.ndarray,
    ) -> tuple[bool, bool, bool]:
        velocity_changed = bool(
            np.any(np.abs(world.flow_velocity[solve_gas_mask] - previous_flow_velocity[solve_gas_mask]) > GAS_ACTIVITY_EPSILON)
        )
        ambient_changed = bool(
            np.any(
                np.abs(world.ambient_temperature[solve_gas_mask] - previous_ambient_temperature[solve_gas_mask])
                > GAS_ACTIVITY_EPSILON
            )
        )
        gas_changed = bool(
            np.any(
                np.abs(world.gas_concentration[:, solve_gas_mask] - previous_gas_concentration[:, solve_gas_mask])
                > GAS_ACTIVITY_EPSILON
            )
        )
        self.last_velocity_changed = velocity_changed
        self.last_ambient_changed = ambient_changed
        self.last_gas_changed = gas_changed
        self.last_pressure_range = self._masked_range(world.pressure_ping, solve_gas_mask)
        self.last_ambient_range = self._masked_range(world.ambient_temperature, solve_gas_mask)
        self.last_flow_speed_range = self._masked_range(np.linalg.norm(world.flow_velocity, axis=-1), solve_gas_mask)
        species_count = int(world.gas_concentration.shape[0])
        self.last_species_total_concentration = world.gas_concentration.reshape((species_count, -1)).sum(axis=1, dtype=np.float64).astype(np.float32)
        if np.any(solve_gas_mask):
            self.last_species_active_concentration = world.gas_concentration[:, solve_gas_mask].sum(axis=1, dtype=np.float64).astype(np.float32)
        else:
            self.last_species_active_concentration = np.zeros((species_count,), dtype=np.float32)
        return velocity_changed, ambient_changed, gas_changed

    def _refresh_active_regions(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        *,
        velocity_changed: bool,
        ambient_changed: bool,
        gas_changed: bool,
    ) -> None:
        if not velocity_changed and not ambient_changed and not gas_changed:
            return
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int]] = []
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            x0 = int(tile_x) * tile_size
            y0 = int(tile_y) * tile_size
            rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size)))
        world._mark_active_rects_runtime(rects)

    def _pressure_and_density_fields(self, world: "WorldEngine") -> tuple[np.ndarray, np.ndarray]:
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = min(int(world.gas_concentration.shape[0]), int(gas_table.shape[0]))
        pressure_coeff = np.zeros_like(world.ambient_temperature, dtype=np.float32)
        density = np.zeros_like(world.ambient_temperature, dtype=np.float32)
        for species_id in range(species_count):
            concentration = world.gas_concentration[species_id]
            pressure_coeff += concentration * float(gas_table[species_id]["pressure_factor"])
            density += concentration * float(gas_table[species_id]["density_factor"])
        temperature = np.maximum(world.ambient_temperature, 0.1)
        pressure = temperature * pressure_coeff
        return pressure.astype(np.float32), density.astype(np.float32)

    def _apply_pressure_density_forces(
        self,
        velocity: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
        dt: float,
        *,
        solve_gas_mask: np.ndarray,
        mean_density: float,
    ) -> None:
        grad_x = centered_gradient_x(pressure)
        grad_y = centered_gradient_y(pressure)
        inv_density = 1.0 / np.maximum(density, 0.25)
        velocity[..., 0][solve_gas_mask] -= grad_x[solve_gas_mask] * inv_density[solve_gas_mask] * dt * 0.2
        velocity[..., 1][solve_gas_mask] -= grad_y[solve_gas_mask] * inv_density[solve_gas_mask] * dt * 0.2
        velocity[..., 1][solve_gas_mask] += (density[solve_gas_mask] - mean_density) * dt * 0.08

    def _divergence(self, velocity: np.ndarray, *, solve_gas_mask: np.ndarray) -> np.ndarray:
        divergence = centered_gradient_x(velocity[..., 0]) + centered_gradient_y(velocity[..., 1])
        divergence[~solve_gas_mask] = 0.0
        return divergence

    def _apply_force_to_flow(
        self,
        world: "WorldEngine",
        velocity: np.ndarray,
        force: "ForceSource",
        dt: float,
        *,
        solve_gas_mask: np.ndarray | None = None,
    ) -> None:
        grid_x = force.x / world.gas_cell_size
        grid_y = force.y / world.gas_cell_size
        height, width = velocity.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        dist2 = (xx - grid_x) ** 2 + (yy - grid_y) ** 2
        influence = np.exp(-dist2 / max(force.radius / world.gas_cell_size, 1.0) ** 2)
        delta_x = influence * force.direction[0] * force.strength * dt
        delta_y = influence * force.direction[1] * force.strength * dt
        if solve_gas_mask is None:
            velocity[..., 0] += delta_x
            velocity[..., 1] += delta_y
            return
        velocity[..., 0][solve_gas_mask] += delta_x[solve_gas_mask]
        velocity[..., 1][solve_gas_mask] += delta_y[solve_gas_mask]

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        if world is None:
            self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_solve_gas_mask = np.zeros((0, 0), dtype=np.bool_)
            species_count = 0
        else:
            self.last_solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
            self.last_solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
            species_count = int(world.gas_concentration.shape[0])
        self.last_pressure_iterations = 0
        self.last_force_source_count_before = 0
        self.last_force_source_count_after = 0
        self.last_velocity_changed = False
        self.last_ambient_changed = False
        self.last_gas_changed = False
        self.last_pressure_range = np.zeros((2,), dtype=np.float32)
        self.last_ambient_range = np.zeros((2,), dtype=np.float32)
        self.last_flow_speed_range = np.zeros((2,), dtype=np.float32)
        self.last_species_total_concentration = np.zeros((species_count,), dtype=np.float32)
        self.last_species_active_concentration = np.zeros((species_count,), dtype=np.float32)

    def runtime_snapshot(self) -> dict[str, np.ndarray | int | bool]:
        return {
            "solve_tile_mask": self.last_solve_tile_mask.copy(),
            "solve_gas_mask": self.last_solve_gas_mask.copy(),
            "pressure_iterations": int(self.last_pressure_iterations),
            "force_source_count_before": int(self.last_force_source_count_before),
            "force_source_count_after": int(self.last_force_source_count_after),
            "velocity_changed": bool(self.last_velocity_changed),
            "ambient_changed": bool(self.last_ambient_changed),
            "gas_changed": bool(self.last_gas_changed),
            "pressure_range": self.last_pressure_range.copy(),
            "ambient_range": self.last_ambient_range.copy(),
            "flow_speed_range": self.last_flow_speed_range.copy(),
            "species_total_concentration": self.last_species_total_concentration.copy(),
            "species_active_concentration": self.last_species_active_concentration.copy(),
        }

    def _masked_range(self, field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if field.size == 0 or mask.size == 0 or not np.any(mask):
            return np.zeros((2,), dtype=np.float32)
        masked = np.asarray(field[mask], dtype=np.float32)
        return np.array([float(masked.min(initial=0.0)), float(masked.max(initial=0.0))], dtype=np.float32)
