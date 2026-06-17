from __future__ import annotations

import numpy as np

from oracle_game.sim.gpu_heat import GPUHeatPipeline, GPUHeatStageTargets
from oracle_game.sim.utils import cross_average, expand_bool_mask, laplace, tile_mask_to_cell_mask, tile_mask_to_gas_mask
from oracle_game.types import Phase

THERMAL_ACTIVITY_EPSILON = 1e-3
FREEZE_COLD_NEIGHBOR_THRESHOLD = 4


class HeatSolver:
    def __init__(self, ambient_iterations: int = 4) -> None:
        self.ambient_iterations = ambient_iterations
        self.gpu_pipeline = GPUHeatPipeline()
        self.last_backend = "idle"
        self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_solve_cell_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_solve_gas_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_phase_targets = np.zeros((0, 0), dtype=np.int32)
        self.last_boil_targets = np.zeros((0, 0), dtype=np.int32)
        self.last_condense_targets = np.zeros((0, 0, 0), dtype=np.bool_)
        self.last_ambient_iterations = 0
        self.last_cell_changed = False
        self.last_ambient_changed = False
        self.last_material_changed = False
        self.last_phase_changed = False
        self.last_integrity_changed = False
        self.last_gas_changed = False
        self.last_cell_temperature_range = np.zeros((2,), dtype=np.float32)
        self.last_ambient_temperature_range = np.zeros((2,), dtype=np.float32)
        self.last_integrity_range = np.zeros((2,), dtype=np.float32)
        self.last_public_phase_targets: list[dict[str, object]] = []
        self.last_public_boil_targets: list[dict[str, object]] = []
        self.last_public_condense_targets: list[dict[str, object]] = []

    def step(self, world: "WorldEngine", dt: float) -> None:
        self.reset_runtime_state(world)
        world.bridge.sync_rule_tables(world)
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "heat")
        formal_gpu_frame = (
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        active_scheduler_gpu_authoritative = (
            formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        if formal_gpu_frame and not active_scheduler_gpu_authoritative:
            world._require_gpu_stage("active scheduler heat solve masks")
        if active_scheduler_gpu_authoritative:
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            solve_tile_mask = self._solve_tile_mask(world)
        if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
            return
        solve_cell_mask = tile_mask_to_cell_mask(
            solve_tile_mask,
            tile_size=world.active.tile_size,
            width=world.width,
            height=world.height,
        )
        solve_gas_mask = tile_mask_to_gas_mask(
            solve_tile_mask,
            tile_size=world.active.tile_size,
            gas_cell_size=world.gas_cell_size,
            width=world.width,
            height=world.height,
            gas_width=world.gas_width,
            gas_height=world.gas_height,
        )
        self.last_solve_tile_mask = solve_tile_mask.copy()
        self.last_solve_cell_mask = solve_cell_mask.copy()
        self.last_solve_gas_mask = solve_gas_mask.copy()
        self.last_ambient_iterations = int(self.ambient_iterations)

        if formal_gpu_frame:
            previous_cell_temperature = None
            previous_ambient_temperature = None
            previous_material_id = None
            previous_phase = None
            previous_integrity = None
            previous_gas_concentration = None
        else:
            previous_cell_temperature = world.cell_temperature[solve_cell_mask].copy()
            previous_ambient_temperature = world.ambient_temperature[solve_gas_mask].copy()
            previous_material_id = world.material_id[solve_cell_mask].copy()
            previous_phase = world.phase[solve_cell_mask].copy()
            previous_integrity = world.integrity[solve_cell_mask].copy()
            previous_gas_concentration = world.gas_concentration[:, solve_gas_mask].copy()
        if gpu_available:
            stage_targets = self.gpu_pipeline.step(
                world,
                dt,
                solve_tile_mask=solve_tile_mask,
                ambient_iterations=self.ambient_iterations,
            )
            self.last_backend = "gpu"
        else:
            world._require_cpu_oracle_backend("heat")
            self.last_backend = "cpu"
            self._step_cpu_active(world, dt, solve_tile_mask)
            stage_targets = GPUHeatStageTargets(
                phase_targets=self._plan_phase_targets(world, solve_cell_mask=solve_cell_mask),
                boil_targets=self._plan_boil_targets(world, solve_cell_mask=solve_cell_mask),
                condense_targets=self._plan_condense_targets(world, solve_gas_mask=solve_gas_mask),
            )

        self.last_phase_targets = stage_targets.phase_targets.copy()
        self.last_boil_targets = stage_targets.boil_targets.copy()
        self.last_condense_targets = stage_targets.condense_targets.copy()
        (
            self.last_public_phase_targets,
            self.last_public_boil_targets,
            self.last_public_condense_targets,
        ) = self._capture_public_runtime_targets(
            world,
            stage_targets.phase_targets,
            stage_targets.boil_targets,
            stage_targets.condense_targets,
            solve_gas_mask=solve_gas_mask,
        )

        if gpu_available:
            self._mark_heat_target_collapse_dirty_region(
                world,
                stage_targets.phase_targets,
                stage_targets.boil_targets,
                stage_targets.condense_targets,
                solve_tile_mask,
            )
        else:
            self._apply_phase_targets(world, stage_targets.phase_targets, solve_cell_mask=solve_cell_mask)
            self._apply_boil_targets(world, stage_targets.boil_targets, solve_cell_mask=solve_cell_mask, dt=dt)
            self._apply_condense_targets(
                world,
                stage_targets.condense_targets,
                solve_gas_mask=solve_gas_mask,
            )
            self._apply_condensation(
                world,
                solve_gas_mask=solve_gas_mask,
                skip_targets=stage_targets.condense_targets,
            )
        if gpu_available and not self.gpu_pipeline.last_cpu_mirror_downloaded:
            self.last_cell_changed = True
            self.last_ambient_changed = True
            self.last_material_changed = True
            self.last_phase_changed = True
            self.last_integrity_changed = True
            self.last_gas_changed = True
            change_flags = (True, True, True, True, True, True)
        else:
            assert previous_cell_temperature is not None
            assert previous_ambient_temperature is not None
            assert previous_material_id is not None
            assert previous_phase is not None
            assert previous_integrity is not None
            assert previous_gas_concentration is not None
            change_flags = self._finalize_runtime_state(
                world,
                solve_cell_mask,
                solve_gas_mask,
                previous_cell_temperature,
                previous_ambient_temperature,
                previous_material_id,
                previous_phase,
                previous_integrity,
                previous_gas_concentration,
            )
        if active_scheduler_gpu_authoritative:
            change_flags = (False, False, False, False, False, False)
        self._refresh_active_regions(world, solve_tile_mask, *change_flags)

    def _step_cpu_active(self, world: "WorldEngine", dt: float, solve_tile_mask: np.ndarray) -> None:
        cell_mask = tile_mask_to_cell_mask(
            solve_tile_mask,
            tile_size=world.active.tile_size,
            width=world.width,
            height=world.height,
        )
        gas_mask = tile_mask_to_gas_mask(
            solve_tile_mask,
            tile_size=world.active.tile_size,
            gas_cell_size=world.gas_cell_size,
            width=world.width,
            height=world.height,
            gas_width=world.gas_width,
            gas_height=world.gas_height,
        )
        conductivity = self._material_scalar_field(world, world.material_id, "conductivity", world.material_conductivity)
        heat_capacity = np.maximum(
            self._material_scalar_field(world, world.material_id, "heat_capacity", world.material_heat_capacity),
            1.0e-4,
        )
        avg_neighbors = cross_average(world.cell_temperature)
        updated_cell_temperature = world.cell_temperature.copy()
        updated_cell_temperature[cell_mask] = (
            world.cell_temperature[cell_mask]
            + (conductivity[cell_mask] / heat_capacity[cell_mask])
            * (avg_neighbors[cell_mask] - world.cell_temperature[cell_mask])
            * dt
            * 0.35
        )
        world.cell_temperature[:] = updated_cell_temperature

        ambient = world.ambient_temperature.copy()
        for _ in range(self.ambient_iterations):
            next_ambient = ambient.copy()
            next_ambient[gas_mask] = ambient[gas_mask] + 0.08 * laplace(ambient)[gas_mask]
            ambient = next_ambient
        world.ambient_temperature[:] = ambient

        ambient_cells = world.sample_ambient_to_cells()
        exchange = (
            self._material_scalar_field(
                world,
                world.material_id,
                "ambient_exchange_rate",
                world.material_ambient_exchange,
            )
            / heat_capacity
        ) * dt
        delta = (ambient_cells - world.cell_temperature) * exchange
        world.cell_temperature[cell_mask] += delta[cell_mask]
        ambient_feedback = world.downsample_cells_to_gas(-delta * 0.02)
        world.ambient_temperature[gas_mask] += ambient_feedback[gas_mask]

    def _solve_tile_mask(self, world: "WorldEngine") -> np.ndarray:
        active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
        return expand_bool_mask(active_tiles, radius=1)

    def _plan_phase_targets(self, world: "WorldEngine", *, solve_cell_mask: np.ndarray) -> np.ndarray:
        phase_targets = np.zeros((world.height, world.width), dtype=np.int32)
        material_table = world.bridge.shadow_typed_tables["material_table"]
        for material_id in range(1, int(material_table.shape[0])):
            material = material_table[material_id]
            mask = solve_cell_mask & (world.material_id == material_id)
            if not mask.any():
                continue
            temperatures = world.cell_temperature[mask]
            melt_point = float(material["melt_point"])
            melt_to_material_id = int(material["melt_to_material_id"])
            if not np.isnan(melt_point) and melt_to_material_id > 0:
                melt_mask = mask.copy()
                melt_mask[mask] = temperatures > melt_point
                if melt_mask.any():
                    phase_targets[melt_mask] = melt_to_material_id
            freeze_to_material_id = int(material["freeze_to_material_id"])
            if not np.isnan(melt_point) and freeze_to_material_id > 0:
                freeze_mask = mask.copy()
                freeze_mask[mask] = (temperatures < melt_point) & (
                    self._cold_cross_neighbor_count(world.cell_temperature, melt_point)[mask]
                    >= FREEZE_COLD_NEIGHBOR_THRESHOLD
                )
                if freeze_mask.any() and int(material["default_phase"]) == int(Phase.LIQUID):
                    phase_targets[freeze_mask] = freeze_to_material_id
        return phase_targets

    def _cold_cross_neighbor_count(self, temperature: np.ndarray, threshold: float) -> np.ndarray:
        cold = np.asarray(temperature < threshold, dtype=np.int8)
        padded = np.pad(cold, 1, mode="edge")
        return (
            padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )

    def _plan_boil_targets(self, world: "WorldEngine", *, solve_cell_mask: np.ndarray) -> np.ndarray:
        boil_targets = np.zeros((world.height, world.width), dtype=np.int32)
        material_table = world.bridge.shadow_typed_tables["material_table"]
        for material_id in range(1, int(material_table.shape[0])):
            material = material_table[material_id]
            boil_point = float(material["boil_point"])
            target_species_id = int(material["boil_to_gas_species_id"])
            if np.isnan(boil_point) or target_species_id < 0:
                continue
            mask = solve_cell_mask & (world.material_id == material_id)
            if not mask.any():
                continue
            temperatures = world.cell_temperature[mask]
            boil_mask = mask.copy()
            boil_mask[mask] = temperatures > boil_point
            if boil_mask.any():
                boil_targets[boil_mask] = target_species_id + 1
        return boil_targets

    def _plan_condense_targets(self, world: "WorldEngine", *, solve_gas_mask: np.ndarray) -> np.ndarray:
        condense_targets = np.zeros(world.gas_concentration.shape, dtype=np.bool_)
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = min(int(condense_targets.shape[0]), int(gas_table.shape[0]))
        for species_id in range(species_count):
            species = gas_table[species_id]
            condense_point = float(species["condense_point"])
            target_material_id = int(species["condense_to_material_id"])
            if np.isnan(condense_point) or target_material_id <= 0:
                continue
            condense_targets[species_id] = solve_gas_mask & (world.ambient_temperature < condense_point) & (
                world.gas_concentration[species_id] > 0.7
            )
        return condense_targets

    def _finalize_runtime_state(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
        previous_cell_temperature: np.ndarray,
        previous_ambient_temperature: np.ndarray,
        previous_material_id: np.ndarray,
        previous_phase: np.ndarray,
        previous_integrity: np.ndarray,
        previous_gas_concentration: np.ndarray,
    ) -> tuple[bool, bool, bool, bool, bool, bool]:
        cell_changed = bool(
            np.any(np.abs(world.cell_temperature[solve_cell_mask] - previous_cell_temperature) > THERMAL_ACTIVITY_EPSILON)
        )
        ambient_changed = bool(
            np.any(np.abs(world.ambient_temperature[solve_gas_mask] - previous_ambient_temperature) > THERMAL_ACTIVITY_EPSILON)
        )
        material_changed = bool(np.any(world.material_id[solve_cell_mask] != previous_material_id))
        phase_changed = bool(np.any(world.phase[solve_cell_mask] != previous_phase))
        integrity_changed = bool(
            np.any(np.abs(world.integrity[solve_cell_mask] - previous_integrity) > THERMAL_ACTIVITY_EPSILON)
        )
        gas_changed = bool(
            np.any(
                np.abs(world.gas_concentration[:, solve_gas_mask] - previous_gas_concentration) > THERMAL_ACTIVITY_EPSILON
            )
        )
        self.last_cell_changed = cell_changed
        self.last_ambient_changed = ambient_changed
        self.last_material_changed = material_changed
        self.last_phase_changed = phase_changed
        self.last_integrity_changed = integrity_changed
        self.last_gas_changed = gas_changed
        self.last_cell_temperature_range = self._masked_range(world.cell_temperature, solve_cell_mask)
        self.last_ambient_temperature_range = self._masked_range(world.ambient_temperature, solve_gas_mask)
        self.last_integrity_range = self._masked_range(world.integrity, solve_cell_mask)
        return (
            cell_changed,
            ambient_changed,
            material_changed,
            phase_changed,
            integrity_changed,
            gas_changed,
        )

    def _refresh_active_regions(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        cell_changed: bool,
        ambient_changed: bool,
        material_changed: bool,
        phase_changed: bool,
        integrity_changed: bool,
        gas_changed: bool,
    ) -> None:
        if not np.any(solve_tile_mask):
            return
        if not (cell_changed or ambient_changed or material_changed or phase_changed or integrity_changed or gas_changed):
            return
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int]] = []
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            x0 = int(tile_x) * tile_size
            y0 = int(tile_y) * tile_size
            rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size)))
        world._mark_active_rects_runtime(rects)

    def _apply_phase_targets(
        self,
        world: "WorldEngine",
        phase_targets: np.ndarray,
        *,
        solve_cell_mask: np.ndarray,
    ) -> None:
        ys, xs = np.nonzero((phase_targets > 0) & solve_cell_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            target_id = int(phase_targets[y, x])
            if target_id <= 0:
                continue
            world.set_cell_by_id(int(x), int(y), target_id)

    def _mark_heat_target_collapse_dirty_region(
        self,
        world: "WorldEngine",
        phase_targets: np.ndarray,
        boil_targets: np.ndarray,
        condense_targets: np.ndarray,
        solve_tile_mask: np.ndarray,
    ) -> None:
        has_targets = (
            bool(phase_targets.size and np.any(phase_targets > 0))
            or bool(boil_targets.size and np.any(boil_targets > 0))
            or bool(condense_targets.size and np.any(condense_targets))
        )
        if not has_targets or not np.any(solve_tile_mask):
            return
        tile_size = world.active.tile_size
        tile_ys, tile_xs = np.nonzero(solve_tile_mask)
        if tile_ys.size == 0:
            return
        x0 = int(tile_xs.min()) * tile_size
        y0 = int(tile_ys.min()) * tile_size
        x1 = min(world.width, (int(tile_xs.max()) + 1) * tile_size)
        y1 = min(world.height, (int(tile_ys.max()) + 1) * tile_size)
        world._mark_collapse_dirty_rect(x0, y0, x1, y1)

    def _apply_boil_targets(
        self,
        world: "WorldEngine",
        boil_targets: np.ndarray,
        *,
        solve_cell_mask: np.ndarray,
        dt: float,
    ) -> None:
        ys, xs = np.nonzero((boil_targets > 0) & solve_cell_mask)
        if len(ys) == 0:
            return
        grouped: dict[int, list[tuple[int, int]]] = {}
        for y, x in zip(ys.tolist(), xs.tolist()):
            species_id = int(boil_targets[y, x]) - 1
            if species_id < 0:
                continue
            grouped.setdefault(species_id, []).append((int(y), int(x)))
        for species_id, coords in grouped.items():
            boil_cells = np.zeros((world.height, world.width), dtype=np.bool_)
            for y, x in coords:
                boil_cells[y, x] = True
                gy, gx = world.cell_to_gas(y, x)
                world.gas_concentration[species_id, gy, gx] += 0.6 * dt
            coord_y = np.array([y for y, _ in coords], dtype=np.int32)
            coord_x = np.array([x for _, x in coords], dtype=np.int32)
            world.integrity[coord_y, coord_x] -= 0.5 * dt
            evaporate_mask = boil_cells & (world.integrity <= 0)
            if evaporate_mask.any():
                world.clear_cells(evaporate_mask)

    def _apply_condense_targets(
        self,
        world: "WorldEngine",
        condense_targets: np.ndarray,
        *,
        solve_gas_mask: np.ndarray,
    ) -> None:
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = min(condense_targets.shape[0], int(gas_table.shape[0]))
        for species_id in range(species_count):
            target_material_id = int(gas_table[species_id]["condense_to_material_id"])
            if target_material_id <= 0:
                continue
            cool_mask = condense_targets[species_id] & solve_gas_mask
            if not np.any(cool_mask):
                continue
            gas_y, gas_x = np.nonzero(cool_mask)
            for gy, gx in zip(gas_y.tolist(), gas_x.tolist()):
                x0 = gx * world.gas_cell_size
                y0 = gy * world.gas_cell_size
                xs = slice(x0, min(world.width, x0 + world.gas_cell_size))
                ys = slice(y0, min(world.height, y0 + world.gas_cell_size))
                empty = world.material_id[ys, xs] == 0
                if not empty.any():
                    continue
                target_y, target_x = np.argwhere(empty)[0]
                world.set_cell_by_id(xs.start + target_x, ys.start + target_y, target_material_id)
                world.gas_concentration[species_id, gy, gx] = max(0.0, world.gas_concentration[species_id, gy, gx] - 0.6)

    def _apply_condensation(
        self,
        world: "WorldEngine",
        *,
        solve_gas_mask: np.ndarray,
        skip_targets: np.ndarray | None = None,
    ) -> None:
        gas_table = world.bridge.shadow_typed_tables["gas_table"]
        species_count = min(int(world.gas_concentration.shape[0]), int(gas_table.shape[0]))
        for species_id in range(species_count):
            species = gas_table[species_id]
            condense_point = float(species["condense_point"])
            target_material_id = int(species["condense_to_material_id"])
            if np.isnan(condense_point) or target_material_id <= 0:
                continue
            cool_mask = solve_gas_mask & (world.ambient_temperature < condense_point) & (
                world.gas_concentration[species_id] > 0.7
            )
            if skip_targets is not None and species_id < skip_targets.shape[0]:
                cool_mask &= ~skip_targets[species_id]
            if not cool_mask.any():
                continue
            gas_y, gas_x = np.nonzero(cool_mask)
            for gy, gx in zip(gas_y.tolist(), gas_x.tolist()):
                x0 = gx * world.gas_cell_size
                y0 = gy * world.gas_cell_size
                xs = slice(x0, min(world.width, x0 + world.gas_cell_size))
                ys = slice(y0, min(world.height, y0 + world.gas_cell_size))
                empty = world.material_id[ys, xs] == 0
                if not empty.any():
                    continue
                target_y, target_x = np.argwhere(empty)[0]
                world.set_cell_by_id(xs.start + target_x, ys.start + target_y, target_material_id)
                world.gas_concentration[species_id, gy, gx] = max(0.0, world.gas_concentration[species_id, gy, gx] - 0.6)

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        if world is None:
            self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_solve_cell_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_solve_gas_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_phase_targets = np.zeros((0, 0), dtype=np.int32)
            self.last_boil_targets = np.zeros((0, 0), dtype=np.int32)
            self.last_condense_targets = np.zeros((0, 0, 0), dtype=np.bool_)
        else:
            self.last_solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
            self.last_solve_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
            self.last_solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
            self.last_phase_targets = np.zeros((world.height, world.width), dtype=np.int32)
            self.last_boil_targets = np.zeros((world.height, world.width), dtype=np.int32)
            self.last_condense_targets = np.zeros(world.gas_concentration.shape, dtype=np.bool_)
        self.last_ambient_iterations = 0
        self.last_cell_changed = False
        self.last_ambient_changed = False
        self.last_material_changed = False
        self.last_phase_changed = False
        self.last_integrity_changed = False
        self.last_gas_changed = False
        self.last_cell_temperature_range = np.zeros((2,), dtype=np.float32)
        self.last_ambient_temperature_range = np.zeros((2,), dtype=np.float32)
        self.last_integrity_range = np.zeros((2,), dtype=np.float32)
        self.last_public_phase_targets = []
        self.last_public_boil_targets = []
        self.last_public_condense_targets = []

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "solve_tile_mask": self.last_solve_tile_mask.copy(),
            "solve_cell_mask": self.last_solve_cell_mask.copy(),
            "solve_gas_mask": self.last_solve_gas_mask.copy(),
            "phase_targets": self.last_phase_targets.copy(),
            "boil_targets": self.last_boil_targets.copy(),
            "condense_targets": self.last_condense_targets.copy(),
            "public_phase_targets": [dict(target) for target in self.last_public_phase_targets],
            "public_boil_targets": [dict(target) for target in self.last_public_boil_targets],
            "public_condense_targets": [dict(target) for target in self.last_public_condense_targets],
            "ambient_iterations": int(self.last_ambient_iterations),
            "cell_changed": bool(self.last_cell_changed),
            "ambient_changed": bool(self.last_ambient_changed),
            "material_changed": bool(self.last_material_changed),
            "phase_changed": bool(self.last_phase_changed),
            "integrity_changed": bool(self.last_integrity_changed),
            "gas_changed": bool(self.last_gas_changed),
            "cell_temperature_range": self.last_cell_temperature_range.copy(),
            "ambient_temperature_range": self.last_ambient_temperature_range.copy(),
            "integrity_range": self.last_integrity_range.copy(),
        }

    def _capture_public_runtime_targets(
        self,
        world: "WorldEngine",
        phase_targets: np.ndarray,
        boil_targets: np.ndarray,
        condense_targets: np.ndarray,
        *,
        solve_gas_mask: np.ndarray,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        phase_payload: list[dict[str, object]] = []
        phase_ys, phase_xs = np.nonzero(phase_targets > 0)
        for y, x in zip(phase_ys.tolist(), phase_xs.tolist()):
            target_material_id = int(phase_targets[y, x])
            world_x, world_y = world._buffer_to_world_position((int(x), int(y)))
            phase_payload.append(
                {
                    "x": int(world_x),
                    "y": int(world_y),
                    "target_material_id": target_material_id,
                }
            )

        boil_payload: list[dict[str, object]] = []
        boil_ys, boil_xs = np.nonzero(boil_targets > 0)
        for y, x in zip(boil_ys.tolist(), boil_xs.tolist()):
            world_x, world_y = world._buffer_to_world_position((int(x), int(y)))
            boil_payload.append(
                {
                    "x": int(world_x),
                    "y": int(world_y),
                    "target_species_id": int(boil_targets[y, x]) - 1,
                }
            )

        condense_payload: list[dict[str, object]] = []
        species_count = int(condense_targets.shape[0])
        for species_id in range(species_count):
            gas_y, gas_x = np.nonzero(condense_targets[species_id] & solve_gas_mask)
            target_material_id = int(world._shadow_condense_target_material_id(species_id))
            if target_material_id <= 0:
                continue
            for gy, gx in zip(gas_y.tolist(), gas_x.tolist()):
                world_gx, world_gy = world._buffer_gas_to_world_position((int(gx), int(gy)))
                condense_payload.append(
                    {
                        "gas_x": int(world_gx),
                        "gas_y": int(world_gy),
                        "species_id": int(species_id),
                        "target_material_id": target_material_id,
                    }
                )
        return phase_payload, boil_payload, condense_payload

    def _masked_range(self, field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if field.size == 0 or mask.size == 0 or not np.any(mask):
            return np.zeros((2,), dtype=np.float32)
        masked = np.asarray(field[mask], dtype=np.float32)
        return np.array([float(masked.min(initial=0.0)), float(masked.max(initial=0.0))], dtype=np.float32)

    def _material_scalar_field(
        self,
        world: "WorldEngine",
        material_ids: np.ndarray,
        field: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return fallback[material_ids].astype(np.float32, copy=True)
        values = np.zeros(material_ids.shape, dtype=np.float32)
        valid_mask = (
            (material_ids >= 0)
            & (material_ids < int(material_table.shape[0]))
            & (material_table["name_hash"][np.clip(material_ids, 0, max(0, int(material_table.shape[0]) - 1))] != 0)
        )
        if np.any(valid_mask):
            values[valid_mask] = material_table[field][material_ids[valid_mask]].astype(np.float32, copy=False)
        return values
