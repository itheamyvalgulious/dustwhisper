from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from oracle_game.gpu import LIGHT_RENDER_STYLE_IDS
from oracle_game.sim.gpu_optics import GPUOpticsPipeline
from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask, tile_mask_to_gas_mask


OPTICS_ACTIVITY_EPSILON = 1e-4
VISUAL_CHANNEL_ACCENTS = (
    np.array((1.0, 0.2, 0.1), dtype=np.float32),
    np.array((0.25, 1.0, 0.2), dtype=np.float32),
    np.array((0.1, 0.3, 1.0), dtype=np.float32),
)
RENDER_STYLE_TINTS: dict[str, np.ndarray] = {
    "diffuse": np.array((1.0, 1.0, 1.0), dtype=np.float32),
    "holy": np.array((0.85, 1.2, 0.95), dtype=np.float32),
    "chaos": np.array((1.2, 0.55, 0.35), dtype=np.float32),
    "magic": np.array((0.55, 1.05, 1.25), dtype=np.float32),
}
RENDER_STYLE_BASE_SCALE: dict[str, float] = {
    "diffuse": 0.11,
    "holy": 0.10,
    "chaos": 0.09,
    "magic": 0.09,
}
RENDER_STYLE_ACCENT_SCALE: dict[str, float] = {
    "diffuse": 0.02,
    "holy": 0.03,
    "chaos": 0.025,
    "magic": 0.025,
}
RENDER_STYLE_HAZE_SCALE: dict[str, float] = {
    "diffuse": 0.04,
    "holy": 0.07,
    "chaos": 0.05,
    "magic": 0.06,
}
LIGHT_RENDER_STYLE_NAMES = {value: name for name, value in LIGHT_RENDER_STYLE_IDS.items()}


class _RayState(NamedTuple):
    x: float
    y: float
    dx: float
    dy: float
    energy: float
    bounce: int


class _LightRuntime(NamedTuple):
    light_type_id: int
    color: np.ndarray
    visual_channel: int
    default_range: int
    max_bounce: int
    dose_channel_id: int
    render_style: str


class _OpticsRuntime(NamedTuple):
    absorption: float
    scattering: float
    refraction: float


class OpticsSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUOpticsPipeline()
        self.last_backend = "idle"
        self.reset_runtime_state()

    def step(self, world: "WorldEngine") -> None:
        self.reset_runtime_state(world)
        if getattr(world, "simulation_backend", "") == "gpu" and bool(getattr(world, "_world_simulation_frame_active", False)):
            world._formal_gpu_frame_has_light_dose = None
        world.bridge.sync_rule_tables(world)
        self._load_shadow_runtime(world)
        has_gpu_reaction_emitters = self._has_gpu_reaction_emitters(world)
        emitters: list[dict[str, object]] = []
        min_world_x = int(world.paging.origin_x)
        min_world_y = int(world.paging.origin_y)
        max_world_x = min_world_x + int(world.width)
        max_world_y = min_world_y + int(world.height)
        for emitter in list(world.persistent_emitters) + list(world.emitters):
            record = dict(emitter)
            if "world_origin" in record:
                world_x = int(record["world_origin"][0])
                world_y = int(record["world_origin"][1])
            else:
                world_x, world_y = world._buffer_to_world_position((int(record["origin"][0]), int(record["origin"][1])))
                record["world_origin"] = (int(world_x), int(world_y))
            if world_x < min_world_x or world_x >= max_world_x or world_y < min_world_y or world_y >= max_world_y:
                continue
            record["origin"] = world._world_to_buffer_clamped(world_x, world_y)
            emitters.append(record)
        emitters.sort(key=self._emitter_sort_key)
        world.emitters.clear()
        self.last_emitter_count = int(len(emitters))
        if emitters:
            for emitter in emitters:
                origin_x, origin_y = emitter["origin"]
                if world.in_bounds(int(origin_x), int(origin_y)):
                    self.last_emitter_origin_mask[int(origin_y), int(origin_x)] = True
            self.last_emitters = [dict(emitter) for emitter in emitters]
            self.last_public_emitters = [
                {
                    "light_type": str(emitter["light_type"]),
                    "origin": [int(emitter["world_origin"][0]), int(emitter["world_origin"][1])],
                    "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
                    "spread": float(emitter["spread"]),
                    "strength": float(emitter["strength"]),
                    "range_cells": int(emitter["range_cells"]),
                }
                for emitter in emitters
            ]
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "optics")
        formal_gpu_frame = (
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        active_scheduler_gpu_authoritative = (
            formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        if formal_gpu_frame and not active_scheduler_gpu_authoritative:
            world._require_gpu_stage("active scheduler optics solve masks")
        if active_scheduler_gpu_authoritative:
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            solve_tile_mask = self._solve_tile_mask(world, emitters)
            if has_gpu_reaction_emitters:
                solve_tile_mask = np.ones_like(solve_tile_mask, dtype=np.bool_)
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
        if formal_gpu_frame:
            previous_visible = None
            previous_cell_dose = None
            previous_gas_dose = None
        else:
            previous_visible = world.visible_illumination.copy()
            previous_cell_dose = world.cell_optical_dose.copy()
            previous_gas_dose = world.gas_optical_dose.copy()
            world.visible_illumination.fill(0.0)
            world.cell_optical_dose.fill(0.0)
            world.gas_optical_dose.fill(0.0)
            world.bridge.clear_gpu_authoritative(
                "light",
                "visible_illumination",
                "cell_optical_dose",
                "gas_optical_dose",
            )
        if gpu_available:
            if (emitters or has_gpu_reaction_emitters) and np.any(solve_tile_mask):
                self.gpu_pipeline.step(world, emitters, solve_cell_mask=solve_cell_mask, solve_gas_mask=solve_gas_mask)
                if formal_gpu_frame:
                    world._formal_gpu_frame_has_light_dose = True
                    world._gpu_optics_outputs_clear = False
            elif formal_gpu_frame:
                if not bool(getattr(world, "_gpu_optics_outputs_clear", False)):
                    self.gpu_pipeline.clear_outputs(world)
                    world._gpu_optics_outputs_clear = True
                world._formal_gpu_frame_has_light_dose = False
            if emitters and self._needs_secondary_branches(world, emitters):
                self.last_secondary_branch_count = self._count_secondary_branch_seeds(world, emitters)
            self.last_backend = "gpu"
        else:
            world._require_cpu_oracle_backend("optics")
            self.last_backend = "cpu"
            for emitter in emitters:
                self._trace_emitter(world, emitter, solve_cell_mask=solve_cell_mask)
        if self.last_backend != "gpu":
            self._compose_visible_illumination(world)
        if not active_scheduler_gpu_authoritative and (emitters or has_gpu_reaction_emitters) and np.any(solve_tile_mask):
            self._mark_tiles_from_mask(world, solve_tile_mask)
        if formal_gpu_frame:
            self.last_visible_changed_mask[solve_cell_mask] = True
            self.last_cell_dose_changed_mask[solve_cell_mask] = True
            self.last_gas_dose_changed_mask[solve_gas_mask] = True
        elif gpu_available and not self.gpu_pipeline.last_cpu_mirror_downloaded:
            self.last_visible_changed_mask[solve_cell_mask] = True
            self.last_cell_dose_changed_mask[solve_cell_mask] = True
            self.last_gas_dose_changed_mask[solve_gas_mask] = True
        else:
            assert previous_visible is not None
            assert previous_cell_dose is not None
            assert previous_gas_dose is not None
            self._refresh_active_regions(
                world,
                solve_tile_mask,
                solve_cell_mask,
                solve_gas_mask,
                previous_visible,
                previous_cell_dose,
                previous_gas_dose,
            )
        self.last_runtime_backend = self.last_backend
        if formal_gpu_frame:
            self.last_visible_energy_total = 0.0
            self.last_cell_dose_total = 0.0
            self.last_gas_dose_total = 0.0
        else:
            self.last_visible_energy_total = float(np.sum(world.visible_illumination))
            self.last_cell_dose_total = float(np.sum(world.cell_optical_dose))
            self.last_gas_dose_total = float(np.sum(world.gas_optical_dose))

    def _trace_emitter(self, world: "WorldEngine", emitter: dict[str, object], *, solve_cell_mask: np.ndarray) -> None:
        light = self._light_runtime_by_name.get(str(emitter["light_type"]))
        if light is None:
            return
        origin_x, origin_y = emitter["origin"]
        rays = self._build_rays(emitter["direction"], float(emitter["spread"]))
        for ray_dx, ray_dy in rays:
            stack = [_RayState(origin_x + 0.5, origin_y + 0.5, ray_dx, ray_dy, float(emitter["strength"]), 0)]
            self._trace_branch_stack(world, light, int(emitter["range_cells"]), stack, solve_cell_mask=solve_cell_mask)

    def _trace_emitter_primary(self, world: "WorldEngine", emitter: dict[str, object], *, solve_cell_mask: np.ndarray) -> None:
        light = self._light_runtime_by_name.get(str(emitter["light_type"]))
        if light is None:
            return
        origin_x, origin_y = emitter["origin"]
        range_cells = int(emitter["range_cells"])
        for ray_dx, ray_dy in self._build_rays(emitter["direction"], float(emitter["spread"])):
            x = origin_x + 0.5
            y = origin_y + 0.5
            energy = float(emitter["strength"])
            for _ in range(range_cells):
                if energy <= 0.02:
                    break
                x += ray_dx
                y += ray_dy
                cx = int(x)
                cy = int(y)
                if not world.in_bounds(cx, cy):
                    break
                self._deposit_light(world, light, cx, cy, energy, solve_cell_mask=solve_cell_mask)
                material_id = int(world.material_id[cy, cx])
                if material_id == 0:
                    energy *= 0.97
                    continue
                optics = self._optics_runtime(material_id, light.light_type_id)
                energy, _ = self._interact_with_material(x, y, ray_dx, ray_dy, energy, light.max_bounce, optics, light.max_bounce)

    def _needs_secondary_branches(self, world: "WorldEngine", emitters: list[dict[str, object]]) -> bool:
        if not emitters or not np.any(world.material_id != 0):
            return False
        for emitter in emitters:
            light = self._light_runtime_by_name.get(str(emitter["light_type"]))
            if light is None or light.max_bounce <= 0:
                continue
            if light.light_type_id in self._branching_light_ids:
                return True
        return False

    def _run_secondary_branch_supplement(
        self,
        world: "WorldEngine",
        emitters: list[dict[str, object]],
        *,
        solve_cell_mask: np.ndarray,
    ) -> None:
        for emitter in emitters:
            light = self._light_runtime_by_name.get(str(emitter["light_type"]))
            if light is None:
                continue
            if light.max_bounce <= 0:
                continue
            secondary = self._seed_secondary_branches(world, emitter, light)
            self.last_secondary_branch_count += int(len(secondary))
            if secondary:
                self._trace_branch_stack(
                    world,
                    light,
                    int(emitter["range_cells"]),
                    secondary,
                    solve_cell_mask=solve_cell_mask,
                )

    def _count_secondary_branch_seeds(self, world: "WorldEngine", emitters: list[dict[str, object]]) -> int:
        count = 0
        for emitter in emitters:
            light = self._light_runtime_by_name.get(str(emitter["light_type"]))
            if light is None or light.max_bounce <= 0:
                continue
            count += len(self._seed_secondary_branches(world, emitter, light))
        return count

    def _seed_secondary_branches(
        self,
        world: "WorldEngine",
        emitter: dict[str, object],
        light: object,
    ) -> list[_RayState]:
        origin_x, origin_y = emitter["origin"]
        branches: list[_RayState] = []
        for ray_dx, ray_dy in self._build_rays(emitter["direction"], float(emitter["spread"])):
            x = origin_x + 0.5
            y = origin_y + 0.5
            energy = float(emitter["strength"])
            for _ in range(int(emitter["range_cells"])):
                if energy <= 0.02:
                    break
                x += ray_dx
                y += ray_dy
                cx = int(x)
                cy = int(y)
                if not world.in_bounds(cx, cy):
                    break
                material_id = int(world.material_id[cy, cx])
                if material_id == 0:
                    energy *= 0.97
                    continue
                optics = self._optics_runtime(material_id, light.light_type_id)
                energy, spawned = self._interact_with_material(x, y, ray_dx, ray_dy, energy, 0, optics, light.max_bounce)
                branches.extend(spawned)
        return branches

    def _trace_branch_stack(
        self,
        world: "WorldEngine",
        light: object,
        range_cells: int,
        stack: list[_RayState],
        *,
        solve_cell_mask: np.ndarray,
    ) -> None:
        while stack:
            ray = stack.pop()
            if ray.energy <= 0.02 or ray.bounce > light.max_bounce:
                continue
            x, y, dx, dy, energy, bounce = ray
            for _ in range(range_cells):
                if energy <= 0.02:
                    break
                x += dx
                y += dy
                cx = int(x)
                cy = int(y)
                if not world.in_bounds(cx, cy):
                    break
                self._deposit_light(world, light, cx, cy, energy, solve_cell_mask=solve_cell_mask)
                material_id = int(world.material_id[cy, cx])
                if material_id == 0:
                    energy *= 0.97
                    continue
                optics = self._optics_runtime(material_id, light.light_type_id)
                energy, spawned = self._interact_with_material(x, y, dx, dy, energy, bounce, optics, light.max_bounce)
                stack.extend(spawned)

    def _interact_with_material(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        energy: float,
        bounce: int,
        optics: object,
        max_bounce: int,
    ) -> tuple[float, list[_RayState]]:
        absorbed = energy * optics.absorption
        scattered = energy * optics.scattering
        refracted = energy * optics.refraction
        remaining = max(0.0, energy - absorbed - scattered * 0.5 - refracted * 0.25)
        spawned: list[_RayState] = []
        if bounce < max_bounce:
            if scattered > 0.05:
                angle = math.atan2(dy, dx) + 0.6
                spawned.append(_RayState(x, y, math.cos(angle), math.sin(angle), scattered * 0.75, bounce + 1))
            if refracted > 0.05:
                angle = math.atan2(dy, dx) - 0.2
                spawned.append(_RayState(x, y, math.cos(angle), math.sin(angle), refracted * 0.8, bounce + 1))
        return (remaining, spawned)

    def _deposit_light(
        self,
        world: "WorldEngine",
        light: object,
        cx: int,
        cy: int,
        energy: float,
        *,
        solve_cell_mask: np.ndarray,
    ) -> None:
        if not solve_cell_mask[cy, cx]:
            return
        world.cell_optical_dose[light.dose_channel_id, cy, cx] += energy
        gy, gx = world.cell_to_gas(cy, cx)
        world.gas_optical_dose[light.dose_channel_id, gy, gx] += energy * 0.08

    def _compose_visible_illumination(self, world: "WorldEngine") -> None:
        frame = np.zeros_like(world.visible_illumination)
        for light in self._light_runtimes:
            dose_channel = int(light.dose_channel_id)
            if dose_channel < 0 or dose_channel >= world.cell_optical_dose.shape[0]:
                continue
            cell_dose = world.cell_optical_dose[dose_channel]
            gas_haze = np.repeat(
                np.repeat(world.gas_optical_dose[dose_channel], world.gas_cell_size, axis=0),
                world.gas_cell_size,
                axis=1,
            )[: world.height, : world.width]
            if not np.any(cell_dose > 0.0) and not np.any(gas_haze > 0.0):
                continue
            style = str(light.render_style or "diffuse")
            tint = RENDER_STYLE_TINTS.get(style, RENDER_STYLE_TINTS["diffuse"])
            base_scale = float(RENDER_STYLE_BASE_SCALE.get(style, RENDER_STYLE_BASE_SCALE["diffuse"]))
            accent_scale = float(RENDER_STYLE_ACCENT_SCALE.get(style, RENDER_STYLE_ACCENT_SCALE["diffuse"]))
            haze_scale = float(RENDER_STYLE_HAZE_SCALE.get(style, RENDER_STYLE_HAZE_SCALE["diffuse"]))
            accent = VISUAL_CHANNEL_ACCENTS[int(light.visual_channel) % len(VISUAL_CHANNEL_ACCENTS)]
            color = np.asarray(light.color, dtype=np.float32)
            frame += (
                cell_dose[..., None] * (color[None, None, :] * tint[None, None, :] * base_scale)
                + cell_dose[..., None] * (accent[None, None, :] * accent_scale)
                + gas_haze[..., None] * (tint[None, None, :] * haze_scale)
            )
        world.visible_illumination[:] = frame

    def _solve_tile_mask(self, world: "WorldEngine", emitters: list[dict[str, object]]) -> np.ndarray:
        active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
        if not emitters:
            return expand_bool_mask(active_tiles, radius=1)
        seeded_tiles = active_tiles.copy()
        tile_size = world.active.tile_size
        for emitter in emitters:
            light = self._light_runtime_by_name.get(str(emitter["light_type"]))
            bounce_count = max(0, int(light.max_bounce)) if light is not None else 0
            reach = int(np.ceil(float(emitter["range_cells"]) * float(bounce_count + 1)))
            origin_x, origin_y = emitter["origin"]
            x0 = max(0, int(origin_x) - reach)
            y0 = max(0, int(origin_y) - reach)
            x1 = min(world.width, int(origin_x) + reach + 1)
            y1 = min(world.height, int(origin_y) + reach + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            tile_x0 = max(0, x0 // tile_size)
            tile_y0 = max(0, y0 // tile_size)
            tile_x1 = min(world.active.tile_width, (x1 + tile_size - 1) // tile_size)
            tile_y1 = min(world.active.tile_height, (y1 + tile_size - 1) // tile_size)
            seeded_tiles[tile_y0:tile_y1, tile_x0:tile_x1] = True
        return expand_bool_mask(seeded_tiles, radius=1)

    def _has_gpu_reaction_emitters(self, world: "WorldEngine") -> bool:
        if not (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
            and "reaction_light_emitter_count" in world.bridge.gpu_authoritative_resources
        ):
            return False
        count_buffer = world.bridge.buffers.get("reaction_light_emitter_count")
        if count_buffer is None:
            return False
        try:
            count = int(np.frombuffer(count_buffer.read(size=4), dtype=np.uint32, count=1)[0])
        except Exception:
            return True
        return count > 0

    def _emitter_sort_key(self, emitter: dict[str, object]) -> tuple[object, ...]:
        origin = emitter.get("world_origin", emitter.get("origin", (0, 0)))
        direction = emitter.get("direction", (0.0, 0.0))
        return (
            int(origin[1]),
            int(origin[0]),
            str(emitter.get("light_type", "")),
            round(float(direction[0]), 8),
            round(float(direction[1]), 8),
            round(float(emitter.get("spread", 0.0)), 8),
            round(float(emitter.get("strength", 0.0)), 8),
            int(emitter.get("range_cells", 0)),
        )

    def _mark_tiles_from_mask(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> None:
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int]] = []
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            x0 = int(tile_x) * tile_size
            y0 = int(tile_y) * tile_size
            rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size)))
        world._mark_active_rects_runtime(rects)

    def _refresh_active_regions(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
        previous_visible: np.ndarray,
        previous_cell_dose: np.ndarray,
        previous_gas_dose: np.ndarray,
    ) -> None:
        if not np.any(solve_tile_mask):
            return
        if np.any(solve_cell_mask):
            self.last_visible_changed_mask[solve_cell_mask] = np.any(
                np.abs(world.visible_illumination[solve_cell_mask] - previous_visible[solve_cell_mask]) > OPTICS_ACTIVITY_EPSILON,
                axis=-1,
            )
            self.last_cell_dose_changed_mask[solve_cell_mask] = np.any(
                np.abs(world.cell_optical_dose[:, solve_cell_mask] - previous_cell_dose[:, solve_cell_mask]) > OPTICS_ACTIVITY_EPSILON,
                axis=0,
            )
        if np.any(solve_gas_mask):
            self.last_gas_dose_changed_mask[solve_gas_mask] = np.any(
                np.abs(world.gas_optical_dose[:, solve_gas_mask] - previous_gas_dose[:, solve_gas_mask]) > OPTICS_ACTIVITY_EPSILON,
                axis=0,
            )
        visible_changed = bool(np.any(self.last_visible_changed_mask))
        cell_dose_changed = bool(np.any(self.last_cell_dose_changed_mask))
        gas_dose_changed = bool(np.any(self.last_gas_dose_changed_mask))
        if not visible_changed and not cell_dose_changed and not gas_dose_changed:
            return
        self._mark_tiles_from_mask(world, solve_tile_mask)

    def _build_rays(self, direction: tuple[float, float], spread: float) -> list[tuple[float, float]]:
        base_x, base_y = direction
        if abs(base_x) < 1e-5 and abs(base_y) < 1e-5:
            return [
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (0.707, 0.707),
                (-0.707, 0.707),
                (0.707, -0.707),
                (-0.707, -0.707),
            ]
        angle = math.atan2(base_y, base_x)
        offsets = (-spread, 0.0, spread)
        return [(math.cos(angle + offset), math.sin(angle + offset)) for offset in offsets]

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        tile_shape = (0, 0) if world is None else (world.active.tile_height, world.active.tile_width)
        cell_shape = (0, 0) if world is None else (world.height, world.width)
        gas_shape = (0, 0) if world is None else (world.gas_height, world.gas_width)
        self._light_runtimes: list[_LightRuntime] = []
        self._light_runtime_by_name: dict[str, _LightRuntime] = {}
        self._branching_light_ids: set[int] = set()
        self._optics_runtime_by_pair: dict[tuple[int, int], _OpticsRuntime] = {}
        self.last_runtime_backend = "idle"
        self.last_solve_tile_mask = np.zeros(tile_shape, dtype=np.bool_)
        self.last_solve_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_solve_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_visible_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_cell_dose_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_gas_dose_changed_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_emitter_origin_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_emitters: list[dict[str, object]] = []
        self.last_public_emitters: list[dict[str, object]] = []
        self.last_emitter_count = 0
        self.last_secondary_branch_count = 0
        self.last_visible_energy_total = 0.0
        self.last_cell_dose_total = 0.0
        self.last_gas_dose_total = 0.0

    def _load_shadow_runtime(self, world: "WorldEngine") -> None:
        light_table = world.bridge.shadow_typed_tables.get("light_table")
        optics_table = world.bridge.shadow_typed_tables.get("optics_table")
        light_snapshot = world._shadow_light_type_payload()
        optics_snapshot = world.serialize_material_optics_table()
        self._light_runtimes = []
        self._light_runtime_by_name = {}
        self._branching_light_ids = set()
        self._optics_runtime_by_pair = {}
        if light_table is None:
            for light in light_snapshot:
                shadow_light = world._coerce_light_type_def(light)
                light_id = int(shadow_light.light_type_id)
                light_name = str(shadow_light.name)
                if not light_name:
                    continue
                runtime = _LightRuntime(
                    light_type_id=light_id,
                    color=np.asarray(shadow_light.color, dtype=np.float32),
                    visual_channel=int(shadow_light.visual_channel),
                    default_range=int(shadow_light.default_range),
                    max_bounce=int(shadow_light.max_bounce),
                    dose_channel_id=int(shadow_light.dose_channel_id),
                    render_style=str(shadow_light.render_style or "diffuse"),
                )
                self._light_runtimes.append(runtime)
                self._light_runtime_by_name[light_name] = runtime
        else:
            for row in light_table:
                if int(row["name_hash"]) == 0:
                    continue
                light_id = int(row["light_type_id"])
                if light_id < 0 or light_id >= len(light_table):
                    continue
                light_name = world._shadow_light_name(light_id)
                if not light_name:
                    continue
                runtime = _LightRuntime(
                    light_type_id=light_id,
                    color=np.asarray(row["color"], dtype=np.float32),
                    visual_channel=int(row["visual_channel"]),
                    default_range=int(row["default_range"]),
                    max_bounce=int(row["max_bounce"]),
                    dose_channel_id=int(row["dose_channel_id"]),
                    render_style=LIGHT_RENDER_STYLE_NAMES.get(int(row["render_style_id"]), "diffuse"),
                )
                self._light_runtimes.append(runtime)
                self._light_runtime_by_name[light_name] = runtime
        if optics_table is None:
            for optics in optics_snapshot:
                material_name = str(optics.get("material_name", ""))
                light_name = str(optics.get("light_type", ""))
                material_id = int(world._shadow_material_id_by_name(material_name))
                light = self._light_runtime_by_name.get(light_name)
                if material_id <= 0 or light is None:
                    continue
                runtime = _OpticsRuntime(
                    absorption=float(optics.get("absorption", 0.0)),
                    scattering=float(optics.get("scattering", 0.0)),
                    refraction=float(optics.get("refraction", 0.0)),
                )
                self._optics_runtime_by_pair[(int(material_id), light.light_type_id)] = runtime
                if runtime.scattering > 0.05 or runtime.refraction > 0.05:
                    self._branching_light_ids.add(light.light_type_id)
            return
        for row in optics_table:
            material_id = int(row["material_id"])
            light_id = int(row["light_type_id"])
            if not world._shadow_material_row_valid(material_id):
                continue
            if not world._shadow_light_row_valid(light_id):
                continue
            runtime = _OpticsRuntime(
                absorption=float(row["absorption"]),
                scattering=float(row["scattering"]),
                refraction=float(row["refraction"]),
            )
            self._optics_runtime_by_pair[(material_id, light_id)] = runtime
            if runtime.scattering > 0.05 or runtime.refraction > 0.05:
                self._branching_light_ids.add(light_id)

    def _optics_runtime(self, material_id: int, light_id: int) -> _OpticsRuntime:
        return self._optics_runtime_by_pair.get((material_id, light_id), _OpticsRuntime(0.0, 0.0, 0.0))

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "backend": self.last_runtime_backend,
            "solve_tile_mask": self.last_solve_tile_mask.copy(),
            "solve_cell_mask": self.last_solve_cell_mask.copy(),
            "solve_gas_mask": self.last_solve_gas_mask.copy(),
            "visible_changed_mask": self.last_visible_changed_mask.copy(),
            "cell_dose_changed_mask": self.last_cell_dose_changed_mask.copy(),
            "gas_dose_changed_mask": self.last_gas_dose_changed_mask.copy(),
            "emitter_origin_mask": self.last_emitter_origin_mask.copy(),
            "emitters": [dict(emitter) for emitter in self.last_emitters],
            "public_emitters": [dict(emitter) for emitter in self.last_public_emitters],
            "emitter_count": int(self.last_emitter_count),
            "secondary_branch_count": int(self.last_secondary_branch_count),
            "visible_energy_total": float(self.last_visible_energy_total),
            "cell_dose_total": float(self.last_cell_dose_total),
            "gas_dose_total": float(self.last_gas_dose_total),
        }
