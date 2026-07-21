from __future__ import annotations

from contextlib import contextmanager
import math
import time
from typing import Any, NamedTuple

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
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        self._formal_full_active_mask_reuse_enabled = True
        # Keep solve-mask construction canonical; this narrower candidate only
        # aliases formal changed masks after the GPU pass, avoiding three
        # full-resolution CPU copies without changing the authoritative mask.
        # Formal GPU frames already have canonical solve masks in hand. Reuse
        # those arrays as the changed-mask views instead of copying each full
        # resolution mask three times; non-formal/partial paths retain the
        # copy helper below.
        self._formal_changed_mask_alias_enabled = True
        self._formal_full_active_mask_cache_signature: tuple[int, ...] | None = None
        self._formal_full_active_mask_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.reset_runtime_state()

    def _full_active_mask_cache(
        self,
        world: "WorldEngine",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        signature = (
            int(world.active.tile_height),
            int(world.active.tile_width),
            int(world.height),
            int(world.width),
            int(world.gas_height),
            int(world.gas_width),
        )
        if (
            self._formal_full_active_mask_cache is None
            or self._formal_full_active_mask_cache_signature != signature
        ):
            masks = (
                np.ones(signature[0:2], dtype=np.bool_),
                np.ones(signature[2:4], dtype=np.bool_),
                np.ones(signature[4:6], dtype=np.bool_),
            )
            for mask in masks:
                mask.flags.writeable = False
            self._formal_full_active_mask_cache_signature = signature
            self._formal_full_active_mask_cache = masks
        return self._formal_full_active_mask_cache

    def _profile_enabled(self, world: "WorldEngine") -> bool:
        return bool(getattr(world, "profile_passes_enabled", False))

    def reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    def _record_pass_profile_entry(self, name: str, cpu_ms: float, gpu_ms: float | None) -> None:
        entry = {"name": str(name), "cpu_ms": float(cpu_ms), "gpu_ms": float(gpu_ms) if gpu_ms is not None else None}
        self.last_pass_profile["passes"].append(entry)
        summary = self.last_pass_profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
        summary["count"] += 1
        summary["cpu_ms"] += float(cpu_ms)
        if gpu_ms is not None:
            summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + float(gpu_ms)

    @contextmanager
    def _profile_pass(self, world: "WorldEngine", name: str):
        if not self._profile_enabled(world):
            yield
            return
        ctx = world.bridge.ctx if bool(getattr(world, "profile_passes_sync", False)) else None
        if ctx is not None:
            ctx.finish()
        start = time.perf_counter()
        try:
            yield
        finally:
            if ctx is not None:
                ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_pass_profile_entry(str(name), elapsed_ms, elapsed_ms if ctx is not None else None)

    def _merge_gpu_pass_profile(self, world: "WorldEngine") -> None:
        if not self._profile_enabled(world):
            return
        gpu_profile = getattr(self.gpu_pipeline, "last_pass_profile", None)
        if not isinstance(gpu_profile, dict):
            return
        for entry in gpu_profile.get("passes", []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", ""))
            if not name:
                continue
            if name.startswith("optics."):
                name = name[len("optics.") :]
            gpu_ms = entry.get("gpu_ms")
            self._record_pass_profile_entry(
                name,
                float(entry.get("cpu_ms") or 0.0),
                float(gpu_ms) if gpu_ms is not None else None,
            )

    def step(self, world: "WorldEngine") -> None:
        self.gpu_pipeline.last_reaction_latch_clear_fused = False
        self.gpu_pipeline.last_full_active_mask_hydration_elision_used = False
        if self._profile_enabled(world):
            self.reset_pass_profile()
            self.gpu_pipeline.reset_pass_profile()
        with self._profile_pass(world, "optics_runtime_table_prep"):
            self._reset_frame_runtime_state(world)
            if getattr(world, "simulation_backend", "") == "gpu" and bool(getattr(world, "_world_simulation_frame_active", False)):
                world._formal_gpu_frame_has_light_dose = None
            world.bridge.sync_rule_tables(world)
            self._load_shadow_runtime(world)
            has_gpu_reaction_emitters = self._has_gpu_reaction_emitters(world)
        with self._profile_pass(world, "optics_emitter_collection"):
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
                    world_x, world_y = world._buffer_to_world_position(
                        (int(record["origin"][0]), int(record["origin"][1]))
                    )
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
        with self._profile_pass(world, "optics_backend_prepare"):
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
        with self._profile_pass(world, "optics_solve_tile_mask"):
            reuse_full_active_masks = bool(
                active_scheduler_gpu_authoritative
                and self._formal_full_active_mask_reuse_enabled
            )
            if reuse_full_active_masks:
                solve_tile_mask, solve_cell_mask, solve_gas_mask = self._full_active_mask_cache(world)
            elif active_scheduler_gpu_authoritative:
                solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
            else:
                solve_tile_mask = self._solve_tile_mask(world, emitters)
                if has_gpu_reaction_emitters:
                    solve_tile_mask = np.ones_like(solve_tile_mask, dtype=np.bool_)
            has_solve_tiles = bool(np.any(solve_tile_mask))
        with self._profile_pass(world, "optics_mask_expansion"):
            if not reuse_full_active_masks:
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
        with self._profile_pass(world, "optics_mask_snapshots"):
            if reuse_full_active_masks:
                self.last_solve_tile_mask = solve_tile_mask
                self.last_solve_cell_mask = solve_cell_mask
                self.last_solve_gas_mask = solve_gas_mask
            elif formal_gpu_frame and self._formal_changed_mask_alias_enabled:
                # These masks are immutable for the remainder of a formal GPU
                # frame; retaining them avoids three full-resolution copies.
                self.last_solve_tile_mask = solve_tile_mask
                self.last_solve_cell_mask = solve_cell_mask
                self.last_solve_gas_mask = solve_gas_mask
            else:
                self.last_solve_tile_mask = solve_tile_mask.copy()
                self.last_solve_cell_mask = solve_cell_mask.copy()
                self.last_solve_gas_mask = solve_gas_mask.copy()
        with self._profile_pass(world, "optics_previous_output_snapshot"):
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
            if (emitters or has_gpu_reaction_emitters) and has_solve_tiles:
                self.gpu_pipeline.step(world, emitters, solve_cell_mask=solve_cell_mask, solve_gas_mask=solve_gas_mask)
                self._merge_gpu_pass_profile(world)
                if formal_gpu_frame:
                    # True means the GPU light-dose guard was produced for reactions; it is not a CPU dose verdict.
                    world._formal_gpu_frame_has_light_dose = True
                    world._gpu_optics_outputs_clear = False
            elif formal_gpu_frame:
                if not bool(getattr(world, "_gpu_optics_outputs_clear", False)):
                    with self._profile_pass(world, "optics_gpu_clear"):
                        self.gpu_pipeline.clear_outputs(world)
                    self._merge_gpu_pass_profile(world)
                    world._gpu_optics_outputs_clear = True
                else:
                    with self._profile_pass(world, "optics_gpu_guard_clear"):
                        self.gpu_pipeline.clear_light_dose_guard(world)
                    self._merge_gpu_pass_profile(world)
                # False only means optics outputs are known clear and the GPU guard was reset.
                world._formal_gpu_frame_has_light_dose = False
            if emitters and self._needs_secondary_branches(world, emitters):
                with self._profile_pass(world, "optics_secondary_branch_count"):
                    self.last_secondary_branch_count = self._count_secondary_branch_seeds(world, emitters)
            self.last_backend = "gpu"
        else:
            world._require_cpu_oracle_backend("optics")
            self.last_backend = "cpu"
            with self._profile_pass(world, "optics_cpu_trace"):
                for emitter in emitters:
                    self._trace_emitter(world, emitter, solve_cell_mask=solve_cell_mask)
        if self.last_backend != "gpu":
            with self._profile_pass(world, "optics_cpu_compose"):
                self._compose_visible_illumination(world)
        if not active_scheduler_gpu_authoritative and (emitters or has_gpu_reaction_emitters) and has_solve_tiles:
            with self._profile_pass(world, "optics_active_tile_marking"):
                self._mark_tiles_from_mask(world, solve_tile_mask)
        with self._profile_pass(world, "optics_changed_mask_updates"):
            if formal_gpu_frame and (
                reuse_full_active_masks or self._formal_changed_mask_alias_enabled
            ):
                self.last_visible_changed_mask = solve_cell_mask
                self.last_cell_dose_changed_mask = solve_cell_mask
                self.last_gas_dose_changed_mask = solve_gas_mask
                if formal_gpu_frame and self._formal_changed_mask_alias_enabled and not reuse_full_active_masks:
                    # Preserve the child-pass accounting contract without
                    # synchronizing the GPU three times for no-op copies.
                    self._record_aliased_changed_mask_passes(world)
            elif formal_gpu_frame:
                self._copy_direct_changed_masks(world, solve_cell_mask, solve_gas_mask)
            elif gpu_available and not self.gpu_pipeline.last_cpu_mirror_downloaded:
                self._copy_direct_changed_masks(world, solve_cell_mask, solve_gas_mask)
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
        with self._profile_pass(world, "optics_runtime_totals"):
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
                material_id = int(world.material_id[cy, cx])
                if material_id == 0:
                    self._deposit_light(world, light, cx, cy, energy, absorption=0.0, solve_cell_mask=solve_cell_mask)
                    energy *= 0.97
                    continue
                optics = self._optics_runtime(material_id, light.light_type_id)
                self._deposit_light(
                    world,
                    light,
                    cx,
                    cy,
                    energy,
                    absorption=float(optics.absorption),
                    solve_cell_mask=solve_cell_mask,
                )
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
                material_id = int(world.material_id[cy, cx])
                if material_id == 0:
                    self._deposit_light(world, light, cx, cy, energy, absorption=0.0, solve_cell_mask=solve_cell_mask)
                    energy *= 0.97
                    continue
                optics = self._optics_runtime(material_id, light.light_type_id)
                self._deposit_light(
                    world,
                    light,
                    cx,
                    cy,
                    energy,
                    absorption=float(optics.absorption),
                    solve_cell_mask=solve_cell_mask,
                )
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
        absorption: float,
        solve_cell_mask: np.ndarray,
    ) -> None:
        if not solve_cell_mask[cy, cx]:
            return
        dose_channel = int(light.dose_channel_id)
        if dose_channel < 0 or dose_channel >= world.cell_optical_dose.shape[0]:
            return
        if self._visible_exposure.shape != world.cell_optical_dose.shape:
            self._visible_exposure = np.zeros_like(world.cell_optical_dose)
        self._visible_exposure[dose_channel, cy, cx] += energy
        absorbed = energy * max(0.0, float(absorption))
        if absorbed > 0.0:
            world.cell_optical_dose[dose_channel, cy, cx] += absorbed
        gy, gx = world.cell_to_gas(cy, cx)
        world.gas_optical_dose[dose_channel, gy, gx] += energy * 0.08

    def _compose_visible_illumination(self, world: "WorldEngine") -> None:
        frame = np.zeros_like(world.visible_illumination)
        visible_exposure = (
            self._visible_exposure
            if self._visible_exposure.shape == world.cell_optical_dose.shape
            else np.zeros_like(world.cell_optical_dose)
        )
        for light in self._light_runtimes:
            dose_channel = int(light.dose_channel_id)
            if dose_channel < 0 or dose_channel >= world.cell_optical_dose.shape[0]:
                continue
            cell_visible = visible_exposure[dose_channel]
            gas_haze = np.repeat(
                np.repeat(world.gas_optical_dose[dose_channel], world.gas_cell_size, axis=0),
                world.gas_cell_size,
                axis=1,
            )[: world.height, : world.width]
            if not np.any(cell_visible > 0.0) and not np.any(gas_haze > 0.0):
                continue
            style = str(light.render_style or "diffuse")
            tint = RENDER_STYLE_TINTS.get(style, RENDER_STYLE_TINTS["diffuse"])
            base_scale = float(RENDER_STYLE_BASE_SCALE.get(style, RENDER_STYLE_BASE_SCALE["diffuse"]))
            accent_scale = float(RENDER_STYLE_ACCENT_SCALE.get(style, RENDER_STYLE_ACCENT_SCALE["diffuse"]))
            haze_scale = float(RENDER_STYLE_HAZE_SCALE.get(style, RENDER_STYLE_HAZE_SCALE["diffuse"]))
            accent = VISUAL_CHANNEL_ACCENTS[int(light.visual_channel) % len(VISUAL_CHANNEL_ACCENTS)]
            color = np.asarray(light.color, dtype=np.float32)
            frame += (
                cell_visible[..., None] * (color[None, None, :] * tint[None, None, :] * base_scale)
                + cell_visible[..., None] * (accent[None, None, :] * accent_scale)
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
        return count_buffer is not None

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

    def _copy_direct_changed_masks(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
    ) -> None:
        with self._profile_pass(world, "optics_changed_mask_visible"):
            np.copyto(self.last_visible_changed_mask, solve_cell_mask)
        with self._profile_pass(world, "optics_changed_mask_cell_dose"):
            np.copyto(self.last_cell_dose_changed_mask, solve_cell_mask)
        with self._profile_pass(world, "optics_changed_mask_gas_dose"):
            np.copyto(self.last_gas_dose_changed_mask, solve_gas_mask)

    def _record_aliased_changed_mask_passes(self, world: "WorldEngine") -> None:
        if not self._profile_enabled(world):
            return
        for name in (
            "optics_changed_mask_visible",
            "optics_changed_mask_cell_dose",
            "optics_changed_mask_gas_dose",
        ):
            self._record_pass_profile_entry(name, 0.0, None)

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
        self._reset_shadow_runtime_cache()
        self._reset_frame_runtime_state(world)

    def _reset_shadow_runtime_cache(self) -> None:
        self._shadow_runtime_signature: tuple[int, ...] | None = None
        self._light_runtimes: list[_LightRuntime] = []
        self._light_runtime_by_name: dict[str, _LightRuntime] = {}
        self._branching_light_ids: set[int] = set()
        self._optics_runtime_by_pair: dict[tuple[int, int], _OpticsRuntime] = {}

    def _reset_frame_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        tile_shape = (0, 0) if world is None else (world.active.tile_height, world.active.tile_width)
        cell_shape = (0, 0) if world is None else (world.height, world.width)
        gas_shape = (0, 0) if world is None else (world.gas_height, world.gas_width)
        formal_gpu_frame = (
            world is not None
            and getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        reuse_full_active_masks = bool(
            formal_gpu_frame
            and self._formal_full_active_mask_reuse_enabled
            and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        self.last_runtime_backend = "idle"
        if reuse_full_active_masks:
            full_tile, full_cell, full_gas = self._full_active_mask_cache(world)
            self.last_solve_tile_mask = full_tile
            self.last_solve_cell_mask = full_cell
            self.last_solve_gas_mask = full_gas
            self.last_visible_changed_mask = full_cell
            self.last_cell_dose_changed_mask = full_cell
            self.last_gas_dose_changed_mask = full_gas
        else:
            self.last_solve_tile_mask = np.zeros(tile_shape, dtype=np.bool_)
            self.last_solve_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
            self.last_solve_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
            self.last_visible_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
            self.last_cell_dose_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
            self.last_gas_dose_changed_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_emitter_origin_mask = np.zeros(cell_shape, dtype=np.bool_)
        dose_channels = 0 if world is None else int(world.cell_optical_dose.shape[0])
        exposure_shape = (0, 0, 0) if formal_gpu_frame else (dose_channels, *cell_shape)
        self._visible_exposure = np.zeros(exposure_shape, dtype=np.float32)
        self.last_emitters: list[dict[str, object]] = []
        self.last_public_emitters: list[dict[str, object]] = []
        self.last_emitter_count = 0
        self.last_secondary_branch_count = 0
        self.last_visible_energy_total = 0.0
        self.last_cell_dose_total = 0.0
        self.last_gas_dose_total = 0.0

    def _shadow_runtime_cache_signature(self, world: "WorldEngine") -> tuple[int, ...]:
        table_generations = world.bridge.table_generations
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        light_table = world.bridge.shadow_typed_tables.get("light_table")
        optics_table = world.bridge.shadow_typed_tables.get("optics_table")
        material_rows = -1 if material_table is None else int(material_table.shape[0])
        light_rows = -1 if light_table is None else int(light_table.shape[0])
        optics_rows = -1 if optics_table is None else int(optics_table.shape[0])
        return (
            int(table_generations.get("materials", 0)),
            int(table_generations.get("lights", 0)),
            int(table_generations.get("optics", 0)),
            material_rows,
            light_rows,
            optics_rows,
        )

    def _load_shadow_runtime(self, world: "WorldEngine") -> None:
        signature = self._shadow_runtime_cache_signature(world)
        if signature == self._shadow_runtime_signature:
            return
        light_table = world.bridge.shadow_typed_tables.get("light_table")
        optics_table = world.bridge.shadow_typed_tables.get("optics_table")
        light_runtimes: list[_LightRuntime] = []
        light_runtime_by_name: dict[str, _LightRuntime] = {}
        branching_light_ids: set[int] = set()
        optics_runtime_by_pair: dict[tuple[int, int], _OpticsRuntime] = {}
        if light_table is None:
            light_snapshot = world._shadow_light_type_payload()
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
                light_runtimes.append(runtime)
                light_runtime_by_name[light_name] = runtime
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
                light_runtimes.append(runtime)
                light_runtime_by_name[light_name] = runtime
        if optics_table is None:
            optics_snapshot = world.serialize_material_optics_table()
            for optics in optics_snapshot:
                material_name = str(optics.get("material_name", ""))
                light_name = str(optics.get("light_type", ""))
                material_id = int(world._shadow_material_id_by_name(material_name))
                light = light_runtime_by_name.get(light_name)
                if material_id <= 0 or light is None:
                    continue
                runtime = _OpticsRuntime(
                    absorption=float(optics.get("absorption", 0.0)),
                    scattering=float(optics.get("scattering", 0.0)),
                    refraction=float(optics.get("refraction", 0.0)),
                )
                optics_runtime_by_pair[(int(material_id), light.light_type_id)] = runtime
                if runtime.scattering > 0.05 or runtime.refraction > 0.05:
                    branching_light_ids.add(light.light_type_id)
        else:
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
                optics_runtime_by_pair[(material_id, light_id)] = runtime
                if runtime.scattering > 0.05 or runtime.refraction > 0.05:
                    branching_light_ids.add(light_id)
        self._light_runtimes = light_runtimes
        self._light_runtime_by_name = light_runtime_by_name
        self._branching_light_ids = branching_light_ids
        self._optics_runtime_by_pair = optics_runtime_by_pair
        self._shadow_runtime_signature = signature

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
