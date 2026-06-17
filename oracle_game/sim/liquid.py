from __future__ import annotations

import numpy as np

from oracle_game.sim.gpu_liquid import GPULiquidPipeline
from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask
from oracle_game.types import Phase


LIQUID_ACTIVITY_EPSILON = 1e-6
LIQUID_SOLVER_TILE_LEVEL = 1
LIQUID_SOLVER_COLUMNAR = 2


class LiquidSolver:
    """CPU analogue of the planned tile-local shared-memory liquid staging."""

    def __init__(self) -> None:
        self.gpu_pipeline = GPULiquidPipeline()
        self.last_backend = "idle"
        self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_post_tile_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_post_cell_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_vertical_seam_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_horizontal_seam_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_buoyancy_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_changed_cell_mask = np.zeros((0, 0), dtype=np.bool_)
        self.last_material_changed = False
        self.last_phase_changed = False
        self.last_velocity_changed = False
        self.last_temperature_changed = False
        self.last_integrity_changed = False
        self.last_placeholder_changed = False
        self.last_pending_placeholder_count_before = 0
        self.last_pending_placeholder_count_after = 0
        self.last_liquid_cell_count_before = 0
        self.last_liquid_cell_count_after = 0

    def step(self, world: "WorldEngine") -> None:
        self.reset_runtime_state(world)
        gpu_available = world._gpu_pipeline_available(self.gpu_pipeline, "liquid")
        formal_gpu_frame = (
            gpu_available
            and getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )
        active_scheduler_gpu_authoritative = (
            formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )
        if formal_gpu_frame and not active_scheduler_gpu_authoritative:
            world._require_gpu_stage("active scheduler liquid solve masks")
        if active_scheduler_gpu_authoritative:
            active_tiles = []
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            active_tiles = list(world.active.iter_active_tiles())
            solve_tile_mask = self._build_solve_tile_mask(world, active_tiles)
        self.last_solve_tile_mask = solve_tile_mask.copy()
        if not np.any(solve_tile_mask) and not active_scheduler_gpu_authoritative:
            return

        if formal_gpu_frame:
            pre_material_id = None
            pre_phase = None
            pre_velocity = None
            pre_temperature = None
            pre_integrity = None
            pre_island_id = None
            pre_placeholder = None
        else:
            pre_material_id = world.material_id.copy()
            pre_phase = world.phase.copy()
            pre_velocity = world.velocity.copy()
            pre_temperature = world.cell_temperature.copy()
            pre_integrity = world.integrity.copy()
            pre_island_id = world.island_id.copy()
            pre_placeholder = world.placeholder_displaced_material.copy()
            self.last_pending_placeholder_count_before = int(np.count_nonzero(pre_placeholder > 0))
            self.last_liquid_cell_count_before = int(np.count_nonzero(pre_phase == int(Phase.LIQUID)))

        post_tile_mask = expand_bool_mask(solve_tile_mask, radius=1)
        post_cell_mask = tile_mask_to_cell_mask(
            post_tile_mask,
            tile_size=world.active.tile_size,
            width=world.width,
            height=world.height,
        )
        self.last_post_tile_mask = post_tile_mask.copy()
        self.last_post_cell_mask = post_cell_mask.copy()
        self.last_vertical_seam_mask = self._vertical_seam_mask(world, post_tile_mask)
        self.last_horizontal_seam_mask = self._horizontal_seam_mask(world, post_tile_mask)
        if formal_gpu_frame:
            self.last_buoyancy_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        else:
            assert pre_material_id is not None
            assert pre_phase is not None
            self.last_buoyancy_mask = self._buoyancy_candidate_mask(world, post_cell_mask, pre_material_id, pre_phase)

        if gpu_available:
            self.gpu_pipeline.step(
                world,
                solve_tile_mask=solve_tile_mask,
                post_tile_mask=post_tile_mask,
            )
            self.last_backend = "gpu"
            if not active_scheduler_gpu_authoritative:
                self._refresh_active_tiles(world, active_tiles)
            if not formal_gpu_frame:
                self._mark_pending_placeholder_regions(world)
        else:
            world._require_cpu_oracle_backend("liquid")
            self.last_backend = "cpu"
            tile_size = world.active.tile_size
            for tile_x, tile_y in active_tiles:
                x0 = tile_x * tile_size
                y0 = tile_y * tile_size
                x1 = min(world.width, x0 + tile_size)
                y1 = min(world.height, y0 + tile_size)
                self._solve_tile(world, x0, y0, x1, y1)
            self._seam_correction(world, post_tile_mask)
            self._apply_buoyancy(world, post_cell_mask)
            self._apply_placeholder_displacement(world, post_cell_mask)
            self._mark_pending_placeholder_regions(world)

        if formal_gpu_frame:
            self.last_changed_cell_mask = post_cell_mask.copy()
            self.last_material_changed = True
            self.last_phase_changed = True
            self.last_velocity_changed = True
            self.last_temperature_changed = True
            self.last_integrity_changed = True
            self.last_placeholder_changed = True
            return

        assert pre_material_id is not None
        assert pre_phase is not None
        assert pre_velocity is not None
        assert pre_temperature is not None
        assert pre_integrity is not None
        assert pre_island_id is not None
        assert pre_placeholder is not None
        self._finalize_runtime_state(
            world,
            pre_material_id,
            pre_phase,
            pre_velocity,
            pre_temperature,
            pre_integrity,
            pre_island_id,
            pre_placeholder,
            repair_runtime_state=not gpu_available,
        )

    def _build_solve_tile_mask(self, world: "WorldEngine", active_tiles: list[tuple[int, int]]) -> np.ndarray:
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        for tile_x, tile_y in active_tiles:
            solve_tile_mask[tile_y, tile_x] = True
        return solve_tile_mask

    def _solve_tile(self, world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> None:
        local_material = world.material_id[y0:y1, x0:x1].copy()
        local_phase = world.phase[y0:y1, x0:x1].copy()
        local_flags = world.cell_flags[y0:y1, x0:x1].copy()
        local_timer = world.timer_pack[y0:y1, x0:x1].copy()
        local_temp = world.cell_temperature[y0:y1, x0:x1].copy()
        local_integrity = world.integrity[y0:y1, x0:x1].copy()
        local_velocity = world.velocity[y0:y1, x0:x1].copy()
        changed = False

        def move_cell(src_y: int, src_x: int, dst_y: int, dst_x: int) -> None:
            local_material[dst_y, dst_x] = local_material[src_y, src_x]
            local_phase[dst_y, dst_x] = local_phase[src_y, src_x]
            local_flags[dst_y, dst_x] = local_flags[src_y, src_x]
            local_timer[dst_y, dst_x] = local_timer[src_y, src_x]
            local_temp[dst_y, dst_x] = local_temp[src_y, src_x]
            local_integrity[dst_y, dst_x] = local_integrity[src_y, src_x]
            local_velocity[dst_y, dst_x] = local_velocity[src_y, src_x]
            local_material[src_y, src_x] = 0
            local_phase[src_y, src_x] = 0
            local_flags[src_y, src_x] = 0
            local_timer[src_y, src_x] = 0
            local_temp[src_y, src_x] = 0.0
            local_integrity[src_y, src_x] = 0.0
            local_velocity[src_y, src_x] = 0.0

        for ly in range(local_material.shape[0] - 2, -1, -1):
            row_material = local_material[ly].copy()
            row_phase = local_phase[ly].copy()
            below_material = local_material[ly + 1].copy()
            claimed_lateral_dest = np.zeros((local_material.shape[1],), dtype=np.bool_)
            planned_moves: list[tuple[int, int, int, int]] = []
            for lx in range(local_material.shape[1]):
                if int(row_material[lx]) == 0 or int(row_phase[lx]) != int(Phase.LIQUID):
                    continue
                liquid_solver_kind = self._material_liquid_solver_kind(world, int(row_material[lx]))
                if int(below_material[lx]) == 0:
                    planned_moves.append((ly, lx, ly + 1, lx))
                    continue
                if liquid_solver_kind == LIQUID_SOLVER_COLUMNAR:
                    continue
                if lx > 0 and int(row_material[lx - 1]) == 0 and not bool(claimed_lateral_dest[lx - 1]):
                    claimed_lateral_dest[lx - 1] = True
                    planned_moves.append((ly, lx, ly, lx - 1))
                    continue
                if lx + 1 < local_material.shape[1] and int(row_material[lx + 1]) == 0 and not bool(
                    claimed_lateral_dest[lx + 1]
                ):
                    claimed_lateral_dest[lx + 1] = True
                    planned_moves.append((ly, lx, ly, lx + 1))
            for src_y, src_x, dst_y, dst_x in planned_moves:
                move_cell(src_y, src_x, dst_y, dst_x)
                changed = True
        if changed:
            world.material_id[y0:y1, x0:x1] = local_material
            world.phase[y0:y1, x0:x1] = local_phase
            world.cell_flags[y0:y1, x0:x1] = local_flags
            world.timer_pack[y0:y1, x0:x1] = local_timer
            world.cell_temperature[y0:y1, x0:x1] = local_temp
            world.integrity[y0:y1, x0:x1] = local_integrity
            world.velocity[y0:y1, x0:x1] = local_velocity
            world._mark_active_rect_runtime(x0, y0, x1, y1)

    def _seam_correction(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> None:
        tile_size = world.active.tile_size
        vertical_boundaries: set[tuple[int, int]] = set()
        horizontal_boundaries: set[tuple[int, int]] = set()
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            if int(tile_x) > 0:
                vertical_boundaries.add((int(tile_x) * tile_size, int(tile_y)))
            if int(tile_x) + 1 < world.active.tile_width:
                vertical_boundaries.add(((int(tile_x) + 1) * tile_size, int(tile_y)))
            if int(tile_y) > 0:
                horizontal_boundaries.add((int(tile_x), int(tile_y) * tile_size))
            if int(tile_y) + 1 < world.active.tile_height:
                horizontal_boundaries.add((int(tile_x), (int(tile_y) + 1) * tile_size))
        for boundary_x, tile_y in sorted(vertical_boundaries):
            left = boundary_x - 1
            right = boundary_x
            y0 = tile_y * tile_size
            y1 = min(world.height, y0 + tile_size)
            for y in range(y0, y1):
                left_id = int(world.material_id[y, left])
                if (
                    int(world.phase[y, left]) == int(Phase.LIQUID)
                    and self._material_liquid_solver_kind(world, left_id) == LIQUID_SOLVER_TILE_LEVEL
                    and int(world.material_id[y, right]) == 0
                ):
                    world.swap_cells(left, y, right, y)
        for tile_x, boundary_y in sorted(horizontal_boundaries):
            top = boundary_y - 1
            bottom = boundary_y
            x0 = tile_x * tile_size
            x1 = min(world.width, x0 + tile_size)
            for x in range(x0, x1):
                if int(world.phase[top, x]) == int(Phase.LIQUID) and int(world.material_id[bottom, x]) == 0:
                    world.swap_cells(x, top, x, bottom)

    def _apply_buoyancy(self, world: "WorldEngine", solve_cell_mask: np.ndarray) -> None:
        pair_mask = solve_cell_mask[1:, :] | solve_cell_mask[:-1, :]
        pair_rows, pair_xs = np.nonzero(pair_mask)
        material_snapshot = world.material_id.copy()
        phase_snapshot = world.phase.copy()
        sink_swaps: list[tuple[int, int]] = []
        for pair_row, x in zip(pair_rows.tolist(), pair_xs.tolist()):
            y = pair_row + 1
            upper_id = int(material_snapshot[y - 1, x])
            lower_id = int(material_snapshot[y, x])
            if upper_id == 0 or lower_id == 0:
                continue
            if int(phase_snapshot[y - 1, x]) != int(Phase.POWDER) or int(phase_snapshot[y, x]) != int(Phase.LIQUID):
                continue
            powder_density = self._material_density(world, upper_id)
            liquid_density = self._material_density(world, lower_id)
            if powder_density > liquid_density:
                sink_swaps.append((x, y))
        for x, y in sink_swaps:
            world.swap_cells(x, y - 1, x, y)

        material_snapshot = world.material_id.copy()
        phase_snapshot = world.phase.copy()
        float_swaps: list[tuple[int, int]] = []
        pair_rows, pair_xs = np.nonzero(pair_mask)
        for pair_row, x in zip(pair_rows.tolist(), pair_xs.tolist()):
            y = pair_row + 1
            upper_id = int(material_snapshot[y - 1, x])
            lower_id = int(material_snapshot[y, x])
            if upper_id == 0 or lower_id == 0:
                continue
            if int(phase_snapshot[y - 1, x]) != int(Phase.LIQUID) or int(phase_snapshot[y, x]) != int(Phase.POWDER):
                continue
            liquid_density = self._material_density(world, upper_id)
            powder_density = self._material_density(world, lower_id)
            if powder_density < liquid_density:
                float_swaps.append((x, y))
        for x, y in float_swaps:
            world.swap_cells(x, y - 1, x, y)

    def _apply_placeholder_displacement(self, world: "WorldEngine", solve_cell_mask: np.ndarray) -> None:
        placeholder_id = self._placeholder_material_id(world)
        material_in = world.material_id.copy()
        temp_in = world.cell_temperature.copy()
        pending_in = world.placeholder_displaced_material.copy()
        pending_mask = (pending_in > 0) & solve_cell_mask
        active_rows = sorted(int(row) for row in np.unique(np.nonzero(pending_mask)[0]))
        for y in active_rows:
            x = 0
            while x < world.width:
                if int(material_in[y, x]) != placeholder_id:
                    x += 1
                    continue
                left = x
                while x < world.width and int(material_in[y, x]) == placeholder_id:
                    x += 1
                right = x
                if not np.any(pending_mask[y, left:right]):
                    continue
                seg_len = right - left
                for source_x in range(left, right):
                    displaced_material = int(pending_in[y, source_x])
                    if displaced_material <= 0:
                        continue
                    idx = source_x - left
                    for target_x, target_y, velocity in self._placeholder_candidates(
                        world,
                        material_in,
                        placeholder_id,
                        source_x,
                        y,
                        left,
                        right,
                        idx,
                        seg_len,
                    ):
                        if int(material_in[target_y, target_x]) != 0 or int(world.material_id[target_y, target_x]) != 0:
                            continue
                        world.material_id[target_y, target_x] = displaced_material
                        world.phase[target_y, target_x] = int(Phase.LIQUID)
                        world.integrity[target_y, target_x] = self._material_base_integrity(world, displaced_material)
                        world.cell_temperature[target_y, target_x] = temp_in[y, source_x]
                        world.velocity[target_y, target_x] = np.array(velocity, dtype=np.float32)
                        world.placeholder_displaced_material[y, source_x] = 0
                        world._mark_active_rect_runtime(
                            max(0, min(source_x, target_x) - 1),
                            max(0, min(y, target_y) - 1),
                            min(world.width, max(source_x, target_x) + 2),
                            min(world.height, max(y, target_y) + 2),
                        )
                        break

    def _placeholder_candidates(
        self,
        world: "WorldEngine",
        material_in: object,
        placeholder_id: int,
        source_x: int,
        source_y: int,
        left: int,
        right: int,
        idx: int,
        seg_len: int,
    ) -> list[tuple[int, int, tuple[float, float]]]:
        prefer_left = idx * 2 < seg_len
        left_target_x = left - 1 - idx
        right_target_x = right + (seg_len - 1 - idx)
        top_exposed = source_y == 0 or int(material_in[source_y - 1, source_x]) != placeholder_id
        candidates: list[tuple[int, int, tuple[float, float]]] = []
        if prefer_left:
            candidates.append((left_target_x, source_y, (-1.2, -0.15)))
            candidates.append((right_target_x, source_y, (1.2, -0.15)))
            if top_exposed:
                candidates.append((left_target_x, source_y - 1, (-0.8, -0.65)))
                candidates.append((right_target_x, source_y - 1, (0.8, -0.65)))
        else:
            candidates.append((right_target_x, source_y, (1.2, -0.15)))
            candidates.append((left_target_x, source_y, (-1.2, -0.15)))
            if top_exposed:
                candidates.append((right_target_x, source_y - 1, (0.8, -0.65)))
                candidates.append((left_target_x, source_y - 1, (-0.8, -0.65)))
        return [(tx, ty, velocity) for tx, ty, velocity in candidates if world.in_bounds(tx, ty)]

    def _mark_pending_placeholder_regions(self, world: "WorldEngine") -> None:
        ys, xs = np.nonzero(world.placeholder_displaced_material > 0)
        rects: list[tuple[int, int, int, int]] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            rects.append((max(0, x - 1), max(0, y - 1), min(world.width, x + 2), min(world.height, y + 2)))
        world._mark_active_rects_runtime(rects)

    def _refresh_active_tiles(self, world: "WorldEngine", active_tiles: list[tuple[int, int]]) -> None:
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int]] = []
        for tile_x, tile_y in active_tiles:
            x0 = max(0, (tile_x - 1) * tile_size)
            y0 = max(0, (tile_y - 1) * tile_size)
            x1 = min(world.width, (tile_x + 2) * tile_size)
            y1 = min(world.height, (tile_y + 2) * tile_size)
            rects.append((x0, y0, x1, y1))
        world._mark_active_rects_runtime(rects)

    def _vertical_seam_mask(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros((world.height, world.width), dtype=np.bool_)
        tile_size = world.active.tile_size
        vertical_boundaries: set[tuple[int, int]] = set()
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            if int(tile_x) > 0:
                vertical_boundaries.add((int(tile_x) * tile_size, int(tile_y)))
            if int(tile_x) + 1 < world.active.tile_width:
                vertical_boundaries.add(((int(tile_x) + 1) * tile_size, int(tile_y)))
        for boundary_x, tile_y in vertical_boundaries:
            left = boundary_x - 1
            right = boundary_x
            if left < 0 or right >= world.width:
                continue
            y0 = tile_y * tile_size
            y1 = min(world.height, y0 + tile_size)
            mask[y0:y1, left] = True
            mask[y0:y1, right] = True
        return mask

    def _horizontal_seam_mask(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros((world.height, world.width), dtype=np.bool_)
        tile_size = world.active.tile_size
        horizontal_boundaries: set[tuple[int, int]] = set()
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            if int(tile_y) > 0:
                horizontal_boundaries.add((int(tile_x), int(tile_y) * tile_size))
            if int(tile_y) + 1 < world.active.tile_height:
                horizontal_boundaries.add((int(tile_x), (int(tile_y) + 1) * tile_size))
        for tile_x, boundary_y in horizontal_boundaries:
            top = boundary_y - 1
            bottom = boundary_y
            if top < 0 or bottom >= world.height:
                continue
            x0 = tile_x * tile_size
            x1 = min(world.width, x0 + tile_size)
            mask[top, x0:x1] = True
            mask[bottom, x0:x1] = True
        return mask

    def _buoyancy_candidate_mask(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
        material_id: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros((world.height, world.width), dtype=np.bool_)
        pair_mask = solve_cell_mask[1:, :] | solve_cell_mask[:-1, :]
        pair_rows, pair_xs = np.nonzero(pair_mask)
        for pair_row, x in zip(pair_rows.tolist(), pair_xs.tolist()):
            y = pair_row + 1
            upper_id = int(material_id[y - 1, x])
            lower_id = int(material_id[y, x])
            if upper_id == 0 or lower_id == 0:
                continue
            upper_phase = int(phase[y - 1, x])
            lower_phase = int(phase[y, x])
            if upper_phase == int(Phase.POWDER) and lower_phase == int(Phase.LIQUID):
                powder_density = self._material_density(world, upper_id)
                liquid_density = self._material_density(world, lower_id)
                if powder_density > liquid_density:
                    mask[y - 1, x] = True
                    mask[y, x] = True
            elif upper_phase == int(Phase.LIQUID) and lower_phase == int(Phase.POWDER):
                liquid_density = self._material_density(world, upper_id)
                powder_density = self._material_density(world, lower_id)
                if powder_density < liquid_density:
                    mask[y - 1, x] = True
                    mask[y, x] = True
        return mask

    def _finalize_runtime_state(
        self,
        world: "WorldEngine",
        pre_material_id: np.ndarray,
        pre_phase: np.ndarray,
        pre_velocity: np.ndarray,
        pre_temperature: np.ndarray,
        pre_integrity: np.ndarray,
        pre_island_id: np.ndarray,
        pre_placeholder: np.ndarray,
        *,
        repair_runtime_state: bool = True,
    ) -> None:
        material_changed_mask = world.material_id != pre_material_id
        phase_changed_mask = world.phase != pre_phase
        runtime_changed_mask = material_changed_mask | phase_changed_mask
        touched_island_ids = np.unique(pre_island_id[runtime_changed_mask])
        if repair_runtime_state:
            non_placeholder_mask = runtime_changed_mask & ~self._placeholder_mask(world, world.material_id)
            world.entity_id[non_placeholder_mask] = 0
            world.placeholder_displaced_material[non_placeholder_mask] = 0
            invalid_island_mask = runtime_changed_mask & (world.island_id > 0) & (
                (world.phase != int(Phase.FALLING_ISLAND)) | (world.material_id <= 0)
            )
            world.island_id[invalid_island_mask] = 0
        world._refresh_island_records_for_ids(touched_island_ids.tolist())
        velocity_changed_mask = np.any(np.abs(world.velocity - pre_velocity) > LIQUID_ACTIVITY_EPSILON, axis=-1)
        temperature_changed_mask = np.abs(world.cell_temperature - pre_temperature) > LIQUID_ACTIVITY_EPSILON
        integrity_changed_mask = np.abs(world.integrity - pre_integrity) > LIQUID_ACTIVITY_EPSILON
        placeholder_changed_mask = world.placeholder_displaced_material != pre_placeholder
        self.last_changed_cell_mask = (
            material_changed_mask
            | phase_changed_mask
            | velocity_changed_mask
            | temperature_changed_mask
            | integrity_changed_mask
            | placeholder_changed_mask
        )
        self.last_material_changed = bool(np.any(material_changed_mask))
        self.last_phase_changed = bool(np.any(phase_changed_mask))
        self.last_velocity_changed = bool(np.any(velocity_changed_mask))
        self.last_temperature_changed = bool(np.any(temperature_changed_mask))
        self.last_integrity_changed = bool(np.any(integrity_changed_mask))
        self.last_placeholder_changed = bool(np.any(placeholder_changed_mask))
        self.last_pending_placeholder_count_after = int(np.count_nonzero(world.placeholder_displaced_material > 0))
        self.last_liquid_cell_count_after = int(np.count_nonzero(world.phase == int(Phase.LIQUID)))

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        if world is None:
            self.last_solve_tile_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_post_tile_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_post_cell_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_vertical_seam_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_horizontal_seam_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_buoyancy_mask = np.zeros((0, 0), dtype=np.bool_)
            self.last_changed_cell_mask = np.zeros((0, 0), dtype=np.bool_)
        else:
            self.last_solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
            self.last_post_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
            self.last_post_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
            self.last_vertical_seam_mask = np.zeros((world.height, world.width), dtype=np.bool_)
            self.last_horizontal_seam_mask = np.zeros((world.height, world.width), dtype=np.bool_)
            self.last_buoyancy_mask = np.zeros((world.height, world.width), dtype=np.bool_)
            self.last_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        self.last_material_changed = False
        self.last_phase_changed = False
        self.last_velocity_changed = False
        self.last_temperature_changed = False
        self.last_integrity_changed = False
        self.last_placeholder_changed = False
        self.last_pending_placeholder_count_before = 0
        self.last_pending_placeholder_count_after = 0
        self.last_liquid_cell_count_before = 0
        self.last_liquid_cell_count_after = 0

    def runtime_snapshot(self) -> dict[str, np.ndarray | int | bool]:
        return {
            "solve_tile_mask": self.last_solve_tile_mask.copy(),
            "post_tile_mask": self.last_post_tile_mask.copy(),
            "post_cell_mask": self.last_post_cell_mask.copy(),
            "vertical_seam_mask": self.last_vertical_seam_mask.copy(),
            "horizontal_seam_mask": self.last_horizontal_seam_mask.copy(),
            "buoyancy_mask": self.last_buoyancy_mask.copy(),
            "changed_cell_mask": self.last_changed_cell_mask.copy(),
            "material_changed": bool(self.last_material_changed),
            "phase_changed": bool(self.last_phase_changed),
            "velocity_changed": bool(self.last_velocity_changed),
            "temperature_changed": bool(self.last_temperature_changed),
            "integrity_changed": bool(self.last_integrity_changed),
            "placeholder_changed": bool(self.last_placeholder_changed),
            "pending_placeholder_count_before": int(self.last_pending_placeholder_count_before),
            "pending_placeholder_count_after": int(self.last_pending_placeholder_count_after),
            "liquid_cell_count_before": int(self.last_liquid_cell_count_before),
            "liquid_cell_count_after": int(self.last_liquid_cell_count_after),
        }

    def _material_table_row(self, world: "WorldEngine", material_id: int) -> np.void | None:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is None or material_id < 0 or material_id >= int(material_table.shape[0]):
            return None
        row = material_table[material_id]
        if int(row["name_hash"]) == 0:
            return None
        return row

    def _material_density(self, world: "WorldEngine", material_id: int) -> float:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return float(row["density"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return float(shadow_material.density)
        if world._shadow_has_table_payload("materials"):
            return 0.0
        if 0 <= material_id < world.material_density.shape[0]:
            return float(world.material_density[material_id])
        return 0.0

    def _material_base_integrity(self, world: "WorldEngine", material_id: int) -> float:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return float(row["base_integrity"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return float(shadow_material.base_integrity)
        if world._shadow_has_table_payload("materials"):
            return 0.0
        if 0 <= material_id < world.material_base_integrity.shape[0]:
            return float(world.material_base_integrity[material_id])
        return 0.0

    def _material_liquid_solver_kind(self, world: "WorldEngine", material_id: int) -> int:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return int(row["liquid_solver_kind_id"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return LIQUID_SOLVER_COLUMNAR if shadow_material.liquid_solver_kind == "columnar" else LIQUID_SOLVER_TILE_LEVEL
        if world._shadow_has_table_payload("materials"):
            return 0
        if 0 <= material_id < world.material_liquid_solver_kind.shape[0]:
            return int(world.material_liquid_solver_kind[material_id])
        return 0

    def _placeholder_material_id(self, world: "WorldEngine") -> int:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None:
            for row in material_table:
                if int(row["material_id"]) > 0 and int(row["name_hash"]) != 0 and int(row["render_group_id"]) == 7:
                    return int(row["material_id"])
            return 0
        return int(world.placeholder_material_id)

    def _material_is_placeholder(self, world: "WorldEngine", material_id: int) -> bool:
        row = self._material_table_row(world, material_id)
        if row is not None:
            return int(row["render_group_id"]) == 7
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return shadow_material.render_group == "placeholder" or "placeholder" in shadow_material.tags
        if world._shadow_has_table_payload("materials"):
            return False
        if 0 <= material_id < world.material_is_placeholder.shape[0]:
            return bool(world.material_is_placeholder[material_id])
        return False

    def _placeholder_mask(self, world: "WorldEngine", material_ids: np.ndarray) -> np.ndarray:
        result = np.zeros(material_ids.shape, dtype=np.bool_)
        positive_mask = material_ids > 0
        if not np.any(positive_mask):
            return result
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None:
            valid_mask = positive_mask & (material_ids < int(material_table.shape[0]))
            if np.any(valid_mask):
                result[valid_mask] = material_table["render_group_id"][material_ids[valid_mask]] == 7
            fallback_mask = positive_mask & ~valid_mask
        else:
            fallback_mask = positive_mask
        if np.any(fallback_mask):
            for material_id in np.unique(material_ids[fallback_mask]).tolist():
                result[fallback_mask & (material_ids == int(material_id))] = self._material_is_placeholder(world, int(material_id))
        return result
