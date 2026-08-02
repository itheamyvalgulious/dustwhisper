from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.liquid_constants import LIQUID_SOLVER_TILE_LEVEL
from oracle_game.types import Phase

# Lateral seam spread clamps, mirroring SEAM_MAX_MOVE_PER_FRAME /
# SEAM_UNSUPPORTED_OVERHANG in shaders/liquid/seam_x.comp: a boundary move
# fills at most this many cells per frame — the same rate as the in-tile
# lateral pass, so the seam is not a faster channel that piles flickering
# chunks past the boundary.  Among the filled target cells at most the
# leading overhang may lack support from below: a short contiguous terrace
# lip past the supported edge, whose cells fall on later frames if support
# never arrives.
SEAM_MAX_MOVE_PER_FRAME = 1
SEAM_UNSUPPORTED_OVERHANG = 2
# A down-run may slide at most this many cells sideways while dropping one
# row — the in-tile lateral rate — so falling runs cannot teleport
# diagonally into long hovering lines.  Mirrors
# LIQUID_DOWNFILL_MAX_LATERAL_SHIFT in tile_solve.comp / seam_y.comp.
LIQUID_DOWNFILL_MAX_LATERAL_SHIFT = 1
# A suspended liquid cell hangs (does not fall) while it is within this
# many cells of a supported cell in its own row run — the terrace lip
# ("只有最末若干格可以没有支撑").  Anything farther falls, so hovering
# lines cannot run away; the whole front advances at the in-tile rate
# instead of dripping cell-by-cell.  Mirrors LIQUID_LIP_OVERHANG in
# tile_solve.comp.
LIQUID_LIP_OVERHANG = 2


def prepare_motion_flow_intent(solver, world: "WorldEngine") -> None:
    gpu_available = world._gpu_pipeline_available(solver.gpu_pipeline, "liquid")
    formal_gpu_frame = (
        gpu_available
        and getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )
    if not formal_gpu_frame:
        return
    if "active_tile_ttl" not in world.bridge.gpu_authoritative_resources:
        world._require_gpu_stage("active scheduler liquid pre-motion intent mask")
        return
    solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
    solver.gpu_pipeline.prepare_motion_flow_intent(world, solve_tile_mask=solve_tile_mask)


def _build_solve_tile_mask(
    solver, world: "WorldEngine", active_tiles: list[tuple[int, int]]
) -> np.ndarray:
    solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
    for tile_x, tile_y in active_tiles:
        solve_tile_mask[tile_y, tile_x] = True
    return solve_tile_mask


def _world_cell_reachable_empty(solver, world: "WorldEngine", x: int, y: int) -> bool:
    material_id = int(world.material_id[y, x])
    phase_id = int(world.phase[y, x])
    if material_id != 0:
        return False
    if phase_id in (int(Phase.LIQUID), int(Phase.FALLING_ISLAND)):
        return False
    if int(world.entity_id[y, x]) > 0:
        return False
    if int(world.placeholder_displaced_material[y, x]) > 0:
        return False
    return True


def _world_cell_is_tile_level_liquid(solver, world: "WorldEngine", x: int, y: int) -> bool:
    material_id = int(world.material_id[y, x])
    return (
        material_id > 0
        and int(world.phase[y, x]) == int(Phase.LIQUID)
        and solver._material_liquid_solver_kind(world, material_id) == LIQUID_SOLVER_TILE_LEVEL
    )


def _solve_tile(solver, world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> None:
    local_material = world.material_id[y0:y1, x0:x1].copy()
    local_phase = world.phase[y0:y1, x0:x1].copy()
    local_flags = world.cell_flags[y0:y1, x0:x1].copy()
    local_timer = world.timer_pack[y0:y1, x0:x1].copy()
    local_temp = world.cell_temperature[y0:y1, x0:x1].copy()
    local_integrity = world.integrity[y0:y1, x0:x1].copy()
    local_velocity = world.velocity[y0:y1, x0:x1].copy()
    local_island = world.island_id[y0:y1, x0:x1].copy()
    local_entity = world.entity_id[y0:y1, x0:x1].copy()
    local_displaced = world.placeholder_displaced_material[y0:y1, x0:x1].copy()
    changed = False

    def reachable_empty(local_y: int, local_x: int) -> bool:
        material_id = int(local_material[local_y, local_x])
        phase_id = int(local_phase[local_y, local_x])
        if material_id != 0:
            return False
        if phase_id in (int(Phase.LIQUID), int(Phase.FALLING_ISLAND)):
            return False
        if int(local_entity[local_y, local_x]) > 0:
            return False
        if int(local_displaced[local_y, local_x]) > 0:
            return False
        return True

    def is_liquid(local_y: int, local_x: int) -> bool:
        return int(local_material[local_y, local_x]) > 0 and int(
            local_phase[local_y, local_x]
        ) == int(Phase.LIQUID)

    def is_tile_level_liquid(local_y: int, local_x: int) -> bool:
        material_id = int(local_material[local_y, local_x])
        return (
            material_id > 0
            and int(local_phase[local_y, local_x]) == int(Phase.LIQUID)
            and solver._material_liquid_solver_kind(world, material_id) == LIQUID_SOLVER_TILE_LEVEL
        )

    def snapshot_tile_level_liquid(
        row_material: np.ndarray, row_phase: np.ndarray, local_x: int
    ) -> bool:
        material_id = int(row_material[local_x])
        return (
            material_id > 0
            and int(row_phase[local_x]) == int(Phase.LIQUID)
            and solver._material_liquid_solver_kind(world, material_id) == LIQUID_SOLVER_TILE_LEVEL
        )

    def move_cell(src_y: int, src_x: int, dst_y: int, dst_x: int) -> None:
        local_material[dst_y, dst_x] = local_material[src_y, src_x]
        local_phase[dst_y, dst_x] = local_phase[src_y, src_x]
        local_flags[dst_y, dst_x] = local_flags[src_y, src_x]
        local_timer[dst_y, dst_x] = local_timer[src_y, src_x]
        local_temp[dst_y, dst_x] = local_temp[src_y, src_x]
        local_integrity[dst_y, dst_x] = local_integrity[src_y, src_x]
        local_velocity[dst_y, dst_x] = local_velocity[src_y, src_x]
        local_island[dst_y, dst_x] = local_island[src_y, src_x]
        local_entity[dst_y, dst_x] = local_entity[src_y, src_x]
        local_displaced[dst_y, dst_x] = local_displaced[src_y, src_x]
        local_material[src_y, src_x] = 0
        local_phase[src_y, src_x] = 0
        local_flags[src_y, src_x] = 0
        local_timer[src_y, src_x] = 0
        local_temp[src_y, src_x] = 0.0
        local_integrity[src_y, src_x] = 0.0
        local_velocity[src_y, src_x] = 0.0
        local_island[src_y, src_x] = 0
        local_entity[src_y, src_x] = 0
        local_displaced[src_y, src_x] = 0

    for ly in range(local_material.shape[0] - 2, -1, -1):
        row_material = local_material[ly].copy()
        row_phase = local_phase[ly].copy()
        width = local_material.shape[1]
        # Row-snapshot replays: every pass below evaluates the pre-move row
        # (like the GPU warp ballot in tile_solve.comp). Within one row pass a
        # liquid cell keeps its liquid/source status and an empty cell stays
        # empty even if a planned move targets it; executing the plan only at
        # the end keeps interleaved liquid/empty runs from teleporting cells
        # across the row, which produced disconnected flickering patches. A
        # cell whose snapshot status was liquid never becomes a destination
        # within the same pass (source cells stay liquid on the snapshot), so
        # an arriving cell cannot chain-hop in one frame.
        snapshot_liquid = np.fromiter(
            (snapshot_tile_level_liquid(row_material, row_phase, lx) for lx in range(width)),
            dtype=np.bool_,
            count=width,
        )
        # Tile-level liquids plan against the row snapshot (like the GPU warp
        # ballot in tile_solve.comp); non-tile-level (columnar) liquids keep
        # their canonical live row pass below so a left-blocked column can
        # still flow right within the same frame.
        tile_level_liquid = bool(np.any(snapshot_liquid))
        # Lip-hang span (LIQUID_LIP_OVERHANG, mirrors tile_solve.comp): a
        # suspended cell hangs while it is within the overhang distance of a
        # supported cell in its own row run; anything farther falls.  A run
        # with at least one supported cell never collapses as a segment —
        # its beyond-lip cells fall individually below.
        lane_liquid = (
            snapshot_liquid
            if tile_level_liquid
            else np.fromiter(
                (is_liquid(ly, lx) for lx in range(width)), dtype=np.bool_, count=width
            )
        )
        within_lip = np.zeros((width,), dtype=np.bool_)
        run_has_support = np.zeros((width,), dtype=np.bool_)
        lx = 0
        while lx < width:
            if not bool(lane_liquid[lx]):
                lx += 1
                continue
            run_start = lx
            while lx < width and bool(lane_liquid[lx]):
                lx += 1
            run_end = lx
            supported = [x for x in range(run_start, run_end) if not reachable_empty(ly + 1, x)]
            if not supported:
                continue
            run_has_support[run_start:run_end] = True
            span_lo = max(run_start, supported[0] - LIQUID_LIP_OVERHANG)
            span_hi = min(run_end, supported[-1] + LIQUID_LIP_OVERHANG + 1)
            within_lip[span_lo:span_hi] = True
        claimed_down_source = np.zeros((width,), dtype=np.bool_)
        claimed_down_target = np.zeros((width,), dtype=np.bool_)
        planned_down_moves: list[tuple[int, int, int, int]] = []
        lx = 0
        while lx < width:
            if not bool(snapshot_liquid[lx]):
                lx += 1
                continue
            run_start = lx
            while lx < width and bool(snapshot_liquid[lx]):
                lx += 1
            run_end = lx
            if bool(run_has_support[run_start]):
                # The run is (partly) supported: no segment collapse; the
                # straight-down pass below drops only its beyond-lip cells.
                continue
            first_empty_x = -1
            for probe_x in range(run_start, run_end):
                if reachable_empty(ly + 1, probe_x) and not bool(claimed_down_target[probe_x]):
                    first_empty_x = probe_x
                    break
            if first_empty_x < 0:
                continue
            empty_start = first_empty_x
            while (
                empty_start > 0
                and reachable_empty(ly + 1, empty_start - 1)
                and not bool(claimed_down_target[empty_start - 1])
            ):
                empty_start -= 1
            empty_end = first_empty_x + 1
            while (
                empty_end < width
                and reachable_empty(ly + 1, empty_end)
                and not bool(claimed_down_target[empty_end])
            ):
                empty_end += 1
            move_count = min(run_end - run_start, empty_end - empty_start)
            if move_count <= 0:
                continue
            # Bound the downfill's lateral shift to the in-tile lateral rate:
            # an unbounded shift teleports whole runs diagonally and draws
            # long hovering lines over empty space.  When the empty segment
            # is out of reach nothing is claimed, so cells above empties
            # still fall straight down this frame.
            target_lo = max(empty_start, run_start - LIQUID_DOWNFILL_MAX_LATERAL_SHIFT)
            target_hi = min(empty_end - move_count, run_start + LIQUID_DOWNFILL_MAX_LATERAL_SHIFT)
            if target_lo > target_hi:
                continue
            target_base = min(max(run_start, target_lo), target_hi)
            # Only the cells that actually move are claimed: claiming the
            # whole run for a bounded shift starved the lateral pass of
            # unmoved cells and pinned unmoved cells above empties in the
            # air (the straight-down fallback below handles those).
            claimed_down_source[run_start : run_start + move_count] = True
            claimed_down_target[target_base : target_base + move_count] = True
            for offset in range(move_count):
                planned_down_moves.append((ly, run_start + offset, ly + 1, target_base + offset))
        for src_y, src_x, dst_y, dst_x in planned_down_moves:
            move_cell(src_y, src_x, dst_y, dst_x)
            changed = True

        planned_straight_down: list[tuple[int, int, int, int]] = []
        for lx in range(width):
            if bool(claimed_down_source[lx]):
                continue
            if tile_level_liquid:
                # Tile-level rows keep the GPU single-swap semantics: a cell
                # that was empty on the row snapshot (a down-run target) is
                # not re-evaluated as a straight-down source this frame.
                if not bool(snapshot_liquid[lx]):
                    continue
            elif not is_liquid(ly, lx):
                continue
            if bool(within_lip[lx]):
                # Suspended but inside the terrace lip: hangs instead of
                # dripping, so the front advances as a connected body.
                continue
            if reachable_empty(ly + 1, lx):
                # Mark the source as claimed: the cell leaves this row, so the
                # lateral pass must not treat its snapshot-liquid status as a
                # live source and steal a neighbor's lateral destination.
                claimed_down_source[lx] = True
                planned_straight_down.append((ly, lx, ly + 1, lx))
        for src_y, src_x, dst_y, dst_x in planned_straight_down:
            move_cell(src_y, src_x, dst_y, dst_x)
            changed = True

        planned_lateral_moves: list[tuple[int, int, int, int]] = []
        for lx in range(width):
            if bool(claimed_down_source[lx]):
                continue
            # Suspended liquid falls straight down: lateral spread is allowed
            # only when the cell is supported from below, i.e. the cell
            # directly below is not reachable-empty (solid or standing liquid
            # both count as support). Row ly + 1 is always in-tile here and
            # reflects this frame's down moves, matching the GPU shared-state
            # read in tile_solve.comp; a cell that lands this frame first
            # spreads on the next one.
            if reachable_empty(ly + 1, lx):
                continue
            if tile_level_liquid:
                if not bool(snapshot_liquid[lx]):
                    continue
            elif not is_tile_level_liquid(ly, lx):
                continue
            # Conflict-free targets, mirroring tile_solve.comp: a left move
            # is valid only when the left-left cell is not a supported
            # liquid cell (otherwise that cell right-moves into the same
            # destination this pass and wins), which makes every
            # destination's two candidate sources mutually exclusive.
            left2_blocker = lx >= 2 and (
                (
                    bool(snapshot_liquid[lx - 2])
                    if tile_level_liquid
                    else is_tile_level_liquid(ly, lx - 2)
                )
                and not reachable_empty(ly + 1, lx - 2)
            )
            if lx > 0 and reachable_empty(ly, lx - 1) and not left2_blocker:
                planned_lateral_moves.append((ly, lx, ly, lx - 1))
                continue
            if lx + 1 < width and reachable_empty(ly, lx + 1):
                planned_lateral_moves.append((ly, lx, ly, lx + 1))
        for src_y, src_x, dst_y, dst_x in planned_lateral_moves:
            move_cell(src_y, src_x, dst_y, dst_x)
            changed = True
    if changed:
        world._invalidate_gpu_authoritative_cell_resources()
        world.material_id[y0:y1, x0:x1] = local_material
        world.phase[y0:y1, x0:x1] = local_phase
        world.cell_flags[y0:y1, x0:x1] = local_flags
        world.timer_pack[y0:y1, x0:x1] = local_timer
        world.cell_temperature[y0:y1, x0:x1] = local_temp
        world.integrity[y0:y1, x0:x1] = local_integrity
        world.velocity[y0:y1, x0:x1] = local_velocity
        world.island_id[y0:y1, x0:x1] = local_island
        world.entity_id[y0:y1, x0:x1] = local_entity
        world.placeholder_displaced_material[y0:y1, x0:x1] = local_displaced
        world._mark_active_rect_runtime(x0, y0, x1, y1)


def _seam_correction(solver, world: "WorldEngine", solve_tile_mask: np.ndarray) -> None:
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
        y0 = tile_y * tile_size
        y1 = min(world.height, y0 + tile_size)
        for y in range(y0, y1):
            if solver._apply_horizontal_seam_run(world, boundary_x, y, tile_size):
                continue
    for tile_x, boundary_y in sorted(horizontal_boundaries):
        top = boundary_y - 1
        bottom = boundary_y
        x0 = tile_x * tile_size
        x1 = min(world.width, x0 + tile_size)
        solver._apply_vertical_seam_run(world, top, bottom, x0, x1)


def _apply_horizontal_seam_run(
    solver,
    world: "WorldEngine",
    boundary_x: int,
    y: int,
    tile_size: int,
) -> bool:
    if boundary_x <= 0 or boundary_x >= world.width:
        return False
    left = boundary_x - 1
    right = boundary_x
    if solver._world_cell_is_tile_level_liquid(
        world, left, y
    ) and solver._world_cell_reachable_empty(world, right, y):
        # Suspended liquid does not spread across the seam: the run moves only
        # when the boundary-adjacent source cell (always part of the moved
        # set) is supported from below, i.e. the cell below is not
        # reachable-empty (solid or standing liquid both count as support;
        # below the world floor counts as supported, matching the GPU
        # out-of-bounds rule in seam_x.comp).
        if y + 1 < world.height and solver._world_cell_reachable_empty(world, left, y + 1):
            return False
        source_start = left
        source_tile_start = (left // tile_size) * tile_size
        while source_start > source_tile_start and solver._world_cell_is_tile_level_liquid(
            world, source_start - 1, y
        ):
            source_start -= 1
        target_end = right
        target_tile_end = min(world.width, right + tile_size)
        while target_end < target_tile_end and solver._world_cell_reachable_empty(
            world, target_end, y
        ):
            target_end += 1
        target_len = target_end - right
        # Same speed/support clamps as seam_x.comp left_to_right_move: at
        # most SEAM_MAX_MOVE_PER_FRAME cells per frame (the in-tile rate),
        # and at most the leading SEAM_UNSUPPORTED_OVERHANG filled cells may
        # lack support below — a short contiguous terrace lip past the
        # supported edge; those lip cells fall on later frames if support
        # never arrives.  Sources are cleared from the run's TRAILING edge so
        # a partial move shifts the run contiguously instead of punching a
        # hole into the row.
        supported_prefix = 0
        while supported_prefix < target_len and (
            y + 1 >= world.height
            or not solver._world_cell_reachable_empty(world, right + supported_prefix, y + 1)
        ):
            supported_prefix += 1
        move_count = min(
            left - source_start + 1,
            target_len,
            supported_prefix + SEAM_UNSUPPORTED_OVERHANG,
            SEAM_MAX_MOVE_PER_FRAME,
        )
        if move_count > 0:
            for offset in range(move_count):
                world.swap_cells(source_start + offset, y, right + offset, y)
            return True
    if solver._world_cell_is_tile_level_liquid(
        world, right, y
    ) and solver._world_cell_reachable_empty(world, left, y):
        # Same below-support gate as the left-to-right branch above.
        if y + 1 < world.height and solver._world_cell_reachable_empty(world, right, y + 1):
            return False
        source_end = right + 1
        source_tile_end = min(world.width, right + tile_size)
        while source_end < source_tile_end and solver._world_cell_is_tile_level_liquid(
            world, source_end, y
        ):
            source_end += 1
        target_start = left
        target_tile_start = max(0, right - tile_size)
        while target_start > target_tile_start and solver._world_cell_reachable_empty(
            world, target_start - 1, y
        ):
            target_start -= 1
        target_len = right - target_start
        # Same mirrored clamps as the left-to-right branch above; sources
        # clear from the run's trailing (right) edge.
        supported_prefix = 0
        while supported_prefix < target_len and (
            y + 1 >= world.height
            or not solver._world_cell_reachable_empty(world, left - supported_prefix, y + 1)
        ):
            supported_prefix += 1
        move_count = min(
            source_end - right,
            target_len,
            supported_prefix + SEAM_UNSUPPORTED_OVERHANG,
            SEAM_MAX_MOVE_PER_FRAME,
        )
        if move_count > 0:
            target_base = right - move_count
            for offset in range(move_count):
                world.swap_cells(source_end - move_count + offset, y, target_base + offset, y)
            return True
    return False


def _apply_vertical_seam_run(
    solver,
    world: "WorldEngine",
    top: int,
    bottom: int,
    x0: int,
    x1: int,
) -> None:
    claimed_source: set[int] = set()
    claimed_target: set[int] = set()
    planned_moves: list[tuple[int, int]] = []
    x = x0
    while x < x1:
        if not solver._world_cell_is_tile_level_liquid(world, x, top):
            x += 1
            continue
        run_start = x
        while x < x1 and solver._world_cell_is_tile_level_liquid(world, x, top):
            x += 1
        run_end = x
        first_empty_x = -1
        for probe_x in range(run_start, run_end):
            if (
                solver._world_cell_reachable_empty(world, probe_x, bottom)
                and probe_x not in claimed_target
            ):
                first_empty_x = probe_x
                break
        if first_empty_x < 0:
            continue
        empty_start = first_empty_x
        while (
            empty_start > x0
            and solver._world_cell_reachable_empty(world, empty_start - 1, bottom)
            and empty_start - 1 not in claimed_target
        ):
            empty_start -= 1
        empty_end = first_empty_x + 1
        while (
            empty_end < x1
            and solver._world_cell_reachable_empty(world, empty_end, bottom)
            and empty_end not in claimed_target
        ):
            empty_end += 1
        move_count = min(run_end - run_start, empty_end - empty_start)
        if move_count <= 0:
            continue
        # Same lateral-shift bound as the in-tile down-run above; when the
        # empty segment is out of reach, only the straight-down subset
        # crosses (the seam has no per-cell straight-down fallback).
        target_lo = max(empty_start, run_start - LIQUID_DOWNFILL_MAX_LATERAL_SHIFT)
        target_hi = min(empty_end - move_count, run_start + LIQUID_DOWNFILL_MAX_LATERAL_SHIFT)
        if target_lo > target_hi:
            run_start = max(run_start, empty_start)
            move_count = min(run_end, empty_end) - run_start
            if move_count <= 0:
                continue
            target_base = run_start
        else:
            target_base = min(max(run_start, target_lo), target_hi)
        # Same per-cell claiming as the in-tile down-run above.
        claimed_source.update(range(run_start, run_start + move_count))
        claimed_target.update(range(target_base, target_base + move_count))
        for offset in range(move_count):
            planned_moves.append((run_start + offset, target_base + offset))
    for source_x, target_x in planned_moves:
        world.swap_cells(source_x, top, target_x, bottom)


def _apply_buoyancy(solver, world: "WorldEngine", solve_cell_mask: np.ndarray) -> None:
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
        if int(phase_snapshot[y - 1, x]) != int(Phase.POWDER) or int(phase_snapshot[y, x]) != int(
            Phase.LIQUID
        ):
            continue
        powder_density = solver._material_density(world, upper_id)
        liquid_density = solver._material_density(world, lower_id)
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
        if int(phase_snapshot[y - 1, x]) != int(Phase.LIQUID) or int(phase_snapshot[y, x]) != int(
            Phase.POWDER
        ):
            continue
        liquid_density = solver._material_density(world, upper_id)
        powder_density = solver._material_density(world, lower_id)
        if powder_density < liquid_density:
            float_swaps.append((x, y))
    for x, y in float_swaps:
        world.swap_cells(x, y - 1, x, y)


def _apply_placeholder_displacement(
    solver, world: "WorldEngine", solve_cell_mask: np.ndarray
) -> None:
    placeholder_id = solver._placeholder_material_id(world)
    material_in = world.material_id.copy()
    phase_in = world.phase.copy()
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
            pending_sources = [
                source_x for source_x in range(left, right) if int(pending_in[y, source_x]) > 0
            ]
            displaced_count = len(pending_sources)
            if displaced_count <= 0:
                continue
            top_exposed = solver._placeholder_segment_top_exposed(
                material_in, placeholder_id, y, left, right
            )
            left_capacity = solver._placeholder_side_capacity(
                world,
                material_in,
                phase_in,
                pending_in,
                -1,
                top_exposed,
                y,
                left,
                right,
                seg_len,
            )
            right_capacity = solver._placeholder_side_capacity(
                world,
                material_in,
                phase_in,
                pending_in,
                1,
                top_exposed,
                y,
                left,
                right,
                seg_len,
            )
            left_quota = solver._placeholder_left_quota(
                displaced_count, left_capacity, right_capacity
            )
            for displaced_rank, source_x in enumerate(pending_sources):
                displaced_material = int(pending_in[y, source_x])
                if displaced_material <= 0:
                    continue
                side = -1 if displaced_rank < left_quota else 1
                side_rank = displaced_rank if side < 0 else displaced_count - 1 - displaced_rank
                side_rank = max(0, min(seg_len - 1, side_rank))
                for target_x, target_y, velocity in solver._placeholder_side_candidates(
                    world,
                    material_in,
                    phase_in,
                    pending_in,
                    y,
                    left,
                    right,
                    seg_len,
                    side,
                    side_rank,
                    top_exposed,
                ):
                    if int(world.material_id[target_y, target_x]) != 0:
                        continue
                    world.material_id[target_y, target_x] = displaced_material
                    world.phase[target_y, target_x] = int(Phase.LIQUID)
                    world.integrity[target_y, target_x] = solver._material_base_integrity(
                        world, displaced_material
                    )
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


def _placeholder_left_quota(
    solver, displaced_count: int, left_capacity: int, right_capacity: int
) -> int:
    total_capacity = left_capacity + right_capacity
    if displaced_count <= 0 or total_capacity <= 0:
        return 0
    numerator = displaced_count * left_capacity
    quota, remainder = divmod(numerator, total_capacity)
    if remainder * 2 >= total_capacity:
        quota += 1
    return max(0, min(displaced_count, quota))


def _placeholder_segment_top_exposed(
    solver,
    material_in: np.ndarray,
    placeholder_id: int,
    source_y: int,
    left: int,
    right: int,
) -> bool:
    if source_y == 0:
        return True
    return any(int(material_in[source_y - 1, x]) != placeholder_id for x in range(left, right))


def _placeholder_target_empty(
    solver,
    world: "WorldEngine",
    material_in: np.ndarray,
    phase_in: np.ndarray,
    pending_in: np.ndarray,
    target_x: int,
    target_y: int,
) -> bool:
    if not world.in_bounds(target_x, target_y):
        return False
    if int(material_in[target_y, target_x]) != 0:
        return False
    target_phase = int(phase_in[target_y, target_x])
    if target_phase in (int(Phase.LIQUID), int(Phase.FALLING_ISLAND)):
        return False
    if int(pending_in[target_y, target_x]) > 0:
        return False
    return True


def _placeholder_side_lane_reachable(
    solver,
    world: "WorldEngine",
    material_in: np.ndarray,
    phase_in: np.ndarray,
    pending_in: np.ndarray,
    side: int,
    target_x: int,
    target_y: int,
    left: int,
    right: int,
) -> bool:
    if target_y < 0 or target_y >= world.height:
        return False
    if side < 0:
        for x in range(target_x, left):
            if not solver._placeholder_target_empty(
                world, material_in, phase_in, pending_in, x, target_y
            ):
                return False
        return True
    for x in range(right, target_x + 1):
        if not solver._placeholder_target_empty(
            world, material_in, phase_in, pending_in, x, target_y
        ):
            return False
    return True


def _placeholder_side_capacity(
    solver,
    world: "WorldEngine",
    material_in: np.ndarray,
    phase_in: np.ndarray,
    pending_in: np.ndarray,
    side: int,
    top_exposed: bool,
    source_y: int,
    left: int,
    right: int,
    seg_len: int,
) -> int:
    capacity = 0
    for top_lane in (False, True):
        if top_lane and not top_exposed:
            continue
        target_y = source_y - 1 if top_lane else source_y
        for slot in range(seg_len):
            target_x = left - 1 - slot if side < 0 else right + slot
            if solver._placeholder_side_lane_reachable(
                world,
                material_in,
                phase_in,
                pending_in,
                side,
                target_x,
                target_y,
                left,
                right,
            ):
                capacity += 1
    return capacity


def _placeholder_side_candidates(
    solver,
    world: "WorldEngine",
    material_in: np.ndarray,
    phase_in: np.ndarray,
    pending_in: np.ndarray,
    source_y: int,
    left: int,
    right: int,
    seg_len: int,
    side: int,
    start_slot: int,
    top_exposed: bool,
) -> list[tuple[int, int, tuple[float, float]]]:
    candidates: list[tuple[int, int, tuple[float, float]]] = []
    for top_lane in (False, True):
        if top_lane and not top_exposed:
            continue
        target_y = source_y - 1 if top_lane else source_y
        push = (float(side) * 0.8, -0.65) if top_lane else (float(side) * 1.2, -0.15)
        for offset in range(seg_len):
            slot = (start_slot + offset) % seg_len
            target_x = left - 1 - slot if side < 0 else right + slot
            if solver._placeholder_side_lane_reachable(
                world,
                material_in,
                phase_in,
                pending_in,
                side,
                target_x,
                target_y,
                left,
                right,
            ):
                candidates.append((target_x, target_y, push))
    return candidates


def _mark_pending_placeholder_regions(solver, world: "WorldEngine") -> None:
    ys, xs = np.nonzero(world.placeholder_displaced_material > 0)
    rects: list[tuple[int, int, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        rects.append(
            (max(0, x - 1), max(0, y - 1), min(world.width, x + 2), min(world.height, y + 2))
        )
    world._mark_active_rects_runtime(rects)


def _refresh_active_tiles(
    solver, world: "WorldEngine", active_tiles: list[tuple[int, int]]
) -> None:
    tile_size = world.active.tile_size
    rects: list[tuple[int, int, int, int]] = []
    for tile_x, tile_y in active_tiles:
        x0 = max(0, (tile_x - 1) * tile_size)
        y0 = max(0, (tile_y - 1) * tile_size)
        x1 = min(world.width, (tile_x + 2) * tile_size)
        y1 = min(world.height, (tile_y + 2) * tile_size)
        rects.append((x0, y0, x1, y1))
    world._mark_active_rects_runtime(rects)


def _vertical_seam_mask(solver, world: "WorldEngine", solve_tile_mask: np.ndarray) -> np.ndarray:
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


def _horizontal_seam_mask(solver, world: "WorldEngine", solve_tile_mask: np.ndarray) -> np.ndarray:
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
    solver,
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
            powder_density = solver._material_density(world, upper_id)
            liquid_density = solver._material_density(world, lower_id)
            if powder_density > liquid_density:
                mask[y - 1, x] = True
                mask[y, x] = True
        elif upper_phase == int(Phase.LIQUID) and lower_phase == int(Phase.POWDER):
            liquid_density = solver._material_density(world, upper_id)
            powder_density = solver._material_density(world, lower_id)
            if powder_density < liquid_density:
                mask[y - 1, x] = True
                mask[y, x] = True
    return mask
