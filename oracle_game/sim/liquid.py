from __future__ import annotations

import numpy as np

from oracle_game.sim.gpu_liquid import GPULiquidPipeline


LIQUID_ACTIVITY_EPSILON = 1e-6
LIQUID_SOLVER_TILE_LEVEL = 1
LIQUID_SOLVER_COLUMNAR = 2


from oracle_game.sim.liquid_solve import (
    prepare_motion_flow_intent,
    _build_solve_tile_mask,
    _world_cell_reachable_empty,
    _world_cell_is_tile_level_liquid,
    _solve_tile,
    _seam_correction,
    _apply_horizontal_seam_run,
    _apply_vertical_seam_run,
    _apply_buoyancy,
    _apply_placeholder_displacement,
    _placeholder_left_quota,
    _placeholder_segment_top_exposed,
    _placeholder_target_empty,
    _placeholder_side_lane_reachable,
    _placeholder_side_capacity,
    _placeholder_side_candidates,
    _mark_pending_placeholder_regions,
    _refresh_active_tiles,
    _vertical_seam_mask,
    _horizontal_seam_mask,
    _buoyancy_candidate_mask,
)
from oracle_game.sim.liquid_runtime import (
    step,
    _finalize_runtime_state,
    release,
    reset_runtime_state,
    runtime_snapshot,
    _material_table_row,
    _material_density,
    _material_base_integrity,
    _material_liquid_solver_kind,
    _placeholder_material_id,
    _material_is_placeholder,
    _placeholder_mask,
)



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

    prepare_motion_flow_intent = prepare_motion_flow_intent
    _build_solve_tile_mask = _build_solve_tile_mask
    _world_cell_reachable_empty = _world_cell_reachable_empty
    _world_cell_is_tile_level_liquid = _world_cell_is_tile_level_liquid
    _solve_tile = _solve_tile
    _seam_correction = _seam_correction
    _apply_horizontal_seam_run = _apply_horizontal_seam_run
    _apply_vertical_seam_run = _apply_vertical_seam_run
    _apply_buoyancy = _apply_buoyancy
    _apply_placeholder_displacement = _apply_placeholder_displacement
    _placeholder_left_quota = _placeholder_left_quota
    _placeholder_segment_top_exposed = _placeholder_segment_top_exposed
    _placeholder_target_empty = _placeholder_target_empty
    _placeholder_side_lane_reachable = _placeholder_side_lane_reachable
    _placeholder_side_capacity = _placeholder_side_capacity
    _placeholder_side_candidates = _placeholder_side_candidates
    _mark_pending_placeholder_regions = _mark_pending_placeholder_regions
    _refresh_active_tiles = _refresh_active_tiles
    _vertical_seam_mask = _vertical_seam_mask
    _horizontal_seam_mask = _horizontal_seam_mask
    _buoyancy_candidate_mask = _buoyancy_candidate_mask

    step = step
    _finalize_runtime_state = _finalize_runtime_state
    release = release
    reset_runtime_state = reset_runtime_state
    runtime_snapshot = runtime_snapshot
    _material_table_row = _material_table_row
    _material_density = _material_density
    _material_base_integrity = _material_base_integrity
    _material_liquid_solver_kind = _material_liquid_solver_kind
    _placeholder_material_id = _placeholder_material_id
    _material_is_placeholder = _material_is_placeholder
    _placeholder_mask = _placeholder_mask
