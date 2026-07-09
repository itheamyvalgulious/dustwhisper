from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oracle_game.sim.gpu_motion import (
    GPUMotionPipeline,
    falling_island_reservation_dtype,
    powder_reservation_dtype,
)

MAX_ISLAND_DDA_STEP = 4
POWDER_SOLVER_SUSPENDED = 2
FALLING_ISLAND_BREAK_STABLE = 2


@dataclass(slots=True)
class _IslandComponentEntry:
    label: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    cell_count: int


from oracle_game.sim.motion_material import (
    _collision_response,
    _dda_line_cells,
    _material_default_phase,
    _material_elasticity,
    _material_falling_island_break_kind,
    _material_friction,
    _material_gravity,
    _material_int,
    _material_is_placeholder,
    _material_max_dda_step,
    _material_powder_generation_id,
    _material_powder_solver_kind,
    _material_scalar,
    _material_scalar_field,
    _material_table_row,
)
from oracle_game.sim.motion_runtime import (
    _capture_public_island_reservations,
    _capture_public_powder_reservations,
    release,
    reset_runtime_state,
    runtime_snapshot,
)
from oracle_game.sim.motion_velocity import (
    _integrate_velocity,
    _solve_tile_mask,
    step,
)
from oracle_game.sim.motion_powder import (
    _apply_powder_reservations,
    _mark_powder_reservation_regions,
    _move_powders,
    _path_is_clear,
    _path_is_clear_material,
    _plan_cpu_powder_reservations,
    _powder_fallback_candidates,
    _resolve_powder_dda_target,
    _resolve_powder_reservations,
)
from oracle_game.sim.motion_falling_island import (
    _assign_split_component_cells_cpu,
    _bbox_from_coords,
    _can_seed_bridge_runtime_fast_path,
    _can_shift_island,
    _can_shift_island_material,
    _clear_stale_island_cells,
    _component_entry_from_coords,
    _component_entry_from_gpu_metadata,
    _connected_island_components,
    _falling_island_contact_material_response,
    _falling_island_coords,
    _falling_island_fragment_neighbor_threshold,
    _falling_island_gravity_fallback_dy,
    _falling_island_reservation_order_key,
    _gpu_connected_island_component_entries,
    _gpu_connected_island_components,
    _move_falling_islands,
    _plan_falling_island_reservations,
    _resolve_falling_island_components,
    _resolve_falling_island_reservations,
    _resolve_island_dda_shift,
    _resolve_island_dda_target,
    _resolve_island_dda_target_material,
    _same_island_neighbors,
    _shed_falling_island_fragments,
    _shadow_shift_island_material,
    _shift_island,
)


class MotionSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUMotionPipeline()
        self.last_backend = "idle"
        self.last_powder_reservations = np.zeros((0,), dtype=powder_reservation_dtype())
        self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
        self.last_public_powder_reservations: list[dict[str, object]] = []
        self.last_public_island_reservations: list[dict[str, object]] = []

    _collision_response = _collision_response
    _dda_line_cells = _dda_line_cells
    _material_default_phase = _material_default_phase
    _material_elasticity = _material_elasticity
    _material_falling_island_break_kind = _material_falling_island_break_kind
    _material_friction = _material_friction
    _material_gravity = _material_gravity
    _material_int = _material_int
    _material_is_placeholder = _material_is_placeholder
    _material_max_dda_step = _material_max_dda_step
    _material_powder_generation_id = _material_powder_generation_id
    _material_powder_solver_kind = _material_powder_solver_kind
    _material_scalar = _material_scalar
    _material_scalar_field = _material_scalar_field
    _material_table_row = _material_table_row

    _capture_public_island_reservations = _capture_public_island_reservations
    _capture_public_powder_reservations = _capture_public_powder_reservations
    release = release
    reset_runtime_state = reset_runtime_state
    runtime_snapshot = runtime_snapshot

    _integrate_velocity = _integrate_velocity
    _solve_tile_mask = _solve_tile_mask
    step = step

    _apply_powder_reservations = _apply_powder_reservations
    _mark_powder_reservation_regions = _mark_powder_reservation_regions
    _move_powders = _move_powders
    _path_is_clear = _path_is_clear
    _path_is_clear_material = _path_is_clear_material
    _plan_cpu_powder_reservations = _plan_cpu_powder_reservations
    _powder_fallback_candidates = _powder_fallback_candidates
    _resolve_powder_dda_target = _resolve_powder_dda_target
    _resolve_powder_reservations = _resolve_powder_reservations

    _assign_split_component_cells_cpu = _assign_split_component_cells_cpu
    _bbox_from_coords = _bbox_from_coords
    _can_seed_bridge_runtime_fast_path = _can_seed_bridge_runtime_fast_path
    _can_shift_island = _can_shift_island
    _can_shift_island_material = _can_shift_island_material
    _clear_stale_island_cells = _clear_stale_island_cells
    _component_entry_from_coords = _component_entry_from_coords
    _component_entry_from_gpu_metadata = _component_entry_from_gpu_metadata
    _connected_island_components = _connected_island_components
    _falling_island_contact_material_response = _falling_island_contact_material_response
    _falling_island_coords = _falling_island_coords
    _falling_island_fragment_neighbor_threshold = _falling_island_fragment_neighbor_threshold
    _falling_island_gravity_fallback_dy = _falling_island_gravity_fallback_dy
    _falling_island_reservation_order_key = _falling_island_reservation_order_key
    _gpu_connected_island_component_entries = _gpu_connected_island_component_entries
    _gpu_connected_island_components = _gpu_connected_island_components
    _move_falling_islands = _move_falling_islands
    _plan_falling_island_reservations = _plan_falling_island_reservations
    _resolve_falling_island_components = _resolve_falling_island_components
    _resolve_falling_island_reservations = _resolve_falling_island_reservations
    _resolve_island_dda_shift = _resolve_island_dda_shift
    _resolve_island_dda_target = _resolve_island_dda_target
    _resolve_island_dda_target_material = _resolve_island_dda_target_material
    _same_island_neighbors = _same_island_neighbors
    _shed_falling_island_fragments = _shed_falling_island_fragments
    _shadow_shift_island_material = _shadow_shift_island_material
    _shift_island = _shift_island
