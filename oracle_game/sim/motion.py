from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.sim.gpu_motion import (
    GPUMotionPipeline,
    falling_island_reservation_dtype,
    powder_reservation_dtype,
)

# Facade re-exports: the constants and component record live in the leaf module
# ``motion_constants`` so the motion_* satellites can import them without
# cycling back through this hub.
from oracle_game.sim.motion_constants import (
    FALLING_ISLAND_BREAK_STABLE,  # noqa: F401  # facade re-export
    MAX_ISLAND_DDA_STEP,  # noqa: F401  # facade re-export
    POWDER_SOLVER_SUSPENDED,  # noqa: F401  # facade re-export
    _IslandComponentEntry,  # noqa: F401  # facade re-export
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
    _shadow_shift_island_material,
    _shed_falling_island_fragments,
    _shift_island,
)
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


class MotionSolver:
    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
        self.gpu_pipeline = GPUMotionPipeline(engine_config=self.engine_config)
        self.last_backend = "idle"
        self.last_powder_reservations = np.zeros((0,), dtype=powder_reservation_dtype())
        self.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
        self.last_public_powder_reservations: list[dict[str, object]] = []
        self.last_public_island_reservations: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Satellite method delegates (W3: retired the `_x = _x` class grafts).
    #
    # Each body resolves the bare function name through this module's global
    # namespace -- method bodies never see class scope -- i.e. the satellite
    # function imported at the top of this file, bound at import time exactly
    # like the historical grafts.  Monkeypatch semantics are unchanged:
    # patching the attribute on the class or on an instance shadows/replaces
    # the delegate, while patching the satellite module's attribute does NOT
    # affect calls through the solver.
    # ------------------------------------------------------------------

    def _collision_response(self, *args: Any, **kwargs: Any) -> Any:
        return _collision_response(self, *args, **kwargs)

    def _dda_line_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _dda_line_cells(self, *args, **kwargs)

    def _material_default_phase(self, *args: Any, **kwargs: Any) -> Any:
        return _material_default_phase(self, *args, **kwargs)

    def _material_elasticity(self, *args: Any, **kwargs: Any) -> Any:
        return _material_elasticity(self, *args, **kwargs)

    def _material_falling_island_break_kind(self, *args: Any, **kwargs: Any) -> Any:
        return _material_falling_island_break_kind(self, *args, **kwargs)

    def _material_friction(self, *args: Any, **kwargs: Any) -> Any:
        return _material_friction(self, *args, **kwargs)

    def _material_gravity(self, *args: Any, **kwargs: Any) -> Any:
        return _material_gravity(self, *args, **kwargs)

    def _material_int(self, *args: Any, **kwargs: Any) -> Any:
        return _material_int(self, *args, **kwargs)

    def _material_is_placeholder(self, *args: Any, **kwargs: Any) -> Any:
        return _material_is_placeholder(self, *args, **kwargs)

    def _material_max_dda_step(self, *args: Any, **kwargs: Any) -> Any:
        return _material_max_dda_step(self, *args, **kwargs)

    def _material_powder_generation_id(self, *args: Any, **kwargs: Any) -> Any:
        return _material_powder_generation_id(self, *args, **kwargs)

    def _material_powder_solver_kind(self, *args: Any, **kwargs: Any) -> Any:
        return _material_powder_solver_kind(self, *args, **kwargs)

    def _material_scalar(self, *args: Any, **kwargs: Any) -> Any:
        return _material_scalar(self, *args, **kwargs)

    def _material_scalar_field(self, *args: Any, **kwargs: Any) -> Any:
        return _material_scalar_field(self, *args, **kwargs)

    def _material_table_row(self, *args: Any, **kwargs: Any) -> Any:
        return _material_table_row(self, *args, **kwargs)

    def _capture_public_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_public_island_reservations(self, *args, **kwargs)

    def _capture_public_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_public_powder_reservations(self, *args, **kwargs)

    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def reset_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return reset_runtime_state(self, *args, **kwargs)

    def runtime_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return runtime_snapshot(self, *args, **kwargs)

    def _integrate_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return _integrate_velocity(self, *args, **kwargs)

    def _solve_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _solve_tile_mask(self, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def _apply_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_powder_reservations(self, *args, **kwargs)

    def _mark_powder_reservation_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_powder_reservation_regions(self, *args, **kwargs)

    def _move_powders(self, *args: Any, **kwargs: Any) -> Any:
        return _move_powders(self, *args, **kwargs)

    def _path_is_clear(self, *args: Any, **kwargs: Any) -> Any:
        return _path_is_clear(self, *args, **kwargs)

    def _path_is_clear_material(self, *args: Any, **kwargs: Any) -> Any:
        return _path_is_clear_material(self, *args, **kwargs)

    def _plan_cpu_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _plan_cpu_powder_reservations(self, *args, **kwargs)

    def _powder_fallback_candidates(self, *args: Any, **kwargs: Any) -> Any:
        return _powder_fallback_candidates(self, *args, **kwargs)

    def _resolve_powder_dda_target(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_powder_dda_target(self, *args, **kwargs)

    def _resolve_powder_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_powder_reservations(self, *args, **kwargs)

    def _assign_split_component_cells_cpu(self, *args: Any, **kwargs: Any) -> Any:
        return _assign_split_component_cells_cpu(self, *args, **kwargs)

    def _bbox_from_coords(self, *args: Any, **kwargs: Any) -> Any:
        return _bbox_from_coords(self, *args, **kwargs)

    def _can_seed_bridge_runtime_fast_path(self, *args: Any, **kwargs: Any) -> Any:
        return _can_seed_bridge_runtime_fast_path(self, *args, **kwargs)

    def _can_shift_island(self, *args: Any, **kwargs: Any) -> Any:
        return _can_shift_island(self, *args, **kwargs)

    def _can_shift_island_material(self, *args: Any, **kwargs: Any) -> Any:
        return _can_shift_island_material(self, *args, **kwargs)

    def _clear_stale_island_cells(self, *args: Any, **kwargs: Any) -> Any:
        return _clear_stale_island_cells(self, *args, **kwargs)

    def _component_entry_from_coords(self, *args: Any, **kwargs: Any) -> Any:
        return _component_entry_from_coords(self, *args, **kwargs)

    def _component_entry_from_gpu_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return _component_entry_from_gpu_metadata(self, *args, **kwargs)

    def _connected_island_components(self, *args: Any, **kwargs: Any) -> Any:
        return _connected_island_components(self, *args, **kwargs)

    def _falling_island_contact_material_response(self, *args: Any, **kwargs: Any) -> Any:
        return _falling_island_contact_material_response(self, *args, **kwargs)

    def _falling_island_coords(self, *args: Any, **kwargs: Any) -> Any:
        return _falling_island_coords(self, *args, **kwargs)

    def _falling_island_fragment_neighbor_threshold(self, *args: Any, **kwargs: Any) -> Any:
        return _falling_island_fragment_neighbor_threshold(self, *args, **kwargs)

    def _falling_island_gravity_fallback_dy(self, *args: Any, **kwargs: Any) -> Any:
        return _falling_island_gravity_fallback_dy(self, *args, **kwargs)

    def _falling_island_reservation_order_key(self, *args: Any, **kwargs: Any) -> Any:
        return _falling_island_reservation_order_key(self, *args, **kwargs)

    def _gpu_connected_island_component_entries(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_connected_island_component_entries(self, *args, **kwargs)

    def _gpu_connected_island_components(self, *args: Any, **kwargs: Any) -> Any:
        return _gpu_connected_island_components(self, *args, **kwargs)

    def _move_falling_islands(self, *args: Any, **kwargs: Any) -> Any:
        return _move_falling_islands(self, *args, **kwargs)

    def _plan_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _plan_falling_island_reservations(self, *args, **kwargs)

    def _resolve_falling_island_components(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_falling_island_components(self, *args, **kwargs)

    def _resolve_falling_island_reservations(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_falling_island_reservations(self, *args, **kwargs)

    def _resolve_island_dda_shift(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_island_dda_shift(self, *args, **kwargs)

    def _resolve_island_dda_target(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_island_dda_target(self, *args, **kwargs)

    def _resolve_island_dda_target_material(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_island_dda_target_material(self, *args, **kwargs)

    def _same_island_neighbors(self, *args: Any, **kwargs: Any) -> Any:
        return _same_island_neighbors(self, *args, **kwargs)

    def _shed_falling_island_fragments(self, *args: Any, **kwargs: Any) -> Any:
        return _shed_falling_island_fragments(self, *args, **kwargs)

    def _shadow_shift_island_material(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_shift_island_material(self, *args, **kwargs)

    def _shift_island(self, *args: Any, **kwargs: Any) -> Any:
        return _shift_island(self, *args, **kwargs)
