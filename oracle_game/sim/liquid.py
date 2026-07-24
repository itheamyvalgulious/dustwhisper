from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.sim.gpu_liquid import GPULiquidPipeline

# Facade re-exports: the solver-kind constants live in the leaf module
# ``liquid_constants`` so the liquid_* satellites can import them without
# cycling back through this hub.
from oracle_game.sim.liquid_constants import (
    LIQUID_ACTIVITY_EPSILON,  # noqa: F401  # facade re-export
    LIQUID_SOLVER_COLUMNAR,  # noqa: F401  # facade re-export
    LIQUID_SOLVER_TILE_LEVEL,  # noqa: F401  # facade re-export
)
from oracle_game.sim.liquid_runtime import (
    _finalize_runtime_state,
    _material_base_integrity,
    _material_density,
    _material_is_placeholder,
    _material_liquid_solver_kind,
    _material_table_row,
    _placeholder_mask,
    _placeholder_material_id,
    release,
    reset_runtime_state,
    runtime_snapshot,
    step,
)
from oracle_game.sim.liquid_solve import (
    _apply_buoyancy,
    _apply_horizontal_seam_run,
    _apply_placeholder_displacement,
    _apply_vertical_seam_run,
    _build_solve_tile_mask,
    _buoyancy_candidate_mask,
    _horizontal_seam_mask,
    _mark_pending_placeholder_regions,
    _placeholder_left_quota,
    _placeholder_segment_top_exposed,
    _placeholder_side_candidates,
    _placeholder_side_capacity,
    _placeholder_side_lane_reachable,
    _placeholder_target_empty,
    _refresh_active_tiles,
    _seam_correction,
    _solve_tile,
    _vertical_seam_mask,
    _world_cell_is_tile_level_liquid,
    _world_cell_reachable_empty,
    prepare_motion_flow_intent,
)


class LiquidSolver:
    """CPU analogue of the planned tile-local shared-memory liquid staging."""

    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
        self.gpu_pipeline = GPULiquidPipeline(engine_config=self.engine_config)
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

    def prepare_motion_flow_intent(self, *args: Any, **kwargs: Any) -> Any:
        return prepare_motion_flow_intent(self, *args, **kwargs)

    def _build_solve_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _build_solve_tile_mask(self, *args, **kwargs)

    def _world_cell_reachable_empty(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_reachable_empty(self, *args, **kwargs)

    def _world_cell_is_tile_level_liquid(self, *args: Any, **kwargs: Any) -> Any:
        return _world_cell_is_tile_level_liquid(self, *args, **kwargs)

    def _solve_tile(self, *args: Any, **kwargs: Any) -> Any:
        return _solve_tile(self, *args, **kwargs)

    def _seam_correction(self, *args: Any, **kwargs: Any) -> Any:
        return _seam_correction(self, *args, **kwargs)

    def _apply_horizontal_seam_run(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_horizontal_seam_run(self, *args, **kwargs)

    def _apply_vertical_seam_run(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_vertical_seam_run(self, *args, **kwargs)

    def _apply_buoyancy(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_buoyancy(self, *args, **kwargs)

    def _apply_placeholder_displacement(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_placeholder_displacement(self, *args, **kwargs)

    def _placeholder_left_quota(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_left_quota(self, *args, **kwargs)

    def _placeholder_segment_top_exposed(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_segment_top_exposed(self, *args, **kwargs)

    def _placeholder_target_empty(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_target_empty(self, *args, **kwargs)

    def _placeholder_side_lane_reachable(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_side_lane_reachable(self, *args, **kwargs)

    def _placeholder_side_capacity(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_side_capacity(self, *args, **kwargs)

    def _placeholder_side_candidates(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_side_candidates(self, *args, **kwargs)

    def _mark_pending_placeholder_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_pending_placeholder_regions(self, *args, **kwargs)

    def _refresh_active_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_active_tiles(self, *args, **kwargs)

    def _vertical_seam_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _vertical_seam_mask(self, *args, **kwargs)

    def _horizontal_seam_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _horizontal_seam_mask(self, *args, **kwargs)

    def _buoyancy_candidate_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _buoyancy_candidate_mask(self, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def _finalize_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return _finalize_runtime_state(self, *args, **kwargs)

    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def reset_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return reset_runtime_state(self, *args, **kwargs)

    def runtime_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return runtime_snapshot(self, *args, **kwargs)

    def _material_table_row(self, *args: Any, **kwargs: Any) -> Any:
        return _material_table_row(self, *args, **kwargs)

    def _material_density(self, *args: Any, **kwargs: Any) -> Any:
        return _material_density(self, *args, **kwargs)

    def _material_base_integrity(self, *args: Any, **kwargs: Any) -> Any:
        return _material_base_integrity(self, *args, **kwargs)

    def _material_liquid_solver_kind(self, *args: Any, **kwargs: Any) -> Any:
        return _material_liquid_solver_kind(self, *args, **kwargs)

    def _placeholder_material_id(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_material_id(self, *args, **kwargs)

    def _material_is_placeholder(self, *args: Any, **kwargs: Any) -> Any:
        return _material_is_placeholder(self, *args, **kwargs)

    def _placeholder_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _placeholder_mask(self, *args, **kwargs)
