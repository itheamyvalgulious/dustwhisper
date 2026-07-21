from __future__ import annotations

import numpy as np

from oracle_game.sim.gpu_reactions import GPUReactionPipeline
# Re-exported so tests can monkeypatch oracle_game.sim.reactions.tile_mask_to_*_mask
# (the reactions_masks bucket references these via this facade module).
from oracle_game.sim.utils import tile_mask_to_cell_mask, tile_mask_to_gas_mask


REACTION_ACTIVITY_EPSILON = 1e-4
REACTION_FLOW_SOURCE_LIFETIME = 1.0 / 60.0
REACTION_STAGE_NAMES = (
    "timed",
    "self",
    "material_material",
    "material_gas",
    "material_light",
    "gas_gas",
    "gas_light",
)


class GPUAuthoritativeFullSolveMask:
    full_gpu_authoritative = True

    __slots__ = ("domain", "shape")

    def __init__(self, domain: str, shape: tuple[int, int]) -> None:
        self.domain = domain
        self.shape = shape

    def __array__(self, dtype: object | None = None, copy: object | None = None) -> np.ndarray:
        raise TypeError("GPU-authoritative full solve mask is not materialized on CPU")

    def copy(self) -> "GPUAuthoritativeFullSolveMask":
        return self

    def __repr__(self) -> str:
        return f"GPUAuthoritativeFullSolveMask(domain={self.domain!r}, shape={self.shape!r})"


SolveMask = np.ndarray | GPUAuthoritativeFullSolveMask


from oracle_game.sim.reactions_masks import (
    _solve_masks,
    _use_full_gpu_authoritative_reaction_solve_masks,
    _full_gpu_authoritative_solve_masks,
    _full_solve_masks,
    _is_full_gpu_authoritative_mask,
    _all_full_gpu_authoritative_masks,
    _solve_mask_any,
    _require_materialized_cpu_solve_masks,
    _formal_gpu_frame,
    _active_scheduler_gpu_authoritative,
    _solve_tile_mask,
    _capture_activity_state,
    _refresh_active_regions,
    _ensure_runtime_state,
    _record_stage_solve_masks,
    _note_runtime_backend,
    _current_runtime_backend,
    _finalize_stage_runtime,
    _mark_tiles_from_mask,
    _mark_tiles_from_cell_mask,
    _mark_tiles_from_gas_mask,
)
from oracle_game.sim.reactions_actions import (
    _rule_value,
    _execute_pair_rule,
    _rule_scale,
    _consume_policy,
    _phase_mask_matches_values,
    _apply_material_material_consume,
    _apply_material_gas_consume,
    _apply_material_light_consume,
    _apply_gas_gas_consume,
    _apply_gas_light_consume,
    _consume_material_cell,
    _consume_gas_species,
    _consume_light_dose,
    _mask_matches,
    _trigger_material_slot,
    _execute_action,
    _execute_gas_action,
    _action_row,
    _apply_trigger_grid,
    _apply_deferred_batch,
    _record_gpu_local_action_counts,
    _append_gpu_emitted_lights,
    _record_gpu_emitted_materials,
    _record_gpu_deferred_action,
    _deferred_action_handled_by_gpu,
    _select_random_convert_material,
    _emit_modify_gas_flow_sources,
)
from oracle_game.sim.reactions_selectors import (
    _match_material_selector,
    _matching_material_neighbor,
    _best_matching_material_reaction_gas_species,
    _matching_material_reaction_gas_species_ids,
    _best_matching_light_reaction_gas_species,
    _matching_light_gas_species_ids,
    _light_dose_channel,
    _light_emit_metadata,
    _material_default_phase,
    _material_base_integrity,
    _random_convert_candidates,
    _material_reaction_slot,
    _material_tag_mask,
    _gas_tag_mask,
    _neighbor_for_direction,
    _neighbor_for_direction_id,
    _material_emit_target_and_velocity,
    _deterministic_selector,
    _deterministic_random_neighbor,
    _neighbor_for_gas_direction,
    _neighbor_for_gas_direction_id,
    _direction_vector,
    _direction_vector_id,
    _gas_direction_vector,
    _gas_direction_vector_id,
    _gas_cell_center,
)
from oracle_game.sim.reactions_runners import (
    _advance_timed_slots,
    _run_self_rules,
    _run_material_material,
    _try_run_material_pair_fused,
    _run_material_gas,
    _run_material_light,
    _run_gas_gas,
    _run_gas_light,
)
from oracle_game.sim.reactions_runtime import (
    step,
    release,
    reset_runtime_state,
    runtime_snapshot,
)


class ReactionSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUReactionPipeline()
        self.last_backend = "idle"
        self.last_runtime_backend = "idle"
        self._current_stage: str | None = None
        self._full_solve_mask_cache_signature: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None
        self._full_solve_mask_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._full_gpu_authoritative_mask_cache_signature: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None
        self._full_gpu_authoritative_mask_cache: (
            tuple[GPUAuthoritativeFullSolveMask, GPUAuthoritativeFullSolveMask, GPUAuthoritativeFullSolveMask] | None
        ) = None
        self.reset_runtime_state()

    # --- reactions_runners ---
    _advance_timed_slots = _advance_timed_slots
    _run_self_rules = _run_self_rules
    _run_material_material = _run_material_material
    _try_run_material_pair_fused = _try_run_material_pair_fused
    _run_material_gas = _run_material_gas
    _run_material_light = _run_material_light
    _run_gas_gas = _run_gas_gas
    _run_gas_light = _run_gas_light

    # --- reactions_masks ---
    _solve_masks = _solve_masks
    _use_full_gpu_authoritative_reaction_solve_masks = _use_full_gpu_authoritative_reaction_solve_masks
    _full_gpu_authoritative_solve_masks = _full_gpu_authoritative_solve_masks
    _full_solve_masks = _full_solve_masks
    _is_full_gpu_authoritative_mask = staticmethod(_is_full_gpu_authoritative_mask)
    _all_full_gpu_authoritative_masks = staticmethod(_all_full_gpu_authoritative_masks)
    _solve_mask_any = staticmethod(_solve_mask_any)
    _require_materialized_cpu_solve_masks = _require_materialized_cpu_solve_masks
    _formal_gpu_frame = _formal_gpu_frame
    _active_scheduler_gpu_authoritative = _active_scheduler_gpu_authoritative
    _solve_tile_mask = _solve_tile_mask
    _capture_activity_state = _capture_activity_state
    _refresh_active_regions = _refresh_active_regions
    _ensure_runtime_state = _ensure_runtime_state
    _record_stage_solve_masks = _record_stage_solve_masks
    _note_runtime_backend = _note_runtime_backend
    _current_runtime_backend = _current_runtime_backend
    _finalize_stage_runtime = _finalize_stage_runtime
    _mark_tiles_from_mask = _mark_tiles_from_mask
    _mark_tiles_from_cell_mask = _mark_tiles_from_cell_mask
    _mark_tiles_from_gas_mask = _mark_tiles_from_gas_mask

    # --- reactions_actions ---
    _rule_value = staticmethod(_rule_value)
    _execute_pair_rule = _execute_pair_rule
    _rule_scale = staticmethod(_rule_scale)
    _consume_policy = staticmethod(_consume_policy)
    _phase_mask_matches_values = staticmethod(_phase_mask_matches_values)
    _apply_material_material_consume = _apply_material_material_consume
    _apply_material_gas_consume = _apply_material_gas_consume
    _apply_material_light_consume = _apply_material_light_consume
    _apply_gas_gas_consume = _apply_gas_gas_consume
    _apply_gas_light_consume = _apply_gas_light_consume
    _consume_material_cell = _consume_material_cell
    _consume_gas_species = _consume_gas_species
    _consume_light_dose = _consume_light_dose
    _mask_matches = staticmethod(_mask_matches)
    _trigger_material_slot = _trigger_material_slot
    _execute_action = _execute_action
    _execute_gas_action = _execute_gas_action
    _action_row = _action_row
    _apply_trigger_grid = _apply_trigger_grid
    _apply_deferred_batch = _apply_deferred_batch
    _record_gpu_local_action_counts = _record_gpu_local_action_counts
    _append_gpu_emitted_lights = _append_gpu_emitted_lights
    _record_gpu_emitted_materials = _record_gpu_emitted_materials
    _record_gpu_deferred_action = _record_gpu_deferred_action
    _deferred_action_handled_by_gpu = _deferred_action_handled_by_gpu
    _select_random_convert_material = _select_random_convert_material
    _emit_modify_gas_flow_sources = _emit_modify_gas_flow_sources

    # --- reactions_selectors ---
    _match_material_selector = _match_material_selector
    _matching_material_neighbor = _matching_material_neighbor
    _best_matching_material_reaction_gas_species = _best_matching_material_reaction_gas_species
    _matching_material_reaction_gas_species_ids = _matching_material_reaction_gas_species_ids
    _best_matching_light_reaction_gas_species = _best_matching_light_reaction_gas_species
    _matching_light_gas_species_ids = _matching_light_gas_species_ids
    _light_dose_channel = _light_dose_channel
    _light_emit_metadata = _light_emit_metadata
    _material_default_phase = _material_default_phase
    _material_base_integrity = _material_base_integrity
    _random_convert_candidates = _random_convert_candidates
    _material_reaction_slot = _material_reaction_slot
    _material_tag_mask = _material_tag_mask
    _gas_tag_mask = _gas_tag_mask
    _neighbor_for_direction = _neighbor_for_direction
    _neighbor_for_direction_id = _neighbor_for_direction_id
    _material_emit_target_and_velocity = _material_emit_target_and_velocity
    _deterministic_selector = staticmethod(_deterministic_selector)
    _deterministic_random_neighbor = _deterministic_random_neighbor
    _neighbor_for_gas_direction = _neighbor_for_gas_direction
    _neighbor_for_gas_direction_id = _neighbor_for_gas_direction_id
    _direction_vector = _direction_vector
    _direction_vector_id = _direction_vector_id
    _gas_direction_vector = _gas_direction_vector
    _gas_direction_vector_id = _gas_direction_vector_id
    _gas_cell_center = _gas_cell_center

    # --- reactions_runtime ---
    step = step
    release = release
    reset_runtime_state = reset_runtime_state
    runtime_snapshot = runtime_snapshot
