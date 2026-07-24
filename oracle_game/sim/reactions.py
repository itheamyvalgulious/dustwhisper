from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.engine_config import DEFAULT_ENGINE_CONFIG, EngineConfig
from oracle_game.sim.gpu_reactions import GPUReactionPipeline
from oracle_game.sim.reactions_actions import (
    _action_row,
    _append_gpu_emitted_lights,
    _apply_deferred_batch,
    _apply_gas_gas_consume,
    _apply_gas_light_consume,
    _apply_material_gas_consume,
    _apply_material_light_consume,
    _apply_material_material_consume,
    _apply_trigger_grid,
    _consume_gas_species,
    _consume_light_dose,
    _consume_material_cell,
    _consume_policy,
    _deferred_action_handled_by_gpu,
    _emit_modify_gas_flow_sources,
    _execute_action,
    _execute_gas_action,
    _execute_pair_rule,
    _mask_matches,
    _phase_mask_matches_values,
    _record_gpu_deferred_action,
    _record_gpu_emitted_materials,
    _record_gpu_local_action_counts,
    _rule_scale,
    _rule_value,
    _select_random_convert_material,
    _trigger_material_slot,
)

# Facade re-exports: the constants and solve-mask types live in the leaf
# module ``reactions_constants`` so the reactions_* satellites can import them
# without cycling back through this hub.
from oracle_game.sim.reactions_constants import (
    REACTION_ACTIVITY_EPSILON,  # noqa: F401  # facade re-export
    REACTION_FLOW_SOURCE_LIFETIME,  # noqa: F401  # facade re-export
    REACTION_STAGE_NAMES,  # noqa: F401  # facade re-export
    GPUAuthoritativeFullSolveMask,
    SolveMask,  # noqa: F401  # facade re-export
)
from oracle_game.sim.reactions_masks import (
    _active_scheduler_gpu_authoritative,
    _all_full_gpu_authoritative_masks,
    _capture_activity_state,
    _current_runtime_backend,
    _ensure_runtime_state,
    _finalize_stage_runtime,
    _formal_gpu_frame,
    _full_gpu_authoritative_solve_masks,
    _full_solve_masks,
    _is_full_gpu_authoritative_mask,
    _mark_tiles_from_cell_mask,
    _mark_tiles_from_gas_mask,
    _mark_tiles_from_mask,
    _note_runtime_backend,
    _record_stage_solve_masks,
    _refresh_active_regions,
    _require_materialized_cpu_solve_masks,
    _solve_mask_any,
    _solve_masks,
    _solve_tile_mask,
    _use_full_gpu_authoritative_reaction_solve_masks,
)
from oracle_game.sim.reactions_runners import (
    _advance_timed_slots,
    _run_gas_gas,
    _run_gas_light,
    _run_material_gas,
    _run_material_light,
    _run_material_material,
    _run_self_rules,
    _try_run_material_pair_fused,
)
from oracle_game.sim.reactions_runtime import (
    release,
    reset_runtime_state,
    runtime_snapshot,
    step,
)
from oracle_game.sim.reactions_selectors import (
    _best_matching_light_reaction_gas_species,
    _best_matching_material_reaction_gas_species,
    _deterministic_random_neighbor,
    _deterministic_selector,
    _direction_vector,
    _direction_vector_id,
    _gas_cell_center,
    _gas_direction_vector,
    _gas_direction_vector_id,
    _gas_tag_mask,
    _light_dose_channel,
    _light_emit_metadata,
    _match_material_selector,
    _matching_light_gas_species_ids,
    _matching_material_neighbor,
    _matching_material_reaction_gas_species_ids,
    _material_base_integrity,
    _material_default_phase,
    _material_emit_target_and_velocity,
    _material_reaction_slot,
    _material_tag_mask,
    _neighbor_for_direction,
    _neighbor_for_direction_id,
    _neighbor_for_gas_direction,
    _neighbor_for_gas_direction_id,
    _random_convert_candidates,
)

# Re-exported so tests can monkeypatch oracle_game.sim.reactions.tile_mask_to_*_mask
# (the reactions_masks bucket references these via this facade module).
from oracle_game.sim.utils import (  # noqa: F401  # monkeypatch target in tests
    tile_mask_to_cell_mask,
    tile_mask_to_gas_mask,
)


class ReactionSolver:
    def __init__(self, *, engine_config: EngineConfig | None = None) -> None:
        self.engine_config = engine_config if engine_config is not None else DEFAULT_ENGINE_CONFIG
        self.gpu_pipeline = GPUReactionPipeline(engine_config=self.engine_config)
        self.last_backend = "idle"
        self.last_runtime_backend = "idle"
        self._current_stage: str | None = None
        self._full_solve_mask_cache_signature: (
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
        ) = None
        self._full_solve_mask_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._full_gpu_authoritative_mask_cache_signature: (
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
        ) = None
        self._full_gpu_authoritative_mask_cache: (
            tuple[
                GPUAuthoritativeFullSolveMask,
                GPUAuthoritativeFullSolveMask,
                GPUAuthoritativeFullSolveMask,
            ]
            | None
        ) = None
        self.reset_runtime_state()

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

    # --- reactions_runners ---
    def _advance_timed_slots(self, *args: Any, **kwargs: Any) -> Any:
        return _advance_timed_slots(self, *args, **kwargs)

    def _run_self_rules(self, *args: Any, **kwargs: Any) -> Any:
        return _run_self_rules(self, *args, **kwargs)

    def _run_material_material(self, *args: Any, **kwargs: Any) -> Any:
        return _run_material_material(self, *args, **kwargs)

    def _try_run_material_pair_fused(self, *args: Any, **kwargs: Any) -> Any:
        return _try_run_material_pair_fused(self, *args, **kwargs)

    def _run_material_gas(self, *args: Any, **kwargs: Any) -> Any:
        return _run_material_gas(self, *args, **kwargs)

    def _run_material_light(self, *args: Any, **kwargs: Any) -> Any:
        return _run_material_light(self, *args, **kwargs)

    def _run_gas_gas(self, *args: Any, **kwargs: Any) -> Any:
        return _run_gas_gas(self, *args, **kwargs)

    def _run_gas_light(self, *args: Any, **kwargs: Any) -> Any:
        return _run_gas_light(self, *args, **kwargs)

    # --- reactions_masks ---
    def _solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _solve_masks(self, *args, **kwargs)

    def _use_full_gpu_authoritative_reaction_solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _use_full_gpu_authoritative_reaction_solve_masks(self, *args, **kwargs)

    def _full_gpu_authoritative_solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _full_gpu_authoritative_solve_masks(self, *args, **kwargs)

    def _full_solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _full_solve_masks(self, *args, **kwargs)

    @staticmethod
    def _is_full_gpu_authoritative_mask(*args: Any, **kwargs: Any) -> Any:
        return _is_full_gpu_authoritative_mask(*args, **kwargs)

    @staticmethod
    def _all_full_gpu_authoritative_masks(*args: Any, **kwargs: Any) -> Any:
        return _all_full_gpu_authoritative_masks(*args, **kwargs)

    @staticmethod
    def _solve_mask_any(*args: Any, **kwargs: Any) -> Any:
        return _solve_mask_any(*args, **kwargs)

    def _require_materialized_cpu_solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _require_materialized_cpu_solve_masks(self, *args, **kwargs)

    def _formal_gpu_frame(self, *args: Any, **kwargs: Any) -> Any:
        return _formal_gpu_frame(self, *args, **kwargs)

    def _active_scheduler_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return _active_scheduler_gpu_authoritative(self, *args, **kwargs)

    def _solve_tile_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _solve_tile_mask(self, *args, **kwargs)

    def _capture_activity_state(self, *args: Any, **kwargs: Any) -> Any:
        return _capture_activity_state(self, *args, **kwargs)

    def _refresh_active_regions(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_active_regions(self, *args, **kwargs)

    def _ensure_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_runtime_state(self, *args, **kwargs)

    def _record_stage_solve_masks(self, *args: Any, **kwargs: Any) -> Any:
        return _record_stage_solve_masks(self, *args, **kwargs)

    def _note_runtime_backend(self, *args: Any, **kwargs: Any) -> Any:
        return _note_runtime_backend(self, *args, **kwargs)

    def _current_runtime_backend(self, *args: Any, **kwargs: Any) -> Any:
        return _current_runtime_backend(self, *args, **kwargs)

    def _finalize_stage_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return _finalize_stage_runtime(self, *args, **kwargs)

    def _mark_tiles_from_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_tiles_from_mask(self, *args, **kwargs)

    def _mark_tiles_from_cell_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_tiles_from_cell_mask(self, *args, **kwargs)

    def _mark_tiles_from_gas_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_tiles_from_gas_mask(self, *args, **kwargs)

    # --- reactions_actions ---
    @staticmethod
    def _rule_value(*args: Any, **kwargs: Any) -> Any:
        return _rule_value(*args, **kwargs)

    def _execute_pair_rule(self, *args: Any, **kwargs: Any) -> Any:
        return _execute_pair_rule(self, *args, **kwargs)

    @staticmethod
    def _rule_scale(*args: Any, **kwargs: Any) -> Any:
        return _rule_scale(*args, **kwargs)

    @staticmethod
    def _consume_policy(*args: Any, **kwargs: Any) -> Any:
        return _consume_policy(*args, **kwargs)

    @staticmethod
    def _phase_mask_matches_values(*args: Any, **kwargs: Any) -> Any:
        return _phase_mask_matches_values(*args, **kwargs)

    def _apply_material_material_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_material_material_consume(self, *args, **kwargs)

    def _apply_material_gas_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_material_gas_consume(self, *args, **kwargs)

    def _apply_material_light_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_material_light_consume(self, *args, **kwargs)

    def _apply_gas_gas_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_gas_gas_consume(self, *args, **kwargs)

    def _apply_gas_light_consume(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_gas_light_consume(self, *args, **kwargs)

    def _consume_material_cell(self, *args: Any, **kwargs: Any) -> Any:
        return _consume_material_cell(self, *args, **kwargs)

    def _consume_gas_species(self, *args: Any, **kwargs: Any) -> Any:
        return _consume_gas_species(self, *args, **kwargs)

    def _consume_light_dose(self, *args: Any, **kwargs: Any) -> Any:
        return _consume_light_dose(self, *args, **kwargs)

    @staticmethod
    def _mask_matches(*args: Any, **kwargs: Any) -> Any:
        return _mask_matches(*args, **kwargs)

    def _trigger_material_slot(self, *args: Any, **kwargs: Any) -> Any:
        return _trigger_material_slot(self, *args, **kwargs)

    def _execute_action(self, *args: Any, **kwargs: Any) -> Any:
        return _execute_action(self, *args, **kwargs)

    def _execute_gas_action(self, *args: Any, **kwargs: Any) -> Any:
        return _execute_gas_action(self, *args, **kwargs)

    def _action_row(self, *args: Any, **kwargs: Any) -> Any:
        return _action_row(self, *args, **kwargs)

    def _apply_trigger_grid(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_trigger_grid(self, *args, **kwargs)

    def _apply_deferred_batch(self, *args: Any, **kwargs: Any) -> Any:
        return _apply_deferred_batch(self, *args, **kwargs)

    def _record_gpu_local_action_counts(self, *args: Any, **kwargs: Any) -> Any:
        return _record_gpu_local_action_counts(self, *args, **kwargs)

    def _append_gpu_emitted_lights(self, *args: Any, **kwargs: Any) -> Any:
        return _append_gpu_emitted_lights(self, *args, **kwargs)

    def _record_gpu_emitted_materials(self, *args: Any, **kwargs: Any) -> Any:
        return _record_gpu_emitted_materials(self, *args, **kwargs)

    def _record_gpu_deferred_action(self, *args: Any, **kwargs: Any) -> Any:
        return _record_gpu_deferred_action(self, *args, **kwargs)

    def _deferred_action_handled_by_gpu(self, *args: Any, **kwargs: Any) -> Any:
        return _deferred_action_handled_by_gpu(self, *args, **kwargs)

    def _select_random_convert_material(self, *args: Any, **kwargs: Any) -> Any:
        return _select_random_convert_material(self, *args, **kwargs)

    def _emit_modify_gas_flow_sources(self, *args: Any, **kwargs: Any) -> Any:
        return _emit_modify_gas_flow_sources(self, *args, **kwargs)

    # --- reactions_selectors ---
    def _match_material_selector(self, *args: Any, **kwargs: Any) -> Any:
        return _match_material_selector(self, *args, **kwargs)

    def _matching_material_neighbor(self, *args: Any, **kwargs: Any) -> Any:
        return _matching_material_neighbor(self, *args, **kwargs)

    def _best_matching_material_reaction_gas_species(self, *args: Any, **kwargs: Any) -> Any:
        return _best_matching_material_reaction_gas_species(self, *args, **kwargs)

    def _matching_material_reaction_gas_species_ids(self, *args: Any, **kwargs: Any) -> Any:
        return _matching_material_reaction_gas_species_ids(self, *args, **kwargs)

    def _best_matching_light_reaction_gas_species(self, *args: Any, **kwargs: Any) -> Any:
        return _best_matching_light_reaction_gas_species(self, *args, **kwargs)

    def _matching_light_gas_species_ids(self, *args: Any, **kwargs: Any) -> Any:
        return _matching_light_gas_species_ids(self, *args, **kwargs)

    def _light_dose_channel(self, *args: Any, **kwargs: Any) -> Any:
        return _light_dose_channel(self, *args, **kwargs)

    def _light_emit_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return _light_emit_metadata(self, *args, **kwargs)

    def _material_default_phase(self, *args: Any, **kwargs: Any) -> Any:
        return _material_default_phase(self, *args, **kwargs)

    def _material_base_integrity(self, *args: Any, **kwargs: Any) -> Any:
        return _material_base_integrity(self, *args, **kwargs)

    def _random_convert_candidates(self, *args: Any, **kwargs: Any) -> Any:
        return _random_convert_candidates(self, *args, **kwargs)

    def _material_reaction_slot(self, *args: Any, **kwargs: Any) -> Any:
        return _material_reaction_slot(self, *args, **kwargs)

    def _material_tag_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _material_tag_mask(self, *args, **kwargs)

    def _gas_tag_mask(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_tag_mask(self, *args, **kwargs)

    def _neighbor_for_direction(self, *args: Any, **kwargs: Any) -> Any:
        return _neighbor_for_direction(self, *args, **kwargs)

    def _neighbor_for_direction_id(self, *args: Any, **kwargs: Any) -> Any:
        return _neighbor_for_direction_id(self, *args, **kwargs)

    def _material_emit_target_and_velocity(self, *args: Any, **kwargs: Any) -> Any:
        return _material_emit_target_and_velocity(self, *args, **kwargs)

    @staticmethod
    def _deterministic_selector(*args: Any, **kwargs: Any) -> Any:
        return _deterministic_selector(*args, **kwargs)

    def _deterministic_random_neighbor(self, *args: Any, **kwargs: Any) -> Any:
        return _deterministic_random_neighbor(self, *args, **kwargs)

    def _neighbor_for_gas_direction(self, *args: Any, **kwargs: Any) -> Any:
        return _neighbor_for_gas_direction(self, *args, **kwargs)

    def _neighbor_for_gas_direction_id(self, *args: Any, **kwargs: Any) -> Any:
        return _neighbor_for_gas_direction_id(self, *args, **kwargs)

    def _direction_vector(self, *args: Any, **kwargs: Any) -> Any:
        return _direction_vector(self, *args, **kwargs)

    def _direction_vector_id(self, *args: Any, **kwargs: Any) -> Any:
        return _direction_vector_id(self, *args, **kwargs)

    def _gas_direction_vector(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_direction_vector(self, *args, **kwargs)

    def _gas_direction_vector_id(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_direction_vector_id(self, *args, **kwargs)

    def _gas_cell_center(self, *args: Any, **kwargs: Any) -> Any:
        return _gas_cell_center(self, *args, **kwargs)

    # --- reactions_runtime ---
    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self, *args, **kwargs)

    def release(self, *args: Any, **kwargs: Any) -> Any:
        return release(self, *args, **kwargs)

    def reset_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return reset_runtime_state(self, *args, **kwargs)

    def runtime_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return runtime_snapshot(self, *args, **kwargs)
