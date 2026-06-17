from __future__ import annotations

import numpy as np

from oracle_game.gpu import (
    DIRECTION_IDS,
    REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE,
    REACTION_ACTION_FLAG_RANDOM_TARGET,
)
from oracle_game.sim.gpu_reactions import (
    GPUReactionPipeline,
    TYPE_CONVERT_MATERIAL,
    TYPE_EMIT_LIGHT,
    TYPE_EMIT_MATERIAL,
    TYPE_HARM,
    TYPE_MODIFY_GAS,
    TYPE_MODIFY_TEMPERATURE,
)
from oracle_game.sim.utils import expand_bool_mask, tile_mask_to_cell_mask, tile_mask_to_gas_mask
from oracle_game.types import CellFlag, Direction, ForceSource, Phase, ReactionType


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


class ReactionSolver:
    def __init__(self) -> None:
        self.gpu_pipeline = GPUReactionPipeline()
        self.last_backend = "idle"
        self.last_runtime_backend = "idle"
        self._current_stage: str | None = None
        self.reset_runtime_state()

    def step(self, world: "WorldEngine", dt: float) -> None:
        self.reset_runtime_state(world)
        self._advance_timed_slots(world)
        self._run_self_rules(world)
        self._run_material_material(world)
        self._run_material_gas(world)
        self._run_material_light(world)
        self._run_gas_gas(world)
        self._run_gas_light(world)
        if self.gpu_pipeline.clear_reaction_latches(world):
            self._note_runtime_backend("gpu")
        else:
            world._require_gpu_stage("reaction latch clearing")
            world.cell_flags &= np.uint8(~int(CellFlag.REACTION_LATCHED) & 0xFF)
            self._note_runtime_backend("cpu")

    def _advance_timed_slots(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("timed", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "timed"
        deferred = self.gpu_pipeline.run_timed_actions(
            world,
            solve_cell_mask=solve_cell_mask,
        )
        if deferred is not None:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("timed reaction actions")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        timer_mask = solve_cell_mask & np.any(world.timer_pack > 0, axis=-1)
        trigger_grid = np.zeros((world.height, world.width, 4), dtype=np.int32)
        updated_timers = world.timer_pack.copy()
        ys, xs = np.nonzero(timer_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            material_id = int(world.material_id[y, x])
            for timer_index in range(4):
                timer_value = int(world.timer_pack[y, x, timer_index])
                if timer_value <= 0:
                    continue
                action_index = self._material_reaction_slot(world, material_id, timer_index)
                if action_index > 0:
                    trigger_grid[y, x, timer_index] = action_index
                updated_timers[y, x, timer_index] = max(0, timer_value - 1)
        world.timer_pack[:] = updated_timers
        self._apply_trigger_grid(world, trigger_grid)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_self_rules(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("self", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "self"
        deferred = self.gpu_pipeline.run_self_actions(
            world,
            solve_cell_mask=solve_cell_mask,
        )
        if deferred is not None:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("self reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        pre_material_id = world.material_id.copy()
        pre_phase = world.phase.copy()
        pre_temperature = world.cell_temperature.copy()
        pre_integrity = world.integrity.copy()
        updated_timers = world.timer_pack.copy()
        trigger_lo = np.zeros((world.height, world.width, 4), dtype=np.int32)
        trigger_hi = np.zeros((world.height, world.width, 4), dtype=np.int32)
        active_ys, active_xs = np.nonzero(solve_cell_mask)
        for rule in world.bridge.shadow_typed_tables["self_rule_table"]:
            material_values = pre_material_id[active_ys, active_xs]
            phase_values = pre_phase[active_ys, active_xs]
            temp_values = pre_temperature[active_ys, active_xs]
            integrity_values = pre_integrity[active_ys, active_xs]
            material_id = int(rule["material_id"])
            if material_id <= 0:
                continue
            mask = material_values == material_id
            phase_mask = int(rule["phase_mask"])
            if phase_mask != 0:
                mask &= self._phase_mask_matches_values(phase_values, phase_mask)
            mask &= temp_values >= float(rule["min_temperature"])
            mask &= temp_values <= float(rule["max_temperature"])
            integrity_at_most = float(rule["integrity_at_most"])
            integrity_at_least = float(rule["integrity_at_least"])
            if not np.isnan(integrity_at_most):
                mask &= integrity_values <= integrity_at_most
            if not np.isnan(integrity_at_least):
                mask &= integrity_values >= integrity_at_least
            slot_index = int(rule["trigger_slot_index"])
            for y, x in zip(active_ys[mask].tolist(), active_xs[mask].tolist()):
                source_material_id = int(pre_material_id[y, x])
                if source_material_id <= 0:
                    continue
                action_index = self._material_reaction_slot(world, source_material_id, slot_index)
                if action_index <= 0:
                    continue
                if slot_index < 4:
                    if int(updated_timers[y, x, slot_index]) > 0:
                        continue
                    action = self._action_row(world, action_index)
                    if action is not None and int(action["duration"]) > 0:
                        updated_timers[y, x, slot_index] = int(action["duration"])
                    trigger_lo[y, x, slot_index] = action_index
                    continue
                trigger_hi[y, x, slot_index - 4] = action_index
        world.timer_pack[:] = updated_timers
        self._apply_trigger_grid(world, trigger_lo)
        self._apply_trigger_grid(world, trigger_hi)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_material_material(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        if int(world.bridge.shadow_typed_tables["material_material_rule_table"].shape[0]) <= 0:
            return
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("material_material", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "material_material"
        deferred = self.gpu_pipeline.run_material_material(
            world,
            solve_cell_mask=solve_cell_mask,
        )
        if deferred is not None:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("material-material reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        material_snapshot = world.material_id.copy()
        phase_snapshot = world.phase.copy()
        temperature_snapshot = world.cell_temperature.copy()
        active_ys, active_xs = np.nonzero(solve_cell_mask)
        for rule in world.bridge.shadow_typed_tables["material_material_rule_table"]:
            lhs_id_raw = int(rule["lhs_material_id"])
            rhs_id_raw = int(rule["rhs_material_id"])
            lhs_id = lhs_id_raw if lhs_id_raw > 0 else None
            rhs_id = rhs_id_raw if rhs_id_raw > 0 else None
            rhs_selector_active = rhs_id is not None or int(rule["rhs_tag_mask"]) != 0
            if not rhs_selector_active:
                continue
            material_values = material_snapshot[active_ys, active_xs]
            phase_values = phase_snapshot[active_ys, active_xs]
            temperature_values = temperature_snapshot[active_ys, active_xs]
            candidate_mask = material_values > 0
            if lhs_id is not None:
                candidate_mask &= material_values == lhs_id
            phase_mask = int(rule["phase_mask"])
            if phase_mask != 0:
                candidate_mask &= self._phase_mask_matches_values(phase_values, phase_mask)
            candidate_mask &= temperature_values >= float(rule["min_temperature"])
            candidate_mask &= temperature_values <= float(rule["max_temperature"])
            for index in np.nonzero(candidate_mask)[0].tolist():
                y = int(active_ys[index])
                x = int(active_xs[index])
                material_id = int(material_snapshot[y, x])
                if material_id <= 0:
                    continue
                if not self._mask_matches(
                    self._material_tag_mask(world, material_id, "material_tag_mask"),
                    int(rule["lhs_tag_mask"]),
                ):
                    continue
                match_xy = self._matching_material_neighbor(
                    world,
                    x,
                    y,
                    required_id=rhs_id,
                    required_mask=int(rule["rhs_tag_mask"]),
                    material_grid=material_snapshot,
                )
                if match_xy is not None:
                    scale = self._rule_scale(rule, 1.0)
                    self._execute_pair_rule(world, rule, x, y, scale)
                    self._apply_material_material_consume(world, rule, x, y, match_xy, scale)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_material_gas(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        if int(world.bridge.shadow_typed_tables["material_gas_rule_table"].shape[0]) <= 0:
            return
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("material_gas", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "material_gas"
        deferred = self.gpu_pipeline.run_material_gas(
            world,
            solve_cell_mask=solve_cell_mask,
        )
        if deferred is not None:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("material-gas reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        gas_snapshot = world.gas_concentration.copy()
        active_ys, active_xs = np.nonzero(solve_cell_mask)
        for rule in world.bridge.shadow_typed_tables["material_gas_rule_table"]:
            lhs_id_raw = int(rule["lhs_material_id"])
            gas_id_raw = int(rule["rhs_gas_id"])
            lhs_id = lhs_id_raw if lhs_id_raw > 0 else None
            gas_id = gas_id_raw if gas_id_raw >= 0 else None
            rhs_selector_active = gas_id is not None or int(rule["rhs_tag_mask"]) != 0
            if not rhs_selector_active:
                continue
            material_values = world.material_id[active_ys, active_xs]
            phase_values = world.phase[active_ys, active_xs]
            temperature_values = world.cell_temperature[active_ys, active_xs]
            candidate_mask = material_values > 0
            if lhs_id is not None:
                candidate_mask &= material_values == lhs_id
            phase_mask = int(rule["phase_mask"])
            if phase_mask != 0:
                candidate_mask &= self._phase_mask_matches_values(phase_values, phase_mask)
            candidate_mask &= temperature_values >= float(rule["min_temperature"])
            candidate_mask &= temperature_values <= float(rule["max_temperature"])
            for index in np.nonzero(candidate_mask)[0].tolist():
                y = int(active_ys[index])
                x = int(active_xs[index])
                material_id = int(world.material_id[y, x])
                if material_id <= 0:
                    continue
                if not self._mask_matches(
                    self._material_tag_mask(world, material_id, "gas_tag_mask"),
                    int(rule["lhs_tag_mask"]),
                ):
                    continue
                gy, gx = world.cell_to_gas(y, x)
                species_id, concentration = self._best_matching_material_reaction_gas_species(
                    world,
                    gy,
                    gx,
                    gas_id=gas_id,
                    required_mask=int(rule["rhs_tag_mask"]),
                    gas_concentration=gas_snapshot,
                )
                if concentration >= float(rule["threshold"]):
                    scale = self._rule_scale(rule, concentration)
                    self._execute_pair_rule(world, rule, x, y, scale)
                    self._apply_material_gas_consume(world, rule, x, y, gy, gx, species_id, scale)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_material_light(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        if int(world.bridge.shadow_typed_tables["material_light_rule_table"].shape[0]) <= 0:
            return
        if self._formal_gpu_frame(world) and getattr(world, "_formal_gpu_frame_has_light_dose", None) is False:
            return
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("material_light", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "material_light"
        deferred = self.gpu_pipeline.run_material_light(
            world,
            solve_cell_mask=solve_cell_mask,
        )
        if deferred is not None:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("material-light reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        active_ys, active_xs = np.nonzero(solve_cell_mask)
        for rule in world.bridge.shadow_typed_tables["material_light_rule_table"]:
            lhs_id_raw = int(rule["lhs_material_id"])
            light_id_raw = int(rule["rhs_light_id"])
            lhs_id = lhs_id_raw if lhs_id_raw > 0 else None
            light_id = light_id_raw if light_id_raw >= 0 else None
            if light_id is None:
                continue
            dose_channel = self._light_dose_channel(world, light_id)
            if dose_channel is None:
                continue
            material_values = world.material_id[active_ys, active_xs]
            phase_values = world.phase[active_ys, active_xs]
            temperature_values = world.cell_temperature[active_ys, active_xs]
            dose_values = world.cell_optical_dose[dose_channel, active_ys, active_xs]
            mask = material_values > 0
            mask &= dose_values >= float(rule["threshold"])
            if lhs_id is not None:
                mask &= material_values == lhs_id
            if int(rule["lhs_tag_mask"]) != 0:
                required_mask = np.uint32(int(rule["lhs_tag_mask"]))
                light_masks = np.asarray(
                    [self._material_tag_mask(world, int(material_id), "light_tag_mask") for material_id in material_values],
                    dtype=np.uint32,
                )
                mask &= (light_masks & required_mask) == required_mask
            phase_mask = int(rule["phase_mask"])
            if phase_mask != 0:
                mask &= self._phase_mask_matches_values(phase_values, phase_mask)
            mask &= temperature_values >= float(rule["min_temperature"])
            mask &= temperature_values <= float(rule["max_temperature"])
            for index in np.nonzero(mask)[0].tolist():
                y = int(active_ys[index])
                x = int(active_xs[index])
                scale = self._rule_scale(rule, float(dose_values[index]))
                self._execute_pair_rule(world, rule, x, y, scale)
                self._apply_material_light_consume(world, rule, x, y, dose_channel, scale)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_gas_gas(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        if int(world.bridge.shadow_typed_tables["gas_gas_rule_table"].shape[0]) <= 0:
            return
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("gas_gas", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "gas_gas"
        deferred = self.gpu_pipeline.run_gas_gas(
            world,
            solve_gas_mask=solve_gas_mask,
        )
        if deferred is not None and deferred is not False:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            if deferred is not True:
                self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("gas-gas reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        active_gy, active_gx = np.nonzero(solve_gas_mask)
        for rule in world.bridge.shadow_typed_tables["gas_gas_rule_table"]:
            lhs_id_raw = int(rule["lhs_gas_id"])
            rhs_id_raw = int(rule["rhs_gas_id"])
            lhs_id = lhs_id_raw if lhs_id_raw >= 0 else None
            rhs_id = rhs_id_raw if rhs_id_raw >= 0 else None
            lhs_tag_mask = int(rule["lhs_tag_mask"])
            rhs_tag_mask = int(rule["rhs_tag_mask"])
            lhs_selector_active = lhs_id is not None or lhs_tag_mask != 0
            rhs_selector_active = rhs_id is not None or rhs_tag_mask != 0
            if not lhs_selector_active or not rhs_selector_active:
                continue
            lhs_species = self._matching_material_reaction_gas_species_ids(
                world,
                gas_id=lhs_id,
                required_mask=lhs_tag_mask,
            )
            rhs_species = self._matching_material_reaction_gas_species_ids(
                world,
                gas_id=rhs_id,
                required_mask=rhs_tag_mask,
            )
            if not lhs_species or not rhs_species:
                continue
            lhs_values = np.zeros(active_gy.shape, dtype=np.float32)
            rhs_values = np.zeros(active_gy.shape, dtype=np.float32)
            for species_id in lhs_species:
                lhs_values = np.maximum(lhs_values, world.gas_concentration[species_id, active_gy, active_gx])
            for species_id in rhs_species:
                rhs_values = np.maximum(rhs_values, world.gas_concentration[species_id, active_gy, active_gx])
            ambient_values = world.ambient_temperature[active_gy, active_gx]
            mask = (lhs_values >= float(rule["threshold"])) & (rhs_values >= float(rule["threshold"]))
            mask &= ambient_values >= float(rule["min_temperature"])
            mask &= ambient_values <= float(rule["max_temperature"])
            for index in np.nonzero(mask)[0].tolist():
                gy = int(active_gy[index])
                gx = int(active_gx[index])
                lhs_species_id, _ = self._best_matching_material_reaction_gas_species(
                    world,
                    gy,
                    gx,
                    gas_id=lhs_id,
                    required_mask=lhs_tag_mask,
                )
                rhs_species_id, _ = self._best_matching_material_reaction_gas_species(
                    world,
                    gy,
                    gx,
                    gas_id=rhs_id,
                    required_mask=rhs_tag_mask,
                )
                scale = self._rule_scale(rule, min(float(lhs_values[index]), float(rhs_values[index])))
                self._execute_gas_action(world, int(rule["result_action"]), gx, gy, scale)
                self._apply_gas_gas_consume(world, rule, gx, gy, lhs_species_id, rhs_species_id, scale)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _run_gas_light(self, world: "WorldEngine") -> None:
        self._ensure_runtime_state(world)
        if int(world.bridge.shadow_typed_tables["gas_light_rule_table"].shape[0]) <= 0:
            return
        if self._formal_gpu_frame(world) and getattr(world, "_formal_gpu_frame_has_light_dose", None) is False:
            return
        solve_tile_mask, solve_cell_mask, solve_gas_mask = self._solve_masks(world, seed_timer_cells=True)
        self._record_stage_solve_masks("gas_light", solve_tile_mask, solve_cell_mask, solve_gas_mask)
        if not np.any(solve_tile_mask):
            return
        previous_state = self._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
        self._current_stage = "gas_light"
        deferred = self.gpu_pipeline.run_gas_light(
            world,
            solve_gas_mask=solve_gas_mask,
        )
        if deferred is not None and deferred is not False:
            self.last_backend = "gpu"
            self._note_runtime_backend("gpu")
            if deferred is not True:
                self._apply_deferred_batch(world, deferred)
            self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
            return
        world._require_gpu_stage("gas-light reaction rules")
        self.last_backend = "cpu"
        self._note_runtime_backend("cpu")
        active_gy, active_gx = np.nonzero(solve_gas_mask)
        for rule in world.bridge.shadow_typed_tables["gas_light_rule_table"]:
            gas_id_raw = int(rule["rhs_gas_id"])
            light_id_raw = int(rule["rhs_light_id"])
            gas_id = gas_id_raw if gas_id_raw >= 0 else None
            light_id = light_id_raw if light_id_raw >= 0 else None
            rhs_tag_mask = int(rule["rhs_tag_mask"])
            if gas_id is None or light_id is None:
                if rhs_tag_mask == 0 or light_id is None:
                    continue
            dose_channel = self._light_dose_channel(world, light_id)
            if dose_channel is None:
                continue
            dose_values = world.gas_optical_dose[dose_channel, active_gy, active_gx]
            ambient_values = world.ambient_temperature[active_gy, active_gx]
            candidate_species = self._matching_light_gas_species_ids(world, gas_id=gas_id, required_mask=rhs_tag_mask)
            if not candidate_species:
                continue
            best_gas_values = np.zeros_like(dose_values, dtype=np.float32)
            for species_id in candidate_species:
                best_gas_values = np.maximum(best_gas_values, world.gas_concentration[species_id, active_gy, active_gx])
            trigger_mask = (best_gas_values >= float(rule["threshold"])) & (dose_values >= float(rule["threshold"]))
            trigger_mask &= ambient_values >= float(rule["min_temperature"])
            trigger_mask &= ambient_values <= float(rule["max_temperature"])
            for index in np.nonzero(trigger_mask)[0].tolist():
                gy = int(active_gy[index])
                gx = int(active_gx[index])
                species_id, _ = self._best_matching_light_reaction_gas_species(
                    world,
                    gy,
                    gx,
                    gas_id=gas_id,
                    required_mask=rhs_tag_mask,
                )
                scale = self._rule_scale(rule, min(float(best_gas_values[index]), float(dose_values[index])))
                self._execute_gas_action(world, int(rule["result_action"]), gx, gy, scale)
                self._apply_gas_light_consume(world, rule, gx, gy, species_id, scale)
        self._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)

    def _solve_masks(
        self,
        world: "WorldEngine",
        *,
        seed_timer_cells: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        formal_gpu_frame = self._formal_gpu_frame(world)
        active_scheduler_gpu_authoritative = self._active_scheduler_gpu_authoritative(world)
        if formal_gpu_frame and not active_scheduler_gpu_authoritative:
            world._require_gpu_stage("active scheduler reaction solve masks")
        if active_scheduler_gpu_authoritative:
            solve_tile_mask = np.ones((world.active.tile_height, world.active.tile_width), dtype=np.bool_)
        else:
            solve_tile_mask = self._solve_tile_mask(world, seed_timer_cells=seed_timer_cells)
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
        return solve_tile_mask, solve_cell_mask, solve_gas_mask

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _active_scheduler_gpu_authoritative(self, world: "WorldEngine") -> bool:
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        )

    def _solve_tile_mask(self, world: "WorldEngine", *, seed_timer_cells: bool) -> np.ndarray:
        active_tiles = np.asarray(world.active.active_tile_ttl, dtype=np.int32) > 0
        seeded_tiles = active_tiles.copy()
        if seed_timer_cells:
            tile_size = world.active.tile_size
            timer_mask = np.any(world.timer_pack > 0, axis=-1)
            for y, x in np.argwhere(timer_mask):
                tile_x = min(world.active.tile_width - 1, int(x) // tile_size)
                tile_y = min(world.active.tile_height - 1, int(y) // tile_size)
                seeded_tiles[tile_y, tile_x] = True
        return expand_bool_mask(seeded_tiles, radius=1)

    def _capture_activity_state(
        self,
        world: "WorldEngine",
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
    ) -> dict[str, object]:
        self._stage_extra_changed_cell_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        if self._formal_gpu_frame(world):
            return {
                "formal_gpu_frame": True,
                "emitters": len(world.emitters),
            }
        return {
            "formal_gpu_frame": False,
            "material_id": world.material_id[solve_cell_mask].copy(),
            "phase": world.phase[solve_cell_mask].copy(),
            "cell_temperature": world.cell_temperature[solve_cell_mask].copy(),
            "integrity": world.integrity[solve_cell_mask].copy(),
            "timer_pack": world.timer_pack[solve_cell_mask].copy(),
            "gas_concentration": world.gas_concentration[:, solve_gas_mask].copy(),
            "ambient_temperature": world.ambient_temperature[solve_gas_mask].copy(),
            "emitters": len(world.emitters),
        }

    def _refresh_active_regions(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        changed_cell_mask: np.ndarray,
        changed_gas_mask: np.ndarray,
        ambient_changed_mask: np.ndarray,
        timer_changed_mask: np.ndarray,
        *,
        source_emitted: bool,
    ) -> None:
        any_cell_changed = bool(np.any(changed_cell_mask))
        any_gas_changed = bool(np.any(changed_gas_mask))
        any_ambient_changed = bool(np.any(ambient_changed_mask))
        any_timer_changed = bool(np.any(timer_changed_mask))
        if not (
            any_cell_changed
            or any_gas_changed
            or any_ambient_changed
            or any_timer_changed
            or source_emitted
        ):
            return
        if np.any(solve_tile_mask):
            self._mark_tiles_from_mask(world, solve_tile_mask)
        if any_cell_changed or any_timer_changed:
            self._mark_tiles_from_cell_mask(world, changed_cell_mask | timer_changed_mask, tile_padding=1)
        if any_gas_changed or any_ambient_changed:
            self._mark_tiles_from_gas_mask(world, changed_gas_mask | ambient_changed_mask, tile_padding=1)

    def _ensure_runtime_state(self, world: "WorldEngine") -> None:
        world.bridge.sync_rule_tables(world)
        if self.last_solve_cell_mask.shape != (world.height, world.width):
            self.reset_runtime_state(world)

    def _record_stage_solve_masks(
        self,
        stage: str,
        solve_tile_mask: np.ndarray,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
    ) -> None:
        self.last_stage_tile_masks[stage] = np.asarray(solve_tile_mask, dtype=np.bool_).copy()
        self.last_solve_cell_mask |= np.asarray(solve_cell_mask, dtype=np.bool_)
        self.last_solve_gas_mask |= np.asarray(solve_gas_mask, dtype=np.bool_)

    def _note_runtime_backend(self, backend: str) -> None:
        if backend == "gpu":
            self._runtime_used_gpu = True
        else:
            self._runtime_used_cpu = True
        self.last_runtime_backend = self._current_runtime_backend()

    def _current_runtime_backend(self) -> str:
        if self._runtime_used_cpu and self._runtime_used_gpu:
            return "hybrid"
        if self._runtime_used_gpu:
            return "gpu"
        return "cpu"

    def _finalize_stage_runtime(
        self,
        world: "WorldEngine",
        solve_tile_mask: np.ndarray,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
        previous_state: dict[str, object],
    ) -> None:
        if bool(previous_state.get("formal_gpu_frame", False)):
            self.last_changed_cell_mask |= solve_cell_mask | self._stage_extra_changed_cell_mask
            self.last_changed_gas_mask |= solve_gas_mask
            self.last_ambient_changed_mask |= solve_gas_mask
            self.last_timer_changed_mask |= solve_cell_mask
            self._current_stage = None
            if not self._active_scheduler_gpu_authoritative(world):
                world._require_gpu_stage("active scheduler reaction refresh")
            return
        material_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        phase_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        timer_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        cell_temperature_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        integrity_changed_mask = np.zeros((world.height, world.width), dtype=np.bool_)
        if np.any(solve_cell_mask):
            material_changed_mask[solve_cell_mask] = world.material_id[solve_cell_mask] != previous_state["material_id"]
            phase_changed_mask[solve_cell_mask] = world.phase[solve_cell_mask] != previous_state["phase"]
            timer_changed_mask[solve_cell_mask] = np.any(
                world.timer_pack[solve_cell_mask] != previous_state["timer_pack"],
                axis=-1,
            )
            cell_temperature_changed_mask[solve_cell_mask] = (
                np.abs(world.cell_temperature[solve_cell_mask] - previous_state["cell_temperature"]) > REACTION_ACTIVITY_EPSILON
            )
            integrity_changed_mask[solve_cell_mask] = (
                np.abs(world.integrity[solve_cell_mask] - previous_state["integrity"]) > REACTION_ACTIVITY_EPSILON
            )
        gas_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
        ambient_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.bool_)
        if np.any(solve_gas_mask):
            gas_changed_mask[solve_gas_mask] = np.any(
                np.abs(world.gas_concentration[:, solve_gas_mask] - previous_state["gas_concentration"]) > REACTION_ACTIVITY_EPSILON,
                axis=0,
            )
            ambient_changed_mask[solve_gas_mask] = (
                np.abs(world.ambient_temperature[solve_gas_mask] - previous_state["ambient_temperature"]) > REACTION_ACTIVITY_EPSILON
            )
        material_changed_mask |= self._stage_extra_changed_cell_mask
        self.last_changed_cell_mask |= (
            material_changed_mask
            | phase_changed_mask
            | timer_changed_mask
            | cell_temperature_changed_mask
            | integrity_changed_mask
        )
        self.last_changed_gas_mask |= gas_changed_mask
        self.last_ambient_changed_mask |= ambient_changed_mask
        self.last_timer_changed_mask |= timer_changed_mask
        source_emitted = len(world.emitters) > int(previous_state["emitters"])
        self._current_stage = None
        if not self._active_scheduler_gpu_authoritative(world):
            self._refresh_active_regions(
                world,
                solve_tile_mask,
                material_changed_mask | phase_changed_mask | cell_temperature_changed_mask | integrity_changed_mask,
                gas_changed_mask,
                ambient_changed_mask,
                timer_changed_mask,
                source_emitted=source_emitted,
            )

    def _mark_tiles_from_mask(self, world: "WorldEngine", solve_tile_mask: np.ndarray) -> None:
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int]] = []
        for tile_y, tile_x in np.argwhere(solve_tile_mask):
            x0 = int(tile_x) * tile_size
            y0 = int(tile_y) * tile_size
            rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size)))
        world._mark_active_rects_runtime(rects)

    def _mark_tiles_from_cell_mask(
        self,
        world: "WorldEngine",
        cell_mask: np.ndarray,
        *,
        tile_padding: int = 0,
    ) -> None:
        tile_size = world.active.tile_size
        rects: list[tuple[int, int, int, int, int]] = []
        for tile_y, tile_x in {
            (int(y) // tile_size, int(x) // tile_size)
            for y, x in np.argwhere(cell_mask)
        }:
            x0 = tile_x * tile_size
            y0 = tile_y * tile_size
            rects.append((x0, y0, min(world.width, x0 + tile_size), min(world.height, y0 + tile_size), tile_padding))
        world._mark_active_rects_runtime(rects)

    def _mark_tiles_from_gas_mask(
        self,
        world: "WorldEngine",
        gas_mask: np.ndarray,
        *,
        tile_padding: int = 0,
    ) -> None:
        gas_cell_size = world.gas_cell_size
        rects: list[tuple[int, int, int, int, int]] = []
        for gy, gx in np.argwhere(gas_mask):
            x0 = int(gx) * gas_cell_size
            y0 = int(gy) * gas_cell_size
            rects.append((x0, y0, min(world.width, x0 + gas_cell_size), min(world.height, y0 + gas_cell_size), tile_padding))
        world._mark_active_rects_runtime(rects)

    @staticmethod
    def _rule_value(rule: object, field: str, default: object | None = None) -> object | None:
        dtype = getattr(rule, "dtype", None)
        names = None if dtype is None else getattr(dtype, "names", None)
        if names is not None and field in names:
            return rule[field]
        return getattr(rule, field, default)

    def _execute_pair_rule(self, world: "WorldEngine", rule: object, x: int, y: int, scale: float) -> None:
        trigger_slot_index = self._rule_value(rule, "trigger_slot_index", None)
        if trigger_slot_index is not None and int(trigger_slot_index) >= 0:
            self._trigger_material_slot(world, x, y, int(trigger_slot_index), scale=scale)
            return
        result_action = self._rule_value(rule, "result_action", -1)
        if result_action is not None and int(result_action) >= 0:
            self._execute_action(world, int(result_action), x, y, scale)

    def _match_material_selector(
        self,
        world: "WorldEngine",
        x: int,
        y: int,
        *,
        required_id: int | None,
        required_mask: int,
        material_grid: np.ndarray | None = None,
    ) -> bool:
        if not world.in_bounds(x, y):
            return False
        material_field = world.material_id if material_grid is None else material_grid
        material_id = int(material_field[y, x])
        if material_id <= 0:
            return False
        if required_id is not None and material_id != required_id:
            return False
        if required_mask == 0:
            return required_id is not None
        return self._mask_matches(self._material_tag_mask(world, material_id, "material_tag_mask"), required_mask)

    def _matching_material_neighbor(
        self,
        world: "WorldEngine",
        x: int,
        y: int,
        *,
        required_id: int | None,
        required_mask: int,
        material_grid: np.ndarray | None = None,
    ) -> tuple[int, int] | None:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if self._match_material_selector(
                world,
                nx,
                ny,
                required_id=required_id,
                required_mask=required_mask,
                material_grid=material_grid,
            ):
                return (nx, ny)
        return None

    def _best_matching_material_reaction_gas_species(
        self,
        world: "WorldEngine",
        gy: int,
        gx: int,
        *,
        gas_id: int | None,
        required_mask: int,
        gas_concentration: np.ndarray | None = None,
    ) -> tuple[int | None, float]:
        species_ids = self._matching_material_reaction_gas_species_ids(
            world,
            gas_id=gas_id,
            required_mask=required_mask,
        )
        if not species_ids:
            return (None, 0.0)
        gas_field = world.gas_concentration if gas_concentration is None else gas_concentration
        best_species_id: int | None = None
        best = -1.0
        for species_id in species_ids:
            value = float(gas_field[species_id, gy, gx])
            if value > best:
                best = value
                best_species_id = int(species_id)
        return (best_species_id, max(0.0, best))

    def _matching_material_reaction_gas_species_ids(
        self,
        world: "WorldEngine",
        *,
        gas_id: int | None,
        required_mask: int,
    ) -> list[int]:
        if gas_id is not None:
            if gas_id < 0:
                return []
            if not self._mask_matches(self._gas_tag_mask(world, gas_id, "material_reaction_tag_mask"), required_mask):
                return []
            return [int(gas_id)]
        if required_mask == 0:
            return []
        species_limit = int(
            world.bridge.shadow_typed_tables["gas_table"].shape[0]
            if world.bridge.shadow_typed_tables.get("gas_table") is not None
            else world.gas_material_reaction_tag_mask.shape[0]
        )
        return [
            int(species_id)
            for species_id in range(species_limit)
            if self._mask_matches(self._gas_tag_mask(world, species_id, "material_reaction_tag_mask"), required_mask)
        ]

    def _best_matching_light_reaction_gas_species(
        self,
        world: "WorldEngine",
        gy: int,
        gx: int,
        *,
        gas_id: int | None,
        required_mask: int,
    ) -> tuple[int | None, float]:
        species_ids = self._matching_light_gas_species_ids(
            world,
            gas_id=gas_id,
            required_mask=required_mask,
        )
        if not species_ids:
            return (None, 0.0)
        best_species_id: int | None = None
        best = -1.0
        for species_id in species_ids:
            value = float(world.gas_concentration[species_id, gy, gx])
            if value > best:
                best = value
                best_species_id = int(species_id)
        return (best_species_id, max(0.0, best))

    def _matching_light_gas_species_ids(
        self,
        world: "WorldEngine",
        *,
        gas_id: int | None,
        required_mask: int,
    ) -> list[int]:
        if gas_id is not None:
            if gas_id < 0:
                return []
            if not self._mask_matches(self._gas_tag_mask(world, gas_id, "light_reaction_tag_mask"), required_mask):
                return []
            return [int(gas_id)]
        if required_mask == 0:
            return []
        species_limit = int(
            world.bridge.shadow_typed_tables["gas_table"].shape[0]
            if world.bridge.shadow_typed_tables.get("gas_table") is not None
            else world.gas_light_reaction_tag_mask.shape[0]
        )
        return [
            int(species_id)
            for species_id in range(species_limit)
            if self._mask_matches(self._gas_tag_mask(world, species_id, "light_reaction_tag_mask"), required_mask)
        ]

    def _light_dose_channel(self, world: "WorldEngine", light_id: int) -> int | None:
        light_table = world.bridge.shadow_typed_tables.get("light_table")
        if light_table is not None and 0 <= light_id < int(light_table.shape[0]):
            if int(light_table[light_id]["name_hash"]) == 0:
                return None
            dose_channel = int(light_table[light_id]["dose_channel_id"])
            if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
                return dose_channel
            return None
        light_payload = world._shadow_light_type_payload()
        if isinstance(light_payload, list):
            for item in light_payload:
                if int(item.get("light_type_id", -1)) != light_id:
                    continue
                dose_channel = int(item.get("dose_channel_id", -1))
                if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
                    return dose_channel
                return None
            return None
        if 0 <= light_id < world.light_dose_channel.shape[0]:
            dose_channel = int(world.light_dose_channel[light_id])
            if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
                return dose_channel
        return None

    def _light_emit_metadata(self, world: "WorldEngine", light_id: int) -> tuple[str, int] | None:
        return world._shadow_light_name_and_range(light_id)

    def _material_default_phase(self, world: "WorldEngine", material_id: int) -> int | None:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
            if int(material_table[material_id]["name_hash"]) == 0:
                return None
            return int(material_table[material_id]["default_phase"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return int(shadow_material.default_phase)
        if world._shadow_has_table_payload("materials"):
            return None
        if 0 <= material_id < world.material_default_phase.shape[0]:
            return int(world.material_default_phase[material_id])
        return None

    def _material_base_integrity(self, world: "WorldEngine", material_id: int) -> float | None:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
            if int(material_table[material_id]["name_hash"]) == 0:
                return None
            return float(material_table[material_id]["base_integrity"])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return float(shadow_material.base_integrity)
        if world._shadow_has_table_payload("materials"):
            return None
        if 0 <= material_id < world.material_base_integrity.shape[0]:
            return float(world.material_base_integrity[material_id])
        return None

    def _random_convert_candidates(self, world: "WorldEngine") -> list[int]:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        chaos_convert_bit = int(world.tag_bits_by_name.get("chaos_convert", 0))
        if material_table is not None and chaos_convert_bit != 0:
            return [
                int(row["material_id"])
                for row in material_table
                if int(row["material_id"]) > 0
                and int(row["name_hash"]) != 0
                and bool(int(row["material_tag_mask"]) & chaos_convert_bit)
                and int(row["default_phase"]) == int(Phase.POWDER)
            ]
        if material_table is not None:
            return []
        if world.random_convert_material_ids:
            return [int(material_id) for material_id in world.random_convert_material_ids if int(material_id) > 0]
        return []

    def _material_reaction_slot(self, world: "WorldEngine", material_id: int, slot_index: int) -> int:
        if slot_index < 0 or slot_index >= 8:
            return -1
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
            if int(material_table[material_id]["name_hash"]) == 0:
                return -1
            return int(material_table[material_id]["reaction_slots"][slot_index])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            return int(shadow_material.reaction_slots[slot_index])
        if world._shadow_has_table_payload("materials"):
            return -1
        if 0 <= material_id < world.material_reaction_slots.shape[0]:
            return int(world.material_reaction_slots[material_id, slot_index])
        return -1

    def _material_tag_mask(self, world: "WorldEngine", material_id: int, field: str) -> int:
        material_table = world.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
            if int(material_table[material_id]["name_hash"]) == 0:
                return 0
            return int(material_table[material_id][field])
        shadow_material = world._shadow_material_def(material_id)
        if shadow_material is not None:
            if field == "material_tag_mask":
                return int(shadow_material.material_tag_mask)
            if field == "gas_tag_mask":
                return int(shadow_material.gas_tag_mask)
            if field == "light_tag_mask":
                return int(shadow_material.light_tag_mask)
        if world._shadow_has_table_payload("materials"):
            return 0
        fallback_map = {
            "material_tag_mask": world.material_material_tag_mask,
            "gas_tag_mask": world.material_gas_tag_mask,
            "light_tag_mask": world.material_light_tag_mask,
        }
        fallback = fallback_map[field]
        if 0 <= material_id < fallback.shape[0]:
            return int(fallback[material_id])
        return 0

    def _gas_tag_mask(self, world: "WorldEngine", species_id: int, field: str) -> int:
        gas_table = world.bridge.shadow_typed_tables.get("gas_table")
        if gas_table is not None and 0 <= species_id < int(gas_table.shape[0]):
            if int(gas_table[species_id]["name_hash"]) == 0:
                return 0
            return int(gas_table[species_id][field])
        shadow_gas = world._shadow_gas_species_def(species_id)
        if shadow_gas is not None:
            if field == "material_reaction_tag_mask":
                return int(shadow_gas.material_reaction_tag_mask)
            if field == "light_reaction_tag_mask":
                return int(shadow_gas.light_reaction_tag_mask)
        if world._shadow_has_table_payload("gases"):
            return 0
        fallback_map = {
            "material_reaction_tag_mask": world.gas_material_reaction_tag_mask,
            "light_reaction_tag_mask": world.gas_light_reaction_tag_mask,
        }
        fallback = fallback_map[field]
        if 0 <= species_id < fallback.shape[0]:
            return int(fallback[species_id])
        return 0

    @staticmethod
    def _rule_scale(rule: object, base_scale: float) -> float:
        return max(0.0, float(base_scale) * float(ReactionSolver._rule_value(rule, "rate", 1.0)))

    @staticmethod
    def _consume_policy(rule: object) -> str:
        policy_id = ReactionSolver._rule_value(rule, "consume_policy_id", None)
        if policy_id is not None:
            return {
                1: "lhs",
                2: "rhs",
                3: "both",
            }.get(int(policy_id), "none")
        return str(ReactionSolver._rule_value(rule, "consume_policy", "none") or "none").lower()

    @staticmethod
    def _phase_mask_matches_values(phase_values: np.ndarray, phase_mask: int) -> np.ndarray:
        if phase_mask == 0:
            return np.ones_like(phase_values, dtype=np.bool_)
        phase_bits = np.left_shift(np.uint32(1), phase_values.astype(np.uint32, copy=False))
        return (phase_bits & np.uint32(phase_mask)) != 0

    def _apply_material_material_consume(
        self,
        world: "WorldEngine",
        rule: object,
        x: int,
        y: int,
        rhs_xy: tuple[int, int],
        scale: float,
    ) -> None:
        policy = self._consume_policy(rule)
        if policy in {"lhs", "both"}:
            self._consume_material_cell(world, x, y, scale)
        if policy in {"rhs", "both"}:
            self._consume_material_cell(world, rhs_xy[0], rhs_xy[1], scale)

    def _apply_material_gas_consume(
        self,
        world: "WorldEngine",
        rule: object,
        x: int,
        y: int,
        gy: int,
        gx: int,
        species_id: int | None,
        scale: float,
    ) -> None:
        policy = self._consume_policy(rule)
        if policy in {"lhs", "both"}:
            self._consume_material_cell(world, x, y, scale)
        if policy in {"rhs", "both"} and species_id is not None:
            self._consume_gas_species(world, species_id, gy, gx, scale)

    def _apply_material_light_consume(
        self,
        world: "WorldEngine",
        rule: object,
        x: int,
        y: int,
        dose_channel: int,
        scale: float,
    ) -> None:
        policy = self._consume_policy(rule)
        if policy in {"lhs", "both"}:
            self._consume_material_cell(world, x, y, scale)
        if policy in {"rhs", "both"}:
            self._consume_light_dose(world, dose_channel, x, y, scale)

    def _apply_gas_gas_consume(
        self,
        world: "WorldEngine",
        rule: object,
        gx: int,
        gy: int,
        lhs_species_id: int | None,
        rhs_species_id: int | None,
        scale: float,
    ) -> None:
        policy = self._consume_policy(rule)
        if policy in {"lhs", "both"} and lhs_species_id is not None:
            self._consume_gas_species(world, lhs_species_id, gy, gx, scale)
        if policy in {"rhs", "both"} and rhs_species_id is not None:
            self._consume_gas_species(world, rhs_species_id, gy, gx, scale)

    def _apply_gas_light_consume(
        self,
        world: "WorldEngine",
        rule: object,
        gx: int,
        gy: int,
        species_id: int | None,
        scale: float,
    ) -> None:
        policy = self._consume_policy(rule)
        if policy in {"rhs", "both"} and species_id is not None:
            self._consume_gas_species(world, species_id, gy, gx, scale)

    def _consume_material_cell(self, world: "WorldEngine", x: int, y: int, amount: float) -> None:
        if amount <= 0.0 or not world.in_bounds(x, y):
            return
        material_id = int(world.material_id[y, x])
        if material_id <= 0:
            return
        world.integrity[y, x] -= float(amount)
        if world.integrity[y, x] <= 0.0:
            world.clear_cell(x, y)

    def _consume_gas_species(self, world: "WorldEngine", species_id: int, gy: int, gx: int, amount: float) -> None:
        if amount <= 0.0 or species_id < 0:
            return
        world.gas_concentration[species_id, gy, gx] = max(
            0.0,
            float(world.gas_concentration[species_id, gy, gx]) - float(amount),
        )

    def _consume_light_dose(self, world: "WorldEngine", dose_channel: int, x: int, y: int, amount: float) -> None:
        if amount <= 0.0:
            return
        world.cell_optical_dose[dose_channel, y, x] = max(
            0.0,
            float(world.cell_optical_dose[dose_channel, y, x]) - float(amount),
        )
        gy, gx = world.cell_to_gas(y, x)
        world.gas_optical_dose[dose_channel, gy, gx] = max(
            0.0,
            float(world.gas_optical_dose[dose_channel, gy, gx]) - float(amount) * 0.08,
        )

    @staticmethod
    def _mask_matches(value: int, required_mask: int) -> bool:
        if required_mask == 0:
            return True
        return (int(value) & int(required_mask)) == int(required_mask)

    def _trigger_material_slot(self, world: "WorldEngine", x: int, y: int, slot_index: int, *, scale: float = 1.0) -> None:
        material_id = int(world.material_id[y, x])
        if material_id <= 0:
            return
        action_index = self._material_reaction_slot(world, material_id, slot_index)
        if action_index <= 0:
            return
        action = self._action_row(world, action_index)
        if action is None:
            return
        if slot_index < 4:
            if world.timer_pack[y, x, slot_index] == 0 and int(action["duration"]) > 0:
                world.timer_pack[y, x, slot_index] = int(action["duration"])
            self._execute_action(world, action_index, x, y, scale)
        else:
            self._execute_action(world, action_index, x, y, scale)

    def _execute_action(self, world: "WorldEngine", action_index: int, x: int, y: int, scale: float) -> None:
        action = self._action_row(world, action_index)
        if action is None:
            return
        reaction_type_id = int(action["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            return
        self.last_executed_action_count += 1
        if self._current_stage is not None:
            self.last_stage_action_counts[self._current_stage] += 1
        world.cell_flags[y, x] |= int(CellFlag.REACTION_LATCHED)
        if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
            self.last_emit_material_action_count += 1
            emit_material_id = int(action["emit_material_id"])
            tx, ty, emitted_velocity = self._material_emit_target_and_velocity(
                world,
                emit_material_id,
                int(action["direction_id"]),
                np.asarray(action["velocity"], dtype=np.float32),
                float(action["speed"]),
                x,
                y,
            )
            if emit_material_id > 0 and world.in_bounds(tx, ty) and world.material_id[ty, tx] == 0:
                world.set_cell_by_id(tx, ty, emit_material_id)
                world.velocity[ty, tx] = emitted_velocity
                self._stage_extra_changed_cell_mask[ty, tx] = True
                self.last_emitted_material_count += 1
                self.last_emitted_material_mask[ty, tx] = True
        elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value):
            self.last_emit_light_action_count += 1
            light_id = int(action["light_type_id"])
            light_meta = self._light_emit_metadata(world, light_id)
            if light_meta is None:
                return
            light_name, default_range = light_meta
            range_cells = int(action["range_cells"])
            if range_cells <= 0:
                range_cells = int(default_range)
            world.emitters.append(
                {
                    "light_type": light_name,
                    "origin": (x, y),
                    "direction": self._direction_vector_id(int(action["direction_id"]), x, y, world),
                    "spread": max(0.0, float(action["beam_width"])),
                    "strength": max(0.1, float(action["strength"]) * scale),
                    "range_cells": range_cells,
                }
            )
            self.last_emitted_light_count += 1
            self.last_emitted_light_mask[y, x] = True
        elif reaction_type_id == int(ReactionType.MODIFY_GAS.value):
            self.last_modify_gas_action_count += 1
            gy, gx = world.cell_to_gas(y, x)
            species_id = int(action["gas_species_id"])
            if 0 <= species_id < world.gas_concentration.shape[0]:
                world.gas_concentration[species_id, gy, gx] = max(
                    0.0,
                    world.gas_concentration[species_id, gy, gx] + float(action["speed"]) * 0.1 * scale,
                )
            self._emit_modify_gas_flow_sources(world, action, x, y, scale)
        elif reaction_type_id == int(ReactionType.CONVERT_MATERIAL.value):
            self.last_convert_material_action_count += 1
            harm_scale = float(scale)
            if not (int(action["flags"]) & REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE):
                harm_scale = max(1.0, harm_scale)
            world.integrity[y, x] -= float(action["harm_per_frame"]) * harm_scale
            if world.integrity[y, x] <= float(action["integrity_threshold"]):
                material_id = int(world.material_id[y, x])
                if int(action["flags"]) & REACTION_ACTION_FLAG_RANDOM_TARGET:
                    target_material_id = self._select_random_convert_material(world, material_id, x, y)
                    if target_material_id > 0:
                        world.set_cell_by_id(x, y, target_material_id)
                    else:
                        world.clear_cell(x, y)
                else:
                    target_material_id = int(action["target_material_id"])
                    if target_material_id > 0:
                        world.set_cell_by_id(x, y, target_material_id)
                    else:
                        world.clear_cell(x, y)
        elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
            self.last_modify_temperature_action_count += 1
            world.cell_temperature[y, x] += float(action["delta"]) * scale
        elif reaction_type_id == int(ReactionType.HARM.value):
            self.last_harm_action_count += 1
            material_id = int(world.material_id[y, x])
            base_integrity = self._material_base_integrity(world, material_id)
            if base_integrity is not None:
                next_integrity = float(world.integrity[y, x]) - float(action["value"]) * scale
                if float(action["value"]) < 0.0:
                    next_integrity = min(base_integrity, next_integrity)
                world.integrity[y, x] = next_integrity
                if world.integrity[y, x] <= 0.0:
                    world.clear_cell(x, y)

    def _execute_gas_action(self, world: "WorldEngine", action_index: int, gx: int, gy: int, scale: float) -> None:
        action = self._action_row(world, action_index)
        if action is None:
            return
        reaction_type_id = int(action["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            return
        self.last_executed_action_count += 1
        if self._current_stage is not None:
            self.last_stage_action_counts[self._current_stage] += 1
        if reaction_type_id == int(ReactionType.MODIFY_GAS.value):
            self.last_modify_gas_action_count += 1
            species_id = int(action["gas_species_id"])
            if 0 <= species_id < world.gas_concentration.shape[0]:
                world.gas_concentration[species_id, gy, gx] = max(
                    0.0,
                    world.gas_concentration[species_id, gy, gx] + float(action["speed"]) * 0.1 * scale,
                )
            cell_x, cell_y = self._gas_cell_center(world, gx, gy)
            self._emit_modify_gas_flow_sources(world, action, cell_x, cell_y, scale)
            return
        if reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
            self.last_modify_temperature_action_count += 1
            world.ambient_temperature[gy, gx] += float(action["delta"]) * scale
            return
        cell_x, cell_y = self._gas_cell_center(world, gx, gy)
        if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
            self.last_emit_material_action_count += 1
            emit_material_id = int(action["emit_material_id"])
            tx, ty, emitted_velocity = self._material_emit_target_and_velocity(
                world,
                emit_material_id,
                int(action["direction_id"]),
                np.asarray(action["velocity"], dtype=np.float32),
                float(action["speed"]),
                cell_x,
                cell_y,
            )
            if emit_material_id > 0 and world.in_bounds(tx, ty) and world.material_id[ty, tx] == 0:
                world.set_cell_by_id(tx, ty, emit_material_id)
                world.velocity[ty, tx] = emitted_velocity
                self._stage_extra_changed_cell_mask[ty, tx] = True
                self.last_emitted_material_count += 1
                self.last_emitted_material_mask[ty, tx] = True
        elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value):
            self.last_emit_light_action_count += 1
            light_id = int(action["light_type_id"])
            light_meta = self._light_emit_metadata(world, light_id)
            if light_meta is None:
                return
            light_name, default_range = light_meta
            range_cells = int(action["range_cells"])
            if range_cells <= 0:
                range_cells = int(default_range)
            world.emitters.append(
                {
                    "light_type": light_name,
                    "origin": (cell_x, cell_y),
                    "direction": self._gas_direction_vector_id(world, int(action["direction_id"]), gx, gy),
                    "spread": max(0.0, float(action["beam_width"])),
                    "strength": max(0.1, float(action["strength"]) * scale),
                    "range_cells": range_cells,
                }
            )
            self.last_emitted_light_count += 1
            self.last_emitted_light_mask[cell_y, cell_x] = True

    def _action_row(self, world: "WorldEngine", action_index: int) -> np.void | None:
        action_table = world.bridge.shadow_typed_tables["reaction_action_table"]
        if action_index < 0 or action_index >= action_table.shape[0]:
            return None
        return action_table[action_index]

    def _apply_trigger_grid(self, world: "WorldEngine", trigger_grid: np.ndarray) -> None:
        ys, xs = np.nonzero(np.any(trigger_grid > 0, axis=-1))
        used_cpu = False
        for y, x in zip(ys.tolist(), xs.tolist()):
            for local_slot, action_index in enumerate(trigger_grid[y, x].tolist()):
                if int(action_index) <= 0:
                    continue
                world._require_gpu_stage("reaction trigger action execution")
                used_cpu = True
                self._execute_action(world, int(action_index), int(x), int(y), 1.0)
        if used_cpu:
            self._note_runtime_backend("cpu")

    def _apply_deferred_batch(self, world: "WorldEngine", batch: "GPUDeferredActionBatch") -> None:
        self._record_gpu_local_action_counts(batch.gpu_local_action_counts)
        if self._formal_gpu_frame(world):
            if getattr(batch, "formal_gpu_empty", False):
                return
            has_cpu_deferred_payload = bool(
                np.any(batch.action_lo > 0)
                or np.any(batch.action_hi > 0)
                or batch.emitted_lights.size > 0
                or np.any(batch.emitted_material_mask)
            )
            if has_cpu_deferred_payload:
                raise RuntimeError("GPU reaction formal frame returned CPU deferred actions; CPU fallback is disabled")
            return
        self._append_gpu_emitted_lights(world, batch.emitted_lights)
        self._record_gpu_emitted_materials(world, batch.emitted_material_mask)
        ys, xs = np.nonzero(
            np.any(batch.action_lo > 0, axis=-1) | np.any(batch.action_hi > 0, axis=-1)
        )
        used_cpu = False
        for y, x in zip(ys.tolist(), xs.tolist()):
            action_indices = batch.action_lo[y, x].tolist() + batch.action_hi[y, x].tolist()
            scales = batch.scale_lo[y, x].tolist() + batch.scale_hi[y, x].tolist()
            for action_index, scale in zip(action_indices, scales):
                if int(action_index) <= 0:
                    continue
                if self._deferred_action_handled_by_gpu(world, int(action_index)):
                    self._record_gpu_deferred_action(world, int(action_index))
                    continue
                world._require_gpu_stage("deferred reaction action execution")
                used_cpu = True
                self._execute_action(world, int(action_index), int(x), int(y), float(scale))
        if used_cpu:
            self._note_runtime_backend("cpu")

    def _record_gpu_local_action_counts(self, counts: np.ndarray) -> None:
        if counts.size == 0:
            return
        values = np.asarray(counts, dtype=np.uint32).reshape(-1)
        if values.size < 8:
            return
        total = int(values[0])
        if total <= 0:
            return
        self.last_executed_action_count += total
        if self._current_stage is not None:
            self.last_stage_action_counts[self._current_stage] += total
        self.last_harm_action_count += int(values[TYPE_HARM])
        self.last_modify_temperature_action_count += int(values[TYPE_MODIFY_TEMPERATURE])
        self.last_convert_material_action_count += int(values[TYPE_CONVERT_MATERIAL])
        self.last_modify_gas_action_count += int(values[TYPE_MODIFY_GAS])
        self.last_emit_light_action_count += int(values[TYPE_EMIT_LIGHT])
        self.last_emit_material_action_count += int(values[TYPE_EMIT_MATERIAL])

    def _append_gpu_emitted_lights(self, world: "WorldEngine", emitted_lights: np.ndarray) -> None:
        if emitted_lights.size == 0:
            return
        for record in np.asarray(emitted_lights, dtype=np.float32):
            x = int(round(float(record[0])))
            y = int(round(float(record[1])))
            if not world.in_bounds(x, y):
                continue
            light_id = int(round(float(record[7])))
            light_meta = self._light_emit_metadata(world, light_id)
            if light_meta is None:
                continue
            light_name, default_range = light_meta
            range_cells = int(round(float(record[5])))
            if range_cells <= 0:
                range_cells = int(default_range)
            world.emitters.append(
                {
                    "light_type": light_name,
                    "origin": (x, y),
                    "direction": (float(record[2]), float(record[3])),
                    "spread": max(0.0, float(record[6])),
                    "strength": max(0.1, float(record[4])),
                    "range_cells": range_cells,
                }
            )
            self.last_emitted_light_count += 1
            self.last_emitted_light_mask[y, x] = True

    def _record_gpu_emitted_materials(self, world: "WorldEngine", emitted_material_mask: np.ndarray) -> None:
        if emitted_material_mask.size == 0:
            return
        mask = np.asarray(emitted_material_mask, dtype=np.bool_)
        if mask.shape != self.last_emitted_material_mask.shape or not np.any(mask):
            return
        self.last_emitted_material_mask |= mask
        self.last_emitted_material_count += int(np.count_nonzero(mask))
        self._stage_extra_changed_cell_mask |= mask

    def _record_gpu_deferred_action(self, world: "WorldEngine", action_index: int) -> None:
        action = self._action_row(world, action_index)
        if action is None:
            return
        reaction_type_id = int(action["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            return
        self.last_executed_action_count += 1
        if self._current_stage is not None:
            self.last_stage_action_counts[self._current_stage] += 1
        if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
            self.last_emit_material_action_count += 1
        elif reaction_type_id == int(ReactionType.MODIFY_GAS.value):
            self.last_modify_gas_action_count += 1

    def _deferred_action_handled_by_gpu(self, world: "WorldEngine", action_index: int) -> bool:
        action = self._action_row(world, action_index)
        if action is None:
            return False
        if not world._gpu_pipeline_available(self.gpu_pipeline, "reactions"):
            return False
        reaction_type_id = int(action["reaction_type_id"])
        if reaction_type_id == int(ReactionType.MODIFY_GAS.value):
            return int(action["gas_species_id"]) >= 0
        if reaction_type_id == int(ReactionType.EMIT_MATERIAL.value):
            return int(action["emit_material_id"]) > 0
        return False

    def _neighbor_for_direction(self, world: "WorldEngine", direction: Direction, x: int, y: int) -> tuple[int, int]:
        if direction == Direction.DOWN:
            return x, y + 1
        if direction == Direction.UP:
            return x, y - 1
        if direction == Direction.LEFT:
            return x - 1, y
        if direction == Direction.RIGHT:
            return x + 1, y
        if direction == Direction.RANDOM:
            return self._deterministic_random_neighbor(x, y)
        if direction == Direction.SPEED:
            vx, vy = world.velocity[y, x]
            return x + int(np.sign(vx)), y + int(np.sign(vy))
        return x, y

    def _neighbor_for_direction_id(self, world: "WorldEngine", direction_id: int, x: int, y: int) -> tuple[int, int]:
        if direction_id == int(DIRECTION_IDS["down"]):
            return x, y + 1
        if direction_id == int(DIRECTION_IDS["up"]):
            return x, y - 1
        if direction_id == int(DIRECTION_IDS["left"]):
            return x - 1, y
        if direction_id == int(DIRECTION_IDS["right"]):
            return x + 1, y
        if direction_id == int(DIRECTION_IDS["random"]):
            return self._deterministic_random_neighbor(x, y)
        if direction_id == int(DIRECTION_IDS["speed"]):
            vx, vy = world.velocity[y, x]
            return x + int(np.sign(vx)), y + int(np.sign(vy))
        return x, y

    def _material_emit_target_and_velocity(
        self,
        world: "WorldEngine",
        emit_material_id: int,
        direction_id: int,
        explicit_velocity: np.ndarray,
        speed: float,
        x: int,
        y: int,
    ) -> tuple[int, int, np.ndarray]:
        tx, ty = self._neighbor_for_direction_id(world, direction_id, x, y)
        emitted_phase = self._material_default_phase(world, emit_material_id)
        if emitted_phase is None:
            return tx, ty, np.zeros((2,), dtype=np.float32)
        if emitted_phase != int(Phase.POWDER):
            return tx, ty, np.zeros((2,), dtype=np.float32)
        velocity = np.asarray(explicit_velocity, dtype=np.float32)
        if float(np.hypot(float(velocity[0]), float(velocity[1]))) > 1e-5:
            return tx, ty, velocity.astype(np.float32, copy=True)
        dx = tx - x
        dy = ty - y
        norm = max(1e-5, float(np.hypot(dx, dy)))
        magnitude = max(0.0, float(speed))
        return tx, ty, np.asarray((dx / norm * magnitude, dy / norm * magnitude), dtype=np.float32)

    def _select_random_convert_material(self, world: "WorldEngine", current_material_id: int, x: int, y: int) -> int:
        candidate_ids = self._random_convert_candidates(world)
        if not candidate_ids:
            return 0
        selector = self._deterministic_selector(x, y, len(candidate_ids))
        candidate = candidate_ids[selector]
        if candidate > 0 and candidate != current_material_id:
            return candidate
        for offset in range(1, len(candidate_ids)):
            material_id = candidate_ids[(selector + offset) % len(candidate_ids)]
            if material_id > 0 and material_id != current_material_id:
                return material_id
        return 0

    @staticmethod
    def _deterministic_selector(x: int, y: int, count: int) -> int:
        if count <= 0:
            return 0
        return abs((int(x) * 73856093) ^ (int(y) * 19349663)) % int(count)

    def _deterministic_random_neighbor(self, x: int, y: int) -> tuple[int, int]:
        selector = self._deterministic_selector(x, y, 9)
        dx = selector % 3 - 1
        dy = selector // 3 - 1
        return x + dx, y + dy

    def _neighbor_for_gas_direction(self, world: "WorldEngine", direction: Direction, gx: int, gy: int) -> tuple[int, int]:
        cell_x, cell_y = self._gas_cell_center(world, gx, gy)
        if direction == Direction.SPEED:
            vx, vy = world.flow_velocity[gy, gx]
            return cell_x + int(np.sign(vx)), cell_y + int(np.sign(vy))
        return self._neighbor_for_direction(world, direction, cell_x, cell_y)

    def _neighbor_for_gas_direction_id(self, world: "WorldEngine", direction_id: int, gx: int, gy: int) -> tuple[int, int]:
        cell_x, cell_y = self._gas_cell_center(world, gx, gy)
        if direction_id == int(DIRECTION_IDS["speed"]):
            vx, vy = world.flow_velocity[gy, gx]
            return cell_x + int(np.sign(vx)), cell_y + int(np.sign(vy))
        return self._neighbor_for_direction_id(world, direction_id, cell_x, cell_y)

    def _direction_vector(self, direction: Direction, x: int, y: int, world: "WorldEngine") -> tuple[float, float]:
        if direction == Direction.UP:
            return (0.0, -1.0)
        if direction == Direction.DOWN:
            return (0.0, 1.0)
        if direction == Direction.LEFT:
            return (-1.0, 0.0)
        if direction == Direction.RIGHT:
            return (1.0, 0.0)
        if direction == Direction.SPEED:
            vx, vy = world.velocity[y, x]
            norm = max(1e-5, float(np.hypot(vx, vy)))
            return (float(vx / norm), float(vy / norm))
        return (0.0, 0.0)

    def _direction_vector_id(self, direction_id: int, x: int, y: int, world: "WorldEngine") -> tuple[float, float]:
        if direction_id == int(DIRECTION_IDS["up"]):
            return (0.0, -1.0)
        if direction_id == int(DIRECTION_IDS["down"]):
            return (0.0, 1.0)
        if direction_id == int(DIRECTION_IDS["left"]):
            return (-1.0, 0.0)
        if direction_id == int(DIRECTION_IDS["right"]):
            return (1.0, 0.0)
        if direction_id == int(DIRECTION_IDS["speed"]):
            vx, vy = world.velocity[y, x]
            norm = max(1e-5, float(np.hypot(vx, vy)))
            return (float(vx / norm), float(vy / norm))
        return (0.0, 0.0)

    def _emit_modify_gas_flow_sources(
        self,
        world: "WorldEngine",
        action: np.void,
        x: int,
        y: int,
        scale: float,
    ) -> None:
        strength = float(action["strength"]) * max(0.0, scale)
        radius = float(action["range_cells"])
        if strength <= 0.0 or radius <= 0.0:
            return
        flow_sources: list[ForceSource] = []
        velocity = np.asarray(action["velocity"], dtype=np.float32)
        velocity_norm = float(np.hypot(velocity[0], velocity[1]))
        direction_id = int(action["direction_id"])
        if velocity_norm > 1e-5:
            direction = (float(velocity[0] / velocity_norm), float(velocity[1] / velocity_norm))
            flow_sources.append(
                ForceSource(
                    x=float(x),
                    y=float(y),
                    direction=direction,
                    radius=radius,
                    strength=strength,
                    lifetime=REACTION_FLOW_SOURCE_LIFETIME,
                )
            )
        elif direction_id != int(DIRECTION_IDS["all"]):
            direction = self._direction_vector_id(direction_id, x, y, world)
            if abs(direction[0]) > 1e-5 or abs(direction[1]) > 1e-5:
                flow_sources.append(
                    ForceSource(
                        x=float(x),
                        y=float(y),
                        direction=direction,
                        radius=radius,
                        strength=strength,
                        lifetime=REACTION_FLOW_SOURCE_LIFETIME,
                    )
                )
        else:
            speed = float(action["speed"])
            if abs(speed) <= 1e-5:
                return
            flow_sign = 1.0 if speed > 0.0 else -1.0
            offset = max(1.0, radius * 0.45)
            for radial_x, radial_y in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
                source_x = min(max(float(x) + radial_x * offset, 0.0), float(world.width - 1))
                source_y = min(max(float(y) + radial_y * offset, 0.0), float(world.height - 1))
                flow_sources.append(
                    ForceSource(
                        x=source_x,
                        y=source_y,
                        direction=(radial_x * flow_sign, radial_y * flow_sign),
                        radius=radius,
                        strength=strength,
                        lifetime=REACTION_FLOW_SOURCE_LIFETIME,
                    )
                )
        if not flow_sources:
            return
        world.force_sources.extend(flow_sources)
        max_radius = int(np.ceil(radius))
        world._mark_active_rect_runtime(
            max(0, x - max_radius),
            max(0, y - max_radius),
            min(world.width, x + max_radius + 1),
            min(world.height, y + max_radius + 1),
        )

    def _gas_direction_vector(self, world: "WorldEngine", direction: Direction, gx: int, gy: int) -> tuple[float, float]:
        if direction == Direction.SPEED:
            vx, vy = world.flow_velocity[gy, gx]
            norm = max(1e-5, float(np.hypot(vx, vy)))
            return (float(vx / norm), float(vy / norm))
        cell_x, cell_y = self._gas_cell_center(world, gx, gy)
        return self._direction_vector(direction, cell_x, cell_y, world)

    def _gas_direction_vector_id(self, world: "WorldEngine", direction_id: int, gx: int, gy: int) -> tuple[float, float]:
        if direction_id == int(DIRECTION_IDS["speed"]):
            vx, vy = world.flow_velocity[gy, gx]
            norm = max(1e-5, float(np.hypot(vx, vy)))
            return (float(vx / norm), float(vy / norm))
        cell_x, cell_y = self._gas_cell_center(world, gx, gy)
        return self._direction_vector_id(direction_id, cell_x, cell_y, world)

    def _gas_cell_center(self, world: "WorldEngine", gx: int, gy: int) -> tuple[int, int]:
        x = min(world.width - 1, gx * world.gas_cell_size + world.gas_cell_size // 2)
        y = min(world.height - 1, gy * world.gas_cell_size + world.gas_cell_size // 2)
        return (x, y)

    def release(self) -> None:
        self.gpu_pipeline.release()
        self.reset_runtime_state()

    def reset_runtime_state(self, world: "WorldEngine" | None = None) -> None:
        tile_shape = (0, 0) if world is None else (world.active.tile_height, world.active.tile_width)
        cell_shape = (0, 0) if world is None else (world.height, world.width)
        gas_shape = (0, 0) if world is None else (world.gas_height, world.gas_width)
        self.last_stage_tile_masks = {
            stage: np.zeros(tile_shape, dtype=np.bool_)
            for stage in REACTION_STAGE_NAMES
        }
        self.last_solve_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_solve_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_changed_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_changed_gas_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_ambient_changed_mask = np.zeros(gas_shape, dtype=np.bool_)
        self.last_timer_changed_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_emitted_light_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_emitted_material_mask = np.zeros(cell_shape, dtype=np.bool_)
        self.last_stage_action_counts = {stage: 0 for stage in REACTION_STAGE_NAMES}
        self.last_executed_action_count = 0
        self.last_emitted_light_count = 0
        self.last_emitted_material_count = 0
        self.last_emit_light_action_count = 0
        self.last_emit_material_action_count = 0
        self.last_modify_gas_action_count = 0
        self.last_convert_material_action_count = 0
        self.last_modify_temperature_action_count = 0
        self.last_harm_action_count = 0
        self.last_runtime_backend = "idle"
        self._stage_extra_changed_cell_mask = np.zeros(cell_shape, dtype=np.bool_)
        self._runtime_used_cpu = False
        self._runtime_used_gpu = False
        self._current_stage = None

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "backend": self.last_runtime_backend,
            "stage_tile_masks": {stage: mask.copy() for stage, mask in self.last_stage_tile_masks.items()},
            "solve_cell_mask": self.last_solve_cell_mask.copy(),
            "solve_gas_mask": self.last_solve_gas_mask.copy(),
            "changed_cell_mask": self.last_changed_cell_mask.copy(),
            "changed_gas_mask": self.last_changed_gas_mask.copy(),
            "ambient_changed_mask": self.last_ambient_changed_mask.copy(),
            "timer_changed_mask": self.last_timer_changed_mask.copy(),
            "emitted_light_mask": self.last_emitted_light_mask.copy(),
            "emitted_material_mask": self.last_emitted_material_mask.copy(),
            "stage_action_counts": dict(self.last_stage_action_counts),
            "executed_action_count": int(self.last_executed_action_count),
            "emitted_light_count": int(self.last_emitted_light_count),
            "emitted_material_count": int(self.last_emitted_material_count),
            "emit_light_action_count": int(self.last_emit_light_action_count),
            "emit_material_action_count": int(self.last_emit_material_action_count),
            "modify_gas_action_count": int(self.last_modify_gas_action_count),
            "convert_material_action_count": int(self.last_convert_material_action_count),
            "modify_temperature_action_count": int(self.last_modify_temperature_action_count),
            "harm_action_count": int(self.last_harm_action_count),
        }
