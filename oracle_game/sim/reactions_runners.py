from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _advance_timed_slots(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="timed",
    )
    solver._record_stage_solve_masks("timed", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "timed"
    deferred = solver.gpu_pipeline.run_timed_actions(
        world,
        solve_cell_mask=solve_cell_mask,
    )
    if deferred is not None:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "timed reaction actions",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("timed reaction actions")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
            action_index = solver._material_reaction_slot(world, material_id, timer_index)
            if action_index > 0:
                trigger_grid[y, x, timer_index] = action_index
            updated_timers[y, x, timer_index] = max(0, timer_value - 1)
    world.timer_pack[:] = updated_timers
    solver._apply_trigger_grid(world, trigger_grid)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_self_rules(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="self",
    )
    solver._record_stage_solve_masks("self", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "self"
    deferred = solver.gpu_pipeline.run_self_actions(
        world,
        solve_cell_mask=solve_cell_mask,
    )
    if deferred is not None:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "self reaction rules",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("self reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
            mask &= solver._phase_mask_matches_values(phase_values, phase_mask)
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
            action_index = solver._material_reaction_slot(world, source_material_id, slot_index)
            if action_index <= 0:
                continue
            if slot_index < 4:
                if int(updated_timers[y, x, slot_index]) > 0:
                    continue
                action = solver._action_row(world, action_index)
                if action is not None and int(action["duration"]) > 0:
                    updated_timers[y, x, slot_index] = int(action["duration"])
                trigger_lo[y, x, slot_index] = action_index
                continue
            trigger_hi[y, x, slot_index - 4] = action_index
    world.timer_pack[:] = updated_timers
    solver._apply_trigger_grid(world, trigger_lo)
    solver._apply_trigger_grid(world, trigger_hi)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_material_material(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    if int(world.bridge.shadow_typed_tables["material_material_rule_table"].shape[0]) <= 0:
        return
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="material_material",
    )
    solver._record_stage_solve_masks("material_material", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "material_material"
    deferred = solver.gpu_pipeline.run_material_material(
        world,
        solve_cell_mask=solve_cell_mask,
    )
    if deferred is not None:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "material-material reaction rules",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("material-material reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
            candidate_mask &= solver._phase_mask_matches_values(phase_values, phase_mask)
        candidate_mask &= temperature_values >= float(rule["min_temperature"])
        candidate_mask &= temperature_values <= float(rule["max_temperature"])
        for index in np.nonzero(candidate_mask)[0].tolist():
            y = int(active_ys[index])
            x = int(active_xs[index])
            material_id = int(material_snapshot[y, x])
            if material_id <= 0:
                continue
            if not solver._mask_matches(
                solver._material_tag_mask(world, material_id, "material_tag_mask"),
                int(rule["lhs_tag_mask"]),
            ):
                continue
            match_xy = solver._matching_material_neighbor(
                world,
                x,
                y,
                required_id=rhs_id,
                required_mask=int(rule["rhs_tag_mask"]),
                material_grid=material_snapshot,
            )
            if match_xy is not None:
                scale = solver._rule_scale(rule, 1.0)
                solver._execute_pair_rule(world, rule, x, y, scale)
                solver._apply_material_material_consume(world, rule, x, y, match_xy, scale)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_material_gas(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    if int(world.bridge.shadow_typed_tables["material_gas_rule_table"].shape[0]) <= 0:
        return
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="material_gas",
    )
    solver._record_stage_solve_masks("material_gas", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "material_gas"
    deferred = solver.gpu_pipeline.run_material_gas(
        world,
        solve_cell_mask=solve_cell_mask,
    )
    if deferred is not None:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "material-gas reaction rules",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("material-gas reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
            candidate_mask &= solver._phase_mask_matches_values(phase_values, phase_mask)
        candidate_mask &= temperature_values >= float(rule["min_temperature"])
        candidate_mask &= temperature_values <= float(rule["max_temperature"])
        for index in np.nonzero(candidate_mask)[0].tolist():
            y = int(active_ys[index])
            x = int(active_xs[index])
            material_id = int(world.material_id[y, x])
            if material_id <= 0:
                continue
            if not solver._mask_matches(
                solver._material_tag_mask(world, material_id, "gas_tag_mask"),
                int(rule["lhs_tag_mask"]),
            ):
                continue
            gy, gx = world.cell_to_gas(y, x)
            species_id, concentration = solver._best_matching_material_reaction_gas_species(
                world,
                gy,
                gx,
                gas_id=gas_id,
                required_mask=int(rule["rhs_tag_mask"]),
                gas_concentration=gas_snapshot,
            )
            if concentration >= float(rule["threshold"]):
                scale = solver._rule_scale(rule, concentration)
                solver._execute_pair_rule(world, rule, x, y, scale)
                solver._apply_material_gas_consume(world, rule, x, y, gy, gx, species_id, scale)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_material_light(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    if int(world.bridge.shadow_typed_tables["material_light_rule_table"].shape[0]) <= 0:
        return
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="material_light",
    )
    solver._record_stage_solve_masks("material_light", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "material_light"
    deferred = solver.gpu_pipeline.run_material_light(
        world,
        solve_cell_mask=solve_cell_mask,
    )
    if deferred is not None:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "material-light reaction rules",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("material-light reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
    active_ys, active_xs = np.nonzero(solve_cell_mask)
    for rule in world.bridge.shadow_typed_tables["material_light_rule_table"]:
        lhs_id_raw = int(rule["lhs_material_id"])
        light_id_raw = int(rule["rhs_light_id"])
        lhs_id = lhs_id_raw if lhs_id_raw > 0 else None
        light_id = light_id_raw if light_id_raw >= 0 else None
        if light_id is None:
            continue
        dose_channel = solver._light_dose_channel(world, light_id)
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
                [solver._material_tag_mask(world, int(material_id), "light_tag_mask") for material_id in material_values],
                dtype=np.uint32,
            )
            mask &= (light_masks & required_mask) == required_mask
        phase_mask = int(rule["phase_mask"])
        if phase_mask != 0:
            mask &= solver._phase_mask_matches_values(phase_values, phase_mask)
        mask &= temperature_values >= float(rule["min_temperature"])
        mask &= temperature_values <= float(rule["max_temperature"])
        for index in np.nonzero(mask)[0].tolist():
            y = int(active_ys[index])
            x = int(active_xs[index])
            scale = solver._rule_scale(rule, float(dose_values[index]))
            solver._execute_pair_rule(world, rule, x, y, scale)
            solver._apply_material_light_consume(world, rule, x, y, dose_channel, scale)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_gas_gas(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    if int(world.bridge.shadow_typed_tables["gas_gas_rule_table"].shape[0]) <= 0:
        return
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(world, seed_timer_cells=True)
    solver._record_stage_solve_masks("gas_gas", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not np.any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "gas_gas"
    deferred = solver.gpu_pipeline.run_gas_gas(
        world,
        solve_gas_mask=solve_gas_mask,
    )
    if deferred is not None and deferred is not False:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        if deferred is not True:
            solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    world._require_gpu_stage("gas-gas reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
        lhs_species = solver._matching_material_reaction_gas_species_ids(
            world,
            gas_id=lhs_id,
            required_mask=lhs_tag_mask,
        )
        rhs_species = solver._matching_material_reaction_gas_species_ids(
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
            lhs_species_id, _ = solver._best_matching_material_reaction_gas_species(
                world,
                gy,
                gx,
                gas_id=lhs_id,
                required_mask=lhs_tag_mask,
            )
            rhs_species_id, _ = solver._best_matching_material_reaction_gas_species(
                world,
                gy,
                gx,
                gas_id=rhs_id,
                required_mask=rhs_tag_mask,
            )
            scale = solver._rule_scale(rule, min(float(lhs_values[index]), float(rhs_values[index])))
            solver._execute_gas_action(world, int(rule["result_action"]), gx, gy, scale)
            solver._apply_gas_gas_consume(world, rule, gx, gy, lhs_species_id, rhs_species_id, scale)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)


def _run_gas_light(solver, world: "WorldEngine") -> None:
    solver._ensure_runtime_state(world)
    if int(world.bridge.shadow_typed_tables["gas_light_rule_table"].shape[0]) <= 0:
        return
    solve_tile_mask, solve_cell_mask, solve_gas_mask = solver._solve_masks(
        world,
        seed_timer_cells=True,
        stage="gas_light",
    )
    solver._record_stage_solve_masks("gas_light", solve_tile_mask, solve_cell_mask, solve_gas_mask)
    if not solver._solve_mask_any(solve_tile_mask):
        return
    previous_state = solver._capture_activity_state(world, solve_cell_mask, solve_gas_mask)
    solver._current_stage = "gas_light"
    deferred = solver.gpu_pipeline.run_gas_light(
        world,
        solve_gas_mask=solve_gas_mask,
    )
    if deferred is not None and deferred is not False:
        solver.last_backend = "gpu"
        solver._note_runtime_backend("gpu")
        if deferred is not True:
            solver._apply_deferred_batch(world, deferred)
        solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
        return
    solver._require_materialized_cpu_solve_masks(
        world,
        "gas-light reaction rules",
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
    )
    world._require_gpu_stage("gas-light reaction rules")
    solver.last_backend = "cpu"
    solver._note_runtime_backend("cpu")
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
        dose_channel = solver._light_dose_channel(world, light_id)
        if dose_channel is None:
            continue
        dose_values = world.gas_optical_dose[dose_channel, active_gy, active_gx]
        ambient_values = world.ambient_temperature[active_gy, active_gx]
        candidate_species = solver._matching_light_gas_species_ids(world, gas_id=gas_id, required_mask=rhs_tag_mask)
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
            species_id, _ = solver._best_matching_light_reaction_gas_species(
                world,
                gy,
                gx,
                gas_id=gas_id,
                required_mask=rhs_tag_mask,
            )
            scale = solver._rule_scale(rule, min(float(best_gas_values[index]), float(dose_values[index])))
            solver._execute_gas_action(world, int(rule["result_action"]), gx, gy, scale)
            solver._apply_gas_light_consume(world, rule, gx, gy, species_id, scale)
    solver._finalize_stage_runtime(world, solve_tile_mask, solve_cell_mask, solve_gas_mask, previous_state)
