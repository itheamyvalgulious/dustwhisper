from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.types import PageStripeUpdate
from oracle_game.gpu.dtypes import (
    GAS_RUNTIME_META_DTYPE,
    GAS_SPECIES_RUNTIME_DTYPE,
    HEAT_RUNTIME_META_DTYPE,
    LIQUID_RUNTIME_META_DTYPE,
    REACTION_RUNTIME_META_DTYPE,
    COLLAPSE_RUNTIME_META_DTYPE,
    COLLAPSE_COMPONENT_DTYPE,
    OPTICS_RUNTIME_META_DTYPE,
    PAGE_STRIPE_META_DTYPE,
    PAGE_STRIPE_SECTION_DTYPE,
)


def pack_gas_runtime_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = world.gas_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    if solve_tile_mask.shape != (world.active.tile_height, world.active.tile_width):
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    if solve_gas_mask.shape != (world.gas_height, world.gas_width):
        solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    pressure_range = np.asarray(snapshot["pressure_range"], dtype=np.float32)
    if pressure_range.shape != (2,):
        pressure_range = np.zeros((2,), dtype=np.float32)
    ambient_range = np.asarray(snapshot["ambient_range"], dtype=np.float32)
    if ambient_range.shape != (2,):
        ambient_range = np.zeros((2,), dtype=np.float32)
    flow_speed_range = np.asarray(snapshot["flow_speed_range"], dtype=np.float32)
    if flow_speed_range.shape != (2,):
        flow_speed_range = np.zeros((2,), dtype=np.float32)
    species_total = np.asarray(snapshot["species_total_concentration"], dtype=np.float32)
    species_active = np.asarray(snapshot["species_active_concentration"], dtype=np.float32)
    species_count = int(world.gas_concentration.shape[0])
    if species_total.shape != (species_count,):
        species_total = np.zeros((species_count,), dtype=np.float32)
    if species_active.shape != (species_count,):
        species_active = np.zeros((species_count,), dtype=np.float32)

    meta = np.zeros((1,), dtype=GAS_RUNTIME_META_DTYPE)
    meta[0]["backend_id"] = 2 if world.gas_solver.last_backend == "gpu" else 1
    meta[0]["pressure_iterations"] = int(snapshot["pressure_iterations"])
    meta[0]["force_source_count_before"] = int(snapshot["force_source_count_before"])
    meta[0]["force_source_count_after"] = int(snapshot["force_source_count_after"])
    meta[0]["solve_tile_count"] = int(np.count_nonzero(solve_tile_mask))
    meta[0]["solve_gas_count"] = int(np.count_nonzero(solve_gas_mask))
    meta[0]["velocity_changed"] = int(bool(snapshot["velocity_changed"]))
    meta[0]["ambient_changed"] = int(bool(snapshot["ambient_changed"]))
    meta[0]["gas_changed"] = int(bool(snapshot["gas_changed"]))
    meta[0]["pressure_min"] = float(pressure_range[0])
    meta[0]["pressure_max"] = float(pressure_range[1])
    meta[0]["ambient_min"] = float(ambient_range[0])
    meta[0]["ambient_max"] = float(ambient_range[1])
    meta[0]["flow_speed_min"] = float(flow_speed_range[0])
    meta[0]["flow_speed_max"] = float(flow_speed_range[1])

    species_runtime = np.zeros((species_count,), dtype=GAS_SPECIES_RUNTIME_DTYPE)
    if species_count > 0:
        species_runtime["species_id"] = np.arange(species_count, dtype=np.int32)
        species_runtime["total_concentration"] = species_total
        species_runtime["active_concentration"] = species_active
    return meta, solve_tile_mask, solve_gas_mask, species_runtime


def pack_heat_runtime_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = world.heat_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    if solve_tile_mask.shape != (world.active.tile_height, world.active.tile_width):
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    if solve_cell_mask.shape != (world.height, world.width):
        solve_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    if solve_gas_mask.shape != (world.gas_height, world.gas_width):
        solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    phase_targets = np.asarray(snapshot["phase_targets"], dtype=np.int32)
    if phase_targets.shape != (world.height, world.width):
        phase_targets = np.zeros((world.height, world.width), dtype=np.int32)
    boil_targets = np.asarray(snapshot["boil_targets"], dtype=np.int32)
    if boil_targets.shape != (world.height, world.width):
        boil_targets = np.zeros((world.height, world.width), dtype=np.int32)
    condense_targets = np.asarray(snapshot["condense_targets"], dtype=np.uint8)
    if condense_targets.shape != world.gas_concentration.shape:
        condense_targets = np.zeros(world.gas_concentration.shape, dtype=np.uint8)
    cell_temperature_range = np.asarray(snapshot["cell_temperature_range"], dtype=np.float32)
    if cell_temperature_range.shape != (2,):
        cell_temperature_range = np.zeros((2,), dtype=np.float32)
    ambient_temperature_range = np.asarray(snapshot["ambient_temperature_range"], dtype=np.float32)
    if ambient_temperature_range.shape != (2,):
        ambient_temperature_range = np.zeros((2,), dtype=np.float32)
    integrity_range = np.asarray(snapshot["integrity_range"], dtype=np.float32)
    if integrity_range.shape != (2,):
        integrity_range = np.zeros((2,), dtype=np.float32)

    meta = np.zeros((1,), dtype=HEAT_RUNTIME_META_DTYPE)
    meta[0]["backend_id"] = 2 if world.heat_solver.last_backend == "gpu" else 1
    meta[0]["ambient_iterations"] = int(snapshot["ambient_iterations"])
    meta[0]["solve_tile_count"] = int(np.count_nonzero(solve_tile_mask))
    meta[0]["solve_cell_count"] = int(np.count_nonzero(solve_cell_mask))
    meta[0]["solve_gas_count"] = int(np.count_nonzero(solve_gas_mask))
    meta[0]["phase_target_count"] = int(np.count_nonzero(phase_targets > 0))
    meta[0]["boil_target_count"] = int(np.count_nonzero(boil_targets > 0))
    meta[0]["condense_target_count"] = int(np.count_nonzero(condense_targets > 0))
    meta[0]["cell_changed"] = int(bool(snapshot["cell_changed"]))
    meta[0]["ambient_changed"] = int(bool(snapshot["ambient_changed"]))
    meta[0]["material_changed"] = int(bool(snapshot["material_changed"]))
    meta[0]["phase_changed"] = int(bool(snapshot["phase_changed"]))
    meta[0]["integrity_changed"] = int(bool(snapshot["integrity_changed"]))
    meta[0]["gas_changed"] = int(bool(snapshot["gas_changed"]))
    meta[0]["cell_temperature_min"] = float(cell_temperature_range[0])
    meta[0]["cell_temperature_max"] = float(cell_temperature_range[1])
    meta[0]["ambient_temperature_min"] = float(ambient_temperature_range[0])
    meta[0]["ambient_temperature_max"] = float(ambient_temperature_range[1])
    meta[0]["integrity_min"] = float(integrity_range[0])
    meta[0]["integrity_max"] = float(integrity_range[1])
    return meta, solve_tile_mask, solve_cell_mask, solve_gas_mask, phase_targets, boil_targets, condense_targets


def pack_liquid_runtime_upload(
    world: "WorldEngine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = world.liquid_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    if solve_tile_mask.shape != (world.active.tile_height, world.active.tile_width):
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    post_tile_mask = np.asarray(snapshot["post_tile_mask"], dtype=np.uint8)
    if post_tile_mask.shape != (world.active.tile_height, world.active.tile_width):
        post_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    post_cell_mask = np.asarray(snapshot["post_cell_mask"], dtype=np.uint8)
    if post_cell_mask.shape != (world.height, world.width):
        post_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    vertical_seam_mask = np.asarray(snapshot["vertical_seam_mask"], dtype=np.uint8)
    if vertical_seam_mask.shape != (world.height, world.width):
        vertical_seam_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    horizontal_seam_mask = np.asarray(snapshot["horizontal_seam_mask"], dtype=np.uint8)
    if horizontal_seam_mask.shape != (world.height, world.width):
        horizontal_seam_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    buoyancy_mask = np.asarray(snapshot["buoyancy_mask"], dtype=np.uint8)
    if buoyancy_mask.shape != (world.height, world.width):
        buoyancy_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
    if changed_cell_mask.shape != (world.height, world.width):
        changed_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)

    meta = np.zeros((1,), dtype=LIQUID_RUNTIME_META_DTYPE)
    meta[0]["backend_id"] = 2 if world.liquid_solver.last_backend == "gpu" else 1
    meta[0]["solve_tile_count"] = int(np.count_nonzero(solve_tile_mask))
    meta[0]["post_tile_count"] = int(np.count_nonzero(post_tile_mask))
    meta[0]["post_cell_count"] = int(np.count_nonzero(post_cell_mask))
    meta[0]["vertical_seam_cell_count"] = int(np.count_nonzero(vertical_seam_mask))
    meta[0]["horizontal_seam_cell_count"] = int(np.count_nonzero(horizontal_seam_mask))
    meta[0]["buoyancy_candidate_count"] = int(np.count_nonzero(buoyancy_mask))
    meta[0]["changed_cell_count"] = int(np.count_nonzero(changed_cell_mask))
    meta[0]["material_changed"] = int(bool(snapshot["material_changed"]))
    meta[0]["phase_changed"] = int(bool(snapshot["phase_changed"]))
    meta[0]["velocity_changed"] = int(bool(snapshot["velocity_changed"]))
    meta[0]["temperature_changed"] = int(bool(snapshot["temperature_changed"]))
    meta[0]["integrity_changed"] = int(bool(snapshot["integrity_changed"]))
    meta[0]["placeholder_changed"] = int(bool(snapshot["placeholder_changed"]))
    meta[0]["pending_placeholder_count_before"] = int(snapshot["pending_placeholder_count_before"])
    meta[0]["pending_placeholder_count_after"] = int(snapshot["pending_placeholder_count_after"])
    meta[0]["liquid_cell_count_before"] = int(snapshot["liquid_cell_count_before"])
    meta[0]["liquid_cell_count_after"] = int(snapshot["liquid_cell_count_after"])
    return (
        meta,
        solve_tile_mask,
        post_tile_mask,
        post_cell_mask,
        vertical_seam_mask,
        horizontal_seam_mask,
        buoyancy_mask,
        changed_cell_mask,
    )


def pack_reaction_runtime_upload(
    world: "WorldEngine",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    snapshot = world.reaction_solver.runtime_snapshot()
    stage_tile_masks = {
        stage: np.asarray(snapshot["stage_tile_masks"].get(stage, []), dtype=np.uint8)
        for stage in (
            "timed",
            "self",
            "material_material",
            "material_gas",
            "material_light",
            "gas_gas",
            "gas_light",
        )
    }
    for stage, mask in stage_tile_masks.items():
        if mask.shape != (world.active.tile_height, world.active.tile_width):
            stage_tile_masks[stage] = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    if solve_cell_mask.shape != (world.height, world.width):
        solve_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    if solve_gas_mask.shape != (world.gas_height, world.gas_width):
        solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
    if changed_cell_mask.shape != (world.height, world.width):
        changed_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    changed_gas_mask = np.asarray(snapshot["changed_gas_mask"], dtype=np.uint8)
    if changed_gas_mask.shape != (world.gas_height, world.gas_width):
        changed_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    ambient_changed_mask = np.asarray(snapshot["ambient_changed_mask"], dtype=np.uint8)
    if ambient_changed_mask.shape != (world.gas_height, world.gas_width):
        ambient_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    timer_changed_mask = np.asarray(snapshot["timer_changed_mask"], dtype=np.uint8)
    if timer_changed_mask.shape != (world.height, world.width):
        timer_changed_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    emitted_light_mask = np.asarray(snapshot["emitted_light_mask"], dtype=np.uint8)
    if emitted_light_mask.shape != (world.height, world.width):
        emitted_light_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    emitted_material_mask = np.asarray(snapshot["emitted_material_mask"], dtype=np.uint8)
    if emitted_material_mask.shape != (world.height, world.width):
        emitted_material_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    stage_action_counts = dict(snapshot["stage_action_counts"])
    solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    for mask in stage_tile_masks.values():
        solve_tile_mask |= mask

    meta = np.zeros((1,), dtype=REACTION_RUNTIME_META_DTYPE)
    backend = str(snapshot["backend"])
    meta[0]["backend_id"] = 3 if backend == "hybrid" else (2 if backend == "gpu" else 1)
    meta[0]["solve_tile_count"] = int(np.count_nonzero(solve_tile_mask))
    meta[0]["solve_cell_count"] = int(np.count_nonzero(solve_cell_mask))
    meta[0]["solve_gas_count"] = int(np.count_nonzero(solve_gas_mask))
    meta[0]["changed_cell_count"] = int(np.count_nonzero(changed_cell_mask))
    meta[0]["changed_gas_count"] = int(np.count_nonzero(changed_gas_mask))
    meta[0]["ambient_changed_count"] = int(np.count_nonzero(ambient_changed_mask))
    meta[0]["timer_changed_count"] = int(np.count_nonzero(timer_changed_mask))
    meta[0]["executed_action_count"] = int(snapshot["executed_action_count"])
    meta[0]["emitted_light_count"] = int(snapshot["emitted_light_count"])
    meta[0]["emitted_material_count"] = int(snapshot["emitted_material_count"])
    meta[0]["emit_light_action_count"] = int(snapshot["emit_light_action_count"])
    meta[0]["emit_material_action_count"] = int(snapshot["emit_material_action_count"])
    meta[0]["modify_gas_action_count"] = int(snapshot["modify_gas_action_count"])
    meta[0]["convert_material_action_count"] = int(snapshot["convert_material_action_count"])
    meta[0]["modify_temperature_action_count"] = int(snapshot["modify_temperature_action_count"])
    meta[0]["harm_action_count"] = int(snapshot["harm_action_count"])
    meta[0]["timed_action_count"] = int(stage_action_counts.get("timed", 0))
    meta[0]["self_action_count"] = int(stage_action_counts.get("self", 0))
    meta[0]["material_material_action_count"] = int(stage_action_counts.get("material_material", 0))
    meta[0]["material_gas_action_count"] = int(stage_action_counts.get("material_gas", 0))
    meta[0]["material_light_action_count"] = int(stage_action_counts.get("material_light", 0))
    meta[0]["gas_gas_action_count"] = int(stage_action_counts.get("gas_gas", 0))
    meta[0]["gas_light_action_count"] = int(stage_action_counts.get("gas_light", 0))
    return (
        meta,
        stage_tile_masks["timed"],
        stage_tile_masks["self"],
        stage_tile_masks["material_material"],
        stage_tile_masks["material_gas"],
        stage_tile_masks["material_light"],
        stage_tile_masks["gas_gas"],
        stage_tile_masks["gas_light"],
        solve_cell_mask,
        solve_gas_mask,
        changed_cell_mask,
        changed_gas_mask,
        ambient_changed_mask,
        timer_changed_mask,
        emitted_light_mask,
        emitted_material_mask,
    )


def pack_collapse_runtime_upload(
    world: "WorldEngine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = world.collapse_solver.runtime_snapshot()
    solve_region_mask = np.asarray(snapshot["solve_region_mask"], dtype=np.int32)
    if solve_region_mask.shape != (world.height, world.width):
        solve_region_mask = np.zeros((world.height, world.width), dtype=np.int32)
    structural_mask = np.asarray(snapshot["structural_mask"], dtype=np.int32)
    if structural_mask.shape != (world.height, world.width):
        structural_mask = np.zeros((world.height, world.width), dtype=np.int32)
    support_seed_mask = np.asarray(snapshot["support_seed_mask"], dtype=np.int32)
    if support_seed_mask.shape != (world.height, world.width):
        support_seed_mask = np.zeros((world.height, world.width), dtype=np.int32)
    supported_mask = np.asarray(snapshot["supported_mask"], dtype=np.int32)
    if supported_mask.shape != (world.height, world.width):
        supported_mask = np.zeros((world.height, world.width), dtype=np.int32)
    unsupported_mask = np.asarray(snapshot["unsupported_mask"], dtype=np.int32)
    if unsupported_mask.shape != (world.height, world.width):
        unsupported_mask = np.zeros((world.height, world.width), dtype=np.int32)
    delayed_pending_mask = np.asarray(snapshot["delayed_pending_mask"], dtype=np.int32)
    if delayed_pending_mask.shape != (world.height, world.width):
        delayed_pending_mask = np.zeros((world.height, world.width), dtype=np.int32)
    immune_unsupported_mask = np.asarray(snapshot["immune_unsupported_mask"], dtype=np.int32)
    if immune_unsupported_mask.shape != (world.height, world.width):
        immune_unsupported_mask = np.zeros((world.height, world.width), dtype=np.int32)
    collapsed_cell_mask = np.asarray(snapshot["collapsed_cell_mask"], dtype=np.int32)
    if collapsed_cell_mask.shape != (world.height, world.width):
        collapsed_cell_mask = np.zeros((world.height, world.width), dtype=np.int32)
    components = np.zeros((len(snapshot["collapsed_components"]),), dtype=COLLAPSE_COMPONENT_DTYPE)
    for index, component in enumerate(snapshot["collapsed_components"]):
        components[index]["island_id"] = int(component["island_id"])
        components[index]["bbox"] = np.asarray(component["bbox"], dtype=np.int32)
        components[index]["cell_count"] = int(component["cell_count"])

    meta = np.zeros((1,), dtype=COLLAPSE_RUNTIME_META_DTYPE)
    meta[0]["backend_id"] = 2 if str(snapshot["backend"]) == "gpu" else 1
    meta[0]["dirty_region_count_before"] = int(snapshot["dirty_region_count_before"])
    meta[0]["solve_region_count"] = int(snapshot["solve_region_count"])
    meta[0]["solve_region_cell_count"] = int(np.count_nonzero(solve_region_mask))
    meta[0]["structural_cell_count"] = int(np.count_nonzero(structural_mask))
    meta[0]["support_seed_count"] = int(np.count_nonzero(support_seed_mask))
    meta[0]["supported_cell_count"] = int(np.count_nonzero(supported_mask))
    meta[0]["unsupported_cell_count"] = int(np.count_nonzero(unsupported_mask))
    meta[0]["delayed_pending_count"] = int(np.count_nonzero(delayed_pending_mask))
    meta[0]["immune_unsupported_count"] = int(np.count_nonzero(immune_unsupported_mask))
    meta[0]["collapsed_cell_count"] = int(np.count_nonzero(collapsed_cell_mask))
    meta[0]["collapsed_component_count"] = int(components.shape[0])
    return (
        meta,
        solve_region_mask,
        structural_mask,
        support_seed_mask,
        supported_mask,
        unsupported_mask,
        delayed_pending_mask,
        immune_unsupported_mask,
        collapsed_cell_mask,
        components,
    )


def pack_optics_runtime_upload(
    world: "WorldEngine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot = world.optics_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    if solve_tile_mask.shape != (world.active.tile_height, world.active.tile_width):
        solve_tile_mask = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8)
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    if solve_cell_mask.shape != (world.height, world.width):
        solve_cell_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    if solve_gas_mask.shape != (world.gas_height, world.gas_width):
        solve_gas_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    visible_changed_mask = np.asarray(snapshot["visible_changed_mask"], dtype=np.uint8)
    if visible_changed_mask.shape != (world.height, world.width):
        visible_changed_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    cell_dose_changed_mask = np.asarray(snapshot["cell_dose_changed_mask"], dtype=np.uint8)
    if cell_dose_changed_mask.shape != (world.height, world.width):
        cell_dose_changed_mask = np.zeros((world.height, world.width), dtype=np.uint8)
    gas_dose_changed_mask = np.asarray(snapshot["gas_dose_changed_mask"], dtype=np.uint8)
    if gas_dose_changed_mask.shape != (world.gas_height, world.gas_width):
        gas_dose_changed_mask = np.zeros((world.gas_height, world.gas_width), dtype=np.uint8)
    emitter_origin_mask = np.asarray(snapshot["emitter_origin_mask"], dtype=np.uint8)
    if emitter_origin_mask.shape != (world.height, world.width):
        emitter_origin_mask = np.zeros((world.height, world.width), dtype=np.uint8)

    meta = np.zeros((1,), dtype=OPTICS_RUNTIME_META_DTYPE)
    backend = str(snapshot["backend"])
    meta[0]["backend_id"] = 3 if backend == "hybrid" else (2 if backend == "gpu" else 1)
    meta[0]["emitter_count"] = int(snapshot["emitter_count"])
    meta[0]["secondary_branch_count"] = int(snapshot["secondary_branch_count"])
    meta[0]["solve_tile_count"] = int(np.count_nonzero(solve_tile_mask))
    meta[0]["solve_cell_count"] = int(np.count_nonzero(solve_cell_mask))
    meta[0]["solve_gas_count"] = int(np.count_nonzero(solve_gas_mask))
    meta[0]["visible_changed_count"] = int(np.count_nonzero(visible_changed_mask))
    meta[0]["cell_dose_changed_count"] = int(np.count_nonzero(cell_dose_changed_mask))
    meta[0]["gas_dose_changed_count"] = int(np.count_nonzero(gas_dose_changed_mask))
    meta[0]["visible_energy_total"] = float(snapshot["visible_energy_total"])
    meta[0]["cell_dose_total"] = float(snapshot["cell_dose_total"])
    meta[0]["gas_dose_total"] = float(snapshot["gas_dose_total"])
    return (
        meta,
        solve_tile_mask,
        solve_cell_mask,
        solve_gas_mask,
        visible_changed_mask,
        cell_dose_changed_mask,
        gas_dose_changed_mask,
        emitter_origin_mask,
    )


PAGE_STRIPE_AXIS_IDS = {"x": 1, "y": 2}
PAGE_STRIPE_KIND_IDS = {"save": 1, "load": 2}
PAGE_STRIPE_DTYPE_CODES = {
    np.dtype(np.uint8).str: 1,
    np.dtype(np.int32).str: 2,
    np.dtype(np.uint32).str: 3,
    np.dtype(np.float32).str: 4,
}
PAGE_STRIPE_FIELD_PATHS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("cell", "material_id")),
    (2, ("cell", "phase")),
    (3, ("cell", "cell_flags")),
    (4, ("cell", "velocity")),
    (5, ("cell", "cell_temperature")),
    (6, ("cell", "timer_pack")),
    (7, ("cell", "integrity")),
    (8, ("cell", "island_id")),
    (9, ("cell", "entity_id")),
    (10, ("cell", "placeholder_displaced_material")),
    (11, ("cell", "collapse_delay_pending")),
    (12, ("cell", "visible_illumination")),
    (13, ("cell", "cell_optical_dose")),
    (14, ("gas", "ambient_temperature")),
    (15, ("gas", "flow_velocity")),
    (16, ("gas", "pressure_ping")),
    (17, ("gas", "gas_concentration")),
    (18, ("gas", "gas_optical_dose")),
    (19, ("runtime", "island_ids")),
    (20, ("runtime", "island_velocity")),
    (21, ("runtime", "island_subcell_offset")),
    (22, ("runtime", "entity_placeholder_entity_id")),
)


def _page_stripe_payload_key(update: PageStripeUpdate) -> tuple[str, int, int, int, int, str]:
    return (
        update.axis,
        int(update.world_start),
        int(update.world_end),
        int(update.buffer_start),
        int(update.buffer_end),
        update.kind,
    )


def _page_stripe_payload_array(payload: dict[str, Any], path: tuple[str, ...]) -> np.ndarray:
    cursor: Any = payload
    for key in path:
        cursor = cursor[key]
    return np.ascontiguousarray(cursor)


def pack_page_stripe_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    updates = world.bridge_frame_paging_updates
    meta = np.zeros((len(updates),), dtype=PAGE_STRIPE_META_DTYPE)
    payload_map: dict[tuple[str, int, int, int, int, str], list[dict[str, Any]]] = {}
    for update, payload in world.bridge_frame_page_stripes:
        payload_map.setdefault(_page_stripe_payload_key(update), []).append(payload)

    sections: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    payload_chunks: list[bytes] = []
    payload_offset = 0
    for stripe_index, update in enumerate(updates):
        key = _page_stripe_payload_key(update)
        stripe_payloads = payload_map.get(key)
        payload = stripe_payloads.pop(0) if stripe_payloads else None
        section_offset = len(sections)
        if payload is not None:
            for field_id, path in PAGE_STRIPE_FIELD_PATHS:
                array = _page_stripe_payload_array(payload, path)
                dtype_code = PAGE_STRIPE_DTYPE_CODES.get(array.dtype.str, 0)
                if dtype_code == 0:
                    raise ValueError(f"Unsupported page stripe dtype: {array.dtype.str}")
                dims = tuple(int(dim) for dim in array.shape)
                padded_dims = (dims + (1, 1, 1))[:3]
                sections.append(
                    (
                        stripe_index,
                        field_id,
                        dtype_code,
                        array.ndim,
                        padded_dims[0],
                        padded_dims[1],
                        padded_dims[2],
                        payload_offset,
                        array.nbytes,
                    )
                )
                payload_chunks.append(array.tobytes())
                payload_offset += array.nbytes
        meta[stripe_index]["axis_id"] = PAGE_STRIPE_AXIS_IDS.get(update.axis, 0)
        meta[stripe_index]["kind_id"] = PAGE_STRIPE_KIND_IDS.get(update.kind, 0)
        meta[stripe_index]["world_start"] = int(update.world_start)
        meta[stripe_index]["world_end"] = int(update.world_end)
        meta[stripe_index]["buffer_start"] = int(update.buffer_start)
        meta[stripe_index]["buffer_end"] = int(update.buffer_end)
        meta[stripe_index]["section_offset"] = section_offset
        meta[stripe_index]["section_count"] = len(sections) - section_offset

    section_array = np.array(sections, dtype=PAGE_STRIPE_SECTION_DTYPE) if sections else np.zeros((0,), dtype=PAGE_STRIPE_SECTION_DTYPE)
    payload_array = np.frombuffer(b"".join(payload_chunks), dtype=np.uint8).copy() if payload_chunks else np.zeros((0,), dtype=np.uint8)
    return meta, section_array, payload_array
