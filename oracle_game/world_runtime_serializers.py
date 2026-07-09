from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def serialize_gas_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.gas_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    species_total = np.asarray(snapshot["species_total_concentration"], dtype=np.float32)
    species_active = np.asarray(snapshot["species_active_concentration"], dtype=np.float32)
    species_runtime = []
    for species_id in range(max(len(engine.gas_name_by_id), int(species_total.shape[0]), int(species_active.shape[0]))):
        species_name = engine._shadow_gas_name(species_id)
        if not species_name:
            continue
        total = float(species_total[species_id]) if species_id < species_total.shape[0] else 0.0
        active = float(species_active[species_id]) if species_id < species_active.shape[0] else 0.0
        species_runtime.append(
            {
                "species_id": int(species_id),
                "species": species_name,
                "total_concentration": round(total, 4),
                "active_concentration": round(active, 4),
            }
        )
    return {
        "backend": engine.gas_solver.last_backend,
        "pressure_iterations": int(snapshot["pressure_iterations"]),
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "gas_grid_size": [int(engine.gas_width), int(engine.gas_height)],
        "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
        "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
        "solve_tile_mask": solve_tile_mask.tolist(),
        "solve_gas_mask": solve_gas_mask.tolist(),
        "force_source_count_before": int(snapshot["force_source_count_before"]),
        "force_source_count_after": int(snapshot["force_source_count_after"]),
        "velocity_changed": bool(snapshot["velocity_changed"]),
        "ambient_changed": bool(snapshot["ambient_changed"]),
        "gas_changed": bool(snapshot["gas_changed"]),
        "pressure_range": np.asarray(snapshot["pressure_range"], dtype=np.float32).round(4).tolist(),
        "ambient_range": np.asarray(snapshot["ambient_range"], dtype=np.float32).round(4).tolist(),
        "flow_speed_range": np.asarray(snapshot["flow_speed_range"], dtype=np.float32).round(4).tolist(),
        "species_runtime": species_runtime,
    }


def serialize_heat_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.heat_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    phase_targets = np.asarray(snapshot["phase_targets"], dtype=np.int32)
    boil_targets = np.asarray(snapshot["boil_targets"], dtype=np.int32)
    condense_targets = np.asarray(snapshot["condense_targets"], dtype=np.bool_)

    phase_payload = []
    public_phase_targets = snapshot.get("public_phase_targets")
    if isinstance(public_phase_targets, list) and public_phase_targets:
        for target in public_phase_targets:
            target_material_id = int(target["target_material_id"])
            phase_payload.append(
                {
                    "x": int(target["x"]),
                    "y": int(target["y"]),
                    "target_material_id": target_material_id,
                    "target_material": engine._shadow_material_name(target_material_id),
                }
            )
    else:
        phase_ys, phase_xs = np.nonzero(phase_targets > 0)
        for y, x in zip(phase_ys.tolist(), phase_xs.tolist()):
            target_material_id = int(phase_targets[y, x])
            world_x, world_y = engine._buffer_to_world_position((int(x), int(y)))
            phase_payload.append(
                {
                    "x": int(world_x),
                    "y": int(world_y),
                    "target_material_id": target_material_id,
                    "target_material": engine._shadow_material_name(target_material_id),
                }
            )

    boil_payload = []
    public_boil_targets = snapshot.get("public_boil_targets")
    if isinstance(public_boil_targets, list) and public_boil_targets:
        for target in public_boil_targets:
            target_species_id = int(target["target_species_id"])
            boil_payload.append(
                {
                    "x": int(target["x"]),
                    "y": int(target["y"]),
                    "target_species_id": target_species_id,
                    "target_species": engine._shadow_gas_name(target_species_id),
                }
            )
    else:
        boil_ys, boil_xs = np.nonzero(boil_targets > 0)
        for y, x in zip(boil_ys.tolist(), boil_xs.tolist()):
            target_species_id = int(boil_targets[y, x]) - 1
            world_x, world_y = engine._buffer_to_world_position((int(x), int(y)))
            boil_payload.append(
                {
                    "x": int(world_x),
                    "y": int(world_y),
                    "target_species_id": target_species_id,
                    "target_species": engine._shadow_gas_name(target_species_id),
                }
            )

    condense_payload = []
    public_condense_targets = snapshot.get("public_condense_targets")
    if isinstance(public_condense_targets, list) and public_condense_targets:
        for target in public_condense_targets:
            species_id = int(target["species_id"])
            target_material_id = int(target["target_material_id"])
            condense_payload.append(
                {
                    "gas_x": int(target["gas_x"]),
                    "gas_y": int(target["gas_y"]),
                    "species_id": species_id,
                    "species": engine._shadow_gas_name(species_id),
                    "target_material_id": target_material_id,
                    "target_material": engine._shadow_material_name(target_material_id),
                }
            )
    else:
        species_count = int(condense_targets.shape[0])
        for species_id in range(species_count):
            gas_y, gas_x = np.nonzero(condense_targets[species_id] & (solve_gas_mask > 0))
            if len(gas_y) == 0:
                continue
            target_material_id = engine._shadow_condense_target_material_id(species_id)
            if target_material_id <= 0:
                continue
            target_material_name = engine._shadow_material_name(target_material_id)
            for gy, gx in zip(gas_y.tolist(), gas_x.tolist()):
                world_gx, world_gy = engine._buffer_gas_to_world_position((int(gx), int(gy)))
                condense_payload.append(
                    {
                        "gas_x": int(world_gx),
                        "gas_y": int(world_gy),
                        "species_id": int(species_id),
                        "species": engine._shadow_gas_name(species_id),
                        "target_material_id": target_material_id,
                        "target_material": target_material_name,
                    }
                )

    return {
        "backend": engine.heat_solver.last_backend,
        "ambient_iterations": int(snapshot["ambient_iterations"]),
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "cell_grid_size": [int(engine.width), int(engine.height)],
        "gas_grid_size": [int(engine.gas_width), int(engine.gas_height)],
        "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
        "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
        "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
        "phase_target_count": int(len(phase_payload)),
        "boil_target_count": int(len(boil_payload)),
        "condense_target_count": int(len(condense_payload)),
        "solve_tile_mask": solve_tile_mask.tolist(),
        "solve_cell_mask": solve_cell_mask.tolist(),
        "solve_gas_mask": solve_gas_mask.tolist(),
        "cell_changed": bool(snapshot["cell_changed"]),
        "ambient_changed": bool(snapshot["ambient_changed"]),
        "material_changed": bool(snapshot["material_changed"]),
        "phase_changed": bool(snapshot["phase_changed"]),
        "integrity_changed": bool(snapshot["integrity_changed"]),
        "gas_changed": bool(snapshot["gas_changed"]),
        "cell_temperature_range": np.asarray(snapshot["cell_temperature_range"], dtype=np.float32).round(4).tolist(),
        "ambient_temperature_range": np.asarray(snapshot["ambient_temperature_range"], dtype=np.float32).round(4).tolist(),
        "integrity_range": np.asarray(snapshot["integrity_range"], dtype=np.float32).round(4).tolist(),
        "phase_targets": phase_payload,
        "boil_targets": boil_payload,
        "condense_targets": condense_payload,
    }


def serialize_liquid_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.liquid_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    post_tile_mask = np.asarray(snapshot["post_tile_mask"], dtype=np.uint8)
    post_cell_mask = np.asarray(snapshot["post_cell_mask"], dtype=np.uint8)
    vertical_seam_mask = np.asarray(snapshot["vertical_seam_mask"], dtype=np.uint8)
    horizontal_seam_mask = np.asarray(snapshot["horizontal_seam_mask"], dtype=np.uint8)
    buoyancy_mask = np.asarray(snapshot["buoyancy_mask"], dtype=np.uint8)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
    return {
        "backend": engine.liquid_solver.last_backend,
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "cell_grid_size": [int(engine.width), int(engine.height)],
        "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
        "post_tile_count": int(np.count_nonzero(post_tile_mask)),
        "post_cell_count": int(np.count_nonzero(post_cell_mask)),
        "vertical_seam_cell_count": int(np.count_nonzero(vertical_seam_mask)),
        "horizontal_seam_cell_count": int(np.count_nonzero(horizontal_seam_mask)),
        "buoyancy_candidate_count": int(np.count_nonzero(buoyancy_mask)),
        "changed_cell_count": int(np.count_nonzero(changed_cell_mask)),
        "material_changed": bool(snapshot["material_changed"]),
        "phase_changed": bool(snapshot["phase_changed"]),
        "velocity_changed": bool(snapshot["velocity_changed"]),
        "temperature_changed": bool(snapshot["temperature_changed"]),
        "integrity_changed": bool(snapshot["integrity_changed"]),
        "placeholder_changed": bool(snapshot["placeholder_changed"]),
        "pending_placeholder_count_before": int(snapshot["pending_placeholder_count_before"]),
        "pending_placeholder_count_after": int(snapshot["pending_placeholder_count_after"]),
        "liquid_cell_count_before": int(snapshot["liquid_cell_count_before"]),
        "liquid_cell_count_after": int(snapshot["liquid_cell_count_after"]),
        "solve_tile_mask": solve_tile_mask.tolist(),
        "post_tile_mask": post_tile_mask.tolist(),
        "post_cell_mask": post_cell_mask.tolist(),
        "vertical_seam_mask": vertical_seam_mask.tolist(),
        "horizontal_seam_mask": horizontal_seam_mask.tolist(),
        "buoyancy_mask": buoyancy_mask.tolist(),
        "changed_cell_mask": changed_cell_mask.tolist(),
    }


def serialize_reaction_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.reaction_solver.runtime_snapshot()
    stage_tile_masks = {
        stage: np.asarray(mask, dtype=np.uint8)
        for stage, mask in snapshot["stage_tile_masks"].items()
    }
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
    changed_gas_mask = np.asarray(snapshot["changed_gas_mask"], dtype=np.uint8)
    ambient_changed_mask = np.asarray(snapshot["ambient_changed_mask"], dtype=np.uint8)
    timer_changed_mask = np.asarray(snapshot["timer_changed_mask"], dtype=np.uint8)
    emitted_light_mask = np.asarray(snapshot["emitted_light_mask"], dtype=np.uint8)
    emitted_material_mask = np.asarray(snapshot["emitted_material_mask"], dtype=np.uint8)
    solve_tile_mask = np.zeros((engine.active.tile_height, engine.active.tile_width), dtype=np.uint8)
    for mask in stage_tile_masks.values():
        solve_tile_mask |= mask
    return {
        "backend": str(snapshot["backend"]),
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "cell_grid_size": [int(engine.width), int(engine.height)],
        "gas_grid_size": [int(engine.gas_width), int(engine.gas_height)],
        "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
        "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
        "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
        "changed_cell_count": int(np.count_nonzero(changed_cell_mask)),
        "changed_gas_count": int(np.count_nonzero(changed_gas_mask)),
        "ambient_changed_count": int(np.count_nonzero(ambient_changed_mask)),
        "timer_changed_count": int(np.count_nonzero(timer_changed_mask)),
        "emitted_light_count": int(snapshot["emitted_light_count"]),
        "emitted_material_count": int(snapshot["emitted_material_count"]),
        "executed_action_count": int(snapshot["executed_action_count"]),
        "emit_light_action_count": int(snapshot["emit_light_action_count"]),
        "emit_material_action_count": int(snapshot["emit_material_action_count"]),
        "modify_gas_action_count": int(snapshot["modify_gas_action_count"]),
        "convert_material_action_count": int(snapshot["convert_material_action_count"]),
        "modify_temperature_action_count": int(snapshot["modify_temperature_action_count"]),
        "harm_action_count": int(snapshot["harm_action_count"]),
        "stage_action_counts": {stage: int(count) for stage, count in snapshot["stage_action_counts"].items()},
        "stage_tile_masks": {stage: mask.tolist() for stage, mask in stage_tile_masks.items()},
        "solve_cell_mask": solve_cell_mask.tolist(),
        "solve_gas_mask": solve_gas_mask.tolist(),
        "changed_cell_mask": changed_cell_mask.tolist(),
        "changed_gas_mask": changed_gas_mask.tolist(),
        "ambient_changed_mask": ambient_changed_mask.tolist(),
        "timer_changed_mask": timer_changed_mask.tolist(),
        "emitted_light_mask": emitted_light_mask.tolist(),
        "emitted_material_mask": emitted_material_mask.tolist(),
    }


def serialize_collapse_runtime(engine: "WorldEngine", *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
    snapshot = engine.collapse_solver.runtime_snapshot(
        engine,
        allow_gpu_sync_readback=allow_gpu_sync_readback,
    )
    solve_region_mask = np.asarray(snapshot["solve_region_mask"], dtype=np.uint8)
    structural_mask = np.asarray(snapshot["structural_mask"], dtype=np.uint8)
    support_seed_mask = np.asarray(snapshot["support_seed_mask"], dtype=np.uint8)
    supported_mask = np.asarray(snapshot["supported_mask"], dtype=np.uint8)
    unsupported_mask = np.asarray(snapshot["unsupported_mask"], dtype=np.uint8)
    delayed_pending_mask = np.asarray(snapshot["delayed_pending_mask"], dtype=np.uint8)
    immune_unsupported_mask = np.asarray(snapshot["immune_unsupported_mask"], dtype=np.uint8)
    collapsed_cell_mask = np.asarray(snapshot["collapsed_cell_mask"], dtype=np.uint8)
    return {
        "backend": str(snapshot["backend"]),
        "gpu_authoritative": bool(snapshot.get("gpu_authoritative", False)),
        "gpu_authoritative_resources": list(snapshot.get("gpu_authoritative_resources", [])),
        "snapshot_source": str(snapshot.get("snapshot_source", "cpu")),
        "snapshot_stale": bool(snapshot.get("snapshot_stale", False)),
        "gpu_authoritative_snapshot_stale": bool(snapshot.get("gpu_authoritative_snapshot_stale", False)),
        "stale_resources": list(snapshot.get("stale_resources", [])),
        "sync_readback_required": bool(snapshot.get("sync_readback_required", False)),
        "sync_readback_performed": bool(snapshot.get("sync_readback_performed", False)),
        "cell_grid_size": [int(engine.width), int(engine.height)],
        "dirty_region_count_before": int(snapshot["dirty_region_count_before"]),
        "solve_region_count": int(snapshot["solve_region_count"]),
        "solve_region_cell_count": int(np.count_nonzero(solve_region_mask)),
        "structural_cell_count": int(np.count_nonzero(structural_mask)),
        "support_seed_count": int(np.count_nonzero(support_seed_mask)),
        "supported_cell_count": int(np.count_nonzero(supported_mask)),
        "unsupported_cell_count": int(np.count_nonzero(unsupported_mask)),
        "delayed_pending_count": int(np.count_nonzero(delayed_pending_mask)),
        "immune_unsupported_count": int(np.count_nonzero(immune_unsupported_mask)),
        "collapsed_cell_count": int(np.count_nonzero(collapsed_cell_mask)),
        "collapsed_component_count": int(len(snapshot["collapsed_components"])),
        "solve_region_mask": solve_region_mask.tolist(),
        "structural_mask": structural_mask.tolist(),
        "support_seed_mask": support_seed_mask.tolist(),
        "supported_mask": supported_mask.tolist(),
        "unsupported_mask": unsupported_mask.tolist(),
        "delayed_pending_mask": delayed_pending_mask.tolist(),
        "immune_unsupported_mask": immune_unsupported_mask.tolist(),
        "collapsed_cell_mask": collapsed_cell_mask.tolist(),
        "collapsed_components": [
            {
                "island_id": int(component["island_id"]),
                "bbox": (
                    list(component["world_bbox"])
                    if component.get("world_bbox") is not None
                    else list(engine._buffer_bbox_to_world_bbox(tuple(int(value) for value in component["bbox"])))
                ),
                "cell_count": int(component["cell_count"]),
            }
            for component in snapshot["collapsed_components"]
        ],
    }


def serialize_optics_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.optics_solver.runtime_snapshot()
    solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
    solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
    solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
    visible_changed_mask = np.asarray(snapshot["visible_changed_mask"], dtype=np.uint8)
    cell_dose_changed_mask = np.asarray(snapshot["cell_dose_changed_mask"], dtype=np.uint8)
    gas_dose_changed_mask = np.asarray(snapshot["gas_dose_changed_mask"], dtype=np.uint8)
    emitter_origin_mask = np.asarray(snapshot["emitter_origin_mask"], dtype=np.uint8)
    public_emitters = snapshot.get("public_emitters")
    emitters_payload = (
        [
            {
                "light_type": str(emitter["light_type"]),
                "origin": list(emitter["origin"]),
                "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
                "spread": float(emitter["spread"]),
                "strength": float(emitter["strength"]),
                "range_cells": int(emitter["range_cells"]),
            }
            for emitter in public_emitters
        ]
        if isinstance(public_emitters, list) and public_emitters
        else [
            {
                "light_type": str(emitter["light_type"]),
                "origin": list(
                    engine._buffer_to_world_position((int(emitter["origin"][0]), int(emitter["origin"][1])))
                ),
                "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
                "spread": float(emitter["spread"]),
                "strength": float(emitter["strength"]),
                "range_cells": int(emitter["range_cells"]),
            }
            for emitter in snapshot["emitters"]
        ]
    )
    return {
        "backend": str(snapshot["backend"]),
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "cell_grid_size": [int(engine.width), int(engine.height)],
        "gas_grid_size": [int(engine.gas_width), int(engine.gas_height)],
        "emitter_count": int(snapshot["emitter_count"]),
        "secondary_branch_count": int(snapshot["secondary_branch_count"]),
        "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
        "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
        "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
        "visible_changed_count": int(np.count_nonzero(visible_changed_mask)),
        "cell_dose_changed_count": int(np.count_nonzero(cell_dose_changed_mask)),
        "gas_dose_changed_count": int(np.count_nonzero(gas_dose_changed_mask)),
        "visible_energy_total": round(float(snapshot["visible_energy_total"]), 4),
        "cell_dose_total": round(float(snapshot["cell_dose_total"]), 4),
        "gas_dose_total": round(float(snapshot["gas_dose_total"]), 4),
        "emitters": emitters_payload,
        "solve_tile_mask": solve_tile_mask.tolist(),
        "solve_cell_mask": solve_cell_mask.tolist(),
        "solve_gas_mask": solve_gas_mask.tolist(),
        "visible_changed_mask": visible_changed_mask.tolist(),
        "cell_dose_changed_mask": cell_dose_changed_mask.tolist(),
        "gas_dose_changed_mask": gas_dose_changed_mask.tolist(),
        "emitter_origin_mask": emitter_origin_mask.tolist(),
    }


def serialize_active_runtime(engine: "WorldEngine") -> dict[str, Any]:
    active_tile_ttl = np.asarray(engine.active.active_tile_ttl, dtype=np.int32)
    active_chunk_mask = np.asarray(engine.active.active_chunk_mask, dtype=np.uint8)
    ys, xs = np.nonzero(engine.placeholder_displaced_material > 0)
    pending_displaced_cells: list[dict[str, Any]] = []
    for buffer_y, buffer_x in zip(ys.tolist(), xs.tolist()):
        world_x, world_y = engine._buffer_to_world_position((buffer_x, buffer_y))
        pending_displaced_cells.append(
            {
                "x": int(world_x),
                "y": int(world_y),
                "material_id": int(engine.placeholder_displaced_material[buffer_y, buffer_x]),
            }
        )
    pending_displaced_cells.sort(key=lambda cell: (int(cell["y"]), int(cell["x"])))
    return {
        "tile_size": int(engine.active.tile_size),
        "chunk_tiles": int(engine.active.chunk_tiles),
        "active_ttl_reset": int(engine.active.active_ttl_reset),
        "tile_grid_size": [int(engine.active.tile_width), int(engine.active.tile_height)],
        "chunk_grid_size": [int(engine.active.chunk_width), int(engine.active.chunk_height)],
        "active_tile_count": int(np.count_nonzero(active_tile_ttl > 0)),
        "active_chunk_count": int(np.count_nonzero(active_chunk_mask > 0)),
        "active_tile_ttl": active_tile_ttl.tolist(),
        "active_chunk_mask": active_chunk_mask.tolist(),
        "pending_displaced_count": int(len(pending_displaced_cells)),
        "pending_displaced_cells": pending_displaced_cells,
    }


def serialize_motion_runtime(engine: "WorldEngine") -> dict[str, Any]:
    snapshot = engine.motion_solver.runtime_snapshot()
    public_powder_reservations = snapshot.get("public_powder_reservations")
    if isinstance(public_powder_reservations, list) and public_powder_reservations:
        powder_payload: list[dict[str, Any]] = [dict(record) for record in public_powder_reservations]
    else:
        powder_payload = []
        for record in snapshot["powder_reservations"]:
            item: dict[str, Any] = {}
            for name in snapshot["powder_reservations"].dtype.names or ():
                value = record[name]
                if isinstance(value, np.ndarray):
                    if name in {"source_xy", "desired_target_xy", "reserved_target_xy", "resolved_target_xy"}:
                        world_x, world_y = engine._buffer_to_world_position((int(value[0]), int(value[1])))
                        item[name] = [int(world_x), int(world_y)]
                    else:
                        item[name] = value.tolist()
                elif isinstance(value, np.generic):
                    item[name] = value.item()
                else:
                    item[name] = value
            powder_payload.append(item)
    public_island_reservations = snapshot.get("public_island_reservations")
    if isinstance(public_island_reservations, list) and public_island_reservations:
        island_payload: list[dict[str, Any]] = [dict(record) for record in public_island_reservations]
    else:
        island_payload = []
        for record in snapshot["island_reservations"]:
            item = {}
            for name in snapshot["island_reservations"].dtype.names or ():
                value = record[name]
                if name == "buffer_bbox":
                    item["world_bbox"] = list(
                        engine._buffer_bbox_to_world_bbox(tuple(int(component) for component in np.asarray(value).tolist()))
                    )
                    continue
                if isinstance(value, np.ndarray):
                    item[name] = value.tolist()
                elif isinstance(value, np.generic):
                    item[name] = value.item()
                else:
                    item[name] = value
            island_payload.append(item)
    return {
        "backend": engine.motion_solver.last_backend,
        "powder_reservation_count": int(snapshot["powder_reservations"].shape[0]),
        "island_reservation_count": int(snapshot["island_reservations"].shape[0]),
        "powder_reservations": powder_payload,
        "island_reservations": island_payload,
    }


def serialize_paging_state(engine: "WorldEngine") -> dict[str, Any]:
    x0, y0, x1, y1 = engine.paging.active_bounds()
    return {
        "origin": [engine.paging.origin_x, engine.paging.origin_y],
        "buffer_origin": [engine.paging.buffer_origin_x, engine.paging.buffer_origin_y],
        "active_bounds": [x0, y0, x1, y1],
        "buffer_size": [engine.width, engine.height],
        "active_size": [engine.paging.active_width, engine.paging.active_height],
        "stored_stripes": engine.page_store.stored_count(),
    }
