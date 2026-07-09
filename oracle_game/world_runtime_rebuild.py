"""Runtime rebuild / runtime-index helpers extracted from WorldEngine.

Behavior-preserving module-level mirrors of the corresponding WorldEngine
methods; the engine delegates to these one-for-one.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.gpu import (
    FALLING_ISLAND_BREAK_KIND_IDS,
    LIQUID_SOLVER_KIND_IDS,
    POWDER_SOLVER_KIND_IDS,
    typed_gas_id,
)
from oracle_game.sim.gpu_collapse_dirty import drain_collapse_structure_dirty_tile_regions
from oracle_game.types import (
    COLLAPSE_BEHAVIOR_IDS,
    FallingIslandRecord,
    PageStripeUpdate,
    Phase,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _rebuild_sparse_runtime_indexes(engine) -> None:
    _rebuild_entity_placeholder_index(engine)
    _rebuild_island_records(engine)

def _rebuild_entity_placeholder_index(engine) -> None:
    engine.entity_placeholders.clear()
    placeholder_mask = (engine.entity_id > 0) & engine._material_placeholder_mask(engine.material_id)
    ys, xs = np.nonzero(placeholder_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        entity_id = int(engine.entity_id[y, x])
        engine.entity_placeholders.setdefault(entity_id, set()).add((int(x), int(y)))

def _normalize_cell_runtime_arrays(
    engine,
    material_id: np.ndarray,
    phase: np.ndarray,
    island_id: np.ndarray,
    entity_id: np.ndarray,
    placeholder_displaced_material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=np.uint8).copy()
    island_id = np.asarray(island_id, dtype=np.int32).copy()
    entity_id = np.asarray(entity_id, dtype=np.int32).copy()
    placeholder_displaced_material = np.asarray(placeholder_displaced_material, dtype=np.int32).copy()
    empty_mask = material_id <= 0
    phase[empty_mask] = 0
    placeholder_mask = engine._material_placeholder_mask(material_id)
    non_placeholder_mask = empty_mask | ~placeholder_mask
    entity_id[non_placeholder_mask] = 0
    placeholder_displaced_material[non_placeholder_mask] = 0
    invalid_island_mask = (island_id > 0) & (
        (phase != int(Phase.FALLING_ISLAND)) | (material_id <= 0)
    )
    island_id[invalid_island_mask] = 0
    return phase, island_id, entity_id, placeholder_displaced_material

def _normalize_page_stripe_cell_runtime(engine, update: PageStripeUpdate) -> None:
    if engine._gpu_pipeline_available(
        engine.page_stripe_pipeline,
        "page stripe normalization",
        require=engine.simulation_backend == "gpu",
    ):
        engine.page_stripe_pipeline.normalize_cell_runtime(engine, update)
        return
    engine._require_cpu_oracle_backend("page stripe normalization")
    cell_axis = 1 if update.axis == "x" else 0
    material_id = engine._capture_stripe_array(engine.material_id, update, stripe_axis=cell_axis)
    phase = engine._capture_stripe_array(engine.phase, update, stripe_axis=cell_axis)
    island_id = engine._capture_stripe_array(engine.island_id, update, stripe_axis=cell_axis)
    entity_id = engine._capture_stripe_array(engine.entity_id, update, stripe_axis=cell_axis)
    placeholder_displaced_material = engine._capture_stripe_array(
        engine.placeholder_displaced_material,
        update,
        stripe_axis=cell_axis,
    )
    phase, island_id, entity_id, placeholder_displaced_material = _normalize_cell_runtime_arrays(
        engine,
        material_id,
        phase,
        island_id,
        entity_id,
        placeholder_displaced_material,
    )
    engine._write_stripe_array(engine.phase, update, phase, stripe_axis=cell_axis)
    engine._write_stripe_array(engine.island_id, update, island_id, stripe_axis=cell_axis)
    engine._write_stripe_array(engine.entity_id, update, entity_id, stripe_axis=cell_axis)
    engine._write_stripe_array(
        engine.placeholder_displaced_material,
        update,
        placeholder_displaced_material,
        stripe_axis=cell_axis,
    )

def _capture_page_stripe_entity_placeholder_runtime(
    engine,
    update: PageStripeUpdate,
    *,
    stripe_axis: int,
) -> np.ndarray:
    live_placeholder_mask = (engine.entity_id > 0) & engine._material_placeholder_mask(engine.material_id)
    grid = np.where(live_placeholder_mask, engine.entity_id, 0).astype(np.int32)
    for entity_id, cells in engine.entity_placeholders.items():
        for x, y in cells:
            if engine.in_bounds(x, y):
                grid[y, x] = int(entity_id)
    return engine._capture_stripe_array(grid, update, stripe_axis=stripe_axis)

def _apply_page_stripe_entity_placeholder_runtime(
    engine,
    update: PageStripeUpdate,
    entity_placeholder_entity_id: np.ndarray | None,
) -> None:
    cell_axis = 1 if update.axis == "x" else 0
    if entity_placeholder_entity_id is None:
        if engine.simulation_backend == "gpu":
            raise RuntimeError(
                "GPU page stripe placeholder runtime requires payload runtime data; CPU fallback is disabled"
            )
        placeholder_mask = (engine.entity_id > 0) & engine._material_placeholder_mask(engine.material_id)
        dense_placeholder_ids = np.where(placeholder_mask, engine.entity_id, 0).astype(np.int32)
        entity_placeholder_entity_id = engine._capture_stripe_array(
            dense_placeholder_ids,
            update,
            stripe_axis=cell_axis,
        )
    spans = engine._stripe_buffer_ranges(update, gas_grid=False)
    if update.axis == "x":
        def cell_in_loaded_stripe(cell: tuple[int, int]) -> bool:
            return any(start <= cell[0] < end for start, end in spans)
    else:
        def cell_in_loaded_stripe(cell: tuple[int, int]) -> bool:
            return any(start <= cell[1] < end for start, end in spans)

    next_entity_cells: dict[int, set[tuple[int, int]]] = {}
    for entity_id, cells in engine.entity_placeholders.items():
        filtered = {cell for cell in cells if not cell_in_loaded_stripe(cell)}
        if filtered:
            next_entity_cells[int(entity_id)] = filtered

    offset = 0
    for start, end in spans:
        span = end - start
        if update.axis == "x":
            stripe_slice = entity_placeholder_entity_id[:, offset : offset + span]
        else:
            stripe_slice = entity_placeholder_entity_id[offset : offset + span, :]
        ys, xs = np.nonzero(stripe_slice > 0)
        for local_y, local_x in zip(ys.tolist(), xs.tolist()):
            entity_id = int(stripe_slice[local_y, local_x])
            if entity_id <= 0:
                continue
            cell = (start + local_x, local_y) if update.axis == "x" else (local_x, start + local_y)
            next_entity_cells.setdefault(entity_id, set()).add(cell)
        offset += span
    engine.entity_placeholders = next_entity_cells

def _rebuild_island_records(engine) -> None:
    previous_records = dict(engine.islands)
    engine.islands.clear()
    invalid_island_mask = (engine.island_id > 0) & (
        (engine.phase != int(Phase.FALLING_ISLAND)) | (engine.material_id <= 0)
    )
    if np.any(invalid_island_mask):
        engine.island_id[invalid_island_mask] = 0
    for island_id in np.unique(engine.island_id):
        island_id = int(island_id)
        if island_id <= 0:
            continue
        coords = np.argwhere(
            (engine.island_id == island_id)
            & (engine.phase == int(Phase.FALLING_ISLAND))
            & (engine.material_id > 0)
        )
        if coords.size == 0:
            continue
        min_y, min_x = coords.min(axis=0).tolist()
        max_y, max_x = coords.max(axis=0).tolist()
        previous = previous_records.get(island_id)
        if previous is None:
            velocity_xy = tuple(np.mean(engine.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32).tolist())
            subcell_offset = (0.0, 0.0)
        else:
            velocity_xy = (float(previous.velocity_xy[0]), float(previous.velocity_xy[1]))
            subcell_offset = (float(previous.subcell_offset[0]), float(previous.subcell_offset[1]))
        engine.islands[island_id] = FallingIslandRecord(
            island_id=island_id,
            bbox=(int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1),
            velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
            subcell_offset=subcell_offset,
        )
    engine.next_island_id = max(1, max(engine.islands, default=0) + 1)

def _capture_page_stripe_island_runtime(engine, stripe_island_ids: np.ndarray) -> dict[str, np.ndarray]:
    island_ids = sorted(int(island_id) for island_id in np.unique(stripe_island_ids) if int(island_id) > 0)
    if not island_ids:
        return {
            "island_ids": np.zeros((0,), dtype=np.int32),
            "island_velocity": np.zeros((0, 2), dtype=np.float32),
            "island_subcell_offset": np.zeros((0, 2), dtype=np.float32),
        }
    velocity = np.zeros((len(island_ids), 2), dtype=np.float32)
    subcell_offset = np.zeros((len(island_ids), 2), dtype=np.float32)
    for index, island_id in enumerate(island_ids):
        record = engine.islands.get(island_id)
        if record is None:
            coords = np.argwhere(
                (engine.island_id == island_id)
                & (engine.phase == int(Phase.FALLING_ISLAND))
                & (engine.material_id > 0)
            )
            if coords.size != 0:
                mean_velocity = np.mean(engine.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32)
                velocity[index] = mean_velocity
            continue
        velocity[index] = np.asarray(record.velocity_xy, dtype=np.float32)
        subcell_offset[index] = np.asarray(record.subcell_offset, dtype=np.float32)
    return {
        "island_ids": np.asarray(island_ids, dtype=np.int32),
        "island_velocity": velocity,
        "island_subcell_offset": subcell_offset,
    }
def _merge_island_runtime_payload(
    engine,
    runtime_payload: dict[str, Any] | None,
    *,
    update: PageStripeUpdate | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if not runtime_payload:
        return
    island_ids = np.asarray(runtime_payload.get("island_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    island_velocity = np.asarray(runtime_payload.get("island_velocity", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    island_subcell_offset = np.asarray(runtime_payload.get("island_subcell_offset", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    payload_bboxes = (
        engine._page_stripe_island_bboxes_from_payload(update, payload)
        if update is not None and payload is not None
        else None
    )
    count = min(len(island_ids), len(island_velocity), len(island_subcell_offset))
    for index in range(count):
        island_id = int(island_ids[index])
        if island_id <= 0:
            continue
        if payload_bboxes is not None and island_id not in payload_bboxes:
            engine.islands.pop(island_id, None)
            continue
        previous = engine.islands.get(island_id)
        bbox = (
            payload_bboxes[island_id]
            if payload_bboxes is not None
            else (0, 0, 0, 0) if previous is None else previous.bbox
        )
        engine.islands[island_id] = FallingIslandRecord(
            island_id=island_id,
            bbox=bbox,
            velocity_xy=(float(island_velocity[index, 0]), float(island_velocity[index, 1])),
            subcell_offset=(float(island_subcell_offset[index, 0]), float(island_subcell_offset[index, 1])),
        )

def _rebuild_material_property_arrays(engine) -> None:
    max_id = max(engine.rulebook.materials_by_id, default=0)
    size = max_id + 1
    engine.material_base_color = np.zeros((size, 3), dtype=np.float32)
    engine.material_density = np.zeros(size, dtype=np.float32)
    engine.material_gravity = np.zeros(size, dtype=np.float32)
    engine.material_wind = np.zeros(size, dtype=np.float32)
    engine.material_drag = np.zeros(size, dtype=np.float32)
    engine.material_friction = np.zeros(size, dtype=np.float32)
    engine.material_elasticity = np.zeros(size, dtype=np.float32)
    engine.material_max_dda_step = np.zeros(size, dtype=np.int32)
    engine.material_default_phase = np.zeros(size, dtype=np.uint8)
    engine.material_base_integrity = np.zeros(size, dtype=np.float32)
    engine.material_spawn_temperature = np.full(size, np.nan, dtype=np.float32)
    engine.material_reaction_slots = np.full((size, 8), -1, dtype=np.int32)
    engine.material_material_tag_mask = np.zeros(size, dtype=np.uint32)
    engine.material_gas_tag_mask = np.zeros(size, dtype=np.uint32)
    engine.material_light_tag_mask = np.zeros(size, dtype=np.uint32)
    engine.material_powder_solver_kind = np.zeros(size, dtype=np.uint8)
    engine.material_liquid_solver_kind = np.zeros(size, dtype=np.uint8)
    engine.material_falling_island_break_kind = np.zeros(size, dtype=np.uint8)
    engine.material_heat_capacity = np.zeros(size, dtype=np.float32)
    engine.material_conductivity = np.zeros(size, dtype=np.float32)
    engine.material_ambient_exchange = np.zeros(size, dtype=np.float32)
    engine.material_is_structural = np.zeros(size, dtype=np.bool_)
    engine.material_is_support_anchor = np.zeros(size, dtype=np.bool_)
    engine.material_is_plant = np.zeros(size, dtype=np.bool_)
    engine.material_is_placeholder = np.zeros(size, dtype=np.bool_)
    engine.material_collapse_behavior = np.zeros(size, dtype=np.uint8)
    engine.material_collapse_generation_id = np.zeros(size, dtype=np.int32)
    engine.material_powder_generation_id = np.zeros(size, dtype=np.int32)
    engine.material_name_by_id = [""] * size
    engine.placeholder_material_id = 0
    for material_id, material in engine.rulebook.materials_by_id.items():
        engine.material_base_color[material_id] = np.asarray(material.base_color, dtype=np.float32)
        engine.material_density[material_id] = material.density
        engine.material_gravity[material_id] = material.gravity_scale
        engine.material_wind[material_id] = material.wind_coupling
        engine.material_drag[material_id] = material.drag_scale
        engine.material_friction[material_id] = material.friction
        engine.material_elasticity[material_id] = material.elasticity
        engine.material_max_dda_step[material_id] = int(material.max_dda_step)
        engine.material_default_phase[material_id] = np.uint8(int(material.default_phase))
        engine.material_base_integrity[material_id] = material.base_integrity
        engine.material_spawn_temperature[material_id] = (
            np.float32(material.spawn_temperature) if material.spawn_temperature is not None else np.float32(np.nan)
        )
        engine.material_reaction_slots[material_id] = np.asarray(material.reaction_slots, dtype=np.int32)
        engine.material_material_tag_mask[material_id] = np.uint32(material.material_tag_mask)
        engine.material_gas_tag_mask[material_id] = np.uint32(material.gas_tag_mask)
        engine.material_light_tag_mask[material_id] = np.uint32(material.light_tag_mask)
        engine.material_powder_solver_kind[material_id] = np.uint8(POWDER_SOLVER_KIND_IDS.get(material.powder_solver_kind, 0))
        engine.material_liquid_solver_kind[material_id] = np.uint8(LIQUID_SOLVER_KIND_IDS.get(material.liquid_solver_kind, 0))
        engine.material_falling_island_break_kind[material_id] = np.uint8(
            FALLING_ISLAND_BREAK_KIND_IDS.get(material.falling_island_break_kind, 0)
        )
        engine.material_heat_capacity[material_id] = material.heat_capacity
        engine.material_conductivity[material_id] = material.conductivity
        engine.material_ambient_exchange[material_id] = material.ambient_exchange_rate
        engine.material_is_structural[material_id] = material.is_structural
        engine.material_is_support_anchor[material_id] = material.is_support_anchor
        engine.material_is_plant[material_id] = material.render_group == "plant" or "plant" in material.tags
        engine.material_is_placeholder[material_id] = material.render_group == "placeholder" or "placeholder" in material.tags
        engine.material_collapse_behavior[material_id] = np.uint8(COLLAPSE_BEHAVIOR_IDS.get(material.collapse_behavior, 0))
        engine.material_collapse_generation_id[material_id] = int(
            engine.rulebook.material_id(material.collapse_generation) if material.collapse_generation else 0
        )
        engine.material_powder_generation_id[material_id] = int(
            engine.rulebook.material_id(material.powder_generation) if material.powder_generation else 0
        )
        engine.material_name_by_id[material_id] = material.name
        if material.name == "placeholder_solid":
            engine.placeholder_material_id = int(material_id)
    chaos_convert_bit = int(engine.tag_bits_by_name.get("chaos_convert", 0))
    engine.random_convert_material_ids = [
        int(material_id)
        for material_id, material in sorted(engine.rulebook.materials_by_id.items())
        if chaos_convert_bit != 0
        and bool(int(material.material_tag_mask) & chaos_convert_bit)
        and int(material.default_phase) == int(Phase.POWDER)
    ]

def _rebuild_gas_property_arrays(engine) -> None:
    max_id = max(engine.rulebook.gases_by_id, default=-1)
    size = max(0, max_id + 1)
    engine.gas_material_reaction_tag_mask = np.zeros(size, dtype=np.uint32)
    engine.gas_light_reaction_tag_mask = np.zeros(size, dtype=np.uint32)
    engine.gas_density_factor = np.zeros(size, dtype=np.float32)
    engine.gas_condense_material_id = np.zeros(size, dtype=np.int32)
    engine.gas_name_by_id = [""] * size
    engine.air_gas_species_id = -1
    for species_id, gas in engine.rulebook.gases_by_id.items():
        engine.gas_material_reaction_tag_mask[species_id] = np.uint32(gas.material_reaction_tag_mask)
        engine.gas_light_reaction_tag_mask[species_id] = np.uint32(gas.light_reaction_tag_mask)
        engine.gas_density_factor[species_id] = np.float32(gas.density_factor)
        engine.gas_condense_material_id[species_id] = int(
            engine.rulebook.material_id(gas.condense_to_material) if gas.condense_to_material else 0
        )
        engine.gas_name_by_id[species_id] = gas.name
        if gas.name == "air":
            engine.air_gas_species_id = int(species_id)
    if engine.air_gas_species_id < 0:
        gas_table = engine.bridge.shadow_typed_tables.get("gas_table")
        if gas_table is not None:
            air_species_id = int(typed_gas_id(gas_table, "air"))
            if 0 <= air_species_id < size:
                engine.air_gas_species_id = air_species_id

def _rebuild_light_property_arrays(engine) -> None:
    max_id = max(engine.rulebook.lights_by_id, default=-1)
    size = max(0, max_id + 1)
    engine.light_default_range = np.zeros(size, dtype=np.int32)
    engine.light_dose_channel = np.zeros(size, dtype=np.int32)
    engine.light_color = np.zeros((size, 3), dtype=np.float32)
    engine.light_name_by_id = [""] * size
    for light_id, light in engine.rulebook.lights_by_id.items():
        engine.light_default_range[light_id] = int(light.default_range)
        engine.light_dose_channel[light_id] = int(light.dose_channel_id)
        engine.light_color[light_id] = np.asarray(light.color, dtype=np.float32)
        engine.light_name_by_id[light_id] = light.name

def _cell_participates_in_collapse(engine, material_id: int, phase: int) -> bool:
    return (
        material_id != 0
        and phase != int(Phase.FALLING_ISLAND)
        and (
            engine.material_is_structural[material_id]
            or engine.material_is_support_anchor[material_id]
        )
    )

def _mark_collapse_dirty_rect(engine, x0: int, y0: int, x1: int, y1: int, *, pad: int = 8) -> None:
    engine.collapse_dirty_regions.append(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(engine.width, x1 + pad),
            min(engine.height, y1 + pad),
        )
    )

def _drain_gpu_collapse_structure_dirty_tiles(engine) -> None:
    regions = drain_collapse_structure_dirty_tile_regions(engine)
    if regions:
        engine.collapse_dirty_regions.extend(regions)
