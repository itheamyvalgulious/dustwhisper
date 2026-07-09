from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np

from oracle_game.page_store import StoredStripeKey
from oracle_game.types import PageStripeUpdate, TargetQuery

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def focus_paging(engine: "WorldEngine", center_x: int, center_y: int) -> list[PageStripeUpdate]:
    return engine.paging.focus_on(center_x, center_y)


def advance_paging(
    engine: "WorldEngine",
    center_x: int | None,
    center_y: int | None,
    *,
    immediate: bool = False,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> list[PageStripeUpdate]:
    center_x, center_y, resolved_target_query_id = engine._resolve_direct_targeted_coords(
        "advance_paging",
        center_x,
        center_y,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    if immediate:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        return engine._advance_paging(center_x, center_y)
    payload: dict[str, Any] = {"center_x": center_x, "center_y": center_y}
    if resolved_target_query_id is not None:
        payload["resolved_target_query_id"] = resolved_target_query_id
    engine.queue_command("advance_paging", **payload)
    return []


def capture_page_stripe(engine: "WorldEngine", update: PageStripeUpdate) -> dict[str, Any]:
    update = engine._contextualize_page_stripe_update(update)
    if engine.simulation_backend == "gpu":
        if engine._gpu_cpu_dirty_resources:
            engine.bridge.sync_world(engine)
            engine._gpu_cpu_dirty_resources.clear()
        return engine.page_stripe_pipeline.capture(engine, update)
    return _capture_page_stripe_cpu_snapshot(engine, update)


def _capture_page_stripe_cpu_snapshot(engine: "WorldEngine", update: PageStripeUpdate) -> dict[str, Any]:
    gas_ranges = _stripe_buffer_ranges(engine, update, gas_grid=True)
    cell_axis = 1 if update.axis == "x" else 0
    cell_dose_axis = 2 if update.axis == "x" else 1
    material_id = engine._capture_stripe_array(engine.material_id, update, stripe_axis=cell_axis)
    phase = engine._capture_stripe_array(engine.phase, update, stripe_axis=cell_axis)
    island_id = engine._capture_stripe_array(engine.island_id, update, stripe_axis=cell_axis)
    entity_id = engine._capture_stripe_array(engine.entity_id, update, stripe_axis=cell_axis)
    placeholder_displaced_material = engine._capture_stripe_array(
        engine.placeholder_displaced_material,
        update,
        stripe_axis=cell_axis,
    )
    phase, island_id, entity_id, placeholder_displaced_material = engine._normalize_cell_runtime_arrays(
        material_id,
        phase,
        island_id,
        entity_id,
        placeholder_displaced_material,
    )
    runtime_payload = engine._capture_page_stripe_island_runtime(
        island_id
    )
    runtime_payload["entity_placeholder_entity_id"] = engine._capture_page_stripe_entity_placeholder_runtime(
        update,
        stripe_axis=cell_axis,
    )
    payload = {
        "meta": {
            "axis": update.axis,
            "world_start": update.world_start,
            "world_end": update.world_end,
            "buffer_start": update.buffer_start,
            "buffer_end": update.buffer_end,
            "kind": update.kind,
            "cross_world_start": update.cross_world_start,
            "cross_world_end": update.cross_world_end,
        },
        "cell": {
            "material_id": material_id,
            "phase": phase,
            "cell_flags": engine._capture_stripe_array(engine.cell_flags, update, stripe_axis=cell_axis),
            "velocity": engine._capture_stripe_array(engine.velocity, update, stripe_axis=cell_axis),
            "cell_temperature": engine._capture_stripe_array(engine.cell_temperature, update, stripe_axis=cell_axis),
            "timer_pack": engine._capture_stripe_array(engine.timer_pack, update, stripe_axis=cell_axis),
            "integrity": engine._capture_stripe_array(engine.integrity, update, stripe_axis=cell_axis),
            "island_id": island_id,
            "entity_id": entity_id,
            "placeholder_displaced_material": placeholder_displaced_material,
            "collapse_delay_pending": engine._capture_stripe_array(
                engine.collapse_delay_pending.astype(np.uint8),
                update,
                stripe_axis=cell_axis,
            ),
            "visible_illumination": engine._capture_stripe_array(
                engine.visible_illumination,
                update,
                stripe_axis=cell_axis,
            ),
            "cell_optical_dose": engine._capture_stripe_array(
                engine.cell_optical_dose,
                update,
                stripe_axis=cell_dose_axis,
            ),
        },
        "runtime": runtime_payload,
        "gas": {
            "ambient_temperature": engine._capture_stripe_array(
                engine.ambient_temperature,
                update,
                stripe_axis=1 if update.axis == "x" else 0,
                ranges=gas_ranges,
            ),
            "flow_velocity": engine._capture_stripe_array(
                engine.flow_velocity,
                update,
                stripe_axis=1 if update.axis == "x" else 0,
                ranges=gas_ranges,
            ),
            "pressure_ping": engine._capture_stripe_array(
                engine.pressure_ping,
                update,
                stripe_axis=1 if update.axis == "x" else 0,
                ranges=gas_ranges,
            ),
            "gas_concentration": engine._capture_stripe_array(
                engine.gas_concentration,
                update,
                stripe_axis=2 if update.axis == "x" else 1,
                ranges=gas_ranges,
            ),
            "gas_optical_dose": engine._capture_stripe_array(
                engine.gas_optical_dose,
                update,
                stripe_axis=2 if update.axis == "x" else 1,
                ranges=gas_ranges,
            ),
        },
    }
    return payload


def apply_page_stripe(
    engine: "WorldEngine",
    update: PageStripeUpdate,
    payload: dict[str, Any],
    *,
    immediate: bool = False,
) -> None:
    update = engine._contextualize_page_stripe_update(update)
    payload = _coerce_page_stripe_payload(engine, payload)
    if immediate:
        if not engine._bridge_inputs_prepared:
            engine._prepare_bridge_frame_inputs()
        engine.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
        _apply_page_stripe(engine, update, payload)
        engine._record_bridge_page_stripe(update, payload)
        return
    engine.queue_command(
        "apply_page_stripe",
        update=asdict(update),
        payload=payload,
    )


def store_page_stripe(engine: "WorldEngine", update: PageStripeUpdate, payload: dict[str, Any]) -> dict[str, Any]:
    update = engine._contextualize_page_stripe_update(update)
    normalized_payload = _coerce_page_stripe_payload(engine, payload)
    engine.page_store.save(update, normalized_payload)
    stored_payload = engine.page_store.load(update)
    assert stored_payload is not None
    return _coerce_page_stripe_payload(engine, stored_payload)


def load_page_stripe(engine: "WorldEngine", update: PageStripeUpdate) -> dict[str, Any] | None:
    update = engine._contextualize_page_stripe_update(update)
    payload = engine.page_store.load(update)
    if payload is None:
        return None
    return _coerce_page_stripe_payload(engine, payload)


def apply_stored_page_stripe(
    engine: "WorldEngine",
    update: PageStripeUpdate,
    *,
    immediate: bool = False,
) -> dict[str, Any] | None:
    payload = load_page_stripe(engine, update)
    if payload is None:
        return None
    apply_page_stripe(engine, update, payload, immediate=immediate)
    return payload


def page_store_has_stripe(engine: "WorldEngine", update: PageStripeUpdate) -> bool:
    update = engine._contextualize_page_stripe_update(update)
    return bool(engine.page_store.has(update))


def list_page_store_stripe_keys(engine: "WorldEngine") -> list[StoredStripeKey] | None:
    list_keys = getattr(engine.page_store, "keys", None)
    if not callable(list_keys):
        return None
    return [_coerce_page_store_key(engine, key) for key in list_keys()]


def export_page_store_entries(engine: "WorldEngine") -> dict[str, Any]:
    keys = list_page_store_stripe_keys(engine)
    entries: list[dict[str, Any]] = []
    if keys is not None:
        for key in keys:
            payload = engine.page_store.load(engine._page_store_key_lookup_update(key))
            if payload is None:
                continue
            entries.append(
                {
                    "key": engine.serialize_page_store_key(key),
                    "payload": engine.serialize_page_stripe_payload(_coerce_page_stripe_payload(engine, payload)),
                }
            )
    return {
        "stored_stripes": int(engine.page_store.stored_count()),
        "key_listing_supported": keys is not None,
        "entries": entries,
    }


def import_page_store_entries(engine: "WorldEngine", entries: Iterable[dict[str, Any]], *, clear: bool = False) -> dict[str, int]:
    cleared = 0
    if clear:
        cleared = clear_page_store(engine)
    imported = 0
    for entry in entries:
        key = _coerce_page_store_key(engine, entry["key"])
        payload = _coerce_page_stripe_payload(engine, dict(entry["payload"]))
        engine.page_store.save(engine._page_store_key_lookup_update(key), payload)
        imported += 1
    return {
        "cleared": int(cleared),
        "imported": int(imported),
        "stored_stripes": int(engine.page_store.stored_count()),
    }


def clear_page_store(engine: "WorldEngine") -> int:
    cleared = int(engine.page_store.stored_count())
    engine.page_store.clear()
    return cleared


def _coerce_page_store_key(
    engine: "WorldEngine",
    key: StoredStripeKey | PageStripeUpdate | dict[str, Any],
) -> StoredStripeKey:
    if isinstance(key, StoredStripeKey):
        return StoredStripeKey(
            axis=str(key.axis),
            world_start=int(key.world_start),
            world_end=int(key.world_end),
            cross_world_start=int(getattr(key, "cross_world_start", 0)),
            cross_world_end=int(getattr(key, "cross_world_end", 0)),
        )
    if isinstance(key, PageStripeUpdate):
        return StoredStripeKey(
            axis=str(key.axis),
            world_start=int(key.world_start),
            world_end=int(key.world_end),
            cross_world_start=0 if key.cross_world_start is None else int(key.cross_world_start),
            cross_world_end=0 if key.cross_world_end is None else int(key.cross_world_end),
        )
    payload = dict(key)
    return StoredStripeKey(
        axis=str(payload["axis"]),
        world_start=int(payload["world_start"]),
        world_end=int(payload["world_end"]),
        cross_world_start=int(payload.get("cross_world_start", 0)),
        cross_world_end=int(payload.get("cross_world_end", 0)),
    )


def _coerce_page_stripe_payload(engine: "WorldEngine", payload: dict[str, Any]) -> dict[str, Any]:
    cell_payload = dict(payload["cell"])
    gas_payload = dict(payload["gas"])
    runtime_payload = None if payload.get("runtime") is None else dict(payload["runtime"])
    gas_concentration = np.asarray(
        gas_payload["gas_concentration"],
        dtype=engine.gas_concentration.dtype,
    ).copy()
    np.maximum(gas_concentration, 0.0, out=gas_concentration)
    if runtime_payload is not None and "entity_placeholder_entity_id" in runtime_payload:
        runtime_payload["entity_placeholder_entity_id"] = np.asarray(
            runtime_payload["entity_placeholder_entity_id"],
            dtype=np.int32,
        )
    if runtime_payload is not None:
        if "island_ids" in runtime_payload:
            runtime_payload["island_ids"] = np.asarray(runtime_payload["island_ids"], dtype=np.int32)
        if "island_velocity" in runtime_payload:
            runtime_payload["island_velocity"] = np.asarray(runtime_payload["island_velocity"], dtype=np.float32)
        if "island_subcell_offset" in runtime_payload:
            runtime_payload["island_subcell_offset"] = np.asarray(
                runtime_payload["island_subcell_offset"],
                dtype=np.float32,
            )
    return {
        "meta": dict(payload["meta"]),
        "cell": {
            "material_id": np.asarray(cell_payload["material_id"], dtype=engine.material_id.dtype),
            "phase": np.asarray(cell_payload["phase"], dtype=engine.phase.dtype),
            "cell_flags": np.asarray(cell_payload["cell_flags"], dtype=engine.cell_flags.dtype),
            "velocity": np.asarray(cell_payload["velocity"], dtype=engine.velocity.dtype),
            "cell_temperature": np.asarray(cell_payload["cell_temperature"], dtype=engine.cell_temperature.dtype),
            "timer_pack": np.asarray(cell_payload["timer_pack"], dtype=engine.timer_pack.dtype),
            "integrity": np.asarray(cell_payload["integrity"], dtype=engine.integrity.dtype),
            "island_id": np.asarray(cell_payload["island_id"], dtype=engine.island_id.dtype),
            "entity_id": np.asarray(cell_payload["entity_id"], dtype=engine.entity_id.dtype),
            "placeholder_displaced_material": np.asarray(
                cell_payload["placeholder_displaced_material"],
                dtype=engine.placeholder_displaced_material.dtype,
            ),
            "collapse_delay_pending": np.asarray(
                cell_payload["collapse_delay_pending"],
                dtype=np.uint8,
            ),
            "visible_illumination": np.asarray(
                cell_payload["visible_illumination"],
                dtype=engine.visible_illumination.dtype,
            ),
            "cell_optical_dose": np.asarray(
                cell_payload["cell_optical_dose"],
                dtype=engine.cell_optical_dose.dtype,
            ),
        },
        "runtime": runtime_payload,
        "gas": {
            "ambient_temperature": np.asarray(
                gas_payload["ambient_temperature"],
                dtype=engine.ambient_temperature.dtype,
            ),
            "flow_velocity": np.asarray(gas_payload["flow_velocity"], dtype=engine.flow_velocity.dtype),
            "pressure_ping": np.asarray(gas_payload["pressure_ping"], dtype=engine.pressure_ping.dtype),
            "gas_concentration": gas_concentration,
            "gas_optical_dose": np.asarray(
                gas_payload["gas_optical_dose"],
                dtype=engine.gas_optical_dose.dtype,
            ),
        },
    }


def _apply_page_stripe(engine: "WorldEngine", update: PageStripeUpdate, payload: dict[str, Any]) -> None:
    if engine._gpu_pipeline_available(
        engine.page_stripe_pipeline,
        "page stripe",
        require=engine.simulation_backend == "gpu",
    ):
        engine.page_stripe_pipeline.apply(engine, update, payload)
    else:
        engine._require_cpu_oracle_backend("page stripe")
        engine.page_stripe_pipeline.last_backend = "cpu"
        engine.page_stripe_pipeline.last_cpu_mirror_downloaded = True
        _apply_page_stripe_dense_cpu(engine, update, payload)

    engine._normalize_page_stripe_cell_runtime(update)

    runtime_payload = payload.get("runtime")
    engine._merge_island_runtime_payload(runtime_payload, update=update, payload=payload)
    if engine.page_stripe_pipeline.last_cpu_mirror_downloaded:
        engine._queue_loaded_collapse_pending_regions(update)
    else:
        engine._queue_loaded_collapse_pending_regions_from_payload(update, payload)
    engine._mark_loaded_page_stripe_active(update)
    if engine.page_stripe_pipeline.last_cpu_mirror_downloaded:
        engine._rebuild_island_records()
    engine._apply_page_stripe_entity_placeholder_runtime(
        update,
        None
        if runtime_payload is None
        else runtime_payload.get("entity_placeholder_entity_id"),
    )
    engine._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")


def _apply_page_stripe_dense_cpu(engine: "WorldEngine", update: PageStripeUpdate, payload: dict[str, Any]) -> None:
    gas_ranges = _stripe_buffer_ranges(engine, update, gas_grid=True)
    cell_payload = payload["cell"]
    gas_payload = payload["gas"]
    cell_axis = 1 if update.axis == "x" else 0
    cell_dose_axis = 2 if update.axis == "x" else 1

    engine._write_stripe_array(engine.material_id, update, cell_payload["material_id"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.phase, update, cell_payload["phase"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.cell_flags, update, cell_payload["cell_flags"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.velocity, update, cell_payload["velocity"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.cell_temperature, update, cell_payload["cell_temperature"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.timer_pack, update, cell_payload["timer_pack"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.integrity, update, cell_payload["integrity"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.island_id, update, cell_payload["island_id"], stripe_axis=cell_axis)
    engine._write_stripe_array(engine.entity_id, update, cell_payload["entity_id"], stripe_axis=cell_axis)
    engine._write_stripe_array(
        engine.placeholder_displaced_material,
        update,
        cell_payload["placeholder_displaced_material"],
        stripe_axis=cell_axis,
    )
    engine._write_stripe_array(
        engine.collapse_delay_pending,
        update,
        np.asarray(cell_payload["collapse_delay_pending"], dtype=np.bool_),
        stripe_axis=cell_axis,
    )
    engine._write_stripe_array(
        engine.visible_illumination,
        update,
        cell_payload["visible_illumination"],
        stripe_axis=cell_axis,
    )
    engine._write_stripe_array(
        engine.cell_optical_dose,
        update,
        cell_payload["cell_optical_dose"],
        stripe_axis=cell_dose_axis,
    )

    engine._write_stripe_array(
        engine.ambient_temperature,
        update,
        gas_payload["ambient_temperature"],
        stripe_axis=1 if update.axis == "x" else 0,
        ranges=gas_ranges,
    )
    engine._write_stripe_array(
        engine.flow_velocity,
        update,
        gas_payload["flow_velocity"],
        stripe_axis=1 if update.axis == "x" else 0,
        ranges=gas_ranges,
    )
    engine._write_stripe_array(
        engine.pressure_ping,
        update,
        gas_payload["pressure_ping"],
        stripe_axis=1 if update.axis == "x" else 0,
        ranges=gas_ranges,
    )
    engine._write_stripe_array(
        engine.gas_concentration,
        update,
        gas_payload["gas_concentration"],
        stripe_axis=2 if update.axis == "x" else 1,
        ranges=gas_ranges,
    )
    engine._write_stripe_array(
        engine.gas_optical_dose,
        update,
        gas_payload["gas_optical_dose"],
        stripe_axis=2 if update.axis == "x" else 1,
        ranges=gas_ranges,
    )


def _default_page_stripe_payload(engine: "WorldEngine", update: PageStripeUpdate) -> dict[str, Any]:
    cell_span = engine.paging.stripe_span(update)
    gas_span = cell_span // engine.gas_cell_size
    cell_width = cell_span if update.axis == "x" else engine.width
    cell_height = engine.height if update.axis == "x" else cell_span
    gas_width = gas_span if update.axis == "x" else engine.gas_width
    gas_height = engine.gas_height if update.axis == "x" else gas_span
    light_count = engine.cell_optical_dose.shape[0]
    gas_count = engine.gas_concentration.shape[0]
    payload = {
        "meta": {
            "axis": update.axis,
            "world_start": update.world_start,
            "world_end": update.world_end,
            "buffer_start": update.buffer_start,
            "buffer_end": update.buffer_end,
            "kind": "generated",
        },
        "cell": {
            "material_id": np.zeros((cell_height, cell_width), dtype=np.int32),
            "phase": np.zeros((cell_height, cell_width), dtype=np.uint8),
            "cell_flags": np.zeros((cell_height, cell_width), dtype=np.uint8),
            "velocity": np.zeros((cell_height, cell_width, 2), dtype=np.float32),
            "cell_temperature": np.full((cell_height, cell_width), 20.0, dtype=np.float32),
            "timer_pack": np.zeros((cell_height, cell_width, 4), dtype=np.uint8),
            "integrity": np.zeros((cell_height, cell_width), dtype=np.float32),
            "island_id": np.zeros((cell_height, cell_width), dtype=np.int32),
            "entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
            "placeholder_displaced_material": np.zeros((cell_height, cell_width), dtype=np.int32),
            "collapse_delay_pending": np.zeros((cell_height, cell_width), dtype=np.uint8),
            "visible_illumination": np.zeros((cell_height, cell_width, 3), dtype=np.float32),
            "cell_optical_dose": np.zeros((light_count, cell_height, cell_width), dtype=np.float32),
        },
        "runtime": {
            "island_ids": np.zeros((0,), dtype=np.int32),
            "island_velocity": np.zeros((0, 2), dtype=np.float32),
            "island_subcell_offset": np.zeros((0, 2), dtype=np.float32),
            "entity_placeholder_entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
        },
        "gas": {
            "ambient_temperature": np.full((gas_height, gas_width), 20.0, dtype=np.float32),
            "flow_velocity": np.zeros((gas_height, gas_width, 2), dtype=np.float32),
            "pressure_ping": np.zeros((gas_height, gas_width), dtype=np.float32),
            "gas_concentration": np.zeros((gas_count, gas_height, gas_width), dtype=np.float32),
            "gas_optical_dose": np.zeros((light_count, gas_height, gas_width), dtype=np.float32),
        },
    }
    if 0 <= engine.air_gas_species_id < gas_count:
        payload["gas"]["gas_concentration"][engine.air_gas_species_id] = 1.0
    return payload


def _stripe_buffer_ranges(engine: "WorldEngine", update: PageStripeUpdate, *, gas_grid: bool) -> list[tuple[int, int]]:
    if not gas_grid:
        return engine.paging.stripe_buffer_ranges(update)
    cell_span = engine.paging.stripe_span(update)
    if cell_span % engine.gas_cell_size != 0 or update.buffer_start % engine.gas_cell_size != 0:
        raise ValueError("page stripe is not aligned to the gas grid")
    size = engine.gas_width if update.axis == "x" else engine.gas_height
    span = min(size, cell_span // engine.gas_cell_size)
    if span <= 0:
        return []
    start = (update.buffer_start // engine.gas_cell_size) % size
    if span >= size:
        return [(0, size)]
    end = (start + span) % size
    if start < end:
        return [(start, end)]
    ranges = [(start, size)]
    if end > 0:
        ranges.append((0, end))
    return ranges
