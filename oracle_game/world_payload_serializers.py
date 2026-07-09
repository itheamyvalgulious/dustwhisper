from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from copy import deepcopy
from dataclasses import asdict
from oracle_game.gpu import (
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
)
from oracle_game.page_store import StoredStripeKey
from oracle_game.types import (
    CarrierIntent,
    ChangeIntent,
    DebugView,
    EntityFeedback,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityState,
    EntityStatePatch,
    ForceSource,
    ObservationResult,
    ObservationTarget,
    PageStripeUpdate,
    Phase,
    ReadbackRequest,
    ReadbackResult,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    TargetQuery,
    WorldCommand,
    WorldFrameInput,
    WorldFrameOutput,
    WorldFramePreview,
)
from oracle_game.world_constants import ENTITY_STATE_PATCH_METADATA_FIELDS

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

def serialize_pending_commands(engine) -> dict[str, Any]:
    return {
        "pending": len(engine.command_queue),
        "commands": [engine.serialize_world_command(command) for command in engine.command_queue],
    }


def serialize_readback_state(engine) -> dict[str, Any]:
    queued_commands = [
        engine.serialize_world_command(command)
        for command in engine.command_queue
        if command.kind == "request_readback"
    ]
    bridge_runtime = engine.bridge.serialize_runtime_state()
    readback_slots = [
        slot
        for slot in bridge_runtime.get("readback_slots", [])
        if bool(slot.get("occupied", False))
    ]
    return {
        "queued": len(queued_commands),
        "queued_commands": queued_commands,
        "pending": len(engine.pending_readbacks),
        "pending_requests": [engine.serialize_readback_request(request) for request in engine.pending_readbacks],
        "inflight": len(engine.inflight_readbacks),
        "inflight_requests": [engine.serialize_readback_request(request) for request in engine.inflight_readbacks],
        "inflight_slots": readback_slots,
        "readback_latency_frames": bridge_runtime.get("readback_latency_frames", {}),
        "ready": len(engine.completed_readbacks),
    }


def serialize_ready_readbacks(engine) -> dict[str, Any]:
    return {
        "ready": len(engine.completed_readbacks),
        "results": [engine.serialize_readback_result(result) for result in engine.completed_readbacks],
    }


def serialize_frame_state(engine) -> dict[str, Any]:
    pending_submission_ids = engine.pending_frame_submission_ids()
    ready_submission_ids = [
        int(output.submission_id)
        for output in engine.completed_frame_outputs
        if output.submission_id is not None
    ]
    return {
        "pending": len(engine.pending_frame_inputs),
        "pending_submission_ids": pending_submission_ids,
        "ready": len(engine.completed_frame_outputs),
        "ready_submission_ids": ready_submission_ids,
        "canceled_submission_ids": sorted(int(submission_id) for submission_id in engine.canceled_frame_submission_ids),
    }


def serialize_pending_frame_inputs(engine) -> dict[str, Any]:
    return {
        "pending": len(engine.pending_frame_inputs),
        "frames": [engine.serialize_pending_frame_detail(frame_input) for frame_input in engine.pending_frame_inputs],
    }


def serialize_pending_frame_detail(engine, frame_input: WorldFrameInput) -> dict[str, Any]:
    payload = engine.serialize_frame_input(frame_input)
    payload["preview"] = engine.serialize_frame_preview(
        engine.preview_frame_input(
            frame_input,
            reserved_readback_request_ids=set(engine._frame_readback_request_ids(frame_input)),
        )
    )
    return payload


def serialize_ready_frame_outputs(engine) -> dict[str, Any]:
    return {
        "ready": len(engine.completed_frame_outputs),
        "outputs": [engine.serialize_frame_output(output) for output in engine.completed_frame_outputs],
    }


def serialize_page_store_state(engine) -> dict[str, Any]:
    keys = engine.list_page_store_stripe_keys()
    return {
        "stored_stripes": int(engine.page_store.stored_count()),
        "key_listing_supported": keys is not None,
        "stripe_keys": []
        if keys is None
        else [engine.serialize_page_store_key(key) for key in keys],
    }


def serialize_controller_state(engine) -> dict[str, Any]:
    return {"controller_state": deepcopy(engine.controller_state_snapshot)}


def _serialize_preview_bridge_frame_snapshot(
    engine,
    *,
    current_entity_placeholders: dict[int, set[tuple[int, int]]],
    resolved_commands: list[WorldCommand],
    observation_requests: list[ReadbackRequest],
    readback_requests: list[ReadbackRequest],
    placeholder_inputs: list[EntityPlaceholder],
    paging_updates: list[PageStripeUpdate],
    page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]],
    reserved_readback_request_ids: set[int] | None = None,
) -> dict[str, Any]:
    bridge_input_stage = "reserved" if reserved_readback_request_ids is not None else "predicted"
    snapshot_prepared = bool(
        resolved_commands
        or observation_requests
        or readback_requests
        or placeholder_inputs
        or paging_updates
        or page_stripes
    )
    serialized_page_stripes = [
        {
            "update": engine.serialize_page_stripe_update(update),
            "payload": engine.serialize_page_stripe_payload(payload),
        }
        for update, payload in page_stripes
    ]
    return {
        "prepared": snapshot_prepared,
        "commands": [engine.serialize_world_command(command) for command in resolved_commands],
        "command_stages": engine._serialize_bridge_index_stages(
            resolved_commands,
            stage=bridge_input_stage,
        ),
        "readback_requests": [
            engine.serialize_readback_request(request)
            for request in [*observation_requests, *readback_requests]
        ],
        "readback_request_stages": engine._serialize_bridge_readback_request_stages(
            [*observation_requests, *readback_requests],
            reserved_request_ids=reserved_readback_request_ids,
            observation_request_ids={
                int(request.request_id)
                for request in observation_requests
                if request.request_id is not None
            },
        ),
        "placeholders": [
            engine.serialize_entity_placeholder_input(placeholder) for placeholder in placeholder_inputs
        ],
        "placeholder_stages": engine._serialize_bridge_index_stages(
            placeholder_inputs,
            stage=bridge_input_stage,
        ),
        "placeholder_dirty_rects": engine._preview_bridge_placeholder_dirty_rects(
            current_entity_placeholders,
            placeholder_inputs,
        ),
        "paging_updates": [
            engine.serialize_page_stripe_update(update) for update in paging_updates
        ],
        "paging_update_stages": engine._serialize_bridge_index_stages(
            paging_updates,
            stage=bridge_input_stage,
        ),
        "page_stripes": serialized_page_stripes,
        "page_stripe_stages": engine._serialize_bridge_index_stages(
            page_stripes,
            stage=bridge_input_stage,
        ),
    }


def serialize_local_cells(engine, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    world_x0, world_y0, world_x1, world_y1 = engine._clamped_world_window(
        int(x),
        int(y),
        int(width),
        int(height),
    )
    size_x = max(0, world_x1 - world_x0)
    size_y = max(0, world_y1 - world_y0)
    material_id = engine._extract_world_window(engine.material_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    phase = engine._extract_world_window(engine.phase, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    cell_flags = engine._extract_world_window(engine.cell_flags, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    velocity = engine._extract_world_window(engine.velocity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    cell_temperature = engine._extract_world_window(
        engine.cell_temperature,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    timer_pack = engine._extract_world_window(engine.timer_pack, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    integrity = engine._extract_world_window(engine.integrity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    island_id = engine._extract_world_window(engine.island_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    entity_id = engine._extract_world_window(engine.entity_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
    placeholder_displaced_material = engine._extract_world_window(
        engine.placeholder_displaced_material,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    collapse_delay_pending = engine._extract_world_window(
        engine.collapse_delay_pending,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    return {
        "origin": [world_x0, world_y0],
        "size": [size_x, size_y],
        "material_id": material_id.tolist(),
        "phase": phase.tolist(),
        "cell_flags": cell_flags.tolist(),
        "velocity": velocity.round(4).tolist(),
        "cell_temperature": cell_temperature.round(3).tolist(),
        "temperature": cell_temperature.round(3).tolist(),
        "timer_pack": timer_pack.tolist(),
        "integrity": integrity.round(3).tolist(),
        "island_id": island_id.tolist(),
        "entity_id": entity_id.tolist(),
        "placeholder_displaced_material": placeholder_displaced_material.tolist(),
        "collapse_delay_pending": collapse_delay_pending.astype(np.uint8).tolist(),
    }


def serialize_temperature_window(engine, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    world_x0, world_y0, world_x1, world_y1 = engine._clamped_world_window(
        int(x),
        int(y),
        int(width),
        int(height),
    )
    temperature = engine._extract_world_window(
        engine.cell_temperature,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=1,
        y_axis=0,
    )
    return {"temperature": temperature.round(3).tolist()}


def serialize_gas(engine, species: str) -> list[list[float]]:
    species_id = engine._resolve_sanctioned_gas_id(species)
    if species_id < 0:
        raise KeyError(species)
    return engine._extract_world_window(
        engine.gas_concentration[species_id],
        int(engine.paging.origin_x) // int(engine.gas_cell_size),
        int(engine.paging.origin_y) // int(engine.gas_cell_size),
        int(engine.paging.origin_x) // int(engine.gas_cell_size) + int(engine.gas_width),
        int(engine.paging.origin_y) // int(engine.gas_cell_size) + int(engine.gas_height),
        x_axis=1,
        y_axis=0,
        gas_grid=True,
    ).round(4).tolist()


def serialize_pressure(engine) -> list[list[float]]:
    return engine._extract_world_window(
        engine.pressure_ping,
        int(engine.paging.origin_x) // int(engine.gas_cell_size),
        int(engine.paging.origin_y) // int(engine.gas_cell_size),
        int(engine.paging.origin_x) // int(engine.gas_cell_size) + int(engine.gas_width),
        int(engine.paging.origin_y) // int(engine.gas_cell_size) + int(engine.gas_height),
        x_axis=1,
        y_axis=0,
        gas_grid=True,
    ).round(4).tolist()


def serialize_velocity(engine) -> list[list[list[float]]]:
    return engine._extract_world_window(
        engine.flow_velocity,
        int(engine.paging.origin_x) // int(engine.gas_cell_size),
        int(engine.paging.origin_y) // int(engine.gas_cell_size),
        int(engine.paging.origin_x) // int(engine.gas_cell_size) + int(engine.gas_width),
        int(engine.paging.origin_y) // int(engine.gas_cell_size) + int(engine.gas_height),
        x_axis=1,
        y_axis=0,
        gas_grid=True,
    ).round(4).tolist()


def serialize_visible_illumination(engine) -> list[list[list[float]]]:
    return engine._extract_world_window(
        engine.visible_illumination,
        int(engine.paging.origin_x),
        int(engine.paging.origin_y),
        int(engine.paging.origin_x) + int(engine.width),
        int(engine.paging.origin_y) + int(engine.height),
        x_axis=1,
        y_axis=0,
    ).round(4).tolist()


def serialize_material_table(engine) -> list[dict[str, Any]]:
    payload = engine._shadow_material_payload()
    return engine._normalize_json_payload_value(payload)


def _serialize_force_source_record(engine, force_source: ForceSource) -> dict[str, Any]:
    world_x, world_y = engine._force_source_world_position(force_source)
    return {
        "x": float(world_x),
        "y": float(world_y),
        "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
        "radius": float(force_source.radius),
        "strength": float(force_source.strength),
        "lifetime": float(force_source.lifetime),
    }


def serialize_force_sources(engine) -> list[dict[str, Any]]:
    return [engine._serialize_force_source_record(force_source) for force_source in engine.force_sources]


def _serialize_emitter_record(engine, emitter: dict[str, object]) -> dict[str, object]:
    if "world_origin" in emitter:
        world_x, world_y = (int(emitter["world_origin"][0]), int(emitter["world_origin"][1]))
    else:
        world_x, world_y = engine._buffer_to_world_position((int(emitter["origin"][0]), int(emitter["origin"][1])))
    return {
        "x": int(world_x),
        "y": int(world_y),
        "light_type": str(emitter["light_type"]),
        "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
        "spread": float(emitter["spread"]),
        "strength": float(emitter["strength"]),
        "radius": int(emitter["range_cells"]),
    }


def serialize_emitters(engine) -> dict[str, list[dict[str, object]]]:
    return {
        "persistent_emitters": [engine._serialize_emitter_record(emitter) for emitter in engine.persistent_emitters],
        "queued_emitters": [engine._serialize_emitter_record(emitter) for emitter in engine.emitters],
    }


def serialize_gas_species_table(engine) -> list[dict[str, Any]]:
    payload = engine._shadow_gas_species_payload()
    return engine._normalize_json_payload_value(payload)


def serialize_light_type_table(engine) -> list[dict[str, Any]]:
    payload = engine._shadow_light_type_payload()
    return engine._normalize_json_payload_value(payload)


def serialize_material_optics_table(engine) -> list[dict[str, Any]]:
    payload = engine._stable_shadow_payload("optics", engine._material_optics_table_snapshot_payload)
    return engine._normalize_json_payload_value(payload)


def serialize_reaction_table(engine) -> dict[str, object]:
    payload = engine._shadow_reaction_payload()
    return engine._normalize_json_payload_value(payload)


def serialize_optics(
    engine,
    x: int = 0,
    y: int = 0,
    width: int | None = None,
    height: int | None = None,
    *,
    light_type: str | None = None,
) -> dict[str, Any]:
    resolved_width = engine.width if width is None else max(0, int(width))
    resolved_height = engine.height if height is None else max(0, int(height))
    world_x0, world_y0, world_x1, world_y1 = engine._clamped_world_window(
        int(x),
        int(y),
        resolved_width,
        resolved_height,
    )
    gas_world_x0, gas_world_y0, gas_world_x1, gas_world_y1 = engine._world_gas_window_for_cell_world_rect(
        world_x0,
        world_y0,
        world_x1,
        world_y1,
    )
    light_entries: list[tuple[str, int]] = []
    if light_type is None:
        light_entries = [
            (shadow_name, dose_channel)
            for light_id in range(len(engine.light_name_by_id))
            for shadow_name in [engine._shadow_light_name(light_id)]
            for dose_channel in [engine._shadow_light_dose_channel(light_id)]
            if shadow_name
            and dose_channel is not None
            and 0 <= int(dose_channel) < engine.cell_optical_dose.shape[0]
            and 0 <= int(dose_channel) < engine.gas_optical_dose.shape[0]
        ]
    else:
        light_id = engine._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            raise KeyError(light_type)
        dose_channel = engine._shadow_light_dose_channel(light_id)
        if dose_channel is None:
            raise KeyError(light_type)
        if not (0 <= dose_channel < engine.cell_optical_dose.shape[0] and 0 <= dose_channel < engine.gas_optical_dose.shape[0]):
            raise KeyError(light_type)
        shadow_light_name = engine._shadow_light_name(light_id)
        if shadow_light_name is None:
            raise KeyError(light_type)
        light_entries = [(shadow_light_name, dose_channel)]
    return {
        "origin": [world_x0, world_y0],
        "size": [world_x1 - world_x0, world_y1 - world_y0],
        "gas_origin": [gas_world_x0, gas_world_y0],
        "gas_size": [gas_world_x1 - gas_world_x0, gas_world_y1 - gas_world_y0],
        "visible_illumination": engine._extract_world_window(
            engine.visible_illumination,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        .round(4)
        .tolist(),
        "cell_dose": {
            light_name: engine._extract_world_window(
                engine.cell_optical_dose[dose_channel],
                world_x0,
                world_y0,
                world_x1,
                world_y1,
                x_axis=1,
                y_axis=0,
            )
            .round(4)
            .tolist()
            for light_name, dose_channel in light_entries
        },
        "gas_dose": {
            light_name: engine._extract_world_window(
                engine.gas_optical_dose[dose_channel],
                gas_world_x0,
                gas_world_y0,
                gas_world_x1,
                gas_world_y1,
                x_axis=1,
                y_axis=0,
                gas_grid=True,
            )
            .round(4)
            .tolist()
            for light_name, dose_channel in light_entries
        },
    }


def serialize_readback_request(engine, request: ReadbackRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "center_x": int(request.center_x) if request.center_x is not None else None,
        "center_y": int(request.center_y) if request.center_y is not None else None,
        "width": int(request.width),
        "height": int(request.height),
        "channels": list(request.channels),
        "observer_id": request.observer_id,
        "label": request.label,
        "target_query_id": request.target_query_id,
        "target_dx": int(request.target_dx),
        "target_dy": int(request.target_dy),
    }


def _infer_readback_payload_coord_space(
    engine,
    path: tuple[str, ...],
    *,
    resource_name: str | None = None,
) -> str | None:
    if path:
        root = path[0]
        if root == "cell":
            return "world"
        if root in {"ambient_temperature", "pressure", "velocity", "gas"}:
            return "gas"
        if root == "optics":
            if len(path) >= 2 and path[1] in {"visible_illumination", "cell_dose"}:
                return "world"
            if len(path) >= 2 and path[1] == "gas_dose":
                return "gas"
    if resource_name is None:
        return None
    if resource_name in {
        "cell_core",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "collapse_delay_pending",
        "visible_illumination",
        "cell_optical_dose",
    }:
        return "world"
    if resource_name in {
        "ambient_temperature",
        "pressure_ping",
        "flow_velocity",
        "gas_concentration",
        "gas_optical_dose",
    }:
        return "gas"
    return None


def _serialize_readback_source_descriptor(engine, path: tuple[str, ...], value: Any) -> Any:
    if isinstance(value, np.ndarray):
        payload = {
            "source_type": "cpu_array",
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
        }
        coord_space = engine._infer_readback_payload_coord_space(path)
        if coord_space is not None:
            payload["coord_space"] = coord_space
        return payload
    if isinstance(value, GPUBufferReadbackSource):
        payload = {
            "source_type": "buffer_window",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "chunk_size": int(value.chunk_size),
            "start": int(value.start),
            "step": int(value.step),
            "count": int(value.count),
        }
        if value.dst_step is not None:
            payload["dst_step"] = int(value.dst_step)
        coord_space = engine._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
        if coord_space is not None:
            payload["coord_space"] = coord_space
        return payload
    if isinstance(value, GPUCellCoreWindowReadbackSource):
        world_origin_x, world_origin_y = engine._buffer_to_world_position((value.origin_x, value.origin_y))
        payload = {
            "source_type": "cell_core_window",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "coord_space": "world",
            "buffer_origin": [int(value.origin_x), int(value.origin_y)],
            "world_origin": [int(world_origin_x), int(world_origin_y)],
            "cell_grid_width": int(value.cell_grid_width),
        }
        if value.dst_cell_grid_width is not None:
            payload["dst_cell_grid_width"] = int(value.dst_cell_grid_width)
        return payload
    if isinstance(value, GPUGasWindowReadbackSource):
        gas_world_x, gas_world_y = engine._buffer_gas_to_world_position((value.origin_x, value.origin_y))
        payload = {
            "source_type": "gas_window",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "coord_space": "gas",
            "buffer_origin": [int(value.origin_x), int(value.origin_y)],
            "gas_origin": [int(gas_world_x), int(gas_world_y)],
            "gas_grid_size": [int(value.gas_grid_width), int(value.gas_grid_height)],
            "species_id": int(value.species_id),
        }
        if value.dst_step is not None:
            payload["dst_step"] = int(value.dst_step)
        return payload
    if isinstance(value, GPUTextureReadbackSource):
        coord_space = engine._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
        payload = {
            "source_type": "texture_view",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "components": int(value.components),
            "viewport": [int(part) for part in value.viewport],
        }
        if value.dst_step is not None:
            payload["dst_step"] = int(value.dst_step)
        if coord_space is not None:
            payload["coord_space"] = coord_space
            if coord_space == "world":
                world_origin_x, world_origin_y = engine._buffer_to_world_position((value.viewport[0], value.viewport[1]))
                payload["world_origin"] = [int(world_origin_x), int(world_origin_y)]
            elif coord_space == "gas":
                gas_world_x, gas_world_y = engine._buffer_gas_to_world_position((value.viewport[0], value.viewport[1]))
                payload["gas_origin"] = [int(gas_world_x), int(gas_world_y)]
        return payload
    if isinstance(value, GPUSegmentedBufferReadbackSource):
        payload = {
            "source_type": "segmented_buffer_window",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "grid_width": int(value.grid_width),
            "base_offset": int(value.base_offset),
            "segments": [
                {
                    "src": [int(segment.src_x), int(segment.src_y)],
                    "dst": [int(segment.dst_x), int(segment.dst_y)],
                    "size": [int(segment.width), int(segment.height)],
                }
                for segment in value.segments
            ],
        }
        coord_space = engine._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
        if coord_space is not None:
            payload["coord_space"] = coord_space
        return payload
    if isinstance(value, GPUSegmentedCellCoreWindowReadbackSource):
        payload = {
            "source_type": "segmented_cell_core_window",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "coord_space": "world",
            "cell_grid_width": int(value.cell_grid_width),
            "segments": [
                {
                    "src": [int(segment.src_x), int(segment.src_y)],
                    "dst": [int(segment.dst_x), int(segment.dst_y)],
                    "size": [int(segment.width), int(segment.height)],
                }
                for segment in value.segments
            ],
        }
        return payload
    if isinstance(value, GPUSegmentedTextureReadbackSource):
        coord_space = engine._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
        payload = {
            "source_type": "segmented_texture_view",
            "resource_name": str(value.resource_name),
            "dtype": str(np.dtype(value.dtype)),
            "shape": [int(dimension) for dimension in value.shape],
            "components": int(value.components),
            "segments": [
                {
                    "src": [int(segment.src_x), int(segment.src_y)],
                    "dst": [int(segment.dst_x), int(segment.dst_y)],
                    "size": [int(segment.width), int(segment.height)],
                }
                for segment in value.segments
            ],
        }
        if coord_space is not None:
            payload["coord_space"] = coord_space
        return payload
    if isinstance(value, dict):
        return {
            str(key): engine._serialize_readback_source_descriptor(path + (str(key),), child)
            for key, child in value.items()
        }
    return engine._normalize_json_payload_value(value)


def _serialize_readback_plan_for_request(engine, request: ReadbackRequest) -> dict[str, Any]:
    payload = engine._make_readback_payload(request)
    plan = engine.bridge._plan_readback_payload(payload)
    return {
        "request": engine.serialize_readback_request(request),
        "layout": engine.bridge._serialize_readback_layout(plan.layout),
        "nbytes": int(plan.nbytes),
        "gpu_source_count": int(len(plan.gpu_sources)),
        "cpu_chunk_count": int(len(plan.cpu_chunks)),
        "payload": engine._serialize_readback_source_descriptor((), payload),
    }


def _serialize_readback_plans_for_requests(engine, requests: list[ReadbackRequest]) -> list[dict[str, Any]]:
    return [engine._serialize_readback_plan_for_request(request) for request in requests]


def _serialize_observation_plan_for_target_request(
    engine,
    target: ObservationTarget,
    request: ReadbackRequest,
) -> dict[str, Any]:
    return {
        "target": engine.serialize_observation_target(target),
        **engine._serialize_readback_plan_for_request(request),
    }


def serialize_readback_plan(
    engine,
    center_x: int | None,
    center_y: int | None,
    width: int,
    height: int,
    channels: tuple[str, ...],
    *,
    request_id: int | None = None,
    observer_id: int | None = None,
    label: str | None = None,
    target_query_id: str | None = None,
    target_dx: int = 0,
    target_dy: int = 0,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request = engine.preview_readback(
        center_x,
        center_y,
        width,
        height,
        channels,
        request_id=request_id,
        observer_id=observer_id,
        label=label,
        target_query_id=target_query_id,
        target_dx=target_dx,
        target_dy=target_dy,
        target_queries=target_queries,
    )
    return engine._serialize_readback_plan_for_request(request)


def serialize_observation_plan(
    engine,
    target: ObservationTarget | dict[str, Any],
    *,
    request_id: int | None = None,
    target_queries: list[TargetQuery | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target = engine._coerce_observation_target(target)
    request = engine.preview_observation(
        target,
        request_id=request_id,
        target_queries=target_queries,
    )
    return engine._serialize_observation_plan_for_target_request(target, request)


from oracle_game.world_payload_serializers_extra import (
    _serialize_cpu_visible_entity_placeholders,
    _serialize_readback_payload,
    serialize_carrier_intent_input,
    serialize_change_intent_input,
    serialize_consumed_entity_feedback_snapshot,
    serialize_debug_frame,
    serialize_entity_feedback,
    serialize_entity_feedback_snapshot,
    serialize_entity_observation_consume_state,
    serialize_entity_observation_spec,
    serialize_entity_observation_state,
    serialize_entity_placeholder_index_snapshot,
    serialize_entity_placeholder_input,
    serialize_entity_placeholders,
    serialize_entity_state,
    serialize_entity_state_input,
    serialize_entity_state_patch,
    serialize_entity_states,
    serialize_frame_input,
    serialize_frame_output,
    serialize_frame_preview,
    serialize_observation_result,
    serialize_observation_target,
    serialize_page_store_key,
    serialize_page_stripe_payload,
    serialize_page_stripe_update,
    serialize_readback_result,
    serialize_resolved_carrier_intent,
    serialize_resolved_change_intent,
    serialize_resolved_target,
    serialize_target_query_input,
    serialize_world_command,
)
