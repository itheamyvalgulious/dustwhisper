from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import numpy as np

from oracle_game.gpu._common import (
    CPU_READBACK_LATENCY_FRAMES,
    GPU_READBACK_LATENCY_FRAMES,
)

from oracle_game.gpu.readback import (
    GLReadbackSlot,
    ReadbackPayloadLayout,
)


def mark_gpu_authoritative(bridge, *resource_names: str) -> None:
    bridge.gpu_authoritative_resources.update(str(name) for name in resource_names)


def clear_gpu_authoritative(bridge, *resource_names: str) -> None:
    if resource_names:
        for name in resource_names:
            bridge.gpu_authoritative_resources.discard(str(name))
        return
    bridge.gpu_authoritative_resources.clear()


def _should_upload_cpu_resource(bridge, world: "WorldEngine", resource_name: str) -> bool:
    if bridge._force_cpu_resource_upload:
        return True
    return not (
        str(resource_name) in bridge.gpu_authoritative_resources
        and getattr(world, "simulation_backend", "") == "gpu"
    )


def _should_upload_cpu_solver_runtime(world: "WorldEngine") -> bool:
    return getattr(world, "simulation_backend", "") == "cpu"


def _serialize_table_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {
            "kind": "dict",
            "size": len(payload),
            "keys": sorted(str(key) for key in payload.keys()),
        }
    if isinstance(payload, (list, tuple)):
        return {
            "kind": "list",
            "size": len(payload),
        }
    if payload is None:
        return {"kind": "none", "size": 0}
    return {"kind": type(payload).__name__}


def _serialize_ndarray_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }


def _resource_size_bytes(resource: Any) -> int | None:
    size = getattr(resource, "size", None)
    if size is None or isinstance(size, tuple):
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def _serialize_buffer_summary(cls, resource: Any) -> dict[str, Any]:
    return {"size_bytes": cls._resource_size_bytes(resource)}


def _serialize_texture_summary(texture: Any) -> dict[str, Any]:
    size = getattr(texture, "size", None)
    if isinstance(size, tuple):
        size_payload: list[int] | None = [int(value) for value in size]
    elif size is None:
        size_payload = None
    else:
        size_payload = [int(size)]
    components = getattr(texture, "components", None)
    return {
        "size": size_payload,
        "components": None if components is None else int(components),
        "dtype": None if getattr(texture, "dtype", None) is None else str(texture.dtype),
    }


def _serialize_readback_layout(layout: ReadbackPayloadLayout | None) -> dict[str, Any] | None:
    if layout is None:
        return None
    return {
        "metadata_keys": sorted(str(key) for key in layout.metadata.keys()),
        "array_count": len(layout.arrays),
        "arrays": [
            {
                "path": [str(part) for part in array.path],
                "dtype": np.dtype(array.dtype).name,
                "shape": [int(value) for value in array.shape],
                "offset": int(array.offset),
                "nbytes": int(array.nbytes),
            }
            for array in layout.arrays
        ],
    }


def _serialize_readback_slot(cls, slot: GLReadbackSlot) -> dict[str, Any]:
    request = slot.request
    return {
        "slot_index": int(slot.slot_index),
        "occupied": request is not None and slot.frame_id >= 0,
        "frame_id": None if slot.frame_id < 0 else int(slot.frame_id),
        "ready_frame_id": None if slot.ready_frame_id < 0 else int(slot.ready_frame_id),
        "min_poll_frame_id": None if slot.min_poll_frame_id < 0 else int(slot.min_poll_frame_id),
        "latency_frames": int(slot.latency_frames),
        "gpu_backed": bool(slot.gpu_backed),
        "pending_gpu_latency": bool(slot.gpu_backed and slot.min_poll_frame_id > slot.ready_frame_id >= 0),
        "request_id": None if request is None or request.request_id is None else int(request.request_id),
        "observer_id": None if request is None or request.observer_id is None else int(request.observer_id),
        "label": None if request is None or request.label is None else str(request.label),
        "channels": None if request is None else [str(channel) for channel in request.channels],
        "window": None
        if request is None
        else {
            "center_x": None if request.center_x is None else int(request.center_x),
            "center_y": None if request.center_y is None else int(request.center_y),
            "width": int(request.width),
            "height": int(request.height),
        },
        "target_query_id": None if request is None or request.target_query_id is None else str(request.target_query_id),
        "target_dx": None if request is None else int(request.target_dx),
        "target_dy": None if request is None else int(request.target_dy),
        "nbytes": int(slot.nbytes),
        "buffer_size_bytes": cls._resource_size_bytes(slot.buffer),
        "layout": cls._serialize_readback_layout(slot.layout),
    }


def serialize_runtime_state(bridge) -> dict[str, Any]:
    return {
        "enabled": bool(bridge.enabled),
        "has_context": bridge.ctx is not None,
        "own_context": bool(bridge.own_context),
        "world_signature": None
        if bridge.world_signature is None
        else [int(value) for value in bridge.world_signature],
        "rule_table_signature": None
        if bridge.rule_table_signature is None
        else [int(value) for value in bridge.rule_table_signature],
        "atlas_grid": [int(bridge.atlas_grid[0]), int(bridge.atlas_grid[1])],
        "atlas_dirty": bool(bridge.atlas_dirty),
        "write_index": int(bridge.write_index),
        "table_generations": {
            str(name): int(generation)
            for name, generation in sorted(bridge.table_generations.items())
        },
        "shadow_tables": {
            str(name): bridge._serialize_table_summary(payload)
            for name, payload in sorted(bridge.shadow_tables.items())
        },
        "shadow_typed_tables": {
            str(name): bridge._serialize_ndarray_summary(payload)
            for name, payload in sorted(bridge.shadow_typed_tables.items())
        },
        "shadow_buffers": {
            str(name): bridge._serialize_ndarray_summary(payload)
            for name, payload in sorted(bridge.shadow_buffers.items())
        },
        "textures": {
            str(name): bridge._serialize_texture_summary(texture)
            for name, texture in sorted(bridge.textures.items())
        },
        "buffers": {
            str(name): bridge._serialize_buffer_summary(buffer)
            for name, buffer in sorted(bridge.buffers.items())
        },
        "table_buffers": {
            str(name): bridge._serialize_buffer_summary(buffer)
            for name, buffer in sorted(bridge.table_buffers.items())
        },
        "typed_table_buffers": {
            str(name): bridge._serialize_buffer_summary(buffer)
            for name, buffer in sorted(bridge.typed_table_buffers.items())
        },
        "readback_programs": sorted(str(name) for name in bridge.readback_programs.keys()),
        "readback_latency_frames": {
            "cpu_payload": int(CPU_READBACK_LATENCY_FRAMES),
            "gpu_payload": int(GPU_READBACK_LATENCY_FRAMES),
        },
        "readback_slots": [bridge._serialize_readback_slot(slot) for slot in bridge.readback_slots],
    }
