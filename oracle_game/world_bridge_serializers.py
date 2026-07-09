from __future__ import annotations

import json
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

from oracle_game.gpu import PAGE_STRIPE_AXIS_IDS, PAGE_STRIPE_FIELD_PATHS, PAGE_STRIPE_KIND_IDS
from oracle_game.readback_contract import READBACK_CHANNEL_BITS
from oracle_game.types import ReadbackRequest

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def serialize_bridge_runtime(engine: "WorldEngine") -> dict[str, Any]:
    return {
        "frame_id": int(engine.frame_id),
        "bridge": engine.bridge.serialize_runtime_state(),
        "pending_readbacks": len(engine.pending_readbacks),
        "inflight_readbacks": len(engine.inflight_readbacks),
        "ready_readbacks": len(engine.completed_readbacks),
        "pending_commands": len(engine.command_queue),
    }


def _serialize_bridge_resource_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": str(name),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "structured": array.dtype.names is not None,
        "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
        "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
    }


def serialize_bridge_resources(engine: "WorldEngine") -> dict[str, Any]:
    typed_tables = [
        {
            **_serialize_bridge_resource_summary(str(name), table),
            "endpoint": "/api/read/bridge_typed_table",
            "query": {"name": str(name)},
            "response_type": "bridge_typed_table_snapshot",
            "slice_endpoint": "/api/read/bridge_typed_table_slice",
            "slice_query": {"name": str(name), "offset": 0, "limit": 64},
            "slice_response_type": "bridge_typed_table_slice_snapshot",
        }
        for name, table in sorted(engine.bridge.shadow_typed_tables.items())
    ]
    shadow_buffers = []
    for name, buffer in sorted(engine.bridge.shadow_buffers.items()):
        resource = {
            **_serialize_bridge_resource_summary(str(name), buffer),
            "endpoint": "/api/read/bridge_shadow_buffer",
            "query": {"name": str(name)},
            "response_type": "bridge_shadow_buffer_snapshot",
            "slice_endpoint": "/api/read/bridge_shadow_buffer_slice",
            "slice_query": {"name": str(name), "offset": 0, "limit": 64},
            "slice_response_type": "bridge_shadow_buffer_slice_snapshot",
        }
        if buffer.ndim >= 2:
            resource["window_endpoint"] = "/api/read/bridge_shadow_buffer_window"
            resource["window_query"] = {"name": str(name), "x": 0, "y": 0, "w": 16, "h": 16}
            resource["window_response_type"] = "bridge_shadow_buffer_window_snapshot"
            resource["window_axes"] = [int(buffer.ndim - 2), int(buffer.ndim - 1)]
        trailing_shape = tuple(int(value) for value in buffer.shape[-2:]) if buffer.ndim >= 2 else ()
        if trailing_shape == (int(engine.height), int(engine.width)):
            resource["world_window_endpoint"] = "/api/read/bridge_shadow_buffer_world_window"
            resource["world_window_query"] = {"name": str(name), "x": int(engine.paging.origin_x), "y": int(engine.paging.origin_y), "w": 16, "h": 16}
            resource["world_window_response_type"] = "bridge_shadow_buffer_world_window_snapshot"
        if trailing_shape == (int(engine.gas_height), int(engine.gas_width)):
            resource["gas_window_endpoint"] = "/api/read/bridge_shadow_buffer_gas_window"
            resource["gas_window_query"] = {
                "name": str(name),
                "x": int(engine.paging.origin_x) // int(engine.gas_cell_size),
                "y": int(engine.paging.origin_y) // int(engine.gas_cell_size),
                "w": 4,
                "h": 4,
            }
            resource["gas_window_response_type"] = "bridge_shadow_buffer_gas_window_snapshot"
        shadow_buffers.append(resource)
    return {
        "typed_tables": typed_tables,
        "shadow_buffers": shadow_buffers,
        "snapshots": [
            {
                "name": "bridge_runtime",
                "endpoint": "/api/read/bridge_runtime",
                "response_type": "bridge_runtime",
            },
            {
                "name": "bridge_uploads",
                "endpoint": "/api/read/bridge_uploads",
                "response_type": "bridge_upload_snapshot",
            },
            {
                "name": "bridge_frame",
                "endpoint": "/api/read/bridge_frame",
                "response_type": "bridge_frame_snapshot",
            },
        ],
    }


def _serialize_bridge_readback_request_stages(
    requests: list[ReadbackRequest],
    *,
    stage: str | None = None,
    reserved_request_ids: set[int] | None = None,
    observation_request_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    reserved_ids = set() if reserved_request_ids is None else {int(request_id) for request_id in reserved_request_ids}
    observation_ids = set() if observation_request_ids is None else {int(request_id) for request_id in observation_request_ids}
    payload: list[dict[str, Any]] = []
    for request in requests:
        if request.request_id is None:
            continue
        if stage is None:
            current_stage = "predicted"
            if int(request.request_id) in reserved_ids:
                current_stage = "reserved"
            elif int(request.request_id) in observation_ids:
                current_stage = "predicted"
        else:
            current_stage = str(stage)
        payload.append({"request_id": int(request.request_id), "stage": current_stage})
    return payload


def _serialize_bridge_index_stages(
    values: Sequence[Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    return [{"index": int(index), "stage": str(stage)} for index, _ in enumerate(values)]


def _serialize_bridge_ndarray(engine: "WorldEngine", name: str, array: np.ndarray) -> dict[str, Any]:
    if array.dtype.names is not None:
        rows = [
            {
                str(field_name): engine._normalize_json_payload_value(row[field_name])
                for field_name in array.dtype.names
            }
            for row in array
        ]
        return {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": True,
            "field_names": [str(field_name) for field_name in array.dtype.names],
            "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
            "rows": rows,
        }

    utf8: str | None = None
    if array.dtype == np.uint8 and array.ndim == 1:
        try:
            utf8 = bytes(np.ascontiguousarray(array).tolist()).decode("utf-8")
        except UnicodeDecodeError:
            utf8 = None
    return {
        "name": str(name),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "structured": False,
        "field_names": [],
        "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
        "values": engine._normalize_json_payload_value(array),
        "utf8": utf8,
    }


def _serialize_bridge_ndarray_slice(
    engine: "WorldEngine",
    name: str,
    array: np.ndarray,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    start, end = engine._normalize_bridge_slice_bounds(array, offset=offset, limit=limit)
    sliced = array[start:end]
    payload = {
        "name": str(name),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "structured": array.dtype.names is not None,
        "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
        "row_count": engine._bridge_row_count(array),
        "offset": int(start),
        "limit": int(end - start if limit is None else max(0, int(limit))),
        "returned_count": int(end - start),
        "slice_shape": [int(value) for value in sliced.shape],
    }
    if array.dtype.names is not None:
        payload["rows"] = [
            {
                str(field_name): engine._normalize_json_payload_value(row[field_name])
                for field_name in array.dtype.names
            }
            for row in sliced
        ]
        payload["values"] = []
        payload["utf8"] = None
        return payload

    utf8: str | None = None
    if sliced.dtype == np.uint8 and sliced.ndim == 1:
        try:
            utf8 = bytes(np.ascontiguousarray(sliced).tolist()).decode("utf-8")
        except UnicodeDecodeError:
            utf8 = None
    payload["rows"] = []
    payload["values"] = engine._normalize_json_payload_value(sliced)
    payload["utf8"] = utf8
    return payload


def _serialize_bridge_ndarray_window(
    engine: "WorldEngine",
    name: str,
    array: np.ndarray,
    *,
    x: int = 0,
    y: int = 0,
    w: int | None = None,
    h: int | None = None,
) -> dict[str, Any]:
    x0, y0, x1, y1 = engine._normalize_bridge_window_bounds(array, x=x, y=y, w=w, h=h)
    selection = (slice(None),) * max(0, array.ndim - 2) + (slice(y0, y1), slice(x0, x1))
    window = array[selection]
    payload = {
        "name": str(name),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "structured": array.dtype.names is not None,
        "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
        "row_count": engine._bridge_row_count(array),
        "window_origin": [int(x0), int(y0)],
        "requested_size": [max(0, 0 if w is None else int(w)), max(0, 0 if h is None else int(h))],
        "window_size": [int(x1 - x0), int(y1 - y0)],
        "window_axes": [int(array.ndim - 2), int(array.ndim - 1)],
        "returned_shape": [int(value) for value in window.shape],
    }
    if array.dtype.names is not None:
        payload["rows"] = engine._normalize_json_payload_value(window)
        payload["values"] = []
        payload["utf8"] = None
        return payload
    payload["rows"] = []
    payload["values"] = engine._normalize_json_payload_value(window)
    payload["utf8"] = None
    return payload


def _serialize_bridge_spatial_window_payload(
    engine: "WorldEngine",
    name: str,
    array: np.ndarray,
    *,
    coord_space: str,
    window_origin: tuple[int, int],
    requested_size: tuple[int, int],
    window_size: tuple[int, int],
    window: np.ndarray,
) -> dict[str, Any]:
    payload = {
        "name": str(name),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "structured": array.dtype.names is not None,
        "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
        "row_count": engine._bridge_row_count(array),
        "coord_space": str(coord_space),
        "window_origin": [int(window_origin[0]), int(window_origin[1])],
        "requested_size": [int(requested_size[0]), int(requested_size[1])],
        "window_size": [int(window_size[0]), int(window_size[1])],
        "window_axes": [int(array.ndim - 2), int(array.ndim - 1)],
        "returned_shape": [int(value) for value in window.shape],
    }
    if array.dtype.names is not None:
        payload["rows"] = engine._normalize_json_payload_value(window)
        payload["values"] = []
        payload["utf8"] = None
        return payload
    payload["rows"] = []
    payload["values"] = engine._normalize_json_payload_value(window)
    payload["utf8"] = None
    return payload


def serialize_bridge_typed_table(engine: "WorldEngine", name: str) -> dict[str, Any]:
    table = engine.bridge.shadow_typed_tables.get(str(name))
    if table is None:
        raise KeyError(name)
    if table.dtype.names is None:
        raise ValueError(f"bridge typed table '{name}' is not a structured array")
    return _serialize_bridge_ndarray(engine, str(name), table)


def serialize_bridge_typed_table_slice(engine: "WorldEngine", name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    table = engine.bridge.shadow_typed_tables.get(str(name))
    if table is None:
        raise KeyError(name)
    if table.dtype.names is None:
        raise ValueError(f"bridge typed table '{name}' is not a structured array")
    payload = _serialize_bridge_ndarray_slice(engine, str(name), table, offset=offset, limit=limit)
    payload.pop("values", None)
    payload.pop("utf8", None)
    return payload


def serialize_bridge_shadow_buffer(engine: "WorldEngine", name: str) -> dict[str, Any]:
    buffer = engine.bridge.shadow_buffers.get(str(name))
    if buffer is None:
        raise KeyError(name)
    if not isinstance(buffer, np.ndarray):
        raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
    payload = _serialize_bridge_ndarray(engine, str(name), buffer)
    if payload.get("structured"):
        payload.setdefault("values", [])
        payload.setdefault("utf8", None)
    else:
        payload.setdefault("rows", [])
    return payload


def serialize_bridge_shadow_buffer_slice(engine: "WorldEngine", name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    buffer = engine.bridge.shadow_buffers.get(str(name))
    if buffer is None:
        raise KeyError(name)
    if not isinstance(buffer, np.ndarray):
        raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
    return _serialize_bridge_ndarray_slice(engine, str(name), buffer, offset=offset, limit=limit)


def serialize_bridge_shadow_buffer_window(
    engine: "WorldEngine",
    name: str,
    *,
    x: int = 0,
    y: int = 0,
    w: int | None = None,
    h: int | None = None,
) -> dict[str, Any]:
    buffer = engine.bridge.shadow_buffers.get(str(name))
    if buffer is None:
        raise KeyError(name)
    if not isinstance(buffer, np.ndarray):
        raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
    return _serialize_bridge_ndarray_window(engine, str(name), buffer, x=x, y=y, w=w, h=h)


def serialize_bridge_shadow_buffer_world_window(
    engine: "WorldEngine",
    name: str,
    *,
    x: int = 0,
    y: int = 0,
    w: int | None = None,
    h: int | None = None,
) -> dict[str, Any]:
    buffer = engine.bridge.shadow_buffers.get(str(name))
    if buffer is None:
        raise KeyError(name)
    if not isinstance(buffer, np.ndarray):
        raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
    coord_space = engine._bridge_shadow_buffer_coord_space(buffer)
    if coord_space != "world":
        raise ValueError(f"bridge shadow buffer '{name}' does not use world grid coordinates")
    world_x0, world_y0, world_x1, world_y1 = engine._clamped_world_window(int(x), int(y), engine.width if w is None else int(w), engine.height if h is None else int(h))
    window = engine._extract_world_window(
        buffer,
        world_x0,
        world_y0,
        world_x1,
        world_y1,
        x_axis=buffer.ndim - 1,
        y_axis=buffer.ndim - 2,
    )
    return _serialize_bridge_spatial_window_payload(
        engine,
        str(name),
        buffer,
        coord_space="world",
        window_origin=(world_x0, world_y0),
        requested_size=(max(0, engine.width if w is None else int(w)), max(0, engine.height if h is None else int(h))),
        window_size=(world_x1 - world_x0, world_y1 - world_y0),
        window=window,
    )


def serialize_bridge_shadow_buffer_gas_window(
    engine: "WorldEngine",
    name: str,
    *,
    x: int = 0,
    y: int = 0,
    w: int | None = None,
    h: int | None = None,
) -> dict[str, Any]:
    buffer = engine.bridge.shadow_buffers.get(str(name))
    if buffer is None:
        raise KeyError(name)
    if not isinstance(buffer, np.ndarray):
        raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
    coord_space = engine._bridge_shadow_buffer_coord_space(buffer)
    if coord_space != "gas":
        raise ValueError(f"bridge shadow buffer '{name}' does not use gas grid coordinates")
    gas_x0, gas_y0, gas_x1, gas_y1 = engine._clamped_gas_window(int(x), int(y), engine.gas_width if w is None else int(w), engine.gas_height if h is None else int(h))
    window = engine._extract_world_window(
        buffer,
        gas_x0,
        gas_y0,
        gas_x1,
        gas_y1,
        x_axis=buffer.ndim - 1,
        y_axis=buffer.ndim - 2,
        gas_grid=True,
    )
    return _serialize_bridge_spatial_window_payload(
        engine,
        str(name),
        buffer,
        coord_space="gas",
        window_origin=(gas_x0, gas_y0),
        requested_size=(max(0, engine.gas_width if w is None else int(w)), max(0, engine.gas_height if h is None else int(h))),
        window_size=(gas_x1 - gas_x0, gas_y1 - gas_y0),
        window=window,
    )


def _decode_bridge_uploaded_command(meta_record: np.ndarray, payload_bytes: np.ndarray) -> dict[str, Any]:
    start = int(meta_record["payload_offset"])
    end = start + int(meta_record["payload_length"])
    return json.loads(payload_bytes[start:end].tobytes().decode("utf-8"))


def _decode_bridge_uploaded_label(meta_record: np.ndarray, label_bytes: np.ndarray) -> str:
    start = int(meta_record["label_offset"])
    end = start + int(meta_record["label_length"])
    return label_bytes[start:end].tobytes().decode("utf-8")


def _decode_bridge_uploaded_page_stripe_section(section_record: np.ndarray, payload_bytes: np.ndarray) -> np.ndarray:
    dtype_map = {
        1: np.uint8,
        2: np.int32,
        3: np.uint32,
        4: np.float32,
    }
    dtype = dtype_map[int(section_record["dtype_code"])]
    ndim = int(section_record["ndim"])
    shape = tuple(int(section_record[f"dim{axis}"]) for axis in range(ndim))
    start = int(section_record["byte_offset"])
    end = start + int(section_record["byte_length"])
    return np.frombuffer(payload_bytes[start:end].tobytes(), dtype=dtype).reshape(shape)


def serialize_bridge_upload_snapshot(engine: "WorldEngine") -> dict[str, Any]:
    commands_meta = engine.bridge.shadow_buffers.get("world_command")
    commands_payload = engine.bridge.shadow_buffers.get("world_command_payload")
    readback_meta = engine.bridge.shadow_buffers.get("readback_request")
    readback_labels = engine.bridge.shadow_buffers.get("readback_request_label")
    stripe_meta = engine.bridge.shadow_buffers.get("page_stripe_meta")
    stripe_sections = engine.bridge.shadow_buffers.get("page_stripe_section")
    stripe_payload = engine.bridge.shadow_buffers.get("page_stripe_payload")
    frame_meta = engine.bridge.shadow_buffers.get("frame_meta")

    world_commands = []
    if isinstance(commands_meta, np.ndarray) and isinstance(commands_payload, np.ndarray):
        world_commands = [
            engine._normalize_json_payload_value(_decode_bridge_uploaded_command(record, commands_payload))
            for record in commands_meta
        ]

    readback_requests = []
    if isinstance(readback_meta, np.ndarray) and isinstance(readback_labels, np.ndarray):
        for record in readback_meta:
            channels = [
                channel
                for channel, bit in READBACK_CHANNEL_BITS.items()
                if int(record["channels_mask"]) & int(bit)
            ]
            readback_requests.append(
                {
                    "request_id": int(record["request_id"]),
                    "center_x": int(record["center_x"]),
                    "center_y": int(record["center_y"]),
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "channels_mask": int(record["channels_mask"]),
                    "channels": channels,
                    "observer_id": int(record["observer_id"]),
                    "label": _decode_bridge_uploaded_label(record, readback_labels),
                }
            )

    axis_names = {value: key for key, value in PAGE_STRIPE_AXIS_IDS.items()}
    kind_names = {value: key for key, value in PAGE_STRIPE_KIND_IDS.items()}
    field_paths = {field_id: path for field_id, path in PAGE_STRIPE_FIELD_PATHS}
    page_stripes = []
    if (
        isinstance(stripe_meta, np.ndarray)
        and isinstance(stripe_sections, np.ndarray)
        and isinstance(stripe_payload, np.ndarray)
    ):
        for stripe_index, record in enumerate(stripe_meta):
            payload: dict[str, Any] = {}
            start = int(record["section_offset"])
            end = start + int(record["section_count"])
            for section in stripe_sections[start:end]:
                path = field_paths.get(int(section["field_id"]))
                if path is None:
                    continue
                engine._set_nested_payload_value(
                    payload,
                    path,
                    _decode_bridge_uploaded_page_stripe_section(section, stripe_payload),
                )
            serialized_payload = None
            if payload:
                serialized_payload = engine.serialize_page_stripe_payload(payload)
            page_stripes.append(
                {
                    "stripe_index": int(stripe_index),
                    "axis": axis_names.get(int(record["axis_id"]), "unknown"),
                    "kind": kind_names.get(int(record["kind_id"]), "unknown"),
                    "world_start": int(record["world_start"]),
                    "world_end": int(record["world_end"]),
                    "buffer_start": int(record["buffer_start"]),
                    "buffer_end": int(record["buffer_end"]),
                    "section_count": int(record["section_count"]),
                    "payload": serialized_payload,
                }
            )

    frame_meta_rows: list[dict[str, Any]] = []
    if isinstance(frame_meta, np.ndarray) and frame_meta.dtype.names is not None:
        frame_meta_rows = [
            {
                str(field_name): engine._normalize_json_payload_value(row[field_name])
                for field_name in frame_meta.dtype.names
            }
            for row in frame_meta
        ]

    return {
        "frame_meta": frame_meta_rows,
        "world_commands": world_commands,
        "readback_requests": readback_requests,
        "page_stripes": page_stripes,
    }


def serialize_bridge_frame_snapshot(engine: "WorldEngine") -> dict[str, Any]:
    snapshot_prepared = bool(
        engine._bridge_inputs_prepared
        or engine.bridge_frame_commands
        or engine.bridge_frame_readback_requests
        or engine.bridge_frame_placeholders
        or engine.bridge_frame_placeholder_dirty_rects
        or engine.bridge_frame_paging_updates
        or engine.bridge_frame_page_stripes
    )
    placeholder_dirty_rects = []
    for x0, y0, x1, y1 in engine.bridge_frame_placeholder_dirty_rects:
        world_rect = engine._buffer_bbox_to_world_bbox((int(x0), int(y0), int(x1), int(y1)))
        placeholder_dirty_rects.append(
            {
                "buffer_rect": [int(x0), int(y0), int(x1), int(y1)],
                "world_rect": [int(world_rect[0]), int(world_rect[1]), int(world_rect[2]), int(world_rect[3])],
            }
        )

    page_stripes = [
        {
            "update": engine.serialize_page_stripe_update(update),
            "payload": engine.serialize_page_stripe_payload(payload),
        }
        for update, payload in engine.bridge_frame_page_stripes
    ]

    return {
        "prepared": snapshot_prepared,
        "commands": [engine.serialize_world_command(command) for command in engine.bridge_frame_commands],
        "command_stages": _serialize_bridge_index_stages(
            engine.bridge_frame_commands,
            stage="staged",
        ),
        "readback_requests": [
            engine.serialize_readback_request(request) for request in engine.bridge_frame_readback_requests
        ],
        "readback_request_stages": _serialize_bridge_readback_request_stages(
            engine.bridge_frame_readback_requests,
            stage="staged",
        ),
        "placeholders": [
            engine.serialize_entity_placeholder_input(placeholder) for placeholder in engine.bridge_frame_placeholders
        ],
        "placeholder_stages": _serialize_bridge_index_stages(
            engine.bridge_frame_placeholders,
            stage="staged",
        ),
        "placeholder_dirty_rects": placeholder_dirty_rects,
        "paging_updates": [
            engine.serialize_page_stripe_update(update) for update in engine.bridge_frame_paging_updates
        ],
        "paging_update_stages": _serialize_bridge_index_stages(
            engine.bridge_frame_paging_updates,
            stage="staged",
        ),
        "page_stripes": page_stripes,
        "page_stripe_stages": _serialize_bridge_index_stages(
            engine.bridge_frame_page_stripes,
            stage="staged",
        ),
    }
