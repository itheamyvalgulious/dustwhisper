from __future__ import annotations

from copy import deepcopy
import json
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from oracle_game.readback import READBACK_CPU_LATENCY_FRAMES, READBACK_GPU_LATENCY_FRAMES
from oracle_game.readback_contract import READBACK_CHANNEL_BITS
from oracle_game.types import COLLAPSE_BEHAVIOR_IDS, PageStripeUpdate, ReadbackRequest, ReadbackResult, WorldCommand

CPU_READBACK_LATENCY_FRAMES = READBACK_CPU_LATENCY_FRAMES
GPU_READBACK_LATENCY_FRAMES = READBACK_GPU_LATENCY_FRAMES

try:  # pragma: no cover
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


_SHARED_STANDALONE_CONTEXT: Any | None = None
MAX_REACTION_LIGHT_EMITTERS = 256


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{value!r} is not JSON serializable")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, default=_json_default).encode("utf-8")


def _get_shared_standalone_context(*, require: int) -> Any | None:
    global _SHARED_STANDALONE_CONTEXT
    if moderngl is None:
        return None
    if _SHARED_STANDALONE_CONTEXT is None:
        errors: list[Exception] = []
        for kwargs in ({"require": require, "backend": "egl"}, {"require": require}):
            try:
                _SHARED_STANDALONE_CONTEXT = moderngl.create_standalone_context(**kwargs)
                break
            except Exception as exc:
                errors.append(exc)
        if _SHARED_STANDALONE_CONTEXT is None and errors:
            raise errors[-1]
    return _SHARED_STANDALONE_CONTEXT


def _pack_half2x16(velocity: np.ndarray) -> np.ndarray:
    half = velocity.astype(np.float16)
    raw = half.view(np.uint16)
    return (raw[..., 0].astype(np.uint32) | (raw[..., 1].astype(np.uint32) << 16)).astype(np.uint32)


def _unpack_half2x16(word: np.ndarray) -> np.ndarray:
    pair = np.empty(word.shape + (2,), dtype=np.uint16)
    pair[..., 0] = (word & 0xFFFF).astype(np.uint16)
    pair[..., 1] = ((word >> 16) & 0xFFFF).astype(np.uint16)
    return pair.view(np.float16).astype(np.float32)


def _render_group_tile(material: "MaterialDef", tile: int) -> np.ndarray:
    base = np.asarray(material.base_color, dtype=np.float32)
    return np.broadcast_to(np.clip(base, 0.0, 1.0), (tile, tile, 3)).copy()


def pack_cell_core_window(world: "WorldEngine", x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    material_id = world.material_id[y0:y1, x0:x1]
    phase = world.phase[y0:y1, x0:x1]
    cell_flags = world.cell_flags[y0:y1, x0:x1]
    velocity = world.velocity[y0:y1, x0:x1]
    cell_temperature = world.cell_temperature[y0:y1, x0:x1]
    timer_pack = world.timer_pack[y0:y1, x0:x1]
    integrity = world.integrity[y0:y1, x0:x1]

    packed = np.zeros((max(0, y1 - y0), max(0, x1 - x0), 5), dtype=np.uint32)
    packed[..., 0] = (
        material_id.astype(np.uint32)
        | (phase.astype(np.uint32) << 16)
        | (cell_flags.astype(np.uint32) << 24)
    )
    packed[..., 1] = _pack_half2x16(velocity)
    packed[..., 2] = cell_temperature.astype(np.float32).view(np.uint32)
    packed[..., 3] = (
        timer_pack[..., 0].astype(np.uint32)
        | (timer_pack[..., 1].astype(np.uint32) << 8)
        | (timer_pack[..., 2].astype(np.uint32) << 16)
        | (timer_pack[..., 3].astype(np.uint32) << 24)
    )
    packed[..., 4] = np.clip(np.rint(integrity), 0, 65535).astype(np.uint32)
    return packed


def pack_cell_core(world: "WorldEngine") -> np.ndarray:
    return pack_cell_core_window(world, 0, 0, world.width, world.height)


def unpack_cell_core(packed: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "material_id": (packed[..., 0] & 0xFFFF).astype(np.uint16),
        "phase": ((packed[..., 0] >> 16) & 0xFF).astype(np.uint8),
        "cell_flags": ((packed[..., 0] >> 24) & 0xFF).astype(np.uint8),
        "velocity": _unpack_half2x16(packed[..., 1]),
        "cell_temperature": packed[..., 2].view(np.float32),
        "timer_pack": np.stack(
            [
                (packed[..., 3] & 0xFF).astype(np.uint8),
                ((packed[..., 3] >> 8) & 0xFF).astype(np.uint8),
                ((packed[..., 3] >> 16) & 0xFF).astype(np.uint8),
                ((packed[..., 3] >> 24) & 0xFF).astype(np.uint8),
            ],
            axis=-1,
        ),
        "integrity": (packed[..., 4] & 0xFFFF).astype(np.float32),
    }


ENTITY_STATE_DTYPE = np.dtype(
    [
        ("entity_id", "<i4"),
        ("buffer_x", "<i4"),
        ("buffer_y", "<i4"),
        ("world_x", "<i4"),
        ("world_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("placeholder_material_id", "<i4"),
        ("velocity_xy", "<f4", (2,)),
    ]
)


def entity_state_dtype() -> np.dtype:
    return ENTITY_STATE_DTYPE


def pack_entity_state_upload(world: "WorldEngine") -> np.ndarray:
    entities = sorted(world.entity_states.values(), key=lambda entity: entity.entity_id)
    material_table = world.bridge.shadow_typed_tables["material_table"]
    packed = np.zeros((len(entities),), dtype=ENTITY_STATE_DTYPE)
    for index, entity in enumerate(entities):
        buffer_x = int(entity.x)
        buffer_y = int(entity.y)
        world_x, world_y = world.paging.buffer_to_world(buffer_x, buffer_y)
        packed[index]["entity_id"] = entity.entity_id
        packed[index]["buffer_x"] = buffer_x
        packed[index]["buffer_y"] = buffer_y
        packed[index]["world_x"] = int(world_x)
        packed[index]["world_y"] = int(world_y)
        packed[index]["width"] = int(entity.width)
        packed[index]["height"] = int(entity.height)
        packed[index]["placeholder_material_id"] = int(typed_material_id(material_table, entity.placeholder_material))
        packed[index]["velocity_xy"] = np.asarray(entity.velocity_xy, dtype=np.float32)
    return packed


FORCE_SOURCE_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("world_x", "<f4"),
        ("world_y", "<f4"),
        ("direction_xy", "<f4", (2,)),
        ("radius", "<f4"),
        ("strength", "<f4"),
        ("lifetime", "<f4"),
    ]
)


def force_source_dtype() -> np.dtype:
    return FORCE_SOURCE_DTYPE


def pack_force_source_upload(world: "WorldEngine") -> np.ndarray:
    packed = np.zeros((len(world.force_sources),), dtype=FORCE_SOURCE_DTYPE)
    for index, force in enumerate(world.force_sources):
        world_x, world_y = world._force_source_world_position(force)
        buffer_x, buffer_y = world._force_source_buffer_position(force)
        packed[index]["x"] = float(buffer_x)
        packed[index]["y"] = float(buffer_y)
        packed[index]["world_x"] = np.float32(world_x)
        packed[index]["world_y"] = np.float32(world_y)
        packed[index]["direction_xy"] = np.asarray(force.direction, dtype=np.float32)
        packed[index]["radius"] = float(force.radius)
        packed[index]["strength"] = float(force.strength)
        packed[index]["lifetime"] = float(force.lifetime)
    return packed


ISLAND_RUNTIME_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("buffer_bbox", "<i4", (4,)),
        ("world_bbox", "<i4", (4,)),
        ("velocity_xy", "<f4", (2,)),
        ("subcell_offset", "<f4", (2,)),
    ]
)


def island_runtime_dtype() -> np.dtype:
    return ISLAND_RUNTIME_DTYPE


def pack_island_runtime_upload(world: "WorldEngine") -> np.ndarray:
    island_ids = sorted({int(island_id) for island_id in world.islands} | {int(island_id) for island_id in np.unique(world.island_id) if int(island_id) > 0})
    packed = np.zeros((len(island_ids),), dtype=ISLAND_RUNTIME_DTYPE)
    for index, island_id in enumerate(island_ids):
        if island_id <= 0:
            continue
        record = world.islands.get(island_id)
        if record is None:
            coords = np.argwhere(world.island_id == island_id)
            if coords.size == 0:
                continue
            min_y = int(coords[:, 0].min())
            min_x = int(coords[:, 1].min())
            max_y = int(coords[:, 0].max()) + 1
            max_x = int(coords[:, 1].max()) + 1
            velocity_xy = np.mean(world.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32)
            subcell_offset = np.zeros((2,), dtype=np.float32)
        else:
            min_x, min_y, max_x, max_y = (int(value) for value in record.bbox)
            velocity_xy = np.asarray(record.velocity_xy, dtype=np.float32)
            subcell_offset = np.asarray(record.subcell_offset, dtype=np.float32)
        world_x0, world_y0 = world.paging.buffer_to_world(min_x, min_y)
        width = max_x - min_x
        height = max_y - min_y
        packed[index]["island_id"] = island_id
        packed[index]["buffer_bbox"] = np.asarray((min_x, min_y, max_x, max_y), dtype=np.int32)
        packed[index]["world_bbox"] = np.asarray((int(world_x0), int(world_y0), int(world_x0) + width, int(world_y0) + height), dtype=np.int32)
        packed[index]["velocity_xy"] = velocity_xy
        packed[index]["subcell_offset"] = subcell_offset
    return packed


FRAME_META_DTYPE = np.dtype(
    [
        ("frame_id", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("gas_width", "<i4"),
        ("gas_height", "<i4"),
        ("origin_x", "<i4"),
        ("origin_y", "<i4"),
        ("buffer_origin_x", "<i4"),
        ("buffer_origin_y", "<i4"),
        ("active_width", "<i4"),
        ("active_height", "<i4"),
        ("entity_count", "<i4"),
        ("force_source_count", "<i4"),
        ("world_command_count", "<i4"),
        ("readback_request_count", "<i4"),
        ("placeholder_count", "<i4"),
        ("placeholder_dirty_rect_count", "<i4"),
        ("active_tile_count", "<i4"),
        ("active_chunk_count", "<i4"),
        ("page_update_count", "<i4"),
        ("page_stripe_section_count", "<i4"),
    ]
)


def frame_meta_dtype() -> np.dtype:
    return FRAME_META_DTYPE


def pack_frame_meta_upload(
    world: "WorldEngine",
    *,
    entity_count: int,
    force_source_count: int,
    world_command_count: int,
    readback_request_count: int,
    placeholder_count: int,
    placeholder_dirty_rect_count: int,
    active_tile_count: int,
    active_chunk_count: int,
    page_update_count: int,
    page_stripe_section_count: int,
) -> np.ndarray:
    packed = np.zeros((1,), dtype=FRAME_META_DTYPE)
    packed[0]["frame_id"] = int(world.frame_id)
    packed[0]["width"] = int(world.width)
    packed[0]["height"] = int(world.height)
    packed[0]["gas_width"] = int(world.gas_width)
    packed[0]["gas_height"] = int(world.gas_height)
    packed[0]["origin_x"] = int(world.paging.origin_x)
    packed[0]["origin_y"] = int(world.paging.origin_y)
    packed[0]["buffer_origin_x"] = int(world.paging.buffer_origin_x)
    packed[0]["buffer_origin_y"] = int(world.paging.buffer_origin_y)
    packed[0]["active_width"] = int(world.paging.active_width)
    packed[0]["active_height"] = int(world.paging.active_height)
    packed[0]["entity_count"] = int(entity_count)
    packed[0]["force_source_count"] = int(force_source_count)
    packed[0]["world_command_count"] = int(world_command_count)
    packed[0]["readback_request_count"] = int(readback_request_count)
    packed[0]["placeholder_count"] = int(placeholder_count)
    packed[0]["placeholder_dirty_rect_count"] = int(placeholder_dirty_rect_count)
    packed[0]["active_tile_count"] = int(active_tile_count)
    packed[0]["active_chunk_count"] = int(active_chunk_count)
    packed[0]["page_update_count"] = int(page_update_count)
    packed[0]["page_stripe_section_count"] = int(page_stripe_section_count)
    return packed


WORLD_COMMAND_KIND_IDS: dict[str, int] = {
    "inject_material": 1,
    "inject_temperature": 2,
    "inject_force": 3,
    "inject_gas": 4,
    "request_readback": 5,
    "inject_light": 6,
    "advance_paging": 7,
    "apply_page_stripe": 8,
    "sync_entity_placeholders": 9,
    "sync_entity_states": 25,
    "set_force_sources": 10,
    "update_material_table": 11,
    "update_gas_species_table": 12,
    "update_light_type_table": 13,
    "update_material_optics_table": 14,
    "update_reaction_table": 15,
    "replace_reaction_table": 16,
    "patch_material": 17,
    "patch_light": 18,
    "patch_gas": 19,
    "patch_reaction_action": 20,
    "reset_world": 21,
    "inject_velocity": 22,
    "write_material_region": 23,
    "patch_material_optics": 24,
    "patch_reaction_rule": 26,
    "delete_reaction_rule": 27,
    "delete_reaction_action": 28,
}

WORLD_COMMAND_DTYPE = np.dtype(
    [
        ("kind_id", "<i4"),
        ("payload_offset", "<i4"),
        ("payload_length", "<i4"),
    ]
)


def world_command_dtype() -> np.dtype:
    return WORLD_COMMAND_DTYPE


def pack_world_command_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray]:
    commands = world.bridge_frame_commands
    meta = np.zeros((len(commands),), dtype=WORLD_COMMAND_DTYPE)
    payload_chunks: list[bytes] = []
    payload_offset = 0
    for index, command in enumerate(commands):
        payload = _json_bytes({"kind": command.kind, "payload": command.payload})
        meta[index]["kind_id"] = WORLD_COMMAND_KIND_IDS.get(command.kind, 0)
        meta[index]["payload_offset"] = payload_offset
        meta[index]["payload_length"] = len(payload)
        payload_chunks.append(payload)
        payload_offset += len(payload)
    payload = np.frombuffer(b"".join(payload_chunks), dtype=np.uint8).copy() if payload_chunks else np.zeros((0,), dtype=np.uint8)
    return meta, payload


READBACK_REQUEST_DTYPE = np.dtype(
    [
        ("request_id", "<i4"),
        ("center_x", "<i4"),
        ("center_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("channels_mask", "<i4"),
        ("observer_id", "<i4"),
        ("label_offset", "<i4"),
        ("label_length", "<i4"),
    ]
)


def readback_request_dtype() -> np.dtype:
    return READBACK_REQUEST_DTYPE


def pack_readback_request_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray]:
    requests = world.bridge_frame_readback_requests
    meta = np.zeros((len(requests),), dtype=READBACK_REQUEST_DTYPE)
    label_chunks: list[bytes] = []
    label_offset = 0
    for index, request in enumerate(requests):
        channels_mask = 0
        for channel in request.channels:
            channels_mask |= READBACK_CHANNEL_BITS.get(channel, 0)
        label = (request.label or "").encode("utf-8")
        meta[index]["request_id"] = int(request.request_id) if request.request_id is not None else -1
        meta[index]["center_x"] = int(request.center_x)
        meta[index]["center_y"] = int(request.center_y)
        meta[index]["width"] = int(request.width)
        meta[index]["height"] = int(request.height)
        meta[index]["channels_mask"] = int(channels_mask)
        meta[index]["observer_id"] = int(request.observer_id) if request.observer_id is not None else -1
        meta[index]["label_offset"] = label_offset
        meta[index]["label_length"] = len(label)
        label_chunks.append(label)
        label_offset += len(label)
    labels = np.frombuffer(b"".join(label_chunks), dtype=np.uint8).copy() if label_chunks else np.zeros((0,), dtype=np.uint8)
    return meta, labels


PLACEHOLDER_DTYPE = np.dtype(
    [
        ("entity_id", "<i4"),
        ("buffer_x", "<i4"),
        ("buffer_y", "<i4"),
        ("world_x", "<i4"),
        ("world_y", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("material_id", "<i4"),
    ]
)

PLACEHOLDER_DIRTY_RECT_DTYPE = np.dtype(
    [
        ("buffer_x0", "<i4"),
        ("buffer_y0", "<i4"),
        ("buffer_x1", "<i4"),
        ("buffer_y1", "<i4"),
        ("world_x0", "<i4"),
        ("world_y0", "<i4"),
        ("width", "<i4"),
        ("height", "<i4"),
    ]
)

ACTIVE_META_DTYPE = np.dtype(
    [
        ("tile_size", "<i4"),
        ("chunk_tiles", "<i4"),
        ("active_ttl_reset", "<i4"),
        ("tile_width", "<i4"),
        ("tile_height", "<i4"),
        ("chunk_width", "<i4"),
        ("chunk_height", "<i4"),
        ("active_tile_count", "<i4"),
        ("active_chunk_count", "<i4"),
    ]
)

ACTIVE_RECT_DTYPE = np.dtype(
    [
        ("x0", "<i4"),
        ("y0", "<i4"),
        ("x1", "<i4"),
        ("y1", "<i4"),
        ("tile_padding", "<i4"),
    ]
)

GAS_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("pressure_iterations", "<i4"),
        ("force_source_count_before", "<i4"),
        ("force_source_count_after", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("velocity_changed", "<i4"),
        ("ambient_changed", "<i4"),
        ("gas_changed", "<i4"),
        ("pressure_min", "<f4"),
        ("pressure_max", "<f4"),
        ("ambient_min", "<f4"),
        ("ambient_max", "<f4"),
        ("flow_speed_min", "<f4"),
        ("flow_speed_max", "<f4"),
    ]
)

GAS_SPECIES_RUNTIME_DTYPE = np.dtype(
    [
        ("species_id", "<i4"),
        ("total_concentration", "<f4"),
        ("active_concentration", "<f4"),
    ]
)

HEAT_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("ambient_iterations", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("phase_target_count", "<i4"),
        ("boil_target_count", "<i4"),
        ("condense_target_count", "<i4"),
        ("cell_changed", "<i4"),
        ("ambient_changed", "<i4"),
        ("material_changed", "<i4"),
        ("phase_changed", "<i4"),
        ("integrity_changed", "<i4"),
        ("gas_changed", "<i4"),
        ("cell_temperature_min", "<f4"),
        ("cell_temperature_max", "<f4"),
        ("ambient_temperature_min", "<f4"),
        ("ambient_temperature_max", "<f4"),
        ("integrity_min", "<f4"),
        ("integrity_max", "<f4"),
    ]
)

LIQUID_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("solve_tile_count", "<i4"),
        ("post_tile_count", "<i4"),
        ("post_cell_count", "<i4"),
        ("vertical_seam_cell_count", "<i4"),
        ("horizontal_seam_cell_count", "<i4"),
        ("buoyancy_candidate_count", "<i4"),
        ("changed_cell_count", "<i4"),
        ("material_changed", "<i4"),
        ("phase_changed", "<i4"),
        ("velocity_changed", "<i4"),
        ("temperature_changed", "<i4"),
        ("integrity_changed", "<i4"),
        ("placeholder_changed", "<i4"),
        ("pending_placeholder_count_before", "<i4"),
        ("pending_placeholder_count_after", "<i4"),
        ("liquid_cell_count_before", "<i4"),
        ("liquid_cell_count_after", "<i4"),
    ]
)

REACTION_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("changed_cell_count", "<i4"),
        ("changed_gas_count", "<i4"),
        ("ambient_changed_count", "<i4"),
        ("timer_changed_count", "<i4"),
        ("executed_action_count", "<i4"),
        ("emitted_light_count", "<i4"),
        ("emitted_material_count", "<i4"),
        ("emit_light_action_count", "<i4"),
        ("emit_material_action_count", "<i4"),
        ("modify_gas_action_count", "<i4"),
        ("convert_material_action_count", "<i4"),
        ("modify_temperature_action_count", "<i4"),
        ("harm_action_count", "<i4"),
        ("timed_action_count", "<i4"),
        ("self_action_count", "<i4"),
        ("material_material_action_count", "<i4"),
        ("material_gas_action_count", "<i4"),
        ("material_light_action_count", "<i4"),
        ("gas_gas_action_count", "<i4"),
        ("gas_light_action_count", "<i4"),
    ]
)

COLLAPSE_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("dirty_region_count_before", "<i4"),
        ("solve_region_count", "<i4"),
        ("solve_region_cell_count", "<i4"),
        ("structural_cell_count", "<i4"),
        ("support_seed_count", "<i4"),
        ("supported_cell_count", "<i4"),
        ("unsupported_cell_count", "<i4"),
        ("delayed_pending_count", "<i4"),
        ("immune_unsupported_count", "<i4"),
        ("collapsed_cell_count", "<i4"),
        ("collapsed_component_count", "<i4"),
    ]
)

COLLAPSE_COMPONENT_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("bbox", "<i4", (4,)),
        ("cell_count", "<i4"),
    ]
)

OPTICS_RUNTIME_META_DTYPE = np.dtype(
    [
        ("backend_id", "<i4"),
        ("emitter_count", "<i4"),
        ("secondary_branch_count", "<i4"),
        ("solve_tile_count", "<i4"),
        ("solve_cell_count", "<i4"),
        ("solve_gas_count", "<i4"),
        ("visible_changed_count", "<i4"),
        ("cell_dose_changed_count", "<i4"),
        ("gas_dose_changed_count", "<i4"),
        ("visible_energy_total", "<f4"),
        ("cell_dose_total", "<f4"),
        ("gas_dose_total", "<f4"),
    ]
)


def placeholder_dtype() -> np.dtype:
    return PLACEHOLDER_DTYPE


def placeholder_dirty_rect_dtype() -> np.dtype:
    return PLACEHOLDER_DIRTY_RECT_DTYPE


def active_meta_dtype() -> np.dtype:
    return ACTIVE_META_DTYPE


def active_rect_dtype() -> np.dtype:
    return ACTIVE_RECT_DTYPE


def gas_runtime_meta_dtype() -> np.dtype:
    return GAS_RUNTIME_META_DTYPE


def gas_species_runtime_dtype() -> np.dtype:
    return GAS_SPECIES_RUNTIME_DTYPE


def heat_runtime_meta_dtype() -> np.dtype:
    return HEAT_RUNTIME_META_DTYPE


def liquid_runtime_meta_dtype() -> np.dtype:
    return LIQUID_RUNTIME_META_DTYPE


def reaction_runtime_meta_dtype() -> np.dtype:
    return REACTION_RUNTIME_META_DTYPE


def collapse_runtime_meta_dtype() -> np.dtype:
    return COLLAPSE_RUNTIME_META_DTYPE


def collapse_component_dtype() -> np.dtype:
    return COLLAPSE_COMPONENT_DTYPE


def optics_runtime_meta_dtype() -> np.dtype:
    return OPTICS_RUNTIME_META_DTYPE


def pack_placeholder_upload(world: "WorldEngine") -> np.ndarray:
    placeholders = world.bridge_frame_placeholders
    material_table = world.bridge.shadow_typed_tables["material_table"]
    packed = np.zeros((len(placeholders),), dtype=PLACEHOLDER_DTYPE)
    for index, placeholder in enumerate(placeholders):
        if placeholder.world_x is not None and placeholder.world_y is not None:
            world_x = int(placeholder.world_x)
            world_y = int(placeholder.world_y)
        else:
            world_x, world_y = world.paging.buffer_to_world(int(placeholder.x), int(placeholder.y))
        packed[index]["entity_id"] = int(placeholder.entity_id)
        packed[index]["buffer_x"] = int(placeholder.x)
        packed[index]["buffer_y"] = int(placeholder.y)
        packed[index]["world_x"] = int(world_x)
        packed[index]["world_y"] = int(world_y)
        packed[index]["width"] = int(placeholder.width)
        packed[index]["height"] = int(placeholder.height)
        packed[index]["material_id"] = int(typed_material_id(material_table, placeholder.material))
    return packed


def pack_placeholder_dirty_rect_upload(world: "WorldEngine") -> np.ndarray:
    rects = world.bridge_frame_placeholder_dirty_rects
    packed = np.zeros((len(rects),), dtype=PLACEHOLDER_DIRTY_RECT_DTYPE)
    for index, (x0, y0, x1, y1) in enumerate(rects):
        world_x0, world_y0 = world.paging.buffer_to_world(int(x0), int(y0))
        packed[index]["buffer_x0"] = int(x0)
        packed[index]["buffer_y0"] = int(y0)
        packed[index]["buffer_x1"] = int(x1)
        packed[index]["buffer_y1"] = int(y1)
        packed[index]["world_x0"] = int(world_x0)
        packed[index]["world_y0"] = int(world_y0)
        packed[index]["width"] = int(x1 - x0)
        packed[index]["height"] = int(y1 - y0)
    return packed


def pack_active_meta_upload(
    world: "WorldEngine",
    *,
    active_tile_count: int,
    active_chunk_count: int,
) -> np.ndarray:
    meta = np.zeros((1,), dtype=ACTIVE_META_DTYPE)
    meta[0]["tile_size"] = int(world.active.tile_size)
    meta[0]["chunk_tiles"] = int(world.active.chunk_tiles)
    meta[0]["active_ttl_reset"] = int(world.active.active_ttl_reset)
    meta[0]["tile_width"] = int(world.active.tile_width)
    meta[0]["tile_height"] = int(world.active.tile_height)
    meta[0]["chunk_width"] = int(world.active.chunk_width)
    meta[0]["chunk_height"] = int(world.active.chunk_height)
    meta[0]["active_tile_count"] = int(active_tile_count)
    meta[0]["active_chunk_count"] = int(active_chunk_count)
    return meta


def pack_active_upload(world: "WorldEngine") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile_ttl = np.asarray(world.active.active_tile_ttl or [], dtype=np.int32)
    chunk_mask = np.asarray(world.active.active_chunk_mask or [], dtype=np.uint8)
    meta = pack_active_meta_upload(
        world,
        active_tile_count=int(np.count_nonzero(tile_ttl > 0)),
        active_chunk_count=int(np.count_nonzero(chunk_mask > 0)),
    )
    return meta, tile_ttl, chunk_mask


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


PAGE_STRIPE_META_DTYPE = np.dtype(
    [
        ("axis_id", "<i4"),
        ("kind_id", "<i4"),
        ("world_start", "<i4"),
        ("world_end", "<i4"),
        ("buffer_start", "<i4"),
        ("buffer_end", "<i4"),
        ("section_offset", "<i4"),
        ("section_count", "<i4"),
    ]
)

PAGE_STRIPE_SECTION_DTYPE = np.dtype(
    [
        ("stripe_index", "<i4"),
        ("field_id", "<i4"),
        ("dtype_code", "<i4"),
        ("ndim", "<i4"),
        ("dim0", "<i4"),
        ("dim1", "<i4"),
        ("dim2", "<i4"),
        ("byte_offset", "<i8"),
        ("byte_length", "<i8"),
    ]
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


def page_stripe_meta_dtype() -> np.dtype:
    return PAGE_STRIPE_META_DTYPE


def page_stripe_section_dtype() -> np.dtype:
    return PAGE_STRIPE_SECTION_DTYPE


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


@dataclass(slots=True)
class GLReadbackSlot:
    slot_index: int
    buffer: Any | None = None
    frame_id: int = -1
    ready_frame_id: int = -1
    min_poll_frame_id: int = -1
    latency_frames: int = CPU_READBACK_LATENCY_FRAMES
    gpu_backed: bool = False
    request: ReadbackRequest | None = None
    nbytes: int = 0
    layout: "ReadbackPayloadLayout | None" = None


@dataclass(slots=True)
class ReadbackArrayLayout:
    path: tuple[str, ...]
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(slots=True)
class ReadbackPayloadLayout:
    metadata: dict[str, Any] = field(default_factory=dict)
    arrays: list[ReadbackArrayLayout] = field(default_factory=list)


@dataclass(slots=True)
class GPUBufferReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, ...]
    chunk_size: int
    start: int
    step: int
    count: int
    dst_step: int | None = None


@dataclass(slots=True)
class GPUCellCoreWindowReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, int, int]
    cell_grid_width: int
    origin_x: int
    origin_y: int
    dst_cell_grid_width: int | None = None


@dataclass(slots=True)
class GPUGasWindowReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, int]
    gas_grid_width: int
    gas_grid_height: int
    species_id: int
    origin_x: int
    origin_y: int
    dst_step: int | None = None


@dataclass(slots=True)
class GPUTextureReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, ...]
    components: int
    viewport: tuple[int, int, int, int]
    dst_step: int | None = None


@dataclass(slots=True)
class GPUReadbackSegment:
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    width: int
    height: int


@dataclass(slots=True)
class GPUSegmentedBufferReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, ...]
    grid_width: int
    base_offset: int
    segments: tuple[GPUReadbackSegment, ...]


@dataclass(slots=True)
class GPUSegmentedCellCoreWindowReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, int, int]
    cell_grid_width: int
    segments: tuple[GPUReadbackSegment, ...]


@dataclass(slots=True)
class GPUSegmentedTextureReadbackSource:
    resource_name: str
    dtype: str
    shape: tuple[int, ...]
    components: int
    segments: tuple[GPUReadbackSegment, ...]


@dataclass(slots=True)
class ReadbackPayloadPlan:
    layout: ReadbackPayloadLayout
    cpu_chunks: list[tuple[int, bytes]] = field(default_factory=list)
    cpu_chunk_paths: list[tuple[str, ...]] = field(default_factory=list)
    gpu_sources: list[
        tuple[
            int,
            GPUBufferReadbackSource
            | GPUCellCoreWindowReadbackSource
            | GPUGasWindowReadbackSource
            | GPUTextureReadbackSource
            | GPUSegmentedBufferReadbackSource
            | GPUSegmentedCellCoreWindowReadbackSource
            | GPUSegmentedTextureReadbackSource,
        ]
    ] = field(default_factory=list)
    nbytes: int = 0


RENDER_GROUP_IDS = {
    "powder": 1,
    "stone": 2,
    "plant": 3,
    "metal": 4,
    "liquid": 5,
    "special": 6,
    "placeholder": 7,
}
POWDER_SOLVER_KIND_IDS = {"granular": 1, "suspended": 2}
LIQUID_SOLVER_KIND_IDS = {
    "tile_level": 1,
    "columnar": 2,
}
FALLING_ISLAND_BREAK_KIND_IDS = {"shear": 1, "stable": 2}
LIGHT_RENDER_STYLE_IDS = {"diffuse": 1, "holy": 2, "chaos": 3, "magic": 4}
CONSUME_POLICY_IDS = {"none": 0, "lhs": 1, "rhs": 2, "both": 3}
DIRECTION_IDS = {
    "all": 0,
    "random": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "speed": 6,
}

RULE_TABLE_META_DTYPE = np.dtype(
    [
        ("material_count", "<i4"),
        ("gas_count", "<i4"),
        ("light_count", "<i4"),
        ("optics_count", "<i4"),
        ("reaction_action_count", "<i4"),
        ("material_material_rule_count", "<i4"),
        ("material_gas_rule_count", "<i4"),
        ("material_light_rule_count", "<i4"),
        ("gas_gas_rule_count", "<i4"),
        ("gas_light_rule_count", "<i4"),
        ("self_rule_count", "<i4"),
        ("material_generation", "<i4"),
        ("gas_generation", "<i4"),
        ("light_generation", "<i4"),
        ("optics_generation", "<i4"),
        ("reaction_generation", "<i4"),
    ]
)
MATERIAL_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("name_hash", "<u8"),
        ("default_phase", "<i4"),
        ("render_group_id", "<i4"),
        ("base_color", "<f4", (3,)),
        ("density", "<f4"),
        ("gravity_scale", "<f4"),
        ("wind_coupling", "<f4"),
        ("drag_scale", "<f4"),
        ("friction", "<f4"),
        ("elasticity", "<f4"),
        ("max_dda_step", "<i4"),
        ("powder_solver_kind_id", "<i4"),
        ("liquid_solver_kind_id", "<i4"),
        ("falling_island_break_kind_id", "<i4"),
        ("is_structural", "<i4"),
        ("is_support_anchor", "<i4"),
        ("collapse_behavior_id", "<i4"),
        ("collapse_generation_id", "<i4"),
        ("powder_generation_id", "<i4"),
        ("base_integrity", "<f4"),
        ("spawn_temperature", "<f4"),
        ("heat_capacity", "<f4"),
        ("conductivity", "<f4"),
        ("ambient_exchange_rate", "<f4"),
        ("melt_point", "<f4"),
        ("boil_point", "<f4"),
        ("melt_to_material_id", "<i4"),
        ("freeze_to_material_id", "<i4"),
        ("boil_to_gas_species_id", "<i4"),
        ("material_tag_mask", "<u4"),
        ("gas_tag_mask", "<u4"),
        ("light_tag_mask", "<u4"),
        ("reaction_slots", "<i4", (8,)),
    ]
)
GAS_TABLE_DTYPE = np.dtype(
    [
        ("species_id", "<i4"),
        ("name_hash", "<u8"),
        ("color", "<f4", (3,)),
        ("diffusion_rate", "<f4"),
        ("buoyancy", "<f4"),
        ("decay_rate", "<f4"),
        ("temperature_coupling", "<f4"),
        ("condense_point", "<f4"),
        ("condense_to_material_id", "<i4"),
        ("pressure_factor", "<f4"),
        ("density_factor", "<f4"),
        ("material_reaction_tag_mask", "<u4"),
        ("light_reaction_tag_mask", "<u4"),
    ]
)
LIGHT_TABLE_DTYPE = np.dtype(
    [
        ("light_type_id", "<i4"),
        ("name_hash", "<u8"),
        ("color", "<f4", (3,)),
        ("visual_channel", "<i4"),
        ("default_range", "<i4"),
        ("max_bounce", "<i4"),
        ("dose_channel_id", "<i4"),
        ("render_style_id", "<i4"),
    ]
)
OPTICS_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("light_type_id", "<i4"),
        ("absorption", "<f4"),
        ("scattering", "<f4"),
        ("refraction", "<f4"),
    ]
)
REACTION_ACTION_FLAG_RANDOM_TARGET = 1 << 0
REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE = 1 << 1
REACTION_ACTION_TABLE_DTYPE = np.dtype(
    [
        ("reaction_type_id", "<i4"),
        ("target_material_id", "<i4"),
        ("emit_material_id", "<i4"),
        ("light_type_id", "<i4"),
        ("gas_species_id", "<i4"),
        ("duration", "<i4"),
        ("velocity", "<f4", (2,)),
        ("direction_id", "<i4"),
        ("range_cells", "<i4"),
        ("generation", "<i4"),
        ("flags", "<i4"),
        ("speed", "<f4"),
        ("strength", "<f4"),
        ("beam_width", "<f4"),
        ("delta", "<f4"),
        ("value", "<f4"),
        ("harm_per_frame", "<f4"),
        ("integrity_threshold", "<f4"),
    ]
)
PAIR_REACTION_RULE_TABLE_DTYPE = np.dtype(
    [
        ("lhs_material_id", "<i4"),
        ("lhs_gas_id", "<i4"),
        ("rhs_material_id", "<i4"),
        ("rhs_gas_id", "<i4"),
        ("rhs_light_id", "<i4"),
        ("lhs_tag_mask", "<u4"),
        ("rhs_tag_mask", "<u4"),
        ("phase_mask", "<u4"),
        ("consume_policy_id", "<i4"),
        ("result_action", "<i4"),
        ("trigger_slot_index", "<i4"),
        ("min_temperature", "<f4"),
        ("max_temperature", "<f4"),
        ("threshold", "<f4"),
        ("rate", "<f4"),
    ]
)
SELF_REACTION_RULE_TABLE_DTYPE = np.dtype(
    [
        ("material_id", "<i4"),
        ("trigger_slot_index", "<i4"),
        ("phase_mask", "<u4"),
        ("timer_index", "<i4"),
        ("min_temperature", "<f4"),
        ("max_temperature", "<f4"),
        ("integrity_at_most", "<f4"),
        ("integrity_at_least", "<f4"),
    ]
)


def rule_table_meta_dtype() -> np.dtype:
    return RULE_TABLE_META_DTYPE


def material_table_dtype() -> np.dtype:
    return MATERIAL_TABLE_DTYPE


def gas_table_dtype() -> np.dtype:
    return GAS_TABLE_DTYPE


def light_table_dtype() -> np.dtype:
    return LIGHT_TABLE_DTYPE


def optics_table_dtype() -> np.dtype:
    return OPTICS_TABLE_DTYPE


def reaction_action_table_dtype() -> np.dtype:
    return REACTION_ACTION_TABLE_DTYPE


def pair_reaction_rule_table_dtype() -> np.dtype:
    return PAIR_REACTION_RULE_TABLE_DTYPE


def self_reaction_rule_table_dtype() -> np.dtype:
    return SELF_REACTION_RULE_TABLE_DTYPE


def _phase_mask(phases: tuple[Any, ...]) -> int:
    mask = 0
    for phase in phases:
        mask |= 1 << int(phase)
    return mask


def _material_ref(world: "WorldEngine", name: str | None) -> int:
    if not name:
        return 0
    material = world.rulebook.materials_by_name.get(name)
    return 0 if material is None else int(material.material_id)


def _gas_ref(world: "WorldEngine", name: str | None) -> int:
    if not name:
        return -1
    gas = world.rulebook.gases_by_name.get(name)
    return -1 if gas is None else int(gas.species_id)


def _light_ref(world: "WorldEngine", name: str | None) -> int:
    if not name:
        return -1
    light = world.rulebook.lights_by_name.get(name)
    return -1 if light is None else int(light.light_type_id)


def _float_or_nan(value: float | None) -> float:
    return np.float32(np.nan if value is None else value)


def stable_name_hash(name: str) -> np.uint64:
    value = 14695981039346656037
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return np.uint64(value)


def _typed_name_id(table: np.ndarray, name: str, *, id_field: str, default: int) -> int:
    if table.size <= 0:
        return default
    matches = np.nonzero(table["name_hash"] == stable_name_hash(name))[0]
    if matches.size == 0:
        return default
    return int(table[int(matches[0])][id_field])


def typed_material_id(table: np.ndarray, name: str) -> int:
    return _typed_name_id(table, name, id_field="material_id", default=0)


def typed_gas_id(table: np.ndarray, name: str) -> int:
    return _typed_name_id(table, name, id_field="species_id", default=-1)


def typed_light_id(table: np.ndarray, name: str) -> int:
    return _typed_name_id(table, name, id_field="light_type_id", default=-1)


def pack_material_table(world: "WorldEngine") -> np.ndarray:
    material_count = max(world.rulebook.materials_by_id, default=0) + 1
    packed = np.zeros((material_count,), dtype=MATERIAL_TABLE_DTYPE)
    for material_id, material in world.rulebook.materials_by_id.items():
        packed[material_id]["material_id"] = int(material.material_id)
        packed[material_id]["name_hash"] = stable_name_hash(material.name)
        packed[material_id]["default_phase"] = int(material.default_phase)
        packed[material_id]["render_group_id"] = int(RENDER_GROUP_IDS.get(material.render_group, 0))
        packed[material_id]["base_color"] = np.asarray(material.base_color, dtype=np.float32)
        packed[material_id]["density"] = float(material.density)
        packed[material_id]["gravity_scale"] = float(material.gravity_scale)
        packed[material_id]["wind_coupling"] = float(material.wind_coupling)
        packed[material_id]["drag_scale"] = float(material.drag_scale)
        packed[material_id]["friction"] = float(material.friction)
        packed[material_id]["elasticity"] = float(material.elasticity)
        packed[material_id]["max_dda_step"] = int(material.max_dda_step)
        packed[material_id]["powder_solver_kind_id"] = int(POWDER_SOLVER_KIND_IDS.get(material.powder_solver_kind, 0))
        packed[material_id]["liquid_solver_kind_id"] = int(LIQUID_SOLVER_KIND_IDS.get(material.liquid_solver_kind, 0))
        packed[material_id]["falling_island_break_kind_id"] = int(
            FALLING_ISLAND_BREAK_KIND_IDS.get(material.falling_island_break_kind, 0)
        )
        packed[material_id]["is_structural"] = int(bool(material.is_structural))
        packed[material_id]["is_support_anchor"] = int(bool(material.is_support_anchor))
        packed[material_id]["collapse_behavior_id"] = int(COLLAPSE_BEHAVIOR_IDS.get(material.collapse_behavior, 0))
        packed[material_id]["collapse_generation_id"] = _material_ref(world, material.collapse_generation)
        packed[material_id]["powder_generation_id"] = _material_ref(world, material.powder_generation)
        packed[material_id]["base_integrity"] = float(material.base_integrity)
        packed[material_id]["spawn_temperature"] = _float_or_nan(material.spawn_temperature)
        packed[material_id]["heat_capacity"] = float(material.heat_capacity)
        packed[material_id]["conductivity"] = float(material.conductivity)
        packed[material_id]["ambient_exchange_rate"] = float(material.ambient_exchange_rate)
        packed[material_id]["melt_point"] = _float_or_nan(material.melt_point)
        packed[material_id]["boil_point"] = _float_or_nan(material.boil_point)
        packed[material_id]["melt_to_material_id"] = _material_ref(world, material.melt_to_material)
        packed[material_id]["freeze_to_material_id"] = _material_ref(world, material.freeze_to_material)
        packed[material_id]["boil_to_gas_species_id"] = _gas_ref(world, material.boil_to_gas_species)
        packed[material_id]["material_tag_mask"] = np.uint32(material.material_tag_mask)
        packed[material_id]["gas_tag_mask"] = np.uint32(material.gas_tag_mask)
        packed[material_id]["light_tag_mask"] = np.uint32(material.light_tag_mask)
        packed[material_id]["reaction_slots"] = np.asarray(material.reaction_slots, dtype=np.int32)
    return packed


def pack_gas_table(world: "WorldEngine") -> np.ndarray:
    gas_count = max(world.rulebook.gases_by_id, default=-1) + 1
    packed = np.zeros((max(0, gas_count),), dtype=GAS_TABLE_DTYPE)
    for species_id, gas in world.rulebook.gases_by_id.items():
        packed[species_id]["species_id"] = int(gas.species_id)
        packed[species_id]["name_hash"] = stable_name_hash(gas.name)
        packed[species_id]["color"] = np.asarray(gas.color, dtype=np.float32)
        packed[species_id]["diffusion_rate"] = float(gas.diffusion_rate)
        packed[species_id]["buoyancy"] = float(gas.buoyancy)
        packed[species_id]["decay_rate"] = float(gas.decay_rate)
        packed[species_id]["temperature_coupling"] = float(gas.temperature_coupling)
        packed[species_id]["condense_point"] = _float_or_nan(gas.condense_point)
        packed[species_id]["condense_to_material_id"] = _material_ref(world, gas.condense_to_material)
        packed[species_id]["pressure_factor"] = float(gas.pressure_factor)
        packed[species_id]["density_factor"] = float(gas.density_factor)
        packed[species_id]["material_reaction_tag_mask"] = np.uint32(gas.material_reaction_tag_mask)
        packed[species_id]["light_reaction_tag_mask"] = np.uint32(gas.light_reaction_tag_mask)
    return packed


def pack_light_table(world: "WorldEngine") -> np.ndarray:
    light_count = max(world.rulebook.lights_by_id, default=-1) + 1
    packed = np.zeros((max(0, light_count),), dtype=LIGHT_TABLE_DTYPE)
    for light_id, light in world.rulebook.lights_by_id.items():
        packed[light_id]["light_type_id"] = int(light.light_type_id)
        packed[light_id]["name_hash"] = stable_name_hash(light.name)
        packed[light_id]["color"] = np.asarray(light.color, dtype=np.float32)
        packed[light_id]["visual_channel"] = int(light.visual_channel)
        packed[light_id]["default_range"] = int(light.default_range)
        packed[light_id]["max_bounce"] = int(light.max_bounce)
        packed[light_id]["dose_channel_id"] = int(light.dose_channel_id)
        packed[light_id]["render_style_id"] = int(LIGHT_RENDER_STYLE_IDS.get(light.render_style, 0))
    return packed


def pack_optics_table(world: "WorldEngine") -> np.ndarray:
    entries: list[tuple[int, int, float, float, float]] = []
    for material_id, material in sorted(world.rulebook.materials_by_id.items()):
        for light_id, light in sorted(world.rulebook.lights_by_id.items()):
            entry = world.rulebook.optics.get((material.name, light.name))
            if entry is None:
                continue
            entries.append(
                (
                    int(material_id),
                    int(light_id),
                    float(entry.absorption),
                    float(entry.scattering),
                    float(entry.refraction),
                )
            )
    packed = np.zeros((len(entries),), dtype=OPTICS_TABLE_DTYPE)
    for index, (material_id, light_id, absorption, scattering, refraction) in enumerate(entries):
        packed[index]["material_id"] = material_id
        packed[index]["light_type_id"] = light_id
        packed[index]["absorption"] = absorption
        packed[index]["scattering"] = scattering
        packed[index]["refraction"] = refraction
    return packed


def pack_reaction_action_table(world: "WorldEngine") -> np.ndarray:
    actions = world.rulebook.reaction_actions
    packed = np.zeros((len(actions),), dtype=REACTION_ACTION_TABLE_DTYPE)
    for index, action in enumerate(actions):
        flags = 0
        if action.target_material == "__random__":
            flags |= REACTION_ACTION_FLAG_RANDOM_TARGET
        if bool(action.allow_subunit_scale):
            flags |= REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE
        packed[index]["reaction_type_id"] = int(action.reaction_type.value)
        packed[index]["target_material_id"] = _material_ref(world, action.target_material)
        packed[index]["emit_material_id"] = _material_ref(world, action.emit_material)
        packed[index]["light_type_id"] = _light_ref(world, action.light_type)
        packed[index]["gas_species_id"] = _gas_ref(world, action.gas_species)
        packed[index]["duration"] = int(action.duration)
        packed[index]["velocity"] = np.asarray(action.velocity, dtype=np.float32)
        packed[index]["direction_id"] = int(DIRECTION_IDS.get(action.direction.value, 0))
        packed[index]["range_cells"] = int(action.range_cells)
        packed[index]["generation"] = int(action.generation)
        packed[index]["flags"] = flags
        packed[index]["speed"] = float(action.speed)
        packed[index]["strength"] = float(action.strength)
        packed[index]["beam_width"] = float(action.beam_width)
        packed[index]["delta"] = float(action.delta)
        packed[index]["value"] = float(action.value)
        packed[index]["harm_per_frame"] = float(action.harm_per_frame)
        packed[index]["integrity_threshold"] = float(action.integrity_threshold)
    return packed


def _pack_pair_reaction_rules(world: "WorldEngine", rules: list[object]) -> np.ndarray:
    packed = np.zeros((len(rules),), dtype=PAIR_REACTION_RULE_TABLE_DTYPE)
    for index, rule in enumerate(rules):
        packed[index]["lhs_material_id"] = _material_ref(world, rule.lhs_material)
        packed[index]["lhs_gas_id"] = _gas_ref(world, rule.lhs_gas)
        packed[index]["rhs_material_id"] = _material_ref(world, rule.rhs_material)
        packed[index]["rhs_gas_id"] = _gas_ref(world, rule.rhs_gas)
        packed[index]["rhs_light_id"] = _light_ref(world, rule.rhs_light)
        packed[index]["lhs_tag_mask"] = np.uint32(rule.lhs_tag_mask)
        packed[index]["rhs_tag_mask"] = np.uint32(rule.rhs_tag_mask)
        packed[index]["phase_mask"] = np.uint32(_phase_mask(rule.phases))
        packed[index]["consume_policy_id"] = int(CONSUME_POLICY_IDS.get(rule.consume_policy, -1))
        packed[index]["result_action"] = int(rule.result_action)
        packed[index]["trigger_slot_index"] = -1 if rule.trigger_slot_index is None else int(rule.trigger_slot_index)
        packed[index]["min_temperature"] = float(rule.min_temperature)
        packed[index]["max_temperature"] = float(rule.max_temperature)
        packed[index]["threshold"] = float(rule.threshold)
        packed[index]["rate"] = float(rule.rate)
    return packed


def pack_self_reaction_rule_table(world: "WorldEngine") -> np.ndarray:
    rules = world.rulebook.self_rules
    packed = np.zeros((len(rules),), dtype=SELF_REACTION_RULE_TABLE_DTYPE)
    for index, rule in enumerate(rules):
        packed[index]["material_id"] = _material_ref(world, rule.material)
        packed[index]["trigger_slot_index"] = int(rule.trigger_slot_index)
        packed[index]["phase_mask"] = np.uint32(_phase_mask(rule.phases))
        packed[index]["timer_index"] = -1 if rule.timer_index is None else int(rule.timer_index)
        packed[index]["min_temperature"] = float(rule.min_temperature)
        packed[index]["max_temperature"] = float(rule.max_temperature)
        packed[index]["integrity_at_most"] = _float_or_nan(rule.integrity_at_most)
        packed[index]["integrity_at_least"] = _float_or_nan(rule.integrity_at_least)
    return packed


@dataclass(slots=True)
class GPUBridge:
    ctx: Any | None = None
    create_standalone: bool = True
    table_generations: dict[str, int] = field(default_factory=dict)
    shadow_tables: dict[str, Any] = field(default_factory=dict)
    shadow_typed_tables: dict[str, np.ndarray] = field(default_factory=dict)
    shadow_buffers: dict[str, np.ndarray] = field(default_factory=dict)
    textures: dict[str, Any] = field(default_factory=dict)
    buffers: dict[str, Any] = field(default_factory=dict)
    table_buffers: dict[str, Any] = field(default_factory=dict)
    typed_table_buffers: dict[str, Any] = field(default_factory=dict)
    readback_programs: dict[str, Any] = field(default_factory=dict)
    display_programs: dict[str, Any] = field(default_factory=dict)
    active_scheduler_programs: dict[str, Any] = field(default_factory=dict)
    readback_slots: list[GLReadbackSlot] = field(default_factory=lambda: [GLReadbackSlot(0), GLReadbackSlot(1)])
    gpu_authoritative_resources: set[str] = field(default_factory=set)
    write_index: int = 0
    own_context: bool = False
    enabled: bool = False
    owner_thread_id: int | None = None
    _force_cpu_resource_upload: bool = False
    world_signature: tuple[int, int, int, int, int] | None = None
    rule_table_signature: tuple[int, ...] | None = None
    atlas_grid: tuple[int, int] = (1, 1)
    atlas_dirty: bool = True

    def __post_init__(self) -> None:
        if self.ctx is not None:
            self.enabled = True
            self.owner_thread_id = threading.get_ident()
        elif self.create_standalone and moderngl is not None:
            try:
                self.ctx = _get_shared_standalone_context(require=430)
                self.own_context = False
                self.enabled = True
                self.owner_thread_id = threading.get_ident()
            except Exception:
                self.ctx = None
                self.enabled = False
                self.owner_thread_id = None

    def attach_context(self, ctx: Any) -> None:
        if self.own_context and self.ctx is not None:
            self.release()
        else:
            self._release_readback_programs()
            self._release_display_programs()
        self.ctx = ctx
        self.own_context = False
        self.enabled = True
        self.owner_thread_id = threading.get_ident()
        self.world_signature = None
        self.rule_table_signature = None
        self.textures.clear()
        self.buffers.clear()
        self.table_buffers.clear()
        self.typed_table_buffers.clear()
        self.readback_slots = [GLReadbackSlot(0), GLReadbackSlot(1)]
        self._release_active_scheduler_programs()
        self.gpu_authoritative_resources.clear()
        self.write_index = 0
        self.atlas_dirty = True

    def ensure_world_resources(self, world: "WorldEngine") -> None:
        if not self.enabled or self.ctx is None:
            return
        signature = (
            world.width,
            world.height,
            world.gas_width,
            world.gas_height,
            world.gas_concentration.shape[0],
            world.cell_optical_dose.shape[0],
        )
        if signature == self.world_signature:
            return
        self.release_resources()
        self.gpu_authoritative_resources.clear()
        self.world_signature = signature
        self.textures["material"] = self.ctx.texture((world.width, world.height), 1, dtype="f4")
        self.textures["light"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["debug"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["ambient_temperature"] = self.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        self.textures["pressure_ping"] = self.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        self.textures["flow_velocity"] = self.ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        self.textures["visible_illumination"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["liquid_flow_intent"] = self.ctx.texture((world.width, world.height), 2, dtype="f4")
        for texture in self.textures.values():
            texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.buffers["cell_core"] = self.ctx.buffer(reserve=world.width * world.height * 5 * 4, dynamic=True)
        self.buffers["island_id"] = self.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
        self.buffers["entity_id"] = self.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
        self.buffers["placeholder_displaced_material"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_delay_pending"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["gas_concentration"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height * 4),
            dynamic=True,
        )
        self.buffers["cell_optical_dose"] = self.ctx.buffer(
            reserve=max(4, int(np.prod(world.cell_optical_dose.shape, dtype=np.int64)) * 4),
            dynamic=True,
        )
        self.buffers["gas_optical_dose"] = self.ctx.buffer(
            reserve=max(4, int(np.prod(world.gas_optical_dose.shape, dtype=np.int64)) * 4),
            dynamic=True,
        )
        self.buffers["entity_state"] = self.ctx.buffer(reserve=max(4, ENTITY_STATE_DTYPE.itemsize), dynamic=True)
        self.buffers["entity_state_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["force_source"] = self.ctx.buffer(reserve=max(4, FORCE_SOURCE_DTYPE.itemsize), dynamic=True)
        self.buffers["force_source_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_runtime"] = self.ctx.buffer(reserve=max(4, ISLAND_RUNTIME_DTYPE.itemsize), dynamic=True)
        self.buffers["island_runtime_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["powder_reservation"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["powder_reservation_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_reservation"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_reservation_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["world_command"] = self.ctx.buffer(reserve=max(4, WORLD_COMMAND_DTYPE.itemsize), dynamic=True)
        self.buffers["world_command_payload"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["readback_request"] = self.ctx.buffer(reserve=max(4, READBACK_REQUEST_DTYPE.itemsize), dynamic=True)
        self.buffers["readback_request_label"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["placeholder"] = self.ctx.buffer(reserve=max(4, PLACEHOLDER_DTYPE.itemsize), dynamic=True)
        self.buffers["placeholder_dirty_rect"] = self.ctx.buffer(
            reserve=max(4, PLACEHOLDER_DIRTY_RECT_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["active_meta"] = self.ctx.buffer(reserve=max(4, ACTIVE_META_DTYPE.itemsize), dynamic=True)
        self.buffers["active_tile_ttl"] = self.ctx.buffer(reserve=max(4, world.active.tile_width * world.active.tile_height * 4), dynamic=True)
        self.buffers["active_chunk_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.chunk_width * world.active.chunk_height * 4),
            dynamic=True,
        )
        active_chunk_count = max(1, int(world.active.chunk_width * world.active.chunk_height))
        self.buffers["active_chunk_list"] = self.ctx.buffer(reserve=max(8, active_chunk_count * 2 * 4), dynamic=True)
        self.buffers["active_chunk_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["active_chunk_dispatch_args"] = self.ctx.buffer(reserve=3 * 4, dynamic=True)
        self.buffers["gas_runtime_meta"] = self.ctx.buffer(reserve=max(4, GAS_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["gas_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["gas_species_runtime"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * GAS_SPECIES_RUNTIME_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["heat_runtime_meta"] = self.ctx.buffer(reserve=max(4, HEAT_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["heat_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["heat_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["heat_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["heat_phase_target"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
            dynamic=True,
        )
        self.buffers["heat_boil_target"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
            dynamic=True,
        )
        self.buffers["heat_condense_target"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["liquid_runtime_meta"] = self.ctx.buffer(reserve=max(4, LIQUID_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["liquid_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["liquid_post_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["liquid_post_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_vertical_seam_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_horizontal_seam_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_buoyancy_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_changed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, REACTION_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["reaction_timed_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_self_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_material_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_light_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_gas_gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_gas_light_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_changed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_changed_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_ambient_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_timer_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_emitted_light_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_emitted_material_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_light_emitter"] = self.ctx.buffer(
            reserve=MAX_REACTION_LIGHT_EMITTERS * 2 * 4 * 4,
            dynamic=True,
        )
        self.buffers["reaction_light_emitter_count"] = self.ctx.buffer(
            reserve=16 * 4,
            dynamic=True,
        )
        self.buffers["collapse_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, COLLAPSE_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["collapse_solve_region_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_structural_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_support_seed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_supported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_unsupported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_delayed_pending_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_immune_unsupported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_collapsed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_component_label"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_component"] = self.ctx.buffer(
            reserve=max(4, COLLAPSE_COMPONENT_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["optics_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, OPTICS_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["optics_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["optics_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["optics_visible_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_cell_dose_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_gas_dose_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["optics_emitter_origin_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["page_stripe_meta"] = self.ctx.buffer(reserve=max(4, PAGE_STRIPE_META_DTYPE.itemsize), dynamic=True)
        self.buffers["page_stripe_section"] = self.ctx.buffer(reserve=max(4, PAGE_STRIPE_SECTION_DTYPE.itemsize), dynamic=True)
        self.buffers["page_stripe_payload"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["frame_meta"] = self.ctx.buffer(reserve=max(4, FRAME_META_DTYPE.itemsize), dynamic=True)
        self.atlas_dirty = True
        self._ensure_atlas_texture(world)

    def upload_table(self, name: str, payload: Any) -> None:
        data = _json_bytes(payload)
        self.shadow_tables[name] = json.loads(data.decode("utf-8"))
        self.table_generations[name] = self.table_generations.get(name, 0) + 1
        if not self.enabled or self.ctx is None:
            return
        buffer = self.table_buffers.get(name)
        if buffer is None or buffer.size < len(data):
            if buffer is not None:
                buffer.release()
            self.table_buffers[name] = self.ctx.buffer(data, dynamic=True)
        else:
            buffer.orphan(len(data))
            buffer.write(data)
        if name == "materials":
            self.atlas_dirty = True

    def sync_rule_tables(self, world: "WorldEngine") -> None:
        signature = (
            self.table_generations.get("materials", 0),
            self.table_generations.get("gases", 0),
            self.table_generations.get("lights", 0),
            self.table_generations.get("optics", 0),
            self.table_generations.get("reactions", 0),
        )
        buffers_ready = all(
            name in self.typed_table_buffers
            for name in (
                "rule_table_meta",
                "material_table",
                "gas_table",
                "light_table",
                "optics_table",
                "reaction_action_table",
                "material_material_rule_table",
                "material_gas_rule_table",
                "material_light_rule_table",
                "gas_gas_rule_table",
                "gas_light_rule_table",
                "self_rule_table",
            )
        )
        if signature == self.rule_table_signature and self.shadow_typed_tables and ((not self.enabled or self.ctx is None) or buffers_ready):
            return

        material_table = pack_material_table(world)
        gas_table = pack_gas_table(world)
        light_table = pack_light_table(world)
        optics_table = pack_optics_table(world)
        reaction_action_table = pack_reaction_action_table(world)
        material_material_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_material_rules)
        material_gas_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_gas_rules)
        material_light_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_light_rules)
        gas_gas_rule_table = _pack_pair_reaction_rules(world, world.rulebook.gas_gas_rules)
        gas_light_rule_table = _pack_pair_reaction_rules(world, world.rulebook.gas_light_rules)
        self_rule_table = pack_self_reaction_rule_table(world)
        rule_table_meta = np.zeros((1,), dtype=RULE_TABLE_META_DTYPE)
        rule_table_meta[0]["material_count"] = int(material_table.shape[0])
        rule_table_meta[0]["gas_count"] = int(gas_table.shape[0])
        rule_table_meta[0]["light_count"] = int(light_table.shape[0])
        rule_table_meta[0]["optics_count"] = int(optics_table.shape[0])
        rule_table_meta[0]["reaction_action_count"] = int(reaction_action_table.shape[0])
        rule_table_meta[0]["material_material_rule_count"] = int(material_material_rule_table.shape[0])
        rule_table_meta[0]["material_gas_rule_count"] = int(material_gas_rule_table.shape[0])
        rule_table_meta[0]["material_light_rule_count"] = int(material_light_rule_table.shape[0])
        rule_table_meta[0]["gas_gas_rule_count"] = int(gas_gas_rule_table.shape[0])
        rule_table_meta[0]["gas_light_rule_count"] = int(gas_light_rule_table.shape[0])
        rule_table_meta[0]["self_rule_count"] = int(self_rule_table.shape[0])
        rule_table_meta[0]["material_generation"] = int(self.table_generations.get("materials", 0))
        rule_table_meta[0]["gas_generation"] = int(self.table_generations.get("gases", 0))
        rule_table_meta[0]["light_generation"] = int(self.table_generations.get("lights", 0))
        rule_table_meta[0]["optics_generation"] = int(self.table_generations.get("optics", 0))
        rule_table_meta[0]["reaction_generation"] = int(self.table_generations.get("reactions", 0))

        self.shadow_typed_tables["rule_table_meta"] = rule_table_meta.copy()
        self.shadow_typed_tables["material_table"] = material_table.copy()
        self.shadow_typed_tables["gas_table"] = gas_table.copy()
        self.shadow_typed_tables["light_table"] = light_table.copy()
        self.shadow_typed_tables["optics_table"] = optics_table.copy()
        self.shadow_typed_tables["reaction_action_table"] = reaction_action_table.copy()
        self.shadow_typed_tables["material_material_rule_table"] = material_material_rule_table.copy()
        self.shadow_typed_tables["material_gas_rule_table"] = material_gas_rule_table.copy()
        self.shadow_typed_tables["material_light_rule_table"] = material_light_rule_table.copy()
        self.shadow_typed_tables["gas_gas_rule_table"] = gas_gas_rule_table.copy()
        self.shadow_typed_tables["gas_light_rule_table"] = gas_light_rule_table.copy()
        self.shadow_typed_tables["self_rule_table"] = self_rule_table.copy()

        if self.enabled and self.ctx is not None:
            self._write_typed_table_buffer("rule_table_meta", rule_table_meta)
            self._write_typed_table_buffer("material_table", material_table)
            self._write_typed_table_buffer("gas_table", gas_table)
            self._write_typed_table_buffer("light_table", light_table)
            self._write_typed_table_buffer("optics_table", optics_table)
            self._write_typed_table_buffer("reaction_action_table", reaction_action_table)
            self._write_typed_table_buffer("material_material_rule_table", material_material_rule_table)
            self._write_typed_table_buffer("material_gas_rule_table", material_gas_rule_table)
            self._write_typed_table_buffer("material_light_rule_table", material_light_rule_table)
            self._write_typed_table_buffer("gas_gas_rule_table", gas_gas_rule_table)
            self._write_typed_table_buffer("gas_light_rule_table", gas_light_rule_table)
            self._write_typed_table_buffer("self_rule_table", self_rule_table)

        self.rule_table_signature = signature

    def sync_world(
        self,
        world: "WorldEngine",
        *,
        debug_frame: np.ndarray | None = None,
        upload_debug_texture: bool = True,
        force_cpu_resource_upload: bool = False,
    ) -> None:
        previous_force_cpu_resource_upload = self._force_cpu_resource_upload
        self._force_cpu_resource_upload = bool(force_cpu_resource_upload)
        try:
            self._sync_world_impl(world, debug_frame=debug_frame, upload_debug_texture=upload_debug_texture)
        finally:
            self._force_cpu_resource_upload = previous_force_cpu_resource_upload

    def _sync_world_impl(
        self,
        world: "WorldEngine",
        *,
        debug_frame: np.ndarray | None = None,
        upload_debug_texture: bool = True,
    ) -> None:
        self.ensure_world_resources(world)
        self.sync_rule_tables(world)
        upload_solver_runtime_from_cpu = self._should_upload_cpu_solver_runtime(world)
        upload_island_runtime_from_cpu = self._should_upload_cpu_resource(world, "island_runtime")
        upload_powder_reservation_from_cpu = (
            upload_solver_runtime_from_cpu and self._should_upload_cpu_resource(world, "powder_reservation")
        )
        upload_island_reservation_from_cpu = (
            upload_solver_runtime_from_cpu and self._should_upload_cpu_resource(world, "island_reservation")
        )
        entity_state_upload = pack_entity_state_upload(world)
        entity_count_upload = np.array([len(entity_state_upload)], dtype=np.int32)
        force_source_upload = pack_force_source_upload(world)
        force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
        island_runtime_upload = (
            pack_island_runtime_upload(world)
            if upload_island_runtime_from_cpu
            else self.shadow_buffers.get("island_runtime", np.zeros((0,), dtype=ISLAND_RUNTIME_DTYPE))
        )
        island_runtime_count_upload = (
            np.array([len(island_runtime_upload)], dtype=np.int32)
            if upload_island_runtime_from_cpu
            else self.shadow_buffers.get("island_runtime_count", np.zeros((1,), dtype=np.int32))
        )
        motion_runtime = (
            world.motion_solver.runtime_snapshot()
            if upload_powder_reservation_from_cpu or upload_island_reservation_from_cpu
            else None
        )
        powder_reservation_upload = (
            motion_runtime["powder_reservations"]
            if upload_powder_reservation_from_cpu and motion_runtime is not None
            else self.shadow_buffers.get(
                "powder_reservation",
                np.zeros((0,), dtype=getattr(world.motion_solver, "last_powder_reservations").dtype),
            )
        )
        powder_reservation_count_upload = (
            np.array([len(powder_reservation_upload)], dtype=np.int32)
            if upload_powder_reservation_from_cpu
            else self.shadow_buffers.get("powder_reservation_count", np.zeros((1,), dtype=np.int32))
        )
        island_reservation_upload = (
            motion_runtime["island_reservations"]
            if upload_island_reservation_from_cpu and motion_runtime is not None
            else self.shadow_buffers.get(
                "island_reservation",
                np.zeros((0,), dtype=getattr(world.motion_solver, "last_island_reservations").dtype),
            )
        )
        island_reservation_count_upload = (
            np.array([len(island_reservation_upload)], dtype=np.int32)
            if upload_island_reservation_from_cpu
            else self.shadow_buffers.get("island_reservation_count", np.zeros((1,), dtype=np.int32))
        )
        world_command_upload, world_command_payload_upload = pack_world_command_upload(world)
        readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
        placeholder_upload = pack_placeholder_upload(world)
        placeholder_dirty_rect_upload = pack_placeholder_dirty_rect_upload(world)
        upload_active_tile_ttl_from_cpu = self._should_upload_cpu_resource(world, "active_tile_ttl")
        upload_active_chunk_mask_from_cpu = self._should_upload_cpu_resource(world, "active_chunk_mask")
        upload_active_meta_from_cpu = self._should_upload_cpu_resource(world, "active_meta")
        active_tile_ttl_default = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.int32)
        active_chunk_mask_default = np.zeros((world.active.chunk_height, world.active.chunk_width), dtype=np.uint8)
        active_tile_ttl_upload = (
            np.asarray(world.active.active_tile_ttl or [], dtype=np.int32)
            if upload_active_tile_ttl_from_cpu
            else self._shadow_or_default("active_tile_ttl", active_tile_ttl_default)
        )
        active_chunk_mask_upload = (
            np.asarray(world.active.active_chunk_mask or [], dtype=np.uint8)
            if upload_active_chunk_mask_from_cpu
            else self._shadow_or_default("active_chunk_mask", active_chunk_mask_default)
        )
        active_meta_default = pack_active_meta_upload(
            world,
            active_tile_count=int(np.count_nonzero(active_tile_ttl_default > 0)),
            active_chunk_count=int(np.count_nonzero(active_chunk_mask_default > 0)),
        )
        active_meta_upload = (
            pack_active_meta_upload(
                world,
                active_tile_count=int(np.count_nonzero(active_tile_ttl_upload > 0)),
                active_chunk_count=int(np.count_nonzero(active_chunk_mask_upload > 0)),
            )
            if upload_active_meta_from_cpu
            else self._shadow_or_default("active_meta", active_meta_default)
        )
        if upload_solver_runtime_from_cpu:
            gas_runtime_meta_upload, gas_solve_tile_mask_upload, gas_solve_gas_mask_upload, gas_species_runtime_upload = pack_gas_runtime_upload(world)
            (
                heat_runtime_meta_upload,
                heat_solve_tile_mask_upload,
                heat_solve_cell_mask_upload,
                heat_solve_gas_mask_upload,
                heat_phase_target_upload,
                heat_boil_target_upload,
                heat_condense_target_upload,
            ) = pack_heat_runtime_upload(world)
            (
                liquid_runtime_meta_upload,
                liquid_solve_tile_mask_upload,
                liquid_post_tile_mask_upload,
                liquid_post_cell_mask_upload,
                liquid_vertical_seam_mask_upload,
                liquid_horizontal_seam_mask_upload,
                liquid_buoyancy_mask_upload,
                liquid_changed_cell_mask_upload,
            ) = pack_liquid_runtime_upload(world)
            (
                reaction_runtime_meta_upload,
                reaction_timed_solve_tile_mask_upload,
                reaction_self_solve_tile_mask_upload,
                reaction_material_material_solve_tile_mask_upload,
                reaction_material_gas_solve_tile_mask_upload,
                reaction_material_light_solve_tile_mask_upload,
                reaction_gas_gas_solve_tile_mask_upload,
                reaction_gas_light_solve_tile_mask_upload,
                reaction_solve_cell_mask_upload,
                reaction_solve_gas_mask_upload,
                reaction_changed_cell_mask_upload,
                reaction_changed_gas_mask_upload,
                reaction_ambient_changed_mask_upload,
                reaction_timer_changed_mask_upload,
                reaction_emitted_light_mask_upload,
                reaction_emitted_material_mask_upload,
            ) = pack_reaction_runtime_upload(world)
        else:
            gas_runtime_meta_upload = self._shadow_or_default("gas_runtime_meta", np.zeros((1,), dtype=GAS_RUNTIME_META_DTYPE))
            gas_solve_tile_mask_upload = self._shadow_or_default(
                "gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            gas_solve_gas_mask_upload = self._shadow_or_default(
                "gas_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            gas_species_runtime_upload = self._shadow_or_default(
                "gas_species_runtime",
                np.zeros((world.gas_concentration.shape[0],), dtype=GAS_SPECIES_RUNTIME_DTYPE),
            )
            heat_runtime_meta_upload = self._shadow_or_default("heat_runtime_meta", np.zeros((1,), dtype=HEAT_RUNTIME_META_DTYPE))
            heat_solve_tile_mask_upload = self._shadow_or_default(
                "heat_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            heat_solve_cell_mask_upload = self._shadow_or_default(
                "heat_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            heat_solve_gas_mask_upload = self._shadow_or_default(
                "heat_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            heat_phase_target_upload = self._shadow_or_default(
                "heat_phase_target",
                np.zeros((world.height, world.width), dtype=np.int32),
            )
            heat_boil_target_upload = self._shadow_or_default(
                "heat_boil_target",
                np.zeros((world.height, world.width), dtype=np.int32),
            )
            heat_condense_target_upload = self._shadow_or_default(
                "heat_condense_target",
                np.zeros(world.gas_concentration.shape, dtype=np.uint8),
            )
            liquid_runtime_meta_upload = self._shadow_or_default("liquid_runtime_meta", np.zeros((1,), dtype=LIQUID_RUNTIME_META_DTYPE))
            liquid_solve_tile_mask_upload = self._shadow_or_default(
                "liquid_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            liquid_post_tile_mask_upload = self._shadow_or_default(
                "liquid_post_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            liquid_post_cell_mask_upload = self._shadow_or_default(
                "liquid_post_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_vertical_seam_mask_upload = self._shadow_or_default(
                "liquid_vertical_seam_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_horizontal_seam_mask_upload = self._shadow_or_default(
                "liquid_horizontal_seam_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_buoyancy_mask_upload = self._shadow_or_default(
                "liquid_buoyancy_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_changed_cell_mask_upload = self._shadow_or_default(
                "liquid_changed_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_runtime_meta_upload = self._shadow_or_default("reaction_runtime_meta", np.zeros((1,), dtype=REACTION_RUNTIME_META_DTYPE))
            reaction_timed_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_timed_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_self_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_self_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_material_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_material_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_gas_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_light_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_light_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_gas_gas_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_gas_gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_gas_light_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_gas_light_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_solve_cell_mask_upload = self._shadow_or_default(
                "reaction_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_solve_gas_mask_upload = self._shadow_or_default(
                "reaction_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_changed_cell_mask_upload = self._shadow_or_default(
                "reaction_changed_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_changed_gas_mask_upload = self._shadow_or_default(
                "reaction_changed_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_ambient_changed_mask_upload = self._shadow_or_default(
                "reaction_ambient_changed_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_timer_changed_mask_upload = self._shadow_or_default(
                "reaction_timer_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_emitted_light_mask_upload = self._shadow_or_default(
                "reaction_emitted_light_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_emitted_material_mask_upload = self._shadow_or_default(
                "reaction_emitted_material_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
        collapse_mask_resources = (
            "collapse_structural_mask",
            "collapse_support_seed_mask",
            "collapse_supported_mask",
            "collapse_unsupported_mask",
            "collapse_delayed_pending_mask",
            "collapse_immune_unsupported_mask",
            "collapse_collapsed_cell_mask",
        )
        upload_collapse_runtime_from_cpu = upload_solver_runtime_from_cpu and not (
            any(name in self.gpu_authoritative_resources for name in collapse_mask_resources)
        )
        if upload_collapse_runtime_from_cpu:
            (
                collapse_runtime_meta_upload,
                collapse_solve_region_mask_upload,
                collapse_structural_mask_upload,
                collapse_support_seed_mask_upload,
                collapse_supported_mask_upload,
                collapse_unsupported_mask_upload,
                collapse_delayed_pending_mask_upload,
                collapse_immune_unsupported_mask_upload,
                collapse_collapsed_cell_mask_upload,
                collapse_component_upload,
            ) = pack_collapse_runtime_upload(world)
        else:
            cell_zero = np.zeros((world.height, world.width), dtype=np.int32)
            collapse_runtime_meta_upload = self.shadow_buffers.get(
                "collapse_runtime_meta",
                np.zeros((1,), dtype=COLLAPSE_RUNTIME_META_DTYPE),
            )
            collapse_solve_region_mask_upload = self.shadow_buffers.get("collapse_solve_region_mask", cell_zero)
            collapse_structural_mask_upload = self.shadow_buffers.get("collapse_structural_mask", cell_zero)
            collapse_support_seed_mask_upload = self.shadow_buffers.get("collapse_support_seed_mask", cell_zero)
            collapse_supported_mask_upload = self.shadow_buffers.get("collapse_supported_mask", cell_zero)
            collapse_unsupported_mask_upload = self.shadow_buffers.get("collapse_unsupported_mask", cell_zero)
            collapse_delayed_pending_mask_upload = self.shadow_buffers.get("collapse_delayed_pending_mask", cell_zero)
            collapse_immune_unsupported_mask_upload = self.shadow_buffers.get("collapse_immune_unsupported_mask", cell_zero)
            collapse_collapsed_cell_mask_upload = self.shadow_buffers.get("collapse_collapsed_cell_mask", cell_zero)
            collapse_component_upload = self.shadow_buffers.get(
                "collapse_component",
                np.zeros((0,), dtype=COLLAPSE_COMPONENT_DTYPE),
            )
        if upload_solver_runtime_from_cpu:
            (
                optics_runtime_meta_upload,
                optics_solve_tile_mask_upload,
                optics_solve_cell_mask_upload,
                optics_solve_gas_mask_upload,
                optics_visible_changed_mask_upload,
                optics_cell_dose_changed_mask_upload,
                optics_gas_dose_changed_mask_upload,
                optics_emitter_origin_mask_upload,
            ) = pack_optics_runtime_upload(world)
        else:
            optics_runtime_meta_upload = self._shadow_or_default("optics_runtime_meta", np.zeros((1,), dtype=OPTICS_RUNTIME_META_DTYPE))
            optics_solve_tile_mask_upload = self._shadow_or_default(
                "optics_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            optics_solve_cell_mask_upload = self._shadow_or_default(
                "optics_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_solve_gas_mask_upload = self._shadow_or_default(
                "optics_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            optics_visible_changed_mask_upload = self._shadow_or_default(
                "optics_visible_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_cell_dose_changed_mask_upload = self._shadow_or_default(
                "optics_cell_dose_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_gas_dose_changed_mask_upload = self._shadow_or_default(
                "optics_gas_dose_changed_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            optics_emitter_origin_mask_upload = self._shadow_or_default(
                "optics_emitter_origin_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
        page_stripe_meta_upload, page_stripe_section_upload, page_stripe_payload_upload = pack_page_stripe_upload(world)
        frame_meta_upload = pack_frame_meta_upload(
            world,
            entity_count=len(entity_state_upload),
            force_source_count=len(force_source_upload),
            world_command_count=len(world_command_upload),
            readback_request_count=len(readback_request_upload),
            placeholder_count=len(placeholder_upload),
            placeholder_dirty_rect_count=len(placeholder_dirty_rect_upload),
            active_tile_count=int(active_meta_upload[0]["active_tile_count"]),
            active_chunk_count=int(active_meta_upload[0]["active_chunk_count"]),
            page_update_count=len(page_stripe_meta_upload),
            page_stripe_section_count=len(page_stripe_section_upload),
        )
        self.shadow_buffers["entity_state"] = entity_state_upload.copy()
        self.shadow_buffers["entity_state_count"] = entity_count_upload.copy()
        self.shadow_buffers["force_source"] = force_source_upload.copy()
        self.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
        if upload_island_runtime_from_cpu or "island_runtime" not in self.shadow_buffers:
            self.shadow_buffers["island_runtime"] = island_runtime_upload.copy()
            self.shadow_buffers["island_runtime_count"] = island_runtime_count_upload.copy()
        if upload_powder_reservation_from_cpu or "powder_reservation" not in self.shadow_buffers:
            self.shadow_buffers["powder_reservation"] = powder_reservation_upload.copy()
            self.shadow_buffers["powder_reservation_count"] = powder_reservation_count_upload.copy()
        if upload_island_reservation_from_cpu or "island_reservation" not in self.shadow_buffers:
            self.shadow_buffers["island_reservation"] = island_reservation_upload.copy()
            self.shadow_buffers["island_reservation_count"] = island_reservation_count_upload.copy()
        self.shadow_buffers["world_command"] = world_command_upload.copy()
        self.shadow_buffers["world_command_payload"] = world_command_payload_upload.copy()
        self.shadow_buffers["readback_request"] = readback_request_upload.copy()
        self.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
        self.shadow_buffers["placeholder"] = placeholder_upload.copy()
        self.shadow_buffers["placeholder_dirty_rect"] = placeholder_dirty_rect_upload.copy()
        self.shadow_buffers["island_id"] = world.island_id.astype(np.int32).copy()
        self.shadow_buffers["entity_id"] = world.entity_id.astype(np.int32).copy()
        self.shadow_buffers["placeholder_displaced_material"] = world.placeholder_displaced_material.astype(np.int32).copy()
        self.shadow_buffers["collapse_delay_pending"] = world.collapse_delay_pending.astype(np.int32).copy()
        self.shadow_buffers["cell_optical_dose"] = world.cell_optical_dose.astype(np.float32).copy()
        self.shadow_buffers["gas_optical_dose"] = world.gas_optical_dose.astype(np.float32).copy()
        self.shadow_buffers["active_meta"] = active_meta_upload.copy()
        self.shadow_buffers["active_tile_ttl"] = active_tile_ttl_upload.copy()
        self.shadow_buffers["active_chunk_mask"] = active_chunk_mask_upload.copy()
        self.shadow_buffers["gas_runtime_meta"] = gas_runtime_meta_upload.copy()
        self.shadow_buffers["gas_solve_tile_mask"] = gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["gas_solve_gas_mask"] = gas_solve_gas_mask_upload.copy()
        self.shadow_buffers["gas_species_runtime"] = gas_species_runtime_upload.copy()
        self.shadow_buffers["heat_runtime_meta"] = heat_runtime_meta_upload.copy()
        self.shadow_buffers["heat_solve_tile_mask"] = heat_solve_tile_mask_upload.copy()
        self.shadow_buffers["heat_solve_cell_mask"] = heat_solve_cell_mask_upload.copy()
        self.shadow_buffers["heat_solve_gas_mask"] = heat_solve_gas_mask_upload.copy()
        self.shadow_buffers["heat_phase_target"] = heat_phase_target_upload.copy()
        self.shadow_buffers["heat_boil_target"] = heat_boil_target_upload.copy()
        self.shadow_buffers["heat_condense_target"] = heat_condense_target_upload.copy()
        self.shadow_buffers["liquid_runtime_meta"] = liquid_runtime_meta_upload.copy()
        self.shadow_buffers["liquid_solve_tile_mask"] = liquid_solve_tile_mask_upload.copy()
        self.shadow_buffers["liquid_post_tile_mask"] = liquid_post_tile_mask_upload.copy()
        self.shadow_buffers["liquid_post_cell_mask"] = liquid_post_cell_mask_upload.copy()
        self.shadow_buffers["liquid_vertical_seam_mask"] = liquid_vertical_seam_mask_upload.copy()
        self.shadow_buffers["liquid_horizontal_seam_mask"] = liquid_horizontal_seam_mask_upload.copy()
        self.shadow_buffers["liquid_buoyancy_mask"] = liquid_buoyancy_mask_upload.copy()
        self.shadow_buffers["liquid_changed_cell_mask"] = liquid_changed_cell_mask_upload.copy()
        self.shadow_buffers["reaction_runtime_meta"] = reaction_runtime_meta_upload.copy()
        self.shadow_buffers["reaction_timed_solve_tile_mask"] = reaction_timed_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_self_solve_tile_mask"] = reaction_self_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_material_solve_tile_mask"] = reaction_material_material_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_gas_solve_tile_mask"] = reaction_material_gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_light_solve_tile_mask"] = reaction_material_light_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_gas_gas_solve_tile_mask"] = reaction_gas_gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_gas_light_solve_tile_mask"] = reaction_gas_light_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_solve_cell_mask"] = reaction_solve_cell_mask_upload.copy()
        self.shadow_buffers["reaction_solve_gas_mask"] = reaction_solve_gas_mask_upload.copy()
        self.shadow_buffers["reaction_changed_cell_mask"] = reaction_changed_cell_mask_upload.copy()
        self.shadow_buffers["reaction_changed_gas_mask"] = reaction_changed_gas_mask_upload.copy()
        self.shadow_buffers["reaction_ambient_changed_mask"] = reaction_ambient_changed_mask_upload.copy()
        self.shadow_buffers["reaction_timer_changed_mask"] = reaction_timer_changed_mask_upload.copy()
        self.shadow_buffers["reaction_emitted_light_mask"] = reaction_emitted_light_mask_upload.copy()
        self.shadow_buffers["reaction_emitted_material_mask"] = reaction_emitted_material_mask_upload.copy()
        self.shadow_buffers["collapse_runtime_meta"] = collapse_runtime_meta_upload.copy()
        self.shadow_buffers["collapse_solve_region_mask"] = collapse_solve_region_mask_upload.copy()
        self.shadow_buffers["collapse_structural_mask"] = collapse_structural_mask_upload.copy()
        self.shadow_buffers["collapse_support_seed_mask"] = collapse_support_seed_mask_upload.copy()
        self.shadow_buffers["collapse_supported_mask"] = collapse_supported_mask_upload.copy()
        self.shadow_buffers["collapse_unsupported_mask"] = collapse_unsupported_mask_upload.copy()
        self.shadow_buffers["collapse_delayed_pending_mask"] = collapse_delayed_pending_mask_upload.copy()
        self.shadow_buffers["collapse_immune_unsupported_mask"] = collapse_immune_unsupported_mask_upload.copy()
        self.shadow_buffers["collapse_collapsed_cell_mask"] = collapse_collapsed_cell_mask_upload.copy()
        self.shadow_buffers["collapse_component"] = collapse_component_upload.copy()
        self.shadow_buffers["optics_runtime_meta"] = optics_runtime_meta_upload.copy()
        self.shadow_buffers["optics_solve_tile_mask"] = optics_solve_tile_mask_upload.copy()
        self.shadow_buffers["optics_solve_cell_mask"] = optics_solve_cell_mask_upload.copy()
        self.shadow_buffers["optics_solve_gas_mask"] = optics_solve_gas_mask_upload.copy()
        self.shadow_buffers["optics_visible_changed_mask"] = optics_visible_changed_mask_upload.copy()
        self.shadow_buffers["optics_cell_dose_changed_mask"] = optics_cell_dose_changed_mask_upload.copy()
        self.shadow_buffers["optics_gas_dose_changed_mask"] = optics_gas_dose_changed_mask_upload.copy()
        self.shadow_buffers["optics_emitter_origin_mask"] = optics_emitter_origin_mask_upload.copy()
        self.shadow_buffers["page_stripe_meta"] = page_stripe_meta_upload.copy()
        self.shadow_buffers["page_stripe_section"] = page_stripe_section_upload.copy()
        self.shadow_buffers["page_stripe_payload"] = page_stripe_payload_upload.copy()
        self.shadow_buffers["frame_meta"] = frame_meta_upload.copy()
        if not self.enabled or self.ctx is None:
            return
        self._ensure_atlas_texture(world)
        upload_cell_dose_from_cpu = self._should_upload_cpu_resource(world, "cell_optical_dose")
        upload_gas_dose_from_cpu = self._should_upload_cpu_resource(world, "gas_optical_dose")
        upload_light_from_cpu = self._should_upload_cpu_resource(world, "light")
        upload_visible_from_cpu = self._should_upload_cpu_resource(world, "visible_illumination")
        if upload_cell_dose_from_cpu or upload_gas_dose_from_cpu or upload_light_from_cpu or upload_visible_from_cpu:
            world._gpu_optics_outputs_clear = False
        if self._should_upload_cpu_resource(world, "cell_core"):
            packed = pack_cell_core(world)
            self.buffers["cell_core"].write(packed.tobytes())
        if self._should_upload_cpu_resource(world, "island_id"):
            self.buffers["island_id"].write(np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes())
        if self._should_upload_cpu_resource(world, "entity_id"):
            self.buffers["entity_id"].write(np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes())
        if self._should_upload_cpu_resource(world, "placeholder_displaced_material"):
            self.buffers["placeholder_displaced_material"].write(
                np.ascontiguousarray(world.placeholder_displaced_material.astype(np.int32)).tobytes()
            )
        if self._should_upload_cpu_resource(world, "collapse_delay_pending"):
            self.buffers["collapse_delay_pending"].write(
                np.ascontiguousarray(world.collapse_delay_pending.astype(np.int32)).tobytes()
            )
        if self._should_upload_cpu_resource(world, "gas_concentration"):
            self.buffers["gas_concentration"].write(world.gas_concentration.astype("f4").tobytes())
        if upload_cell_dose_from_cpu:
            self.buffers["cell_optical_dose"].write(np.ascontiguousarray(world.cell_optical_dose.astype(np.float32)).tobytes())
        if upload_gas_dose_from_cpu:
            self.buffers["gas_optical_dose"].write(np.ascontiguousarray(world.gas_optical_dose.astype(np.float32)).tobytes())
        self._write_dynamic_buffer("entity_state", entity_state_upload)
        self.buffers["entity_state_count"].write(entity_count_upload.tobytes())
        self._write_dynamic_buffer("force_source", force_source_upload)
        self.buffers["force_source_count"].write(force_source_count_upload.tobytes())
        if upload_island_runtime_from_cpu:
            self._write_dynamic_buffer("island_runtime", island_runtime_upload)
            self.buffers["island_runtime_count"].write(island_runtime_count_upload.tobytes())
        if upload_powder_reservation_from_cpu:
            self._write_dynamic_buffer("powder_reservation", powder_reservation_upload)
            self.buffers["powder_reservation_count"].write(powder_reservation_count_upload.tobytes())
        if upload_island_reservation_from_cpu:
            self._write_dynamic_buffer("island_reservation", island_reservation_upload)
            self.buffers["island_reservation_count"].write(island_reservation_count_upload.tobytes())
        self._write_dynamic_buffer("world_command", world_command_upload)
        self._write_dynamic_buffer("world_command_payload", world_command_payload_upload)
        self._write_dynamic_buffer("readback_request", readback_request_upload)
        self._write_dynamic_buffer("readback_request_label", readback_request_label_upload)
        self._write_dynamic_buffer("placeholder", placeholder_upload)
        self._write_dynamic_buffer("placeholder_dirty_rect", placeholder_dirty_rect_upload)
        if self._should_upload_cpu_resource(world, "active_meta"):
            self._write_dynamic_buffer("active_meta", active_meta_upload)
        if self._should_upload_cpu_resource(world, "active_tile_ttl"):
            self._write_dynamic_buffer("active_tile_ttl", active_tile_ttl_upload)
        if self._should_upload_cpu_resource(world, "active_chunk_mask"):
            self._write_dynamic_buffer("active_chunk_mask", active_chunk_mask_upload.astype(np.int32, copy=False))
        if (
            getattr(world, "simulation_backend", "") == "gpu"
            and (
                upload_active_meta_from_cpu
                or upload_active_tile_ttl_from_cpu
                or upload_active_chunk_mask_from_cpu
            )
        ):
            self._ensure_active_scheduler_programs()
            self._refresh_active_chunks_and_meta(world, read_meta=False)
        self._write_dynamic_buffer("gas_runtime_meta", gas_runtime_meta_upload)
        self._write_dynamic_buffer("gas_solve_tile_mask", gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("gas_solve_gas_mask", gas_solve_gas_mask_upload)
        self._write_dynamic_buffer("gas_species_runtime", gas_species_runtime_upload)
        self._write_dynamic_buffer("heat_runtime_meta", heat_runtime_meta_upload)
        self._write_dynamic_buffer("heat_solve_tile_mask", heat_solve_tile_mask_upload)
        self._write_dynamic_buffer("heat_solve_cell_mask", heat_solve_cell_mask_upload)
        self._write_dynamic_buffer("heat_solve_gas_mask", heat_solve_gas_mask_upload)
        self._write_dynamic_buffer("heat_phase_target", heat_phase_target_upload)
        self._write_dynamic_buffer("heat_boil_target", heat_boil_target_upload)
        self._write_dynamic_buffer("heat_condense_target", heat_condense_target_upload)
        self._write_dynamic_buffer("liquid_runtime_meta", liquid_runtime_meta_upload)
        self._write_dynamic_buffer("liquid_solve_tile_mask", liquid_solve_tile_mask_upload)
        self._write_dynamic_buffer("liquid_post_tile_mask", liquid_post_tile_mask_upload)
        self._write_dynamic_buffer("liquid_post_cell_mask", liquid_post_cell_mask_upload)
        self._write_dynamic_buffer("liquid_vertical_seam_mask", liquid_vertical_seam_mask_upload)
        self._write_dynamic_buffer("liquid_horizontal_seam_mask", liquid_horizontal_seam_mask_upload)
        self._write_dynamic_buffer("liquid_buoyancy_mask", liquid_buoyancy_mask_upload)
        self._write_dynamic_buffer("liquid_changed_cell_mask", liquid_changed_cell_mask_upload)
        self._write_dynamic_buffer("reaction_runtime_meta", reaction_runtime_meta_upload)
        self._write_dynamic_buffer("reaction_timed_solve_tile_mask", reaction_timed_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_self_solve_tile_mask", reaction_self_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_material_solve_tile_mask", reaction_material_material_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_gas_solve_tile_mask", reaction_material_gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_light_solve_tile_mask", reaction_material_light_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_gas_gas_solve_tile_mask", reaction_gas_gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_gas_light_solve_tile_mask", reaction_gas_light_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_solve_cell_mask", reaction_solve_cell_mask_upload)
        self._write_dynamic_buffer("reaction_solve_gas_mask", reaction_solve_gas_mask_upload)
        self._write_dynamic_buffer("reaction_changed_cell_mask", reaction_changed_cell_mask_upload)
        self._write_dynamic_buffer("reaction_changed_gas_mask", reaction_changed_gas_mask_upload)
        self._write_dynamic_buffer("reaction_ambient_changed_mask", reaction_ambient_changed_mask_upload)
        self._write_dynamic_buffer("reaction_timer_changed_mask", reaction_timer_changed_mask_upload)
        self._write_dynamic_buffer("reaction_emitted_light_mask", reaction_emitted_light_mask_upload)
        self._write_dynamic_buffer("reaction_emitted_material_mask", reaction_emitted_material_mask_upload)
        self._write_dynamic_buffer("collapse_runtime_meta", collapse_runtime_meta_upload)
        self._write_dynamic_buffer("collapse_solve_region_mask", collapse_solve_region_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_structural_mask"):
            self._write_dynamic_buffer("collapse_structural_mask", collapse_structural_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_support_seed_mask"):
            self._write_dynamic_buffer("collapse_support_seed_mask", collapse_support_seed_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_supported_mask"):
            self._write_dynamic_buffer("collapse_supported_mask", collapse_supported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_unsupported_mask"):
            self._write_dynamic_buffer("collapse_unsupported_mask", collapse_unsupported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_delayed_pending_mask"):
            self._write_dynamic_buffer("collapse_delayed_pending_mask", collapse_delayed_pending_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_immune_unsupported_mask"):
            self._write_dynamic_buffer("collapse_immune_unsupported_mask", collapse_immune_unsupported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_collapsed_cell_mask"):
            self._write_dynamic_buffer("collapse_collapsed_cell_mask", collapse_collapsed_cell_mask_upload)
        self._write_dynamic_buffer("collapse_component", collapse_component_upload)
        self._write_dynamic_buffer("optics_runtime_meta", optics_runtime_meta_upload)
        self._write_dynamic_buffer("optics_solve_tile_mask", optics_solve_tile_mask_upload)
        self._write_dynamic_buffer("optics_solve_cell_mask", optics_solve_cell_mask_upload)
        self._write_dynamic_buffer("optics_solve_gas_mask", optics_solve_gas_mask_upload)
        self._write_dynamic_buffer("optics_visible_changed_mask", optics_visible_changed_mask_upload)
        self._write_dynamic_buffer("optics_cell_dose_changed_mask", optics_cell_dose_changed_mask_upload)
        self._write_dynamic_buffer("optics_gas_dose_changed_mask", optics_gas_dose_changed_mask_upload)
        self._write_dynamic_buffer("optics_emitter_origin_mask", optics_emitter_origin_mask_upload)
        self._write_dynamic_buffer("page_stripe_meta", page_stripe_meta_upload)
        self._write_dynamic_buffer("page_stripe_section", page_stripe_section_upload)
        self._write_dynamic_buffer("page_stripe_payload", page_stripe_payload_upload)
        self.buffers["frame_meta"].write(frame_meta_upload.tobytes())
        if self._should_upload_cpu_resource(world, "material"):
            self.textures["material"].write(world.material_id.astype("f4").tobytes())
        if upload_light_from_cpu:
            light_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            light_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
            light_rgba[..., 3] = 1.0
            self.textures["light"].write(light_rgba.tobytes())
        if upload_visible_from_cpu:
            visible_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            visible_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
            visible_rgba[..., 3] = 1.0
            self.textures["visible_illumination"].write(visible_rgba.tobytes())
        if upload_debug_texture:
            if debug_frame is None:
                debug_frame = world.debug_frame(world.default_debug_view)
            debug_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            debug_rgba[..., :3] = np.clip(debug_frame, 0.0, 1.0)
            debug_rgba[..., 3] = 1.0
            self.textures["debug"].write(debug_rgba.tobytes())
        if self._should_upload_cpu_resource(world, "ambient_temperature"):
            self.textures["ambient_temperature"].write(world.ambient_temperature.astype("f4").tobytes())
        if self._should_upload_cpu_resource(world, "pressure_ping"):
            self.textures["pressure_ping"].write(world.pressure_ping.astype("f4").tobytes())
        if self._should_upload_cpu_resource(world, "flow_velocity"):
            self.textures["flow_velocity"].write(world.flow_velocity.astype("f4").tobytes())
        if getattr(world, "simulation_backend", "") == "gpu":
            self.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
                "collapse_delay_pending",
                "gas_concentration",
                "ambient_temperature",
                "flow_velocity",
                "pressure_ping",
                "visible_illumination",
                "cell_optical_dose",
                "gas_optical_dose",
                "active_meta",
                "active_tile_ttl",
                "active_chunk_mask",
            )

    def sync_display_textures(self, world: "WorldEngine") -> None:
        """Refresh textures sampled by the desktop demo from GPU-authoritative buffers."""
        if not self.enabled or self.ctx is None:
            return
        self.ensure_world_resources(world)
        if getattr(world, "simulation_backend", "") != "gpu":
            return
        if "cell_core" in self.gpu_authoritative_resources and "cell_core" in self.buffers:
            self._ensure_display_programs()
            program = self.display_programs["material_from_cell_core"]
            program["width"] = int(world.width)
            program["height"] = int(world.height)
            self.buffers["cell_core"].bind_to_storage_buffer(0)
            self.textures["material"].bind_to_image(0, read=False, write=True)
            program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
            self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        if "visible_illumination" in self.gpu_authoritative_resources and "visible_illumination" in self.textures:
            self._ensure_display_programs()
            program = self.display_programs["light_from_visible_texture"]
            program["width"] = int(world.width)
            program["height"] = int(world.height)
            self.textures["visible_illumination"].use(0)
            self.textures["light"].bind_to_image(0, read=False, write=True)
            program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
            self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def sync_debug_display_texture(
        self,
        world: "WorldEngine",
        *,
        view: str,
        gas_species_id: int = -1,
        light_dose_channel: int = -1,
    ) -> bool:
        """Refresh the desktop demo debug texture using only GPU-resident state."""
        if not self.enabled or self.ctx is None:
            return False
        if getattr(world, "simulation_backend", "") != "gpu":
            return False
        self.ensure_world_resources(world)
        self._ensure_display_programs()
        view_ids = {
            "active": 7,
            "temperature": 1,
            "heat": 1,
            "velocity": 2,
            "motion": 2,
            "light": 3,
            "optics": 4,
            "gas": 5,
            "pressure": 6,
        }
        view_id = view_ids.get(str(view).lower(), 0)
        if view_id == 0:
            return False
        program = self.display_programs["debug_from_gpu_state"]
        program["width"] = int(world.width)
        program["height"] = int(world.height)
        program["gas_width"] = int(world.gas_width)
        program["gas_height"] = int(world.gas_height)
        program["gas_cell_size"] = int(world.gas_cell_size)
        program["tile_width"] = int(world.active.tile_width)
        program["tile_height"] = int(world.active.tile_height)
        program["tile_size"] = int(world.active.tile_size)
        program["active_ttl_reset"] = int(world.active.active_ttl_reset)
        program["view_mode"] = int(view_id)
        program["gas_species_id"] = int(gas_species_id)
        program["light_dose_channel"] = int(light_dose_channel)
        program["light_channel_count"] = int(world.cell_optical_dose.shape[0])
        program["gas_species_count"] = int(world.gas_concentration.shape[0])
        self.buffers["cell_core"].bind_to_storage_buffer(0)
        self.buffers["gas_concentration"].bind_to_storage_buffer(1)
        self.buffers["cell_optical_dose"].bind_to_storage_buffer(2)
        self.buffers["gas_optical_dose"].bind_to_storage_buffer(3)
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(4)
        self.textures["visible_illumination"].use(0)
        self.textures["flow_velocity"].use(1)
        self.textures["pressure_ping"].use(2)
        self.textures["debug"].bind_to_image(0, read=False, write=True)
        program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
        self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        return True

    def _ensure_display_programs(self) -> None:
        if not self.enabled or self.ctx is None:
            return
        if "material_from_cell_core" not in self.display_programs:
            self.display_programs["material_from_cell_core"] = self.ctx.compute_shader(
                """
                #version 430
                layout(local_size_x = 16, local_size_y = 16) in;
                layout(std430, binding = 0) readonly buffer CellCoreBuffer {
                    uint cell_core[];
                };
                layout(r32f, binding = 0) writeonly uniform image2D material_tex;
                uniform int width;
                uniform int height;
                void main() {
                    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                    if (pos.x >= width || pos.y >= height) {
                        return;
                    }
                    int index = pos.y * width + pos.x;
                    uint word0 = cell_core[index * 5];
                    float material_id = float(word0 & 0xFFFFu);
                    imageStore(material_tex, pos, vec4(material_id, 0.0, 0.0, 1.0));
                }
                """
            )
        if "light_from_visible_texture" not in self.display_programs:
            self.display_programs["light_from_visible_texture"] = self.ctx.compute_shader(
                """
                #version 430
                layout(local_size_x = 16, local_size_y = 16) in;
                layout(rgba32f, binding = 0) writeonly uniform image2D light_tex;
                uniform sampler2D visible_tex;
                uniform int width;
                uniform int height;
                void main() {
                    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                    if (pos.x >= width || pos.y >= height) {
                        return;
                    }
                    vec4 light = texelFetch(visible_tex, pos, 0);
                    imageStore(light_tex, pos, vec4(light.rgb, 1.0));
                }
                """
            )
            self.display_programs["light_from_visible_texture"]["visible_tex"] = 0
        if "debug_from_gpu_state" not in self.display_programs:
            self.display_programs["debug_from_gpu_state"] = self.ctx.compute_shader(
                """
                #version 430
                layout(local_size_x = 16, local_size_y = 16) in;
                layout(std430, binding = 0) readonly buffer CellCoreBuffer {
                    uint cell_core[];
                };
                layout(std430, binding = 1) readonly buffer GasBuffer {
                    float gas_concentration[];
                };
                layout(std430, binding = 2) readonly buffer CellDoseBuffer {
                    float cell_optical_dose[];
                };
                layout(std430, binding = 3) readonly buffer GasDoseBuffer {
                    float gas_optical_dose[];
                };
                layout(std430, binding = 4) readonly buffer ActiveTileBuffer {
                    int active_tile_ttl[];
                };
                layout(rgba32f, binding = 0) writeonly uniform image2D debug_tex;
                uniform sampler2D visible_tex;
                uniform sampler2D flow_velocity_tex;
                uniform sampler2D pressure_tex;
                uniform int width;
                uniform int height;
                uniform int gas_width;
                uniform int gas_height;
                uniform int gas_cell_size;
                uniform int tile_width;
                uniform int tile_height;
                uniform int tile_size;
                uniform int active_ttl_reset;
                uniform int view_mode;
                uniform int gas_species_id;
                uniform int light_dose_channel;
                uniform int light_channel_count;
                uniform int gas_species_count;

                vec3 heat_color(float t) {
                    float cold = clamp((20.0 - t) / 80.0, 0.0, 1.0);
                    float hot = clamp((t - 20.0) / 180.0, 0.0, 1.0);
                    float warm = clamp(1.0 - abs(t - 20.0) / 80.0, 0.0, 1.0);
                    return clamp(vec3(hot, warm * 0.22 + hot * 0.45, cold), 0.0, 1.0);
                }

                vec3 vector_color(vec2 v) {
                    float mag = clamp(length(v) / 4.0, 0.0, 1.0);
                    if (mag <= 0.00001) {
                        return vec3(0.0);
                    }
                    vec2 dir = normalize(v);
                    return clamp(vec3(max(dir.x, 0.0), max(dir.y, 0.0), max(-dir.x, 0.0)) * mag + vec3(0.0, 0.0, max(-dir.y, 0.0)) * mag, 0.0, 1.0);
                }

                void main() {
                    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                    if (pos.x >= width || pos.y >= height) {
                        return;
                    }
                    int cell_index = pos.y * width + pos.x;
                    uint word0 = cell_core[cell_index * 5];
                    uint word1 = cell_core[cell_index * 5 + 1];
                    uint word2 = cell_core[cell_index * 5 + 2];
                    int material_id = int(word0 & 0xFFFFu);
                    vec2 cell_velocity = unpackHalf2x16(word1);
                    float cell_temperature = uintBitsToFloat(word2);
                    ivec2 gas_cell = clamp(pos / max(1, gas_cell_size), ivec2(0), ivec2(max(0, gas_width - 1), max(0, gas_height - 1)));
                    int gas_index = gas_cell.y * gas_width + gas_cell.x;
                    vec3 color = vec3(0.0);
                    if (view_mode == 1) {
                        color = heat_color(cell_temperature);
                    } else if (view_mode == 2) {
                        vec2 flow = texelFetch(flow_velocity_tex, gas_cell, 0).xy;
                        color = vector_color(material_id > 0 ? cell_velocity : flow);
                    } else if (view_mode == 3) {
                        color = clamp(texelFetch(visible_tex, pos, 0).rgb, 0.0, 1.0);
                    } else if (view_mode == 4) {
                        if (light_dose_channel >= 0 && light_dose_channel < light_channel_count) {
                            float cell_dose = cell_optical_dose[light_dose_channel * width * height + cell_index];
                            float gas_dose = gas_optical_dose[light_dose_channel * gas_width * gas_height + gas_index];
                            float strength = 1.0 - exp(-max(0.0, cell_dose + gas_dose * 0.65));
                            color = vec3(strength * 0.2, strength * 0.95, strength);
                        } else {
                            color = clamp(texelFetch(visible_tex, pos, 0).rgb, 0.0, 1.0);
                        }
                    } else if (view_mode == 5) {
                        if (gas_species_id >= 0 && gas_species_id < gas_species_count) {
                            float amount = gas_concentration[gas_species_id * gas_width * gas_height + gas_index];
                            float strength = 1.0 - exp(-max(0.0, amount));
                            color = vec3(strength * 0.3, strength, strength * 0.6);
                        }
                    } else if (view_mode == 6) {
                        float pressure = texelFetch(pressure_tex, gas_cell, 0).x;
                        float pos_pressure = clamp(pressure, 0.0, 1.0);
                        float neg_pressure = clamp(-pressure, 0.0, 1.0);
                        color = vec3(pos_pressure, 0.18 * (1.0 - clamp(abs(pressure), 0.0, 1.0)), neg_pressure);
                    } else if (view_mode == 7) {
                        ivec2 tile = clamp(pos / max(1, tile_size), ivec2(0), ivec2(max(0, tile_width - 1), max(0, tile_height - 1)));
                        int ttl = active_tile_ttl[tile.y * tile_width + tile.x];
                        float active_value = clamp(float(ttl) / max(1.0, float(active_ttl_reset)), 0.0, 1.0);
                        color = vec3(active_value * 0.10, active_value * 0.95, 0.0);
                    }
                    imageStore(debug_tex, pos, vec4(clamp(color, 0.0, 1.0), 1.0));
                }
                """
            )
            self.display_programs["debug_from_gpu_state"]["visible_tex"] = 0
            self.display_programs["debug_from_gpu_state"]["flow_velocity_tex"] = 1
            self.display_programs["debug_from_gpu_state"]["pressure_tex"] = 2

    def mark_gpu_authoritative(self, *resource_names: str) -> None:
        self.gpu_authoritative_resources.update(str(name) for name in resource_names)

    def clear_gpu_authoritative(self, *resource_names: str) -> None:
        if resource_names:
            for name in resource_names:
                self.gpu_authoritative_resources.discard(str(name))
            return
        self.gpu_authoritative_resources.clear()

    def _should_upload_cpu_resource(self, world: "WorldEngine", resource_name: str) -> bool:
        if self._force_cpu_resource_upload:
            return True
        return not (
            str(resource_name) in self.gpu_authoritative_resources
            and getattr(world, "simulation_backend", "") == "gpu"
        )

    @staticmethod
    def _should_upload_cpu_solver_runtime(world: "WorldEngine") -> bool:
        return getattr(world, "simulation_backend", "") == "cpu"

    def _shadow_or_default(self, name: str, default: np.ndarray) -> np.ndarray:
        existing = self.shadow_buffers.get(name)
        if isinstance(existing, np.ndarray) and existing.shape == default.shape and existing.dtype == default.dtype:
            return existing
        return default

    def _write_dynamic_buffer(self, name: str, data: np.ndarray) -> None:
        if not self.enabled or self.ctx is None:
            return
        buffer = self.buffers.get(name)
        nbytes = max(4, data.nbytes)
        if buffer is None:
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.buffers[name] = buffer
        elif buffer.size < nbytes:
            buffer.release()
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.buffers[name] = buffer
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    def sync_readback_requests(self, world: "WorldEngine") -> None:
        readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
        self.shadow_buffers["readback_request"] = readback_request_upload.copy()
        self.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
        self._write_dynamic_buffer("readback_request", readback_request_upload)
        self._write_dynamic_buffer("readback_request_label", readback_request_label_upload)

    def sync_force_sources(self, world: "WorldEngine") -> None:
        force_source_upload = pack_force_source_upload(world)
        force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
        self.shadow_buffers["force_source"] = force_source_upload.copy()
        self.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
        self._write_dynamic_buffer("force_source", force_source_upload)
        if self.enabled and self.ctx is not None:
            self.buffers["force_source_count"].write(force_source_count_upload.tobytes())

    def mark_active_rects(
        self,
        world: "WorldEngine",
        rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
    ) -> bool:
        if not rects:
            return True
        if not self.enabled or self.ctx is None:
            return False
        self.ensure_world_resources(world)
        if (
            "active_meta" not in self.buffers
            or "active_tile_ttl" not in self.buffers
            or "active_chunk_mask" not in self.buffers
        ):
            return False
        self._ensure_active_scheduler_programs()
        tile_count = int(world.active.tile_width * world.active.tile_height)
        chunk_count = int(world.active.chunk_width * world.active.chunk_height)
        if tile_count <= 0 or chunk_count <= 0:
            return False

        packed_rects = np.zeros((len(rects),), dtype=ACTIVE_RECT_DTYPE)
        for index, rect in enumerate(rects):
            if len(rect) == 4:
                x0, y0, x1, y1 = rect
                tile_padding = 0
            else:
                x0, y0, x1, y1, tile_padding = rect
            packed_rects[index]["x0"] = int(x0)
            packed_rects[index]["y0"] = int(y0)
            packed_rects[index]["x1"] = int(x1)
            packed_rects[index]["y1"] = int(y1)
            packed_rects[index]["tile_padding"] = max(0, int(tile_padding))
        self._write_dynamic_buffer("active_rect", packed_rects)

        mark_program = self.active_scheduler_programs["mark_active_rects"]
        mark_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        mark_program["world_size"].value = (world.width, world.height)
        mark_program["tile_size"].value = int(world.active.tile_size)
        mark_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        mark_program["rect_count"].value = int(len(packed_rects))
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        self.buffers["active_rect"].bind_to_storage_buffer(binding=1)
        mark_program.run((tile_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )
        self._refresh_active_chunks_and_meta(world, read_meta=False)
        self.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
        return True

    def decay_active_scheduler(self, world: "WorldEngine") -> bool:
        if not self.enabled or self.ctx is None:
            return False
        self.ensure_world_resources(world)
        if (
            "active_meta" not in self.buffers
            or "active_tile_ttl" not in self.buffers
            or "active_chunk_mask" not in self.buffers
        ):
            return False
        self._ensure_active_scheduler_programs()
        tile_count = int(world.active.tile_width * world.active.tile_height)
        chunk_count = int(world.active.chunk_width * world.active.chunk_height)
        if tile_count <= 0 or chunk_count <= 0:
            return False

        decay_program = self.active_scheduler_programs["decay_active_tiles"]
        decay_program["tile_count"].value = tile_count
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        decay_program.run((tile_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )

        self._refresh_active_chunks_and_meta(world, read_meta=False)
        self.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
        return True

    def _refresh_active_chunks_and_meta(self, world: "WorldEngine", *, read_meta: bool = False) -> None:
        assert self.ctx is not None
        clear_program = self.active_scheduler_programs["clear_active_counts"]
        self.buffers["active_meta"].bind_to_storage_buffer(binding=0)
        self.buffers["active_chunk_count"].bind_to_storage_buffer(binding=1)
        self.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=2)
        clear_program.run(1, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )

        refresh_program = self.active_scheduler_programs["refresh_active_chunks"]
        refresh_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        refresh_program["chunk_grid_size"].value = (world.active.chunk_width, world.active.chunk_height)
        refresh_program["chunk_tiles"].value = int(world.active.chunk_tiles)
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        self.buffers["active_chunk_mask"].bind_to_storage_buffer(binding=1)
        self.buffers["active_meta"].bind_to_storage_buffer(binding=2)
        self.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
        self.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
        self.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=5)
        refresh_program.run(world.active.chunk_width, world.active.chunk_height, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )
        if read_meta:
            self.shadow_buffers["active_meta"] = np.frombuffer(
                self.buffers["active_meta"].read(size=ACTIVE_META_DTYPE.itemsize),
                dtype=ACTIVE_META_DTYPE,
                count=1,
            ).copy()

    def _ensure_active_scheduler_programs(self) -> None:
        if self.ctx is None:
            return
        required_programs = {
            "mark_active_rects",
            "decay_active_tiles",
            "clear_active_counts",
            "count_active_scheduler",
            "refresh_active_chunks",
        }
        if required_programs.issubset(self.active_scheduler_programs):
            return
        for name in required_programs:
            program = self.active_scheduler_programs.pop(name, None)
            if program is not None:
                try:
                    program.release()
                except Exception:
                    pass
        self.active_scheduler_programs["mark_active_rects"] = self.ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform ivec2 world_size;
            uniform int tile_size;
            uniform int active_ttl_reset;
            uniform int rect_count;
            layout(std430, binding=0) buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };
            struct ActiveRect {
                int x0;
                int y0;
                int x1;
                int y1;
                int tile_padding;
            };
            layout(std430, binding=1) readonly buffer ActiveRectBuffer {
                ActiveRect active_rects[];
            };
            int ceil_div(int value, int divisor) {
                return (value + divisor - 1) / divisor;
            }
            void main() {
                uint index = gl_GlobalInvocationID.x;
                int tile_count = tile_grid_size.x * tile_grid_size.y;
                if (index >= uint(tile_count)) {
                    return;
                }
                int tile_x = int(index) % tile_grid_size.x;
                int tile_y = int(index) / tile_grid_size.x;
                for (int rect_index = 0; rect_index < rect_count; ++rect_index) {
                    ActiveRect rect = active_rects[rect_index];
                    int x0 = clamp(rect.x0, 0, world_size.x);
                    int y0 = clamp(rect.y0, 0, world_size.y);
                    int x1 = clamp(rect.x1, 0, world_size.x);
                    int y1 = clamp(rect.y1, 0, world_size.y);
                    if (x1 <= x0 || y1 <= y0) {
                        continue;
                    }
                    int padding = max(0, rect.tile_padding);
                    int tile_x0 = max(0, x0 / tile_size - padding);
                    int tile_y0 = max(0, y0 / tile_size - padding);
                    int tile_x1 = min(tile_grid_size.x, ceil_div(x1, tile_size) + padding);
                    int tile_y1 = min(tile_grid_size.y, ceil_div(y1, tile_size) + padding);
                    if (tile_x >= tile_x0 && tile_x < tile_x1 && tile_y >= tile_y0 && tile_y < tile_y1) {
                        active_tile_ttl[index] = active_ttl_reset;
                        return;
                    }
                }
            }
            """
        )
        self.active_scheduler_programs["decay_active_tiles"] = self.ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;
            layout(std430, binding=0) buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };
            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index >= uint(tile_count)) {
                    return;
                }
                if (active_tile_ttl[index] > 0) {
                    active_tile_ttl[index] -= 1;
                }
            }
            """
        )
        self.active_scheduler_programs["clear_active_counts"] = self.ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            layout(std430, binding=0) buffer ActiveMetaBuffer {
                int active_meta[];
            };
            layout(std430, binding=1) buffer ActiveChunkCountBuffer {
                uint active_chunk_count[];
            };
            layout(std430, binding=2) buffer ActiveChunkDispatchArgsBuffer {
                uint active_chunk_dispatch_args[];
            };
            void main() {
                active_meta[7] = 0;
                active_meta[8] = 0;
                active_chunk_count[0] = 0u;
                active_chunk_dispatch_args[0] = 0u;
                active_chunk_dispatch_args[1] = 1u;
                active_chunk_dispatch_args[2] = 1u;
            }
            """
        )
        self.active_scheduler_programs["count_active_scheduler"] = self.ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
            uniform int tile_count;
            uniform int chunk_count;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };
            layout(std430, binding=1) readonly buffer ActiveChunkMaskBuffer {
                int active_chunk_mask[];
            };
            layout(std430, binding=2) buffer ActiveMetaBuffer {
                int active_meta[];
            };
            void main() {
                uint index = gl_GlobalInvocationID.x;
                if (index < uint(tile_count) && active_tile_ttl[index] > 0) {
                    atomicAdd(active_meta[7], 1);
                }
                if (index < uint(chunk_count) && active_chunk_mask[index] > 0) {
                    atomicAdd(active_meta[8], 1);
                }
            }
            """
        )
        self.active_scheduler_programs["refresh_active_chunks"] = self.ctx.compute_shader(
            """
            #version 430
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            uniform ivec2 tile_grid_size;
            uniform ivec2 chunk_grid_size;
            uniform int chunk_tiles;
            layout(std430, binding=0) readonly buffer ActiveTileTTLBuffer {
                int active_tile_ttl[];
            };
            layout(std430, binding=1) buffer ActiveChunkMaskBuffer {
                int active_chunk_mask[];
            };
            layout(std430, binding=2) buffer ActiveMetaBuffer {
                int active_meta[];
            };
            layout(std430, binding=3) buffer ActiveChunkCountBuffer {
                uint active_chunk_count[];
            };
            layout(std430, binding=4) buffer ActiveChunkListBuffer {
                ivec2 active_chunk_list[];
            };
            layout(std430, binding=5) buffer ActiveChunkDispatchArgsBuffer {
                uint active_chunk_dispatch_args[];
            };
            void main() {
                ivec2 chunk = ivec2(gl_GlobalInvocationID.xy);
                if (chunk.x >= chunk_grid_size.x || chunk.y >= chunk_grid_size.y) {
                    return;
                }
                int x0 = chunk.x * chunk_tiles;
                int y0 = chunk.y * chunk_tiles;
                int x1 = min(tile_grid_size.x, x0 + chunk_tiles);
                int y1 = min(tile_grid_size.y, y0 + chunk_tiles);
                int active_tile_count = 0;
                for (int tile_y = y0; tile_y < y1; ++tile_y) {
                    for (int tile_x = x0; tile_x < x1; ++tile_x) {
                        int tile_index = tile_y * tile_grid_size.x + tile_x;
                        if (active_tile_ttl[tile_index] > 0) {
                            active_tile_count += 1;
                        }
                    }
                }
                int chunk_index = chunk.y * chunk_grid_size.x + chunk.x;
                int active_flag = active_tile_count > 0 ? 1 : 0;
                active_chunk_mask[chunk_index] = active_flag;
                if (active_flag == 0) {
                    return;
                }
                atomicAdd(active_meta[7], active_tile_count);
                atomicAdd(active_meta[8], 1);
                uint slot = atomicAdd(active_chunk_count[0], 1u);
                active_chunk_list[slot] = chunk;
                atomicMax(active_chunk_dispatch_args[0], slot + 1u);
            }
            """
        )

    def _write_typed_table_buffer(self, name: str, data: np.ndarray) -> None:
        if not self.enabled or self.ctx is None:
            return
        buffer = self.typed_table_buffers.get(name)
        nbytes = max(4, data.nbytes)
        if buffer is None or buffer.size < nbytes:
            if buffer is not None:
                buffer.release()
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.typed_table_buffers[name] = buffer
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    def queue_readback(
        self,
        frame_id: int,
        request: ReadbackRequest,
        payload: dict[str, Any],
        *,
        require_gpu_sources: bool = False,
    ) -> bool:
        slot: GLReadbackSlot | None = None
        slot_count = len(self.readback_slots)
        for offset in range(slot_count):
            candidate = self.readback_slots[(self.write_index + offset) % slot_count]
            if candidate.frame_id < 0 and candidate.request is None:
                slot = candidate
                break
        if slot is None:
            return False
        plan = self._plan_readback_payload(payload)
        gpu_backed = bool(plan.gpu_sources)
        latency_frames = GPU_READBACK_LATENCY_FRAMES if gpu_backed else CPU_READBACK_LATENCY_FRAMES
        if require_gpu_sources and plan.cpu_chunks:
            paths = ", ".join(".".join(path) if path else "<root>" for path in plan.cpu_chunk_paths)
            raise RuntimeError(
                f"GPU readback requires GPU-backed payload arrays, found CPU payload chunks at: {paths}; "
                "CPU fallback is disabled"
            )
        if require_gpu_sources and plan.gpu_sources and (not self.enabled or self.ctx is None):
            raise RuntimeError("GPU readback requires an enabled ModernGL context; CPU fallback is disabled")
        if self.enabled and self.ctx is not None:
            if slot.buffer is None or slot.buffer.size < max(plan.nbytes, 4):
                if slot.buffer is not None:
                    slot.buffer.release()
                slot.buffer = self.ctx.buffer(reserve=max(plan.nbytes, 4), dynamic=True)
            else:
                slot.buffer.orphan(max(plan.nbytes, 4))
            for offset, data in plan.cpu_chunks:
                if data:
                    slot.buffer.write(data, offset=offset)
            for offset, source in plan.gpu_sources:
                self._fill_readback_slot_from_gpu(
                    slot.buffer,
                    offset,
                    source,
                    require_gpu_source=require_gpu_sources,
                )
        else:
            if plan.gpu_sources:
                names = ", ".join(source.resource_name for _, source in plan.gpu_sources)
                raise RuntimeError(
                    f"GPU readback requires an enabled ModernGL context for GPU sources: {names}; "
                    "CPU fallback is disabled"
                )
            raw = bytearray(plan.nbytes)
            for offset, data in plan.cpu_chunks:
                raw[offset : offset + len(data)] = data
            slot.buffer = bytes(raw)
        slot.frame_id = frame_id
        slot.ready_frame_id = frame_id + CPU_READBACK_LATENCY_FRAMES
        slot.min_poll_frame_id = frame_id + latency_frames
        slot.latency_frames = latency_frames
        slot.gpu_backed = gpu_backed
        slot.request = request
        slot.nbytes = plan.nbytes
        slot.layout = plan.layout
        self.write_index = (self.write_index + 1) % len(self.readback_slots)
        return True

    def poll_readback(self, current_frame_id: int) -> ReadbackResult | None:
        ready_slots = [
            slot
            for slot in self.readback_slots
            if slot.frame_id >= 0
            and slot.request is not None
            and slot.min_poll_frame_id >= 0
            and slot.min_poll_frame_id <= current_frame_id
        ]
        if not ready_slots:
            return None
        slot = min(ready_slots, key=lambda item: (item.frame_id, item.slot_index))
        if slot.nbytes <= 0:
            raw = b""
        elif self.enabled and self.ctx is not None and slot.buffer is not None:
            raw = slot.buffer.read(size=slot.nbytes)
        else:
            raw = slot.buffer if isinstance(slot.buffer, (bytes, bytearray)) else b""
        payload = self._decode_readback_payload(raw, slot.layout)
        result = ReadbackResult(frame_id=slot.frame_id, request=slot.request, payload=payload)
        slot.frame_id = -1
        slot.ready_frame_id = -1
        slot.min_poll_frame_id = -1
        slot.latency_frames = CPU_READBACK_LATENCY_FRAMES
        slot.gpu_backed = False
        slot.request = None
        slot.nbytes = 0
        slot.layout = None
        return result

    @staticmethod
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

    @staticmethod
    def _serialize_ndarray_summary(array: np.ndarray) -> dict[str, Any]:
        return {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        }

    @staticmethod
    def _resource_size_bytes(resource: Any) -> int | None:
        size = getattr(resource, "size", None)
        if size is None or isinstance(size, tuple):
            return None
        try:
            return int(size)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _serialize_buffer_summary(cls, resource: Any) -> dict[str, Any]:
        return {"size_bytes": cls._resource_size_bytes(resource)}

    @staticmethod
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

    @staticmethod
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

    @classmethod
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

    def serialize_runtime_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "has_context": self.ctx is not None,
            "own_context": bool(self.own_context),
            "world_signature": None
            if self.world_signature is None
            else [int(value) for value in self.world_signature],
            "rule_table_signature": None
            if self.rule_table_signature is None
            else [int(value) for value in self.rule_table_signature],
            "atlas_grid": [int(self.atlas_grid[0]), int(self.atlas_grid[1])],
            "atlas_dirty": bool(self.atlas_dirty),
            "write_index": int(self.write_index),
            "table_generations": {
                str(name): int(generation)
                for name, generation in sorted(self.table_generations.items())
            },
            "shadow_tables": {
                str(name): self._serialize_table_summary(payload)
                for name, payload in sorted(self.shadow_tables.items())
            },
            "shadow_typed_tables": {
                str(name): self._serialize_ndarray_summary(payload)
                for name, payload in sorted(self.shadow_typed_tables.items())
            },
            "shadow_buffers": {
                str(name): self._serialize_ndarray_summary(payload)
                for name, payload in sorted(self.shadow_buffers.items())
            },
            "textures": {
                str(name): self._serialize_texture_summary(texture)
                for name, texture in sorted(self.textures.items())
            },
            "buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.buffers.items())
            },
            "table_buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.table_buffers.items())
            },
            "typed_table_buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.typed_table_buffers.items())
            },
            "readback_programs": sorted(str(name) for name in self.readback_programs.keys()),
            "readback_latency_frames": {
                "cpu_payload": int(CPU_READBACK_LATENCY_FRAMES),
                "gpu_payload": int(GPU_READBACK_LATENCY_FRAMES),
            },
            "readback_slots": [self._serialize_readback_slot(slot) for slot in self.readback_slots],
        }

    def texture(self, name: str) -> Any | None:
        return self.textures.get(name)

    def atlas_texture(self) -> Any | None:
        return self.textures.get("atlas")

    def release_resources(self) -> None:
        for texture in self.textures.values():
            try:
                texture.release()
            except Exception:
                pass
        for buffer in self.buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for buffer in self.table_buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for buffer in self.typed_table_buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for slot in self.readback_slots:
            if self.enabled and self.ctx is not None and hasattr(slot.buffer, "release"):
                try:
                    slot.buffer.release()
                except Exception:
                    pass
            slot.buffer = None
            slot.frame_id = -1
            slot.ready_frame_id = -1
            slot.min_poll_frame_id = -1
            slot.latency_frames = CPU_READBACK_LATENCY_FRAMES
            slot.gpu_backed = False
            slot.request = None
            slot.nbytes = 0
            slot.layout = None
        self.textures.clear()
        self.buffers.clear()
        self.table_buffers.clear()
        self.typed_table_buffers.clear()
        self._release_active_scheduler_programs()
        self._release_display_programs()
        self.gpu_authoritative_resources.clear()
        self.rule_table_signature = None

    def release(self) -> None:
        self.release_resources()
        self._release_readback_programs()
        if self.own_context and self.ctx is not None:
            try:
                self.ctx.release()
            except Exception:
                pass
        self.ctx = None
        self.enabled = False
        self.own_context = False
        self.owner_thread_id = None

    def _ensure_atlas_texture(self, world: "WorldEngine") -> None:
        if not self.enabled or self.ctx is None or not self.atlas_dirty:
            return
        material_count = max(world.rulebook.materials_by_id, default=0) + 1
        cols = 8
        rows = max(1, math.ceil(material_count / cols))
        tile = 8
        atlas = np.zeros((rows * tile, cols * tile, 3), dtype="f4")
        for material in world.rulebook.materials_by_id.values():
            tx = material.material_id % cols
            ty = material.material_id // cols
            atlas[ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile] = _render_group_tile(material, tile)
        existing = self.textures.get("atlas")
        if existing is not None:
            try:
                existing.release()
            except Exception:
                pass
        self.textures["atlas"] = self.ctx.texture((cols * tile, rows * tile), 3, atlas.tobytes(), dtype="f4")
        self.textures["atlas"].filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.atlas_grid = (cols, rows)
        self.atlas_dirty = False

    def _plan_readback_payload(self, payload: dict[str, Any]) -> ReadbackPayloadPlan:
        plan = ReadbackPayloadPlan(layout=ReadbackPayloadLayout())
        offset = 0
        gpu_source_types = (
            GPUBufferReadbackSource,
            GPUCellCoreWindowReadbackSource,
            GPUGasWindowReadbackSource,
            GPUTextureReadbackSource,
            GPUSegmentedBufferReadbackSource,
            GPUSegmentedCellCoreWindowReadbackSource,
            GPUSegmentedTextureReadbackSource,
        )

        def visit(path: tuple[str, ...], value: Any) -> Any:
            nonlocal offset
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                plan.layout.arrays.append(
                    ReadbackArrayLayout(
                        path=path,
                        dtype=array.dtype.str,
                        shape=tuple(int(dim) for dim in array.shape),
                        offset=offset,
                        nbytes=array.nbytes,
                    )
                )
                plan.cpu_chunks.append((offset, array.tobytes()))
                plan.cpu_chunk_paths.append(path)
                offset += array.nbytes
                return None
            if isinstance(value, gpu_source_types):
                dtype = np.dtype(value.dtype)
                nbytes = int(np.prod(value.shape, dtype=np.int64)) * dtype.itemsize
                plan.layout.arrays.append(
                    ReadbackArrayLayout(
                        path=path,
                        dtype=dtype.str,
                        shape=tuple(int(dim) for dim in value.shape),
                        offset=offset,
                        nbytes=nbytes,
                    )
                )
                plan.gpu_sources.append((offset, value))
                offset += nbytes
                return None
            if isinstance(value, dict):
                metadata: dict[str, Any] = {}
                for key, child in value.items():
                    child_meta = visit(path + (str(key),), child)
                    if child_meta is not None:
                        metadata[str(key)] = child_meta
                return metadata
            return self._normalize_metadata(value)

        metadata = visit((), payload)
        plan.layout.metadata = metadata if isinstance(metadata, dict) else {}
        plan.nbytes = offset
        return plan

    def _fill_readback_slot_from_gpu(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUBufferReadbackSource
        | GPUCellCoreWindowReadbackSource
        | GPUGasWindowReadbackSource
        | GPUTextureReadbackSource
        | GPUSegmentedBufferReadbackSource
        | GPUSegmentedCellCoreWindowReadbackSource
        | GPUSegmentedTextureReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        assert self.ctx is not None
        if isinstance(source, GPUSegmentedCellCoreWindowReadbackSource):
            self._pack_segmented_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUSegmentedBufferReadbackSource):
            self._pack_segmented_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUSegmentedTextureReadbackSource):
            self._pack_segmented_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUCellCoreWindowReadbackSource):
            self._pack_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUGasWindowReadbackSource):
            self._pack_gas_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUBufferReadbackSource):
            self._pack_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUTextureReadbackSource):
            self._pack_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        raise TypeError(f"Unsupported GPU readback source: {type(source)!r}")

    def _decode_readback_payload(self, raw: bytes, layout: ReadbackPayloadLayout | None) -> dict[str, Any]:
        if layout is None:
            return {}
        payload = deepcopy(layout.metadata)
        for spec in layout.arrays:
            array = np.frombuffer(raw, dtype=np.dtype(spec.dtype), count=int(np.prod(spec.shape, dtype=np.int64)), offset=spec.offset)
            array = array.reshape(spec.shape).copy()
            cursor = payload
            for key in spec.path[:-1]:
                child = cursor.get(key)
                if not isinstance(child, dict):
                    child = {}
                    cursor[key] = child
                cursor = child
            cursor[spec.path[-1]] = array
        return payload

    def _normalize_metadata(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._normalize_metadata(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return [self._normalize_metadata(child) for child in value]
        if isinstance(value, list):
            return [self._normalize_metadata(child) for child in value]
        return value

    def _ensure_readback_programs(self) -> None:
        if self.ctx is None or self.readback_programs:
            return
        local_size = 8
        self.readback_programs["cell_core_window"] = self.ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={local_size}, local_size_y={local_size}, local_size_z=1) in;
            uniform ivec2 window_origin;
            uniform ivec2 window_size;
            uniform int cell_grid_width;
            uniform int dst_word_offset;
            uniform int dst_cell_grid_width;
            layout(std430, binding=0) readonly buffer CellCore {{
                uint cell_core[];
            }};
            layout(std430, binding=1) writeonly buffer SlotWords {{
                uint slot_words[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= window_size.x || gid.y >= window_size.y) {{
                    return;
                }}
                int src_cell = (window_origin.y + gid.y) * cell_grid_width + (window_origin.x + gid.x);
                int dst_cell = gid.y * dst_cell_grid_width + gid.x;
                int src_word = src_cell * 5;
                int dst_word = dst_word_offset + dst_cell * 5;
                for (int lane = 0; lane < 5; ++lane) {{
                    slot_words[dst_word + lane] = cell_core[src_word + lane];
                }}
            }}
            """
        )
        self.readback_programs["gas_window"] = self.ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={local_size}, local_size_y={local_size}, local_size_z=1) in;
            uniform ivec2 window_origin;
            uniform ivec2 window_size;
            uniform ivec2 gas_grid_size;
            uniform int species_id;
            uniform int dst_word_offset;
            layout(std430, binding=0) readonly buffer GasValues {{
                float gas_values[];
            }};
            layout(std430, binding=1) writeonly buffer SlotWords {{
                uint slot_words[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= window_size.x || gid.y >= window_size.y) {{
                    return;
                }}
                int src_x = window_origin.x + gid.x;
                int src_y = window_origin.y + gid.y;
                int src_index = ((species_id * gas_grid_size.y + src_y) * gas_grid_size.x) + src_x;
                int dst_index = dst_word_offset + gid.y * window_size.x + gid.x;
                slot_words[dst_index] = floatBitsToUint(gas_values[src_index]);
            }}
            """
        )
        self.readback_programs["buffer_window"] = self.ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={local_size}, local_size_y={local_size}, local_size_z=1) in;
            uniform int src_word_offset;
            uniform int src_word_stride;
            uniform int dst_word_offset;
            uniform int dst_words_per_row;
            uniform int dst_word_stride;
            uniform int row_count;
            layout(std430, binding=0) readonly buffer SrcWords {{
                uint src_words[];
            }};
            layout(std430, binding=1) writeonly buffer SlotWords {{
                uint slot_words[];
            }};
            void main() {{
                int word_index = int(gl_GlobalInvocationID.x);
                int row_index = int(gl_GlobalInvocationID.y);
                if (word_index >= dst_words_per_row || row_index >= row_count) {{
                    return;
                }}
                int src_index = src_word_offset + row_index * src_word_stride + word_index;
                int dst_index = dst_word_offset + row_index * dst_word_stride + word_index;
                slot_words[dst_index] = src_words[src_index];
            }}
            """
        )
        self.readback_programs["texture_window"] = self.ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={local_size}, local_size_y={local_size}, local_size_z=1) in;
            uniform ivec2 window_origin;
            uniform ivec2 window_size;
            uniform int component_count;
            uniform int dst_float_offset;
            uniform int dst_float_row_stride;
            layout(binding=0) uniform sampler2D src_texture;
            layout(std430, binding=1) writeonly buffer SlotFloats {{
                float slot_floats[];
            }};
            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= window_size.x || gid.y >= window_size.y) {{
                    return;
                }}
                vec4 sample_value = texelFetch(src_texture, window_origin + gid, 0);
                int dst_index = dst_float_offset + gid.y * dst_float_row_stride + gid.x * component_count;
                if (component_count > 0) {{
                    slot_floats[dst_index] = sample_value.x;
                }}
                if (component_count > 1) {{
                    slot_floats[dst_index + 1] = sample_value.y;
                }}
                if (component_count > 2) {{
                    slot_floats[dst_index + 2] = sample_value.z;
                }}
                if (component_count > 3) {{
                    slot_floats[dst_index + 3] = sample_value.w;
                }}
            }}
            """
        )

    def _pack_cell_core_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUCellCoreWindowReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        height, width = source.shape[:2]
        if width <= 0 or height <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("cell_core_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing cell core readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
            return
        src_buffer.bind_to_storage_buffer(binding=0)
        slot_buffer.bind_to_storage_buffer(binding=1)
        program["window_origin"].value = (source.origin_x, source.origin_y)
        program["window_size"].value = (width, height)
        program["cell_grid_width"].value = source.cell_grid_width
        program["dst_word_offset"].value = offset // 4
        program["dst_cell_grid_width"].value = int(source.dst_cell_grid_width or width)
        group_x = (width + 7) // 8
        group_y = (height + 7) // 8
        program.run(group_x, group_y, 1)
        self.ctx.memory_barrier()

    def _pack_gas_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUGasWindowReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        height, width = source.shape
        if width <= 0 or height <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("gas_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing gas readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
            return
        src_buffer.bind_to_storage_buffer(binding=0)
        slot_buffer.bind_to_storage_buffer(binding=1)
        program["window_origin"].value = (source.origin_x, source.origin_y)
        program["window_size"].value = (width, height)
        program["gas_grid_size"].value = (source.gas_grid_width, source.gas_grid_height)
        program["species_id"].value = source.species_id
        program["dst_word_offset"].value = offset // 4
        group_x = (width + 7) // 8
        group_y = (height + 7) // 8
        program.run(group_x, group_y, 1)
        self.ctx.memory_barrier()

    def _pack_buffer_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUBufferReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        dtype = np.dtype(source.dtype)
        if (
            offset % 4 != 0
            or source.start % 4 != 0
            or source.step % 4 != 0
            or source.chunk_size % 4 != 0
            or dtype.itemsize != 4
        ):
            self._raise_gpu_readback_unavailable(source, "unsupported buffer copy alignment or element size")
            return
        if source.chunk_size <= 0 or source.count <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("buffer_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing buffer readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
            return
        src_buffer.bind_to_storage_buffer(binding=0)
        slot_buffer.bind_to_storage_buffer(binding=1)
        program["src_word_offset"].value = source.start // 4
        program["src_word_stride"].value = source.step // 4
        program["dst_word_offset"].value = offset // 4
        program["dst_words_per_row"].value = source.chunk_size // 4
        program["dst_word_stride"].value = (source.dst_step or source.chunk_size) // 4
        program["row_count"].value = source.count
        group_x = ((source.chunk_size // 4) + 7) // 8
        group_y = (source.count + 7) // 8
        program.run(group_x, group_y, 1)
        self.ctx.memory_barrier()

    def _pack_texture_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUTextureReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        origin_x, origin_y, width, height = source.viewport
        if width <= 0 or height <= 0 or source.components <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("texture_window")
        if program is None or source.components > 4:
            self._raise_gpu_readback_unavailable(source, "missing texture readback shader or unsupported component count")
            return
        texture = self.textures.get(source.resource_name)
        if texture is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU texture")
            return
        texture.use(location=0)
        slot_buffer.bind_to_storage_buffer(binding=1)
        program["src_texture"].value = 0
        program["window_origin"].value = (origin_x, origin_y)
        program["window_size"].value = (width, height)
        program["component_count"].value = source.components
        program["dst_float_offset"].value = offset // 4
        program["dst_float_row_stride"].value = (source.dst_step or (width * source.components * 4)) // 4
        group_x = (width + 7) // 8
        group_y = (height + 7) // 8
        program.run(group_x, group_y, 1)
        self.ctx.memory_barrier()

    def _pack_segmented_cell_core_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUSegmentedCellCoreWindowReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        height, width = source.shape[:2]
        if width <= 0 or height <= 0:
            return
        for segment in source.segments:
            if segment.width <= 0 or segment.height <= 0:
                continue
            segment_offset = offset + ((int(segment.dst_y) * width + int(segment.dst_x)) * 5 * 4)
            self._pack_cell_core_window_into_buffer(
                slot_buffer,
                segment_offset,
                GPUCellCoreWindowReadbackSource(
                    resource_name=source.resource_name,
                    dtype=source.dtype,
                    shape=(int(segment.height), int(segment.width), 5),
                    cell_grid_width=source.cell_grid_width,
                    origin_x=int(segment.src_x),
                    origin_y=int(segment.src_y),
                    dst_cell_grid_width=width,
                ),
                require_gpu_source=require_gpu_source,
            )

    def _pack_segmented_buffer_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUSegmentedBufferReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        dtype = np.dtype(source.dtype)
        if dtype.itemsize != 4:
            self._raise_gpu_readback_unavailable(source, "unsupported segmented buffer element size")
            return
        if len(source.shape) < 2:
            self._raise_gpu_readback_unavailable(source, "segmented buffer source requires a 2D destination")
            return
        width = int(source.shape[1])
        height = int(source.shape[0])
        if width <= 0 or height <= 0:
            return
        itemsize = dtype.itemsize
        for segment in source.segments:
            if segment.width <= 0 or segment.height <= 0:
                continue
            src_start = int(source.base_offset) + (int(segment.src_y) * int(source.grid_width) + int(segment.src_x)) * itemsize
            dst_offset = offset + (int(segment.dst_y) * width + int(segment.dst_x)) * itemsize
            self._pack_buffer_window_into_buffer(
                slot_buffer,
                dst_offset,
                GPUBufferReadbackSource(
                    resource_name=source.resource_name,
                    dtype=source.dtype,
                    shape=(int(segment.height), int(segment.width)),
                    chunk_size=int(segment.width) * itemsize,
                    start=src_start,
                    step=int(source.grid_width) * itemsize,
                    count=int(segment.height),
                    dst_step=width * itemsize,
                ),
                require_gpu_source=require_gpu_source,
            )

    def _pack_segmented_texture_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUSegmentedTextureReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if source.components <= 0:
            return
        if len(source.shape) < 2:
            self._raise_gpu_readback_unavailable(source, "segmented texture source requires a 2D destination")
            return
        width = int(source.shape[1])
        height = int(source.shape[0])
        if width <= 0 or height <= 0:
            return
        row_step = width * int(source.components) * 4
        for segment in source.segments:
            if segment.width <= 0 or segment.height <= 0:
                continue
            dst_offset = offset + (int(segment.dst_y) * width + int(segment.dst_x)) * int(source.components) * 4
            segment_shape: tuple[int, ...]
            if int(source.components) == 1 and len(source.shape) == 2:
                segment_shape = (int(segment.height), int(segment.width))
            else:
                segment_shape = (int(segment.height), int(segment.width), int(source.components))
            self._pack_texture_window_into_buffer(
                slot_buffer,
                dst_offset,
                GPUTextureReadbackSource(
                    resource_name=source.resource_name,
                    dtype=source.dtype,
                    shape=segment_shape,
                    components=int(source.components),
                    viewport=(int(segment.src_x), int(segment.src_y), int(segment.width), int(segment.height)),
                    dst_step=row_step,
                ),
                require_gpu_source=require_gpu_source,
            )

    @staticmethod
    def _raise_gpu_readback_unavailable(
        source: GPUBufferReadbackSource
        | GPUCellCoreWindowReadbackSource
        | GPUGasWindowReadbackSource
        | GPUTextureReadbackSource
        | GPUSegmentedBufferReadbackSource
        | GPUSegmentedCellCoreWindowReadbackSource
        | GPUSegmentedTextureReadbackSource,
        reason: str,
    ) -> None:
        raise RuntimeError(
            f"GPU readback requires GPU source '{source.resource_name}' ({reason}); CPU fallback is disabled"
        )

    def _release_readback_programs(self) -> None:
        for program in self.readback_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.readback_programs.clear()

    def _release_display_programs(self) -> None:
        for program in self.display_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.display_programs.clear()

    def _release_active_scheduler_programs(self) -> None:
        for program in self.active_scheduler_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.active_scheduler_programs.clear()
