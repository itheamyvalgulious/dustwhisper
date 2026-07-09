from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.readback_contract import READBACK_CHANNEL_BITS
from oracle_game.types import COLLAPSE_BEHAVIOR_IDS, PageStripeUpdate

from oracle_game.gpu._common import _json_bytes
from oracle_game.gpu.dtypes import (
    ENTITY_STATE_DTYPE,
    FORCE_SOURCE_DTYPE,
    ISLAND_RUNTIME_DTYPE,
    FRAME_META_DTYPE,
    WORLD_COMMAND_DTYPE,
    READBACK_REQUEST_DTYPE,
    PLACEHOLDER_DTYPE,
    PLACEHOLDER_DIRTY_RECT_DTYPE,
    ACTIVE_META_DTYPE,
    ACTIVE_RECT_DTYPE,
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
    RULE_TABLE_META_DTYPE,
    MATERIAL_TABLE_DTYPE,
    GAS_TABLE_DTYPE,
    LIGHT_TABLE_DTYPE,
    OPTICS_TABLE_DTYPE,
    REACTION_ACTION_TABLE_DTYPE,
    PAIR_REACTION_RULE_TABLE_DTYPE,
    SELF_REACTION_RULE_TABLE_DTYPE,
)


def _pack_half2x16(velocity: np.ndarray) -> np.ndarray:
    half = velocity.astype(np.float16)
    raw = half.view(np.uint16)
    return (raw[..., 0].astype(np.uint32) | (raw[..., 1].astype(np.uint32) << 16)).astype(np.uint32)


def _unpack_half2x16(word: np.ndarray) -> np.ndarray:
    pair = np.empty(word.shape + (2,), dtype=np.uint16)
    pair[..., 0] = (word & 0xFFFF).astype(np.uint16)
    pair[..., 1] = ((word >> 16) & 0xFFFF).astype(np.uint16)
    return pair.view(np.float16).astype(np.float32)


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


REACTION_ACTION_FLAG_RANDOM_TARGET = 1 << 0
REACTION_ACTION_FLAG_ALLOW_SUBUNIT_SCALE = 1 << 1


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
