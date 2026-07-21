from __future__ import annotations

from typing import Any

import numpy as np

from oracle_game.types import Phase


LOCAL_SIZE = 8
COMPACT_LOCAL_SIZE = 256
COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER = "collapse_structure_dirty_tile_mask"
COLLAPSE_STRUCTURE_MATERIAL_FLAGS_BUFFER = "collapse_structure_material_flags"
COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER = "collapse_structure_dirty_tile_count"
COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER = "collapse_structure_dirty_tile_list"
COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER = "collapse_structure_dirty_tile_dispatch_args"
COLLAPSE_STRUCTURE_ACTIVE_TILE_COUNT_BUFFER = "collapse_structure_dirty_active_tile_count"
COLLAPSE_STRUCTURE_ACTIVE_TILE_LIST_BUFFER = "collapse_structure_dirty_active_tile_list"
COLLAPSE_STRUCTURE_ACTIVE_TILE_DISPATCH_ARGS_BUFFER = "collapse_structure_dirty_active_tile_dispatch_args"
COLLAPSE_STRUCTURE_ACTIVE_TILE_FLAGS_BUFFER = "collapse_structure_dirty_active_tile_flags"
COLLAPSE_STRUCTURE_GUARDED_DIRTY_DISPATCH_ARGS_BUFFER = "collapse_structure_guarded_dirty_dispatch_args"
COLLAPSE_STRUCTURE_DIRTY_TILE_BOUNDS_ATTR = "_gpu_collapse_structure_dirty_tile_bounds"


def _formal_gpu_frame(world: "WorldEngine") -> bool:
    return (
        getattr(world, "simulation_backend", "") == "gpu"
        and bool(getattr(world, "_world_simulation_frame_active", False))
    )


def _active_scheduler_gpu_authoritative(world: "WorldEngine") -> bool:
    authoritative = world.bridge.gpu_authoritative_resources
    return (
        _formal_gpu_frame(world)
        and "active_meta" in authoritative
        and "active_tile_ttl" in authoritative
        and "active_chunk_mask" in authoritative
    )


def _active_tile_workgroups_per_tile(world: "WorldEngine") -> int:
    tile_size = max(1, int(world.active.tile_size))
    axis = max(1, (tile_size + LOCAL_SIZE - 1) // LOCAL_SIZE)
    return axis * axis


def clear_collapse_structure_dirty_tile_bounds(world: "WorldEngine") -> None:
    setattr(world, COLLAPSE_STRUCTURE_DIRTY_TILE_BOUNDS_ATTR, None)


def set_collapse_structure_dirty_tile_bounds(
    world: "WorldEngine",
    tile_bounds: tuple[int, int, int, int] | None,
) -> None:
    if tile_bounds is None:
        clear_collapse_structure_dirty_tile_bounds(world)
        return
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    x0, y0, x1, y1 = (int(value) for value in tile_bounds)
    bounds = (
        max(0, min(tile_width, x0)),
        max(0, min(tile_height, y0)),
        max(0, min(tile_width, x1)),
        max(0, min(tile_height, y1)),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        clear_collapse_structure_dirty_tile_bounds(world)
        return
    setattr(world, COLLAPSE_STRUCTURE_DIRTY_TILE_BOUNDS_ATTR, bounds)


def get_collapse_structure_dirty_tile_bounds(world: "WorldEngine") -> tuple[int, int, int, int] | None:
    bounds = getattr(world, COLLAPSE_STRUCTURE_DIRTY_TILE_BOUNDS_ATTR, None)
    if bounds is None:
        return None
    try:
        x0, y0, x1, y1 = (int(value) for value in bounds)
    except (TypeError, ValueError):
        return None
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def merge_collapse_structure_dirty_tile_bounds(
    world: "WorldEngine",
    tile_bounds: tuple[int, int, int, int] | None,
) -> None:
    if tile_bounds is None:
        return
    existing = get_collapse_structure_dirty_tile_bounds(world)
    if existing is None:
        set_collapse_structure_dirty_tile_bounds(world, tile_bounds)
        return
    x0, y0, x1, y1 = (int(value) for value in tile_bounds)
    ex0, ey0, ex1, ey1 = existing
    set_collapse_structure_dirty_tile_bounds(
        world,
        (min(ex0, x0), min(ey0, y0), max(ex1, x1), max(ey1, y1)),
    )


def _active_source_tile_bounds(
    world: "WorldEngine",
    *,
    expansion_radius: int,
) -> tuple[int, int, int, int] | None:
    tile_width = max(1, int(getattr(world.active, "tile_width", 1)))
    tile_height = max(1, int(getattr(world.active, "tile_height", 1)))
    bridge = world.bridge
    ttl_source = bridge.shadow_buffers.get("active_tile_ttl")
    if not isinstance(ttl_source, np.ndarray) or ttl_source.shape != (tile_height, tile_width):
        ttl_source = np.asarray(getattr(world.active, "active_tile_ttl", []), dtype=np.int32)
    if isinstance(ttl_source, np.ndarray) and ttl_source.shape == (tile_height, tile_width):
        ys, xs = np.nonzero(ttl_source > 0)
        if xs.size > 0 and ys.size > 0:
            margin = max(0, int(expansion_radius)) + 1
            return (
                max(0, int(xs.min()) - margin),
                max(0, int(ys.min()) - margin),
                min(tile_width, int(xs.max()) + margin + 1),
                min(tile_height, int(ys.max()) + margin + 1),
            )

    chunk_width = max(1, int(getattr(world.active, "chunk_width", 1)))
    chunk_height = max(1, int(getattr(world.active, "chunk_height", 1)))
    chunk_tiles = max(1, int(getattr(world.active, "chunk_tiles", 1)))
    chunk_source = bridge.shadow_buffers.get("active_chunk_mask")
    if not isinstance(chunk_source, np.ndarray) or chunk_source.shape != (chunk_height, chunk_width):
        chunk_source = np.asarray(getattr(world.active, "active_chunk_mask", []), dtype=np.uint8)
    if not isinstance(chunk_source, np.ndarray) or chunk_source.shape != (chunk_height, chunk_width):
        return None
    chunk_ys, chunk_xs = np.nonzero(chunk_source > 0)
    if chunk_xs.size <= 0 or chunk_ys.size <= 0:
        return None
    margin = max(0, int(expansion_radius)) + 1
    return (
        max(0, int(chunk_xs.min()) * chunk_tiles - margin),
        max(0, int(chunk_ys.min()) * chunk_tiles - margin),
        min(tile_width, (int(chunk_xs.max()) + 1) * chunk_tiles + margin),
        min(tile_height, (int(chunk_ys.max()) + 1) * chunk_tiles + margin),
    )


def _shader_storage_barrier(ctx: Any, *, command: bool = False) -> None:
    flags = getattr(ctx, "SHADER_STORAGE_BARRIER_BIT", 0) | getattr(ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
    if command:
        flags |= getattr(ctx, "COMMAND_BARRIER_BIT", 0)
    ctx.memory_barrier(flags)


def ensure_collapse_structure_dirty_tile_mask(
    world: "WorldEngine",
    *,
    clear: bool = False,
) -> Any | None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        return None
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    required_bytes = tile_count * np.dtype(np.uint32).itemsize
    existing = bridge.buffers.get(COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER)
    created = existing is None or existing.size < required_bytes
    if created:
        if existing is not None:
            existing.release()
        existing = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER] = existing
    if created or clear:
        existing.write(np.zeros(tile_count, dtype=np.uint32).tobytes())
        if clear:
            setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", False)
            setattr(world, "_gpu_collapse_structure_dirty_tiles_deferred", False)
            clear_collapse_structure_dirty_tile_bounds(world)
    bridge.mark_gpu_authoritative(COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER)
    return existing


def clear_collapse_structure_dirty_tile_mask(world: "WorldEngine") -> None:
    ensure_collapse_structure_dirty_tile_mask(world, clear=True)
    bridge = world.bridge
    if not bridge.enabled or bridge.ctx is None:
        return
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    for name, payload in (
        (COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER, np.zeros(1, dtype=np.uint32).tobytes()),
        (COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER, np.zeros(tile_count * 2, dtype=np.int32).tobytes()),
        (
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
            np.asarray([0, 1, 1], dtype=np.uint32).tobytes(),
        ),
    ):
        buffer = bridge.buffers.get(name)
        if buffer is not None:
            buffer.write(payload)
    bridge.mark_gpu_authoritative(
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    )


def ensure_collapse_structure_dirty_tile_queue(
    world: "WorldEngine",
    *,
    clear: bool = False,
) -> tuple[Any, Any, Any] | None:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        return None
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    specs = (
        (COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER, 4),
        (COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER, max(8, tile_count * 2 * np.dtype(np.int32).itemsize)),
        (COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER, 3 * np.dtype(np.uint32).itemsize),
    )
    buffers: dict[str, Any] = {}
    for name, required_bytes in specs:
        existing = bridge.buffers.get(name)
        created = existing is None or existing.size < required_bytes
        if created:
            if existing is not None:
                existing.release()
            existing = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            bridge.buffers[name] = existing
        if created or clear:
            if name == COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER:
                existing.write(np.zeros(1, dtype=np.uint32).tobytes())
            elif name == COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER:
                existing.write(np.zeros(tile_count * 2, dtype=np.int32).tobytes())
            else:
                existing.write(np.asarray([0, 1, 1], dtype=np.uint32).tobytes())
        buffers[name] = existing
    if clear:
        setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", False)
        setattr(world, "_gpu_collapse_structure_dirty_tiles_deferred", False)
        clear_collapse_structure_dirty_tile_bounds(world)
    bridge.mark_gpu_authoritative(*(name for name, _required_bytes in specs))
    return (
        buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER],
        buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER],
        buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER],
    )


def has_pending_collapse_structure_dirty_tiles(world: "WorldEngine") -> bool:
    if not _formal_gpu_frame(world):
        return False
    if not bool(getattr(world, "_gpu_collapse_structure_dirty_tiles_pending", False)):
        return False
    bridge = world.bridge
    return (
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER in bridge.buffers
        and COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER in bridge.buffers
        and COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER in bridge.buffers
        and COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER in bridge.buffers
    )


def _material_participation_flags(world: "WorldEngine") -> np.ndarray:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is None or material_table.size == 0:
        count = max(
            1,
            int(getattr(world, "material_is_structural", np.zeros(1, dtype=np.bool_)).shape[0]),
            int(getattr(world, "material_is_support_anchor", np.zeros(1, dtype=np.bool_)).shape[0]),
        )
        flags = np.zeros((count,), dtype=np.uint32)
        structural = getattr(world, "material_is_structural", np.zeros(1, dtype=np.bool_))
        support = getattr(world, "material_is_support_anchor", np.zeros(1, dtype=np.bool_))
        flags[: structural.shape[0]] |= structural.astype(np.uint32, copy=False)
        flags[: support.shape[0]] |= support.astype(np.uint32, copy=False) << np.uint32(1)
        return flags
    material_ids = material_table["material_id"].astype(np.int32, copy=False)
    valid_ids = material_ids[material_ids >= 0]
    count = int(valid_ids.max(initial=0)) + 1
    flags = np.zeros((max(1, count),), dtype=np.uint32)
    structural = material_table["is_structural"].astype(np.bool_, copy=False)
    support = material_table["is_support_anchor"].astype(np.bool_, copy=False)
    valid_rows = material_table["name_hash"] != 0 if "name_hash" in material_table.dtype.names else np.ones_like(structural)
    valid_participating = valid_rows & (material_ids >= 0)
    ids = material_ids[valid_participating]
    flags[ids] = (
        structural[valid_participating].astype(np.uint32, copy=False)
        | (support[valid_participating].astype(np.uint32, copy=False) << np.uint32(1))
    )
    return flags


def _ensure_material_flags_buffer(world: "WorldEngine") -> tuple[Any, int]:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse dirty tracking requires bridge GPU resources")
    flags = _material_participation_flags(world)
    required_bytes = max(4, flags.nbytes)
    existing = bridge.buffers.get(COLLAPSE_STRUCTURE_MATERIAL_FLAGS_BUFFER)
    if existing is None or existing.size < required_bytes:
        if existing is not None:
            existing.release()
        existing = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
        bridge.buffers[COLLAPSE_STRUCTURE_MATERIAL_FLAGS_BUFFER] = existing
    existing.write(flags.tobytes())
    bridge.mark_gpu_authoritative(COLLAPSE_STRUCTURE_MATERIAL_FLAGS_BUFFER)
    return existing, int(flags.size)


def _ensure_active_tile_dispatch_buffers(world: "WorldEngine") -> tuple[Any, Any, Any, Any]:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse dirty tracking requires bridge GPU resources")
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    specs = (
        (COLLAPSE_STRUCTURE_ACTIVE_TILE_COUNT_BUFFER, 4),
        (COLLAPSE_STRUCTURE_ACTIVE_TILE_LIST_BUFFER, max(8, tile_count * 2 * np.dtype(np.int32).itemsize)),
        (COLLAPSE_STRUCTURE_ACTIVE_TILE_DISPATCH_ARGS_BUFFER, 3 * np.dtype(np.uint32).itemsize),
        (COLLAPSE_STRUCTURE_ACTIVE_TILE_FLAGS_BUFFER, max(4, tile_count * np.dtype(np.uint32).itemsize)),
    )
    buffers: dict[str, Any] = {}
    for name, required_bytes in specs:
        existing = bridge.buffers.get(name)
        if existing is None or existing.size < required_bytes:
            if existing is not None:
                existing.release()
            existing = bridge.ctx.buffer(reserve=required_bytes, dynamic=True)
            existing.write(bytes(required_bytes))
            bridge.buffers[name] = existing
        buffers[name] = existing
    bridge.mark_gpu_authoritative(*(name for name, _required_bytes in specs))
    return (
        buffers[COLLAPSE_STRUCTURE_ACTIVE_TILE_COUNT_BUFFER],
        buffers[COLLAPSE_STRUCTURE_ACTIVE_TILE_LIST_BUFFER],
        buffers[COLLAPSE_STRUCTURE_ACTIVE_TILE_DISPATCH_ARGS_BUFFER],
        buffers[COLLAPSE_STRUCTURE_ACTIVE_TILE_FLAGS_BUFFER],
    )


def _collapse_dirty_program(world: "WorldEngine", key: str, source: str) -> Any:
    bridge = world.bridge
    ctx = bridge.ctx
    if ctx is None:
        raise RuntimeError("GPU collapse dirty tracking requires bridge GPU resources")
    program = bridge.active_scheduler_programs.get(key)
    if program is not None:
        return program
    program = ctx.compute_shader(source)
    bridge.active_scheduler_programs[key] = program
    return program


def _clear_active_tile_dispatch_program(world: "WorldEngine") -> Any:
    return _collapse_dirty_program(
        world,
        "collapse_structure_dirty_clear_active_tile_dispatch",
        f"""
        #version 430
        layout(local_size_x={COMPACT_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
        uniform int tile_count;
        layout(std430, binding=0) buffer ActiveTileCountBuffer {{
            uint active_tile_count[];
        }};
        layout(std430, binding=1) buffer ActiveTileDispatchArgsBuffer {{
            uint active_tile_dispatch_args[];
        }};
        layout(std430, binding=2) buffer ActiveTileFlagsBuffer {{
            uint active_tile_flags[];
        }};
        void main() {{
            uint index = gl_GlobalInvocationID.x;
            if (index == 0u) {{
                active_tile_count[0] = 0u;
                active_tile_dispatch_args[0] = 0u;
                active_tile_dispatch_args[1] = 1u;
                active_tile_dispatch_args[2] = 1u;
            }}
            if (index < uint(tile_count)) {{
                active_tile_flags[index] = 0u;
            }}
        }}
        """,
    )


def _compact_active_tiles_from_chunks_program(world: "WorldEngine") -> Any:
    return _collapse_dirty_program(
        world,
        "collapse_structure_dirty_compact_active_tiles_from_chunks",
        f"""
        #version 430
        layout(local_size_x={COMPACT_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
        uniform ivec2 tile_grid_size;
        uniform int chunk_tiles;
        uniform int expansion_radius;
        uniform uint workgroups_per_tile;
        layout(std430, binding=0) buffer ActiveTileCountBuffer {{
            uint active_tile_count[];
        }};
        layout(std430, binding=1) buffer ActiveTileListBuffer {{
            ivec2 active_tile_list[];
        }};
        layout(std430, binding=2) buffer ActiveTileDispatchArgsBuffer {{
            uint active_tile_dispatch_args[];
        }};
        layout(std430, binding=3) readonly buffer ActiveChunkCountBuffer {{
            uint active_chunk_count[];
        }};
        layout(std430, binding=4) readonly buffer ActiveChunkListBuffer {{
            ivec2 active_chunk_list[];
        }};
        layout(std430, binding=5) readonly buffer ActiveTileTTLBuffer {{
            int active_tile_ttl[];
        }};
        layout(std430, binding=6) buffer ActiveTileFlagsBuffer {{
            uint active_tile_flags[];
        }};
        void append_tile(ivec2 tile) {{
            if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                return;
            }}
            int tile_index = tile.y * tile_grid_size.x + tile.x;
            if (atomicCompSwap(active_tile_flags[tile_index], 0u, 1u) != 0u) {{
                return;
            }}
            uint slot = atomicAdd(active_tile_count[0], 1u);
            active_tile_list[int(slot)] = tile;
            atomicMax(active_tile_dispatch_args[0], (slot + 1u) * max(workgroups_per_tile, 1u));
        }}
        void main() {{
            uint chunk_index = gl_WorkGroupID.x;
            if (chunk_index >= active_chunk_count[0]) {{
                return;
            }}
            ivec2 chunk = active_chunk_list[int(chunk_index)];
            int chunk_side = max(chunk_tiles, 1);
            int tile_slots = chunk_side * chunk_side;
            for (int tile_slot = int(gl_LocalInvocationIndex); tile_slot < tile_slots; tile_slot += {COMPACT_LOCAL_SIZE}) {{
                ivec2 local_tile = ivec2(tile_slot % chunk_side, tile_slot / chunk_side);
                ivec2 source_tile = chunk * chunk_side + local_tile;
                if (
                    source_tile.x < 0
                    || source_tile.y < 0
                    || source_tile.x >= tile_grid_size.x
                    || source_tile.y >= tile_grid_size.y
                ) {{
                    continue;
                }}
                int source_index = source_tile.y * tile_grid_size.x + source_tile.x;
                if (active_tile_ttl[source_index] <= 0) {{
                    continue;
                }}
                int radius = max(0, expansion_radius);
                for (int dy = -radius; dy <= radius; ++dy) {{
                    for (int dx = -radius; dx <= radius; ++dx) {{
                        append_tile(source_tile + ivec2(dx, dy));
                    }}
                }}
            }}
        }}
        """,
    )


def _dirty_mark_program(world: "WorldEngine", *, packed_cell_state: bool = False) -> Any:
    state_declarations = (
        "layout(binding=0) uniform usampler2D cell_state_tex;"
        if packed_cell_state
        else "layout(binding=0) uniform sampler2D material_tex;\n"
        "layout(binding=1) uniform sampler2D phase_tex;"
    )
    state_fetch = (
        "uint cell_state = texelFetch(cell_state_tex, gid, 0).x;\n"
        "            int new_material = int(cell_state & 0xFFFFu);\n"
        "            int new_phase = int((cell_state >> 16u) & 0xFFu);"
        if packed_cell_state
        else "int new_material = int(clamp(round(texelFetch(material_tex, gid, 0).x), 0.0, 65535.0));\n"
        "            int new_phase = int(clamp(round(texelFetch(phase_tex, gid, 0).x), 0.0, 255.0));"
    )
    return _collapse_dirty_program(
        world,
        "collapse_structure_dirty_tiles_packed" if packed_cell_state else "collapse_structure_dirty_tiles",
        f"""
        #version 430
        layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
        uniform ivec2 cell_grid_size;
        uniform ivec2 tile_grid_size;
        uniform int tile_size;
        uniform int material_count;
        uniform int phase_falling_island;
        {state_declarations}
        layout(std430, binding=0) readonly buffer BridgeCellCoreBuffer {{
            uint bridge_cell_core[];
        }};
        layout(std430, binding=1) readonly buffer MaterialParticipationBuffer {{
            uint material_participates[];
        }};
        layout(std430, binding=2) buffer DirtyTileMaskBuffer {{
            uint dirty_tile_mask[];
        }};
        layout(std430, binding=3) readonly buffer ActiveTileCountBuffer {{
            uint active_tile_count[];
        }};
        layout(std430, binding=4) readonly buffer ActiveTileListBuffer {{
            ivec2 active_tile_list[];
        }};
        layout(std430, binding=5) buffer DirtyTileCountBuffer {{
            uint dirty_tile_count[];
        }};
        layout(std430, binding=6) buffer DirtyTileListBuffer {{
            ivec2 dirty_tile_list[];
        }};
        layout(std430, binding=7) buffer DirtyTileDispatchArgsBuffer {{
            uint dirty_tile_dispatch_args[];
        }};
        uint structure_flags(int material_id, int phase_value) {{
            if (material_id <= 0 || material_id >= material_count || phase_value == phase_falling_island) {{
                return 0u;
            }}
            return material_participates[material_id];
        }}
        void append_dirty_tile(ivec2 tile) {{
            if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                return;
            }}
            int tile_index = tile.y * tile_grid_size.x + tile.x;
            uint previous = atomicOr(dirty_tile_mask[tile_index], 1u);
            if ((previous & 1u) != 0u) {{
                return;
            }}
            uint slot = atomicAdd(dirty_tile_count[0], 1u);
            dirty_tile_list[int(slot)] = tile;
            atomicMax(dirty_tile_dispatch_args[0], slot + 1u);
        }}
        void main() {{
            int groups_per_tile_axis = max(1, (tile_size + {LOCAL_SIZE - 1}) / {LOCAL_SIZE});
            int workgroups_per_tile = max(1, groups_per_tile_axis * groups_per_tile_axis);
            uint active_tile_index = gl_WorkGroupID.x / uint(workgroups_per_tile);
            if (active_tile_index >= active_tile_count[0]) {{
                return;
            }}
            int subtile = int(gl_WorkGroupID.x % uint(workgroups_per_tile));
            ivec2 tile = active_tile_list[int(active_tile_index)];
            if (tile.x < 0 || tile.y < 0 || tile.x >= tile_grid_size.x || tile.y >= tile_grid_size.y) {{
                return;
            }}
            ivec2 tile_origin = tile * tile_size;
            ivec2 tile_end = min(tile_origin + ivec2(tile_size), cell_grid_size);
            ivec2 local_group = ivec2(subtile % groups_per_tile_axis, subtile / groups_per_tile_axis);
            ivec2 gid = tile_origin + local_group * {LOCAL_SIZE} + ivec2(gl_LocalInvocationID.xy);
            if (gid.x >= tile_end.x || gid.y >= tile_end.y) {{
                return;
            }}
            int cell_index = gid.y * cell_grid_size.x + gid.x;
            uint previous_word = bridge_cell_core[cell_index * 5];
            int old_material = int(previous_word & 0xFFFFu);
            int old_phase = int((previous_word >> 16u) & 0xFFu);
            {state_fetch}
            uint old_flags = structure_flags(old_material, old_phase);
            uint new_flags = structure_flags(new_material, new_phase);
            if (old_flags == new_flags) {{
                return;
            }}
            ivec2 local_cell = gid - tile_origin;
            append_dirty_tile(tile);
            if (local_cell.x == 0) {{
                append_dirty_tile(tile + ivec2(-1, 0));
            }}
            if (local_cell.x + 1 >= tile_size) {{
                append_dirty_tile(tile + ivec2(1, 0));
            }}
            if (local_cell.y == 0) {{
                append_dirty_tile(tile + ivec2(0, -1));
            }}
            if (local_cell.y + 1 >= tile_size) {{
                append_dirty_tile(tile + ivec2(0, 1));
            }}
        }}
        """,
    )


def _guarded_dirty_dispatch_args(world: "WorldEngine", guard_buffer: Any, source_args: Any) -> Any:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse dirty tracking requires bridge GPU resources")
    existing = bridge.buffers.get(COLLAPSE_STRUCTURE_GUARDED_DIRTY_DISPATCH_ARGS_BUFFER)
    required_bytes = 3 * np.dtype(np.uint32).itemsize
    if existing is None or existing.size < required_bytes:
        if existing is not None:
            existing.release()
        existing = bridge.ctx.buffer(np.asarray([0, 1, 1], dtype=np.uint32).tobytes(), dynamic=True)
        bridge.buffers[COLLAPSE_STRUCTURE_GUARDED_DIRTY_DISPATCH_ARGS_BUFFER] = existing
    program = _collapse_dirty_program(
        world,
        "collapse_structure_dirty_guard_dispatch_args",
        """
        #version 430
        layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        layout(std430, binding=0) readonly buffer DispatchGuard {
            uint dispatch_guard[];
        };
        layout(std430, binding=1) readonly buffer SourceArgs {
            uint source_args[];
        };
        layout(std430, binding=2) buffer GuardedArgs {
            uint guarded_args[];
        };
        void main() {
            if (dispatch_guard[0] == 0u) {
                guarded_args[0] = 0u;
                guarded_args[1] = 1u;
                guarded_args[2] = 1u;
                return;
            }
            guarded_args[0] = source_args[0];
            guarded_args[1] = max(source_args[1], 1u);
            guarded_args[2] = max(source_args[2], 1u);
        }
        """,
    )
    guard_buffer.bind_to_storage_buffer(binding=0)
    source_args.bind_to_storage_buffer(binding=1)
    existing.bind_to_storage_buffer(binding=2)
    program.run(1, 1, 1)
    _shader_storage_barrier(bridge.ctx, command=True)
    bridge.mark_gpu_authoritative(COLLAPSE_STRUCTURE_GUARDED_DIRTY_DISPATCH_ARGS_BUFFER)
    return existing


def _clear_dirty_tile_queue_program(world: "WorldEngine") -> Any:
    return _collapse_dirty_program(
        world,
        "collapse_structure_dirty_clear_dirty_tile_queue",
        f"""
        #version 430
        layout(local_size_x={COMPACT_LOCAL_SIZE}, local_size_y=1, local_size_z=1) in;
        uniform int tile_count;
        layout(std430, binding=0) buffer DirtyTileMaskBuffer {{
            uint dirty_tile_mask[];
        }};
        layout(std430, binding=1) buffer DirtyTileCountBuffer {{
            uint dirty_tile_count[];
        }};
        layout(std430, binding=2) buffer DirtyTileListBuffer {{
            ivec2 dirty_tile_list[];
        }};
        layout(std430, binding=3) buffer DirtyTileDispatchArgsBuffer {{
            uint dirty_tile_dispatch_args[];
        }};
        void main() {{
            uint index = gl_GlobalInvocationID.x;
            if (index == 0u) {{
                dirty_tile_count[0] = 0u;
                dirty_tile_dispatch_args[0] = 0u;
                dirty_tile_dispatch_args[1] = 1u;
                dirty_tile_dispatch_args[2] = 1u;
            }}
            if (index < uint(max(tile_count, 0))) {{
                dirty_tile_mask[int(index)] = 0u;
                dirty_tile_list[int(index)] = ivec2(0);
            }}
        }}
        """,
    )


def clear_collapse_structure_dirty_tile_queue_on_gpu(world: "WorldEngine") -> bool:
    if not _formal_gpu_frame(world):
        return False
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        return False
    dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
    dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
    if dirty_buffer is None or dirty_queue is None:
        return False
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    program = _clear_dirty_tile_queue_program(world)
    program["tile_count"].value = int(tile_count)
    dirty_buffer.bind_to_storage_buffer(binding=0)
    dirty_count.bind_to_storage_buffer(binding=1)
    dirty_list.bind_to_storage_buffer(binding=2)
    dirty_dispatch_args.bind_to_storage_buffer(binding=3)
    program.run((tile_count + COMPACT_LOCAL_SIZE - 1) // COMPACT_LOCAL_SIZE, 1, 1)
    _shader_storage_barrier(bridge.ctx, command=True)
    setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", False)
    setattr(world, "_gpu_collapse_structure_dirty_tiles_deferred", False)
    clear_collapse_structure_dirty_tile_bounds(world)
    bridge.mark_gpu_authoritative(
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    )
    return True


def _mark_dirty_tile_rect_program(world: "WorldEngine") -> Any:
    return _collapse_dirty_program(
        world,
        "collapse_structure_dirty_mark_tile_rect",
        f"""
        #version 430
        layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
        uniform ivec4 tile_rect;
        uniform ivec2 tile_grid_size;
        layout(std430, binding=0) buffer DirtyTileMaskBuffer {{
            uint dirty_tile_mask[];
        }};
        layout(std430, binding=1) buffer DirtyTileCountBuffer {{
            uint dirty_tile_count[];
        }};
        layout(std430, binding=2) buffer DirtyTileListBuffer {{
            ivec2 dirty_tile_list[];
        }};
        layout(std430, binding=3) buffer DirtyTileDispatchArgsBuffer {{
            uint dirty_tile_dispatch_args[];
        }};
        void main() {{
            ivec2 tile = tile_rect.xy + ivec2(gl_GlobalInvocationID.xy);
            if (tile.x >= tile_rect.z || tile.y >= tile_rect.w) {{
                return;
            }}
            int tile_index = tile.y * tile_grid_size.x + tile.x;
            uint previous = atomicOr(dirty_tile_mask[tile_index], 1u);
            if ((previous & 1u) != 0u) {{
                return;
            }}
            uint slot = atomicAdd(dirty_tile_count[0], 1u);
            dirty_tile_list[int(slot)] = tile;
            atomicMax(dirty_tile_dispatch_args[0], slot + 1u);
        }}
        """,
    )


def mark_collapse_structure_dirty_tile_regions_on_gpu(
    world: "WorldEngine",
    regions: list[tuple[int, int, int, int]],
    *,
    deferred: bool = False,
) -> bool:
    if not _formal_gpu_frame(world) or not regions:
        return False
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        return False
    dirty_mask = ensure_collapse_structure_dirty_tile_mask(world)
    dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
    if dirty_mask is None or dirty_queue is None:
        return False
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    tile_size = max(1, int(world.active.tile_size))
    tile_width = max(1, int(world.active.tile_width))
    tile_height = max(1, int(world.active.tile_height))
    program = _mark_dirty_tile_rect_program(world)
    program["tile_grid_size"].value = (tile_width, tile_height)
    dirty_mask.bind_to_storage_buffer(binding=0)
    dirty_count.bind_to_storage_buffer(binding=1)
    dirty_list.bind_to_storage_buffer(binding=2)
    dirty_dispatch_args.bind_to_storage_buffer(binding=3)
    marked = False
    for x0, y0, x1, y1 in regions:
        tile_x0 = max(0, min(tile_width, int(x0) // tile_size))
        tile_y0 = max(0, min(tile_height, int(y0) // tile_size))
        tile_x1 = max(0, min(tile_width, (int(x1) + tile_size - 1) // tile_size))
        tile_y1 = max(0, min(tile_height, (int(y1) + tile_size - 1) // tile_size))
        if tile_x0 >= tile_x1 or tile_y0 >= tile_y1:
            continue
        program["tile_rect"].value = (tile_x0, tile_y0, tile_x1, tile_y1)
        program.run(
            (tile_x1 - tile_x0 + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (tile_y1 - tile_y0 + LOCAL_SIZE - 1) // LOCAL_SIZE,
            1,
        )
        merge_collapse_structure_dirty_tile_bounds(
            world,
            (tile_x0, tile_y0, tile_x1, tile_y1),
        )
        marked = True
    if not marked:
        return False
    _shader_storage_barrier(bridge.ctx, command=True)
    setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
    if deferred:
        setattr(world, "_gpu_collapse_structure_dirty_tiles_deferred", True)
    bridge.mark_gpu_authoritative(
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    )
    return True


def _build_active_tile_dispatch(
    world: "WorldEngine",
    *,
    expansion_radius: int,
) -> tuple[Any, Any, Any]:
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        raise RuntimeError("GPU collapse dirty tracking requires bridge GPU resources")
    bridge._ensure_active_scheduler_programs()
    bridge._refresh_active_chunks_and_meta(world, read_meta=False)

    active_count, active_list, active_dispatch_args, active_flags = _ensure_active_tile_dispatch_buffers(world)
    tile_count = max(1, int(world.active.tile_width) * int(world.active.tile_height))
    clear_program = _clear_active_tile_dispatch_program(world)
    clear_program["tile_count"].value = int(tile_count)
    active_count.bind_to_storage_buffer(binding=0)
    active_dispatch_args.bind_to_storage_buffer(binding=1)
    active_flags.bind_to_storage_buffer(binding=2)
    clear_program.run((tile_count + COMPACT_LOCAL_SIZE - 1) // COMPACT_LOCAL_SIZE, 1, 1)
    _shader_storage_barrier(bridge.ctx, command=True)

    compact_program = _compact_active_tiles_from_chunks_program(world)
    if not hasattr(compact_program, "run_indirect"):
        raise RuntimeError("GPU collapse dirty active tile compaction requires ComputeShader.run_indirect")
    compact_program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
    compact_program["chunk_tiles"].value = int(world.active.chunk_tiles)
    compact_program["expansion_radius"].value = max(0, int(expansion_radius))
    compact_program["workgroups_per_tile"].value = int(_active_tile_workgroups_per_tile(world))
    active_count.bind_to_storage_buffer(binding=0)
    active_list.bind_to_storage_buffer(binding=1)
    active_dispatch_args.bind_to_storage_buffer(binding=2)
    bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
    bridge.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=5)
    active_flags.bind_to_storage_buffer(binding=6)
    compact_program.run_indirect(bridge.buffers["active_chunk_dispatch_args"])
    _shader_storage_barrier(bridge.ctx, command=True)
    bridge.mark_gpu_authoritative(
        COLLAPSE_STRUCTURE_ACTIVE_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_DISPATCH_ARGS_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_FLAGS_BUFFER,
    )
    return active_count, active_list, active_dispatch_args


def mark_collapse_structure_dirty_tiles_from_bridge_cell_core(
    world: "WorldEngine",
    material_texture: Any | None,
    phase_texture: Any | None,
    *,
    expansion_radius: int = 1,
    dispatch_guard_buffer: Any | None = None,
    cell_state_texture: Any | None = None,
) -> bool:
    if not _formal_gpu_frame(world):
        return False
    bridge = world.bridge
    bridge.ensure_world_resources(world)
    if not bridge.enabled or bridge.ctx is None:
        return False
    if "cell_core" not in bridge.gpu_authoritative_resources:
        world._require_gpu_authoritative_resources("collapse structure dirty tracking", "cell_core")
        return False
    if not _active_scheduler_gpu_authoritative(world):
        world._require_gpu_authoritative_resources(
            "collapse structure dirty tracking",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )
        return False
    dirty_buffer = ensure_collapse_structure_dirty_tile_mask(world)
    if dirty_buffer is None:
        return False
    dirty_queue = ensure_collapse_structure_dirty_tile_queue(world)
    if dirty_queue is None:
        return False
    dirty_count, dirty_list, dirty_dispatch_args = dirty_queue
    active_count, active_list, active_dispatch_args = _build_active_tile_dispatch(
        world,
        expansion_radius=expansion_radius,
    )
    material_flags_buffer, material_count = _ensure_material_flags_buffer(world)
    packed_cell_state = cell_state_texture is not None
    if not packed_cell_state and (material_texture is None or phase_texture is None):
        raise ValueError("collapse dirty tracking requires split material/phase or packed cell state")
    program = _dirty_mark_program(world, packed_cell_state=packed_cell_state)
    if not hasattr(program, "run_indirect"):
        raise RuntimeError("GPU collapse dirty tracking requires ComputeShader.run_indirect")
    program["cell_grid_size"].value = (int(world.width), int(world.height))
    program["tile_grid_size"].value = (int(world.active.tile_width), int(world.active.tile_height))
    program["tile_size"].value = int(world.active.tile_size)
    program["material_count"].value = int(material_count)
    program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
    if packed_cell_state:
        cell_state_texture.use(location=0)
    else:
        material_texture.use(location=0)
        phase_texture.use(location=1)
    bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
    material_flags_buffer.bind_to_storage_buffer(binding=1)
    dirty_buffer.bind_to_storage_buffer(binding=2)
    active_count.bind_to_storage_buffer(binding=3)
    active_list.bind_to_storage_buffer(binding=4)
    dirty_count.bind_to_storage_buffer(binding=5)
    dirty_list.bind_to_storage_buffer(binding=6)
    dirty_dispatch_args.bind_to_storage_buffer(binding=7)
    dispatch_args = active_dispatch_args
    if dispatch_guard_buffer is not None:
        dispatch_args = _guarded_dirty_dispatch_args(world, dispatch_guard_buffer, active_dispatch_args)
    program.run_indirect(dispatch_args)
    _shader_storage_barrier(bridge.ctx, command=True)
    setattr(world, "_gpu_collapse_structure_dirty_tiles_pending", True)
    authoritative_names = [
        COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_COUNT_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_LIST_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_DISPATCH_ARGS_BUFFER,
        COLLAPSE_STRUCTURE_ACTIVE_TILE_FLAGS_BUFFER,
    ]
    if dispatch_guard_buffer is not None:
        authoritative_names.append(COLLAPSE_STRUCTURE_GUARDED_DIRTY_DISPATCH_ARGS_BUFFER)
    bridge.mark_gpu_authoritative(*authoritative_names)
    return True


def dirty_tile_mask_to_regions(world: "WorldEngine", tile_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    if tile_mask.size == 0:
        return []
    tile_size = max(1, int(world.active.tile_size))
    regions: list[tuple[int, int, int, int]] = []
    for tile_y in range(tile_mask.shape[0]):
        tile_x = 0
        while tile_x < tile_mask.shape[1]:
            if not bool(tile_mask[tile_y, tile_x]):
                tile_x += 1
                continue
            run_x0 = tile_x
            while tile_x < tile_mask.shape[1] and bool(tile_mask[tile_y, tile_x]):
                tile_x += 1
            x0 = run_x0 * tile_size
            y0 = tile_y * tile_size
            x1 = min(int(world.width), tile_x * tile_size)
            y1 = min(int(world.height), (tile_y + 1) * tile_size)
            if x0 < x1 and y0 < y1:
                regions.append((x0, y0, x1, y1))
    return regions


def drain_collapse_structure_dirty_tile_regions(world: "WorldEngine") -> list[tuple[int, int, int, int]]:
    if not _formal_gpu_frame(world):
        return []
    return []
