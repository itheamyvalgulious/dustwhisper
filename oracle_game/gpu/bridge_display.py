from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import numpy as np

from oracle_game.gpu.dtypes import (
    ACTIVE_META_DTYPE,
    ACTIVE_RECT_DTYPE,
)
from oracle_game.gpu.shader_loader import build_compute_shader


def sync_display_textures(bridge, world: "WorldEngine") -> None:
    """Refresh textures sampled by the desktop demo from GPU-authoritative buffers."""
    if not bridge.enabled or bridge.ctx is None:
        return
    bridge.ensure_world_resources(world)
    if getattr(world, "simulation_backend", "") != "gpu":
        return
    if "cell_core" in bridge.gpu_authoritative_resources and "cell_core" in bridge.buffers:
        bridge._ensure_display_programs()
        program = bridge.display_programs["material_from_cell_core"]
        program["width"] = int(world.width)
        program["height"] = int(world.height)
        bridge.buffers["cell_core"].bind_to_storage_buffer(0)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
        bridge.ctx.memory_barrier(
            bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
        )
    if (
        "visible_illumination" in bridge.gpu_authoritative_resources
        and "visible_illumination" in bridge.textures
    ):
        bridge._ensure_display_programs()
        program = bridge.display_programs["light_from_visible_texture"]
        program["width"] = int(world.width)
        program["height"] = int(world.height)
        bridge.textures["visible_illumination"].use(0)
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
        bridge.ctx.memory_barrier(
            bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
        )


def sync_debug_display_texture(
    bridge,
    world: "WorldEngine",
    *,
    view: str,
    gas_species_id: int = -1,
    light_dose_channel: int = -1,
) -> bool:
    """Refresh the desktop demo debug texture using only GPU-resident state."""
    if not bridge.enabled or bridge.ctx is None:
        return False
    if getattr(world, "simulation_backend", "") != "gpu":
        return False
    bridge.ensure_world_resources(world)
    bridge._ensure_display_programs()
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
    program = bridge.display_programs["debug_from_gpu_state"]
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
    bridge.buffers["cell_core"].bind_to_storage_buffer(0)
    bridge.buffers["gas_concentration"].bind_to_storage_buffer(1)
    bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(2)
    bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(3)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(4)
    bridge.textures["visible_illumination"].use(0)
    bridge.textures["flow_velocity"].use(1)
    bridge.textures["pressure_ping"].use(2)
    bridge.textures["debug"].bind_to_image(0, read=False, write=True)
    program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
    bridge.ctx.memory_barrier(
        bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
    )
    return True


def _ensure_display_programs(bridge) -> None:
    if not bridge.enabled or bridge.ctx is None:
        return
    if "material_from_cell_core" not in bridge.display_programs:
        bridge.display_programs["material_from_cell_core"] = build_compute_shader(
            bridge.ctx,
            "display/material_from_cell_core.comp",
        )
    if "light_from_visible_texture" not in bridge.display_programs:
        bridge.display_programs["light_from_visible_texture"] = build_compute_shader(
            bridge.ctx,
            "display/light_from_visible_texture.comp",
        )
        bridge.display_programs["light_from_visible_texture"]["visible_tex"] = 0
    if "debug_from_gpu_state" not in bridge.display_programs:
        bridge.display_programs["debug_from_gpu_state"] = build_compute_shader(
            bridge.ctx,
            "display/debug_from_gpu_state.comp",
        )
        bridge.display_programs["debug_from_gpu_state"]["visible_tex"] = 0
        bridge.display_programs["debug_from_gpu_state"]["flow_velocity_tex"] = 1
        bridge.display_programs["debug_from_gpu_state"]["pressure_tex"] = 2


def mark_active_rects(
    bridge,
    world: "WorldEngine",
    rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
) -> bool:
    if not rects:
        return True
    if not bridge.enabled or bridge.ctx is None:
        return False
    bridge.ensure_world_resources(world)
    if (
        "active_meta" not in bridge.buffers
        or "active_tile_ttl" not in bridge.buffers
        or "active_chunk_mask" not in bridge.buffers
    ):
        return False
    bridge._ensure_active_scheduler_programs()
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
    bridge._write_dynamic_buffer("active_rect", packed_rects)

    mark_program = bridge.active_scheduler_programs["mark_active_rects"]
    mark_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    mark_program["world_size"].value = (world.width, world.height)
    mark_program["tile_size"].value = int(world.active.tile_size)
    mark_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
    mark_program["rect_count"].value = int(len(packed_rects))
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
    bridge.buffers["active_rect"].bind_to_storage_buffer(binding=1)
    mark_program.run((tile_count + 255) // 256, 1, 1)
    bridge.ctx.memory_barrier(
        getattr(bridge.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
    )
    bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
    return True


def decay_active_scheduler(bridge, world: "WorldEngine") -> bool:
    if not bridge.enabled or bridge.ctx is None:
        return False
    bridge.ensure_world_resources(world)
    if (
        "active_meta" not in bridge.buffers
        or "active_tile_ttl" not in bridge.buffers
        or "active_chunk_mask" not in bridge.buffers
    ):
        return False
    bridge._ensure_active_scheduler_programs()
    tile_count = int(world.active.tile_width * world.active.tile_height)
    chunk_count = int(world.active.chunk_width * world.active.chunk_height)
    if tile_count <= 0 or chunk_count <= 0:
        return False

    decay_program = bridge.active_scheduler_programs["decay_active_tiles"]
    decay_program["tile_count"].value = tile_count
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
    decay_program.run((tile_count + 255) // 256, 1, 1)
    bridge.ctx.memory_barrier(
        getattr(bridge.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
    )

    bridge._refresh_active_chunks_and_meta(world, read_meta=True)
    bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
    return True


def _refresh_active_chunks_and_meta(
    bridge, world: "WorldEngine", *, read_meta: bool = False
) -> None:
    assert bridge.ctx is not None
    clear_program = bridge.active_scheduler_programs["clear_active_counts"]
    bridge.buffers["active_meta"].bind_to_storage_buffer(binding=0)
    bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=1)
    bridge.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=2)
    clear_program.run(1, 1, 1)
    bridge.ctx.memory_barrier(
        getattr(bridge.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "COMMAND_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
    )

    refresh_program = bridge.active_scheduler_programs["refresh_active_chunks"]
    refresh_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
    refresh_program["chunk_grid_size"].value = (world.active.chunk_width, world.active.chunk_height)
    refresh_program["chunk_tiles"].value = int(world.active.chunk_tiles)
    bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
    bridge.buffers["active_chunk_mask"].bind_to_storage_buffer(binding=1)
    bridge.buffers["active_meta"].bind_to_storage_buffer(binding=2)
    bridge.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
    bridge.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
    bridge.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=5)
    refresh_program.run(world.active.chunk_width, world.active.chunk_height, 1)
    bridge.ctx.memory_barrier(
        getattr(bridge.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "COMMAND_BARRIER_BIT", 0)
        | getattr(bridge.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
    )
    if read_meta:
        bridge.shadow_buffers["active_meta"] = np.frombuffer(
            bridge.buffers["active_meta"].read(size=ACTIVE_META_DTYPE.itemsize),
            dtype=ACTIVE_META_DTYPE,
            count=1,
        ).copy()


def _ensure_active_scheduler_programs(bridge) -> None:
    if bridge.ctx is None:
        return
    required_programs = {
        "mark_active_rects",
        "decay_active_tiles",
        "clear_active_counts",
        "count_active_scheduler",
        "refresh_active_chunks",
    }
    if required_programs.issubset(bridge.active_scheduler_programs):
        return
    for name in required_programs:
        program = bridge.active_scheduler_programs.pop(name, None)
        if program is not None:
            try:
                program.release()
            except Exception:
                pass
    for name in required_programs:
        bridge.active_scheduler_programs[name] = build_compute_shader(
            bridge.ctx,
            f"display/{name}.comp",
        )


def _release_display_programs(bridge) -> None:
    for program in bridge.display_programs.values():
        try:
            program.release()
        except Exception:
            pass
    bridge.display_programs.clear()


def _release_active_scheduler_programs(bridge) -> None:
    for program in bridge.active_scheduler_programs.values():
        try:
            program.release()
        except Exception:
            pass
    bridge.active_scheduler_programs.clear()
