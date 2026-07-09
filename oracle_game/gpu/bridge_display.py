from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import numpy as np

from oracle_game.gpu.dtypes import (
    ACTIVE_RECT_DTYPE,
    ACTIVE_META_DTYPE,
)


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
        bridge.ctx.memory_barrier(bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
    if "visible_illumination" in bridge.gpu_authoritative_resources and "visible_illumination" in bridge.textures:
        bridge._ensure_display_programs()
        program = bridge.display_programs["light_from_visible_texture"]
        program["width"] = int(world.width)
        program["height"] = int(world.height)
        bridge.textures["visible_illumination"].use(0)
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
        bridge.ctx.memory_barrier(bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)


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
    bridge.ctx.memory_barrier(bridge.ctx.TEXTURE_FETCH_BARRIER_BIT | bridge.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
    return True


def _ensure_display_programs(bridge) -> None:
    if not bridge.enabled or bridge.ctx is None:
        return
    if "material_from_cell_core" not in bridge.display_programs:
        bridge.display_programs["material_from_cell_core"] = bridge.ctx.compute_shader(
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
    if "light_from_visible_texture" not in bridge.display_programs:
        bridge.display_programs["light_from_visible_texture"] = bridge.ctx.compute_shader(
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
        bridge.display_programs["light_from_visible_texture"]["visible_tex"] = 0
    if "debug_from_gpu_state" not in bridge.display_programs:
        bridge.display_programs["debug_from_gpu_state"] = bridge.ctx.compute_shader(
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

    bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
    return True


def _refresh_active_chunks_and_meta(bridge, world: "WorldEngine", *, read_meta: bool = False) -> None:
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
    bridge.active_scheduler_programs["mark_active_rects"] = bridge.ctx.compute_shader(
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
    bridge.active_scheduler_programs["decay_active_tiles"] = bridge.ctx.compute_shader(
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
    bridge.active_scheduler_programs["clear_active_counts"] = bridge.ctx.compute_shader(
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
    bridge.active_scheduler_programs["count_active_scheduler"] = bridge.ctx.compute_shader(
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
    bridge.active_scheduler_programs["refresh_active_chunks"] = bridge.ctx.compute_shader(
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
