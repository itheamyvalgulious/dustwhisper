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
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
    ReadbackPayloadPlan,
    GLReadbackSlot,
    ReadbackPayloadLayout,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedBufferReadbackSource,
    GPUGasWindowReadbackSource,
    ReadbackArrayLayout,
    GPUCellCoreWindowReadbackSource,
    GPUBufferReadbackSource,
)

from oracle_game.types import (
    ReadbackResult,
    ReadbackRequest,
)

from copy import (
    deepcopy,
)


def queue_readback(
    bridge,
    frame_id: int,
    request: ReadbackRequest,
    payload: dict[str, Any],
    *,
    require_gpu_sources: bool = False,
) -> bool:
    slot: GLReadbackSlot | None = None
    slot_count = len(bridge.readback_slots)
    for offset in range(slot_count):
        candidate = bridge.readback_slots[(bridge.write_index + offset) % slot_count]
        if candidate.frame_id < 0 and candidate.request is None:
            slot = candidate
            break
    if slot is None:
        return False
    plan = bridge._plan_readback_payload(payload)
    gpu_backed = bool(plan.gpu_sources)
    latency_frames = GPU_READBACK_LATENCY_FRAMES if gpu_backed else CPU_READBACK_LATENCY_FRAMES
    if require_gpu_sources and plan.cpu_chunks:
        paths = ", ".join(".".join(path) if path else "<root>" for path in plan.cpu_chunk_paths)
        raise RuntimeError(
            f"GPU readback requires GPU-backed payload arrays, found CPU payload chunks at: {paths}; "
            "CPU fallback is disabled"
        )
    if require_gpu_sources and plan.gpu_sources and (not bridge.enabled or bridge.ctx is None):
        raise RuntimeError("GPU readback requires an enabled ModernGL context; CPU fallback is disabled")
    if bridge.enabled and bridge.ctx is not None:
        if slot.buffer is None or slot.buffer.size < max(plan.nbytes, 4):
            if slot.buffer is not None:
                slot.buffer.release()
            slot.buffer = bridge.ctx.buffer(reserve=max(plan.nbytes, 4), dynamic=True)
        else:
            slot.buffer.orphan(max(plan.nbytes, 4))
        for offset, data in plan.cpu_chunks:
            if data:
                slot.buffer.write(data, offset=offset)
        for offset, source in plan.gpu_sources:
            bridge._fill_readback_slot_from_gpu(
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
    bridge.write_index = (bridge.write_index + 1) % len(bridge.readback_slots)
    return True


def poll_readback(bridge, current_frame_id: int) -> ReadbackResult | None:
    ready_slots = [
        slot
        for slot in bridge.readback_slots
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
    elif bridge.enabled and bridge.ctx is not None and slot.buffer is not None:
        raw = slot.buffer.read(size=slot.nbytes)
    else:
        raw = slot.buffer if isinstance(slot.buffer, (bytes, bytearray)) else b""
    payload = bridge._decode_readback_payload(raw, slot.layout)
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


def _plan_readback_payload(bridge, payload: dict[str, Any]) -> ReadbackPayloadPlan:
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
        return bridge._normalize_metadata(value)

    metadata = visit((), payload)
    plan.layout.metadata = metadata if isinstance(metadata, dict) else {}
    plan.nbytes = offset
    return plan


def _fill_readback_slot_from_gpu(
    bridge,
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
    assert bridge.ctx is not None
    if isinstance(source, GPUSegmentedCellCoreWindowReadbackSource):
        bridge._pack_segmented_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUSegmentedBufferReadbackSource):
        bridge._pack_segmented_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUSegmentedTextureReadbackSource):
        bridge._pack_segmented_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUCellCoreWindowReadbackSource):
        bridge._pack_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUGasWindowReadbackSource):
        bridge._pack_gas_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUBufferReadbackSource):
        bridge._pack_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    if isinstance(source, GPUTextureReadbackSource):
        bridge._pack_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
        return
    raise TypeError(f"Unsupported GPU readback source: {type(source)!r}")


def _decode_readback_payload(bridge, raw: bytes, layout: ReadbackPayloadLayout | None) -> dict[str, Any]:
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


def _normalize_metadata(bridge, value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): bridge._normalize_metadata(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [bridge._normalize_metadata(child) for child in value]
    if isinstance(value, list):
        return [bridge._normalize_metadata(child) for child in value]
    return value


def _ensure_readback_programs(bridge) -> None:
    if bridge.ctx is None or bridge.readback_programs:
        return
    local_size = 8
    bridge.readback_programs["cell_core_window"] = bridge.ctx.compute_shader(
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
    bridge.readback_programs["gas_window"] = bridge.ctx.compute_shader(
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
    bridge.readback_programs["buffer_window"] = bridge.ctx.compute_shader(
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
    bridge.readback_programs["texture_window"] = bridge.ctx.compute_shader(
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
    bridge,
    slot_buffer: Any,
    offset: int,
    source: GPUCellCoreWindowReadbackSource,
    *,
    require_gpu_source: bool = False,
) -> None:
    if offset % 4 != 0:
        bridge._raise_gpu_readback_unavailable(source, "unaligned destination offset")
        return
    height, width = source.shape[:2]
    if width <= 0 or height <= 0:
        return
    bridge._ensure_readback_programs()
    program = bridge.readback_programs.get("cell_core_window")
    if program is None:
        bridge._raise_gpu_readback_unavailable(source, "missing cell core readback shader")
        return
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
    bridge.ctx.memory_barrier()


def _pack_gas_window_into_buffer(
    bridge,
    slot_buffer: Any,
    offset: int,
    source: GPUGasWindowReadbackSource,
    *,
    require_gpu_source: bool = False,
) -> None:
    if offset % 4 != 0:
        bridge._raise_gpu_readback_unavailable(source, "unaligned destination offset")
        return
    height, width = source.shape
    if width <= 0 or height <= 0:
        return
    bridge._ensure_readback_programs()
    program = bridge.readback_programs.get("gas_window")
    if program is None:
        bridge._raise_gpu_readback_unavailable(source, "missing gas readback shader")
        return
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
    bridge.ctx.memory_barrier()


def _pack_buffer_window_into_buffer(
    bridge,
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
        bridge._raise_gpu_readback_unavailable(source, "unsupported buffer copy alignment or element size")
        return
    if source.chunk_size <= 0 or source.count <= 0:
        return
    bridge._ensure_readback_programs()
    program = bridge.readback_programs.get("buffer_window")
    if program is None:
        bridge._raise_gpu_readback_unavailable(source, "missing buffer readback shader")
        return
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
    bridge.ctx.memory_barrier()


def _pack_texture_window_into_buffer(
    bridge,
    slot_buffer: Any,
    offset: int,
    source: GPUTextureReadbackSource,
    *,
    require_gpu_source: bool = False,
) -> None:
    if offset % 4 != 0:
        bridge._raise_gpu_readback_unavailable(source, "unaligned destination offset")
        return
    origin_x, origin_y, width, height = source.viewport
    if width <= 0 or height <= 0 or source.components <= 0:
        return
    bridge._ensure_readback_programs()
    program = bridge.readback_programs.get("texture_window")
    if program is None or source.components > 4:
        bridge._raise_gpu_readback_unavailable(source, "missing texture readback shader or unsupported component count")
        return
    texture = bridge.textures.get(source.resource_name)
    if texture is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU texture")
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
    bridge.ctx.memory_barrier()


def _pack_segmented_cell_core_window_into_buffer(
    bridge,
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
        bridge._pack_cell_core_window_into_buffer(
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
    bridge,
    slot_buffer: Any,
    offset: int,
    source: GPUSegmentedBufferReadbackSource,
    *,
    require_gpu_source: bool = False,
) -> None:
    dtype = np.dtype(source.dtype)
    if dtype.itemsize != 4:
        bridge._raise_gpu_readback_unavailable(source, "unsupported segmented buffer element size")
        return
    if len(source.shape) < 2:
        bridge._raise_gpu_readback_unavailable(source, "segmented buffer source requires a 2D destination")
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
        bridge._pack_buffer_window_into_buffer(
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
    bridge,
    slot_buffer: Any,
    offset: int,
    source: GPUSegmentedTextureReadbackSource,
    *,
    require_gpu_source: bool = False,
) -> None:
    if source.components <= 0:
        return
    if len(source.shape) < 2:
        bridge._raise_gpu_readback_unavailable(source, "segmented texture source requires a 2D destination")
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
        bridge._pack_texture_window_into_buffer(
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


def _release_readback_programs(bridge) -> None:
    for program in bridge.readback_programs.values():
        try:
            program.release()
        except Exception:
            pass
    bridge.readback_programs.clear()
