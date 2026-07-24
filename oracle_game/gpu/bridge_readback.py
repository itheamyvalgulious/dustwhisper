from __future__ import annotations

from copy import (
    deepcopy,
)
from typing import Any

import numpy as np

from oracle_game.gpu._common import (
    CPU_READBACK_LATENCY_FRAMES,
    GPU_READBACK_LATENCY_FRAMES,
)
from oracle_game.gpu.readback import (
    GLReadbackSlot,
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
    ReadbackArrayLayout,
    ReadbackPayloadLayout,
    ReadbackPayloadPlan,
)
from oracle_game.gpu.shader_loader import build_compute_shader
from oracle_game.types import (
    ReadbackRequest,
    ReadbackResult,
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
        raise RuntimeError(
            "GPU readback requires an enabled ModernGL context; CPU fallback is disabled"
        )
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


def _stash_inflight_readback_slots(bridge) -> None:
    """Preserve in-flight readback slot state across readback ring rebuilds.

    Rebuilding the ring (context attach, resource release) destroys the GL
    storage backing occupied slots.  Stash the occupied slot descriptors so
    ``requeue_detached_readbacks`` can re-queue them on the rebuilt ring with
    their original frame/latency bookkeeping instead of silently dropping
    them (which also leaks the engine's ``inflight_readbacks`` entries).
    """
    for slot in bridge.readback_slots:
        if slot.frame_id < 0 or slot.request is None:
            continue
        bridge.detached_readback_slots.append(
            GLReadbackSlot(
                slot.slot_index,
                frame_id=slot.frame_id,
                ready_frame_id=slot.ready_frame_id,
                min_poll_frame_id=slot.min_poll_frame_id,
                latency_frames=slot.latency_frames,
                gpu_backed=slot.gpu_backed,
                request=slot.request,
                nbytes=slot.nbytes,
                layout=slot.layout,
            )
        )


def requeue_detached_readbacks(bridge, world) -> None:
    """Re-queue readbacks detached by a readback ring rebuild.

    Each detached request's payload is re-materialized against the current
    (already re-synced) resources and queued with its original frame id, so
    the request keeps its contracted latency/ready timing and the engine's
    ``inflight_readbacks`` bookkeeping drains through the normal poll path.
    Requests that cannot be queued any more are failed explicitly: their
    inflight bookkeeping is dropped so it cannot leak.  A full ring is
    transient, so those descriptors stay stashed for the next drain.
    """
    if not bridge.detached_readback_slots:
        return
    detached = sorted(
        bridge.detached_readback_slots,
        key=lambda slot: (slot.frame_id, slot.slot_index),
    )
    bridge.detached_readback_slots.clear()
    for slot in detached:
        request = slot.request
        if request is None:
            continue
        try:
            payload = world._make_readback_payload(request)
        except Exception:
            payload = None
        queued = False
        if payload is not None:
            try:
                queued = bridge.queue_readback(
                    slot.frame_id,
                    request,
                    payload,
                    require_gpu_sources=getattr(world, "simulation_backend", "") == "gpu",
                )
            except Exception:
                payload = None
        if queued:
            continue
        if payload is not None:
            # The ring was full; keep the descriptor stashed so a later drain
            # retries once slots free up.
            bridge.detached_readback_slots.append(slot)
            continue
        if request.request_id is not None:
            request_id = int(request.request_id)
            world.inflight_readbacks = [
                existing
                for existing in world.inflight_readbacks
                if existing.request_id != request_id
            ]


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
        bridge._pack_segmented_cell_core_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUSegmentedBufferReadbackSource):
        bridge._pack_segmented_buffer_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUSegmentedTextureReadbackSource):
        bridge._pack_segmented_texture_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUCellCoreWindowReadbackSource):
        bridge._pack_cell_core_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUGasWindowReadbackSource):
        bridge._pack_gas_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUBufferReadbackSource):
        bridge._pack_buffer_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    if isinstance(source, GPUTextureReadbackSource):
        bridge._pack_texture_window_into_buffer(
            slot_buffer, offset, source, require_gpu_source=require_gpu_source
        )
        return
    raise TypeError(f"Unsupported GPU readback source: {type(source)!r}")


def _decode_readback_payload(
    bridge, raw: bytes, layout: ReadbackPayloadLayout | None
) -> dict[str, Any]:
    if layout is None:
        return {}
    payload = deepcopy(layout.metadata)
    for spec in layout.arrays:
        array = np.frombuffer(
            raw,
            dtype=np.dtype(spec.dtype),
            count=int(np.prod(spec.shape, dtype=np.int64)),
            offset=spec.offset,
        )
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
    subs = {"LOCAL_SIZE": 8}
    for name in ("cell_core_window", "gas_window", "buffer_window", "texture_window"):
        bridge.readback_programs[name] = build_compute_shader(
            bridge.ctx,
            f"readback/{name}.comp",
            subs,
        )


def _fill_slot_from_buffer_rows(
    bridge,
    slot_buffer: Any,
    offset: int,
    src_buffer: Any,
    *,
    src_row_offset: int,
    src_row_stride: int,
    row_bytes: int,
    row_count: int,
    dst_row_stride: int,
) -> None:
    """Fill a readback slot from a GPU buffer via DMA copies and one mapped read.

    On some NVIDIA EGL standalone contexts compute-shader SSBO reads return
    stale or wrong data (uniform-indexed reads landing on the wrong words);
    the ``glCopyBufferSubData`` + map path has been empirically verified to
    stay coherent with the buffer's actual contents there, so buffer-backed
    readback windows are copied this way instead of with the pack shaders.
    """
    nbytes = max(row_bytes * row_count, 4)
    staging = bridge.readback_staging
    if staging is None or staging.size < nbytes:
        if staging is not None:
            staging.release()
        staging = bridge.ctx.buffer(reserve=nbytes, dynamic=True)
        bridge.readback_staging = staging
    for row in range(row_count):
        bridge.ctx.copy_buffer(
            staging,
            src_buffer,
            row_bytes,
            read_offset=src_row_offset + row * src_row_stride,
            write_offset=row * row_bytes,
        )
    if dst_row_stride == row_bytes:
        slot_buffer.write(staging.read(size=row_bytes * row_count), offset=offset)
        return
    for row in range(row_count):
        slot_buffer.write(
            staging.read(size=row_bytes, offset=row * row_bytes),
            offset=offset + row * dst_row_stride,
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
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
        return
    dst_grid = int(source.dst_cell_grid_width or width)
    _fill_slot_from_buffer_rows(
        bridge,
        slot_buffer,
        offset,
        src_buffer,
        src_row_offset=(source.origin_y * source.cell_grid_width + source.origin_x) * 5 * 4,
        src_row_stride=source.cell_grid_width * 5 * 4,
        row_bytes=width * 5 * 4,
        row_count=height,
        dst_row_stride=dst_grid * 5 * 4,
    )


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
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
        return
    _fill_slot_from_buffer_rows(
        bridge,
        slot_buffer,
        offset,
        src_buffer,
        src_row_offset=(
            (source.species_id * source.gas_grid_height + source.origin_y) * source.gas_grid_width
            + source.origin_x
        )
        * 4,
        src_row_stride=source.gas_grid_width * 4,
        row_bytes=width * 4,
        row_count=height,
        dst_row_stride=width * 4,
    )


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
        bridge._raise_gpu_readback_unavailable(
            source, "unsupported buffer copy alignment or element size"
        )
        return
    if source.chunk_size <= 0 or source.count <= 0:
        return
    src_buffer = bridge.buffers.get(source.resource_name)
    if src_buffer is None:
        bridge._raise_gpu_readback_unavailable(source, "missing GPU buffer")
        return
    _fill_slot_from_buffer_rows(
        bridge,
        slot_buffer,
        offset,
        src_buffer,
        src_row_offset=source.start,
        src_row_stride=source.step,
        row_bytes=source.chunk_size,
        row_count=source.count,
        dst_row_stride=source.dst_step or source.chunk_size,
    )


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
        bridge._raise_gpu_readback_unavailable(
            source, "missing texture readback shader or unsupported component count"
        )
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
    program["dst_float_row_stride"].value = (
        source.dst_step or (width * source.components * 4)
    ) // 4
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
        bridge._raise_gpu_readback_unavailable(
            source, "segmented buffer source requires a 2D destination"
        )
        return
    width = int(source.shape[1])
    height = int(source.shape[0])
    if width <= 0 or height <= 0:
        return
    itemsize = dtype.itemsize
    for segment in source.segments:
        if segment.width <= 0 or segment.height <= 0:
            continue
        src_start = (
            int(source.base_offset)
            + (int(segment.src_y) * int(source.grid_width) + int(segment.src_x)) * itemsize
        )
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
        bridge._raise_gpu_readback_unavailable(
            source, "segmented texture source requires a 2D destination"
        )
        return
    width = int(source.shape[1])
    height = int(source.shape[0])
    if width <= 0 or height <= 0:
        return
    row_step = width * int(source.components) * 4
    for segment in source.segments:
        if segment.width <= 0 or segment.height <= 0:
            continue
        dst_offset = (
            offset + (int(segment.dst_y) * width + int(segment.dst_x)) * int(source.components) * 4
        )
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
                viewport=(
                    int(segment.src_x),
                    int(segment.src_y),
                    int(segment.width),
                    int(segment.height),
                ),
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
