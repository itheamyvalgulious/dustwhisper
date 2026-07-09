from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oracle_game.types import ReadbackRequest

from oracle_game.gpu._common import CPU_READBACK_LATENCY_FRAMES



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
