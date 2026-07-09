from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.gpu import (
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUReadbackSegment,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
)
from oracle_game.types import ReadbackRequest

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def make_readback_payload(engine: "WorldEngine", request: ReadbackRequest) -> dict[str, Any]:
    world_x0, world_y0, world_x1, world_y1 = engine._centered_world_window(
        int(request.center_x),
        int(request.center_y),
        int(request.width),
        int(request.height),
    )
    cell_width = world_x1 - world_x0
    cell_height = world_y1 - world_y0
    x_spans = engine._world_axis_spans(world_x0, world_x1, axis="x")
    y_spans = engine._world_axis_spans(world_y0, world_y1, axis="y")
    cell_contiguous = len(x_spans) <= 1 and len(y_spans) <= 1
    x0 = x_spans[0][0] if x_spans else 0
    y0 = y_spans[0][0] if y_spans else 0
    x1 = x0 + cell_width
    y1 = y0 + cell_height
    gas_world_x0, gas_world_y0, gas_world_x1, gas_world_y1 = engine._world_gas_window_for_cell_world_rect(
        world_x0,
        world_y0,
        world_x1,
        world_y1,
    )
    gas_width = gas_world_x1 - gas_world_x0
    gas_height = gas_world_y1 - gas_world_y0
    gx_spans = engine._world_axis_spans(gas_world_x0, gas_world_x1, axis="x", gas_grid=True)
    gy_spans = engine._world_axis_spans(gas_world_y0, gas_world_y1, axis="y", gas_grid=True)
    gas_contiguous = len(gx_spans) <= 1 and len(gy_spans) <= 1
    gx0 = gx_spans[0][0] if gx_spans else 0
    gy0 = gy_spans[0][0] if gy_spans else 0
    gx1 = gx0 + gas_width
    gy1 = gy0 + gas_height
    gpu_mode = engine.simulation_backend == "gpu"

    def axis_segments(spans: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
        dst_offset = 0
        result: list[tuple[int, int, int, int]] = []
        for src_start, src_end in spans:
            length = max(0, int(src_end) - int(src_start))
            if length <= 0:
                continue
            result.append((int(src_start), int(src_end), dst_offset, length))
            dst_offset += length
        return result

    def rect_segments(
        x_axis_spans: list[tuple[int, int]],
        y_axis_spans: list[tuple[int, int]],
    ) -> tuple[GPUReadbackSegment, ...]:
        x_parts = axis_segments(x_axis_spans)
        y_parts = axis_segments(y_axis_spans)
        return tuple(
            GPUReadbackSegment(
                src_x=src_x0,
                src_y=src_y0,
                dst_x=dst_x0,
                dst_y=dst_y0,
                width=width,
                height=height,
            )
            for src_y0, _src_y1, dst_y0, height in y_parts
            for src_x0, _src_x1, dst_x0, width in x_parts
        )

    cell_segments = rect_segments(x_spans, y_spans)
    gas_segments = rect_segments(gx_spans, gy_spans)

    def buffer_window_source(
        *,
        resource_name: str,
        dtype: str | np.dtype[Any],
        shape: tuple[int, int],
        grid_width: int,
        origin_x: int,
        origin_y: int,
        contiguous: bool,
        segments: tuple[GPUReadbackSegment, ...],
        base_offset: int = 0,
        cpu_array_factory: Any,
    ) -> Any:
        if not gpu_mode:
            return cpu_array_factory()
        resolved_dtype = np.dtype(dtype)
        itemsize = resolved_dtype.itemsize
        if contiguous:
            row_bytes = int(grid_width) * itemsize
            window_bytes = int(shape[1]) * itemsize
            start = int(base_offset) + (int(origin_y) * int(grid_width) + int(origin_x)) * itemsize
            return GPUBufferReadbackSource(
                resource_name=resource_name,
                dtype=resolved_dtype.str,
                shape=shape,
                chunk_size=window_bytes,
                start=start,
                step=row_bytes,
                count=int(shape[0]),
            )
        return GPUSegmentedBufferReadbackSource(
            resource_name=resource_name,
            dtype=resolved_dtype.str,
            shape=shape,
            grid_width=int(grid_width),
            base_offset=int(base_offset),
            segments=segments,
        )

    def texture_window_source(
        *,
        resource_name: str,
        dtype: str | np.dtype[Any],
        shape: tuple[int, ...],
        components: int,
        origin_x: int,
        origin_y: int,
        width: int,
        height: int,
        contiguous: bool,
        segments: tuple[GPUReadbackSegment, ...],
        cpu_array_factory: Any,
    ) -> Any:
        if not gpu_mode:
            return cpu_array_factory()
        if contiguous:
            return GPUTextureReadbackSource(
                resource_name=resource_name,
                dtype=np.dtype(dtype).str,
                shape=shape,
                components=int(components),
                viewport=(int(origin_x), int(origin_y), int(width), int(height)),
            )
        return GPUSegmentedTextureReadbackSource(
            resource_name=resource_name,
            dtype=np.dtype(dtype).str,
            shape=shape,
            components=int(components),
            segments=segments,
        )

    def cell_core_source() -> Any:
        if not gpu_mode:
            return engine._pack_cell_core_world_window(world_x0, world_y0, world_x1, world_y1)
        if cell_contiguous:
            return GPUCellCoreWindowReadbackSource(
                resource_name="cell_core",
                dtype="u4",
                shape=(cell_height, cell_width, 5),
                cell_grid_width=engine.width,
                origin_x=x0,
                origin_y=y0,
            )
        return GPUSegmentedCellCoreWindowReadbackSource(
            resource_name="cell_core",
            dtype="u4",
            shape=(cell_height, cell_width, 5),
            cell_grid_width=engine.width,
            segments=cell_segments,
        )

    def gas_species_source(species_id: int) -> Any:
        if not gpu_mode:
            return engine._extract_world_window(
                engine.gas_concentration[species_id],
                gas_world_x0,
                gas_world_y0,
                gas_world_x1,
                gas_world_y1,
                x_axis=1,
                y_axis=0,
                gas_grid=True,
            ).astype(np.float32, copy=False)
        if gas_contiguous:
            return GPUGasWindowReadbackSource(
                resource_name="gas_concentration",
                dtype="f4",
                shape=(gas_height, gas_width),
                gas_grid_width=engine.gas_width,
                gas_grid_height=engine.gas_height,
                species_id=int(species_id),
                origin_x=gx0,
                origin_y=gy0,
            )
        return GPUSegmentedBufferReadbackSource(
            resource_name="gas_concentration",
            dtype=np.dtype(np.float32).str,
            shape=(gas_height, gas_width),
            grid_width=engine.gas_width,
            base_offset=int(species_id) * engine.gas_height * engine.gas_width * np.dtype(np.float32).itemsize,
            segments=gas_segments,
        )

    payload: dict[str, Any] = {}
    if "cell" in request.channels:
        cell_payload: dict[str, Any] = {
            "origin": [world_x0, world_y0],
            "size": [cell_width, cell_height],
        }
        cell_payload.update(
            {
                "core_words": cell_core_source(),
                "island_id": buffer_window_source(
                    resource_name="island_id",
                    dtype=np.int32,
                    shape=(cell_height, cell_width),
                    grid_width=engine.width,
                    origin_x=x0,
                    origin_y=y0,
                    contiguous=cell_contiguous,
                    segments=cell_segments,
                    cpu_array_factory=lambda: engine._extract_world_window(
                        engine.island_id,
                        world_x0,
                        world_y0,
                        world_x1,
                        world_y1,
                        x_axis=1,
                        y_axis=0,
                    ).astype(np.int32, copy=False),
                ),
                "entity_id": buffer_window_source(
                    resource_name="entity_id",
                    dtype=np.int32,
                    shape=(cell_height, cell_width),
                    grid_width=engine.width,
                    origin_x=x0,
                    origin_y=y0,
                    contiguous=cell_contiguous,
                    segments=cell_segments,
                    cpu_array_factory=lambda: engine._extract_world_window(
                        engine.entity_id,
                        world_x0,
                        world_y0,
                        world_x1,
                        world_y1,
                        x_axis=1,
                        y_axis=0,
                    ).astype(np.int32, copy=False),
                ),
                "placeholder_displaced_material": buffer_window_source(
                    resource_name="placeholder_displaced_material",
                    dtype=np.int32,
                    shape=(cell_height, cell_width),
                    grid_width=engine.width,
                    origin_x=x0,
                    origin_y=y0,
                    contiguous=cell_contiguous,
                    segments=cell_segments,
                    cpu_array_factory=lambda: engine._extract_world_window(
                        engine.placeholder_displaced_material,
                        world_x0,
                        world_y0,
                        world_x1,
                        world_y1,
                        x_axis=1,
                        y_axis=0,
                    ).astype(np.int32, copy=False),
                ),
                "collapse_delay_pending": buffer_window_source(
                    resource_name="collapse_delay_pending",
                    dtype=np.int32,
                    shape=(cell_height, cell_width),
                    grid_width=engine.width,
                    origin_x=x0,
                    origin_y=y0,
                    contiguous=cell_contiguous,
                    segments=cell_segments,
                    cpu_array_factory=lambda: engine._extract_world_window(
                        engine.collapse_delay_pending,
                        world_x0,
                        world_y0,
                        world_x1,
                        world_y1,
                        x_axis=1,
                        y_axis=0,
                    ).astype(np.int32, copy=False),
                ),
            }
        )
        payload["cell"] = cell_payload
    if "ambient_temperature" in request.channels:
        payload["ambient_temperature"] = {
            "origin": [gas_world_x0, gas_world_y0],
            "size": [gas_width, gas_height],
            "grid": "gas",
            "values": texture_window_source(
                resource_name="ambient_temperature",
                dtype=np.float32,
                shape=(gas_height, gas_width),
                components=1,
                origin_x=gx0,
                origin_y=gy0,
                width=gas_width,
                height=gas_height,
                contiguous=gas_contiguous,
                segments=gas_segments,
                cpu_array_factory=lambda: engine._extract_world_window(
                    engine.ambient_temperature,
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                ).astype(np.float32, copy=False),
            ),
        }
    if "pressure" in request.channels:
        payload["pressure"] = {
            "origin": [gas_world_x0, gas_world_y0],
            "size": [gas_width, gas_height],
            "grid": "gas",
            "values": texture_window_source(
                resource_name="pressure_ping",
                dtype=np.float32,
                shape=(gas_height, gas_width),
                components=1,
                origin_x=gx0,
                origin_y=gy0,
                width=gas_width,
                height=gas_height,
                contiguous=gas_contiguous,
                segments=gas_segments,
                cpu_array_factory=lambda: engine._extract_world_window(
                    engine.pressure_ping,
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                ).astype(np.float32, copy=False),
            ),
        }
    if "velocity" in request.channels:
        payload["velocity"] = {
            "origin": [gas_world_x0, gas_world_y0],
            "size": [gas_width, gas_height],
            "grid": "gas",
            "values": texture_window_source(
                resource_name="flow_velocity",
                dtype=np.float32,
                shape=(gas_height, gas_width, 2),
                components=2,
                origin_x=gx0,
                origin_y=gy0,
                width=gas_width,
                height=gas_height,
                contiguous=gas_contiguous,
                segments=gas_segments,
                cpu_array_factory=lambda: engine._extract_world_window(
                    engine.flow_velocity,
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                ).astype(np.float32, copy=False),
            ),
        }
    if "optics" in request.channels:
        light_entries = [
            (shadow_name, dose_channel)
            for light_id in range(len(engine.light_name_by_id))
            for shadow_name in [engine._shadow_light_name(light_id)]
            for dose_channel in [engine._shadow_light_dose_channel(light_id)]
            if shadow_name
            and dose_channel is not None
            and 0 <= int(dose_channel) < engine.cell_optical_dose.shape[0]
            and 0 <= int(dose_channel) < engine.gas_optical_dose.shape[0]
        ]
        optics_payload: dict[str, Any] = {
            "origin": [world_x0, world_y0],
            "size": [cell_width, cell_height],
            "gas_origin": [gas_world_x0, gas_world_y0],
            "gas_size": [gas_width, gas_height],
        }
        optics_payload["visible_illumination"] = texture_window_source(
            resource_name="visible_illumination",
            dtype=np.float32,
            shape=(cell_height, cell_width, 3),
            components=3,
            origin_x=x0,
            origin_y=y0,
            width=cell_width,
            height=cell_height,
            contiguous=cell_contiguous,
            segments=cell_segments,
            cpu_array_factory=lambda: engine._extract_world_window(
                engine.visible_illumination,
                world_x0,
                world_y0,
                world_x1,
                world_y1,
                x_axis=1,
                y_axis=0,
            ).astype(np.float32, copy=False),
        )
        optics_payload["cell_dose"] = {
            light_name: buffer_window_source(
                resource_name="cell_optical_dose",
                dtype=np.float32,
                shape=(cell_height, cell_width),
                grid_width=engine.width,
                origin_x=x0,
                origin_y=y0,
                contiguous=cell_contiguous,
                segments=cell_segments,
                base_offset=int(dose_channel) * engine.height * engine.width * np.dtype(np.float32).itemsize,
                cpu_array_factory=lambda dose_channel=dose_channel: engine._extract_world_window(
                    engine.cell_optical_dose[dose_channel],
                    world_x0,
                    world_y0,
                    world_x1,
                    world_y1,
                    x_axis=1,
                    y_axis=0,
                ).astype(np.float32, copy=False),
            )
            for light_name, dose_channel in light_entries
        }
        optics_payload["gas_dose"] = {
            light_name: buffer_window_source(
                resource_name="gas_optical_dose",
                dtype=np.float32,
                shape=(gas_height, gas_width),
                grid_width=engine.gas_width,
                origin_x=gx0,
                origin_y=gy0,
                contiguous=gas_contiguous,
                segments=gas_segments,
                base_offset=int(dose_channel) * engine.gas_height * engine.gas_width * np.dtype(np.float32).itemsize,
                cpu_array_factory=lambda dose_channel=dose_channel: engine._extract_world_window(
                    engine.gas_optical_dose[dose_channel],
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                ).astype(np.float32, copy=False),
            )
            for light_name, dose_channel in light_entries
        }
        payload["optics"] = optics_payload
    if "gas" in request.channels:
        gas_entries = [
            (species_id, shadow_name)
            for species_id in range(engine.gas_concentration.shape[0])
            for shadow_name in [engine._shadow_gas_name(species_id)]
            if shadow_name
        ]
        payload["gas"] = {
            "origin": [gas_world_x0, gas_world_y0],
            "size": [gas_width, gas_height],
            "grid": "gas",
            "species": {
                name: gas_species_source(species_id) for species_id, name in gas_entries
            },
        }
    return payload
