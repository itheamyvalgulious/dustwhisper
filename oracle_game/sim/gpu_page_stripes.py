from __future__ import annotations

import threading
from typing import Any

import numpy as np

from oracle_game.gpu import moderngl, unpack_cell_core
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.types import Phase


LOCAL_SIZE = 128


class GPUPageStripePipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.programs: dict[str, Any] = {}
        self.last_backend = "idle"
        self.last_cpu_mirror_downloaded = False

    def available(self, world: "WorldEngine") -> bool:
        # Override: this pipeline may run on a standalone (non-bridge) context,
        # so it also accepts ``moderngl`` being importable as a fallback.
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        ctx = world.bridge.ctx
        return bool(
            (world.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)
            or moderngl is not None
        )

    def apply(self, world: "WorldEngine", update: "PageStripeUpdate", payload: dict[str, Any]) -> None:
        ctx, release_context = self._context_for_current_thread(world)
        try:
            if ctx is None or getattr(ctx, "version_code", 0) < 430:
                raise RuntimeError("GPU page stripe pipeline requires a valid ModernGL context")
            if getattr(world, "simulation_backend", "") == "gpu":
                if release_context or ctx is not world.bridge.ctx:
                    raise RuntimeError("GPU page stripe runtime must use the bridge context")
                self._apply_bridge(world, update, payload)
                self.last_backend = "gpu"
                self.last_cpu_mirror_downloaded = False
                return
            cell_ranges = world._stripe_buffer_ranges(update, gas_grid=False)
            gas_ranges = world._stripe_buffer_ranges(update, gas_grid=True)
            cell_payload = payload["cell"]
            gas_payload = payload["gas"]
            cell_axis = 1 if update.axis == "x" else 0
            cell_dose_axis = 2 if update.axis == "x" else 1
            gas_axis = 1 if update.axis == "x" else 0
            gas_dose_axis = 2 if update.axis == "x" else 1

            self._write_stripe_array(ctx, world.material_id, update, cell_payload["material_id"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.phase, update, cell_payload["phase"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.cell_flags, update, cell_payload["cell_flags"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.velocity, update, cell_payload["velocity"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.cell_temperature, update, cell_payload["cell_temperature"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.timer_pack, update, cell_payload["timer_pack"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.integrity, update, cell_payload["integrity"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.island_id, update, cell_payload["island_id"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(ctx, world.entity_id, update, cell_payload["entity_id"], stripe_axis=cell_axis, ranges=cell_ranges)
            self._write_stripe_array(
                ctx,
                world.placeholder_displaced_material,
                update,
                cell_payload["placeholder_displaced_material"],
                stripe_axis=cell_axis,
                ranges=cell_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.collapse_delay_pending,
                update,
                np.asarray(cell_payload["collapse_delay_pending"], dtype=np.uint8),
                stripe_axis=cell_axis,
                ranges=cell_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.visible_illumination,
                update,
                cell_payload["visible_illumination"],
                stripe_axis=cell_axis,
                ranges=cell_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.cell_optical_dose,
                update,
                cell_payload["cell_optical_dose"],
                stripe_axis=cell_dose_axis,
                ranges=cell_ranges,
            )

            self._write_stripe_array(
                ctx,
                world.ambient_temperature,
                update,
                gas_payload["ambient_temperature"],
                stripe_axis=gas_axis,
                ranges=gas_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.flow_velocity,
                update,
                gas_payload["flow_velocity"],
                stripe_axis=gas_axis,
                ranges=gas_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.pressure_ping,
                update,
                gas_payload["pressure_ping"],
                stripe_axis=gas_axis,
                ranges=gas_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.gas_concentration,
                update,
                gas_payload["gas_concentration"],
                stripe_axis=gas_dose_axis,
                ranges=gas_ranges,
            )
            self._write_stripe_array(
                ctx,
                world.gas_optical_dose,
                update,
                gas_payload["gas_optical_dose"],
                stripe_axis=gas_dose_axis,
                ranges=gas_ranges,
            )
            self.last_backend = "gpu"
            self.last_cpu_mirror_downloaded = True
        finally:
            if release_context:
                self._release_context_programs(ctx)
                try:
                    ctx.release()
                except Exception:
                    pass

    def capture(self, world: "WorldEngine", update: "PageStripeUpdate") -> dict[str, Any]:
        if not self._formal_gpu_frame(world) and getattr(world, "simulation_backend", "") != "gpu":
            raise RuntimeError("GPU page stripe capture is only available for GPU-authoritative worlds")
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU page stripe capture requires bridge GPU resources")
        required = {
            "cell_core",
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
        }
        missing = sorted(required.difference(bridge.gpu_authoritative_resources))
        if missing:
            raise RuntimeError(
                "GPU page stripe capture requires GPU-authoritative resources: " + ", ".join(missing)
            )

        cell_ranges = world._stripe_buffer_ranges(update, gas_grid=False)
        gas_ranges = world._stripe_buffer_ranges(update, gas_grid=True)
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1
        gas_axis = 1 if update.axis == "x" else 0
        gas_dose_axis = 2 if update.axis == "x" else 1
        bridge.ctx.finish()

        packed_cell_core = np.frombuffer(
            self._read_bridge_buffer(bridge.ctx, bridge.buffers["cell_core"]),
            dtype=np.uint32,
        ).reshape((world.height, world.width, 5))
        cell_core = unpack_cell_core(packed_cell_core)
        material_id = self._capture_array_stripe(cell_core["material_id"], stripe_axis=cell_axis, ranges=cell_ranges).astype(np.int32)
        phase = self._capture_array_stripe(cell_core["phase"], stripe_axis=cell_axis, ranges=cell_ranges).astype(np.uint8)
        island_id = self._capture_bridge_buffer_stripe(
            bridge.ctx,
            bridge.buffers["island_id"],
            (world.height, world.width),
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        entity_id = self._capture_bridge_buffer_stripe(
            bridge.ctx,
            bridge.buffers["entity_id"],
            (world.height, world.width),
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        placeholder_displaced_material = self._capture_bridge_buffer_stripe(
            bridge.ctx,
            bridge.buffers["placeholder_displaced_material"],
            (world.height, world.width),
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        phase, island_id, entity_id, placeholder_displaced_material = world._normalize_cell_runtime_arrays(
            material_id,
            phase,
            island_id,
            entity_id,
            placeholder_displaced_material,
        )
        runtime_payload = world._capture_page_stripe_island_runtime(island_id)
        runtime_payload["entity_placeholder_entity_id"] = world._capture_page_stripe_entity_placeholder_runtime(
            update,
            stripe_axis=cell_axis,
        )

        visible_rgba = np.frombuffer(
            self._read_bridge_texture(bridge.textures["visible_illumination"]),
            dtype=np.float32,
        ).reshape((world.height, world.width, 4))
        payload = {
            "meta": {
                "axis": update.axis,
                "world_start": update.world_start,
                "world_end": update.world_end,
                "buffer_start": update.buffer_start,
                "buffer_end": update.buffer_end,
                "kind": update.kind,
                "cross_world_start": update.cross_world_start,
                "cross_world_end": update.cross_world_end,
            },
            "cell": {
                "material_id": material_id,
                "phase": phase,
                "cell_flags": self._capture_array_stripe(cell_core["cell_flags"], stripe_axis=cell_axis, ranges=cell_ranges),
                "velocity": self._capture_array_stripe(cell_core["velocity"], stripe_axis=cell_axis, ranges=cell_ranges),
                "cell_temperature": self._capture_array_stripe(
                    cell_core["cell_temperature"],
                    stripe_axis=cell_axis,
                    ranges=cell_ranges,
                ),
                "timer_pack": self._capture_array_stripe(cell_core["timer_pack"], stripe_axis=cell_axis, ranges=cell_ranges),
                "integrity": self._capture_array_stripe(cell_core["integrity"], stripe_axis=cell_axis, ranges=cell_ranges),
                "island_id": island_id,
                "entity_id": entity_id,
                "placeholder_displaced_material": placeholder_displaced_material,
                "collapse_delay_pending": self._capture_bridge_buffer_stripe(
                    bridge.ctx,
                    bridge.buffers["collapse_delay_pending"],
                    (world.height, world.width),
                    np.int32,
                    stripe_axis=cell_axis,
                    ranges=cell_ranges,
                ).astype(np.uint8),
                "visible_illumination": self._capture_array_stripe(
                    visible_rgba[..., :3],
                    stripe_axis=cell_axis,
                    ranges=cell_ranges,
                ),
                "cell_optical_dose": self._capture_bridge_buffer_stripe(
                    bridge.ctx,
                    bridge.buffers["cell_optical_dose"],
                    tuple(int(dim) for dim in world.cell_optical_dose.shape),
                    np.float32,
                    stripe_axis=cell_dose_axis,
                    ranges=cell_ranges,
                ),
            },
            "runtime": runtime_payload,
            "gas": {
                "ambient_temperature": self._capture_bridge_texture_stripe(
                    bridge.textures["ambient_temperature"],
                    (world.gas_height, world.gas_width),
                    stripe_axis=gas_axis,
                    ranges=gas_ranges,
                ),
                "flow_velocity": self._capture_bridge_texture_stripe(
                    bridge.textures["flow_velocity"],
                    (world.gas_height, world.gas_width, 2),
                    stripe_axis=gas_axis,
                    ranges=gas_ranges,
                ),
                "pressure_ping": self._capture_bridge_texture_stripe(
                    bridge.textures["pressure_ping"],
                    (world.gas_height, world.gas_width),
                    stripe_axis=gas_axis,
                    ranges=gas_ranges,
                ),
                "gas_concentration": self._capture_bridge_buffer_stripe(
                    bridge.ctx,
                    bridge.buffers["gas_concentration"],
                    tuple(int(dim) for dim in world.gas_concentration.shape),
                    np.float32,
                    stripe_axis=gas_dose_axis,
                    ranges=gas_ranges,
                ),
                "gas_optical_dose": self._capture_bridge_buffer_stripe(
                    bridge.ctx,
                    bridge.buffers["gas_optical_dose"],
                    tuple(int(dim) for dim in world.gas_optical_dose.shape),
                    np.float32,
                    stripe_axis=gas_dose_axis,
                    ranges=gas_ranges,
                ),
            },
        }
        self.last_backend = "gpu"
        self.last_cpu_mirror_downloaded = False
        return payload

    def release(self) -> None:
        for program in self.programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.programs.clear()
        self.last_backend = "idle"

    def _context_for_current_thread(self, world: "WorldEngine") -> tuple[Any | None, bool]:
        if world.bridge.ctx is not None and world.bridge.owner_thread_id == threading.get_ident():
            return world.bridge.ctx, False
        if getattr(world, "simulation_backend", "") == "gpu":
            return None, False
        if moderngl is None:
            return world.bridge.ctx, False
        errors: list[Exception] = []
        for kwargs in ({"require": 430, "backend": "egl"}, {"require": 430}):
            try:
                return moderngl.create_standalone_context(**kwargs), True
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        return None, False

    def _release_context_programs(self, ctx: Any) -> None:
        prefix = f"{id(ctx)}:"
        for key in [key for key in self.programs if key.startswith(prefix)]:
            program = self.programs.pop(key)
            try:
                program.release()
            except Exception:
                pass

    def normalize_cell_runtime(self, world: "WorldEngine", update: "PageStripeUpdate") -> None:
        ctx, release_context = self._context_for_current_thread(world)
        if ctx is None or getattr(ctx, "version_code", 0) < 430:
            raise RuntimeError("GPU page stripe normalization requires a valid ModernGL context")
        if getattr(world, "simulation_backend", "") == "gpu":
            if release_context or ctx is not world.bridge.ctx:
                raise RuntimeError("GPU page stripe normalization must use the bridge context")
            self._normalize_bridge_cell_runtime(world, update)
            self.last_backend = "gpu"
            self.last_cpu_mirror_downloaded = False
            return
        material = np.ascontiguousarray(world.material_id, dtype=np.int32)
        phase = np.ascontiguousarray(world.phase, dtype=np.int32)
        island = np.ascontiguousarray(world.island_id, dtype=np.int32)
        entity = np.ascontiguousarray(world.entity_id, dtype=np.int32)
        displaced = np.ascontiguousarray(world.placeholder_displaced_material, dtype=np.int32)
        placeholder_flags = np.ascontiguousarray(world.material_is_placeholder.astype(np.int32, copy=False))
        material_buffer = ctx.buffer(reserve=max(4, material.nbytes), dynamic=True)
        phase_buffer = ctx.buffer(reserve=max(4, phase.nbytes), dynamic=True)
        island_buffer = ctx.buffer(reserve=max(4, island.nbytes), dynamic=True)
        entity_buffer = ctx.buffer(reserve=max(4, entity.nbytes), dynamic=True)
        displaced_buffer = ctx.buffer(reserve=max(4, displaced.nbytes), dynamic=True)
        placeholder_buffer = ctx.buffer(reserve=max(4, placeholder_flags.nbytes), dynamic=True)
        try:
            material_buffer.write(material.tobytes())
            phase_buffer.write(phase.tobytes())
            island_buffer.write(island.tobytes())
            entity_buffer.write(entity.tobytes())
            displaced_buffer.write(displaced.tobytes())
            if placeholder_flags.nbytes > 0:
                placeholder_buffer.write(placeholder_flags.tobytes())
            program = self._normalize_program(ctx)
            material_buffer.bind_to_storage_buffer(binding=0)
            phase_buffer.bind_to_storage_buffer(binding=1)
            island_buffer.bind_to_storage_buffer(binding=2)
            entity_buffer.bind_to_storage_buffer(binding=3)
            displaced_buffer.bind_to_storage_buffer(binding=4)
            placeholder_buffer.bind_to_storage_buffer(binding=5)
            ranges = world._stripe_buffer_ranges(update, gas_grid=False)
            axis_id = 1 if update.axis == "x" else 0
            width = int(world.width)
            height = int(world.height)
            program["grid_size"].value = (width, height)
            program["axis_id"].value = int(axis_id)
            program["falling_island_phase"].value = int(Phase.FALLING_ISLAND)
            program["placeholder_count"].value = int(placeholder_flags.shape[0])
            for start, end in ranges:
                span = int(end) - int(start)
                if span <= 0:
                    continue
                total = span * (height if axis_id == 1 else width)
                program["range_start"].value = int(start)
                program["range_span"].value = int(span)
                program["total_count"].value = int(total)
                program.run((total + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
                ctx.memory_barrier()
            ctx.finish()
            world.phase[:] = np.frombuffer(phase_buffer.read(size=phase.nbytes), dtype=np.int32).reshape(phase.shape).astype(
                world.phase.dtype
            )
            world.island_id[:] = np.frombuffer(island_buffer.read(size=island.nbytes), dtype=np.int32).reshape(island.shape)
            world.entity_id[:] = np.frombuffer(entity_buffer.read(size=entity.nbytes), dtype=np.int32).reshape(entity.shape)
            world.placeholder_displaced_material[:] = np.frombuffer(displaced_buffer.read(size=displaced.nbytes), dtype=np.int32).reshape(
                displaced.shape
            )
            self.last_backend = "gpu"
            self.last_cpu_mirror_downloaded = True
        finally:
            for buffer in (material_buffer, phase_buffer, island_buffer, entity_buffer, displaced_buffer, placeholder_buffer):
                try:
                    buffer.release()
                except Exception:
                    pass
            if release_context:
                self._release_context_programs(ctx)
                try:
                    ctx.release()
                except Exception:
                    pass

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _apply_bridge(self, world: "WorldEngine", update: "PageStripeUpdate", payload: dict[str, Any]) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU page stripe pipeline requires bridge GPU resources")
        cell_payload = payload["cell"]
        gas_payload = payload["gas"]
        cell_ranges = world._stripe_buffer_ranges(update, gas_grid=False)
        gas_ranges = world._stripe_buffer_ranges(update, gas_grid=True)
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1
        gas_axis = 1 if update.axis == "x" else 0
        gas_dose_axis = 2 if update.axis == "x" else 1

        cell_core = self._pack_cell_core_payload(cell_payload)
        self._write_bridge_buffer_stripe(
            bridge.buffers["cell_core"],
            (world.height, world.width, 5),
            cell_core,
            np.uint32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["island_id"],
            (world.height, world.width),
            cell_payload["island_id"],
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["entity_id"],
            (world.height, world.width),
            cell_payload["entity_id"],
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["placeholder_displaced_material"],
            (world.height, world.width),
            cell_payload["placeholder_displaced_material"],
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["collapse_delay_pending"],
            (world.height, world.width),
            np.asarray(cell_payload["collapse_delay_pending"], dtype=np.int32),
            np.int32,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["cell_optical_dose"],
            tuple(int(dim) for dim in world.cell_optical_dose.shape),
            cell_payload["cell_optical_dose"],
            np.float32,
            stripe_axis=cell_dose_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_texture_stripe(
            bridge.textures["material"],
            np.asarray(cell_payload["material_id"], dtype=np.float32),
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        visible = np.asarray(cell_payload["visible_illumination"], dtype=np.float32)
        visible_rgba = np.empty((visible.shape[0], visible.shape[1], 4), dtype=np.float32)
        visible_rgba[..., :3] = visible
        visible_rgba[..., 3] = 1.0
        self._write_bridge_texture_stripe(
            bridge.textures["light"],
            visible_rgba,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )
        self._write_bridge_texture_stripe(
            bridge.textures["visible_illumination"],
            visible_rgba,
            stripe_axis=cell_axis,
            ranges=cell_ranges,
        )

        self._write_bridge_texture_stripe(
            bridge.textures["ambient_temperature"],
            np.asarray(gas_payload["ambient_temperature"], dtype=np.float32),
            stripe_axis=gas_axis,
            ranges=gas_ranges,
        )
        self._write_bridge_texture_stripe(
            bridge.textures["pressure_ping"],
            np.asarray(gas_payload["pressure_ping"], dtype=np.float32),
            stripe_axis=gas_axis,
            ranges=gas_ranges,
        )
        self._write_bridge_texture_stripe(
            bridge.textures["flow_velocity"],
            np.asarray(gas_payload["flow_velocity"], dtype=np.float32),
            stripe_axis=gas_axis,
            ranges=gas_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["gas_concentration"],
            tuple(int(dim) for dim in world.gas_concentration.shape),
            gas_payload["gas_concentration"],
            np.float32,
            stripe_axis=gas_dose_axis,
            ranges=gas_ranges,
        )
        self._write_bridge_buffer_stripe(
            bridge.buffers["gas_optical_dose"],
            tuple(int(dim) for dim in world.gas_optical_dose.shape),
            gas_payload["gas_optical_dose"],
            np.float32,
            stripe_axis=gas_dose_axis,
            ranges=gas_ranges,
        )
        optics_pipeline = getattr(getattr(world, "optics_solver", None), "gpu_pipeline", None)
        invalidate_sparse = getattr(optics_pipeline, "invalidate_sparse_runtime", None)
        if callable(invalidate_sparse):
            invalidate_sparse()
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "light",
            "visible_illumination",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
            "ambient_temperature",
            "pressure_ping",
            "flow_velocity",
            "gas_concentration",
            "cell_optical_dose",
            "gas_optical_dose",
        )

    def _normalize_bridge_cell_runtime(self, world: "WorldEngine", update: "PageStripeUpdate") -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU page stripe normalization requires bridge GPU resources")
        placeholder_flags = np.ascontiguousarray(world.material_is_placeholder.astype(np.int32, copy=False))
        placeholder_buffer = bridge.ctx.buffer(reserve=max(4, placeholder_flags.nbytes), dynamic=True)
        try:
            if placeholder_flags.nbytes > 0:
                placeholder_buffer.write(placeholder_flags.tobytes())
            program = self._normalize_bridge_program(bridge.ctx)
            program["grid_size"].value = (int(world.width), int(world.height))
            program["axis_id"].value = 1 if update.axis == "x" else 0
            program["falling_island_phase"].value = int(Phase.FALLING_ISLAND)
            program["placeholder_count"].value = int(placeholder_flags.shape[0])
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            placeholder_buffer.bind_to_storage_buffer(binding=4)
            for start, end in world._stripe_buffer_ranges(update, gas_grid=False):
                span = int(end) - int(start)
                if span <= 0:
                    continue
                total = span * (world.height if update.axis == "x" else world.width)
                program["range_start"].value = int(start)
                program["range_span"].value = int(span)
                program["total_count"].value = int(total)
                program.run((total + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
            self._sync_compute_writes(bridge.ctx)
            bridge.mark_gpu_authoritative(
                "cell_core",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
            )
        finally:
            try:
                placeholder_buffer.release()
            except Exception:
                pass

    def _normalize_bridge_program(self, ctx: Any) -> Any:
        key = f"{id(ctx)}:normalize_bridge"
        program = self.programs.get(key)
        if program is not None:
            return program
        program = build_compute_shader(ctx, "page_stripes/normalize_bridge.comp", {"LOCAL_SIZE": LOCAL_SIZE})
        self.programs[key] = program
        return program

    def _pack_cell_core_payload(self, cell_payload: dict[str, Any]) -> np.ndarray:
        material_id = np.asarray(cell_payload["material_id"], dtype=np.uint32)
        phase = np.asarray(cell_payload["phase"], dtype=np.uint32)
        cell_flags = np.asarray(cell_payload["cell_flags"], dtype=np.uint32)
        velocity = np.asarray(cell_payload["velocity"], dtype=np.float32)
        cell_temperature = np.asarray(cell_payload["cell_temperature"], dtype=np.float32)
        timer_pack = np.asarray(cell_payload["timer_pack"], dtype=np.uint32)
        integrity = np.asarray(cell_payload["integrity"], dtype=np.float32)
        packed = np.zeros(material_id.shape + (5,), dtype=np.uint32)
        packed[..., 0] = material_id | (phase << 16) | (cell_flags << 24)
        packed[..., 1] = self._pack_half2x16(velocity)
        packed[..., 2] = cell_temperature.view(np.uint32)
        packed[..., 3] = (
            timer_pack[..., 0]
            | (timer_pack[..., 1] << 8)
            | (timer_pack[..., 2] << 16)
            | (timer_pack[..., 3] << 24)
        )
        packed[..., 4] = np.clip(np.rint(integrity), 0, 65535).astype(np.uint32)
        return np.ascontiguousarray(packed)

    @staticmethod
    def _pack_half2x16(velocity: np.ndarray) -> np.ndarray:
        half = np.asarray(velocity, dtype=np.float32).astype(np.float16)
        raw = half.view(np.uint16)
        return (raw[..., 0].astype(np.uint32) | (raw[..., 1].astype(np.uint32) << 16)).astype(np.uint32)

    def _write_bridge_buffer_stripe(
        self,
        buffer: Any,
        dst_shape: tuple[int, ...],
        values: np.ndarray,
        dtype: Any,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]],
    ) -> None:
        source = np.ascontiguousarray(values, dtype=dtype)
        itemsize = np.dtype(dtype).itemsize
        trailing_slices = (slice(None),) * (len(dst_shape) - stripe_axis - 1)
        prefix_shape = dst_shape[:stripe_axis]
        prefixes = [()] if not prefix_shape else np.ndindex(prefix_shape)
        source_offset = 0
        for start, end in ranges:
            span = int(end) - int(start)
            if span <= 0:
                continue
            for prefix in prefixes if isinstance(prefixes, list) else np.ndindex(prefix_shape):
                src_slice = prefix + (slice(source_offset, source_offset + span),) + trailing_slices
                data = np.ascontiguousarray(source[src_slice])
                if data.nbytes <= 0:
                    continue
                flat_coord = prefix + (int(start),) + ((0,) * (len(dst_shape) - stripe_axis - 1))
                flat_index = int(np.ravel_multi_index(flat_coord, dst_shape))
                buffer.write(data.tobytes(), offset=flat_index * itemsize)
            source_offset += span

    def _write_bridge_texture_stripe(
        self,
        texture: Any,
        values: np.ndarray,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]],
    ) -> None:
        source = np.ascontiguousarray(values, dtype=np.float32)
        source_offset = 0
        for start, end in ranges:
            span = int(end) - int(start)
            if span <= 0:
                continue
            if stripe_axis == 1:
                data = np.ascontiguousarray(source[:, source_offset : source_offset + span, ...], dtype=np.float32)
                viewport = (int(start), 0, int(span), int(source.shape[0]))
            elif stripe_axis == 0:
                data = np.ascontiguousarray(source[source_offset : source_offset + span, ...], dtype=np.float32)
                viewport = (0, int(start), int(source.shape[1]), int(span))
            else:
                raise ValueError("texture stripe axis must be 0 or 1")
            if data.nbytes > 0:
                texture.write(data.tobytes(), viewport=viewport)
            source_offset += span

    def _capture_bridge_buffer_stripe(
        self,
        ctx: Any,
        buffer: Any,
        src_shape: tuple[int, ...],
        dtype: Any,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]],
    ) -> np.ndarray:
        source = np.frombuffer(self._read_bridge_buffer(ctx, buffer), dtype=dtype).reshape(src_shape)
        return self._capture_array_stripe(source, stripe_axis=stripe_axis, ranges=ranges)

    def _capture_bridge_texture_stripe(
        self,
        texture: Any,
        src_shape: tuple[int, ...],
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]],
    ) -> np.ndarray:
        source = np.frombuffer(self._read_bridge_texture(texture), dtype=np.float32).reshape(src_shape)
        return self._capture_array_stripe(source, stripe_axis=stripe_axis, ranges=ranges)

    @staticmethod
    def _read_bridge_buffer(ctx: Any, buffer: Any) -> bytes:
        size = int(getattr(buffer, "size", 0))
        if size <= 0:
            return b""
        staging = ctx.buffer(reserve=size)
        try:
            ctx.copy_buffer(staging, buffer, size=size)
            ctx.finish()
            return staging.read(size=size)
        finally:
            try:
                staging.release()
            except Exception:
                pass

    @staticmethod
    def _read_bridge_texture(texture: Any) -> bytes:
        return texture.read()

    @staticmethod
    def _capture_array_stripe(
        source: np.ndarray,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]],
    ) -> np.ndarray:
        parts = []
        for start, end in ranges:
            slices = [slice(None)] * source.ndim
            slices[stripe_axis] = slice(int(start), int(end))
            parts.append(np.asarray(source[tuple(slices)]))
        if not parts:
            shape = list(source.shape)
            shape[stripe_axis] = 0
            return np.empty(tuple(shape), dtype=source.dtype)
        return np.ascontiguousarray(np.concatenate(parts, axis=stripe_axis))

    # ``_sync_compute_writes`` is inherited from :class:`GPUPipelineBase`
    # (default barrier bits match: image access + texture fetch + shader storage).

    def _program(self, ctx: Any, kind: str) -> Any:
        key = f"{id(ctx)}:{kind}"
        program = self.programs.get(key)
        if program is not None:
            return program
        scalar_type = "float" if kind == "float" else "int"
        program = build_compute_shader(ctx, "page_stripes/program.comp", {"LOCAL_SIZE": LOCAL_SIZE, "scalar_type": scalar_type})
        self.programs[key] = program
        return program

    def _normalize_program(self, ctx: Any) -> Any:
        key = f"{id(ctx)}:normalize"
        program = self.programs.get(key)
        if program is not None:
            return program
        program = build_compute_shader(ctx, "page_stripes/normalize.comp", {"LOCAL_SIZE": LOCAL_SIZE})
        self.programs[key] = program
        return program

    def _write_stripe_array(
        self,
        ctx: Any,
        array: np.ndarray,
        update: "PageStripeUpdate",
        values: np.ndarray,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        spans = ranges if ranges is not None else update_world_ranges(update)
        if not spans:
            return
        dst_dtype = array.dtype
        kind = "float" if np.issubdtype(dst_dtype, np.floating) else "int"
        work_dtype = np.float32 if kind == "float" else np.int32
        source = np.ascontiguousarray(values, dtype=work_dtype)
        dest = np.ascontiguousarray(array, dtype=work_dtype)
        src_shape = tuple(int(dim) for dim in source.shape)
        dst_shape = tuple(int(dim) for dim in dest.shape)
        padded_src_shape = (src_shape + (1, 1, 1))[:3]
        padded_dst_shape = (dst_shape + (1, 1, 1))[:3]
        src_buffer = ctx.buffer(reserve=max(4, source.nbytes), dynamic=True)
        dst_buffer = ctx.buffer(reserve=max(4, dest.nbytes), dynamic=True)
        try:
            if source.nbytes > 0:
                src_buffer.write(source.tobytes())
            if dest.nbytes > 0:
                dst_buffer.write(dest.tobytes())
            program = self._program(ctx, kind)
            offset = 0
            for start, end in spans:
                span = int(end) - int(start)
                if span <= 0:
                    continue
                copy_shape = list(src_shape)
                copy_shape[stripe_axis] = span
                total = int(np.prod(copy_shape, dtype=np.int64))
                if total <= 0:
                    offset += span
                    continue
                src_buffer.bind_to_storage_buffer(binding=0)
                dst_buffer.bind_to_storage_buffer(binding=1)
                program["total_count"].value = total
                program["ndim"].value = len(src_shape)
                program["stripe_axis"].value = int(stripe_axis)
                program["src_axis_offset"].value = int(offset)
                program["dst_axis_start"].value = int(start)
                program["axis_span"].value = int(span)
                program["src_shape"].value = padded_src_shape
                program["dst_shape"].value = padded_dst_shape
                program.run((total + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
                ctx.memory_barrier()
                offset += span
            ctx.finish()
            array[:] = np.frombuffer(dst_buffer.read(size=dest.nbytes), dtype=work_dtype).reshape(dest.shape).astype(dst_dtype)
        finally:
            try:
                src_buffer.release()
            except Exception:
                pass
            try:
                dst_buffer.release()
            except Exception:
                pass


def update_world_ranges(update: "PageStripeUpdate") -> list[tuple[int, int]]:
    size = int(update.buffer_end) - int(update.buffer_start)
    span = abs(size)
    if span <= 0:
        return []
    start = int(update.buffer_start)
    end = int(update.buffer_end)
    if start < end:
        return [(start, end)]
    return [(start, start + span)]
