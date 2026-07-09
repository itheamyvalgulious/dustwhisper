from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import PLACEHOLDER_DTYPE, RENDER_GROUP_IDS, typed_material_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader
from oracle_game.types import EntityPlaceholder, Phase


PASS_LOCAL_SIZE = 8
MAX_MATERIALS = 256


@dataclass(slots=True)
class GPUPlaceholderResources:
    signature: tuple[int, int, int, int]
    material: Any
    phase: Any
    flags: Any
    timer: Any
    temp: Any
    integrity: Any
    velocity: Any
    island: Any
    entity: Any
    displaced: Any
    ambient: Any
    placeholders: Any
    material_params: Any
    placeholder_capacity: int = 0
    material_params_signature: tuple[int, int] | None = None


class GPUPlaceholderPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUPlaceholderResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_backend = "idle"
        self.last_cpu_mirror_downloaded = False

    # ``available`` is inherited from :class:`GPUPipelineBase` (formerly inlined
    # here verbatim — no ``moderngl is not None`` fallback guard was present).

    def apply(self, world: "WorldEngine", placeholders: list[EntityPlaceholder]) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU placeholder pipeline requires a valid ModernGL context")
        self._ensure_programs(ctx)
        resources = self._ensure_resources(world)
        world.bridge.sync_rule_tables(world)
        placeholder_upload = self._pack_placeholder_upload(world, placeholders)
        self._upload_inputs(world, resources, placeholder_upload)
        self._load_authoritative_bridge_inputs(world, resources)
        program = self.programs["apply_placeholders"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["placeholder_count"].value = int(len(placeholder_upload))
        program["phase_liquid"].value = int(Phase.LIQUID)
        program["phase_falling_island"].value = int(Phase.FALLING_ISLAND)
        program["render_group_placeholder"].value = int(RENDER_GROUP_IDS["placeholder"])
        resources.material.bind_to_storage_buffer(binding=0)
        resources.phase.bind_to_storage_buffer(binding=1)
        resources.flags.bind_to_storage_buffer(binding=2)
        resources.timer.bind_to_storage_buffer(binding=3)
        resources.temp.bind_to_storage_buffer(binding=4)
        resources.integrity.bind_to_storage_buffer(binding=5)
        resources.velocity.bind_to_storage_buffer(binding=6)
        resources.island.bind_to_storage_buffer(binding=7)
        resources.entity.bind_to_storage_buffer(binding=8)
        resources.displaced.bind_to_storage_buffer(binding=9)
        resources.ambient.bind_to_storage_buffer(binding=10)
        resources.placeholders.bind_to_storage_buffer(binding=11)
        resources.material_params.bind_to_storage_buffer(binding=12)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
        self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources)
        self.last_backend = "gpu"

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.material,
            self.resources.phase,
            self.resources.flags,
            self.resources.timer,
            self.resources.temp,
            self.resources.integrity,
            self.resources.velocity,
            self.resources.island,
            self.resources.entity,
            self.resources.displaced,
            self.resources.ambient,
            self.resources.placeholders,
            self.resources.material_params,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def _ensure_resources(self, world: "WorldEngine") -> GPUPlaceholderResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height)
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        cell_count = world.width * world.height
        self.resources = GPUPlaceholderResources(
            signature=signature,
            material=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            phase=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            flags=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            timer=ctx.buffer(reserve=max(4, cell_count * 4 * 4), dynamic=True),
            temp=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            integrity=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            velocity=ctx.buffer(reserve=max(4, cell_count * 2 * 4), dynamic=True),
            island=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            entity=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            displaced=ctx.buffer(reserve=max(4, cell_count * 4), dynamic=True),
            ambient=ctx.buffer(reserve=max(4, world.gas_width * world.gas_height * 4), dynamic=True),
            placeholders=ctx.buffer(reserve=max(4, PLACEHOLDER_DTYPE.itemsize), dynamic=True),
            material_params=ctx.buffer(reserve=MAX_MATERIALS * 4 * 4, dynamic=True),
        )
        return self.resources

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["apply_placeholders"] = build_compute_shader(ctx, "placeholders/pass_00.comp", {"PASS_LOCAL_SIZE": PASS_LOCAL_SIZE, "MAX_MATERIALS": MAX_MATERIALS})
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "placeholders/pass_01.comp", {"PASS_LOCAL_SIZE": PASS_LOCAL_SIZE, "MAX_MATERIALS": MAX_MATERIALS})
        self.programs["load_bridge_ambient"] = build_compute_shader(ctx, "placeholders/pass_02.comp", {"PASS_LOCAL_SIZE": PASS_LOCAL_SIZE, "MAX_MATERIALS": MAX_MATERIALS})
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "placeholders/pass_03.comp", {"PASS_LOCAL_SIZE": PASS_LOCAL_SIZE, "MAX_MATERIALS": MAX_MATERIALS})

    def _pack_placeholder_upload(
        self,
        world: "WorldEngine",
        placeholders: list[EntityPlaceholder],
    ) -> np.ndarray:
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

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPUPlaceholderResources,
        placeholder_upload: np.ndarray,
    ) -> None:
        cell_count = world.width * world.height
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources(
            "placeholder input",
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
        )
        if not formal_gpu_frame:
            resources.material.write(np.ascontiguousarray(world.material_id.astype(np.int32)).tobytes())
            resources.phase.write(np.ascontiguousarray(world.phase.astype(np.int32)).tobytes())
            resources.flags.write(np.ascontiguousarray(world.cell_flags.astype(np.int32)).tobytes())
            resources.timer.write(np.ascontiguousarray(world.timer_pack.astype(np.int32).reshape(cell_count, 4)).tobytes())
            resources.temp.write(np.ascontiguousarray(world.cell_temperature.astype(np.float32)).tobytes())
            resources.integrity.write(np.ascontiguousarray(world.integrity.astype(np.float32)).tobytes())
            resources.velocity.write(np.ascontiguousarray(world.velocity.astype(np.float32).reshape(cell_count, 2)).tobytes())
            resources.island.write(np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes())
            resources.entity.write(np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes())
            resources.displaced.write(np.ascontiguousarray(world.placeholder_displaced_material.astype(np.int32)).tobytes())
            resources.ambient.write(np.ascontiguousarray(world.ambient_temperature.astype(np.float32)).tobytes())
        self._write_placeholder_buffer(world, resources, placeholder_upload)
        self._write_material_params(world, resources)

    def _write_placeholder_buffer(
        self,
        world: "WorldEngine",
        resources: GPUPlaceholderResources,
        placeholder_upload: np.ndarray,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        nbytes = max(4, int(placeholder_upload.nbytes))
        if resources.placeholders.size < nbytes:
            resources.placeholders.release()
            resources.placeholders = ctx.buffer(reserve=nbytes, dynamic=True)
            resources.placeholder_capacity = nbytes
        else:
            resources.placeholders.orphan(nbytes)
        if placeholder_upload.nbytes > 0:
            resources.placeholders.write(np.ascontiguousarray(placeholder_upload).tobytes())

    def _write_material_params(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        material_table = world.bridge.shadow_typed_tables["material_table"]
        signature = (world.bridge.table_generations.get("materials", 0), int(material_table.shape[0]))
        params = np.zeros((MAX_MATERIALS, 4), dtype=np.float32)
        params[:, 3] = np.nan
        count = min(MAX_MATERIALS, int(material_table.shape[0]))
        valid_indices = np.nonzero(material_table[:count]["name_hash"] != 0)[0]
        params[valid_indices, 0] = material_table[:count]["default_phase"][valid_indices].astype(np.float32)
        params[valid_indices, 1] = material_table[:count]["base_integrity"][valid_indices].astype(np.float32)
        params[valid_indices, 2] = material_table[:count]["render_group_id"][valid_indices].astype(np.float32)
        params[valid_indices, 3] = material_table[:count]["spawn_temperature"][valid_indices].astype(np.float32)
        resources.material_params.write(params.tobytes())
        resources.material_params_signature = signature

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _load_authoritative_bridge_inputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        if not self._formal_gpu_frame(world):
            return
        bridge = world.bridge
        authoritative = bridge.gpu_authoritative_resources
        copy_cell_core = "cell_core" in authoritative
        copy_island_id = "island_id" in authoritative
        copy_entity_id = "entity_id" in authoritative
        copy_displaced = "placeholder_displaced_material" in authoritative
        copy_ambient = "ambient_temperature" in authoritative
        if not (copy_cell_core or copy_island_id or copy_entity_id or copy_displaced or copy_ambient):
            return
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU placeholder pipeline requires bridge GPU resources for authoritative input state")
        if copy_cell_core or copy_island_id or copy_entity_id or copy_displaced:
            program = self.programs["load_bridge_cell"]
            program["cell_grid_size"].value = (world.width, world.height)
            program["copy_cell_core"].value = bool(copy_cell_core)
            program["copy_island_id"].value = bool(copy_island_id)
            program["copy_entity_id"].value = bool(copy_entity_id)
            program["copy_displaced_material"].value = bool(copy_displaced)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            bridge.buffers["island_id"].bind_to_storage_buffer(binding=1)
            bridge.buffers["entity_id"].bind_to_storage_buffer(binding=2)
            bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=3)
            resources.material.bind_to_storage_buffer(binding=4)
            resources.phase.bind_to_storage_buffer(binding=5)
            resources.flags.bind_to_storage_buffer(binding=6)
            resources.timer.bind_to_storage_buffer(binding=7)
            resources.temp.bind_to_storage_buffer(binding=8)
            resources.integrity.bind_to_storage_buffer(binding=9)
            resources.velocity.bind_to_storage_buffer(binding=10)
            resources.island.bind_to_storage_buffer(binding=11)
            resources.entity.bind_to_storage_buffer(binding=12)
            resources.displaced.bind_to_storage_buffer(binding=13)
            program.run(
                (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        if copy_ambient:
            program = self.programs["load_bridge_ambient"]
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            bridge.textures["ambient_temperature"].use(location=0)
            resources.ambient.bind_to_storage_buffer(binding=1)
            program.run(
                (world.gas_width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                (world.gas_height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(bridge.ctx)

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            if self._formal_gpu_frame(world):
                raise RuntimeError("GPU placeholder pipeline requires bridge GPU resources for authoritative output state")
            return
        program = self.programs["publish_bridge_cell"]
        program["cell_grid_size"].value = (world.width, world.height)
        resources.material.bind_to_storage_buffer(binding=0)
        resources.phase.bind_to_storage_buffer(binding=1)
        resources.flags.bind_to_storage_buffer(binding=2)
        resources.timer.bind_to_storage_buffer(binding=3)
        resources.temp.bind_to_storage_buffer(binding=4)
        resources.integrity.bind_to_storage_buffer(binding=5)
        resources.velocity.bind_to_storage_buffer(binding=6)
        resources.island.bind_to_storage_buffer(binding=7)
        resources.entity.bind_to_storage_buffer(binding=8)
        resources.displaced.bind_to_storage_buffer(binding=9)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=10)
        bridge.buffers["island_id"].bind_to_storage_buffer(binding=11)
        bridge.buffers["entity_id"].bind_to_storage_buffer(binding=12)
        bridge.buffers["placeholder_displaced_material"].bind_to_storage_buffer(binding=13)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        program.run(
            (world.width + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            (world.height + PASS_LOCAL_SIZE - 1) // PASS_LOCAL_SIZE,
            1,
        )
        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
        )

    # ``_sync_compute_writes`` is inherited from :class:`GPUPipelineBase`
    # (default ``_barrier_bits`` matches the three bits formerly inlined here:
    # SHADER_IMAGE_ACCESS / TEXTURE_FETCH / SHADER_STORAGE).

    def _download_outputs(self, world: "WorldEngine", resources: GPUPlaceholderResources) -> None:
        world.material_id[:] = np.frombuffer(resources.material.read(), dtype=np.int32).reshape((world.height, world.width))
        world.phase[:] = np.frombuffer(resources.phase.read(), dtype=np.int32).reshape((world.height, world.width)).astype(np.uint8)
        world.cell_flags[:] = np.frombuffer(resources.flags.read(), dtype=np.int32).reshape((world.height, world.width)).astype(np.uint8)
        timer = np.frombuffer(resources.timer.read(), dtype=np.int32).reshape((world.height, world.width, 4))
        world.timer_pack[:] = np.clip(timer, 0, 255).astype(np.uint8)
        world.cell_temperature[:] = np.frombuffer(resources.temp.read(), dtype=np.float32).reshape((world.height, world.width))
        world.integrity[:] = np.frombuffer(resources.integrity.read(), dtype=np.float32).reshape((world.height, world.width))
        world.velocity[:] = np.frombuffer(resources.velocity.read(), dtype=np.float32).reshape((world.height, world.width, 2))
        world.island_id[:] = np.frombuffer(resources.island.read(), dtype=np.int32).reshape((world.height, world.width))
        world.entity_id[:] = np.frombuffer(resources.entity.read(), dtype=np.int32).reshape((world.height, world.width))
        world.placeholder_displaced_material[:] = np.frombuffer(resources.displaced.read(), dtype=np.int32).reshape(
            (world.height, world.width)
        )
