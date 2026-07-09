from __future__ import annotations

from copy import deepcopy
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oracle_game.types import ReadbackRequest, ReadbackResult

from oracle_game.gpu._common import (
    CPU_READBACK_LATENCY_FRAMES,
    GPU_READBACK_LATENCY_FRAMES,
    MAX_REACTION_LIGHT_EMITTERS,
    _get_shared_standalone_context,
    _json_bytes,
    _render_group_tile,
    moderngl,
)
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
from oracle_game.gpu.packers import (
    pack_cell_core,
    pack_entity_state_upload,
    pack_force_source_upload,
    pack_island_runtime_upload,
    pack_frame_meta_upload,
    pack_world_command_upload,
    pack_readback_request_upload,
    pack_placeholder_upload,
    pack_placeholder_dirty_rect_upload,
    pack_active_meta_upload,
    pack_gas_runtime_upload,
    pack_heat_runtime_upload,
    pack_liquid_runtime_upload,
    pack_reaction_runtime_upload,
    pack_collapse_runtime_upload,
    pack_optics_runtime_upload,
    pack_page_stripe_upload,
    pack_material_table,
    pack_gas_table,
    pack_light_table,
    pack_optics_table,
    pack_reaction_action_table,
    pack_self_reaction_rule_table,
    _pack_pair_reaction_rules,
)
from oracle_game.gpu.readback import (
    GLReadbackSlot,
    ReadbackArrayLayout,
    ReadbackPayloadLayout,
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUTextureReadbackSource,
    GPUReadbackSegment,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    ReadbackPayloadPlan,
)


@dataclass(slots=True)
class GPUBridge:
    ctx: Any | None = None
    create_standalone: bool = True
    table_generations: dict[str, int] = field(default_factory=dict)
    shadow_tables: dict[str, Any] = field(default_factory=dict)
    shadow_typed_tables: dict[str, np.ndarray] = field(default_factory=dict)
    shadow_buffers: dict[str, np.ndarray] = field(default_factory=dict)
    textures: dict[str, Any] = field(default_factory=dict)
    buffers: dict[str, Any] = field(default_factory=dict)
    table_buffers: dict[str, Any] = field(default_factory=dict)
    typed_table_buffers: dict[str, Any] = field(default_factory=dict)
    readback_programs: dict[str, Any] = field(default_factory=dict)
    display_programs: dict[str, Any] = field(default_factory=dict)
    active_scheduler_programs: dict[str, Any] = field(default_factory=dict)
    readback_slots: list[GLReadbackSlot] = field(default_factory=lambda: [GLReadbackSlot(0), GLReadbackSlot(1)])
    gpu_authoritative_resources: set[str] = field(default_factory=set)
    write_index: int = 0
    own_context: bool = False
    enabled: bool = False
    owner_thread_id: int | None = None
    _force_cpu_resource_upload: bool = False
    world_signature: tuple[int, int, int, int, int] | None = None
    rule_table_signature: tuple[int, ...] | None = None
    atlas_grid: tuple[int, int] = (1, 1)
    atlas_dirty: bool = True

    def __post_init__(self) -> None:
        if self.ctx is not None:
            self.enabled = True
            self.owner_thread_id = threading.get_ident()
        elif self.create_standalone and moderngl is not None:
            try:
                self.ctx = _get_shared_standalone_context(require=430)
                self.own_context = False
                self.enabled = True
                self.owner_thread_id = threading.get_ident()
            except Exception:
                self.ctx = None
                self.enabled = False
                self.owner_thread_id = None

    def attach_context(self, ctx: Any) -> None:
        if self.own_context and self.ctx is not None:
            self.release()
        else:
            self._release_readback_programs()
            self._release_display_programs()
        self.ctx = ctx
        self.own_context = False
        self.enabled = True
        self.owner_thread_id = threading.get_ident()
        self.world_signature = None
        self.rule_table_signature = None
        self.textures.clear()
        self.buffers.clear()
        self.table_buffers.clear()
        self.typed_table_buffers.clear()
        self.readback_slots = [GLReadbackSlot(0), GLReadbackSlot(1)]
        self._release_active_scheduler_programs()
        self.gpu_authoritative_resources.clear()
        self.write_index = 0
        self.atlas_dirty = True

    def ensure_world_resources(self, world: "WorldEngine") -> None:
        if not self.enabled or self.ctx is None:
            return
        signature = (
            world.width,
            world.height,
            world.gas_width,
            world.gas_height,
            world.gas_concentration.shape[0],
            world.cell_optical_dose.shape[0],
        )
        if signature == self.world_signature:
            return
        self.release_resources()
        self.gpu_authoritative_resources.clear()
        self.world_signature = signature
        self.textures["material"] = self.ctx.texture((world.width, world.height), 1, dtype="f4")
        self.textures["light"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["debug"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["ambient_temperature"] = self.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        self.textures["pressure_ping"] = self.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        self.textures["flow_velocity"] = self.ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
        self.textures["visible_illumination"] = self.ctx.texture((world.width, world.height), 4, dtype="f4")
        self.textures["liquid_flow_intent"] = self.ctx.texture((world.width, world.height), 2, dtype="f4")
        for texture in self.textures.values():
            texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.buffers["cell_core"] = self.ctx.buffer(reserve=world.width * world.height * 5 * 4, dynamic=True)
        self.buffers["island_id"] = self.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
        self.buffers["entity_id"] = self.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
        self.buffers["placeholder_displaced_material"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_delay_pending"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["gas_concentration"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height * 4),
            dynamic=True,
        )
        self.buffers["cell_optical_dose"] = self.ctx.buffer(
            reserve=max(4, int(np.prod(world.cell_optical_dose.shape, dtype=np.int64)) * 4),
            dynamic=True,
        )
        self.buffers["gas_optical_dose"] = self.ctx.buffer(
            reserve=max(4, int(np.prod(world.gas_optical_dose.shape, dtype=np.int64)) * 4),
            dynamic=True,
        )
        self.buffers["entity_state"] = self.ctx.buffer(reserve=max(4, ENTITY_STATE_DTYPE.itemsize), dynamic=True)
        self.buffers["entity_state_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["force_source"] = self.ctx.buffer(reserve=max(4, FORCE_SOURCE_DTYPE.itemsize), dynamic=True)
        self.buffers["force_source_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_runtime"] = self.ctx.buffer(reserve=max(4, ISLAND_RUNTIME_DTYPE.itemsize), dynamic=True)
        self.buffers["island_runtime_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["powder_reservation"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["powder_reservation_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_reservation"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["island_reservation_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["world_command"] = self.ctx.buffer(reserve=max(4, WORLD_COMMAND_DTYPE.itemsize), dynamic=True)
        self.buffers["world_command_payload"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["readback_request"] = self.ctx.buffer(reserve=max(4, READBACK_REQUEST_DTYPE.itemsize), dynamic=True)
        self.buffers["readback_request_label"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["placeholder"] = self.ctx.buffer(reserve=max(4, PLACEHOLDER_DTYPE.itemsize), dynamic=True)
        self.buffers["placeholder_dirty_rect"] = self.ctx.buffer(
            reserve=max(4, PLACEHOLDER_DIRTY_RECT_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["active_meta"] = self.ctx.buffer(reserve=max(4, ACTIVE_META_DTYPE.itemsize), dynamic=True)
        self.buffers["active_tile_ttl"] = self.ctx.buffer(reserve=max(4, world.active.tile_width * world.active.tile_height * 4), dynamic=True)
        self.buffers["active_chunk_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.chunk_width * world.active.chunk_height * 4),
            dynamic=True,
        )
        active_chunk_count = max(1, int(world.active.chunk_width * world.active.chunk_height))
        self.buffers["active_chunk_list"] = self.ctx.buffer(reserve=max(8, active_chunk_count * 2 * 4), dynamic=True)
        self.buffers["active_chunk_count"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["active_chunk_dispatch_args"] = self.ctx.buffer(reserve=3 * 4, dynamic=True)
        self.buffers["gas_runtime_meta"] = self.ctx.buffer(reserve=max(4, GAS_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["gas_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["gas_species_runtime"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * GAS_SPECIES_RUNTIME_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["heat_runtime_meta"] = self.ctx.buffer(reserve=max(4, HEAT_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["heat_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["heat_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["heat_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["heat_phase_target"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
            dynamic=True,
        )
        self.buffers["heat_boil_target"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
            dynamic=True,
        )
        self.buffers["heat_condense_target"] = self.ctx.buffer(
            reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["liquid_runtime_meta"] = self.ctx.buffer(reserve=max(4, LIQUID_RUNTIME_META_DTYPE.itemsize), dynamic=True)
        self.buffers["liquid_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["liquid_post_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["liquid_post_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_vertical_seam_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_horizontal_seam_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_buoyancy_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["liquid_changed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, REACTION_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["reaction_timed_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_self_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_material_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_material_light_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_gas_gas_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_gas_light_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["reaction_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_changed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_changed_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_ambient_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["reaction_timer_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_emitted_light_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_emitted_material_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["reaction_light_emitter"] = self.ctx.buffer(
            reserve=MAX_REACTION_LIGHT_EMITTERS * 2 * 4 * 4,
            dynamic=True,
        )
        self.buffers["reaction_light_emitter_count"] = self.ctx.buffer(
            reserve=16 * 4,
            dynamic=True,
        )
        self.buffers["collapse_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, COLLAPSE_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["collapse_solve_region_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_structural_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_support_seed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_supported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_unsupported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_delayed_pending_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_immune_unsupported_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_collapsed_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_component_label"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height * 4),
            dynamic=True,
        )
        self.buffers["collapse_component"] = self.ctx.buffer(
            reserve=max(4, COLLAPSE_COMPONENT_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["optics_runtime_meta"] = self.ctx.buffer(
            reserve=max(4, OPTICS_RUNTIME_META_DTYPE.itemsize),
            dynamic=True,
        )
        self.buffers["optics_solve_tile_mask"] = self.ctx.buffer(
            reserve=max(4, world.active.tile_width * world.active.tile_height),
            dynamic=True,
        )
        self.buffers["optics_solve_cell_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_solve_gas_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["optics_visible_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_cell_dose_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["optics_gas_dose_changed_mask"] = self.ctx.buffer(
            reserve=max(4, world.gas_width * world.gas_height),
            dynamic=True,
        )
        self.buffers["optics_emitter_origin_mask"] = self.ctx.buffer(
            reserve=max(4, world.width * world.height),
            dynamic=True,
        )
        self.buffers["page_stripe_meta"] = self.ctx.buffer(reserve=max(4, PAGE_STRIPE_META_DTYPE.itemsize), dynamic=True)
        self.buffers["page_stripe_section"] = self.ctx.buffer(reserve=max(4, PAGE_STRIPE_SECTION_DTYPE.itemsize), dynamic=True)
        self.buffers["page_stripe_payload"] = self.ctx.buffer(reserve=4, dynamic=True)
        self.buffers["frame_meta"] = self.ctx.buffer(reserve=max(4, FRAME_META_DTYPE.itemsize), dynamic=True)
        self.atlas_dirty = True
        self._ensure_atlas_texture(world)

    def upload_table(self, name: str, payload: Any) -> None:
        data = _json_bytes(payload)
        self.shadow_tables[name] = json.loads(data.decode("utf-8"))
        self.table_generations[name] = self.table_generations.get(name, 0) + 1
        if not self.enabled or self.ctx is None:
            return
        buffer = self.table_buffers.get(name)
        if buffer is None or buffer.size < len(data):
            if buffer is not None:
                buffer.release()
            self.table_buffers[name] = self.ctx.buffer(data, dynamic=True)
        else:
            buffer.orphan(len(data))
            buffer.write(data)
        if name == "materials":
            self.atlas_dirty = True

    def sync_rule_tables(self, world: "WorldEngine") -> None:
        signature = (
            self.table_generations.get("materials", 0),
            self.table_generations.get("gases", 0),
            self.table_generations.get("lights", 0),
            self.table_generations.get("optics", 0),
            self.table_generations.get("reactions", 0),
        )
        buffers_ready = all(
            name in self.typed_table_buffers
            for name in (
                "rule_table_meta",
                "material_table",
                "gas_table",
                "light_table",
                "optics_table",
                "reaction_action_table",
                "material_material_rule_table",
                "material_gas_rule_table",
                "material_light_rule_table",
                "gas_gas_rule_table",
                "gas_light_rule_table",
                "self_rule_table",
            )
        )
        if signature == self.rule_table_signature and self.shadow_typed_tables and ((not self.enabled or self.ctx is None) or buffers_ready):
            return

        material_table = pack_material_table(world)
        gas_table = pack_gas_table(world)
        light_table = pack_light_table(world)
        optics_table = pack_optics_table(world)
        reaction_action_table = pack_reaction_action_table(world)
        material_material_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_material_rules)
        material_gas_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_gas_rules)
        material_light_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_light_rules)
        gas_gas_rule_table = _pack_pair_reaction_rules(world, world.rulebook.gas_gas_rules)
        gas_light_rule_table = _pack_pair_reaction_rules(world, world.rulebook.gas_light_rules)
        self_rule_table = pack_self_reaction_rule_table(world)
        rule_table_meta = np.zeros((1,), dtype=RULE_TABLE_META_DTYPE)
        rule_table_meta[0]["material_count"] = int(material_table.shape[0])
        rule_table_meta[0]["gas_count"] = int(gas_table.shape[0])
        rule_table_meta[0]["light_count"] = int(light_table.shape[0])
        rule_table_meta[0]["optics_count"] = int(optics_table.shape[0])
        rule_table_meta[0]["reaction_action_count"] = int(reaction_action_table.shape[0])
        rule_table_meta[0]["material_material_rule_count"] = int(material_material_rule_table.shape[0])
        rule_table_meta[0]["material_gas_rule_count"] = int(material_gas_rule_table.shape[0])
        rule_table_meta[0]["material_light_rule_count"] = int(material_light_rule_table.shape[0])
        rule_table_meta[0]["gas_gas_rule_count"] = int(gas_gas_rule_table.shape[0])
        rule_table_meta[0]["gas_light_rule_count"] = int(gas_light_rule_table.shape[0])
        rule_table_meta[0]["self_rule_count"] = int(self_rule_table.shape[0])
        rule_table_meta[0]["material_generation"] = int(self.table_generations.get("materials", 0))
        rule_table_meta[0]["gas_generation"] = int(self.table_generations.get("gases", 0))
        rule_table_meta[0]["light_generation"] = int(self.table_generations.get("lights", 0))
        rule_table_meta[0]["optics_generation"] = int(self.table_generations.get("optics", 0))
        rule_table_meta[0]["reaction_generation"] = int(self.table_generations.get("reactions", 0))

        self.shadow_typed_tables["rule_table_meta"] = rule_table_meta.copy()
        self.shadow_typed_tables["material_table"] = material_table.copy()
        self.shadow_typed_tables["gas_table"] = gas_table.copy()
        self.shadow_typed_tables["light_table"] = light_table.copy()
        self.shadow_typed_tables["optics_table"] = optics_table.copy()
        self.shadow_typed_tables["reaction_action_table"] = reaction_action_table.copy()
        self.shadow_typed_tables["material_material_rule_table"] = material_material_rule_table.copy()
        self.shadow_typed_tables["material_gas_rule_table"] = material_gas_rule_table.copy()
        self.shadow_typed_tables["material_light_rule_table"] = material_light_rule_table.copy()
        self.shadow_typed_tables["gas_gas_rule_table"] = gas_gas_rule_table.copy()
        self.shadow_typed_tables["gas_light_rule_table"] = gas_light_rule_table.copy()
        self.shadow_typed_tables["self_rule_table"] = self_rule_table.copy()

        if self.enabled and self.ctx is not None:
            self._write_typed_table_buffer("rule_table_meta", rule_table_meta)
            self._write_typed_table_buffer("material_table", material_table)
            self._write_typed_table_buffer("gas_table", gas_table)
            self._write_typed_table_buffer("light_table", light_table)
            self._write_typed_table_buffer("optics_table", optics_table)
            self._write_typed_table_buffer("reaction_action_table", reaction_action_table)
            self._write_typed_table_buffer("material_material_rule_table", material_material_rule_table)
            self._write_typed_table_buffer("material_gas_rule_table", material_gas_rule_table)
            self._write_typed_table_buffer("material_light_rule_table", material_light_rule_table)
            self._write_typed_table_buffer("gas_gas_rule_table", gas_gas_rule_table)
            self._write_typed_table_buffer("gas_light_rule_table", gas_light_rule_table)
            self._write_typed_table_buffer("self_rule_table", self_rule_table)

        self.rule_table_signature = signature

    def sync_world(
        self,
        world: "WorldEngine",
        *,
        debug_frame: np.ndarray | None = None,
        upload_debug_texture: bool = True,
        force_cpu_resource_upload: bool = False,
    ) -> None:
        previous_force_cpu_resource_upload = self._force_cpu_resource_upload
        self._force_cpu_resource_upload = bool(force_cpu_resource_upload)
        try:
            self._sync_world_impl(world, debug_frame=debug_frame, upload_debug_texture=upload_debug_texture)
        finally:
            self._force_cpu_resource_upload = previous_force_cpu_resource_upload

    def _sync_world_impl(
        self,
        world: "WorldEngine",
        *,
        debug_frame: np.ndarray | None = None,
        upload_debug_texture: bool = True,
    ) -> None:
        self.ensure_world_resources(world)
        self.sync_rule_tables(world)
        upload_solver_runtime_from_cpu = self._should_upload_cpu_solver_runtime(world)
        upload_island_runtime_from_cpu = self._should_upload_cpu_resource(world, "island_runtime")
        upload_powder_reservation_from_cpu = (
            upload_solver_runtime_from_cpu and self._should_upload_cpu_resource(world, "powder_reservation")
        )
        upload_island_reservation_from_cpu = (
            upload_solver_runtime_from_cpu and self._should_upload_cpu_resource(world, "island_reservation")
        )
        entity_state_upload = pack_entity_state_upload(world)
        entity_count_upload = np.array([len(entity_state_upload)], dtype=np.int32)
        force_source_upload = pack_force_source_upload(world)
        force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
        island_runtime_upload = (
            pack_island_runtime_upload(world)
            if upload_island_runtime_from_cpu
            else self.shadow_buffers.get("island_runtime", np.zeros((0,), dtype=ISLAND_RUNTIME_DTYPE))
        )
        island_runtime_count_upload = (
            np.array([len(island_runtime_upload)], dtype=np.int32)
            if upload_island_runtime_from_cpu
            else self.shadow_buffers.get("island_runtime_count", np.zeros((1,), dtype=np.int32))
        )
        motion_runtime = (
            world.motion_solver.runtime_snapshot()
            if upload_powder_reservation_from_cpu or upload_island_reservation_from_cpu
            else None
        )
        powder_reservation_upload = (
            motion_runtime["powder_reservations"]
            if upload_powder_reservation_from_cpu and motion_runtime is not None
            else self.shadow_buffers.get(
                "powder_reservation",
                np.zeros((0,), dtype=getattr(world.motion_solver, "last_powder_reservations").dtype),
            )
        )
        powder_reservation_count_upload = (
            np.array([len(powder_reservation_upload)], dtype=np.int32)
            if upload_powder_reservation_from_cpu
            else self.shadow_buffers.get("powder_reservation_count", np.zeros((1,), dtype=np.int32))
        )
        island_reservation_upload = (
            motion_runtime["island_reservations"]
            if upload_island_reservation_from_cpu and motion_runtime is not None
            else self.shadow_buffers.get(
                "island_reservation",
                np.zeros((0,), dtype=getattr(world.motion_solver, "last_island_reservations").dtype),
            )
        )
        island_reservation_count_upload = (
            np.array([len(island_reservation_upload)], dtype=np.int32)
            if upload_island_reservation_from_cpu
            else self.shadow_buffers.get("island_reservation_count", np.zeros((1,), dtype=np.int32))
        )
        world_command_upload, world_command_payload_upload = pack_world_command_upload(world)
        readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
        placeholder_upload = pack_placeholder_upload(world)
        placeholder_dirty_rect_upload = pack_placeholder_dirty_rect_upload(world)
        upload_active_tile_ttl_from_cpu = self._should_upload_cpu_resource(world, "active_tile_ttl")
        upload_active_chunk_mask_from_cpu = self._should_upload_cpu_resource(world, "active_chunk_mask")
        upload_active_meta_from_cpu = self._should_upload_cpu_resource(world, "active_meta")
        active_tile_ttl_default = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.int32)
        active_chunk_mask_default = np.zeros((world.active.chunk_height, world.active.chunk_width), dtype=np.uint8)
        active_tile_ttl_upload = (
            np.asarray(world.active.active_tile_ttl or [], dtype=np.int32)
            if upload_active_tile_ttl_from_cpu
            else self._shadow_or_default("active_tile_ttl", active_tile_ttl_default)
        )
        active_chunk_mask_upload = (
            np.asarray(world.active.active_chunk_mask or [], dtype=np.uint8)
            if upload_active_chunk_mask_from_cpu
            else self._shadow_or_default("active_chunk_mask", active_chunk_mask_default)
        )
        active_meta_default = pack_active_meta_upload(
            world,
            active_tile_count=int(np.count_nonzero(active_tile_ttl_default > 0)),
            active_chunk_count=int(np.count_nonzero(active_chunk_mask_default > 0)),
        )
        active_meta_upload = (
            pack_active_meta_upload(
                world,
                active_tile_count=int(np.count_nonzero(active_tile_ttl_upload > 0)),
                active_chunk_count=int(np.count_nonzero(active_chunk_mask_upload > 0)),
            )
            if upload_active_meta_from_cpu
            else self._shadow_or_default("active_meta", active_meta_default)
        )
        if upload_solver_runtime_from_cpu:
            gas_runtime_meta_upload, gas_solve_tile_mask_upload, gas_solve_gas_mask_upload, gas_species_runtime_upload = pack_gas_runtime_upload(world)
            (
                heat_runtime_meta_upload,
                heat_solve_tile_mask_upload,
                heat_solve_cell_mask_upload,
                heat_solve_gas_mask_upload,
                heat_phase_target_upload,
                heat_boil_target_upload,
                heat_condense_target_upload,
            ) = pack_heat_runtime_upload(world)
            (
                liquid_runtime_meta_upload,
                liquid_solve_tile_mask_upload,
                liquid_post_tile_mask_upload,
                liquid_post_cell_mask_upload,
                liquid_vertical_seam_mask_upload,
                liquid_horizontal_seam_mask_upload,
                liquid_buoyancy_mask_upload,
                liquid_changed_cell_mask_upload,
            ) = pack_liquid_runtime_upload(world)
            (
                reaction_runtime_meta_upload,
                reaction_timed_solve_tile_mask_upload,
                reaction_self_solve_tile_mask_upload,
                reaction_material_material_solve_tile_mask_upload,
                reaction_material_gas_solve_tile_mask_upload,
                reaction_material_light_solve_tile_mask_upload,
                reaction_gas_gas_solve_tile_mask_upload,
                reaction_gas_light_solve_tile_mask_upload,
                reaction_solve_cell_mask_upload,
                reaction_solve_gas_mask_upload,
                reaction_changed_cell_mask_upload,
                reaction_changed_gas_mask_upload,
                reaction_ambient_changed_mask_upload,
                reaction_timer_changed_mask_upload,
                reaction_emitted_light_mask_upload,
                reaction_emitted_material_mask_upload,
            ) = pack_reaction_runtime_upload(world)
        else:
            gas_runtime_meta_upload = self._shadow_or_default("gas_runtime_meta", np.zeros((1,), dtype=GAS_RUNTIME_META_DTYPE))
            gas_solve_tile_mask_upload = self._shadow_or_default(
                "gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            gas_solve_gas_mask_upload = self._shadow_or_default(
                "gas_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            gas_species_runtime_upload = self._shadow_or_default(
                "gas_species_runtime",
                np.zeros((world.gas_concentration.shape[0],), dtype=GAS_SPECIES_RUNTIME_DTYPE),
            )
            heat_runtime_meta_upload = self._shadow_or_default("heat_runtime_meta", np.zeros((1,), dtype=HEAT_RUNTIME_META_DTYPE))
            heat_solve_tile_mask_upload = self._shadow_or_default(
                "heat_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            heat_solve_cell_mask_upload = self._shadow_or_default(
                "heat_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            heat_solve_gas_mask_upload = self._shadow_or_default(
                "heat_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            heat_phase_target_upload = self._shadow_or_default(
                "heat_phase_target",
                np.zeros((world.height, world.width), dtype=np.int32),
            )
            heat_boil_target_upload = self._shadow_or_default(
                "heat_boil_target",
                np.zeros((world.height, world.width), dtype=np.int32),
            )
            heat_condense_target_upload = self._shadow_or_default(
                "heat_condense_target",
                np.zeros(world.gas_concentration.shape, dtype=np.uint8),
            )
            liquid_runtime_meta_upload = self._shadow_or_default("liquid_runtime_meta", np.zeros((1,), dtype=LIQUID_RUNTIME_META_DTYPE))
            liquid_solve_tile_mask_upload = self._shadow_or_default(
                "liquid_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            liquid_post_tile_mask_upload = self._shadow_or_default(
                "liquid_post_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            liquid_post_cell_mask_upload = self._shadow_or_default(
                "liquid_post_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_vertical_seam_mask_upload = self._shadow_or_default(
                "liquid_vertical_seam_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_horizontal_seam_mask_upload = self._shadow_or_default(
                "liquid_horizontal_seam_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_buoyancy_mask_upload = self._shadow_or_default(
                "liquid_buoyancy_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            liquid_changed_cell_mask_upload = self._shadow_or_default(
                "liquid_changed_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_runtime_meta_upload = self._shadow_or_default("reaction_runtime_meta", np.zeros((1,), dtype=REACTION_RUNTIME_META_DTYPE))
            reaction_timed_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_timed_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_self_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_self_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_material_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_material_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_gas_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_material_light_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_material_light_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_gas_gas_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_gas_gas_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_gas_light_solve_tile_mask_upload = self._shadow_or_default(
                "reaction_gas_light_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            reaction_solve_cell_mask_upload = self._shadow_or_default(
                "reaction_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_solve_gas_mask_upload = self._shadow_or_default(
                "reaction_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_changed_cell_mask_upload = self._shadow_or_default(
                "reaction_changed_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_changed_gas_mask_upload = self._shadow_or_default(
                "reaction_changed_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_ambient_changed_mask_upload = self._shadow_or_default(
                "reaction_ambient_changed_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            reaction_timer_changed_mask_upload = self._shadow_or_default(
                "reaction_timer_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_emitted_light_mask_upload = self._shadow_or_default(
                "reaction_emitted_light_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            reaction_emitted_material_mask_upload = self._shadow_or_default(
                "reaction_emitted_material_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
        collapse_mask_resources = (
            "collapse_structural_mask",
            "collapse_support_seed_mask",
            "collapse_supported_mask",
            "collapse_unsupported_mask",
            "collapse_delayed_pending_mask",
            "collapse_immune_unsupported_mask",
            "collapse_collapsed_cell_mask",
        )
        upload_collapse_runtime_from_cpu = upload_solver_runtime_from_cpu and not (
            any(name in self.gpu_authoritative_resources for name in collapse_mask_resources)
        )
        if upload_collapse_runtime_from_cpu:
            (
                collapse_runtime_meta_upload,
                collapse_solve_region_mask_upload,
                collapse_structural_mask_upload,
                collapse_support_seed_mask_upload,
                collapse_supported_mask_upload,
                collapse_unsupported_mask_upload,
                collapse_delayed_pending_mask_upload,
                collapse_immune_unsupported_mask_upload,
                collapse_collapsed_cell_mask_upload,
                collapse_component_upload,
            ) = pack_collapse_runtime_upload(world)
        else:
            cell_zero = np.zeros((world.height, world.width), dtype=np.int32)
            collapse_runtime_meta_upload = self.shadow_buffers.get(
                "collapse_runtime_meta",
                np.zeros((1,), dtype=COLLAPSE_RUNTIME_META_DTYPE),
            )
            collapse_solve_region_mask_upload = self.shadow_buffers.get("collapse_solve_region_mask", cell_zero)
            collapse_structural_mask_upload = self.shadow_buffers.get("collapse_structural_mask", cell_zero)
            collapse_support_seed_mask_upload = self.shadow_buffers.get("collapse_support_seed_mask", cell_zero)
            collapse_supported_mask_upload = self.shadow_buffers.get("collapse_supported_mask", cell_zero)
            collapse_unsupported_mask_upload = self.shadow_buffers.get("collapse_unsupported_mask", cell_zero)
            collapse_delayed_pending_mask_upload = self.shadow_buffers.get("collapse_delayed_pending_mask", cell_zero)
            collapse_immune_unsupported_mask_upload = self.shadow_buffers.get("collapse_immune_unsupported_mask", cell_zero)
            collapse_collapsed_cell_mask_upload = self.shadow_buffers.get("collapse_collapsed_cell_mask", cell_zero)
            collapse_component_upload = self.shadow_buffers.get(
                "collapse_component",
                np.zeros((0,), dtype=COLLAPSE_COMPONENT_DTYPE),
            )
        if upload_solver_runtime_from_cpu:
            (
                optics_runtime_meta_upload,
                optics_solve_tile_mask_upload,
                optics_solve_cell_mask_upload,
                optics_solve_gas_mask_upload,
                optics_visible_changed_mask_upload,
                optics_cell_dose_changed_mask_upload,
                optics_gas_dose_changed_mask_upload,
                optics_emitter_origin_mask_upload,
            ) = pack_optics_runtime_upload(world)
        else:
            optics_runtime_meta_upload = self._shadow_or_default("optics_runtime_meta", np.zeros((1,), dtype=OPTICS_RUNTIME_META_DTYPE))
            optics_solve_tile_mask_upload = self._shadow_or_default(
                "optics_solve_tile_mask",
                np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
            )
            optics_solve_cell_mask_upload = self._shadow_or_default(
                "optics_solve_cell_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_solve_gas_mask_upload = self._shadow_or_default(
                "optics_solve_gas_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            optics_visible_changed_mask_upload = self._shadow_or_default(
                "optics_visible_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_cell_dose_changed_mask_upload = self._shadow_or_default(
                "optics_cell_dose_changed_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
            optics_gas_dose_changed_mask_upload = self._shadow_or_default(
                "optics_gas_dose_changed_mask",
                np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
            )
            optics_emitter_origin_mask_upload = self._shadow_or_default(
                "optics_emitter_origin_mask",
                np.zeros((world.height, world.width), dtype=np.uint8),
            )
        page_stripe_meta_upload, page_stripe_section_upload, page_stripe_payload_upload = pack_page_stripe_upload(world)
        frame_meta_upload = pack_frame_meta_upload(
            world,
            entity_count=len(entity_state_upload),
            force_source_count=len(force_source_upload),
            world_command_count=len(world_command_upload),
            readback_request_count=len(readback_request_upload),
            placeholder_count=len(placeholder_upload),
            placeholder_dirty_rect_count=len(placeholder_dirty_rect_upload),
            active_tile_count=int(active_meta_upload[0]["active_tile_count"]),
            active_chunk_count=int(active_meta_upload[0]["active_chunk_count"]),
            page_update_count=len(page_stripe_meta_upload),
            page_stripe_section_count=len(page_stripe_section_upload),
        )
        self.shadow_buffers["entity_state"] = entity_state_upload.copy()
        self.shadow_buffers["entity_state_count"] = entity_count_upload.copy()
        self.shadow_buffers["force_source"] = force_source_upload.copy()
        self.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
        if upload_island_runtime_from_cpu or "island_runtime" not in self.shadow_buffers:
            self.shadow_buffers["island_runtime"] = island_runtime_upload.copy()
            self.shadow_buffers["island_runtime_count"] = island_runtime_count_upload.copy()
        if upload_powder_reservation_from_cpu or "powder_reservation" not in self.shadow_buffers:
            self.shadow_buffers["powder_reservation"] = powder_reservation_upload.copy()
            self.shadow_buffers["powder_reservation_count"] = powder_reservation_count_upload.copy()
        if upload_island_reservation_from_cpu or "island_reservation" not in self.shadow_buffers:
            self.shadow_buffers["island_reservation"] = island_reservation_upload.copy()
            self.shadow_buffers["island_reservation_count"] = island_reservation_count_upload.copy()
        self.shadow_buffers["world_command"] = world_command_upload.copy()
        self.shadow_buffers["world_command_payload"] = world_command_payload_upload.copy()
        self.shadow_buffers["readback_request"] = readback_request_upload.copy()
        self.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
        self.shadow_buffers["placeholder"] = placeholder_upload.copy()
        self.shadow_buffers["placeholder_dirty_rect"] = placeholder_dirty_rect_upload.copy()
        self.shadow_buffers["island_id"] = world.island_id.astype(np.int32).copy()
        self.shadow_buffers["entity_id"] = world.entity_id.astype(np.int32).copy()
        self.shadow_buffers["placeholder_displaced_material"] = world.placeholder_displaced_material.astype(np.int32).copy()
        self.shadow_buffers["collapse_delay_pending"] = world.collapse_delay_pending.astype(np.int32).copy()
        self.shadow_buffers["cell_optical_dose"] = world.cell_optical_dose.astype(np.float32).copy()
        self.shadow_buffers["gas_optical_dose"] = world.gas_optical_dose.astype(np.float32).copy()
        self.shadow_buffers["active_meta"] = active_meta_upload.copy()
        self.shadow_buffers["active_tile_ttl"] = active_tile_ttl_upload.copy()
        self.shadow_buffers["active_chunk_mask"] = active_chunk_mask_upload.copy()
        self.shadow_buffers["gas_runtime_meta"] = gas_runtime_meta_upload.copy()
        self.shadow_buffers["gas_solve_tile_mask"] = gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["gas_solve_gas_mask"] = gas_solve_gas_mask_upload.copy()
        self.shadow_buffers["gas_species_runtime"] = gas_species_runtime_upload.copy()
        self.shadow_buffers["heat_runtime_meta"] = heat_runtime_meta_upload.copy()
        self.shadow_buffers["heat_solve_tile_mask"] = heat_solve_tile_mask_upload.copy()
        self.shadow_buffers["heat_solve_cell_mask"] = heat_solve_cell_mask_upload.copy()
        self.shadow_buffers["heat_solve_gas_mask"] = heat_solve_gas_mask_upload.copy()
        self.shadow_buffers["heat_phase_target"] = heat_phase_target_upload.copy()
        self.shadow_buffers["heat_boil_target"] = heat_boil_target_upload.copy()
        self.shadow_buffers["heat_condense_target"] = heat_condense_target_upload.copy()
        self.shadow_buffers["liquid_runtime_meta"] = liquid_runtime_meta_upload.copy()
        self.shadow_buffers["liquid_solve_tile_mask"] = liquid_solve_tile_mask_upload.copy()
        self.shadow_buffers["liquid_post_tile_mask"] = liquid_post_tile_mask_upload.copy()
        self.shadow_buffers["liquid_post_cell_mask"] = liquid_post_cell_mask_upload.copy()
        self.shadow_buffers["liquid_vertical_seam_mask"] = liquid_vertical_seam_mask_upload.copy()
        self.shadow_buffers["liquid_horizontal_seam_mask"] = liquid_horizontal_seam_mask_upload.copy()
        self.shadow_buffers["liquid_buoyancy_mask"] = liquid_buoyancy_mask_upload.copy()
        self.shadow_buffers["liquid_changed_cell_mask"] = liquid_changed_cell_mask_upload.copy()
        self.shadow_buffers["reaction_runtime_meta"] = reaction_runtime_meta_upload.copy()
        self.shadow_buffers["reaction_timed_solve_tile_mask"] = reaction_timed_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_self_solve_tile_mask"] = reaction_self_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_material_solve_tile_mask"] = reaction_material_material_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_gas_solve_tile_mask"] = reaction_material_gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_material_light_solve_tile_mask"] = reaction_material_light_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_gas_gas_solve_tile_mask"] = reaction_gas_gas_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_gas_light_solve_tile_mask"] = reaction_gas_light_solve_tile_mask_upload.copy()
        self.shadow_buffers["reaction_solve_cell_mask"] = reaction_solve_cell_mask_upload.copy()
        self.shadow_buffers["reaction_solve_gas_mask"] = reaction_solve_gas_mask_upload.copy()
        self.shadow_buffers["reaction_changed_cell_mask"] = reaction_changed_cell_mask_upload.copy()
        self.shadow_buffers["reaction_changed_gas_mask"] = reaction_changed_gas_mask_upload.copy()
        self.shadow_buffers["reaction_ambient_changed_mask"] = reaction_ambient_changed_mask_upload.copy()
        self.shadow_buffers["reaction_timer_changed_mask"] = reaction_timer_changed_mask_upload.copy()
        self.shadow_buffers["reaction_emitted_light_mask"] = reaction_emitted_light_mask_upload.copy()
        self.shadow_buffers["reaction_emitted_material_mask"] = reaction_emitted_material_mask_upload.copy()
        self.shadow_buffers["collapse_runtime_meta"] = collapse_runtime_meta_upload.copy()
        self.shadow_buffers["collapse_solve_region_mask"] = collapse_solve_region_mask_upload.copy()
        self.shadow_buffers["collapse_structural_mask"] = collapse_structural_mask_upload.copy()
        self.shadow_buffers["collapse_support_seed_mask"] = collapse_support_seed_mask_upload.copy()
        self.shadow_buffers["collapse_supported_mask"] = collapse_supported_mask_upload.copy()
        self.shadow_buffers["collapse_unsupported_mask"] = collapse_unsupported_mask_upload.copy()
        self.shadow_buffers["collapse_delayed_pending_mask"] = collapse_delayed_pending_mask_upload.copy()
        self.shadow_buffers["collapse_immune_unsupported_mask"] = collapse_immune_unsupported_mask_upload.copy()
        self.shadow_buffers["collapse_collapsed_cell_mask"] = collapse_collapsed_cell_mask_upload.copy()
        self.shadow_buffers["collapse_component"] = collapse_component_upload.copy()
        self.shadow_buffers["optics_runtime_meta"] = optics_runtime_meta_upload.copy()
        self.shadow_buffers["optics_solve_tile_mask"] = optics_solve_tile_mask_upload.copy()
        self.shadow_buffers["optics_solve_cell_mask"] = optics_solve_cell_mask_upload.copy()
        self.shadow_buffers["optics_solve_gas_mask"] = optics_solve_gas_mask_upload.copy()
        self.shadow_buffers["optics_visible_changed_mask"] = optics_visible_changed_mask_upload.copy()
        self.shadow_buffers["optics_cell_dose_changed_mask"] = optics_cell_dose_changed_mask_upload.copy()
        self.shadow_buffers["optics_gas_dose_changed_mask"] = optics_gas_dose_changed_mask_upload.copy()
        self.shadow_buffers["optics_emitter_origin_mask"] = optics_emitter_origin_mask_upload.copy()
        self.shadow_buffers["page_stripe_meta"] = page_stripe_meta_upload.copy()
        self.shadow_buffers["page_stripe_section"] = page_stripe_section_upload.copy()
        self.shadow_buffers["page_stripe_payload"] = page_stripe_payload_upload.copy()
        self.shadow_buffers["frame_meta"] = frame_meta_upload.copy()
        if not self.enabled or self.ctx is None:
            return
        self._ensure_atlas_texture(world)
        upload_cell_dose_from_cpu = self._should_upload_cpu_resource(world, "cell_optical_dose")
        upload_gas_dose_from_cpu = self._should_upload_cpu_resource(world, "gas_optical_dose")
        upload_light_from_cpu = self._should_upload_cpu_resource(world, "light")
        upload_visible_from_cpu = self._should_upload_cpu_resource(world, "visible_illumination")
        if upload_cell_dose_from_cpu or upload_gas_dose_from_cpu or upload_light_from_cpu or upload_visible_from_cpu:
            world._gpu_optics_outputs_clear = False
        if self._should_upload_cpu_resource(world, "cell_core"):
            packed = pack_cell_core(world)
            self.buffers["cell_core"].write(packed.tobytes())
        if self._should_upload_cpu_resource(world, "island_id"):
            self.buffers["island_id"].write(np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes())
        if self._should_upload_cpu_resource(world, "entity_id"):
            self.buffers["entity_id"].write(np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes())
        if self._should_upload_cpu_resource(world, "placeholder_displaced_material"):
            self.buffers["placeholder_displaced_material"].write(
                np.ascontiguousarray(world.placeholder_displaced_material.astype(np.int32)).tobytes()
            )
        if self._should_upload_cpu_resource(world, "collapse_delay_pending"):
            self.buffers["collapse_delay_pending"].write(
                np.ascontiguousarray(world.collapse_delay_pending.astype(np.int32)).tobytes()
            )
        if self._should_upload_cpu_resource(world, "gas_concentration"):
            self.buffers["gas_concentration"].write(world.gas_concentration.astype("f4").tobytes())
        if upload_cell_dose_from_cpu:
            self.buffers["cell_optical_dose"].write(np.ascontiguousarray(world.cell_optical_dose.astype(np.float32)).tobytes())
        if upload_gas_dose_from_cpu:
            self.buffers["gas_optical_dose"].write(np.ascontiguousarray(world.gas_optical_dose.astype(np.float32)).tobytes())
        self._write_dynamic_buffer("entity_state", entity_state_upload)
        self.buffers["entity_state_count"].write(entity_count_upload.tobytes())
        self._write_dynamic_buffer("force_source", force_source_upload)
        self.buffers["force_source_count"].write(force_source_count_upload.tobytes())
        if upload_island_runtime_from_cpu:
            self._write_dynamic_buffer("island_runtime", island_runtime_upload)
            self.buffers["island_runtime_count"].write(island_runtime_count_upload.tobytes())
        if upload_powder_reservation_from_cpu:
            self._write_dynamic_buffer("powder_reservation", powder_reservation_upload)
            self.buffers["powder_reservation_count"].write(powder_reservation_count_upload.tobytes())
        if upload_island_reservation_from_cpu:
            self._write_dynamic_buffer("island_reservation", island_reservation_upload)
            self.buffers["island_reservation_count"].write(island_reservation_count_upload.tobytes())
        self._write_dynamic_buffer("world_command", world_command_upload)
        self._write_dynamic_buffer("world_command_payload", world_command_payload_upload)
        self._write_dynamic_buffer("readback_request", readback_request_upload)
        self._write_dynamic_buffer("readback_request_label", readback_request_label_upload)
        self._write_dynamic_buffer("placeholder", placeholder_upload)
        self._write_dynamic_buffer("placeholder_dirty_rect", placeholder_dirty_rect_upload)
        if self._should_upload_cpu_resource(world, "active_meta"):
            self._write_dynamic_buffer("active_meta", active_meta_upload)
        if self._should_upload_cpu_resource(world, "active_tile_ttl"):
            self._write_dynamic_buffer("active_tile_ttl", active_tile_ttl_upload)
        if self._should_upload_cpu_resource(world, "active_chunk_mask"):
            self._write_dynamic_buffer("active_chunk_mask", active_chunk_mask_upload.astype(np.int32, copy=False))
        if (
            getattr(world, "simulation_backend", "") == "gpu"
            and (
                upload_active_meta_from_cpu
                or upload_active_tile_ttl_from_cpu
                or upload_active_chunk_mask_from_cpu
            )
        ):
            self._ensure_active_scheduler_programs()
            self._refresh_active_chunks_and_meta(world, read_meta=False)
        self._write_dynamic_buffer("gas_runtime_meta", gas_runtime_meta_upload)
        self._write_dynamic_buffer("gas_solve_tile_mask", gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("gas_solve_gas_mask", gas_solve_gas_mask_upload)
        self._write_dynamic_buffer("gas_species_runtime", gas_species_runtime_upload)
        self._write_dynamic_buffer("heat_runtime_meta", heat_runtime_meta_upload)
        self._write_dynamic_buffer("heat_solve_tile_mask", heat_solve_tile_mask_upload)
        self._write_dynamic_buffer("heat_solve_cell_mask", heat_solve_cell_mask_upload)
        self._write_dynamic_buffer("heat_solve_gas_mask", heat_solve_gas_mask_upload)
        self._write_dynamic_buffer("heat_phase_target", heat_phase_target_upload)
        self._write_dynamic_buffer("heat_boil_target", heat_boil_target_upload)
        self._write_dynamic_buffer("heat_condense_target", heat_condense_target_upload)
        self._write_dynamic_buffer("liquid_runtime_meta", liquid_runtime_meta_upload)
        self._write_dynamic_buffer("liquid_solve_tile_mask", liquid_solve_tile_mask_upload)
        self._write_dynamic_buffer("liquid_post_tile_mask", liquid_post_tile_mask_upload)
        self._write_dynamic_buffer("liquid_post_cell_mask", liquid_post_cell_mask_upload)
        self._write_dynamic_buffer("liquid_vertical_seam_mask", liquid_vertical_seam_mask_upload)
        self._write_dynamic_buffer("liquid_horizontal_seam_mask", liquid_horizontal_seam_mask_upload)
        self._write_dynamic_buffer("liquid_buoyancy_mask", liquid_buoyancy_mask_upload)
        self._write_dynamic_buffer("liquid_changed_cell_mask", liquid_changed_cell_mask_upload)
        self._write_dynamic_buffer("reaction_runtime_meta", reaction_runtime_meta_upload)
        self._write_dynamic_buffer("reaction_timed_solve_tile_mask", reaction_timed_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_self_solve_tile_mask", reaction_self_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_material_solve_tile_mask", reaction_material_material_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_gas_solve_tile_mask", reaction_material_gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_material_light_solve_tile_mask", reaction_material_light_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_gas_gas_solve_tile_mask", reaction_gas_gas_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_gas_light_solve_tile_mask", reaction_gas_light_solve_tile_mask_upload)
        self._write_dynamic_buffer("reaction_solve_cell_mask", reaction_solve_cell_mask_upload)
        self._write_dynamic_buffer("reaction_solve_gas_mask", reaction_solve_gas_mask_upload)
        self._write_dynamic_buffer("reaction_changed_cell_mask", reaction_changed_cell_mask_upload)
        self._write_dynamic_buffer("reaction_changed_gas_mask", reaction_changed_gas_mask_upload)
        self._write_dynamic_buffer("reaction_ambient_changed_mask", reaction_ambient_changed_mask_upload)
        self._write_dynamic_buffer("reaction_timer_changed_mask", reaction_timer_changed_mask_upload)
        self._write_dynamic_buffer("reaction_emitted_light_mask", reaction_emitted_light_mask_upload)
        self._write_dynamic_buffer("reaction_emitted_material_mask", reaction_emitted_material_mask_upload)
        self._write_dynamic_buffer("collapse_runtime_meta", collapse_runtime_meta_upload)
        self._write_dynamic_buffer("collapse_solve_region_mask", collapse_solve_region_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_structural_mask"):
            self._write_dynamic_buffer("collapse_structural_mask", collapse_structural_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_support_seed_mask"):
            self._write_dynamic_buffer("collapse_support_seed_mask", collapse_support_seed_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_supported_mask"):
            self._write_dynamic_buffer("collapse_supported_mask", collapse_supported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_unsupported_mask"):
            self._write_dynamic_buffer("collapse_unsupported_mask", collapse_unsupported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_delayed_pending_mask"):
            self._write_dynamic_buffer("collapse_delayed_pending_mask", collapse_delayed_pending_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_immune_unsupported_mask"):
            self._write_dynamic_buffer("collapse_immune_unsupported_mask", collapse_immune_unsupported_mask_upload)
        if self._should_upload_cpu_resource(world, "collapse_collapsed_cell_mask"):
            self._write_dynamic_buffer("collapse_collapsed_cell_mask", collapse_collapsed_cell_mask_upload)
        self._write_dynamic_buffer("collapse_component", collapse_component_upload)
        self._write_dynamic_buffer("optics_runtime_meta", optics_runtime_meta_upload)
        self._write_dynamic_buffer("optics_solve_tile_mask", optics_solve_tile_mask_upload)
        self._write_dynamic_buffer("optics_solve_cell_mask", optics_solve_cell_mask_upload)
        self._write_dynamic_buffer("optics_solve_gas_mask", optics_solve_gas_mask_upload)
        self._write_dynamic_buffer("optics_visible_changed_mask", optics_visible_changed_mask_upload)
        self._write_dynamic_buffer("optics_cell_dose_changed_mask", optics_cell_dose_changed_mask_upload)
        self._write_dynamic_buffer("optics_gas_dose_changed_mask", optics_gas_dose_changed_mask_upload)
        self._write_dynamic_buffer("optics_emitter_origin_mask", optics_emitter_origin_mask_upload)
        self._write_dynamic_buffer("page_stripe_meta", page_stripe_meta_upload)
        self._write_dynamic_buffer("page_stripe_section", page_stripe_section_upload)
        self._write_dynamic_buffer("page_stripe_payload", page_stripe_payload_upload)
        self.buffers["frame_meta"].write(frame_meta_upload.tobytes())
        if self._should_upload_cpu_resource(world, "material"):
            self.textures["material"].write(world.material_id.astype("f4").tobytes())
        if upload_light_from_cpu:
            light_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            light_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
            light_rgba[..., 3] = 1.0
            self.textures["light"].write(light_rgba.tobytes())
        if upload_visible_from_cpu:
            visible_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            visible_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
            visible_rgba[..., 3] = 1.0
            self.textures["visible_illumination"].write(visible_rgba.tobytes())
        if upload_debug_texture:
            if debug_frame is None:
                debug_frame = world.debug_frame(world.default_debug_view)
            debug_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
            debug_rgba[..., :3] = np.clip(debug_frame, 0.0, 1.0)
            debug_rgba[..., 3] = 1.0
            self.textures["debug"].write(debug_rgba.tobytes())
        if self._should_upload_cpu_resource(world, "ambient_temperature"):
            self.textures["ambient_temperature"].write(world.ambient_temperature.astype("f4").tobytes())
        if self._should_upload_cpu_resource(world, "pressure_ping"):
            self.textures["pressure_ping"].write(world.pressure_ping.astype("f4").tobytes())
        if self._should_upload_cpu_resource(world, "flow_velocity"):
            self.textures["flow_velocity"].write(world.flow_velocity.astype("f4").tobytes())
        if getattr(world, "simulation_backend", "") == "gpu":
            self.mark_gpu_authoritative(
                "cell_core",
                "material",
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
                "active_meta",
                "active_tile_ttl",
                "active_chunk_mask",
            )

    def sync_display_textures(self, world: "WorldEngine") -> None:
        """Refresh textures sampled by the desktop demo from GPU-authoritative buffers."""
        if not self.enabled or self.ctx is None:
            return
        self.ensure_world_resources(world)
        if getattr(world, "simulation_backend", "") != "gpu":
            return
        if "cell_core" in self.gpu_authoritative_resources and "cell_core" in self.buffers:
            self._ensure_display_programs()
            program = self.display_programs["material_from_cell_core"]
            program["width"] = int(world.width)
            program["height"] = int(world.height)
            self.buffers["cell_core"].bind_to_storage_buffer(0)
            self.textures["material"].bind_to_image(0, read=False, write=True)
            program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
            self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        if "visible_illumination" in self.gpu_authoritative_resources and "visible_illumination" in self.textures:
            self._ensure_display_programs()
            program = self.display_programs["light_from_visible_texture"]
            program["width"] = int(world.width)
            program["height"] = int(world.height)
            self.textures["visible_illumination"].use(0)
            self.textures["light"].bind_to_image(0, read=False, write=True)
            program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
            self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def sync_debug_display_texture(
        self,
        world: "WorldEngine",
        *,
        view: str,
        gas_species_id: int = -1,
        light_dose_channel: int = -1,
    ) -> bool:
        """Refresh the desktop demo debug texture using only GPU-resident state."""
        if not self.enabled or self.ctx is None:
            return False
        if getattr(world, "simulation_backend", "") != "gpu":
            return False
        self.ensure_world_resources(world)
        self._ensure_display_programs()
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
        program = self.display_programs["debug_from_gpu_state"]
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
        self.buffers["cell_core"].bind_to_storage_buffer(0)
        self.buffers["gas_concentration"].bind_to_storage_buffer(1)
        self.buffers["cell_optical_dose"].bind_to_storage_buffer(2)
        self.buffers["gas_optical_dose"].bind_to_storage_buffer(3)
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(4)
        self.textures["visible_illumination"].use(0)
        self.textures["flow_velocity"].use(1)
        self.textures["pressure_ping"].use(2)
        self.textures["debug"].bind_to_image(0, read=False, write=True)
        program.run(group_x=(int(world.width) + 15) // 16, group_y=(int(world.height) + 15) // 16)
        self.ctx.memory_barrier(self.ctx.TEXTURE_FETCH_BARRIER_BIT | self.ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        return True

    def _ensure_display_programs(self) -> None:
        if not self.enabled or self.ctx is None:
            return
        if "material_from_cell_core" not in self.display_programs:
            self.display_programs["material_from_cell_core"] = self.ctx.compute_shader(
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
        if "light_from_visible_texture" not in self.display_programs:
            self.display_programs["light_from_visible_texture"] = self.ctx.compute_shader(
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
            self.display_programs["light_from_visible_texture"]["visible_tex"] = 0
        if "debug_from_gpu_state" not in self.display_programs:
            self.display_programs["debug_from_gpu_state"] = self.ctx.compute_shader(
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
            self.display_programs["debug_from_gpu_state"]["visible_tex"] = 0
            self.display_programs["debug_from_gpu_state"]["flow_velocity_tex"] = 1
            self.display_programs["debug_from_gpu_state"]["pressure_tex"] = 2

    def mark_gpu_authoritative(self, *resource_names: str) -> None:
        self.gpu_authoritative_resources.update(str(name) for name in resource_names)

    def clear_gpu_authoritative(self, *resource_names: str) -> None:
        if resource_names:
            for name in resource_names:
                self.gpu_authoritative_resources.discard(str(name))
            return
        self.gpu_authoritative_resources.clear()

    def _should_upload_cpu_resource(self, world: "WorldEngine", resource_name: str) -> bool:
        if self._force_cpu_resource_upload:
            return True
        return not (
            str(resource_name) in self.gpu_authoritative_resources
            and getattr(world, "simulation_backend", "") == "gpu"
        )

    @staticmethod
    def _should_upload_cpu_solver_runtime(world: "WorldEngine") -> bool:
        return getattr(world, "simulation_backend", "") == "cpu"

    def _shadow_or_default(self, name: str, default: np.ndarray) -> np.ndarray:
        existing = self.shadow_buffers.get(name)
        if isinstance(existing, np.ndarray) and existing.shape == default.shape and existing.dtype == default.dtype:
            return existing
        return default

    def _write_dynamic_buffer(self, name: str, data: np.ndarray) -> None:
        if not self.enabled or self.ctx is None:
            return
        buffer = self.buffers.get(name)
        nbytes = max(4, data.nbytes)
        if buffer is None:
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.buffers[name] = buffer
        elif buffer.size < nbytes:
            buffer.release()
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.buffers[name] = buffer
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    def sync_readback_requests(self, world: "WorldEngine") -> None:
        readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
        self.shadow_buffers["readback_request"] = readback_request_upload.copy()
        self.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
        self._write_dynamic_buffer("readback_request", readback_request_upload)
        self._write_dynamic_buffer("readback_request_label", readback_request_label_upload)

    def sync_force_sources(self, world: "WorldEngine") -> None:
        force_source_upload = pack_force_source_upload(world)
        force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
        self.shadow_buffers["force_source"] = force_source_upload.copy()
        self.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
        self._write_dynamic_buffer("force_source", force_source_upload)
        if self.enabled and self.ctx is not None:
            self.buffers["force_source_count"].write(force_source_count_upload.tobytes())

    def mark_active_rects(
        self,
        world: "WorldEngine",
        rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
    ) -> bool:
        if not rects:
            return True
        if not self.enabled or self.ctx is None:
            return False
        self.ensure_world_resources(world)
        if (
            "active_meta" not in self.buffers
            or "active_tile_ttl" not in self.buffers
            or "active_chunk_mask" not in self.buffers
        ):
            return False
        self._ensure_active_scheduler_programs()
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
        self._write_dynamic_buffer("active_rect", packed_rects)

        mark_program = self.active_scheduler_programs["mark_active_rects"]
        mark_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        mark_program["world_size"].value = (world.width, world.height)
        mark_program["tile_size"].value = int(world.active.tile_size)
        mark_program["active_ttl_reset"].value = int(world.active.active_ttl_reset)
        mark_program["rect_count"].value = int(len(packed_rects))
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        self.buffers["active_rect"].bind_to_storage_buffer(binding=1)
        mark_program.run((tile_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )
        self._refresh_active_chunks_and_meta(world, read_meta=False)
        self.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
        return True

    def decay_active_scheduler(self, world: "WorldEngine") -> bool:
        if not self.enabled or self.ctx is None:
            return False
        self.ensure_world_resources(world)
        if (
            "active_meta" not in self.buffers
            or "active_tile_ttl" not in self.buffers
            or "active_chunk_mask" not in self.buffers
        ):
            return False
        self._ensure_active_scheduler_programs()
        tile_count = int(world.active.tile_width * world.active.tile_height)
        chunk_count = int(world.active.chunk_width * world.active.chunk_height)
        if tile_count <= 0 or chunk_count <= 0:
            return False

        decay_program = self.active_scheduler_programs["decay_active_tiles"]
        decay_program["tile_count"].value = tile_count
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        decay_program.run((tile_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )

        self._refresh_active_chunks_and_meta(world, read_meta=False)
        self.mark_gpu_authoritative("active_meta", "active_tile_ttl", "active_chunk_mask")
        return True

    def _refresh_active_chunks_and_meta(self, world: "WorldEngine", *, read_meta: bool = False) -> None:
        assert self.ctx is not None
        clear_program = self.active_scheduler_programs["clear_active_counts"]
        self.buffers["active_meta"].bind_to_storage_buffer(binding=0)
        self.buffers["active_chunk_count"].bind_to_storage_buffer(binding=1)
        self.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=2)
        clear_program.run(1, 1, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )

        refresh_program = self.active_scheduler_programs["refresh_active_chunks"]
        refresh_program["tile_grid_size"].value = (world.active.tile_width, world.active.tile_height)
        refresh_program["chunk_grid_size"].value = (world.active.chunk_width, world.active.chunk_height)
        refresh_program["chunk_tiles"].value = int(world.active.chunk_tiles)
        self.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
        self.buffers["active_chunk_mask"].bind_to_storage_buffer(binding=1)
        self.buffers["active_meta"].bind_to_storage_buffer(binding=2)
        self.buffers["active_chunk_count"].bind_to_storage_buffer(binding=3)
        self.buffers["active_chunk_list"].bind_to_storage_buffer(binding=4)
        self.buffers["active_chunk_dispatch_args"].bind_to_storage_buffer(binding=5)
        refresh_program.run(world.active.chunk_width, world.active.chunk_height, 1)
        self.ctx.memory_barrier(
            getattr(self.ctx, "SHADER_STORAGE_BARRIER_BIT", 0)
            | getattr(self.ctx, "COMMAND_BARRIER_BIT", 0)
            | getattr(self.ctx, "BUFFER_UPDATE_BARRIER_BIT", 0)
        )
        if read_meta:
            self.shadow_buffers["active_meta"] = np.frombuffer(
                self.buffers["active_meta"].read(size=ACTIVE_META_DTYPE.itemsize),
                dtype=ACTIVE_META_DTYPE,
                count=1,
            ).copy()

    def _ensure_active_scheduler_programs(self) -> None:
        if self.ctx is None:
            return
        required_programs = {
            "mark_active_rects",
            "decay_active_tiles",
            "clear_active_counts",
            "count_active_scheduler",
            "refresh_active_chunks",
        }
        if required_programs.issubset(self.active_scheduler_programs):
            return
        for name in required_programs:
            program = self.active_scheduler_programs.pop(name, None)
            if program is not None:
                try:
                    program.release()
                except Exception:
                    pass
        self.active_scheduler_programs["mark_active_rects"] = self.ctx.compute_shader(
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
        self.active_scheduler_programs["decay_active_tiles"] = self.ctx.compute_shader(
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
        self.active_scheduler_programs["clear_active_counts"] = self.ctx.compute_shader(
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
        self.active_scheduler_programs["count_active_scheduler"] = self.ctx.compute_shader(
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
        self.active_scheduler_programs["refresh_active_chunks"] = self.ctx.compute_shader(
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

    def _write_typed_table_buffer(self, name: str, data: np.ndarray) -> None:
        if not self.enabled or self.ctx is None:
            return
        buffer = self.typed_table_buffers.get(name)
        nbytes = max(4, data.nbytes)
        if buffer is None or buffer.size < nbytes:
            if buffer is not None:
                buffer.release()
            buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.typed_table_buffers[name] = buffer
        else:
            buffer.orphan(nbytes)
        if data.nbytes > 0:
            buffer.write(np.ascontiguousarray(data).tobytes())

    def queue_readback(
        self,
        frame_id: int,
        request: ReadbackRequest,
        payload: dict[str, Any],
        *,
        require_gpu_sources: bool = False,
    ) -> bool:
        slot: GLReadbackSlot | None = None
        slot_count = len(self.readback_slots)
        for offset in range(slot_count):
            candidate = self.readback_slots[(self.write_index + offset) % slot_count]
            if candidate.frame_id < 0 and candidate.request is None:
                slot = candidate
                break
        if slot is None:
            return False
        plan = self._plan_readback_payload(payload)
        gpu_backed = bool(plan.gpu_sources)
        latency_frames = GPU_READBACK_LATENCY_FRAMES if gpu_backed else CPU_READBACK_LATENCY_FRAMES
        if require_gpu_sources and plan.cpu_chunks:
            paths = ", ".join(".".join(path) if path else "<root>" for path in plan.cpu_chunk_paths)
            raise RuntimeError(
                f"GPU readback requires GPU-backed payload arrays, found CPU payload chunks at: {paths}; "
                "CPU fallback is disabled"
            )
        if require_gpu_sources and plan.gpu_sources and (not self.enabled or self.ctx is None):
            raise RuntimeError("GPU readback requires an enabled ModernGL context; CPU fallback is disabled")
        if self.enabled and self.ctx is not None:
            if slot.buffer is None or slot.buffer.size < max(plan.nbytes, 4):
                if slot.buffer is not None:
                    slot.buffer.release()
                slot.buffer = self.ctx.buffer(reserve=max(plan.nbytes, 4), dynamic=True)
            else:
                slot.buffer.orphan(max(plan.nbytes, 4))
            for offset, data in plan.cpu_chunks:
                if data:
                    slot.buffer.write(data, offset=offset)
            for offset, source in plan.gpu_sources:
                self._fill_readback_slot_from_gpu(
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
        self.write_index = (self.write_index + 1) % len(self.readback_slots)
        return True

    def poll_readback(self, current_frame_id: int) -> ReadbackResult | None:
        ready_slots = [
            slot
            for slot in self.readback_slots
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
        elif self.enabled and self.ctx is not None and slot.buffer is not None:
            raw = slot.buffer.read(size=slot.nbytes)
        else:
            raw = slot.buffer if isinstance(slot.buffer, (bytes, bytearray)) else b""
        payload = self._decode_readback_payload(raw, slot.layout)
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

    @staticmethod
    def _serialize_table_summary(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return {
                "kind": "dict",
                "size": len(payload),
                "keys": sorted(str(key) for key in payload.keys()),
            }
        if isinstance(payload, (list, tuple)):
            return {
                "kind": "list",
                "size": len(payload),
            }
        if payload is None:
            return {"kind": "none", "size": 0}
        return {"kind": type(payload).__name__}

    @staticmethod
    def _serialize_ndarray_summary(array: np.ndarray) -> dict[str, Any]:
        return {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        }

    @staticmethod
    def _resource_size_bytes(resource: Any) -> int | None:
        size = getattr(resource, "size", None)
        if size is None or isinstance(size, tuple):
            return None
        try:
            return int(size)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _serialize_buffer_summary(cls, resource: Any) -> dict[str, Any]:
        return {"size_bytes": cls._resource_size_bytes(resource)}

    @staticmethod
    def _serialize_texture_summary(texture: Any) -> dict[str, Any]:
        size = getattr(texture, "size", None)
        if isinstance(size, tuple):
            size_payload: list[int] | None = [int(value) for value in size]
        elif size is None:
            size_payload = None
        else:
            size_payload = [int(size)]
        components = getattr(texture, "components", None)
        return {
            "size": size_payload,
            "components": None if components is None else int(components),
            "dtype": None if getattr(texture, "dtype", None) is None else str(texture.dtype),
        }

    @staticmethod
    def _serialize_readback_layout(layout: ReadbackPayloadLayout | None) -> dict[str, Any] | None:
        if layout is None:
            return None
        return {
            "metadata_keys": sorted(str(key) for key in layout.metadata.keys()),
            "array_count": len(layout.arrays),
            "arrays": [
                {
                    "path": [str(part) for part in array.path],
                    "dtype": np.dtype(array.dtype).name,
                    "shape": [int(value) for value in array.shape],
                    "offset": int(array.offset),
                    "nbytes": int(array.nbytes),
                }
                for array in layout.arrays
            ],
        }

    @classmethod
    def _serialize_readback_slot(cls, slot: GLReadbackSlot) -> dict[str, Any]:
        request = slot.request
        return {
            "slot_index": int(slot.slot_index),
            "occupied": request is not None and slot.frame_id >= 0,
            "frame_id": None if slot.frame_id < 0 else int(slot.frame_id),
            "ready_frame_id": None if slot.ready_frame_id < 0 else int(slot.ready_frame_id),
            "min_poll_frame_id": None if slot.min_poll_frame_id < 0 else int(slot.min_poll_frame_id),
            "latency_frames": int(slot.latency_frames),
            "gpu_backed": bool(slot.gpu_backed),
            "pending_gpu_latency": bool(slot.gpu_backed and slot.min_poll_frame_id > slot.ready_frame_id >= 0),
            "request_id": None if request is None or request.request_id is None else int(request.request_id),
            "observer_id": None if request is None or request.observer_id is None else int(request.observer_id),
            "label": None if request is None or request.label is None else str(request.label),
            "channels": None if request is None else [str(channel) for channel in request.channels],
            "window": None
            if request is None
            else {
                "center_x": None if request.center_x is None else int(request.center_x),
                "center_y": None if request.center_y is None else int(request.center_y),
                "width": int(request.width),
                "height": int(request.height),
            },
            "target_query_id": None if request is None or request.target_query_id is None else str(request.target_query_id),
            "target_dx": None if request is None else int(request.target_dx),
            "target_dy": None if request is None else int(request.target_dy),
            "nbytes": int(slot.nbytes),
            "buffer_size_bytes": cls._resource_size_bytes(slot.buffer),
            "layout": cls._serialize_readback_layout(slot.layout),
        }

    def serialize_runtime_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "has_context": self.ctx is not None,
            "own_context": bool(self.own_context),
            "world_signature": None
            if self.world_signature is None
            else [int(value) for value in self.world_signature],
            "rule_table_signature": None
            if self.rule_table_signature is None
            else [int(value) for value in self.rule_table_signature],
            "atlas_grid": [int(self.atlas_grid[0]), int(self.atlas_grid[1])],
            "atlas_dirty": bool(self.atlas_dirty),
            "write_index": int(self.write_index),
            "table_generations": {
                str(name): int(generation)
                for name, generation in sorted(self.table_generations.items())
            },
            "shadow_tables": {
                str(name): self._serialize_table_summary(payload)
                for name, payload in sorted(self.shadow_tables.items())
            },
            "shadow_typed_tables": {
                str(name): self._serialize_ndarray_summary(payload)
                for name, payload in sorted(self.shadow_typed_tables.items())
            },
            "shadow_buffers": {
                str(name): self._serialize_ndarray_summary(payload)
                for name, payload in sorted(self.shadow_buffers.items())
            },
            "textures": {
                str(name): self._serialize_texture_summary(texture)
                for name, texture in sorted(self.textures.items())
            },
            "buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.buffers.items())
            },
            "table_buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.table_buffers.items())
            },
            "typed_table_buffers": {
                str(name): self._serialize_buffer_summary(buffer)
                for name, buffer in sorted(self.typed_table_buffers.items())
            },
            "readback_programs": sorted(str(name) for name in self.readback_programs.keys()),
            "readback_latency_frames": {
                "cpu_payload": int(CPU_READBACK_LATENCY_FRAMES),
                "gpu_payload": int(GPU_READBACK_LATENCY_FRAMES),
            },
            "readback_slots": [self._serialize_readback_slot(slot) for slot in self.readback_slots],
        }

    def texture(self, name: str) -> Any | None:
        return self.textures.get(name)

    def atlas_texture(self) -> Any | None:
        return self.textures.get("atlas")

    def release_resources(self) -> None:
        for texture in self.textures.values():
            try:
                texture.release()
            except Exception:
                pass
        for buffer in self.buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for buffer in self.table_buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for buffer in self.typed_table_buffers.values():
            try:
                buffer.release()
            except Exception:
                pass
        for slot in self.readback_slots:
            if self.enabled and self.ctx is not None and hasattr(slot.buffer, "release"):
                try:
                    slot.buffer.release()
                except Exception:
                    pass
            slot.buffer = None
            slot.frame_id = -1
            slot.ready_frame_id = -1
            slot.min_poll_frame_id = -1
            slot.latency_frames = CPU_READBACK_LATENCY_FRAMES
            slot.gpu_backed = False
            slot.request = None
            slot.nbytes = 0
            slot.layout = None
        self.textures.clear()
        self.buffers.clear()
        self.table_buffers.clear()
        self.typed_table_buffers.clear()
        self._release_active_scheduler_programs()
        self._release_display_programs()
        self.gpu_authoritative_resources.clear()
        self.rule_table_signature = None

    def release(self) -> None:
        self.release_resources()
        self._release_readback_programs()
        if self.own_context and self.ctx is not None:
            try:
                self.ctx.release()
            except Exception:
                pass
        self.ctx = None
        self.enabled = False
        self.own_context = False
        self.owner_thread_id = None

    def _ensure_atlas_texture(self, world: "WorldEngine") -> None:
        if not self.enabled or self.ctx is None or not self.atlas_dirty:
            return
        material_count = max(world.rulebook.materials_by_id, default=0) + 1
        cols = 8
        rows = max(1, math.ceil(material_count / cols))
        tile = 8
        atlas = np.zeros((rows * tile, cols * tile, 3), dtype="f4")
        for material in world.rulebook.materials_by_id.values():
            tx = material.material_id % cols
            ty = material.material_id // cols
            atlas[ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile] = _render_group_tile(material, tile)
        existing = self.textures.get("atlas")
        if existing is not None:
            try:
                existing.release()
            except Exception:
                pass
        self.textures["atlas"] = self.ctx.texture((cols * tile, rows * tile), 3, atlas.tobytes(), dtype="f4")
        self.textures["atlas"].filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.atlas_grid = (cols, rows)
        self.atlas_dirty = False

    def _plan_readback_payload(self, payload: dict[str, Any]) -> ReadbackPayloadPlan:
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
            return self._normalize_metadata(value)

        metadata = visit((), payload)
        plan.layout.metadata = metadata if isinstance(metadata, dict) else {}
        plan.nbytes = offset
        return plan

    def _fill_readback_slot_from_gpu(
        self,
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
        assert self.ctx is not None
        if isinstance(source, GPUSegmentedCellCoreWindowReadbackSource):
            self._pack_segmented_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUSegmentedBufferReadbackSource):
            self._pack_segmented_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUSegmentedTextureReadbackSource):
            self._pack_segmented_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUCellCoreWindowReadbackSource):
            self._pack_cell_core_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUGasWindowReadbackSource):
            self._pack_gas_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUBufferReadbackSource):
            self._pack_buffer_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        if isinstance(source, GPUTextureReadbackSource):
            self._pack_texture_window_into_buffer(slot_buffer, offset, source, require_gpu_source=require_gpu_source)
            return
        raise TypeError(f"Unsupported GPU readback source: {type(source)!r}")

    def _decode_readback_payload(self, raw: bytes, layout: ReadbackPayloadLayout | None) -> dict[str, Any]:
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

    def _normalize_metadata(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._normalize_metadata(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return [self._normalize_metadata(child) for child in value]
        if isinstance(value, list):
            return [self._normalize_metadata(child) for child in value]
        return value

    def _ensure_readback_programs(self) -> None:
        if self.ctx is None or self.readback_programs:
            return
        local_size = 8
        self.readback_programs["cell_core_window"] = self.ctx.compute_shader(
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
        self.readback_programs["gas_window"] = self.ctx.compute_shader(
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
        self.readback_programs["buffer_window"] = self.ctx.compute_shader(
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
        self.readback_programs["texture_window"] = self.ctx.compute_shader(
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
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUCellCoreWindowReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        height, width = source.shape[:2]
        if width <= 0 or height <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("cell_core_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing cell core readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
        self.ctx.memory_barrier()

    def _pack_gas_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUGasWindowReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        height, width = source.shape
        if width <= 0 or height <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("gas_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing gas readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
        self.ctx.memory_barrier()

    def _pack_buffer_window_into_buffer(
        self,
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
            self._raise_gpu_readback_unavailable(source, "unsupported buffer copy alignment or element size")
            return
        if source.chunk_size <= 0 or source.count <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("buffer_window")
        if program is None:
            self._raise_gpu_readback_unavailable(source, "missing buffer readback shader")
            return
        src_buffer = self.buffers.get(source.resource_name)
        if src_buffer is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU buffer")
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
        self.ctx.memory_barrier()

    def _pack_texture_window_into_buffer(
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUTextureReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if offset % 4 != 0:
            self._raise_gpu_readback_unavailable(source, "unaligned destination offset")
            return
        origin_x, origin_y, width, height = source.viewport
        if width <= 0 or height <= 0 or source.components <= 0:
            return
        self._ensure_readback_programs()
        program = self.readback_programs.get("texture_window")
        if program is None or source.components > 4:
            self._raise_gpu_readback_unavailable(source, "missing texture readback shader or unsupported component count")
            return
        texture = self.textures.get(source.resource_name)
        if texture is None:
            self._raise_gpu_readback_unavailable(source, "missing GPU texture")
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
        self.ctx.memory_barrier()

    def _pack_segmented_cell_core_window_into_buffer(
        self,
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
            self._pack_cell_core_window_into_buffer(
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
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUSegmentedBufferReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        dtype = np.dtype(source.dtype)
        if dtype.itemsize != 4:
            self._raise_gpu_readback_unavailable(source, "unsupported segmented buffer element size")
            return
        if len(source.shape) < 2:
            self._raise_gpu_readback_unavailable(source, "segmented buffer source requires a 2D destination")
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
            self._pack_buffer_window_into_buffer(
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
        self,
        slot_buffer: Any,
        offset: int,
        source: GPUSegmentedTextureReadbackSource,
        *,
        require_gpu_source: bool = False,
    ) -> None:
        if source.components <= 0:
            return
        if len(source.shape) < 2:
            self._raise_gpu_readback_unavailable(source, "segmented texture source requires a 2D destination")
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
            self._pack_texture_window_into_buffer(
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

    @staticmethod
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

    def _release_readback_programs(self) -> None:
        for program in self.readback_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.readback_programs.clear()

    def _release_display_programs(self) -> None:
        for program in self.display_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.display_programs.clear()

    def _release_active_scheduler_programs(self) -> None:
        for program in self.active_scheduler_programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.active_scheduler_programs.clear()
