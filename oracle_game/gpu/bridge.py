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



from oracle_game.gpu.bridge_resources import (
    ensure_world_resources,
    ensure_cell_core_spare,
    _ensure_atlas_texture,
    release_resources,
    texture,
    atlas_texture,
)

from oracle_game.gpu.bridge_sync import (
    upload_table,
    sync_rule_tables,
    sync_world,
    _sync_world_impl,
    sync_readback_requests,
    sync_force_sources,
    _write_typed_table_buffer,
    _write_dynamic_buffer,
    _shadow_or_default,
)

from oracle_game.gpu.bridge_readback import (
    queue_readback,
    poll_readback,
    _plan_readback_payload,
    _fill_readback_slot_from_gpu,
    _decode_readback_payload,
    _normalize_metadata,
    _ensure_readback_programs,
    _pack_cell_core_window_into_buffer,
    _pack_gas_window_into_buffer,
    _pack_buffer_window_into_buffer,
    _pack_texture_window_into_buffer,
    _pack_segmented_cell_core_window_into_buffer,
    _pack_segmented_buffer_window_into_buffer,
    _pack_segmented_texture_window_into_buffer,
    _raise_gpu_readback_unavailable,
    _release_readback_programs,
)

from oracle_game.gpu.bridge_display import (
    sync_display_textures,
    sync_debug_display_texture,
    _ensure_display_programs,
    mark_active_rects,
    decay_active_scheduler,
    _refresh_active_chunks_and_meta,
    _ensure_active_scheduler_programs,
    _release_display_programs,
    _release_active_scheduler_programs,
)

from oracle_game.gpu.bridge_state import (
    mark_gpu_authoritative,
    clear_gpu_authoritative,
    _should_upload_cpu_resource,
    _should_upload_cpu_solver_runtime,
    _serialize_table_summary,
    _serialize_ndarray_summary,
    _resource_size_bytes,
    _serialize_buffer_summary,
    _serialize_texture_summary,
    _serialize_readback_layout,
    _serialize_readback_slot,
    serialize_runtime_state,
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
    # Private ping-pong storage for liquid provenance publication.  Keep this
    # out of ``buffers`` so authoritative names and readback resource lookup
    # continue to refer only to the live ``cell_core`` buffer.
    cell_core_spare: Any | None = None

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
            if self.cell_core_spare is not None:
                try:
                    self.cell_core_spare.release()
                except Exception:
                    pass
                self.cell_core_spare = None
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

    ensure_world_resources = ensure_world_resources
    ensure_cell_core_spare = ensure_cell_core_spare
    _ensure_atlas_texture = _ensure_atlas_texture
    release_resources = release_resources
    texture = texture
    atlas_texture = atlas_texture
    upload_table = upload_table
    sync_rule_tables = sync_rule_tables
    sync_world = sync_world
    _sync_world_impl = _sync_world_impl
    sync_readback_requests = sync_readback_requests
    sync_force_sources = sync_force_sources
    _write_typed_table_buffer = _write_typed_table_buffer
    _write_dynamic_buffer = _write_dynamic_buffer
    _shadow_or_default = _shadow_or_default
    queue_readback = queue_readback
    poll_readback = poll_readback
    _plan_readback_payload = _plan_readback_payload
    _fill_readback_slot_from_gpu = _fill_readback_slot_from_gpu
    _decode_readback_payload = _decode_readback_payload
    _normalize_metadata = _normalize_metadata
    _ensure_readback_programs = _ensure_readback_programs
    _pack_cell_core_window_into_buffer = _pack_cell_core_window_into_buffer
    _pack_gas_window_into_buffer = _pack_gas_window_into_buffer
    _pack_buffer_window_into_buffer = _pack_buffer_window_into_buffer
    _pack_texture_window_into_buffer = _pack_texture_window_into_buffer
    _pack_segmented_cell_core_window_into_buffer = _pack_segmented_cell_core_window_into_buffer
    _pack_segmented_buffer_window_into_buffer = _pack_segmented_buffer_window_into_buffer
    _pack_segmented_texture_window_into_buffer = _pack_segmented_texture_window_into_buffer
    _raise_gpu_readback_unavailable = staticmethod(_raise_gpu_readback_unavailable)
    _release_readback_programs = _release_readback_programs
    sync_display_textures = sync_display_textures
    sync_debug_display_texture = sync_debug_display_texture
    _ensure_display_programs = _ensure_display_programs
    mark_active_rects = mark_active_rects
    decay_active_scheduler = decay_active_scheduler
    _refresh_active_chunks_and_meta = _refresh_active_chunks_and_meta
    _ensure_active_scheduler_programs = _ensure_active_scheduler_programs
    _release_display_programs = _release_display_programs
    _release_active_scheduler_programs = _release_active_scheduler_programs
    mark_gpu_authoritative = mark_gpu_authoritative
    clear_gpu_authoritative = clear_gpu_authoritative
    _should_upload_cpu_resource = _should_upload_cpu_resource
    _should_upload_cpu_solver_runtime = staticmethod(_should_upload_cpu_solver_runtime)
    _serialize_table_summary = staticmethod(_serialize_table_summary)
    _serialize_ndarray_summary = staticmethod(_serialize_ndarray_summary)
    _resource_size_bytes = staticmethod(_resource_size_bytes)
    _serialize_buffer_summary = classmethod(_serialize_buffer_summary)
    _serialize_texture_summary = staticmethod(_serialize_texture_summary)
    _serialize_readback_layout = staticmethod(_serialize_readback_layout)
    _serialize_readback_slot = classmethod(_serialize_readback_slot)
    serialize_runtime_state = serialize_runtime_state
