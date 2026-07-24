# ruff: noqa: F401
# Facade module: oracle_game/gpu/__init__.py re-exports this module via
# `from oracle_game.gpu.bridge import *`, so every public import here is part of
# the oracle_game.gpu namespace (and a monkeypatch target in tests).
from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oracle_game.gpu._common import (
    CPU_READBACK_LATENCY_FRAMES,
    GPU_READBACK_LATENCY_FRAMES,
    MAX_REACTION_LIGHT_EMITTERS,
    _get_shared_standalone_context,
    _json_bytes,
    _render_group_tile,
    moderngl,
)
from oracle_game.gpu.bridge_display import (
    _ensure_active_scheduler_programs,
    _ensure_display_programs,
    _refresh_active_chunks_and_meta,
    _release_active_scheduler_programs,
    _release_display_programs,
    decay_active_scheduler,
    mark_active_rects,
    sync_debug_display_texture,
    sync_display_textures,
)
from oracle_game.gpu.bridge_readback import (
    _decode_readback_payload,
    _ensure_readback_programs,
    _fill_readback_slot_from_gpu,
    _normalize_metadata,
    _pack_buffer_window_into_buffer,
    _pack_cell_core_window_into_buffer,
    _pack_gas_window_into_buffer,
    _pack_segmented_buffer_window_into_buffer,
    _pack_segmented_cell_core_window_into_buffer,
    _pack_segmented_texture_window_into_buffer,
    _pack_texture_window_into_buffer,
    _plan_readback_payload,
    _raise_gpu_readback_unavailable,
    _release_readback_programs,
    _stash_inflight_readback_slots,
    poll_readback,
    queue_readback,
    requeue_detached_readbacks,
)
from oracle_game.gpu.bridge_resources import (
    _ensure_atlas_texture,
    atlas_texture,
    ensure_cell_core_spare,
    ensure_world_resources,
    release_resources,
    texture,
)
from oracle_game.gpu.bridge_state import (
    _resource_size_bytes,
    _serialize_buffer_summary,
    _serialize_ndarray_summary,
    _serialize_readback_layout,
    _serialize_readback_slot,
    _serialize_table_summary,
    _serialize_texture_summary,
    _should_upload_cpu_resource,
    _should_upload_cpu_solver_runtime,
    clear_gpu_authoritative,
    mark_gpu_authoritative,
    serialize_runtime_state,
)
from oracle_game.gpu.bridge_sync import (
    _shadow_or_default,
    _sync_world_impl,
    _write_dynamic_buffer,
    _write_typed_table_buffer,
    download_gpu_authoritative_resources,
    sync_force_sources,
    sync_readback_requests,
    sync_rule_tables,
    sync_world,
    sync_world_commands,
    upload_table,
)
from oracle_game.gpu.dtypes import (
    ACTIVE_META_DTYPE,
    ACTIVE_RECT_DTYPE,
    COLLAPSE_COMPONENT_DTYPE,
    COLLAPSE_RUNTIME_META_DTYPE,
    ENTITY_STATE_DTYPE,
    FORCE_SOURCE_DTYPE,
    FRAME_META_DTYPE,
    GAS_RUNTIME_META_DTYPE,
    GAS_SPECIES_RUNTIME_DTYPE,
    GAS_TABLE_DTYPE,
    HEAT_RUNTIME_META_DTYPE,
    ISLAND_RUNTIME_DTYPE,
    LIGHT_TABLE_DTYPE,
    LIQUID_RUNTIME_META_DTYPE,
    MATERIAL_TABLE_DTYPE,
    OPTICS_RUNTIME_META_DTYPE,
    OPTICS_TABLE_DTYPE,
    PAGE_STRIPE_META_DTYPE,
    PAGE_STRIPE_SECTION_DTYPE,
    PAIR_REACTION_RULE_TABLE_DTYPE,
    PLACEHOLDER_DIRTY_RECT_DTYPE,
    PLACEHOLDER_DTYPE,
    REACTION_ACTION_TABLE_DTYPE,
    REACTION_RUNTIME_META_DTYPE,
    READBACK_REQUEST_DTYPE,
    RULE_TABLE_META_DTYPE,
    SELF_REACTION_RULE_TABLE_DTYPE,
    WORLD_COMMAND_DTYPE,
)
from oracle_game.gpu.packers import (
    _pack_pair_reaction_rules,
    pack_active_meta_upload,
    pack_cell_core,
    pack_collapse_runtime_upload,
    pack_entity_state_upload,
    pack_force_source_upload,
    pack_frame_meta_upload,
    pack_gas_runtime_upload,
    pack_gas_table,
    pack_heat_runtime_upload,
    pack_island_runtime_upload,
    pack_light_table,
    pack_liquid_runtime_upload,
    pack_material_table,
    pack_optics_runtime_upload,
    pack_optics_table,
    pack_page_stripe_upload,
    pack_placeholder_dirty_rect_upload,
    pack_placeholder_upload,
    pack_reaction_action_table,
    pack_reaction_runtime_upload,
    pack_readback_request_upload,
    pack_self_reaction_rule_table,
    pack_world_command_upload,
)
from oracle_game.gpu.readback import (
    GLReadbackSlot,
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUReadbackSegment,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
    ReadbackArrayLayout,
    ReadbackPayloadLayout,
    ReadbackPayloadPlan,
)
from oracle_game.types import ReadbackRequest, ReadbackResult


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
    readback_slots: list[GLReadbackSlot] = field(
        default_factory=lambda: [GLReadbackSlot(0), GLReadbackSlot(1)]
    )
    # In-flight readback slot descriptors preserved across readback ring
    # rebuilds (context attach, resource release).  Their GL storage is gone,
    # but the requests must be re-queued on the new context instead of being
    # silently dropped (which also leaks ``engine.inflight_readbacks``).
    detached_readback_slots: list[GLReadbackSlot] = field(default_factory=list)
    # Grow-only staging buffer for DMA-based readback window copies (see
    # ``_fill_slot_from_buffer_rows``); released with the rest of the ring.
    readback_staging: Any | None = None
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
        self._stash_inflight_readback_slots()
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

    # ------------------------------------------------------------------
    # Satellite method delegates (W5b: retired the `_x = _x` class grafts).
    #
    # Each body resolves the bare function name through this module's global
    # namespace -- method bodies never see class scope -- i.e. the satellite
    # function imported at the top of this file, bound at import time exactly
    # like the historical grafts.  Monkeypatch semantics are unchanged:
    # patching the attribute on the class or on an instance shadows/replaces
    # the delegate, while patching the satellite module's attribute does NOT
    # affect calls through the bridge.
    # ------------------------------------------------------------------
    def ensure_world_resources(self, *args: Any, **kwargs: Any) -> Any:
        return ensure_world_resources(self, *args, **kwargs)

    def ensure_cell_core_spare(self, *args: Any, **kwargs: Any) -> Any:
        return ensure_cell_core_spare(self, *args, **kwargs)

    def _ensure_atlas_texture(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_atlas_texture(self, *args, **kwargs)

    def release_resources(self, *args: Any, **kwargs: Any) -> Any:
        return release_resources(self, *args, **kwargs)

    def texture(self, *args: Any, **kwargs: Any) -> Any:
        return texture(self, *args, **kwargs)

    def atlas_texture(self, *args: Any, **kwargs: Any) -> Any:
        return atlas_texture(self, *args, **kwargs)

    def upload_table(self, *args: Any, **kwargs: Any) -> Any:
        return upload_table(self, *args, **kwargs)

    def sync_rule_tables(self, *args: Any, **kwargs: Any) -> Any:
        return sync_rule_tables(self, *args, **kwargs)

    def sync_world(self, *args: Any, **kwargs: Any) -> Any:
        return sync_world(self, *args, **kwargs)

    def download_gpu_authoritative_resources(self, *args: Any, **kwargs: Any) -> Any:
        return download_gpu_authoritative_resources(self, *args, **kwargs)

    def _sync_world_impl(self, *args: Any, **kwargs: Any) -> Any:
        return _sync_world_impl(self, *args, **kwargs)

    def sync_readback_requests(self, *args: Any, **kwargs: Any) -> Any:
        return sync_readback_requests(self, *args, **kwargs)

    def sync_force_sources(self, *args: Any, **kwargs: Any) -> Any:
        return sync_force_sources(self, *args, **kwargs)

    def sync_world_commands(self, *args: Any, **kwargs: Any) -> Any:
        return sync_world_commands(self, *args, **kwargs)

    def _write_typed_table_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _write_typed_table_buffer(self, *args, **kwargs)

    def _write_dynamic_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _write_dynamic_buffer(self, *args, **kwargs)

    def _shadow_or_default(self, *args: Any, **kwargs: Any) -> Any:
        return _shadow_or_default(self, *args, **kwargs)

    def queue_readback(self, *args: Any, **kwargs: Any) -> Any:
        return queue_readback(self, *args, **kwargs)

    def poll_readback(self, *args: Any, **kwargs: Any) -> Any:
        return poll_readback(self, *args, **kwargs)

    def requeue_detached_readbacks(self, *args: Any, **kwargs: Any) -> Any:
        return requeue_detached_readbacks(self, *args, **kwargs)

    def _stash_inflight_readback_slots(self, *args: Any, **kwargs: Any) -> Any:
        return _stash_inflight_readback_slots(self, *args, **kwargs)

    def _plan_readback_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _plan_readback_payload(self, *args, **kwargs)

    def _fill_readback_slot_from_gpu(self, *args: Any, **kwargs: Any) -> Any:
        return _fill_readback_slot_from_gpu(self, *args, **kwargs)

    def _decode_readback_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _decode_readback_payload(self, *args, **kwargs)

    def _normalize_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return _normalize_metadata(self, *args, **kwargs)

    def _ensure_readback_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_readback_programs(self, *args, **kwargs)

    def _pack_cell_core_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_cell_core_window_into_buffer(self, *args, **kwargs)

    def _pack_gas_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_gas_window_into_buffer(self, *args, **kwargs)

    def _pack_buffer_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_buffer_window_into_buffer(self, *args, **kwargs)

    def _pack_texture_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_texture_window_into_buffer(self, *args, **kwargs)

    def _pack_segmented_cell_core_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_segmented_cell_core_window_into_buffer(self, *args, **kwargs)

    def _pack_segmented_buffer_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_segmented_buffer_window_into_buffer(self, *args, **kwargs)

    def _pack_segmented_texture_window_into_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return _pack_segmented_texture_window_into_buffer(self, *args, **kwargs)

    @staticmethod
    def _raise_gpu_readback_unavailable(*args: Any, **kwargs: Any) -> Any:
        return _raise_gpu_readback_unavailable(*args, **kwargs)

    def _release_readback_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _release_readback_programs(self, *args, **kwargs)

    def sync_display_textures(self, *args: Any, **kwargs: Any) -> Any:
        return sync_display_textures(self, *args, **kwargs)

    def sync_debug_display_texture(self, *args: Any, **kwargs: Any) -> Any:
        return sync_debug_display_texture(self, *args, **kwargs)

    def _ensure_display_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_display_programs(self, *args, **kwargs)

    def mark_active_rects(self, *args: Any, **kwargs: Any) -> Any:
        return mark_active_rects(self, *args, **kwargs)

    def decay_active_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        return decay_active_scheduler(self, *args, **kwargs)

    def _refresh_active_chunks_and_meta(self, *args: Any, **kwargs: Any) -> Any:
        return _refresh_active_chunks_and_meta(self, *args, **kwargs)

    def _ensure_active_scheduler_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _ensure_active_scheduler_programs(self, *args, **kwargs)

    def _release_display_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _release_display_programs(self, *args, **kwargs)

    def _release_active_scheduler_programs(self, *args: Any, **kwargs: Any) -> Any:
        return _release_active_scheduler_programs(self, *args, **kwargs)

    def mark_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return mark_gpu_authoritative(self, *args, **kwargs)

    def clear_gpu_authoritative(self, *args: Any, **kwargs: Any) -> Any:
        return clear_gpu_authoritative(self, *args, **kwargs)

    def _should_upload_cpu_resource(self, *args: Any, **kwargs: Any) -> Any:
        return _should_upload_cpu_resource(self, *args, **kwargs)

    @staticmethod
    def _should_upload_cpu_solver_runtime(*args: Any, **kwargs: Any) -> Any:
        return _should_upload_cpu_solver_runtime(*args, **kwargs)

    @staticmethod
    def _serialize_table_summary(*args: Any, **kwargs: Any) -> Any:
        return _serialize_table_summary(*args, **kwargs)

    @staticmethod
    def _serialize_ndarray_summary(*args: Any, **kwargs: Any) -> Any:
        return _serialize_ndarray_summary(*args, **kwargs)

    @staticmethod
    def _resource_size_bytes(*args: Any, **kwargs: Any) -> Any:
        return _resource_size_bytes(*args, **kwargs)

    @classmethod
    def _serialize_buffer_summary(cls, *args: Any, **kwargs: Any) -> Any:
        return _serialize_buffer_summary(cls, *args, **kwargs)

    @staticmethod
    def _serialize_texture_summary(*args: Any, **kwargs: Any) -> Any:
        return _serialize_texture_summary(*args, **kwargs)

    @staticmethod
    def _serialize_readback_layout(*args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_layout(*args, **kwargs)

    @classmethod
    def _serialize_readback_slot(cls, *args: Any, **kwargs: Any) -> Any:
        return _serialize_readback_slot(cls, *args, **kwargs)

    def serialize_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return serialize_runtime_state(self, *args, **kwargs)
