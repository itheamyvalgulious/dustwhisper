from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import math
import numpy as np

from oracle_game.gpu._common import (
    CPU_READBACK_LATENCY_FRAMES,
    MAX_REACTION_LIGHT_EMITTERS,
    _render_group_tile,
)

from oracle_game.gpu.dtypes import (
    GAS_RUNTIME_META_DTYPE,
    PAGE_STRIPE_SECTION_DTYPE,
    REACTION_RUNTIME_META_DTYPE,
    HEAT_RUNTIME_META_DTYPE,
    WORLD_COMMAND_DTYPE,
    GAS_SPECIES_RUNTIME_DTYPE,
    PLACEHOLDER_DIRTY_RECT_DTYPE,
    LIQUID_RUNTIME_META_DTYPE,
    ACTIVE_META_DTYPE,
    ENTITY_STATE_DTYPE,
    COLLAPSE_COMPONENT_DTYPE,
    PLACEHOLDER_DTYPE,
    ISLAND_RUNTIME_DTYPE,
    FORCE_SOURCE_DTYPE,
    OPTICS_RUNTIME_META_DTYPE,
    PAGE_STRIPE_META_DTYPE,
    FRAME_META_DTYPE,
    COLLAPSE_RUNTIME_META_DTYPE,
    READBACK_REQUEST_DTYPE,
)


def ensure_world_resources(bridge, world: "WorldEngine") -> None:
    if not bridge.enabled or bridge.ctx is None:
        return
    signature = (
        world.width,
        world.height,
        world.gas_width,
        world.gas_height,
        world.gas_concentration.shape[0],
        world.cell_optical_dose.shape[0],
    )
    if signature == bridge.world_signature:
        return
    bridge.release_resources()
    bridge.gpu_authoritative_resources.clear()
    bridge.world_signature = signature
    bridge.textures["material"] = bridge.ctx.texture((world.width, world.height), 1, dtype="f4")
    bridge.textures["light"] = bridge.ctx.texture((world.width, world.height), 4, dtype="f4")
    bridge.textures["debug"] = bridge.ctx.texture((world.width, world.height), 4, dtype="f4")
    bridge.textures["ambient_temperature"] = bridge.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
    bridge.textures["pressure_ping"] = bridge.ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
    bridge.textures["flow_velocity"] = bridge.ctx.texture((world.gas_width, world.gas_height), 2, dtype="f4")
    bridge.textures["visible_illumination"] = bridge.ctx.texture((world.width, world.height), 4, dtype="f4")
    bridge.textures["liquid_flow_intent"] = bridge.ctx.texture((world.width, world.height), 2, dtype="f4")
    for texture in bridge.textures.values():
        texture.filter = (bridge.ctx.NEAREST, bridge.ctx.NEAREST)
    bridge.buffers["cell_core"] = bridge.ctx.buffer(reserve=world.width * world.height * 5 * 4, dynamic=True)
    bridge.buffers["island_id"] = bridge.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
    bridge.buffers["entity_id"] = bridge.ctx.buffer(reserve=max(4, world.width * world.height * 4), dynamic=True)
    bridge.buffers["placeholder_displaced_material"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_delay_pending"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["gas_concentration"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height * 4),
        dynamic=True,
    )
    bridge.buffers["cell_optical_dose"] = bridge.ctx.buffer(
        reserve=max(4, int(np.prod(world.cell_optical_dose.shape, dtype=np.int64)) * 4),
        dynamic=True,
    )
    bridge.buffers["gas_optical_dose"] = bridge.ctx.buffer(
        reserve=max(4, int(np.prod(world.gas_optical_dose.shape, dtype=np.int64)) * 4),
        dynamic=True,
    )
    bridge.buffers["entity_state"] = bridge.ctx.buffer(reserve=max(4, ENTITY_STATE_DTYPE.itemsize), dynamic=True)
    bridge.buffers["entity_state_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["force_source"] = bridge.ctx.buffer(reserve=max(4, FORCE_SOURCE_DTYPE.itemsize), dynamic=True)
    bridge.buffers["force_source_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["island_runtime"] = bridge.ctx.buffer(reserve=max(4, ISLAND_RUNTIME_DTYPE.itemsize), dynamic=True)
    bridge.buffers["island_runtime_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["powder_reservation"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["powder_reservation_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["island_reservation"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["island_reservation_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["world_command"] = bridge.ctx.buffer(reserve=max(4, WORLD_COMMAND_DTYPE.itemsize), dynamic=True)
    bridge.buffers["world_command_payload"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["readback_request"] = bridge.ctx.buffer(reserve=max(4, READBACK_REQUEST_DTYPE.itemsize), dynamic=True)
    bridge.buffers["readback_request_label"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["placeholder"] = bridge.ctx.buffer(reserve=max(4, PLACEHOLDER_DTYPE.itemsize), dynamic=True)
    bridge.buffers["placeholder_dirty_rect"] = bridge.ctx.buffer(
        reserve=max(4, PLACEHOLDER_DIRTY_RECT_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["active_meta"] = bridge.ctx.buffer(reserve=max(4, ACTIVE_META_DTYPE.itemsize), dynamic=True)
    bridge.buffers["active_tile_ttl"] = bridge.ctx.buffer(reserve=max(4, world.active.tile_width * world.active.tile_height * 4), dynamic=True)
    bridge.buffers["active_chunk_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.chunk_width * world.active.chunk_height * 4),
        dynamic=True,
    )
    active_chunk_count = max(1, int(world.active.chunk_width * world.active.chunk_height))
    bridge.buffers["active_chunk_list"] = bridge.ctx.buffer(reserve=max(8, active_chunk_count * 2 * 4), dynamic=True)
    bridge.buffers["active_chunk_count"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["active_chunk_dispatch_args"] = bridge.ctx.buffer(reserve=3 * 4, dynamic=True)
    bridge.buffers["gas_runtime_meta"] = bridge.ctx.buffer(reserve=max(4, GAS_RUNTIME_META_DTYPE.itemsize), dynamic=True)
    bridge.buffers["gas_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["gas_solve_gas_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["gas_species_runtime"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_concentration.shape[0] * GAS_SPECIES_RUNTIME_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["heat_runtime_meta"] = bridge.ctx.buffer(reserve=max(4, HEAT_RUNTIME_META_DTYPE.itemsize), dynamic=True)
    bridge.buffers["heat_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["heat_solve_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["heat_solve_gas_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["heat_phase_target"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
        dynamic=True,
    )
    bridge.buffers["heat_boil_target"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * np.dtype(np.int32).itemsize),
        dynamic=True,
    )
    bridge.buffers["heat_condense_target"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_concentration.shape[0] * world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["liquid_runtime_meta"] = bridge.ctx.buffer(reserve=max(4, LIQUID_RUNTIME_META_DTYPE.itemsize), dynamic=True)
    bridge.buffers["liquid_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["liquid_post_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["liquid_post_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["liquid_vertical_seam_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["liquid_horizontal_seam_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["liquid_buoyancy_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["liquid_changed_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_runtime_meta"] = bridge.ctx.buffer(
        reserve=max(4, REACTION_RUNTIME_META_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["reaction_timed_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_self_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_material_material_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_material_gas_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_material_light_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_gas_gas_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_gas_light_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["reaction_solve_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_solve_gas_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["reaction_changed_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_changed_gas_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["reaction_ambient_changed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["reaction_timer_changed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_emitted_light_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_emitted_material_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["reaction_light_emitter"] = bridge.ctx.buffer(
        reserve=MAX_REACTION_LIGHT_EMITTERS * 2 * 4 * 4,
        dynamic=True,
    )
    bridge.buffers["reaction_light_emitter_count"] = bridge.ctx.buffer(
        reserve=16 * 4,
        dynamic=True,
    )
    bridge.buffers["collapse_runtime_meta"] = bridge.ctx.buffer(
        reserve=max(4, COLLAPSE_RUNTIME_META_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["collapse_solve_region_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_structural_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_support_seed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_supported_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_unsupported_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_delayed_pending_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_immune_unsupported_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_collapsed_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_component_label"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height * 4),
        dynamic=True,
    )
    bridge.buffers["collapse_component"] = bridge.ctx.buffer(
        reserve=max(4, COLLAPSE_COMPONENT_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["optics_runtime_meta"] = bridge.ctx.buffer(
        reserve=max(4, OPTICS_RUNTIME_META_DTYPE.itemsize),
        dynamic=True,
    )
    bridge.buffers["optics_solve_tile_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.active.tile_width * world.active.tile_height),
        dynamic=True,
    )
    bridge.buffers["optics_solve_cell_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["optics_solve_gas_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["optics_visible_changed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["optics_cell_dose_changed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["optics_gas_dose_changed_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.gas_width * world.gas_height),
        dynamic=True,
    )
    bridge.buffers["optics_emitter_origin_mask"] = bridge.ctx.buffer(
        reserve=max(4, world.width * world.height),
        dynamic=True,
    )
    bridge.buffers["page_stripe_meta"] = bridge.ctx.buffer(reserve=max(4, PAGE_STRIPE_META_DTYPE.itemsize), dynamic=True)
    bridge.buffers["page_stripe_section"] = bridge.ctx.buffer(reserve=max(4, PAGE_STRIPE_SECTION_DTYPE.itemsize), dynamic=True)
    bridge.buffers["page_stripe_payload"] = bridge.ctx.buffer(reserve=4, dynamic=True)
    bridge.buffers["frame_meta"] = bridge.ctx.buffer(reserve=max(4, FRAME_META_DTYPE.itemsize), dynamic=True)
    bridge.atlas_dirty = True
    bridge._ensure_atlas_texture(world)


def _ensure_atlas_texture(bridge, world: "WorldEngine") -> None:
    if not bridge.enabled or bridge.ctx is None or not bridge.atlas_dirty:
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
    existing = bridge.textures.get("atlas")
    if existing is not None:
        try:
            existing.release()
        except Exception:
            pass
    bridge.textures["atlas"] = bridge.ctx.texture((cols * tile, rows * tile), 3, atlas.tobytes(), dtype="f4")
    bridge.textures["atlas"].filter = (bridge.ctx.NEAREST, bridge.ctx.NEAREST)
    bridge.atlas_grid = (cols, rows)
    bridge.atlas_dirty = False


def release_resources(bridge) -> None:
    for texture in bridge.textures.values():
        try:
            texture.release()
        except Exception:
            pass
    for buffer in bridge.buffers.values():
        try:
            buffer.release()
        except Exception:
            pass
    for buffer in bridge.table_buffers.values():
        try:
            buffer.release()
        except Exception:
            pass
    for buffer in bridge.typed_table_buffers.values():
        try:
            buffer.release()
        except Exception:
            pass
    for slot in bridge.readback_slots:
        if bridge.enabled and bridge.ctx is not None and hasattr(slot.buffer, "release"):
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
    bridge.textures.clear()
    bridge.buffers.clear()
    bridge.table_buffers.clear()
    bridge.typed_table_buffers.clear()
    bridge._release_active_scheduler_programs()
    bridge._release_display_programs()
    bridge.gpu_authoritative_resources.clear()
    bridge.rule_table_signature = None


def texture(bridge, name: str) -> Any | None:
    return bridge.textures.get(name)


def atlas_texture(bridge) -> Any | None:
    return bridge.textures.get("atlas")
