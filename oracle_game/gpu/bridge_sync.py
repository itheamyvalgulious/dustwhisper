from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import json
import threading

import numpy as np

from oracle_game.gpu._common import (
    _json_bytes,
)
from oracle_game.gpu.dtypes import (
    COLLAPSE_COMPONENT_DTYPE,
    COLLAPSE_RUNTIME_META_DTYPE,
    GAS_RUNTIME_META_DTYPE,
    GAS_SPECIES_RUNTIME_DTYPE,
    HEAT_RUNTIME_META_DTYPE,
    ISLAND_RUNTIME_DTYPE,
    LIQUID_RUNTIME_META_DTYPE,
    OPTICS_RUNTIME_META_DTYPE,
    REACTION_RUNTIME_META_DTYPE,
    RULE_TABLE_META_DTYPE,
)
from oracle_game.gpu.packers import (
    _pack_pair_reaction_rules,
    pack_active_meta_upload,
    pack_cell_core,
    pack_cell_core_window,
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
    unpack_cell_core,
)


def upload_table(bridge, name: str, payload: Any) -> None:
    data = _json_bytes(payload)
    bridge.shadow_tables[name] = json.loads(data.decode("utf-8"))
    bridge.table_generations[name] = bridge.table_generations.get(name, 0) + 1
    if not bridge.enabled or bridge.ctx is None:
        return
    buffer = bridge.table_buffers.get(name)
    if buffer is None or buffer.size < len(data):
        if buffer is not None:
            buffer.release()
        bridge.table_buffers[name] = bridge.ctx.buffer(data, dynamic=True)
    else:
        buffer.orphan(len(data))
        buffer.write(data)
    if name == "materials":
        bridge.atlas_dirty = True


def sync_rule_tables(bridge, world: "WorldEngine") -> None:
    signature = (
        bridge.table_generations.get("materials", 0),
        bridge.table_generations.get("gases", 0),
        bridge.table_generations.get("lights", 0),
        bridge.table_generations.get("optics", 0),
        bridge.table_generations.get("reactions", 0),
    )
    buffers_ready = all(
        name in bridge.typed_table_buffers
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
    if (
        signature == bridge.rule_table_signature
        and bridge.shadow_typed_tables
        and ((not bridge.enabled or bridge.ctx is None) or buffers_ready)
    ):
        return

    material_table = pack_material_table(world)
    gas_table = pack_gas_table(world)
    light_table = pack_light_table(world)
    optics_table = pack_optics_table(world)
    reaction_action_table = pack_reaction_action_table(world)
    material_material_rule_table = _pack_pair_reaction_rules(
        world, world.rulebook.material_material_rules
    )
    material_gas_rule_table = _pack_pair_reaction_rules(world, world.rulebook.material_gas_rules)
    material_light_rule_table = _pack_pair_reaction_rules(
        world, world.rulebook.material_light_rules
    )
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
    rule_table_meta[0]["material_generation"] = int(bridge.table_generations.get("materials", 0))
    rule_table_meta[0]["gas_generation"] = int(bridge.table_generations.get("gases", 0))
    rule_table_meta[0]["light_generation"] = int(bridge.table_generations.get("lights", 0))
    rule_table_meta[0]["optics_generation"] = int(bridge.table_generations.get("optics", 0))
    rule_table_meta[0]["reaction_generation"] = int(bridge.table_generations.get("reactions", 0))

    bridge.shadow_typed_tables["rule_table_meta"] = rule_table_meta.copy()
    bridge.shadow_typed_tables["material_table"] = material_table.copy()
    bridge.shadow_typed_tables["gas_table"] = gas_table.copy()
    bridge.shadow_typed_tables["light_table"] = light_table.copy()
    bridge.shadow_typed_tables["optics_table"] = optics_table.copy()
    bridge.shadow_typed_tables["reaction_action_table"] = reaction_action_table.copy()
    bridge.shadow_typed_tables["material_material_rule_table"] = material_material_rule_table.copy()
    bridge.shadow_typed_tables["material_gas_rule_table"] = material_gas_rule_table.copy()
    bridge.shadow_typed_tables["material_light_rule_table"] = material_light_rule_table.copy()
    bridge.shadow_typed_tables["gas_gas_rule_table"] = gas_gas_rule_table.copy()
    bridge.shadow_typed_tables["gas_light_rule_table"] = gas_light_rule_table.copy()
    bridge.shadow_typed_tables["self_rule_table"] = self_rule_table.copy()

    if bridge.enabled and bridge.ctx is not None:
        bridge._write_typed_table_buffer("rule_table_meta", rule_table_meta)
        bridge._write_typed_table_buffer("material_table", material_table)
        bridge._write_typed_table_buffer("gas_table", gas_table)
        bridge._write_typed_table_buffer("light_table", light_table)
        bridge._write_typed_table_buffer("optics_table", optics_table)
        bridge._write_typed_table_buffer("reaction_action_table", reaction_action_table)
        bridge._write_typed_table_buffer(
            "material_material_rule_table", material_material_rule_table
        )
        bridge._write_typed_table_buffer("material_gas_rule_table", material_gas_rule_table)
        bridge._write_typed_table_buffer("material_light_rule_table", material_light_rule_table)
        bridge._write_typed_table_buffer("gas_gas_rule_table", gas_gas_rule_table)
        bridge._write_typed_table_buffer("gas_light_rule_table", gas_light_rule_table)
        bridge._write_typed_table_buffer("self_rule_table", self_rule_table)

    bridge.rule_table_signature = signature


def sync_world(
    bridge,
    world: "WorldEngine",
    *,
    debug_frame: np.ndarray | None = None,
    upload_debug_texture: bool = True,
    force_cpu_resource_upload: bool = False,
) -> None:
    previous_force_cpu_resource_upload = bridge._force_cpu_resource_upload
    bridge._force_cpu_resource_upload = bool(force_cpu_resource_upload)
    try:
        bridge._sync_world_impl(
            world, debug_frame=debug_frame, upload_debug_texture=upload_debug_texture
        )
    finally:
        bridge._force_cpu_resource_upload = previous_force_cpu_resource_upload


def _pack_cpu_state_uploads(bridge, world: "WorldEngine") -> dict[str, Any]:
    """Pack the non-solver per-frame uploads and decide the CPU-upload flags."""
    uploads: dict[str, Any] = {}
    upload_solver_runtime_from_cpu = bridge._should_upload_cpu_solver_runtime(world)
    upload_island_runtime_from_cpu = bridge._should_upload_cpu_resource(world, "island_runtime")
    upload_powder_reservation_from_cpu = (
        upload_solver_runtime_from_cpu
        and bridge._should_upload_cpu_resource(world, "powder_reservation")
    )
    upload_island_reservation_from_cpu = (
        upload_solver_runtime_from_cpu
        and bridge._should_upload_cpu_resource(world, "island_reservation")
    )
    uploads["upload_solver_runtime_from_cpu"] = upload_solver_runtime_from_cpu
    uploads["upload_island_runtime_from_cpu"] = upload_island_runtime_from_cpu
    uploads["upload_powder_reservation_from_cpu"] = upload_powder_reservation_from_cpu
    uploads["upload_island_reservation_from_cpu"] = upload_island_reservation_from_cpu
    if upload_powder_reservation_from_cpu:
        world.motion_solver.gpu_pipeline.materialize_compact_powder_reservations(
            world,
            download=True,
        )
    entity_state_upload = pack_entity_state_upload(world)
    uploads["entity_state"] = entity_state_upload
    uploads["entity_state_count"] = np.array([len(entity_state_upload)], dtype=np.int32)
    force_source_upload = pack_force_source_upload(world)
    uploads["force_source"] = force_source_upload
    uploads["force_source_count"] = np.array([len(force_source_upload)], dtype=np.int32)
    uploads["island_runtime"] = (
        pack_island_runtime_upload(world)
        if upload_island_runtime_from_cpu
        else bridge.shadow_buffers.get("island_runtime", np.zeros((0,), dtype=ISLAND_RUNTIME_DTYPE))
    )
    uploads["island_runtime_count"] = (
        np.array([len(uploads["island_runtime"])], dtype=np.int32)
        if upload_island_runtime_from_cpu
        else bridge.shadow_buffers.get("island_runtime_count", np.zeros((1,), dtype=np.int32))
    )
    motion_runtime = (
        world.motion_solver.runtime_snapshot()
        if upload_powder_reservation_from_cpu or upload_island_reservation_from_cpu
        else None
    )
    uploads["powder_reservation"] = (
        motion_runtime["powder_reservations"]
        if upload_powder_reservation_from_cpu and motion_runtime is not None
        else bridge.shadow_buffers.get(
            "powder_reservation",
            np.zeros((0,), dtype=getattr(world.motion_solver, "last_powder_reservations").dtype),
        )
    )
    uploads["powder_reservation_count"] = (
        np.array([len(uploads["powder_reservation"])], dtype=np.int32)
        if upload_powder_reservation_from_cpu
        else bridge.shadow_buffers.get("powder_reservation_count", np.zeros((1,), dtype=np.int32))
    )
    uploads["island_reservation"] = (
        motion_runtime["island_reservations"]
        if upload_island_reservation_from_cpu and motion_runtime is not None
        else bridge.shadow_buffers.get(
            "island_reservation",
            np.zeros((0,), dtype=getattr(world.motion_solver, "last_island_reservations").dtype),
        )
    )
    uploads["island_reservation_count"] = (
        np.array([len(uploads["island_reservation"])], dtype=np.int32)
        if upload_island_reservation_from_cpu
        else bridge.shadow_buffers.get("island_reservation_count", np.zeros((1,), dtype=np.int32))
    )
    world_command_upload, world_command_payload_upload = pack_world_command_upload(world)
    uploads["world_command"] = world_command_upload
    uploads["world_command_payload"] = world_command_payload_upload
    readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
    uploads["readback_request"] = readback_request_upload
    uploads["readback_request_label"] = readback_request_label_upload
    uploads["placeholder"] = pack_placeholder_upload(world)
    uploads["placeholder_dirty_rect"] = pack_placeholder_dirty_rect_upload(world)
    upload_active_tile_ttl_from_cpu = bridge._should_upload_cpu_resource(world, "active_tile_ttl")
    upload_active_chunk_mask_from_cpu = bridge._should_upload_cpu_resource(
        world, "active_chunk_mask"
    )
    upload_active_meta_from_cpu = bridge._should_upload_cpu_resource(world, "active_meta")
    uploads["upload_active_tile_ttl_from_cpu"] = upload_active_tile_ttl_from_cpu
    uploads["upload_active_chunk_mask_from_cpu"] = upload_active_chunk_mask_from_cpu
    uploads["upload_active_meta_from_cpu"] = upload_active_meta_from_cpu
    active_tile_ttl_default = np.zeros(
        (world.active.tile_height, world.active.tile_width), dtype=np.int32
    )
    active_chunk_mask_default = np.zeros(
        (world.active.chunk_height, world.active.chunk_width), dtype=np.uint8
    )
    active_tile_ttl_upload = (
        np.asarray(world.active.active_tile_ttl or [], dtype=np.int32)
        if upload_active_tile_ttl_from_cpu
        else bridge._shadow_or_default("active_tile_ttl", active_tile_ttl_default)
    )
    active_chunk_mask_upload = (
        np.asarray(world.active.active_chunk_mask or [], dtype=np.uint8)
        if upload_active_chunk_mask_from_cpu
        else bridge._shadow_or_default("active_chunk_mask", active_chunk_mask_default)
    )
    uploads["active_tile_ttl"] = active_tile_ttl_upload
    uploads["active_chunk_mask"] = active_chunk_mask_upload
    active_meta_default = pack_active_meta_upload(
        world,
        active_tile_count=int(np.count_nonzero(active_tile_ttl_default > 0)),
        active_chunk_count=int(np.count_nonzero(active_chunk_mask_default > 0)),
    )
    uploads["active_meta"] = (
        pack_active_meta_upload(
            world,
            active_tile_count=int(np.count_nonzero(active_tile_ttl_upload > 0)),
            active_chunk_count=int(np.count_nonzero(active_chunk_mask_upload > 0)),
        )
        if upload_active_meta_from_cpu
        else bridge._shadow_or_default("active_meta", active_meta_default)
    )
    return uploads


def _pack_solver_runtime_uploads(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Pack the gas/heat/liquid/reaction runtime uploads (or reuse shadows)."""
    if uploads["upload_solver_runtime_from_cpu"]:
        (
            uploads["gas_runtime_meta"],
            uploads["gas_solve_tile_mask"],
            uploads["gas_solve_gas_mask"],
            uploads["gas_species_runtime"],
        ) = pack_gas_runtime_upload(world)
        (
            uploads["heat_runtime_meta"],
            uploads["heat_solve_tile_mask"],
            uploads["heat_solve_cell_mask"],
            uploads["heat_solve_gas_mask"],
            uploads["heat_phase_target"],
            uploads["heat_boil_target"],
            uploads["heat_condense_target"],
        ) = pack_heat_runtime_upload(world)
        (
            uploads["liquid_runtime_meta"],
            uploads["liquid_solve_tile_mask"],
            uploads["liquid_post_tile_mask"],
            uploads["liquid_post_cell_mask"],
            uploads["liquid_vertical_seam_mask"],
            uploads["liquid_horizontal_seam_mask"],
            uploads["liquid_buoyancy_mask"],
            uploads["liquid_changed_cell_mask"],
        ) = pack_liquid_runtime_upload(world)
        (
            uploads["reaction_runtime_meta"],
            uploads["reaction_timed_solve_tile_mask"],
            uploads["reaction_self_solve_tile_mask"],
            uploads["reaction_material_material_solve_tile_mask"],
            uploads["reaction_material_gas_solve_tile_mask"],
            uploads["reaction_material_light_solve_tile_mask"],
            uploads["reaction_gas_gas_solve_tile_mask"],
            uploads["reaction_gas_light_solve_tile_mask"],
            uploads["reaction_solve_cell_mask"],
            uploads["reaction_solve_gas_mask"],
            uploads["reaction_changed_cell_mask"],
            uploads["reaction_changed_gas_mask"],
            uploads["reaction_ambient_changed_mask"],
            uploads["reaction_timer_changed_mask"],
            uploads["reaction_emitted_light_mask"],
            uploads["reaction_emitted_material_mask"],
        ) = pack_reaction_runtime_upload(world)
        return
    uploads["gas_runtime_meta"] = bridge._shadow_or_default(
        "gas_runtime_meta", np.zeros((1,), dtype=GAS_RUNTIME_META_DTYPE)
    )
    uploads["gas_solve_tile_mask"] = bridge._shadow_or_default(
        "gas_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["gas_solve_gas_mask"] = bridge._shadow_or_default(
        "gas_solve_gas_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["gas_species_runtime"] = bridge._shadow_or_default(
        "gas_species_runtime",
        np.zeros((world.gas_concentration.shape[0],), dtype=GAS_SPECIES_RUNTIME_DTYPE),
    )
    uploads["heat_runtime_meta"] = bridge._shadow_or_default(
        "heat_runtime_meta", np.zeros((1,), dtype=HEAT_RUNTIME_META_DTYPE)
    )
    uploads["heat_solve_tile_mask"] = bridge._shadow_or_default(
        "heat_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["heat_solve_cell_mask"] = bridge._shadow_or_default(
        "heat_solve_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["heat_solve_gas_mask"] = bridge._shadow_or_default(
        "heat_solve_gas_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["heat_phase_target"] = bridge._shadow_or_default(
        "heat_phase_target",
        np.zeros((world.height, world.width), dtype=np.int32),
    )
    uploads["heat_boil_target"] = bridge._shadow_or_default(
        "heat_boil_target",
        np.zeros((world.height, world.width), dtype=np.int32),
    )
    uploads["heat_condense_target"] = bridge._shadow_or_default(
        "heat_condense_target",
        np.zeros(world.gas_concentration.shape, dtype=np.uint8),
    )
    uploads["liquid_runtime_meta"] = bridge._shadow_or_default(
        "liquid_runtime_meta", np.zeros((1,), dtype=LIQUID_RUNTIME_META_DTYPE)
    )
    uploads["liquid_solve_tile_mask"] = bridge._shadow_or_default(
        "liquid_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["liquid_post_tile_mask"] = bridge._shadow_or_default(
        "liquid_post_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["liquid_post_cell_mask"] = bridge._shadow_or_default(
        "liquid_post_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["liquid_vertical_seam_mask"] = bridge._shadow_or_default(
        "liquid_vertical_seam_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["liquid_horizontal_seam_mask"] = bridge._shadow_or_default(
        "liquid_horizontal_seam_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["liquid_buoyancy_mask"] = bridge._shadow_or_default(
        "liquid_buoyancy_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["liquid_changed_cell_mask"] = bridge._shadow_or_default(
        "liquid_changed_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["reaction_runtime_meta"] = bridge._shadow_or_default(
        "reaction_runtime_meta", np.zeros((1,), dtype=REACTION_RUNTIME_META_DTYPE)
    )
    uploads["reaction_timed_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_timed_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_self_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_self_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_material_material_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_material_material_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_material_gas_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_material_gas_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_material_light_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_material_light_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_gas_gas_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_gas_gas_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_gas_light_solve_tile_mask"] = bridge._shadow_or_default(
        "reaction_gas_light_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["reaction_solve_cell_mask"] = bridge._shadow_or_default(
        "reaction_solve_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["reaction_solve_gas_mask"] = bridge._shadow_or_default(
        "reaction_solve_gas_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["reaction_changed_cell_mask"] = bridge._shadow_or_default(
        "reaction_changed_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["reaction_changed_gas_mask"] = bridge._shadow_or_default(
        "reaction_changed_gas_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["reaction_ambient_changed_mask"] = bridge._shadow_or_default(
        "reaction_ambient_changed_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["reaction_timer_changed_mask"] = bridge._shadow_or_default(
        "reaction_timer_changed_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["reaction_emitted_light_mask"] = bridge._shadow_or_default(
        "reaction_emitted_light_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["reaction_emitted_material_mask"] = bridge._shadow_or_default(
        "reaction_emitted_material_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )


def _pack_collapse_runtime_uploads(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Pack the collapse runtime uploads unless the GPU owns the collapse masks."""
    collapse_mask_resources = (
        "collapse_structural_mask",
        "collapse_support_seed_mask",
        "collapse_supported_mask",
        "collapse_unsupported_mask",
        "collapse_delayed_pending_mask",
        "collapse_immune_unsupported_mask",
        "collapse_collapsed_cell_mask",
    )
    upload_collapse_runtime_from_cpu = uploads["upload_solver_runtime_from_cpu"] and not (
        any(name in bridge.gpu_authoritative_resources for name in collapse_mask_resources)
    )
    if upload_collapse_runtime_from_cpu:
        (
            uploads["collapse_runtime_meta"],
            uploads["collapse_solve_region_mask"],
            uploads["collapse_structural_mask"],
            uploads["collapse_support_seed_mask"],
            uploads["collapse_supported_mask"],
            uploads["collapse_unsupported_mask"],
            uploads["collapse_delayed_pending_mask"],
            uploads["collapse_immune_unsupported_mask"],
            uploads["collapse_collapsed_cell_mask"],
            uploads["collapse_component"],
        ) = pack_collapse_runtime_upload(world)
        return
    cell_zero = np.zeros((world.height, world.width), dtype=np.int32)
    uploads["collapse_runtime_meta"] = bridge.shadow_buffers.get(
        "collapse_runtime_meta",
        np.zeros((1,), dtype=COLLAPSE_RUNTIME_META_DTYPE),
    )
    uploads["collapse_solve_region_mask"] = bridge.shadow_buffers.get(
        "collapse_solve_region_mask", cell_zero
    )
    uploads["collapse_structural_mask"] = bridge.shadow_buffers.get(
        "collapse_structural_mask", cell_zero
    )
    uploads["collapse_support_seed_mask"] = bridge.shadow_buffers.get(
        "collapse_support_seed_mask", cell_zero
    )
    uploads["collapse_supported_mask"] = bridge.shadow_buffers.get(
        "collapse_supported_mask", cell_zero
    )
    uploads["collapse_unsupported_mask"] = bridge.shadow_buffers.get(
        "collapse_unsupported_mask", cell_zero
    )
    uploads["collapse_delayed_pending_mask"] = bridge.shadow_buffers.get(
        "collapse_delayed_pending_mask", cell_zero
    )
    uploads["collapse_immune_unsupported_mask"] = bridge.shadow_buffers.get(
        "collapse_immune_unsupported_mask", cell_zero
    )
    uploads["collapse_collapsed_cell_mask"] = bridge.shadow_buffers.get(
        "collapse_collapsed_cell_mask", cell_zero
    )
    uploads["collapse_component"] = bridge.shadow_buffers.get(
        "collapse_component",
        np.zeros((0,), dtype=COLLAPSE_COMPONENT_DTYPE),
    )


def _pack_optics_runtime_uploads(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Pack the optics runtime uploads (or reuse shadows)."""
    if uploads["upload_solver_runtime_from_cpu"]:
        (
            uploads["optics_runtime_meta"],
            uploads["optics_solve_tile_mask"],
            uploads["optics_solve_cell_mask"],
            uploads["optics_solve_gas_mask"],
            uploads["optics_visible_changed_mask"],
            uploads["optics_cell_dose_changed_mask"],
            uploads["optics_gas_dose_changed_mask"],
            uploads["optics_emitter_origin_mask"],
        ) = pack_optics_runtime_upload(world)
        return
    uploads["optics_runtime_meta"] = bridge._shadow_or_default(
        "optics_runtime_meta", np.zeros((1,), dtype=OPTICS_RUNTIME_META_DTYPE)
    )
    uploads["optics_solve_tile_mask"] = bridge._shadow_or_default(
        "optics_solve_tile_mask",
        np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
    )
    uploads["optics_solve_cell_mask"] = bridge._shadow_or_default(
        "optics_solve_cell_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["optics_solve_gas_mask"] = bridge._shadow_or_default(
        "optics_solve_gas_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["optics_visible_changed_mask"] = bridge._shadow_or_default(
        "optics_visible_changed_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["optics_cell_dose_changed_mask"] = bridge._shadow_or_default(
        "optics_cell_dose_changed_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )
    uploads["optics_gas_dose_changed_mask"] = bridge._shadow_or_default(
        "optics_gas_dose_changed_mask",
        np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
    )
    uploads["optics_emitter_origin_mask"] = bridge._shadow_or_default(
        "optics_emitter_origin_mask",
        np.zeros((world.height, world.width), dtype=np.uint8),
    )


def _pack_frame_meta_upload(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Pack the page stripe payloads and the frame meta header."""
    (
        uploads["page_stripe_meta"],
        uploads["page_stripe_section"],
        uploads["page_stripe_payload"],
    ) = pack_page_stripe_upload(world)
    uploads["frame_meta"] = pack_frame_meta_upload(
        world,
        entity_count=len(uploads["entity_state"]),
        force_source_count=len(uploads["force_source"]),
        world_command_count=len(uploads["world_command"]),
        readback_request_count=len(uploads["readback_request"]),
        placeholder_count=len(uploads["placeholder"]),
        placeholder_dirty_rect_count=len(uploads["placeholder_dirty_rect"]),
        active_tile_count=int(uploads["active_meta"][0]["active_tile_count"]),
        active_chunk_count=int(uploads["active_meta"][0]["active_chunk_count"]),
        page_update_count=len(uploads["page_stripe_meta"]),
        page_stripe_section_count=len(uploads["page_stripe_section"]),
    )


def _stage_shadow_buffers(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Publish every packed upload into the CPU shadow buffers."""
    bridge.shadow_buffers["entity_state"] = uploads["entity_state"].copy()
    bridge.shadow_buffers["entity_state_count"] = uploads["entity_state_count"].copy()
    bridge.shadow_buffers["force_source"] = uploads["force_source"].copy()
    bridge.shadow_buffers["force_source_count"] = uploads["force_source_count"].copy()
    if uploads["upload_island_runtime_from_cpu"] or "island_runtime" not in bridge.shadow_buffers:
        bridge.shadow_buffers["island_runtime"] = uploads["island_runtime"].copy()
        bridge.shadow_buffers["island_runtime_count"] = uploads["island_runtime_count"].copy()
    if (
        uploads["upload_powder_reservation_from_cpu"]
        or "powder_reservation" not in bridge.shadow_buffers
    ):
        bridge.shadow_buffers["powder_reservation"] = uploads["powder_reservation"].copy()
        bridge.shadow_buffers["powder_reservation_count"] = uploads[
            "powder_reservation_count"
        ].copy()
    if (
        uploads["upload_island_reservation_from_cpu"]
        or "island_reservation" not in bridge.shadow_buffers
    ):
        bridge.shadow_buffers["island_reservation"] = uploads["island_reservation"].copy()
        bridge.shadow_buffers["island_reservation_count"] = uploads[
            "island_reservation_count"
        ].copy()
    bridge.shadow_buffers["world_command"] = uploads["world_command"].copy()
    bridge.shadow_buffers["world_command_payload"] = uploads["world_command_payload"].copy()
    bridge.shadow_buffers["readback_request"] = uploads["readback_request"].copy()
    bridge.shadow_buffers["readback_request_label"] = uploads["readback_request_label"].copy()
    bridge.shadow_buffers["placeholder"] = uploads["placeholder"].copy()
    bridge.shadow_buffers["placeholder_dirty_rect"] = uploads["placeholder_dirty_rect"].copy()
    bridge.shadow_buffers["island_id"] = world.island_id.astype(np.int32).copy()
    bridge.shadow_buffers["entity_id"] = world.entity_id.astype(np.int32).copy()
    bridge.shadow_buffers["placeholder_displaced_material"] = (
        world.placeholder_displaced_material.astype(np.int32).copy()
    )
    bridge.shadow_buffers["collapse_delay_pending"] = world.collapse_delay_pending.astype(
        np.int32
    ).copy()
    bridge.shadow_buffers["cell_optical_dose"] = world.cell_optical_dose.astype(np.float32).copy()
    bridge.shadow_buffers["gas_optical_dose"] = world.gas_optical_dose.astype(np.float32).copy()
    bridge.shadow_buffers["active_meta"] = uploads["active_meta"].copy()
    bridge.shadow_buffers["active_tile_ttl"] = uploads["active_tile_ttl"].copy()
    bridge.shadow_buffers["active_chunk_mask"] = uploads["active_chunk_mask"].copy()
    bridge.shadow_buffers["gas_runtime_meta"] = uploads["gas_runtime_meta"].copy()
    bridge.shadow_buffers["gas_solve_tile_mask"] = uploads["gas_solve_tile_mask"].copy()
    bridge.shadow_buffers["gas_solve_gas_mask"] = uploads["gas_solve_gas_mask"].copy()
    bridge.shadow_buffers["gas_species_runtime"] = uploads["gas_species_runtime"].copy()
    bridge.shadow_buffers["heat_runtime_meta"] = uploads["heat_runtime_meta"].copy()
    bridge.shadow_buffers["heat_solve_tile_mask"] = uploads["heat_solve_tile_mask"].copy()
    bridge.shadow_buffers["heat_solve_cell_mask"] = uploads["heat_solve_cell_mask"].copy()
    bridge.shadow_buffers["heat_solve_gas_mask"] = uploads["heat_solve_gas_mask"].copy()
    bridge.shadow_buffers["heat_phase_target"] = uploads["heat_phase_target"].copy()
    bridge.shadow_buffers["heat_boil_target"] = uploads["heat_boil_target"].copy()
    bridge.shadow_buffers["heat_condense_target"] = uploads["heat_condense_target"].copy()
    bridge.shadow_buffers["liquid_runtime_meta"] = uploads["liquid_runtime_meta"].copy()
    bridge.shadow_buffers["liquid_solve_tile_mask"] = uploads["liquid_solve_tile_mask"].copy()
    bridge.shadow_buffers["liquid_post_tile_mask"] = uploads["liquid_post_tile_mask"].copy()
    bridge.shadow_buffers["liquid_post_cell_mask"] = uploads["liquid_post_cell_mask"].copy()
    bridge.shadow_buffers["liquid_vertical_seam_mask"] = uploads["liquid_vertical_seam_mask"].copy()
    bridge.shadow_buffers["liquid_horizontal_seam_mask"] = uploads[
        "liquid_horizontal_seam_mask"
    ].copy()
    bridge.shadow_buffers["liquid_buoyancy_mask"] = uploads["liquid_buoyancy_mask"].copy()
    bridge.shadow_buffers["liquid_changed_cell_mask"] = uploads["liquid_changed_cell_mask"].copy()
    bridge.shadow_buffers["reaction_runtime_meta"] = uploads["reaction_runtime_meta"].copy()
    bridge.shadow_buffers["reaction_timed_solve_tile_mask"] = uploads[
        "reaction_timed_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_self_solve_tile_mask"] = uploads[
        "reaction_self_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_material_material_solve_tile_mask"] = uploads[
        "reaction_material_material_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_material_gas_solve_tile_mask"] = uploads[
        "reaction_material_gas_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_material_light_solve_tile_mask"] = uploads[
        "reaction_material_light_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_gas_gas_solve_tile_mask"] = uploads[
        "reaction_gas_gas_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_gas_light_solve_tile_mask"] = uploads[
        "reaction_gas_light_solve_tile_mask"
    ].copy()
    bridge.shadow_buffers["reaction_solve_cell_mask"] = uploads["reaction_solve_cell_mask"].copy()
    bridge.shadow_buffers["reaction_solve_gas_mask"] = uploads["reaction_solve_gas_mask"].copy()
    bridge.shadow_buffers["reaction_changed_cell_mask"] = uploads[
        "reaction_changed_cell_mask"
    ].copy()
    bridge.shadow_buffers["reaction_changed_gas_mask"] = uploads["reaction_changed_gas_mask"].copy()
    bridge.shadow_buffers["reaction_ambient_changed_mask"] = uploads[
        "reaction_ambient_changed_mask"
    ].copy()
    bridge.shadow_buffers["reaction_timer_changed_mask"] = uploads[
        "reaction_timer_changed_mask"
    ].copy()
    bridge.shadow_buffers["reaction_emitted_light_mask"] = uploads[
        "reaction_emitted_light_mask"
    ].copy()
    bridge.shadow_buffers["reaction_emitted_material_mask"] = uploads[
        "reaction_emitted_material_mask"
    ].copy()
    bridge.shadow_buffers["collapse_runtime_meta"] = uploads["collapse_runtime_meta"].copy()
    bridge.shadow_buffers["collapse_solve_region_mask"] = uploads[
        "collapse_solve_region_mask"
    ].copy()
    bridge.shadow_buffers["collapse_structural_mask"] = uploads["collapse_structural_mask"].copy()
    bridge.shadow_buffers["collapse_support_seed_mask"] = uploads[
        "collapse_support_seed_mask"
    ].copy()
    bridge.shadow_buffers["collapse_supported_mask"] = uploads["collapse_supported_mask"].copy()
    bridge.shadow_buffers["collapse_unsupported_mask"] = uploads["collapse_unsupported_mask"].copy()
    bridge.shadow_buffers["collapse_delayed_pending_mask"] = uploads[
        "collapse_delayed_pending_mask"
    ].copy()
    bridge.shadow_buffers["collapse_immune_unsupported_mask"] = uploads[
        "collapse_immune_unsupported_mask"
    ].copy()
    bridge.shadow_buffers["collapse_collapsed_cell_mask"] = uploads[
        "collapse_collapsed_cell_mask"
    ].copy()
    bridge.shadow_buffers["collapse_component"] = uploads["collapse_component"].copy()
    bridge.shadow_buffers["optics_runtime_meta"] = uploads["optics_runtime_meta"].copy()
    bridge.shadow_buffers["optics_solve_tile_mask"] = uploads["optics_solve_tile_mask"].copy()
    bridge.shadow_buffers["optics_solve_cell_mask"] = uploads["optics_solve_cell_mask"].copy()
    bridge.shadow_buffers["optics_solve_gas_mask"] = uploads["optics_solve_gas_mask"].copy()
    bridge.shadow_buffers["optics_visible_changed_mask"] = uploads[
        "optics_visible_changed_mask"
    ].copy()
    bridge.shadow_buffers["optics_cell_dose_changed_mask"] = uploads[
        "optics_cell_dose_changed_mask"
    ].copy()
    bridge.shadow_buffers["optics_gas_dose_changed_mask"] = uploads[
        "optics_gas_dose_changed_mask"
    ].copy()
    bridge.shadow_buffers["optics_emitter_origin_mask"] = uploads[
        "optics_emitter_origin_mask"
    ].copy()
    bridge.shadow_buffers["page_stripe_meta"] = uploads["page_stripe_meta"].copy()
    bridge.shadow_buffers["page_stripe_section"] = uploads["page_stripe_section"].copy()
    bridge.shadow_buffers["page_stripe_payload"] = uploads["page_stripe_payload"].copy()
    bridge.shadow_buffers["frame_meta"] = uploads["frame_meta"].copy()


def _upload_cpu_written_cell_rects(bridge, world: "WorldEngine") -> bool:
    """Upload only the cells the CPU explicitly wrote since the last sync.

    The CPU cell arrays are not refreshed from the GPU every frame, so the
    legacy full-array upload after any CPU write rewound the whole world to
    a stale snapshot (paint strokes teleported falling water back up, and
    partial readback coverage produced tile-aligned stale mosaics).  The
    written cells themselves are correct in the CPU arrays by construction,
    so uploading just those rects merges the writes into the live GPU
    state.  Uploaded resources are re-marked GPU-authoritative so the
    full-array uploads in this sync skip them.
    """
    rects = getattr(world, "_cpu_written_cell_rects", None)
    if not rects:
        return False
    width, height = int(world.width), int(world.height)

    def _clamped_rects() -> list[tuple[int, int, int, int]]:
        clamped: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in rects:
            cx0, cy0 = max(0, int(x0)), max(0, int(y0))
            cx1, cy1 = min(width, int(x1)), min(height, int(y1))
            if cx1 > cx0 and cy1 > cy0:
                clamped.append((cx0, cy0, cx1, cy1))
        return clamped

    clamped = _clamped_rects()
    rects.clear()
    if bridge._force_cpu_resource_upload:
        return False
    if getattr(world, "simulation_backend", "") != "gpu":
        return False
    if not bridge.enabled or bridge.ctx is None:
        return False
    if not clamped:
        return False
    uint32_bytes = int(np.dtype(np.uint32).itemsize)
    uploaded: list[str] = []
    if bridge._should_upload_cpu_resource(world, "cell_core"):
        buffer = bridge.buffers["cell_core"]
        for x0, y0, x1, y1 in clamped:
            packed = pack_cell_core_window(world, x0, y0, x1, y1)
            for row in range(y1 - y0):
                offset = ((y0 + row) * width + x0) * 5 * uint32_bytes
                buffer.write(packed[row].tobytes(), offset=offset)
        uploaded.append("cell_core")
    for name, array in (
        ("island_id", world.island_id),
        ("entity_id", world.entity_id),
        ("placeholder_displaced_material", world.placeholder_displaced_material),
    ):
        if not bridge._should_upload_cpu_resource(world, name):
            continue
        buffer = bridge.buffers[name]
        data = np.ascontiguousarray(array.astype(np.int32))
        for x0, y0, x1, y1 in clamped:
            for row in range(y0, y1):
                offset = (row * width + x0) * uint32_bytes
                buffer.write(data[row, x0:x1].tobytes(), offset=offset)
        uploaded.append(name)
    if bridge._should_upload_cpu_resource(world, "material"):
        texture = bridge.textures["material"]
        for x0, y0, x1, y1 in clamped:
            texture.write(
                world.material_id[y0:y1, x0:x1].astype("f4").tobytes(),
                viewport=(x0, y0, x1 - x0, y1 - y0),
            )
        uploaded.append("material")
    if uploaded:
        bridge.mark_gpu_authoritative(*uploaded)
    return True


def _upload_bridge_core_buffers(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Write the core per-frame buffers (entities, commands, doses) to the GPU."""
    bridge._ensure_atlas_texture(world)
    _upload_cpu_written_cell_rects(bridge, world)
    upload_cell_dose_from_cpu = bridge._should_upload_cpu_resource(world, "cell_optical_dose")
    upload_gas_dose_from_cpu = bridge._should_upload_cpu_resource(world, "gas_optical_dose")
    upload_light_from_cpu = bridge._should_upload_cpu_resource(world, "light")
    upload_visible_from_cpu = bridge._should_upload_cpu_resource(world, "visible_illumination")
    uploads["upload_light_from_cpu"] = upload_light_from_cpu
    uploads["upload_visible_from_cpu"] = upload_visible_from_cpu
    if (
        upload_cell_dose_from_cpu
        or upload_gas_dose_from_cpu
        or upload_light_from_cpu
        or upload_visible_from_cpu
    ):
        world._gpu_optics_outputs_clear = False
        optics_pipeline = getattr(getattr(world, "optics_solver", None), "gpu_pipeline", None)
        invalidate_sparse = getattr(optics_pipeline, "invalidate_sparse_runtime", None)
        if callable(invalidate_sparse):
            invalidate_sparse()
    if bridge._should_upload_cpu_resource(world, "cell_core"):
        packed = pack_cell_core(world)
        bridge.buffers["cell_core"].write(packed.tobytes())
    if bridge._should_upload_cpu_resource(world, "island_id"):
        bridge.buffers["island_id"].write(
            np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes()
        )
    if bridge._should_upload_cpu_resource(world, "entity_id"):
        bridge.buffers["entity_id"].write(
            np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes()
        )
    if bridge._should_upload_cpu_resource(world, "placeholder_displaced_material"):
        bridge.buffers["placeholder_displaced_material"].write(
            np.ascontiguousarray(world.placeholder_displaced_material.astype(np.int32)).tobytes()
        )
    if bridge._should_upload_cpu_resource(world, "collapse_delay_pending"):
        bridge.buffers["collapse_delay_pending"].write(
            np.ascontiguousarray(world.collapse_delay_pending.astype(np.int32)).tobytes()
        )
    if bridge._should_upload_cpu_resource(world, "gas_concentration"):
        bridge.buffers["gas_concentration"].write(world.gas_concentration.astype("f4").tobytes())
    if upload_cell_dose_from_cpu:
        bridge.buffers["cell_optical_dose"].write(
            np.ascontiguousarray(world.cell_optical_dose.astype(np.float32)).tobytes()
        )
    if upload_gas_dose_from_cpu:
        bridge.buffers["gas_optical_dose"].write(
            np.ascontiguousarray(world.gas_optical_dose.astype(np.float32)).tobytes()
        )
    bridge._write_dynamic_buffer("entity_state", uploads["entity_state"])
    bridge.buffers["entity_state_count"].write(uploads["entity_state_count"].tobytes())
    bridge._write_dynamic_buffer("force_source", uploads["force_source"])
    bridge.buffers["force_source_count"].write(uploads["force_source_count"].tobytes())
    if uploads["upload_island_runtime_from_cpu"]:
        bridge._write_dynamic_buffer("island_runtime", uploads["island_runtime"])
        bridge.buffers["island_runtime_count"].write(uploads["island_runtime_count"].tobytes())
    if uploads["upload_powder_reservation_from_cpu"]:
        bridge._write_dynamic_buffer("powder_reservation", uploads["powder_reservation"])
        bridge.buffers["powder_reservation_count"].write(
            uploads["powder_reservation_count"].tobytes()
        )
        bridge.clear_gpu_authoritative(
            "powder_reservation",
            "powder_reservation_compact",
            "powder_reservation_standard",
            "powder_reservation_cpu_mirror",
        )
    if uploads["upload_island_reservation_from_cpu"]:
        bridge._write_dynamic_buffer("island_reservation", uploads["island_reservation"])
        bridge.buffers["island_reservation_count"].write(
            uploads["island_reservation_count"].tobytes()
        )
    bridge._write_dynamic_buffer("world_command", uploads["world_command"])
    bridge._write_dynamic_buffer("world_command_payload", uploads["world_command_payload"])
    bridge._write_dynamic_buffer("readback_request", uploads["readback_request"])
    bridge._write_dynamic_buffer("readback_request_label", uploads["readback_request_label"])
    bridge._write_dynamic_buffer("placeholder", uploads["placeholder"])
    bridge._write_dynamic_buffer("placeholder_dirty_rect", uploads["placeholder_dirty_rect"])


def _upload_bridge_solver_buffers(bridge, world: "WorldEngine", uploads: dict[str, Any]) -> None:
    """Write the active-scheduler and solver runtime buffers to the GPU."""
    if bridge._should_upload_cpu_resource(world, "active_meta"):
        bridge._write_dynamic_buffer("active_meta", uploads["active_meta"])
    if bridge._should_upload_cpu_resource(world, "active_tile_ttl"):
        bridge._write_dynamic_buffer("active_tile_ttl", uploads["active_tile_ttl"])
    if bridge._should_upload_cpu_resource(world, "active_chunk_mask"):
        bridge._write_dynamic_buffer(
            "active_chunk_mask", uploads["active_chunk_mask"].astype(np.int32, copy=False)
        )
    if getattr(world, "simulation_backend", "") == "gpu" and (
        uploads["upload_active_meta_from_cpu"]
        or uploads["upload_active_tile_ttl_from_cpu"]
        or uploads["upload_active_chunk_mask_from_cpu"]
    ):
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge._write_dynamic_buffer("gas_runtime_meta", uploads["gas_runtime_meta"])
    bridge._write_dynamic_buffer("gas_solve_tile_mask", uploads["gas_solve_tile_mask"])
    bridge._write_dynamic_buffer("gas_solve_gas_mask", uploads["gas_solve_gas_mask"])
    bridge._write_dynamic_buffer("gas_species_runtime", uploads["gas_species_runtime"])
    bridge._write_dynamic_buffer("heat_runtime_meta", uploads["heat_runtime_meta"])
    bridge._write_dynamic_buffer("heat_solve_tile_mask", uploads["heat_solve_tile_mask"])
    bridge._write_dynamic_buffer("heat_solve_cell_mask", uploads["heat_solve_cell_mask"])
    bridge._write_dynamic_buffer("heat_solve_gas_mask", uploads["heat_solve_gas_mask"])
    bridge._write_dynamic_buffer("heat_phase_target", uploads["heat_phase_target"])
    bridge._write_dynamic_buffer("heat_boil_target", uploads["heat_boil_target"])
    bridge._write_dynamic_buffer("heat_condense_target", uploads["heat_condense_target"])
    bridge._write_dynamic_buffer("liquid_runtime_meta", uploads["liquid_runtime_meta"])
    bridge._write_dynamic_buffer("liquid_solve_tile_mask", uploads["liquid_solve_tile_mask"])
    bridge._write_dynamic_buffer("liquid_post_tile_mask", uploads["liquid_post_tile_mask"])
    bridge._write_dynamic_buffer("liquid_post_cell_mask", uploads["liquid_post_cell_mask"])
    bridge._write_dynamic_buffer("liquid_vertical_seam_mask", uploads["liquid_vertical_seam_mask"])
    bridge._write_dynamic_buffer(
        "liquid_horizontal_seam_mask", uploads["liquid_horizontal_seam_mask"]
    )
    bridge._write_dynamic_buffer("liquid_buoyancy_mask", uploads["liquid_buoyancy_mask"])
    bridge._write_dynamic_buffer("liquid_changed_cell_mask", uploads["liquid_changed_cell_mask"])
    bridge._write_dynamic_buffer("reaction_runtime_meta", uploads["reaction_runtime_meta"])
    bridge._write_dynamic_buffer(
        "reaction_timed_solve_tile_mask", uploads["reaction_timed_solve_tile_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_self_solve_tile_mask", uploads["reaction_self_solve_tile_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_material_material_solve_tile_mask",
        uploads["reaction_material_material_solve_tile_mask"],
    )
    bridge._write_dynamic_buffer(
        "reaction_material_gas_solve_tile_mask",
        uploads["reaction_material_gas_solve_tile_mask"],
    )
    bridge._write_dynamic_buffer(
        "reaction_material_light_solve_tile_mask",
        uploads["reaction_material_light_solve_tile_mask"],
    )
    bridge._write_dynamic_buffer(
        "reaction_gas_gas_solve_tile_mask", uploads["reaction_gas_gas_solve_tile_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_gas_light_solve_tile_mask", uploads["reaction_gas_light_solve_tile_mask"]
    )
    bridge._write_dynamic_buffer("reaction_solve_cell_mask", uploads["reaction_solve_cell_mask"])
    bridge._write_dynamic_buffer("reaction_solve_gas_mask", uploads["reaction_solve_gas_mask"])
    bridge._write_dynamic_buffer(
        "reaction_changed_cell_mask", uploads["reaction_changed_cell_mask"]
    )
    bridge._write_dynamic_buffer("reaction_changed_gas_mask", uploads["reaction_changed_gas_mask"])
    bridge._write_dynamic_buffer(
        "reaction_ambient_changed_mask", uploads["reaction_ambient_changed_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_timer_changed_mask", uploads["reaction_timer_changed_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_emitted_light_mask", uploads["reaction_emitted_light_mask"]
    )
    bridge._write_dynamic_buffer(
        "reaction_emitted_material_mask", uploads["reaction_emitted_material_mask"]
    )
    bridge._write_dynamic_buffer("collapse_runtime_meta", uploads["collapse_runtime_meta"])
    bridge._write_dynamic_buffer(
        "collapse_solve_region_mask", uploads["collapse_solve_region_mask"]
    )
    if bridge._should_upload_cpu_resource(world, "collapse_structural_mask"):
        bridge._write_dynamic_buffer(
            "collapse_structural_mask", uploads["collapse_structural_mask"]
        )
    if bridge._should_upload_cpu_resource(world, "collapse_support_seed_mask"):
        bridge._write_dynamic_buffer(
            "collapse_support_seed_mask", uploads["collapse_support_seed_mask"]
        )
    if bridge._should_upload_cpu_resource(world, "collapse_supported_mask"):
        bridge._write_dynamic_buffer("collapse_supported_mask", uploads["collapse_supported_mask"])
    if bridge._should_upload_cpu_resource(world, "collapse_unsupported_mask"):
        bridge._write_dynamic_buffer(
            "collapse_unsupported_mask", uploads["collapse_unsupported_mask"]
        )
    if bridge._should_upload_cpu_resource(world, "collapse_delayed_pending_mask"):
        bridge._write_dynamic_buffer(
            "collapse_delayed_pending_mask", uploads["collapse_delayed_pending_mask"]
        )
    if bridge._should_upload_cpu_resource(world, "collapse_immune_unsupported_mask"):
        bridge._write_dynamic_buffer(
            "collapse_immune_unsupported_mask", uploads["collapse_immune_unsupported_mask"]
        )
    if bridge._should_upload_cpu_resource(world, "collapse_collapsed_cell_mask"):
        bridge._write_dynamic_buffer(
            "collapse_collapsed_cell_mask", uploads["collapse_collapsed_cell_mask"]
        )
    bridge._write_dynamic_buffer("collapse_component", uploads["collapse_component"])
    bridge._write_dynamic_buffer("optics_runtime_meta", uploads["optics_runtime_meta"])
    bridge._write_dynamic_buffer("optics_solve_tile_mask", uploads["optics_solve_tile_mask"])
    bridge._write_dynamic_buffer("optics_solve_cell_mask", uploads["optics_solve_cell_mask"])
    bridge._write_dynamic_buffer("optics_solve_gas_mask", uploads["optics_solve_gas_mask"])
    bridge._write_dynamic_buffer(
        "optics_visible_changed_mask", uploads["optics_visible_changed_mask"]
    )
    bridge._write_dynamic_buffer(
        "optics_cell_dose_changed_mask", uploads["optics_cell_dose_changed_mask"]
    )
    bridge._write_dynamic_buffer(
        "optics_gas_dose_changed_mask", uploads["optics_gas_dose_changed_mask"]
    )
    bridge._write_dynamic_buffer(
        "optics_emitter_origin_mask", uploads["optics_emitter_origin_mask"]
    )
    bridge._write_dynamic_buffer("page_stripe_meta", uploads["page_stripe_meta"])
    bridge._write_dynamic_buffer("page_stripe_section", uploads["page_stripe_section"])
    bridge._write_dynamic_buffer("page_stripe_payload", uploads["page_stripe_payload"])
    bridge.buffers["frame_meta"].write(uploads["frame_meta"].tobytes())


def _upload_bridge_textures(
    bridge,
    world: "WorldEngine",
    uploads: dict[str, Any],
    *,
    debug_frame: np.ndarray | None,
    upload_debug_texture: bool,
) -> None:
    """Write the material/light/debug/ambient textures to the GPU."""
    if bridge._should_upload_cpu_resource(world, "material"):
        bridge.textures["material"].write(world.material_id.astype("f4").tobytes())
    if uploads["upload_light_from_cpu"]:
        light_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
        light_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
        light_rgba[..., 3] = 1.0
        bridge.textures["light"].write(light_rgba.tobytes())
    if uploads["upload_visible_from_cpu"]:
        visible_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
        visible_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
        visible_rgba[..., 3] = 1.0
        bridge.textures["visible_illumination"].write(visible_rgba.tobytes())
    if upload_debug_texture:
        if debug_frame is None:
            debug_frame = world.debug_frame(world.default_debug_view)
        debug_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
        debug_rgba[..., :3] = np.clip(debug_frame, 0.0, 1.0)
        debug_rgba[..., 3] = 1.0
        bridge.textures["debug"].write(debug_rgba.tobytes())
    if bridge._should_upload_cpu_resource(world, "ambient_temperature"):
        bridge.textures["ambient_temperature"].write(
            world.ambient_temperature.astype("f4").tobytes()
        )
    if bridge._should_upload_cpu_resource(world, "pressure_ping"):
        bridge.textures["pressure_ping"].write(world.pressure_ping.astype("f4").tobytes())
    if bridge._should_upload_cpu_resource(world, "flow_velocity"):
        bridge.textures["flow_velocity"].write(world.flow_velocity.astype("f4").tobytes())


def _mark_synced_resources_gpu_authoritative(bridge, world: "WorldEngine") -> None:
    if getattr(world, "simulation_backend", "") == "gpu":
        bridge.mark_gpu_authoritative(
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


def _sync_world_impl(
    bridge,
    world: "WorldEngine",
    *,
    debug_frame: np.ndarray | None = None,
    upload_debug_texture: bool = True,
) -> None:
    bridge.ensure_world_resources(world)
    bridge.sync_rule_tables(world)
    uploads = _pack_cpu_state_uploads(bridge, world)
    _pack_solver_runtime_uploads(bridge, world, uploads)
    _pack_collapse_runtime_uploads(bridge, world, uploads)
    _pack_optics_runtime_uploads(bridge, world, uploads)
    _pack_frame_meta_upload(bridge, world, uploads)
    _stage_shadow_buffers(bridge, world, uploads)
    if not bridge.enabled or bridge.ctx is None:
        return
    _upload_bridge_core_buffers(bridge, world, uploads)
    _upload_bridge_solver_buffers(bridge, world, uploads)
    _upload_bridge_textures(
        bridge,
        world,
        uploads,
        debug_frame=debug_frame,
        upload_debug_texture=upload_debug_texture,
    )
    _mark_synced_resources_gpu_authoritative(bridge, world)


def download_gpu_authoritative_resources(bridge, world: "WorldEngine") -> None:
    """Refresh the CPU mirror from GPU-authoritative resources.

    Formal GPU frames keep the world state on the GPU and do not maintain the
    CPU mirror, so a plain ``sync_world(force_cpu_resource_upload=True)`` after
    GPU-resident writes (console ticks, immediate mutations) would overwrite
    the newer GPU state with the stale mirror.  Call this before tearing down
    a context whose GPU state must survive a re-attach; the re-attach sync
    then re-uploads an accurate mirror.

    ``material`` and ``active_meta`` need no download: the material texture is
    re-derived from ``material_id`` (covered by ``cell_core``) and the active
    scheduler meta is rebuilt from the tile TTL / chunk mask on re-sync.
    """
    if not bridge.enabled or bridge.ctx is None:
        return
    authoritative = bridge.gpu_authoritative_resources
    if not authoritative:
        return
    # Flush the driver's command queue before mapping buffers: without an
    # explicit finish the reads below can observe pre-write (uninitialised)
    # buffer contents on some EGL/driver combinations.
    bridge.ctx.finish()
    buffers = bridge.buffers
    textures = bridge.textures
    width = int(world.width)
    height = int(world.height)

    if "cell_core" in authoritative:
        buffer = buffers.get("cell_core")
        if buffer is not None:
            packed = np.frombuffer(
                buffer.read(size=width * height * 5 * np.dtype(np.uint32).itemsize),
                dtype=np.uint32,
            ).reshape((height, width, 5))
            unpacked = unpack_cell_core(packed)
            # A genuine core snapshot can never reference a material outside
            # the rulebook (``debug_frame`` indexes ``material_base_color``
            # with it).  If any cell does, the buffer content is corrupt
            # (e.g. an uninitialised ping-pong buffer published by a
            # provenance swap); keep the CPU mirror instead of importing it.
            if not bool((unpacked["material_id"] >= len(world.material_base_color)).any()):
                world.material_id[:] = unpacked["material_id"]
                world.phase[:] = unpacked["phase"]
                world.cell_flags[:] = unpacked["cell_flags"]
                world.velocity[:] = unpacked["velocity"]
                world.cell_temperature[:] = unpacked["cell_temperature"]
                world.timer_pack[:] = unpacked["timer_pack"]
                world.integrity[:] = unpacked["integrity"]
    for name in ("island_id", "entity_id", "placeholder_displaced_material"):
        if name not in authoritative:
            continue
        buffer = buffers.get(name)
        if buffer is None:
            continue
        mirror = getattr(world, name)
        mirror[:] = np.frombuffer(buffer.read(size=mirror.nbytes), dtype=np.int32).reshape(
            mirror.shape
        )
    if "collapse_delay_pending" in authoritative:
        buffer = buffers.get("collapse_delay_pending")
        if buffer is not None:
            mirror = world.collapse_delay_pending
            mirror[:] = (
                np.frombuffer(
                    buffer.read(size=mirror.size * np.dtype(np.int32).itemsize), dtype=np.int32
                ).reshape(mirror.shape)
                != 0
            )
    for name in ("gas_concentration", "cell_optical_dose", "gas_optical_dose"):
        if name not in authoritative:
            continue
        buffer = buffers.get(name)
        if buffer is None:
            continue
        mirror = getattr(world, name)
        mirror[:] = np.frombuffer(buffer.read(size=mirror.nbytes), dtype=np.float32).reshape(
            mirror.shape
        )
    for name in ("ambient_temperature", "pressure_ping", "flow_velocity"):
        if name not in authoritative:
            continue
        texture = textures.get(name)
        if texture is None:
            continue
        mirror = getattr(world, name)
        mirror[:] = np.frombuffer(texture.read(), dtype="f4").reshape(mirror.shape)
    if "visible_illumination" in authoritative:
        texture = textures.get("visible_illumination")
        if texture is not None:
            rgba = np.frombuffer(texture.read(), dtype="f4").reshape((height, width, 4))
            world.visible_illumination[:] = rgba[..., :3]
    if "active_tile_ttl" in authoritative:
        buffer = buffers.get("active_tile_ttl")
        active = world.active
        if buffer is not None and active.tile_width > 0 and active.tile_height > 0:
            ttl = np.frombuffer(
                buffer.read(
                    size=active.tile_width * active.tile_height * np.dtype(np.int32).itemsize
                ),
                dtype=np.int32,
            ).reshape((active.tile_height, active.tile_width))
            active.active_tile_ttl = ttl.tolist()
    if "active_chunk_mask" in authoritative:
        buffer = buffers.get("active_chunk_mask")
        active = world.active
        if buffer is not None and active.chunk_width > 0 and active.chunk_height > 0:
            mask = np.frombuffer(
                buffer.read(
                    size=active.chunk_width * active.chunk_height * np.dtype(np.int32).itemsize
                ),
                dtype=np.int32,
            ).reshape((active.chunk_height, active.chunk_width))
            active.active_chunk_mask = (mask != 0).tolist()


def sync_readback_requests(bridge, world: "WorldEngine") -> None:
    readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
    bridge.shadow_buffers["readback_request"] = readback_request_upload.copy()
    bridge.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
    bridge._write_dynamic_buffer("readback_request", readback_request_upload)
    bridge._write_dynamic_buffer("readback_request_label", readback_request_label_upload)


def sync_world_commands(bridge, world: "WorldEngine") -> None:
    world_command_upload, world_command_payload_upload = pack_world_command_upload(world)
    bridge.shadow_buffers["world_command"] = world_command_upload.copy()
    bridge.shadow_buffers["world_command_payload"] = world_command_payload_upload.copy()
    # GL writes are only valid on the thread that owns the context (a
    # moderngl standalone context is pinned to the thread that created it);
    # shadow buffers above remain the cross-thread publication channel.
    if bridge.owner_thread_id is not None and bridge.owner_thread_id != threading.get_ident():
        return
    bridge._write_dynamic_buffer("world_command", world_command_upload)
    bridge._write_dynamic_buffer("world_command_payload", world_command_payload_upload)


def sync_force_sources(bridge, world: "WorldEngine") -> None:
    force_source_upload = pack_force_source_upload(world)
    force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
    bridge.shadow_buffers["force_source"] = force_source_upload.copy()
    bridge.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
    # GL writes are only valid on the thread that owns the context; shadow
    # buffers above remain the cross-thread publication channel.
    if bridge.owner_thread_id is not None and bridge.owner_thread_id != threading.get_ident():
        return
    bridge._write_dynamic_buffer("force_source", force_source_upload)
    bridge._write_dynamic_buffer("force_source_count", force_source_count_upload)


def _write_typed_table_buffer(bridge, name: str, data: np.ndarray) -> None:
    if not bridge.enabled or bridge.ctx is None:
        return
    buffer = bridge.typed_table_buffers.get(name)
    nbytes = max(4, data.nbytes)
    if buffer is None or buffer.size < nbytes:
        if buffer is not None:
            buffer.release()
        buffer = bridge.ctx.buffer(reserve=nbytes, dynamic=True)
        bridge.typed_table_buffers[name] = buffer
    else:
        buffer.orphan(nbytes)
    if data.nbytes > 0:
        buffer.write(np.ascontiguousarray(data).tobytes())


def _write_dynamic_buffer(bridge, name: str, data: np.ndarray) -> None:
    if not bridge.enabled or bridge.ctx is None:
        return
    buffer = bridge.buffers.get(name)
    nbytes = max(4, data.nbytes)
    if buffer is None:
        buffer = bridge.ctx.buffer(reserve=nbytes, dynamic=True)
        bridge.buffers[name] = buffer
    elif buffer.size < nbytes:
        buffer.release()
        buffer = bridge.ctx.buffer(reserve=nbytes, dynamic=True)
        bridge.buffers[name] = buffer
    else:
        buffer.orphan(nbytes)
    if data.nbytes > 0:
        buffer.write(np.ascontiguousarray(data).tobytes())


def _shadow_or_default(bridge, name: str, default: np.ndarray) -> np.ndarray:
    existing = bridge.shadow_buffers.get(name)
    if (
        isinstance(existing, np.ndarray)
        and existing.shape == default.shape
        and existing.dtype == default.dtype
    ):
        return existing
    return default
