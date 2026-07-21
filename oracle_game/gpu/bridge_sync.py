from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

import json
import numpy as np

from oracle_game.gpu._common import (
    _json_bytes,
)

from oracle_game.gpu.dtypes import (
    GAS_RUNTIME_META_DTYPE,
    REACTION_RUNTIME_META_DTYPE,
    HEAT_RUNTIME_META_DTYPE,
    GAS_SPECIES_RUNTIME_DTYPE,
    LIQUID_RUNTIME_META_DTYPE,
    COLLAPSE_COMPONENT_DTYPE,
    ISLAND_RUNTIME_DTYPE,
    RULE_TABLE_META_DTYPE,
    OPTICS_RUNTIME_META_DTYPE,
    COLLAPSE_RUNTIME_META_DTYPE,
)

from oracle_game.gpu.packers import (
    pack_self_reaction_rule_table,
    pack_reaction_action_table,
    pack_gas_table,
    pack_entity_state_upload,
    pack_cell_core,
    pack_liquid_runtime_upload,
    pack_world_command_upload,
    pack_force_source_upload,
    pack_light_table,
    pack_heat_runtime_upload,
    pack_reaction_runtime_upload,
    pack_optics_table,
    pack_island_runtime_upload,
    pack_gas_runtime_upload,
    _pack_pair_reaction_rules,
    pack_readback_request_upload,
    pack_optics_runtime_upload,
    pack_frame_meta_upload,
    pack_placeholder_dirty_rect_upload,
    pack_collapse_runtime_upload,
    pack_material_table,
    pack_page_stripe_upload,
    pack_active_meta_upload,
    pack_placeholder_upload,
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
    if signature == bridge.rule_table_signature and bridge.shadow_typed_tables and ((not bridge.enabled or bridge.ctx is None) or buffers_ready):
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
        bridge._write_typed_table_buffer("material_material_rule_table", material_material_rule_table)
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
        bridge._sync_world_impl(world, debug_frame=debug_frame, upload_debug_texture=upload_debug_texture)
    finally:
        bridge._force_cpu_resource_upload = previous_force_cpu_resource_upload


def _sync_world_impl(
    bridge,
    world: "WorldEngine",
    *,
    debug_frame: np.ndarray | None = None,
    upload_debug_texture: bool = True,
) -> None:
    bridge.ensure_world_resources(world)
    bridge.sync_rule_tables(world)
    upload_solver_runtime_from_cpu = bridge._should_upload_cpu_solver_runtime(world)
    upload_island_runtime_from_cpu = bridge._should_upload_cpu_resource(world, "island_runtime")
    upload_powder_reservation_from_cpu = (
        upload_solver_runtime_from_cpu and bridge._should_upload_cpu_resource(world, "powder_reservation")
    )
    upload_island_reservation_from_cpu = (
        upload_solver_runtime_from_cpu and bridge._should_upload_cpu_resource(world, "island_reservation")
    )
    if upload_powder_reservation_from_cpu:
        world.motion_solver.gpu_pipeline.materialize_compact_powder_reservations(
            world,
            download=True,
        )
    entity_state_upload = pack_entity_state_upload(world)
    entity_count_upload = np.array([len(entity_state_upload)], dtype=np.int32)
    force_source_upload = pack_force_source_upload(world)
    force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
    island_runtime_upload = (
        pack_island_runtime_upload(world)
        if upload_island_runtime_from_cpu
        else bridge.shadow_buffers.get("island_runtime", np.zeros((0,), dtype=ISLAND_RUNTIME_DTYPE))
    )
    island_runtime_count_upload = (
        np.array([len(island_runtime_upload)], dtype=np.int32)
        if upload_island_runtime_from_cpu
        else bridge.shadow_buffers.get("island_runtime_count", np.zeros((1,), dtype=np.int32))
    )
    motion_runtime = (
        world.motion_solver.runtime_snapshot()
        if upload_powder_reservation_from_cpu or upload_island_reservation_from_cpu
        else None
    )
    powder_reservation_upload = (
        motion_runtime["powder_reservations"]
        if upload_powder_reservation_from_cpu and motion_runtime is not None
        else bridge.shadow_buffers.get(
            "powder_reservation",
            np.zeros((0,), dtype=getattr(world.motion_solver, "last_powder_reservations").dtype),
        )
    )
    powder_reservation_count_upload = (
        np.array([len(powder_reservation_upload)], dtype=np.int32)
        if upload_powder_reservation_from_cpu
        else bridge.shadow_buffers.get("powder_reservation_count", np.zeros((1,), dtype=np.int32))
    )
    island_reservation_upload = (
        motion_runtime["island_reservations"]
        if upload_island_reservation_from_cpu and motion_runtime is not None
        else bridge.shadow_buffers.get(
            "island_reservation",
            np.zeros((0,), dtype=getattr(world.motion_solver, "last_island_reservations").dtype),
        )
    )
    island_reservation_count_upload = (
        np.array([len(island_reservation_upload)], dtype=np.int32)
        if upload_island_reservation_from_cpu
        else bridge.shadow_buffers.get("island_reservation_count", np.zeros((1,), dtype=np.int32))
    )
    world_command_upload, world_command_payload_upload = pack_world_command_upload(world)
    readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
    placeholder_upload = pack_placeholder_upload(world)
    placeholder_dirty_rect_upload = pack_placeholder_dirty_rect_upload(world)
    upload_active_tile_ttl_from_cpu = bridge._should_upload_cpu_resource(world, "active_tile_ttl")
    upload_active_chunk_mask_from_cpu = bridge._should_upload_cpu_resource(world, "active_chunk_mask")
    upload_active_meta_from_cpu = bridge._should_upload_cpu_resource(world, "active_meta")
    active_tile_ttl_default = np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.int32)
    active_chunk_mask_default = np.zeros((world.active.chunk_height, world.active.chunk_width), dtype=np.uint8)
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
        else bridge._shadow_or_default("active_meta", active_meta_default)
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
        gas_runtime_meta_upload = bridge._shadow_or_default("gas_runtime_meta", np.zeros((1,), dtype=GAS_RUNTIME_META_DTYPE))
        gas_solve_tile_mask_upload = bridge._shadow_or_default(
            "gas_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        gas_solve_gas_mask_upload = bridge._shadow_or_default(
            "gas_solve_gas_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        gas_species_runtime_upload = bridge._shadow_or_default(
            "gas_species_runtime",
            np.zeros((world.gas_concentration.shape[0],), dtype=GAS_SPECIES_RUNTIME_DTYPE),
        )
        heat_runtime_meta_upload = bridge._shadow_or_default("heat_runtime_meta", np.zeros((1,), dtype=HEAT_RUNTIME_META_DTYPE))
        heat_solve_tile_mask_upload = bridge._shadow_or_default(
            "heat_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        heat_solve_cell_mask_upload = bridge._shadow_or_default(
            "heat_solve_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        heat_solve_gas_mask_upload = bridge._shadow_or_default(
            "heat_solve_gas_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        heat_phase_target_upload = bridge._shadow_or_default(
            "heat_phase_target",
            np.zeros((world.height, world.width), dtype=np.int32),
        )
        heat_boil_target_upload = bridge._shadow_or_default(
            "heat_boil_target",
            np.zeros((world.height, world.width), dtype=np.int32),
        )
        heat_condense_target_upload = bridge._shadow_or_default(
            "heat_condense_target",
            np.zeros(world.gas_concentration.shape, dtype=np.uint8),
        )
        liquid_runtime_meta_upload = bridge._shadow_or_default("liquid_runtime_meta", np.zeros((1,), dtype=LIQUID_RUNTIME_META_DTYPE))
        liquid_solve_tile_mask_upload = bridge._shadow_or_default(
            "liquid_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        liquid_post_tile_mask_upload = bridge._shadow_or_default(
            "liquid_post_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        liquid_post_cell_mask_upload = bridge._shadow_or_default(
            "liquid_post_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        liquid_vertical_seam_mask_upload = bridge._shadow_or_default(
            "liquid_vertical_seam_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        liquid_horizontal_seam_mask_upload = bridge._shadow_or_default(
            "liquid_horizontal_seam_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        liquid_buoyancy_mask_upload = bridge._shadow_or_default(
            "liquid_buoyancy_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        liquid_changed_cell_mask_upload = bridge._shadow_or_default(
            "liquid_changed_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        reaction_runtime_meta_upload = bridge._shadow_or_default("reaction_runtime_meta", np.zeros((1,), dtype=REACTION_RUNTIME_META_DTYPE))
        reaction_timed_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_timed_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_self_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_self_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_material_material_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_material_material_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_material_gas_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_material_gas_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_material_light_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_material_light_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_gas_gas_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_gas_gas_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_gas_light_solve_tile_mask_upload = bridge._shadow_or_default(
            "reaction_gas_light_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        reaction_solve_cell_mask_upload = bridge._shadow_or_default(
            "reaction_solve_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        reaction_solve_gas_mask_upload = bridge._shadow_or_default(
            "reaction_solve_gas_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        reaction_changed_cell_mask_upload = bridge._shadow_or_default(
            "reaction_changed_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        reaction_changed_gas_mask_upload = bridge._shadow_or_default(
            "reaction_changed_gas_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        reaction_ambient_changed_mask_upload = bridge._shadow_or_default(
            "reaction_ambient_changed_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        reaction_timer_changed_mask_upload = bridge._shadow_or_default(
            "reaction_timer_changed_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        reaction_emitted_light_mask_upload = bridge._shadow_or_default(
            "reaction_emitted_light_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        reaction_emitted_material_mask_upload = bridge._shadow_or_default(
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
        any(name in bridge.gpu_authoritative_resources for name in collapse_mask_resources)
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
        collapse_runtime_meta_upload = bridge.shadow_buffers.get(
            "collapse_runtime_meta",
            np.zeros((1,), dtype=COLLAPSE_RUNTIME_META_DTYPE),
        )
        collapse_solve_region_mask_upload = bridge.shadow_buffers.get("collapse_solve_region_mask", cell_zero)
        collapse_structural_mask_upload = bridge.shadow_buffers.get("collapse_structural_mask", cell_zero)
        collapse_support_seed_mask_upload = bridge.shadow_buffers.get("collapse_support_seed_mask", cell_zero)
        collapse_supported_mask_upload = bridge.shadow_buffers.get("collapse_supported_mask", cell_zero)
        collapse_unsupported_mask_upload = bridge.shadow_buffers.get("collapse_unsupported_mask", cell_zero)
        collapse_delayed_pending_mask_upload = bridge.shadow_buffers.get("collapse_delayed_pending_mask", cell_zero)
        collapse_immune_unsupported_mask_upload = bridge.shadow_buffers.get("collapse_immune_unsupported_mask", cell_zero)
        collapse_collapsed_cell_mask_upload = bridge.shadow_buffers.get("collapse_collapsed_cell_mask", cell_zero)
        collapse_component_upload = bridge.shadow_buffers.get(
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
        optics_runtime_meta_upload = bridge._shadow_or_default("optics_runtime_meta", np.zeros((1,), dtype=OPTICS_RUNTIME_META_DTYPE))
        optics_solve_tile_mask_upload = bridge._shadow_or_default(
            "optics_solve_tile_mask",
            np.zeros((world.active.tile_height, world.active.tile_width), dtype=np.uint8),
        )
        optics_solve_cell_mask_upload = bridge._shadow_or_default(
            "optics_solve_cell_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        optics_solve_gas_mask_upload = bridge._shadow_or_default(
            "optics_solve_gas_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        optics_visible_changed_mask_upload = bridge._shadow_or_default(
            "optics_visible_changed_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        optics_cell_dose_changed_mask_upload = bridge._shadow_or_default(
            "optics_cell_dose_changed_mask",
            np.zeros((world.height, world.width), dtype=np.uint8),
        )
        optics_gas_dose_changed_mask_upload = bridge._shadow_or_default(
            "optics_gas_dose_changed_mask",
            np.zeros((world.gas_height, world.gas_width), dtype=np.uint8),
        )
        optics_emitter_origin_mask_upload = bridge._shadow_or_default(
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
    bridge.shadow_buffers["entity_state"] = entity_state_upload.copy()
    bridge.shadow_buffers["entity_state_count"] = entity_count_upload.copy()
    bridge.shadow_buffers["force_source"] = force_source_upload.copy()
    bridge.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
    if upload_island_runtime_from_cpu or "island_runtime" not in bridge.shadow_buffers:
        bridge.shadow_buffers["island_runtime"] = island_runtime_upload.copy()
        bridge.shadow_buffers["island_runtime_count"] = island_runtime_count_upload.copy()
    if upload_powder_reservation_from_cpu or "powder_reservation" not in bridge.shadow_buffers:
        bridge.shadow_buffers["powder_reservation"] = powder_reservation_upload.copy()
        bridge.shadow_buffers["powder_reservation_count"] = powder_reservation_count_upload.copy()
    if upload_island_reservation_from_cpu or "island_reservation" not in bridge.shadow_buffers:
        bridge.shadow_buffers["island_reservation"] = island_reservation_upload.copy()
        bridge.shadow_buffers["island_reservation_count"] = island_reservation_count_upload.copy()
    bridge.shadow_buffers["world_command"] = world_command_upload.copy()
    bridge.shadow_buffers["world_command_payload"] = world_command_payload_upload.copy()
    bridge.shadow_buffers["readback_request"] = readback_request_upload.copy()
    bridge.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
    bridge.shadow_buffers["placeholder"] = placeholder_upload.copy()
    bridge.shadow_buffers["placeholder_dirty_rect"] = placeholder_dirty_rect_upload.copy()
    bridge.shadow_buffers["island_id"] = world.island_id.astype(np.int32).copy()
    bridge.shadow_buffers["entity_id"] = world.entity_id.astype(np.int32).copy()
    bridge.shadow_buffers["placeholder_displaced_material"] = world.placeholder_displaced_material.astype(np.int32).copy()
    bridge.shadow_buffers["collapse_delay_pending"] = world.collapse_delay_pending.astype(np.int32).copy()
    bridge.shadow_buffers["cell_optical_dose"] = world.cell_optical_dose.astype(np.float32).copy()
    bridge.shadow_buffers["gas_optical_dose"] = world.gas_optical_dose.astype(np.float32).copy()
    bridge.shadow_buffers["active_meta"] = active_meta_upload.copy()
    bridge.shadow_buffers["active_tile_ttl"] = active_tile_ttl_upload.copy()
    bridge.shadow_buffers["active_chunk_mask"] = active_chunk_mask_upload.copy()
    bridge.shadow_buffers["gas_runtime_meta"] = gas_runtime_meta_upload.copy()
    bridge.shadow_buffers["gas_solve_tile_mask"] = gas_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["gas_solve_gas_mask"] = gas_solve_gas_mask_upload.copy()
    bridge.shadow_buffers["gas_species_runtime"] = gas_species_runtime_upload.copy()
    bridge.shadow_buffers["heat_runtime_meta"] = heat_runtime_meta_upload.copy()
    bridge.shadow_buffers["heat_solve_tile_mask"] = heat_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["heat_solve_cell_mask"] = heat_solve_cell_mask_upload.copy()
    bridge.shadow_buffers["heat_solve_gas_mask"] = heat_solve_gas_mask_upload.copy()
    bridge.shadow_buffers["heat_phase_target"] = heat_phase_target_upload.copy()
    bridge.shadow_buffers["heat_boil_target"] = heat_boil_target_upload.copy()
    bridge.shadow_buffers["heat_condense_target"] = heat_condense_target_upload.copy()
    bridge.shadow_buffers["liquid_runtime_meta"] = liquid_runtime_meta_upload.copy()
    bridge.shadow_buffers["liquid_solve_tile_mask"] = liquid_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["liquid_post_tile_mask"] = liquid_post_tile_mask_upload.copy()
    bridge.shadow_buffers["liquid_post_cell_mask"] = liquid_post_cell_mask_upload.copy()
    bridge.shadow_buffers["liquid_vertical_seam_mask"] = liquid_vertical_seam_mask_upload.copy()
    bridge.shadow_buffers["liquid_horizontal_seam_mask"] = liquid_horizontal_seam_mask_upload.copy()
    bridge.shadow_buffers["liquid_buoyancy_mask"] = liquid_buoyancy_mask_upload.copy()
    bridge.shadow_buffers["liquid_changed_cell_mask"] = liquid_changed_cell_mask_upload.copy()
    bridge.shadow_buffers["reaction_runtime_meta"] = reaction_runtime_meta_upload.copy()
    bridge.shadow_buffers["reaction_timed_solve_tile_mask"] = reaction_timed_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_self_solve_tile_mask"] = reaction_self_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_material_material_solve_tile_mask"] = reaction_material_material_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_material_gas_solve_tile_mask"] = reaction_material_gas_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_material_light_solve_tile_mask"] = reaction_material_light_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_gas_gas_solve_tile_mask"] = reaction_gas_gas_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_gas_light_solve_tile_mask"] = reaction_gas_light_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["reaction_solve_cell_mask"] = reaction_solve_cell_mask_upload.copy()
    bridge.shadow_buffers["reaction_solve_gas_mask"] = reaction_solve_gas_mask_upload.copy()
    bridge.shadow_buffers["reaction_changed_cell_mask"] = reaction_changed_cell_mask_upload.copy()
    bridge.shadow_buffers["reaction_changed_gas_mask"] = reaction_changed_gas_mask_upload.copy()
    bridge.shadow_buffers["reaction_ambient_changed_mask"] = reaction_ambient_changed_mask_upload.copy()
    bridge.shadow_buffers["reaction_timer_changed_mask"] = reaction_timer_changed_mask_upload.copy()
    bridge.shadow_buffers["reaction_emitted_light_mask"] = reaction_emitted_light_mask_upload.copy()
    bridge.shadow_buffers["reaction_emitted_material_mask"] = reaction_emitted_material_mask_upload.copy()
    bridge.shadow_buffers["collapse_runtime_meta"] = collapse_runtime_meta_upload.copy()
    bridge.shadow_buffers["collapse_solve_region_mask"] = collapse_solve_region_mask_upload.copy()
    bridge.shadow_buffers["collapse_structural_mask"] = collapse_structural_mask_upload.copy()
    bridge.shadow_buffers["collapse_support_seed_mask"] = collapse_support_seed_mask_upload.copy()
    bridge.shadow_buffers["collapse_supported_mask"] = collapse_supported_mask_upload.copy()
    bridge.shadow_buffers["collapse_unsupported_mask"] = collapse_unsupported_mask_upload.copy()
    bridge.shadow_buffers["collapse_delayed_pending_mask"] = collapse_delayed_pending_mask_upload.copy()
    bridge.shadow_buffers["collapse_immune_unsupported_mask"] = collapse_immune_unsupported_mask_upload.copy()
    bridge.shadow_buffers["collapse_collapsed_cell_mask"] = collapse_collapsed_cell_mask_upload.copy()
    bridge.shadow_buffers["collapse_component"] = collapse_component_upload.copy()
    bridge.shadow_buffers["optics_runtime_meta"] = optics_runtime_meta_upload.copy()
    bridge.shadow_buffers["optics_solve_tile_mask"] = optics_solve_tile_mask_upload.copy()
    bridge.shadow_buffers["optics_solve_cell_mask"] = optics_solve_cell_mask_upload.copy()
    bridge.shadow_buffers["optics_solve_gas_mask"] = optics_solve_gas_mask_upload.copy()
    bridge.shadow_buffers["optics_visible_changed_mask"] = optics_visible_changed_mask_upload.copy()
    bridge.shadow_buffers["optics_cell_dose_changed_mask"] = optics_cell_dose_changed_mask_upload.copy()
    bridge.shadow_buffers["optics_gas_dose_changed_mask"] = optics_gas_dose_changed_mask_upload.copy()
    bridge.shadow_buffers["optics_emitter_origin_mask"] = optics_emitter_origin_mask_upload.copy()
    bridge.shadow_buffers["page_stripe_meta"] = page_stripe_meta_upload.copy()
    bridge.shadow_buffers["page_stripe_section"] = page_stripe_section_upload.copy()
    bridge.shadow_buffers["page_stripe_payload"] = page_stripe_payload_upload.copy()
    bridge.shadow_buffers["frame_meta"] = frame_meta_upload.copy()
    if not bridge.enabled or bridge.ctx is None:
        return
    bridge._ensure_atlas_texture(world)
    upload_cell_dose_from_cpu = bridge._should_upload_cpu_resource(world, "cell_optical_dose")
    upload_gas_dose_from_cpu = bridge._should_upload_cpu_resource(world, "gas_optical_dose")
    upload_light_from_cpu = bridge._should_upload_cpu_resource(world, "light")
    upload_visible_from_cpu = bridge._should_upload_cpu_resource(world, "visible_illumination")
    if upload_cell_dose_from_cpu or upload_gas_dose_from_cpu or upload_light_from_cpu or upload_visible_from_cpu:
        world._gpu_optics_outputs_clear = False
        optics_pipeline = getattr(getattr(world, "optics_solver", None), "gpu_pipeline", None)
        invalidate_sparse = getattr(optics_pipeline, "invalidate_sparse_runtime", None)
        if callable(invalidate_sparse):
            invalidate_sparse()
    if bridge._should_upload_cpu_resource(world, "cell_core"):
        packed = pack_cell_core(world)
        bridge.buffers["cell_core"].write(packed.tobytes())
    if bridge._should_upload_cpu_resource(world, "island_id"):
        bridge.buffers["island_id"].write(np.ascontiguousarray(world.island_id.astype(np.int32)).tobytes())
    if bridge._should_upload_cpu_resource(world, "entity_id"):
        bridge.buffers["entity_id"].write(np.ascontiguousarray(world.entity_id.astype(np.int32)).tobytes())
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
        bridge.buffers["cell_optical_dose"].write(np.ascontiguousarray(world.cell_optical_dose.astype(np.float32)).tobytes())
    if upload_gas_dose_from_cpu:
        bridge.buffers["gas_optical_dose"].write(np.ascontiguousarray(world.gas_optical_dose.astype(np.float32)).tobytes())
    bridge._write_dynamic_buffer("entity_state", entity_state_upload)
    bridge.buffers["entity_state_count"].write(entity_count_upload.tobytes())
    bridge._write_dynamic_buffer("force_source", force_source_upload)
    bridge.buffers["force_source_count"].write(force_source_count_upload.tobytes())
    if upload_island_runtime_from_cpu:
        bridge._write_dynamic_buffer("island_runtime", island_runtime_upload)
        bridge.buffers["island_runtime_count"].write(island_runtime_count_upload.tobytes())
    if upload_powder_reservation_from_cpu:
        bridge._write_dynamic_buffer("powder_reservation", powder_reservation_upload)
        bridge.buffers["powder_reservation_count"].write(powder_reservation_count_upload.tobytes())
        bridge.clear_gpu_authoritative(
            "powder_reservation",
            "powder_reservation_compact",
            "powder_reservation_standard",
            "powder_reservation_cpu_mirror",
        )
    if upload_island_reservation_from_cpu:
        bridge._write_dynamic_buffer("island_reservation", island_reservation_upload)
        bridge.buffers["island_reservation_count"].write(island_reservation_count_upload.tobytes())
    bridge._write_dynamic_buffer("world_command", world_command_upload)
    bridge._write_dynamic_buffer("world_command_payload", world_command_payload_upload)
    bridge._write_dynamic_buffer("readback_request", readback_request_upload)
    bridge._write_dynamic_buffer("readback_request_label", readback_request_label_upload)
    bridge._write_dynamic_buffer("placeholder", placeholder_upload)
    bridge._write_dynamic_buffer("placeholder_dirty_rect", placeholder_dirty_rect_upload)
    if bridge._should_upload_cpu_resource(world, "active_meta"):
        bridge._write_dynamic_buffer("active_meta", active_meta_upload)
    if bridge._should_upload_cpu_resource(world, "active_tile_ttl"):
        bridge._write_dynamic_buffer("active_tile_ttl", active_tile_ttl_upload)
    if bridge._should_upload_cpu_resource(world, "active_chunk_mask"):
        bridge._write_dynamic_buffer("active_chunk_mask", active_chunk_mask_upload.astype(np.int32, copy=False))
    if (
        getattr(world, "simulation_backend", "") == "gpu"
        and (
            upload_active_meta_from_cpu
            or upload_active_tile_ttl_from_cpu
            or upload_active_chunk_mask_from_cpu
        )
    ):
        bridge._ensure_active_scheduler_programs()
        bridge._refresh_active_chunks_and_meta(world, read_meta=False)
    bridge._write_dynamic_buffer("gas_runtime_meta", gas_runtime_meta_upload)
    bridge._write_dynamic_buffer("gas_solve_tile_mask", gas_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("gas_solve_gas_mask", gas_solve_gas_mask_upload)
    bridge._write_dynamic_buffer("gas_species_runtime", gas_species_runtime_upload)
    bridge._write_dynamic_buffer("heat_runtime_meta", heat_runtime_meta_upload)
    bridge._write_dynamic_buffer("heat_solve_tile_mask", heat_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("heat_solve_cell_mask", heat_solve_cell_mask_upload)
    bridge._write_dynamic_buffer("heat_solve_gas_mask", heat_solve_gas_mask_upload)
    bridge._write_dynamic_buffer("heat_phase_target", heat_phase_target_upload)
    bridge._write_dynamic_buffer("heat_boil_target", heat_boil_target_upload)
    bridge._write_dynamic_buffer("heat_condense_target", heat_condense_target_upload)
    bridge._write_dynamic_buffer("liquid_runtime_meta", liquid_runtime_meta_upload)
    bridge._write_dynamic_buffer("liquid_solve_tile_mask", liquid_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("liquid_post_tile_mask", liquid_post_tile_mask_upload)
    bridge._write_dynamic_buffer("liquid_post_cell_mask", liquid_post_cell_mask_upload)
    bridge._write_dynamic_buffer("liquid_vertical_seam_mask", liquid_vertical_seam_mask_upload)
    bridge._write_dynamic_buffer("liquid_horizontal_seam_mask", liquid_horizontal_seam_mask_upload)
    bridge._write_dynamic_buffer("liquid_buoyancy_mask", liquid_buoyancy_mask_upload)
    bridge._write_dynamic_buffer("liquid_changed_cell_mask", liquid_changed_cell_mask_upload)
    bridge._write_dynamic_buffer("reaction_runtime_meta", reaction_runtime_meta_upload)
    bridge._write_dynamic_buffer("reaction_timed_solve_tile_mask", reaction_timed_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_self_solve_tile_mask", reaction_self_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_material_material_solve_tile_mask", reaction_material_material_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_material_gas_solve_tile_mask", reaction_material_gas_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_material_light_solve_tile_mask", reaction_material_light_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_gas_gas_solve_tile_mask", reaction_gas_gas_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_gas_light_solve_tile_mask", reaction_gas_light_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("reaction_solve_cell_mask", reaction_solve_cell_mask_upload)
    bridge._write_dynamic_buffer("reaction_solve_gas_mask", reaction_solve_gas_mask_upload)
    bridge._write_dynamic_buffer("reaction_changed_cell_mask", reaction_changed_cell_mask_upload)
    bridge._write_dynamic_buffer("reaction_changed_gas_mask", reaction_changed_gas_mask_upload)
    bridge._write_dynamic_buffer("reaction_ambient_changed_mask", reaction_ambient_changed_mask_upload)
    bridge._write_dynamic_buffer("reaction_timer_changed_mask", reaction_timer_changed_mask_upload)
    bridge._write_dynamic_buffer("reaction_emitted_light_mask", reaction_emitted_light_mask_upload)
    bridge._write_dynamic_buffer("reaction_emitted_material_mask", reaction_emitted_material_mask_upload)
    bridge._write_dynamic_buffer("collapse_runtime_meta", collapse_runtime_meta_upload)
    bridge._write_dynamic_buffer("collapse_solve_region_mask", collapse_solve_region_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_structural_mask"):
        bridge._write_dynamic_buffer("collapse_structural_mask", collapse_structural_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_support_seed_mask"):
        bridge._write_dynamic_buffer("collapse_support_seed_mask", collapse_support_seed_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_supported_mask"):
        bridge._write_dynamic_buffer("collapse_supported_mask", collapse_supported_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_unsupported_mask"):
        bridge._write_dynamic_buffer("collapse_unsupported_mask", collapse_unsupported_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_delayed_pending_mask"):
        bridge._write_dynamic_buffer("collapse_delayed_pending_mask", collapse_delayed_pending_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_immune_unsupported_mask"):
        bridge._write_dynamic_buffer("collapse_immune_unsupported_mask", collapse_immune_unsupported_mask_upload)
    if bridge._should_upload_cpu_resource(world, "collapse_collapsed_cell_mask"):
        bridge._write_dynamic_buffer("collapse_collapsed_cell_mask", collapse_collapsed_cell_mask_upload)
    bridge._write_dynamic_buffer("collapse_component", collapse_component_upload)
    bridge._write_dynamic_buffer("optics_runtime_meta", optics_runtime_meta_upload)
    bridge._write_dynamic_buffer("optics_solve_tile_mask", optics_solve_tile_mask_upload)
    bridge._write_dynamic_buffer("optics_solve_cell_mask", optics_solve_cell_mask_upload)
    bridge._write_dynamic_buffer("optics_solve_gas_mask", optics_solve_gas_mask_upload)
    bridge._write_dynamic_buffer("optics_visible_changed_mask", optics_visible_changed_mask_upload)
    bridge._write_dynamic_buffer("optics_cell_dose_changed_mask", optics_cell_dose_changed_mask_upload)
    bridge._write_dynamic_buffer("optics_gas_dose_changed_mask", optics_gas_dose_changed_mask_upload)
    bridge._write_dynamic_buffer("optics_emitter_origin_mask", optics_emitter_origin_mask_upload)
    bridge._write_dynamic_buffer("page_stripe_meta", page_stripe_meta_upload)
    bridge._write_dynamic_buffer("page_stripe_section", page_stripe_section_upload)
    bridge._write_dynamic_buffer("page_stripe_payload", page_stripe_payload_upload)
    bridge.buffers["frame_meta"].write(frame_meta_upload.tobytes())
    if bridge._should_upload_cpu_resource(world, "material"):
        bridge.textures["material"].write(world.material_id.astype("f4").tobytes())
    if upload_light_from_cpu:
        light_rgba = np.empty((world.height, world.width, 4), dtype=np.float32)
        light_rgba[..., :3] = np.clip(world.visible_illumination, 0.0, 4.0)
        light_rgba[..., 3] = 1.0
        bridge.textures["light"].write(light_rgba.tobytes())
    if upload_visible_from_cpu:
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
        bridge.textures["ambient_temperature"].write(world.ambient_temperature.astype("f4").tobytes())
    if bridge._should_upload_cpu_resource(world, "pressure_ping"):
        bridge.textures["pressure_ping"].write(world.pressure_ping.astype("f4").tobytes())
    if bridge._should_upload_cpu_resource(world, "flow_velocity"):
        bridge.textures["flow_velocity"].write(world.flow_velocity.astype("f4").tobytes())
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


def sync_readback_requests(bridge, world: "WorldEngine") -> None:
    readback_request_upload, readback_request_label_upload = pack_readback_request_upload(world)
    bridge.shadow_buffers["readback_request"] = readback_request_upload.copy()
    bridge.shadow_buffers["readback_request_label"] = readback_request_label_upload.copy()
    bridge._write_dynamic_buffer("readback_request", readback_request_upload)
    bridge._write_dynamic_buffer("readback_request_label", readback_request_label_upload)


def sync_force_sources(bridge, world: "WorldEngine") -> None:
    force_source_upload = pack_force_source_upload(world)
    force_source_count_upload = np.array([len(force_source_upload)], dtype=np.int32)
    bridge.shadow_buffers["force_source"] = force_source_upload.copy()
    bridge.shadow_buffers["force_source_count"] = force_source_count_upload.copy()
    bridge._write_dynamic_buffer("force_source", force_source_upload)
    if bridge.enabled and bridge.ctx is not None:
        bridge.buffers["force_source_count"].write(force_source_count_upload.tobytes())


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
    if isinstance(existing, np.ndarray) and existing.shape == default.shape and existing.dtype == default.dtype:
        return existing
    return default
