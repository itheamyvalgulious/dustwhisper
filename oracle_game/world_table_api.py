from __future__ import annotations

from typing import Any, TYPE_CHECKING

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np

from oracle_game.rules import build_default_optics_entries
from oracle_game.sim.gpu_collapse_dirty import clear_collapse_structure_dirty_tile_mask
from oracle_game.types import (
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    ReactionAction,
    ReactionType,
)
from oracle_game.world_constants import REACTION_RULE_SET_NAMES

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def update_material_table(engine, materials: list[MaterialDef | dict[str, Any]], *, immediate: bool = True) -> None:
    materials = [engine._coerce_material_def(material) for material in materials]
    if not immediate:
        engine.queue_command("update_material_table", materials=[asdict(material) for material in materials])
        return
    merged_payload = engine._merged_material_table_payload(materials)
    engine._validate_material_table_payload(merged_payload)
    engine.rulebook.materials_by_name.clear()
    engine.rulebook.materials_by_id.clear()
    engine.rulebook.update_materials(engine._coerce_material_def(item) for item in merged_payload)
    engine.rulebook.optics.clear()
    engine.rulebook.update_optics(
        build_default_optics_entries(
            engine.rulebook.materials_by_id.values(),
            engine.rulebook.lights_by_id.values(),
            existing=engine._material_optics_snapshot_map(),
        )
    )
    engine.tag_bits_by_name = deepcopy(engine.rulebook.tag_bits)
    engine._rebuild_material_property_arrays()
    optics_payload = engine._material_optics_table_snapshot_payload()
    engine._set_stable_shadow_payload("materials", merged_payload)
    engine._set_stable_shadow_payload("optics", optics_payload)
    engine.bridge.upload_table("materials", merged_payload)
    engine.bridge.upload_table("optics", optics_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("update_material_table")
    engine.bridge.ensure_world_resources(engine)


def update_gas_species_table(engine, gases: list[GasSpeciesDef | dict[str, Any]], *, immediate: bool = True) -> None:
    gases = [engine._coerce_gas_species_def(gas) for gas in gases]
    if not immediate:
        engine.queue_command("update_gas_species_table", gases=[asdict(gas) for gas in gases])
        return
    merged_payload = engine._merged_gas_species_table_payload(gases)
    engine._validate_gas_species_payload(merged_payload)
    engine.rulebook.gases_by_name.clear()
    engine.rulebook.gases_by_id.clear()
    engine.rulebook.update_gases(engine._coerce_gas_species_def(item) for item in merged_payload)
    engine._rebuild_gas_property_arrays()
    previous = engine.gas_concentration
    gas_count = engine._gas_field_count()
    engine.gas_concentration = np.zeros((gas_count, engine.gas_height, engine.gas_width), dtype=np.float32)
    count = min(previous.shape[0], engine.gas_concentration.shape[0])
    engine.gas_concentration[:count] = previous[:count]
    if 0 <= engine.air_gas_species_id < engine.gas_concentration.shape[0]:
        engine.gas_concentration[engine.air_gas_species_id] = np.maximum(
            engine.gas_concentration[engine.air_gas_species_id], 1.0
        )
    engine._set_stable_shadow_payload("gases", merged_payload)
    engine.bridge.upload_table("gases", merged_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("update_gas_species_table")
    engine.bridge.ensure_world_resources(engine)


def update_light_type_table(engine, lights: list[LightTypeDef | dict[str, Any]], *, immediate: bool = True) -> None:
    lights = [engine._coerce_light_type_def(light) for light in lights]
    if not immediate:
        engine.queue_command("update_light_type_table", lights=[asdict(light) for light in lights])
        return
    merged_payload = engine._merged_light_type_table_payload(lights)
    engine._validate_light_type_payload(merged_payload)
    engine.rulebook.lights_by_name.clear()
    engine.rulebook.lights_by_id.clear()
    engine.rulebook.update_lights(engine._coerce_light_type_def(item) for item in merged_payload)
    engine.rulebook.optics.clear()
    engine.rulebook.update_optics(
        build_default_optics_entries(
            engine.rulebook.materials_by_id.values(),
            engine.rulebook.lights_by_id.values(),
            existing=engine._material_optics_snapshot_map(),
        )
    )
    engine._rebuild_light_property_arrays()
    previous_cell_dose = engine.cell_optical_dose
    previous_gas_dose = engine.gas_optical_dose
    light_count = engine._light_field_count()
    engine.cell_optical_dose = np.zeros((light_count, engine.height, engine.width), dtype=np.float32)
    engine.gas_optical_dose = np.zeros((light_count, engine.gas_height, engine.gas_width), dtype=np.float32)
    cell_count = min(previous_cell_dose.shape[0], engine.cell_optical_dose.shape[0])
    gas_count = min(previous_gas_dose.shape[0], engine.gas_optical_dose.shape[0])
    engine.cell_optical_dose[:cell_count] = previous_cell_dose[:cell_count]
    engine.gas_optical_dose[:gas_count] = previous_gas_dose[:gas_count]
    optics_payload = engine._material_optics_table_snapshot_payload()
    engine._set_stable_shadow_payload("lights", merged_payload)
    engine._set_stable_shadow_payload("optics", optics_payload)
    engine.bridge.upload_table("lights", merged_payload)
    engine.bridge.upload_table("optics", optics_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("update_light_type_table")
    engine.bridge.ensure_world_resources(engine)


def update_material_optics_table(engine, optics: list[MaterialOpticsDef | dict[str, Any]], *, immediate: bool = True) -> None:
    optics = [engine._coerce_material_optics_def(entry) for entry in optics]
    if not immediate:
        engine.queue_command("update_material_optics_table", optics=[asdict(entry) for entry in optics])
        return
    merged_payload = engine._merged_material_optics_table_payload(optics)
    engine._validate_material_optics_payload(merged_payload)
    engine.rulebook.optics.clear()
    engine.rulebook.update_optics(engine._coerce_material_optics_def(item) for item in merged_payload)
    engine._set_stable_shadow_payload("optics", merged_payload)
    engine.bridge.upload_table("optics", merged_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("update_material_optics_table")


def update_reaction_table(
    engine,
    actions: list[ReactionAction | dict[str, Any]],
    rules: dict[str, object],
    *,
    immediate: bool = True,
) -> None:
    actions = [engine._coerce_reaction_action(action) for action in actions]
    rules = engine._coerce_reaction_rules(rules)
    if not immediate:
        engine.queue_command(
            "update_reaction_table",
            actions=[asdict(action) for action in actions],
            rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
        )
        return
    merged_payload = engine._merged_reaction_table_payload(actions, rules)
    engine._validate_reaction_payload(merged_payload)
    engine.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + [
        engine._coerce_reaction_action(action) for action in merged_payload["actions"]
    ]
    merged_rules = merged_payload["rules"]
    engine.rulebook.material_material_rules = [
        engine._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_material"]
    ]
    engine.rulebook.material_gas_rules = [
        engine._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_gas"]
    ]
    engine.rulebook.material_light_rules = [
        engine._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_light"]
    ]
    engine.rulebook.gas_gas_rules = [
        engine._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_gas"]
    ]
    engine.rulebook.gas_light_rules = [
        engine._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_light"]
    ]
    engine.rulebook.self_rules = [
        engine._coerce_self_reaction_rule(rule) for rule in merged_rules["self_rules"]
    ]
    engine._set_stable_shadow_payload("reactions", merged_payload)
    engine.bridge.upload_table("reactions", merged_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("update_reaction_table")


def replace_reaction_table(
    engine,
    actions: list[ReactionAction | dict[str, Any]],
    rules: dict[str, object],
    *,
    immediate: bool = True,
) -> None:
    actions = [engine._coerce_reaction_action(action) for action in actions]
    rules = engine._coerce_reaction_rules(rules)
    if not immediate:
        engine.queue_command(
            "replace_reaction_table",
            actions=[asdict(action) for action in actions],
            rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
        )
        return
    replacement_payload = {
        "actions": [asdict(action) for action in actions],
        "rules": {name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
    }
    engine._validate_reaction_payload(replacement_payload)
    materials_payload = engine._shadow_material_payload()
    engine._clamp_material_payload_reaction_slots(
        materials_payload,
        action_count=len(replacement_payload["actions"]) + 1,
    )
    engine.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + list(actions)
    engine.rulebook.material_material_rules = list(rules["material_material"])
    engine.rulebook.material_gas_rules = list(rules["material_gas"])
    engine.rulebook.material_light_rules = list(rules["material_light"])
    engine.rulebook.gas_gas_rules = list(rules["gas_gas"])
    engine.rulebook.gas_light_rules = list(rules["gas_light"])
    engine.rulebook.self_rules = list(rules["self_rules"])
    engine.rulebook.materials_by_name.clear()
    engine.rulebook.materials_by_id.clear()
    engine.rulebook.update_materials(engine._coerce_material_def(item) for item in materials_payload)
    engine.tag_bits_by_name = deepcopy(engine.rulebook.tag_bits)
    engine._rebuild_material_property_arrays()
    reaction_payload = engine._reaction_table_snapshot_payload()
    engine._set_stable_shadow_payload("materials", materials_payload)
    engine._set_stable_shadow_payload("reactions", reaction_payload)
    engine.bridge.upload_table("materials", materials_payload)
    engine.bridge.upload_table("reactions", reaction_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bootstrap_log.append("replace_reaction_table")


def reset_world(engine, *, immediate: bool = True) -> None:
    if not immediate:
        engine.queue_command("reset_world")
        return
    _reset_world_state(engine, reset_bridge_frame_inputs=True, keep_command_log=False)


def _reset_world_state(
    engine,
    *,
    reset_bridge_frame_inputs: bool,
    keep_command_log: bool,
) -> None:
    engine.material_id.fill(0)
    engine.phase.fill(0)
    engine.cell_flags.fill(0)
    engine.velocity.fill(0.0)
    engine.cell_temperature.fill(20.0)
    engine.timer_pack.fill(0)
    engine.integrity.fill(0.0)
    engine.island_id.fill(0)
    engine.entity_id.fill(0)
    engine.placeholder_displaced_material.fill(0)
    engine.collapse_delay_pending.fill(False)
    engine.flow_velocity.fill(0.0)
    engine.ambient_temperature.fill(20.0)
    engine.pressure_ping.fill(0.0)
    engine.gas_concentration.fill(0.0)
    if 0 <= engine.air_gas_species_id < engine.gas_concentration.shape[0]:
        engine.gas_concentration[engine.air_gas_species_id] = 1.0
    engine.visible_illumination.fill(0.0)
    engine.cell_optical_dose.fill(0.0)
    engine.gas_optical_dose.fill(0.0)
    engine.force_sources.clear()
    engine.persistent_emitters.clear()
    engine.emitters.clear()
    engine.collapse_dirty_regions.clear()
    engine.collapse_deferred_regions.clear()
    clear_collapse_structure_dirty_tile_mask(engine)
    engine.islands.clear()
    engine.entity_states.clear()
    engine.entity_placeholders.clear()
    engine.pending_frame_inputs.clear()
    engine.completed_frame_outputs.clear()
    engine.canceled_frame_submission_ids.clear()
    engine.next_frame_submission_id = 1
    engine.next_readback_request_id = 1
    engine.pending_readbacks.clear()
    engine.inflight_readbacks.clear()
    engine.completed_readbacks.clear()
    engine.canceled_readback_request_ids.clear()
    engine.last_entity_observation_consume_snapshot = {
        "frame_id": int(engine.frame_id),
        "consumed": 0,
        "consumed_readbacks": [],
        "observations": {},
        "entity_feedback": {},
    }
    engine.controller_state_snapshot = None
    engine.gas_solver.reset_runtime_state(engine)
    engine.heat_solver.reset_runtime_state(engine)
    engine.liquid_solver.reset_runtime_state(engine)
    engine.reaction_solver.reset_runtime_state(engine)
    engine.collapse_solver.reset_runtime_state(engine)
    engine.optics_solver.reset_runtime_state(engine)
    engine.motion_solver.reset_runtime_state()
    if reset_bridge_frame_inputs:
        engine._clear_bridge_frame_inputs(keep_commands=keep_command_log, prepared=False)
    engine.page_store.clear()
    engine.next_island_id = 1
    engine._build_demo_scene()


def patch_material(engine, name: str, *, immediate: bool = True, **fields: Any) -> None:
    if not immediate:
        engine.queue_command(
            "patch_material",
            name=str(engine._canonical_material_input_name(name)),
            fields=engine._normalize_material_patch_fields(fields),
        )
        return
    material_id = engine._resolve_sanctioned_material_id(name)
    if material_id <= 0:
        raise KeyError(name)
    material = engine._shadow_material_def(material_id)
    if material is None:
        raise KeyError(name)
    patch_fields = dict(fields)
    patch_fields.setdefault("name", engine._shadow_material_name(material_id))
    patch_fields.setdefault("material_id", int(material_id))
    updated = engine._coerce_material_def(asdict(replace(material, **patch_fields)))
    update_material_table(engine, [updated])


def patch_light(engine, name: str, *, immediate: bool = True, **fields: Any) -> None:
    if not immediate:
        engine.queue_command("patch_light", name=name, fields=fields)
        return
    light_id = engine._resolve_sanctioned_light_id(name)
    if light_id < 0:
        raise KeyError(name)
    light = engine._shadow_light_type_def(light_id)
    if light is None:
        raise KeyError(name)
    patch_fields = dict(fields)
    patch_fields.setdefault("name", engine._shadow_light_name(light_id))
    patch_fields.setdefault("light_type_id", int(light_id))
    updated = engine._coerce_light_type_def(asdict(replace(light, **patch_fields)))
    update_light_type_table(engine, [updated])


def patch_gas(engine, name: str, *, immediate: bool = True, **fields: Any) -> None:
    if not immediate:
        engine.queue_command(
            "patch_gas",
            name=name,
            fields=engine._normalize_gas_patch_fields(fields),
        )
        return
    species_id = engine._resolve_sanctioned_gas_id(name)
    if species_id < 0:
        raise KeyError(name)
    gas = engine._shadow_gas_species_def(species_id)
    if gas is None:
        raise KeyError(name)
    patch_fields = dict(fields)
    patch_fields.setdefault("name", engine._shadow_gas_name(species_id))
    patch_fields.setdefault("species_id", int(species_id))
    updated = engine._coerce_gas_species_def(asdict(replace(gas, **patch_fields)))
    update_gas_species_table(engine, [updated])


def patch_material_optics(
    engine,
    material_name: str,
    light_type: str,
    *,
    immediate: bool = True,
    **fields: Any,
) -> None:
    if not immediate:
        engine.queue_command(
            "patch_material_optics",
            material_name=str(engine._canonical_material_input_name(material_name)),
            light_type=light_type,
            fields=engine._normalize_material_optics_patch_fields(fields),
        )
        return
    material_id = engine._resolve_sanctioned_material_id(material_name)
    if material_id <= 0:
        raise KeyError(material_name)
    light_id = engine._resolve_sanctioned_light_id(light_type)
    if light_id < 0:
        raise KeyError(light_type)
    canonical_material_name = engine._shadow_material_name(material_id)
    canonical_light_type = engine._shadow_light_name(light_id)
    if canonical_material_name is None or canonical_light_type is None:
        raise KeyError((material_name, light_type))
    optics = engine._shadow_material_optics_def(canonical_material_name, canonical_light_type)
    if optics is None:
        raise KeyError((material_name, light_type))
    patch_fields = dict(fields)
    patch_fields.setdefault("material_name", canonical_material_name)
    patch_fields.setdefault("light_type", canonical_light_type)
    updated = engine._coerce_material_optics_def(asdict(replace(optics, **patch_fields)))
    update_material_optics_table(engine, [updated])


def patch_reaction_action(engine, index: int, *, immediate: bool = True, **fields: Any) -> None:
    if not immediate:
        engine.queue_command(
            "patch_reaction_action",
            index=index,
            fields=engine._normalize_reaction_action_patch_fields(fields),
        )
        return
    if index <= 0:
        raise ValueError("reaction action 0 is reserved")
    if index >= len(engine.rulebook.reaction_actions):
        raise IndexError(index)
    action = engine._shadow_reaction_action(index)
    if action is None:
        raise IndexError(index)
    updated = engine._coerce_reaction_action(asdict(replace(action, **fields)))
    reactions_payload = engine._shadow_reaction_payload()
    if index == 0:
        engine.rulebook.reaction_actions = [updated] + [
            engine._coerce_reaction_action(item) for item in reactions_payload["actions"]
        ]
        engine._validate_reaction_payload(reactions_payload)
        engine._set_stable_shadow_payload("reactions", reactions_payload)
        engine.bridge.upload_table("reactions", reactions_payload)
        engine.bridge.sync_rule_tables(engine)
        return
    reactions_payload["actions"][index - 1] = asdict(updated)
    engine._validate_reaction_payload(reactions_payload)
    engine.rulebook.reaction_actions = [engine.rulebook.reaction_actions[0]] + [
        engine._coerce_reaction_action(item) for item in reactions_payload["actions"]
    ]
    engine._set_stable_shadow_payload("reactions", reactions_payload)
    engine.bridge.upload_table("reactions", reactions_payload)
    engine.bridge.sync_rule_tables(engine)


def patch_reaction_rule(
    engine,
    rule_set: str,
    index: int,
    *,
    immediate: bool = True,
    **fields: Any,
) -> None:
    rule_set = str(rule_set)
    if rule_set not in REACTION_RULE_SET_NAMES:
        raise KeyError(rule_set)
    if not immediate:
        engine.queue_command(
            "patch_reaction_rule",
            rule_set=rule_set,
            index=index,
            fields=engine._normalize_reaction_rule_patch_fields(fields),
        )
        return
    rule = engine._shadow_reaction_rule(rule_set, index)
    if rule is None:
        raise IndexError(index)
    if rule_set == "self_rules":
        updated = engine._coerce_self_reaction_rule(asdict(replace(rule, **fields)))
    else:
        updated = engine._coerce_pair_reaction_rule(asdict(replace(rule, **fields)))
    reactions_payload = engine._shadow_reaction_payload()
    reactions_payload["rules"][rule_set][index] = asdict(updated)
    engine._validate_reaction_payload(reactions_payload)
    engine._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
    engine._set_stable_shadow_payload("reactions", reactions_payload)
    engine.bridge.upload_table("reactions", reactions_payload)
    engine.bridge.sync_rule_tables(engine)


def delete_reaction_action(engine, index: int, *, immediate: bool = True) -> None:
    if not immediate:
        engine.queue_command("delete_reaction_action", index=index)
        return
    if index <= 0:
        raise ValueError("reaction action 0 is reserved")
    if index >= len(engine.rulebook.reaction_actions):
        raise IndexError(index)
    reactions_payload = engine._shadow_reaction_payload()
    actions_payload = reactions_payload["actions"]
    if index > len(actions_payload):
        raise IndexError(index)
    materials_payload = engine._shadow_material_payload()
    del actions_payload[index - 1]
    engine._remap_reaction_payload_result_actions(reactions_payload["rules"], deleted_action_index=index)
    engine._remap_material_payload_reaction_slots(materials_payload, deleted_action_index=index)
    engine.rulebook.reaction_actions = [engine.rulebook.reaction_actions[0]] + [
        engine._coerce_reaction_action(item) for item in actions_payload
    ]
    engine._set_reaction_rules_payload(reactions_payload["rules"])
    engine.rulebook.materials_by_name.clear()
    engine.rulebook.materials_by_id.clear()
    engine.rulebook.update_materials(engine._coerce_material_def(item) for item in materials_payload)
    engine.tag_bits_by_name = deepcopy(engine.rulebook.tag_bits)
    engine._rebuild_material_property_arrays()
    engine._set_stable_shadow_payload("materials", materials_payload)
    engine._set_stable_shadow_payload("reactions", reactions_payload)
    engine.bridge.upload_table("materials", materials_payload)
    engine.bridge.upload_table("reactions", reactions_payload)
    engine.bridge.sync_rule_tables(engine)
    engine.bridge.ensure_world_resources(engine)


def delete_reaction_rule(engine, rule_set: str, index: int, *, immediate: bool = True) -> None:
    rule_set = str(rule_set)
    if rule_set not in REACTION_RULE_SET_NAMES:
        raise KeyError(rule_set)
    if not immediate:
        engine.queue_command("delete_reaction_rule", rule_set=rule_set, index=index)
        return
    rules_list = engine._reaction_rule_list(rule_set)
    if index < 0 or index >= len(rules_list):
        raise IndexError(index)
    reactions_payload = engine._shadow_reaction_payload()
    del reactions_payload["rules"][rule_set][index]
    engine._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
    engine._set_stable_shadow_payload("reactions", reactions_payload)
    engine.bridge.upload_table("reactions", reactions_payload)
    engine.bridge.sync_rule_tables(engine)
