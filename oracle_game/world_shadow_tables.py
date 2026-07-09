from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.gpu import typed_gas_id, typed_light_id, typed_material_id
from oracle_game.types import (
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    PairReactionRule,
    ReactionAction,
    ReactionType,
    SelfReactionRule,
)
from oracle_game.world_constants import BASE_MATERIAL_RUNTIME_ALIASES, PAIR_REACTION_RULE_SET_NAMES

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _resolve_sanctioned_material_id(engine: "WorldEngine", name: str) -> int:
    engine.bridge.sync_rule_tables(engine)
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is None:
        return 0
    canonical_names = [str(name)]
    alias = BASE_MATERIAL_RUNTIME_ALIASES.get(str(name))
    if alias is not None and alias != name:
        canonical_names.append(alias)
    for candidate_name in canonical_names:
        material_id = int(typed_material_id(material_table, candidate_name))
        if material_id <= 0 or material_id >= int(material_table.shape[0]):
            continue
        if int(material_table[material_id]["name_hash"]) == 0:
            continue
        return material_id
    return 0


def _shadow_material_id_by_name(engine: "WorldEngine", name: str | None) -> int:
    canonical_name = engine._canonical_material_input_name(name)
    if canonical_name is None:
        return 0
    materials_payload = engine._shadow_material_payload()
    if materials_payload is not None:
        for item in materials_payload:
            if engine._canonical_material_input_name(item.get("name")) != canonical_name:
                continue
            return int(item.get("material_id", 0))
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is None:
        return 0
    return int(typed_material_id(material_table, canonical_name))


def _resolve_sanctioned_placeholder_material_id(engine: "WorldEngine", name: str) -> int:
    material_id = _resolve_sanctioned_material_id(engine, name)
    if not _shadow_material_is_placeholder(engine, material_id):
        return 0
    return material_id


def _resolve_sanctioned_light_id(engine: "WorldEngine", name: str) -> int:
    engine.bridge.sync_rule_tables(engine)
    light_table = engine.bridge.shadow_typed_tables.get("light_table")
    if light_table is None:
        return -1
    light_id = int(typed_light_id(light_table, name))
    if light_id < 0 or light_id >= int(light_table.shape[0]):
        return -1
    if int(light_table[light_id]["name_hash"]) == 0:
        return -1
    return light_id


def _resolve_sanctioned_gas_id(engine: "WorldEngine", name: str) -> int:
    engine.bridge.sync_rule_tables(engine)
    gas_table = engine.bridge.shadow_typed_tables.get("gas_table")
    if gas_table is None:
        return -1
    species_id = int(typed_gas_id(gas_table, name))
    if species_id < 0 or species_id >= int(gas_table.shape[0]):
        return -1
    if int(gas_table[species_id]["name_hash"]) == 0:
        return -1
    return species_id


def _shadow_material_row_valid(engine: "WorldEngine", material_id: int) -> bool:
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is None:
        return True
    if material_id <= 0 or material_id >= int(material_table.shape[0]):
        return False
    return int(material_table[int(material_id)]["name_hash"]) != 0


def _shadow_gas_row_valid(engine: "WorldEngine", species_id: int) -> bool:
    gas_table = engine.bridge.shadow_typed_tables.get("gas_table")
    if gas_table is None:
        return True
    if species_id < 0 or species_id >= int(gas_table.shape[0]):
        return False
    return int(gas_table[int(species_id)]["name_hash"]) != 0


def _shadow_light_row_valid(engine: "WorldEngine", light_id: int) -> bool:
    light_table = engine.bridge.shadow_typed_tables.get("light_table")
    if light_table is None:
        return True
    if light_id < 0 or light_id >= int(light_table.shape[0]):
        return False
    return int(light_table[int(light_id)]["name_hash"]) != 0


def _shadow_material_def(engine: "WorldEngine", material_id: int) -> MaterialDef | None:
    if not _shadow_material_row_valid(engine, int(material_id)):
        return None
    for item in engine._shadow_material_payload():
        if int(item.get("material_id", 0)) == int(material_id):
            return engine._coerce_material_def(item)
    return None


def _shadow_light_type_def(engine: "WorldEngine", light_id: int) -> LightTypeDef | None:
    if not _shadow_light_row_valid(engine, int(light_id)):
        return None
    for item in engine._shadow_light_type_payload():
        if int(item.get("light_type_id", -1)) == int(light_id):
            return engine._coerce_light_type_def(item)
    return None


def _shadow_gas_species_def(engine: "WorldEngine", species_id: int) -> GasSpeciesDef | None:
    if not _shadow_gas_row_valid(engine, int(species_id)):
        return None
    for item in engine._shadow_gas_species_payload():
        if int(item.get("species_id", -1)) == int(species_id):
            return engine._coerce_gas_species_def(item)
    return None


def _shadow_material_optics_def(engine: "WorldEngine", material_name: str, light_type: str) -> MaterialOpticsDef | None:
    payload = engine.bridge.shadow_tables.get("optics")
    if payload is not None:
        for item in payload:
            if str(item.get("material_name", "")) == material_name and str(item.get("light_type", "")) == light_type:
                return engine._coerce_material_optics_def(item)
        return None
    optics_table = engine.bridge.shadow_typed_tables.get("optics_table")
    material_id = _resolve_sanctioned_material_id(engine, material_name)
    light_id = _resolve_sanctioned_light_id(engine, light_type)
    if material_id <=0 or light_id <0:
        return None
    if optics_table is not None:
        for row in optics_table:
            if int(row["material_id"]) == material_id and int(row["light_type_id"]) == light_id:
                return MaterialOpticsDef(
                material_name=material_name,
                light_type=light_type,
                absorption=float(row["absorption"]),
                scattering=float(row["scattering"]),
                refraction=float(row["refraction"]),
                )
    return None


def _shadow_material_name(engine: "WorldEngine", material_id: int) -> str | None:
    material = _shadow_material_def(engine, int(material_id))
    if material is not None and material.name:
        return str(material.name)
    if not _shadow_material_row_valid(engine, int(material_id)):
        return None
    if engine._shadow_has_table_payload("materials"):
        return None
    if 0 <= int(material_id) < len(engine.material_name_by_id) and engine.material_name_by_id[int(material_id)]:
        return engine.material_name_by_id[int(material_id)]
    return None


def _shadow_gas_name(engine: "WorldEngine", species_id: int) -> str | None:
    gas = _shadow_gas_species_def(engine, int(species_id))
    if gas is not None and gas.name:
        return str(gas.name)
    if not _shadow_gas_row_valid(engine, int(species_id)):
        return None
    if engine._shadow_has_table_payload("gases"):
        return None

    return engine.gas_name_by_id[int(species_id)]
    return None


def _shadow_light_name(engine: "WorldEngine", light_id: int) -> str | None:
    light = _shadow_light_type_def(engine, int(light_id))
    if light is not None and light.name:
        return str(light.name)
    if not _shadow_light_row_valid(engine, int(light_id)):
        return None
    if engine._shadow_has_table_payload("lights"):
        return None
    if 0 <= int(light_id) < len(engine.light_name_by_id) and engine.light_name_by_id[int(light_id)]:
        return engine.light_name_by_id[int(light_id)]
    return None


def _shadow_light_default_range(engine: "WorldEngine", light_id: int) -> int | None:
    light_table = engine.bridge.shadow_typed_tables.get("light_table")
    if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
        row = light_table[int(light_id)]
        if int(row["name_hash"]) == 0:
            return None
        return int(row["default_range"])
    light = _shadow_light_type_def(engine, int(light_id))
    if light is not None:
        return int(light.default_range)
    if engine._shadow_has_table_payload("lights"):
        return None
    if 0 <= int(light_id) < engine.light_default_range.shape[0]:
        return int(engine.light_default_range[int(light_id)])
    return None


def _shadow_light_dose_channel(engine: "WorldEngine", light_id: int) -> int | None:
    light_table = engine.bridge.shadow_typed_tables.get("light_table")
    if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
        row = light_table[int(light_id)]
        if int(row["name_hash"]) == 0:
            return None
        return int(row["dose_channel_id"])
    light = _shadow_light_type_def(engine, int(light_id))
    if light is not None:
        return int(light.dose_channel_id)
    if engine._shadow_has_table_payload("lights"):
        return None
    if 0 <= int(light_id) < engine.light_dose_channel.shape[0]:
        return int(engine.light_dose_channel[int(light_id)])
    return None


def _shadow_light_color(engine: "WorldEngine", light_id: int) -> np.ndarray | None:
    light_table = engine.bridge.shadow_typed_tables.get("light_table")
    if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
        row = light_table[int(light_id)]
        if int(row["name_hash"]) == 0:
            return None
        return np.asarray(row["color"], dtype=np.float32)
    light = _shadow_light_type_def(engine, int(light_id))
    if light is not None:
        return np.asarray(light.color, dtype=np.float32)
    if engine._shadow_has_table_payload("lights"):
        return None
    if 0 <= int(light_id) < engine.light_color.shape[0]:
        return np.asarray(engine.light_color[int(light_id)], dtype=np.float32)
    return None


def _shadow_light_name_and_range(engine: "WorldEngine", light_id: int) -> tuple[str, int] | None:
    light_name = _shadow_light_name(engine, int(light_id))
    if light_name is None:
        return None
    default_range = _shadow_light_default_range(engine, int(light_id))
    if default_range is None:
        return None
    return (light_name, default_range)


def _shadow_material_default_phase(engine: "WorldEngine", material_id: int) -> int | None:
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
        row = material_table[int(material_id)]
        if int(row["name_hash"]) == 0:
            return None
        return int(row["default_phase"])
    shadow_material = _shadow_material_def(engine, int(material_id))
    if shadow_material is not None:
        return int(shadow_material.default_phase)
    if engine._shadow_has_table_payload("materials"):
        return None
    if 0 <= int(material_id) < engine.material_default_phase.shape[0]:
        return int(engine.material_default_phase[int(material_id)])
    return None


def _shadow_material_base_integrity(engine: "WorldEngine", material_id: int) -> float | None:
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
        row = material_table[int(material_id)]
        if int(row["name_hash"]) == 0:
            return None
        return float(row["base_integrity"])
    shadow_material = _shadow_material_def(engine, int(material_id))
    if shadow_material is not None:
        return float(shadow_material.base_integrity)
    if engine._shadow_has_table_payload("materials"):
        return None
    if 0 <= int(material_id) < engine.material_base_integrity.shape[0]:
        return float(engine.material_base_integrity[int(material_id)])
    return None


def _shadow_material_spawn_temperature(engine: "WorldEngine", material_id: int) -> float | None:
    material_table = engine.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
        row = material_table[int(material_id)]
        if int(row["name_hash"]) == 0:
            return None
        value = float(row["spawn_temperature"])
        return None if np.isnan(value) else value
    shadow_material = _shadow_material_def(engine, int(material_id))
    if shadow_material is not None and shadow_material.spawn_temperature is not None:
        return float(shadow_material.spawn_temperature)
    if engine._shadow_has_table_payload("materials"):
        return None
    if 0 <= int(material_id) < engine.material_spawn_temperature.shape[0]:
        value = float(engine.material_spawn_temperature[int(material_id)])
        return None if np.isnan(value) else value
    return None


def _shadow_condense_target_material_id(engine: "WorldEngine", species_id: int) -> int:
    gas_table = engine.bridge.shadow_typed_tables.get("gas_table")
    if gas_table is not None and 0 <= int(species_id) < int(gas_table.shape[0]):
        row = gas_table[int(species_id)]
        if int(row["name_hash"]) != 0:
            return int(row["condense_to_material_id"])
    gas = _shadow_gas_species_def(engine, int(species_id))
    if gas is None or gas.condense_to_material is None:
        return 0
    return _shadow_material_id_by_name(engine, gas.condense_to_material)


def _shadow_material_is_placeholder(engine: "WorldEngine", material_id: int) -> bool:
    shadow_material = _shadow_material_def(engine, int(material_id))
    if shadow_material is not None:
        return shadow_material.render_group == "placeholder" or "placeholder" in shadow_material.tags
    if engine.bridge.shadow_typed_tables.get("material_table") is not None:
        return False
    if engine._shadow_has_table_payload("materials"):
        return False
    if 0 <= int(material_id) < engine.material_is_placeholder.shape[0]:
        return bool(engine.material_is_placeholder[int(material_id)])
    return False


def _shadow_material_is_plant(engine: "WorldEngine", material_id: int) -> bool:
    shadow_material = _shadow_material_def(engine, int(material_id))
    if shadow_material is not None:
        return shadow_material.render_group == "plant" or "plant" in shadow_material.tags
    if engine.bridge.shadow_typed_tables.get("material_table") is not None:
        return False
    if engine._shadow_has_table_payload("materials"):
        return False
    if 0 <= int(material_id) < engine.material_is_plant.shape[0]:
        return bool(engine.material_is_plant[int(material_id)])
    return False


def _shadow_reaction_action(engine: "WorldEngine", index: int) -> ReactionAction | None:
    if index ==0:
        return engine.rulebook.reaction_actions[0] if engine.rulebook.reaction_actions else ReactionAction(ReactionType.NONE)
    payload = engine._shadow_reaction_payload()
    actions = payload.get("actions", [])
    if index >0 and index <= len(actions):
        return engine._coerce_reaction_action(actions[index -1])
    if engine._shadow_has_table_payload("reactions"):
        return None
    if index >=0 and index < len(engine.rulebook.reaction_actions):
        return engine.rulebook.reaction_actions[index]
    return None


def _reaction_rule_list(engine: "WorldEngine", rule_set: str) -> list[PairReactionRule] | list[SelfReactionRule]:
    normalized = str(rule_set)
    payload = engine._shadow_reaction_payload()
    if payload is not None:
        rules_payload = payload.get("rules", {})
        entries = list(rules_payload.get(normalized, []))
        if normalized == "self_rules":
            return [engine._coerce_self_reaction_rule(entry) for entry in entries]
        if normalized in PAIR_REACTION_RULE_SET_NAMES:
            return [engine._coerce_pair_reaction_rule(entry) for entry in entries]
    if engine._shadow_has_table_payload("reactions"):
        return []
    if normalized == "material_material":
        return engine.rulebook.material_material_rules
    if normalized == "material_gas":
        return engine.rulebook.material_gas_rules
    if normalized == "material_light":
        return engine.rulebook.material_light_rules
    if normalized == "gas_gas":
        return engine.rulebook.gas_gas_rules
    if normalized == "gas_light":
        return engine.rulebook.gas_light_rules
    if normalized == "self_rules":
        return engine.rulebook.self_rules
    raise KeyError(rule_set)


def _shadow_reaction_rule(engine: "WorldEngine", rule_set: str, index: int) -> PairReactionRule | SelfReactionRule | None:
    payload = engine._shadow_reaction_payload()
    normalized = str(rule_set)
    rules = payload.get("rules", {})
    entries = rules.get(normalized, [])
    if (0 <= index) and index < len(entries):
        if normalized == "self_rules":
            return engine._coerce_self_reaction_rule(entries[index])
        return engine._coerce_pair_reaction_rule(entries[index])
    if engine._shadow_has_table_payload("reactions"):
        return None
    if (0 <= index) and index < len(rules_list):
        return rules_list[index]
    return None
