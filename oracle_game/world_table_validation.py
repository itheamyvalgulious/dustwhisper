from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np

from oracle_game.types import (
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    PairReactionRule,
    Phase,
    ReactionAction,
    ReactionType,
    SelfReactionRule,
)
from oracle_game.world_constants import (
    PAIR_REACTION_RULE_SET_NAMES,
    REACTION_RULE_SET_NAMES,
)

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _material_table_snapshot_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return [asdict(material) for _, material in sorted(engine.rulebook.materials_by_id.items())]


def _gas_species_table_snapshot_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return [asdict(gas) for _, gas in sorted(engine.rulebook.gases_by_id.items())]


def _light_type_table_snapshot_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return [asdict(light) for _, light in sorted(engine.rulebook.lights_by_id.items())]


def _material_optics_table_snapshot_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return [asdict(entry) for _, entry in sorted(engine.rulebook.optics.items())]


def _reaction_table_snapshot_payload(engine: "WorldEngine") -> dict[str, object]:
    return {
        "actions": [asdict(action) for action in engine.rulebook.reaction_actions[1:]],
        "rules": {
            "material_material": [asdict(rule) for rule in engine.rulebook.material_material_rules],
            "material_gas": [asdict(rule) for rule in engine.rulebook.material_gas_rules],
            "material_light": [asdict(rule) for rule in engine.rulebook.material_light_rules],
            "gas_gas": [asdict(rule) for rule in engine.rulebook.gas_gas_rules],
            "gas_light": [asdict(rule) for rule in engine.rulebook.gas_light_rules],
            "self_rules": [asdict(rule) for rule in engine.rulebook.self_rules],
        },
    }


def _stable_shadow_payload(engine: "WorldEngine", name: str, snapshot_factory: Callable[[], Any]) -> Any:
    payload = engine.bridge.shadow_tables.get(str(name))
    if payload is not None:
        stable = deepcopy(payload)
        engine._stable_shadow_payloads[str(name)] = stable
        return deepcopy(stable)
    cached = engine._stable_shadow_payloads.get(str(name))
    if cached is not None:
        return deepcopy(cached)
    stable = deepcopy(snapshot_factory())
    engine._stable_shadow_payloads[str(name)] = stable
    return deepcopy(stable)


def _set_stable_shadow_payload(engine: "WorldEngine", name: str, payload: Any) -> None:
    engine._stable_shadow_payloads[str(name)] = deepcopy(payload)


def _shadow_has_table_payload(engine: "WorldEngine", name: str) -> bool:
    if engine.bridge.shadow_tables.get(str(name)) is not None:
     return True
    return engine._stable_shadow_payloads.get(str(name)) is not None


def _merged_reaction_table_payload(
    engine: "WorldEngine",
    actions: list[ReactionAction],
    rules: dict[str, list[object]],
) -> dict[str, object]:
    base_payload = _shadow_reaction_payload(engine)
    merged_actions = list(base_payload.get("actions", []))
    merged_actions.extend(asdict(action) for action in actions)
    merged_rules = {
        "material_material": list(base_payload.get("rules", {}).get("material_material", [])),
        "material_gas": list(base_payload.get("rules", {}).get("material_gas", [])),
        "material_light": list(base_payload.get("rules", {}).get("material_light", [])),
        "gas_gas": list(base_payload.get("rules", {}).get("gas_gas", [])),
        "gas_light": list(base_payload.get("rules", {}).get("gas_light", [])),
        "self_rules": list(base_payload.get("rules", {}).get("self_rules", [])),
    }
    for name, entries in rules.items():
        merged_rules[name].extend(asdict(rule) for rule in entries)
    return {
        "actions": merged_actions,
        "rules": merged_rules,
    }


def _merged_material_table_payload(engine: "WorldEngine", materials: list[MaterialDef]) -> list[dict[str, Any]]:
    base_payload = _shadow_material_payload(engine)
    merged = {int(item["material_id"]): dict(item) for item in base_payload}
    for material in materials:
        merged[int(material.material_id)] = asdict(material)
    return [merged[material_id] for material_id in sorted(merged)]


def _merged_gas_species_table_payload(engine: "WorldEngine", gases: list[GasSpeciesDef]) -> list[dict[str, Any]]:
    base_payload = _shadow_gas_species_payload(engine)
    merged = {int(item["species_id"]): dict(item) for item in base_payload}
    for gas in gases:
        merged[int(gas.species_id)] = asdict(gas)
    return [merged[species_id] for species_id in sorted(merged)]


def _merged_light_type_table_payload(engine: "WorldEngine", lights: list[LightTypeDef]) -> list[dict[str, Any]]:
    base_payload = _shadow_light_type_payload(engine)
    merged = {int(item["light_type_id"]): dict(item) for item in base_payload}
    for light in lights:
        merged[int(light.light_type_id)] = asdict(light)
    return [merged[light_id] for light_id in sorted(merged)]


def _merged_material_optics_table_payload(engine: "WorldEngine", optics: list[MaterialOpticsDef]) -> list[dict[str, Any]]:
    base_payload = _stable_shadow_payload(engine, "optics", engine._material_optics_table_snapshot_payload)
    merged = {(str(item["material_name"]), str(item["light_type"])): dict(item) for item in base_payload}
    for entry in optics:
        merged[(str(entry.material_name), str(entry.light_type))] = asdict(entry)
    return [merged[key] for key in sorted(merged)]


def _shadow_material_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return _stable_shadow_payload(engine, "materials", engine._material_table_snapshot_payload)


def _shadow_gas_species_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return _stable_shadow_payload(engine, "gases", engine._gas_species_table_snapshot_payload)


def _shadow_light_type_payload(engine: "WorldEngine") -> list[dict[str, Any]]:
    return _stable_shadow_payload(engine, "lights", engine._light_type_table_snapshot_payload)


def _shadow_reaction_payload(engine: "WorldEngine") -> dict[str, Any]:
    return _stable_shadow_payload(engine, "reactions", engine._reaction_table_snapshot_payload)


def _payload_name_set(payload: Iterable[dict[str, Any]], field: str = "name") -> set[str]:
    names: set[str] = set()
    for item in payload:
        name = item.get(field)
        if name:
            names.add(str(name))
    return names


def _validate_named_reference(valid_names: set[str], reference: str | None) -> None:
    if reference is None:
        return
    if str(reference) not in valid_names:
        raise KeyError(reference)


def _validate_unique_identity_fields(
    payload: Iterable[dict[str, Any]],
    *,
    id_field: str,
    name_field: str = "name",
    allow_zero_id: bool,
) -> None:
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in payload:
        item_id = int(item[id_field])
        item_name = str(item[name_field])
        if not allow_zero_id and item_id == 0:
            raise ValueError(f"{id_field}=0 is reserved")
        if item_id in seen_ids:
            raise ValueError(f"duplicate {id_field}: {item_id}")
        if item_name in seen_names:
            raise ValueError(f"duplicate {name_field}: {item_name}")
        seen_ids.add(item_id)
        seen_names.add(item_name)


def _validate_material_table_payload(engine: "WorldEngine", materials_payload: list[dict[str, Any]]) -> None:
    _validate_unique_identity_fields(
        materials_payload,
        id_field="material_id",
        allow_zero_id=False,
    )
    material_names = _payload_name_set(materials_payload)
    gas_names = _payload_name_set(_shadow_gas_species_payload(engine))
    reactions_payload = engine.bridge.shadow_tables.get("reactions")
    action_count = 0
    if reactions_payload is not None:
        action_count = len(reactions_payload.get("actions", [])) + 1
    for item in materials_payload:
        item_name = str(item.get("name", ""))
        reaction_slots = tuple(int(slot) for slot in item.get("reaction_slots", (-1,) * 8))
        if len(reaction_slots) != 8:
            raise ValueError("reaction_slots must contain exactly 8 entries")
        if action_count > 0:
            for action_index in reaction_slots:
                if action_index < -1 or action_index >= action_count:
                    raise IndexError(action_index)
        is_placeholder = str(item.get("render_group", "")) == "placeholder" or "placeholder" in tuple(
            str(tag) for tag in item.get("tags", ())
        )
        if item_name == "placeholder_solid" or is_placeholder:
            if bool(item.get("is_structural", False)):
                raise ValueError("placeholder materials cannot be structural")
            if bool(item.get("is_support_anchor", False)):
                raise ValueError("placeholder materials cannot be support anchors")
        for field in ("collapse_generation", "powder_generation", "melt_to_material", "freeze_to_material"):
            _validate_named_reference(material_names, item.get(field))
        if gas_names:
            _validate_named_reference(gas_names, item.get("boil_to_gas_species"))
    if reactions_payload is not None:
        uses_random_convert = any(
            action.get("target_material") == "__random__"
            for action in reactions_payload.get("actions", [])
        )
        if uses_random_convert:
            chaos_convert_candidates = [
                item
                for item in materials_payload
                if "chaos_convert" in tuple(str(tag) for tag in item.get("tags", ()))
                and int(engine._coerce_enum(Phase, item.get("default_phase", Phase.STATIC_SOLID))) == int(Phase.POWDER)
            ]
            if not chaos_convert_candidates:
                raise ValueError(
                    "material table must contain at least one powder material tagged chaos_convert when reactions use target_material=__random__"
                )


def _validate_gas_species_payload(engine: "WorldEngine", gases_payload: list[dict[str, Any]]) -> None:
    air_entries = [item for item in gases_payload if str(item.get("name", "")) == "air"]
    if len(air_entries) != 1:
        raise ValueError("gas table must contain exactly one air species")
    air_entry = air_entries[0]
    if int(air_entry["species_id"]) != 0:
        raise ValueError("air species_id must remain 0")
    if air_entry.get("condense_to_material") is not None:
        raise ValueError("air cannot condense to a material")
    _validate_unique_identity_fields(
        gases_payload,
        id_field="species_id",
        allow_zero_id=True,
    )
    material_names = _payload_name_set(_shadow_material_payload(engine))
    for item in gases_payload:
        _validate_named_reference(material_names, item.get("condense_to_material"))


def _validate_light_type_payload(engine: "WorldEngine", lights_payload: list[dict[str, Any]]) -> None:
    _validate_unique_identity_fields(
        lights_payload,
        id_field="light_type_id",
        allow_zero_id=True,
    )
    if len(lights_payload) > 8:
        raise ValueError("light type count exceeds maximum of 8")
    seen_dose_channels: set[int] = set()
    for item in lights_payload:
        light_type_id = int(item["light_type_id"])
        if light_type_id < 0 or light_type_id >= 8:
            raise ValueError(f"light_type_id out of range: {light_type_id}")
        dose_channel_id = int(item.get("dose_channel_id", -1))
        if dose_channel_id < 0 or dose_channel_id >= 8:
            raise ValueError(f"dose_channel_id out of range: {dose_channel_id}")
        if dose_channel_id in seen_dose_channels:
            raise ValueError(f"duplicate dose_channel_id: {dose_channel_id}")
        seen_dose_channels.add(dose_channel_id)


def _validate_material_optics_payload(engine: "WorldEngine", optics_payload: list[dict[str, Any]]) -> None:
    material_names = _payload_name_set(_shadow_material_payload(engine))
    light_names = _payload_name_set(_shadow_light_type_payload(engine))
    for item in optics_payload:
        _validate_named_reference(material_names, item.get("material_name"))
        _validate_named_reference(light_names, item.get("light_type"))


def _validate_reaction_payload(engine: "WorldEngine", reactions_payload: dict[str, Any]) -> None:
    material_names = _payload_name_set(_shadow_material_payload(engine))
    gas_names = _payload_name_set(_shadow_gas_species_payload(engine))
    light_names = _payload_name_set(_shadow_light_type_payload(engine))
    valid_consume_policies = {"none", "lhs", "rhs", "both"}
    actions_payload = list(reactions_payload.get("actions", []))
    action_count = len(actions_payload) + 1
    for action in actions_payload:
        reaction_type = engine._coerce_enum(ReactionType, action.get("reaction_type", ReactionType.NONE))
        if reaction_type == ReactionType.NONE:
            raise ValueError("reaction action 0 is reserved for ReactionType.NONE")
        duration = int(action.get("duration", 0))
        if duration < 0:
            raise ValueError(f"reaction actions require non-negative duration: {duration}")
        generation = int(action.get("generation", 0))
        if generation != 0:
            raise ValueError("reaction actions do not support non-zero generation")
        if bool(action.get("allow_subunit_scale", False)) and reaction_type != ReactionType.CONVERT_MATERIAL:
            raise ValueError("allow_subunit_scale is only supported for convert_material actions")
        target_material = action.get("target_material")
        if target_material == "__random__" and reaction_type != ReactionType.CONVERT_MATERIAL:
            raise ValueError("target_material=__random__ is only supported for convert_material actions")
        if target_material is not None and reaction_type != ReactionType.CONVERT_MATERIAL:
            if reaction_type != ReactionType.HARM:
                raise ValueError("target_material is only supported for convert_material and harm actions")
        if action.get("emit_material") is not None and reaction_type != ReactionType.EMIT_MATERIAL:
            raise ValueError("emit_material is only supported for emit_material actions")
        if action.get("light_type") is not None and reaction_type != ReactionType.EMIT_LIGHT:
            raise ValueError("light_type is only supported for emit_light actions")
        if action.get("gas_species") is not None and reaction_type != ReactionType.MODIFY_GAS:
            raise ValueError("gas_species is only supported for modify_gas actions")
        if float(action.get("delta", 0.0)) != 0.0 and reaction_type != ReactionType.MODIFY_TEMPERATURE:
            raise ValueError("delta is only supported for modify_temperature actions")
        if float(action.get("value", 0.0)) != 0.0 and reaction_type != ReactionType.HARM:
            raise ValueError("value is only supported for harm actions")
        velocity_value = action.get("velocity", (0.0, 0.0))
        velocity_xy = tuple(float(component) for component in velocity_value)
        if any(abs(component) > 1.0e-6 for component in velocity_xy) and reaction_type != ReactionType.EMIT_MATERIAL:
            raise ValueError("velocity is only supported for emit_material actions")
        if float(action.get("beam_width", 1.0)) != 1.0 and reaction_type != ReactionType.EMIT_LIGHT:
            raise ValueError("beam_width is only supported for emit_light actions")
        if target_material != "__random__":
            _validate_named_reference(material_names, target_material)
        _validate_named_reference(material_names, action.get("emit_material"))
        _validate_named_reference(gas_names, action.get("gas_species"))
        _validate_named_reference(light_names, action.get("light_type"))
        if reaction_type == ReactionType.EMIT_MATERIAL and action.get("emit_material") is None:
            raise ValueError("emit_material actions require emit_material")
        if reaction_type == ReactionType.EMIT_MATERIAL and float(action.get("speed", 0.0)) < 0.0:
            raise ValueError("emit_material actions require non-negative speed")
        if reaction_type == ReactionType.EMIT_LIGHT and action.get("light_type") is None:
            raise ValueError("emit_light actions require light_type")
        if reaction_type == ReactionType.EMIT_LIGHT and float(action.get("strength", 0.0)) <= 0.0:
            raise ValueError("emit_light actions require positive strength")
        if reaction_type == ReactionType.EMIT_LIGHT and float(action.get("beam_width", 1.0)) < 0.0:
            raise ValueError("emit_light actions require non-negative beam_width")
        if reaction_type == ReactionType.EMIT_LIGHT and int(action.get("range_cells", 0)) < 0:
            raise ValueError("emit_light actions require non-negative range_cells")
        if reaction_type == ReactionType.MODIFY_GAS and action.get("gas_species") is None:
            raise ValueError("modify_gas actions require gas_species")
        if reaction_type == ReactionType.CONVERT_MATERIAL and float(action.get("harm_per_frame", 0.0)) < 0.0:
            raise ValueError("convert_material actions require non-negative harm_per_frame")
    rules_payload = dict(reactions_payload.get("rules", {}))
    for rule_set in PAIR_REACTION_RULE_SET_NAMES:
        for rule in rules_payload.get(rule_set, []):
            _validate_named_reference(material_names, rule.get("lhs_material"))
            _validate_named_reference(material_names, rule.get("rhs_material"))
            _validate_named_reference(gas_names, rule.get("lhs_gas"))
            _validate_named_reference(gas_names, rule.get("rhs_gas"))
            _validate_named_reference(light_names, rule.get("rhs_light"))
            trigger_slot_value = rule.get("trigger_slot_index", -1)
            trigger_slot_index = -1 if trigger_slot_value is None else int(trigger_slot_value)
            if trigger_slot_index < -1 or trigger_slot_index >= 8:
                raise IndexError(trigger_slot_index)
            result_action = int(rule.get("result_action", 0))
            if result_action < -1 or result_action >= action_count:
                raise IndexError(result_action)
            min_temperature = float(rule.get("min_temperature", float("-inf")))
            max_temperature = float(rule.get("max_temperature", float("inf")))
            if min_temperature > max_temperature:
                raise ValueError(f"{rule_set} rule requires min_temperature <= max_temperature")
            threshold = float(rule.get("threshold", 0.0))
            if threshold < 0.0:
                raise ValueError(f"{rule_set} rule requires non-negative threshold")
            rate = float(rule.get("rate", 1.0))
            if rate < 0.0:
                raise ValueError(f"{rule_set} rule requires non-negative rate")
            has_trigger_slot = trigger_slot_index >= 0
            has_result_action = result_action > 0
            consume_policy = str(rule.get("consume_policy", "none") or "none").lower()
            if consume_policy not in valid_consume_policies:
                raise ValueError(f"invalid consume_policy: {consume_policy}")
            if has_trigger_slot and has_result_action:
                raise ValueError(f"{rule_set} rule cannot define both trigger_slot_index and result_action")
            if rule_set in {"gas_gas", "gas_light"} and has_trigger_slot:
                raise ValueError(f"{rule_set} rules cannot use trigger_slot_index")
    for rule in rules_payload.get("self_rules", []):
        _validate_named_reference(material_names, rule.get("material"))
        trigger_slot_value = rule.get("trigger_slot_index", -1)
        trigger_slot_index = -1 if trigger_slot_value is None else int(trigger_slot_value)
        if trigger_slot_index < 0 or trigger_slot_index >= 8:
            raise IndexError(trigger_slot_index)
        timer_index_value = rule.get("timer_index", None)
        timer_index = -1 if timer_index_value is None else int(timer_index_value)
        if timer_index < -1:
            raise IndexError(timer_index)
        if timer_index >= 4:
            raise IndexError(timer_index)
        if timer_index >= 0:
            if trigger_slot_index >= 4:
                raise ValueError("untimed self rules cannot define timer_index")
            if timer_index != trigger_slot_index:
                raise ValueError("self rule timer_index must match trigger_slot_index for timed slots")
        min_temperature = float(rule.get("min_temperature", float("-inf")))
        max_temperature = float(rule.get("max_temperature", float("inf")))
        if min_temperature > max_temperature:
            raise ValueError("self rules require min_temperature <= max_temperature")
        integrity_at_most = rule.get("integrity_at_most", None)
        integrity_at_least = rule.get("integrity_at_least", None)
        if integrity_at_most is not None and integrity_at_least is not None:
            if float(integrity_at_most) < float(integrity_at_least):
                raise ValueError("self rules require integrity_at_most >= integrity_at_least")


def _material_placeholder_mask(engine: "WorldEngine", material_id: np.ndarray) -> np.ndarray:
    ids = np.asarray(material_id, dtype=np.int64)
    mask = np.zeros(ids.shape, dtype=np.bool_)
    valid = (ids >= 0) & (ids < int(engine.material_is_placeholder.shape[0]))
    if np.any(valid):
        mask[valid] = engine.material_is_placeholder[ids[valid]]
    return mask


def _set_reaction_rule_list(engine: "WorldEngine", rule_set: str, entries: list[dict[str, Any]] | list[PairReactionRule] | list[SelfReactionRule]) -> None:
    normalized = str(rule_set)
    if normalized == "self_rules":
        normalized_entries = [engine._coerce_self_reaction_rule(entry) for entry in entries]
        engine.rulebook.self_rules = normalized_entries
        return
    normalized_entries = [engine._coerce_pair_reaction_rule(entry) for entry in entries]
    if normalized == "material_material":
        engine.rulebook.material_material_rules = normalized_entries
        return
    if normalized == "material_gas":
        engine.rulebook.material_gas_rules = normalized_entries
        return
    if normalized == "material_light":
        engine.rulebook.material_light_rules = normalized_entries
        return
    if normalized == "gas_gas":
        engine.rulebook.gas_gas_rules = normalized_entries
        return
    if normalized == "gas_light":
        engine.rulebook.gas_light_rules = normalized_entries
        return
    raise KeyError(rule_set)


def _set_reaction_rules_payload(engine: "WorldEngine", rules_payload: dict[str, list[dict[str, Any]]]) -> None:
    for rule_set in REACTION_RULE_SET_NAMES:
        _set_reaction_rule_list(engine, str(rule_set), list(rules_payload.get(str(rule_set), [])))


def _remap_reaction_payload_result_actions(
    rules_payload: dict[str, list[dict[str, Any]]],
    *,
    deleted_action_index: int,
) -> None:
    for rule_set in PAIR_REACTION_RULE_SET_NAMES:
        for rule in rules_payload.get(str(rule_set), []):
            result_action = int(rule.get("result_action", -1))
            if result_action == deleted_action_index:
                rule["result_action"] = 0
            elif result_action > deleted_action_index:
                rule["result_action"] = result_action - 1


def _remap_material_payload_reaction_slots(
    materials_payload: list[dict[str, Any]],
    *,
    deleted_action_index: int,
) -> None:
    for material in materials_payload:
        slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
        remapped_slots: list[int] = []
        for slot in slots[:8]:
            action_index = int(slot)
            if action_index == deleted_action_index:
                remapped_slots.append(-1)
            elif action_index > deleted_action_index:
                remapped_slots.append(action_index - 1)
            else:
                remapped_slots.append(action_index)
        if len(remapped_slots) < 8:
            remapped_slots.extend([-1] * (8 - len(remapped_slots)))
        material["reaction_slots"] = tuple(remapped_slots)


def _clamp_material_payload_reaction_slots(
    materials_payload: list[dict[str, Any]],
    *,
    action_count: int,
) -> None:
    for material in materials_payload:
        slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
        clamped_slots: list[int] = []
        for slot in slots[:8]:
            action_index = int(slot)
            if action_index < -1 or action_index >= action_count:
                clamped_slots.append(-1)
            else:
                clamped_slots.append(action_index)
        if len(clamped_slots) < 8:
            clamped_slots.extend([-1] * (8 - len(clamped_slots)))
        material["reaction_slots"] = tuple(clamped_slots)
