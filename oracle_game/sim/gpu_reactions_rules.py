from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.types import ReactionType
from oracle_game.sim.gpu_reactions import (
    ACTION_FLAG_ALLOW_SUBUNIT_SCALE,
    ACTION_FLAG_RANDOM_TARGET,
    CONSUME_POLICY_BOTH,
    CONSUME_POLICY_NONE,
    CONSUME_POLICY_RHS,
    FLOW_SOURCE_LAYERS,
    MAX_ACTIONS,
    MAX_MATERIAL_LIGHT_PACKED_RULES,
    MAX_MATERIAL_PAIR_PACKED_RULES,
    MAX_MATERIALS,
    MAX_RULES,
    MAX_SELF_RULES,
    RULE_CANDIDATE_VECS,
    RULE_CANDIDATE_WORDS,
    TYPE_CONVERT_MATERIAL,
    TYPE_DEFERRED,
    TYPE_EMIT_LIGHT,
    TYPE_EMIT_MATERIAL,
    TYPE_HARM,
    TYPE_MODIFY_GAS,
    TYPE_MODIFY_TEMPERATURE,
    TYPE_NONE,
)


def _compile_action_buffers(
    pipeline,
    action_table: np.ndarray,
    used_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
    action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
    action_count = int(action_table.shape[0])
    if action_count > MAX_ACTIONS:
        return None
    for index in range(action_count):
        row = action_table[index]
        action_i[index, 3] = max(0, int(row["duration"]))
        if used_indices is not None and index not in used_indices:
            action_i[index, 0] = TYPE_NONE
            continue
        reaction_type_id = int(row["reaction_type_id"])
        flags = 0
        if reaction_type_id == int(ReactionType.HARM.value):
            action_i[index, 0] = TYPE_HARM
            action_f[index, 1] = float(row["value"])
        elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
            action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
            action_f[index, 0] = float(row["delta"])
        elif reaction_type_id == int(ReactionType.CONVERT_MATERIAL.value):
            action_i[index, 0] = TYPE_CONVERT_MATERIAL
            if int(row["flags"]) & ACTION_FLAG_RANDOM_TARGET:
                flags |= 1
            if int(row["flags"]) & ACTION_FLAG_ALLOW_SUBUNIT_SCALE:
                flags |= ACTION_FLAG_ALLOW_SUBUNIT_SCALE
            action_i[index, 1] = int(row["target_material_id"])
            action_i[index, 2] = flags
            action_f[index, 2] = float(row["harm_per_frame"])
            action_f[index, 3] = float(row["integrity_threshold"])
        elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
            action_i[index, 0] = TYPE_MODIFY_GAS
            action_i[index, 1] = int(row["gas_species_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
            action_f[index, 0] = float(row["speed"]) * 0.1
            action_f[index, 1] = float(row["strength"])
            action_f[index, 2] = float(row["range_cells"])
            action_f[index, 3] = float(row["speed"])
        elif (
            reaction_type_id == int(ReactionType.EMIT_LIGHT.value)
            and int(row["light_type_id"]) >= 0
        ):
            action_i[index, 0] = TYPE_EMIT_LIGHT
            action_i[index, 1] = int(row["light_type_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["strength"])
            action_f[index, 1] = float(row["range_cells"])
            action_f[index, 2] = float(row["beam_width"])
        elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
            action_i[index, 0] = TYPE_EMIT_MATERIAL
            action_i[index, 1] = int(row["emit_material_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["velocity"][0])
            action_f[index, 1] = float(row["velocity"][1])
            action_f[index, 2] = float(row["speed"])
        elif reaction_type_id == int(ReactionType.NONE.value):
            action_i[index, 0] = TYPE_NONE
        else:
            action_i[index, 0] = TYPE_DEFERRED
    return action_i, action_f



def _compile_action_buffers_cached(
    pipeline,
    world: "WorldEngine",
    action_table: np.ndarray,
    used_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    used_key = None if used_indices is None else tuple(sorted(int(index) for index in used_indices))
    key = (
        id(world.bridge),
        int(world.bridge.table_generations.get("reactions", 0)),
        int(action_table.shape[0]),
        used_key,
    )
    if key not in pipeline._compiled_action_cache:
        if len(pipeline._compiled_action_cache) > 64:
            pipeline._compiled_action_cache.clear()
        pipeline._compiled_action_cache[key] = pipeline._compile_action_buffers(action_table, used_indices)
    return pipeline._compiled_action_cache[key]



def _compiled_actions_include_modify_gas(pipeline, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.any(compiled_actions[0][:, 0] == TYPE_MODIFY_GAS))



def _compiled_actions_include_flow_sources(pipeline, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
    action_i = np.asarray(compiled_actions[0], dtype=np.int32)
    return bool(np.any((action_i[:, 0] == TYPE_MODIFY_GAS) & (action_i[:, 3] != 0)))



def _compiled_self_rule_flow_source_layers(
    rule_table: np.ndarray,
    material_table: np.ndarray,
    compiled_actions: tuple[np.ndarray, np.ndarray],
) -> int:
    action_i = np.asarray(compiled_actions[0], dtype=np.int32)
    if not bool(np.any((action_i[:, 0] == TYPE_MODIFY_GAS) & (action_i[:, 3] != 0))):
        return FLOW_SOURCE_LAYERS
    if "trigger_slot_index" not in rule_table.dtype.names or "reaction_slots" not in material_table.dtype.names:
        return FLOW_SOURCE_LAYERS
    if "material_id" not in rule_table.dtype.names:
        return FLOW_SOURCE_LAYERS
    material_count = int(material_table.shape[0])
    slots_by_material: dict[int, set[int]] = {}
    for rule in rule_table[:MAX_SELF_RULES]:
        slot_index = int(rule["trigger_slot_index"])
        if slot_index < 0 or slot_index >= 8:
            return FLOW_SOURCE_LAYERS
        material_id = int(rule["material_id"])
        if material_id > 0:
            if material_id >= material_count:
                return FLOW_SOURCE_LAYERS
            slots_by_material.setdefault(material_id, set()).add(slot_index)
        else:
            for candidate_material_id in range(1, material_count):
                slots_by_material.setdefault(candidate_material_id, set()).add(slot_index)
    deferred_types = {TYPE_DEFERRED, TYPE_EMIT_MATERIAL, TYPE_MODIFY_GAS}
    max_flow_position = -1
    for material_id, slot_indices in slots_by_material.items():
        deferred_position = 0
        for slot_index in sorted(slot_indices):
            action_index = int(material_table[material_id]["reaction_slots"][slot_index])
            if action_index < 0:
                continue
            if action_index >= action_i.shape[0]:
                return FLOW_SOURCE_LAYERS
            ai = action_i[action_index]
            action_type = int(ai[0])
            if action_type not in deferred_types:
                continue
            if action_type == TYPE_MODIFY_GAS and int(ai[3]) != 0:
                max_flow_position = max(max_flow_position, deferred_position)
            deferred_position += 1
    if max_flow_position < 0:
        return FLOW_SOURCE_LAYERS
    return max(1, min(FLOW_SOURCE_LAYERS, (max_flow_position + 1) * 4))


def _compiled_modify_gas_layer_mask(compiled_actions: tuple[np.ndarray, np.ndarray], gas_count: int) -> int:
    action_i = np.asarray(compiled_actions[0], dtype=np.int32)
    mask = 0
    for raw_layer in action_i[action_i[:, 0] == TYPE_MODIFY_GAS, 1].tolist():
        layer = int(raw_layer)
        if layer < 0:
            continue
        if layer >= int(gas_count):
            continue
        if layer >= 31:
            return (1 << min(31, int(gas_count))) - 1
        mask |= 1 << layer
    return mask



def _compiled_actions_include_emit_material(pipeline, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.any(compiled_actions[0][:, 0] == TYPE_EMIT_MATERIAL))


def _compiled_actions_include_emit_light(pipeline, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.any(compiled_actions[0][:, 0] == TYPE_EMIT_LIGHT))


def _compiled_actions_require_deferred_outputs(
    pipeline,
    compiled_actions: tuple[np.ndarray, np.ndarray],
    *,
    direct_modify_gas: bool = False,
) -> bool:
    """Return whether a pass must publish its deferred action textures."""
    action_types = np.asarray(compiled_actions[0][:, 0], dtype=np.int32)
    return bool(
        np.any(
            (action_types == TYPE_DEFERRED)
            | (action_types == TYPE_EMIT_MATERIAL)
            | ((action_types == TYPE_MODIFY_GAS) & (not direct_modify_gas))
        )
    )


def _self_rules_require_deferred_hi_outputs(
    pipeline,
    rule_table: np.ndarray,
    material_table: np.ndarray,
    compiled_actions: tuple[np.ndarray, np.ndarray],
) -> bool:
    """Return whether any material can enqueue more than four self actions."""
    if "material_id" not in rule_table.dtype.names or "trigger_slot_index" not in rule_table.dtype.names:
        return True
    if "reaction_slots" not in material_table.dtype.names:
        return True
    action_types = np.asarray(compiled_actions[0][:, 0], dtype=np.int32)
    deferred_types = {TYPE_DEFERRED, TYPE_EMIT_MATERIAL, TYPE_MODIFY_GAS}
    slots_by_material: dict[int, set[int]] = {}
    for rule in rule_table[:MAX_SELF_RULES]:
        material_id = int(rule["material_id"])
        slot_index = int(rule["trigger_slot_index"])
        if material_id <= 0 or material_id >= int(material_table.shape[0]) or slot_index < 0 or slot_index >= 8:
            return True
        slots_by_material.setdefault(material_id, set()).add(slot_index)
    for material_id, slot_indices in slots_by_material.items():
        deferred_count = 0
        for slot_index in slot_indices:
            action_index = int(material_table[material_id]["reaction_slots"][slot_index])
            if action_index < 0:
                continue
            if action_index >= int(action_types.shape[0]):
                return True
            if int(action_types[action_index]) in deferred_types:
                deferred_count += 1
                if deferred_count > 4:
                    return True
    return False



def _compiled_actions_may_change_structure(pipeline, compiled_actions: tuple[np.ndarray, np.ndarray]) -> bool:
    action_types = np.asarray(compiled_actions[0][:, 0], dtype=np.int32)
    return bool(
        np.any(
            (action_types == TYPE_HARM)
            | (action_types == TYPE_CONVERT_MATERIAL)
            | (action_types == TYPE_EMIT_MATERIAL)
        )
    )



def _compiled_rules_include_rhs_consume(pipeline, rule_tags: np.ndarray) -> bool:
    consume_policies = np.asarray(rule_tags[:, 3], dtype=np.uint32)
    return bool(np.any((consume_policies == CONSUME_POLICY_RHS) | (consume_policies == CONSUME_POLICY_BOTH)))



def _compile_gas_action_buffers(
    pipeline,
    action_table: np.ndarray,
    used_indices: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
    action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
    action_count = int(action_table.shape[0])
    if action_count > MAX_ACTIONS:
        return None
    for index in range(action_count):
        row = action_table[index]
        if index not in used_indices:
            action_i[index, 0] = TYPE_NONE
            continue
        reaction_type_id = int(row["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            action_i[index, 0] = TYPE_NONE
        elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
            action_i[index, 0] = TYPE_MODIFY_GAS
            action_i[index, 1] = int(row["gas_species_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
            action_f[index, 0] = float(row["speed"]) * 0.1
            action_f[index, 1] = float(row["strength"])
            action_f[index, 2] = float(row["range_cells"])
            action_f[index, 3] = float(row["speed"])
        elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
            action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
            action_f[index, 0] = float(row["delta"])
        elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value) and int(row["light_type_id"]) >= 0:
            action_i[index, 0] = TYPE_EMIT_LIGHT
            action_i[index, 1] = int(row["light_type_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["strength"])
            action_f[index, 1] = float(row["range_cells"])
            action_f[index, 2] = float(row["beam_width"])
        elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
            action_i[index, 0] = TYPE_EMIT_MATERIAL
            action_i[index, 1] = int(row["emit_material_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["velocity"][0])
            action_f[index, 1] = float(row["velocity"][1])
            action_f[index, 2] = float(row["speed"])
        else:
            return None
    return action_i, action_f



def _compile_gas_light_action_buffers(
    pipeline,
    action_table: np.ndarray,
    used_indices: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    action_i = np.zeros((MAX_ACTIONS, 4), dtype=np.int32)
    action_f = np.zeros((MAX_ACTIONS, 4), dtype=np.float32)
    action_count = int(action_table.shape[0])
    if action_count > MAX_ACTIONS:
        return None
    for index in range(action_count):
        row = action_table[index]
        if index not in used_indices:
            action_i[index, 0] = TYPE_NONE
            continue
        reaction_type_id = int(row["reaction_type_id"])
        if reaction_type_id == int(ReactionType.NONE.value):
            action_i[index, 0] = TYPE_NONE
        elif reaction_type_id == int(ReactionType.MODIFY_GAS.value) and int(row["gas_species_id"]) >= 0:
            action_i[index, 0] = TYPE_MODIFY_GAS
            action_i[index, 1] = int(row["gas_species_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_i[index, 3] = int(float(row["strength"]) > 0.0 and int(row["range_cells"]) > 0)
            action_f[index, 0] = float(row["speed"]) * 0.1
            action_f[index, 1] = float(row["strength"])
            action_f[index, 2] = float(row["range_cells"])
            action_f[index, 3] = float(row["speed"])
        elif reaction_type_id == int(ReactionType.MODIFY_TEMPERATURE.value):
            action_i[index, 0] = TYPE_MODIFY_TEMPERATURE
            action_f[index, 0] = float(row["delta"])
        elif reaction_type_id == int(ReactionType.EMIT_LIGHT.value) and int(row["light_type_id"]) >= 0:
            action_i[index, 0] = TYPE_EMIT_LIGHT
            action_i[index, 1] = int(row["light_type_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["strength"])
            action_f[index, 1] = float(row["range_cells"])
            action_f[index, 2] = float(row["beam_width"])
        elif reaction_type_id == int(ReactionType.EMIT_MATERIAL.value) and int(row["emit_material_id"]) > 0:
            action_i[index, 0] = TYPE_EMIT_MATERIAL
            action_i[index, 1] = int(row["emit_material_id"])
            action_i[index, 2] = int(row["direction_id"])
            action_f[index, 0] = float(row["velocity"][0])
            action_f[index, 1] = float(row["velocity"][1])
            action_f[index, 2] = float(row["speed"])
        else:
            return None
    return action_i, action_f



def _modify_gas_action_requires_cpu_flow_side_effect(row: np.void) -> bool:
    strength = float(row["strength"])
    radius = int(row["range_cells"])
    if strength <= 0.0 or radius <= 0:
        return False
    velocity = np.asarray(row["velocity"], dtype=np.float32)
    if float(np.hypot(float(velocity[0]), float(velocity[1]))) > 1.0e-6:
        return True
    direction_id = int(row["direction_id"])
    if direction_id != 0:
        return True
    return abs(float(row["speed"])) > 1.0e-6



def _rule_candidate_word_count(rule_count: int) -> int:
    return min(RULE_CANDIDATE_WORDS, max(0, (int(rule_count) + 31) // 32))



def _empty_rule_candidate_masks() -> np.ndarray:
    return np.zeros((MAX_MATERIALS, RULE_CANDIDATE_VECS, 4), dtype=np.uint32)



def _set_rule_candidate(mask_table: np.ndarray, material_id: int, rule_index: int) -> None:
    if material_id <= 0 or material_id >= MAX_MATERIALS or rule_index < 0 or rule_index >= MAX_RULES:
        return
    word_index = rule_index // 32
    mask_table[material_id, word_index // 4, word_index % 4] |= np.uint32(1 << (rule_index % 32))



def _compile_material_rule_candidate_masks(
    pipeline,
    rule_table: np.ndarray,
    material_table: np.ndarray,
    *,
    selector_id_field: str,
    selector_tag_field: str,
    material_tag_field: str,
) -> np.ndarray:
    masks = pipeline._empty_rule_candidate_masks()
    count = min(MAX_RULES, int(rule_table.shape[0]))
    material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
    rule_field_names = rule_table.dtype.names or ()
    material_field_names = material_table.dtype.names or ()
    authoritative = bool(pipeline._authoritative_lhs_candidate_masks_enabled)
    material_tags = (
        np.asarray(material_table[material_tag_field], dtype=np.uint32)
        if material_tag_field in material_field_names
        else None
    )
    for rule_index, rule in enumerate(rule_table[:count]):
        selector_id = int(rule[selector_id_field]) if selector_id_field in rule_field_names else -1
        selector_tag_mask = int(rule[selector_tag_field]) if selector_tag_field in rule_field_names else 0
        if selector_id > 0:
            if authoritative and selector_tag_mask != 0:
                if selector_id >= material_count or material_tags is None:
                    continue
                required = np.uint32(selector_tag_mask)
                if (material_tags[selector_id] & required) != required:
                    continue
            pipeline._set_rule_candidate(masks, selector_id, rule_index)
            continue
        if selector_tag_mask != 0 and material_tags is not None:
            required = np.uint32(selector_tag_mask)
            for material_id in range(1, material_count):
                if (material_tags[material_id] & required) == required:
                    pipeline._set_rule_candidate(masks, material_id, rule_index)
            continue
        if authoritative and selector_tag_mask != 0:
            continue
        for material_id in range(1, MAX_MATERIALS):
            pipeline._set_rule_candidate(masks, material_id, rule_index)
    return masks



def _compile_material_material_rules(pipeline, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
    rule_i[:, 3] = -1
    rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
    rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
    count = min(MAX_RULES, int(rule_table.shape[0]))
    lhs_ids = rule_table[:count]["lhs_material_id"]
    rhs_ids = rule_table[:count]["rhs_material_id"]
    rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
    rule_i[:count, 1] = np.where(rhs_ids > 0, rhs_ids, -1)
    rule_i[:count, 2] = rule_table[:count]["result_action"]
    rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
    rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
    rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
    rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
    rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
    rule_f[:count, 0] = rule_table[:count]["min_temperature"]
    rule_f[:count, 1] = rule_table[:count]["max_temperature"]
    rule_f[:count, 2] = rule_table[:count]["threshold"]
    rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
    return rule_i, rule_f, rule_tags



def _compile_material_gas_rules(pipeline, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
    rule_i[:, 3] = -1
    rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
    rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
    count = min(MAX_RULES, int(rule_table.shape[0]))
    lhs_ids = rule_table[:count]["lhs_material_id"]
    rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
    rule_i[:count, 1] = rule_table[:count]["rhs_gas_id"]
    rule_i[:count, 2] = rule_table[:count]["result_action"]
    rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
    rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
    rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
    rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
    rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
    rule_f[:count, 0] = rule_table[:count]["min_temperature"]
    rule_f[:count, 1] = rule_table[:count]["max_temperature"]
    rule_f[:count, 2] = rule_table[:count]["threshold"]
    rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
    return rule_i, rule_f, rule_tags



def _compile_material_light_rules(
    pipeline,
    rule_table: np.ndarray,
    light_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
    rule_i[:, 3] = -1
    rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
    rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
    count = min(MAX_RULES, int(rule_table.shape[0]))
    lhs_ids = rule_table[:count]["lhs_material_id"]
    rule_i[:count, 0] = np.where(lhs_ids > 0, lhs_ids, -1)
    rhs_light_ids = rule_table[:count]["rhs_light_id"].astype(np.int32)
    dose_channels = np.full((count,), -1, dtype=np.int32)
    valid = (rhs_light_ids >= 0) & (rhs_light_ids < int(light_table.shape[0]))
    dose_channels[valid] = light_table[rhs_light_ids[valid]]["dose_channel_id"].astype(np.int32)
    rule_i[:count, 1] = dose_channels
    rule_i[:count, 2] = rule_table[:count]["result_action"]
    rule_i[:count, 3] = rule_table[:count]["trigger_slot_index"]
    rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
    rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
    rule_tags[:count, 2] = rule_table[:count]["phase_mask"]
    rule_tags[:count, 3] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
    rule_f[:count, 0] = rule_table[:count]["min_temperature"]
    rule_f[:count, 1] = rule_table[:count]["max_temperature"]
    rule_f[:count, 2] = rule_table[:count]["threshold"]
    rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
    return rule_i, rule_f, rule_tags


def _compile_material_light_packed_descriptors(
    rule_table: np.ndarray,
    material_table: np.ndarray,
    light_table: np.ndarray,
) -> np.ndarray | None:
    required_rule_fields = {
        "lhs_material_id",
        "lhs_tag_mask",
        "rhs_tag_mask",
        "rhs_light_id",
        "phase_mask",
        "consume_policy_id",
        "result_action",
        "trigger_slot_index",
        "min_temperature",
        "max_temperature",
        "threshold",
        "rate",
    }
    if not required_rule_fields.issubset(rule_table.dtype.names or ()):
        return None
    if "dose_channel_id" not in (light_table.dtype.names or ()):
        return None
    rule_count = int(rule_table.shape[0])
    material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
    if rule_count <= 0 or rule_count > MAX_RULES or material_count <= 1:
        return None

    per_material: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(MAX_MATERIALS)
    ]
    channel_masks = np.zeros((MAX_MATERIALS,), dtype=np.uint32)
    for rule in rule_table[:rule_count]:
        material_id = int(rule["lhs_material_id"])
        if material_id <= 0 or material_id >= material_count:
            return None
        if int(rule["lhs_tag_mask"]) != 0 or int(rule["rhs_tag_mask"]) != 0:
            return None
        if int(rule["phase_mask"]) != 0 or int(rule["consume_policy_id"]) != CONSUME_POLICY_NONE:
            return None
        if not np.isneginf(np.float32(rule["min_temperature"])):
            return None
        if not np.isposinf(np.float32(rule["max_temperature"])):
            return None
        threshold = np.float32(rule["threshold"])
        rate = np.float32(rule["rate"])
        if not np.isfinite(threshold) or rate != np.float32(1.0):
            return None

        light_id = int(rule["rhs_light_id"])
        if light_id < 0 or light_id >= int(light_table.shape[0]):
            return None
        dose_channel = int(light_table[light_id]["dose_channel_id"])
        if dose_channel < 0 or dose_channel >= 4:
            return None

        result_action = int(rule["result_action"])
        trigger_slot = int(rule["trigger_slot_index"])
        has_action = 0 <= result_action < MAX_ACTIONS
        has_slot = 0 <= trigger_slot < 8
        if has_action == has_slot:
            return None
        material_descriptors = per_material[material_id]
        if len(material_descriptors) >= MAX_MATERIAL_LIGHT_PACKED_RULES:
            return None
        operation_index = result_action if has_action else trigger_slot
        packed_operation = operation_index | (int(has_action) << 8) | (dose_channel << 9)
        material_descriptors.append(
            (
                packed_operation,
                int(threshold.view(np.uint32)),
                int(rate.view(np.uint32)),
                0,
            )
        )
        channel_masks[material_id] |= np.uint32(1 << dose_channel)

    packed = np.zeros((MAX_MATERIALS + MAX_RULES, 4), dtype=np.uint32)
    descriptor_cursor = 0
    for material_id, material_descriptors in enumerate(per_material):
        if not material_descriptors:
            continue
        count = len(material_descriptors)
        packed[material_id, 0] = np.uint32(count)
        packed[material_id, 1] = channel_masks[material_id]
        packed[material_id, 2] = np.uint32(descriptor_cursor)
        packed[
            MAX_MATERIALS + descriptor_cursor : MAX_MATERIALS + descriptor_cursor + count
        ] = np.asarray(material_descriptors, dtype=np.uint32)
        descriptor_cursor += count
    return packed


def _compile_material_light_packed_descriptors_cached(
    pipeline,
    world: "WorldEngine",
    rule_table: np.ndarray,
    material_table: np.ndarray,
    light_table: np.ndarray,
) -> np.ndarray | None:
    key = (
        id(world.bridge),
        int(world.bridge.table_generations.get("reactions", 0)),
        int(world.bridge.table_generations.get("materials", 0)),
        int(world.bridge.table_generations.get("lights", 0)),
        int(rule_table.shape[0]),
        int(material_table.shape[0]),
        int(light_table.shape[0]),
    )
    if pipeline._material_light_packed_descriptor_cache_key != key:
        pipeline._material_light_packed_descriptor_cache = (
            pipeline._compile_material_light_packed_descriptors(
                rule_table,
                material_table,
                light_table,
            )
        )
        pipeline._material_light_packed_descriptor_cache_key = key
    return pipeline._material_light_packed_descriptor_cache


def _compile_material_pair_packed_descriptors(
    mm_table: np.ndarray,
    mg_table: np.ndarray,
    material_table: np.ndarray,
    gas_count: int,
) -> np.ndarray | None:
    common_fields = {
        "lhs_material_id",
        "lhs_tag_mask",
        "rhs_tag_mask",
        "phase_mask",
        "consume_policy_id",
        "result_action",
        "trigger_slot_index",
        "min_temperature",
        "max_temperature",
        "rate",
    }
    if not common_fields.union({"rhs_material_id"}).issubset(mm_table.dtype.names or ()):
        return None
    if not common_fields.union({"rhs_gas_id", "threshold"}).issubset(mg_table.dtype.names or ()):
        return None
    material_count = min(MAX_MATERIALS, int(material_table.shape[0]))
    mm_count = int(mm_table.shape[0])
    mg_count = int(mg_table.shape[0])
    if (
        material_count <= 1
        or mm_count <= 0
        or mg_count <= 0
        or mm_count + mg_count > MAX_RULES
        or gas_count <= 0
        or gas_count > 8
    ):
        return None

    per_material_mm: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(MAX_MATERIALS)
    ]
    per_material_mg: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(MAX_MATERIALS)
    ]
    per_material_gases: list[list[int]] = [[] for _ in range(MAX_MATERIALS)]

    def validate_common(rule: np.void) -> tuple[int, int, bool] | None:
        material_id = int(rule["lhs_material_id"])
        if material_id <= 0 or material_id >= material_count:
            return None
        if int(rule["lhs_tag_mask"]) != 0 or int(rule["rhs_tag_mask"]) != 0:
            return None
        if int(rule["phase_mask"]) != 0:
            return None
        if int(rule["consume_policy_id"]) != CONSUME_POLICY_NONE:
            return None
        if np.float32(rule["rate"]).view(np.uint32) != np.float32(1.0).view(np.uint32):
            return None
        result_action = int(rule["result_action"])
        trigger_slot = int(rule["trigger_slot_index"])
        has_action = 0 <= result_action < MAX_ACTIONS
        has_slot = 0 <= trigger_slot < 8
        if has_action == has_slot:
            return None
        return material_id, result_action if has_action else trigger_slot, has_action

    for rule in mm_table[:mm_count]:
        validated = validate_common(rule)
        if validated is None:
            return None
        material_id, operation_index, direct_action = validated
        rhs_id = int(rule["rhs_material_id"])
        if rhs_id <= 0 or rhs_id >= material_count:
            return None
        packed_operation = operation_index | (int(direct_action) << 8) | (rhs_id << 9)
        per_material_mm[material_id].append(
            (
                packed_operation,
                int(np.float32(rule["min_temperature"]).view(np.uint32)),
                int(np.float32(rule["max_temperature"]).view(np.uint32)),
                0,
            )
        )

    for rule in mg_table[:mg_count]:
        validated = validate_common(rule)
        if validated is None:
            return None
        material_id, operation_index, direct_action = validated
        rhs_id = int(rule["rhs_gas_id"])
        if rhs_id < 0 or rhs_id >= gas_count:
            return None
        gas_ids = per_material_gases[material_id]
        if rhs_id not in gas_ids:
            if len(gas_ids) >= 4:
                return None
            gas_ids.append(rhs_id)
        gas_slot = gas_ids.index(rhs_id)
        packed_operation = (
            operation_index
            | (int(direct_action) << 8)
            | (rhs_id << 9)
            | (gas_slot << 17)
        )
        per_material_mg[material_id].append(
            (
                packed_operation,
                int(np.float32(rule["min_temperature"]).view(np.uint32)),
                int(np.float32(rule["max_temperature"]).view(np.uint32)),
                int(np.float32(rule["threshold"]).view(np.uint32)),
            )
        )

    packed = np.zeros((MAX_MATERIALS + MAX_RULES, 4), dtype=np.uint32)
    descriptor_cursor = 0
    for material_id in range(MAX_MATERIALS):
        mm_descriptors = per_material_mm[material_id]
        mg_descriptors = per_material_mg[material_id]
        if len(mm_descriptors) + len(mg_descriptors) > MAX_MATERIAL_PAIR_PACKED_RULES:
            return None
        mm_offset = descriptor_cursor
        if mm_descriptors:
            count = len(mm_descriptors)
            packed[
                MAX_MATERIALS + descriptor_cursor : MAX_MATERIALS + descriptor_cursor + count
            ] = np.asarray(mm_descriptors, dtype=np.uint32)
            descriptor_cursor += count
        mg_offset = descriptor_cursor
        if mg_descriptors:
            count = len(mg_descriptors)
            packed[
                MAX_MATERIALS + descriptor_cursor : MAX_MATERIALS + descriptor_cursor + count
            ] = np.asarray(mg_descriptors, dtype=np.uint32)
            descriptor_cursor += count
        if descriptor_cursor > MAX_RULES:
            return None
        gas_ids = per_material_gases[material_id]
        packed_gases = 0
        for slot, gas_id in enumerate(gas_ids):
            packed_gases |= gas_id << (slot * 8)
        packed[material_id] = (
            np.uint32(mm_offset | (len(mm_descriptors) << 16)),
            np.uint32(mg_offset | (len(mg_descriptors) << 16)),
            np.uint32(packed_gases),
            np.uint32(len(gas_ids)),
        )
    return packed


def _compile_material_pair_packed_descriptors_cached(
    pipeline,
    world: "WorldEngine",
    mm_table: np.ndarray,
    mg_table: np.ndarray,
    material_table: np.ndarray,
    gas_count: int,
) -> np.ndarray | None:
    key = (
        id(world.bridge),
        int(world.bridge.table_generations.get("reactions", 0)),
        int(world.bridge.table_generations.get("materials", 0)),
        int(world.bridge.table_generations.get("gases", 0)),
        int(mm_table.shape[0]),
        int(mg_table.shape[0]),
        int(material_table.shape[0]),
        int(gas_count),
    )
    if pipeline._material_pair_packed_descriptor_cache_key != key:
        pipeline._material_pair_packed_descriptor_cache = (
            pipeline._compile_material_pair_packed_descriptors(
                mm_table,
                mg_table,
                material_table,
                gas_count,
            )
        )
        pipeline._material_pair_packed_descriptor_cache_key = key
    return pipeline._material_pair_packed_descriptor_cache


def _compile_gas_gas_rules(pipeline, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
    rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
    rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
    count = min(MAX_RULES, int(rule_table.shape[0]))
    rule_i[:count, 0] = rule_table[:count]["lhs_gas_id"]
    rule_i[:count, 1] = rule_table[:count]["rhs_gas_id"]
    rule_i[:count, 2] = rule_table[:count]["result_action"]
    rule_tags[:count, 0] = rule_table[:count]["lhs_tag_mask"]
    rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
    rule_tags[:count, 2] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
    rule_f[:count, 0] = rule_table[:count]["min_temperature"]
    rule_f[:count, 1] = rule_table[:count]["max_temperature"]
    rule_f[:count, 2] = rule_table[:count]["threshold"]
    rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
    return rule_i, rule_f, rule_tags



def _compile_single_gas_gas_rule(pipeline, rule_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pipeline._compile_gas_gas_rules(rule_table[:1])



def _compile_gas_light_rules(
    pipeline,
    rule_table: np.ndarray,
    light_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule_i = np.zeros((MAX_RULES, 4), dtype=np.int32)
    rule_f = np.zeros((MAX_RULES, 4), dtype=np.float32)
    rule_tags = np.zeros((MAX_RULES, 4), dtype=np.uint32)
    count = min(MAX_RULES, int(rule_table.shape[0]))
    rhs_gas_ids = rule_table[:count]["rhs_gas_id"].astype(np.int32)
    rule_i[:count, 0] = np.where(rhs_gas_ids >= 0, rhs_gas_ids, -1)
    rhs_light_ids = rule_table[:count]["rhs_light_id"].astype(np.int32)
    dose_channels = np.full((count,), -1, dtype=np.int32)
    valid = (rhs_light_ids >= 0) & (rhs_light_ids < int(light_table.shape[0]))
    dose_channels[valid] = light_table[rhs_light_ids[valid]]["dose_channel_id"].astype(np.int32)
    rule_i[:count, 1] = dose_channels
    rule_i[:count, 2] = rule_table[:count]["result_action"]
    rule_tags[:count, 1] = rule_table[:count]["rhs_tag_mask"]
    rule_tags[:count, 2] = rule_table[:count]["consume_policy_id"].astype(np.uint32)
    rule_f[:count, 0] = rule_table[:count]["min_temperature"]
    rule_f[:count, 1] = rule_table[:count]["max_temperature"]
    rule_f[:count, 2] = rule_table[:count]["threshold"]
    rule_f[:count, 3] = np.maximum(rule_table[:count]["rate"], 0.0)
    return rule_i, rule_f, rule_tags



def _compile_single_gas_light_rule(
    pipeline,
    rule_table: np.ndarray,
    light_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pipeline._compile_gas_light_rules(rule_table[:1], light_table)



def _used_action_indices(pipeline, rule_table: np.ndarray) -> set[int] | None:
    used_indices: set[int] = set()
    for raw_value in rule_table["result_action"].tolist():
        action_index = int(raw_value)
        if action_index < 0:
            continue
        if action_index >= MAX_ACTIONS:
            return None
        used_indices.add(action_index)
    return used_indices



def _used_action_indices_for_material_slots(
    pipeline,
    material_table: np.ndarray,
    *,
    slot_count: int | None = None,
) -> set[int] | None:
    if "reaction_slots" not in material_table.dtype.names:
        return None
    used_indices: set[int] = set()
    reaction_slots = np.asarray(material_table["reaction_slots"], dtype=np.int32)
    if slot_count is not None:
        reaction_slots = reaction_slots[:, : max(0, min(int(slot_count), reaction_slots.shape[1]))]
    for raw_action in reaction_slots.reshape(-1):
        action_index = int(raw_action)
        if action_index < 0:
            continue
        if action_index >= MAX_ACTIONS:
            return None
        used_indices.add(action_index)
    return used_indices



def _cached_used_action_indices_for_material_slots(
    pipeline,
    world: "WorldEngine",
    material_table: np.ndarray,
    *,
    slot_count: int | None = None,
) -> set[int] | None:
    key = (
        "material_slots",
        id(world.bridge),
        int(world.bridge.table_generations.get("materials", 0)),
        int(material_table.shape[0]),
        None if slot_count is None else int(slot_count),
    )
    if key not in pipeline._used_action_indices_cache:
        if len(pipeline._used_action_indices_cache) > 64:
            pipeline._used_action_indices_cache.clear()
        pipeline._used_action_indices_cache[key] = pipeline._used_action_indices_for_material_slots(
            material_table,
            slot_count=slot_count,
        )
    return pipeline._used_action_indices_cache[key]



def _used_action_indices_for_self_rules(
    pipeline,
    rule_table: np.ndarray,
    material_table: np.ndarray,
) -> set[int] | None:
    if "trigger_slot_index" not in rule_table.dtype.names or "reaction_slots" not in material_table.dtype.names:
        return None
    used_indices: set[int] = set()
    material_count = int(material_table.shape[0])
    for rule in rule_table:
        slot_index = int(rule["trigger_slot_index"])
        if slot_index < 0:
            continue
        if slot_index >= 8:
            return None
        material_id = int(rule["material_id"]) if "material_id" in rule_table.dtype.names else -1
        if material_id > 0:
            if material_id >= material_count:
                return None
            raw_actions = np.asarray(material_table["reaction_slots"][material_id : material_id + 1, slot_index])
        else:
            raw_actions = np.asarray(material_table["reaction_slots"][:, slot_index])
        for raw_action in np.asarray(raw_actions, dtype=np.int32).reshape(-1):
            action_index = int(raw_action)
            if action_index < 0:
                continue
            if action_index >= MAX_ACTIONS:
                return None
            used_indices.add(action_index)
    return used_indices



def _cached_used_action_indices_for_self_rules(
    pipeline,
    world: "WorldEngine",
    rule_table: np.ndarray,
    material_table: np.ndarray,
) -> set[int] | None:
    key = (
        "self_rules",
        id(world.bridge),
        int(world.bridge.table_generations.get("reactions", 0)),
        int(world.bridge.table_generations.get("materials", 0)),
        int(rule_table.shape[0]),
        int(material_table.shape[0]),
    )
    if key not in pipeline._used_action_indices_cache:
        if len(pipeline._used_action_indices_cache) > 64:
            pipeline._used_action_indices_cache.clear()
        pipeline._used_action_indices_cache[key] = pipeline._used_action_indices_for_self_rules(rule_table, material_table)
    return pipeline._used_action_indices_cache[key]



def _used_action_indices_for_pair_rules(
    pipeline,
    rule_table: np.ndarray,
    material_table: np.ndarray,
    *,
    lhs_tag_field: str,
) -> set[int] | None:
    used_indices = pipeline._used_action_indices(rule_table)
    if used_indices is None:
        return None
    if "trigger_slot_index" not in rule_table.dtype.names or "reaction_slots" not in material_table.dtype.names:
        return used_indices
    material_count = int(material_table.shape[0])
    for rule in rule_table:
        slot_index = int(rule["trigger_slot_index"])
        if slot_index < 0:
            continue
        if slot_index >= 8:
            return None
        lhs_material_id = int(rule["lhs_material_id"]) if "lhs_material_id" in rule_table.dtype.names else -1
        lhs_tag_mask = int(rule["lhs_tag_mask"]) if "lhs_tag_mask" in rule_table.dtype.names else 0
        if lhs_material_id > 0:
            if lhs_material_id >= material_count:
                return None
            candidates = material_table[lhs_material_id : lhs_material_id + 1]
        elif lhs_tag_mask != 0:
            if lhs_tag_field not in material_table.dtype.names:
                return None
            masks = np.asarray(material_table[lhs_tag_field], dtype=np.uint32)
            candidates = material_table[(masks & np.uint32(lhs_tag_mask)) == np.uint32(lhs_tag_mask)]
        else:
            candidates = material_table
        for raw_action in np.asarray(candidates["reaction_slots"][:, slot_index], dtype=np.int32).reshape(-1):
            action_index = int(raw_action)
            if action_index < 0:
                continue
            if action_index >= MAX_ACTIONS:
                return None
            used_indices.add(action_index)
    return used_indices



def _cached_used_action_indices_for_pair_rules(
    pipeline,
    world: "WorldEngine",
    rule_table: np.ndarray,
    material_table: np.ndarray,
    *,
    rule_kind: str,
    lhs_tag_field: str,
) -> set[int] | None:
    key = (
        "pair_rules",
        id(world.bridge),
        str(rule_kind),
        str(lhs_tag_field),
        int(world.bridge.table_generations.get("reactions", 0)),
        int(world.bridge.table_generations.get("materials", 0)),
        int(rule_table.shape[0]),
        int(material_table.shape[0]),
    )
    if key not in pipeline._used_action_indices_cache:
        if len(pipeline._used_action_indices_cache) > 64:
            pipeline._used_action_indices_cache.clear()
        pipeline._used_action_indices_cache[key] = pipeline._used_action_indices_for_pair_rules(
            rule_table,
            material_table,
            lhs_tag_field=lhs_tag_field,
        )
    return pipeline._used_action_indices_cache[key]



def _has_unsupported_consume_policies(rule_table: np.ndarray, supported_ids: set[int]) -> bool:
    if "consume_policy_id" not in rule_table.dtype.names:
        return False
    for raw_value in rule_table["consume_policy_id"].tolist():
        if int(raw_value) not in supported_ids:
            return True
    return False

# ``_set_uniform_if_present`` is inherited from GPUPipelineBase.
# ``_sync_compute_writes`` / ``_sync_storage_and_indirect_writes`` are kept
# as overrides: reactions uses a narrower barrier bit set (image-access |
# texture-fetch, no shader-storage) and a nullable-ctx guard.
