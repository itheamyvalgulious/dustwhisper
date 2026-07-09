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
    CONSUME_POLICY_RHS,
    FLOW_SOURCE_LAYERS,
    MAX_ACTIONS,
    MAX_MATERIALS,
    MAX_RULES,
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
    material_count = int(material_table.shape[0])
    max_flow_slot = -1
    for rule in rule_table:
        slot_index = int(rule["trigger_slot_index"])
        if slot_index < 0:
            continue
        if slot_index >= 8:
            return FLOW_SOURCE_LAYERS
        material_id = int(rule["material_id"]) if "material_id" in rule_table.dtype.names else -1
        if material_id > 0:
            if material_id >= material_count:
                return FLOW_SOURCE_LAYERS
            raw_actions = np.asarray(material_table["reaction_slots"][material_id : material_id + 1, slot_index])
        else:
            raw_actions = np.asarray(material_table["reaction_slots"][:, slot_index])
        for raw_action in np.asarray(raw_actions, dtype=np.int32).reshape(-1):
            action_index = int(raw_action)
            if action_index < 0:
                continue
            if action_index >= action_i.shape[0]:
                return FLOW_SOURCE_LAYERS
            ai = action_i[action_index]
            if int(ai[0]) == TYPE_MODIFY_GAS and int(ai[3]) != 0:
                max_flow_slot = max(max_flow_slot, slot_index)
    if max_flow_slot < 0:
        return FLOW_SOURCE_LAYERS
    return max(1, min(FLOW_SOURCE_LAYERS, (max_flow_slot + 1) * 4))



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
    for rule_index, rule in enumerate(rule_table[:count]):
        selector_id = int(rule[selector_id_field]) if selector_id_field in rule_field_names else -1
        if selector_id > 0:
            pipeline._set_rule_candidate(masks, selector_id, rule_index)
            continue
        selector_tag_mask = int(rule[selector_tag_field]) if selector_tag_field in rule_field_names else 0
        if selector_tag_mask != 0 and material_tag_field in material_field_names:
            tag_values = np.asarray(material_table[material_tag_field], dtype=np.uint32)
            required = np.uint32(selector_tag_mask)
            for material_id in range(1, material_count):
                if (tag_values[material_id] & required) == required:
                    pipeline._set_rule_candidate(masks, material_id, rule_index)
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

