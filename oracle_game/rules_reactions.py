from __future__ import annotations

from oracle_game.types import PairReactionRule, SelfReactionRule


def _build_rules() -> dict[str, object]:
    MM = []
    MG = []
    ML = []
    GG = []
    GL = []
    SR = []

    ordinary_flammables = (
        "root_solid",
        "log_solid",
        "leaf_powder",
        "vine_solid",
        "moss_powder",
        "oil_solid",
        "oil_powder",
        "placeholder_solid",
    )
    for name in ordinary_flammables:
        MM.append(
            PairReactionRule(
                lhs_material=name,
                rhs_material="fire_powder",
                min_temperature=30.0,
                max_temperature=90.0,
                trigger_slot_index=4,
            )
        )
        MM.append(
            PairReactionRule(lhs_material=name, rhs_material="fire_powder", min_temperature=90.0, trigger_slot_index=5)
        )
        MG.append(
            PairReactionRule(
                lhs_material=name,
                rhs_gas="fire_gas",
                threshold=0.08,
                min_temperature=30.0,
                max_temperature=90.0,
                trigger_slot_index=4,
            )
        )
        MG.append(
            PairReactionRule(
                lhs_material=name,
                rhs_gas="fire_gas",
                threshold=0.08,
                min_temperature=90.0,
                trigger_slot_index=5,
            )
        )
    MM.append(PairReactionRule(lhs_material="explosive_solid", rhs_material="fire_powder", min_temperature=30.0, result_action=24))
    MM.append(PairReactionRule(lhs_material="explosive_solid", rhs_material="fire_powder", min_temperature=30.0, result_action=9))
    MM.append(
        PairReactionRule(
            lhs_material="explosive_solid",
            rhs_material="fire_powder",
            min_temperature=30.0,
            max_temperature=90.0,
            result_action=4,
        )
    )
    MM.append(PairReactionRule(lhs_material="explosive_solid", rhs_material="fire_powder", min_temperature=90.0, result_action=5))
    MG.append(PairReactionRule(lhs_material="explosive_solid", rhs_gas="fire_gas", threshold=0.08, min_temperature=30.0, result_action=24))
    MG.append(PairReactionRule(lhs_material="explosive_solid", rhs_gas="fire_gas", threshold=0.08, min_temperature=30.0, result_action=9))
    MG.append(
        PairReactionRule(
            lhs_material="explosive_solid",
            rhs_gas="fire_gas",
            threshold=0.08,
            min_temperature=30.0,
            max_temperature=90.0,
            result_action=4,
        )
    )
    MG.append(PairReactionRule(lhs_material="explosive_solid", rhs_gas="fire_gas", threshold=0.08, min_temperature=90.0, result_action=5))

    acid_slot_map = {
        "raw_stone_solid": 0,
        "sand_powder": 1,
        "soil_powder": 1,
        "sandstone_solid": 0,
        "root_solid": 3,
        "leaf_powder": 2,
        "vine_solid": 2,
        "moss_powder": 2,
        "iron_solid": 0,
        "iron_powder": 0,
        "iron_liquid": 0,
        "placeholder_solid": 2,
    }
    for name, slot_index in acid_slot_map.items():
        MM.append(PairReactionRule(lhs_material=name, rhs_material="acid_liquid", trigger_slot_index=slot_index))
    MM.append(PairReactionRule(lhs_material="log_solid", rhs_material="acid_liquid", result_action=10))
    poison_slot_map = {
        "root_solid": 1,
        "log_solid": 2,
        "leaf_powder": 0,
        "vine_solid": 0,
        "moss_powder": 0,
        "placeholder_solid": 0,
    }
    for name, slot_index in poison_slot_map.items():
        MM.append(PairReactionRule(lhs_material=name, rhs_material="poison_liquid", trigger_slot_index=slot_index))
        MG.append(PairReactionRule(lhs_material=name, rhs_gas="poison_gas", threshold=0.12, trigger_slot_index=slot_index))
    pollution_slot_map = {
        "root_solid": 2,
        "log_solid": 3,
        "leaf_powder": 1,
        "vine_solid": 1,
        "moss_powder": 1,
        "soil_powder": 0,
        "sand_powder": 0,
        "placeholder_solid": 1,
    }
    for name, slot_index in pollution_slot_map.items():
        MM.append(PairReactionRule(lhs_material=name, rhs_material="pollution_powder", trigger_slot_index=slot_index))
    MG.append(PairReactionRule(lhs_material="fire_powder", rhs_gas="oil_gas", threshold=0.18, result_action=7))

    all_light_materials = (
        "sand_powder",
        "gravel_powder",
        "soil_powder",
        "sandstone_solid",
        "raw_stone_solid",
        "obsidian_solid",
        "root_solid",
        "log_solid",
        "leaf_powder",
        "vine_solid",
        "moss_powder",
        "iron_powder",
        "iron_solid",
        "iron_liquid",
        "gold_solid",
        "gold_powder",
        "gold_liquid",
        "water_powder",
        "water_liquid",
        "water_solid",
        "poison_powder",
        "poison_liquid",
        "poison_solid",
        "acid_powder",
        "acid_liquid",
        "acid_solid",
        "oil_powder",
        "oil_liquid",
        "oil_solid",
        "explosive_solid",
        "phosphor_visible_powder",
        "phosphor_holy_powder",
        "phosphor_chaos_powder",
        "phosphor_magic_powder",
        "pollution_powder",
        "vortex_heart_solid",
        "fire_powder",
        "placeholder_solid",
    )
    holy_restoration_materials = tuple(name for name in all_light_materials if name != "pollution_powder")
    for name in all_light_materials:
        ML.append(PairReactionRule(lhs_material=name, rhs_light="visible_light", threshold=0.05, trigger_slot_index=6))
        ML.append(PairReactionRule(lhs_material=name, rhs_light="holy_light", threshold=0.05, trigger_slot_index=7))
        ML.append(PairReactionRule(lhs_material=name, rhs_light="chaos_light", threshold=0.05, trigger_slot_index=6))
    for name in holy_restoration_materials:
        ML.append(PairReactionRule(lhs_material=name, rhs_light="holy_light", threshold=0.05, result_action=23))
    ML.append(PairReactionRule(lhs_material="pollution_powder", rhs_light="holy_light", threshold=0.03, result_action=17))
    ML.append(PairReactionRule(lhs_material="placeholder_solid", rhs_light="chaos_light", threshold=0.05, result_action=11))
    for name in all_light_materials:
        ML.append(PairReactionRule(lhs_material=name, rhs_light="chaos_light", threshold=0.12, result_action=18))

    GL.append(PairReactionRule(rhs_gas="pollution_gas", rhs_light="holy_light", threshold=0.10, result_action=27))

    SR.extend(
        [
            SelfReactionRule(material="root_solid", trigger_slot_index=0, max_temperature=50.0),
            SelfReactionRule(material="log_solid", trigger_slot_index=0, max_temperature=55.0),
            SelfReactionRule(material="log_solid", trigger_slot_index=1, max_temperature=55.0),
            SelfReactionRule(material="oil_liquid", trigger_slot_index=0, min_temperature=120.0),
            SelfReactionRule(material="explosive_solid", trigger_slot_index=0, min_temperature=90.0),
            SelfReactionRule(material="explosive_solid", trigger_slot_index=1, min_temperature=90.0),
            SelfReactionRule(material="explosive_solid", trigger_slot_index=4, min_temperature=90.0),
            SelfReactionRule(material="poison_liquid", trigger_slot_index=4),
            SelfReactionRule(material="fire_powder", trigger_slot_index=0, max_temperature=25.0),
            SelfReactionRule(material="fire_powder", trigger_slot_index=3, min_temperature=25.0),
            SelfReactionRule(material="fire_powder", trigger_slot_index=4, min_temperature=25.0),
            SelfReactionRule(material="fire_powder", trigger_slot_index=5, min_temperature=25.0),
            SelfReactionRule(material="phosphor_visible_powder", trigger_slot_index=4),
            SelfReactionRule(material="phosphor_holy_powder", trigger_slot_index=4),
            SelfReactionRule(material="phosphor_chaos_powder", trigger_slot_index=4),
            SelfReactionRule(material="phosphor_magic_powder", trigger_slot_index=4),
            SelfReactionRule(material="vortex_heart_solid", trigger_slot_index=4),
            SelfReactionRule(material="vortex_heart_solid", trigger_slot_index=5),
            SelfReactionRule(material="water_liquid", trigger_slot_index=4, min_temperature=100.0),
        ]
    )

    return {
        "material_material": MM,
        "material_gas": MG,
        "material_light": ML,
        "gas_gas": GG,
        "gas_light": GL,
        "self_rules": SR,
    }
