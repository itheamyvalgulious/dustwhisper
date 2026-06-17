from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from oracle_game.types import (
    Direction,
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


class RuleBook:
    def __init__(self) -> None:
        self.materials_by_name: dict[str, MaterialDef] = {}
        self.materials_by_id: dict[int, MaterialDef] = {}
        self.gases_by_name: dict[str, GasSpeciesDef] = {}
        self.gases_by_id: dict[int, GasSpeciesDef] = {}
        self.lights_by_name: dict[str, LightTypeDef] = {}
        self.lights_by_id: dict[int, LightTypeDef] = {}
        self.optics: dict[tuple[str, str], MaterialOpticsDef] = {}
        self.reaction_actions: list[ReactionAction] = [ReactionAction(reaction_type=ReactionType.NONE)]
        self.material_material_rules: list[PairReactionRule] = []
        self.material_gas_rules: list[PairReactionRule] = []
        self.material_light_rules: list[PairReactionRule] = []
        self.gas_gas_rules: list[PairReactionRule] = []
        self.gas_light_rules: list[PairReactionRule] = []
        self.self_rules: list[SelfReactionRule] = []
        self.tag_bits: dict[str, int] = {}

    def _mask(self, tags: Iterable[str]) -> int:
        mask = 0
        for tag in tags:
            if tag not in self.tag_bits:
                self.tag_bits[tag] = 1 << len(self.tag_bits)
            mask |= self.tag_bits[tag]
        return mask

    def update_materials(self, materials: Iterable[MaterialDef]) -> None:
        for material in materials:
            material = replace(
                material,
                material_tag_mask=material.material_tag_mask or self._mask(material.tags),
                gas_tag_mask=material.gas_tag_mask,
                light_tag_mask=material.light_tag_mask,
            )
            self.materials_by_name[material.name] = material
            self.materials_by_id[material.material_id] = material

    def update_gases(self, gases: Iterable[GasSpeciesDef]) -> None:
        for gas in gases:
            self.gases_by_name[gas.name] = gas
            self.gases_by_id[gas.species_id] = gas

    def update_lights(self, lights: Iterable[LightTypeDef]) -> None:
        for light in lights:
            self.lights_by_name[light.name] = light
            self.lights_by_id[light.light_type_id] = light

    def update_optics(self, optics: Iterable[MaterialOpticsDef]) -> None:
        for entry in optics:
            self.optics[(entry.material_name, entry.light_type)] = entry

    def update_reaction_actions(self, actions: Iterable[ReactionAction]) -> None:
        for action in actions:
            self.reaction_actions.append(action)

    def update_reaction_rules(
        self,
        *,
        material_material: Iterable[PairReactionRule] = (),
        material_gas: Iterable[PairReactionRule] = (),
        material_light: Iterable[PairReactionRule] = (),
        gas_gas: Iterable[PairReactionRule] = (),
        gas_light: Iterable[PairReactionRule] = (),
        self_rules: Iterable[SelfReactionRule] = (),
    ) -> None:
        self.material_material_rules.extend(material_material)
        self.material_gas_rules.extend(material_gas)
        self.material_light_rules.extend(material_light)
        self.gas_gas_rules.extend(gas_gas)
        self.gas_light_rules.extend(gas_light)
        self.self_rules.extend(self_rules)

    def material_id(self, name: str) -> int:
        return self.materials_by_name[name].material_id

    def gas_id(self, name: str) -> int:
        return self.gases_by_name[name].species_id

    def light_id(self, name: str) -> int:
        return self.lights_by_name[name].light_type_id


def build_default_payloads() -> dict[str, object]:
    materials = _build_materials()
    gases = _build_gases()
    lights = _build_lights()
    actions = _build_actions()
    materials = _assign_reaction_slots(materials, actions)
    optics = build_default_optics_entries(materials, lights)
    rules = _build_rules()
    return {
        "materials": materials,
        "gases": gases,
        "lights": lights,
        "actions": actions,
        "optics": optics,
        "rules": rules,
    }


def _build_materials() -> list[MaterialDef]:
    defs: list[MaterialDef] = []

    def add(
        name: str,
        display_name: str,
        phase: Phase,
        color: tuple[float, float, float],
        *,
        render_group: str,
        density: float,
        gravity_scale: float,
        wind_coupling: float,
        drag_scale: float,
        friction: float,
        elasticity: float,
        max_dda_step: int,
        structural: bool,
        support_anchor: bool,
        collapse_generation: str | None,
        powder_generation: str | None,
        base_integrity: float,
        heat_capacity: float,
        conductivity: float,
        ambient_exchange_rate: float,
        melt_point: float | None = None,
        boil_point: float | None = None,
        melt_to_material: str | None = None,
        freeze_to_material: str | None = None,
        boil_to_gas_species: str | None = None,
        spawn_temperature: float | None = None,
        powder_solver_kind: str = "granular",
        liquid_solver_kind: str = "tile_level",
        falling_island_break_kind: str = "shear",
        collapse_behavior: str = "falling_island",
        tags: tuple[str, ...] = (),
    ) -> None:
        defs.append(
            MaterialDef(
                material_id=len(defs) + 1,
                name=name,
                display_name=display_name,
                default_phase=phase,
                render_group=render_group,
                base_color=color,
                density=density,
                gravity_scale=gravity_scale,
                wind_coupling=wind_coupling,
                drag_scale=drag_scale,
                friction=friction,
                elasticity=elasticity,
                max_dda_step=max_dda_step,
                powder_solver_kind=powder_solver_kind,
                liquid_solver_kind=liquid_solver_kind,
                falling_island_break_kind=falling_island_break_kind,
                is_structural=structural,
                is_support_anchor=support_anchor,
                collapse_behavior=collapse_behavior,
                collapse_generation=collapse_generation,
                powder_generation=powder_generation,
                base_integrity=base_integrity,
                heat_capacity=heat_capacity,
                conductivity=conductivity,
                ambient_exchange_rate=ambient_exchange_rate,
                melt_point=melt_point,
                boil_point=boil_point,
                melt_to_material=melt_to_material,
                freeze_to_material=freeze_to_material,
                boil_to_gas_species=boil_to_gas_species,
                spawn_temperature=spawn_temperature,
                tags=tags,
            )
        )

    add(
        "sand_powder",
        "沙子",
        Phase.POWDER,
        (0.82, 0.72, 0.45),
        render_group="powder",
        density=1.55,
        gravity_scale=1.0,
        wind_coupling=0.12,
        drag_scale=0.22,
        friction=0.55,
        elasticity=0.05,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=40,
        heat_capacity=0.65,
        conductivity=0.18,
        ambient_exchange_rate=0.08,
        melt_point=1250,
        melt_to_material="sandstone_solid",
        tags=("soil", "granular", "chaos_convert"),
    )
    add(
        "gravel_powder",
        "沙砾",
        Phase.POWDER,
        (0.55, 0.53, 0.50),
        render_group="powder",
        density=1.95,
        gravity_scale=1.05,
        wind_coupling=0.08,
        drag_scale=0.18,
        friction=0.6,
        elasticity=0.07,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=60,
        heat_capacity=0.75,
        conductivity=0.25,
        ambient_exchange_rate=0.05,
        tags=("stone", "granular"),
    )
    add(
        "soil_powder",
        "土壤",
        Phase.POWDER,
        (0.39, 0.27, 0.15),
        render_group="powder",
        density=1.25,
        gravity_scale=0.95,
        wind_coupling=0.14,
        drag_scale=0.24,
        friction=0.52,
        elasticity=0.04,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=35,
        heat_capacity=0.8,
        conductivity=0.15,
        ambient_exchange_rate=0.08,
        tags=("soil", "granular", "soft", "chaos_convert"),
    )
    add(
        "sandstone_solid",
        "砂岩",
        Phase.STATIC_SOLID,
        (0.71, 0.62, 0.36),
        render_group="stone",
        density=2.1,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.85,
        elasticity=0.05,
        max_dda_step=0,
        structural=True,
        support_anchor=True,
        collapse_generation="sandstone_solid",
        powder_generation="sand_powder",
        base_integrity=140,
        heat_capacity=0.9,
        conductivity=0.35,
        ambient_exchange_rate=0.04,
        melt_point=1450,
        tags=("stone", "structural", "soft"),
    )
    add(
        "raw_stone_solid",
        "原石",
        Phase.STATIC_SOLID,
        (0.42, 0.45, 0.48),
        render_group="stone",
        density=2.5,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.88,
        elasticity=0.03,
        max_dda_step=0,
        structural=True,
        support_anchor=True,
        collapse_generation="raw_stone_solid",
        powder_generation="gravel_powder",
        base_integrity=180,
        heat_capacity=1.0,
        conductivity=0.45,
        ambient_exchange_rate=0.03,
        melt_point=1600,
        tags=("stone", "structural"),
    )
    add(
        "obsidian_solid",
        "黑曜石",
        Phase.STATIC_SOLID,
        (0.10, 0.08, 0.14),
        render_group="stone",
        density=2.8,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.9,
        elasticity=0.02,
        max_dda_step=0,
        structural=True,
        support_anchor=True,
        collapse_generation="obsidian_solid",
        powder_generation="gravel_powder",
        base_integrity=320,
        heat_capacity=1.2,
        conductivity=0.22,
        ambient_exchange_rate=0.02,
        melt_point=2200,
        tags=("stone", "structural", "acid_immune"),
    )
    add(
        "root_solid",
        "树根",
        Phase.STATIC_SOLID,
        (0.44, 0.30, 0.15),
        render_group="plant",
        density=1.05,
        gravity_scale=0.0,
        wind_coupling=0.04,
        drag_scale=0.01,
        friction=0.8,
        elasticity=0.03,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="root_solid",
        powder_generation="soil_powder",
        base_integrity=90,
        heat_capacity=0.7,
        conductivity=0.10,
        ambient_exchange_rate=0.12,
        melt_point=260,
        tags=("plant", "flammable", "soft", "chaos_convert"),
    )
    add(
        "log_solid",
        "原木",
        Phase.STATIC_SOLID,
        (0.58, 0.37, 0.17),
        render_group="plant",
        density=0.95,
        gravity_scale=0.0,
        wind_coupling=0.03,
        drag_scale=0.01,
        friction=0.75,
        elasticity=0.04,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="log_solid",
        powder_generation="soil_powder",
        base_integrity=110,
        heat_capacity=0.75,
        conductivity=0.11,
        ambient_exchange_rate=0.12,
        melt_point=280,
        tags=("plant", "flammable", "soft", "chaos_convert"),
    )
    add(
        "leaf_powder",
        "树叶",
        Phase.POWDER,
        (0.32, 0.68, 0.24),
        render_group="plant",
        density=0.35,
        gravity_scale=0.7,
        wind_coupling=0.4,
        drag_scale=0.35,
        friction=0.25,
        elasticity=0.08,
        max_dda_step=3,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=22,
        heat_capacity=0.35,
        conductivity=0.08,
        ambient_exchange_rate=0.18,
        tags=("plant", "flammable", "soft"),
    )
    add(
        "vine_solid",
        "藤蔓",
        Phase.STATIC_SOLID,
        (0.20, 0.52, 0.17),
        render_group="plant",
        density=0.65,
        gravity_scale=0.0,
        wind_coupling=0.05,
        drag_scale=0.02,
        friction=0.65,
        elasticity=0.05,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="vine_solid",
        powder_generation="leaf_powder",
        base_integrity=55,
        heat_capacity=0.55,
        conductivity=0.09,
        ambient_exchange_rate=0.15,
        melt_point=220,
        tags=("plant", "flammable", "soft"),
    )
    add(
        "moss_powder",
        "苔藓",
        Phase.POWDER,
        (0.34, 0.58, 0.16),
        render_group="plant",
        density=0.45,
        gravity_scale=0.65,
        wind_coupling=0.25,
        drag_scale=0.32,
        friction=0.3,
        elasticity=0.05,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=18,
        heat_capacity=0.42,
        conductivity=0.07,
        ambient_exchange_rate=0.2,
        tags=("plant", "flammable", "soft", "chaos_convert"),
    )
    add(
        "iron_solid",
        "铁",
        Phase.STATIC_SOLID,
        (0.60, 0.64, 0.70),
        render_group="metal",
        density=4.0,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.92,
        elasticity=0.10,
        max_dda_step=0,
        structural=True,
        support_anchor=True,
        collapse_generation="iron_solid",
        powder_generation="iron_powder",
        base_integrity=380,
        heat_capacity=1.1,
        conductivity=0.78,
        ambient_exchange_rate=0.04,
        melt_point=1538,
        melt_to_material="iron_liquid",
        tags=("metal", "structural"),
    )
    add(
        "iron_powder",
        "铁屑",
        Phase.POWDER,
        (0.52, 0.56, 0.62),
        render_group="metal",
        density=3.6,
        gravity_scale=0.82,
        wind_coupling=0.03,
        drag_scale=0.12,
        friction=0.42,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=180,
        heat_capacity=1.0,
        conductivity=0.70,
        ambient_exchange_rate=0.05,
        melt_point=1538,
        melt_to_material="iron_liquid",
        tags=("metal", "powder"),
    )
    add(
        "iron_liquid",
        "熔铁",
        Phase.LIQUID,
        (0.96, 0.56, 0.18),
        render_group="liquid",
        density=3.9,
        gravity_scale=0.85,
        wind_coupling=0.04,
        drag_scale=0.08,
        friction=0.10,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=340,
        heat_capacity=1.0,
        conductivity=0.74,
        ambient_exchange_rate=0.06,
        melt_point=1538,
        freeze_to_material="iron_solid",
        liquid_solver_kind="tile_level",
        tags=("metal", "liquid", "hot"),
    )
    add(
        "gold_solid",
        "金",
        Phase.STATIC_SOLID,
        (0.92, 0.75, 0.18),
        render_group="metal",
        density=5.2,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.90,
        elasticity=0.08,
        max_dda_step=0,
        structural=True,
        support_anchor=True,
        collapse_generation="gold_solid",
        powder_generation="gold_powder",
        base_integrity=260,
        heat_capacity=1.05,
        conductivity=0.82,
        ambient_exchange_rate=0.03,
        melt_point=1064,
        melt_to_material="gold_liquid",
        tags=("metal", "structural", "acid_immune"),
    )
    add(
        "gold_powder",
        "金粉",
        Phase.POWDER,
        (0.90, 0.71, 0.18),
        render_group="metal",
        density=4.8,
        gravity_scale=0.80,
        wind_coupling=0.02,
        drag_scale=0.10,
        friction=0.40,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=140,
        heat_capacity=1.0,
        conductivity=0.76,
        ambient_exchange_rate=0.04,
        melt_point=1064,
        melt_to_material="gold_liquid",
        tags=("metal", "powder", "acid_immune"),
    )
    add(
        "gold_liquid",
        "熔金",
        Phase.LIQUID,
        (1.00, 0.74, 0.16),
        render_group="liquid",
        density=5.0,
        gravity_scale=0.85,
        wind_coupling=0.03,
        drag_scale=0.08,
        friction=0.09,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=240,
        heat_capacity=0.98,
        conductivity=0.78,
        ambient_exchange_rate=0.05,
        melt_point=1064,
        freeze_to_material="gold_solid",
        liquid_solver_kind="tile_level",
        tags=("metal", "liquid", "hot", "acid_immune"),
    )
    add(
        "water_liquid",
        "水",
        Phase.LIQUID,
        (0.18, 0.42, 0.82),
        render_group="liquid",
        density=1.0,
        gravity_scale=0.85,
        wind_coupling=0.24,
        drag_scale=0.10,
        friction=0.12,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=100,
        heat_capacity=1.5,
        conductivity=0.42,
        ambient_exchange_rate=0.28,
        melt_point=0,
        boil_point=100,
        freeze_to_material="water_solid",
        boil_to_gas_species="water_gas",
        liquid_solver_kind="tile_level",
        tags=("liquid", "coolant"),
    )
    add(
        "water_solid",
        "冰",
        Phase.STATIC_SOLID,
        (0.76, 0.88, 0.96),
        render_group="liquid",
        density=0.92,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.08,
        elasticity=0.02,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="water_solid",
        powder_generation="water_powder",
        base_integrity=90,
        heat_capacity=1.3,
        conductivity=0.35,
        ambient_exchange_rate=0.22,
        melt_point=0,
        melt_to_material="water_liquid",
        tags=("structural", "cold"),
    )
    add(
        "water_powder",
        "雪",
        Phase.POWDER,
        (0.88, 0.93, 0.98),
        render_group="powder",
        density=0.35,
        gravity_scale=0.78,
        wind_coupling=0.18,
        drag_scale=0.28,
        friction=0.18,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=52,
        heat_capacity=1.1,
        conductivity=0.18,
        ambient_exchange_rate=0.18,
        melt_point=0,
        melt_to_material="water_liquid",
        boil_point=100,
        boil_to_gas_species="water_gas",
        tags=("cold", "granular"),
    )
    add(
        "poison_liquid",
        "毒液",
        Phase.LIQUID,
        (0.42, 0.82, 0.10),
        render_group="liquid",
        density=1.08,
        gravity_scale=0.85,
        wind_coupling=0.16,
        drag_scale=0.10,
        friction=0.10,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=100,
        heat_capacity=0.95,
        conductivity=0.18,
        ambient_exchange_rate=0.12,
        melt_point=-20,
        freeze_to_material="poison_solid",
        boil_point=115,
        boil_to_gas_species="poison_gas",
        liquid_solver_kind="tile_level",
        tags=("liquid", "poison"),
    )
    add(
        "poison_solid",
        "毒冰",
        Phase.STATIC_SOLID,
        (0.68, 0.92, 0.24),
        render_group="liquid",
        density=1.02,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.06,
        elasticity=0.01,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="poison_solid",
        powder_generation="poison_powder",
        base_integrity=84,
        heat_capacity=0.90,
        conductivity=0.16,
        ambient_exchange_rate=0.10,
        melt_point=-20,
        melt_to_material="poison_liquid",
        tags=("structural", "poison", "cold"),
    )
    add(
        "poison_powder",
        "毒霜",
        Phase.POWDER,
        (0.74, 0.95, 0.30),
        render_group="powder",
        density=0.42,
        gravity_scale=0.78,
        wind_coupling=0.20,
        drag_scale=0.26,
        friction=0.18,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=56,
        heat_capacity=0.82,
        conductivity=0.14,
        ambient_exchange_rate=0.12,
        melt_point=-20,
        melt_to_material="poison_liquid",
        boil_point=115,
        boil_to_gas_species="poison_gas",
        tags=("poison", "cold", "granular"),
    )
    add(
        "acid_liquid",
        "酸液",
        Phase.LIQUID,
        (0.95, 0.86, 0.20),
        render_group="liquid",
        density=1.12,
        gravity_scale=0.85,
        wind_coupling=0.12,
        drag_scale=0.12,
        friction=0.09,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=100,
        heat_capacity=0.82,
        conductivity=0.14,
        ambient_exchange_rate=0.10,
        melt_point=-40,
        freeze_to_material="acid_solid",
        boil_point=108,
        boil_to_gas_species=None,
        liquid_solver_kind="tile_level",
        tags=("liquid", "acid"),
    )
    add(
        "acid_solid",
        "酸冰",
        Phase.STATIC_SOLID,
        (0.98, 0.94, 0.42),
        render_group="liquid",
        density=1.05,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.05,
        elasticity=0.01,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="acid_solid",
        powder_generation="acid_powder",
        base_integrity=88,
        heat_capacity=0.78,
        conductivity=0.12,
        ambient_exchange_rate=0.08,
        melt_point=-40,
        melt_to_material="acid_liquid",
        tags=("structural", "acid", "cold"),
    )
    add(
        "acid_powder",
        "酸霜",
        Phase.POWDER,
        (0.98, 0.96, 0.56),
        render_group="powder",
        density=0.46,
        gravity_scale=0.80,
        wind_coupling=0.18,
        drag_scale=0.24,
        friction=0.16,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=60,
        heat_capacity=0.74,
        conductivity=0.10,
        ambient_exchange_rate=0.08,
        melt_point=-40,
        melt_to_material="acid_liquid",
        boil_point=108,
        boil_to_gas_species=None,
        tags=("acid", "cold", "granular"),
    )
    add(
        "oil_liquid",
        "油",
        Phase.LIQUID,
        (0.30, 0.22, 0.06),
        render_group="liquid",
        density=0.78,
        gravity_scale=0.85,
        wind_coupling=0.20,
        drag_scale=0.12,
        friction=0.10,
        elasticity=0.0,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=100,
        heat_capacity=0.70,
        conductivity=0.08,
        ambient_exchange_rate=0.07,
        melt_point=-50,
        freeze_to_material="oil_solid",
        boil_point=90,
        boil_to_gas_species="oil_gas",
        liquid_solver_kind="tile_level",
        tags=("liquid", "flammable"),
    )
    add(
        "oil_solid",
        "凝油",
        Phase.STATIC_SOLID,
        (0.48, 0.38, 0.16),
        render_group="liquid",
        density=0.82,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.04,
        elasticity=0.01,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="oil_solid",
        powder_generation="oil_powder",
        base_integrity=72,
        heat_capacity=0.66,
        conductivity=0.06,
        ambient_exchange_rate=0.05,
        melt_point=-50,
        melt_to_material="oil_liquid",
        tags=("structural", "flammable", "cold"),
    )
    add(
        "oil_powder",
        "油晶",
        Phase.POWDER,
        (0.54, 0.42, 0.18),
        render_group="powder",
        density=0.38,
        gravity_scale=0.76,
        wind_coupling=0.18,
        drag_scale=0.22,
        friction=0.14,
        elasticity=0.02,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=48,
        heat_capacity=0.60,
        conductivity=0.06,
        ambient_exchange_rate=0.05,
        melt_point=-50,
        melt_to_material="oil_liquid",
        boil_point=90,
        boil_to_gas_species="oil_gas",
        tags=("flammable", "cold", "granular"),
    )
    add(
        "explosive_solid",
        "炸药",
        Phase.STATIC_SOLID,
        (0.80, 0.18, 0.15),
        render_group="special",
        density=1.35,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.72,
        elasticity=0.02,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="explosive_solid",
        powder_generation="soil_powder",
        base_integrity=55,
        heat_capacity=0.45,
        conductivity=0.10,
        ambient_exchange_rate=0.10,
        melt_point=140,
        tags=("flammable", "special", "soft"),
    )
    add(
        "phosphor_visible_powder",
        "可见荧光粉",
        Phase.POWDER,
        (0.92, 0.90, 0.68),
        render_group="special",
        density=0.62,
        gravity_scale=0.75,
        wind_coupling=0.20,
        drag_scale=0.18,
        friction=0.22,
        elasticity=0.03,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=32,
        heat_capacity=0.30,
        conductivity=0.05,
        ambient_exchange_rate=0.20,
        tags=("phosphor", "powder"),
    )
    add(
        "phosphor_holy_powder",
        "圣荧光粉",
        Phase.POWDER,
        (0.95, 0.95, 0.78),
        render_group="special",
        density=0.62,
        gravity_scale=0.75,
        wind_coupling=0.20,
        drag_scale=0.18,
        friction=0.22,
        elasticity=0.03,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=32,
        heat_capacity=0.30,
        conductivity=0.05,
        ambient_exchange_rate=0.20,
        tags=("phosphor", "powder"),
    )
    add(
        "phosphor_chaos_powder",
        "混沌荧光粉",
        Phase.POWDER,
        (0.90, 0.52, 0.12),
        render_group="special",
        density=0.62,
        gravity_scale=0.75,
        wind_coupling=0.20,
        drag_scale=0.18,
        friction=0.22,
        elasticity=0.03,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=32,
        heat_capacity=0.30,
        conductivity=0.05,
        ambient_exchange_rate=0.20,
        tags=("phosphor", "powder"),
    )
    add(
        "phosphor_magic_powder",
        "魔法荧光粉",
        Phase.POWDER,
        (0.16, 0.92, 0.90),
        render_group="special",
        density=0.62,
        gravity_scale=0.75,
        wind_coupling=0.20,
        drag_scale=0.18,
        friction=0.22,
        elasticity=0.03,
        max_dda_step=2,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=32,
        heat_capacity=0.30,
        conductivity=0.05,
        ambient_exchange_rate=0.20,
        tags=("phosphor", "powder"),
    )
    add(
        "pollution_powder",
        "污染物",
        Phase.POWDER,
        (0.42, 0.14, 0.44),
        render_group="special",
        density=0.28,
        gravity_scale=0.0,
        wind_coupling=0.35,
        drag_scale=0.06,
        friction=0.01,
        elasticity=0.0,
        max_dda_step=3,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=48,
        heat_capacity=0.40,
        conductivity=0.03,
        ambient_exchange_rate=0.22,
        boil_point=80,
        boil_to_gas_species="pollution_gas",
        powder_solver_kind="suspended",
        tags=("pollution", "powder", "chaos_convert"),
    )
    add(
        "vortex_heart_solid",
        "漩涡之心",
        Phase.STATIC_SOLID,
        (0.20, 0.80, 0.90),
        render_group="special",
        density=1.8,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.50,
        elasticity=0.2,
        max_dda_step=0,
        structural=True,
        support_anchor=False,
        collapse_generation="vortex_heart_solid",
        powder_generation="gravel_powder",
        base_integrity=160,
        heat_capacity=1.2,
        conductivity=0.5,
        ambient_exchange_rate=0.16,
        tags=("special",),
    )
    add(
        "fire_powder",
        "火焰",
        Phase.POWDER,
        (1.00, 0.38, 0.10),
        render_group="special",
        density=0.08,
        gravity_scale=-0.25,
        wind_coupling=0.5,
        drag_scale=0.25,
        friction=0.0,
        elasticity=0.0,
        max_dda_step=3,
        structural=False,
        support_anchor=False,
        collapse_generation=None,
        powder_generation=None,
        base_integrity=12,
        heat_capacity=0.12,
        conductivity=0.05,
        ambient_exchange_rate=0.4,
        spawn_temperature=180.0,
        tags=("fire", "hot"),
    )
    add(
        "placeholder_solid",
        "占位体",
        Phase.STATIC_SOLID,
        (0.75, 0.12, 0.55),
        render_group="placeholder",
        density=3.0,
        gravity_scale=0.0,
        wind_coupling=0.0,
        drag_scale=0.0,
        friction=0.92,
        elasticity=0.04,
        max_dda_step=0,
        structural=False,
        support_anchor=False,
        collapse_generation="placeholder_solid",
        powder_generation="soil_powder",
        base_integrity=240,
        heat_capacity=1.0,
        conductivity=0.2,
        ambient_exchange_rate=0.12,
        tags=("placeholder", "soft"),
    )
    return defs


def _build_gases() -> list[GasSpeciesDef]:
    return [
        GasSpeciesDef(0, "air", "空气", (0.75, 0.78, 0.85), 0.18, 0.02, 0.0, 0.0, None, None, 1.0, 1.0),
        GasSpeciesDef(1, "water_gas", "水蒸气", (0.65, 0.80, 0.95), 0.35, 0.25, 0.01, 0.06, 95.0, "water_liquid", 0.9, 0.7),
        GasSpeciesDef(2, "poison_gas", "毒雾", (0.55, 0.90, 0.20), 0.28, 0.15, 0.02, 0.03, 40.0, "poison_liquid", 1.0, 1.1),
        GasSpeciesDef(3, "oil_gas", "油雾", (0.45, 0.33, 0.18), 0.22, 0.08, 0.02, 0.02, 55.0, "oil_liquid", 1.15, 1.25),
        GasSpeciesDef(4, "pollution_gas", "污染气", (0.70, 0.18, 0.82), 0.25, 0.12, 0.015, 0.04, 50.0, "pollution_powder", 1.05, 0.95),
        GasSpeciesDef(5, "fire_gas", "焰气", (1.0, 0.42, 0.10), 0.40, 0.45, 0.08, 0.12, None, None, 0.85, 0.18),
    ]


def _build_lights() -> list[LightTypeDef]:
    return [
        LightTypeDef(0, "visible_light", "可见光", (1.0, 0.96, 0.82), 0, 24, 1, 0, "diffuse"),
        LightTypeDef(1, "holy_light", "圣光", (0.92, 1.0, 0.86), 1, 22, 1, 1, "holy"),
        LightTypeDef(2, "chaos_light", "混沌光", (1.0, 0.45, 0.18), 2, 20, 2, 2, "chaos"),
        LightTypeDef(3, "magic_light", "魔法光", (0.20, 0.95, 1.0), 3, 24, 1, 3, "magic"),
    ]


def _build_actions() -> list[ReactionAction]:
    return [
        ReactionAction(ReactionType.EMIT_MATERIAL, emit_material="root_solid", direction=Direction.DOWN, duration=4, speed=0.0),
        ReactionAction(ReactionType.EMIT_MATERIAL, emit_material="log_solid", direction=Direction.UP, duration=4, speed=0.0),
        ReactionAction(ReactionType.EMIT_MATERIAL, emit_material="leaf_powder", direction=Direction.RANDOM, duration=2, speed=1.0),
        ReactionAction(ReactionType.HARM, value=1.5, duration=1),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="fire_powder",
            harm_per_frame=3.0,
            integrity_threshold=20.0,
        ),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material=None,
            harm_per_frame=2.0,
            integrity_threshold=8.0,
        ),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="fire_powder",
            harm_per_frame=5.0,
            integrity_threshold=40.0,
        ),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="fire_powder",
            harm_per_frame=14.0,
            integrity_threshold=30.0,
        ),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="oil_gas", speed=4.0, duration=8),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material=None,
            harm_per_frame=4.0,
            integrity_threshold=10.0,
        ),
        ReactionAction(ReactionType.HARM, value=2.5, duration=0),
        ReactionAction(ReactionType.EMIT_LIGHT, light_type="visible_light", strength=1.0, duration=1, range_cells=16),
        ReactionAction(ReactionType.EMIT_LIGHT, light_type="holy_light", strength=1.1, duration=1, range_cells=16),
        ReactionAction(ReactionType.EMIT_LIGHT, light_type="chaos_light", strength=1.1, duration=1, range_cells=16),
        ReactionAction(ReactionType.EMIT_LIGHT, light_type="magic_light", strength=1.1, duration=1, range_cells=16),
        ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=-1.5, duration=1),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material=None,
            harm_per_frame=5.0,
            integrity_threshold=16.0,
            allow_subunit_scale=True,
        ),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="__random__",
            harm_per_frame=1.0,
            integrity_threshold=14.0,
        ),
        ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=3.0, duration=1),
        ReactionAction(
            ReactionType.CONVERT_MATERIAL,
            target_material="pollution_powder",
            harm_per_frame=1.5,
            integrity_threshold=12.0,
        ),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="air", speed=-2.5, duration=1, strength=2.4, range_cells=14),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="air", speed=1.5, duration=1, strength=1.2, range_cells=8),
        ReactionAction(ReactionType.HARM, value=-1.5, duration=1),
        ReactionAction(ReactionType.MODIFY_TEMPERATURE, delta=12.0, duration=1),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="poison_gas", speed=1.5, duration=1),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="water_gas", speed=1.5, duration=1),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="pollution_gas", speed=-10.0, duration=1),
        ReactionAction(ReactionType.MODIFY_GAS, gas_species="fire_gas", speed=2.0, duration=1),
    ]


def _assign_reaction_slots(materials: list[MaterialDef], actions: list[ReactionAction]) -> list[MaterialDef]:
    slot_map = {
        "sandstone_solid": {0: 10},
        "raw_stone_solid": {0: 6},
        "root_solid": {0: 1, 1: 11, 2: 20},
        "log_solid": {0: 2, 1: 3, 2: 11, 3: 20},
        "leaf_powder": {0: 11, 1: 20, 2: 17},
        "vine_solid": {0: 11, 1: 20, 2: 17},
        "moss_powder": {0: 11, 1: 20, 2: 17},
        "soil_powder": {0: 20, 1: 10},
        "sand_powder": {0: 20, 1: 10},
        "iron_solid": {0: 10},
        "iron_powder": {0: 17},
        "iron_liquid": {0: 17},
        "oil_liquid": {0: 7},
        "explosive_solid": {0: 8, 1: 9, 4: 24},
        "poison_liquid": {4: 25},
        "fire_powder": {0: 6, 3: 28, 4: 12, 5: 24},
        "phosphor_visible_powder": {4: 12},
        "phosphor_holy_powder": {4: 13},
        "phosphor_chaos_powder": {4: 14},
        "phosphor_magic_powder": {4: 15},
        "vortex_heart_solid": {4: 21, 5: 22},
        "water_liquid": {4: 26},
        "placeholder_solid": {0: 11, 1: 20, 2: 17},
    }
    fire_reactive_materials = {
        "root_solid",
        "log_solid",
        "leaf_powder",
        "vine_solid",
        "moss_powder",
        "oil_solid",
        "oil_powder",
        "placeholder_solid",
    }
    light_temperature_materials = {
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
    }
    updated: list[MaterialDef] = []
    for material in materials:
        slots = list(material.reaction_slots)
        for slot_index, action_index in slot_map.get(material.name, {}).items():
            slots[slot_index] = action_index
        if material.name in fire_reactive_materials:
            slots[4] = 4
            slots[5] = 5
        if material.name == "root_solid":
            slots[3] = 10
        if material.name in light_temperature_materials:
            slots[6] = 19
            slots[7] = 16
        updated.append(replace(material, reaction_slots=tuple(slots)))
    return updated


def _default_optics_values(material: MaterialDef, light: LightTypeDef) -> tuple[float, float, float]:
    defaults = {
        "powder": (0.20, 0.25, 0.00),
        "stone": (0.35, 0.08, 0.05),
        "plant": (0.28, 0.18, 0.02),
        "metal": (0.18, 0.10, 0.45),
        "liquid": (0.08, 0.10, 0.18),
        "special": (0.18, 0.22, 0.06),
        "placeholder": (0.30, 0.00, 0.00),
    }
    absorb, scatter, refract = defaults.get(material.render_group, (0.2, 0.1, 0.0))
    if material.name == "gold_solid" and light.name == "visible_light":
        absorb, scatter, refract = (0.10, 0.12, 0.65)
    if material.name == "obsidian_solid":
        absorb, scatter, refract = (0.60, 0.05, 0.02)
    if material.name.startswith("phosphor_"):
        absorb, scatter, refract = (0.10, 0.25, 0.05)
    return (absorb, scatter, refract)


def build_default_optics_entries(
    materials: Iterable[MaterialDef],
    lights: Iterable[LightTypeDef],
    *,
    existing: dict[tuple[str, str], MaterialOpticsDef] | None = None,
) -> list[MaterialOpticsDef]:
    material_list = list(materials)
    light_list = list(lights)
    existing = {} if existing is None else dict(existing)
    entries: list[MaterialOpticsDef] = []
    for material in material_list:
        for light in light_list:
            entry = existing.get((material.name, light.name))
            if entry is None:
                absorb, scatter, refract = _default_optics_values(material, light)
                entry = MaterialOpticsDef(
                    material_name=material.name,
                    light_type=light.name,
                    absorption=absorb,
                    scattering=scatter,
                    refraction=refract,
                )
            entries.append(entry)
    return entries


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
