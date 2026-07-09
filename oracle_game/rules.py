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
from oracle_game.rules_materials import (
    _build_terrain_materials,
    _build_stone_materials,
    _build_plant_materials,
    _build_metal_materials,
    _build_water_materials,
    _build_poison_materials,
    _build_acid_materials,
    _build_oil_materials,
    _build_special_materials,
)
from oracle_game.rules_reactions import _build_rules


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

    _build_terrain_materials(add)
    _build_stone_materials(add)
    _build_plant_materials(add)
    _build_metal_materials(add)
    _build_water_materials(add)
    _build_poison_materials(add)
    _build_acid_materials(add)
    _build_oil_materials(add)
    _build_special_materials(add)
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

