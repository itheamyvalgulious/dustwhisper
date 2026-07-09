from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.gpu import DIRECTION_IDS
from oracle_game.types import Direction, Phase


def _match_material_selector(
    solver,
    world: "WorldEngine",
    x: int,
    y: int,
    *,
    required_id: int | None,
    required_mask: int,
    material_grid: np.ndarray | None = None,
) -> bool:
    if not world.in_bounds(x, y):
        return False
    material_field = world.material_id if material_grid is None else material_grid
    material_id = int(material_field[y, x])
    if material_id <= 0:
        return False
    if required_id is not None and material_id != required_id:
        return False
    if required_mask == 0:
        return required_id is not None
    return solver._mask_matches(solver._material_tag_mask(world, material_id, "material_tag_mask"), required_mask)


def _matching_material_neighbor(
    solver,
    world: "WorldEngine",
    x: int,
    y: int,
    *,
    required_id: int | None,
    required_mask: int,
    material_grid: np.ndarray | None = None,
) -> tuple[int, int] | None:
    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if solver._match_material_selector(
            world,
            nx,
            ny,
            required_id=required_id,
            required_mask=required_mask,
            material_grid=material_grid,
        ):
            return (nx, ny)
    return None


def _best_matching_material_reaction_gas_species(
    solver,
    world: "WorldEngine",
    gy: int,
    gx: int,
    *,
    gas_id: int | None,
    required_mask: int,
    gas_concentration: np.ndarray | None = None,
) -> tuple[int | None, float]:
    species_ids = solver._matching_material_reaction_gas_species_ids(
        world,
        gas_id=gas_id,
        required_mask=required_mask,
    )
    if not species_ids:
        return (None, 0.0)
    gas_field = world.gas_concentration if gas_concentration is None else gas_concentration
    best_species_id: int | None = None
    best = -1.0
    for species_id in species_ids:
        value = float(gas_field[species_id, gy, gx])
        if value > best:
            best = value
            best_species_id = int(species_id)
    return (best_species_id, max(0.0, best))


def _matching_material_reaction_gas_species_ids(
    solver,
    world: "WorldEngine",
    *,
    gas_id: int | None,
    required_mask: int,
) -> list[int]:
    if gas_id is not None:
        if gas_id < 0:
            return []
        if not solver._mask_matches(solver._gas_tag_mask(world, gas_id, "material_reaction_tag_mask"), required_mask):
            return []
        return [int(gas_id)]
    if required_mask == 0:
        return []
    species_limit = int(
        world.bridge.shadow_typed_tables["gas_table"].shape[0]
        if world.bridge.shadow_typed_tables.get("gas_table") is not None
        else world.gas_material_reaction_tag_mask.shape[0]
    )
    return [
        int(species_id)
        for species_id in range(species_limit)
        if solver._mask_matches(solver._gas_tag_mask(world, species_id, "material_reaction_tag_mask"), required_mask)
    ]


def _best_matching_light_reaction_gas_species(
    solver,
    world: "WorldEngine",
    gy: int,
    gx: int,
    *,
    gas_id: int | None,
    required_mask: int,
) -> tuple[int | None, float]:
    species_ids = solver._matching_light_gas_species_ids(
        world,
        gas_id=gas_id,
        required_mask=required_mask,
    )
    if not species_ids:
        return (None, 0.0)
    best_species_id: int | None = None
    best = -1.0
    for species_id in species_ids:
        value = float(world.gas_concentration[species_id, gy, gx])
        if value > best:
            best = value
            best_species_id = int(species_id)
    return (best_species_id, max(0.0, best))


def _matching_light_gas_species_ids(
    solver,
    world: "WorldEngine",
    *,
    gas_id: int | None,
    required_mask: int,
) -> list[int]:
    if gas_id is not None:
        if gas_id < 0:
            return []
        if not solver._mask_matches(solver._gas_tag_mask(world, gas_id, "light_reaction_tag_mask"), required_mask):
            return []
        return [int(gas_id)]
    if required_mask == 0:
        return []
    species_limit = int(
        world.bridge.shadow_typed_tables["gas_table"].shape[0]
        if world.bridge.shadow_typed_tables.get("gas_table") is not None
        else world.gas_light_reaction_tag_mask.shape[0]
    )
    return [
        int(species_id)
        for species_id in range(species_limit)
        if solver._mask_matches(solver._gas_tag_mask(world, species_id, "light_reaction_tag_mask"), required_mask)
    ]


def _light_dose_channel(solver, world: "WorldEngine", light_id: int) -> int | None:
    light_table = world.bridge.shadow_typed_tables.get("light_table")
    if light_table is not None and 0 <= light_id < int(light_table.shape[0]):
        if int(light_table[light_id]["name_hash"]) == 0:
            return None
        dose_channel = int(light_table[light_id]["dose_channel_id"])
        if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
            return dose_channel
        return None
    light_payload = world._shadow_light_type_payload()
    if isinstance(light_payload, list):
        for item in light_payload:
            if int(item.get("light_type_id", -1)) != light_id:
                continue
            dose_channel = int(item.get("dose_channel_id", -1))
            if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
                return dose_channel
            return None
        return None
    if 0 <= light_id < world.light_dose_channel.shape[0]:
        dose_channel = int(world.light_dose_channel[light_id])
        if 0 <= dose_channel < world.cell_optical_dose.shape[0] and 0 <= dose_channel < world.gas_optical_dose.shape[0]:
            return dose_channel
    return None


def _light_emit_metadata(solver, world: "WorldEngine", light_id: int) -> tuple[str, int] | None:
    return world._shadow_light_name_and_range(light_id)


def _material_default_phase(solver, world: "WorldEngine", material_id: int) -> int | None:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
        if int(material_table[material_id]["name_hash"]) == 0:
            return None
        return int(material_table[material_id]["default_phase"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return int(shadow_material.default_phase)
    if world._shadow_has_table_payload("materials"):
        return None
    if 0 <= material_id < world.material_default_phase.shape[0]:
        return int(world.material_default_phase[material_id])
    return None


def _material_base_integrity(solver, world: "WorldEngine", material_id: int) -> float | None:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
        if int(material_table[material_id]["name_hash"]) == 0:
            return None
        return float(material_table[material_id]["base_integrity"])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return float(shadow_material.base_integrity)
    if world._shadow_has_table_payload("materials"):
        return None
    if 0 <= material_id < world.material_base_integrity.shape[0]:
        return float(world.material_base_integrity[material_id])
    return None


def _random_convert_candidates(solver, world: "WorldEngine") -> list[int]:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    chaos_convert_bit = int(world.tag_bits_by_name.get("chaos_convert", 0))
    if material_table is not None and chaos_convert_bit != 0:
        return [
            int(row["material_id"])
            for row in material_table
            if int(row["material_id"]) > 0
            and int(row["name_hash"]) != 0
            and bool(int(row["material_tag_mask"]) & chaos_convert_bit)
            and int(row["default_phase"]) == int(Phase.POWDER)
        ]
    if material_table is not None:
        return []
    if world.random_convert_material_ids:
        return [int(material_id) for material_id in world.random_convert_material_ids if int(material_id) > 0]
    return []


def _material_reaction_slot(solver, world: "WorldEngine", material_id: int, slot_index: int) -> int:
    if slot_index < 0 or slot_index >= 8:
        return -1
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
        if int(material_table[material_id]["name_hash"]) == 0:
            return -1
        return int(material_table[material_id]["reaction_slots"][slot_index])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        return int(shadow_material.reaction_slots[slot_index])
    if world._shadow_has_table_payload("materials"):
        return -1
    if 0 <= material_id < world.material_reaction_slots.shape[0]:
        return int(world.material_reaction_slots[material_id, slot_index])
    return -1


def _material_tag_mask(solver, world: "WorldEngine", material_id: int, field: str) -> int:
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is not None and 0 <= material_id < int(material_table.shape[0]):
        if int(material_table[material_id]["name_hash"]) == 0:
            return 0
        return int(material_table[material_id][field])
    shadow_material = world._shadow_material_def(material_id)
    if shadow_material is not None:
        if field == "material_tag_mask":
            return int(shadow_material.material_tag_mask)
        if field == "gas_tag_mask":
            return int(shadow_material.gas_tag_mask)
        if field == "light_tag_mask":
            return int(shadow_material.light_tag_mask)
    if world._shadow_has_table_payload("materials"):
        return 0
    fallback_map = {
        "material_tag_mask": world.material_material_tag_mask,
        "gas_tag_mask": world.material_gas_tag_mask,
        "light_tag_mask": world.material_light_tag_mask,
    }
    fallback = fallback_map[field]
    if 0 <= material_id < fallback.shape[0]:
        return int(fallback[material_id])
    return 0


def _gas_tag_mask(solver, world: "WorldEngine", species_id: int, field: str) -> int:
    gas_table = world.bridge.shadow_typed_tables.get("gas_table")
    if gas_table is not None and 0 <= species_id < int(gas_table.shape[0]):
        if int(gas_table[species_id]["name_hash"]) == 0:
            return 0
        return int(gas_table[species_id][field])
    shadow_gas = world._shadow_gas_species_def(species_id)
    if shadow_gas is not None:
        if field == "material_reaction_tag_mask":
            return int(shadow_gas.material_reaction_tag_mask)
        if field == "light_reaction_tag_mask":
            return int(shadow_gas.light_reaction_tag_mask)
    if world._shadow_has_table_payload("gases"):
        return 0
    fallback_map = {
        "material_reaction_tag_mask": world.gas_material_reaction_tag_mask,
        "light_reaction_tag_mask": world.gas_light_reaction_tag_mask,
    }
    fallback = fallback_map[field]
    if 0 <= species_id < fallback.shape[0]:
        return int(fallback[species_id])
    return 0


def _neighbor_for_direction(solver, world: "WorldEngine", direction: Direction, x: int, y: int) -> tuple[int, int]:
    if direction == Direction.DOWN:
        return x, y + 1
    if direction == Direction.UP:
        return x, y - 1
    if direction == Direction.LEFT:
        return x - 1, y
    if direction == Direction.RIGHT:
        return x + 1, y
    if direction == Direction.RANDOM:
        return solver._deterministic_random_neighbor(x, y)
    if direction == Direction.SPEED:
        vx, vy = world.velocity[y, x]
        return x + int(np.sign(vx)), y + int(np.sign(vy))
    return x, y


def _neighbor_for_direction_id(solver, world: "WorldEngine", direction_id: int, x: int, y: int) -> tuple[int, int]:
    if direction_id == int(DIRECTION_IDS["down"]):
        return x, y + 1
    if direction_id == int(DIRECTION_IDS["up"]):
        return x, y - 1
    if direction_id == int(DIRECTION_IDS["left"]):
        return x - 1, y
    if direction_id == int(DIRECTION_IDS["right"]):
        return x + 1, y
    if direction_id == int(DIRECTION_IDS["random"]):
        return solver._deterministic_random_neighbor(x, y)
    if direction_id == int(DIRECTION_IDS["speed"]):
        vx, vy = world.velocity[y, x]
        return x + int(np.sign(vx)), y + int(np.sign(vy))
    return x, y


def _material_emit_target_and_velocity(
    solver,
    world: "WorldEngine",
    emit_material_id: int,
    direction_id: int,
    explicit_velocity: np.ndarray,
    speed: float,
    x: int,
    y: int,
) -> tuple[int, int, np.ndarray]:
    tx, ty = solver._neighbor_for_direction_id(world, direction_id, x, y)
    emitted_phase = solver._material_default_phase(world, emit_material_id)
    if emitted_phase is None:
        return tx, ty, np.zeros((2,), dtype=np.float32)
    if emitted_phase != int(Phase.POWDER):
        return tx, ty, np.zeros((2,), dtype=np.float32)
    velocity = np.asarray(explicit_velocity, dtype=np.float32)
    if float(np.hypot(float(velocity[0]), float(velocity[1]))) > 1e-5:
        return tx, ty, velocity.astype(np.float32, copy=True)
    dx = tx - x
    dy = ty - y
    norm = max(1e-5, float(np.hypot(dx, dy)))
    magnitude = max(0.0, float(speed))
    return tx, ty, np.asarray((dx / norm * magnitude, dy / norm * magnitude), dtype=np.float32)


def _deterministic_selector(x: int, y: int, count: int) -> int:
    if count <= 0:
        return 0
    return abs((int(x) * 73856093) ^ (int(y) * 19349663)) % int(count)


def _deterministic_random_neighbor(solver, x: int, y: int) -> tuple[int, int]:
    selector = solver._deterministic_selector(x, y, 9)
    dx = selector % 3 - 1
    dy = selector // 3 - 1
    return x + dx, y + dy


def _neighbor_for_gas_direction(solver, world: "WorldEngine", direction: Direction, gx: int, gy: int) -> tuple[int, int]:
    cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
    if direction == Direction.SPEED:
        vx, vy = world.flow_velocity[gy, gx]
        return cell_x + int(np.sign(vx)), cell_y + int(np.sign(vy))
    return solver._neighbor_for_direction(world, direction, cell_x, cell_y)


def _neighbor_for_gas_direction_id(solver, world: "WorldEngine", direction_id: int, gx: int, gy: int) -> tuple[int, int]:
    cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
    if direction_id == int(DIRECTION_IDS["speed"]):
        vx, vy = world.flow_velocity[gy, gx]
        return cell_x + int(np.sign(vx)), cell_y + int(np.sign(vy))
    return solver._neighbor_for_direction_id(world, direction_id, cell_x, cell_y)


def _direction_vector(solver, direction: Direction, x: int, y: int, world: "WorldEngine") -> tuple[float, float]:
    if direction == Direction.UP:
        return (0.0, -1.0)
    if direction == Direction.DOWN:
        return (0.0, 1.0)
    if direction == Direction.LEFT:
        return (-1.0, 0.0)
    if direction == Direction.RIGHT:
        return (1.0, 0.0)
    if direction == Direction.SPEED:
        vx, vy = world.velocity[y, x]
        norm = max(1e-5, float(np.hypot(vx, vy)))
        return (float(vx / norm), float(vy / norm))
    return (0.0, 0.0)


def _direction_vector_id(solver, direction_id: int, x: int, y: int, world: "WorldEngine") -> tuple[float, float]:
    if direction_id == int(DIRECTION_IDS["up"]):
        return (0.0, -1.0)
    if direction_id == int(DIRECTION_IDS["down"]):
        return (0.0, 1.0)
    if direction_id == int(DIRECTION_IDS["left"]):
        return (-1.0, 0.0)
    if direction_id == int(DIRECTION_IDS["right"]):
        return (1.0, 0.0)
    if direction_id == int(DIRECTION_IDS["speed"]):
        vx, vy = world.velocity[y, x]
        norm = max(1e-5, float(np.hypot(vx, vy)))
        return (float(vx / norm), float(vy / norm))
    return (0.0, 0.0)


def _gas_direction_vector(solver, world: "WorldEngine", direction: Direction, gx: int, gy: int) -> tuple[float, float]:
    if direction == Direction.SPEED:
        vx, vy = world.flow_velocity[gy, gx]
        norm = max(1e-5, float(np.hypot(vx, vy)))
        return (float(vx / norm), float(vy / norm))
    cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
    return solver._direction_vector(direction, cell_x, cell_y, world)


def _gas_direction_vector_id(solver, world: "WorldEngine", direction_id: int, gx: int, gy: int) -> tuple[float, float]:
    if direction_id == int(DIRECTION_IDS["speed"]):
        vx, vy = world.flow_velocity[gy, gx]
        norm = max(1e-5, float(np.hypot(vx, vy)))
        return (float(vx / norm), float(vy / norm))
    cell_x, cell_y = solver._gas_cell_center(world, gx, gy)
    return solver._direction_vector_id(direction_id, cell_x, cell_y, world)


def _gas_cell_center(solver, world: "WorldEngine", gx: int, gy: int) -> tuple[int, int]:
    x = min(world.width - 1, gx * world.gas_cell_size + world.gas_cell_size // 2)
    y = min(world.height - 1, gy * world.gas_cell_size + world.gas_cell_size // 2)
    return (x, y)
