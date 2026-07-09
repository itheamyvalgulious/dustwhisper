from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_motion import (
    falling_island_reservation_dtype,
    powder_reservation_dtype,
)


def release(solver) -> None:
    solver.gpu_pipeline.release()
    solver.reset_runtime_state()



def reset_runtime_state(solver) -> None:
    solver.last_powder_reservations = np.zeros((0,), dtype=powder_reservation_dtype())
    solver.last_island_reservations = np.zeros((0,), dtype=falling_island_reservation_dtype())
    solver.last_public_powder_reservations = []
    solver.last_public_island_reservations = []



def runtime_snapshot(solver) -> dict[str, object]:
    return {
        "powder_reservations": solver.last_powder_reservations.copy(),
        "island_reservations": solver.last_island_reservations.copy(),
        "public_powder_reservations": [dict(record) for record in solver.last_public_powder_reservations],
        "public_island_reservations": [dict(record) for record in solver.last_public_island_reservations],
    }



def _capture_public_powder_reservations(
    solver,
    world: "WorldEngine",
    reservations: np.ndarray,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for record in reservations:
        item: dict[str, object] = {}
        for name in reservations.dtype.names or ():
            value = record[name]
            if isinstance(value, np.ndarray):
                if name in {"source_xy", "desired_target_xy", "reserved_target_xy", "resolved_target_xy"}:
                    world_x, world_y = world._buffer_to_world_position((int(value[0]), int(value[1])))
                    item[name] = [int(world_x), int(world_y)]
                else:
                    item[name] = value.tolist()
            elif isinstance(value, np.generic):
                item[name] = value.item()
            else:
                item[name] = value
        payload.append(item)
    return payload



def _capture_public_island_reservations(
    solver,
    world: "WorldEngine",
    reservations: np.ndarray,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for record in reservations:
        item: dict[str, object] = {}
        for name in reservations.dtype.names or ():
            value = record[name]
            if name == "buffer_bbox":
                item["world_bbox"] = list(
                    world._buffer_bbox_to_world_bbox(tuple(int(component) for component in np.asarray(value).tolist()))
                )
                continue
            if isinstance(value, np.ndarray):
                item[name] = value.tolist()
            elif isinstance(value, np.generic):
                item[name] = value.item()
            else:
                item[name] = value
        payload.append(item)
    return payload

