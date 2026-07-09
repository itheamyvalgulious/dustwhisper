from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _paint_material(engine: "WorldEngine", x: int, y: int, material: str, radius: int) -> None:
    yy, xx = np.mgrid[0:engine.height, 0:engine.width]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    engine.set_material_by_mask(mask, material)


def _write_material_region_immediate(
    engine: "WorldEngine",
    x: int,
    y: int,
    width: int,
    height: int,
    material: str,
) -> None:
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(engine.width, int(x) + max(0, int(width)))
    y1 = min(engine.height, int(y) + max(0, int(height)))
    if x0 >= x1 or y0 >= y1:
        return
    for write_y in range(y0, y1):
        for write_x in range(x0, x1):
            engine.set_cell(write_x, write_y, material)


def _build_demo_scene(engine: "WorldEngine") -> None:
    active_w = int(engine.paging.active_width)
    active_h = int(engine.paging.active_height)
    floor_y = max(0, active_h - 28)
    _fill_rect(engine, 0, floor_y, active_w, 28, "raw_stone_solid")
    _fill_rect(engine, 32, floor_y - 58, 160, 14, "sandstone_solid")
    _fill_rect(engine, 60, floor_y - 112, 118, 54, "sand_powder")
    _fill_rect(engine, 230, floor_y - 24, 112, 24, "water_liquid")
    _fill_rect(engine, 374, floor_y - 18, 78, 18, "oil_liquid")
    _fill_rect(engine, 500, floor_y - 86, 12, 86, "raw_stone_solid")
    _fill_rect(engine, 520, floor_y - 46, 130, 18, "sandstone_solid")
    _fill_rect(engine, 550, floor_y - 86, 76, 40, "gravel_powder")
    _fill_rect(engine, 690, floor_y - 140, 72, 16, "log_solid")
    _fill_rect(engine, 700, floor_y - 188, 12, 48, "root_solid")


def _fill_rect(engine: "WorldEngine", x: int, y: int, width: int, height: int, material: str) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(engine.width, x + width)
    y1 = min(engine.height, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    material_id = engine._resolve_sanctioned_material_id(material)
    if material_id <= 0:
        raise KeyError(material)
    phase = int(engine.material_default_phase[material_id]) if material_id < engine.material_default_phase.shape[0] else 0
    integrity = (
        float(engine.material_base_integrity[material_id])
        if material_id < engine.material_base_integrity.shape[0]
        else 0.0
    )
    engine.material_id[y0:y1, x0:x1] = int(material_id)
    engine.phase[y0:y1, x0:x1] = phase
    engine.cell_flags[y0:y1, x0:x1] = 0
    engine.velocity[y0:y1, x0:x1] = 0.0
    engine.timer_pack[y0:y1, x0:x1] = 0
    engine.integrity[y0:y1, x0:x1] = integrity
    engine.island_id[y0:y1, x0:x1] = 0
    engine.entity_id[y0:y1, x0:x1] = 0
    engine.placeholder_displaced_material[y0:y1, x0:x1] = 0
    if material_id < engine.material_spawn_temperature.shape[0]:
        spawn_temperature = float(engine.material_spawn_temperature[material_id])
        if np.isfinite(spawn_temperature):
            engine.cell_temperature[y0:y1, x0:x1] = np.maximum(
                engine.cell_temperature[y0:y1, x0:x1],
                spawn_temperature,
            )


def close(engine: "WorldEngine") -> None:
    if engine._closed:
        return
    engine._closed = True
    engine.gas_solver.release()
    engine.heat_solver.release()
    engine.collapse_solver.release()
    engine.motion_solver.release()
    engine.liquid_solver.release()
    engine.optics_solver.release()
    engine.reaction_solver.release()
    engine.placeholder_pipeline.release()
    engine.page_stripe_pipeline.release()
    engine.grid_command_pipeline.release()
    engine.bridge.release()


def _world_engine_del(engine: "WorldEngine") -> None:  # pragma: no cover
    try:
        close(engine)
    except Exception:
        pass
