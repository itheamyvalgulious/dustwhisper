"""CPU-oracle tests for the lateral seam spread clamps.

Lateral seam spread is speed-limited (SEAM_MAX_MOVE_PER_FRAME cells per
frame per boundary) and support-limited (at most the leading
SEAM_UNSUPPORTED_OVERHANG filled cells may lack support from below).  These
tests pin the CPU oracle semantics in liquid_solve.py that mirror
shaders/liquid/seam_x.comp, plus the constant parity between both sides.
"""

from __future__ import annotations

import re

import numpy as np

from oracle_game.sim.gpu_liquid import _SHADER_SUBS
from oracle_game.sim.liquid_constants import LIQUID_SOLVER_TILE_LEVEL
from oracle_game.sim.liquid_solve import (
    LIQUID_DOWNFILL_MAX_LATERAL_SHIFT,
    SEAM_MAX_MOVE_PER_FRAME,
    SEAM_UNSUPPORTED_OVERHANG,
    _apply_horizontal_seam_run,
    _world_cell_is_tile_level_liquid,
    _world_cell_reachable_empty,
)
from oracle_game.sim.shader_loader import shader_source
from oracle_game.types import Phase

TILE_SIZE = 8


class _FakeSolver:
    def _material_liquid_solver_kind(self, world: "_FakeWorld", material_id: int) -> int:
        return LIQUID_SOLVER_TILE_LEVEL

    def _world_cell_reachable_empty(self, world: "_FakeWorld", x: int, y: int) -> bool:
        return _world_cell_reachable_empty(self, world, x, y)

    def _world_cell_is_tile_level_liquid(self, world: "_FakeWorld", x: int, y: int) -> bool:
        return _world_cell_is_tile_level_liquid(self, world, x, y)


class _FakeWorld:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.material_id = np.zeros((height, width), dtype=np.int32)
        self.phase = np.zeros((height, width), dtype=np.int32)
        self.entity_id = np.zeros((height, width), dtype=np.int32)
        self.placeholder_displaced_material = np.zeros((height, width), dtype=np.int32)

    def swap_cells(self, x1: int, y1: int, x2: int, y2: int) -> None:
        for field in (
            self.material_id,
            self.phase,
            self.entity_id,
            self.placeholder_displaced_material,
        ):
            field[y1, x1], field[y2, x2] = field[y2, x2], field[y1, x1]


def _put_liquid(world: _FakeWorld, x: int, y: int) -> None:
    world.material_id[y, x] = 1
    world.phase[y, x] = int(Phase.LIQUID)


def _put_solid(world: _FakeWorld, x: int, y: int) -> None:
    world.material_id[y, x] = 2
    world.phase[y, x] = int(Phase.STATIC_SOLID)


def _liquid_xs(world: _FakeWorld, y: int) -> set[int]:
    return {
        x
        for x in range(world.width)
        if int(world.phase[y, x]) == int(Phase.LIQUID) and int(world.material_id[y, x]) == 1
    }


def _flat_world(*, floor: bool = True) -> _FakeWorld:
    world = _FakeWorld(width=32, height=8)
    if floor:
        for x in range(world.width):
            _put_solid(world, x, 5)
    for x in range(8):
        _put_liquid(world, x, 4)
    return world


def test_speed_cap_matches_in_tile_rate_on_flat_ground() -> None:
    solver = _FakeSolver()
    world = _flat_world()
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is True
    assert _liquid_xs(world, 4) == {0, 1, 2, 3, 4, 5, 6, 8}


def test_suspended_source_does_not_spread() -> None:
    solver = _FakeSolver()
    world = _flat_world(floor=False)
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is False
    assert _liquid_xs(world, 4) == set(range(8))


def test_grounded_source_pushes_one_lip_cell_over_unsupported_targets() -> None:
    solver = _FakeSolver()
    world = _flat_world(floor=False)
    for x in range(8):
        _put_solid(world, x, 5)
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is True
    assert _liquid_xs(world, 4) == {0, 1, 2, 3, 4, 5, 6, 8}


def test_source_on_liquid_gets_no_overhang_over_unsupported_targets() -> None:
    # The source run stands on other liquid (not ground), so the spread must
    # strictly follow its support: with unsupported targets nothing crosses,
    # which keeps pool surfaces from hanging past their base (the wall).
    solver = _FakeSolver()
    world = _flat_world(floor=False)
    for x in range(8):
        _put_liquid(world, x, 5)
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is False
    assert _liquid_xs(world, 4) == set(range(8))


def test_source_on_liquid_follows_supported_targets() -> None:
    solver = _FakeSolver()
    world = _flat_world(floor=False)
    for x in range(8):
        _put_liquid(world, x, 5)
    _put_solid(world, 8, 5)
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is True
    assert _liquid_xs(world, 4) == {0, 1, 2, 3, 4, 5, 6, 8}


def test_right_to_left_speed_cap_mirrors_left_to_right() -> None:
    solver = _FakeSolver()
    world = _FakeWorld(width=32, height=8)
    for x in range(world.width):
        _put_solid(world, x, 5)
    for x in range(8, 16):
        _put_liquid(world, x, 4)
    assert _apply_horizontal_seam_run(solver, world, 8, 4, TILE_SIZE) is True
    assert _liquid_xs(world, 4) == {7, 9, 10, 11, 12, 13, 14, 15}


def test_seam_clamp_constants_match_shader() -> None:
    source = shader_source("liquid/seam_x.comp", dict(_SHADER_SUBS))
    shader_cap = int(
        re.search(r"const int SEAM_MAX_MOVE_PER_FRAME = (\d+);", source).group(1)  # type: ignore[union-attr]
    )
    shader_overhang = int(
        re.search(r"const int SEAM_UNSUPPORTED_OVERHANG = (\d+);", source).group(1)  # type: ignore[union-attr]
    )
    assert shader_cap == SEAM_MAX_MOVE_PER_FRAME
    assert shader_overhang == SEAM_UNSUPPORTED_OVERHANG


def test_downfill_lateral_shift_constant_matches_shaders() -> None:
    for shader in ("liquid/tile_solve.comp", "liquid/seam_y.comp"):
        source = shader_source(shader, dict(_SHADER_SUBS))
        shader_shift = int(
            re.search(r"const int LIQUID_DOWNFILL_MAX_LATERAL_SHIFT = (\d+);", source).group(1)  # type: ignore[union-attr]
        )
        assert shader_shift == LIQUID_DOWNFILL_MAX_LATERAL_SHIFT
