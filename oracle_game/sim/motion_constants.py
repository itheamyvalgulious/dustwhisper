"""Leaf constants and component record for the CPU motion family.

Every ``motion_*`` satellite imports these names from here instead of the
``oracle_game.sim.motion`` facade, so the satellites stay importable in
isolation and the facade can re-export them without an import cycle. This
module must not import any other member of the motion family. The resolve/break
values mirror ``gpu_motion_constants`` (CPU and GPU paths share the ABI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ISLAND_DDA_STEP = 4
POWDER_SOLVER_SUSPENDED = 2
FALLING_ISLAND_BREAK_STABLE = 2


@dataclass(slots=True)
class _IslandComponentEntry:
    label: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    cell_count: int
