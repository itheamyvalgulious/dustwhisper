"""Leaf solver-kind constants for the CPU liquid family.

The ``liquid_*`` satellites import these names from here instead of the
``oracle_game.sim.liquid`` facade, so the satellites stay importable in
isolation and the facade can re-export them without an import cycle. This
module must not import any other member of the liquid family.
"""

from __future__ import annotations

LIQUID_ACTIVITY_EPSILON = 1e-6
LIQUID_SOLVER_TILE_LEVEL = 1
LIQUID_SOLVER_COLUMNAR = 2
