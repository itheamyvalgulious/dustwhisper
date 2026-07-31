"""Leaf constants and reservation dtypes for the GPU motion family.

Every ``gpu_motion_*`` satellite imports these names from here instead of the
``oracle_game.sim.gpu_motion`` facade, so the satellites stay importable in
isolation and the facade can re-export them without an import cycle. This
module must not import any other member of the gpu_motion family.
"""

from __future__ import annotations

import numpy as np

LOCAL_SIZE = 8
POWDER_RESERVATION_LOCAL_SIZE = 64
ISLAND_RESERVATION_LINEAR_LOCAL_SIZE = 256
ACTIVE_TILE_WORKGROUP_AXIS = 4
ACTIVE_TILE_WORKGROUPS_PER_TILE = ACTIVE_TILE_WORKGROUP_AXIS * ACTIVE_TILE_WORKGROUP_AXIS
MAX_MATERIALS = 256
MAX_ISLAND_DDA_STEP = 4
INDEX_EMPTY = 2147483647
# Global gravity acceleration in cells/s^2 applied to velocity.y before drag.
# Strong enough that a freely falling grain clears the velocity-DDA engagement
# threshold (30 cells/s) within a few frames; 24.0 needed ~1.25 s at 60 fps and
# looked like a uniform 1 cell/frame trickle for the first seconds.
GRAVITY_CELLS_PER_SECOND_SQ = 120.0
FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING = 1
FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING = 2
FALLING_ISLAND_INDEX_CLEAR_MATERIALIZATION = 4
FALLING_ISLAND_INDEX_CLEAR_SOURCE = 8
FALLING_ISLAND_INDEX_CLEAR_APPLY = (
    FALLING_ISLAND_INDEX_CLEAR_APPLY_INCOMING | FALLING_ISLAND_INDEX_CLEAR_APPLY_OUTGOING
)

POWDER_RESOLVE_BLOCKED = 0
POWDER_RESOLVE_DDA = 1
POWDER_RESOLVE_FALLBACK = 2
POWDER_RESOLVE_STALE = 3
POWDER_SOLVER_SUSPENDED = 2
ISLAND_RESOLVE_BLOCKED = 0
ISLAND_RESOLVE_DIRECT = 1
ISLAND_RESOLVE_RERESOLVED = 2
ISLAND_RESOLVE_STALE = 3
FALLING_ISLAND_BREAK_STABLE = 2


POWDER_RESERVATION_DTYPE = np.dtype(
    [
        ("source_xy", "<i4", (2,)),
        ("desired_target_xy", "<i4", (2,)),
        ("reserved_target_xy", "<i4", (2,)),
        ("resolved_target_xy", "<i4", (2,)),
        ("velocity_xy", "<f4", (2,)),
        ("material_id", "<i4"),
        ("resolve_state", "<i4"),
    ]
)


def powder_reservation_dtype() -> np.dtype:
    return POWDER_RESERVATION_DTYPE


FALLING_ISLAND_RESERVATION_DTYPE = np.dtype(
    [
        ("island_id", "<i4"),
        ("buffer_bbox", "<i4", (4,)),
        ("velocity_xy", "<f4", (2,)),
        ("subcell_offset", "<f4", (2,)),
        ("target_shift", "<i4", (2,)),
        ("reserved_shift", "<i4", (2,)),
        ("resolved_shift", "<i4", (2,)),
        ("resolve_state", "<i4"),
    ]
)


def falling_island_reservation_dtype() -> np.dtype:
    return FALLING_ISLAND_RESERVATION_DTYPE
