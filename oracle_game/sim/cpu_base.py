"""Shared base/helpers for the CPU-side stage solvers.

Phase 1 introduces this module to host helpers that were copy-pasted across
the seven CPU solvers (``CollapseSolver``, ``GasSolver``, ``HeatSolver``,
``LiquidSolver``, ``MotionSolver``, ``ReactionSolver``, ``OpticsSolver``).

Phase 3 will add a ``Solver`` base class here codifying the shared lifecycle
(``step`` / ``reset_runtime_state`` / ``release`` / ``runtime_snapshot``) and a
``_select_backend`` helper for the duplicated "formal-GPU-frame /
active-scheduler-authoritative" prologue at the top of each ``step``.  For now
this module holds the one helper that is provably identical across solvers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # WorldEngine is passed in; avoid a circular import at runtime.
    from oracle_game.world import WorldEngine


def material_table_row(world: "WorldEngine", material_id: int) -> np.void | None:
    """Look up a material row in the bridge's typed shadow table.

    This body was duplicated verbatim in ``CollapseSolver``, ``MotionSolver``
    and ``LiquidSolver`` (confirmed byte-identical via diff).  It uses no
    ``self`` state — only ``world.bridge.shadow_typed_tables`` — so it is a pure
    function and the lift is behavior-preserving by construction.
    """
    material_table = world.bridge.shadow_typed_tables.get("material_table")
    if material_table is None or material_id < 0 or material_id >= int(material_table.shape[0]):
        return None
    row = material_table[material_id]
    if int(row["name_hash"]) == 0:
        return None
    return row
