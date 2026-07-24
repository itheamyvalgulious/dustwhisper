"""Leaf constants and solve-mask types for the CPU reactions family.

Every ``reactions_*`` satellite imports these names from here instead of the
``oracle_game.sim.reactions`` facade, so the satellites stay importable in
isolation and the facade can re-export them without an import cycle. This
module must not import any other member of the reactions family.
"""

from __future__ import annotations

import numpy as np

REACTION_ACTIVITY_EPSILON = 1e-4
REACTION_FLOW_SOURCE_LIFETIME = 1.0 / 60.0
REACTION_STAGE_NAMES = (
    "timed",
    "self",
    "material_material",
    "material_gas",
    "material_light",
    "gas_gas",
    "gas_light",
)


class GPUAuthoritativeFullSolveMask:
    full_gpu_authoritative = True

    __slots__ = ("domain", "shape")

    def __init__(self, domain: str, shape: tuple[int, int]) -> None:
        self.domain = domain
        self.shape = shape

    def __array__(self, dtype: object | None = None, copy: object | None = None) -> np.ndarray:
        raise TypeError("GPU-authoritative full solve mask is not materialized on CPU")

    def copy(self) -> "GPUAuthoritativeFullSolveMask":
        return self

    def __repr__(self) -> str:
        return f"GPUAuthoritativeFullSolveMask(domain={self.domain!r}, shape={self.shape!r})"


SolveMask = np.ndarray | GPUAuthoritativeFullSolveMask
