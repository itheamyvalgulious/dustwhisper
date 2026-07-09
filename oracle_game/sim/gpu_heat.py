from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader, shader_source


LOCAL_SIZE = 8
MAX_MATERIALS = 256
MAX_GAS_SPECIES = 256
FREEZE_COLD_NEIGHBOR_THRESHOLD = 4

# Superset of every {{NAME}} marker referenced by any heat shader; the loader
# ignores unused keys, so one shared dict suffices for all passes.
_SHADER_SUBS = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_GAS_SPECIES": MAX_GAS_SPECIES,
    "MAX_MATERIALS_MINUS_ONE": MAX_MATERIALS - 1,
}


@dataclass(slots=True)
class GPUHeatResources:
    signature: tuple[int, int, int, int, int]
    material_tex: Any
    material_out_tex: Any
    phase_tex: Any
    phase_out_tex: Any
    cell_flags_tex: Any
    cell_flags_out_tex: Any
    timer_tex: Any
    timer_out_tex: Any
    integrity_tex: Any
    integrity_out_tex: Any
    island_id_tex: Any
    island_id_out_tex: Any
    entity_id_tex: Any
    entity_id_out_tex: Any
    displaced_tex: Any
    displaced_out_tex: Any
    velocity_tex: Any
    velocity_out_tex: Any
    temp_ping: Any
    temp_pong: Any
    phase_target_tex: Any
    boil_target_tex: Any
    gas_tex: Any
    gas_out_tex: Any
    condense_target_tex: Any
    ambient_ping: Any
    ambient_pong: Any
    active_tile_tex: Any
    material_params: Any
    material_response_params: Any
    material_phase_params: Any
    gas_params: Any
    material_params_signature: tuple[int, int] | None = None
    gas_params_signature: tuple[int, int] | None = None


@dataclass(slots=True)
class GPUHeatStageTargets:
    phase_targets: np.ndarray
    boil_targets: np.ndarray
    condense_targets: np.ndarray

    @property
    def empty(self) -> bool:
        return (
            self.phase_targets.size == 0
            and self.boil_targets.size == 0
            and self.condense_targets.size == 0
        )

    @classmethod
    def empty_sentinel(cls) -> "GPUHeatStageTargets":
        return cls(
            phase_targets=np.zeros((0, 0), dtype=np.int32),
            boil_targets=np.zeros((0, 0), dtype=np.int32),
            condense_targets=np.zeros((0, 0, 0), dtype=np.bool_),
        )


from oracle_game.sim.gpu_heat_resources import (
    release,
    _ensure_resources,
    _upload_inputs,
    _load_authoritative_active_tile_mask,
)
from oracle_game.sim.gpu_heat_stages import (
    step,
    _load_authoritative_bridge_inputs,
    _run_cell_heat,
    _run_ambient_exchange,
    _run_ambient_diffuse,
    _run_ambient_feedback,
    _run_phase_targets,
    _run_boil_targets,
    _run_condense_targets,
    _run_apply_cell_targets,
    _run_apply_cell_aux_targets,
    _run_apply_gas_targets,
    _run_apply_condense_cells,
    _run_apply_condense_cell_aux,
    _download_outputs,
    _empty_stage_targets,
    _publish_bridge_outputs,
)


class GPUHeatPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUHeatResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_cell_state_upload_skipped = False
        self.last_cpu_island_id_upload_skipped = False
        self.last_cpu_entity_id_upload_skipped = False
        self.last_cpu_displaced_material_upload_skipped = False
        self.last_cpu_ambient_upload_skipped = False
        self.last_cpu_gas_upload_skipped = False
        self.last_cpu_active_upload_skipped = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = build_compute_shader(ctx, "heat/load_active_tiles.comp", _SHADER_SUBS)
        self.programs["cell_heat"] = build_compute_shader(ctx, "heat/cell_heat.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_exchange"] = build_compute_shader(ctx, "heat/ambient_exchange.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_feedback"] = build_compute_shader(ctx, "heat/ambient_feedback.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_diffuse"] = build_compute_shader(ctx, "heat/ambient_diffuse.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["phase_targets"] = build_compute_shader(ctx, "heat/phase_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_cell_targets"] = build_compute_shader(ctx, "heat/apply_cell_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_cell_aux_targets"] = build_compute_shader(ctx, "heat/apply_cell_aux_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["boil_targets"] = build_compute_shader(ctx, "heat/boil_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["condense_targets"] = build_compute_shader(ctx, "heat/condense_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_gas_targets"] = build_compute_shader(ctx, "heat/apply_gas_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_condense_cells"] = build_compute_shader(ctx, "heat/apply_condense_cells.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["apply_condense_cell_aux"] = build_compute_shader(ctx, "heat/apply_condense_cell_aux.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "heat/publish_bridge_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_gas"] = build_compute_shader(ctx, "heat/publish_bridge_gas.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell"] = build_compute_shader(ctx, "heat/load_bridge_cell.comp", _SHADER_SUBS)
        self.programs["load_bridge_cell_aux"] = build_compute_shader(ctx, "heat/load_bridge_cell_aux.comp", _SHADER_SUBS)
        self.programs["load_bridge_gas"] = build_compute_shader(ctx, "heat/load_bridge_gas.comp", _SHADER_SUBS)

    release = release
    _ensure_resources = _ensure_resources
    _upload_inputs = _upload_inputs
    _load_authoritative_active_tile_mask = _load_authoritative_active_tile_mask

    step = step
    _load_authoritative_bridge_inputs = _load_authoritative_bridge_inputs
    _run_cell_heat = _run_cell_heat
    _run_ambient_exchange = _run_ambient_exchange
    _run_ambient_diffuse = _run_ambient_diffuse
    _run_ambient_feedback = _run_ambient_feedback
    _run_phase_targets = _run_phase_targets
    _run_boil_targets = _run_boil_targets
    _run_condense_targets = _run_condense_targets
    _run_apply_cell_targets = _run_apply_cell_targets
    _run_apply_cell_aux_targets = _run_apply_cell_aux_targets
    _run_apply_gas_targets = _run_apply_gas_targets
    _run_apply_condense_cells = _run_apply_condense_cells
    _run_apply_condense_cell_aux = _run_apply_condense_cell_aux
    _download_outputs = _download_outputs
    _empty_stage_targets = _empty_stage_targets
    _publish_bridge_outputs = _publish_bridge_outputs

    # ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
    # from :class:`GPUPipelineBase`; the heat pass uses the default barrier
    # bits (image-access | texture-fetch | shader-storage).
