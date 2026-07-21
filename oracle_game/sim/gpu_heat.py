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
    "TERMINAL_LOCAL_SIZE_X": 8,
    "TERMINAL_LOCAL_SIZE_Y": 8,
    "TERMINAL_CELL_COUNT": 64,
    "TERMINAL_GAS_SIZE_X": 2,
    "TERMINAL_GAS_SIZE_Y": 2,
    "TERMINAL_GAS_COUNT": 4,
    "TERMINAL_CONDENSE_RANK_COUNT": 24,
    "MAX_MATERIALS": MAX_MATERIALS,
    "MAX_GAS_SPECIES": MAX_GAS_SPECIES,
    "MAX_MATERIALS_MINUS_ONE": MAX_MATERIALS - 1,
    "DIRTY_WORKGROUP_AGGREGATE": 0,
    "HEAT_GAS_BRIDGE_RESIDENT": 0,
    "TERMINAL_SPARSE_RESIDENT_SPECIALIZED": 0,
    "HEAT_LAZY_ACTION_INPUTS": 0,
    "HEAT_PACKED_PHASE_BOIL_TARGETS": 0,
    "TERMINAL_HIERARCHICAL_ROW_SUMMARY": 0,
    "TERMINAL_NV32_GAS_BALLOT": 0,
    "TERMINAL_NV32_EXTENSIONS": "",
}


@dataclass(slots=True)
class GPUHeatResources:
    signature: tuple[int, int, int, int, int]
    cell_state_tex: Any
    cell_state_out_tex: Any
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
    _run_ambient_exchange_feedback4,
    _run_ambient_diffuse,
    _run_ambient_feedback,
    _run_phase_boil_targets,
    _run_phase_targets,
    _run_boil_targets,
    _run_condense_targets,
    _run_apply_cell_targets,
    _run_apply_cell_aux_targets,
    _run_apply_gas_targets,
    _run_apply_terminal4x6,
    _run_apply_condense_cells,
    _run_apply_condense_cell_aux,
    _download_outputs,
    _empty_stage_targets,
    _publish_bridge_outputs,
    abort_deferred_cell_core,
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
        self._ambient_exchange_feedback4_enabled = True
        self._cell_heat_bridge_exchange_feedback4_fusion_enabled = True
        self._cell_heat_bridge_diffuse4_fusion_enabled = False
        self.last_cell_heat_bridge_diffuse4_fusion_used = False
        self._condense_apply_gas4x6_fusion_enabled = True
        self._terminal4x6_fusion_enabled = True
        self._terminal4x6_workgroup16x8_enabled = False
        self.last_terminal4x6_workgroup16x8_used = False
        self._terminal_bridge_aux_dirty_fusion_enabled = True
        self._terminal_bridge_aux_residency_enabled = False
        self.last_terminal_bridge_aux_residency_used = False
        self._terminal_phase_fusion_enabled = False
        self._terminal_dirty_publish_fusion_enabled = True
        self._terminal_dirty_workgroup_aggregation_enabled = True
        self.last_terminal_dirty_workgroup_aggregation_used = False
        self._terminal_split_target_active_reuse_enabled = True
        self._terminal_dead_condense_target_store_elision_enabled = True
        # Experimental: update the terminal's resident textures in place and
        # omit stores for fields whose bits did not change. The shader reaches
        # a workgroup barrier after all sampled cell reads before any writes.
        self._terminal_inplace_sparse_write_enabled = True
        # Experimental: compile the formal sparse-resident terminal's fixed
        # gate combination into the shader so unreachable optional paths do
        # not consume registers. The generic shader remains the fallback for
        # every other terminal configuration.
        self._terminal_sparse_resident_specialization_enabled = True
        self.last_terminal_sparse_resident_specialization_used = False
        # Experimental: the sparse terminal leaves unchanged resident fields
        # in place, so fetch large per-cell inputs only when a phase/boil/
        # condense action can actually modify them. The generic eager program
        # remains the fallback outside the strict formal resident gate.
        self._terminal_lazy_action_inputs_enabled = True
        self.last_terminal_lazy_action_inputs_used = False
        # Experimental: the formal lazy terminal consumes phase/boil targets as
        # bounded integer IDs. Pack both into the existing R32F phase target so
        # the split target pass and terminal exchange one word per cell. Lock
        # the first table generations used by the candidate; edited material or
        # reaction capabilities fall back to the canonical two-target ABI.
        self._packed_phase_boil_targets_enabled = True
        self._packed_phase_boil_table_signature: tuple[int, int] | None = None
        self.last_packed_phase_boil_targets_used = False
        self._terminal_hierarchical_row_summary_enabled = True
        self.last_terminal_hierarchical_row_summary_used = False
        # Replace the packed terminal's shared row summaries with seven uniform
        # NV32 warp ballots. Unsupported devices retain the row-summary path.
        self._terminal_nv32_ballot_gas_reduction_enabled = True
        self._terminal_nv32_ballot_supported = False
        self.last_terminal_nv32_ballot_gas_reduction_used = False
        self._heat_gas_bridge_residency_enabled = False
        self._terminal_bridge_gas_residency_enabled = False
        # Experimental: keep bridge-owned cell aux state resident for the whole
        # heat stage. The terminal only reads displaced material on an actual
        # placeholder transition, avoiding full-screen bridge-to-private aux
        # hydration. A later non-resident frame hydrates every private aux cache
        # before any pass can sample it.
        self._heat_sparse_bridge_residency_enabled = True
        self.last_heat_sparse_bridge_residency_used = False
        self.last_heat_gas_bridge_residency_used = False
        self.last_terminal_bridge_gas_residency_used = False
        self._heat_gas_bridge_residency_frame_id: int | None = None
        # Experimental: motion/reaction terminal already compares the final
        # same-frame state against the deferred bridge core and publishes the
        # collapse dirty queue. Keep disabled until formal A/B retention.
        self._deferred_dirty_publish_handoff_enabled = False
        self._terminal_bridge_aux_dirty_frame_id: int | None = None
        self._terminal_phase_fusion_frame_id: int | None = None
        self._terminal_dirty_publish_frame_id: int | None = None
        self._deferred_dirty_publish_handoff_frame_id: int | None = None
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}
        # Identifies the frame whose published cell textures can be consumed
        # directly by the formal reaction pipeline.
        self._last_formal_output_frame_id: int | None = None
        self._deferred_cell_core_frame_id: int | None = None
        self._motion_handoff_candidate: dict[str, Any] | None = None

    def _barrier_bits(self) -> tuple[str, ...]:
        """Synchronize heat outputs through their actual consumer paths.

        Heat writes textures and bridge SSBOs, then consumes them as sampled
        textures or storage buffers in later compute passes. No heat pass
        reads a prior image binding directly, so an image-cache barrier is
        unnecessary and adds a full image-cache flush on every pass.
        """
        return (
            "TEXTURE_FETCH_BARRIER_BIT",
            "SHADER_STORAGE_BARRIER_BIT",
        )

    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly inlined here verbatim).

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_tiles"] = build_compute_shader(ctx, "heat/load_active_tiles.comp", _SHADER_SUBS)
        self.programs["cell_heat"] = build_compute_shader(ctx, "heat/cell_heat.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["cell_heat_bridge"] = build_compute_shader(
            ctx, "heat/cell_heat_bridge.comp", _SHADER_SUBS, includes=["heat/_common.comp"]
        )
        self.programs["cell_heat_bridge_exchange_feedback4"] = build_compute_shader(
            ctx,
            "heat/cell_heat_bridge_exchange_feedback4.comp",
            _SHADER_SUBS,
            includes=["heat/_common.comp"],
        )
        self.programs["cell_heat_bridge_diffuse4_exchange_feedback4"] = build_compute_shader(
            ctx,
            "heat/cell_heat_bridge_diffuse4_exchange_feedback4.comp",
            _SHADER_SUBS,
            includes=["heat/_common.comp"],
        )
        self.programs["ambient_exchange"] = build_compute_shader(ctx, "heat/ambient_exchange.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_exchange_feedback4"] = build_compute_shader(
            ctx,
            "heat/ambient_exchange_feedback4.comp",
            _SHADER_SUBS,
            includes=["heat/_common.comp"],
        )
        self.programs["ambient_feedback"] = build_compute_shader(ctx, "heat/ambient_feedback.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["ambient_diffuse"] = build_compute_shader(ctx, "heat/ambient_diffuse.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["phase_targets"] = build_compute_shader(ctx, "heat/phase_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["phase_boil_targets"] = build_compute_shader(
            ctx, "heat/phase_boil_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"]
        )
        lazy_action_subs = dict(_SHADER_SUBS)
        lazy_action_subs["HEAT_LAZY_ACTION_INPUTS"] = 1
        self.programs["phase_boil_targets_lazy_action_inputs"] = build_compute_shader(
            ctx,
            "heat/phase_boil_targets.comp",
            lazy_action_subs,
            includes=["heat/_common.comp"],
        )
        packed_lazy_action_subs = dict(lazy_action_subs)
        packed_lazy_action_subs["HEAT_PACKED_PHASE_BOIL_TARGETS"] = 1
        self.programs[
            "phase_boil_targets_packed_lazy_action_inputs"
        ] = build_compute_shader(
            ctx,
            "heat/phase_boil_targets.comp",
            packed_lazy_action_subs,
            includes=["heat/_common.comp"],
        )
        self.programs["apply_cell_targets"] = build_compute_shader(ctx, "heat/apply_cell_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_cell_aux_targets"] = build_compute_shader(ctx, "heat/apply_cell_aux_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["boil_targets"] = build_compute_shader(ctx, "heat/boil_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["condense_targets"] = build_compute_shader(ctx, "heat/condense_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_gas_targets"] = build_compute_shader(ctx, "heat/apply_gas_targets.comp", _SHADER_SUBS, includes=["heat/_common.comp"])
        self.programs["apply_gas_targets4x6"] = build_compute_shader(
            ctx,
            "heat/apply_gas_targets4x6.comp",
            _SHADER_SUBS,
            includes=["heat/_common.comp"],
        )
        required_terminal_ballot_extensions = {
            "GL_NV_gpu_shader5",
            "GL_NV_shader_thread_group",
        }
        available_extensions = set(getattr(ctx, "extensions", ()))
        terminal_warp_size = 0
        if required_terminal_ballot_extensions.issubset(available_extensions):
            warp_size_program = build_compute_shader(
                ctx,
                "heat/query_nv_warp_size.comp",
            )
            warp_size_buffer = ctx.buffer(reserve=np.dtype(np.uint32).itemsize)
            try:
                warp_size_buffer.bind_to_storage_buffer(binding=0)
                warp_size_program.run(1, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
                terminal_warp_size = int(
                    np.frombuffer(warp_size_buffer.read(), dtype=np.uint32, count=1)[0]
                )
            finally:
                warp_size_buffer.release()
                warp_size_program.release()
        self._terminal_nv32_ballot_supported = terminal_warp_size == 32
        self.programs["apply_terminal4x6"] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            _SHADER_SUBS,
        )
        dirty_workgroup_subs = dict(_SHADER_SUBS)
        dirty_workgroup_subs["DIRTY_WORKGROUP_AGGREGATE"] = 1
        self.programs["apply_terminal4x6_dirty_workgroup"] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            dirty_workgroup_subs,
        )
        bridge_resident_subs = dict(_SHADER_SUBS)
        bridge_resident_subs["HEAT_GAS_BRIDGE_RESIDENT"] = 1
        self.programs["apply_terminal4x6_bridge_resident"] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            bridge_resident_subs,
        )
        bridge_resident_dirty_subs = dict(bridge_resident_subs)
        bridge_resident_dirty_subs["DIRTY_WORKGROUP_AGGREGATE"] = 1
        self.programs[
            "apply_terminal4x6_bridge_resident_dirty_workgroup"
        ] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            bridge_resident_dirty_subs,
        )
        sparse_resident_specialized_subs = dict(dirty_workgroup_subs)
        sparse_resident_specialized_subs["TERMINAL_SPARSE_RESIDENT_SPECIALIZED"] = 1
        self.programs["apply_terminal4x6_sparse_resident_specialized"] = (
            build_compute_shader(
                ctx,
                "heat/apply_terminal4x6.comp",
                sparse_resident_specialized_subs,
            )
        )
        sparse_resident_lazy_subs = dict(sparse_resident_specialized_subs)
        sparse_resident_lazy_subs["HEAT_LAZY_ACTION_INPUTS"] = 1
        self.programs["apply_terminal4x6_sparse_resident_lazy_action_inputs"] = (
            build_compute_shader(
                ctx,
                "heat/apply_terminal4x6.comp",
                sparse_resident_lazy_subs,
            )
        )
        sparse_resident_packed_lazy_subs = dict(sparse_resident_lazy_subs)
        sparse_resident_packed_lazy_subs["HEAT_PACKED_PHASE_BOIL_TARGETS"] = 1
        self.programs[
            "apply_terminal4x6_sparse_resident_packed_lazy_action_inputs"
        ] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            sparse_resident_packed_lazy_subs,
        )
        sparse_resident_packed_lazy_row_summary_subs = dict(
            sparse_resident_packed_lazy_subs
        )
        sparse_resident_packed_lazy_row_summary_subs[
            "TERMINAL_HIERARCHICAL_ROW_SUMMARY"
        ] = 1
        self.programs[
            "apply_terminal4x6_sparse_resident_packed_lazy_row_summary"
        ] = build_compute_shader(
            ctx,
            "heat/apply_terminal4x6.comp",
            sparse_resident_packed_lazy_row_summary_subs,
        )
        if self._terminal_nv32_ballot_supported:
            nv32_ballot_subs = dict(sparse_resident_packed_lazy_subs)
            nv32_ballot_subs["TERMINAL_NV32_GAS_BALLOT"] = 1
            nv32_ballot_subs["TERMINAL_NV32_EXTENSIONS"] = "\n".join(
                (
                    "#extension GL_NV_gpu_shader5 : require",
                    "#extension GL_NV_shader_thread_group : require",
                )
            )
            self.programs[
                "apply_terminal4x6_sparse_resident_packed_lazy_nv32_ballot"
            ] = build_compute_shader(
                ctx,
                "heat/apply_terminal4x6.comp",
                nv32_ballot_subs,
            )
        terminal16x8_subs = {
            **_SHADER_SUBS,
            "TERMINAL_LOCAL_SIZE_X": 16,
            "TERMINAL_LOCAL_SIZE_Y": 8,
            "TERMINAL_CELL_COUNT": 128,
            "TERMINAL_GAS_SIZE_X": 4,
            "TERMINAL_GAS_SIZE_Y": 2,
            "TERMINAL_GAS_COUNT": 8,
            "TERMINAL_CONDENSE_RANK_COUNT": 48,
        }
        terminal16x8_variants = {
            "apply_terminal4x6_workgroup16x8": terminal16x8_subs,
            "apply_terminal4x6_dirty_workgroup_workgroup16x8": {
                **terminal16x8_subs,
                "DIRTY_WORKGROUP_AGGREGATE": 1,
            },
            "apply_terminal4x6_bridge_resident_workgroup16x8": {
                **terminal16x8_subs,
                "HEAT_GAS_BRIDGE_RESIDENT": 1,
            },
            "apply_terminal4x6_bridge_resident_dirty_workgroup_workgroup16x8": {
                **terminal16x8_subs,
                "HEAT_GAS_BRIDGE_RESIDENT": 1,
                "DIRTY_WORKGROUP_AGGREGATE": 1,
            },
        }
        for program_name, substitutions in terminal16x8_variants.items():
            self.programs[program_name] = build_compute_shader(
                ctx,
                "heat/apply_terminal4x6.comp",
                substitutions,
            )
        self.programs["apply_condense_cells"] = build_compute_shader(ctx, "heat/apply_condense_cells.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["apply_condense_cell_aux"] = build_compute_shader(ctx, "heat/apply_condense_cell_aux.comp", _SHADER_SUBS, includes=["heat/_condense_common.comp"])
        self.programs["publish_bridge_cell"] = build_compute_shader(ctx, "heat/publish_bridge_cell.comp", _SHADER_SUBS)
        self.programs["publish_bridge_cell_aux_dirty"] = build_compute_shader(
            ctx, "heat/publish_bridge_cell_aux_dirty.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_gas"] = build_compute_shader(ctx, "heat/publish_bridge_gas.comp", _SHADER_SUBS)
        self.programs["publish_bridge_ambient"] = build_compute_shader(
            ctx, "heat/publish_bridge_ambient.comp", _SHADER_SUBS
        )
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
    _run_ambient_exchange_feedback4 = _run_ambient_exchange_feedback4
    _run_ambient_diffuse = _run_ambient_diffuse
    _run_ambient_feedback = _run_ambient_feedback
    _run_phase_boil_targets = _run_phase_boil_targets
    _run_phase_targets = _run_phase_targets
    _run_boil_targets = _run_boil_targets
    _run_condense_targets = _run_condense_targets
    _run_apply_cell_targets = _run_apply_cell_targets
    _run_apply_cell_aux_targets = _run_apply_cell_aux_targets
    _run_apply_gas_targets = _run_apply_gas_targets
    _run_apply_terminal4x6 = _run_apply_terminal4x6
    _run_apply_condense_cells = _run_apply_condense_cells
    _run_apply_condense_cell_aux = _run_apply_condense_cell_aux
    _download_outputs = _download_outputs
    _empty_stage_targets = _empty_stage_targets
    _publish_bridge_outputs = _publish_bridge_outputs
    abort_deferred_cell_core = abort_deferred_cell_core

    # ``_set_uniform_if_present`` and ``_sync_compute_writes`` are inherited
    # from :class:`GPUPipelineBase`; the heat pass uses the default barrier
    # bits (image-access | texture-fetch | shader-storage).
