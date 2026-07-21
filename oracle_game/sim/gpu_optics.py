from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.gpu import typed_light_id
from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader, shader_source
from oracle_game.types import CellFlag


LOCAL_SIZE = 8
RAY_LOCAL_SIZE = 64
MAX_LIGHTS = 8
MAX_MATERIALS = 256
MAX_EMITTERS = 256
MAX_RAY_STACK = 64
# DFS keeps at most one deferred sibling per bounce; four entries safely cover
# the default table's maximum of two while custom higher-bounce tables fall back.
BOUNDED_RAY_STACK = 4
BOUNDED_RAY_STACK_MAX_BOUNCE = 2
ACTIVE_CELL_TEXTURE_UNIT = 6
ACTIVE_GAS_TEXTURE_UNIT = 7
LIGHT_DOSE_GUARD_BUFFER = "optics_light_dose_guard"
OPTICS_CELL_ACCUM_SCALE = 2_097_152.0
OPTICS_GAS_ACCUM_SCALE = 4_194_304.0
OPTICS_ILLUM_ACCUM_SCALE = 2_097_152.0

# Superset of every {{NAME}} marker referenced by any optics shader; the loader
# ignores unused keys, so one shared dict suffices for all passes. The trace
# shift variant overrides ``GAS_CELL_COMPUTE`` locally (bitwise >> mapping).
_SHADER_SUBS = {
    "LOCAL_SIZE": LOCAL_SIZE,
    "RAY_LOCAL_SIZE": RAY_LOCAL_SIZE,
    "MAX_LIGHTS": MAX_LIGHTS,
    "LIGHT_PARAM_COUNT": MAX_LIGHTS * 2,
    "OPTICS_PARAMS_COUNT": MAX_MATERIALS * MAX_LIGHTS,
    "ACTIVE_CELL_TEXTURE_UNIT": ACTIVE_CELL_TEXTURE_UNIT,
    "ACTIVE_GAS_TEXTURE_UNIT": ACTIVE_GAS_TEXTURE_UNIT,
    "MAX_EMITTERS": MAX_EMITTERS,
    "MAX_RAY_STACK": MAX_RAY_STACK,
    "CELL_ACCUM_SCALE": repr(OPTICS_CELL_ACCUM_SCALE),
    "GAS_ACCUM_SCALE": repr(OPTICS_GAS_ACCUM_SCALE),
    "ILLUM_ACCUM_SCALE": repr(OPTICS_ILLUM_ACCUM_SCALE),
    "CELL_ACCUM_INV_SCALE": repr(1.0 / OPTICS_CELL_ACCUM_SCALE),
    "GAS_ACCUM_INV_SCALE": repr(1.0 / OPTICS_GAS_ACCUM_SCALE),
    "ILLUM_ACCUM_INV_SCALE": repr(1.0 / OPTICS_ILLUM_ACCUM_SCALE),
    "GAS_CELL_COMPUTE": "ivec2(cell.x / gas_cell_size, cell.y / gas_cell_size)",
    "SPARSE_LOCAL_SIZE": 64,
    "SPARSE_TILE_LOCAL_ATOMIC_MAX": 0,
    "SPARSE_GAS_OWNER_WARP_COMPACTION": 0,
    "SPARSE_WARP_EXTENSIONS": "",
    "TILE_SEEDED_BUILD": 0,
    "ACTIVE_TILE_SNAPSHOT": 0,
}


@dataclass(slots=True)
class GPUOpticsResources:
    signature: tuple[int, int, int, int, int]
    material_tex: Any
    active_cell_tex: Any
    active_gas_tex: Any
    cell_dose: Any
    gas_dose: Any
    illum_layers: Any
    cell_dose_accum: Any
    gas_dose_accum: Any
    illum_accum: Any
    visible_tex: Any
    emitter_buffer: Any
    emitter_count_buffer: Any
    light_buffer: Any
    optics_buffer: Any
    sparse_visible_marks: Any
    sparse_cell_list: Any
    sparse_gas_list: Any
    sparse_visible_list: Any
    sparse_cell_tile_marks: Any
    sparse_gas_tile_marks: Any
    sparse_cell_tile_list: Any
    sparse_gas_tile_list: Any
    sparse_runtime: Any
    sparse_initialized: bool = False
    sparse_generation: int = 0
    light_buffer_signature: tuple[int, int] | None = None
    optics_buffer_signature: tuple[int, int, int, int, int] | None = None


class GPUOpticsPipeline(GPUPipelineBase):
    def __init__(self) -> None:
        self.resources: GPUOpticsResources | None = None
        self.programs: dict[str, Any] = {}
        self.last_cpu_mirror_downloaded = False
        self.last_cpu_active_upload_skipped = False
        self.last_reaction_latch_clear_fused = False
        self.last_liquid_active_snapshot_used = False
        self._direct_bridge_visible_publish_enabled = True
        self._direct_visual_accumulator_compose_enabled = True
        self._sparse_optics_worklists_enabled = True
        self._sparse_gas_visible_scan_fusion_enabled = True
        self._sparse_tile_seeded_build_enabled = True
        # Experimental: aggregate indirect dispatch maxima per tile workgroup
        # instead of issuing one contended global atomicMax per sparse entry.
        self._sparse_tile_local_atomic_max_enabled = False
        self._sparse_gas_owner_warp_compaction_enabled = True
        self._sparse_gas_owner_warp_compaction_supported = False
        self.last_sparse_gas_owner_warp_compaction_used = False
        self._bounded_trace_stack_enabled = True
        # Experimental: full-active trace variants hard-code both activity
        # predicates, so their private mask textures have no consumers.
        self._full_active_mask_hydration_elision_enabled = True
        self.last_full_active_mask_hydration_elision_used = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly inlined here verbatim). The
    # ``_profile_enabled`` private helper is dead once inheritance takes over
    # and is removed. ``_reset_pass_profile`` is kept as an alias for the
    # inherited ``reset_pass_profile`` (tests call the underscore form directly).
    _reset_pass_profile = GPUPipelineBase.reset_pass_profile

    def step(
        self,
        world: "WorldEngine",
        emitters: list[dict[str, object]],
        *,
        solve_cell_mask: np.ndarray | None = None,
        solve_gas_mask: np.ndarray | None = None,
    ) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        self.last_reaction_latch_clear_fused = False
        self.last_liquid_active_snapshot_used = False
        self.last_full_active_mask_hydration_elision_used = False
        self.last_sparse_gas_owner_warp_compaction_used = False
        self.reset_pass_profile()
        if solve_cell_mask is None:
            solve_cell_mask = np.ones((world.height, world.width), dtype=np.bool_)
        if solve_gas_mask is None:
            solve_gas_mask = np.ones((world.gas_height, world.gas_width), dtype=np.bool_)
        with self._profile_pass(world, "optics_prepare_resources"):
            self._ensure_programs(ctx)
            resources = self._ensure_resources(world)
        use_sparse_worklists = bool(
            self._sparse_optics_worklists_enabled
            and self._formal_gpu_frame(world)
            and self._direct_bridge_visible_publish_enabled
            and self._direct_visual_accumulator_compose_enabled
        )
        use_tile_seeded_build = bool(
            use_sparse_worklists and self._sparse_tile_seeded_build_enabled
        )
        sparse_steady = bool(use_sparse_worklists and resources.sparse_initialized)
        force_all_active = self._trace_force_all_active(world)
        self._upload_inputs(
            world,
            resources,
            emitters,
            solve_cell_mask=solve_cell_mask,
            solve_gas_mask=solve_gas_mask,
            sparse_worklists=use_sparse_worklists,
            sparse_steady=sparse_steady,
            force_all_active=force_all_active,
        )
        with self._profile_pass(world, "optics_trace_emitters"):
            self._run_emitter_buffer_rays(
                world,
                resources,
                resources.emitter_buffer,
                resources.emitter_count_buffer,
                force_all_active=force_all_active,
                tile_seeded_build=use_tile_seeded_build,
            )
        if self._formal_gpu_frame(world) and "reaction_light_emitter_count" in world.bridge.gpu_authoritative_resources:
            with self._profile_pass(world, "optics_trace_reaction_emitters"):
                self._run_emitter_buffer_rays(
                    world,
                    resources,
                    world.bridge.buffers["reaction_light_emitter"],
                    world.bridge.buffers["reaction_light_emitter_count"],
                    force_all_active=force_all_active,
                    tile_seeded_build=use_tile_seeded_build,
                )
        if use_sparse_worklists:
            with self._profile_pass(world, "optics_sparse_build_worklists"):
                if use_tile_seeded_build:
                    self._build_tile_seeded_optics_worklists(world, resources)
                else:
                    self._build_sparse_optics_worklists(world, resources)
        if sparse_steady:
            if not self._sparse_gas_visible_scan_fusion_enabled:
                with self._profile_pass(world, "optics_sparse_expand_visible"):
                    self._expand_sparse_visible_worklist(world, resources)
            if not use_tile_seeded_build:
                with self._profile_pass(world, "optics_sparse_convert_accumulators"):
                    self._convert_sparse_accumulators(world, resources)
            with self._profile_pass(world, "optics_sparse_compose_visible"):
                self._compose_sparse_visible_illumination(world, resources)
        else:
            with self._profile_pass(world, "optics_convert_accumulators"):
                self._convert_accumulators(
                    world,
                    resources,
                    allow_direct_visual_accumulators=True,
                )
            with self._profile_pass(world, "optics_compose_visible"):
                self._compose_visible_illumination(
                    world,
                    resources,
                    allow_direct_visual_accumulators=True,
                )
        resources.sparse_initialized = use_sparse_worklists
        if not (
            self._formal_gpu_frame(world)
            and self._direct_bridge_visible_publish_enabled
        ):
            with self._profile_pass(world, "optics_publish_bridge"):
                self._publish_bridge_outputs(world, resources)
        self.last_cpu_mirror_downloaded = not self._formal_gpu_frame(world)
        if self.last_cpu_mirror_downloaded:
            ctx.finish()
            self._download_outputs(world, resources)

    def release(self) -> None:
        if self.resources is None:
            return
        for resource in (
            self.resources.material_tex,
            self.resources.active_cell_tex,
            self.resources.active_gas_tex,
            self.resources.cell_dose,
            self.resources.gas_dose,
            self.resources.illum_layers,
            self.resources.cell_dose_accum,
            self.resources.gas_dose_accum,
            self.resources.illum_accum,
            self.resources.visible_tex,
            self.resources.emitter_buffer,
            self.resources.emitter_count_buffer,
            self.resources.light_buffer,
            self.resources.optics_buffer,
            self.resources.sparse_visible_marks,
            self.resources.sparse_cell_list,
            self.resources.sparse_gas_list,
            self.resources.sparse_visible_list,
            self.resources.sparse_cell_tile_marks,
            self.resources.sparse_gas_tile_marks,
            self.resources.sparse_cell_tile_list,
            self.resources.sparse_gas_tile_list,
            self.resources.sparse_runtime,
        ):
            try:
                resource.release()
            except Exception:
                pass
        self.resources = None

    def invalidate_sparse_runtime(self) -> None:
        if self.resources is not None:
            self.resources.sparse_initialized = False

    def _ensure_resources(self, world: "WorldEngine") -> GPUOpticsResources:
        ctx = world.bridge.ctx
        assert ctx is not None
        signature = (world.width, world.height, world.gas_width, world.gas_height, world.cell_optical_dose.shape[0])
        if self.resources is not None and self.resources.signature == signature:
            return self.resources
        self.release()
        light_count = signature[4]
        material_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        active_cell_tex = ctx.texture((world.width, world.height), 1, dtype="f4")
        active_gas_tex = ctx.texture((world.gas_width, world.gas_height), 1, dtype="f4")
        cell_dose = ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4")
        gas_dose = ctx.texture_array((world.gas_width, world.gas_height, light_count), 1, dtype="f4")
        illum_layers = ctx.texture_array((world.width, world.height, light_count), 1, dtype="f4")
        visible_tex = ctx.texture((world.width, world.height), 4, dtype="f4")
        for texture in (material_tex, active_cell_tex, active_gas_tex, cell_dose, gas_dose, illum_layers, visible_tex):
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        emitter_buffer = ctx.buffer(reserve=MAX_EMITTERS * 8 * 4, dynamic=True)
        emitter_count_buffer = ctx.buffer(reserve=16 * 4, dynamic=True)
        light_buffer = ctx.buffer(reserve=MAX_LIGHTS * 2 * 4 * 4, dynamic=True)
        optics_buffer = ctx.buffer(reserve=MAX_MATERIALS * MAX_LIGHTS * 4 * 4, dynamic=True)
        cell_accum_size = world.width * world.height * light_count * 4
        gas_accum_size = world.gas_width * world.gas_height * light_count * 4
        cell_dose_accum = ctx.buffer(reserve=cell_accum_size, dynamic=True)
        gas_dose_accum = ctx.buffer(reserve=gas_accum_size, dynamic=True)
        illum_accum = ctx.buffer(reserve=cell_accum_size, dynamic=True)
        sparse_visible_marks = ctx.buffer(reserve=world.width * world.height * 4, dynamic=True)
        sparse_cell_list = ctx.buffer(reserve=cell_accum_size, dynamic=True)
        sparse_gas_list = ctx.buffer(reserve=gas_accum_size, dynamic=True)
        sparse_visible_list = ctx.buffer(reserve=world.width * world.height * 4, dynamic=True)
        sparse_cell_tile_count = ((world.width + LOCAL_SIZE - 1) // LOCAL_SIZE) * (
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        )
        sparse_gas_tile_count = ((world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE) * (
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
        )
        sparse_cell_tile_marks = ctx.buffer(reserve=max(4, sparse_cell_tile_count * 4), dynamic=True)
        sparse_gas_tile_marks = ctx.buffer(reserve=max(4, sparse_gas_tile_count * 4), dynamic=True)
        sparse_cell_tile_list = ctx.buffer(reserve=max(4, sparse_cell_tile_count * 4), dynamic=True)
        sparse_gas_tile_list = ctx.buffer(reserve=max(4, sparse_gas_tile_count * 4), dynamic=True)
        sparse_visible_marks.clear()
        sparse_cell_tile_marks.clear()
        sparse_gas_tile_marks.clear()
        sparse_runtime_values = np.zeros((32,), dtype=np.uint32)
        sparse_runtime_values[[5, 6, 9, 10, 13, 14, 21, 22, 25, 26]] = np.uint32(1)
        sparse_runtime = ctx.buffer(sparse_runtime_values.tobytes(), dynamic=True)
        self.resources = GPUOpticsResources(
            signature=signature,
            material_tex=material_tex,
            active_cell_tex=active_cell_tex,
            active_gas_tex=active_gas_tex,
            cell_dose=cell_dose,
            gas_dose=gas_dose,
            illum_layers=illum_layers,
            cell_dose_accum=cell_dose_accum,
            gas_dose_accum=gas_dose_accum,
            illum_accum=illum_accum,
            visible_tex=visible_tex,
            emitter_buffer=emitter_buffer,
            emitter_count_buffer=emitter_count_buffer,
            light_buffer=light_buffer,
            optics_buffer=optics_buffer,
            sparse_visible_marks=sparse_visible_marks,
            sparse_cell_list=sparse_cell_list,
            sparse_gas_list=sparse_gas_list,
            sparse_visible_list=sparse_visible_list,
            sparse_cell_tile_marks=sparse_cell_tile_marks,
            sparse_gas_tile_marks=sparse_gas_tile_marks,
            sparse_cell_tile_list=sparse_cell_tile_list,
            sparse_gas_tile_list=sparse_gas_tile_list,
            sparse_runtime=sparse_runtime,
        )
        return self.resources

    # ``_formal_gpu_frame`` is inherited from :class:`GPUPipelineBase`.

    def _trace_force_all_active(self, world: "WorldEngine") -> bool:
        authoritative = world.bridge.gpu_authoritative_resources
        return (
            self._formal_gpu_frame(world)
            and "active_tile_ttl" in authoritative
            and "reaction_light_emitter_count" in authoritative
        )

    @staticmethod
    def _gas_cell_size_power_of_two(gas_cell_size: int) -> bool:
        size = int(gas_cell_size)
        return size > 0 and (size & (size - 1)) == 0

    @classmethod
    def _gas_cell_shift(cls, gas_cell_size: int) -> int:
        if not cls._gas_cell_size_power_of_two(gas_cell_size):
            raise ValueError("gas_cell_size must be a positive power of two")
        return int(gas_cell_size).bit_length() - 1

    def _trace_emitter_program_name(
        self,
        world: "WorldEngine",
        *,
        force_all_active: bool,
        tile_seeded_build: bool = False,
    ) -> str:
        if force_all_active and self._gas_cell_size_power_of_two(int(world.gas_cell_size)):
            name = "trace_emitters_full_active_shift"
        elif force_all_active:
            name = "trace_emitters_full_active"
        else:
            name = "trace_emitters"
        if tile_seeded_build:
            name += "_tile_seeded"
        if (
            self._bounded_trace_stack_enabled
            and name == "trace_emitters_full_active_shift_tile_seeded"
            and self._trace_max_bounce_at_most(world, BOUNDED_RAY_STACK_MAX_BOUNCE)
        ):
            name += "_stack4"
        return name

    @staticmethod
    def _trace_max_bounce_at_most(world: "WorldEngine", maximum: int) -> bool:
        light_table = world.bridge.shadow_typed_tables.get("light_table")
        if light_table is None:
            return False
        count = min(MAX_LIGHTS, int(light_table.shape[0]))
        if count == 0:
            return True
        return bool(np.all(light_table[:count]["max_bounce"] <= int(maximum)))

    def _ensure_programs(self, ctx: Any) -> None:
        if self.programs:
            return
        self.programs["load_active_cell"] = build_compute_shader(
            ctx, "optics/load_active_cell.comp", _SHADER_SUBS, includes=["optics/_active_common.comp"]
        )
        self.programs["load_active_cell_snapshot"] = build_compute_shader(
            ctx,
            "optics/load_active_cell.comp",
            {**_SHADER_SUBS, "ACTIVE_TILE_SNAPSHOT": 1},
            includes=["optics/_active_common.comp"],
        )
        tile_seed_subs = {**_SHADER_SUBS, "TILE_SEEDED_BUILD": 1}
        self.programs["load_active_gas"] = build_compute_shader(
            ctx, "optics/load_active_gas.comp", _SHADER_SUBS, includes=["optics/_active_common.comp"]
        )
        self.programs["load_active_gas_snapshot"] = build_compute_shader(
            ctx,
            "optics/load_active_gas.comp",
            {**_SHADER_SUBS, "ACTIVE_TILE_SNAPSHOT": 1},
            includes=["optics/_active_common.comp"],
        )
        self.programs["trace_emitters"] = build_compute_shader(
            ctx, "optics/trace_body.comp", _SHADER_SUBS, includes=["optics/_trace_common.comp"]
        )
        self.programs["trace_emitters_full_active"] = build_compute_shader(
            ctx, "optics/trace_body.comp", _SHADER_SUBS, includes=["optics/_trace_common_full_active.comp"]
        )
        # Shift variant: power-of-two gas_cell_size enables a bitwise >> mapping
        # with a ``gas_cell_shift`` uniform. Both the preamble and the gas-cell
        # computation differ, so GAS_CELL_COMPUTE is overridden for this program.
        shift_subs = {
            **_SHADER_SUBS,
            "GAS_CELL_COMPUTE": "ivec2(cell.x >> gas_cell_shift, cell.y >> gas_cell_shift)",
        }
        self.programs["trace_emitters_full_active_shift"] = build_compute_shader(
            ctx, "optics/trace_body.comp", shift_subs, includes=["optics/_trace_common_full_active_shift.comp"]
        )
        self.programs["trace_emitters_tile_seeded"] = build_compute_shader(
            ctx, "optics/trace_body.comp", tile_seed_subs, includes=["optics/_trace_common.comp"]
        )
        self.programs["trace_emitters_full_active_tile_seeded"] = build_compute_shader(
            ctx,
            "optics/trace_body.comp",
            tile_seed_subs,
            includes=["optics/_trace_common_full_active.comp"],
        )
        tile_seed_shift_subs = {**shift_subs, "TILE_SEEDED_BUILD": 1}
        self.programs["trace_emitters_full_active_shift_tile_seeded"] = build_compute_shader(
            ctx,
            "optics/trace_body.comp",
            tile_seed_shift_subs,
            includes=["optics/_trace_common_full_active_shift.comp"],
        )
        self.programs["trace_emitters_full_active_shift_tile_seeded_stack4"] = build_compute_shader(
            ctx,
            "optics/trace_body.comp",
            {**tile_seed_shift_subs, "MAX_RAY_STACK": BOUNDED_RAY_STACK},
            includes=["optics/_trace_common_full_active_shift.comp"],
        )
        self.programs["convert_accumulators"] = build_compute_shader(
            ctx, "optics/convert_accumulators.comp", _SHADER_SUBS
        )
        self.programs["compose_visible"] = build_compute_shader(
            ctx, "optics/compose_visible.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_cell"] = build_compute_shader(
            ctx, "optics/publish_bridge_cell.comp", _SHADER_SUBS
        )
        self.programs["publish_bridge_gas"] = build_compute_shader(
            ctx, "optics/publish_bridge_gas.comp", _SHADER_SUBS
        )
        self.programs["clear_runtime_outputs"] = build_compute_shader(
            ctx, "optics/clear_runtime_outputs.comp", _SHADER_SUBS
        )
        self.programs["clear_bridge_outputs"] = build_compute_shader(
            ctx, "optics/clear_bridge_outputs.comp", _SHADER_SUBS
        )
        self.programs["clear_light_dose_guard"] = build_compute_shader(
            ctx, "optics/clear_light_dose_guard.comp", _SHADER_SUBS
        )
        self.programs["sparse_runtime_reset"] = build_compute_shader(
            ctx, "optics/sparse_runtime_reset.comp", _SHADER_SUBS
        )
        self.programs["sparse_clear_previous"] = build_compute_shader(
            ctx, "optics/sparse_clear_previous.comp", _SHADER_SUBS
        )
        self.programs["sparse_expand_visible"] = build_compute_shader(
            ctx, "optics/sparse_expand_visible.comp", _SHADER_SUBS
        )
        self.programs["sparse_build_worklists"] = build_compute_shader(
            ctx, "optics/sparse_build_worklists.comp", _SHADER_SUBS
        )
        self.programs["sparse_build_tile_worklists"] = build_compute_shader(
            ctx, "optics/sparse_build_tile_worklists.comp", _SHADER_SUBS
        )
        self.programs["sparse_build_tile_worklists_local_atomic_max"] = build_compute_shader(
            ctx,
            "optics/sparse_build_tile_worklists.comp",
            {**_SHADER_SUBS, "SPARSE_TILE_LOCAL_ATOMIC_MAX": 1},
        )
        required_owner_warp_extensions = {
            "GL_NV_gpu_shader5",
            "GL_NV_shader_thread_group",
            "GL_NV_shader_thread_shuffle",
        }
        available_extensions = set(getattr(ctx, "extensions", ()))
        owner_warp_size = 0
        if (
            self._sparse_gas_owner_warp_compaction_enabled
            and required_owner_warp_extensions.issubset(available_extensions)
        ):
            warp_size_program = build_compute_shader(
                ctx,
                "optics/query_nv_warp_size.comp",
            )
            warp_size_buffer = ctx.buffer(reserve=np.dtype(np.uint32).itemsize)
            try:
                warp_size_buffer.bind_to_storage_buffer(binding=0)
                warp_size_program.run(1, 1, 1)
                ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT)
                owner_warp_size = int(
                    np.frombuffer(warp_size_buffer.read(), dtype=np.uint32, count=1)[0]
                )
            finally:
                warp_size_buffer.release()
                warp_size_program.release()
        self._sparse_gas_owner_warp_compaction_supported = owner_warp_size == 32
        if self._sparse_gas_owner_warp_compaction_supported:
            self.programs["sparse_build_tile_worklists_gas_owner_nv32"] = build_compute_shader(
                ctx,
                "optics/sparse_build_tile_worklists.comp",
                {
                    **_SHADER_SUBS,
                    "SPARSE_GAS_OWNER_WARP_COMPACTION": 1,
                    "SPARSE_WARP_EXTENSIONS": "\n".join(
                        (
                            "#extension GL_NV_gpu_shader5 : require",
                            "#extension GL_NV_shader_thread_group : require",
                            "#extension GL_NV_shader_thread_shuffle : require",
                        )
                    ),
                },
            )
        self.programs["sparse_clear_reaction_latches"] = build_compute_shader(
            ctx, "optics/sparse_clear_reaction_latches.comp", _SHADER_SUBS
        )
        self.programs["sparse_convert_accumulators"] = build_compute_shader(
            ctx, "optics/sparse_convert_accumulators.comp", _SHADER_SUBS
        )
        self.programs["sparse_compose_visible"] = build_compute_shader(
            ctx, "optics/sparse_compose_visible.comp", _SHADER_SUBS
        )

    def _upload_inputs(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        emitters: list[dict[str, object]],
        *,
        solve_cell_mask: np.ndarray,
        solve_gas_mask: np.ndarray,
        sparse_worklists: bool = False,
        sparse_steady: bool = False,
        force_all_active: bool = False,
    ) -> int:
        formal_gpu_frame = self._formal_gpu_frame(world)
        world._require_gpu_authoritative_resources("optics input", "material", "active_tile_ttl")
        with self._profile_pass(world, "optics_upload_inputs.tables"):
            world.bridge.sync_rule_tables(world)
            light_table = world.bridge.shadow_typed_tables["light_table"]
            light_signature = (world.bridge.table_generations.get("lights", 0), int(light_table.shape[0]))
            count = min(MAX_LIGHTS, int(light_table.shape[0]))
            if resources.light_buffer_signature != light_signature:
                light_colors = np.zeros((MAX_LIGHTS * 2, 4), dtype="f4")
                light_colors[:count, :3] = light_table[:count]["color"]
                light_colors[:count, 3] = light_table[:count]["dose_channel_id"].astype(np.float32)
                light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 0] = light_table[:count]["visual_channel"].astype(
                    np.float32
                )
                light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 1] = light_table[:count]["render_style_id"].astype(
                    np.float32
                )
                light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 2] = light_table[:count]["max_bounce"].astype(np.float32)
                light_colors[MAX_LIGHTS : MAX_LIGHTS + count, 3] = light_table[:count]["default_range"].astype(
                    np.float32
                )
                resources.light_buffer.write(light_colors.tobytes())
                resources.light_buffer_signature = light_signature

            material_table = world.bridge.shadow_typed_tables["material_table"]
            optics_table = world.bridge.shadow_typed_tables["optics_table"]
            optics_signature = (
                world.bridge.table_generations.get("materials", 0),
                world.bridge.table_generations.get("lights", 0),
                world.bridge.table_generations.get("optics", 0),
                int(material_table.shape[0]),
                int(optics_table.shape[0]),
            )
            if resources.optics_buffer_signature != optics_signature:
                optics = np.zeros((MAX_MATERIALS * MAX_LIGHTS, 4), dtype="f4")
                for row in optics_table:
                    material_id = int(row["material_id"])
                    light_id = int(row["light_type_id"])
                    if material_id < 0 or material_id >= MAX_MATERIALS or light_id < 0 or light_id >= MAX_LIGHTS:
                        continue
                    optics[material_id * MAX_LIGHTS + light_id] = (
                        float(row["absorption"]),
                        float(row["scattering"]),
                        float(row["refraction"]),
                        0.0,
                    )
                resources.optics_buffer.write(optics.tobytes())
                resources.optics_buffer_signature = optics_signature

        active_authoritative = formal_gpu_frame and "active_tile_ttl" in world.bridge.gpu_authoritative_resources
        self.last_cpu_active_upload_skipped = bool(active_authoritative)
        skip_active_mask_hydration = bool(
            self._full_active_mask_hydration_elision_enabled
            and active_authoritative
            and force_all_active
        )
        self.last_full_active_mask_hydration_elision_used = (
            skip_active_mask_hydration
        )

        with self._profile_pass(world, "optics_upload_inputs.material"):
            if not self._bridge_material_authoritative(world):
                resources.material_tex.write(world.material_id.astype("f4").tobytes())

        with self._profile_pass(world, "optics_upload_inputs.clear_runtime"):
            if sparse_steady:
                self._clear_sparse_previous_outputs(world, resources)
            else:
                self._clear_runtime_outputs(world, resources)
                if sparse_worklists:
                    self._reset_sparse_runtime(world, resources)

        with self._profile_pass(world, "optics_upload_inputs.emitters"):
            emitter_data = np.zeros((MAX_EMITTERS * 2, 4), dtype="f4")
            emitter_count = 0
            for emitter in emitters:
                if emitter_count >= MAX_EMITTERS:
                    break
                light_id = typed_light_id(light_table, str(emitter["light_type"]))
                if light_id < 0:
                    continue
                direction = emitter["direction"]
                emitter_data[emitter_count * 2] = (
                    float(emitter["origin"][0]),
                    float(emitter["origin"][1]),
                    float(direction[0]),
                    float(direction[1]),
                )
                emitter_data[emitter_count * 2 + 1] = (
                    float(emitter["strength"]),
                    float(emitter["range_cells"]),
                    float(emitter["spread"]),
                    float(light_id),
                )
                emitter_count += 1
            resources.emitter_buffer.write(emitter_data.tobytes())
            emitter_counts = np.zeros((16,), dtype=np.uint32)
            emitter_counts[0] = np.uint32(emitter_count)
            resources.emitter_count_buffer.write(emitter_counts.tobytes())

        with self._profile_pass(world, "optics_upload_inputs.active_masks"):
            if active_authoritative:
                if not skip_active_mask_hydration:
                    self._load_authoritative_active_masks(
                        world,
                        resources,
                        force_all_active=force_all_active,
                    )
            else:
                resources.active_cell_tex.write(np.asarray(solve_cell_mask, dtype="f4").tobytes())
                resources.active_gas_tex.write(np.asarray(solve_gas_mask, dtype="f4").tobytes())
        return emitter_count

    def _clear_runtime_outputs(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        program = self.programs["clear_runtime_outputs"]
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["dose_channel_count"].value = dose_channel_count
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.illum_accum.bind_to_storage_buffer(binding=2)
        self._ensure_light_dose_guard(world).bind_to_storage_buffer(binding=3)
        groups_x = (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, dose_channel_count)
        self._sync_compute_writes(ctx)
        world.bridge.mark_gpu_authoritative(LIGHT_DOSE_GUARD_BUFFER)

    def _next_sparse_generation(self, resources: GPUOpticsResources) -> int:
        generation = int(resources.sparse_generation) + 1
        if generation >= 0xFFFFFFFF:
            resources.sparse_visible_marks.clear()
            resources.sparse_cell_tile_marks.clear()
            resources.sparse_gas_tile_marks.clear()
            generation = 1
        resources.sparse_generation = generation
        return generation

    @staticmethod
    def _sync_sparse_indirect_writes(ctx: Any) -> None:
        ctx.memory_barrier(
            ctx.SHADER_STORAGE_BARRIER_BIT
            | ctx.COMMAND_BARRIER_BIT
            | ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
            | ctx.TEXTURE_FETCH_BARRIER_BIT
        )

    def _reset_sparse_runtime(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        self._next_sparse_generation(resources)
        resources.sparse_runtime.bind_to_storage_buffer(binding=0)
        self._ensure_light_dose_guard(world).bind_to_storage_buffer(binding=1)
        self.programs["sparse_runtime_reset"].run(1, 1, 1)
        self._sync_sparse_indirect_writes(ctx)
        world.bridge.mark_gpu_authoritative(LIGHT_DOSE_GUARD_BUFFER)

    def _clear_sparse_previous_outputs(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        program = self.programs["sparse_clear_previous"]
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.illum_accum.bind_to_storage_buffer(binding=2)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=3)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=4)
        resources.sparse_cell_list.bind_to_storage_buffer(binding=5)
        resources.sparse_gas_list.bind_to_storage_buffer(binding=6)
        resources.sparse_visible_list.bind_to_storage_buffer(binding=7)
        resources.sparse_runtime.bind_to_storage_buffer(binding=8)
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(1, read=False, write=True)
        for clear_kind, offset in ((0, 16), (1, 32), (2, 48)):
            program["clear_kind"].value = clear_kind
            program.run_indirect(resources.sparse_runtime, offset=offset)
        self._sync_compute_writes(ctx)
        self._sync_sparse_indirect_writes(ctx)
        self._reset_sparse_runtime(world, resources)

    def _expand_sparse_visible_worklist(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        program = self.programs["sparse_expand_visible"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        self._set_uniform_if_present(
            program, "gas_cell_size", int(world.gas_cell_size)
        )
        self._set_uniform_if_present(
            program, "sparse_generation", int(resources.sparse_generation)
        )
        resources.sparse_gas_list.bind_to_storage_buffer(binding=0)
        resources.sparse_visible_marks.bind_to_storage_buffer(binding=1)
        resources.sparse_visible_list.bind_to_storage_buffer(binding=2)
        resources.sparse_runtime.bind_to_storage_buffer(binding=3)
        program.run_indirect(resources.sparse_runtime, offset=32)
        self._sync_sparse_indirect_writes(ctx)

    def _build_sparse_optics_worklists(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        bridge = world.bridge
        program = self.programs["sparse_build_worklists"]
        cell_count = int(world.width * world.height)
        gas_count = int(world.gas_width * world.gas_height)
        dose_channels = int(world.cell_optical_dose.shape[0])
        cell_entry_count = cell_count * dose_channels
        gas_entry_count = gas_count * dose_channels
        clear_latches = "cell_core" in bridge.gpu_authoritative_resources
        program["cell_entry_count"].value = cell_entry_count
        program["gas_entry_count"].value = gas_entry_count
        program["cell_count"].value = cell_count
        program["gas_cell_count"].value = gas_count
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["dose_channel_count"].value = dose_channels
        self._set_uniform_if_present(
            program,
            "expand_gas_visible",
            bool(self._sparse_gas_visible_scan_fusion_enabled),
        )
        self._set_uniform_if_present(
            program, "sparse_generation", int(resources.sparse_generation)
        )
        program["clear_reaction_latches"].value = clear_latches
        program["reaction_latched_flag_shifted"].value = int(CellFlag.REACTION_LATCHED) << 24
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.illum_accum.bind_to_storage_buffer(binding=2)
        resources.sparse_visible_marks.bind_to_storage_buffer(binding=3)
        resources.sparse_cell_list.bind_to_storage_buffer(binding=4)
        resources.sparse_gas_list.bind_to_storage_buffer(binding=5)
        resources.sparse_visible_list.bind_to_storage_buffer(binding=6)
        resources.sparse_runtime.bind_to_storage_buffer(binding=7)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=8)
        invocation_count = max(cell_entry_count, gas_entry_count)
        program.run((invocation_count + 63) // 64, 1, 1)
        self._sync_sparse_indirect_writes(ctx)
        if clear_latches:
            bridge.mark_gpu_authoritative("cell_core", "material")
            self.last_reaction_latch_clear_fused = True

    def _build_tile_seeded_optics_worklists(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        cell_program = self.programs[
            "sparse_build_tile_worklists_local_atomic_max"
            if self._sparse_tile_local_atomic_max_enabled
            else "sparse_build_tile_worklists"
        ]
        use_owner_warp_compaction = bool(
            self._sparse_gas_owner_warp_compaction_enabled
            and self._sparse_gas_owner_warp_compaction_supported
            and int(world.gas_cell_size) == 4
            and self._sparse_gas_visible_scan_fusion_enabled
        )
        gas_program = (
            self.programs["sparse_build_tile_worklists_gas_owner_nv32"]
            if use_owner_warp_compaction
            else cell_program
        )
        self.last_sparse_gas_owner_warp_compaction_used = use_owner_warp_compaction
        cell_tile_grid = (
            (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        )
        gas_tile_grid = (
            (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
            (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
        )
        programs = [cell_program]
        if gas_program is not cell_program:
            programs.append(gas_program)
        for program in programs:
            program["cell_grid_size"].value = (world.width, world.height)
            program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            program["cell_sparse_tile_grid_size"].value = cell_tile_grid
            program["gas_sparse_tile_grid_size"].value = gas_tile_grid
            self._set_uniform_if_present(
                program, "gas_cell_size", int(world.gas_cell_size)
            )
            program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
            self._set_uniform_if_present(
                program,
                "expand_gas_visible",
                bool(self._sparse_gas_visible_scan_fusion_enabled),
            )
            self._set_uniform_if_present(
                program, "sparse_generation", int(resources.sparse_generation)
            )
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.illum_accum.bind_to_storage_buffer(binding=2)
        resources.sparse_visible_marks.bind_to_storage_buffer(binding=3)
        resources.sparse_cell_list.bind_to_storage_buffer(binding=4)
        resources.sparse_gas_list.bind_to_storage_buffer(binding=5)
        resources.sparse_visible_list.bind_to_storage_buffer(binding=6)
        resources.sparse_runtime.bind_to_storage_buffer(binding=7)
        resources.sparse_cell_tile_list.bind_to_storage_buffer(binding=8)
        resources.sparse_gas_tile_list.bind_to_storage_buffer(binding=9)
        world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=10)
        world.bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=11)

        # Trace shaders publish these indirect counts, including reaction-light
        # traces that run immediately before this pass.
        self._sync_sparse_indirect_writes(ctx)
        for program, build_kind, offset, profile_name in (
            (cell_program, 0, 80, "optics_sparse_build_cell_tiles"),
            (gas_program, 1, 96, "optics_sparse_build_gas_tiles"),
        ):
            with self._profile_pass(world, profile_name):
                program["build_kind"].value = build_kind
                program.run_indirect(resources.sparse_runtime, offset=offset)
                self._sync_sparse_indirect_writes(ctx)
        world.bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")
        handoff_cleared_latches = bool(
            getattr(world, "_reaction_latches_handoff_cleared_frame_id", None)
            == int(getattr(world, "frame_id", 0))
        )
        if (
            "cell_core" in world.bridge.gpu_authoritative_resources
            and not handoff_cleared_latches
        ):
            clear_program = self.programs["sparse_clear_reaction_latches"]
            clear_program["cell_count"].value = int(world.width * world.height)
            clear_program["reaction_latched_flag_shifted"].value = int(CellFlag.REACTION_LATCHED) << 24
            world.bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
            clear_program.run((world.width * world.height + 255) // 256, 1, 1)
            self._sync_compute_writes(ctx)
            world.bridge.mark_gpu_authoritative("cell_core", "material")
            self.last_reaction_latch_clear_fused = True
        elif handoff_cleared_latches:
            self.last_reaction_latch_clear_fused = True

    def _convert_sparse_accumulators(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        bridge = world.bridge
        program = self.programs["sparse_convert_accumulators"]
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=2)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=3)
        resources.sparse_cell_list.bind_to_storage_buffer(binding=4)
        resources.sparse_gas_list.bind_to_storage_buffer(binding=5)
        resources.sparse_runtime.bind_to_storage_buffer(binding=6)
        for convert_kind, offset in ((0, 16), (1, 32)):
            program["convert_kind"].value = convert_kind
            program.run_indirect(resources.sparse_runtime, offset=offset)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")

    def _compose_sparse_visible_illumination(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        bridge = world.bridge
        program = self.programs["sparse_compose_visible"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = int(world.gas_cell_size)
        program["light_count"].value = min(
            MAX_LIGHTS,
            int(world.bridge.shadow_typed_tables["light_table"].shape[0]),
        )
        program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
        resources.illum_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.light_buffer.bind_to_storage_buffer(binding=2)
        resources.sparse_visible_list.bind_to_storage_buffer(binding=3)
        resources.sparse_runtime.bind_to_storage_buffer(binding=4)
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(1, read=False, write=True)
        program.run_indirect(resources.sparse_runtime, offset=48)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative("light", "visible_illumination")

    def _ensure_light_dose_guard(self, world: "WorldEngine") -> Any:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge GPU resources for light dose guard")
        guard = bridge.buffers.get(LIGHT_DOSE_GUARD_BUFFER)
        if guard is None:
            guard = bridge.ctx.buffer(np.zeros((4,), dtype=np.uint32).tobytes(), dynamic=True)
            bridge.buffers[LIGHT_DOSE_GUARD_BUFFER] = guard
        return guard

    def clear_light_dose_guard(self, world: "WorldEngine") -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        self.last_reaction_latch_clear_fused = False
        self._ensure_programs(ctx)
        guard = self._ensure_light_dose_guard(world)
        guard.bind_to_storage_buffer(binding=0)
        self.programs["clear_light_dose_guard"].run(1, 1, 1)
        self._sync_compute_writes(ctx)
        world.bridge.mark_gpu_authoritative(LIGHT_DOSE_GUARD_BUFFER)

    def _load_authoritative_active_masks(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        *,
        force_all_active: bool,
    ) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge active scheduler resources")
        liquid_pipeline = getattr(getattr(world, "liquid_solver", None), "gpu_pipeline", None)
        liquid_resources = getattr(liquid_pipeline, "resources", None)
        active_tile_snapshot = None
        if (
            liquid_resources is not None
            and bool(getattr(liquid_pipeline, "last_flow_active_decay_fusion_used", False))
            and getattr(liquid_pipeline, "_flow_active_decay_fusion_frame_id", None)
            == int(getattr(world, "frame_id", 0))
        ):
            active_tile_snapshot = liquid_resources.active_tile_tex
        self.last_liquid_active_snapshot_used = active_tile_snapshot is not None
        for name, texture, width, height in (
            ("load_active_cell", resources.active_cell_tex, world.width, world.height),
            ("load_active_gas", resources.active_gas_tex, world.gas_width, world.gas_height),
        ):
            program = self.programs[
                f"{name}_snapshot" if active_tile_snapshot is not None else name
            ]
            self._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
            self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
            self._set_uniform_if_present(program, "tile_grid_size", (world.active.tile_width, world.active.tile_height))
            self._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
            self._set_uniform_if_present(program, "tile_size", int(world.active.tile_size))
            self._set_uniform_if_present(program, "expansion_radius", 1)
            self._set_uniform_if_present(program, "force_all_active", bool(force_all_active))
            if active_tile_snapshot is not None:
                active_tile_snapshot.use(location=5)
            else:
                bridge.buffers["active_tile_ttl"].bind_to_storage_buffer(binding=0)
            resources.emitter_buffer.bind_to_storage_buffer(binding=1)
            resources.emitter_count_buffer.bind_to_storage_buffer(binding=2)
            resources.light_buffer.bind_to_storage_buffer(binding=3)
            texture.bind_to_image(4, read=False, write=True)
            program.run(
                (int(width) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (int(height) + LOCAL_SIZE - 1) // LOCAL_SIZE,
                1,
            )
        self._sync_compute_writes(bridge.ctx)

    # ``_set_uniform_if_present`` is inherited from :class:`GPUPipelineBase`.

    def _bridge_material_authoritative(self, world: "WorldEngine") -> bool:
        return (
            self._formal_gpu_frame(world)
            and "material" in world.bridge.gpu_authoritative_resources
            and "material" in world.bridge.textures
        )

    def _bind_material_input(self, world: "WorldEngine", resources: GPUOpticsResources, *, location: int = 0) -> None:
        if self._bridge_material_authoritative(world):
            world.bridge.textures["material"].use(location=location)
            return
        resources.material_tex.use(location=location)

    def _run_emitter_buffer_rays(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        emitter_buffer: Any,
        emitter_count_buffer: Any,
        *,
        force_all_active: bool,
        tile_seeded_build: bool = False,
    ) -> None:
        program_name = self._trace_emitter_program_name(
            world,
            force_all_active=force_all_active,
            tile_seeded_build=tile_seeded_build,
        )
        program = self.programs[program_name]
        ctx = world.bridge.ctx
        assert ctx is not None
        program["cell_grid_size"].value = (world.width, world.height)
        self._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
        self._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
        if program_name.startswith("trace_emitters_full_active_shift"):
            program["gas_cell_shift"].value = self._gas_cell_shift(int(world.gas_cell_size))
        program["max_emitters"].value = MAX_EMITTERS
        program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
        self._bind_material_input(world, resources, location=0)
        resources.active_cell_tex.use(location=ACTIVE_CELL_TEXTURE_UNIT)
        resources.active_gas_tex.use(location=ACTIVE_GAS_TEXTURE_UNIT)
        emitter_buffer.bind_to_storage_buffer(binding=1)
        emitter_count_buffer.bind_to_storage_buffer(binding=2)
        resources.optics_buffer.bind_to_storage_buffer(binding=3)
        resources.light_buffer.bind_to_storage_buffer(binding=4)
        self._ensure_light_dose_guard(world).bind_to_storage_buffer(binding=0)
        resources.cell_dose_accum.bind_to_storage_buffer(binding=5)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=6)
        resources.illum_accum.bind_to_storage_buffer(binding=7)
        if tile_seeded_build:
            cell_tile_grid = (
                (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            )
            gas_tile_grid = (
                (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE,
                (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE,
            )
            program["cell_sparse_tile_grid_size"].value = cell_tile_grid
            program["gas_sparse_tile_grid_size"].value = gas_tile_grid
            program["sparse_generation"].value = int(resources.sparse_generation)
            resources.sparse_cell_tile_marks.bind_to_storage_buffer(binding=8)
            resources.sparse_gas_tile_marks.bind_to_storage_buffer(binding=9)
            resources.sparse_cell_tile_list.bind_to_storage_buffer(binding=10)
            resources.sparse_gas_tile_list.bind_to_storage_buffer(binding=11)
            resources.sparse_runtime.bind_to_storage_buffer(binding=12)
        lane_count = MAX_EMITTERS * 8
        groups_x = (lane_count + RAY_LOCAL_SIZE - 1) // RAY_LOCAL_SIZE
        program.run(groups_x, 1, 1)
        self._sync_compute_writes(ctx)
        world.bridge.mark_gpu_authoritative(LIGHT_DOSE_GUARD_BUFFER)

    def _use_direct_visual_accumulators(
        self,
        world: "WorldEngine",
        *,
        allowed: bool,
    ) -> bool:
        return bool(
            allowed
            and self._formal_gpu_frame(world)
            and self._direct_bridge_visible_publish_enabled
            and self._direct_visual_accumulator_compose_enabled
        )

    def _convert_accumulators(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        *,
        allow_direct_visual_accumulators: bool = False,
    ) -> None:
        program = self.programs["convert_accumulators"]
        ctx = world.bridge.ctx
        assert ctx is not None
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        publish_bridge_dose = self._formal_gpu_frame(world)
        write_visual_textures = not self._use_direct_visual_accumulators(
            world,
            allowed=allow_direct_visual_accumulators,
        )
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["dose_channel_count"].value = dose_channel_count
        program["publish_bridge_dose"].value = publish_bridge_dose
        program["write_cell_dose_texture"].value = not publish_bridge_dose
        program["write_visual_textures"].value = write_visual_textures
        resources.cell_dose_accum.bind_to_storage_buffer(binding=0)
        resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        resources.illum_accum.bind_to_storage_buffer(binding=2)
        resources.cell_dose.bind_to_image(3, read=False, write=True)
        if write_visual_textures:
            resources.gas_dose.bind_to_image(4, read=False, write=True)
            resources.illum_layers.bind_to_image(5, read=False, write=True)
        if publish_bridge_dose:
            world.bridge.ensure_world_resources(world)
            world.bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=3)
            world.bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=4)
        groups_x = (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, dose_channel_count)
        self._sync_compute_writes(ctx)
        if publish_bridge_dose:
            world.bridge.mark_gpu_authoritative("cell_optical_dose", "gas_optical_dose")

    def _compose_visible_illumination(
        self,
        world: "WorldEngine",
        resources: GPUOpticsResources,
        *,
        allow_direct_visual_accumulators: bool = False,
    ) -> None:
        ctx = world.bridge.ctx
        assert ctx is not None
        direct_bridge_outputs = bool(
            self._formal_gpu_frame(world)
            and self._direct_bridge_visible_publish_enabled
        )
        use_accumulator_inputs = self._use_direct_visual_accumulators(
            world,
            allowed=allow_direct_visual_accumulators,
        )
        program = self.programs["compose_visible"]
        bridge = world.bridge
        fuse_latch_clear = False
        if direct_bridge_outputs:
            bridge.ensure_world_resources(world)
            if not bridge.enabled or bridge.ctx is None:
                raise RuntimeError("GPU optics pipeline requires bridge GPU resources for authoritative visible output")
            fuse_latch_clear = "cell_core" in bridge.gpu_authoritative_resources
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["gas_cell_size"].value = world.gas_cell_size
        program["light_count"].value = min(MAX_LIGHTS, int(world.bridge.shadow_typed_tables["light_table"].shape[0]))
        program["dose_channel_count"].value = int(world.cell_optical_dose.shape[0])
        program["publish_bridge_outputs"].value = direct_bridge_outputs
        program["use_accumulator_inputs"].value = use_accumulator_inputs
        program["clear_reaction_latches"].value = fuse_latch_clear
        program["reaction_latched_flag_shifted"].value = int(CellFlag.REACTION_LATCHED) << 24
        if use_accumulator_inputs:
            resources.illum_accum.bind_to_storage_buffer(binding=0)
            resources.gas_dose_accum.bind_to_storage_buffer(binding=1)
        else:
            resources.illum_layers.use(location=1)
            resources.gas_dose.use(location=2)
        if direct_bridge_outputs:
            bridge.textures["light"].bind_to_image(6, read=False, write=True)
            bridge.textures["visible_illumination"].bind_to_image(7, read=False, write=True)
            bridge.buffers["cell_core"].bind_to_storage_buffer(binding=5)
        else:
            resources.visible_tex.bind_to_image(6, read=False, write=True)
        resources.light_buffer.bind_to_storage_buffer(binding=4)
        groups_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, 1)
        self._sync_compute_writes(ctx)
        if direct_bridge_outputs:
            bridge.mark_gpu_authoritative(
                "light",
                "visible_illumination",
            )
            if fuse_latch_clear:
                bridge.mark_gpu_authoritative("cell_core", "material")
                self.last_reaction_latch_clear_fused = True

    def _download_outputs(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        light_count = world.cell_optical_dose.shape[0]
        cell = np.frombuffer(resources.cell_dose.read(), dtype="f4").reshape((light_count, world.height, world.width))
        gas = np.frombuffer(resources.gas_dose.read(), dtype="f4").reshape((light_count, world.gas_height, world.gas_width))
        visible = np.frombuffer(resources.visible_tex.read(), dtype="f4").reshape((world.height, world.width, 4))
        world.cell_optical_dose[:] = cell
        world.gas_optical_dose[:] = gas
        world.visible_illumination[:] = visible[..., :3]

    def _publish_bridge_outputs(self, world: "WorldEngine", resources: GPUOpticsResources) -> None:
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge GPU resources for authoritative optics state")
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        direct_bridge_dose = self._formal_gpu_frame(world)
        fuse_latch_clear = bool(direct_bridge_dose and "cell_core" in bridge.gpu_authoritative_resources)
        cell_program = self.programs["publish_bridge_cell"]
        cell_program["cell_grid_size"].value = (world.width, world.height)
        cell_program["dose_channel_count"].value = dose_channel_count
        cell_program["publish_cell_dose"].value = not direct_bridge_dose
        cell_program["clear_reaction_latches"].value = fuse_latch_clear
        cell_program["reaction_latched_flag_shifted"].value = int(CellFlag.REACTION_LATCHED) << 24
        cell_program["visible_tex"].value = 0
        cell_program["cell_dose_tex"].value = 1
        resources.visible_tex.use(location=0)
        resources.cell_dose.use(location=1)
        bridge.textures["light"].bind_to_image(2, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(3, read=False, write=True)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=4)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=5)
        cell_group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        cell_program.run(cell_group_x, cell_group_y, 1 if direct_bridge_dose else dose_channel_count)

        if not direct_bridge_dose:
            gas_program = self.programs["publish_bridge_gas"]
            gas_program["gas_grid_size"].value = (world.gas_width, world.gas_height)
            gas_program["dose_channel_count"].value = dose_channel_count
            gas_program["gas_dose_tex"].value = 0
            resources.gas_dose.use(location=0)
            bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=1)
            gas_group_x = (world.gas_width + LOCAL_SIZE - 1) // LOCAL_SIZE
            gas_group_y = (world.gas_height + LOCAL_SIZE - 1) // LOCAL_SIZE
            gas_program.run(gas_group_x, gas_group_y, dose_channel_count)

        self._sync_compute_writes(bridge.ctx)
        bridge.mark_gpu_authoritative(
            "light",
            "visible_illumination",
            "cell_optical_dose",
            "gas_optical_dose",
        )
        if fuse_latch_clear:
            bridge.mark_gpu_authoritative("cell_core", "material")
            self.last_reaction_latch_clear_fused = True

    def clear_outputs(self, world: "WorldEngine") -> None:
        ctx = world.bridge.ctx
        if ctx is None:
            raise RuntimeError("GPU optics pipeline requires a valid ModernGL context")
        self.last_reaction_latch_clear_fused = False
        self._ensure_programs(ctx)
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        if not bridge.enabled or bridge.ctx is None:
            raise RuntimeError("GPU optics pipeline requires bridge GPU resources for clearing optics state")
        dose_channel_count = int(world.cell_optical_dose.shape[0])
        fuse_latch_clear = bool(
            self._formal_gpu_frame(world) and "cell_core" in bridge.gpu_authoritative_resources
        )
        program = self.programs["clear_bridge_outputs"]
        program["cell_grid_size"].value = (world.width, world.height)
        program["gas_grid_size"].value = (world.gas_width, world.gas_height)
        program["dose_channel_count"].value = dose_channel_count
        program["clear_reaction_latches"].value = fuse_latch_clear
        program["reaction_latched_flag_shifted"].value = int(CellFlag.REACTION_LATCHED) << 24
        bridge.textures["light"].bind_to_image(0, read=False, write=True)
        bridge.textures["visible_illumination"].bind_to_image(1, read=False, write=True)
        bridge.buffers["cell_optical_dose"].bind_to_storage_buffer(binding=0)
        bridge.buffers["gas_optical_dose"].bind_to_storage_buffer(binding=1)
        self._ensure_light_dose_guard(world).bind_to_storage_buffer(binding=2)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=3)
        groups_x = (max(world.width, world.gas_width) + LOCAL_SIZE - 1) // LOCAL_SIZE
        groups_y = (max(world.height, world.gas_height) + LOCAL_SIZE - 1) // LOCAL_SIZE
        program.run(groups_x, groups_y, dose_channel_count)
        self._sync_compute_writes(ctx)
        bridge.mark_gpu_authoritative(
            "light",
            "visible_illumination",
            "cell_optical_dose",
            "gas_optical_dose",
            LIGHT_DOSE_GUARD_BUFFER,
        )
        if fuse_latch_clear:
            bridge.mark_gpu_authoritative("cell_core", "material")
            self.last_reaction_latch_clear_fused = True
        self.last_cpu_mirror_downloaded = False

    # ``_sync_compute_writes`` is inherited from :class:`GPUPipelineBase`;
    # the default ``_barrier_bits`` covers exactly the three bits this method
    # used inline (SHADER_IMAGE_ACCESS / TEXTURE_FETCH / SHADER_STORAGE).
