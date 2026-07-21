from __future__ import annotations

import inspect

import numpy as np
import pytest

from oracle_game.sim.gpu_collapse_dirty import (
    COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER,
    COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER,
    ensure_collapse_structure_dirty_tile_mask,
    ensure_collapse_structure_dirty_tile_queue,
)
from oracle_game.sim.gpu_heat import GPUHeatPipeline, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source
from oracle_game.world import WorldEngine


def test_heat_terminal4x6_is_default_enabled() -> None:
    pipeline = GPUHeatPipeline()
    assert pipeline._terminal4x6_fusion_enabled is True
    assert pipeline._terminal4x6_workgroup16x8_enabled is False
    assert pipeline._terminal_bridge_aux_dirty_fusion_enabled is True
    assert pipeline._terminal_phase_fusion_enabled is False
    assert pipeline._terminal_dirty_publish_fusion_enabled is True
    assert pipeline._terminal_dirty_workgroup_aggregation_enabled is True
    assert pipeline._terminal_split_target_active_reuse_enabled is True
    assert pipeline._terminal_dead_condense_target_store_elision_enabled is True
    assert pipeline._terminal_inplace_sparse_write_enabled is True
    assert pipeline._terminal_sparse_resident_specialization_enabled is True
    assert pipeline._deferred_dirty_publish_handoff_enabled is False


def test_heat_terminal4x6_has_strict_formal_dispatch_gate() -> None:
    source = inspect.getsource(GPUHeatPipeline.step)
    assert "pipeline._terminal4x6_fusion_enabled" in source
    assert "and fuse_condense_apply_gas" in source
    assert "int(world.gas_width) == (int(world.width) + 3) // 4" in source
    assert "int(world.gas_height) == (int(world.height) + 3) // 4" in source
    assert "if fuse_terminal4x6:" in source
    assert "pipeline._run_apply_terminal4x6" in source
    assert "pipeline._terminal_phase_fusion_enabled" in source
    assert "pipeline._terminal_dirty_publish_fusion_enabled" in source
    assert "and deferred_cell_core" in source
    assert 'getattr(world, "heat_motion_handoff_active", False)' in source
    assert '"placeholder_displaced_material"' in source
    assert "_active_scheduler_gpu_authoritative(world)" in source
    assert "ensure_collapse_structure_dirty_tile_mask(world)" in source
    assert "ensure_collapse_structure_dirty_tile_queue(world)" in source
    assert "if not terminal_phase_fusion:" in source
    assert "skip_cell=terminal_dirty_publish_fusion" in source


def test_heat_deferred_dirty_publish_requires_motion_dirty_handoff() -> None:
    source = inspect.getsource(GPUHeatPipeline._publish_bridge_outputs)
    assert "pipeline._deferred_dirty_publish_handoff_enabled" in source
    assert "defer_cell_core" in source
    assert 'getattr(motion_pipeline, "can_consume_deferred_heat_core", None)' in source
    assert "and can_consume_deferred(world)" in source
    assert "skip_cell_publish = bool(skip_cell or defer_dirty_publish_to_handoff)" in source
    assert "publish_bridge_outputs.cell_deferred_to_handoff" in source


def test_heat_terminal4x6_preserves_row_major_condense_and_aux_integrity() -> None:
    source = shader_source("heat/apply_terminal4x6.comp", _SHADER_SUBS)
    assert "layout(local_size_x=8, local_size_y=8" in source
    assert "int lane = int(gl_LocalInvocationIndex);" in source
    assert "empty_mask[gas_lane] |= 1u << uint(cell_lane);" in source
    assert "bitCount(empty_mask[gas_lane] & lower_lane_mask)" in source
    assert "float auxiliary_integrity = original_integrity - 0.5 * dt;" in source
    assert "for (int species = 0; species < 6; ++species)" in source
    assert source.count("layout(std430") == 14
    assert source.count(") writeonly uniform ") == 8


def test_heat_terminal4x6_workgroup16x8_is_compile_specialized() -> None:
    source = shader_source(
        "heat/apply_terminal4x6.comp",
        {
            **_SHADER_SUBS,
            "TERMINAL_LOCAL_SIZE_X": 16,
            "TERMINAL_LOCAL_SIZE_Y": 8,
            "TERMINAL_CELL_COUNT": 128,
            "TERMINAL_GAS_SIZE_X": 4,
            "TERMINAL_GAS_SIZE_Y": 2,
            "TERMINAL_GAS_COUNT": 8,
            "TERMINAL_CONDENSE_RANK_COUNT": 48,
        },
    )
    assert "local_size_x=16" in source
    assert "local_size_y=8" in source
    assert "shared uint after_cell_state[128];" in source
    assert "shared int condense_rank_material[48];" in source
    assert "gas_in_group.y * 4 + gas_in_group.x" in source
    assert "shared_cell.y * 16 + shared_cell.x" in source


def test_heat_terminal_dirty_workgroup_aggregation_is_compile_specialized() -> None:
    substitutions = dict(_SHADER_SUBS)
    substitutions["DIRTY_WORKGROUP_AGGREGATE"] = 1
    source = shader_source("heat/apply_terminal4x6.comp", substitutions)
    assert "shared uint workgroup_dirty_tile_bits;" in source
    assert "atomicOr(workgroup_dirty_tile_bits, 1u << 4u);" in source
    assert "flush_workgroup_dirty_tiles();" in source


def test_heat_terminal_sparse_resident_path_is_compile_specialized() -> None:
    substitutions = {
        **_SHADER_SUBS,
        "DIRTY_WORKGROUP_AGGREGATE": 1,
        "TERMINAL_SPARSE_RESIDENT_SPECIALIZED": 1,
    }
    source = shader_source("heat/apply_terminal4x6.comp", substitutions)
    assert "#define TERMINAL_FUSE_PHASE_BOIL false" in source
    assert "#define TERMINAL_FUSE_DIRTY_PUBLISH true" in source
    assert "#define TERMINAL_USE_BRIDGE_DISPLACED true" in source
    assert "#define TERMINAL_WRITE_PRIVATE_DISPLACED false" in source
    assert "#define TERMINAL_INPLACE_SPARSE_WRITE true" in source
    assert "#define TERMINAL_WRITE_CONDENSE_TARGETS false" in source


def test_heat_terminal_sparse_resident_specialization_has_strict_gate() -> None:
    source = inspect.getsource(GPUHeatPipeline._run_apply_terminal4x6)
    assert "pipeline._terminal_sparse_resident_specialization_enabled" in source
    assert "and pipeline._formal_gpu_frame(world)" in source
    assert "and bridge_aux_resident" in source
    assert "and bridge_aux_dirty" in source
    assert "and dirty_publish_resources is not None" in source
    assert "and aggregate_dirty_tiles" in source
    assert "and not fuse_phase_boil" in source
    assert "and not bridge_gas_resident" in source
    assert "and not use_workgroup16x8" in source
    assert "and pipeline._terminal_inplace_sparse_write_enabled" in source
    assert "and pipeline._terminal_split_target_active_reuse_enabled" in source
    assert "and pipeline._terminal_dead_condense_target_store_elision_enabled" in source


def test_heat_terminal_aux_clear_replays_post_transition_boil_integrity() -> None:
    source = shader_source(
        "heat/phase_boil_targets.comp",
        _SHADER_SUBS,
        includes=["heat/_common.comp"],
    )
    phase_integrity = "integrity_value = response_params[target_material].y;"
    boil_integrity = "integrity_value -= 0.5 * dt;"
    assert source.index(phase_integrity) < source.index(boil_integrity)
    assert "if (clear_terminal_aux)" in source
    assert "phase_params[target_material].x != phase_falling_island" in source
    assert "clear_entity = !is_placeholder_material(target_material);" in source


def _seed_terminal4x6_world(engine: WorldEngine) -> None:
    rng = np.random.default_rng(0x4AFA)
    material_ids = np.asarray(
        sorted(material_id for material_id in engine.rulebook.materials_by_id if material_id > 0),
        dtype=np.int32,
    )
    assert material_ids.size > 0
    phase_by_material = np.zeros((int(material_ids.max()) + 1,), dtype=np.uint8)
    for material_id, material in engine.rulebook.materials_by_id.items():
        if 0 <= int(material_id) < phase_by_material.size:
            phase_by_material[int(material_id)] = int(material.default_phase)

    engine.material_id[:] = rng.choice(material_ids, size=engine.material_id.shape, replace=True)
    engine.phase[:] = phase_by_material[engine.material_id]
    engine.cell_flags[:] = rng.integers(0, 256, size=engine.cell_flags.shape, dtype=np.uint8)
    engine.timer_pack[:] = rng.integers(0, 256, size=engine.timer_pack.shape, dtype=np.uint8)
    engine.cell_temperature[:] = rng.uniform(-2000.0, 5000.0, size=engine.cell_temperature.shape).astype("f4")
    engine.integrity[:] = rng.uniform(-0.01, 1.0, size=engine.integrity.shape).astype("f4")
    engine.velocity[:] = rng.normal(0.0, 0.5, size=engine.velocity.shape).astype("f4")
    engine.island_id[:] = rng.integers(0, 200, size=engine.island_id.shape, dtype=np.int32)
    engine.entity_id[:] = rng.integers(0, 200, size=engine.entity_id.shape, dtype=np.int32)
    engine.placeholder_displaced_material[:] = rng.choice(
        np.concatenate((np.zeros((1,), dtype=np.int32), material_ids)),
        size=engine.placeholder_displaced_material.shape,
        replace=True,
    )
    engine.ambient_temperature[:] = rng.uniform(-200.0, 200.0, size=engine.ambient_temperature.shape).astype("f4")
    engine.gas_concentration[:] = rng.uniform(0.65, 0.95, size=engine.gas_concentration.shape).astype("f4")

    for tile_y in range(engine.active.tile_height):
        for tile_x in range(engine.active.tile_width):
            engine.active.active_tile_ttl[tile_y][tile_x] = 3 if (tile_x + 2 * tile_y) % 3 else 0
    for chunk_y in range(engine.active.chunk_height):
        for chunk_x in range(engine.active.chunk_width):
            engine.active.active_chunk_mask[chunk_y][chunk_x] = True


def _capture_terminal4x6_state(engine: WorldEngine) -> dict[str, bytes]:
    pipeline = engine.heat_solver.gpu_pipeline
    resources = pipeline.resources
    ctx = engine.bridge.ctx
    assert resources is not None and ctx is not None
    ctx.finish()
    dirty_count_bytes = engine.bridge.buffers[
        COLLAPSE_STRUCTURE_DIRTY_TILE_COUNT_BUFFER
    ].read()
    dirty_count = int(np.frombuffer(dirty_count_bytes, dtype=np.uint32, count=1)[0])
    dirty_list = np.frombuffer(
        engine.bridge.buffers[COLLAPSE_STRUCTURE_DIRTY_TILE_LIST_BUFFER].read(),
        dtype=np.int32,
    ).reshape((-1, 2))[:dirty_count]
    if dirty_list.size:
        dirty_list = dirty_list[np.lexsort((dirty_list[:, 0], dirty_list[:, 1]))]
    snapshot = {
        "resident.cell_state": resources.cell_state_tex.read(),
        "resident.timer": resources.timer_tex.read(),
        "resident.temperature": resources.temp_ping.read(),
        "resident.integrity": resources.integrity_tex.read(),
        "resident.island": resources.island_id_tex.read(),
        "resident.entity": resources.entity_id_tex.read(),
        "resident.displaced": resources.displaced_tex.read(),
        "resident.velocity": resources.velocity_tex.read(),
        "resident.ambient": resources.ambient_pong.read(),
        "resident.gas": resources.gas_out_tex.read(),
        "resident.phase_target": resources.phase_target_tex.read(),
        "resident.boil_target": resources.boil_target_tex.read(),
        "resident.condense_target": resources.condense_target_tex.read(),
        "bridge.cell_core": engine.bridge.buffers["cell_core"].read(
            size=engine.width * engine.height * 5 * np.dtype(np.uint32).itemsize
        ),
        "bridge.material": engine.bridge.textures["material"].read(),
        "bridge.island": engine.bridge.buffers["island_id"].read(size=engine.island_id.nbytes),
        "bridge.entity": engine.bridge.buffers["entity_id"].read(size=engine.entity_id.nbytes),
        "bridge.displaced": engine.bridge.buffers["placeholder_displaced_material"].read(
            size=engine.placeholder_displaced_material.nbytes
        ),
        "bridge.ambient": engine.bridge.textures["ambient_temperature"].read(),
        "bridge.gas": engine.bridge.buffers["gas_concentration"].read(
            size=engine.gas_concentration.nbytes
        ),
        "bridge.collapse_dirty_mask": engine.bridge.buffers[
            COLLAPSE_STRUCTURE_DIRTY_TILE_MASK_BUFFER
        ].read(),
        "bridge.collapse_dirty_count": dirty_count_bytes,
        "bridge.collapse_dirty_list_live_sorted": dirty_list.tobytes(),
        "bridge.collapse_dirty_dispatch": engine.bridge.buffers[
            COLLAPSE_STRUCTURE_DIRTY_TILE_DISPATCH_ARGS_BUFFER
        ].read(),
    }
    if pipeline._terminal_dead_condense_target_store_elision_enabled:
        snapshot.pop("resident.condense_target")
    return snapshot


def _run_terminal4x6_sequence(
    fused_frames: tuple[bool, bool],
    *,
    width: int = 67,
    height: int = 67,
    heat_motion_handoff: bool = False,
    bridge_aux_dirty_fusion: bool = True,
    phase_fusion: bool = False,
    dirty_publish_fusion: bool = False,
    dirty_workgroup_aggregation: bool = False,
    workgroup16x8: bool = False,
    inplace_sparse_write: bool = False,
    sparse_resident_specialization: bool = False,
    full_active: bool = False,
) -> list[dict[str, bytes]]:
    # 67x67 maps to a 17x17 gas grid, so the final 8x8 workgroup contains
    # partial cells plus invalid right, bottom, and bottom-right gas quadrants.
    engine = WorldEngine(width=width, height=height, gas_cell_size=4)
    try:
        pipeline = engine.heat_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU heat pipeline is not available")
        if int(engine.gas_concentration.shape[0]) != 6:
            pytest.skip("terminal4x6 requires the six-species game plan")
        _seed_terminal4x6_world(engine)
        if full_active:
            for row in engine.active.active_tile_ttl:
                row[:] = [3] * len(row)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine.bridge.mark_gpu_authoritative(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "ambient_temperature",
            "gas_concentration",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )

        previous_frame_active = engine._world_simulation_frame_active
        had_handoff = hasattr(engine, "heat_motion_handoff_active")
        previous_handoff = bool(getattr(engine, "heat_motion_handoff_active", False))
        engine._world_simulation_frame_active = True
        engine.heat_motion_handoff_active = heat_motion_handoff
        pipeline._terminal_bridge_aux_dirty_fusion_enabled = bridge_aux_dirty_fusion
        pipeline._terminal_phase_fusion_enabled = phase_fusion
        pipeline._terminal_dirty_publish_fusion_enabled = dirty_publish_fusion
        pipeline._terminal_dirty_workgroup_aggregation_enabled = (
            dirty_workgroup_aggregation
        )
        pipeline._terminal4x6_workgroup16x8_enabled = workgroup16x8
        pipeline._terminal_inplace_sparse_write_enabled = inplace_sparse_write
        pipeline._terminal_sparse_resident_specialization_enabled = (
            sparse_resident_specialization
        )
        # These cases isolate terminal fusion/residency. Packed target ABI has
        # its own full/partial and generation-fallback exact coverage.
        pipeline._packed_phase_boil_targets_enabled = False
        try:
            snapshots: list[dict[str, bytes]] = []
            for frame_id, fused in enumerate(fused_frames, start=1):
                pipeline._terminal4x6_fusion_enabled = fused
                engine.frame_id = frame_id
                engine.heat_solver.step(engine, 1.0 / 60.0)
                marker = pipeline._terminal_bridge_aux_dirty_frame_id
                phase_marker = pipeline._terminal_phase_fusion_frame_id
                dirty_marker = pipeline._terminal_dirty_publish_frame_id
                if heat_motion_handoff and fused and (
                    bridge_aux_dirty_fusion or phase_fusion or dirty_publish_fusion
                ):
                    assert marker == frame_id
                else:
                    assert marker is None
                if heat_motion_handoff and fused and phase_fusion:
                    assert phase_marker == frame_id
                else:
                    assert phase_marker is None
                if heat_motion_handoff and fused and dirty_publish_fusion:
                    assert dirty_marker == frame_id
                else:
                    assert dirty_marker is None
                assert pipeline.last_terminal_dirty_workgroup_aggregation_used is (
                    heat_motion_handoff
                    and fused
                    and dirty_publish_fusion
                    and dirty_workgroup_aggregation
                )
                assert pipeline.last_terminal4x6_workgroup16x8_used is (
                    fused and workgroup16x8
                )
                assert pipeline.last_terminal_sparse_resident_specialization_used is (
                    fused
                    and heat_motion_handoff
                    and dirty_publish_fusion
                    and dirty_workgroup_aggregation
                    and inplace_sparse_write
                    and sparse_resident_specialization
                )
                snapshots.append(_capture_terminal4x6_state(engine))
                if heat_motion_handoff:
                    assert pipeline.abort_deferred_cell_core(engine)
            return snapshots
        finally:
            engine._world_simulation_frame_active = previous_frame_active
            if had_handoff:
                engine.heat_motion_handoff_active = previous_handoff
            else:
                del engine.heat_motion_handoff_active
    finally:
        engine.close()


def test_gpu_heat_terminal4x6_matches_fallback_across_edges_frames_and_role_swaps() -> None:
    fallback = _run_terminal4x6_sequence((False, False))
    scenarios = {
        "consecutive-fused": _run_terminal4x6_sequence((True, True)),
        "fused-to-fallback": _run_terminal4x6_sequence((True, False)),
    }

    for scenario, snapshots in scenarios.items():
        assert len(snapshots) == len(fallback) == 2
        for frame_id, (expected, actual) in enumerate(zip(fallback, snapshots, strict=True), start=1):
            assert actual.keys() == expected.keys()
            for resource_name in expected:
                assert actual[resource_name] == expected[resource_name], (
                    f"{scenario} frame {frame_id} differs in {resource_name}"
                )


@pytest.mark.parametrize("width,height", ((64, 64), (67, 67)))
def test_gpu_heat_terminal4x6_inplace_sparse_write_is_two_frame_raw_byte_exact(
    width: int,
    height: int,
) -> None:
    control = _run_terminal4x6_sequence((True, True), width=width, height=height)
    candidate = _run_terminal4x6_sequence(
        (True, True),
        width=width,
        height=height,
        inplace_sparse_write=True,
    )
    for frame_id, (expected, actual) in enumerate(
        zip(control, candidate, strict=True), start=1
    ):
        assert actual.keys() == expected.keys()
        for resource_name in expected:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal in-place sparse write frame {frame_id} differs in "
                f"{resource_name}"
            )


@pytest.mark.parametrize("width,height", ((64, 64), (67, 67)))
@pytest.mark.parametrize("fused_frames", ((True, True), (True, False)))
@pytest.mark.parametrize("full_active", (False, True), ids=("partial", "full"))
def test_gpu_heat_terminal_sparse_resident_specialization_is_raw_byte_exact(
    width: int,
    height: int,
    fused_frames: tuple[bool, bool],
    full_active: bool,
) -> None:
    kwargs = {
        "width": width,
        "height": height,
        "heat_motion_handoff": True,
        "dirty_publish_fusion": True,
        "dirty_workgroup_aggregation": True,
        "inplace_sparse_write": True,
        "full_active": full_active,
    }
    control = _run_terminal4x6_sequence(fused_frames, **kwargs)
    candidate = _run_terminal4x6_sequence(
        fused_frames,
        sparse_resident_specialization=True,
        **kwargs,
    )
    for frame_id, (expected, actual) in enumerate(
        zip(control, candidate, strict=True), start=1
    ):
        assert actual.keys() == expected.keys()
        # Sparse bridge residency intentionally leaves these private caches
        # unhydrated; bridge-owned aux below remains byte-exact.
        private_unresident_aux = {
            "resident.island",
            "resident.entity",
            "resident.displaced",
        }
        for resource_name in expected.keys() - private_unresident_aux:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal sparse specialization frame {frame_id} differs in "
                f"{resource_name}"
            )


@pytest.mark.parametrize("width,height", ((64, 64), (67, 67)))
@pytest.mark.parametrize("dirty_workgroup_aggregation", (False, True))
def test_gpu_heat_terminal4x6_workgroup16x8_is_two_frame_raw_byte_exact(
    dirty_workgroup_aggregation: bool,
    width: int,
    height: int,
) -> None:
    kwargs = {
        "width": width,
        "height": height,
        "heat_motion_handoff": dirty_workgroup_aggregation,
        "dirty_publish_fusion": dirty_workgroup_aggregation,
        "dirty_workgroup_aggregation": dirty_workgroup_aggregation,
    }
    control = _run_terminal4x6_sequence((True, True), **kwargs)
    candidate = _run_terminal4x6_sequence(
        (True, True),
        workgroup16x8=True,
        **kwargs,
    )
    for frame_id, (expected, actual) in enumerate(
        zip(control, candidate, strict=True), start=1
    ):
        assert actual.keys() == expected.keys()
        for resource_name in expected:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal 16x8 frame {frame_id} differs in {resource_name}"
            )


def test_gpu_heat_terminal_bridge_aux_dirty_matches_publish_pass() -> None:
    control = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
        bridge_aux_dirty_fusion=False,
    )
    candidate = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
        bridge_aux_dirty_fusion=True,
    )
    assert int(np.frombuffer(candidate[0]["bridge.collapse_dirty_count"], dtype=np.uint32)[0]) > 0
    for frame_id, (expected, actual) in enumerate(zip(control, candidate, strict=True), start=1):
        assert actual.keys() == expected.keys()
        for resource_name in expected:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal bridge fusion frame {frame_id} differs in {resource_name}"
            )


def test_gpu_heat_terminal_dirty_workgroup_aggregation_is_two_frame_exact() -> None:
    control = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
        dirty_publish_fusion=True,
    )
    candidate = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
        dirty_publish_fusion=True,
        dirty_workgroup_aggregation=True,
    )
    for frame_id, (expected, actual) in enumerate(
        zip(control, candidate, strict=True), start=1
    ):
        for resource_name in expected:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal dirty workgroup aggregation frame {frame_id} "
                f"differs in {resource_name}"
            )


@pytest.mark.parametrize(
    ("phase_fusion", "dirty_publish_fusion"),
    ((True, False), (False, True), (True, True)),
)
def test_gpu_heat_terminal_split_fusions_match_deferred_observable_state(
    phase_fusion: bool,
    dirty_publish_fusion: bool,
) -> None:
    control = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
    )
    candidate = _run_terminal4x6_sequence(
        (True, True),
        heat_motion_handoff=True,
        phase_fusion=phase_fusion,
        dirty_publish_fusion=dirty_publish_fusion,
    )
    private_intermediates = (
        {
            "resident.island",
            "resident.entity",
            "resident.phase_target",
            "resident.boil_target",
        }
        if phase_fusion
        else set()
    )
    for frame_id, (expected, actual) in enumerate(zip(control, candidate, strict=True), start=1):
        for resource_name in expected.keys() - private_intermediates:
            assert actual[resource_name] == expected[resource_name], (
                f"terminal split fusion frame {frame_id} differs in {resource_name}"
            )


def test_gpu_heat_terminal_split_fusions_are_same_context_dirty_queue_exact() -> None:
    engine = WorldEngine(width=67, height=67, gas_cell_size=4)
    try:
        pipeline = engine.heat_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU heat pipeline is not available")
        if int(engine.gas_concentration.shape[0]) != 6:
            pytest.skip("terminal4x6 requires the six-species game plan")
        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        engine.heat_motion_handoff_active = True
        engine.profile_passes_enabled = True
        # Keep the compared private phase/boil intermediates on one ABI; the
        # default packed ABI is verified independently.
        pipeline._packed_phase_boil_targets_enabled = False
        snapshots: dict[tuple[bool, bool], dict[str, bytes]] = {}
        pass_names: dict[tuple[bool, bool], set[str]] = {}
        sparse_aux_residency: dict[tuple[bool, bool], bool] = {}
        try:
            for mode in ((False, False), (True, False), (False, True), (True, True)):
                phase_fusion, dirty_publish_fusion = mode
                _seed_terminal4x6_world(engine)
                engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
                engine.bridge.mark_gpu_authoritative(
                    "cell_core",
                    "material",
                    "island_id",
                    "entity_id",
                    "placeholder_displaced_material",
                    "ambient_temperature",
                    "gas_concentration",
                    "active_meta",
                    "active_tile_ttl",
                    "active_chunk_mask",
                )
                ensure_collapse_structure_dirty_tile_mask(engine, clear=True)
                ensure_collapse_structure_dirty_tile_queue(engine, clear=True)
                pipeline._terminal4x6_fusion_enabled = True
                pipeline._terminal_phase_fusion_enabled = phase_fusion
                pipeline._terminal_dirty_publish_fusion_enabled = dirty_publish_fusion
                engine.frame_id = 41
                pipeline.reset_pass_profile()
                engine.heat_solver.step(engine, 1.0 / 60.0)
                snapshot = _capture_terminal4x6_state(engine)
                snapshots[mode] = snapshot
                sparse_aux_residency[mode] = bool(
                    pipeline.last_heat_sparse_bridge_residency_used
                )
                pass_names[mode] = {
                    str(entry["name"])
                    for entry in pipeline.last_pass_profile["passes"]
                }
                assert pipeline.abort_deferred_cell_core(engine)
        finally:
            engine._world_simulation_frame_active = previous_frame_active

        control = snapshots[(False, False)]
        for mode in ((True, False), (False, True), (True, True)):
            phase_fusion, dirty_publish_fusion = mode
            names = pass_names[mode]
            assert ("phase_boil_targets" not in names) is phase_fusion
            assert ("publish_bridge_outputs.cell" not in names) is dirty_publish_fusion
            private_intermediates = (
                {
                    "resident.island",
                    "resident.entity",
                    "resident.phase_target",
                    "resident.boil_target",
                }
                if phase_fusion
                else set()
            )
            if sparse_aux_residency[mode]:
                private_intermediates.update(
                    {
                        "resident.island",
                        "resident.entity",
                        "resident.displaced",
                    }
                )
            for resource_name in control.keys() - private_intermediates:
                assert snapshots[mode][resource_name] == control[resource_name], (
                    f"same-context terminal split fusion {mode} differs in {resource_name}"
                )
    finally:
        engine.close()
